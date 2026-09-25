"""Opt-in Linux signal ownership and raw-prefix export for trusted research.

The caller exits with the returned proposedExitCode. Only a separate parent can
observe that actual exit. This helper neither sets nor extends resource limits.
"""
import copy
import os
from pathlib import Path
import signal
import sys
import threading

from solver_temporal_transport import (PROFILE, PrivateDirectory, error_record,
                                       require_identity, require_policy)


MANAGED = (signal.SIGXCPU, signal.SIGTERM, signal.SIGINT, signal.SIGXFSZ)
_OWNER = None


class _Signals:
    def __init__(self, execution):
        from solver_temporal_execution import TemporalStopRequested
        self.stop_type = TemporalStopRequested
        self.execution = execution
        self.phase = "installing"
        self.first = None
        self.old_mask = None
        self.old_handlers = {}
        self.attempted = []

    def handler(self, signum, frame):
        # Prevent a nested different signal from replacing the first observation
        # between its check and assignment. Formatting and I/O stay outside.
        old = signal.pthread_sigmask(signal.SIG_BLOCK, MANAGED)
        first = self.first is None
        try:
            if first:
                self.first = (signum, self.phase, sys.exception())
            self.execution.request_stop()
            interrupt = first and self.phase == "numerical"
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)
        if interrupt:
            raise self.stop_type(self.execution.stop_reason)

    def install(self):
        # Record the old mask before a later installation operation can fail.
        self.old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, ())
        signal.pthread_sigmask(signal.SIG_BLOCK, MANAGED)
        installed = False
        try:
            self.old_handlers = {number: signal.getsignal(number) for number in MANAGED}
            for number in MANAGED:
                self.attempted.append(number)
                signal.signal(number, self.handler)
            installed = True
        finally:
            # Borrow ownership of managed delivery, even if the caller had a
            # managed signal blocked; restore the caller's exact mask on exit.
            signal.pthread_sigmask(signal.SIG_SETMASK,
                                   self.old_mask-set(MANAGED) if installed else self.old_mask)

    def set_phase(self, phase):
        old = signal.pthread_sigmask(signal.SIG_BLOCK, MANAGED)
        try:
            self.phase = phase
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)

    def restore(self):
        errors = []
        self.phase = "restoring"
        if self.old_mask is None:
            # Installation never obtained a mask or changed a handler. Do not
            # introduce a new block that cannot be restored truthfully.
            return errors
        try:
            signal.pthread_sigmask(signal.SIG_BLOCK, MANAGED)
        except BaseException as error:
            errors.append(error)
        for number in reversed(self.attempted):
            try:
                signal.signal(number, self.old_handlers[number])
            except BaseException as error:
                errors.append(error)
        if self.old_mask is not None:
            try:
                # Pending signals may invoke the restored prior handlers here.
                # A resulting hard/default exit belongs to the parent's receipt.
                signal.pthread_sigmask(signal.SIG_SETMASK, self.old_mask)
            except BaseException as error:
                errors.append(error)
        return errors

    def signal_record(self):
        if self.first is None:
            return None
        number, phase, active_error = self.first
        return {"number": number, "observedPhase": phase,
                "activeExceptionAtFirstSignal": error_record(active_error) if active_error is not None else None,
                "observationScope": "first Python handler observation; kernel arrivals may coalesce or be delayed"}


def run_temporal_worker(operation, directory, execution, *, run_identity, export_policy):
    """Invoke operation(handle) once, then export raw evidence in directory/worker.

    Exit precedence: recovery/ordinary-operation failure -> 1; otherwise a latched
    stop -> 86; ordinary full completion -> 0; ordinary incomplete/unavailable -> 1.
    A dedicated stop exception is not an ordinary operation error. Metadata is
    explicitly as-of publication, before handler restoration and actual exit.
    Large snapshots are not structurally or numerically admitted by this helper.
    """
    global _OWNER
    from solver_temporal_execution import PROFILE as RAW_PROFILE, TemporalStopRequested, require_execution
    if sys.platform != "linux" or threading.current_thread() is not threading.main_thread():
        raise ValueError("Temporal process ownership requires the Linux main Python thread")
    if _OWNER is not None:
        raise ValueError("Temporal signal ownership cannot be nested")
    if not callable(operation):
        raise ValueError("A trusted temporal operation is required")
    require_execution(execution)
    require_identity(run_identity); require_policy(export_policy)
    store = PrivateDirectory.create(Path(directory)/"worker", export_policy)
    scope = _Signals(execution)
    _OWNER = scope
    escaping = None
    secondary = []
    snapshot = None
    raw_descriptor = None
    returned = False
    outcome = None

    def failure(error, stage):
        secondary.append({"stage": stage, **error_record(error)})

    def proposed():
        ordinary = escaping is not None and not isinstance(escaping, TemporalStopRequested)
        if ordinary or secondary:
            return 1
        if execution.stop_requested or isinstance(escaping, TemporalStopRequested):
            return 86
        return 0 if returned and snapshot and snapshot.get("available") is True and snapshot.get("complete") is True else 1

    try:
        try:
            scope.install()
            scope.set_phase("numerical")
            execution._check_stop()
            try:
                operation(execution)
                returned = True
            except BaseException as error:
                escaping = error
            finally:
                scope.set_phase("exporting")
        except BaseException as error:
            if escaping is None:
                escaping = error
            else:
                failure(error, "signal-transition")
            scope.phase = "exporting"
        try:
            snapshot = execution.snapshot()
            if snapshot.get("profile") != RAW_PROFILE or snapshot.get("accepted") is not False:
                raise ValueError("Unexpected retained snapshot profile")
            raw_descriptor = store.publish("snapshot.json", {
                "profile": PROFILE, "accepted": False, "runIdentity": run_identity,
                "snapshot": snapshot})
        except BaseException as error:
            failure(error, "snapshot-acquisition-or-publication")
        outcome = {"profile": PROFILE, "accepted": False, "runIdentity": run_identity,
                   "asOf": "before-outcome-publication-and-handler-restoration",
                   "proposedExitCode": proposed(), "operationReturned": returned,
                   "escapingOperationOrInstallationError": error_record(escaping) if escaping is not None else None,
                   "firstSignal": scope.signal_record(), "stopRequested": execution.stop_requested,
                   "rawSnapshot": raw_descriptor, "rawPublication": copy.deepcopy(store.observations.get("snapshot.json")),
                   "secondaryFailures": copy.deepcopy(secondary),
                   "validation": "raw-bytes-only-independent-structural-and-numerical-audit-required"}
        try:
            store.publish("outcome.json", outcome)
        except BaseException as error:
            failure(error, "outcome-publication")
    finally:
        # Even partial installation is restored as far as possible. Allocation
        # failure or hard termination can prevent all recovery; no guarantee of
        # a file or of survival under a restored pending signal is made.
        try:
            for error in scope.restore():
                failure(error, "signal-restoration")
        finally:
            _OWNER = None
            try:
                store.close()
            except BaseException as error:
                failure(error, "directory-close")
    return {"profile": PROFILE, "accepted": False, "runIdentity": run_identity,
            "proposedExitCode": proposed(), "firstSignal": scope.signal_record(),
            "escapingOperationOrInstallationError": error_record(escaping) if escaping is not None else None,
            "stopRequested": execution.stop_requested, "secondaryFailures": secondary,
            "snapshotArtifact": raw_descriptor, "metadataAsOfPublication": outcome,
            "actualExitObserved": False}
