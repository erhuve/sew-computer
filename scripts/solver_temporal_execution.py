"""Opt-in, single-threaded retention for conditional temporal trials.

This is an in-memory observation API, not a journal, resume protocol, signal
handler or process supervisor. A stop is observed at guarded boundaries; it
does not immediately cancel a native call. Hard termination can lose all data.
"""
import copy
from dataclasses import dataclass, replace
from fractions import Fraction
import math

import numpy as np


PROFILE = "conditional-temporal-retained-run-v1"


class TemporalStopRequested(BaseException):
    """A requested orchestration stop, separate from numerical rejection."""


@dataclass(frozen=True)
class _Capsule:
    context: dict
    committed: tuple
    reservations: tuple = ()
    attempts: tuple = ()
    assessments: tuple = ()
    phase: str = "running"
    controller_reason: str | None = None
    report_ready: bool = False
    enrichment: str = "not-requested"
    failures: tuple = ()


class TemporalExecution:
    """Single-use run handle with a persistent first-stop latch.

    The reason is fixed before admission. request_stop() only latches; a caller
    translating a signal may also raise TemporalStopRequested. Public snapshots
    are detached JSON-compatible data. Snapshot allocation can itself fail.
    Private Python attributes are not a security boundary.
    """
    __slots__ = ("_reason", "_requested", "_bound", "_capsule", "_admission_failure")

    def __init__(self, reason="requested"):
        if hasattr(self, "_reason"):
            raise ValueError("Temporal execution cannot be reinitialized")
        if type(reason) is not str or not 1 <= len(reason) <= 128:
            raise ValueError("A fixed bounded temporal stop reason is required")
        self._reason = reason
        self._requested = False
        self._bound = False
        self._capsule = None
        self._admission_failure = None

    @property
    def stop_requested(self):
        return self._requested

    @property
    def stop_reason(self):
        return self._reason if self._requested else None

    def request_stop(self):
        self._requested = True

    def _check_stop(self):
        if self._requested:
            raise TemporalStopRequested(self._reason)

    def _claim(self):
        if self._bound:
            raise ValueError("Temporal execution is single-use")
        self._bound = True

    def _initialize(self, context, committed):
        self._capsule = _Capsule(context_value(context), committed)

    def _reserve(self, record):
        reservation = {"trial": copy.deepcopy(record), "observation": "reserved"}
        candidate = replace(self._capsule, reservations=self._capsule.reservations+(reservation,))
        self._check_stop()
        # ID and its entire allowance become retained in the same replacement.
        self._capsule = candidate

    def _observe(self, observation):
        rows = self._capsule.reservations
        latest = {**rows[-1], "observation": observation}
        self._capsule = replace(self._capsule, reservations=rows[:-1]+(latest,))

    def _record_trial(self, record):
        candidate = copy.deepcopy(record)
        rows = self._capsule.reservations
        for key, value in rows[-1]["trial"].items():
            if key != "converged" and (type(candidate.get(key)) is not type(value) or candidate[key] != value):
                raise ValueError("Recorded trial identity differs from its retained reservation")
        latest = {**rows[-1], "recordPublished": True}
        self._capsule = replace(self._capsule, reservations=rows[:-1]+(latest,),
                                attempts=self._capsule.attempts+(candidate,))

    def _record_assessment(self, record):
        record = copy.deepcopy(record)
        if record["outcome"] == "committed":
            record["outcome"] = "indicators-passed-awaiting-commit"
        self._capsule = replace(self._capsule,
                                assessments=self._capsule.assessments+(record,))

    def _commit(self, committed):
        candidate = replace(self._capsule, committed=committed)
        self._check_stop()
        # This is the sole authoritative commit on the opt-in route.
        self._capsule = candidate

    def _finished(self, reason):
        self._capsule = replace(self._capsule, phase="controller-finished", controller_reason=reason)

    def _report_ready(self):
        self._capsule = replace(self._capsule, phase="report-ready", report_ready=True)

    def _enrichment(self, status):
        self._capsule = replace(self._capsule, enrichment=status)

    def _failure(self, error, stage):
        # Best effort only; do not replace a primary interruption with another
        # exception while recording diagnostics under exhausted resources.
        try:
            self._failure_record({"type": type(error).__name__, "message": str(error)}, stage)
        except BaseException:
            pass

    def _failure_record(self, error, stage):
        row = {**copy.deepcopy(error), "stage": stage}
        if self._capsule is None:
            self._admission_failure = row
        else:
            self._capsule = replace(self._capsule, failures=self._capsule.failures+(row,))

    def snapshot(self):
        """Copy fixed context and raw observations without any mechanical work."""
        from solver_temporal_control import rational, state_record
        capsule = self._capsule
        result = {"profile": PROFILE, "accepted": False, "bound": self._bound,
                  "stopRequested": self._requested, "stopReason": self.stop_reason,
                  "available": capsule is not None}
        if capsule is None:
            result["admissionFailure"] = copy.deepcopy(self._admission_failure)
            return result
        q, v, fraction, accepted, transactions, total = capsule.committed
        reservations = copy.deepcopy(list(capsule.reservations))
        result.update(context=copy.deepcopy(capsule.context), phase=capsule.phase,
                      controllerReason=capsule.controller_reason, reportReady=capsule.report_ready,
                      enrichment=capsule.enrichment, failures=copy.deepcopy(list(capsule.failures)),
                      complete=fraction == 1., completedFraction=fraction,
                      state=state_record(q, v), acceptedSteps=copy.deepcopy(list(accepted)),
                      transactions=copy.deepcopy(list(transactions)),
                      conditionalAbsoluteEnergyBoundJoules=rational(total),
                      reservations=reservations, attempts=copy.deepcopy(list(capsule.attempts)),
                      temporalAssessments=copy.deepcopy(list(capsule.assessments)),
                      resources={"reservedTrials": len(reservations),
                                 "recordedTrials": len(capsule.attempts),
                                 "chargedEvaluationAllowance": sum(row["trial"]["evaluationAllowanceCharged"]
                                                                    for row in reservations)})
        return result


def require_execution(execution):
    if type(execution) is not TemporalExecution:
        raise ValueError("An exact TemporalExecution handle is required")
    if execution._bound:
        raise ValueError("Temporal execution is single-use")


def context_value(value):
    """Faithful, detached encoding of admitted controls, never object repr()."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is Fraction:
        return {"numerator": str(value.numerator), "denominator": str(value.denominator)}
    if type(value) is np.ndarray and value.dtype.kind in "biuf":
        if value.dtype.kind == "f" and value.dtype.itemsize > 8:
            raise ValueError("Retained context cannot narrow extended-precision values")
        return {"dtype": value.dtype.str, "shape": list(value.shape), "values": context_value(value.tolist())}
    if type(value) in (list, tuple):
        return [context_value(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: context_value(item) for key, item in value.items()}
    raise ValueError("Temporal retained context has no declared encoding for this value")


def schedule_context(controls, kind):
    """Copy admitted knots without resampling or unifying interpolation laws."""
    if controls is None:
        return None
    fields = {
        "assembly": {"knots": "knots"},
        "sewing": {"rowIds": "_row_ids", "fractions": "_fractions", "activation": "_activation"},
        "gripper": {"gripperIds": "_ids", "fractions": "_fractions", "targetsMeters": "_targets", "activation": "_activation"},
        "fold": {"hinges": "_hinges", "fractions": "_fractions", "targetsRadians": "_targets", "activation": "_activation"}}
    return {"implementation": type(controls).__module__+"."+type(controls).__name__,
            "knots": {key: context_value(getattr(controls, attribute)) for key, attribute in fields[kind].items()},
            "interpolation": "inherited-binary64-increment" if kind != "fold" else "exact-rational-round-once"}


def enrich_result(execution, result, enrich, primary=None):
    """Retain raw evidence while building optional enrichment on a fresh copy."""
    try:
        execution._enrichment("pending")
        report = copy.deepcopy(result[2])
        enrich(report)
        execution._enrichment("ready")
        enriched = result[0], result[1], report
    except BaseException as error:
        execution._failure(error, "enrichment")
        try:
            execution._enrichment("failed")
        except BaseException as secondary:
            execution._failure(secondary, "enrichment-status")
        if primary is not None:
            raise primary from error
        if execution.stop_requested and not isinstance(error, TemporalStopRequested):
            raise TemporalStopRequested(execution.stop_reason) from error
        raise
    if primary is None:
        execution._check_stop()
    return enriched
