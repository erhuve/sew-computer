"""Linux trusted-process supervision for raw temporal evidence.

CPU and address-space limits are inherited PER PROCESS, not aggregate descendant
limits. Same-group samples/reaped usage are diagnostics. A separately declared
container/cgroup supplies aggregate constraints for actual research runs. The
worker group has fixed wall/grace deadlines; escaped untrusted processes are
outside this trusted research interface.
"""
from dataclasses import asdict, dataclass
import ctypes
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
import time

from solver_process_budget import _kill_group, _prctl, arm_parent_death, group_members
from solver_temporal_transport import (ExportPolicy, NAMES, PROFILE, PrivateDirectory,
                                       artifact_name, error_record, require_identity, require_policy)


_PARENT_OWNER = False
_PARENT_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


@dataclass(frozen=True)
class ProcessPolicy:
    cpu_soft_seconds: int
    cpu_hard_seconds: int
    wall_soft_seconds: float
    wall_grace_seconds: float
    address_space_bytes: int
    cleanup_seconds: float
    poll_seconds: float
    export: ExportPolicy

    def __post_init__(self):
        require_policy(self.export)
        if (type(self.cpu_soft_seconds) is not int or type(self.cpu_hard_seconds) is not int
                or not 1 <= self.cpu_soft_seconds < self.cpu_hard_seconds <= 86400
                or self.cpu_hard_seconds-self.cpu_soft_seconds > 600
                or type(self.address_space_bytes) is not int
                or not 2**25 <= self.address_space_bytes <= 2**36):
            raise ValueError("Explicit bounded per-process CPU/address-space limits required")
        for value, lower, upper in ((self.wall_soft_seconds, .01, 86400),
                                    (self.wall_grace_seconds, .01, 600),
                                    (self.cleanup_seconds, .01, 10),
                                    (self.poll_seconds, .001, .25)):
            if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError("Explicit bounded finite supervision intervals required")


def _child_exec(data, command):
    # This interpreter has imported only standard-library research helpers.
    # exec preserves absolute per-process CPU usage and resource limits.
    arm_parent_death(data["parentPid"])
    policy = ProcessPolicy(**{**data["policy"], "export": ExportPolicy(**data["policy"]["export"])})
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (policy.cpu_soft_seconds, policy.cpu_hard_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (policy.address_space_bytes, policy.address_space_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (policy.export.file_size_limit, policy.export.file_size_limit))
    signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGXCPU, signal.SIGTERM, signal.SIGINT, signal.SIGXFSZ))
    os.execvpe(command[0], command, os.environ)


def observe_job(store, policy):
    """Observe bounded named bytes after group cleanup; never admit a snapshot.

    Inventory failures do not hide a surviving final file. Unknown names are not
    followed, and temporary files are labelled as incomplete transport evidence.
    These checks audit a trusted writer; they do not impose aggregate disk limits
    on arbitrary code that writes elsewhere or creates unlimited files.
    """
    artifacts, errors = [], []

    def failed(stage, error):
        errors.append({"stage": stage, **error_record(error)})

    def inventory(directory, maximum):
        names = []
        # Keep the trusted anchor, but give every inventory its own open file
        # description. Reusing the anchor's stream can retain a stale directory
        # entry boundary even when its numeric position is zero.
        fd = os.open(".", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                     dir_fd=directory.fd)
        try:
            with os.scandir(fd) as entries:
                for entry in entries:
                    if len(names) == maximum:
                        raise ValueError("Too many temporal artifact names")
                    names.append(entry.name)
        finally:
            os.close(fd)
        return names

    def observe(directory, name, limit, prefix=""):
        try:
            row = directory.observe(name, limit)
            artifacts.append({**row, "path": prefix+name})
        except FileNotFoundError:
            # Hard termination can leave no artifact. Absence grants no work.
            pass
        except BaseException as error:
            failed("observe:"+prefix+name, error)

    try:
        if any(name not in ("worker", "worker.log") for name in inventory(store, 2)):
            raise ValueError("Unexpected job artifact name")
    except BaseException as error:
        failed("job-inventory", error)
    observe(store, "worker.log", policy.log_bytes)
    child = None
    try:
        try:
            fd = os.open("worker", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=store.fd)
        except FileNotFoundError:
            fd = None
        if fd is not None:
            child = PrivateDirectory(fd, policy)
            temporary_names = []
            try:
                roles = set()
                for name in inventory(child, 4):
                    role, temporary = artifact_name(name)
                    if temporary:
                        if role in roles:
                            raise ValueError("More than one temporary file for an artifact")
                        roles.add(role); temporary_names.append(name)
            except BaseException as error:
                failed("worker-inventory", error)
            for name in (*NAMES, *temporary_names):
                observe(child, name, policy.file_limit(name), "worker/")
    except BaseException as error:
        failed("worker-directory", error)
    finally:
        if child is not None:
            try: child.close()
            except BaseException as error: failed("worker-directory-close", error)
    named_bytes = sum(row["bytes"] for row in artifacts)
    if named_bytes > policy.maximum_named_bytes:
        failed("aggregate-byte-budget", ValueError("Named temporal output exceeds declared maximum"))
    return artifacts, errors, named_bytes


def supervise_temporal(command, directory, *, run_identity, policy):
    """Create a fresh private job directory and observe one trusted process group.

    Returns process/byte observations only. No snapshot is parsed or admitted;
    caller-bound run identity is not a verification of source/input files. The
    research launcher must bind those separately before/after the execution.
    """
    global _PARENT_OWNER
    if sys.platform != "linux" or threading.current_thread() is not threading.main_thread() or _PARENT_OWNER:
        raise ValueError("Non-nested Linux main-thread supervision required")
    if type(policy) is not ProcessPolicy:
        raise ValueError("An exact ProcessPolicy is required")
    policy.__post_init__(); require_identity(run_identity)
    if (type(command) not in (list, tuple) or not 1 <= len(command) <= 64
            or any(type(part) is not str or not 1 <= len(part) <= 4096 or "\0" in part for part in command)):
        raise ValueError("Bounded explicit trusted argv required")
    started = time.monotonic()
    soft_deadline = started+policy.wall_soft_seconds
    hard_deadline = soft_deadline+policy.wall_grace_seconds
    store = PrivateDirectory.create(directory, policy.export)
    _PARENT_OWNER = True
    process = None
    first_signal = None
    old_mask = None
    handlers = {}
    attempted_handlers = []
    subreaper = ctypes.c_int()
    subreaper_set = False
    log_fd = None
    reason = None
    failures = []
    sent = []
    raw_status = returncode = None
    reaped = []
    remaining = []
    cleanup_complete = False
    log_bytes = log_discarded = 0
    log_eof = False
    sampled_group_cpu = 0.

    def received(number, frame):
        nonlocal first_signal
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, _PARENT_SIGNALS)
        try:
            if first_signal is None:
                first_signal = (number, time.monotonic())
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)

    def stop(why, when=None):
        nonlocal reason, hard_deadline
        if reason is None:
            reason = why
            hard_deadline = min(hard_deadline, (time.monotonic() if when is None else when)+policy.wall_grace_seconds)

    def send(number):
        if _kill_group(process.pid, number):
            sent.append({"signal": number, "elapsedSeconds": time.monotonic()-started})

    def failed(stage, error):
        failures.append({"stage": stage, **error_record(error)})

    def live_before_reaping():
        # Empty /proc enumeration alone is not proof that the owned leader
        # exited. Keep the leader unreaped until all group signalling ends.
        status = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        return status is None or any(r["state"] not in ("Z", "X") for r in group_members(process.pid))

    def drain_log():
        nonlocal log_bytes, log_discarded, log_eof
        if process is None or process.stdout is None or log_eof: return
        # Bounded work per poll, including a continuously flooding worker.
        for _ in range(4):
            try: chunk = os.read(process.stdout.fileno(), 65536)
            except BlockingIOError: return
            if not chunk:
                log_eof = True
                return
            keep = min(len(chunk), policy.export.log_bytes-log_bytes)
            view = memoryview(chunk)[:keep]
            while view:
                count = os.write(log_fd, view)
                if count <= 0: raise OSError("Log write made no progress")
                log_bytes += count; view = view[count:]
            log_discarded += len(chunk)-keep
            if keep < len(chunk): stop("log-byte-budget-exhausted")

    try:
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, ())
        signal.pthread_sigmask(signal.SIG_BLOCK, _PARENT_SIGNALS)
        handlers = {number: signal.getsignal(number) for number in _PARENT_SIGNALS}
        installed = False
        try:
            for number in _PARENT_SIGNALS:
                attempted_handlers.append(number); signal.signal(number, received)
            installed = True
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK,
                                   old_mask-set(_PARENT_SIGNALS) if installed else old_mask)
        _prctl(37, ctypes.byref(subreaper))
        subreaper_set = True
        _prctl(36, 1)
        log_fd = os.open("worker.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                         0o600, dir_fd=store.fd)
        if first_signal is not None:
            stop("supervisor-signal", first_signal[1])
        elif time.monotonic() >= soft_deadline:
            stop("wall-soft-limit")
        else:
            child_data = {"parentPid": os.getpid(), "policy": asdict(policy)}
            argv = [sys.executable, str(Path(__file__).resolve()), "--exec-worker",
                    json.dumps(child_data, separators=(",", ":")), "--", *command]
            environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
            process = subprocess.Popen(argv, start_new_session=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, bufsize=0, env=environment)
            os.set_blocking(process.stdout.fileno(), False)
            term_sent = kill_sent = False
            while True:
                now = time.monotonic()
                if first_signal is not None: stop("supervisor-signal", first_signal[1])
                if now >= soft_deadline: stop("wall-soft-limit", soft_deadline)
                drain_log()
                members = group_members(process.pid)
                sampled_group_cpu = max(sampled_group_cpu, sum(r["cpuSeconds"] for r in members))
                status = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                if status is not None:
                    if any(r["pid"] != process.pid and r["state"] not in ("Z", "X") for r in members):
                        stop("worker-left-live-descendants")
                    break
                if reason is not None and not term_sent:
                    send(signal.SIGTERM); term_sent = True
                if now >= hard_deadline and not kill_sent:
                    send(signal.SIGKILL); kill_sent = True
                if kill_sent and now >= hard_deadline+policy.cleanup_seconds:
                    break
                time.sleep(policy.poll_seconds)
    except BaseException as error:
        failed("supervision", error)
        stop("supervisor-failure")
    finally:
        if first_signal is not None:
            stop("supervisor-signal", first_signal[1])
        if process is not None:
            # Do not reap the leader until group signalling ends: this keeps its
            # PID/PGID reserved during cleanup and avoids signalling a reused id.
            try:
                if live_before_reaping():
                    stop(reason or "worker-left-live-descendants")
                    send(signal.SIGTERM)
                    while time.monotonic() < hard_deadline:
                        if first_signal is not None: stop("supervisor-signal", first_signal[1])
                        drain_log()
                        if not live_before_reaping(): break
                        time.sleep(policy.poll_seconds)
                    if live_before_reaping():
                        send(signal.SIGKILL)
            except BaseException as error:
                failed("group-stop", error)
                stop("cleanup-failure")
                try: send(signal.SIGKILL)
                except BaseException as secondary: failed("hard-kill", secondary)
            # An already exhausted hard-limit cleanup allowance is not renewed.
            cleanup_deadline = min(time.monotonic(), hard_deadline)+policy.cleanup_seconds
            while True:
                try:
                    for index in range(64):
                        if index and time.monotonic() >= cleanup_deadline: break
                        pid, status, usage = os.wait4(-process.pid, os.WNOHANG)
                        if pid == 0: break
                        reaped.append({"pid": pid, "rawWaitStatus": status,
                                       "userCpuSeconds": usage.ru_utime, "systemCpuSeconds": usage.ru_stime})
                        if pid == process.pid:
                            raw_status, returncode = status, os.waitstatus_to_exitcode(status)
                            process.returncode = returncode
                except ChildProcessError:
                    pass
                except BaseException as error:
                    failed("reaping", error)
                    break
                try:
                    drain_log(); remaining = group_members(process.pid)
                except BaseException as error:
                    failed("cleanup-inspection", error)
                    remaining = [{"inspectionUnavailable": True}]
                if (not remaining and raw_status is not None) or time.monotonic() >= cleanup_deadline: break
                try: time.sleep(policy.poll_seconds)
                except BaseException as error:
                    failed("cleanup-wait", error)
                    break
            cleanup_complete = not remaining and raw_status is not None
            if process.stdout is not None:
                try: process.stdout.close()
                except BaseException as error: failed("log-pipe-close", error)
        if log_fd is not None:
            try:
                os.fchmod(log_fd, 0o400); os.fsync(log_fd)
            except BaseException as error: failed("log-sync", error)
            try: os.close(log_fd)
            except BaseException as error: failed("log-close", error)
        if subreaper_set:
            try: _prctl(36, subreaper.value)
            except BaseException as error: failed("subreaper-restoration", error)
        if old_mask is not None:
            try: signal.pthread_sigmask(signal.SIG_BLOCK, _PARENT_SIGNALS)
            except BaseException as error: failed("mask-block", error)
            for number in reversed(attempted_handlers):
                try: signal.signal(number, handlers[number])
                except BaseException as error: failed("handler-restoration", error)
            try: signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            except BaseException as error: failed("mask-restoration", error)
        _PARENT_OWNER = False
    if first_signal is not None:
        stop("supervisor-signal", first_signal[1])
    # Byte-bounded observation is separate from the worker wall budget. An outer
    # research launcher must declare the parent/observation budget as well.
    artifacts, evidence_errors, named_bytes = observe_job(store, policy.export)
    try: store.close()
    except BaseException as error: evidence_errors.append({"stage": "job-directory-close", **error_record(error)})
    return {"profile": PROFILE, "accepted": False, "runIdentity": run_identity,
            "command": list(command), "limits": asdict(policy), "workerRawWaitStatus": raw_status,
            "workerReturnCode": returncode, "workerSignal": -returncode if returncode is not None and returncode < 0 else None,
            "workerProcessId": process.pid if process is not None else None,
            "reason": reason or ("worker-exited" if returncode == 0 else "worker-nonzero-or-unavailable-exit"),
            "supervisorFirstSignal": first_signal[0] if first_signal else None, "signalsSent": sent,
            "processExitSucceeded": returncode == 0 and cleanup_complete and reason is None and not failures,
            "byteObservationWithoutErrors": not evidence_errors,
            "cleanupComplete": cleanup_complete, "remainingGroupMembers": remaining,
            "reapedProcessUsage": reaped, "sampledGroupCpuSeconds": sampled_group_cpu,
            "resourceScope": "CPU/AS limits inherited per process; group samples and reaped usage are diagnostics, not a rigorous aggregate bound. Separate container/cgroup limits are required for research aggregate budgets.",
            "wallScope": "Fixed soft/hard deadlines from launch setup; cleanup bounded separately; byte observation afterwards under outer launcher's budget.",
            "elapsedSeconds": time.monotonic()-started, "logBytesRetained": log_bytes,
            "logBytesObservedDiscarded": log_discarded, "logEofObserved": log_eof,
            "namedBytesObserved": named_bytes, "maximumNamedBytes": policy.export.maximum_named_bytes,
            "artifacts": artifacts, "evidenceErrors": evidence_errors, "supervisionErrors": failures,
            "snapshotValidation": "deferred; hashes alone do not admit structure, provenance, numerical completion or path correctness"}


if __name__ == "__main__":
    if len(sys.argv) < 5 or sys.argv[1] != "--exec-worker" or sys.argv[3] != "--":
        raise SystemExit("Internal trusted worker exec only")
    _child_exec(json.loads(sys.argv[2]), sys.argv[4:])
