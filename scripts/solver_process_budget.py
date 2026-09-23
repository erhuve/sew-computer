"""Linux-only, trusted research process supervision and durable diagnostic evidence."""

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import stat
import subprocess
import tempfile
import time


MAX_FILE_BYTES = 64 * 1024 ** 2
MAX_CHECKPOINTS = 4128
STATE_NAME = re.compile(r"accepted-[0-9]{4}\.json|state\.json")
CHECKPOINT_NAME = re.compile(r"progress-[0-9]{6}\.json")


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def read_regular(path, limit=MAX_FILE_BYTES):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"Regular evidence file required: {Path(path).name}")
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError(f"Evidence file exceeds {limit} bytes: {Path(path).name}")
    return content


def atomic_bytes(path, content, *, replace=False):
    path = Path(path)
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("Evidence exceeds 64 MiB")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fchmod(stream.fileno(), 0o400)
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            os.unlink(temporary)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return {"path": path.name, "sha256": sha256(content)}


def atomic_json(path, value, *, replace=False):
    return atomic_bytes(path, json_bytes(value), replace=replace)


class ProgressStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        names = [path for path in self.directory.glob("progress-*.json") if CHECKPOINT_NAME.fullmatch(path.name)]
        self.sequence = max((int(path.stem.split("-")[1]) for path in names), default=0)

    def save(self, report, phase, *, exit_code=None):
        if report.get("accepted") is not False:
            raise ValueError("Research evidence must remain unaccepted")
        if self.sequence >= MAX_CHECKPOINTS:
            raise ValueError("Progress checkpoint budget exhausted")
        if exit_code is not None and (type(exit_code) is not int or not 0 <= exit_code <= 255):
            raise ValueError("Invalid worker exit code")
        self.sequence += 1
        payload = {"sequence": self.sequence,
                   "report": dict(report, accepted=False, terminal=False, completed=False), "phase": phase,
                   "workerFinished": exit_code is not None, "workerExitCode": exit_code}
        envelope = {"payload": payload, "sha256": sha256(json_bytes(payload))}
        descriptor = atomic_json(self.directory / f"progress-{self.sequence:06d}.json", envelope)
        atomic_json(self.directory / "progress.json", descriptor, replace=True)


def recover_progress(directory, *, expected_report=None):
    from solver_attempt_journal import strict_loads

    directory = Path(directory)
    paths = sorted((path for path in directory.glob("progress-*.json")
                    if CHECKPOINT_NAME.fullmatch(path.name)), reverse=True)
    if not paths or len(paths) > MAX_CHECKPOINTS:
        raise ValueError("Missing or excessive progress checkpoints")
    errors = []
    verified_artifacts = {}
    for path in paths:
        try:
            content = read_regular(path)
            envelope = strict_loads(content)
            payload = envelope["payload"]
            if envelope["sha256"] != sha256(json_bytes(payload)):
                raise ValueError("Progress payload hash mismatch")
            if (type(payload["sequence"]) is not int
                    or payload["sequence"] != int(path.stem.split("-")[1])):
                raise ValueError("Progress sequence mismatch")
            finished, exit_code = payload["workerFinished"], payload["workerExitCode"]
            if (type(finished) is not bool or not isinstance(payload["phase"], str)
                    or (finished and (type(exit_code) is not int or not 0 <= exit_code <= 255))
                    or (not finished and exit_code is not None)):
                raise ValueError("Invalid progress lifecycle")
            report = payload["report"]
            if any(report.get(key) is not False for key in ("accepted", "completed", "terminal")):
                raise ValueError("Research progress cannot grant terminal completion or acceptance")
            if expected_report is not None:
                for key in ("sourceDigests", "canonicalSha256", "placementSha256", "arguments"):
                    if json_bytes(report.get(key)) != json_bytes(expected_report.get(key)):
                        raise ValueError(f"Progress changed captured provenance: {key}")
            if "adaptive" in report and not isinstance(report["adaptive"], dict):
                raise ValueError("Invalid adaptive report")
            artifacts = report.get("acceptedStateArtifacts", [])
            if not isinstance(artifacts, list) or len(artifacts) > 4096:
                raise ValueError("Invalid accepted-state ledger")
            expected_names = [f"accepted-{index:04d}.json" for index in range(1, len(artifacts) + 1)]
            if [entry["path"] for entry in artifacts] != expected_names:
                raise ValueError("Noncontiguous accepted-state ledger")
            if "stateArtifact" in report and report["stateArtifact"]["path"] != "state.json":
                raise ValueError("Final state must reference state.json")
            descriptors = artifacts + ([report["stateArtifact"]] if "stateArtifact" in report else [])
            for entry in descriptors:
                if not STATE_NAME.fullmatch(entry["path"]):
                    raise ValueError("Invalid state evidence path")
                identity = (entry["path"], entry["sha256"])
                if identity not in verified_artifacts:
                    data = read_regular(directory / entry["path"])
                    verified_artifacts[identity] = (sha256(data) == entry["sha256"]
                                                     and strict_loads(data).get("accepted") is False)
                if not verified_artifacts[identity]:
                    raise ValueError(f"State evidence hash or acceptance mismatch: {entry['path']}")
            return payload, {"path": path.name, "sha256": sha256(content)}, errors
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError, OverflowError) as error:
            errors.append({"path": path.name, "message": str(error)})
    raise ValueError(f"No intact progress checkpoint: {errors}")


def captured_source_path(directory, name):
    """Resolve only flat research files or the captured engine dependency subtree."""
    if not isinstance(name, str):
        raise ValueError("Invalid captured source path")
    parts = name.split("/")
    python_name = re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.py", parts[-1]) is not None
    flat = len(parts) == 1 and (python_name or name in
        ("solver-contact.requirements.txt", "solver-spike.requirements.txt"))
    engine = len(parts) == 3 and parts[:2] == ["services", "engine"] and python_name
    if not (flat or engine):
        raise ValueError("Invalid captured source path")
    root = Path(directory) / "source-snapshot"
    path = root
    for part in parts[:-1]:
        if path.is_symlink():
            raise ValueError("Captured source directories cannot be symbolic links")
        path /= part
    if path.is_symlink():
        raise ValueError("Captured source directories cannot be symbolic links")
    return path / parts[-1]


def verify_capture(directory, report):
    directory = Path(directory)
    for name, key in (("canonical.json", "canonicalSha256"), ("placement.json", "placementSha256")):
        if sha256(read_regular(directory / name)) != report[key]:
            raise ValueError(f"Captured input hash mismatch: {name}")
    for name, digest in report["sourceDigests"].items():
        if sha256(read_regular(captured_source_path(directory, name))) != digest:
            raise ValueError(f"Captured source hash mismatch: {name}")


def _prctl(option, value):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(option, value, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl failed")


def arm_parent_death(parent_pid):
    _prctl(1, signal.SIGKILL)
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


def prepare_worker(cpu_limit_seconds, parent_pid):
    arm_parent_death(parent_pid)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_seconds, cpu_limit_seconds + 1))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 ** 3, 4 * 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def group_members(pgid):
    members = []
    ticks = os.sysconf("SC_CLK_TCK")
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == pgid:
            members.append({"pid": int(path.name), "state": fields[0],
                            "cpuSeconds": sum(int(fields[index]) for index in (11, 12, 13, 14)) / ticks})
    return members


def _kill_group(pgid, signum):
    try:
        os.killpg(pgid, signum)
        return True
    except ProcessLookupError:
        return False


def supervise(command, directory, initial_report, *, cpu_limit_seconds, wall_limit_seconds,
              terminate_grace_seconds=.25, cleanup_seconds=2., poll_seconds=.025):
    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        return _supervise(command, directory, initial_report, cpu_limit_seconds=cpu_limit_seconds,
                          wall_limit_seconds=wall_limit_seconds, terminate_grace_seconds=terminate_grace_seconds,
                          cleanup_seconds=cleanup_seconds, poll_seconds=poll_seconds)
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def _supervise(command, directory, initial_report, *, cpu_limit_seconds, wall_limit_seconds,
               terminate_grace_seconds, cleanup_seconds, poll_seconds):
    if (type(cpu_limit_seconds) is not int or not 1 <= cpu_limit_seconds <= 3600
            or not math.isfinite(wall_limit_seconds) or not 0 < wall_limit_seconds <= 7200
            or any(not math.isfinite(value) or not 0 < value <= 5
                   for value in (terminate_grace_seconds, cleanup_seconds, poll_seconds))):
        raise ValueError("Bounded positive CPU, wall and supervision intervals required")
    directory = Path(directory)
    started, cpu_started = time.monotonic(), time.process_time()
    received_signal = None
    old_subreaper = ctypes.c_int()
    subreaper_set = False
    process = None
    returncode = None
    cpu_user = cpu_system = sampled_cpu = 0.
    reason = None
    internal_error = None
    hard_kill = False
    cleanup_complete = True
    remaining = []

    def interrupted(signum, frame):
        nonlocal received_signal
        if received_signal is None:
            received_signal = signum

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, interrupted)
        _prctl(37, ctypes.byref(old_subreaper))
        _prctl(36, 1)
        subreaper_set = True
        if received_signal is not None:
            reason = "supervisor-interrupted"
        else:
            environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                               PYTHONDONTWRITEBYTECODE="1")
            process = subprocess.Popen(command, start_new_session=True, env=environment)
            while True:
                elapsed = time.monotonic() - started
                members = group_members(process.pid)
                sampled_cpu = max(sampled_cpu, sum(member["cpuSeconds"] for member in members))
                if received_signal is not None:
                    reason = "supervisor-interrupted"
                elif elapsed >= wall_limit_seconds:
                    reason = "wall-budget-exhausted"
                elif sampled_cpu >= cpu_limit_seconds:
                    reason = "cpu-budget-exhausted"
                if reason:
                    break
                status = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                if status is not None:
                    if any(member["pid"] != process.pid for member in members):
                        reason = "worker-left-descendants"
                    break
                time.sleep(min(poll_seconds, max(0., wall_limit_seconds - elapsed)))
    except BaseException as error:
        reason = "supervisor-failure"
        internal_error = {"type": type(error).__name__, "message": str(error)}
    finally:
        worker_wall = time.monotonic() - started
        if reason is None and worker_wall >= wall_limit_seconds:
            reason = "wall-budget-exhausted"
        if process is not None:
            # Keep the leader unreaped until group signalling ends, preventing PID/PGID reuse.
            try:
                _kill_group(process.pid, signal.SIGTERM)
                deadline = time.monotonic() + terminate_grace_seconds
                while time.monotonic() < deadline:
                    members = group_members(process.pid)
                    if any(member["pid"] != process.pid for member in members):
                        reason = reason or "worker-left-descendants"
                    if not any(member["state"] not in ("Z", "X") for member in members):
                        break
                    time.sleep(poll_seconds)
                if any(member["state"] not in ("Z", "X") for member in group_members(process.pid)):
                    hard_kill = _kill_group(process.pid, signal.SIGKILL)
            except BaseException as error:
                hard_kill = _kill_group(process.pid, signal.SIGKILL)
                internal_error = {"type": type(error).__name__, "message": str(error)}
                reason = "supervisor-cleanup-failure"
            deadline = time.monotonic() + cleanup_seconds
            while True:
                no_children = False
                while True:
                    try:
                        pid, status, usage = os.wait4(-process.pid, os.WNOHANG)
                    except ChildProcessError:
                        no_children = True
                        break
                    if pid == 0:
                        break
                    cpu_user += usage.ru_utime
                    cpu_system += usage.ru_stime
                    if pid == process.pid:
                        returncode = os.waitstatus_to_exitcode(status)
                        process.returncode = returncode
                try:
                    remaining = group_members(process.pid)
                except OSError as error:
                    remaining = [{"inspectionError": str(error)}]
                if not remaining or no_children or time.monotonic() >= deadline:
                    break
                time.sleep(poll_seconds)
            cleanup_complete = not remaining and returncode is not None
        if subreaper_set:
            try:
                _prctl(36, old_subreaper.value)
            except OSError as error:
                reason = reason or "supervisor-cleanup-failure"
                internal_error = {"type": type(error).__name__, "message": str(error)}

    report = dict(initial_report)
    recovery_errors = []
    try:
        payload, checkpoint, recovery_errors = recover_progress(directory, expected_report=initial_report)
        report = payload["report"]
        report["progressCheckpoint"] = checkpoint
        finished = payload["workerFinished"] is True and payload["workerExitCode"] == returncode
        report["lastWorkerPhase"] = payload["phase"]
        if recovery_errors:
            reason = reason or "evidence-recovery-failure"
    except (OSError, ValueError, KeyError, TypeError) as error:
        finished = False
        reason = reason or "evidence-recovery-failure"
        recovery_errors.append({"message": str(error)})
    capture_verified = False
    try:
        verify_capture(directory, initial_report)
        capture_verified = True
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        reason = reason or "capture-integrity-failure"
        recovery_errors.append({"message": str(error)})
    if received_signal is not None:
        reason = "supervisor-interrupted"
    if cpu_user + cpu_system >= cpu_limit_seconds:
        reason = reason or "cpu-budget-exhausted"
    if not cleanup_complete:
        reason = reason or "process-group-cleanup-failure"
    if returncode is not None and returncode < 0:
        reason = reason or "worker-signal"
    elif returncode not in (0, 1):
        reason = reason or "worker-exit"
    if not finished:
        reason = reason or "worker-incomplete"
    if report.get("attemptJournalRequired") or any(directory.glob("attempt-*.json")):
        from solver_attempt_journal import recover_attempt_journal, state_data

        try:
            recovered = recover_attempt_journal(directory, initial_report,
                interruption=(reason or "Worker ended without a durable attempt outcome") if cleanup_complete else None)
            journal_errors = recovered["attemptJournal"]["errors"]
            if finished and "failure" not in report:
                for key in ("adaptive", "acceptedStateArtifacts"):
                    if json_bytes(report.get(key)) != json_bytes(recovered[key]):
                        journal_errors.append({"message": f"Finished progress differs from attempt journal: {key}"})
            report.update(recovered)
            if "stateArtifact" in report:
                try:
                    state = state_data(directory, report["stateArtifact"], "state.json")
                    last = report["attemptJournal"]["lastAcceptedState"]
                    accepted_state = state_data(directory, last, last["path"])
                    if (state["positionsMeters"] != accepted_state["positionsMeters"]
                            or state["velocitiesMetersPerSecond"] != accepted_state["velocitiesMetersPerSecond"]
                            or json_bytes(state["completedDurationSeconds"]) !=
                               json_bytes(report["adaptive"]["completedDurationSeconds"])):
                        raise ValueError("Final state differs from the journal accepted prefix")
                except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError, OverflowError) as error:
                    journal_errors.append({"message": str(error)})
                    report.pop("stateArtifact", None)
            recovery_errors.extend(journal_errors)
            if journal_errors or not recovered["attemptJournal"]["finished"]:
                reason = reason or "attempt-journal-incomplete-or-invalid"
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError, OverflowError) as error:
            reason = reason or "attempt-journal-recovery-failure"
            recovery_errors.append({"message": str(error)})
            for key in ("adaptive", "stateArtifact", "attemptJournal"):
                report.pop(key, None)
            report["acceptedStateArtifacts"] = []
    if reason:
        if "failure" in report:
            report["workerFailure"] = report["failure"]
        report["failure"] = {"type": "ProcessBudgetFailure", "message": reason}
        report["classification"] = "uncompleted-research-diagnostic"
        if "adaptive" in report:
            report["adaptive"] = dict(report["adaptive"], complete=False,
                                      reason="supervision-interrupted-or-invalid")
    complete = (reason is None and returncode == 0 and finished and "failure" not in report
                and report.get("adaptive", {}).get("complete") is True
                and report.get("classification") == "completed-research-interval")
    if not complete and (report.get("classification") == "completed-research-interval"
                         or report.get("adaptive", {}).get("complete") is True):
        reason = reason or "inconsistent-worker-completion"
        report["classification"] = "uncompleted-research-diagnostic"
        report["failure"] = {"type": "ProcessBudgetFailure", "message": reason}
        if "adaptive" in report:
            report["adaptive"] = dict(report["adaptive"], complete=False, reason=reason)
    referenced = {entry["path"] for entry in report.get("acceptedStateArtifacts", [])}
    if "stateArtifact" in report:
        referenced.add(report["stateArtifact"]["path"])
    report["uncommittedStateArtifacts"] = [path.name for path in sorted(directory.iterdir())
                                            if STATE_NAME.fullmatch(path.name) and path.name not in referenced]
    report.update({"accepted": False, "terminal": True, "completed": complete,
                   "wallSeconds": time.monotonic() - started, "workerBudgetWallSeconds": worker_wall,
                   "captureIntegrityVerified": capture_verified,
                   "cpuSeconds": cpu_user + cpu_system,
                   "supervisorCpuSeconds": time.process_time() - cpu_started,
                   "supervision": {"reason": reason or "worker-exited", "workerReturnCode": returncode,
                       "workerSignal": -returncode if returncode is not None and returncode < 0 else None,
                       "supervisorSignal": received_signal, "hardKillSent": hard_kill,
                       "processGroupId": process.pid if process else None,
                       "cleanupComplete": cleanup_complete, "remainingGroupMembers": remaining,
                       "cpuUserSeconds": cpu_user, "cpuSystemSeconds": cpu_system,
                       "sampledGroupCpuSeconds": sampled_cpu,
                       "cpuAccounting": "wait4: worker plus reaped same-group descendants, not supervisor CPU",
                       "wallAccounting": "supervisor monotonic time from launch setup through cleanup and recovery",
                       "cpuLimitSeconds": cpu_limit_seconds, "wallLimitSeconds": wall_limit_seconds,
                       "terminateGraceSeconds": terminate_grace_seconds, "cleanupLimitSeconds": cleanup_seconds},
                   "recoveryErrors": recovery_errors})
    if internal_error:
        report["supervision"]["error"] = internal_error
    while True:
        publication_signal = received_signal
        if publication_signal is not None:
            report.update({"completed": False, "classification": "uncompleted-research-diagnostic",
                           "failure": {"type": "ProcessBudgetFailure", "message": "supervisor-interrupted"}})
            report["supervision"].update({"reason": "supervisor-interrupted",
                                          "supervisorSignal": publication_signal})
            if "adaptive" in report:
                report["adaptive"] = dict(report["adaptive"], complete=False,
                                          reason="supervision-interrupted-or-invalid")
        # Recovery or a late publication signal can shorten/invalidate the
        # durable prefix. Publish no cached worker control aggregate for an
        # incomplete run; replay derives its totals from accepted transitions.
        if not report["completed"]:
            report.pop("gripperWorkSummary", None)
            report.pop("sewingWorkSummary", None)
        atomic_json(directory / "report.json", report, replace=True)
        if publication_signal == received_signal:
            break
    return report
