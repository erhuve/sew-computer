"""Bounded immutable attempt evidence; recovery never advances past an invalid record."""

import copy
import json
import math
import numbers
from pathlib import Path
import re

from solver_process_budget import atomic_json, json_bytes, read_regular, sha256


PROFILE = "contact-attempt-journal-v1"
NAME = re.compile(r"attempt-[0-9]{6}\.json")
MAX_RECORD_BYTES = 1024 ** 2
MAX_JOURNAL_BYTES = 16 * 1024 ** 2
RECOVERY_RESERVE_BYTES = 8192
IDENTITY_KEYS = ("attemptId", "parentAttemptId", "initialInterval", "startFraction", "endFraction",
                 "durationSeconds", "depth", "lastAcceptedState")
PROVENANCE_KEYS = ("sourceDigests", "canonicalSha256", "placementSha256", "arguments")


def strict_loads(content):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    def finite(value):
        number = float(value)
        return number if math.isfinite(number) else invalid(value)

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(content, parse_constant=invalid, parse_float=finite, object_pairs_hook=unique)


def diagnostic_json(value, path="/step"):
    nonfinite = []
    nodes = 0

    def visit(item, pointer, depth):
        nonlocal nodes
        nodes += 1
        if nodes > 200000 or depth > 24:
            raise ValueError("Diagnostic structure budget exhausted")
        if item is None or type(item) in (str, bool):
            return item
        if isinstance(item, numbers.Integral):
            return int(item)
        if isinstance(item, numbers.Real):
            value = float(item)
            if math.isfinite(value):
                return value
            tag = {"$nonfinite": "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity"),
                   "path": pointer}
            nonfinite.append(tag)
            return tag
        if isinstance(item, dict):
            if "$nonfinite" in item or any(not isinstance(key, str) for key in item):
                raise ValueError("Diagnostic keys must be strings without reserved nonfinite tags")
            return {key: visit(child, pointer + "/" + key.replace("~", "~0").replace("/", "~1"), depth + 1)
                    for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(child, pointer + f"/{index}", depth + 1) for index, child in enumerate(item)]
        if type(item).__module__.startswith("numpy") and hasattr(item, "tolist"):
            return visit(item.tolist(), pointer, depth + 1)
        raise ValueError(f"Unsupported diagnostic type: {type(item).__name__}")

    result = visit(value, path, 0)
    if len(json_bytes([result, nonfinite])) > MAX_RECORD_BYTES // 2:
        raise ValueError("Diagnostic byte budget exhausted")
    return result, nonfinite


def same(left, right):
    return json_bytes(left) == json_bytes(right)


def validate_diagnostics(record):
    observed = []
    nodes = 0

    def walk(value, path, depth=0):
        nonlocal nodes
        nodes += 1
        if depth > 24 or nodes > 200000:
            raise ValueError("Diagnostic structure budget exhausted")
        if isinstance(value, dict):
            if "$nonfinite" in value:
                if (set(value) != {"$nonfinite", "path"} or value["path"] != path
                        or value["$nonfinite"] not in ("NaN", "+Infinity", "-Infinity")):
                    raise ValueError("Malformed nonfinite diagnostic tag")
                observed.append(value)
            else:
                for key, child in value.items():
                    walk(child, path + "/" + key.replace("~", "~0").replace("/", "~1"), depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}/{index}", depth + 1)

    if "step" in record and not isinstance(record["step"], dict):
        raise ValueError("Step diagnostics must be an object")
    walk(record.get("step"), "/step")
    declared = record.get("nonfiniteDiagnostics", [])
    if not isinstance(declared, list):
        raise ValueError("Invalid nonfinite diagnostic paths")
    for tag in declared:
        if (not isinstance(tag, dict) or set(tag) != {"$nonfinite", "path"}
                or not isinstance(tag["path"], str)
                or tag["$nonfinite"] not in ("NaN", "+Infinity", "-Infinity")):
            raise ValueError("Malformed nonfinite diagnostic path")
    if len({tag["path"] for tag in declared}) != len(declared):
        raise ValueError("Duplicate nonfinite diagnostic path")
    declared_step = [tag for tag in declared if tag["path"].startswith("/step/")]
    if (sorted(declared_step, key=lambda tag: tag["path"]) != sorted(observed, key=lambda tag: tag["path"])
            or any(tag not in declared_step and not re.fullmatch(
                r"/candidate(?:Positions|Velocities)(?:/[0-9]+)*", tag["path"]) for tag in declared)):
        raise ValueError("Nonfinite tags and paths do not match")
    if "error" in record and (not isinstance(record["error"], dict)
            or set(record["error"]) != {"type", "message"}
            or any(not isinstance(value, str) for value in record["error"].values())):
        raise ValueError("Malformed attempt exception")
    return declared


def configuration(dt, subdivisions, depth, attempts):
    if (type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0
            or type(subdivisions) is not int or type(depth) is not int or type(attempts) is not int
            or not 0 <= depth <= 30 or not 1 <= attempts <= 4096
            or not 1 <= subdivisions <= attempts or subdivisions & (subdivisions - 1)):
        raise ValueError("Invalid journal interval configuration")
    return {"requestedDurationSeconds": float(dt), "initialSubdivisions": subdivisions,
            "maxDepth": depth, "maxAttempts": attempts, "stationarityToleranceN": 1e-6}


def state_data(directory, descriptor, expected_name):
    if (not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}
            or descriptor["path"] != expected_name):
        raise ValueError("Invalid journal state descriptor")
    content = read_regular(directory / expected_name)
    if sha256(content) != descriptor["sha256"]:
        raise ValueError("Journal state hash mismatch")
    state = strict_loads(content)
    if state.get("accepted") is not False:
        raise ValueError("Journal states cannot grant garment acceptance")
    positions, velocities = state["positionsMeters"], state["velocitiesMetersPerSecond"]
    for values in (positions, velocities):
        if (not isinstance(values, list) or not 1 <= len(values) <= 25000
                or any(not isinstance(row, list) or len(row) != 3
                       or any(type(value) not in (float, int) or not math.isfinite(value) for value in row)
                       for row in values)):
            raise ValueError("Journal state requires bounded finite three-dimensional arrays")
    if len(positions) != len(velocities):
        raise ValueError("Journal state dimensions differ")
    return state


class AttemptJournal:
    def __init__(self, directory, report, positions, velocities, *, dt, initial_subdivisions,
                 max_depth, max_attempts):
        self.directory = Path(directory)
        self.sequence = -1
        self.previous = None
        self.byte_count = 0
        self.accepted_count = 0
        self.active = None
        self.finished = False
        self.config = configuration(dt, initial_subdivisions, max_depth, max_attempts)
        self.last_state = atomic_json(self.directory / "initial-state.json", {
            "positionsMeters": positions.tolist(), "velocitiesMetersPerSecond": velocities.tolist(),
            "accepted": False})
        self.append("header", {"profile": PROFILE, "configuration": self.config,
                              "provenance": {key: report[key] for key in PROVENANCE_KEYS},
                              "initialState": self.last_state})

    def append(self, kind, data):
        if self.sequence >= 2 * self.config["maxAttempts"] + 1:
            raise ValueError("Attempt journal event budget exhausted")
        payload = {"sequence": self.sequence + 1, "previousSha256": self.previous, "kind": kind, "data": data}
        envelope = {"payload": payload, "sha256": sha256(json_bytes(payload))}
        size = len(json_bytes(envelope))
        reserve = RECOVERY_RESERVE_BYTES + (MAX_RECORD_BYTES if kind == "start" else 0)
        if size > MAX_RECORD_BYTES or self.byte_count + size > MAX_JOURNAL_BYTES - reserve:
            raise ValueError("Attempt journal byte budget exhausted")
        descriptor = atomic_json(self.directory / f"attempt-{self.sequence + 1:06d}.json", envelope)
        self.sequence += 1
        self.previous = descriptor["sha256"]
        self.byte_count += size
        return descriptor

    def start(self, record):
        if self.active is not None or self.finished:
            raise ValueError("Attempt already active or journal finished")
        record["lastAcceptedState"] = copy.deepcopy(self.last_state)
        self.append("start", record)
        self.active = copy.deepcopy(record)

    def outcome(self, record, positions=None, velocities=None):
        if self.active is None or any(not same(record.get(key), self.active[key]) for key in IDENTITY_KEYS):
            raise ValueError("Outcome does not match active attempt")
        if record["outcome"] == "accepted":
            artifact = atomic_json(self.directory / f"accepted-{self.accepted_count + 1:04d}.json", {
                "positionsMeters": positions.tolist(), "velocitiesMetersPerSecond": velocities.tolist(),
                "record": record, "accepted": False})
            record["acceptedState"] = artifact
        self.append("outcome", record)
        if record["outcome"] == "accepted":
            self.last_state = artifact
            self.accepted_count += 1
        self.active = None

    def finish(self, reason, complete):
        if self.active is not None or self.finished:
            raise ValueError("Cannot finish active or finished journal")
        self.append("finished", {"reason": reason, "complete": complete})
        self.finished = True


def recover_attempt_journal(directory, report, *, interruption=None):
    directory = Path(directory)
    paths = sorted(directory.glob("attempt-*.json"))
    if not paths or len(paths) > 8194:
        raise ValueError("Missing or excessive attempt journal records")
    errors, events, attempts, artifacts = [], [], [], []
    active = None
    config = None
    initial = last_state = None
    completed = 0.
    next_initial = 0
    pending = []
    previous = None
    total_bytes = 0
    finished = None
    blocked = False
    initial_count = None
    for sequence, path in enumerate(paths):
        try:
            if not NAME.fullmatch(path.name) or path.name != f"attempt-{sequence:06d}.json":
                raise ValueError("Noncontiguous journal records")
            content = read_regular(path, MAX_RECORD_BYTES)
            total_bytes += len(content)
            if total_bytes > MAX_JOURNAL_BYTES:
                raise ValueError("Attempt journal byte budget exhausted")
            envelope = strict_loads(content)
            payload = envelope["payload"]
            if (type(payload["sequence"]) is not int or payload["sequence"] != sequence
                    or payload["previousSha256"] != previous
                    or envelope["sha256"] != sha256(json_bytes(payload))):
                raise ValueError("Journal sequence or hash-chain mismatch")
            kind, data = payload["kind"], payload["data"]
            if sequence == 0:
                if kind != "header" or data["profile"] != PROFILE:
                    raise ValueError("Invalid journal header")
                if not same(data["provenance"], {key: report[key] for key in PROVENANCE_KEYS}):
                    raise ValueError("Journal changed captured provenance")
                declared = data["configuration"]
                config = configuration(declared["requestedDurationSeconds"], declared["initialSubdivisions"],
                                       declared["maxDepth"], declared["maxAttempts"])
                if not same(declared, config):
                    raise ValueError("Journal changed stationarity tolerance")
                arguments = report["arguments"]
                for key, argument in (("requestedDurationSeconds", "step_seconds"),
                                      ("initialSubdivisions", "subdivisions"),
                                      ("maxDepth", "max_depth"), ("maxAttempts", "max_attempts")):
                    if not same(arguments.get(argument), config[key]):
                        raise ValueError("Journal configuration differs from captured arguments")
                initial = data["initialState"]
                initial_count = len(state_data(directory, initial, "initial-state.json")["positionsMeters"])
                last_state = initial
            elif finished is not None:
                raise ValueError("Journal contains records after finish")
            elif kind == "start":
                if active is not None or blocked or len(attempts) >= config["maxAttempts"]:
                    raise ValueError("Invalid overlapping or out-of-budget attempt")
                if not pending:
                    if next_initial >= config["initialSubdivisions"]:
                        raise ValueError("Attempt after interval completion")
                    index = next_initial
                    pending.append((index / config["initialSubdivisions"], (index + 1) / config["initialSubdivisions"],
                                    0, None, index))
                    next_initial += 1
                start, end, depth, parent, index = pending[-1]
                expected = {"attemptId": len(attempts) + 1, "parentAttemptId": parent, "initialInterval": index,
                            "startFraction": start, "endFraction": end, "durationSeconds":
                            config["requestedDurationSeconds"] * (end - start), "depth": depth,
                            "lastAcceptedState": last_state, "converged": False}
                if (not same(data, expected) or start != completed or expected["durationSeconds"] <= 0
                        or any(type(data[key]) is not int for key in ("attemptId", "initialInterval", "depth"))
                        or (parent is not None and type(data["parentAttemptId"]) is not int)):
                    raise ValueError("Attempt interval, lineage or last accepted state mismatch")
                active = copy.deepcopy(data)
                pending.pop()
            elif kind == "outcome":
                if active is None or any(not same(data.get(key), active[key]) for key in IDENTITY_KEYS):
                    raise ValueError("Outcome does not match active attempt")
                outcome = data.get("outcome")
                if outcome not in ("accepted", "rejected", "interrupted"):
                    raise ValueError("Invalid attempt outcome")
                if type(data.get("fatal")) is not bool or type(data.get("converged")) is not bool:
                    raise ValueError("Invalid attempt outcome flags")
                tags = validate_diagnostics(data)
                if outcome == "accepted":
                    step = data["step"]
                    residual = step["gradientInfinityNorm"]
                    if (data["converged"] is not True or data["fatal"] or step.get("converged") is not True
                            or type(residual) not in (float, int) or not 0 <= residual <= 1e-6
                            or tags or data.get("nonfiniteDiagnostics") or "error" in data
                            or not same(data["completedDurationSeconds"],
                                        config["requestedDurationSeconds"] * data["endFraction"])):
                        raise ValueError("Invalid accepted numerical outcome")
                    artifact = data["acceptedState"]
                    state = state_data(directory, artifact, f"accepted-{len(artifacts) + 1:04d}.json")
                    if (len(state["positionsMeters"]) != initial_count
                            or not same(state["record"], {key: value for key, value in data.items() if key != "acceptedState"})):
                        raise ValueError("Accepted state and journal outcome mismatch")
                    artifacts.append(artifact)
                    last_state = artifact
                    completed = data["endFraction"]
                else:
                    if data["converged"] or "acceptedState" in data or "completedDurationSeconds" in data:
                        raise ValueError("Rejected or interrupted outcome cannot advance state")
                    blocked = outcome == "interrupted" or data["fatal"] or data["depth"] == config["maxDepth"]
                    if not blocked:
                        start, end = data["startFraction"], data["endFraction"]
                        midpoint = .5 * (start + end)
                        pending.extend(((midpoint, end, data["depth"] + 1, data["attemptId"], data["initialInterval"]),
                                        (start, midpoint, data["depth"] + 1, data["attemptId"], data["initialInterval"])))
                attempts.append(copy.deepcopy(data))
                active = None
            elif kind == "finished":
                next_width = (pending[-1][1] - pending[-1][0]) if pending else 1 / config["initialSubdivisions"]
                expected_reason = ("complete" if completed == 1. else
                    "solver-resource-or-runtime-failure" if attempts and attempts[-1]["fatal"] else
                    "subdivision-depth-exhausted" if blocked else
                    "attempt-budget-exhausted" if len(attempts) == config["maxAttempts"] else
                    "substep-duration-underflow" if config["requestedDurationSeconds"] * next_width == 0 else None)
                if (active is not None or type(data.get("complete")) is not bool
                        or data["complete"] != (completed == 1.) or data.get("reason") != expected_reason
                        or expected_reason is None):
                    raise ValueError("Invalid journal finish")
                finished = data
            else:
                raise ValueError("Unknown journal record kind")
            previous = sha256(content)
            events.append({"path": path.name, "sha256": previous})
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError, OverflowError) as error:
            errors.append({"path": path.name, "message": str(error)})
            break
    if not events:
        raise ValueError(f"No intact attempt journal header: {errors}")
    if active is not None:
        record = dict(active, outcome="interrupted", converged=False, fatal=True,
                      error={"type": "InterruptedAttempt", "message": interruption or "No durable outcome"})
        if interruption is not None and not errors:
            payload = {"sequence": len(events), "previousSha256": previous, "kind": "outcome", "data": record}
            envelope = {"payload": payload, "sha256": sha256(json_bytes(payload))}
            try:
                if (len(events) >= 2 * config["maxAttempts"] + 2
                        or len(json_bytes(envelope)) > MAX_RECORD_BYTES
                        or total_bytes + len(json_bytes(envelope)) > MAX_JOURNAL_BYTES):
                    raise ValueError("Recovery journal budget exhausted")
                descriptor = atomic_json(directory / f"attempt-{len(events):06d}.json", envelope)
                events.append(descriptor)
            except (OSError, ValueError) as error:
                errors.append({"message": str(error)})
                record["recoveredWithoutDurableOutcome"] = True
        else:
            record["recoveredWithoutDurableOutcome"] = True
        attempts.append(record)
    adaptive = dict(config, profile="experimental-adaptive-contact-time-subdivision-v1", accepted=False,
                    complete=bool(finished and finished["complete"] and not errors),
                    reason=finished["reason"] if finished and not errors else "journal-incomplete-or-invalid",
                    completedFraction=completed, completedDurationSeconds=config["requestedDurationSeconds"] * completed,
                    attempts=attempts, acceptedSteps=[entry for entry in attempts if entry["outcome"] == "accepted"],
                    rejectedSteps=[entry for entry in attempts if entry["outcome"] == "rejected"],
                    interruptedSteps=[entry for entry in attempts if entry["outcome"] == "interrupted"],
                    targetInterpolation="linear over the original physical interval")
    arguments = report.get("arguments", {})
    if arguments.get("assembly_schedule") is True:
        adaptive["targetInterpolation"] = "captured piecewise-linear sewing/fold progress"
    elif arguments.get("material_grippers") is True or arguments.get("sewing_activation") is True:
        adaptive["targetInterpolation"] = "linear sewing progress over the original physical interval"
    if arguments.get("sewing_activation") is True:
        adaptive["sewingActivationInterpolation"] = (
            "captured monotone piecewise-linear canonical-row activation over the original physical interval")
    if arguments.get("material_grippers") is True:
        adaptive["gripperInterpolation"] = (
            "captured piecewise-linear material-point targets and activation over the original physical interval")
    return {"adaptive": adaptive, "acceptedStateArtifacts": artifacts,
            "attemptJournal": {"profile": PROFILE, "events": events, "initialState": initial,
                               "lastAcceptedState": last_state, "finished": finished is not None,
                               "errors": errors}}
