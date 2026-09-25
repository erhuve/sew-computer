"""Map admitted raw trials to inputs for separate mechanical response audits.

No mechanics, source/model authentication or physical-path proof is performed
here. Every valid returned trial needs a separate response audit, including
discarded coarse/fine trials and valid tails without an assessment. Only the
transaction-bound fine pairs belong to the committed physical trajectory.

Admission and planning read the same externally identified artifact twice,
sequentially. Whole-object decoding remains bounded by external process limits;
this is not a streaming or constant-memory reader. Returned dictionaries are
owned mutable observations, not immutable capabilities or resume inputs.
"""
from dataclasses import dataclass

from solver_temporal_evidence import audit_snapshot, read_snapshot, require, same


PROFILE = "retained-temporal-trial-plan-v1"


@dataclass(frozen=True)
class TrialInput:
    assessment_id: int | None
    committed: bool
    start_state: dict
    trial: dict


@dataclass(frozen=True)
class TrialPlan:
    admission: dict
    context: dict
    trials: tuple[TrialInput, ...]
    summary: dict


def read_trial_plan(path, *, expected_bytes, expected_sha256, expected_run_identity,
                    expected_context_sha256, evaluator_profile, limits):
    """Admit a complete raw envelope, then map all valid returned trial inputs.

    Expectations must come from the frozen external run/artifact declaration.
    An unavailable snapshot raises rather than supplying an empty valid path.
    Invalid or unreturned records remain outside the response candidate list,
    even if they contain state or convergence diagnostics. This function does
    not call a solver and never interprets an outcome as physical acceptance.
    """
    admission = audit_snapshot(path, expected_bytes=expected_bytes,
        expected_sha256=expected_sha256, expected_run_identity=expected_run_identity,
        expected_context_sha256=expected_context_sha256,
        evaluator_profile=evaluator_profile, limits=limits)
    require(admission["available"], "Unavailable snapshot has no response audit inputs")
    envelope = read_snapshot(path, expected_bytes=expected_bytes,
                             expected_sha256=expected_sha256, limits=limits)
    snapshot = envelope["snapshot"]
    context = snapshot["context"]
    attempts = snapshot["attempts"]
    reservations = snapshot["reservations"]
    transactions = {row["transactionId"]: row for row in snapshot["transactions"]}
    current = context["initialState"]
    cursor = 0
    inputs = []
    skipped = []
    committed_ids = []

    def group(records, assessment_id):
        nonlocal current
        transaction = transactions.get(assessment_id)
        fine_ids = transaction["fineTrialIds"] if transaction is not None else []
        first_fine = None
        for offset, trial in enumerate(records):
            require(offset < 3, "At most one coarse/fine pair per assessment")
            require(trial["role"] == ("coarse", "fine-first", "fine-second")[offset],
                    "Trial roles do not form a coarse/fine group")
            start = first_fine if offset == 2 else current
            require(start is not None, "Second fine trial lacks a valid first fine state")
            same(trial["startStateSha256"], start["sha256"])
            identifier = trial["attemptId"]
            returned = reservations[identifier-1]["observation"] == "return-observed"
            if trial["numericallyValid"]:
                require(returned, "Response input needs an observed callback return")
                inputs.append(TrialInput(assessment_id, identifier in fine_ids, start, trial))
                if offset == 1:
                    first_fine = trial["state"]
            else:
                skipped.append(identifier)
        if transaction is not None:
            require(len(records) == 3 and fine_ids == [r["attemptId"] for r in records[1:]],
                    "A committed transaction needs both fine trials")
            require(all(r["numericallyValid"] for r in records), "Invalid committed group")
            current = records[2]["state"]
            committed_ids.extend(fine_ids)

    for assessment in snapshot["temporalAssessments"]:
        identifiers = assessment["trialIds"]
        same(identifiers, list(range(cursor+1, cursor+len(identifiers)+1)))
        group(attempts[cursor:cursor+len(identifiers)], assessment["assessmentId"])
        cursor += len(identifiers)
    # An interrupt can retain complete valid returns before publishing their
    # assessment. They still need response audits, but cannot advance state.
    if cursor < len(attempts):
        group(attempts[cursor:], None)
    same(current, snapshot["state"])
    same(committed_ids, [row["attemptId"] for row in snapshot["acceptedSteps"]])
    require(len(inputs)+len(skipped) == len(attempts), "Every recorded trial needs a disposition")
    summary = {"profile": PROFILE, "accepted": False,
        "artifact": admission["artifact"], "runIdentity": expected_run_identity,
        "contextSha256": expected_context_sha256,
        "responseCandidateTrialIds": [row.trial["attemptId"] for row in inputs],
        "committedFineTrialIds": committed_ids,
        "uncommittedResponseTrialIds": [row.trial["attemptId"] for row in inputs if not row.committed],
        "unassessedResponseTrialIds": [row.trial["attemptId"] for row in inputs if row.assessment_id is None],
        "invalidRecordedTrialIds": skipped,
        "unrecordedReservations": len(reservations)-len(attempts),
        "scope": "Input mapping only; source/control/response/primitive/force/path truth remains a separate audit."}
    return TrialPlan(admission, context, tuple(inputs), summary)
