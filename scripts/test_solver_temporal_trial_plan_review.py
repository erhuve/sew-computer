"""Independent raw-to-trial mapping attacks using hand-authored observations.

No controller, solver or native model constructs these fixtures. Branch states
and expected starts are deliberately different; their positions and velocities
do not obey a mechanics law. The adapter may establish mappings, not physics.
"""
import copy
from dataclasses import replace
from fractions import Fraction as F
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import solver_temporal_trial_plan as planner
from solver_temporal_evidence import AuditLimits
from test_solver_temporal_evidence_review import (
    EVALUATOR, MOTION, PUBLIC_SCALARS, RUN, context_sha,
    empty_available, ratio, serialized, sha, state, unavailable,
)


def at(x):
    return state([[float(x), 0., 0.]], [[0., 0., 0.]])


def zero_energy():
    terms = dict.fromkeys(MOTION, 0.)
    return {**dict.fromkeys(PUBLIC_SCALARS, 0.), **terms, 'accepted': False,
            'temporalMotion': {'termsJoules': terms.copy(), 'knownErrorBoundJoules': ratio()}}


def rational(value):
    value = F(value)
    return ratio(value.numerator, value.denominator)


def exact_metric(first, second, tolerance):
    # Independent one-vertex arithmetic; our chosen differences are 0 or 8,
    # so the approximate square root is exact as well.
    squared = (F(first)-F(second))**2
    return {'maximumSquared': rational(squared),
            'toleranceSquared': rational(F(tolerance)**2),
            'maximumApproximate': math.sqrt(float(squared)), 'maximumVertex': 0,
            'withinThreshold': squared <= F(tolerance)**2}


class WireFixture:
    """Small declared scheduler graph, independent of producer orchestration."""
    def __init__(self, *, subdivisions=1, depth=1, tolerance=16.):
        self.envelope = empty_available()
        self.raw = self.envelope['snapshot']
        self.context = self.raw['context']
        self.context.update(maxAttempts=16, maxDepth=depth, initialSubdivisions=subdivisions)
        self.context['policy'].update(maxEvaluationBudget=16,
                                      positionToleranceM=float(tolerance), velocityToleranceMPerS=1.)
        self.current = self.context['initialState']
        self.fraction = 0.

    def group(self, left, right, values, *, depth=0, parent=None, initial=0,
              assessment=True, commit=False, invalid_last=False, fatal_last=False):
        """Author a root/child observation from an explicit list of endpoints."""
        raw = self.raw
        assessment_id = len(raw['temporalAssessments'])+1
        first_id = len(raw['attempts'])+1
        midpoint = (left+right)/2
        first_fine = None
        rows = []
        for offset, value in enumerate(values):
            a, b = ((left, right), (left, midpoint), (midpoint, right))[offset]
            start = first_fine if offset == 2 else self.current
            ident = first_id+offset
            reservation = {'attemptId': ident, 'parentAttemptId': parent if offset == 0 else first_id,
                'initialInterval': initial, 'startFraction': float(a), 'endFraction': float(b),
                'durationSeconds': float(b-a), 'depth': depth+(offset != 0),
                'role': ('coarse', 'fine-first', 'fine-second')[offset], 'converged': False,
                'evaluationAllowanceCharged': 1, 'startStateSha256': start['sha256']}
            last = offset == len(values)-1
            valid = not (invalid_last and last)
            row = {**reservation, 'numericallyValid': valid, 'fatal': bool(fatal_last and last),
                   'outcome': 'provisional' if valid else 'numerically-rejected', 'state': at(value),
                   'step': {'evaluations': 0, 'energyBalance': zero_energy()}}
            if not valid:
                # Plausible diagnostic data must not imply a valid response.
                row['converged'] = True
            raw['reservations'].append({'trial': copy.deepcopy(reservation),
                'observation': 'return-observed', 'recordPublished': True})
            raw['attempts'].append(row)
            rows.append(row)
            if offset == 1: first_fine = row['state']
        if assessment:
            assessment_row = {'assessmentId': assessment_id, 'startFraction': float(left),
                'endFraction': float(right), 'depth': depth,
                'trialIds': [row['attemptId'] for row in rows], 'outcome': 'incomplete'}
            if len(rows) == 3 and not invalid_last:
                position = exact_metric(values[0], values[2], self.context['policy']['positionToleranceM'])
                velocity = exact_metric(0., 0., 1.)
                energies = [{'exactStoredKineticChangeJoules': ratio(), 'nominalDefectJoules': ratio(),
                    'knownErrorBoundJoules': ratio(), 'absoluteUpperBoundJoules': ratio(),
                    'allocationJoules': rational(F(row['endFraction'])-F(row['startFraction'])),
                    'outcome': 'within-budget'} for row in rows[1:]]
                assessment_row.update(position=position, velocity=velocity, fineEnergy=energies,
                    outcome='indicators-passed-awaiting-commit' if position['withinThreshold'] else 'temporally-rejected')
            elif invalid_last:
                assessment_row['outcome'] = 'numerically-rejected'
            raw['temporalAssessments'].append(assessment_row)
        if commit:
            assert assessment and len(rows) == 3
            raw['transactions'].append({'transactionId': assessment_id,
                'fineTrialIds': [row['attemptId'] for row in rows[1:]],
                'startFraction': float(left), 'endFraction': float(right)})
            for row in rows[1:]:
                committed = copy.deepcopy(row)
                committed.update(outcome='committed', transactionId=assessment_id,
                                 completedDurationSeconds=row['endFraction'])
                raw['acceptedSteps'].append(committed)
            self.current, self.fraction = rows[-1]['state'], float(right)
        return rows

    def finish(self, reason='interrupted'):
        complete = self.fraction == 1.
        stopped = reason == 'interrupted'
        self.raw.update(state=copy.deepcopy(self.current), completedFraction=self.fraction,
            complete=complete, controllerReason=reason,
            phase='report-ready' if reason == 'complete' else 'controller-finished',
            reportReady=reason == 'complete', stopRequested=stopped,
            stopReason='requested' if stopped else None,
            resources={'reservedTrials': len(self.raw['reservations']),
                       'recordedTrials': len(self.raw['attempts']),
                       'chargedEvaluationAllowance': len(self.raw['reservations'])})
        return self.envelope


def whole_pair():
    fixture = WireFixture()
    fixture.group(0., 1., [10., 1., 2.], commit=True)
    return fixture.finish('complete')


class TemporalTrialPlanIndependentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'snapshot.json'
        self.limits = AuditLimits(input_bytes=400_000, nesting_depth=32, json_values=40_000,
            string_characters=10_000, vertices=8, trials=16, assessments=16)

    def write(self, envelope):
        raw = serialized(envelope)
        self.path.write_bytes(raw)
        context = envelope['snapshot'].get('context')
        return {'expected_bytes': len(raw), 'expected_sha256': sha(raw),
                'expected_run_identity': RUN,
                'expected_context_sha256': context_sha(context) if context is not None else None,
                'evaluator_profile': EVALUATOR, 'limits': self.limits}

    def plan(self, envelope, **options):
        arguments = self.write(envelope)
        arguments.update(options)
        return planner.read_trial_plan(self.path, **arguments)

    def starts(self, plan):
        return [row.start_state['positionsMeters'][0][0] for row in plan.trials]

    def test_branch_distinct_full_pair_does_not_follow_previous_attempt(self):
        plan = self.plan(whole_pair())
        self.assertEqual(self.starts(plan), [0., 0., 1.])
        self.assertEqual([row.trial['state']['positionsMeters'][0][0] for row in plan.trials], [10., 1., 2.])
        self.assertEqual([row.committed for row in plan.trials], [False, True, True])
        self.assertEqual(plan.summary['committedFineTrialIds'], [2, 3])
        self.assertEqual(plan.summary['uncommittedResponseTrialIds'], [1])
        self.assertTrue(all(row.trial['outcome'] == 'provisional' for row in plan.trials))

    def test_rejected_parent_then_children_use_actual_committed_midpoint(self):
        fixture = WireFixture(depth=2, tolerance=1.)
        fixture.group(0., 1., [10., 1., 2.])
        fixture.group(0., .5, [3., 1.5, 3.], depth=1, parent=1, commit=True)
        fixture.group(.5, 1., [5., 4., 5.], depth=1, parent=1, commit=True)
        plan = self.plan(fixture.finish('complete'))
        # These expected starts are specified directly, not extracted from IDs,
        # recorded hashes or a producer-generated trace.
        self.assertEqual(self.starts(plan), [0., 0., 1., 0., 0., 1.5, 3., 3., 4.])
        self.assertEqual(plan.summary['responseCandidateTrialIds'], list(range(1, 10)))
        self.assertEqual(plan.summary['committedFineTrialIds'], [5, 6, 8, 9])
        self.assertEqual(plan.summary['uncommittedResponseTrialIds'], [1, 2, 3, 4, 7])
        self.assertEqual([row.assessment_id for row in plan.trials], [1]*3+[2]*3+[3]*3)

    def test_passing_assessment_without_transaction_never_advances_path(self):
        fixture = WireFixture()
        fixture.group(0., 1., [10., 1., 2.])
        plan = self.plan(fixture.finish())
        self.assertEqual(self.starts(plan), [0., 0., 1.])
        self.assertEqual(plan.admission['completedFraction'], 0.)
        self.assertEqual(plan.summary['committedFineTrialIds'], [])
        self.assertEqual(plan.summary['unassessedResponseTrialIds'], [])
        self.assertTrue(all(not row.committed and row.assessment_id == 1 for row in plan.trials))

    def test_every_unassessed_valid_tail_length_is_retained_without_commit(self):
        for length, expected in ((1, [0.]), (2, [0., 0.]), (3, [0., 0., 1.])):
            with self.subTest(length=length):
                fixture = WireFixture()
                fixture.group(0., 1., [10., 1., 2.][:length], assessment=False)
                plan = self.plan(fixture.finish())
                self.assertEqual(self.starts(plan), expected)
                self.assertEqual(plan.summary['unassessedResponseTrialIds'], list(range(1, length+1)))
                self.assertEqual(plan.summary['committedFineTrialIds'], [])
                self.assertTrue(all(row.assessment_id is None and not row.committed for row in plan.trials))
                self.assertFalse(plan.admission['numericalPrefixComplete'])

    def test_tail_after_committed_pair_starts_at_committed_state(self):
        fixture = WireFixture(subdivisions=2)
        fixture.group(0., .5, [10., 1., 2.], commit=True)
        fixture.group(.5, 1., [20., 3., 4.], initial=1, assessment=False)
        plan = self.plan(fixture.finish())
        self.assertEqual(self.starts(plan), [0., 0., 1., 2., 2., 3.])
        self.assertEqual(plan.summary['committedFineTrialIds'], [2, 3])
        self.assertEqual(plan.summary['unassessedResponseTrialIds'], [4, 5, 6])
        self.assertEqual(plan.admission['completedFraction'], .5)

    def test_final_unrecorded_reservation_is_not_a_response_candidate(self):
        for observation in ('reserved', 'dispatch-authorized', 'return-observed'):
            with self.subTest(observation=observation):
                fixture = WireFixture()
                fixture.group(0., 1., [10., 1., 2.], assessment=False)
                fixture.raw['attempts'].pop()
                last = fixture.raw['reservations'][-1]
                del last['recordPublished']
                last['observation'] = observation
                plan = self.plan(fixture.finish())
                self.assertEqual(self.starts(plan), [0., 0.])
                self.assertEqual(plan.summary['responseCandidateTrialIds'], [1, 2])
                self.assertEqual(plan.summary['unrecordedReservations'], 1)
                self.assertEqual(plan.admission['chargedEvaluationAllowance'], 3)

    def test_invalid_diagnostic_state_is_excluded_at_each_role(self):
        for length in (1, 2, 3):
            with self.subTest(length=length):
                fixture = WireFixture()
                fixture.group(0., 1., [10., 1., 2.][:length], invalid_last=True)
                plan = self.plan(fixture.finish('temporal-depth-exhausted'))
                self.assertEqual(plan.summary['invalidRecordedTrialIds'], [length])
                self.assertEqual(plan.summary['responseCandidateTrialIds'], list(range(1, length)))
                self.assertEqual(plan.summary['committedFineTrialIds'], [])

    def test_fatal_invalid_and_fatal_valid_observations_remain_distinct(self):
        invalid = WireFixture()
        invalid.group(0., 1., [10.], invalid_last=True, fatal_last=True)
        invalid.raw['temporalAssessments'][0]['outcome'] = 'incomplete'
        plan = self.plan(invalid.finish('solver-resource-or-runtime-failure'))
        self.assertEqual(plan.summary['invalidRecordedTrialIds'], [1])
        self.assertEqual(plan.trials, ())
        valid = WireFixture()
        valid.group(0., 1., [10.], fatal_last=True)
        plan = self.plan(valid.finish('solver-resource-or-runtime-failure'))
        self.assertEqual(len(plan.trials), 1)
        self.assertTrue(plan.trials[0].trial['fatal'])
        self.assertFalse(plan.trials[0].committed)
        self.assertEqual(plan.summary['invalidRecordedTrialIds'], [])

    def test_identical_states_at_distinct_fractions_are_not_deduplicated(self):
        fixture = WireFixture(subdivisions=2)
        fixture.group(0., .5, [0., 0., 0.], commit=True)
        fixture.group(.5, 1., [0., 0., 0.], initial=1, commit=True)
        plan = self.plan(fixture.finish('complete'))
        self.assertEqual(len(plan.trials), 6)
        self.assertEqual({row.start_state['sha256'] for row in plan.trials}, {at(0.)['sha256']})
        self.assertEqual([(row.trial['startFraction'], row.trial['endFraction']) for row in plan.trials],
                         [(0., .5), (0., .25), (.25, .5), (.5, 1.), (.5, .75), (.75, 1.)])
        self.assertEqual(plan.summary['committedFineTrialIds'], [2, 3, 5, 6])

    def test_nonphysics_callback_values_are_not_mislabeled_as_mechanics(self):
        plan = self.plan(whole_pair())
        self.assertTrue(plan.admission['numericalPrefixComplete'])
        self.assertTrue(all(row.trial['converged'] is False for row in plan.trials))
        self.assertTrue(all(row.trial['state']['velocitiesMPerS'] == [[0., 0., 0.]] for row in plan.trials))
        self.assertFalse(plan.summary['accepted'])
        self.assertIn('source/control/response/primitive/force/path', plan.summary['scope'])
        self.assertNotIn('physicsVerified', plan.summary)
        self.assertNotIn('processExitSucceeded', plan.summary)

    def test_unavailable_and_available_empty_prefix_are_distinct(self):
        with patch.object(planner, 'read_snapshot', wraps=planner.read_snapshot) as second_read:
            with self.assertRaises(ValueError): self.plan(unavailable())
            second_read.assert_not_called()
        plan = self.plan(empty_available())
        self.assertEqual(plan.trials, ())
        self.assertFalse(plan.admission['numericalPrefixComplete'])
        self.assertEqual(plan.summary['committedFineTrialIds'], [])

    def test_full_commit_survives_later_failure_without_process_success_claim(self):
        envelope = whole_pair()
        raw = envelope['snapshot']
        raw.update(controllerReason='interrupted', stopRequested=True, stopReason='requested', enrichment='failed')
        raw['failures'] = [{'type': 'MemoryError', 'message': 'after final commit', 'stage': 'enrichment'}]
        plan = self.plan(envelope)
        self.assertTrue(plan.admission['numericalPrefixComplete'])
        self.assertEqual(plan.summary['committedFineTrialIds'], [2, 3])
        self.assertFalse(plan.summary['accepted'])
        self.assertNotIn('processExitSucceeded', plan.summary)

    def test_wrong_external_identity_and_budget_reject_before_planning_read(self):
        for changes in ({'expected_run_identity': 'b'*64}, {'expected_context_sha256': 'b'*64},
                        {'expected_sha256': 'b'*64}, {'limits': replace(self.limits, trials=2)}):
            with self.subTest(changes=changes), patch.object(planner, 'read_snapshot', wraps=planner.read_snapshot) as second_read:
                with self.assertRaises(ValueError): self.plan(whole_pair(), **changes)
                second_read.assert_not_called()

    def test_between_reads_same_length_run_or_context_replacement_rejects(self):
        for mutation in ('run', 'mass'):
            with self.subTest(mutation=mutation):
                envelope = whole_pair()
                options = self.write(envelope)
                replacement = copy.deepcopy(envelope)
                if mutation == 'run': replacement['runIdentity'] = 'b'*64
                else: replacement['snapshot']['context']['massKg'][0] = 2.
                replacement_bytes = serialized(replacement)
                self.assertEqual(len(replacement_bytes), options['expected_bytes'])
                original_admission = planner.audit_snapshot
                def admit_then_replace(*args, **kwargs):
                    result = original_admission(*args, **kwargs)
                    self.path.write_bytes(replacement_bytes)
                    return result
                with patch.object(planner, 'audit_snapshot', side_effect=admit_then_replace):
                    with self.assertRaises(ValueError): planner.read_trial_plan(self.path, **options)

    def test_between_reads_symlink_substitution_rejects(self):
        options = self.write(whole_pair())
        other = self.path.with_name('copied.json')
        other.write_bytes(self.path.read_bytes())
        original_admission = planner.audit_snapshot
        def admit_then_replace(*args, **kwargs):
            result = original_admission(*args, **kwargs)
            self.path.unlink()
            self.path.symlink_to(other)
            return result
        with patch.object(planner, 'audit_snapshot', side_effect=admit_then_replace):
            with self.assertRaises((ValueError, OSError)): planner.read_trial_plan(self.path, **options)

    def test_mutating_returned_observations_does_not_change_file_or_fresh_plan(self):
        options = self.write(whole_pair())
        before = self.path.read_bytes()
        plan = planner.read_trial_plan(self.path, **options)
        # Dictionaries within one plan are intentionally owned mutable values;
        # no within-plan immutability/security promise is asserted here.
        plan.context['massKg'][0] = 99.
        plan.trials[0].trial['state']['positionsMeters'][0][0] = -999.
        plan.trials[2].start_state['positionsMeters'][0][0] = -888.
        plan.summary['committedFineTrialIds'].clear()
        self.assertEqual(self.path.read_bytes(), before)
        fresh = planner.read_trial_plan(self.path, **options)
        self.assertEqual(self.starts(fresh), [0., 0., 1.])
        self.assertEqual(fresh.trials[0].trial['state']['positionsMeters'][0][0], 10.)
        self.assertEqual(fresh.context['massKg'], [1.])
        self.assertEqual(fresh.summary['committedFineTrialIds'], [2, 3])

    def test_false_start_and_transaction_substitutions_are_rejected(self):
        for attack in ('previous-attempt-start', 'coarse-as-fine', 'missing-half', 'raw-committed'):
            envelope = whole_pair()
            raw = envelope['snapshot']
            if attack == 'previous-attempt-start':
                wrong = raw['attempts'][0]['state']['sha256']
                raw['attempts'][1]['startStateSha256'] = wrong
                raw['reservations'][1]['trial']['startStateSha256'] = wrong
            elif attack == 'coarse-as-fine': raw['transactions'][0]['fineTrialIds'] = [1, 3]
            elif attack == 'missing-half': raw['acceptedSteps'].pop()
            else: raw['attempts'][0]['outcome'] = 'committed'
            with self.subTest(attack=attack), self.assertRaises(ValueError): self.plan(envelope)


if __name__ == '__main__':
    unittest.main()
