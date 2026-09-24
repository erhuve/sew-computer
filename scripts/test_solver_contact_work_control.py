"""Bound contact work to actual endpoints and reject stale/mutated evidence."""
import copy
from dataclasses import asdict, replace
from fractions import Fraction as F
import unittest
from unittest.mock import patch

import ipctk
import numpy as np

from solver_contact_work import WorkPolicy
from solver_contact_work_native import bounded_native_work
from solver_contact_work_control import (
    ContactWorkControl, checked_change, contact_error, validate_contact_energy, validate_record,
)
from solver_rest_filtered_contact import RestFilteredSurfaceContact
from test_solver_contact_work_native import fixture


POLICY = asdict(WorkPolicy())


class ContactWorkControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): ipctk.set_num_threads(1)

    def setUp(self):
        self.contact, self.q = fixture()
        self.end = self.q.copy(); self.end[4:, 2] += 1e-14
        self.control = ContactWorkControl(self.contact, copy.deepcopy(POLICY))

    def work(self): return checked_change(self.control, self.contact, self.q, self.end)

    def report(self):
        work = self.work()
        return {'boundedContactWork': work, 'contactChangeJoules': work['changeJoules'],
                'contactBeforeJoules': work['start']['nativeEnergyJoules'],
                'contactAfterJoules': work['end']['nativeEnergyJoules']}

    def test_nonzero_tiny_work_has_bound_and_actual_complete_endpoint_identity(self):
        work = self.work()
        result, first, last = bounded_native_work(self.contact, self.q, self.end)
        self.assertEqual(work['changeJoules'].hex(), result.value.hex())
        self.assertLess(work['changeJoules'], 0.)
        self.assertGreater(validate_record(work), 0)
        self.assertGreaterEqual(validate_record(work), result.absolute_error)
        for name, endpoint in (('start', first), ('end', last)):
            self.assertEqual(work[name]['positionsSha256'], endpoint.positions_sha256)
            self.assertEqual(work[name]['termCount'], len(endpoint.terms))
        self.assertEqual(self.control.validate_change(self.q, self.end, work), work)

    def test_no_contact_terms_remains_a_valid_float_zero_record(self):
        q = self.q[:3].copy()
        contact = RestFilteredSurfaceContact(q, np.array([[0, 1, 2]]),
            activation_distance_m=.002, minimum_distance_m=.0001, stiffness=10000.)
        control = ContactWorkControl(contact, POLICY)
        work = checked_change(control, contact, q, q)
        self.assertEqual(validate_record(work), 0)
        self.assertEqual(work['unionTerms'], 0)
        for name in ('start', 'end'):
            self.assertIs(type(work[name]['nativeEnergyJoules']), float)
            self.assertEqual(work[name]['nativeEnergyJoules'], 0.)

    def test_activation_exit_and_reverse_preserve_missing_endpoint_terms(self):
        outside = self.q.copy(); outside[4:, 2] = .003
        forward = checked_change(self.control, self.contact, self.q, outside)
        reverse = checked_change(self.control, self.contact, outside, self.q)
        self.assertEqual(forward['end']['termCount'], 0)
        self.assertGreater(forward['start']['termCount'], 0)
        self.assertLess(forward['changeJoules'], 0.)
        self.assertLessEqual(abs(F(forward['changeJoules'])+F(reverse['changeJoules'])),
                             validate_record(forward)+validate_record(reverse))

    def test_missing_extra_and_invalid_policy_keys_reject(self):
        for policy in ({}, {**POLICY, 'unknown': 1}, {**POLICY, 'bits': True}):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                ContactWorkControl(self.contact, policy)

    def test_control_is_immutable_and_reinitialization_cannot_replace_native_model(self):
        before = self.control.description()
        with self.assertRaises(AttributeError): self.control.__init__(self.contact, POLICY)
        with self.assertRaises(AttributeError): self.control._policy = WorkPolicy()
        with self.assertRaises(AttributeError): del self.control._contact
        before['policy']['bits'] = 1
        self.assertEqual(self.control.description()['policy'], POLICY)
        self.assertFalse(hasattr(self.control, '__dict__'))

    def test_raw_lossy_or_nonfinite_state_rejects_before_capture(self):
        for q in (self.q.tolist(), self.q.astype(np.float32), self.q.astype(object),
                  self.q[:, :2], np.full_like(self.q, np.nan)):
            with self.subTest(kind=type(q)), patch('solver_contact_work_control.bounded_native_work',
                    side_effect=AssertionError('capture must not run')), self.assertRaises(ValueError):
                self.control.energy_change(q, self.end)

    def test_positions_are_detached_and_cannot_be_made_writable(self):
        state = self.control.positions(self.q)
        original = state.tobytes(); self.q[:] = 0
        self.assertEqual(state.tobytes(), original)
        with self.assertRaises(ValueError): state.setflags(write=True)

    def test_stale_native_result_is_rejected_even_when_helper_repeats_it_consistently(self):
        stale = bounded_native_work(self.contact, self.q, self.end)
        with patch('solver_contact_work_control.bounded_native_work', return_value=stale):
            with self.assertRaisesRegex(ValueError, 'endpoint identity'):
                checked_change(self.control, self.contact, self.q, self.q)

    def test_internally_inconsistent_work_midpoint_radius_or_interval_rejects(self):
        result, first, last = bounded_native_work(self.contact, self.q, self.end)
        for altered in (replace(result, value=0.), replace(result, absolute_error=F()),
                        replace(result, lower=result.upper+1)):
            with self.subTest(result=altered), patch('solver_contact_work_control.bounded_native_work',
                    return_value=(altered, first, last)), self.assertRaises(ValueError):
                self.work()

    def test_native_coordinate_and_inventory_count_tampering_rejects(self):
        result, first, last = bounded_native_work(self.contact, self.q, self.end)
        altered_term = replace(first.terms[0], positions=tuple((0., 0., 0.) for _ in first.terms[0].positions))
        for altered in (replace(first, terms=(altered_term,)+first.terms[1:]),
                        replace(first, observations=first.observations[:-1]),
                        replace(first, terms=first.terms[:-1], observations=first.observations[:-1])):
            with self.subTest(endpoint=altered), patch('solver_contact_work_control.bounded_native_work',
                    return_value=(result, altered, last)), self.assertRaises(ValueError):
                self.work()

    def test_mutating_helper_rejects_and_leaves_caller_state_intact(self):
        expected = self.q.copy()
        def attack(contact, first, last, **options):
            value = bounded_native_work(contact, first, last, **options)
            first[:] = 0
            return value
        with patch('solver_contact_work_control.bounded_native_work', side_effect=attack):
            with self.assertRaisesRegex(ValueError, 'mutated'): self.work()
        np.testing.assert_array_equal(self.q, expected)

    def test_model_identity_changes_reject_before_work(self):
        changed = self.contact.rest_positions.copy(); changed[0, 0] += 1e-6
        self.contact.rest_positions = changed
        with self.assertRaisesRegex(ValueError, 'identity'): self.work()

    def test_changed_or_understated_publication_is_freshly_rejected(self):
        original = self.report()
        for mutation in ('missing', 'public', 'radius', 'inventory', 'policy'):
            report = copy.deepcopy(original)
            if mutation == 'missing': del report['boundedContactWork']
            elif mutation == 'public': report['contactChangeJoules'] = 0.
            elif mutation == 'radius': report['boundedContactWork']['changeErrorBoundJoules'] = 0.
            elif mutation == 'inventory': report['boundedContactWork']['end']['inventorySha256'] = '0'*64
            else: report['boundedContactWork']['definition']['policy']['bits'] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_contact_energy(self.control, self.contact, self.q, self.end, report)
        self.assertEqual(contact_error(original), validate_contact_energy(
            self.control, self.contact, self.q, self.end, original))

    def test_exact_feature_and_budget_failures_remain_explicit(self):
        contact, q = fixture(rotated=False)
        control = ContactWorkControl(contact, POLICY)
        with self.assertRaisesRegex(ValueError, 'Native closest feature'):
            checked_change(control, contact, q, q)
        tiny = ContactWorkControl(self.contact, {**POLICY, 'max_endpoint_terms': 1})
        with self.assertRaisesRegex(ValueError, 'budget'):
            checked_change(tiny, self.contact, self.q, self.end)


if __name__ == '__main__': unittest.main()
