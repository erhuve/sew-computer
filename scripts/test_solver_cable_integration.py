"""Independent admission, precision and closed-form integrated cable checks."""
import copy
from fractions import Fraction as F
import unittest

import numpy as np

from solver_cable_integration import (
    CableControl, assemble_gradient, directional_interval, round_sum, stationarity, _rat, _rational,
)
from solver_continuous_cable_sewing import ContinuousCableSewing


def fixture():
    anchor = lambda vertex: [{'vertex': vertex, 'weight': {'numerator': '1', 'denominator': '1'}}]
    potential = ContinuousCableSewing(2, [{
        'id': 'analytic-free-pair', 'positiveStart': anchor(0), 'positiveEnd': anchor(0),
        'negativeStart': anchor(1), 'negativeEnd': anchor(1), 'targetsMeters': [1., 1.],
        'referenceLengthMeters': {'numerator': '1', 'denominator': '1'},
        'stiffnessDensityNPerM2': 1., 'activation': 1.,
    }])
    policy = dict(energy_tolerance_joules=1e-12, gradient_tolerance_newtons=1e-10,
                  hessian_tolerance_newtons_per_meter=1e-8, absolute_tolerance_joules=1e-14,
                  max_boundary_depth=80, max_boundary_panels=256, moment_max_terms=128,
                  moment_max_panels=256, moment_max_depth=64)
    return potential, policy, np.array([[0., 0., 0.], [2., 0., 0.]])


class CableIntegrationTests(unittest.TestCase):
    def test_control_copies_recipe_policy_and_all_returned_state(self):
        potential, policy, q = fixture()
        control = CableControl(potential, policy)
        expected = control.evaluate(q)
        policy['gradient_tolerance_newtons'] = 1.
        for mapping in (control.policy, control.description(), potential.cells[0]):
            mapping.clear()
        changed = control.evaluate(q)
        changed['gradient'][:] = 999.
        changed['hessian'].data[:] = 888.
        changed['certificate']['inputSha256'] = 'bad'
        np.testing.assert_array_equal(control.evaluate(q)['gradient'], expected['gradient'])
        self.assertEqual(control.policy['gradient_tolerance_newtons'], 1e-10)
        with self.assertRaises(AttributeError):
            control._potential = potential
        with self.assertRaises(AttributeError):
            del control._policy_bytes

    def test_complete_explicit_raw_precision_and_real_positions_required(self):
        potential, policy, q = fixture()
        for key in policy:
            missing = dict(policy); missing.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                CableControl(potential, missing)
        for key, value in [('gradient_tolerance_newtons', True), ('energy_tolerance_joules', 0.),
                           ('absolute_tolerance_joules', np.float64(1e-12)),
                           ('moment_max_terms', 128.), ('max_boundary_panels', False)]:
            bad = dict(policy); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                CableControl(potential, bad)
        control = CableControl(potential, policy)
        bad_positions = [q.astype(bool), [[0, 0, 0], [True, 0, 0]],
                         [[0., 0., 0.], [float('inf'), 0., 0.]]]
        # ARM macOS may alias longdouble to binary64; only genuinely wider
        # formats require rejection, independent of the scalar type's name.
        if np.dtype(np.longdouble).itemsize > 8:
            bad_positions.append(q.astype(np.longdouble))
        else:
            np.testing.assert_array_equal(control.positions(q.astype(np.longdouble)), q)
        for bad in bad_positions:
            with self.assertRaises(ValueError):
                control.positions(bad)

    def test_response_state_recipe_policy_and_shape_forgeries_reject(self):
        potential, policy, q = fixture(); control = CableControl(potential, policy)
        response = control.evaluate(q)
        attacks = [
            lambda x: x['certificate'].update(positionsSha256='0'*64),
            lambda x: x['certificate'].update(inputSha256='0'*64),
            lambda x: x['certificate'].update(verified=1),
            lambda x: x['certificate'].update(accepted=True),
            lambda x: x['certificate']['budgets'].update(max_boundary_depth=True),
            lambda x: x['certificate']['requestedTolerances'].update(g=_rat(F(1))),
            lambda x: x['certificate'].update(gradientMaxAbsoluteErrorBoundNewtons=_rat(F(-1))),
            lambda x: x['certificate'].update(hessianMaxEntryErrorBoundNewtonsPerMeter=_rat(F(1))),
            lambda x: x.update(energy=float('nan')),
            lambda x: x.update(gradient=x['gradient'].astype(np.float32)),
            lambda x: x.update(gradient=x['gradient'].reshape(2, 3)),
            lambda x: x['hessian'].__setitem__((0, 1), 1.),
        ]
        for index, attack in enumerate(attacks):
            bad = copy.deepcopy(response); attack(bad)
            with self.subTest(index=index), self.assertRaises(ValueError):
                control.validate_response(q, bad)
        moved = q.copy(); moved[0, 1] = np.nextafter(0., 1.)
        with self.assertRaises(ValueError):
            control.validate_response(moved, response)

    def test_work_and_energy_only_diagnostics_still_bind_endpoints_and_bounds(self):
        potential, policy, q = fixture(); control = CableControl(potential, policy)
        end = q.copy(); end[1, 0] = 1.5
        work = control.energy_change(q, end)
        self.assertEqual(work['changeJoules'], -.375)
        for field in ('startPositionsSha256', 'endPositionsSha256', 'inputSha256'):
            bad = copy.deepcopy(work); bad['certificate'][field] = '0'*64
            with self.subTest(field=field), self.assertRaises(ValueError):
                control.validate_change(q, end, bad)
        for bound in ({'numerator': '2', 'denominator': '2'}, _rat(F(-1)), _rat(F(1))):
            bad = copy.deepcopy(work); bad['certificate']['changeErrorBoundJoules'] = bound
            with self.assertRaises(ValueError):
                control.validate_change(q, end, bad)
        diagnostic = control.diagnostics(q)
        self.assertEqual(control.validate_diagnostics(q, diagnostic), diagnostic)
        diagnostic['definition']['precision']['max_boundary_depth'] = 1
        with self.assertRaises(ValueError):
            control.validate_diagnostics(q, diagnostic)

    def test_exact_binary_assembly_retains_cancellation_and_rounding_error(self):
        value, bound = round_sum((1e16, 1., -1e16))
        self.assertEqual((value, bound), (1., F()))
        values, total, rounding = assemble_gradient(np.array([1e16]), np.array([1.]), F(1, 8))
        self.assertEqual(values[0], 1e16)
        self.assertEqual(rounding, F(1)); self.assertEqual(total, F(9, 8))
        values, total, rounding = assemble_gradient(np.array([1e16]), np.array([-1e16]), F(1))
        report = stationarity(values, total, F(1), rounding)
        self.assertEqual(report['gradientInfinityNorm'], 0.)
        self.assertEqual(_rational(report['stationarityUpperBoundNewtons']), 1)

    def test_direction_interval_uses_actual_endpoint_difference_before_rounding(self):
        start, end = np.array([2.**-54]), np.array([1.])
        self.assertEqual(float(end[0]-start[0]), 1.)
        expected = -F(1)+F(1, 2**54)
        self.assertEqual(directional_interval(np.array([-1.]), F(), start, end), (expected, expected))
        lo, hi = directional_interval(np.array([-1.]), F(1, 8), start, end)
        self.assertEqual((lo, hi), (expected*F(9, 8), expected*F(7, 8)))
        with self.assertRaises(ValueError):
            directional_interval(np.array([-1.]), F(-1), start, end)

    def test_closed_form_two_particle_backward_euler_and_internal_impulse(self):
        import newton
        from solver_global_sewing import GlobalSewingSolver
        potential, policy, q = fixture()
        builder = newton.ModelBuilder(gravity=(0., 0., 0.))
        for position in q:
            builder.add_particle(pos=position, vel=(0., 0., 0.), mass=1.)
        builder.set_coloring([[0], [1]])
        model = builder.finalize(device='cpu')
        solver = GlobalSewingSolver(model, [], 1., continuous_cable=potential, cable_precision=policy)
        state, velocity, report = solver.step(q, np.zeros_like(q), np.empty((0, 3)), 1.)
        # r_next=(m*r_previous+2*beta*d)/(m+2*beta)=4/3,
        # center of mass remains1, giving x=(1/3,5/3).
        np.testing.assert_allclose(state[:, 0], [1/3, 5/3], rtol=0, atol=1e-12)
        np.testing.assert_allclose(velocity.sum(axis=0), 0, rtol=0, atol=1e-14)
        self.assertTrue(report['converged']); self.assertFalse(report['accepted'])
        self.assertLessEqual(_rational(report['cableStationarity']['stationarityUpperBoundNewtons']), F(1e-6))
        self.assertTrue(report['directionHistory'])
        for entry in report['directionHistory']:
            decision = entry['cableAwareDecision']
            self.assertLess(_rational(decision['conditionalSlopeUpperJoules']), 0)
            self.assertLessEqual(_rational(decision['conditionalChangeUpperJoules']),
                                 F(1e-4)*_rational(decision['conditionalSlopeLowerJoules']))


if __name__ == '__main__':
    unittest.main()
