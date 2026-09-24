"""Independent admission and work-boundary tests for varying supplied cables.

These tests install no source controls and execute no construction recipe.
"""
import copy
from fractions import Fraction as F
import hashlib
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from solver_cable_integration import CableControl
from solver_cable_parameters import CableParameterRecipe
from solver_cable_varying import VaryingCableControl
from solver_continuous_cable_sewing import ContinuousCableSewing


RESPONSE_PRECISION = dict(energy_tolerance_joules=1e-12, gradient_tolerance_newtons=1e-10,
    hessian_tolerance_newtons_per_meter=1e-8, absolute_tolerance_joules=1e-14,
    max_boundary_depth=80, max_boundary_panels=256, moment_max_terms=128,
    moment_max_panels=256, moment_max_depth=64)
WORK_PRECISION = dict(absolute_tolerance_joules=1e-12, component_tolerances_joules=None,
    max_boundary_depth=80, max_boundary_panels=256, moment_max_terms=128,
    moment_max_panels=256, moment_max_depth=64)
FIELDS = ('targetWorkJoules', 'activationWorkJoules', 'totalWorkJoules',
          'activationIncreaseWorkJoules', 'releaseEnergyRemovedJoules')


def rat(value):
    value = F(value)
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}


def rational(value):
    return F(int(value['numerator']), int(value['denominator']))


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def point_anchor(vertex):
    return [{'vertex': vertex, 'weight': rat(1)}]


def pair_cell(*, target=1., activation=1., stiffness=4., length=F(1)):
    return dict(id='analytic-relative-pair', positiveStart=point_anchor(0), positiveEnd=point_anchor(0),
        negativeStart=point_anchor(1), negativeEnd=point_anchor(1), targetsMeters=[target, target],
        activation=activation, stiffnessDensityNPerM2=stiffness, referenceLengthMeters=rat(length))


def pair_recipe(**cell_options):
    return CableParameterRecipe(ContinuousCableSewing(2, [pair_cell(**cell_options)]))


def wrapper(**cell_options):
    return VaryingCableControl(pair_recipe(**cell_options), copy.deepcopy(RESPONSE_PRECISION),
                               copy.deepcopy(WORK_PRECISION))


class CableVaryingControlTests(unittest.TestCase):
    def test_effective_control_is_frozen_and_matches_independent_relative_spring(self):
        control = wrapper(target=2., activation=0.)
        q = np.array([[0., 0., 0.], [3., 0., 0.]])
        d, a = np.array([[1., 1.]]), np.array([.5])
        effective = control.effective(d, a)
        self.assertIs(type(effective), CableControl)
        d[:] = 99.; a[:] = 0.
        response = effective.evaluate(q)
        self.assertEqual(response['energy'], 4.)
        np.testing.assert_array_equal(response['gradient'].reshape(2, 3), [[-4., 0., 0.], [4., 0., 0.]])
        # r=3,d=1,beta=2: radial tangent Hessian beta*(1-d/r).
        expected = np.diag([2., 4/3, 4/3])
        hessian = response['hessian'].toarray()
        np.testing.assert_allclose(hessian[:3, :3], expected, atol=1e-15, rtol=0)
        np.testing.assert_array_equal(hessian[:3, 3:], -hessian[:3, :3])
        self.assertEqual(control.recipe.initial_parameters[0].tolist(), [[2., 2.]])
        self.assertEqual(control.recipe.initial_parameters[1].tolist(), [0.])

    def test_parameter_record_preserves_complete_numerical_identity_and_is_detached(self):
        control = wrapper()
        d, a = [[.5, .75]], [.25]
        record = control.parameter_record(d, a)
        self.assertEqual(set(record), {'definition', 'parameterSha256', 'targetsMeters',
                                      'activation', 'effectivePotentialSha256'})
        self.assertEqual(record['definition'], control.description())
        self.assertEqual(record['targetsMeters'], d); self.assertEqual(record['activation'], a)
        expected = hashlib.sha256(encoded({'targetsMeters': d, 'activation': a})).hexdigest()
        self.assertEqual(record['parameterSha256'], expected)
        self.assertEqual(record['parameterSha256'], control.parameter_sha256(d, a))
        self.assertEqual(record['effectivePotentialSha256'], control.effective(d, a).description()['inputSha256'])
        record['targetsMeters'][0][0] = 99.; record['definition'].clear()
        self.assertEqual(control.parameter_record(d, a)['targetsMeters'], d)
        changed = control.parameter_record(d, [0.])
        self.assertNotEqual(changed['parameterSha256'], expected)
        self.assertEqual(changed['definition'], control.description())

    def test_recipe_policy_results_and_constructor_are_immutable(self):
        recipe = pair_recipe(); response, work = copy.deepcopy(RESPONSE_PRECISION), copy.deepcopy(WORK_PRECISION)
        control = VaryingCableControl(recipe, response, work)
        before = encoded(control.description())
        response['gradient_tolerance_newtons'] = 1.; work['absolute_tolerance_joules'] = 1.
        control.response_precision.clear(); control.parameter_work_precision.clear(); control.description().clear()
        self.assertEqual(control.response_precision, RESPONSE_PRECISION)
        self.assertEqual(control.parameter_work_precision, WORK_PRECISION)
        self.assertEqual(encoded(control.description()), before)
        self.assertFalse(hasattr(control, '__dict__'))
        for name in ('recipe', 'vertex_count', 'response_precision', '_sealed'):
            with self.subTest(name=name), self.assertRaises(AttributeError): setattr(control, name, None)
            with self.subTest(delete=name), self.assertRaises(AttributeError): delattr(control, name)
        with self.assertRaises(AttributeError): control.__init__(recipe, RESPONSE_PRECISION, WORK_PRECISION)
        d, a = np.array([[1., 2.]]), np.array([.25])
        first = control.parameters(d, a); first[0][:] = 8.; first[1][:] = 1.
        np.testing.assert_array_equal(d, [[1., 2.]]); np.testing.assert_array_equal(a, [.25])
        np.testing.assert_array_equal(control.parameters(d, a)[0], d)

    def test_both_precision_policies_are_exact_explicit_raw_bounded_objects(self):
        recipe = pair_recipe()
        for index, policy in enumerate((RESPONSE_PRECISION, WORK_PRECISION)):
            for key in policy:
                malformed = copy.deepcopy(policy); malformed.pop(key)
                pair = [copy.deepcopy(RESPONSE_PRECISION), copy.deepcopy(WORK_PRECISION)]; pair[index] = malformed
                with self.subTest(policy=index, missing=key), self.assertRaises(ValueError):
                    VaryingCableControl(recipe, *pair)
            malformed = dict(policy, undeclared=1)
            pair = [copy.deepcopy(RESPONSE_PRECISION), copy.deepcopy(WORK_PRECISION)]; pair[index] = malformed
            with self.assertRaises(ValueError): VaryingCableControl(recipe, *pair)
        for index, key, value in ((0, 'energy_tolerance_joules', True),
                (0, 'gradient_tolerance_newtons', 0.), (0, 'moment_max_terms', 128.),
                (1, 'absolute_tolerance_joules', np.float64(1e-12)),
                (1, 'max_boundary_depth', False), (1, 'max_boundary_panels', 0),
                (1, 'component_tolerances_joules', {}),
                (1, 'component_tolerances_joules', {name: True for name in FIELDS if name != 'totalWorkJoules'})):
            pair = [copy.deepcopy(RESPONSE_PRECISION), copy.deepcopy(WORK_PRECISION)]; pair[index][key] = value
            with self.subTest(policy=index, key=key), self.assertRaises(ValueError):
                VaryingCableControl(recipe, *pair)
        for invalid in (None, object(), ContinuousCableSewing(2, [pair_cell()])):
            with self.assertRaises(ValueError): VaryingCableControl(invalid, RESPONSE_PRECISION, WORK_PRECISION)

    def test_target_first_release_has_independent_exact_work_components(self):
        control = wrapper(); q = np.array([[0., 0., 0.], [3., 0., 0.]])
        result = control.parameter_work(q, [[1., 1.]], [1.], [[2., 2.]], [0.])
        expected = dict(zip(FIELDS, (-6., -2., -8., 0., 2.)))
        for field, value in expected.items():
            self.assertEqual(result[field], value)
            self.assertEqual(rational(result['certificate']['errorsJoules'][field]), 0)
        self.assertEqual(control.validate_parameter_work(q, [[1., 1.]], [1.], [[2., 2.]], [0.], result), result)
        result['certificate']['errorsJoules']['totalWorkJoules'] = rat(1)
        self.assertEqual(control.parameter_work(q, [[1., 1.]], [1.], [[2., 2.]], [0.])['totalWorkJoules'], -8.)

    def test_exact_beta_increment_and_distinct_component_tolerances_survive_wrapper(self):
        control = wrapper(stiffness=1e12); q = np.array([[0., 0., 0.], [2., 0., 0.]])
        after = math.nextafter(.25, 1.)
        result = control.parameter_work(q, [[1., 1.]], [.25], [[1., 1.]], [after])
        expected = F(1e12)*(F(after)-F(.25))/2
        bound = rational(result['certificate']['errorsJoules']['totalWorkJoules'])
        self.assertLessEqual(abs(F(result['totalWorkJoules'])-expected), bound)
        wrong = (F(float(F(1e12)*F(after)))-F(float(F(1e12)*F(.25))))/2
        self.assertGreater(abs(F(result['totalWorkJoules'])-wrong), bound)
        work = copy.deepcopy(WORK_PRECISION)
        work.update(absolute_tolerance_joules=1e-30,
            component_tolerances_joules={name:1e-12 for name in FIELDS if name != 'totalWorkJoules'})
        control = VaryingCableControl(pair_recipe(stiffness=1.), RESPONSE_PRECISION, work)
        result = control.parameter_work(q, [[2., 2.]], [1.], [[1., 1.]], [2.**-52])
        self.assertEqual(result['totalWorkJoules'], 2.**-53)
        self.assertEqual(result['certificate']['requestedTolerancesJoules']['totalWorkJoules'], rat(F(1e-30)))
        self.assertEqual(result['certificate']['requestedTolerancesJoules']['targetWorkJoules'], rat(F(1e-12)))

    def test_certificate_forgeries_and_stale_state_controls_are_rejected(self):
        control = wrapper(); q = np.array([[0., 0., 0.], [3., 0., 0.]])
        d0, a0, d1, a1 = [[1., 1.]], [1.], [[2., 2.]], [.25]
        result = control.parameter_work(q, d0, a0, d1, a1)
        attacks = [lambda x: x.update(totalWorkJoules=True), lambda x: x.update(targetWorkJoules=float('nan')),
            lambda x: x.pop('activationWorkJoules'), lambda x: x.update(unknown=0.),
            lambda x: x['certificate'].update(accepted=0), lambda x: x['certificate'].update(verified=1),
            lambda x: x['certificate'].update(sourceAdmissionGranted=True),
            lambda x: x['certificate'].update(parameterOrder='activation-first'),
            lambda x: x['certificate']['budgets'].update(max_boundary_depth=True),
            lambda x: x['certificate']['errorsJoules'].update(totalWorkJoules=rat(-1)),
            lambda x: x['certificate']['errorsJoules'].update(totalWorkJoules=rat(1)),
            lambda x: x['certificate']['errorsJoules'].update(totalWorkJoules={'numerator':'0','denominator':'2'}),
            lambda x: x['certificate']['requestedTolerancesJoules'].update(targetWorkJoules=rat(1))]
        for field in ('geometrySha256', 'positionsSha256', 'beforeParameterSha256', 'afterParameterSha256',
                      'beforePotentialSha256', 'afterPotentialSha256'):
            attacks.append(lambda x, field=field: x['certificate'].__setitem__(field, '0'*64))
        for index, attack in enumerate(attacks):
            malformed = copy.deepcopy(result); attack(malformed)
            with self.subTest(attack=index), self.assertRaises(ValueError):
                control.validate_parameter_work(q, d0, a0, d1, a1, malformed)
        moved = q.copy(); moved[1, 0] = math.nextafter(3., 4.)
        for args in ((moved, d0, a0, d1, a1), (q, d1, a0, d1, a1), (q, d0, a0, d1, [.5])):
            with self.assertRaises(ValueError): control.validate_parameter_work(*args, result)

    def test_raw_controls_states_and_positive_beta_underflow_are_not_repaired(self):
        control = wrapper()
        for d, a in (([[True, 1.]], [1.]), ([[1., 1.]], [False]), ([[0., 1.]], [0.]),
                ([[101., 1.]], [0.]), ([[1., math.inf]], [0.]), ([[1., 1.]], [-.1]),
                ([[1., 1.]], [1.1]), ([1., 1.], [1.]), ([[1., 1.]], [[1.]])):
            with self.subTest(d=d, a=a), self.assertRaises(ValueError): control.effective(d, a)
        q = np.array([[0., 0., 0.], [3., 0., 0.]])
        for invalid in (q.astype(bool), [[0., 0., 0.], [2**53+1, 0., 0.]], q[:1], q+math.inf):
            with self.assertRaises(ValueError): control.positions(invalid)
        if np.dtype(np.longdouble).itemsize > 8:
            with self.assertRaises(ValueError): control.parameters(np.ones((1,2), dtype=np.longdouble), [1.])
        tiny = wrapper(stiffness=.5)
        with self.assertRaises(ValueError): tiny.effective([[1., 1.]], [math.ulp(0.)])
        self.assertEqual(tiny.effective([[1., 1.]], [0.]).evaluate(q)['energy'], 0.)

    def test_backend_mutation_cannot_change_original_inputs_or_return_success(self):
        control = wrapper(); original = CableParameterRecipe.parameter_energy_change
        q = np.array([[0., 0., 0.], [3., 0., 0.]])
        arrays = [q, np.array([[1., 1.]]), np.array([1.]), np.array([[2., 2.]]), np.array([.25])]
        for index in (0, 1, 2, 3, 4):
            def corrupt(recipe, *values, **keywords):
                result = original(recipe, *values, **keywords)
                values[index].flat[0] = math.nextafter(float(values[index].flat[0]), math.inf)
                return result
            snapshots = [value.copy() for value in arrays]
            with self.subTest(index=index), patch.object(CableParameterRecipe, 'parameter_energy_change', corrupt), \
                    self.assertRaises(ValueError):
                control.parameter_work(*arrays)
            for value, saved in zip(arrays, snapshots): np.testing.assert_array_equal(value, saved)

    def test_effective_potential_backend_cannot_mutate_sampled_parameters(self):
        control=wrapper();original=CableParameterRecipe.potential
        d,a=np.array([[1.,1.]]),np.array([.5])
        def corrupt(recipe,targets,activation):
            result=original(recipe,targets,activation)
            targets[0,0]=math.nextafter(float(targets[0,0]),math.inf)
            return result
        with patch.object(CableParameterRecipe,'potential',corrupt),self.assertRaises(ValueError):
            control.effective(d,a)
        np.testing.assert_array_equal(d,[[1.,1.]])
        np.testing.assert_array_equal(a,[.5])

    def test_unresolved_partition_fails_under_original_parameter_work_policy(self):
        c = pair_cell(); c.update(positiveEnd=point_anchor(1), negativeStart=point_anchor(2), negativeEnd=point_anchor(3))
        recipe = CableParameterRecipe(ContinuousCableSewing(4, [c]))
        policy = dict(WORK_PRECISION, max_boundary_depth=0)
        control = VaryingCableControl(recipe, RESPONSE_PRECISION, policy)
        q = np.array([[.5, 0., 0.], [2., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        with self.assertRaisesRegex(ValueError, '[Bb]oundary|unresolved'):
            control.parameter_work(q, [[.75, .75]], [1.], [[1.25, 1.25]], [.5])
        self.assertEqual(control.parameter_work_precision, policy)


if __name__ == '__main__': unittest.main()
