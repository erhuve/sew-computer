"""Independent fixed-control physical laws and adversarial response boundaries.

Native fixtures are synthetic. No timestep is published by these tests.
"""
from fractions import Fraction as F
import unittest
from unittest.mock import patch

import ipctk
import numpy as np
from scipy.sparse import csr_matrix

import solver_physical_response as physical
from solver_bending import ElasticDihedralBending
from solver_cable_integration import CableControl
from solver_cable_varying import VaryingCableControl
from solver_global_sewing import GlobalSewingSolver
from solver_temporal_control import problem_identity
from test_solver_contact_work_integration import cloth as contact_cloth
from test_solver_controlled_fold_integration import fixture as fold_fixture
from test_solver_gripper_integration import make_model, make_recipe
from test_solver_sewing_activation_integration import particles


EMPTY = np.empty((0, 3), dtype=np.float64)


def fraction(value):
    return F(int(value['numerator']), int(value['denominator']))


def particle_solver(mode='vector', *, fixed=False):
    model = particles(masses=[0., 2., 3., 4.] if fixed else None)
    q = model.particle_q.numpy().astype(np.float64)
    weights = np.array([.75, -.25, .25, -.75])
    solver = GlobalSewingSolver(model, [dict(enumerate(weights))], .125, sewing_mode=mode)
    return solver, q, weights


def normal_fixture():
    model = make_model(vertices=[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., .2]],
                       faces=[[0, 1, 2], [1, 3, 2]])
    q = model.particle_q.numpy().astype(np.float64)
    solver = GlobalSewingSolver(model, [{0: -1., 3: 1.}], .125, sewing_mode='normal-offset',
        sewing_frame_faces=[[0, 1, 2]], sewing_sides=[1])
    return solver, q


def folded_control(kind='controlled'):
    model, template, q = fold_fixture(count=1)
    grippers = make_recipe(model, stiffness=8.)
    options = dict(material_grippers=grippers, fold_barrier_joules=.003, fold_activation_angle=.1)
    if kind == 'controlled':
        options['controlled_fold_actuation'] = template.controlled_fold_actuation
        controls = dict(fold_targets=[-.12], fold_activation=[.7])
    else:
        options.update(fold_hinges=template.controlled_fold_actuation.hinges, fold_stiffness_joules=[.02])
        controls = dict(fold_targets=[-.12])
    solver = GlobalSewingSolver(model, [], .125, **options)
    q[0, 2] += .004
    anchor = np.array([.25, .25, .5])@q[solver.faces[0]]
    controls.update(gripper_targets=[anchor+[.001, -.002, .003]], gripper_activation=[.6])
    return solver, q, controls


class _ClosureCaptured(BaseException):
    pass


def old_physical_closure(solver, q, targets, controls):
    """Observe old objective/gradient at q, v=0; inertia is exactly zero.

    Abort before descent. Do not subtract a large inertial response numerically.
    """
    answer = {}
    def capture(evaluate, start, maximum, hessian, objective, **options):
        free = q.ravel()[solver.free].copy()
        answer['energy'] = objective(free)
        answer['gradient'] = options['gradient_function'](free)
        raise _ClosureCaptured()
    with patch('solver_global_sewing._direct_descent', capture):
        try:
            solver.step(q, np.zeros_like(q), targets, .001, **controls)
        except _ClosureCaptured:
            pass
        else:
            raise AssertionError('Old guarded descent closure was not reached')
    return answer


class PhysicalResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ipctk.set_num_threads(1)

    def assert_response(self, result, size):
        self.assertEqual(result['profile'], physical.PROFILE)
        self.assertFalse(result['includesInertia'])
        self.assertFalse(result['accepted'])
        self.assertTrue(result['configurationIdentityChecked'])
        self.assertEqual(result['gradientNewtons'].shape, (size,))
        self.assertEqual(result['searchMatrixNewtonsPerMetre'].shape, (size, size))
        self.assertEqual(result['searchMatrixNewtonsPerMetre'].format, 'csr')
        self.assertEqual(set(result['componentEnergyJoules']), set(physical.COMPONENTS))
        self.assertIn('not certified', result['knownErrorBounds']['scope'])

    def test_vector_closed_form_full_gradient_including_fixed_reaction(self):
        solver, q, coefficients = particle_solver(fixed=True)
        target, weight = np.array([[.03, -.07, .02]]), .625
        result = physical.FixedPhysicalPotential(solver, target, sewing_activation=[weight]).evaluate(q)
        error = coefficients@q-target[0]
        expected_g = weight/solver.compliance*np.outer(coefficients, error)
        expected_h = weight/solver.compliance*np.kron(np.outer(coefficients, coefficients), np.eye(3))
        self.assert_response(result, q.size)
        self.assertAlmostEqual(result['energyJoules'], weight*np.dot(error, error)/(2*solver.compliance), places=14)
        np.testing.assert_allclose(result['gradientNewtons'], expected_g.ravel(), rtol=2e-15, atol=1e-15)
        np.testing.assert_allclose(result['searchMatrixNewtonsPerMetre'].toarray(), expected_h, rtol=0, atol=1e-15)
        self.assertGreater(np.linalg.norm(result['gradientNewtons'][:3]), 0.)
        self.assertFalse(solver.active[0])
        np.testing.assert_allclose(result['gradientNewtons'].reshape(-1, 3).sum(axis=0), 0., atol=1e-15)

    def test_distance_compression_retains_signed_exact_curvature(self):
        solver, q, coefficients = particle_solver('distance')
        vector = coefficients@q
        length = np.linalg.norm(vector)
        target, activation = length*1.7, .8
        result = physical.FixedPhysicalPotential(solver, [target], sewing_activation=[activation]).evaluate(q)
        unit = vector/length
        block = activation/solver.compliance*(np.outer(unit, unit)+(1-target/length)*(np.eye(3)-np.outer(unit, unit)))
        expected_g = activation/solver.compliance*(1-target/length)*np.outer(coefficients, vector)
        self.assertAlmostEqual(result['energyJoules'], activation*(length-target)**2/(2*solver.compliance), places=14)
        np.testing.assert_allclose(result['gradientNewtons'], expected_g.ravel(), rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(result['searchMatrixNewtonsPerMetre'].toarray(), np.kron(np.outer(coefficients, coefficients), block), rtol=1e-13, atol=1e-14)
        self.assertLess(np.linalg.eigvalsh(result['searchMatrixNewtonsPerMetre'].toarray()).min(), -.01)

    def test_normal_offset_energy_gradient_and_exact_directional_curvature(self):
        solver, q = normal_fixture()
        target, activation = .17, .6
        potential = physical.FixedPhysicalPotential(solver, [target], sewing_activation=[activation])
        result = potential.evaluate(q)
        def energy(state):
            cross = np.cross(state[1]-state[0], state[2]-state[0])
            error = state[3]-state[0]-target*cross/np.linalg.norm(cross)
            return activation*np.dot(error, error)/(2*solver.compliance)
        self.assertAlmostEqual(result['energyJoules'], energy(q), places=13)
        epsilon = 2e-6
        finite = np.empty(q.size)
        for i in range(q.size):
            delta = np.zeros_like(q); delta.flat[i] = epsilon
            finite[i] = (energy(q+delta)-energy(q-delta))/(2*epsilon)
        np.testing.assert_allclose(result['gradientNewtons'], finite, rtol=2e-8, atol=2e-8)
        direction = np.arange(1, q.size+1, dtype=float).reshape(q.shape)
        direction /= np.linalg.norm(direction)
        epsilon = 2e-4
        second = (energy(q+epsilon*direction)-2*energy(q)+energy(q-epsilon*direction))/epsilon**2
        observed = direction.ravel()@result['searchMatrixNewtonsPerMetre']@direction.ravel()
        self.assertAlmostEqual(observed, second, delta=3e-7)
        self.assertEqual(result['componentCalls']['sewing.exact_hessian'], 1)

    def test_membrane_stable_reference_and_tiny_stiffness_linear_strain(self):
        for shear, lam in [(1000., 2000.), (2.5e-7, 1.25e-7)]:
            model = make_model(vertices=[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], membrane=0.)
            solver = GlobalSewingSolver(model, [{0: 1., 1: -1.}], .125)
            solver.materials[:, :2] = [shear, lam]
            s, t = 1.17, .83
            q = np.zeros((3, 3)); q[1:] = np.linalg.inv(solver.poses[0].T)@np.array([[s, 0., 0.], [0., t, 0.]])
            potential = physical.FixedPhysicalPotential(solver, [[0., 0., 0.]], sewing_activation=[0.])
            result = potential.evaluate(q)
            bulk = shear+lam
            linear = shear+bulk*(1-(1+shear/max(bulk, 1e-6)))
            area = solver.areas[0]
            expected = area*(shear*(s-t)**2/2+bulk*(s*t-1)**2/2+linear*(s*t-1))
            self.assertAlmostEqual(result['energyJoules'], expected, delta=1e-13*max(abs(expected), 1e-12))
            derivative_s = shear*(s-t)+bulk*t*(s*t-1)+linear*t
            derivative_t = shear*(t-s)+bulk*s*(s*t-1)+linear*s
            coefficients = np.vstack((-solver.poses[0].sum(axis=0), solver.poses[0]))
            expected_g = area*coefficients@np.array([[derivative_s, 0., 0.], [0., derivative_t, 0.]])
            np.testing.assert_allclose(result['gradientNewtons'], expected_g.ravel(), rtol=2e-13, atol=1e-20)
            rest = q.copy(); rest[1:] = np.linalg.inv(solver.poses[0].T)@np.array([[1., 0., 0.], [0., 1., 0.]])
            self.assertAlmostEqual(potential.evaluate(rest)['energyJoules'], 0., delta=1e-24)
            if shear < 1e-6:
                without_linear = area*(shear*(s-t)**2/2+bulk*(s*t-1)**2/2)
                self.assertGreater(abs(expected-without_linear), 1e-10)

    def test_gripper_quadratic_law_has_full_external_force_and_matrix(self):
        model = make_model()
        q = model.particle_q.numpy().astype(float)
        weights, stiffness, activation = np.array([.25, .25, .5]), 8., .7
        recipe = make_recipe(model, stiffness=stiffness)
        solver = GlobalSewingSolver(model, [], .125, material_grippers=recipe)
        target = (weights@q+[.003, -.002, .004])[None]
        result = physical.FixedPhysicalPotential(solver, EMPTY, gripper_targets=target, gripper_activation=[activation]).evaluate(q)
        error = weights@q-target[0]
        expected = stiffness*activation*np.outer(weights, error)
        np.testing.assert_allclose(result['gradientNewtons'], expected.ravel(), rtol=2e-14, atol=1e-16)
        np.testing.assert_allclose(result['searchMatrixNewtonsPerMetre'].toarray(), stiffness*activation*np.kron(np.outer(weights, weights), np.eye(3)), atol=1e-15)
        self.assertAlmostEqual(result['energyJoules'], stiffness*activation*np.dot(error, error)/2, delta=1e-18)
        self.assertGreater(np.linalg.norm(expected.sum(axis=0)), 0.)

    def test_legacy_controlled_fold_bending_and_barrier_are_present_and_labelled(self):
        for kind in ('legacy', 'controlled'):
            solver, q, controls = folded_control(kind)
            potential = physical.FixedPhysicalPotential(solver, EMPTY, **controls)
            result = potential.evaluate(q)
            for name in ('bending', 'foldActuation', 'foldBarrier', 'grippers'):
                self.assertGreater(result['componentEnergyJoules'][name], 0., name)
                self.assertEqual(result['componentCalls'][name+'.gradient'], 1)
            self.assertEqual(result['componentSearchMatrixKinds']['bending'], 'gauss-newton-search-metric')
            self.assertEqual(result['componentSearchMatrixKinds']['foldActuation'], 'gauss-newton-search-metric')
            self.assertEqual(result['componentSearchMatrixKinds']['foldBarrier'], 'angular-pullback-search-metric')
            original = old_physical_closure(solver, q, EMPTY, controls)
            self.assertAlmostEqual(result['energyJoules'], original['energy'], delta=2e-12*max(1., abs(original['energy'])))
            np.testing.assert_allclose(result['gradientNewtons'][solver.free], original['gradient'], rtol=2e-12, atol=2e-12)

    def test_old_zero_inertia_closures_cover_all_sewing_and_contact_cable_routes(self):
        fixtures = []
        for mode in ('vector', 'distance'):
            solver, q, _ = particle_solver(mode)
            target = [[.01, -.02, .03]] if mode == 'vector' else [.2]
            fixtures.append((solver, q, target, dict(sewing_activation=[.6])))
        solver, q = normal_fixture()
        fixtures.append((solver, q, [.17], dict(sewing_activation=[.6])))
        for route in ('contact', 'fixed', 'varying'):
            solver, q = contact_cloth(route)
            controls = dict(cable_targets=[[.0015, .0015]], cable_activation=[.8]) if route == 'varying' else {}
            fixtures.append((solver, q, EMPTY, controls))
        for solver, q, target, controls in fixtures:
            before = problem_identity(solver)
            result = physical.FixedPhysicalPotential(solver, target, **controls).evaluate(q)
            original = old_physical_closure(solver, q, target, controls)
            self.assertAlmostEqual(result['energyJoules'], original['energy'], delta=2e-12*max(1., abs(original['energy'])))
            np.testing.assert_allclose(result['gradientNewtons'][solver.free], original['gradient'], rtol=2e-12, atol=2e-12)
            self.assertEqual(problem_identity(solver), before)

    def test_contact_validation_is_fresh_and_required_before_response(self):
        solver, q = contact_cloth()
        cls, actual = type(solver.contact), type(solver.contact).validate_state
        calls = []
        def spy(contact, values):
            calls.append(values.copy())
            return actual(contact, values)
        potential = physical.FixedPhysicalPotential(solver, EMPTY)
        with patch.object(cls, 'validate_state', spy):
            first, second = potential.evaluate(q), potential.evaluate(q)
        self.assertEqual(len(calls), 2)
        self.assertEqual(first['componentCalls']['contact.validate_state'], 1)
        expected_kind = 'native-unprojected' if solver.contact.requires_guarded_metric else 'native-psd-projected-search-metric'
        self.assertEqual(second['componentSearchMatrixKinds']['contact'], expected_kind)
        def reject(contact, values):
            raise ValueError('independent inadmissible-state witness')
        with patch.object(cls, 'validate_state', reject), self.assertRaisesRegex(ValueError, 'inadmissible-state'):
            potential.evaluate(q)

    def test_cable_evaluation_validation_and_certificates_are_fresh_each_time(self):
        for route, activation in [('fixed', None), ('varying', .8), ('varying', 0.)]:
            solver, q = contact_cloth(route)
            options = dict(cable_targets=[[.0015, .0015]], cable_activation=[activation]) if route == 'varying' else {}
            potential = physical.FixedPhysicalPotential(solver, EMPTY, **options)
            actual = CableControl.evaluate
            calls = []
            def spy(control, values):
                calls.append(values.copy())
                return actual(control, values)
            with patch.object(CableControl, 'evaluate', spy):
                first = potential.evaluate(q)
                first['cableCertificate']['positionsSha256'] = 'forged external copy'
                second = potential.evaluate(q)
            self.assertEqual(len(calls), 2)
            self.assertEqual(second['componentCalls']['cable.evaluate'], 1)
            self.assertEqual(second['componentCalls']['cable.validate_response'], 1)
            self.assertEqual(second['cableCertificate']['positionsSha256'], second['positionsSha256'])
            if activation == 0.:
                self.assertEqual(second['componentEnergyJoules']['cable'], 0.)
            for total, base, rounding in [('energyJoules', 'cableEnergyJoules', 'energyAssemblyRoundingJoules'),
                                          ('gradientMaxAbsoluteNewtons', 'cableGradientMaxAbsoluteNewtons', 'gradientAssemblyRoundingNewtons')]:
                bounds = second['knownErrorBounds']
                self.assertEqual(fraction(bounds[total]), fraction(bounds[base])+fraction(bounds[rounding]))
            exact_energy = sum((F(v) for v in second['componentEnergyJoules'].values()), F())
            self.assertEqual(second['energyJoules'], float(exact_energy))
            self.assertEqual(fraction(second['knownErrorBounds']['energyAssemblyRoundingJoules']), abs(F(second['energyJoules'])-exact_energy))

    def test_inactive_vector_row_never_computes_overflowing_anchor_target_error(self):
        solver, q, _ = particle_solver()
        q[:] = np.finfo(float).max
        target = np.full((1, 3), -np.finfo(float).max)
        with np.errstate(over='raise', invalid='raise'):
            result = physical.FixedPhysicalPotential(solver, target, sewing_activation=[0.]).evaluate(q)
        self.assertEqual(result['energyJoules'], 0.)
        np.testing.assert_array_equal(result['gradientNewtons'], np.zeros(q.size))
        self.assertEqual(result['searchMatrixNewtonsPerMetre'].nnz, 0)

    def test_fixed_controls_descriptions_and_results_are_detached_and_immutable(self):
        solver, q, _ = particle_solver()
        target, activation = np.array([[.01, .02, .03]]), np.array([.7])
        potential = physical.FixedPhysicalPotential(solver, target, sewing_activation=activation)
        initial = potential.evaluate(q)
        target[:] = 99.; activation[:] = 0.
        description = potential.description(); description['controls']['sewingTargets'][0][0] = 123.
        initial['gradientNewtons'][:] = 55.
        initial['searchMatrixNewtonsPerMetre'].data[:] = 99.
        initial['componentEnergyJoules']['sewing'] = -999.
        second = potential.evaluate(q)
        self.assertNotEqual(potential.description()['controls']['sewingTargets'][0][0], 123.)
        self.assertFalse(np.all(second['gradientNewtons'] == 55.))
        self.assertGreater(second['energyJoules'], 0.)
        with self.assertRaises(AttributeError): potential._solver = solver
        with self.assertRaises(AttributeError): potential.__init__(solver, target)
        with self.assertRaises(AttributeError): del potential._weights

    def test_constructor_helper_parameter_mutation_cannot_change_caller_or_controls(self):
        solver, q = contact_cloth('varying')
        targets, activation = np.array([[.0015, .0015]]), np.array([.8])
        actual = VaryingCableControl.parameters
        before = targets.copy(), activation.copy()
        def mutate(control, values, weights):
            result = actual(control, values, weights)
            values.shape = (values.size,)
            return result
        with patch.object(VaryingCableControl, 'parameters', mutate), self.assertRaisesRegex(ValueError, 'mutated'):
            physical.FixedPhysicalPotential(solver, EMPTY, cable_targets=targets, cable_activation=activation)
        np.testing.assert_array_equal(targets, before[0]); np.testing.assert_array_equal(activation, before[1])
        self.assertEqual(targets.shape, (1, 2))

    def test_input_and_advanced_index_helper_mutations_are_rejected(self):
        solver, q, _ = particle_solver()
        potential = physical.FixedPhysicalPotential(solver, [[0., 0., 0.]])
        initial = q.copy()
        actual = ElasticDihedralBending.gradient
        def mutate(bending, values):
            result = actual(bending, values)
            values[0, 0] += .125
            return result
        with patch.object(ElasticDihedralBending, 'gradient', mutate), self.assertRaisesRegex(ValueError, 'mutated'):
            potential.evaluate(q)
        np.testing.assert_array_equal(q, initial)
        solver, q = normal_fixture()
        potential = physical.FixedPhysicalPotential(solver, [.17])
        initial = q.copy(); actual = physical.membrane_element_derivatives
        def mutate_indexed(values, poses, areas, materials):
            result = actual(values, poses, areas, materials)
            values[0, 0, 0] += .125
            return result
        with patch.object(physical, 'membrane_element_derivatives', mutate_indexed), self.assertRaisesRegex(ValueError, 'mutated'):
            potential.evaluate(q)
        np.testing.assert_array_equal(q, initial)

    def test_model_and_derived_control_identity_changes_reject(self):
        solver, q, _ = particle_solver('distance')
        potential = physical.FixedPhysicalPotential(solver, [.2])
        solver.compliance *= 2
        with self.assertRaisesRegex(ValueError, 'identity'): potential.evaluate(q)
        solver, q, _ = particle_solver('distance')
        potential = physical.FixedPhysicalPotential(solver, [.2])
        potential._distance.targets[0] = .3
        with self.assertRaisesRegex(ValueError, 'identity'): potential.evaluate(q)
        solver, q, _ = particle_solver()
        potential = physical.FixedPhysicalPotential(solver, [[0., 0., 0.]])
        actual = ElasticDihedralBending.gradient
        def alter_identity(bending, values):
            result = actual(bending, values)
            solver.mass[0] *= 2
            return result
        with patch.object(ElasticDihedralBending, 'gradient', alter_identity), self.assertRaisesRegex(ValueError, 'identity'):
            potential.evaluate(q)

    def test_bad_response_types_nonfinite_and_forged_fresh_certificate_reject(self):
        solver, q, _ = particle_solver()
        potential = physical.FixedPhysicalPotential(solver, [[0., 0., 0.]])
        for bad in [np.zeros(q.shape, dtype=np.complex128), np.full(q.shape, np.nan), np.zeros(2)]:
            def corrupt(bending, values): return bad
            with patch.object(ElasticDihedralBending, 'gradient', corrupt), self.assertRaises(ValueError):
                potential.evaluate(q)
        def complex_matrix(bending, values): return csr_matrix((q.size, q.size), dtype=np.complex128)
        with patch.object(ElasticDihedralBending, 'hessian', complex_matrix), self.assertRaises(ValueError):
            potential.evaluate(q)
        def complex_energy(bending, values): return 1.+2.j
        with patch.object(ElasticDihedralBending, 'energy', complex_energy), self.assertRaises(physical.PhysicalResponseFailure) as caught:
            potential.evaluate(q)
        self.assertEqual(caught.exception.component_calls['bending.energy'], 1)
        solver, q = contact_cloth('fixed')
        potential = physical.FixedPhysicalPotential(solver, EMPTY)
        actual = CableControl.evaluate
        def forged(control, values):
            result = actual(control, values)
            result['certificate']['positionsSha256'] = '0'*64
            return result
        with patch.object(CableControl, 'evaluate', forged), self.assertRaises(ValueError): potential.evaluate(q)

    def test_rejected_evaluations_retain_detached_attempted_component_counts(self):
        solver, q = contact_cloth()
        potential = physical.FixedPhysicalPotential(solver, EMPTY)
        cls = type(solver.contact)
        def fail(contact, values): raise ValueError('deliberate failed contact response')
        with patch.object(cls, 'gradient', fail):
            failures = []
            for _ in range(2):
                with self.assertRaises(physical.PhysicalResponseFailure) as caught:
                    potential.evaluate(q)
                failures.append(caught.exception)
            for failure in failures:
                self.assertEqual(failure.component_calls['contact.validate_state'], 1)
                self.assertEqual(failure.component_calls['contact.energy'], 1)
                self.assertEqual(failure.component_calls['contact.gradient'], 1)
                self.assertNotIn('contact.hessian', failure.component_calls)
            failures[0].component_calls['caller injected'] = 9
            self.assertNotIn('caller injected', failures[1].component_calls)
        with self.assertRaises(physical.PhysicalResponseFailure) as malformed:
            potential.evaluate(q.astype(np.float32))
        self.assertEqual(malformed.exception.component_calls, {})

    def test_whole_gradient_sum_rounds_once_under_large_component_cancellation(self):
        solver, q, controls = folded_control()
        potential = physical.FixedPhysicalPotential(solver, EMPTY, **controls)
        shape = q.shape
        # A detached assembly witness, not an asserted constitutive response.
        def constant(value):
            def supplied(component, positions): return np.full(shape, value, dtype=np.float64)
            return supplied
        original_membrane = physical.membrane_element_derivatives
        def zero_membrane(*arguments):
            energy, gradient, hessian = original_membrane(*arguments)
            return energy, np.zeros_like(gradient), hessian
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch.object(physical, 'membrane_element_derivatives', zero_membrane))
            stack.enter_context(patch.object(type(solver.bending), 'gradient', constant(float(2**53))))
            stack.enter_context(patch.object(type(potential._actuator), 'gradient', constant(1.)))
            stack.enter_context(patch.object(type(potential._grippers), 'gradient', constant(-float(2**53))))
            stack.enter_context(patch.object(type(solver.fold_barrier), 'gradient', constant(0.)))
            result = potential.evaluate(q)
        np.testing.assert_array_equal(result['gradientNewtons'], np.ones(q.size))
        self.assertEqual(fraction(result['knownErrorBounds']['gradientAssemblyRoundingNewtons']), 0)

    def test_malformed_states_and_unmatched_controls_reject(self):
        solver, q, _ = particle_solver()
        potential = physical.FixedPhysicalPotential(solver, [[0., 0., 0.]])
        class ArraySubclass(np.ndarray): pass
        for bad in [q.tolist(), q.astype(np.float32), q.view(ArraySubclass), q[:, :2], np.full(q.shape, np.inf)]:
            with self.assertRaises(ValueError): potential.evaluate(bad)
        for controls in [dict(fold_targets=[0.]), dict(fold_activation=[1.]),
                         dict(gripper_targets=[[0., 0., 0.]], gripper_activation=[1.]),
                         dict(cable_targets=[[.001, .001]], cable_activation=[1.])]:
            with self.assertRaises(ValueError): physical.FixedPhysicalPotential(solver, [[0., 0., 0.]], **controls)
        solver, q, controls = folded_control()
        with self.assertRaises(ValueError): physical.FixedPhysicalPotential(solver, EMPTY)
        solver, q = contact_cloth('varying')
        with self.assertRaises(ValueError): physical.FixedPhysicalPotential(solver, EMPTY)
        solver, q = normal_fixture(); q[2] = q[1]
        with self.assertRaises(ValueError): physical.FixedPhysicalPotential(solver, [.1]).evaluate(q)


if __name__ == '__main__':
    unittest.main()
