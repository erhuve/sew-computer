"""Exact feature geometry, full candidate construction and integration checks."""
from dataclasses import asdict
from fractions import Fraction as F
import itertools
import math
from types import SimpleNamespace
import unittest

import ipctk
import numpy as np

from solver_contact_work import (BarrierParameters, ContactTerm, WorkBudget,
                                 WorkPolicy, closest_feature, contact_work)
from solver_contact_work_control import ContactWorkControl, checked_change
from solver_contact_work_native import capture_endpoint
from solver_exact_contact import (PROFILE, build_exact_collisions, clamp_squared,
                                  inflation_radius, rest_distance)
from solver_rest_filtered_contact import RestFilteredSurfaceContact
from test_solver_contact_work_native import fixture


def classify(kind, points):
    return closest_feature(kind, tuple(tuple(float(x) for x in p) for p in points), WorkBudget())


def wire(points):
    q = np.array(points, dtype=np.float64)
    mesh = ipctk.CollisionMesh(q, np.array([[0, 1], [2, 3]], dtype=np.int64),
                              np.empty((0, 3), dtype=np.int64))
    candidates = ipctk.Candidates()
    candidates.ee_candidates = [ipctk.EdgeEdgeCandidate(0, 1)]
    return mesh, q, candidates


def build(mesh, q, candidates, activation=.1, minimum=.001):
    return build_exact_collisions(mesh, q, candidates, activation=float(activation),
                                  minimum=float(minimum), max_candidates=1000)


def native_terms(bucket):
    # Copy values before releasing each borrowed native item.
    result = []
    for i in range(len(bucket)):
        c = bucket[i]
        ids = tuple(int(getattr(c, k)) for k in ('vertex0_id', 'vertex1_id',
            'edge_id', 'face_id', 'edge0_id', 'edge1_id', 'vertex_id') if hasattr(c, k))
        feature = str(c.known_dtype()).split('.')[-1] if hasattr(c, 'known_dtype') else 'P_P'
        result.append((type(c).__name__, ids, feature, float(c.weight), float(c.dmin)))
        del c
    return result


class ExactFeatureGeometryTests(unittest.TestCase):
    def test_point_edge_rounded_projection_and_cancellation(self):
        e, d = 2.**-27, 2.**-52
        for points, expected in (
            (((1., 2*e, .125), (0., 0., 0.), (1., e, 0.)), 'P_E1'),
            (((1-d, -1., .125), (0., 0., 0.), (1+d, 1., 0.)), 'P_E0')):
            selected, value, ties = classify('ev', points)
            self.assertEqual(selected, expected)
            self.assertGreater(value, 0)
            self.assertNotEqual(str(ipctk.point_edge_distance_type(*map(np.array, points))).split('.')[-1], expected)

    def test_true_interior_edges_at_tiny_nonzero_angles(self):
        h = 2.**-8
        for exponent in (30, 80, 200, 300, 600):
            eps = 2.**-exponent
            points = ((-1, 0, 0), (1, 0, 0), (-1, -eps, h), (1, eps, h))
            selected, distance, ties = classify('ee', points)
            self.assertEqual((selected, distance, ties), ('EA_EB', F(1, 65536), ('EA_EB',)))
            # Analytic endpoint/segment minimum is strictly larger.
            self.assertGreater(F(eps)**2/(1+F(eps)**2), 0)
            for swap, reverse_a, reverse_b in itertools.product((False, True), repeat=3):
                a, b = points[:2][::(-1 if reverse_a else 1)], points[2:][::(-1 if reverse_b else 1)]
                self.assertEqual(classify('ee', b+a if swap else a+b)[1], distance)

    def test_skinny_face_interior_remains_a_face(self):
        for exponent in (30, 200, 300, 600):
            eps = 2.**-exponent
            points = ((.75, eps/4, .125), (0, 0, 0), (1, 0, 0), (1, eps, 0))
            for vertices in itertools.permutations(points[1:]):
                selected, distance, _ = classify('fv', (points[0],)+vertices)
                self.assertEqual((selected, distance), ('P_T', F(1, 64)))

    def test_boundary_ties_prefer_lower_dimension(self):
        self.assertEqual(classify('ev', ((0, 1, 0), (0, 0, 0), (1, 0, 0))),
                         ('P_E0', F(1), ('P_E0', 'P_E')))
        selected, distance, ties = classify('fv', ((0, 0, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0)))
        self.assertEqual((selected, distance), ('P_T0', F(1)))
        self.assertIn('P_T', ties)
        selected, distance, _ = classify('ee', ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)))
        self.assertEqual((selected, distance), ('EA0_EB0', F(1)))

    def test_degenerate_required_primitives_reject(self):
        for kind, points in (
            ('ev', ((0, 1, 0), (0, 0, 0), (0, 0, 0))),
            ('ee', ((0, 0, 0), (0, 0, 0), (1, 1, 0), (2, 1, 0))),
            ('fv', ((0, 1, 0), (0, 0, 0), (1, 0, 0), (2, 0, 0)))):
            with self.assertRaisesRegex(ValueError, 'Degenerate'):
                classify(kind, points)

    def test_clamp_and_outward_broad_phase_radius(self):
        for activation, minimum in ((.002, .0001), (.2, .1), (1e-60, 1e-60)):
            p = BarrierParameters.capture(activation, minimum, 1.)
            boundary = clamp_squared(activation, minimum)
            self.assertEqual(boundary, F(p.minimum_squared)+F(p.h))
            radius = inflation_radius(((activation, minimum),))
            self.assertGreaterEqual((2*F(radius))**2, boundary)
        self.assertNotEqual(clamp_squared(.002, .0001), F((.002+.0001)**2))

    def test_unrepresentable_rest_squared_distance_rejects(self):
        for square in (F(), F(1, 2**2200), F(2)**2200):
            with self.assertRaisesRegex(ValueError, 'Unrepresentable'):
                rest_distance(square)


class ExactCandidateConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ipctk.set_num_threads(1)

    def test_area_weighted_active_candidate_previously_omitted(self):
        h, eps = 2.**-9, 2.**-27
        q = np.array([[0., 0., 0.], [1., 0., 0.], [.25, h, 0.],
                      [1.25, h-eps, 0.], [0., -1., 0.], [.25, 1+h, 0.]])
        faces = np.array([[0, 1, 4], [2, 3, 5]], dtype=np.int64)
        edges = np.array([[0, 1], [0, 4], [1, 4], [2, 3], [2, 5], [3, 5]], dtype=np.int64)
        mesh = ipctk.CollisionMesh(q, edges, faces)
        candidates = ipctk.Candidates()
        candidates.ee_candidates = [ipctk.EdgeEdgeCandidate(0, 3)]
        minimum, activation = 2.**-13, h-2.**-31-2.**-13
        legacy = ipctk.NormalCollisions()
        legacy.use_area_weighting = True
        legacy.collision_set_type = ipctk.NormalCollisions.IPC
        legacy.build(candidates, mesh, q, activation, minimum)
        self.assertEqual(len(legacy), 0)
        bucket, record = build(mesh, q, candidates, activation, minimum)
        self.assertEqual((len(bucket), record['activeCandidates']), (1, 1))
        self.assertEqual(native_terms(bucket)[0][2], 'EA1_EB')
        self.assertEqual(native_terms(bucket)[0][3], 1/12)
        self.assertEqual(native_terms(bucket)[0][4], minimum)

    def test_forced_interior_feature_and_analytic_gradient(self):
        eps, h = 2.**-30, 2.**-8
        mesh, q, candidates = wire([[-1., 0., 0.], [1., 0., 0.], [-1., -eps, h], [1., eps, h]])
        bucket, _ = build(mesh, q, candidates)
        self.assertEqual(native_terms(bucket)[0][2], 'EA_EB')
        collision = bucket[0]
        gradient = np.asarray(collision.compute_distance_gradient(q.ravel())).reshape((-1, 3))
        np.testing.assert_array_equal(gradient, [[0, 0, -h], [0, 0, -h], [0, 0, h], [0, 0, h]])
        self.assertEqual(float(collision.compute_distance_hessian(q.ravel())[-1, -1]),
                         float(F(1, 2)-F(h)**2/(2*F(eps)**2)))
        del collision

    def test_nonfinite_forced_derivatives_reject_even_with_tiny_mollifier(self):
        for exponent in (200, 300, 600):
            eps, h = 2.**-exponent, 2.**-8
            mesh, q, candidates = wire([[-1., 0., 0.], [1., 0., 0.], [-1., -eps, h], [1., eps, h]])
            with self.assertRaisesRegex(ValueError, 'not representable|loses positive geometry'):
                build(mesh, q, candidates)

    def test_duplicate_candidates_keep_every_source_coefficient(self):
        mesh, q, candidates = wire([[0., 0., 0.], [1., 0., 0.], [0., .01, 0.], [1., .011, 0.]])
        candidates.ee_candidates = [ipctk.EdgeEdgeCandidate(0, 1)]*3
        bucket, record = build(mesh, q, candidates)
        terms = native_terms(bucket)
        self.assertEqual((len(terms), record['activeCandidates']), (3, 3))
        expected = .25*(mesh.edge_area(0)+mesh.edge_area(1))
        self.assertTrue(all(term[3].hex() == expected.hex() for term in terms))
        potential = ipctk.BarrierPotential(.1, 10000., use_physical_barrier=True)
        candidates.ee_candidates = [ipctk.EdgeEdgeCandidate(0, 1)]
        one, _ = build(mesh, q, candidates)
        self.assertAlmostEqual(potential(bucket, mesh, q), 3*potential(one, mesh, q), delta=1e-12)
        np.testing.assert_allclose(potential.gradient(bucket, mesh, q), 3*potential.gradient(one, mesh, q), rtol=2e-15, atol=1e-14)

    def test_conversion_preserves_vv_ev_fv_source_roles(self):
        q = np.array([[0., 0., .01], [0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        edges = np.array([[1, 2], [1, 3], [2, 3]], dtype=np.int64)
        faces = np.array([[1, 2, 3]], dtype=np.int64)
        mesh = ipctk.CollisionMesh(q, edges, faces)
        candidates = ipctk.Candidates()
        candidates.vv_candidates = [ipctk.VertexVertexCandidate(0, 1)]
        candidates.ev_candidates = [ipctk.EdgeVertexCandidate(0, 0)]
        candidates.fv_candidates = [ipctk.FaceVertexCandidate(0, 0)]
        bucket, record = build(mesh, q, candidates)
        rows = native_terms(bucket)
        self.assertEqual(record['activeCandidates'], 3)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r[0] == 'VertexVertexNormalCollision' for r in rows))
        self.assertEqual(sorted(r[3] for r in rows), sorted((
            .5*(mesh.vertex_area(0)+mesh.vertex_area(1)),
            .5*mesh.vertex_area(0), .25*mesh.vertex_area(0))))

    def test_bad_candidate_index_and_shared_vertices_reject_before_native_lookup(self):
        mesh, q, candidates = wire([[0., 0., 0.], [1., 0., 0.], [0., .01, 0.], [1., .011, 0.]])
        for value in (-1, 10000):
            bad = ipctk.EdgeEdgeCandidate(0, 1)
            bad.edge0_id = value
            candidates.ee_candidates = [bad]
            with self.assertRaisesRegex(ValueError, 'In-range'):
                build(mesh, q, candidates)
        candidates.ee_candidates = [ipctk.EdgeEdgeCandidate(0, 0)]
        with self.assertRaisesRegex(ValueError, 'Distinct'):
            build(mesh, q, candidates)

    def test_raw_state_budget_and_minimum_admission(self):
        mesh, q, candidates = wire([[0., 0., 0.], [1., 0., 0.], [0., .01, 0.], [1., .011, 0.]])
        for value in (q.tolist(), q.astype(np.float32), q.astype(int)):
            with self.assertRaisesRegex(ValueError, 'Raw finite binary64'):
                build(mesh, value, candidates)
        candidates.ee_candidates = [ipctk.EdgeEdgeCandidate(0, 1)]*2
        with self.assertRaisesRegex(ValueError, 'budget'):
            build_exact_collisions(mesh, q, candidates, activation=.1, minimum=.001, max_candidates=1)
        with self.assertRaisesRegex(ValueError, 'minimum separation'):
            build(mesh, q, candidates, minimum=.01)

    def test_parallel_heuristic_jump_becomes_quadratic_endpoint_work(self):
        h, eps, delta = 2.**-4, 2.**-30, 2.**-40
        mesh, rest, candidates = wire([[0., 0., 0.], [1., 0., 0.], [0., h, 0.], [1., h+eps, 0.]])
        captures = []
        for x in (-delta, delta):
            q = rest.copy(); q[2:, 0] += x
            bucket, _ = build(mesh, q, candidates)
            c = bucket[0]
            feature = str(c.known_dtype()).split('.')[-1]
            captures.append((ContactTerm(0, 'ee', (0, 1, 2, 3), (('edge0_id', 0), ('edge1_id', 1)),
                feature, tuple(tuple(float(v) for v in p) for p in q), float(c.weight), float(c.eps_x),
                BarrierParameters.capture(.1, .001, 1.)),))
            del c
        self.assertEqual(captures[0][0].feature, 'EA0_EB0')
        self.assertEqual(captures[1][0].feature, 'EA_EB0')
        result = contact_work(*captures)
        self.assertGreater(result.value, 0)
        self.assertLess(result.value, 1e-30)


class ExactContactIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ipctk.set_num_threads(1)

    def contact(self, original):
        return RestFilteredSurfaceContact(original.rest_positions, original.faces,
            activation_distance_m=.002, minimum_distance_m=.0001, stiffness=10000.,
            feature_profile=PROFILE)

    def test_previously_rejected_parallel_fixture_captures_all_contributions(self):
        old, q = fixture(rotated=False)
        with self.assertRaisesRegex(ValueError, 'Native closest feature'):
            capture_endpoint(old, q)
        exact = self.contact(old)
        endpoint = capture_endpoint(exact, q)
        certificate = exact.feature_certificate(q)
        self.assertEqual(len(endpoint.terms), sum(b['activeCandidates'] for b in certificate['buckets']))
        self.assertTrue(endpoint.terms)
        self.assertTrue(np.isfinite(exact.gradient(q)).all())
        self.assertTrue(np.isfinite(exact.hessian(q).data).all())
        profile = exact.profile()
        self.assertEqual(profile['featureSelection']['profile'], PROFILE)
        self.assertNotIn('featureSelection', old.profile())

    def test_contact_work_control_binds_the_new_model_and_actual_endpoints(self):
        old, q = fixture(rotated=False)
        exact = self.contact(old)
        control = ContactWorkControl(exact, asdict(WorkPolicy()))
        end = q.copy(); end[4:, 2] += 1e-12
        record = checked_change(control, exact, q, end)
        self.assertLess(record['changeJoules'], 0)
        self.assertEqual(record['start']['termCount'], len(capture_endpoint(exact, q).terms))
        with self.assertRaises(ValueError):
            control.check(old)
        for name, value in (('_feature_profile', 'native'), ('_exact_inflation_radius', 0.)):
            with self.assertRaises(AttributeError):
                setattr(exact, name, value)

    def test_exact_rest_parameters_are_inactive_and_certificate_is_detached(self):
        from test_solver_contact_range_adversarial import square_grid
        rest, faces = square_grid(3)
        exact = RestFilteredSurfaceContact(rest, faces, activation_distance_m=.01,
            minimum_distance_m=.0001, stiffness=10000., feature_profile=PROFILE)
        self.assertEqual(exact.energy(rest), 0.)
        first = exact.feature_certificate(rest)
        first['definition']['profile'] = 'corrupted'
        self.assertEqual(exact.feature_certificate(rest)['definition']['profile'], PROFILE)

    def test_profile_is_explicit_and_legacy_certificate_rejects(self):
        old, q = fixture()
        with self.assertRaisesRegex(ValueError, 'not enabled'):
            old.feature_certificate(q)
        with self.assertRaisesRegex(ValueError, 'feature profile'):
            RestFilteredSurfaceContact(old.rest_positions, old.faces,
                activation_distance_m=.002, minimum_distance_m=.0001, stiffness=10000.,
                feature_profile=True)

    def test_adapter_conversion_is_declared_and_bounded_work_stays_strict(self):
        old, q = fixture()
        exact = RestFilteredSurfaceContact(old.rest_positions.tolist(), old.faces,
            activation_distance_m=.002, minimum_distance_m=.0001, stiffness=10000.,
            feature_profile=PROFILE)
        np.testing.assert_array_equal(exact.rest_positions, old.rest_positions)
        self.assertEqual(exact.energy(q.tolist()), exact.energy(q))
        self.assertIn('pre-conversion', exact.profile()['inputConversion'])
        with self.assertRaisesRegex(ValueError, 'Raw native binary64'):
            capture_endpoint(exact, q.tolist())

    def test_state_caches_do_not_change_model_identity_but_feature_law_does(self):
        from solver_temporal_control import problem_identity
        old, q = fixture()
        exact = self.contact(old)
        solver = SimpleNamespace(contact=exact)
        identity = problem_identity(solver)
        first = exact.feature_certificate(q)
        end = q.copy(); end[4:, 2] += .00001
        second = exact.feature_certificate(end)
        self.assertNotEqual(first, second)
        self.assertEqual(problem_identity(solver), identity)
        # Bypass normal immutability to verify that identity still catches a
        # changed model definition, rather than excluding all new fields.
        object.__setattr__(exact, '_exact_feature_definition', '{}')
        self.assertNotEqual(problem_identity(solver), identity)

    def test_actual_guarded_motion_with_exact_contact_and_work(self):
        import newton
        import warp as wp
        from solver_adaptive_contact import adaptive_contact_step
        from solver_global_sewing import GlobalSewingSolver
        from solver_temporal_control import problem_identity
        from test_solver_temporal_control import declaration
        old, q = fixture(rotated=False)
        exact = self.contact(old)
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
            vel=wp.vec3(0, 0, 0), vertices=exact.rest_positions.tolist(),
            indices=exact.faces.reshape(-1).tolist(), density=.2,
            tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=.001, edge_kd=0)
        builder.set_coloring([[i] for i in range(len(q))])
        solver = GlobalSewingSolver(builder.finalize(device='cpu'), [], 1e-8,
            contact=exact, contact_work_policy=asdict(WorkPolicy()))
        identity = problem_identity(solver)
        empty = np.empty((0, 3))
        end, velocity, report = adaptive_contact_step(solver, q, q*0, empty, empty, 1e-5,
            max_depth=1, max_attempts=3, max_evaluations=64, temporal_policy=declaration())
        self.assertTrue(report['complete'], report.get('reason'))
        self.assertEqual(len(report['acceptedSteps']), 2)
        self.assertGreater(np.max(np.abs(end-q)), 0.)
        self.assertTrue(np.isfinite(velocity).all())
        self.assertEqual(problem_identity(solver), identity)
        for row in report['acceptedSteps']:
            self.assertEqual(row['step']['boundedContactWork'],
                             row['step']['energyBalance']['boundedContactWork'])


if __name__ == '__main__':
    unittest.main()
