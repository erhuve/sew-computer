import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.sparse import csr_matrix

from solver_normal_sewing import NormalOffsetSewing
from solver_triangle_sweep import triangle_sweep_safe, verify_triangle_sweep_exact


class TriangleSweepTests(unittest.TestCase):
    def fixture(self):
        start = np.array([[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.],
                          [.003, .003, .002], [.013, .003, .002], [.003, .013, .002]])
        end = start.copy()
        end[:, :2] *= -1
        return start, end, np.array([[0, 1, 2], [3, 4, 5]])

    def test_collapsing_seam_frames_have_valid_endpoint_seams_and_contact(self):
        from solver_rest_filtered_contact import RestFilteredSurfaceContact
        start, end, faces = self.fixture()
        seam = NormalOffsetSewing(csr_matrix([[-.4, -.3, -.3, 1., 0., 0.]]),
                                 [.002], 1e-8, [[0, 1, 2]], [1])
        np.testing.assert_allclose(seam.residual(start), 0., atol=1e-14)
        np.testing.assert_allclose(seam.residual(end), 0., atol=1e-14)
        contact = RestFilteredSurfaceContact(start, faces, activation_distance_m=.001,
            minimum_distance_m=.0001, stiffness=10000,
            ccd_profile="temporal-separation-tight-inclusion")
        self.assertTrue(contact.path_certificate(start, end)["safe"])
        np.testing.assert_allclose((start + end)[:3] / 2, 0., atol=0)
        self.assertFalse(triangle_sweep_safe(start, end, faces))
        with self.assertRaisesRegex(ValueError, "degenerates"):
            verify_triangle_sweep_exact(start, end, faces)

    def test_off_grid_tangent_collapse_and_scale_covariance(self):
        for root in (.137391, .371283, .823717):
            def points(t):
                delta = t - root
                return np.array([[0., 0., 0.], [1., delta, 0.], [1 + delta, delta, 0.]])
            for scale in (1e-6, 1., 1e6):
                for offset in (np.zeros(3), np.array([2., -3., 4.]) * scale):
                    start, end = [points(t) * scale + offset for t in (0., 1.)]
                    self.assertFalse(triangle_sweep_safe(start, end, [[0, 1, 2]]))
                    with self.assertRaises(ValueError):
                        verify_triangle_sweep_exact(start, end, [[0, 1, 2]])

    def test_safe_rotating_normals_are_not_required_to_keep_initial_direction(self):
        start = np.array([[0., 0., 0.], [.02, 0., 0.], [0., .03, 0.]])
        angle = 2.2
        rotation = np.array([[1., 0., 0.], [0., np.cos(angle), -np.sin(angle)],
                             [0., np.sin(angle), np.cos(angle)]])
        end = start @ rotation + [.01, -.02, .003]
        self.assertTrue(triangle_sweep_safe(start, end, [[0, 1, 2]]))
        result = verify_triangle_sweep_exact(start, end, [[0, 1, 2]])
        self.assertGreaterEqual(result["verifiedLeaves"], 1)

    def test_random_guarded_paths_have_independent_exact_proofs(self):
        generator = np.random.default_rng(92183)
        passed = 0
        for _ in range(50):
            start = generator.normal(size=(3, 3))
            end = start + .4 * generator.normal(size=(3, 3))
            if triangle_sweep_safe(start, end, [[0, 1, 2]]):
                verify_triangle_sweep_exact(start, end, [[0, 1, 2]])
                passed += 1
        self.assertGreater(passed, 40)

    def test_malformed_budget_and_endpoint_degeneracy_fail_closed(self):
        start, end, faces = self.fixture()
        for options in ({"max_depth": True}, {"max_depth": 25}, {"max_intervals": 0},
                        {"max_intervals": 1000001}):
            for function in (triangle_sweep_safe, verify_triangle_sweep_exact):
                with self.assertRaises(ValueError):
                    function(start, end, faces, **options)
        for indices in ([[True, 1, 2]], [[False, 1, 2]], [[0., 1., 2.]], [[0, 0, 2]], [[0, 1, 99]]):
            for function in (triangle_sweep_safe, verify_triangle_sweep_exact):
                with self.assertRaises(ValueError):
                    function(start, end, indices)
        self.assertFalse(triangle_sweep_safe(start, np.zeros_like(end), faces))
        with self.assertRaises(ValueError):
            verify_triangle_sweep_exact(start, np.zeros_like(end), faces)
        self.assertFalse(triangle_sweep_safe(start, start, faces, max_intervals=1))

    def solver_fixture(self):
        import newton
        import warp as wp
        from solver_global_sewing import GlobalSewingSolver
        start, end, faces = self.fixture()
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
            vel=wp.vec3(0, 0, 0), vertices=start.tolist(), indices=faces.ravel().tolist(),
            density=.2, tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=0, edge_kd=0)
        builder.set_coloring([[i] for i in range(6)])
        solver = GlobalSewingSolver(builder.finalize(device="cpu"),
            [{0: -.4, 1: -.3, 2: -.3, 3: 1.}], 1e-8, sewing_mode="normal-offset",
            sewing_frame_faces=[[0, 1, 2]], sewing_sides=[1])
        return solver, start, end

    def test_global_search_and_final_publication_reject_collapse_without_contact_or_hinges(self):
        solver, previous, end = self.solver_fixture()
        self.assertIsNone(solver.contact)
        self.assertEqual(len(solver.bending.indices), 0)
        def fake_descent(evaluate, start, maximum, hessian, objective, **options):
            self.assertTrue(np.isfinite(evaluate(end.ravel())).all())
            self.assertTrue(np.isinf(options["energy_change_function"](start, end.ravel())))
            return OptimizeResult(x=end.ravel(), success=True, status=1, message="probe", nfev=1)
        with patch("solver_global_sewing._direct_descent", fake_descent), \
                self.assertRaisesRegex(ValueError, "triangle path"):
            solver.step(previous, np.zeros_like(previous), np.array([.002]), .01)

    def test_both_optimizer_and_physical_paths_are_checked(self):
        for failed_segment in (0, 1):
            solver, previous, _ = self.solver_fixture()
            checked = []
            def guard(start, end, faces):
                checked.append((start.copy(), end.copy()))
                return len(checked) - 1 != failed_segment
            def fake_descent(evaluate, start, maximum, hessian, objective, **options):
                intermediate = start.copy() + .0001
                candidate = intermediate.copy() + .0002
                self.assertTrue(np.isinf(options["energy_change_function"](intermediate, candidate)))
                np.testing.assert_allclose(checked[0][0].ravel(), intermediate)
                if failed_segment:
                    np.testing.assert_array_equal(checked[1][0], previous)
                return OptimizeResult(x=start, success=False, status=0, message="probe", nfev=1)
            with patch("solver_triangle_sweep.triangle_sweep_safe", guard), \
                    patch("solver_global_sewing._direct_descent", fake_descent):
                solver.step(previous, np.zeros_like(previous), np.array([.002]), .01)
            self.assertEqual(len(checked), failed_segment + 2)  # includes final output check

    def test_membrane_only_least_squares_cannot_bypass_search_guards(self):
        normal_solver, previous, _ = self.solver_fixture()
        # Reuse valid physical model data with vector sewing and no optional
        # bending/contact/normal-frame guard to exercise the formerly open path.
        normal_solver.sewing_mode = "vector"
        with self.assertRaisesRegex(ValueError, "swept nondegeneracy"):
            normal_solver.step(previous, np.zeros_like(previous),
                               np.array([[0., 0., .002]]), .01, linear_solver="lsmr")


if __name__ == "__main__":
    unittest.main()
