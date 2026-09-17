import unittest

import numpy as np

from solver_membrane_hessian import membrane_element_derivatives


class MembraneAdversarialTests(unittest.TestCase):
    def test_heterogeneous_batch_matches_individual_derivatives(self):
        generator = np.random.default_rng(733)
        count = 17
        positions = generator.normal(size=(count, 3, 3)) * .1
        poses = generator.normal(size=(count, 2, 2)) * 5 + np.eye(2) * 8
        areas = generator.uniform(.001, .02, count)
        materials = np.column_stack((generator.uniform(100., 10000., count),
                                     generator.uniform(100., 20000., count), np.zeros(count)))
        for projected in (False, True):
            batch = membrane_element_derivatives(positions, poses, areas, materials,
                                                 project_psd=projected)
            for element in range(count):
                individual = membrane_element_derivatives(
                    positions[element:element + 1], poses[element:element + 1],
                    areas[element:element + 1], materials[element:element + 1],
                    project_psd=projected,
                )
                for actual, expected in zip(batch, individual):
                    np.testing.assert_allclose(actual[element], expected[0], rtol=1e-11, atol=1e-7)
            np.testing.assert_allclose(batch[2], batch[2].transpose(0, 2, 1), atol=1e-7)
            if projected:
                eigenvalues = np.linalg.eigvalsh(batch[2])
                scale = np.maximum(1, np.max(np.abs(eigenvalues), axis=1))
                self.assertTrue(np.all(eigenvalues[:, 0] >= -1e-12 * scale))
            else:
                numerical = np.zeros_like(batch[2])
                for coordinate in range(9):
                    offset = np.zeros_like(positions)
                    offset.reshape((count, 9))[:, coordinate] = 1e-7
                    positive = membrane_element_derivatives(positions + offset, poses, areas, materials)[1]
                    negative = membrane_element_derivatives(positions - offset, poses, areas, materials)[1]
                    numerical[:, :, coordinate] = (positive - negative) / 2e-7
                errors = np.linalg.norm(numerical - batch[2], axis=(1, 2))
                scales = np.linalg.norm(numerical, axis=(1, 2))
                self.assertTrue(np.all(errors / scales < 1e-7))

    def test_randomized_derivatives_and_rigid_covariance(self):
        generator = np.random.default_rng(9231)
        for _ in range(20):
            positions = generator.normal(size=(1, 3, 3)) * .1
            poses = generator.normal(size=(1, 2, 2)) * 5
            if abs(np.linalg.det(poses[0])) < .2:
                continue
            areas = np.array([.005])
            materials = np.array([[10000., 7000., 0.]])
            arguments = (positions, poses, areas, materials)
            _, _, hessian = membrane_element_derivatives(*arguments)
            numerical = np.zeros((9, 9))
            for coordinate in range(9):
                offset = np.zeros_like(positions)
                offset.ravel()[coordinate] = 1e-7
                positive = membrane_element_derivatives(positions + offset, poses, areas, materials)[1]
                negative = membrane_element_derivatives(positions - offset, poses, areas, materials)[1]
                numerical[:, coordinate] = (positive - negative)[0] / 2e-7
            self.assertLess(np.linalg.norm(numerical - hessian[0]) / np.linalg.norm(numerical), 1e-7)
            rotation = np.linalg.qr(generator.normal(size=(3, 3)))[0]
            transform = np.kron(np.eye(3), rotation.T)
            for project in (False, True):
                energy, gradient, hessian = membrane_element_derivatives(*arguments, project_psd=project)
                moved = membrane_element_derivatives(
                    positions @ rotation + np.array([3, -4, 2]), poses, areas, materials,
                    project_psd=project,
                )
                np.testing.assert_allclose(moved[0], energy, rtol=1e-9, atol=1e-8)
                np.testing.assert_allclose(moved[1][0], transform @ gradient[0], rtol=1e-8, atol=1e-7)
                np.testing.assert_allclose(moved[2][0], transform @ hessian[0] @ transform.T, rtol=1e-8, atol=1e-6)
                np.testing.assert_allclose(hessian[0] @ np.tile(np.eye(3), (3, 1)), 0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
