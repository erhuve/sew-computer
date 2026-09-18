import unittest
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

from solver_energy_balance import global_energy_transition
from solver_bending import ElasticDihedralBending


class EnergyBalanceAdversarialTests(unittest.TestCase):
    def fixture(self, pinned=False):
        return SimpleNamespace(
            bending=ElasticDihedralBending(2, np.empty((0, 4), dtype=int), [], [], []),
            mass=np.array([0.0 if pinned else 2.0, 3.0]),
            active=np.array([not pinned, True]),
            sewing=csr_matrix([[1.0, -1.0]]), compliance=.04,
            poses=np.empty((0, 2, 2)), faces=np.empty((0, 3), dtype=int),
            areas=np.empty(0), materials=np.empty((0, 3)),
        )

    def solve(self, solver, previous, previous_velocity, targets, timestep):
        stiffness = solver.sewing.toarray().T @ solver.sewing.toarray() / solver.compliance
        matrix = np.diag(solver.mass / timestep ** 2) + stiffness
        rhs = solver.mass[:, None] / timestep ** 2 * (previous + timestep * previous_velocity)
        rhs += solver.sewing.T @ targets / solver.compliance
        current = previous.copy()
        active, fixed = np.flatnonzero(solver.active), np.flatnonzero(~solver.active)
        rhs = rhs[active] - matrix[np.ix_(active, fixed)] @ previous[fixed]
        current[active] = np.linalg.solve(matrix[np.ix_(active, active)], rhs)
        return current, (current - previous) / timestep

    def check_quadratic_balance(self, previous_targets, targets, pinned=False):
        solver = self.fixture(pinned)
        previous = np.array([[.2, -.1, .3], [1.1, .4, -.2]])
        old_velocity = np.array([[.3, -.2, .1], [-.2, .5, .1]])
        old_velocity[~solver.active] = 0
        timestep = .05
        current, velocity = self.solve(solver, previous, old_velocity, targets, timestep)
        report = global_energy_transition(solver, previous, current, old_velocity, velocity,
                                          previous_targets, targets, timestep)
        old_residual = solver.sewing @ previous - previous_targets
        new_residual = solver.sewing @ current - targets
        before = np.sum(solver.mass[:, None] * old_velocity ** 2) / 2 + np.sum(old_residual ** 2) / (2 * solver.compliance)
        after = np.sum(solver.mass[:, None] * velocity ** 2) / 2 + np.sum(new_residual ** 2) / (2 * solver.compliance)
        work = (np.sum((solver.sewing @ previous - targets) ** 2) - np.sum(old_residual ** 2)) / (2 * solver.compliance)
        dissipation = (np.sum(solver.mass[:, None] * (velocity - old_velocity) ** 2) / 2
                       + np.sum((solver.sewing @ (current - previous)) ** 2) / (2 * solver.compliance))
        self.assertAlmostEqual(report['mechanicalChangeJoules'], after - before, places=11)
        self.assertAlmostEqual(report['targetParameterWorkJoules'], work, places=11)
        self.assertAlmostEqual(report['mechanicalChangeMinusTargetWorkJoules'], -dissipation, places=11)
        self.assertFalse(report['accepted'])
        return report

    def test_fixed_target_exact_backward_euler_dissipation(self):
        targets = np.array([[.1, -.2, .05]])
        self.check_quadratic_balance(targets, targets)

    def test_moving_target_positive_and_negative_work(self):
        aligned = np.array([[-.9, -.5, .5]])
        zero = np.zeros((1, 3))
        positive = self.check_quadratic_balance(aligned, zero)
        negative = self.check_quadratic_balance(zero, aligned)
        self.assertGreater(positive['targetParameterWorkJoules'], 0)
        self.assertLess(negative['targetParameterWorkJoules'], 0)

    def test_fixed_particle_exact_dissipation(self):
        self.check_quadratic_balance(np.zeros((1, 3)), np.array([[.1, .2, .3]]), pinned=True)

    def test_frame_invariance(self):
        solver = self.fixture()
        previous = np.array([[.2, -.1, .3], [1.1, .4, -.2]])
        old_velocity = np.array([[.3, -.2, .1], [-.2, .5, .1]])
        previous_targets = np.array([[.1, -.2, .3]])
        targets = np.array([[.2, .1, -.1]])
        timestep = .05
        current, velocity = self.solve(solver, previous, old_velocity, targets, timestep)
        rotation = np.linalg.qr(np.random.default_rng(241).normal(size=(3, 3)))[0]
        original = global_energy_transition(solver, previous, current, old_velocity, velocity,
                                            previous_targets, targets, timestep)
        transformed = global_energy_transition(solver, previous @ rotation + [3, 1, -2],
            current @ rotation + [3, 1, -2], old_velocity @ rotation, velocity @ rotation,
            previous_targets @ rotation, targets @ rotation, timestep)
        for key, value in original.items():
            if key.endswith('Joules'):
                self.assertAlmostEqual(value, transformed[key], places=11)

    def test_nonfinite_shapes_timestep_and_inconsistent_velocity_rejected(self):
        solver = self.fixture()
        positions = np.array([[0., 0., 0.], [1., 0., 0.]])
        velocity = np.zeros_like(positions)
        targets = np.zeros((1, 3))
        arguments = [positions, positions, velocity, velocity, targets, targets, .01]
        for index in range(6):
            for corrupt in (np.full_like(arguments[index], np.nan), np.zeros((1,))):
                changed = list(arguments)
                changed[index] = corrupt
                with self.subTest(index=index, shape=corrupt.shape), self.assertRaises(ValueError):
                    global_energy_transition(solver, *changed)
        for timestep in (0, -.1, np.nan, np.inf):
            with self.subTest(timestep=timestep), self.assertRaises(ValueError):
                global_energy_transition(solver, *arguments[:-1], timestep)
        changed = list(arguments)
        changed[3] = np.ones_like(velocity)
        with self.assertRaisesRegex(ValueError, 'reconstructed'):
            global_energy_transition(solver, *changed)

    def test_fixed_particle_motion_or_velocity_rejected(self):
        solver = self.fixture(pinned=True)
        positions = np.array([[0., 0., 0.], [1., 0., 0.]])
        velocity = np.zeros_like(positions)
        targets = np.zeros((1, 3))
        arguments = [positions, positions, velocity, velocity, targets, targets, .01]
        for index in (1, 2, 3):
            changed = list(arguments)
            changed[index] = changed[index].copy()
            changed[index][0, 0] += .1
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'stationary fixed'):
                global_energy_transition(solver, *changed)

    def test_unconverged_energy_gain_is_reported_without_acceptance(self):
        solver = self.fixture()
        previous = np.zeros((2, 3))
        current = np.array([[1., 0., 0.], [-1., 0., 0.]])
        targets = np.zeros((1, 3))
        report = global_energy_transition(solver, previous, current, previous, current,
                                          targets, targets, 1.)
        self.assertGreater(report['mechanicalChangeMinusTargetWorkJoules'], 0)
        self.assertFalse(report['accepted'])

    def test_membrane_transition_matches_independent_gram_energy(self):
        solver = SimpleNamespace(
            bending=ElasticDihedralBending(3, np.empty((0, 4), dtype=int), [], [], []),
            mass=np.array([.2, .3, .4]), active=np.ones(3, dtype=bool),
            sewing=csr_matrix([[1., -.4, -.6]]), compliance=.04,
            poses=np.eye(2)[None], faces=np.array([[0, 1, 2]]),
            areas=np.array([.5]), materials=np.array([[4., 7., 0.]]),
        )
        previous = np.array([[.1, -.1, .2], [1.2, .1, .3], [.3, .8, -.1]])
        current = np.array([[.05, -.08, .21], [1.17, .13, .32], [.32, .84, -.07]])
        old_velocity = np.array([[.1, .2, .3], [.1, -.2, .3], [.3, .2, -.1]])
        timestep = .03
        velocity = (current - previous) / timestep
        old_target, target = np.array([[.1, -.2, .3]]), np.array([[.2, .1, -.1]])

        def membrane_energy(positions):
            edges = (positions[1:] - positions[0]).T
            gram = edges.T @ edges
            area_ratio = np.sqrt(np.linalg.det(gram))
            return .5 * (2 * (np.trace(gram) - 2) + 5.5 * (area_ratio - 1) ** 2
                         - 4 * (area_ratio - 1))

        report = global_energy_transition(solver, previous, current, old_velocity, velocity,
                                          old_target, target, timestep)
        change = membrane_energy(current) - membrane_energy(previous)
        self.assertAlmostEqual(report['membraneChangeJoules'], change, places=13)
        for positions, endpoint in ((previous, 'Before'), (current, 'After')):
            selected_target = old_target if endpoint == 'Before' else target
            residual = positions[0] - .4 * positions[1] - .6 * positions[2] - selected_target[0]
            self.assertAlmostEqual(report['sewing' + endpoint + 'Joules'], residual @ residual / .08, places=12)
        kinetic_change = np.sum(solver.mass[:, None] * (velocity ** 2 - old_velocity ** 2)) / 2
        self.assertAlmostEqual(report['mechanicalChangeJoules'], change + kinetic_change
                               + report['sewingAfterJoules'] - report['sewingBeforeJoules'], places=12)


if __name__ == '__main__':
    unittest.main()
