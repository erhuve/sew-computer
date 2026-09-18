import numpy as np
from scipy.sparse import csr_matrix, diags

from solver_bending import ElasticDihedralBending


class LocalAngularFoldBarrier:
    def __init__(self, vertex_count, indices, activation_angle=np.pi / 2, stiffness_joules=1.):
        count = len(indices)
        self._geometry = ElasticDihedralBending(vertex_count, indices, np.zeros(count),
                                               np.ones(count), np.ones(count))
        self.vertex_count = vertex_count
        self.indices = self._geometry.indices.copy()
        raw_angles = np.asarray(activation_angle, dtype=float)
        raw_stiffness = np.asarray(stiffness_joules, dtype=float)
        if (not np.isfinite(raw_angles).all() or not np.isfinite(raw_stiffness).all()
                or np.any(raw_angles <= 0) or np.any(raw_angles >= np.pi - 1e-8)
                or np.any(raw_stiffness < 0)):
            raise ValueError("Finite local fold-barrier parameters with 0 < activation < pi required")
        self.activation_angles = np.broadcast_to(raw_angles, (count,)).copy()
        self.stiffness_joules = np.broadcast_to(raw_stiffness, (count,)).copy()
        for values in (self.indices, self.activation_angles, self.stiffness_joules,
                       self._geometry.indices, self._geometry.weights):
            values.setflags(write=False)

    def _angles(self, positions):
        if not len(self.indices):
            self._geometry._positions(positions)
            return np.empty(0, dtype=np.longdouble)
        return self._geometry._geometry(np.asarray(positions, dtype=np.longdouble))[0]

    def _terms(self, angles):
        widths = np.longdouble(np.pi) - self.activation_angles.astype(np.longdouble)
        scaled_gaps = (np.longdouble(np.pi) - np.abs(angles)) / widths
        active = (scaled_gaps < 1) & (self.stiffness_joules > 0)
        return widths, scaled_gaps, active

    def _energies(self, scaled_gaps, active):
        result = np.zeros(len(self.indices), dtype=np.longdouble)
        offsets = scaled_gaps[active] - 1
        result[active] = -self.stiffness_joules[active] * offsets ** 2 * np.log1p(offsets)
        return result

    def energy(self, positions):
        _, scaled_gaps, active = self._terms(self._angles(positions))
        result = float(np.sum(self._energies(scaled_gaps, active)))
        if not np.isfinite(result):
            raise ValueError("Nonfinite local fold-barrier energy")
        return result

    def residual(self, positions):
        _, scaled_gaps, active = self._terms(self._angles(positions))
        result = np.asarray(np.sqrt(2 * self._energies(scaled_gaps, active)), dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite local fold-barrier residual")
        return result

    def _angle_gradient(self, positions):
        angles = self._angles(positions)
        widths, scaled_gaps, active = self._terms(angles)
        derivative = np.zeros(len(self.indices))
        offsets = scaled_gaps[active] - 1
        derivative[active] = (np.sign(angles[active]) * self.stiffness_joules[active] / widths[active]
                              * (2 * offsets * np.log1p(offsets) + offsets ** 2 / scaled_gaps[active]))
        return derivative

    def jacobian(self, positions):
        residual = self.residual(positions)
        derivative = self._angle_gradient(positions)
        factors = np.divide(derivative, residual, out=np.zeros_like(derivative), where=residual > 0)
        result = (diags(factors) @ self._geometry.jacobian(positions)).tocsr()
        if not np.isfinite(result.data).all():
            raise ValueError("Nonfinite local fold-barrier residual Jacobian")
        return result

    def gradient(self, positions):
        derivative = self._angle_gradient(positions)
        result = np.asarray(self._geometry.jacobian(positions).T @ derivative).reshape(self.vertex_count, 3)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite local fold-barrier gradient")
        return result

    def hessian(self, positions):
        widths, scaled_gaps, active = self._terms(self._angles(positions))
        if not np.any(active):
            return csr_matrix((3 * self.vertex_count, 3 * self.vertex_count))
        curvature = np.zeros(len(self.indices))
        offsets = scaled_gaps[active] - 1
        curvature[active] = (self.stiffness_joules[active] / widths[active] ** 2
                             * (-2 * np.log1p(offsets) - 4 * offsets / scaled_gaps[active]
                                + (offsets / scaled_gaps[active]) ** 2))
        jacobian = self._geometry.jacobian(positions)
        result = (jacobian.T @ diags(curvature) @ jacobian).tocsr()
        if not np.isfinite(result.data).all():
            raise ValueError("Nonfinite local fold-barrier search metric")
        return result

    def energy_change(self, start, end):
        start_angles, end_angles = self._angles(start), self._angles(end)
        if np.any(np.abs(end_angles - start_angles) >= np.pi):
            raise ValueError("Local fold-barrier update crosses principal-angle branch")
        _, start_gaps, start_active = self._terms(start_angles)
        _, end_gaps, end_active = self._terms(end_angles)
        changes = self._energies(end_gaps, end_active) - self._energies(start_gaps, start_active)
        shared = start_active & end_active
        delta = end_gaps[shared] - start_gaps[shared]
        changes[shared] = -self.stiffness_joules[shared] * (
            delta * (end_gaps[shared] + start_gaps[shared] - 2) * np.log(start_gaps[shared])
            + (end_gaps[shared] - 1) ** 2 * np.log1p(delta / start_gaps[shared]))
        result = float(np.sum(changes))
        if not np.isfinite(result):
            raise ValueError("Nonfinite local fold-barrier energy change")
        return result
