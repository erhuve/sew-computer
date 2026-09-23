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
        if not len(self.indices):
            self._geometry._positions(start)
            self._geometry._positions(end)
            return 0.
        start = np.asarray(start, dtype=np.longdouble)
        end = np.asarray(end, dtype=np.longdouble)
        start_geometry = self._geometry._geometry(start)
        end_geometry = self._geometry._geometry(end)
        start_angles, end_angles = start_geometry[0], end_geometry[0]
        if np.any(np.abs(end_angles - start_angles) >= np.pi):
            raise ValueError("Local fold-barrier update crosses principal-angle branch")
        widths, start_gaps, start_active = self._terms(start_angles)
        _, end_gaps, end_active = self._terms(end_angles)
        changes = self._energies(end_gaps, end_active) - self._energies(start_gaps, start_active)
        shared = start_active & end_active
        if not np.any(shared):
            result = float(np.sum(changes))
            if not np.isfinite(result):
                raise ValueError("Nonfinite local fold-barrier energy change")
            return result
        delta = end_gaps[shared] - start_gaps[shared]
        # longdouble is float64 on some supported hosts. Subtracting separately
        # rounded atan2 values (and then scaled gaps) loses tiny physical steps.
        # Instead propagate the coordinate displacement through the two normal
        # vectors and use atan2 of the incremental sine/cosine rotation.
        displacement = (end - start)[self.indices[shared]]
        first_start_delta = displacement[:, 2] - displacement[:, 0]
        first_end_delta = displacement[:, 3] - displacement[:, 0]
        second_end_delta = displacement[:, 3] - displacement[:, 1]
        second_start_delta = displacement[:, 2] - displacement[:, 1]

        def unit_increment(vector, change, unit, length, next_length):
            length_change = np.sum((2 * vector + change) * change, axis=1) / (length + next_length)
            return (change - unit * length_change[:, None]) / next_length[:, None]

        def cross_increment(first, second, first_delta, second_delta):
            return (np.cross(first_delta, second) + np.cross(first, second_delta)
                    + np.cross(first_delta, second_delta))

        (_, sine, cosine, first_unit, second_unit, edge_unit, first_length, second_length,
         edge_length, first_start, first_end, second_start, second_end, edge) = (
            values[shared] for values in start_geometry)
        first_change = cross_increment(first_start, first_end, first_start_delta, first_end_delta)
        second_change = cross_increment(second_end, second_start, second_end_delta, second_start_delta)
        edge_change = displacement[:, 3] - displacement[:, 2]
        same_side = np.sign(start_angles[shared]) == np.sign(end_angles[shared])
        incremental = same_side.copy()
        # A displacement expansion is accurate locally, but a large shrink can
        # cancel its entire source normal. Retain independently validated
        # endpoint gaps for those updates instead of reconstructing small final
        # vectors from large cancelling terms. This changes evaluation only.
        for vector, change in ((first_start, first_start_delta), (first_end, first_end_delta),
                               (second_start, second_start_delta), (second_end, second_end_delta),
                               (edge, edge_change)):
            incremental &= np.linalg.norm(change, axis=1) <= .125 * np.linalg.norm(vector, axis=1)
        incremental &= ((np.linalg.norm(first_change, axis=1) <= .125 * first_length)
                        & (np.linalg.norm(second_change, axis=1) <= .125 * second_length))
        first_delta = unit_increment(np.cross(first_start, first_end)[incremental], first_change[incremental],
            first_unit[incremental], first_length[incremental], end_geometry[6][shared][incremental])
        second_delta = unit_increment(np.cross(second_end, second_start)[incremental], second_change[incremental],
            second_unit[incremental], second_length[incremental], end_geometry[7][shared][incremental])
        edge_delta = unit_increment(edge[incremental], edge_change[incremental],
            edge_unit[incremental], edge_length[incremental], end_geometry[8][shared][incremental])
        first_unit, second_unit, edge_unit = (values[incremental] for values in (first_unit, second_unit, edge_unit))
        sine, cosine = sine[incremental], cosine[incremental]
        cross = np.cross(first_unit, second_unit)
        cross_delta = (np.cross(first_delta, second_unit) + np.cross(first_unit, second_delta)
                       + np.cross(first_delta, second_delta))
        sine_delta = np.sum(cross_delta * edge_unit + (cross + cross_delta) * edge_delta, axis=1)
        cosine_delta = np.sum(first_delta * second_unit + (first_unit + first_delta) * second_delta, axis=1)
        angle_delta = np.arctan2(cosine * sine_delta - sine * cosine_delta,
                                cosine * (cosine + cosine_delta) + sine * (sine + sine_delta))
        delta[incremental] = (-np.sign(start_angles[shared][incremental]) * angle_delta
                              / widths[shared][incremental])
        start_offsets = (self.activation_angles[shared] - np.abs(start_angles[shared])) / widths[shared]
        changes[shared] = -self.stiffness_joules[shared] * (
            delta * (2 * start_offsets + delta) * np.log(start_gaps[shared])
            + (start_offsets + delta) ** 2 * np.log1p(delta / start_gaps[shared]))
        result = float(np.sum(changes))
        if not np.isfinite(result):
            raise ValueError("Nonfinite local fold-barrier energy change")
        return result
