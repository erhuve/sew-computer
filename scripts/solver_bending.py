import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def _skew(vectors):
    result = np.zeros((*vectors.shape[:-1], 3, 3), dtype=vectors.dtype)
    result[..., 0, 1] = -vectors[..., 2]
    result[..., 0, 2] = vectors[..., 1]
    result[..., 1, 0] = vectors[..., 2]
    result[..., 1, 2] = -vectors[..., 0]
    result[..., 2, 0] = -vectors[..., 1]
    result[..., 2, 1] = vectors[..., 0]
    return result


class ElasticDihedralBending:
    def __init__(self, vertex_count, indices, rest_angles, rest_lengths, stiffness):
        raw_indices = np.asarray(indices)
        self.indices = raw_indices.astype(np.int64)
        self.vertex_count = vertex_count
        count = len(self.indices)
        self.rest_angles = np.asarray(rest_angles, dtype=float).copy()
        self.rest_lengths = np.asarray(rest_lengths, dtype=float).copy()
        self.stiffness = np.broadcast_to(np.asarray(stiffness, dtype=float), (count,)).copy()
        if (not isinstance(vertex_count, (int, np.integer)) or vertex_count < 0
                or self.indices.shape != (count, 4) or not np.array_equal(raw_indices, self.indices)
                or np.any(self.indices < 0) or np.any(self.indices >= vertex_count)
                or any(len(set(row)) != 4 for row in self.indices)
                or self.rest_angles.shape != (count,) or self.rest_lengths.shape != (count,)):
            raise ValueError("Invalid hinge topology or parameter shapes")
        if (not all(np.isfinite(values).all() for values in
                    (self.rest_angles, self.rest_lengths, self.stiffness))
                or np.any(self.rest_lengths <= 0) or np.any(self.stiffness < 0)
                or np.any(np.abs(self.rest_angles) >= np.pi - 1e-8)):
            raise ValueError("Finite hinge inputs away from the principal-angle branch required")
        self.weights = self.stiffness * self.rest_lengths
        if not np.isfinite(self.weights).all():
            raise ValueError("Nonfinite hinge weights")
        self.dofs = (self.indices[:, :, None] * 3 + np.arange(3)).reshape(count, 12)

    @classmethod
    def from_model(cls, model):
        vertex_count = len(model.particle_mass.numpy())
        if model.edge_bending_properties is None:
            return cls(vertex_count, np.empty((0, 4), dtype=int), [], [], [])
        properties = np.asarray(model.edge_bending_properties.numpy(), dtype=float)
        indices = np.asarray(model.edge_indices.numpy())
        if (indices.ndim != 2 or indices.shape[1] != 4 or properties.shape != (len(indices), 2)
                or not np.isfinite(properties).all() or not np.isfinite(indices).all()
                or np.any(indices != np.floor(indices)) or np.any(indices[:, :2] < -1)
                or np.any(indices[:, 2:] < 0) or np.any(indices >= vertex_count)):
            raise ValueError("Invalid bending properties")
        if np.any(properties < 0) or np.any(properties[:, 1] != 0):
            raise ValueError("Elastic reference requires nonnegative stiffness and zero bending damping")
        active = (indices[:, 0] >= 0) & (indices[:, 1] >= 0) & (properties[:, 0] > 0)
        return cls(vertex_count, indices[active], model.edge_rest_angle.numpy()[active],
                   model.edge_rest_length.numpy()[active], properties[active, 0])

    def _positions(self, positions):
        positions = np.asarray(positions)
        if positions.shape != (self.vertex_count, 3) or not np.isfinite(positions).all():
            raise ValueError("Finite vertex positions required")
        return positions

    def _geometry(self, positions):
        positions = self._positions(positions)
        points = positions[self.indices]
        first_start = points[:, 2] - points[:, 0]
        first_end = points[:, 3] - points[:, 0]
        second_end = points[:, 3] - points[:, 1]
        second_start = points[:, 2] - points[:, 1]
        edge = points[:, 3] - points[:, 2]
        first_normal = np.cross(first_start, first_end)
        second_normal = np.cross(second_end, second_start)
        first_length = np.linalg.norm(first_normal, axis=1)
        second_length = np.linalg.norm(second_normal, axis=1)
        edge_length = np.linalg.norm(edge, axis=1)
        scale = np.max(np.linalg.norm(np.stack((first_start, first_end, second_start, second_end)), axis=2), axis=0)
        if (not all(np.isfinite(value).all() for value in (first_length, second_length, edge_length, scale))
                or np.any(edge_length <= 1e-12 * scale) or np.any(scale <= 0)
                or np.any(first_length <= 1e-12 * scale * scale)
                or np.any(second_length <= 1e-12 * scale * scale)):
            raise ValueError("Degenerate bending hinge")
        first_unit = first_normal / first_length[:, None]
        second_unit = second_normal / second_length[:, None]
        edge_unit = edge / edge_length[:, None]
        sine = np.sum(np.cross(first_unit, second_unit) * edge_unit, axis=1)
        cosine = np.sum(first_unit * second_unit, axis=1)
        angle = np.arctan2(sine, cosine)
        if np.any(np.abs(angle) >= np.pi - 1e-8):
            raise ValueError("Bending hinge reaches principal-angle branch")
        return (angle, sine, cosine, first_unit, second_unit, edge_unit, first_length,
                second_length, edge_length, first_start, first_end, second_start, second_end, edge)

    def angles(self, positions):
        if not len(self.indices):
            self._positions(positions)
            return np.empty(0)
        return self._geometry(np.asarray(positions, dtype=float))[0]

    def residual(self, positions):
        return np.sqrt(self.weights) * (self.angles(positions) - self.rest_angles)

    def jacobian(self, positions):
        if not len(self.indices):
            self._positions(positions)
            return csr_matrix((0, self.vertex_count * 3))
        (_, sine, cosine, first_unit, second_unit, edge_unit, first_length, second_length,
         edge_length, first_start, first_end, second_start, second_end, edge) = self._geometry(
             np.asarray(positions, dtype=float))
        zeros = np.zeros((len(self.indices), 3, 3))
        identity = np.broadcast_to(np.eye(3), zeros.shape)
        first_derivative = np.stack((_skew(edge), zeros, -_skew(first_end), _skew(first_start)), axis=1)
        second_derivative = np.stack((zeros, -_skew(edge), _skew(second_end), -_skew(second_start)), axis=1)
        first_projection = (identity - first_unit[:, :, None] * first_unit[:, None, :]) / first_length[:, None, None]
        second_projection = (identity - second_unit[:, :, None] * second_unit[:, None, :]) / second_length[:, None, None]
        first_derivative = first_projection[:, None] @ first_derivative
        second_derivative = second_projection[:, None] @ second_derivative
        edge_projection = (identity - edge_unit[:, :, None] * edge_unit[:, None, :]) / edge_length[:, None, None]
        edge_derivative = np.stack((zeros, zeros, -edge_projection, edge_projection), axis=1)
        sine_derivative = (
            np.einsum("hi,hvij->hvj", np.cross(second_unit, edge_unit), first_derivative)
            + np.einsum("hi,hvij->hvj", np.cross(edge_unit, first_unit), second_derivative)
            + np.einsum("hi,hvij->hvj", np.cross(first_unit, second_unit), edge_derivative))
        cosine_derivative = (np.einsum("hi,hvij->hvj", second_unit, first_derivative)
                             + np.einsum("hi,hvij->hvj", first_unit, second_derivative))
        derivative = ((cosine[:, None, None] * sine_derivative - sine[:, None, None] * cosine_derivative)
                      / (sine * sine + cosine * cosine)[:, None, None])
        values = np.sqrt(self.weights)[:, None, None] * derivative
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite bending derivatives")
        return coo_matrix((values.ravel(), (np.repeat(np.arange(len(self.indices)), 12), self.dofs.ravel())),
                          shape=(len(self.indices), self.vertex_count * 3)).tocsr()

    def energy(self, positions):
        residual = self.residual(positions)
        result = float(0.5 * np.dot(residual, residual))
        if not np.isfinite(result):
            raise ValueError("Nonfinite bending energy")
        return result

    def gradient(self, positions):
        if not len(self.indices):
            self._positions(positions)
            return np.zeros((self.vertex_count, 3))
        result = np.asarray(self.jacobian(positions).T @ self.residual(positions)).reshape(self.vertex_count, 3)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite bending gradient")
        return result

    def hessian(self, positions):
        if not len(self.indices):
            self._positions(positions)
            return csr_matrix((self.vertex_count * 3, self.vertex_count * 3))
        jacobian = self.jacobian(positions)
        result = (jacobian.T @ jacobian).tocsr()
        if not np.isfinite(result.data).all():
            raise ValueError("Nonfinite bending search metric")
        return result

    def energy_change(self, start, end):
        if not len(self.indices):
            self._positions(start)
            self._positions(end)
            return 0.
        start_geometry = self._geometry(np.asarray(start, dtype=np.longdouble))
        end_geometry = self._geometry(np.asarray(end, dtype=np.longdouble))
        change = end_geometry[0] - start_geometry[0]
        if np.any(np.abs(change) >= np.pi):
            raise ValueError("Bending update crosses principal-angle branch")
        residual = start_geometry[0] - self.rest_angles
        result = float(np.sum(self.weights * change * (residual + 0.5 * change)))
        if not np.isfinite(result):
            raise ValueError("Nonfinite bending energy change")
        return result
