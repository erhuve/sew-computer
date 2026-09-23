from numbers import Real

import numpy as np
from scipy.sparse import block_diag, kron

from solver_sewing_activation import _contains_bool, validate_sewing_activation


class DistanceSewing:
    def __init__(self, sewing, targets, compliance, *, activation=None):
        self.sewing = sewing.copy().tocsr()
        self.targets = np.asarray(targets, dtype=float).copy()
        if (self.targets.shape != (sewing.shape[0],)
                or not np.isfinite(self.targets).all() or np.any(self.targets <= 0)
                or _contains_bool(targets) or isinstance(compliance, (bool, np.bool_))
                or not isinstance(compliance, Real) or not np.isfinite(compliance) or compliance <= 0
                or not np.isfinite(self.sewing.data).all()):
            raise ValueError("Positive finite scalar sewing distances and compliance required")
        self.compliance = float(compliance)
        weights = validate_sewing_activation(activation, sewing.shape[0])
        self._activation = np.frombuffer(weights.tobytes(), dtype=float)
        self._active = np.flatnonzero(weights > 0)
        self._active_sewing = self.sewing[self._active]
        self.sewing_xyz = kron(self.sewing, np.eye(3), format="csr")

    @property
    def activation(self):
        return self._activation.copy()

    def geometry(self, positions):
        """Full row shapes; inactive vectors=0 and lengths=1 are nonphysical placeholders."""
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (self.sewing.shape[1], 3) or not np.isfinite(positions).all():
            raise ValueError("Finite matching sewing positions required")
        vectors = np.zeros((len(self.targets), 3))
        lengths = np.ones(len(self.targets))
        active_vectors = self._active_sewing @ positions
        active_lengths = np.linalg.norm(active_vectors, axis=1)
        if not np.isfinite(active_lengths).all() or np.any(active_lengths <= 0):
            raise ValueError("Distance sewing requires noncoincident anchors")
        vectors[self._active], lengths[self._active] = active_vectors, active_lengths
        return vectors, lengths

    def residual(self, positions):
        _, lengths = self.geometry(positions)
        result = np.zeros(len(self.targets))
        result[self._active] = ((lengths[self._active] - self.targets[self._active]) / np.sqrt(self.compliance)
                                * np.sqrt(self._activation[self._active]))
        return result

    def jacobian(self, positions):
        vectors, lengths = self.geometry(positions)
        from scipy.sparse import coo_matrix
        normals = vectors / lengths[:, None]
        normals *= np.sqrt(self._activation)[:, None]
        rows = np.repeat(np.arange(len(lengths)), 3)
        projection = coo_matrix((normals.ravel(), (rows, np.arange(3 * len(lengths)))),
                                shape=(len(lengths), 3 * len(lengths))).tocsr()
        result = projection @ self.sewing_xyz / np.sqrt(self.compliance)
        result.eliminate_zeros()
        return result

    def gradient(self, positions):
        vectors, lengths = self.geometry(positions)
        active = self._active
        forces = np.zeros_like(vectors)
        forces[active] = vectors[active] * (((1 - self.targets[active] / lengths[active])
                                            / self.compliance) * self._activation[active])[:, None]
        return (self.sewing.T @ forces).ravel()

    def hessian(self, positions, project_psd=False):
        vectors, lengths = self.geometry(positions)
        active = self._active
        normals = vectors[active] / lengths[active, None]
        radial = normals[:, :, None] * normals[:, None, :]
        tangential = 1 - self.targets[active] / lengths[active]
        if project_psd:
            tangential = np.maximum(tangential, 0)
        blocks = ((radial + tangential[:, None, None] * (np.eye(3) - radial)) / self.compliance
                  * self._activation[active, None, None])
        if not len(blocks):
            from scipy.sparse import csr_matrix
            return csr_matrix((self.sewing_xyz.shape[1], self.sewing_xyz.shape[1]))
        active_xyz = self.sewing_xyz[(active[:, None] * 3 + np.arange(3)).ravel()]
        result = active_xyz.T @ block_diag(blocks, format="csr") @ active_xyz
        result.eliminate_zeros()
        return result

    def exact_hessian(self, positions):
        return self.hessian(positions)

    def energy_change(self, start, end):
        vectors, lengths = self.geometry(start)
        _, end_lengths = self.geometry(end)
        active = self._active
        if not len(active):
            return 0.
        vertices = np.unique(self._active_sewing.indices)
        displacement = np.zeros((self.sewing.shape[1], 3))
        displacement[vertices] = np.asarray(end)[vertices] - np.asarray(start)[vertices]
        delta = self._active_sewing @ displacement
        change = np.sum(delta * (2 * vectors[active] + delta), axis=1) / (lengths[active] + end_lengths[active])
        return float(np.sum(self._activation[active] * (lengths[active] - self.targets[active] + .5 * change)
                            * change) / self.compliance)
