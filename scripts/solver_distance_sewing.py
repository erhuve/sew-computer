import numpy as np
from scipy.sparse import block_diag, kron


class DistanceSewing:
    def __init__(self, sewing, targets, compliance):
        self.sewing = sewing.copy()
        self.targets = np.asarray(targets, dtype=float).copy()
        self.compliance = float(compliance)
        if (self.targets.shape != (sewing.shape[0],)
                or not np.isfinite(self.targets).all() or np.any(self.targets <= 0)
                or not np.isfinite(self.compliance) or self.compliance <= 0):
            raise ValueError("Positive finite scalar sewing distances and compliance required")
        self.sewing_xyz = kron(sewing, np.eye(3), format="csr")

    def geometry(self, positions):
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (self.sewing.shape[1], 3) or not np.isfinite(positions).all():
            raise ValueError("Finite matching sewing positions required")
        vectors = self.sewing @ positions
        lengths = np.linalg.norm(vectors, axis=1)
        if not np.isfinite(lengths).all() or np.any(lengths <= 0):
            raise ValueError("Distance sewing requires noncoincident anchors")
        return vectors, lengths

    def residual(self, positions):
        _, lengths = self.geometry(positions)
        return (lengths - self.targets) / np.sqrt(self.compliance)

    def jacobian(self, positions):
        vectors, lengths = self.geometry(positions)
        from scipy.sparse import coo_matrix
        normals = vectors / lengths[:, None]
        rows = np.repeat(np.arange(len(lengths)), 3)
        projection = coo_matrix((normals.ravel(), (rows, np.arange(3 * len(lengths)))),
                                shape=(len(lengths), 3 * len(lengths))).tocsr()
        return projection @ self.sewing_xyz / np.sqrt(self.compliance)

    def gradient(self, positions):
        vectors, lengths = self.geometry(positions)
        return (self.sewing.T @ (vectors * ((1 - self.targets / lengths)
                                           / self.compliance)[:, None])).ravel()

    def hessian(self, positions, project_psd=False):
        vectors, lengths = self.geometry(positions)
        normals = vectors / lengths[:, None]
        radial = normals[:, :, None] * normals[:, None, :]
        tangential = 1 - self.targets / lengths
        if project_psd:
            tangential = np.maximum(tangential, 0)
        blocks = (radial + tangential[:, None, None] * (np.eye(3) - radial)) / self.compliance
        if not len(blocks):
            from scipy.sparse import csr_matrix
            return csr_matrix((self.sewing_xyz.shape[1], self.sewing_xyz.shape[1]))
        return self.sewing_xyz.T @ block_diag(blocks, format="csr") @ self.sewing_xyz

    def energy_change(self, start, end):
        vectors, lengths = self.geometry(start)
        _, end_lengths = self.geometry(end)
        delta = self.sewing @ (np.asarray(end) - np.asarray(start))
        change = np.sum(delta * (2 * vectors + delta), axis=1) / (lengths + end_lengths)
        return float(np.sum((lengths - self.targets + .5 * change) * change) / self.compliance)
