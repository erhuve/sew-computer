from numbers import Real

import numpy as np
from scipy.sparse import coo_matrix, kron

from solver_bending import _skew


class NormalOffsetSewing:
    def __init__(self, sewing, targets, compliance, frame_faces, sides):
        self.sewing = sewing.copy().tocsr()
        self.targets = np.asarray(targets, dtype=float).copy()
        faces = np.asarray(frame_faces)
        sides = np.asarray(sides)
        count, vertex_count = sewing.shape
        if (faces.shape != (count, 3) or faces.dtype.kind not in "iu"
                or np.any(faces < 0) or np.any(faces >= vertex_count)
                or any(len(set(face)) != 3 for face in faces)
                or sides.shape != (count,) or sides.dtype.kind not in "iu"
                or np.any(np.abs(sides) != 1)
                or self.targets.shape != (count,) or not np.isfinite(self.targets).all()
                or np.any(self.targets <= 0) or isinstance(compliance, (bool, np.bool_))
                or not isinstance(compliance, Real) or not np.isfinite(compliance) or compliance <= 0
                or not np.isfinite(self.sewing.data).all()):
            raise ValueError("Valid source frames, explicit sides, positive distances and compliance required")
        self.sewing.sum_duplicates()
        for index, face in enumerate(faces):
            row = self.sewing.getrow(index)
            negative = row.data < 0
            if (abs(row.data[negative].sum() + 1) > 1e-12
                    or abs(row.data[~negative].sum() - 1) > 1e-12
                    or not set(row.indices[negative]).issubset(set(face))
                    or set(row.indices[~negative & (row.data > 0)]).intersection(face)):
                raise ValueError("Frame must contain the normalized negative material anchor only")
        self.faces = faces.copy()
        self.sides = sides.copy()
        self.compliance = float(compliance)
        self.sewing_xyz = kron(self.sewing, np.eye(3), format="csr")

    def geometry(self, positions):
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (self.sewing.shape[1], 3) or not np.isfinite(positions).all():
            raise ValueError("Finite matching frame positions required")
        triangles = positions[self.faces]
        first = triangles[:, 1] - triangles[:, 0]
        second = triangles[:, 2] - triangles[:, 0]
        area = np.cross(first, second)
        magnitude = np.linalg.norm(area, axis=1)
        scale = np.maximum(np.linalg.norm(first, axis=1), np.linalg.norm(second, axis=1))
        if (not np.isfinite(magnitude).all() or np.any(scale <= 0)
                or np.any(magnitude <= 1e-12 * scale ** 2)):
            raise ValueError("Degenerate material frame")
        return first, second, area / magnitude[:, None], magnitude

    def residual(self, positions):
        _, _, normals, _ = self.geometry(positions)
        return ((self.sewing @ positions - (self.targets * self.sides)[:, None] * normals)
                / np.sqrt(self.compliance)).ravel()

    def jacobian(self, positions):
        first, second, normals, magnitude = self.geometry(positions)
        projection = (np.eye(3) - normals[:, :, None] * normals[:, None, :]) / magnitude[:, None, None]
        area_derivatives = np.stack((_skew(second - first), -_skew(second), _skew(first)), axis=1)
        derivatives = np.einsum("nij,nvjk->nvik", projection, area_derivatives)
        derivatives *= -(self.targets * self.sides)[:, None, None, None]
        rows = np.broadcast_to(np.arange(len(first))[:, None, None, None] * 3
                               + np.arange(3)[None, None, :, None], derivatives.shape)
        columns = np.broadcast_to(self.faces[:, :, None, None] * 3
                                  + np.arange(3)[None, None, None, :], derivatives.shape)
        frame_jacobian = coo_matrix((derivatives.ravel(), (rows.ravel(), columns.ravel())),
                                    shape=self.sewing_xyz.shape).tocsr()
        return (self.sewing_xyz + frame_jacobian) / np.sqrt(self.compliance)

    def gradient(self, positions):
        return np.asarray(self.jacobian(positions).T @ self.residual(positions)).ravel()

    def hessian(self, positions, project_psd=False):
        jacobian = self.jacobian(positions)
        return jacobian.T @ jacobian

    def exact_hessian(self, positions):
        first, second, normals, magnitude = self.geometry(positions)
        signed = self.targets * self.sides
        residual = self.sewing @ positions - signed[:, None] * normals
        area_derivatives = np.concatenate((_skew(second - first), -_skew(second), _skew(first)), axis=2)
        normal_dot = np.sum(residual * normals, axis=1)
        curvature = (-residual[:, :, None] * normals[:, None, :]
                     - normals[:, :, None] * residual[:, None, :]
                     - normal_dot[:, None, None] * np.eye(3)
                     + 3 * normal_dot[:, None, None] * normals[:, :, None] * normals[:, None, :])
        curvature /= magnitude[:, None, None] ** 2
        area_gradient = (residual - normal_dot[:, None] * normals) / magnitude[:, None]
        identity = np.eye(3)
        edge_map = np.block([[-identity, identity, np.zeros((3, 3))],
                             [-identity, np.zeros((3, 3)), identity]])
        elements = np.einsum("nai,nab,nbj->nij", area_derivatives, curvature, area_derivatives)
        for index, gradient in enumerate(area_gradient):
            cross_curvature = np.block([[np.zeros((3, 3)), -_skew(gradient)],
                                         [_skew(gradient), np.zeros((3, 3))]])
            elements[index] += edge_map.T @ cross_curvature @ edge_map
        elements *= (-signed / self.compliance)[:, None, None]
        dofs = (self.faces[:, :, None] * 3 + np.arange(3)).reshape((-1, 9))
        rows = np.broadcast_to(dofs[:, :, None], elements.shape)
        columns = np.broadcast_to(dofs[:, None, :], elements.shape)
        correction = coo_matrix((elements.ravel(), (rows.ravel(), columns.ravel())),
                                shape=(self.sewing_xyz.shape[1],) * 2).tocsr()
        return self.hessian(positions) + correction

    def energy_change(self, start, end):
        before, after = self.residual(start), self.residual(end)
        return float((before + .5 * (after - before)) @ (after - before))
