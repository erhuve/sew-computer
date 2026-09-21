import numpy as np

from solver_bending import ElasticDihedralBending


class FoldActuation:
    def __init__(self, model, hinges, stiffness_joules):
        raw = np.asarray(hinges)
        source = (np.asarray(model.edge_indices.numpy()) if model.edge_indices is not None
                  else np.empty((0, 4), dtype=int))
        source_hinges = {tuple(row) for row in source}
        stiffness = np.asarray(stiffness_joules)
        if (raw.ndim != 2 or raw.shape[1] != 4 or not len(raw) or raw.dtype.kind not in "iu"
                or len({tuple(row) for row in raw}) != len(raw)
                or any(tuple(row) not in source_hinges for row in raw)
                or stiffness.dtype.kind not in "fiu" or stiffness.shape != (len(raw),)
                or not np.isfinite(stiffness).all() or np.any(stiffness <= 0)):
            raise ValueError("Unique ordered source hinges and positive per-hinge actuator stiffness required")
        self.vertex_count = len(model.particle_mass.numpy())
        self.hinges = raw.copy()
        self.stiffness = stiffness.astype(float).copy()
        self.potential(np.zeros(len(raw))).energy(model.particle_q.numpy())
        self.hinges.setflags(write=False)
        self.stiffness.setflags(write=False)

    def potential(self, targets):
        raw = np.asarray(targets)
        if raw.dtype.kind not in "fiu" or raw.shape != (len(self.hinges),):
            raise ValueError("Explicit signed fold angles in radians required")
        return ElasticDihedralBending(self.vertex_count, self.hinges, raw,
                                      np.ones(len(self.hinges)), self.stiffness)
