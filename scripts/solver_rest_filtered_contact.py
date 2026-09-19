import hashlib
import json

import numpy as np

from solver_ipc_contact import IpcSurfaceContact
from solver_temporal_separation import _GROUPS, _primitive_ids
from solver_swept_separation import swept_plane_separated


class RestFilteredSurfaceContact(IpcSurfaceContact):
    def __setattr__(self, name, value):
        if getattr(self, "_configuration_locked", False) and name in (
                "activation_distance_m", "minimum_distance_m", "stiffness",
                "_local_minimum", "_local_activation", "_filtered", "_max_candidates"):
            raise AttributeError("Rest-filtered contact configuration is fixed")
        super().__setattr__(name, value)

    def __init__(self, rest_positions, faces, *, activation_distance_m,
                 minimum_distance_m, stiffness, max_candidates=100000,
                 ccd_profile="tight-inclusion"):
        import ipctk
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        if type(max_candidates) is not int or not 1 <= max_candidates <= 1000000:
            raise ValueError("Bounded candidate budget required")
        if ccd_profile not in ("tight-inclusion", "temporal-separation-tight-inclusion"):
            raise ValueError("Unsupported rest-filtered continuous contact profile")
        super().__init__(rest_positions, faces, activation_distance_m=activation_distance_m,
                         minimum_distance_m=minimum_distance_m, stiffness=stiffness,
                         energy_profile="area-improved-max", ccd_profile=ccd_profile)
        self._max_candidates = max_candidates
        self._edges = np.asarray(self.mesh.edges)
        adjacency = coo_matrix((np.ones(2 * len(self._edges)),
            (self._edges.ravel(), self._edges[:, ::-1].ravel())),
            shape=(self.vertex_count, self.vertex_count))
        component_count, owners = connected_components(adjacency, directed=False)
        filtered = set()
        distances = []
        candidate_count = 0
        outer = self.minimum_distance_m + self.activation_distance_m
        for owner in range(component_count):
            global_ids = np.flatnonzero(owners == owner)
            panel = self.rest_positions[global_ids]
            centered = panel - panel.mean(axis=0)
            singular = np.linalg.svd(centered, compute_uv=False)
            if singular[-1] > 1e-12 * singular[0]:
                raise ValueError("Rest filtering requires planar disconnected source panels")
            local_ids = np.full(self.vertex_count, -1, dtype=int)
            local_ids[global_ids] = np.arange(len(global_ids))
            edges = local_ids[self._edges[owners[self._edges[:, 0]] == owner]]
            triangles = local_ids[self.faces[owners[self.faces[:, 0]] == owner]]
            mesh = ipctk.CollisionMesh(panel, edges, triangles)
            candidates = ipctk.Candidates()
            candidates.build(mesh, panel, inflation_radius=np.nextafter(outer / 2, np.inf))
            candidate_count += len(candidates)
            if candidate_count > max_candidates:
                raise ValueError("Rest-filtered candidate budget exceeded")
            for name in _GROUPS:
                for candidate in getattr(candidates, name):
                    first, second = _primitive_ids(name, candidate, edges, triangles)
                    distance_squared = candidate.compute_distance(candidate.dof(panel, edges, triangles))
                    if not np.isfinite(distance_squared) or distance_squared <= 0:
                        raise ValueError("Positive source primitive separation required")
                    distance = float(np.sqrt(distance_squared))
                    if distance < outer:
                        filtered.add(self._ids_key(name, global_ids[first], global_ids[second]))
                        distances.append(distance)
        self._filtered = frozenset(filtered)
        local_scale = min(distances) / 4 if distances else minimum_distance_m
        if not np.isfinite(local_scale ** 4) or local_scale ** 4 <= 0:
            raise ValueError("Unrepresentable local barrier scale")
        self._local_minimum = min(minimum_distance_m, local_scale)
        self._local_activation = local_scale
        self._potentials = (
            ipctk.BarrierPotential(self.activation_distance_m, self.stiffness,
                                   use_physical_barrier=True),
            ipctk.BarrierPotential(self._local_activation, self.stiffness,
                                   use_physical_barrier=True))
        self._bucket_cache = None
        self._bucket_positions = None
        self._configuration_locked = True

    def _key(self, name, candidate):
        first, second = _primitive_ids(name, candidate, self._edges, self.faces)
        return self._ids_key(name, first, second)

    @staticmethod
    def _ids_key(name, first, second):
        first, second = tuple(sorted(map(int, first))), tuple(sorted(map(int, second)))
        if name in ("vv_candidates", "ee_candidates") and first > second:
            first, second = second, first
        return name, first, second

    def _candidates(self, start, end):
        import ipctk

        start, end = self._positions(start), self._positions(end)
        candidates = ipctk.Candidates()
        radius = np.nextafter((self.minimum_distance_m + self.activation_distance_m) / 2,
                              np.inf)
        candidates.build(self.mesh, start, end, inflation_radius=radius)
        if len(candidates) > self._max_candidates:
            raise ValueError("Rest-filtered candidate budget exceeded")
        if sum(len(getattr(candidates, name)) for name in _GROUPS) != len(candidates):
            raise ValueError("Unsupported contact candidate kind")
        return candidates

    def _partition(self, candidates):
        import ipctk

        full, local = ipctk.Candidates(), ipctk.Candidates()
        for name in _GROUPS:
            groups = [[], []]
            for candidate in getattr(candidates, name):
                groups[int(self._key(name, candidate) in self._filtered)].append(candidate)
            setattr(full, name, groups[0])
            setattr(local, name, groups[1])
        if len(full) + len(local) != len(candidates):
            raise ValueError("Incomplete contact partition")
        return full, local

    def _parameters(self):
        return ((self.activation_distance_m, self.minimum_distance_m),
                (self._local_activation, self._local_minimum))

    def _buckets(self, positions):
        import ipctk

        positions = self._positions(positions)
        if self._bucket_positions is None or not np.array_equal(positions, self._bucket_positions):
            groups = self._partition(self._candidates(positions, positions))
            buckets = []
            for candidates, (activation, minimum) in zip(groups, self._parameters()):
                collisions = ipctk.NormalCollisions()
                collisions.use_area_weighting = True
                collisions.collision_set_type = ipctk.NormalCollisions.IPC
                collisions.build(candidates, self.mesh, positions, activation, minimum)
                distance = collisions.compute_minimum_distance(self.mesh, positions)
                if np.isnan(distance) or distance <= minimum ** 2:
                    raise ValueError("Contact state violates assigned minimum separation")
                buckets.append(collisions)
            self._bucket_positions = positions.copy()
            self._bucket_cache = buckets
        return positions, self._bucket_cache

    def validate_state(self, positions):
        import ipctk

        positions, _ = self._buckets(positions)
        if ipctk.has_intersections(self.mesh, positions):
            raise ValueError("Intersecting contact state")

    def energy(self, positions):
        positions, buckets = self._buckets(positions)
        value = sum(potential(bucket, self.mesh, positions)
                    for potential, bucket in zip(self._potentials, buckets))
        if not np.isfinite(value) or value < 0:
            raise ValueError("Invalid filtered contact energy")
        return float(value)

    def gradient(self, positions):
        positions, buckets = self._buckets(positions)
        value = sum((np.asarray(potential.gradient(bucket, self.mesh, positions)).reshape((-1, 3))
                     for potential, bucket in zip(self._potentials, buckets)),
                    np.zeros_like(positions))
        if not np.isfinite(value).all():
            raise ValueError("Invalid filtered contact gradient")
        return value

    def hessian(self, positions):
        import ipctk
        from scipy.sparse import csr_matrix

        positions, buckets = self._buckets(positions)
        value = sum((potential.hessian(bucket, self.mesh, positions, ipctk.PSDProjectionMethod.NONE)
                     for potential, bucket in zip(self._potentials, buckets)),
                    csr_matrix((positions.size, positions.size)))
        if not np.isfinite(value.data).all():
            raise ValueError("Invalid filtered contact Hessian")
        return value

    def step_limit(self, start, end):
        self.validate_state(start)
        start, end = self._positions(start), self._positions(end)
        groups = self._partition(self._candidates(start, end))
        result = 1.
        for candidates, (_, minimum) in zip(groups, self._parameters()):
            for name in _GROUPS:
                group = list(getattr(candidates, name))
                if not group:
                    continue
                ids = [_primitive_ids(name, candidate, self._edges, self.faces) for candidate in group]
                first = np.asarray([pair[0] for pair in ids])
                second = np.asarray([pair[1] for pair in ids])
                normals = [candidate.compute_distance_vector(candidate.dof(start, self._edges, self.faces))
                           for candidate in group]
                safe = swept_plane_separated(
                    np.concatenate((start[first], end[first]), axis=1),
                    np.concatenate((start[second], end[second]), axis=1), normals, minimum)
                setattr(candidates, name, [candidate for candidate, separated in zip(group, safe)
                                          if not separated])
            limit = float(candidates.compute_collision_free_stepsize(
                self.mesh, start, end, min_distance=minimum, narrow_phase_ccd=self.ccd))
            if not np.isfinite(limit) or not 0 <= limit <= 1:
                raise ValueError("Invalid filtered contact step bound")
            result = min(result, limit)
        return result

    def path_safe(self, start, end):
        if self.ccd_profile == "temporal-separation-tight-inclusion":
            return self.path_certificate(start, end)["safe"]
        self.validate_state(end)
        return self.step_limit(start, end) == 1.

    def path_certificate(self, start, end, *, max_depth=12, max_nodes=100000, keep_leaves=False):
        from solver_temporal_separation import certify_linear_path

        self.validate_state(start)
        self.validate_state(end)
        report = certify_linear_path(self.mesh, start, end, self.minimum_distance_m,
            max_depth=max_depth, max_nodes=max_nodes, keep_leaves=keep_leaves,
            candidate_minimum_distance=lambda name, candidate:
                self._local_minimum if self._key(name, candidate) in self._filtered
                else self.minimum_distance_m)
        report["filteredPairsSha256"] = self.profile()["filteredPairsSha256"]
        return report

    def profile(self):
        encoded = json.dumps(sorted(self._filtered), separators=(",", ":")).encode()
        return {"adapter": "experimental-rest-filtered-contact-v1", "accepted": False,
                "ipctk": "1.6.0", "stiffnessUnits": "Pa", "pressurePa": self.stiffness,
                "fullMinimumM": self.minimum_distance_m,
                "fullActivationM": self.activation_distance_m,
                "localMinimumM": self._local_minimum, "localActivationM": self._local_activation,
                "filteredPrimitivePairs": len(self._filtered),
                "filteredPairsSha256": hashlib.sha256(encoded).hexdigest(),
                "collisionSet": "area-weighted IPC; not improved-max",
                "pathPredicate": ("outward-rounded-temporal-separation-v1 with fixed per-pair thickness"
                                  if self.ccd_profile == "temporal-separation-tight-inclusion" else
                                  "static support certificate then Tight Inclusion; conservative"),
                "maxCandidates": self._max_candidates,
                "limitations": ["Changed contact model, not a tolerance fix or paper reproduction.",
                    "Local positive core and activation use one quarter of minimum filtered rest distance.",
                    "Planar connected rest components only; no sewing or layer operations.",
                    "Primitive partition and collision-set transitions require derivative validation.",
                    "No refinement-convergent quadrature, global thickness guarantee or garment acceptance.",
                    "Native broad-phase allocation and CCD require external process budgets."]}
