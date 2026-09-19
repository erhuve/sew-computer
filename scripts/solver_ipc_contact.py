import importlib.metadata

import numpy as np


class IpcSurfaceContact:
    def __init__(self, rest_positions, faces, *, activation_distance_m, minimum_distance_m, stiffness,
                 energy_profile="legacy", ccd_profile="tight-inclusion"):
        import ipctk

        if importlib.metadata.version("ipctk") != "1.6.0":
            raise ValueError("The experimental contact adapter requires ipctk 1.6.0")
        if energy_profile not in ("legacy", "area-improved-max"):
            raise ValueError("Unsupported contact energy profile")
        if ccd_profile not in ("tight-inclusion", "swept-plane-tight-inclusion",
                               "temporal-separation-tight-inclusion"):
            raise ValueError("Unsupported continuous contact profile")
        self._energy_profile = energy_profile
        self._ccd_profile = ccd_profile
        parameters = (activation_distance_m, minimum_distance_m, stiffness)
        if any(isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
               or not np.isfinite(value) or value <= 0 for value in parameters):
            raise ValueError("Positive finite contact distances and stiffness required")
        self.activation_distance_m = float(activation_distance_m)
        self.minimum_distance_m = float(minimum_distance_m)
        self.stiffness = float(stiffness)
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            minimum = np.float64(self.minimum_distance_m)
            activation = np.float64(self.activation_distance_m)
            outer = minimum + activation
            squared_gap = activation * (activation + 2 * minimum)
            derived = np.array([minimum * minimum, outer * outer, squared_gap,
                                np.float64(self.stiffness) * squared_gap * squared_gap])
            if self.requires_guarded_metric:
                derived = np.append(derived, [activation / (squared_gap * squared_gap),
                                               np.float64(self.stiffness) * activation])
        if not np.isfinite(derived).all() or np.any(derived <= 0) or outer <= minimum:
            raise ValueError("Contact distances and barrier scale must be representable in float64")
        rest = np.asarray(rest_positions, dtype=float)
        if rest.ndim != 2 or rest.shape[1] != 3 or len(rest) < 3 or not np.isfinite(rest).all():
            raise ValueError("Finite three-dimensional contact rest vertices required")
        raw_faces = np.asarray(faces)
        if (raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or not len(raw_faces)
                or raw_faces.dtype.kind not in "iu" or np.any(raw_faces < 0)
                or np.any(raw_faces >= len(rest))):
            raise ValueError("Integer in-range contact triangles required")
        self.faces = raw_faces.astype(np.int32, copy=True)
        sorted_faces = np.sort(self.faces, axis=1)
        if (np.any(np.diff(sorted_faces, axis=1) == 0)
                or len(np.unique(sorted_faces, axis=0)) != len(self.faces)
                or len(np.unique(self.faces)) != len(rest)):
            raise ValueError("Contact requires unique triangles and no isolated vertices")
        edges, counts = np.unique(np.sort(self.faces[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2)), axis=1),
                                  axis=0, return_counts=True)
        if np.any(counts > 2):
            raise ValueError("Nonmanifold contact edges are unsupported")
        directed_edges = {}
        vertex_links = [{} for _ in rest]
        for face in self.faces:
            for local in range(3):
                vertex, first, second = (int(face[local]), int(face[(local + 1) % 3]),
                                         int(face[(local + 2) % 3]))
                edge = tuple(sorted((vertex, first)))
                direction = vertex < first
                if edge in directed_edges and directed_edges[edge] == direction:
                    raise ValueError("Inconsistent contact triangle winding")
                directed_edges[edge] = direction
                vertex_links[vertex].setdefault(first, set()).add(second)
                vertex_links[vertex].setdefault(second, set()).add(first)
        for link in vertex_links:
            pending, visited = [next(iter(link))], set()
            while pending:
                vertex = pending.pop()
                if vertex not in visited:
                    visited.add(vertex)
                    pending.extend(link[vertex] - visited)
            if len(visited) != len(link) or any(len(neighbors) > 2 for neighbors in link.values()):
                raise ValueError("Nonmanifold contact vertex is unsupported")
        self.faces.flags.writeable = False
        self.vertex_count = len(rest)
        self.rest_positions = rest.copy()
        self.rest_positions.flags.writeable = False
        self._positions(rest)
        self.mesh = ipctk.CollisionMesh(rest, edges, self.faces)
        self.potential = ipctk.BarrierPotential(self.activation_distance_m, self.stiffness,
                                               use_physical_barrier=self.requires_guarded_metric)
        self.ccd = ipctk.TightInclusionCCD(tolerance=1e-6, max_iterations=10000,
                                          conservative_rescaling=.8)
        self._cached_positions = None
        self._cached_collisions = None

    @property
    def energy_profile(self):
        return self._energy_profile

    @property
    def ccd_profile(self):
        return self._ccd_profile

    @property
    def requires_guarded_metric(self):
        return self.energy_profile == "area-improved-max"

    def _positions(self, positions):
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (self.vertex_count, 3) or not np.isfinite(positions).all():
            raise ValueError("Finite correctly shaped contact positions required")
        triangles = positions[self.faces]
        first, second = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        scale = np.maximum(np.linalg.norm(first, axis=1), np.linalg.norm(second, axis=1))
        areas = np.linalg.norm(np.cross(first, second), axis=1)
        if not np.isfinite(areas).all() or np.any(areas <= 1e-12 * scale ** 2):
            raise ValueError("Degenerate contact triangles are unsupported")
        return positions

    def _collisions(self, positions):
        import ipctk

        positions = self._positions(positions)
        if self._cached_positions is None or not np.array_equal(positions, self._cached_positions):
            collisions = ipctk.NormalCollisions()
            if self.requires_guarded_metric:
                collisions.use_area_weighting = True
                collisions.collision_set_type = ipctk.NormalCollisions.IMPROVED_MAX_APPROX
            collisions.build(self.mesh, positions, self.activation_distance_m, self.minimum_distance_m)
            distance_squared = collisions.compute_minimum_distance(self.mesh, positions)
            if np.isnan(distance_squared) or distance_squared <= self.minimum_distance_m ** 2:
                raise ValueError("Contact state violates minimum surface separation")
            self._cached_positions = positions.copy()
            self._cached_collisions = collisions
        return positions, self._cached_collisions

    def validate_state(self, positions):
        import ipctk

        positions, _ = self._collisions(positions)
        if ipctk.has_intersections(self.mesh, positions):
            raise ValueError("Intersecting contact surface cannot initialize the solver")

    def energy(self, positions):
        positions, collisions = self._collisions(positions)
        energy = float(self.potential(collisions, self.mesh, positions))
        if not np.isfinite(energy):
            raise ValueError("Nonfinite contact energy")
        return energy

    def gradient(self, positions):
        positions, collisions = self._collisions(positions)
        gradient = np.asarray(self.potential.gradient(collisions, self.mesh, positions)).reshape((-1, 3))
        if not np.isfinite(gradient).all():
            raise ValueError("Nonfinite contact gradient")
        return gradient

    def hessian(self, positions):
        import ipctk

        positions, collisions = self._collisions(positions)
        projection = (ipctk.PSDProjectionMethod.NONE if self.requires_guarded_metric
                      else ipctk.PSDProjectionMethod.CLAMP)
        hessian = self.potential.hessian(collisions, self.mesh, positions, projection)
        if not np.isfinite(hessian.data).all():
            raise ValueError("Nonfinite contact search metric")
        return hessian

    def energy_change(self, start, end):
        return self.energy(end) - self.energy(start)

    def step_limit(self, start, end):
        import ipctk

        self.validate_state(start)
        start = self._positions(start)
        end = np.asarray(end, dtype=float)
        if end.shape != start.shape or not np.isfinite(end).all():
            raise ValueError("Finite correctly shaped contact endpoint required")
        if self.ccd_profile in ("swept-plane-tight-inclusion", "temporal-separation-tight-inclusion"):
            from solver_swept_separation import certified_candidates
            candidates, _ = certified_candidates(self.mesh, start, end, self.minimum_distance_m)
            limit = float(candidates.compute_collision_free_stepsize(
                self.mesh, start, end, min_distance=self.minimum_distance_m, narrow_phase_ccd=self.ccd))
        else:
            limit = float(ipctk.compute_collision_free_stepsize(self.mesh, start, end,
                                                               min_distance=self.minimum_distance_m,
                                                               narrow_phase_ccd=self.ccd))
        if not np.isfinite(limit) or not 0 <= limit <= 1:
            raise ValueError("Invalid continuous contact step bound")
        return limit

    def path_safe(self, start, end):
        if self.ccd_profile == "temporal-separation-tight-inclusion":
            from solver_temporal_separation import certify_linear_path
            self.validate_state(start)
            self.validate_state(end)
            return certify_linear_path(self.mesh, start, end, self.minimum_distance_m)["safe"]
        self._positions(end)
        return self.step_limit(start, end) == 1.

    def profile(self):
        return {"adapter": "experimental-ipc-area-contact-v1" if self.requires_guarded_metric else "experimental-ipc-surface-contact-v2", "ipctk": "1.6.0",
                "activationDistanceM": self.activation_distance_m,
                "minimumDistanceM": self.minimum_distance_m, "stiffness": self.stiffness,
                "physicalBarrier": self.requires_guarded_metric,
                "energyProfile": self.energy_profile,
                "areaWeighting": self.requires_guarded_metric,
                "stiffnessUnits": "Pa" if self.requires_guarded_metric else "J/m^4",
                "searchMetric": "unprojected signed contact Hessian; assembled guard required" if self.requires_guarded_metric else "per-stencil PSD projection",
                "seamExclusions": False, "accepted": False,
                "ccd": {"method": "TightInclusionCCD", "profile": self.ccd_profile,
                        "sweptPlaneCertificate": self.ccd_profile != "tight-inclusion",
                        "pathPredicate": ("outward-rounded-temporal-separation-v1"
                                          if self.ccd_profile == "temporal-separation-tight-inclusion"
                                          else "conservative-step-equals-one"),
                        "temporalMaxDepth": 12 if self.ccd_profile == "temporal-separation-tight-inclusion" else None,
                        "temporalMaxNodes": 100000 if self.ccd_profile == "temporal-separation-tight-inclusion" else None,
                        "tolerance": 1e-6,
                        "maxIterations": 10000, "conservativeRescaling": .8,
                        "requiresExternalProcessBudget": True}}
