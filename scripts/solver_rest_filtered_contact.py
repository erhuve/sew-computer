import hashlib
import json
from types import MappingProxyType

import numpy as np

from solver_ipc_contact import IpcSurfaceContact
from solver_ipc_broad_phase import CONTACT_BROAD_PHASE_PROFILE, contact_broad_phase
from solver_temporal_separation import _GROUPS, _primitive_ids
from solver_swept_separation import swept_plane_separated


class RestFilteredSurfaceContact(IpcSurfaceContact):
    def __setattr__(self, name, value):
        if getattr(self, "_configuration_locked", False) and name in (
                "activation_distance_m", "minimum_distance_m", "stiffness",
                "_filtered", "_filtered_parameter_indices", "_contact_parameters",
                "_potentials", "_max_candidates", "_filtered_pairs_sha256",
                "_filtered_parameters_sha256", "_local_minimum_range", "_local_activation_range",
                "_native_barriers", "_native_ccd_parameters", "_energy_profile", "_ccd_profile",
                "ccd", "_configuration_locked", "_feature_profile",
                "_exact_feature_definition", "_exact_inflation_radius"):
            raise AttributeError("Rest-filtered contact configuration is fixed")
        super().__setattr__(name, value)

    def __init__(self, rest_positions, faces, *, activation_distance_m,
                 minimum_distance_m, stiffness, max_candidates=100000,
                 ccd_profile="tight-inclusion", feature_profile="native"):
        import ipctk
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        if type(max_candidates) is not int or not 1 <= max_candidates <= 1000000:
            raise ValueError("Bounded candidate budget required")
        if ccd_profile not in ("tight-inclusion", "temporal-separation-tight-inclusion"):
            raise ValueError("Unsupported rest-filtered continuous contact profile")
        from solver_exact_contact import (PROFILE as EXACT_PROFILE, DEFINITION,
            classify_candidate, clamp_squared, inflation_radius, rest_distance)
        from solver_contact_work import WorkBudget
        if type(feature_profile) is not str or feature_profile not in ("native", EXACT_PROFILE):
            raise ValueError("Unsupported explicit contact feature profile")
        self._feature_profile = feature_profile
        exact_features = feature_profile == EXACT_PROFILE
        self._exact_feature_definition = json.dumps(DEFINITION, sort_keys=True) if exact_features else None
        geometry_budget = WorkBudget() if exact_features else None
        super().__init__(rest_positions, faces, activation_distance_m=activation_distance_m,
                         minimum_distance_m=minimum_distance_m, stiffness=stiffness,
                         energy_profile="area-improved-max", ccd_profile=ccd_profile)
        self._max_candidates = max_candidates
        self._edges = np.asarray(self.mesh.edges)
        adjacency = coo_matrix((np.ones(2 * len(self._edges)),
            (self._edges.ravel(), self._edges[:, ::-1].ravel())),
            shape=(self.vertex_count, self.vertex_count))
        component_count, owners = connected_components(adjacency, directed=False)
        filtered_parameters = {}
        candidate_count = 0
        outer = self.minimum_distance_m + self.activation_distance_m
        exact_outer_squared = (clamp_squared(self.activation_distance_m, self.minimum_distance_m)
                               if exact_features else None)
        rest_radius = (inflation_radius(((self.activation_distance_m, self.minimum_distance_m),))
                       if exact_features else np.nextafter(outer / 2, np.inf))
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
            candidates.build(mesh, panel, inflation_radius=rest_radius,
                             broad_phase=contact_broad_phase())
            candidate_count += len(candidates)
            if candidate_count > max_candidates:
                raise ValueError("Rest-filtered candidate budget exceeded")
            for name in _GROUPS:
                for candidate in getattr(candidates, name):
                    first, second = _primitive_ids(name, candidate, edges, triangles)
                    if exact_features:
                        _, _, _, exact_distance, _ = classify_candidate(
                            name[:2], candidate, panel, edges, triangles, geometry_budget)
                        distance = rest_distance(exact_distance)
                        inside = exact_distance < exact_outer_squared
                    else:
                        distance_squared = candidate.compute_distance(candidate.dof(panel, edges, triangles))
                        if not np.isfinite(distance_squared) or distance_squared <= 0:
                            raise ValueError("Positive source primitive separation required")
                        distance = float(np.sqrt(distance_squared))
                        inside = distance < outer
                    if inside:
                        local_scale = distance / 4
                        local_minimum = min(self.minimum_distance_m, local_scale)
                        squared_gap = np.float64(local_scale) * (local_scale + 2 * local_minimum)
                        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                            derived = np.array([local_minimum ** 2, squared_gap,
                                self.stiffness * squared_gap ** 2,
                                local_scale / squared_gap ** 2])
                        if not np.isfinite(derived).all() or np.any(derived <= 0):
                            raise ValueError("Unrepresentable local barrier scale")
                        if exact_features and exact_distance < clamp_squared(local_scale, local_minimum):
                            raise ValueError("Derived exact rest contact must be inactive")
                        key = self._ids_key(name, global_ids[first], global_ids[second])
                        filtered_parameters[key] = (local_scale, local_minimum)
        self._filtered = frozenset(filtered_parameters)
        # Each primitive pair keeps its own source-derived scale. Grouping only
        # identical parameters is an evaluation optimization, not a global law:
        # adding or refining unrelated geometry must not change existing forces.
        full_parameters = (self.activation_distance_m, self.minimum_distance_m)
        local_parameters = sorted(set(filtered_parameters.values()) - {full_parameters})
        self._contact_parameters = (full_parameters, *local_parameters)
        self._exact_inflation_radius = inflation_radius(self._contact_parameters) if exact_features else None
        parameter_indices = {parameters: index for index, parameters in enumerate(self._contact_parameters)}
        self._filtered_parameter_indices = MappingProxyType({
            key: parameter_indices[parameters] for key, parameters in filtered_parameters.items()})
        keys = sorted(self._filtered)
        assigned = [(key, filtered_parameters[key]) for key in keys]
        self._filtered_pairs_sha256 = hashlib.sha256(
            json.dumps(keys, separators=(",", ":")).encode()).hexdigest()
        self._filtered_parameters_sha256 = hashlib.sha256(
            json.dumps(assigned, separators=(",", ":")).encode()).hexdigest()
        values = tuple(filtered_parameters.values())
        self._local_minimum_range = (min(row[1] for row in values), max(row[1] for row in values)) if values else None
        self._local_activation_range = (min(row[0] for row in values), max(row[0] for row in values)) if values else None
        self._potentials = tuple(ipctk.BarrierPotential(activation, self.stiffness,
                                                       use_physical_barrier=True)
                                 for activation, _ in self._contact_parameters)
        # pybind exposes mutable native dhat/barrier and CCD properties even
        # when their Python containers are immutable. Check these inexpensive
        # values before every public evaluation, including cached geometry.
        self._native_barriers = tuple(potential.barrier for potential in self._potentials)
        self._native_ccd_parameters = (self.ccd.tolerance, self.ccd.max_iterations,
                                       self.ccd.conservative_rescaling)
        self._bucket_cache = None
        self._bucket_positions = None
        self._configuration_locked = True

    def _check_native_parameters(self):
        if (any(potential.dhat != parameters[0] or potential.barrier is not barrier
                for potential, parameters, barrier in zip(
                    self._potentials, self._contact_parameters, self._native_barriers))
                or (self.ccd.tolerance, self.ccd.max_iterations, self.ccd.conservative_rescaling)
                != self._native_ccd_parameters):
            raise ValueError("Native contact parameters differ from the captured contact law")

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
        radius = (self._exact_inflation_radius if self._exact_inflation_radius is not None else
                  np.nextafter((self.minimum_distance_m + self.activation_distance_m) / 2, np.inf))
        candidates.build(self.mesh, start, end, inflation_radius=radius,
                         broad_phase=contact_broad_phase())
        if len(candidates) > self._max_candidates:
            raise ValueError("Rest-filtered candidate budget exceeded")
        if sum(len(getattr(candidates, name)) for name in _GROUPS) != len(candidates):
            raise ValueError("Unsupported contact candidate kind")
        return candidates

    def _partition(self, candidates):
        import ipctk

        groups = {}
        for name in _GROUPS:
            for candidate in getattr(candidates, name):
                index = self._filtered_parameter_indices.get(self._key(name, candidate), 0)
                groups.setdefault(index, {}).setdefault(name, []).append(candidate)
        partition = []
        for index in sorted(groups):
            bucket = ipctk.Candidates()
            for name, entries in groups[index].items():
                setattr(bucket, name, entries)
            partition.append((index, bucket))
        if sum(len(bucket) for _, bucket in partition) != len(candidates):
            raise ValueError("Incomplete contact partition")
        return partition

    def minimum_distance_for_candidate(self, name, candidate):
        index = self._filtered_parameter_indices.get(self._key(name, candidate), 0)
        return self._contact_parameters[index][1]

    def _buckets(self, positions):
        import ipctk

        self._check_native_parameters()
        positions = self._positions(positions)
        changed = (self._bucket_positions is None or
            (positions.tobytes() != self._bucket_positions.tobytes() if self._exact_feature_definition is not None
             else not np.array_equal(positions, self._bucket_positions)))
        if changed:
            groups = self._partition(self._candidates(positions, positions))
            buckets = []
            exact_certificates = []
            for index, candidates in groups:
                activation, minimum = self._contact_parameters[index]
                if self._exact_feature_definition is not None:
                    from solver_exact_contact import build_exact_collisions
                    collisions, certificate = build_exact_collisions(self.mesh, positions, candidates,
                        activation=activation, minimum=minimum, max_candidates=self._max_candidates)
                    exact_certificates.append(dict(certificate, bucket=index))
                else:
                    collisions = ipctk.NormalCollisions()
                    collisions.use_area_weighting = True
                    collisions.collision_set_type = ipctk.NormalCollisions.IPC
                    collisions.build(candidates, self.mesh, positions, activation, minimum)
                distance = collisions.compute_minimum_distance(self.mesh, positions)
                if np.isnan(distance) or distance <= minimum ** 2:
                    raise ValueError("Contact state violates assigned minimum separation")
                buckets.append((index, collisions))
            self._bucket_positions = positions.copy()
            self._bucket_cache = buckets
            if self._exact_feature_definition is not None:
                self._exact_certificates_json = json.dumps(exact_certificates, sort_keys=True)
        return positions, self._bucket_cache

    def feature_certificate(self, positions):
        if self._exact_feature_definition is None:
            raise ValueError("Exact feature policy is not enabled")
        self._buckets(positions)
        return {"definition": json.loads(self._exact_feature_definition),
                "buckets": json.loads(self._exact_certificates_json), "accepted": False}

    def validate_state(self, positions):
        import ipctk

        positions, _ = self._buckets(positions)
        if ipctk.has_intersections(self.mesh, positions, broad_phase=contact_broad_phase()):
            raise ValueError("Intersecting contact state")

    def energy(self, positions):
        positions, buckets = self._buckets(positions)
        value = sum(self._potentials[index](bucket, self.mesh, positions)
                    for index, bucket in buckets)
        if not np.isfinite(value) or value < 0:
            raise ValueError("Invalid filtered contact energy")
        return float(value)

    def gradient(self, positions):
        positions, buckets = self._buckets(positions)
        value = sum((np.asarray(self._potentials[index].gradient(bucket, self.mesh, positions)).reshape((-1, 3))
                     for index, bucket in buckets),
                    np.zeros_like(positions))
        if not np.isfinite(value).all():
            raise ValueError("Invalid filtered contact gradient")
        return value

    def hessian(self, positions):
        import ipctk
        from scipy.sparse import csr_matrix

        positions, buckets = self._buckets(positions)
        value = sum((self._potentials[index].hessian(bucket, self.mesh, positions, ipctk.PSDProjectionMethod.NONE)
                     for index, bucket in buckets),
                    csr_matrix((positions.size, positions.size)))
        if not np.isfinite(value.data).all():
            raise ValueError("Invalid filtered contact Hessian")
        return value

    def step_limit(self, start, end):
        self.validate_state(start)
        start, end = self._positions(start), self._positions(end)
        groups = self._partition(self._candidates(start, end))
        result = 1.
        for index, candidates in groups:
            _, minimum = self._contact_parameters[index]
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
            candidate_minimum_distance=self.minimum_distance_for_candidate)
        report["filteredPairsSha256"] = self._filtered_pairs_sha256
        report["filteredParametersSha256"] = self._filtered_parameters_sha256
        return report

    def profile(self):
        self._check_native_parameters()
        profile = {"adapter": "experimental-rest-filtered-contact-v2", "accepted": False,
                "ipctk": "1.6.0", "stiffnessUnits": "Pa", "pressurePa": self.stiffness,
                "broadPhase": CONTACT_BROAD_PHASE_PROFILE,
                "fullMinimumM": self.minimum_distance_m,
                "fullActivationM": self.activation_distance_m,
                "localScaleRule": "each filtered primitive pair: activation = rest distance / 4; minimum = min(full minimum, activation)",
                "localMinimumRangeM": list(self._local_minimum_range) if self._local_minimum_range else None,
                "localActivationRangeM": list(self._local_activation_range) if self._local_activation_range else None,
                "parameterBuckets": len(self._contact_parameters),
                "filteredPrimitivePairs": len(self._filtered),
                "filteredPairsSha256": self._filtered_pairs_sha256,
                "filteredParametersSha256": self._filtered_parameters_sha256,
                "collisionSet": "area-weighted IPC; not improved-max",
                "pathPredicate": ("outward-rounded-temporal-separation-v1 with fixed per-pair thickness"
                                  if self.ccd_profile == "temporal-separation-tight-inclusion" else
                                  "static support certificate then Tight Inclusion; conservative"),
                "maxCandidates": self._max_candidates,
                "limitations": ["Changed contact model, not a tolerance fix or paper reproduction.",
                    "Local positive core and activation derive from each primitive pair's own rest distance.",
                    "Planar connected rest components only; no sewing or layer operations.",
                    "Closest-feature transitions are not globally C2; Hessians use the selected piecewise stencil.",
                    "No refinement-convergent quadrature, global thickness guarantee or garment acceptance.",
                    "Native broad-phase allocation and CCD require external process budgets."]}
        if self._exact_feature_definition is not None:
            profile["adapter"] = "experimental-rest-filtered-exact-features-v1"
            profile["featureSelection"] = json.loads(self._exact_feature_definition)
            profile["inputConversion"] = "Inherited adapter conversion to binary64 precedes exact geometry; no exactness claim for pre-conversion inputs. Bounded work separately requires raw binary64 endpoints."
            profile["broadPhaseInflationRadiusM"] = self._exact_inflation_radius
            profile["collisionSet"] = "area-weighted IPC source contributions; exact finite-feature construction"
            profile["localScaleRule"] = "exact rest d2, RN(sqrt(RN(d2)))/4; minimum=min(full minimum, activation)"
            profile["limitations"].append("Native areas, primitive derivatives and barrier arithmetic remain numerical inputs.")
        return profile
