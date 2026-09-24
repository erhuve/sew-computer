"""Exact finite-feature admission and complete IPC candidate construction.

This opt-in model constructs native stencils from every supplied broad-phase
candidate, before activation and primitive conversion. It does not repair an
already-built native inventory. Native rest areas, barrier constants and
derivative arithmetic remain captured numerical inputs. IPC area coefficients
and conversion roles follow IPC Toolkit 1.6.0 at commit 478876f30bf8ea768772dd8983c26a1a801ad976;
source-candidate contributions remain separate rather than merging coefficients.
"""
from fractions import Fraction as F
import hashlib
import json
import math

import numpy as np

from solver_contact_work import (BarrierParameters, WorkBudget, closest_feature,
                                 _cross, _dot, _sub)


PROFILE = 'area-weighted-ipc-exact-binary-features-v1'
GROUPS = ('vv_candidates', 'ev_candidates', 'ee_candidates', 'fv_candidates')
KINDS = ('vv', 'ev', 'ee', 'fv')
DEFINITION = {
    'profile': PROFILE,
    'geometry': 'Exact finite-primitive distances on binary64 coordinates; no angular cutoff.',
    'ties': 'Lower feature dimension, then native enum order.',
    'activation': 'Exact d2 < captured RN(dmin*dmin) + captured RN((2*dmin+dhat)*dhat).',
    'minimum': 'Exact d2 > captured RN(dmin*dmin); native finite positive separation also required.',
    'coefficients': 'Native binary64 rest areas and IPC source-role factors; retain one contribution per active source candidate, without merging coefficients.',
    'order': 'VV, EV, EE, FV; each ordered by primitive identities, feature, mollifier threshold and captured coefficient.',
    'restScale': 'Exact closest rest d2; RN(sqrt(RN(d2)))/4; reject unrepresentable squares or scales.',
    'mollifier': 'Captured native rest threshold; retain EE when exact current cross-norm squared is below it.',
    'rationalBits': 32768,
    'derivatives': 'Native forced-feature derivatives; finite/positive checks are not an arithmetic-error certificate.',
    'scope': 'Supplied candidate inventory only. Broad-phase completeness, path safety and physical accuracy require separate verification.'}


def clamp_squared(activation, minimum):
    parameters = BarrierParameters.capture(activation, minimum, 1.)
    return F(parameters.minimum_squared)+F(parameters.h)


def inflation_radius(parameters):
    """A binary64 half-radius whose doubled square covers every scalar clamp."""
    boundary = max(clamp_squared(a, m) for a, m in parameters)
    try:
        root = math.sqrt(float(boundary))
    except OverflowError as error:
        raise ValueError('Unrepresentable exact contact broad-phase radius') from error
    if not math.isfinite(root) or root <= 0:
        raise ValueError('Unrepresentable exact contact broad-phase radius')
    root = math.nextafter(root, math.inf)
    radius = math.nextafter(root/2, math.inf)
    if not math.isfinite(radius) or radius <= 0 or (2*F(radius))**2 < boundary:
        raise ValueError('Exact contact broad-phase radius does not enclose its clamp')
    return radius


def classify_candidate(kind, candidate, positions, edges, faces, budget):
    # Read scalar IDs first. Never send an unchecked candidate index into a
    # native Eigen row accessor, including for an inactive candidate.
    def index(value, size):
        value = int(value)
        if not 0 <= value < size:
            raise ValueError('In-range contact primitive index required')
        return value

    if kind == 'vv':
        ids = (index(candidate.vertex0_id, len(positions)),
               index(candidate.vertex1_id, len(positions)))
    elif kind == 'ev':
        ei = index(candidate.edge_id, len(edges))
        ids = (index(candidate.vertex_id, len(positions)), *map(int, edges[ei]))
    elif kind == 'fv':
        fi = index(candidate.face_id, len(faces))
        ids = (index(candidate.vertex_id, len(positions)), *map(int, faces[fi]))
    elif kind == 'ee':
        ea = index(candidate.edge0_id, len(edges))
        eb = index(candidate.edge1_id, len(edges))
        ids = (*map(int, edges[ea]), *map(int, edges[eb]))
    else:
        raise ValueError('Supported exact contact primitive required')
    sizes = {'vv': 2, 'ev': 3, 'ee': 4, 'fv': 4}
    if (len(ids) != sizes[kind] or len(set(ids)) != len(ids)
            or any(not 0 <= i < len(positions) for i in ids)):
        raise ValueError('Distinct in-range contact candidate vertices required')
    points = tuple(tuple(float(v) for v in positions[i]) for i in ids)
    selected, distance, ties = closest_feature(kind, points, budget)
    return ids, points, selected, distance, ties


def rest_distance(distance_squared):
    try:
        square = float(distance_squared)
    except OverflowError as error:
        raise ValueError('Unrepresentable exact contact rest distance') from error
    if not math.isfinite(square) or square <= 0:
        raise ValueError('Unrepresentable exact contact rest distance')
    return math.sqrt(square)


def _positive(value, label):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Positive representable '+label+' required')
    return value


def build_exact_collisions(mesh, positions, candidates, *, activation, minimum,
                           max_candidates):
    import ipctk

    if type(mesh) is not ipctk.CollisionMesh or type(candidates) is not ipctk.Candidates:
        raise ValueError('Native collision mesh and complete candidate inventory required')
    if (type(positions) is not np.ndarray or positions.dtype != np.dtype('float64')
            or positions.shape != (mesh.num_vertices, 3) or not np.isfinite(positions).all()):
        raise ValueError('Raw finite binary64 contact positions required')
    if type(max_candidates) is not int or not 1 <= max_candidates <= 1000000:
        raise ValueError('Explicit bounded exact contact candidate count required')
    inventory = tuple(tuple(getattr(candidates, group)) for group in GROUPS)
    count = sum(map(len, inventory))
    if count != len(candidates) or len(candidates.pv_candidates):
        raise ValueError('Unsupported or incomplete exact contact candidate inventory')
    if count > max_candidates:
        raise ValueError('Exact contact candidate budget exceeded')
    q = positions.copy()
    edges, faces = np.asarray(mesh.edges), np.asarray(mesh.faces)
    rest = np.asarray(mesh.rest_positions)
    budget = WorkBudget()
    lower, upper = F(minimum*minimum), clamp_squared(activation, minimum)
    contributions = {'vv': [], 'ev': [], 'ee': [], 'fv': []}
    active = 0
    decisions = hashlib.sha256()
    expected_types = (ipctk.VertexVertexCandidate, ipctk.EdgeVertexCandidate,
                      ipctk.EdgeEdgeCandidate, ipctk.FaceVertexCandidate)

    def add(kind, ids, weight, feature=None, eps=None):
        weight = _positive(weight, 'IPC source-role area coefficient')
        if kind == 'vv':
            ids = tuple(sorted(ids))
        elif kind == 'ee' and ids[0] > ids[1]:
            raise ValueError('Canonical ordered edge-edge candidates required')
        key = (ids, feature, eps)
        contributions[kind].append((key, weight))

    for kind, group, expected_type in zip(KINDS, inventory, expected_types):
        for candidate in group:
            if type(candidate) is not expected_type:
                raise ValueError('Unsupported exact contact candidate type')
            ids, points, feature, distance, ties = classify_candidate(
                kind, candidate, q, edges, faces, budget)
            admitted = distance < upper
            # Hex integer encoding avoids process-global decimal digit-limit changes.
            row = [kind, ids, feature, ties, hex(distance.numerator),
                   hex(distance.denominator), admitted]
            decisions.update(json.dumps(row, separators=(',', ':')).encode()+b'\n')
            if not admitted:
                continue
            if distance <= lower:
                raise ValueError('Exact contact state violates assigned minimum separation')
            active += 1
            if kind == 'vv':
                weight = .5*(mesh.vertex_area(ids[0])+mesh.vertex_area(ids[1]))
                add('vv', ids, weight)
            elif kind == 'ev':
                vi, ei = int(candidate.vertex_id), int(candidate.edge_id)
                weight = .5*mesh.vertex_area(vi)
                if feature == 'P_E':
                    add('ev', (ei, vi), weight)
                else:
                    add('vv', (vi, int(edges[ei, int(feature[-1])])), weight)
            elif kind == 'fv':
                vi, fi = int(candidate.vertex_id), int(candidate.face_id)
                weight = .25*mesh.vertex_area(vi)
                if feature == 'P_T':
                    add('fv', (fi, vi), weight)
                elif feature.startswith('P_T'):
                    add('vv', (vi, int(faces[fi, int(feature[-1])])), weight)
                else:
                    ei = int(mesh.faces_to_edges[fi, int(feature[-1])])
                    add('ev', (ei, vi), weight)
            else:
                ea, eb = int(candidate.edge0_id), int(candidate.edge1_id)
                weight = .25*(mesh.edge_area(ea)+mesh.edge_area(eb))
                eps = _positive(ipctk.edge_edge_mollifier_threshold(*rest[list(ids)]),
                                'native rest mollifier threshold')
                p = tuple(tuple(F(x) for x in row) for row in points)
                normal = _cross(_sub(p[1], p[0]), _sub(p[3], p[2]))
                cross_squared = budget.check(_dot(normal, normal))
                if cross_squared < F(eps) or feature == 'EA_EB':
                    add('ee', (ea, eb), weight, feature, eps)
                elif feature.startswith('EA_EB'):
                    add('ev', (ea, ids[2+int(feature[-1])]), weight)
                elif feature.endswith('_EB'):
                    add('ev', (eb, ids[int(feature[2])]), weight)
                else:
                    add('vv', (ids[int(feature[2])], ids[2+int(feature[-1])]), weight)

    bucket = ipctk.NormalCollisions()
    bucket.use_area_weighting = True
    bucket.collision_set_type = ipctk.NormalCollisions.IPC
    bucket.enable_shape_derivatives = False
    constructors = {'vv': ipctk.VertexVertexNormalCollision,
                    'ev': ipctk.EdgeVertexNormalCollision,
                    'fv': ipctk.FaceVertexNormalCollision}
    for kind in ('vv', 'ev', 'ee', 'fv'):
        entries = sorted(contributions[kind])
        collisions = []
        for (ids, feature, eps), coefficient in entries:
            weight = coefficient
            if kind == 'ee':
                collision = ipctk.EdgeEdgeNormalCollision(
                    *ids, eps, getattr(ipctk.EdgeEdgeDistanceType, feature))
            else:
                collision = constructors[kind](*ids)
            collision.weight, collision.dmin = weight, minimum
            local = np.asarray(collision.dof(q, edges, faces))
            native_distance = float(collision.compute_distance(local))
            if not math.isfinite(native_distance) or native_distance <= minimum*minimum:
                raise ValueError('Native forced-feature separation is not representable')
            if kind == 'ee':
                p = tuple(tuple(F(float(v)) for v in row) for row in local.reshape((-1, 3)))
                normal = _cross(_sub(p[1], p[0]), _sub(p[3], p[2]))
                exact_cross = _dot(normal, normal)
                native_mollifier = float(collision.mollifier(local))
                if (not math.isfinite(native_mollifier) or not 0 <= native_mollifier <= 1
                        or (exact_cross > 0 and native_mollifier == 0)):
                    raise ValueError('Native forced-feature mollifier loses positive geometry')
            # Required primitive derivatives must be usable before publishing
            # the bucket; zero weights/mollifiers never hide a singular formula.
            for derivative in (collision.compute_distance_gradient(local),
                               collision.compute_distance_hessian(local),
                               collision.mollifier_gradient(local),
                               collision.mollifier_hessian(local)):
                if not np.isfinite(np.asarray(derivative)).all():
                    raise ValueError('Native forced-feature derivative is not representable')
            collisions.append(collision)
        setattr(bucket, kind+'_collisions', collisions)
    if len(bucket) != active:
        raise ValueError('Exact contact construction lost source-candidate multiplicity')
    certificate = {'profile': PROFILE, 'accepted': False, 'candidates': count,
        'activeCandidates': active, 'collisions': len(bucket),
        'positionsSha256': hashlib.sha256(np.asarray(q, dtype='<f8').tobytes()).hexdigest(),
        'decisionsSha256': decisions.hexdigest()}
    return bucket, certificate
