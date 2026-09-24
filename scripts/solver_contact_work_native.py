"""Complete native endpoint capture for bounded contact scalar work.

Only the existing area-weighted IPC rest-filtered adapter is admitted. The
explicit solver contact-work policy uses these captures for conditional work
intervals. Motion paths, native gradient errors and source ownership are not
certified here; exact geometry checks each selected native closest feature.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import math

import numpy as np

from solver_contact_work import (BarrierParameters, ContactTerm, WorkBudget,
    WorkPolicy, contact_work, distance_squared, _single_initialization)
from solver_rest_filtered_contact import RestFilteredSurfaceContact


@_single_initialization
@dataclass(frozen=True, slots=True)
class NativeObservation:
    known_feature: str
    selected_feature: str
    distance_squared: float
    exact_distance_squared: F
    mollifier: float


@_single_initialization
@dataclass(frozen=True, slots=True)
class NativeEndpoint:
    terms: tuple
    observations: tuple
    positions_sha256: str
    native_energy: float


def capture_endpoint(contact, positions, *, policy=None):
    import ipctk

    if type(contact) is not RestFilteredSurfaceContact:
        raise ValueError("Bounded native work requires the fixed rest-filtered IPC adapter")
    policy = WorkPolicy() if policy is None else policy
    budget = WorkBudget(policy)
    if type(positions) is not np.ndarray or positions.dtype != np.dtype('float64'):
        raise ValueError("Raw native binary64 position array required")
    # Copy actual binary64 endpoints before entering native code. The adapter's
    # existing finite shape/minimum checks remain in force.
    q = contact._positions(positions).copy()
    q, buckets = contact._buckets(q)
    if sum(len(bucket) for _,bucket in buckets) > policy.max_endpoint_terms:
        raise ValueError("Contact-work endpoint-term budget exhausted")
    terms, observations, energies = [], [], []
    kinds = {'VertexVertexNormalCollision':('vv',None),
             'EdgeVertexNormalCollision':('ev','point_edge_distance_type'),
             'FaceVertexNormalCollision':('fv','point_triangle_distance_type'),
             'EdgeEdgeNormalCollision':('ee','edge_edge_distance_type')}
    for index,bucket in buckets:
        if not bucket.use_area_weighting or bucket.collision_set_type != ipctk.NormalCollisions.IPC:
            raise ValueError("Native collision set differs from area-weighted IPC")
        activation,minimum = contact._contact_parameters[index]
        potential = contact._potentials[index]
        if type(potential.barrier).__name__ != 'ClampedLogBarrier':
            raise ValueError("Native contact barrier is not clamped logarithmic")
        parameters = BarrierParameters.capture(activation,minimum,contact.stiffness)
        for i in range(len(bucket)):
            collision = bucket[i]
            name = type(collision).__name__
            if name not in kinds:
                raise ValueError("Unsupported native contact stencil")
            kind,selector = kinds[name]
            ids = tuple(int(v) for v in collision.vertex_ids(contact._edges,contact.faces) if v >= 0)
            local = np.asarray(collision.dof(q,contact._edges,contact.faces)).reshape((-1,3))
            if not np.array_equal(local,q[list(ids)]):
                raise ValueError("Native stencil coordinates differ from captured vertices")
            known = str(collision.known_dtype()).split('.')[-1] if hasattr(collision,'known_dtype') else 'P_P'
            selected = known
            if known == 'AUTO':
                if selector is None:
                    raise ValueError("Unsupported native automatic distance type")
                selected = str(getattr(ipctk,selector)(*local)).split('.')[-1]
            weight, dmin = float(collision.weight),float(collision.dmin)
            if not math.isfinite(weight) or weight < 0 or dmin != minimum:
                raise ValueError("Native weight/minimum differs from admitted IPC law")
            mollified = bool(collision.is_mollified())
            if mollified != (kind == 'ee'):
                raise ValueError("Unsupported native mollifier")
            term = ContactTerm(index,kind,ids,
                tuple(sorted((field,int(getattr(collision,field))) for field in (
                    'vertex0_id','vertex1_id','vertex_id','edge_id','edge0_id','edge1_id','face_id')
                    if hasattr(collision,field))),selected,
                tuple(tuple(float(v) for v in row) for row in local),weight,
                float(collision.eps_x) if kind=='ee' else None,parameters)
            native_distance = float(collision.compute_distance(local.ravel()))
            native_mollifier = float(collision.mollifier(local.ravel()))
            if (not math.isfinite(native_distance) or native_distance <= parameters.minimum_squared
                    or not math.isfinite(native_mollifier) or not 0 <= native_mollifier <= 1):
                raise ValueError("Invalid native contact endpoint observation")
            exact = distance_squared(kind,selected,term.positions,budget)
            if exact <= F(parameters.minimum_squared):
                raise ValueError("Exact contact state violates captured minimum separation")
            terms.append(term)
            observations.append(NativeObservation(known,selected,native_distance,exact,native_mollifier))
        energies.append(float(potential(bucket,contact.mesh,q)))
    energy = float(sum(energies))  # same bucket order; empty inventory is binary64 zero
    if not math.isfinite(energy) or energy < 0:
        raise ValueError("Invalid native contact energy observation")
    contact._check_native_parameters()
    digest = hashlib.sha256(np.asarray(q,dtype='<f8').tobytes(order='C')).hexdigest()
    return NativeEndpoint(tuple(terms),tuple(observations),digest,energy)


def bounded_native_work(contact, start, end, *, policy=None):
    """Capture each actual endpoint separately; never reuse only one active set."""
    first = capture_endpoint(contact,start,policy=policy)
    last = capture_endpoint(contact,end,policy=policy)
    result = contact_work(first.terms,last.terms,policy=policy)
    return result,first,last
