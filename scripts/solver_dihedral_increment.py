"""Stable signed angle changes for a validated subset of dihedral hinges.

This is an endpoint arithmetic helper, not a swept-geometry certificate. The
caller must retain independent cloth/contact/triangle and hinge-path guards.
It neither changes material rest angles nor installs an actuator.
"""

from decimal import Decimal, localcontext
from fractions import Fraction
import math

import numpy as np

from solver_bending import ElasticDihedralBending


def _norm(values):
    # Selection must remain finite even when a valid endpoint is many orders
    # of magnitude larger than the other. Squared norms can overflow before
    # that update reaches the deliberately conservative endpoint fallback.
    return np.hypot(np.hypot(values[..., 0], values[..., 1]), values[..., 2])


def _cross_increment(first, second, first_change, second_change):
    return (np.cross(first_change, second) + np.cross(first, second_change)
            + np.cross(first_change, second_change))


def _exact_local_increment(start, end):
    """Resolve cancellation from the original binary64 endpoint coordinates.

    For each endpoint let T=(n1 cross n2) dot edge, D=n1 dot n2 and
    E=edge dot edge, all exact rationals. The signed angle has the same atan2
    arguments as (T, D*sqrt(E)). Rationalizing the difference of radicals in
    the increment numerator retains tiny angular changes superimposed on a
    much larger scale/shape change. No exact-similarity special case is used.
    """
    def sub(a, b):
        return [x-y for x,y in zip(a,b)]

    def cross(a, b):
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

    def dot(a, b):
        return sum((x*y for x,y in zip(a,b)), Fraction())

    def terms(values):
        q = [[Fraction(float(value)) for value in point] for point in values]
        first = cross(sub(q[2],q[0]), sub(q[3],q[0]))
        second = cross(sub(q[3],q[1]), sub(q[2],q[1]))
        edge = sub(q[3],q[2])
        return dot(cross(first,second),edge), dot(first,second), dot(edge,edge)

    t0,d0,e0 = terms(start)
    t1,d1,e1 = terms(end)
    a,b = t1*d0, t0*d1
    with localcontext() as context:
        context.prec = 120

        def decimal(value):
            return Decimal(value.numerator) / Decimal(value.denominator)

        root0,root1 = decimal(e0).sqrt(), decimal(e1).sqrt()
        if a*b > 0:
            numerator = decimal(a*a*e0-b*b*e1) / (decimal(a)*root0+decimal(b)*root1)
        else:
            # These terms have opposite signs (or one is zero), so their
            # difference cannot lose a small result through cancellation.
            numerator = decimal(a)*root0-decimal(b)*root1
        denominator = decimal(d0*d1)*(decimal(e0*e1).sqrt()) + decimal(t0*t1)
        if denominator <= 0:
            raise ValueError("Local dihedral update exceeds its supported angle increment")
        ratio = numerator / denominator
        rounded = float(ratio)
        if ratio != 0 and rounded == 0:
            raise ValueError("Nonzero dihedral increment underflows binary64")
        result = math.atan(rounded)
        if not math.isfinite(result):
            raise ValueError("Finite cancellation-resolved dihedral increment required")
        return result


def dihedral_angle_increment(geometry, start, end):
    """Return binary64 start angles and stable signed endpoint increments.

    The start angle uses the existing binary64 geometry sampler exactly. Small
    changes are evaluated from source coordinate displacements, rather than by
    subtracting rounded atan2 values. Large changes use the independently
    validated endpoint angles. Long-double intermediates improve precision
    where available; the local expansion also works when they are binary64.
    Near-cancelling local updates use exact binary-input scalar products and
    a rationalized radical expression, so scaling cannot hide a tiny angular
    change. Its final noncancelling radical evaluation uses 120-digit Decimal.

    Only the supplied hinge subset is visited. Empty subsets still validate
    both complete position arrays. Active endpoint degeneracy and principal
    branch violations reject; endpoint checks alone cannot certify the path.
    """
    if not isinstance(geometry, ElasticDihedralBending):
        raise ValueError("Validated elastic dihedral geometry required")
    try:
        start = geometry._positions(np.asarray(start, dtype=np.float64))
        end = geometry._positions(np.asarray(end, dtype=np.float64))
    except (TypeError, OverflowError) as error:
        raise ValueError("Finite correctly shaped dihedral endpoint positions required") from error
    if not len(geometry.indices):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    first_geometry = geometry._geometry(start)
    last_geometry = geometry._geometry(end)
    start_angles = first_geometry[0].copy()
    delta = last_geometry[0] - start_angles
    if np.any(np.abs(delta) >= np.pi):
        raise ValueError("Dihedral update crosses the principal-angle branch")

    # Compute source displacements before edge/normal differences. In
    # particular, do not obtain these by subtracting rounded endpoint normals.
    previous = start.astype(np.longdouble)[geometry.indices]
    displacement = (end.astype(np.longdouble) - start.astype(np.longdouble))[geometry.indices]
    first_start = previous[:, 2] - previous[:, 0]
    first_end = previous[:, 3] - previous[:, 0]
    second_end = previous[:, 3] - previous[:, 1]
    second_start = previous[:, 2] - previous[:, 1]
    edge = previous[:, 3] - previous[:, 2]
    first_start_change = displacement[:, 2] - displacement[:, 0]
    first_end_change = displacement[:, 3] - displacement[:, 0]
    second_end_change = displacement[:, 3] - displacement[:, 1]
    second_start_change = displacement[:, 2] - displacement[:, 1]
    edge_change = displacement[:, 3] - displacement[:, 2]

    # A common power-of-two scale for each hinge keeps the displacement
    # expansion away from squared/cross-product overflow and underflow without
    # changing its angle. Admission is still the existing endpoint geometry's
    # admission, performed above on the original coordinates.
    largest = np.max(np.abs(np.stack((first_start, first_end, second_start, second_end, edge))), axis=(0, 2))
    _, exponents = np.frexp(largest)

    def scaled(values):
        return np.ldexp(values, -exponents[:, None])

    first_start, first_end, second_end, second_start, edge = (
        scaled(value) for value in (first_start, first_end, second_end, second_start, edge))
    first_start_change, first_end_change, second_end_change, second_start_change, edge_change = (
        scaled(value) for value in (first_start_change, first_end_change, second_end_change, second_start_change, edge_change))

    local = np.ones(len(geometry.indices), dtype=bool)
    relative_motion = np.zeros(len(geometry.indices), dtype=np.longdouble)
    for vector, change in ((first_start, first_start_change), (first_end, first_end_change),
                           (second_start, second_start_change), (second_end, second_end_change), (edge, edge_change)):
        ratio = _norm(change) / _norm(vector)
        relative_motion = np.maximum(relative_motion, ratio)
        local &= ratio <= np.longdouble(.125)
    indices = np.flatnonzero(local)
    if not len(indices):
        return start_angles, delta

    first_start, first_end, second_end, second_start, edge = (
        value[indices] for value in (first_start, first_end, second_end, second_start, edge))
    first_start_change, first_end_change, second_end_change, second_start_change, edge_change = (
        value[indices] for value in (first_start_change, first_end_change, second_end_change, second_start_change, edge_change))
    first_normal = np.cross(first_start, first_end)
    second_normal = np.cross(second_end, second_start)
    first_change = _cross_increment(first_start, first_end, first_start_change, first_end_change)
    second_change = _cross_increment(second_end, second_start, second_end_change, second_start_change)
    normal_motion = np.maximum(_norm(first_change)/_norm(first_normal), _norm(second_change)/_norm(second_normal))
    relative_motion = np.maximum(relative_motion[indices], normal_motion)
    poorly_conditioned = ((_norm(first_normal) <= np.longdouble(2.**-16)*_norm(first_start)*_norm(first_end))
                          | (_norm(second_normal) <= np.longdouble(2.**-16)*_norm(second_start)*_norm(second_end)))
    # Large normal cancellation is another reason to retain the endpoint
    # fallback, even if each source edge moved only a modest amount.
    reliable = ((_norm(first_change) <= np.longdouble(.125) * _norm(first_normal))
                & (_norm(second_change) <= np.longdouble(.125) * _norm(second_normal)))
    indices = indices[reliable]
    relative_motion, poorly_conditioned = relative_motion[reliable], poorly_conditioned[reliable]
    if not len(indices):
        return start_angles, delta
    first_normal, second_normal, first_change, second_change, edge, edge_change = (
        value[reliable] for value in (first_normal, second_normal, first_change, second_change, edge, edge_change))

    # The positive normal-length factors cancel between the sine and cosine
    # arguments of atan2, so unnormalized normals avoid two unnecessary unit
    # normal divisions. Only the edge direction needs normalization. Its
    # increment uses the rationalized length change from the fold barrier.
    edge_length, next_edge_length = _norm(edge), _norm(edge + edge_change)
    edge_unit = edge / edge_length[:, None]
    length_change = np.sum((2 * edge + edge_change) * edge_change, axis=1) / (edge_length + next_edge_length)
    unit_change = (edge_change - edge_unit * length_change[:, None]) / next_edge_length[:, None]
    cross = np.cross(first_normal, second_normal)
    cross_change = _cross_increment(first_normal, second_normal, first_change, second_change)
    sine = np.sum(cross * edge_unit, axis=1)
    cosine = np.sum(first_normal * second_normal, axis=1)
    sine_change = np.sum(cross_change * edge_unit + (cross + cross_change) * unit_change, axis=1)
    cosine_change = np.sum(first_change * second_normal + (first_normal + first_change) * second_change, axis=1)
    stable = np.arctan2(cosine * sine_change - sine * cosine_change,
                        cosine * (cosine + cosine_change) + sine * (sine + sine_change))
    if not np.isfinite(stable).all() or np.any(np.abs(stable) >= np.pi):
        raise ValueError("Nonfinite or principal-branch dihedral increment")
    # A small angular signal riding on a larger coordinate/normal change can
    # cancel inside the displacement expansion (uniform scale is one case).
    # The 1/32 switch is a precision strategy, not an acceptance tolerance.
    # Poorly conditioned support also uses original-input exact products.
    cancellation = ((relative_motion > 0)
                    & ((np.abs(stable) <= relative_motion/32) | poorly_conditioned))
    for row in np.flatnonzero(cancellation):
        vertices = geometry.indices[indices[row]]
        stable[row] = _exact_local_increment(start[vertices], end[vertices])
    delta[indices] = np.asarray(stable, dtype=np.float64)
    if not np.isfinite(delta).all():
        raise ValueError("Finite signed dihedral increments required")
    return start_angles, delta
