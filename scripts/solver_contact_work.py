"""Bounded work for captured IPC clamped-log contact stencils.

This standalone arithmetic primitive does not build candidates or drive a solve.
It encloses a declared real scalar formula: exact binary-input geometry and
mollifiers, captured native-rounded minimum²/H/normalization, and exact products
of stored pressure and weight. Native energy/gradient and feature selection have
their own floating-point errors, which are NOT enclosed by this work radius.
Actual native endpoint inventories must be supplied separately and completely.
"""
from collections import defaultdict
from dataclasses import dataclass, fields
from fractions import Fraction as F
import math
from functools import lru_cache, wraps


PROFILE = "captured-ipc-contact-work-v1"


def _single_initialization(cls):
    """Reject an ordinary second __init__ before frozen fields can be changed.

    These Python records are not a security boundary against arbitrary code
    using object.__setattr__ or editing classes.
    """
    original = cls.__init__
    sentinel = fields(cls)[0].name

    @wraps(original)
    def initialize(self, *args, **kwargs):
        if hasattr(self,sentinel):
            raise AttributeError("Immutable contact record is already initialized")
        original(self,*args,**kwargs)

    cls.__init__ = initialize
    return cls


def _binary(value):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("Finite binary64 input required")
    return F(value)


@_single_initialization
@dataclass(frozen=True, slots=True)
class WorkPolicy:
    bits: int = 160
    max_log_terms: int = 96
    max_total_log_terms: int = 100000
    max_fraction_bits: int = 32768
    max_endpoint_terms: int = 10000

    def __post_init__(self):
        for name, lower, upper in (("bits", 64, 512), ("max_log_terms", 1, 512),
                ("max_total_log_terms", 1, 1000000),
                ("max_fraction_bits", 4096, 131072), ("max_endpoint_terms", 1, 100000)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError("Invalid contact-work arithmetic budget")


class WorkBudget:
    def __init__(self, policy=None):
        self.policy = WorkPolicy() if policy is None else policy
        if type(self.policy) is not WorkPolicy:
            raise ValueError("Explicit contact work policy required")
        self.log_terms = 0
        self.log_calls = 0
        self._log_two = None

    def check(self, value):
        if type(value) is not F:
            raise ValueError("Exact rational arithmetic input required")
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > self.policy.max_fraction_bits:
            raise ValueError("Contact-work rational-size budget exhausted")
        return value

    def term(self):
        self.log_terms += 1
        if self.log_terms > self.policy.max_total_log_terms:
            raise ValueError("Contact-work total logarithm budget exhausted")


@_single_initialization
@dataclass(frozen=True, slots=True)
class Interval:
    lower: F
    upper: F

    def __post_init__(self):
        if type(self.lower) is not F or type(self.upper) is not F or self.lower > self.upper:
            raise ValueError("Ordered rational interval required")

    def __add__(self, other):
        return Interval(self.lower + other.lower, self.upper + other.upper)

    def scale(self, factor):
        if type(factor) is not F:
            raise ValueError("Exact interval multiplier required")
        a, b = self.lower * factor, self.upper * factor
        return Interval(min(a, b), max(a, b))


ZERO = Interval(F(), F())


def _ceil_div(numerator, denominator):
    return -(-numerator // denominator)


def _atanh_series(x, budget):
    """Enclose log(x) for exact 1<=x<=2 using outward dyadic arithmetic.

    z=(x-1)/(x+1) is nonnegative and at most 1/3. The positive
    atanh-series remainder after N terms is <=
    2*z**(2*N+1)/((2*N+1)*(1-z*z)). Every fixed-point product/division
    rounds outward, so the returned interval includes all rounding and tail.
    """
    if not 1 <= x <= 2:
        raise ValueError("Logarithm range reduction failed")
    if x == 1:
        return ZERO
    budget.check(x)
    unit = 1 << budget.policy.bits
    z = budget.check((x-1)/(x+1))
    lo = z.numerator*unit // z.denominator
    hi = _ceil_div(z.numerator*unit, z.denominator)
    z2lo = lo*lo // unit
    z2hi = _ceil_div(hi*hi, unit)
    power_lo, power_hi = lo, hi
    sum_lo = sum_hi = 0
    for n in range(budget.policy.max_log_terms):
        budget.term()
        denominator = 2*n+1
        sum_lo += 2*power_lo // denominator
        sum_hi += _ceil_div(2*power_hi, denominator)
        power_lo = power_lo*z2lo // unit
        power_hi = _ceil_div(power_hi*z2hi, unit)
        # ceil of the mathematical upper tail, in fixed-point units.
        tail = _ceil_div(2*power_hi*unit, (denominator+2)*(unit-z2hi))
        if tail <= 1:
            return Interval(F(sum_lo,unit), F(sum_hi+tail,unit))
    raise ValueError("Contact-work per-logarithm term budget exhausted")


def log_interval(x, budget):
    """Rigorous interval for log of an exact positive rational."""
    budget.check(x)
    if x <= 0:
        raise ValueError("Positive logarithm argument required")
    budget.log_calls += 1
    if x == 1:
        return ZERO
    sign = F(1)
    if x < 1:
        x, sign = 1/x, F(-1)
    exponent = x.numerator.bit_length() - x.denominator.bit_length()
    scale = 1 << exponent
    if x < scale:
        exponent -= 1
        scale >>= 1
    reduced = budget.check(x/scale)
    result = _atanh_series(reduced, budget)
    if exponent:
        if budget._log_two is None:
            budget._log_two = _atanh_series(F(2), budget)
        result = result + budget._log_two.scale(F(exponent))
    return result.scale(sign)


@_single_initialization
@dataclass(frozen=True, slots=True)
class BarrierParameters:
    """Captured native factorization, with no rounded composite coefficient."""
    activation: float
    minimum: float
    pressure: float
    minimum_squared: float
    h: float
    h_squared: float
    normalization: float
    candidate_outer_squared: float

    @classmethod
    def capture(cls, activation, minimum, pressure):
        for v in (activation, minimum, pressure):
            if _binary(v) <= 0:
                raise ValueError("Positive contact parameters required")
        h = (2*minimum + activation)*activation
        h2 = h*h
        if not math.isfinite(h2) or h2 <= 0:
            raise ValueError("Unrepresentable native barrier normalization")
        return cls(activation,minimum,pressure,minimum*minimum,h,h2,
                   activation/h2,(minimum+activation)*(minimum+activation))

    def __post_init__(self):
        for value in (getattr(self,field.name) for field in fields(self)):
            if _binary(value) <= 0:
                raise ValueError("Positive representable native barrier constants required")
        expected_h = (2*self.minimum+self.activation)*self.activation
        if (self.minimum_squared != self.minimum*self.minimum or self.h != expected_h
                or self.h_squared != self.h*self.h
                or self.normalization != self.activation/self.h_squared
                or self.candidate_outer_squared != (self.minimum+self.activation)*(self.minimum+self.activation)):
            raise ValueError("Captured barrier constants differ from declared native factorization")


def barrier_value(s, h, budget):
    budget.check(s)
    budget.check(h)
    if s <= 0 or h <= 0:
        raise ValueError("Positive contact separation and activation required")
    if s >= h:
        return ZERO
    return log_interval(s/h, budget).scale(-(s-h)**2)


def barrier_change(s0, s1, h, budget):
    for value in (s0,s1,h):
        budget.check(value)
        if value <= 0:
            raise ValueError("Positive contact separation and activation required")
    if s0 == s1 or (s0 >= h and s1 >= h):
        return ZERO
    if s0 >= h or s1 >= h:
        return barrier_value(s1,h,budget) + barrier_value(s0,h,budget).scale(F(-1))
    delta, a0, a1 = s1-s0, s0-h, s1-h
    # log(s1/s0) is log1p(delta/s0), formed without any binary64 rounding.
    return (log_interval(s0/h,budget).scale(-delta*(a1+a0))
            + log_interval(s1/s0,budget).scale(-a1*a1))


def _sub(a,b):
    return tuple(x-y for x,y in zip(a,b))


def _dot(a,b):
    return sum((x*y for x,y in zip(a,b)),F())


def _cross(a,b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _point_edge(p,a,b):
    direction, offset = _sub(b,a), _sub(p,a)
    norm = _dot(direction,direction)
    if norm <= 0:
        raise ValueError("Degenerate contact edge")
    t = _dot(offset,direction)/norm
    features = {"P_E0":_dot(offset,offset),"P_E1":_dot(_sub(p,b),_sub(p,b))}
    if 0 <= t <= 1:
        features["P_E"] = _dot(offset,offset)-_dot(offset,direction)**2/norm
    return features


def _distance_choices_uncached(kind, positions):
    """All feasible finite-primitive features, on exact binary input geometry."""
    sizes = {"vv":2,"ev":3,"fv":4,"ee":4}
    if kind not in sizes or type(positions) is not tuple or len(positions) != sizes[kind]:
        raise ValueError("Supported ordered contact stencil required")
    if any(type(row) is not tuple or len(row) != 3 for row in positions):
        raise ValueError("Three-dimensional immutable contact coordinates required")
    p = tuple(tuple(_binary(v) for v in row) for row in positions)
    if kind == "vv":
        d = _sub(*p)
        choices = {"P_P":_dot(d,d)}
    elif kind == "ev":
        choices = _point_edge(*p)
    elif kind == "fv":
        point,a,b,c = p
        u,v,w = _sub(b,a),_sub(c,a),_sub(point,a)
        uu,vv,uv,wu,wv = _dot(u,u),_dot(v,v),_dot(u,v),_dot(w,u),_dot(w,v)
        determinant = uu*vv-uv*uv
        if determinant <= 0:
            raise ValueError("Degenerate contact face")
        choices = {f"P_T{i}":_dot(_sub(point,q),_sub(point,q)) for i,q in enumerate((a,b,c))}
        for i,(q,r) in enumerate(((a,b),(b,c),(c,a))):
            edges = _point_edge(point,q,r)
            if "P_E" in edges:
                choices[f"P_E{i}"] = edges["P_E"]
        alpha,beta = (wu*vv-wv*uv)/determinant,(wv*uu-wu*uv)/determinant
        if alpha >= 0 and beta >= 0 and alpha+beta <= 1:
            normal = _cross(u,v)
            choices["P_T"] = _dot(w,normal)**2/_dot(normal,normal)
    else:
        a,b,c,d = p
        choices = {}
        for i,q in enumerate((a,b)):
            for j,r in enumerate((c,d)):
                delta = _sub(q,r)
                choices[f"EA{i}_EB{j}"] = _dot(delta,delta)
            edge = _point_edge(q,c,d)
            if "P_E" in edge:
                choices[f"EA{i}_EB"] = edge["P_E"]
        for j,r in enumerate((c,d)):
            edge = _point_edge(r,a,b)
            if "P_E" in edge:
                choices[f"EA_EB{j}"] = edge["P_E"]
        u,v,w = _sub(b,a),_sub(d,c),_sub(a,c)
        uu,vv,uv,uw,vw = _dot(u,u),_dot(v,v),_dot(u,v),_dot(u,w),_dot(v,w)
        determinant = uu*vv-uv*uv
        if determinant > 0:
            t,s = (uv*vw-vv*uw)/determinant,(uu*vw-uv*uw)/determinant
            if 0 <= t <= 1 and 0 <= s <= 1:
                normal = _cross(u,v)
                choices["EA_EB"] = _dot(w,normal)**2/_dot(normal,normal)
    return choices


@lru_cache(maxsize=256)
def _distance_choice_integers(kind, positions):
    """Bounded pure-geometry memo; never retain caller-visible Fractions.

    Each key contains at most four finite binary64 points and each value at
    most nine finite-feature distances. No candidates, weights, activation,
    selected feature or work/budget result is cached here.
    """
    choices = _distance_choices_uncached(kind, positions)
    return tuple((name, value.numerator, value.denominator)
                 for name, value in choices.items())


def _distance_choices(kind, positions):
    # Validate before lookup: Python considers bool/int coordinates equal to
    # floats in a dictionary key, but the arithmetic API requires raw floats.
    sizes = {"vv":2,"ev":3,"fv":4,"ee":4}
    if kind not in sizes or type(positions) is not tuple or len(positions) != sizes[kind]:
        raise ValueError("Supported ordered contact stencil required")
    if any(type(row) is not tuple or len(row) != 3 for row in positions):
        raise ValueError("Three-dimensional immutable contact coordinates required")
    for row in positions:
        for value in row:
            _binary(value)
    # Preserve the prior behavior for nonstandard string-like kind objects
    # without allowing their custom equality/hash methods into shared keys.
    if type(kind) is not str:
        return _distance_choices_uncached(kind, positions)
    return {name: F(numerator, denominator)
            for name, numerator, denominator in _distance_choice_integers(kind, positions)}


def distance_squared(kind, feature, positions, budget):
    """Exact selected-feature distance, admitted only if it is an exact minimum.

    No native feature is replaced. Exact ties are admissible; inconsistent
    native predicates and degenerate primitive geometry explicitly reject.
    """
    choices = _distance_choices(kind, positions)
    if feature not in choices or choices[feature] != min(choices.values()):
        raise ValueError("Native closest feature is not an exact binary-input minimum")
    return budget.check(choices[feature])


def closest_feature(kind, positions, budget):
    """Classify before construction; this does not replace a captured feature.

    Exact ties prefer a lower-dimensional feature, then the fixed native enum
    order. No angular cutoff treats a nonparallel edge pair as parallel. All
    feasible distances are checked against the supplied rational-size budget.
    The legacy captured-feature validator above retains its original behavior.
    """
    orders = {
        'vv': ('P_P',),
        'ev': ('P_E0', 'P_E1', 'P_E'),
        'fv': ('P_T0', 'P_T1', 'P_T2', 'P_E0', 'P_E1', 'P_E2', 'P_T'),
        'ee': ('EA0_EB0', 'EA0_EB1', 'EA1_EB0', 'EA1_EB1',
               'EA_EB0', 'EA_EB1', 'EA0_EB', 'EA1_EB', 'EA_EB')}
    choices = _distance_choices(kind, positions)
    for value in choices.values():
        budget.check(value)
    minimum = min(choices.values())
    tied = tuple(feature for feature in orders[kind]
                 if choices.get(feature) == minimum)
    return tied[0], minimum, tied


@_single_initialization
@dataclass(frozen=True, slots=True)
class ContactTerm:
    bucket: int
    kind: str
    vertex_ids: tuple
    native_identity: tuple
    feature: str
    positions: tuple
    weight: float
    eps_x: float | None
    parameters: BarrierParameters

    def __post_init__(self):
        if type(self.bucket) is not int or self.bucket < 0 or type(self.parameters) is not BarrierParameters:
            raise ValueError("Captured bucket and barrier parameters required")
        if (type(self.positions) is not tuple or any(type(row) is not tuple or len(row) != 3 for row in self.positions)
                or type(self.vertex_ids) is not tuple or len(self.vertex_ids) != len(self.positions)
                or any(type(v) is not int or v < 0 for v in self.vertex_ids)
                or len(set(self.vertex_ids)) != len(self.vertex_ids)):
            raise ValueError("Distinct ordered vertex identities required")
        for row in self.positions:
            for value in row:
                _binary(value)
        if self.kind not in ("vv","ev","fv","ee") or type(self.feature) is not str:
            raise ValueError("Explicit primitive kind and feature required")
        if (type(self.native_identity) is not tuple or any(type(row) is not tuple or len(row) != 2
                or type(row[0]) is not str or type(row[1]) is not int or row[1] < 0 for row in self.native_identity)):
            raise ValueError("Immutable native primitive identity required")
        _binary(self.weight)  # signed algebra is supported; profile admission is external
        if self.kind == "ee":
            if _binary(self.eps_x) <= 0:
                raise ValueError("Positive native edge mollifier threshold required")
        elif self.eps_x is not None:
            raise ValueError("Mollifier is only defined for edge-edge collisions")

    @property
    def key(self):
        # Do not coalesce duplicates. Features/weights/eps can change between
        # endpoints; both declared endpoint contributions remain represented.
        return (self.bucket,self.kind,self.vertex_ids,self.native_identity,self.parameters)

    def scalar(self, budget):
        distance = distance_squared(self.kind,self.feature,self.positions,budget)
        s = budget.check(distance-F(self.parameters.minimum_squared))
        if s <= 0:
            raise ValueError("Exact contact state violates captured minimum separation")
        multiplier = F(self.weight)*F(self.parameters.pressure)*F(self.parameters.normalization)
        if self.kind == "ee":
            p = tuple(tuple(F(v) for v in row) for row in self.positions)
            normal = _cross(_sub(p[1],p[0]),_sub(p[3],p[2]))
            z = budget.check(_dot(normal,normal)/F(self.eps_x))
            multiplier *= 2*z-z*z if z < 1 else 1
        return s, budget.check(multiplier)


@_single_initialization
@dataclass(frozen=True, slots=True)
class WorkResult:
    value: float
    absolute_error: F
    lower: F
    upper: F
    endpoint_terms: int
    union_terms: int
    log_calls: int
    log_terms: int
    profile: str = PROFILE

    def binary64_error_bound(self):
        """Compact outward bound for reports, including positive underflow.

        Do not serialize large raw Fraction denominators or change Python's
        integer-string limit. Conditional arithmetic can keep the exact radius;
        a JSON certificate can use this larger finite binary64 radius.
        """
        try:
            bound = float(self.absolute_error)
        except OverflowError as error:
            raise ValueError("Unrepresentable contact-work error certificate") from error
        if F(bound) < self.absolute_error:
            bound = math.nextafter(bound,math.inf)
        if not math.isfinite(bound):
            raise ValueError("Unrepresentable contact-work error certificate")
        return bound


def contact_work(start, end, *, policy=None):
    """Sum complete endpoint inventories, retaining multiplicity and missing=0.

    The inputs are immutable native captures. This does not certify that a
    caller supplied every collision; a separate native adapter must do so.
    """
    budget = WorkBudget(policy)
    if type(start) is not tuple or type(end) is not tuple:
        raise ValueError("Immutable complete endpoint inventories required")
    count = len(start)+len(end)
    if count > budget.policy.max_endpoint_terms:
        raise ValueError("Contact-work endpoint-term budget exhausted")
    grouped = [defaultdict(list),defaultdict(list)]
    # Evaluate both endpoints even for zero weight/equal positions, validating
    # geometry and minimum distance without concealing inadmissible terms.
    for inventory,groups in zip((start,end),grouped):
        vertices, parameters = {}, {}
        for term in inventory:
            if type(term) is not ContactTerm:
                raise ValueError("Captured immutable contact term required")
            for index,position in zip(term.vertex_ids,term.positions):
                if index in vertices and position != vertices[index]:
                    raise ValueError("Inconsistent endpoint vertex coordinates")
                vertices[index] = position
            if term.bucket in parameters and term.parameters != parameters[term.bucket]:
                raise ValueError("Inconsistent endpoint bucket parameters")
            parameters[term.bucket] = term.parameters
            groups[term.key].append(term.scalar(budget))
    result = ZERO
    union_terms = 0
    for key in dict.fromkeys((*grouped[0],*grouped[1])):
        first,last = grouped[0].get(key,()),grouped[1].get(key,())
        h = F(key[-1].h)
        for i in range(max(len(first),len(last))):
            union_terms += 1
            if i >= len(first):
                s,m = last[i]
                increment = ZERO if not m else barrier_value(s,h,budget).scale(m)
            elif i >= len(last):
                s,m = first[i]
                increment = ZERO if not m else barrier_value(s,h,budget).scale(-m)
            else:
                s0,m0 = first[i]
                s1,m1 = last[i]
                increment = ZERO
                if m1 != m0:
                    increment = barrier_value(s0,h,budget).scale(m1-m0)
                if m1:
                    increment = increment + barrier_change(s0,s1,h,budget).scale(m1)
            result = result + increment
            budget.check(result.lower)
            budget.check(result.upper)
    midpoint = (result.lower+result.upper)/2
    try:
        value = float(midpoint)
    except OverflowError as error:
        raise ValueError("Unrepresentable contact-work output") from error
    if not math.isfinite(value):
        raise ValueError("Unrepresentable contact-work output")
    error = max(abs(F(value)-result.lower),abs(result.upper-F(value)))
    return WorkResult(value,error,result.lower,result.upper,count,union_terms,budget.log_calls,budget.log_terms)
