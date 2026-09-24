"""Bounded rational enclosures for quadratic radial moments on [0, 1].

This is a numerical integration kernel, not a connector or geometry admission
policy. Every certificate decision uses integers and Fraction arithmetic.
"""

from fractions import Fraction as F
import math


PROFILE = 'quadratic-radial-moment-bounds-v1'
_POWERS = frozenset((F(-3, 2), F(-1), F(-1, 2), F(1, 2)))
_INPUT_BITS = 4096
_ARITHMETIC_BITS = 262144
_OPERATIONS = 4000000
_ROOT_BITS = 16384


class _Arithmetic:
    """Conservative, explicit arithmetic budget; exhaustion never certifies."""
    def __init__(self):
        self.operations = 0

    def tick(self):
        self.operations += 1
        if self.operations > _OPERATIONS:
            raise ValueError('Radial moment arithmetic operation budget exhausted')

    def add(self, a, b):
        self.tick()
        common = math.gcd(a.denominator, b.denominator)
        x, y = b.denominator // common, a.denominator // common
        if max(a.numerator.bit_length()+x.bit_length(), b.numerator.bit_length()+y.bit_length())+1 > _ARITHMETIC_BITS:
            raise ValueError('Radial moment numerator complexity budget exhausted')
        if a.denominator.bit_length()+x.bit_length() > _ARITHMETIC_BITS:
            raise ValueError('Radial moment denominator complexity budget exhausted')
        return F(a.numerator*x+b.numerator*y, a.denominator*x)

    def mul(self, a, b):
        self.tick()
        c, d = math.gcd(a.numerator, b.denominator), math.gcd(b.numerator, a.denominator)
        an, bn, ad, bd = a.numerator//c, b.numerator//d, a.denominator//d, b.denominator//c
        if max(an.bit_length()+bn.bit_length(), ad.bit_length()+bd.bit_length()) > _ARITHMETIC_BITS:
            raise ValueError('Radial moment product complexity budget exhausted')
        return F(an*bn, ad*bd)

    def div(self, a, b):
        if not b:
            raise ValueError('Radial moment zero divisor')
        return self.mul(a, F(b.denominator, b.numerator))


def _fraction(value, name):
    if type(value) is not F or max(value.numerator.bit_length(), value.denominator.bit_length()) > _INPUT_BITS:
        raise ValueError(f'{name} requires a bounded exact Fraction')
    return value


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} requires an integer in [{low}, {high}]')


def _evaluate(q, u, arithmetic):
    return arithmetic.add(q[0], arithmetic.mul(u, arithmetic.add(q[1], arithmetic.mul(q[2], u))))


def _range(q, a, b, arithmetic):
    candidates = [(a, _evaluate(q, a, arithmetic)), (b, _evaluate(q, b, arithmetic))]
    critical = arithmetic.div(-q[1], 2*q[2]) if q[2] else None
    if critical is not None and a < critical < b:
        candidates.append((critical, _evaluate(q, critical, arithmetic)))
    low = min(candidates, key=lambda pair: pair[1])
    high = max(candidates, key=lambda pair: pair[1])
    return low[1], high[1], low[0], critical


def _sqrt_bounds(value, bits):
    """Directed root bounds with relative scaling, including extreme exponents."""
    numerator, denominator = value.numerator, value.denominator
    a, b = math.isqrt(numerator), math.isqrt(denominator)
    if a*a == numerator and b*b == denominator:
        result = F(a, b)
        return result, result
    exponent = numerator.bit_length()-denominator.bit_length()
    below_power = (numerator < denominator << exponent) if exponent >= 0 else (numerator << -exponent < denominator)
    if below_power:
        exponent -= 1
    half_exponent = exponent//2
    if half_exponent >= 0:
        denominator <<= 2*half_exponent
        multiplier = F(1 << half_exponent)
    else:
        numerator <<= -2*half_exponent
        multiplier = F(1, 1 << -half_exponent)
    if numerator.bit_length()+2*bits > _ARITHMETIC_BITS:
        raise ValueError('Radial root complexity budget exhausted')
    scale = 1 << bits
    integer = math.isqrt((numerator << (2*bits))//denominator)
    lower, upper = multiplier*F(integer, scale), multiplier*F(integer+1, scale)
    if not 0 < lower <= upper or not lower*lower <= value <= upper*upper:
        raise ValueError('Directed radial square-root enclosure failed')
    return lower, upper


def _scales(s, powers, tolerance, arithmetic):
    bits = 32
    while True:
        root_low, root_high = _sqrt_bounds(s, bits)
        bounds = {}
        for power in powers:
            if power == -1:
                value = arithmetic.div(F(1), s)
                bounds[power] = value, value
            elif power == F(1, 2):
                bounds[power] = root_low, root_high
            else:
                extra = s if power == F(-3, 2) else F(1)
                bounds[power] = (arithmetic.div(F(1), arithmetic.mul(extra, root_high)),
                                 arithmetic.div(F(1), arithmetic.mul(extra, root_low)))
        if all(arithmetic.add(upper, -lower) <= tolerance/8 for lower, upper in bounds.values()):
            return bounds, bits
        if bits == _ROOT_BITS:
            raise ValueError('Radial normalization precision budget exhausted')
        bits = min(2*bits, _ROOT_BITS)


def _multiply_quadratic(poly, quadratic, arithmetic):
    result = [F() for _ in range(len(poly)+2)]
    for i, coefficient in enumerate(poly):
        for j, value in enumerate(quadratic):
            if coefficient and value:
                result[i+j] = arithmetic.add(result[i+j], arithmetic.mul(coefficient, value))
    while len(result) > 1 and not result[-1]:
        result.pop()
    return tuple(result)


def _panel(q, a, b, minimum, maximum, powers, degree, tolerance, max_terms, arithmetic):
    width = b-a
    s = (minimum+maximum)/2
    rho = (maximum-minimum)/(maximum+minimum)
    assert 0 <= rho <= F(1, 4)
    scales, root_bits = _scales(s, powers, tolerance, arithmetic)
    # Q(a + width*t)/s - 1, with t in [0, 1]. Polynomial powers
    # and weighted monomial integrals are shared by every requested exponent.
    local = (arithmetic.add(arithmetic.div(_evaluate(q, a, arithmetic), s), -F(1)),
             arithmetic.div(arithmetic.mul(width, arithmetic.add(q[1], arithmetic.mul(2*q[2], a))), s),
             arithmetic.div(arithmetic.mul(q[2], arithmetic.mul(width, width)), s))
    integrated_weights = []
    for k in range(degree+1):
        integrated_weights.append(tuple(width*F(math.comb(k, j))*a**(k-j)*width**j for j in range(k+1)))
    monomial_cache = []
    def moments(poly):
        while len(monomial_cache) < len(poly):
            n = len(monomial_cache)
            entry = []
            for weights in integrated_weights:
                value = F()
                for j, weight in enumerate(weights):
                    value = arithmetic.add(value, arithmetic.div(weight, F(n+j+1)))
                entry.append(value)
            monomial_cache.append(tuple(entry))
        result = [F() for _ in range(degree+1)]
        for n, coefficient in enumerate(poly):
            if coefficient:
                for k in range(degree+1):
                    result[k] = arithmetic.add(result[k], arithmetic.mul(coefficient, monomial_cache[n][k]))
        return result
    measure = tuple((b**(k+1)-a**(k+1))/F(k+1) for k in range(degree+1))
    partial = {p: [F() for _ in range(degree+1)] for p in powers}
    coefficient = {p: F(1) for p in powers}
    poly, rho_power = (F(1),), F(1)
    for n in range(max_terms):
        integral = moments(poly)
        rho_power = arithmetic.mul(rho_power, rho)
        result = {}
        for p in powers:
            next_coefficient = arithmetic.mul(coefficient[p], (p-n)/F(n+1))
            # For every admitted p and all subsequent binomial terms,
            # |c_(j+1)/c_j| <= 3/2. Thus the remaining pointwise tail is
            # bounded by its first absolute term times this geometric sum.
            tail = arithmetic.div(arithmetic.mul(abs(next_coefficient), rho_power), 1-F(3, 2)*rho)
            lower_scale, upper_scale = scales[p]
            bounds = []
            for k in range(degree+1):
                partial[p][k] = arithmetic.add(partial[p][k], arithmetic.mul(coefficient[p], integral[k]))
                # u^k is nonnegative on this domain, so multiplying the
                # pointwise remainder by its exact integral is outward.
                error = arithmetic.mul(tail, measure[k])
                lo, hi = arithmetic.add(partial[p][k], -error), arithmetic.add(partial[p][k], error)
                products = [arithmetic.mul(x, y) for x in (lo, hi) for y in (lower_scale, upper_scale)]
                bounds.append((max(F(), min(products)), max(products)))
            result[p] = tuple(bounds)
            coefficient[p] = next_coefficient
        if all(hi-lo <= tolerance*width for values in result.values() for lo, hi in values):
            return result, n+1, root_bits
        if n+1 < max_terms:
            poly = _multiply_quadratic(poly, local, arithmetic)
    raise ValueError('Radial binomial term budget exhausted without requested enclosure')


def radial_moment_bounds(q, powers, max_degree, absolute_tolerance, *, max_panels=256, max_terms=128, max_depth=64):
    """Enclose integral_0^1 u^k Q(u)^p du with width <= absolute_tolerance.

    q is the ascending quadratic power basis. All coefficients, exponents and
    the tolerance must be Fraction instances. No midpoint is a certified value:
    callers must retain the returned intervals through their own contractions.
    Budgets are finite and failure raises ValueError; no tolerance is relaxed.
    """
    if type(q) is not tuple or len(q) != 3:
        raise ValueError('q requires three ascending exact Fraction coefficients')
    q = tuple(_fraction(x, 'q coefficient') for x in q)
    if type(powers) is not tuple or not 1 <= len(powers) <= 4:
        raise ValueError('powers requires a unique nonempty tuple')
    if any(type(p) is not F or p not in _POWERS for p in powers):
        raise ValueError('Unsupported radial power')
    if len(set(powers)) != len(powers):
        raise ValueError('powers requires a unique nonempty tuple')
    _integer(max_degree, 0, 5, 'max_degree')
    _integer(max_panels, 1, 4096, 'max_panels')
    _integer(max_terms, 1, 256, 'max_terms')
    _integer(max_depth, 0, 128, 'max_depth')
    tolerance = _fraction(absolute_tolerance, 'absolute_tolerance')
    if tolerance <= 0:
        raise ValueError('absolute_tolerance must be positive')
    arithmetic = _Arithmetic()
    minimum, maximum, witness, _ = _range(q, F(), F(1), arithmetic)
    if minimum <= 0:
        raise ValueError('Quadratic radial norm must be strictly positive on the closed interval')
    pending, panels = [(F(), F(1), 0)], []
    while pending:
        a, b, depth = pending.pop()
        low, high, _, critical = _range(q, a, b, arithmetic)
        if 4*(high-low) <= high+low:
            panels.append((a, b, low, high))
            continue
        if depth == max_depth:
            raise ValueError('Radial panel depth budget exhausted')
        if len(panels)+len(pending)+2 > max_panels:
            raise ValueError('Radial panel count budget exhausted')
        split = critical if critical is not None and a < critical < b else (a+b)/2
        pending.extend(((split, b, depth+1), (a, split, depth+1)))
    total = {p: [(F(), F()) for _ in range(max_degree+1)] for p in powers}
    used_terms = used_root_bits = 0
    for a, b, low, high in panels:
        local, terms, bits = _panel(q, a, b, low, high, powers, max_degree, tolerance, max_terms, arithmetic)
        used_terms, used_root_bits = max(used_terms, terms), max(used_root_bits, bits)
        for p in powers:
            total[p] = [(arithmetic.add(x, lo), arithmetic.add(y, hi)) for (x, y), (lo, hi) in zip(total[p], local[p])]
    if any(hi-lo > tolerance for values in total.values() for lo, hi in values):
        raise ValueError('Accumulated radial moment enclosure exceeds requested tolerance')
    return {'profile': PROFILE, 'moments': {p: tuple(values) for p, values in total.items()},
            'panels': len(panels), 'maxTerms': used_terms, 'maxRootBits': used_root_bits,
            'arithmeticOperations': arithmetic.operations, 'verified': True,
            'minimumQ': minimum, 'minimumQAt': witness, 'maximumQ': maximum,
            'panelRanges': tuple(panels), 'absoluteTolerance': tolerance,
            'errorConvention': 'Every exact integral is enclosed; each interval width is at most absoluteTolerance.'}
