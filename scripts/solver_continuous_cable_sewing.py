"""Bounded supplied-cell tension-only connectors; no source/solver admission.

The exact functional uses admitted binary coefficients and coordinates. Public
responses are separately bounded approximations, including final rounding.
Contact, material sides, bending and thread/seam calibration remain external.
"""
from fractions import Fraction as F
import hashlib
import json
import math

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from solver_continuous_normal_sewing import (
    ANCHORS, _anchor, _capture, _encoded, _number, _positive_float, _rat, _rational,
)
from solver_sewing_activation import _contains_bool
from solver_radial_moments import radial_moment_bounds

PROFILE = "continuous-tension-only-separation-v1"
FIELDS = frozenset(("id", *ANCHORS, "targetsMeters", "referenceLengthMeters",
                    "stiffnessDensityNPerM2", "activation"))
HALF, THREE_HALVES = F(-1, 2), F(-3, 2)


def _trim(poly):
    values = list(poly)
    while len(values) > 1 and not values[-1]:
        values.pop()
    return tuple(values) if values else (F(),)


def _add(first, second):
    return _trim(tuple((first[i] if i < len(first) else F())
                       + (second[i] if i < len(second) else F())
                       for i in range(max(len(first), len(second)))))


def _scale(poly, value):
    return _trim(tuple(value*x for x in poly))


def _mul(first, second):
    values = [F()]*(len(first)+len(second)-1)
    for i, a in enumerate(first):
        for j, b in enumerate(second):
            values[i+j] += a*b
    return _trim(values)


def _at(poly, value):
    result = F()
    for coefficient in reversed(poly):
        result = result*value+coefficient
    return result


def _compose(poly, start, width):
    result = (F(),)
    for coefficient in reversed(poly):
        result = _add(_mul(result, (start, width)), (coefficient,))
    return result


def _integral(poly):
    return sum((value/F(i+1) for i, value in enumerate(poly)), F())


def _quadratic_range(poly, lo=F(), hi=F(1)):
    values = [_at(poly, lo), _at(poly, hi)]
    if len(poly) == 3 and poly[2]:
        stationary = -poly[1]/(2*poly[2])
        if lo < stationary < hi:
            values.append(_at(poly, stationary))
    return min(values), max(values)


def _plus_interval(first, second):
    return first[0]+second[0], first[1]+second[1]


def _times_interval(interval, factor):
    a, b = (x*factor for x in interval)
    return min(a, b), max(a, b)


def _accumulate(result, key, interval):
    result[key] = _plus_interval(result.get(key, (F(), F())), interval)


def _rounded_interval(interval, tolerance):
    lo, hi = interval
    if lo > hi:
        raise ValueError("Invalid response enclosure")
    try:
        value = float((lo+hi)/2)
    except OverflowError as error:
        raise ValueError("Response midpoint is outside binary64 range") from error
    if not math.isfinite(value):
        raise ValueError("Response midpoint is outside binary64 range")
    error = max(abs(F(value)-lo), abs(hi-F(value)))
    if error > tolerance:
        raise ValueError("Requested response precision unresolved after binary64 rounding")
    return value, error


def _error_bound(error, tolerance):
    """Encode an outward binary64 ceiling, without serializing huge fractions.

    The requested tolerance is itself binary64, so a finite ceiling no larger
    than that tolerance exists. Even a positive subnormal error stays positive.
    All integration and response-rounding decisions still use exact fractions.
    """
    if error < 0 or error > tolerance:
        raise ValueError("Response error is outside the requested tolerance")
    try:
        ceiling = float(error)
    except OverflowError as failure:
        raise ValueError("Response error bound is outside binary64 range") from failure
    if not math.isfinite(ceiling):
        raise ValueError("Response error bound is outside binary64 range")
    if F(ceiling) < error:
        ceiling = math.nextafter(ceiling, math.inf)
    bound = F(ceiling)
    if not error <= bound <= tolerance:
        raise ValueError("Representable response error bound unresolved")
    return _rat(bound)


def _budget(value, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("Bounded raw integer evaluation budget required")
    return value


def _budgets(max_boundary_depth, max_boundary_panels, moment_max_terms,
             moment_max_panels, moment_max_depth):
    return {
        "max_boundary_depth": _budget(max_boundary_depth, 0, 128),
        "max_boundary_panels": _budget(max_boundary_panels, 1, 4096),
        "moment_max_terms": _budget(moment_max_terms, 1, 256),
        "moment_max_panels": _budget(moment_max_panels, 1, 4096),
        "moment_max_depth": _budget(moment_max_depth, 0, 128),
    }


class ContinuousCableSewing:
    """Immutable candidate connector with caller-selected absolute error budgets."""
    __slots__ = ("_input_bytes", "_description_bytes", "_vertex_count", "_rows",
                 "_targets", "_beta", "_activation", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Continuous cable definitions are immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError("Continuous cable definitions are immutable")

    def __init__(self, vertex_count, cells):
        if type(vertex_count) is not int or not 1 <= vertex_count <= 100000:
            raise ValueError("Bounded non-Boolean vertex count required")
        if type(cells) is not list or not 1 <= len(cells) <= 4096:
            raise ValueError("Nonempty bounded raw list of explicit cells required")
        self._input_bytes = _capture({"vertexCount": vertex_count, "cells": cells})
        cells = json.loads(self._input_bytes)["cells"]
        rows, targets, betas, activation, conversions, ids = [], [], [], [], [], set()
        for cell in cells:
            if type(cell) is not dict or set(cell) != FIELDS:
                raise ValueError("Exactly the declared continuous cable fields required")
            name = cell["id"]
            if type(name) is not str or not 1 <= len(name) <= 128 or name in ids:
                raise ValueError("Unique bounded nonempty cell identity required")
            ids.add(name)
            distances = cell["targetsMeters"]
            if type(distances) is not list or len(distances) != 2:
                raise ValueError("Two positive endpoint separation limits required")
            d0, d1 = (_number(x, 0, 100, positive=True) for x in distances)
            length = _rational(cell["referenceLengthMeters"])
            if not 0 < length <= 100:
                raise ValueError("Positive exact per-cell reference length at most100m required")
            density = _number(cell["stiffnessDensityNPerM2"], 0, 1e12, positive=True)
            alpha = _number(cell["activation"], 0, 1)
            beta = alpha*density*length
            if beta:
                _positive_float(beta)  # Reject unrepresentable positive scale; retain exact beta.
            parsed = {key: _anchor(cell[key], vertex_count) for key in ANCHORS}
            endpoint = []
            sums = []
            for suffix in ("Start", "End"):
                values = {}
                for sign, prefix in ((1, "positive"), (-1, "negative")):
                    for v, weight in parsed[prefix+suffix][1].items():
                        values[v] = values.get(v, F())+sign*F(weight)
                endpoint.append({v: w for v, w in values.items() if w})
                sums.append(_rat(sum(values.values(), F())))
            first, last = endpoint
            row = tuple((v, (first.get(v, F()), last.get(v, F())-first.get(v, F())))
                        for v in sorted(first.keys() | last.keys()))
            rows.append(row)
            targets.append((d0, d1-d0))
            betas.append(beta)
            activation.append(float(alpha))
            conversions.append({"id": name, "exactEffectiveStiffnessNPerM": _rat(beta),
                "numericalSignedCoefficientSums": sums,
                "anchorConversions": {key: parsed[key][2] for key in ANCHORS}})
        self._vertex_count = vertex_count
        self._rows, self._targets, self._beta = tuple(rows), tuple(targets), tuple(betas)
        self._activation = tuple(activation)
        self._description_bytes = _encoded({
            "profile": PROFILE, "inputSha256": hashlib.sha256(self._input_bytes).hexdigest(),
            "vertexCount": vertex_count, "cellCount": len(cells), "conversions": conversions,
            "energyLaw": "Exact beta/2 integral max(norm(C(u)q)-d(u),0)^2 du on each affine cell",
            "densityUnits": "N/m^2", "referenceMeasure": "Explicit exact per-cell material length in meters",
            "betaPolicy": "Exact product of binary64 activation/density and rational cell measure; no modal scale roots",
            "hessianPolicy": "slack-sided-generalized-curvature",
            "normalizationPolicy": "Slack regions, including coincidence, never normalize a gap vector",
            "outputPolicy": "Independent bounded approximations to one analytic functional; includes final output rounding, not exact derivatives of a single rounded energy",
            "errorBoundEncoding": "Upward-rounded binary64 ceiling of the exact error bound, expressed as an exact rational; positive errors never round to zero",
            "sharedInputHelpers": ["solver_continuous_normal_sewing.py", "solver_sewing_activation.py"],
            "sharedNumericalHelpers": ["solver_radial_moments.py"],
            "accepted": False, "sourceControlsInstalled": False, "sourceAdmissionGranted": False,
            "constructionPhaseCompleted": False,
            "limitations": [
                "A distinct uncalibrated tension-only separation-limit model, not an equivalent replacement of old point springs or a complete thread/seam model.",
                "Allows slack, coincidence, panel roll and wrong-side or intersecting states; contact, layer side, material correspondence, bending and all cloth guards remain external.",
                "Source ownership, complete material-path coverage and reference measure are supplied prerequisites; this class installs no source controls or solver integration.",
                "The energy is C1; identically taut cells select zero slack-sided generalized curvature, not a classical exact Hessian there.",
                "Reported zero outputs with nonzero error bounds are not proofs of exact zero; error budgets must be included in any caller's numerical acceptance.",
                "Response entry bounds do not certify positive semidefiniteness of the rounded sparse matrix, temporal paths, contact, solver convergence or garment acceptance.",
                "Fixed-parameter work uses unrounded endpoint energy enclosures; no target or activation work is installed.",
            ]})
        self._sealed = True

    @property
    def vertex_count(self):
        return self._vertex_count

    @property
    def cells(self):
        return json.loads(self._input_bytes)["cells"]

    @property
    def activation(self):
        return np.asarray(self._activation).copy()

    def description(self):
        return dict(json.loads(self._description_bytes), cells=self.cells)

    def _positions(self, positions):
        if _contains_bool(positions):
            raise ValueError("Boolean positions are not geometry")
        try:
            raw = np.asarray(positions)
            if raw.dtype.kind not in "fiu" or raw.dtype.kind == "f" and raw.dtype.itemsize > 8:
                raise ValueError("Real positions without extended-precision conversion required")
            result = np.array(raw, dtype=float, copy=True)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Finite matching numerical positions required") from error
        if (result.shape != (self.vertex_count, 3) or not np.isfinite(result).all()
                or np.any(np.abs(result) > 1e6)):
            raise ValueError("Matching finite positions bounded by1e6m required")
        return result

    def _geometry(self, index, positions):
        vectors = tuple(tuple(sum((row[power]*F(float(positions[v, axis]))
                                  for v, row in self._rows[index]), F())
                              for power in range(2)) for axis in range(3))
        q = (F(),)
        for vector in vectors:
            q = _add(q, _mul(vector, vector))
        return vectors, q

    def _functionals(self, index, vectors, q, derivatives):
        beta, d = self._beta[index], self._targets[index]
        # key, polynomial, Q^-1/2 numerator, Q^-3/2 numerator
        result = [(('e',), _scale(_add(q, _mul(d, d)), beta/2),
                   _scale(_mul(d, q), -beta), (F(),))]
        if not derivatives:
            return result
        dofs = [(3*v+a, row, a) for v, row in self._rows[index] for a in range(3)]
        for dof, row, axis in dofs:
            first = _scale(_mul(row, vectors[axis]), beta)
            result.append((('g', dof), first, _scale(_mul(first, d), -1), (F(),)))
        for offset, (a, left, axis) in enumerate(dofs):
            for b, right, other in dofs[offset:]:
                product = _scale(_mul(left, right), beta)
                polynomial = product if axis == other else (F(),)
                half = _scale(_mul(product, d), -1) if axis == other else (F(),)
                three = _mul(_mul(_mul(product, d), vectors[axis]), vectors[other])
                result.append((('h', a, b), polynomial, half, three))
        return result

    def _smooth(self, q, functionals, lo, hi, tolerances, budgets, stats):
        width = hi-lo
        local_q = _compose(q, lo, width)
        local_q = tuple(local_q)+(F(),)*(3-len(local_q))
        local = [(key, _compose(poly, lo, width), _compose(half, lo, width),
                  _compose(three, lo, width)) for key, poly, half, three in functionals]
        maxima = {'e': F(), 'g': F(), 'h': F()}
        powers, degree = set(), 0
        for key, _, half, three in local:
            maxima[key[0]] = max(maxima[key[0]], sum(map(abs, half), F())+sum(map(abs, three), F()))
            for power, poly in ((HALF, half), (THREE_HALVES, three)):
                if any(poly):
                    powers.add(power)
                    degree = max(degree, len(poly)-1)
        requested = [tolerances[key]/(8*value) for key, value in maxima.items() if value]
        if powers:
            moments = radial_moment_bounds(local_q, tuple(sorted(powers)), degree, min(requested),
                max_panels=budgets['moment_max_panels'], max_terms=budgets['moment_max_terms'],
                max_depth=budgets['moment_max_depth'])
            if moments.get('verified') is not True:
                raise ValueError("Verified radial moment enclosures required")
            stats['momentPanels'] += moments['panels']
            stats['maxMomentTerms'] = max(stats['maxMomentTerms'], moments['maxTerms'])
        result = {}
        for key, polynomial, half, three in local:
            value = _integral(polynomial)
            interval = value, value
            for power, poly in ((HALF, half), (THREE_HALVES, three)):
                for i, coefficient in enumerate(poly):
                    if coefficient:
                        interval = _plus_interval(interval, _times_interval(moments['moments'][power][i], coefficient))
            result[key] = _times_interval(interval, width)
        return result

    def _boundary_bounds(self, index, margin_max, lo, hi, derivatives):
        beta, d = self._beta[index], self._targets[index]
        dmin = min(_at(d, lo), _at(d, hi))
        height = margin_max/(2*dmin)
        width = hi-lo
        result = {('e',): (F(), beta*height*height*width/2)}
        if derivatives:
            dofs = [(3*v+a, max(abs(_at(row, lo)), abs(_at(row, hi))))
                    for v, row in self._rows[index] for a in range(3)]
            for a, amount in dofs:
                bound = beta*amount*height*width
                result[('g', a)] = -bound, bound
            for offset, (a, first) in enumerate(dofs):
                for b, second in dofs[offset:]:
                    bound = beta*first*second*width
                    result[('h', a, b)] = (F() if a == b else -bound), bound
        return result

    def _intervals(self, positions, tolerances, budgets, derivatives):
        total = {}
        stats = {'pendingCells': 0, 'slackLeaves': 0, 'tautLeaves': 0,
                 'boundaryLeaves': 0, 'momentPanels': 0, 'maxMomentTerms': 0,
                 'boundaryMaxDepth': 0, 'identicallyTautCells': 0, 'smoothPanels': 0}
        count = max(1, sum(bool(x) for x in self._beta))
        tolerances = {key: value/count for key, value in tolerances.items()}
        for index, beta in enumerate(self._beta):
            if not beta:
                stats['pendingCells'] += 1
                continue
            vectors, q = self._geometry(index, positions)
            margin = _add(q, _scale(_mul(self._targets[index], self._targets[index]), -1))
            if not any(margin):
                stats['identicallyTautCells'] += 1
                stats['slackLeaves'] += 1
                continue
            pending = [(F(), F(1), 0)]
            leaves, mixed_count, classified = 1, 0, []
            while pending:
                lo, hi, depth = pending.pop()
                stats['boundaryMaxDepth'] = max(stats['boundaryMaxDepth'], depth)
                lower, upper = _quadratic_range(margin, lo, hi)
                if upper <= 0:
                    stats['slackLeaves'] += 1
                    kind = 'slack'
                elif lower >= 0:
                    stats['tautLeaves'] += 1
                    kind = 'taut'
                else:
                    d = self._targets[index]
                    height = upper/(2*min(_at(d, lo), _at(d, hi)))
                    width = hi-lo
                    maximum_row = max((max(abs(_at(row, lo)), abs(_at(row, hi)))
                                       for _, row in self._rows[index]), default=F())
                    maxima = {'e': beta*height*height*width/2}
                    if derivatives:
                        maxima.update(g=beta*maximum_row*height*width,
                                      h=beta*maximum_row*maximum_row*width)
                    small = all(value <= tolerances[key]/8 for key, value in maxima.items())
                    if not small:
                        if depth >= budgets['max_boundary_depth'] or leaves >= budgets['max_boundary_panels']:
                            raise ValueError("Cable active-boundary integration budget unresolved")
                        middle = (lo+hi)/2
                        pending.extend(((middle, hi, depth+1), (lo, middle, depth+1)))
                        leaves += 1
                        continue
                    stats['boundaryLeaves'] += 1
                    mixed_count += 1
                    if mixed_count > 2:
                        raise ValueError("Quadratic cable boundary partition invariant failed")
                    kind = 'boundary'
                # Preserve exact coverage while coalescing adjacent smooth
                # regions. Otherwise root isolation creates dozens of needless
                # integrals, though it introduces at most two boundary slivers.
                if kind != 'boundary' and classified and classified[-1][0] == kind and classified[-1][2] == lo:
                    classified[-1] = kind, classified[-1][1], hi
                else:
                    classified.append((kind, lo, hi))
            functionals = None
            for kind, lo, hi in classified:
                if kind == 'slack':
                    continue
                if kind == 'taut':
                    if functionals is None:
                        functionals = self._functionals(index, vectors, q, derivatives)
                    responses = self._smooth(q, functionals, lo, hi, tolerances, budgets, stats)
                    stats['smoothPanels'] += 1
                else:
                    upper = _quadratic_range(margin, lo, hi)[1]
                    responses = self._boundary_bounds(index, upper, lo, hi, derivatives)
                for key, interval in responses.items():
                    _accumulate(total, key, interval)
        # Known sign of the exact functional/diagonal curvature can tighten
        # bounds without changing an input or choosing a different law.
        for key, (lo, hi) in tuple(total.items()):
            if key[0] == 'e' or key[0] == 'h' and key[1] == key[2]:
                if hi < 0:
                    raise ValueError("Nonnegative cable response enclosure inconsistent")
                total[key] = max(F(), lo), hi
        return total, stats

    def _certificate(self, positions, stats):
        return {'profile': 'continuous-cable-response-enclosure-v1', 'law': PROFILE,
                'inputSha256': hashlib.sha256(self._input_bytes).hexdigest(),
                'positionsSha256': hashlib.sha256(_encoded(positions.tolist())).hexdigest(),
                'hessianPolicy': 'slack-sided-generalized-curvature', 'stats': stats,
                'verified': True, 'accepted': False, 'sourceControlsInstalled': False,
                'scope': 'Entrywise bounds on supplied-state analytic cable responses, including output rounding; no physical, contact, path, solver or construction acceptance'}

    def evaluate(self, positions, *, energy_tolerance_joules, gradient_tolerance_newtons,
                 hessian_tolerance_newtons_per_meter, max_boundary_depth=80,
                 max_boundary_panels=256, moment_max_terms=128, moment_max_panels=256,
                 moment_max_depth=64):
        q = self._positions(positions)
        tolerances = {key: _number(value, 0, 1e6, positive=True) for key, value in
                      (('e', energy_tolerance_joules), ('g', gradient_tolerance_newtons),
                       ('h', hessian_tolerance_newtons_per_meter))}
        budgets = _budgets(max_boundary_depth, max_boundary_panels, moment_max_terms,
                           moment_max_panels, moment_max_depth)
        intervals, stats = self._intervals(q, tolerances, budgets, True)
        energy, energy_error = _rounded_interval(intervals.get(('e',), (F(), F())), tolerances['e'])
        gradient, hessian_values = np.zeros(3*self.vertex_count), []
        errors = {'g': F(), 'h': F()}
        for key, interval in sorted(intervals.items()):
            if key[0] == 'e':
                continue
            value, error = _rounded_interval(interval, tolerances[key[0]])
            errors[key[0]] = max(errors[key[0]], error)
            if key[0] == 'g':
                gradient[key[1]] = value
            elif value:
                hessian_values.append((key[1], key[2], value))
                if key[1] != key[2]:
                    hessian_values.append((key[2], key[1], value))
        if hessian_values:
            rows, columns, data = zip(*hessian_values)
            hessian = coo_matrix((data, (rows, columns)), shape=(3*self.vertex_count,)*2).tocsr()
        else:
            hessian = csr_matrix((3*self.vertex_count,)*2)
        certificate = self._certificate(q, stats)
        certificate.update(energyErrorBoundJoules=_error_bound(energy_error, tolerances['e']),
            gradientMaxAbsoluteErrorBoundNewtons=_error_bound(errors['g'], tolerances['g']),
            hessianMaxEntryErrorBoundNewtonsPerMeter=_error_bound(errors['h'], tolerances['h']),
            requestedTolerances={key: _rat(value) for key, value in tolerances.items()},
            budgets=budgets)
        return {'energy': energy, 'gradient': gradient, 'hessian': hessian, 'certificate': certificate}

    def energy_change(self, start, end, *, absolute_tolerance_joules,
                      max_boundary_depth=80, max_boundary_panels=256,
                      moment_max_terms=128, moment_max_panels=256, moment_max_depth=64):
        first, last = self._positions(start), self._positions(end)
        tolerance = _number(absolute_tolerance_joules, 0, 1e6, positive=True)
        budgets = _budgets(max_boundary_depth, max_boundary_panels, moment_max_terms,
                           moment_max_panels, moment_max_depth)
        invariant = all(not beta or self._geometry(i, first)[1] == self._geometry(i, last)[1]
                        for i, beta in enumerate(self._beta))
        if invariant:
            interval, statistics = (F(), F()), {'normInvariant': True}
        else:
            target = {'e': tolerance/8, 'g': tolerance/8, 'h': tolerance/8}
            before, before_stats = self._intervals(first, target, budgets, False)
            after, after_stats = self._intervals(last, target, budgets, False)
            a, b = before.get(('e',), (F(), F())), after.get(('e',), (F(), F()))
            interval = b[0]-a[1], b[1]-a[0]
            statistics = {'normInvariant': False, 'before': before_stats, 'after': after_stats}
        value, error = _rounded_interval(interval, tolerance)
        return {'changeJoules': value, 'certificate': {
            'profile': 'continuous-cable-fixed-parameter-work-enclosure-v1', 'law': PROFILE,
            'inputSha256': hashlib.sha256(self._input_bytes).hexdigest(),
            'startPositionsSha256': hashlib.sha256(_encoded(first.tolist())).hexdigest(),
            'endPositionsSha256': hashlib.sha256(_encoded(last.tolist())).hexdigest(),
            'changeErrorBoundJoules': _error_bound(error, tolerance), 'requestedToleranceJoules': _rat(tolerance),
            'stats': statistics, 'budgets': budgets, 'verified': True, 'accepted': False,
            'scope': 'Fixed-parameter difference of unrounded certified energy intervals; no motion/contact/path or parameter-work proof'}}
