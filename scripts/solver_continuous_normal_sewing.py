"""Experimental integrated piecewise-face normal-offset penalty mechanics.

Integration of the affine numerical residual is algebraically exact in the
material cell parameter. A directed high-precision normal enclosure avoids
premature geometric rounding. This installs no source controls or runner.
"""
from fractions import Fraction as F
import hashlib
import json
import math
import re

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from solver_sewing_activation import _contains_bool

PROFILE = "continuous-piecewise-face-normal-sewing-v1"
ANCHORS = ("positiveStart", "positiveEnd", "negativeStart", "negativeEnd")
FIELDS = frozenset(("id", *ANCHORS, "frameVertices", "side", "targetsMeters",
                    "referenceLengthMeters", "stiffnessDensityNPerM2", "activation"))
LIMITATIONS = [
    "Standalone experimental supplied-cell mechanics only; no source mapping, complete seam coverage, captured input admission or construction acceptance.",
    "Per-cell reference measure and external stiffness density are explicit inputs; neither is inferred from current length, gathered members or original point compliance.",
    "Every exact anchor is retained alongside its independently rounded numerical coefficients; no normalization or rest-offset subtraction repairs their defects.",
    "The affine numerical residual is integrated algebraically in space; normal magnitudes use a directed 256-bit enclosure, not exact irrational geometry. Scale roots and final outputs remain binary64.",
    "The normal enclosure is not a certified global force, Hessian or work error bound; small admissible triangles can amplify geometric error. Unrepresentable nonzero outputs reject.",
    "The selected one-sided face director is a constitutive assumption, not a textile-side declaration or a symmetric material frame.",
    "Different adjacent signed face normals can make continuous positive-offset zero-residual sewing incompatible across a crease; no smoothing or corner policy is inferred.",
    "Pending cells contribute no mechanical terms, but their material geometry still requires the caller's complete triangle, hinge and contact guards.",
    "No contact, temporal path, solver convergence, material calibration, garment construction or fit is certified; combining this penalty with point springs adds separately chosen stiffness.",
]
_INTEGER = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)\Z", re.ASCII)
_POSITIVE = re.compile(r"[1-9][0-9]*\Z", re.ASCII)


def _encoded(value):
    try:
        result = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        raise ValueError("Finite raw JSON continuous sewing inputs required") from error
    if len(result) > 8*1024**2:
        raise ValueError("Continuous sewing input or description exceeds byte budget")
    return result


def _capture(value):
    pending, remaining = [(value, 0)], 1000000
    while pending:
        item, depth = pending.pop()
        remaining -= 1
        if remaining < 0 or depth > 12:
            raise ValueError("Continuous sewing input exceeds structure budget")
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("Raw string-keyed JSON objects required")
            pending.extend((child, depth+1) for pair in item.items() for child in pair)
        elif type(item) is list:
            if len(item) > remaining:
                raise ValueError("Continuous sewing input exceeds array budget")
            pending.extend((child, depth+1) for child in item)
        elif type(item) is str:
            if len(item) > 8*1024**2:
                raise ValueError("Continuous sewing input exceeds string budget")
        elif type(item) is int:
            if item.bit_length() > 63:
                raise ValueError("Continuous sewing raw integer exceeds bound")
        elif item is None or type(item) in (float, bool):
            pass
        else:
            raise ValueError("Raw JSON values required before immutable capture")
    return _encoded(value)


def _rat(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _rational(value):
    if type(value) is not dict or set(value) != {"numerator", "denominator"}:
        raise ValueError("Canonical rational object required")
    a, b = value["numerator"], value["denominator"]
    if (type(a) is not str or type(b) is not str or len(a) > 1235 or len(b) > 1234
            or not _INTEGER.fullmatch(a) or not _POSITIVE.fullmatch(b)):
        raise ValueError("Bounded canonical rational strings required")
    numerator, denominator = int(a), int(b)
    if (max(abs(numerator).bit_length(), denominator.bit_length()) > 4096
            or math.gcd(numerator, denominator) != 1):
        raise ValueError("Reduced rational with bounded integer complexity required")
    return F(numerator, denominator)


def _number(value, lower, upper, *, positive=False):
    if type(value) not in (int, float):
        raise ValueError("Raw non-Boolean binary64 number required")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("Finite binary64 number required") from error
    if (not math.isfinite(number) or not lower <= number <= upper or positive and number <= 0
            or type(value) is int and int(number) != value):
        raise ValueError("Finite exactly representable number in declared range required")
    return F(number)


def _positive_float(value):
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError("Positive mechanical scale is not representable") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError("Positive mechanical scale cannot overflow or underflow")
    return number


def _rounded(value):
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("Continuous sewing reduction overflow") from error
    if not math.isfinite(result) or value and result == 0:
        raise ValueError("Continuous sewing reduction is outside binary64 range")
    return result


def _anchor(raw, vertex_count):
    if type(raw) is not list or not 1 <= len(raw) <= 3:
        raise ValueError("One to three ordered positive sparse anchor terms required")
    exact, numerical, report = {}, {}, []
    for term in raw:
        if type(term) is not dict or set(term) != {"vertex", "weight"}:
            raise ValueError("Exact vertex and weight anchor fields required")
        vertex = term["vertex"]
        if (type(vertex) is not int or not 0 <= vertex < vertex_count
                or exact and vertex <= max(exact)):
            raise ValueError("Strictly ordered unique bounded vertex indices required")
        weight = _rational(term["weight"])
        if not 0 < weight <= 1:
            raise ValueError("Every stored anchor coefficient must be positive")
        number = _positive_float(weight)
        exact[vertex], numerical[vertex] = weight, number
        report.append({"vertex": vertex, "exactWeight": _rat(weight), "numericalWeight": number,
                       "numericalMinusExact": _rat(F(number)-weight)})
    if sum(exact.values(), F()) != 1:
        raise ValueError("Each exact anchor must have unit sum without normalization")
    return exact, numerical, report


def _add(values, key, value):
    values[key] = values.get(key, F())+value


def _matrix(values, shape):
    entries = [(a, b, _rounded(value)) for (a, b), value in sorted(values.items()) if value]
    if not entries:
        return csr_matrix(shape)
    rows, columns, data = zip(*entries)
    return coo_matrix((data, (rows, columns)), shape=shape).tocsr()


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F())


def _cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def _sqrt_enclosure(value):
    """Exact directed enclosure of sqrt(value) for scaled value in [1, 3]."""
    a, b = math.isqrt(value.numerator), math.isqrt(value.denominator)
    if a*a == value.numerator and b*b == value.denominator:
        exact = F(a, b)
        return exact, exact
    scale = 1 << 256
    lower_integer = math.isqrt((value.numerator*scale*scale)//value.denominator)
    lower, upper = F(lower_integer, scale), F(lower_integer+1, scale)
    if not lower*lower <= value <= upper*upper:
        raise ValueError("Directed normal square-root enclosure failed")
    return lower, upper


def _normal_differentials(first, second, normal, magnitude, with_hessian):
    """Analytic area/normal derivatives evaluated with rational normal data."""
    variations = []
    for vertex in range(3):
        for axis in range(3):
            basis = [F(int(i == axis)) for i in range(3)]
            de = [(-1 if vertex == 0 else 1 if vertex == 1 else 0)*x for x in basis]
            df = [(-1 if vertex == 0 else 1 if vertex == 2 else 0)*x for x in basis]
            variations.append((de, df))
    area_columns = [tuple(x+y for x, y in zip(_cross(de, second), _cross(first, df))) for de, df in variations]
    magnitude_gradient = [_dot(normal, column) for column in area_columns]
    jacobian = tuple(tuple((area_columns[dof][axis]-normal[axis]*magnitude_gradient[dof])/magnitude
                          for dof in range(9)) for axis in range(3))
    if not with_hessian:
        return jacobian, None
    hessians = [[[F() for _ in range(9)] for _ in range(9)] for _ in range(3)]
    for i, (dei, dfi) in enumerate(variations):
        for j, (dej, dfj) in enumerate(variations):
            second_area = tuple(a+b for a, b in zip(_cross(dei, dfj), _cross(dej, dfi)))
            second_magnitude = ((_dot(area_columns[i], area_columns[j])-magnitude_gradient[i]*magnitude_gradient[j])/magnitude
                                + _dot(normal, second_area))
            for axis in range(3):
                hessians[axis][i][j] = (second_area[axis]-normal[axis]*second_magnitude
                    - jacobian[axis][i]*magnitude_gradient[j]-jacobian[axis][j]*magnitude_gradient[i])/magnitude
    return jacobian, tuple(tuple(map(tuple, matrix)) for matrix in hessians)


class ContinuousNormalOffsetSewing:
    """Immutable-input candidate penalty; callers retain all geometry guards."""
    __slots__ = ("_input_bytes", "_vertex_count", "_faces", "_sides",
                 "_mean_scale", "_slope_scale", "_activation", "_active",
                 "_mode_rows", "_references", "_description_bytes", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Continuous sewing recipes are immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError("Continuous sewing recipes are immutable")

    def __init__(self, vertex_count, cells):
        if type(vertex_count) is not int or not 1 <= vertex_count <= 100000:
            raise ValueError("Bounded non-Boolean vertex count required")
        if type(cells) is not list or not 1 <= len(cells) <= 4096:
            raise ValueError("A nonempty bounded list of explicit spatial cells is required")
        self._input_bytes = _capture({"vertexCount": vertex_count, "cells": cells})
        cells = json.loads(self._input_bytes)["cells"]
        rows, columns, data, targets, faces, sides, active = [], [], [], [], [], [], []
        mean_scales, slope_scales, conversions, ids, activations = [], [], [], set(), []
        for cell_index, cell in enumerate(cells):
            if type(cell) is not dict or set(cell) != FIELDS:
                raise ValueError("Each continuous sewing cell requires exactly its declared fields")
            name = cell["id"]
            if type(name) is not str or not 1 <= len(name) <= 128 or name in ids:
                raise ValueError("Unique bounded nonempty cell IDs required")
            ids.add(name)
            frame, side = cell["frameVertices"], cell["side"]
            if (type(frame) is not list or len(frame) != 3 or any(type(v) is not int or not 0 <= v < vertex_count for v in frame)
                    or len(set(frame)) != 3 or type(side) is not int or side not in (-1, 1)):
                raise ValueError("Explicit oriented triangle and integer mechanical sign required")
            if type(cell["targetsMeters"]) is not list or len(cell["targetsMeters"]) != 2:
                raise ValueError("Two explicit endpoint distances required")
            distances = [_number(x, 0, 100, positive=True) for x in cell["targetsMeters"]]
            measure = _rational(cell["referenceLengthMeters"])
            if not 0 < measure <= 100:
                raise ValueError("Positive per-cell reference length at most 100 meters required")
            density = _number(cell["stiffnessDensityNPerM2"], 0, 1e12, positive=True)
            activation = _number(cell["activation"], 0, 1)
            parsed = {key: _anchor(cell[key], vertex_count) for key in ANCHORS}
            if (any(not set(parsed[key][0]) <= set(frame) for key in ("negativeStart", "negativeEnd"))
                    or any(set(parsed[key][0]) & set(frame) for key in ("positiveStart", "positiveEnd"))):
                raise ValueError("Negative endpoint supports must fit the frame; positive supports must be disjoint")
            sums = []
            for endpoint, suffix in enumerate(("Start", "End")):
                signed = {**parsed["positive"+suffix][1],
                          **{v: -w for v, w in parsed["negative"+suffix][1].items()}}
                sums.append(_rat(sum((F(w) for w in signed.values()), F())))
                for vertex, weight in sorted(signed.items()):
                    rows.append(2*cell_index+endpoint)
                    columns.append(vertex)
                    data.append(weight)
            gamma = activation*density*measure
            numerical_gamma = _positive_float(gamma) if gamma else 0.
            numerical_slope = _positive_float(gamma/12) if gamma else 0.
            mean_scale, slope_scale = math.sqrt(numerical_gamma), math.sqrt(numerical_slope)
            mean_scales.append(mean_scale)
            slope_scales.append(slope_scale)
            targets.extend(float(x) for x in distances)
            faces.extend([list(frame), list(frame)])
            sides.extend([side, side])
            active.extend([int(activation > 0)]*2)
            activations.append(float(activation))
            conversions.append({"id": name, "anchorConversions": {key: parsed[key][2] for key in ANCHORS},
                "numericalSignedCoefficientSums": sums,
                "exactEffectiveStiffnessNPerM": _rat(gamma), "numericalEffectiveStiffnessNPerM": numerical_gamma,
                "effectiveStiffnessConversionResidualNPerM": _rat(F(numerical_gamma)-gamma),
                "exactSlopeWeightNPerM": _rat(gamma/12), "numericalSlopeWeightNPerM": numerical_slope,
                "slopeWeightConversionResidualNPerM": _rat(F(numerical_slope)-gamma/12),
                "meanResidualScale": mean_scale, "slopeResidualScale": slope_scale,
                "exactSquareOfMeanResidualScale": _rat(F(mean_scale)**2),
                "exactSquareOfSlopeResidualScale": _rat(F(slope_scale)**2)})
        self._vertex_count = vertex_count
        self._faces, self._sides = tuple(map(tuple, faces)), tuple(sides)
        self._mean_scale, self._slope_scale = tuple(mean_scales), tuple(slope_scales)
        self._activation = tuple(activations)
        self._active = tuple(i for i, value in enumerate(activations) if value > 0)
        converted = [{} for _ in targets]
        for row, column, value in zip(rows, columns, data):
            converted[row][column] = F(value)
        modes, references = [], []
        for i in range(len(cells)):
            first, last = converted[2*i:2*i+2]
            vertices = sorted(first.keys() | last.keys())
            modes.append(tuple(tuple((v, amount) for v in vertices if
                               (amount := ((first.get(v, F())+last.get(v, F()))/2 if mode == 0 else
                                           last.get(v, F())-first.get(v, F())))) for mode in (0, 1)))
            references.append(((F(targets[2*i])+F(targets[2*i+1]))/2, F(targets[2*i+1])-F(targets[2*i]),
                               F(mean_scales[i])**2, F(slope_scales[i])**2))
        self._mode_rows, self._references = tuple(modes), tuple(references)
        description = {"profile": PROFILE, "inputSha256": hashlib.sha256(self._input_bytes).hexdigest(),
            "vertexCount": vertex_count, "cellCount": len(cells), "conversions": conversions,
            "referenceMeasure": "Explicit per-cell material reference length in meters; no additional fraction factor",
            "densityUnits": "N/m^2", "energyUnits": "J", "gradientUnits": "N",
            "numericalPolicy": "Independent nearest-binary64 endpoint coefficients; one rounding of exact alpha*kappa*L and exact alpha*kappa*L/12, then binary64 scale roots. Exact binary-input area, anchor/modal and assembly contractions with one-round outputs; no coefficient normalization. Normal magnitude uses the midpoint of a directed 256-bit rational square-root enclosure, retained rationally through analytic derivatives.",
            "derivativePolicy": "Gradient and full Hessian expand the analytic unit-normal energy before evaluation, eliminating d^2 director cancellation. They are not promised bitwise equal to separately rounded J-transpose-r or Gauss-Newton-plus-curvature arithmetic.",
            "workPolicy": "Exact binary displacement contractions and stable differences of independently scaled area vectors with norms in [1,sqrt(3)]; positive proportional area vectors preserve unchanged normals. No certified global arithmetic error bound.",
            "normalSquareRootFractionBits": 256,
            "sharedMechanicalHelpers": [],
            "sharedInputHelpers": ["solver_sewing_activation.py"],
            "activeFrameRelativeDoubleAreaThreshold": 1e-12,
            "accepted": False, "sourceControlsInstalled": False, "sourceAdmissionGranted": False,
            "constructionPhaseCompleted": False, "limitations": list(LIMITATIONS)}
        self._description_bytes = _encoded(description)
        self._sealed = True

    @property
    def vertex_count(self):
        return self._vertex_count

    @property
    def activation(self):
        return np.asarray(self._activation).copy()

    @property
    def cells(self):
        return json.loads(self._input_bytes)["cells"]

    def description(self):
        return dict(json.loads(self._description_bytes), cells=self.cells)

    def _positions(self, positions):
        if _contains_bool(positions):
            raise ValueError("Boolean positions are not mechanical geometry")
        try:
            raw = np.asarray(positions)
            if raw.dtype.kind not in "fiu" or raw.dtype.kind == "f" and raw.dtype.itemsize > 8:
                raise ValueError("Real positions without extended-precision conversion required")
            result = np.asarray(raw, dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Finite matching numerical positions required") from error
        if (result.shape != (self.vertex_count, 3) or not np.isfinite(result).all()
                or np.any(np.abs(result) > 1e6)):
            raise ValueError("Matching finite positions bounded by 1e6 meters required")
        return result

    def _geometry(self, positions, derivatives=False, with_hessian=False):
        result, cache = {}, {}
        for i in self._active:
            frame = self._faces[2*i]
            if frame not in cache:
                points = [[F(float(positions[v, axis])) for axis in range(3)] for v in frame]
                first, second = (tuple(x-y for x, y in zip(points[j], points[0])) for j in (1, 2))
                area = _cross(first, second)
                area_squared = _dot(area, area)
                edge_scale_squared = max(_dot(first, first), _dot(second, second))
                if edge_scale_squared <= 0 or area_squared <= F(1e-12)**2*edge_scale_squared**2:
                    raise ValueError("Degenerate active material frame")
                scale = max(map(abs, area))
                scaled = tuple(value/scale for value in area)
                lower, upper = _sqrt_enclosure(_dot(scaled, scaled))
                scaled_magnitude = (lower+upper)/2
                magnitude = scale*scaled_magnitude
                n = tuple(value/scaled_magnitude for value in scaled)
                if derivatives:
                    jacobian, hessians = _normal_differentials(first, second, n, magnitude, with_hessian)
                else:
                    jacobian = hessians = None
                cache[frame] = (n, magnitude, jacobian, hessians, (scale*lower, scale*upper), area_squared)
            result[i] = cache[frame]
        return result

    def normal_diagnostics(self, positions):
        """Inspect directed magnitude enclosures; this is not a force error proof."""
        geometry = self._geometry(self._positions(positions))
        frames, cells = [], self.cells
        for i in self._active:
            normal, _, _, _, bounds, squared = geometry[i]
            frames.append({"cellId": cells[i]["id"], "frameVertices": list(self._faces[2*i]),
                "doubleAreaSquaredMeters4": _rat(squared),
                "doubleAreaMagnitudeBoundsMeters2": [_rat(value) for value in bounds],
                "normalMagnitudeSquaredMinusOne": _rat(_dot(normal, normal)-1),
                "enclosureVerified": bounds[0]**2 <= squared <= bounds[1]**2})
        return {"profile": "continuous-normal-geometric-enclosures-v1", "frames": frames,
                "accepted": False, "globalForceErrorBoundVerified": False}

    def _point(self, row, positions):
        return tuple(sum((weight*F(float(positions[v, axis])) for v, weight in row), F()) for axis in range(3))

    def _errors(self, i, positions, normal):
        side = self._sides[2*i]
        return tuple(tuple(x-side*d*n for x, n in zip(self._point(row, positions), normal))
                     for row, d in zip(self._mode_rows[i], self._references[i][:2]))

    def residual(self, positions):
        q = self._positions(positions)
        geometry = self._geometry(q)
        result = np.zeros(6*len(self._activation))
        for i in self._active:
            for mode, error in enumerate(self._errors(i, q, geometry[i][0])):
                scale = F((self._mean_scale, self._slope_scale)[mode][i])
                for axis, value in enumerate(error):
                    result[6*i+3*mode+axis] = _rounded(scale*value)
        return result

    def jacobian(self, positions):
        q = self._positions(positions)
        geometry, entries = self._geometry(q, derivatives=True), {}
        for i in self._active:
            normal_jacobian = geometry[i][2]
            frame = self._faces[2*i]
            for mode, (row, target) in enumerate(zip(self._mode_rows[i], self._references[i][:2])):
                scale = F((self._mean_scale, self._slope_scale)[mode][i])
                for axis in range(3):
                    index = 6*i+3*mode+axis
                    for vertex, coefficient in row:
                        _add(entries, (index, 3*vertex+axis), scale*coefficient)
                    for dof in range(9):
                        column = 3*frame[dof//3]+dof%3
                        _add(entries, (index, column), -scale*self._sides[2*i]*target*normal_jacobian[axis][dof])
        return _matrix(entries, (6*len(self._activation), 3*self.vertex_count))

    def energy(self, positions):
        q = self._positions(positions)
        geometry, result = self._geometry(q), F()
        for i in self._active:
            for weight, error in zip(self._references[i][2:], self._errors(i, q, geometry[i][0])):
                result += weight*_dot(error, error)/2
        return _rounded(result)

    def _expanded_cell(self, i, positions):
        """Exact contractions of admitted numerical maps, targets and scales."""
        mean, delta = self._mode_rows[i]
        dm, dd, wm, wd = self._references[i]
        points = self._point(mean, positions), self._point(delta, positions)
        b = {}
        for row, distance, weight in ((mean, dm, wm), (delta, dd, wd)):
            for vertex, coefficient in row:
                _add(b, vertex, weight*distance*coefficient)
        z = tuple(wm*dm*x+wd*dd*y for x, y in zip(*points))
        return points, b, z

    def gradient(self, positions):
        q = self._positions(positions)
        geometry, contributions = self._geometry(q, derivatives=True), {}
        for i in self._active:
            normal, _, normal_jacobian, _ = geometry[i][:4]
            frame, side = self._faces[2*i], self._sides[2*i]
            points, b, z = self._expanded_cell(i, q)
            for row, point, weight in zip(self._mode_rows[i], points, self._references[i][2:]):
                for vertex, coefficient in row:
                    for axis in range(3):
                        _add(contributions, 3*vertex+axis, weight*coefficient*point[axis])
            for vertex, coefficient in b.items():
                for axis in range(3):
                    _add(contributions, 3*vertex+axis, -side*coefficient*normal[axis])
            for dof in range(9):
                _add(contributions, 3*frame[dof//3]+dof%3,
                     -side*sum((z[axis]*normal_jacobian[axis][dof] for axis in range(3)), F()))
        result = np.zeros(3*self.vertex_count)
        for dof, value in contributions.items():
            result[dof] = _rounded(value)
        return result

    def hessian(self, positions, project_psd=False):
        """Gauss--Newton metric; exact products of the returned numerical J."""
        jacobian = self.jacobian(positions)
        entries = {}
        for row in range(jacobian.shape[0]):
            start, stop = jacobian.indptr[row:row+2]
            terms = [(int(v), F(float(x))) for v, x in zip(jacobian.indices[start:stop], jacobian.data[start:stop])]
            for a, x in terms:
                for b, y in terms:
                    _add(entries, (a, b), x*y)
        return _matrix(entries, (3*self.vertex_count,)*2)

    def exact_hessian(self, positions):
        """Analytic full unit-normal curvature, without cancelling d^2 terms."""
        q = self._positions(positions)
        geometry, entries = self._geometry(q, derivatives=True, with_hessian=True), {}
        for i in self._active:
            _, _, normal_jacobian, normal_hessians = geometry[i][:4]
            frame, side = self._faces[2*i], self._sides[2*i]
            _, b, z = self._expanded_cell(i, q)
            for row, weight in zip(self._mode_rows[i], self._references[i][2:]):
                for first, a in row:
                    for second, c in row:
                        for axis in range(3):
                            _add(entries, (3*first+axis, 3*second+axis), weight*a*c)
            dofs = [3*v+axis for v in frame for axis in range(3)]
            for vertex, coefficient in b.items():
                for axis in range(3):
                    for local, dof in enumerate(dofs):
                        value = -side*coefficient*normal_jacobian[axis][local]
                        _add(entries, (3*vertex+axis, dof), value)
                        _add(entries, (dof, 3*vertex+axis), value)
            for a, first in enumerate(dofs):
                for b_index, second in enumerate(dofs):
                    value = -side*sum((z[axis]*normal_hessians[axis][a][b_index] for axis in range(3)), F())
                    _add(entries, (first, second), value)
        return _matrix(entries, (3*self.vertex_count,)*2)

    def _normal_increment(self, frame, start, end, before, after):
        first = [[F(float(start[v, axis])) for axis in range(3)] for v in frame]
        last = [[F(float(end[v, axis])) for axis in range(3)] for v in frame]
        e0, f0 = ([p-q for p, q in zip(first[i], first[0])] for i in (1, 2))
        e1, f1 = ([p-q for p, q in zip(last[i], last[0])] for i in (1, 2))
        a0, a1 = _cross(e0, f0), _cross(e1, f1)
        if _cross(a0, a1) == [0, 0, 0] and _dot(a0, a1) > 0:
            return (F(),)*3
        scale0, scale1 = max(map(abs, a0)), max(map(abs, a1))
        u0, u1 = (tuple(value/scale for value in area) for area, scale in ((a0, scale0), (a1, scale1)))
        delta = tuple(b-a for a, b in zip(u0, u1))
        rho0, rho1 = before[1]/scale0, after[1]/scale1
        change = (2*_dot(u0, delta)+_dot(delta, delta))/(rho0+rho1)
        return tuple((d-n*change)/rho1 for d, n in zip(delta, before[0]))

    def energy_change(self, start, end):
        """Stable fixed-parameter work; no subtraction of rounded energies."""
        first, last = self._positions(start), self._positions(end)
        before, after = self._geometry(first), self._geometry(last)
        result = F()
        for i in self._active:
            normal_delta = self._normal_increment(self._faces[2*i], first, last, before[i], after[i])
            errors = self._errors(i, first, before[i][0])
            for mode, (row, target, weight) in enumerate(zip(self._mode_rows[i], self._references[i][:2], self._references[i][2:])):
                displacement = tuple(sum((coefficient*(F(float(last[v, axis]))-F(float(first[v, axis])))
                                          for v, coefficient in row), F()) for axis in range(3))
                delta = tuple(value-self._sides[2*i]*target*n for value, n in zip(displacement, normal_delta))
                result += weight*sum(((a+b/2)*b for a, b in zip(errors[mode], delta)), F())
        return _rounded(result)
