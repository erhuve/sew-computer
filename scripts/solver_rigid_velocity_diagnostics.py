"""Bounded binary64 rigid-velocity observations; no force or acceptance law."""

import hashlib
import json
import math

import numpy as np


PROFILE = "mass-weighted-rigid-velocity-observation-v1"
MAX_VERTICES = 25000
MAX_MAGNITUDE = 1e100
MIN_INERTIA_EIGENVALUE_RATIO = 1e-12
UNIT_NORMAL_TOLERANCE = 1e-12
MIN_NORMALIZED_MASS = np.finfo(np.float64).tiny


def _number(value):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or isinstance(value, np.floating) and value.dtype.itemsize > 8):
        raise ValueError("Finite non-Boolean binary64 inputs required")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("Representable binary64 inputs required") from error
    if (not math.isfinite(result) or abs(result) > MAX_MAGNITUDE
            or isinstance(value, (int, np.integer)) and int(value) != int(result)):
        raise ValueError("Exactly representable bounded binary64 inputs required")
    return result


def _array(value, shape):
    if not shape:
        return _number(value)
    if (not isinstance(value, (list, tuple, np.ndarray))
            or isinstance(value, np.ndarray) and value.ndim == 0 or len(value) != shape[0]):
        raise ValueError("Matching bounded array shapes required")
    return [_array(item, shape[1:]) for item in value]


def _weighted_sum(weights, values):
    return np.array([math.fsum(float(w) * float(x) for w, x in zip(weights, values[:, axis]))
                     for axis in range(3)])


def _inner(weights, first, second):
    return math.fsum(float(w) * float(x) * float(y)
                     for w, a, b in zip(weights, first, second) for x, y in zip(a, b))


def _kinetic(weights, total_mass, velocities):
    scale = float(np.max(np.abs(velocities)))
    if scale == 0:
        return 0.
    unit = velocities / scale
    square = _inner(weights, unit, unit)
    energy = ((total_mass * scale) * scale) * square / 2
    if not math.isfinite(energy) or energy <= 0:
        raise ValueError("Positive kinetic energy is not representable in binary64")
    return energy


def _finite_json(value):
    if isinstance(value, dict):
        return all(_finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_json(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def analyze_rigid_velocity(positions, velocities, masses, *, normal=None):
    """Observe translation, angular velocity and residual motion of one support.

    Positive masses and noncollinear support are mandatory. Planar support is
    admitted: its three-dimensional rotational inertia is positive definite.
    The optional near-unit normal is retained unchanged; projections divide by
    its measured squared norm. Nothing is inferred about material damping.
    """
    if (not isinstance(positions, (list, tuple, np.ndarray))
            or isinstance(positions, np.ndarray) and positions.ndim == 0
            or not 3 <= len(positions) <= MAX_VERTICES):
        raise ValueError("Between 3 and 25000 material samples required")
    count = len(positions)
    q = np.array(_array(positions, (count, 3)), dtype=float)
    v = np.array(_array(velocities, (count, 3)), dtype=float)
    m = np.array(_array(masses, (count,)), dtype=float)
    if np.any(m <= 0):
        raise ValueError("Every sample requires positive physical mass")
    direction = None if normal is None else np.array(_array(normal, (3,)), dtype=float)
    if direction is not None:
        normal_length = math.hypot(*direction)
        if abs(normal_length - 1.) > UNIT_NORMAL_TOLERANCE:
            raise ValueError("An explicitly oriented unit normal is required")
    payload = {"profile": PROFILE, "positionsMeters": q.tolist(), "velocitiesMetersPerSecond": v.tolist(),
               "massesKg": m.tolist(), "normal": None if direction is None else direction.tolist()}
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            total = math.fsum(map(float, m))
            weights = m / total
            if np.any(weights < MIN_NORMALIZED_MASS):
                raise ValueError("Mass ratios are subnormal or underflow the binary64 observation")
            # A reference sample can be arbitrarily larger than the weighted
            # mean: subtracting it first can erase every small majority value.
            # Compensate the original weighted terms instead.
            center = _weighted_sum(weights, q)
            translation_velocity = _weighted_sum(weights, v)
            offsets = q - center
            position_scale = float(np.max(np.abs(offsets)))
            if position_scale == 0:
                raise ValueError("Noncollinear material support required")
            unit_offsets = offsets / position_scale
            xx, yy, zz = [math.fsum(float(w) * float(x) ** 2 for w, x in zip(weights, unit_offsets[:, axis]))
                          for axis in range(3)]
            xy, xz, yz = [math.fsum(float(w) * float(a) * float(b)
                                   for w, a, b in zip(weights, unit_offsets[:, i], unit_offsets[:, j]))
                          for i, j in ((0, 1), (0, 2), (1, 2))]
            unit_inertia = np.array([[yy + zz, -xy, -xz], [-xy, xx + zz, -yz], [-xz, -yz, xx + yy]])
            eigenvalues = np.linalg.eigvalsh(unit_inertia)
            ratio = float(eigenvalues[0] / eigenvalues[-1]) if eigenvalues[-1] > 0 else 0.
            if not math.isfinite(ratio) or ratio <= MIN_INERTIA_EIGENVALUE_RATIO:
                raise ValueError("Collinear or ill-conditioned rotational inertia")
            inertia_scale = (total * position_scale) * position_scale
            physical_eigenvalues = inertia_scale * eigenvalues
            if not inertia_scale > 0 or np.any(physical_eigenvalues <= 0):
                raise ValueError("Positive physical inertia is not representable in binary64")
            inertia = inertia_scale * unit_inertia
            centered_velocity = v - translation_velocity
            unit_momentum = _weighted_sum(weights, np.cross(unit_offsets, centered_velocity))
            scaled_omega = np.linalg.solve(unit_inertia, unit_momentum)
            omega = scaled_omega / position_scale
            if np.any((scaled_omega != 0) & (omega == 0)):
                raise ValueError("Angular velocity underflows binary64")
            rotational_velocity = np.cross(omega, offsets)
            rigid_velocity = translation_velocity + rotational_velocity
            residual = v - rigid_velocity
            translation_field = np.broadcast_to(translation_velocity, v.shape)
            kinetic = _kinetic(weights, total, v)
            translation_kinetic = _kinetic(weights, total, translation_field)
            rotation_kinetic = _kinetic(weights, total, rotational_velocity)
            residual_kinetic = _kinetic(weights, total, residual)
            angular_momentum = (total * position_scale) * unit_momentum
            checks = {
                "reconstructionMaximumAbsoluteErrorMetersPerSecond": float(np.max(np.abs(v - (rigid_velocity + residual)))),
                "residualLinearMomentumKgMetersPerSecond": (total * _weighted_sum(weights, residual)).tolist(),
                "residualAngularMomentumKgMetersSquaredPerSecond":
                    ((total * position_scale) * _weighted_sum(weights, np.cross(unit_offsets, residual))).tolist(),
                "translationRotationInnerProductJoules": total * _inner(weights, translation_field, rotational_velocity),
                "translationResidualInnerProductJoules": total * _inner(weights, translation_field, residual),
                "rotationResidualInnerProductJoules": total * _inner(weights, rotational_velocity, residual),
                "kineticPartitionResidualJoules": kinetic - math.fsum((translation_kinetic, rotation_kinetic, residual_kinetic)),
                "rotationEnergyIdentityResidualJoules": rotation_kinetic - float(omega @ (inertia @ omega)) / 2,
                "angularSolveResidualKgMetersSquaredPerSecond": (inertia @ omega - angular_momentum).tolist(),
                "normalizedWeightSumError": math.fsum(map(float, weights)) - 1.,
            }
            result = {
                "profile": PROFILE, "accepted": False, "inputSha256": hashlib.sha256(json.dumps(
                    payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
                "vertexCount": count, "totalMassKg": total,
                "centerOfMassMeters": center.tolist(), "translationVelocityMetersPerSecond": translation_velocity.tolist(),
                "linearMomentumKgMetersPerSecond": (total * translation_velocity).tolist(),
                "inertiaKgMetersSquared": inertia.tolist(), "angularVelocityRadiansPerSecond": omega.tolist(),
                "angularMomentumAboutCOMKgMetersSquaredPerSecond": angular_momentum.tolist(),
                "rigidVelocitiesMetersPerSecond": rigid_velocity.tolist(), "residualVelocitiesMetersPerSecond": residual.tolist(),
                "kineticJoules": kinetic, "translationKineticJoules": translation_kinetic,
                "rotationKineticJoules": rotation_kinetic, "residualKineticJoules": residual_kinetic,
                "residualKineticFraction": residual_kinetic / kinetic if kinetic else 0.,
                "maximumSpeedMetersPerSecond": max(math.hypot(*row) for row in v),
                "maximumResidualSpeedMetersPerSecond": max(math.hypot(*row) for row in residual),
                "conditioning": {"inertiaEigenvalueRatio": ratio,
                    "minimumAdmittedRatioExclusive": MIN_INERTIA_EIGENVALUE_RATIO,
                    "scaleNormalizedInertiaEigenvalues": eigenvalues.tolist(), "positionScaleMeters": position_scale,
                    "minimumNormalizedMass": float(weights.min()), "minimumAdmittedNormalizedMass": MIN_NORMALIZED_MASS},
                "checks": checks,
            }
            if direction is not None:
                normal_square = math.fsum(float(x) ** 2 for x in direction)
                projected = (residual @ direction)[:, None] * direction[None, :] / normal_square
                normal_kinetic = _kinetic(weights, total, projected)
                result["normalObservation"] = {"normal": direction.tolist(), "measuredLength": normal_length,
                    "unitAdmissionTolerance": UNIT_NORMAL_TOLERANCE,
                    "residualNormalKineticJoules": normal_kinetic,
                    "residualNormalKineticFraction": normal_kinetic / residual_kinetic if residual_kinetic else 0.,
                    "projection": "Projection onto the supplied admitted direction divided by its squared length; the input is not normalized or mutated"}
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        raise ValueError("Rigid-velocity observation is not representable or well-conditioned in binary64") from error
    if not _finite_json(result):
        raise ValueError("Finite binary64 observation outputs required")
    result["definition"] = {
        "translation": "V=sum(m_i*v_i)/M; r_i=q_i-massCOM",
        "rotation": "I=sum(m_i*((r_i dot r_i)*Identity-outer(r_i,r_i))); omega=solve(I,sum(m_i*cross(r_i,v_i-V)))",
        "residual": "u_i=v_i-V-cross(omega,r_i); mass-weighted least-squares rigid velocity projection",
        "kinetic": "K=M*|V|^2/2 + omega dot I omega/2 + sum(m_i*|u_i|^2)/2, with measured binary64 partition and orthogonality residuals",
    }
    result["limitations"] = [
        "Binary64 observation only; measured residuals are not exact-rational certificates or acceptance thresholds.",
        "Saved finite-step velocities can leave a finite-rotation discretization residual even for rigid motion; this instantaneous projection is not a finite-pose Kabsch fit.",
        "Residual velocity denotes motion unexplained by this support's translation and rotation; it does not identify a force, material damping, numerical error or a construction milestone.",
        "Masses and source identity are caller-supplied; this primitive does not reconstruct, calibrate or verify a captured physical model.",
        "Admission is limited to 3..25000 samples, input magnitudes at most 1e100, normal (not subnormal) normalized masses, positive representable inertia/energy, and the declared inertia conditioning bound.",
    ]
    return result
