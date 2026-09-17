import numpy as np
from scipy.sparse import diags


def shifted_positive_definite_direction(matrix, gradient, inertia_diagonal):
    from solver_global_sewing import _positive_definite_direction

    matrix = matrix.tocsc()
    gradient = np.asarray(gradient, dtype=float)
    inertia_diagonal = np.asarray(inertia_diagonal, dtype=float)
    count = matrix.shape[0]
    if (matrix.shape != (count, count) or count == 0
            or gradient.shape != (count,) or inertia_diagonal.shape != (count,)
            or not all(np.isfinite(value).all() for value in
                       (matrix.data, gradient, inertia_diagonal))
            or np.any(inertia_diagonal <= 0)):
        raise ValueError("Finite square metric, gradient and positive inertia diagonal required")
    magnitude = max(1., float(np.max(np.abs(matrix.data), initial=0)))
    asymmetry = matrix - matrix.T
    if np.max(np.abs(asymmetry.data), initial=0) > 1e-12 * magnitude:
        raise ValueError("Symmetric metric required")
    inertia = diags(inertia_diagonal, format="csc")
    attempts = []
    for shift in (0., *(10. ** exponent for exponent in range(-3, 8))):
        candidate = matrix if shift == 0 else matrix + shift * inertia
        direction = _positive_definite_direction(candidate, gradient)
        attempts.append({"lambda": shift, "accepted": direction is not None})
        if direction is not None:
            return direction, {"lambda": shift, "attempts": attempts, "fallbackRequired": False}
    return None, {"lambda": None, "attempts": attempts, "fallbackRequired": True}
