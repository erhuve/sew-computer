from math import comb

import numpy as np


def _product(first, second, operation):
    first_degree, second_degree = first.shape[1] - 1, second.shape[1] - 1
    degree = first_degree + second_degree
    sample = operation(first[:, 0], second[:, 0])
    result = np.zeros((len(first), degree + 1, *sample.shape[1:]))
    for first_index in range(first_degree + 1):
        for second_index in range(second_degree + 1):
            weight = (comb(first_degree, first_index) * comb(second_degree, second_index)
                      / comb(degree, first_index + second_index))
            result[:, first_index + second_index] += weight * operation(
                first[:, first_index], second[:, second_index])
    return result


def _dot(first, second):
    return np.sum(first * second, axis=-1)


def _split(coefficients):
    levels = [coefficients]
    while levels[-1].shape[1] > 1:
        levels.append((levels[-1][:, :-1] + levels[-1][:, 1:]) * 0.5)
    return (np.stack([level[:, 0] for level in levels], axis=1),
            np.stack([level[:, -1] for level in reversed(levels)], axis=1))


def _nonzero(coefficients):
    degree = coefficients.shape[1] - 1
    midpoint = sum(comb(degree, index) * coefficients[:, index]
                   for index in range(degree + 1)) / 2 ** degree
    length = np.linalg.norm(midpoint, axis=1)
    direction = midpoint / np.maximum(length[:, None], np.finfo(float).tiny)
    return np.all(np.einsum("hki,hi->hk", coefficients, direction) > 1e-12, axis=1)


def hinge_sweep_safe(start, end, indices, *, max_depth=12, max_intervals=65536):
    """Conservatively certify affine hinge paths using bounded Bernstein subdivision.

    Numerical margins are used, not directed-rounding interval arithmetic. False
    includes unresolved paths; True does not certify nonadjacent cloth contact.
    """
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    raw_indices = np.asarray(indices)
    if (start.ndim != 2 or start.shape[1] != 3 or end.shape != start.shape
            or not np.isfinite(start).all() or not np.isfinite(end).all()
            or raw_indices.ndim != 2 or raw_indices.shape[1] != 4
            or raw_indices.dtype.kind not in "iu"
            or np.any(raw_indices < 0) or np.any(raw_indices >= len(start))
            or any(len(set(row)) != 4 for row in raw_indices)
            or isinstance(max_depth, bool) or not isinstance(max_depth, (int, np.integer))
            or not 0 <= max_depth <= 24
            or isinstance(max_intervals, bool) or not isinstance(max_intervals, (int, np.integer))
            or max_intervals < 1):
        raise ValueError("Finite vertex arrays, distinct integer hinges and bounded sweep budgets required")
    if not len(raw_indices):
        return True
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        points = np.stack((start[raw_indices], end[raw_indices]), axis=1)
        points = points - points[:, :, 2:3]
        scale = np.max(np.abs(points), axis=(1, 2, 3))
        if not np.isfinite(points).all() or np.any(scale <= 0):
            return False
        points = points / scale[:, None, None, None]
        edge = points[:, :, 3]
        first_normal = _product(points[:, :, 2] - points[:, :, 0],
                                points[:, :, 3] - points[:, :, 0], np.cross)
        second_normal = _product(points[:, :, 3] - points[:, :, 1],
                                 points[:, :, 2] - points[:, :, 1], np.cross)
        cosine = _product(first_normal, second_normal, _dot)
        sine = _product(_product(first_normal, second_normal, np.cross), edge, _dot)
        pending = [(0, (edge, first_normal, second_normal, cosine, sine))]
        visited = 0
        while pending:
            depth, values = pending.pop()
            visited += len(values[0])
            if visited > max_intervals:
                return False
            edge, first_normal, second_normal, cosine, sine = values
            normal_bound = (np.max(np.linalg.norm(first_normal, axis=2), axis=1)
                            * np.max(np.linalg.norm(second_normal, axis=2), axis=1))
            sine_margin = 1e-12 * normal_bound * np.max(np.linalg.norm(edge, axis=2), axis=1)
            branch_safe = (np.all(cosine > 1e-12 * normal_bound[:, None], axis=1)
                           | np.all(sine > sine_margin[:, None], axis=1)
                           | np.all(sine < -sine_margin[:, None], axis=1))
            safe = _nonzero(edge) & _nonzero(first_normal) & _nonzero(second_normal) & branch_safe
            if np.all(safe):
                continue
            if depth == max_depth:
                return False
            halves = [_split(value[~safe]) for value in values]
            pending.append((depth + 1, tuple(half[1] for half in halves)))
            pending.append((depth + 1, tuple(half[0] for half in halves)))
    return True
