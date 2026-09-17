import numpy as np


def membrane_energy_change(deformation, delta_deformation, areas, materials):
    deformation, delta_deformation, areas, materials = [np.asarray(value, dtype=float) for value in
                                                       (deformation, delta_deformation, areas, materials)]
    count = areas.size
    if (deformation.shape != (count, 2, 3) or delta_deformation.shape != deformation.shape
            or areas.shape != (count,) or materials.shape != (count, 3)
            or not all(np.isfinite(value).all() for value in
                       (deformation, delta_deformation, areas, materials))
            or np.any(areas <= 0) or np.any(materials[:, :2] < 0) or np.any(materials[:, 2] != 0)):
        raise ValueError("Finite membrane deformations, positive areas and nonnegative undamped materials required")
    first, second = deformation[:, 0], deformation[:, 1]
    delta_first, delta_second = delta_deformation[:, 0], delta_deformation[:, 1]
    cross = np.cross(first, second)
    delta_cross = (np.cross(delta_first, second) + np.cross(first, delta_second)
                   + np.cross(delta_first, delta_second))
    previous_area = np.linalg.norm(cross, axis=1)
    following_area = np.linalg.norm(cross + delta_cross, axis=1)
    if (not np.isfinite(previous_area).all() or not np.isfinite(following_area).all()
            or np.any(previous_area <= 1e-10) or np.any(following_area <= 1e-10)):
        raise ValueError("Nondegenerate finite membrane endpoints required")
    delta_area = np.sum((2 * cross + delta_cross) * delta_cross, axis=1) / (previous_area + following_area)
    shear = materials[:, 0]
    bulk = shear + materials[:, 1]
    alpha = 1 + shear / np.maximum(bulk, 1e-6)
    change = areas * (shear * np.sum((deformation + .5 * delta_deformation) * delta_deformation, axis=(1, 2))
                      + bulk * ((previous_area - alpha) * delta_area + .5 * delta_area ** 2))
    result = float(np.sum(change))
    if not np.isfinite(result):
        raise ValueError("Finite membrane energy change required")
    return result
