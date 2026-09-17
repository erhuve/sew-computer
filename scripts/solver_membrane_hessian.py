import numpy as np


def _skew(vectors):
    result = np.zeros((*vectors.shape[:-1], 3, 3))
    result[..., 0, 1] = -vectors[..., 2]
    result[..., 0, 2] = vectors[..., 1]
    result[..., 1, 0] = vectors[..., 2]
    result[..., 1, 2] = -vectors[..., 0]
    result[..., 2, 0] = -vectors[..., 1]
    result[..., 2, 1] = vectors[..., 0]
    return result


def membrane_element_derivatives(positions, poses, areas, materials, project_psd=False):
    positions, poses, areas, materials = [np.asarray(value, dtype=float) for value in (positions, poses, areas, materials)]
    count = len(areas)
    if (positions.shape != (count, 3, 3) or poses.shape != (count, 2, 2)
            or materials.shape != (count, 3) or areas.shape != (count,)
            or not all(np.isfinite(value).all() for value in (positions, poses, areas, materials))
            or np.any(areas <= 0) or np.any(materials[:, :2] < 0) or np.any(materials[:, 2] != 0)):
        raise ValueError("Finite positive-area membrane inputs without damping required")
    coefficients = np.concatenate((-poses.sum(axis=1)[:, None, :], poses), axis=1)
    deformation = np.einsum("fvc,fva->fca", coefficients, positions)
    first, second = deformation[:, 0], deformation[:, 1]
    area_vectors = np.cross(first, second)
    area_ratios = np.linalg.norm(area_vectors, axis=1)
    if np.any(area_ratios <= 1e-10):
        raise ValueError("Degenerate membrane Hessian is undefined")
    normals = area_vectors / area_ratios[:, None]
    area_jacobian = np.concatenate((-_skew(second), _skew(first)), axis=2)
    gradient_area = np.einsum("fai,fa->fi", area_jacobian, normals)
    normal_projection = np.eye(3) - normals[:, :, None] * normals[:, None, :]
    hessian_area = (area_jacobian.transpose(0, 2, 1) @ normal_projection @ area_jacobian) / area_ratios[:, None, None]
    hessian_area[:, :3, 3:] -= _skew(normals)
    hessian_area[:, 3:, :3] += _skew(normals)
    shear = materials[:, 0]
    bulk = materials[:, 0] + materials[:, 1]
    alpha = 1 + shear / np.maximum(bulk, 1e-6)
    stress_area = bulk * (area_ratios - alpha)
    flat_deformation = deformation.reshape((count, 6))
    gradient = shear[:, None] * flat_deformation + stress_area[:, None] * gradient_area
    hessian = (shear[:, None, None] * np.eye(6)
               + bulk[:, None, None] * gradient_area[:, :, None] * gradient_area[:, None, :]
               + stress_area[:, None, None] * hessian_area)
    transform = np.zeros((count, 6, 9))
    for component in range(2):
        for vertex in range(3):
            transform[:, component * 3:component * 3 + 3, vertex * 3:vertex * 3 + 3] = coefficients[:, vertex, component, None, None] * np.eye(3)
    gradient = areas[:, None] * np.einsum("fai,fa->fi", transform, gradient)
    hessian = areas[:, None, None] * (transform.transpose(0, 2, 1) @ hessian @ transform)
    hessian = (hessian + hessian.transpose(0, 2, 1)) / 2
    energy = areas * (shear / 2 * (np.sum(flat_deformation ** 2, axis=1) - 2)
                      + bulk / 2 * (area_ratios - alpha) ** 2)
    if project_psd:
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        hessian = (eigenvectors * np.maximum(eigenvalues, 0)[:, None, :]) @ eigenvectors.transpose(0, 2, 1)
    return energy, gradient, hessian
