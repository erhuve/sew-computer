import hashlib
import importlib.util
import inspect
from numbers import Integral, Real
from pathlib import Path
import sys
import textwrap
import types

import numpy as np

from solver_membrane_stability import _PINNED_KERNEL_DIGEST, _PINNED_SOLVER_DIGEST, _REPLACEMENTS


_INSTALLATIONS = {}
_PARAMETERS = '''    sewing_vertex_offsets: wp.array[int],
    sewing_vertex_rows: wp.array[int],
    sewing_vertex_coefficients: wp.array[float],
    sewing_row_offsets: wp.array[int],
    sewing_row_vertices: wp.array[int],
    sewing_row_coefficients: wp.array[float],
    sewing_targets: wp.array[wp.vec3],
    sewing_stiffness: float,
'''
_ENERGY = '''    for sewing_entry in range(sewing_vertex_offsets[particle_index], sewing_vertex_offsets[particle_index + 1]):
        sewing_row = sewing_vertex_rows[sewing_entry]
        sewing_coefficient = sewing_vertex_coefficients[sewing_entry]
        sewing_residual = -sewing_targets[sewing_row]
        for sewing_term in range(sewing_row_offsets[sewing_row], sewing_row_offsets[sewing_row + 1]):
            sewing_residual = sewing_residual + sewing_row_coefficients[sewing_term] * pos[sewing_row_vertices[sewing_term]]
        f = f - sewing_stiffness * sewing_coefficient * sewing_residual
        h = h + sewing_stiffness * sewing_coefficient * sewing_coefficient * wp.identity(n=3, dtype=float)

'''
_AUGMENTED_PARAMETERS = '''    sewing_penalties: wp.array[float],
    sewing_duals: wp.array[wp.vec3],
'''
_DUAL_UPDATE = '''
@wp.kernel
def update_sewing_duals(
    pos: wp.array[wp.vec3],
    row_offsets: wp.array[int],
    row_vertices: wp.array[int],
    row_coefficients: wp.array[float],
    targets: wp.array[wp.vec3],
    penalties: wp.array[float],
    compliance: float,
    duals: wp.array[wp.vec3],
):
    row = wp.tid()
    residual = -targets[row]
    for term in range(row_offsets[row], row_offsets[row + 1]):
        residual = residual + row_coefficients[term] * pos[row_vertices[term]]
    duals[row] = (duals[row] + penalties[row] * residual) / (1.0 + penalties[row] * compliance)
'''


def validate_rows(rows, particle_count, colors):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 32768:
        raise ValueError("Bounded nonempty sewing rows required")
    if len(colors) != particle_count:
        raise ValueError("Particle colors missing")
    result = []
    for row in rows:
        if not isinstance(row, dict) or not 2 <= len(row) <= 6:
            raise ValueError("Each sewing row requires two to six distinct vertices")
        if any(isinstance(vertex, bool) or not isinstance(vertex, Integral) or not 0 <= vertex < particle_count for vertex in row):
            raise ValueError("Invalid sewing vertex")
        if any(isinstance(weight, bool) or not isinstance(weight, Real) or not np.isfinite(weight) or not 0 < abs(weight) <= 1 for weight in row.values()):
            raise ValueError("Invalid sewing coefficient")
        if abs(sum(row.values())) > 1e-7 or abs(sum(weight for weight in row.values() if weight > 0) - 1) > 1e-7:
            raise ValueError("Sewing rows must subtract two normalized anchors")
        if len({int(colors[vertex]) for vertex in row}) != len(row):
            raise ValueError("Sewing hypergraph has same-color vertices")
        result.append(dict(sorted((int(vertex), float(weight)) for vertex, weight in row.items())))
    return result


def _targets(values, count):
    result = np.zeros((count, 3), dtype=np.float32) if values is None else np.asarray(values, dtype=np.float32)
    if result.shape != (count, 3) or not np.isfinite(result).all() or np.max(np.abs(result)) > 10:
        raise ValueError("Finite bounded sewing targets in meters required")
    return result


def _load(source, destination, prefix):
    digest = hashlib.sha256(source.encode()).hexdigest()
    name = f"newton._src.solvers.vbd._experimental_{prefix}_{hashlib.sha256(str(destination).encode()).hexdigest()[:16]}"
    if name in sys.modules:
        raise ValueError("Generated sewing module already loaded")
    with destination.open("x") as output:
        output.write(source)
    specification = importlib.util.spec_from_file_location(name, destination)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, digest


def install_embedded_sewing(solver, rows, compliance, output_directory, targets=None, augmented=False):
    import newton
    import warp as wp
    from newton._src.solvers.vbd import particle_vbd_kernels, solver_vbd

    if newton.__version__ != "1.6.0" or wp.__version__ != "1.17.0":
        raise ValueError("Embedded sewing requires Newton 1.6.0 and Warp 1.17.0")
    if type(solver) is not newton.solvers.SolverVBD or not solver.model.device.is_cpu or solver.use_particle_tile_solve:
        raise ValueError("Embedded sewing supports direct CPU VBD only")
    if id(solver) in _INSTALLATIONS or "_solve_particle_iteration" in solver.__dict__:
        raise ValueError("Solver iteration already overridden")
    if solver_vbd.solve_elasticity is not particle_vbd_kernels.solve_elasticity:
        raise ValueError("Elasticity binding changed; embedded adapter includes stable membrane itself")
    kernel_bytes = Path(particle_vbd_kernels.__file__).read_bytes()
    solver_bytes = Path(solver_vbd.__file__).read_bytes()
    if hashlib.sha256(kernel_bytes).hexdigest() != _PINNED_KERNEL_DIGEST or hashlib.sha256(solver_bytes).hexdigest() != _PINNED_SOLVER_DIGEST:
        raise ValueError("Pinned Newton sewing source changed")
    if isinstance(compliance, bool) or not isinstance(compliance, Real) or not np.isfinite(compliance) or not 1e-12 <= compliance <= 1:
        raise ValueError("Finite positive bounded compliance required")
    particle_count = solver.model.particle_count
    rows = validate_rows(rows, particle_count, solver.model.particle_colors.numpy())
    if type(augmented) is not bool:
        raise ValueError("Augmented mode must be boolean")
    effective_inverse_masses = None
    if augmented:
        masses = solver.model.particle_mass.numpy().astype(float)
        active = (solver.model.particle_flags.numpy() & 1) != 0
        effective_inverse_masses = np.asarray([sum(coefficient ** 2 / masses[vertex] for vertex, coefficient in row.items() if masses[vertex] > 0 and active[vertex]) for row in rows])
        if not np.isfinite(effective_inverse_masses).all() or np.any(effective_inverse_masses <= 0):
            raise ValueError("Augmented sewing requires a movable finite-mass anchor in each row")
    target_values = _targets(targets, len(rows))
    vertex_entries = [[] for _ in range(particle_count)]
    row_offsets, row_vertices, row_coefficients = [0], [], []
    for row_index, row in enumerate(rows):
        for vertex, coefficient in row.items():
            row_vertices.append(vertex)
            row_coefficients.append(coefficient)
            vertex_entries[vertex].append((row_index, coefficient))
        row_offsets.append(len(row_vertices))
    vertex_offsets, vertex_rows, vertex_coefficients = [0], [], []
    for entries in vertex_entries:
        for row_index, coefficient in entries:
            vertex_rows.append(row_index)
            vertex_coefficients.append(coefficient)
        vertex_offsets.append(len(vertex_rows))
    array_inputs = [wp.array(values, dtype=dtype, device="cpu") for values, dtype in (
        (vertex_offsets, int), (vertex_rows, int), (vertex_coefficients, float),
        (row_offsets, int), (row_vertices, int), (row_coefficients, float), (target_values, wp.vec3))]
    source = kernel_bytes.decode()
    for original, replacement in _REPLACEMENTS:
        if source.count(original) != 1:
            raise ValueError("Stable membrane replacement no longer unique")
        source = source.replace(original, replacement)
    start = source.index("def solve_elasticity(\n")
    prefix, kernel_source = source[:start], source[start:]
    kernel_source = kernel_source.replace("    # output\n", _PARAMETERS + (_AUGMENTED_PARAMETERS if augmented else "") + "    # output\n", 1)
    marker = "    # fmt: off\n"
    energy = _ENERGY
    if augmented:
        energy = energy.replace("        f = f - sewing_stiffness * sewing_coefficient * sewing_residual\n        h = h + sewing_stiffness * sewing_coefficient * sewing_coefficient * wp.identity(n=3, dtype=float)",
                                "        penalty = sewing_penalties[sewing_row]\n        denominator = 1.0 + penalty / sewing_stiffness\n        f = f - sewing_coefficient * (sewing_duals[sewing_row] + penalty * sewing_residual) / denominator\n        h = h + penalty / denominator * sewing_coefficient * sewing_coefficient * wp.identity(n=3, dtype=float)")
    kernel_source = kernel_source.replace(marker, energy + marker, 1)
    if augmented:
        kernel_source += _DUAL_UPDATE
    directory = Path(output_directory).resolve()
    module, kernel_digest = _load(prefix + kernel_source, directory / "generated_embedded_sewing.py", "embedded_sewing")
    method_source = textwrap.dedent(inspect.getsource(solver_vbd.SolverVBD._solve_particle_iteration))
    marker = "kernel=solve_elasticity,"
    before, after = method_source.split(marker)
    after = after.replace("            self.particle_hessians,\n", "            self.particle_hessians,\n            *self._embedded_sewing_inputs,\n", 1)
    if "*self._embedded_sewing_inputs" not in after:
        raise ValueError("Pinned elasticity launch not found")
    method_source = before + marker + after
    namespace = dict(solver_vbd.__dict__)
    namespace["solve_elasticity"] = module.solve_elasticity
    exec(compile(method_source, str(directory / "generated_embedded_iteration.py"), "exec"), namespace)
    with (directory / "generated_embedded_iteration.py").open("x") as output:
        output.write(method_source)
    penalties = wp.zeros(len(rows), dtype=float, device="cpu") if augmented else None
    duals = wp.zeros(len(rows), dtype=wp.vec3, device="cpu") if augmented else None
    solver._embedded_sewing_inputs = [*array_inputs, float(1 / compliance), *([penalties, duals] if augmented else [])]
    compiled_iteration = namespace["_solve_particle_iteration"]
    contact_binding = solver_vbd.accumulate_self_contact_force_and_hessian

    def guarded_iteration(owner, state_in, state_out, contacts, dt, iter_num):
        if solver_vbd.solve_elasticity is not particle_vbd_kernels.solve_elasticity or solver_vbd.accumulate_self_contact_force_and_hessian is not contact_binding:
            raise ValueError("Solver kernel bindings changed after embedded sewing installation")
        if augmented and iter_num == 0:
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError("Positive finite timestep required")
            penalty_values = 2 / (dt ** 2 * effective_inverse_masses)
            if not np.isfinite(penalty_values).all() or np.any(penalty_values > np.finfo(np.float32).max):
                raise ValueError("Adaptive sewing penalties exceed finite float32 range")
            penalties.assign(penalty_values.astype(np.float32))
            duals.zero_()
        result = compiled_iteration(owner, state_in, state_out, contacts, dt, iter_num)
        if augmented:
            wp.launch(module.update_sewing_duals, dim=len(rows), inputs=[state_in.particle_q, *array_inputs[3:7], penalties, float(compliance), duals], device="cpu")
        return result

    method = types.MethodType(guarded_iteration, solver)
    solver._solve_particle_iteration = method
    _INSTALLATIONS[id(solver)] = (solver, method, array_inputs[-1], namespace, duals)
    return {"profile": "experimental-augmented-embedded-sewing-v1" if augmented else "experimental-coupled-embedded-sewing-v1", "accepted": False,
            "algorithm": "auxiliary-separation-augmented-lagrangian" if augmented else "direct-penalty-coordinate-descent",
            "augmentedPenalty": "rho = 2 / (dt^2 * sum(coefficient^2 / active_mass)); dual reset each timestep; physical compliance unchanged" if augmented else None,
            "rows": len(rows), "compliance": compliance, "generatedKernelDigest": kernel_digest,
            "generatedIterationDigest": hashlib.sha256(method_source.encode()).hexdigest(),
            "sourceDigests": {"particle_vbd_kernels.py": _PINNED_KERNEL_DIGEST, "solver_vbd.py": _PINNED_SOLVER_DIGEST},
            "limitations": ["CPU research only; no garment acceptance.", "Install point contact before this adapter and retain it until uninstall.", "Stable membrane arithmetic included; separate stable membrane installation forbidden."]}


def set_embedded_targets(solver, targets):
    installation = _INSTALLATIONS.get(id(solver))
    if installation is None or installation[0] is not solver or solver._solve_particle_iteration is not installation[1]:
        raise ValueError("Solver does not own embedded sewing installation")
    installation[2].assign(_targets(targets, installation[2].size))
    if installation[4] is not None:
        installation[4].zero_()


def uninstall_embedded_sewing(solver):
    installation = _INSTALLATIONS.get(id(solver))
    if installation is None or installation[0] is not solver or solver._solve_particle_iteration is not installation[1]:
        raise ValueError("Solver does not own embedded sewing installation")
    del solver._solve_particle_iteration
    del solver._embedded_sewing_inputs
    del _INSTALLATIONS[id(solver)]
