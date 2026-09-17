import hashlib
import importlib.util
from pathlib import Path
import sys


_ACTIVE_INSTALLATION = None
_PINNED_KERNEL_DIGEST = "7bd167d0b81e6bbb51c73a65dbfea7a86652788fed5388ed85266281450d838d"
_PINNED_SOLVER_DIGEST = "2eaf20bde0cfa436bdaad375b689f5e562130667801ceb3e6098343f4418bc13"
_REPLACEMENTS = (
    (
        "    J_s_sq = f0_dot_f0 * f1_dot_f1 - f0_dot_f1 * f0_dot_f1",
        "    area_vector = wp.cross(f0, f1)\n    J_s_sq = wp.dot(area_vector, area_vector)",
    ),
    (
        "    g0 = inv_J_s * (f1_dot_f1 * f0 - f0_dot_f1 * f1)\n    g1 = inv_J_s * (f0_dot_f0 * f1 - f0_dot_f1 * f0)",
        "    g0 = inv_J_s * wp.cross(f1, area_vector)\n    g1 = inv_J_s * wp.cross(area_vector, f0)",
    ),
    (
        "    I_coeff = mu_nh * (df0_dx_sq + df1_dx_sq) + r * (\n        df0_dx_sq * f1_dot_f1 + df1_dx_sq * f0_dot_f0 - 2.0 * df0_dx * df1_dx * f0_dot_f1\n    )",
        "    I_coeff = mu_nh * (df0_dx_sq + df1_dx_sq) + r * wp.dot(w, w)",
    ),
)


def generate_stable_membrane_module(output_directory):
    import newton
    import warp as wp
    from newton._src.solvers.vbd import particle_vbd_kernels, solver_vbd

    if newton.__version__ != "1.6.0" or wp.__version__ != "1.17.0":
        raise ValueError("Stable membrane experiment requires Newton 1.6.0 and Warp 1.17.0")
    kernel_path = Path(particle_vbd_kernels.__file__)
    solver_path = Path(solver_vbd.__file__)
    kernel_bytes, solver_bytes = kernel_path.read_bytes(), solver_path.read_bytes()
    if hashlib.sha256(kernel_bytes).hexdigest() != _PINNED_KERNEL_DIGEST or hashlib.sha256(solver_bytes).hexdigest() != _PINNED_SOLVER_DIGEST:
        raise ValueError("Pinned Newton membrane or solver source changed")
    source = kernel_bytes.decode("utf-8")
    for original, corrected in _REPLACEMENTS:
        if source.count(original) != 1:
            raise ValueError("Pinned membrane expression does not match uniquely")
        source = source.replace(original, corrected, 1)
    generated_bytes = source.encode("utf-8")
    generated_digest = hashlib.sha256(generated_bytes).hexdigest()
    destination = Path(output_directory).resolve() / "generated_stable_membrane.py"
    with destination.open("xb") as generated:
        generated.write(generated_bytes)
    suffix = hashlib.sha256(str(destination).encode() + generated_bytes).hexdigest()[:24]
    module_name = f"newton._src.solvers.vbd._experimental_stable_membrane_{suffix}"
    if module_name in sys.modules:
        raise ValueError("Generated membrane module already loaded")
    specification = importlib.util.spec_from_file_location(module_name, destination)
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module, {
        "profile": "experimental-cross-product-membrane-v1",
        "accepted": False,
        "generatedKernelDigest": generated_digest,
        "sourceDigests": {
            Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            kernel_path.name: _PINNED_KERNEL_DIGEST,
            solver_path.name: _PINNED_SOLVER_DIGEST,
        },
        "versions": {"newton": newton.__version__, "warp-lang": wp.__version__},
        "regularization": "Original squared area floor 1e-20 retained",
        "limitations": ["CPU research process only; all solver execution in this process must use the installation owner.", "Equivalent arithmetic correction does not establish cloth stability or garment acceptance."],
    }


def install_stable_membrane(solver, output_directory):
    global _ACTIVE_INSTALLATION
    import newton
    from newton._src.solvers.vbd import particle_vbd_kernels, solver_vbd

    if _ACTIVE_INSTALLATION is not None:
        raise ValueError("Only one stable membrane solver may be installed per process")
    if type(solver) is not newton.solvers.SolverVBD or not solver.model.device.is_cpu or solver.use_particle_tile_solve:
        raise ValueError("Stable membrane experiment supports only the direct CPU VBD solver")
    if solver_vbd.solve_elasticity is not particle_vbd_kernels.solve_elasticity:
        raise ValueError("Newton elasticity binding is already modified")
    module, report = generate_stable_membrane_module(output_directory)
    original = solver_vbd.solve_elasticity
    solver_vbd.solve_elasticity = module.solve_elasticity
    _ACTIVE_INSTALLATION = (solver, original, module)
    return report


def uninstall_stable_membrane(solver):
    global _ACTIVE_INSTALLATION
    from newton._src.solvers.vbd import solver_vbd

    if _ACTIVE_INSTALLATION is None or _ACTIVE_INSTALLATION[0] is not solver:
        raise ValueError("Solver does not own the stable membrane installation")
    _, original, module = _ACTIVE_INSTALLATION
    if solver_vbd.solve_elasticity is not module.solve_elasticity:
        raise ValueError("Installed elasticity binding changed unexpectedly")
    solver_vbd.solve_elasticity = original
    _ACTIVE_INSTALLATION = None
