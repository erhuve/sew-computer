"""Select one captured or repository engine namespace without cached mixing.

Flat numerical snapshots keep ``services/engine`` beside their solver helpers.
Repository-shaped snapshots and the checkout keep it beside ``scripts``. A
present incomplete tree is an error, never a reason to import a live fallback.
"""

import importlib
from pathlib import Path
import sys


ENGINE_FILES = ("shirt.py", "assembly.py", "cloth_domain.py", "meshing.py",
                "quality_meshing.py", "simulation_validation.py",
                "embedded_constraints.py", "inspection_gltf.py")


def _check_cached(root):
    for filename in ENGINE_FILES:
        name = Path(filename).stem
        if name not in sys.modules:
            continue
        module = sys.modules[name]
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str) or Path(origin).resolve() != root / filename:
            raise ValueError(f"Cached engine module {name} belongs to a different source root")


def engine_source_root(caller_file):
    """Return and select the complete engine tree belonging to this helper."""
    parent = Path(caller_file).resolve().parent
    adjacent_services = parent / "services"
    # A dangling link reports exists() == False. Reject links before either
    # selecting the fallback or resolving paths, which would hide the link.
    if adjacent_services.is_symlink():
        raise ValueError("Linked engine source directories are not supported")
    # Test existence of the services container too: an incomplete adjacent
    # snapshot must not fall through to another tree one level above it.
    services = adjacent_services if adjacent_services.exists() else parent.parent / "services"
    candidate = services / "engine"
    if services.is_symlink() or candidate.is_symlink():
        raise ValueError("Linked engine source directories are not supported")
    root = candidate.resolve()
    if not root.is_dir() or any(not (root / name).is_file()
                               or (root / name).resolve().parent != root for name in ENGINE_FILES):
        raise ValueError("Complete captured or repository engine source tree required")
    _check_cached(root)
    location = str(root)
    if location in sys.path:
        sys.path.remove(location)
    sys.path.insert(0, location)
    importlib.invalidate_caches()
    return root


def load_engine_modules(caller_file, *names):
    """Import only declared engine modules and recheck their resolved origins."""
    if any(name + ".py" not in ENGINE_FILES for name in names):
        raise ValueError("Undeclared engine source dependency")
    root = engine_source_root(caller_file)
    modules = tuple(importlib.import_module(name) for name in names)
    _check_cached(root)
    return modules
