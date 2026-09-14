import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "services/engine"
DEFAULT_SOURCE = Path("/home/workspace/Code/design2garmentcode-impl")


def main():
    parser = argparse.ArgumentParser(description="Verify pinned CPU-only engine; optionally install an isolated runtime. No source, body or model downloads.")
    parser.add_argument("--source", type=Path, default=Path(os.environ.get("SEW_ENGINE_SOURCE", str(DEFAULT_SOURCE))))
    parser.add_argument("--install", action="store_true", help="Install locked packages into services/engine/.venv (pip needs network).")
    args = parser.parse_args()
    source = args.source.resolve()
    python = ENGINE / ".venv/bin/python"
    if args.install:
        subprocess.run([sys.executable, "-m", "venv", str(ENGINE / ".venv")], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "--only-binary=:all:", "-r", str(ENGINE / "requirements.lock")], check=True)
    if not python.is_file():
        python = Path(os.environ.get("SEW_ENGINE_PYTHON", str(source / ".venv/bin/python")))
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MPLBACKEND": "Agg"}
    code = "import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from guard import verify_runtime; verify_runtime(Path(sys.argv[2])); print('Source and locked CPU dependencies verified')"
    subprocess.run([str(python), "-I", "-B", "-c", code, str(ENGINE), str(source)], check=True, env=environment)
    subprocess.run([str(python), "-I", "-B", str(ENGINE / "worker.py"), "--probe"], check=True, env=environment)
    commit = json.loads((ENGINE / "source-lock.json").read_text())["commit"]
    print(f"CPU source: {source} ({commit})\nPython: {python}")
    print("Run: bun test services/engine/runner.test.ts")
    print("Private single-owner trusted code only; no filesystem/network-namespace sandbox; not public/multiuser execution.")


if __name__ == "__main__":
    main()
