import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class GlobalHarnessAdversarialTests(unittest.TestCase):
    def run_injected_harness(self, output, behavior):
        program = """
import runpy
import os
import signal
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from solver_global_sewing import GlobalSewingSolver
behavior = sys.argv[2]
calls = 0
def injected_step(self, positions, velocities, targets, dt, **kwargs):
    global calls
    calls += 1
    if behavior == 'exception':
        raise FloatingPointError('injected global solve failure')
    if behavior == 'cpu':
        os.kill(os.getpid(), signal.SIGXCPU)
    return positions.copy(), np.zeros_like(velocities), {
        'converged': calls > 1, 'accepted': False,
        'gradientInfinityNorm': 0 if calls > 1 else 1,
        'stationarityToleranceN': 1e-6,
    }
GlobalSewingSolver.step = injected_step
sys.argv = [sys.argv[1] + '/spike-full-shirt.py', *sys.argv[3:]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        return subprocess.run([
            sys.executable, "-c", program, str(ROOT / "scripts"), behavior,
            "--output", str(output), "--fixture", "torso", "--embedded-sewing", "--global-reference",
            "--membrane-only-control", "--torso-equilibrium-control", "--disable-contact",
            "--steps", "2", "--ramp-steps", "1",
        ], cwd=ROOT, capture_output=True, text=True, timeout=120)

    def test_global_exception_preserves_rejected_state_and_report(self):
        with tempfile.TemporaryDirectory(prefix="sew-global-failure-") as directory:
            output = Path(directory) / "failure"
            result = self.run_injected_harness(output, "exception")
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((output / "report.json").exists(), result.stdout[-2000:] + result.stderr[-2000:])
            report = json.loads((output / "report.json").read_text())
            self.assertFalse(report["accepted"])
            self.assertEqual(report["error"], "global-solve-failed")
            self.assertIn("injected global solve failure", report["failureDetail"])
            self.assertEqual(report["failedStep"], 1)
            failed = output / "failed-state.npz"
            self.assertEqual(hashlib.sha256(failed.read_bytes()).hexdigest(), report["failedStateSha256"])
            geometry = json.loads((output / "source-geometry.json").read_text())
            with np.load(failed) as state:
                np.testing.assert_allclose(state["positions"], geometry["placedMeters"], atol=1e-7)
                self.assertTrue(np.isfinite(state["positions"]).all())

    def test_earlier_unconverged_step_is_not_hidden_by_final_step(self):
        with tempfile.TemporaryDirectory(prefix="sew-global-summary-") as directory:
            output = Path(directory) / "summary"
            result = self.run_injected_harness(output, "convergence")
            self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-2000:])
            report = json.loads((output / "report.json").read_text())
            self.assertTrue(report["globalReferenceFinalStep"]["converged"])
            self.assertFalse(report["globalReferenceAllStepsConverged"])
            self.assertEqual(report["globalReferenceUnconvergedSubsteps"], [1])
            self.assertFalse(report["accepted"])

    def test_cpu_signal_during_solve_preserves_previous_state(self):
        with tempfile.TemporaryDirectory(prefix="sew-global-cpu-") as directory:
            output = Path(directory) / "cpu"
            result = self.run_injected_harness(output, "cpu")
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((output / "report.json").exists(), result.stdout[-2000:] + result.stderr[-2000:])
            report = json.loads((output / "report.json").read_text())
            self.assertFalse(report["accepted"])
            self.assertEqual(report["error"], "global-solve-failed")
            self.assertIn("CPU time limit", report["failureDetail"])
            self.assertEqual(report["failedStateMeaning"], "previous valid state before failed global solve")
            self.assertFalse(report["globalReferenceAllStepsConverged"])
            self.assertEqual(report["globalReferenceUnconvergedSubsteps"], [1])
            geometry = json.loads((output / "source-geometry.json").read_text())
            with np.load(output / "failed-state.npz") as state:
                np.testing.assert_allclose(state["positions"], geometry["placedMeters"], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
