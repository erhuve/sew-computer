import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class GlobalDynamicsHarnessTests(unittest.TestCase):
    def test_invalid_physical_timesteps_rejected_before_output(self):
        with tempfile.TemporaryDirectory(prefix="sew-dynamics-options-") as directory:
            for index, value in enumerate(("nan", "inf", "-inf", "0", "-1", "0.0001", "0.04")):
                with self.subTest(value=value):
                    output = Path(directory) / str(index)
                    result = subprocess.run([
                        sys.executable, str(ROOT / "scripts/spike-full-shirt.py"),
                        "--output", str(output), "--step-seconds=" + value,
                    ], cwd=ROOT, capture_output=True, text=True, timeout=15)
                    self.assertEqual(result.returncode, 2)
                    self.assertNotIn("unrecognized arguments", result.stderr)
                    self.assertFalse(output.exists())

    def run_stationary_control(self, output, substeps):
        program = """
import runpy
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from solver_global_sewing import GlobalSewingSolver
def stationary_step(self, positions, velocities, targets, dt, **kwargs):
    return positions.copy(), np.zeros_like(velocities), {
        'converged': False, 'accepted': False,
        'gradientInfinityNorm': 1, 'stationarityToleranceN': 1e-6,
    }
GlobalSewingSolver.step = stationary_step
sys.argv = [sys.argv[1] + '/spike-full-shirt.py', *sys.argv[2:]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        return subprocess.run([
            sys.executable, "-c", program, str(ROOT / "scripts"),
            "--output", str(output), "--fixture", "torso", "--embedded-sewing", "--global-reference",
            "--membrane-only-control", "--torso-equilibrium-control", "--torso-front-gap-mm", "1",
            "--disable-contact", "--steps", "2", "--ramp-steps", "1",
            "--step-seconds", "0.01", "--substeps", str(substeps),
        ], cwd=ROOT, capture_output=True, text=True, timeout=120)

    def test_schedule_accounting_source_capture_and_rejected_diagnostics(self):
        with tempfile.TemporaryDirectory(prefix="sew-dynamics-harness-") as directory:
            final_totals = []
            for substeps in (1, 2):
                with self.subTest(substeps=substeps):
                    output = Path(directory) / str(substeps)
                    result = self.run_stationary_control(output, substeps)
                    self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-2000:])
                    report = json.loads((output / "report.json").read_text())
                    trajectory = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
                    source = (ROOT / "scripts/solver_energy_balance.py").read_bytes()
                    self.assertEqual((output / "source-snapshot/solver_energy_balance.py").read_bytes(), source)
                    self.assertEqual(report['sourceDigests']['solver_energy_balance.py'], hashlib.sha256(source).hexdigest())
                    self.assertFalse(report['accepted'])
                    self.assertFalse(report['globalReferenceAllStepsConverged'])
                    self.assertEqual(len(trajectory), 2 * substeps)
                    coupling = report['coupling']
                    self.assertAlmostEqual(coupling['stepSeconds'], .01)
                    self.assertAlmostEqual(coupling['timestepSeconds'], .01 / substeps)
                    self.assertAlmostEqual(coupling['rampDurationSeconds'], .01)
                    self.assertAlmostEqual(coupling['totalDurationSeconds'], .02)
                    self.assertAlmostEqual(coupling['settlingDurationSeconds'], .01)
                    work, remainder = 0., 0.
                    previous_sewing = 0.
                    for index, entry in enumerate(trajectory):
                        self.assertAlmostEqual(entry['timeSeconds'], (index + 1) * .01 / substeps)
                        fraction = min(1., (index + 1) / substeps)
                        self.assertAlmostEqual(entry['closure'], fraction * fraction * (3 - 2 * fraction))
                        energy = entry['energyBalance']
                        self.assertFalse(energy['accepted'])
                        self.assertFalse(entry['globalSolve']['accepted'])
                        self.assertAlmostEqual(energy['sewingBeforeJoules'], previous_sewing, places=10)
                        self.assertAlmostEqual(energy['membraneChangeJoules'], 0., places=10)
                        self.assertAlmostEqual(energy['kineticChangeJoules'], 0., places=10)
                        self.assertAlmostEqual(energy['mechanicalChangeMinusTargetWorkJoules'], 0., places=10)
                        self.assertAlmostEqual(energy['targetParameterWorkJoules'], energy['sewingChangeJoules'], places=10)
                        if index >= substeps:
                            self.assertAlmostEqual(energy['targetParameterWorkJoules'], 0., places=10)
                        work += energy['targetParameterWorkJoules']
                        remainder += energy['mechanicalChangeMinusTargetWorkJoules']
                        self.assertAlmostEqual(energy['cumulativeTargetParameterWorkJoules'], work, places=10)
                        self.assertAlmostEqual(energy['cumulativeMechanicalChangeMinusTargetWorkJoules'], remainder, places=10)
                        self.assertAlmostEqual(entry['mechanicalJoules'], entry['kineticJoules']
                                               + entry['membraneJoules'] + energy['sewingAfterJoules'], places=10)
                        previous_sewing = energy['sewingAfterJoules']
                    self.assertGreater(work, 0)
                    self.assertAlmostEqual(work, previous_sewing, places=10)
                    self.assertEqual(report['globalReferenceEnergyBalance'], trajectory[-1]['energyBalance'])
                    final_totals.append(work)
            self.assertAlmostEqual(final_totals[0], final_totals[1], places=10)

    def test_energy_failure_preserves_previous_state(self):
        program = """
import runpy
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from solver_global_sewing import GlobalSewingSolver
import solver_energy_balance
def translated_step(self, positions, velocities, targets, dt, **kwargs):
    displacement = np.tile([.1, .2, .3], (len(positions), 1))
    return positions + displacement, displacement / dt, {
        'converged': True, 'accepted': False,
        'gradientInfinityNorm': 0, 'stationarityToleranceN': 1e-6,
    }
def failed_transition(*args, **kwargs):
    raise ValueError('injected energy accounting failure')
GlobalSewingSolver.step = translated_step
solver_energy_balance.global_energy_transition = failed_transition
sys.argv = [sys.argv[1] + '/spike-full-shirt.py', *sys.argv[2:]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        with tempfile.TemporaryDirectory(prefix="sew-dynamics-failure-") as directory:
            output = Path(directory) / 'failure'
            result = subprocess.run([
                sys.executable, '-c', program, str(ROOT / 'scripts'),
                '--output', str(output), '--fixture', 'torso', '--embedded-sewing', '--global-reference',
                '--membrane-only-control', '--torso-equilibrium-control', '--disable-contact',
                '--steps', '2', '--ramp-steps', '1',
            ], cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((output / 'report.json').exists(), result.stdout[-2000:] + result.stderr[-2000:])
            report = json.loads((output / 'report.json').read_text())
            self.assertFalse(report['accepted'])
            self.assertEqual(report['error'], 'global-solve-failed')
            self.assertIn('injected energy accounting failure', report['failureDetail'])
            self.assertEqual(report['failedStateMeaning'], 'previous valid state before failed global solve')
            self.assertFalse(report['globalReferenceAllStepsConverged'])
            self.assertEqual(report['globalReferenceUnconvergedSubsteps'], [1])
            geometry = json.loads((output / 'source-geometry.json').read_text())
            with np.load(output / 'failed-state.npz') as state:
                np.testing.assert_array_equal(state['positions'], np.asarray(geometry['placedMeters']))
                np.testing.assert_array_equal(state['velocities'], np.zeros_like(state['velocities']))
            trajectory = output / 'trajectory.jsonl'
            self.assertTrue(not trajectory.exists() or not trajectory.read_text().strip())


if __name__ == '__main__':
    unittest.main()
