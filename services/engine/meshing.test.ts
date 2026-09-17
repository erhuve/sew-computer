import { expect, test } from 'bun:test';
import { join } from 'node:path';

test('research shirt registration preserves collar junctions and sleeve endpoint connectivity', async () => {
  const python = process.env.SEW_ENGINE_PYTHON || join(import.meta.dir, '.venv/bin/python');
  const child = Bun.spawn([python, join(import.meta.dir, '../../scripts/test_shirt_registration_topology.py')], { stdout: 'pipe', stderr: 'pipe', env: { PATH: '/usr/bin:/bin', OPENBLAS_NUM_THREADS: '1' } });
  const [output, errors, status] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
  expect({ status, diagnostics: status === 0 ? '' : output + errors }).toEqual({ status: 0, diagnostics: '' });
}, 120000);

test('sewing constraint colors preserve cloth separation and reject invalid particle partitions', async () => {
  const python = process.env.SEW_ENGINE_PYTHON || join(import.meta.dir, '.venv/bin/python');
  const child = Bun.spawn([python, join(import.meta.dir, 'constraint_coloring_test.py')], { stdout: 'pipe', stderr: 'pipe', env: { PATH: '/usr/bin:/bin', OPENBLAS_NUM_THREADS: '1' } });
  const [output, errors, status] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
  expect({ status, diagnostics: status === 0 ? '' : output + errors }).toEqual({ status: 0, diagnostics: '' });
}, 120000);

test('source-preserving meshes reject topology and correspondence mutations across six shirt sizes', async () => {
  const python = process.env.SEW_ENGINE_PYTHON || join(import.meta.dir, '.venv/bin/python');
  const child = Bun.spawn([python, join(import.meta.dir, 'meshing_test.py')], { stdout: 'pipe', stderr: 'pipe', env: { PATH: '/usr/bin:/bin', OPENBLAS_NUM_THREADS: '1' } });
  const [output, errors, status] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
  expect({ status, diagnostics: status === 0 ? '' : output + errors }).toEqual({ status: 0, diagnostics: '' });
}, 120000);

test('cut-line cloth preserves allowance fabric and rejects interior stitch correspondence mutations', async () => {
  const python = process.env.SEW_ENGINE_PYTHON || join(import.meta.dir, '.venv/bin/python');
  const child = Bun.spawn([python, join(import.meta.dir, 'cloth_domain_test.py')], { stdout: 'pipe', stderr: 'pipe', env: { PATH: '/usr/bin:/bin', OPENBLAS_NUM_THREADS: '1' } });
  const [output, errors, status] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
  expect({ status, diagnostics: status === 0 ? '' : output + errors }).toEqual({ status: 0, diagnostics: '' });
}, 120000);

test('experimental physical placement preserves mirrored source lengths with proper rigid transforms', async () => {
  const python = process.env.SEW_ENGINE_PYTHON || join(import.meta.dir, '.venv/bin/python');
  const child = Bun.spawn([python, join(import.meta.dir, 'placement_test.py')], { stdout: 'pipe', stderr: 'pipe', env: { PATH: '/usr/bin:/bin', OPENBLAS_NUM_THREADS: '1' } });
  const [output, errors, status] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
  expect({ status, diagnostics: status === 0 ? '' : output + errors }).toEqual({ status: 0, diagnostics: '' });
}, 120000);

test('embedded interior sewing preserves source arcs, pinned vertices and mass through staged closure', async () => {
  const python = process.env.SEW_ENGINE_PYTHON || join(import.meta.dir, '.venv/bin/python');
  const child = Bun.spawn([python, join(import.meta.dir, 'embedded_constraints_test.py')], { stdout: 'pipe', stderr: 'pipe', env: { PATH: '/usr/bin:/bin', OPENBLAS_NUM_THREADS: '1' } });
  const [output, errors, status] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
  expect({ status, diagnostics: status === 0 ? '' : output + errors }).toEqual({ status: 0, diagnostics: '' });
}, 120000);

test('opposed sewing probes retain source dimensions and proper rigid orientation', async () => {
  const python = process.env.SEW_ENGINE_PYTHON || join(import.meta.dir, '.venv/bin/python');
  const child = Bun.spawn([python, join(import.meta.dir, 'probe_placement_test.py')], { stdout: 'pipe', stderr: 'pipe', env: { PATH: '/usr/bin:/bin', OPENBLAS_NUM_THREADS: '1' } });
  const [output, errors, status] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
  expect({ status, diagnostics: status === 0 ? '' : output + errors }).toEqual({ status: 0, diagnostics: '' });
}, 120000);

test('quality refinement avoids conflicting Steiner clusters while preserving source panels', async () => {
  const python = process.env.SEW_ENGINE_PYTHON || join(import.meta.dir, '.venv/bin/python');
  const child = Bun.spawn([python, join(import.meta.dir, 'quality_meshing_test.py')], { stdout: 'pipe', stderr: 'pipe', env: { PATH: '/usr/bin:/bin', OPENBLAS_NUM_THREADS: '1' } });
  const [output, errors, status] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
  expect({ status, diagnostics: status === 0 ? '' : output + errors }).toEqual({ status: 0, diagnostics: '' });
}, 120000);
