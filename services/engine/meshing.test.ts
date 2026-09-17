import { expect, test } from 'bun:test';
import { join } from 'node:path';

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
