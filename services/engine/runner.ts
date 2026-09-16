import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { access, chmod, chown, lstat, mkdir, mkdtemp, open, realpath, rm } from 'node:fs/promises';
import { constants } from 'node:fs';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { canonical, DocumentSchema as documentSchema, type Artifact, type GarmentDocument, type PatternGeometry } from '../../packages/contracts/index';
import { sizingInput } from '../../packages/contracts/sizing';

export const ENGINE_COMMIT = '7065b3ef01ff61f4462e871d4cb439a0b97c48db';
const root = dirname(fileURLToPath(import.meta.url));
const defaultUpstream = '/home/workspace/Code/design2garmentcode-impl';
const maxFile = 16 * 1024 * 1024;
export class EngineInputError extends Error {}
const files = [
  { filename: 'pattern.json', mime: 'application/json', kind: 'pattern-json' as const },
  { filename: 'pattern.svg', mime: 'image/svg+xml', kind: 'pattern-svg' as const },
  { filename: 'pattern.pdf', mime: 'application/pdf', kind: 'pattern-pdf' as const },
];

export function validateInput(document: GarmentDocument, inputDigest: string) {
  const doc = documentSchema.parse(document);
  if (!/^[a-f0-9]{64}$/.test(inputDigest) || createHash('sha256').update(canonical(doc)).digest('hex') !== inputDigest) throw new Error('Input digest must match the immutable document');
  let dimensions: ReturnType<typeof sizingInput>;
  try { dimensions = sizingInput(doc); } catch (error) { throw new EngineInputError((error as Error).message); }
  const provenance = [
    ...(doc.interpretation?[`AI parameter proposal: ${doc.interpretation.provider} / ${doc.interpretation.model} / ${doc.interpretation.adapter}; proposal ${doc.interpretation.proposalId}. Accepted by owner; subsequent manual edits possible. Not a fit or sewing validation.`]:[]),
    'The brief, references, construction text and requirement statuses are preserved but are not interpreted by this manual CPU adapter. Only family, length, ease, flare and entered body values are used.',
    ...Object.entries(doc.body).map(([key, m]) => `Owner-entered body.${key}: ${m.state}; source: ${'source' in m ? m.source : 'unspecified'}.`),
    ...(['length', 'ease'] as const).map(key => `Owner-entered garment.${key}: ${doc.garment[key].state}; source: ${'source' in doc.garment[key] ? doc.garment[key].source : 'unspecified'}.`),
    ...doc.requirements.map(r => `Requirement ${r.id} remains ${r.status}, unverified by engine: ${r.text}`),
  ];
  const input = { ...dimensions, provenance, inputDigest };
  if (Buffer.byteLength(JSON.stringify(input)) > 512 * 1024) throw new Error('Engine input exceeds budget');
  return input;
}

export function cleanEnvironment(cacheDir = '.'): NodeJS.ProcessEnv {
  return { PATH: '/usr/bin:/bin', LANG: 'C.UTF-8', LC_ALL: 'C.UTF-8', HOME: cacheDir, TMPDIR: cacheDir, MPLCONFIGDIR: join(cacheDir, '.mpl'), MPLBACKEND: 'Agg', OPENBLAS_NUM_THREADS: '1', OMP_NUM_THREADS: '1', MKL_NUM_THREADS: '1', NUMEXPR_NUM_THREADS: '1' };
}

export async function executeTrusted(python: string, args: string[], cwd: string, payload: string, signal: AbortSignal, timeoutMs = 90_000, cacheDir = cwd): Promise<void> {
  signal.throwIfAborted();
  await new Promise<void>((resolveJob, reject) => {
    const child = spawn(python, ['-I', '-B', ...args], { cwd, env: cleanEnvironment(cacheDir), detached: true, stdio: ['pipe', 'pipe', 'pipe'] });
    let failure: Error | undefined;
    let outputSize = 0;
    let stderr = '';
    const kill = () => {
      if (!child.pid) return;
      try { process.kill(-child.pid, 'SIGKILL'); } catch { child.kill('SIGKILL'); }
    };
    const abort = () => { failure = new Error('Engine cancelled'); kill(); };
    const timer = setTimeout(() => { failure = new Error('Engine exceeded wall timeout'); kill(); }, timeoutMs);
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) abort();
    const consume = (chunk: Buffer, isError = false) => {
      outputSize += chunk.byteLength;
      if (isError && stderr.length < 4096) stderr += chunk.toString().slice(0, 4096 - stderr.length);
      if (outputSize > 65536) { failure = new Error('Engine diagnostic budget exceeded'); kill(); }
    };
    child.stdout.on('data', chunk => consume(chunk));
    child.stderr.on('data', chunk => consume(chunk, true));
    child.stdin.on('error', () => {});
    child.once('error', error => { failure = error; });
    child.once('close', (code, exitSignal) => {
      clearTimeout(timer);
      signal.removeEventListener('abort', abort);
      kill();
      if (failure) reject(failure);
      else if (code !== 0) reject(new Error(`Engine failed (${exitSignal || code}): ${stderr.trim().slice(-1000) || 'no diagnostic'}`));
      else resolveJob();
    });
    child.stdin.end(payload);
  });
}

async function readArtifact(path: string): Promise<Uint8Array> {
  const handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    const stat = await handle.stat();
    if (!stat.isFile() || stat.nlink !== 1 || stat.size < 100 || stat.size > maxFile) throw new Error('Invalid engine artifact file');
    const bytes = await handle.readFile();
    if (bytes.length !== stat.size) throw new Error('Engine artifact changed while reading');
    return bytes;
  } finally { await handle.close(); }
}

export function validateGeometry(value: unknown, inputDigest: string): PatternGeometry {
  const geometry = value as PatternGeometry;
  if (!geometry || geometry.schemaVersion !== 1 || geometry.units !== 'mm' || typeof geometry.engineVersion !== 'string' || !geometry.engineVersion.includes(ENGINE_COMMIT) || geometry.inputDigest !== inputDigest || geometry.classification !== 'printable-reference' || !['shirt', 'skirt', 'trousers'].includes(geometry.family)) throw new Error('Invalid engine provenance');
  if (!Array.isArray(geometry.panels) || geometry.panels.length < 1 || geometry.panels.length > 32 || !Array.isArray(geometry.stitches) || !Array.isArray(geometry.warnings) || !geometry.warnings.every(w => typeof w === 'string') || !Array.isArray(geometry.assumptions) || !geometry.assumptions.every(w => typeof w === 'string')) throw new Error('Invalid engine geometry');
  const ids = new Set<string>();
  for (const panel of geometry.panels) {
    if (!/^[a-zA-Z0-9_-]{1,100}$/.test(panel.id) || ids.has(panel.id)) throw new Error('Invalid panel identity');
    ids.add(panel.id);
    if (!Array.isArray(panel.points) || panel.points.length < 4 || panel.points.length > 20001 || typeof panel.name !== 'string') throw new Error('Invalid panel points');
    if (![panel.widthMm, panel.heightMm].every(n => Number.isFinite(n) && n > 1 && n <= 5000)) throw new Error('Invalid panel dimensions');
    for (const point of panel.points) {
      if (!Array.isArray(point) || point.length !== 2 || !point.every(Number.isFinite) || point[0] < -0.001 || point[1] < -0.001 || point[0] > panel.widthMm + 0.001 || point[1] > panel.heightMm + 0.001) throw new Error('Invalid panel coordinates');
    }
    if (canonical(panel.points[0]) !== canonical(panel.points.at(-1))) throw new Error('Panel is not closed');
    let area = 0;
    for (let i = 0; i < panel.points.length - 1; i++) {
      const a = panel.points[i]!, b = panel.points[i + 1]!;
      area += a[0] * b[1] - b[0] * a[1];
    }
    if (Math.abs(area) < 0.000001) throw new Error('Degenerate zero-area panel');
  }
  if (!geometry.stitches.length || geometry.stitches.length > 1000 || geometry.stitches.some(s => !ids.has(s.panelA) || !ids.has(s.panelB) || !Number.isInteger(s.edgeA) || s.edgeA < 0 || !Number.isInteger(s.edgeB) || s.edgeB < 0)) throw new Error('Invalid stitch graph');
  return geometry;
}

export async function runEngine(input: { document: GarmentDocument; inputDigest: string; outputDir: string; signal: AbortSignal }): Promise<{ geometry: PatternGeometry; files: { filename: string; bytes: Uint8Array; mime: string; kind: Artifact['kind'] }[] }> {
  input.signal.throwIfAborted();
  const payload = validateInput(input.document, input.inputDigest);
  if (!isAbsolute(input.outputDir)) throw new Error('Engine outputDir must be a trusted absolute staging path');
  const upstream = await realpath(process.env.SEW_ENGINE_SOURCE || defaultUpstream);
  let python = process.env.SEW_ENGINE_PYTHON;
  if (!python) {
    const ownPython = join(root, '.venv/bin/python');
    python = await access(ownPython, constants.X_OK).then(() => ownPython, () => join(upstream, '.venv/bin/python'));
  }
  if (!isAbsolute(python)) throw new Error('Engine Python override must be an absolute trusted executable');
  await mkdir(input.outputDir, { recursive: true, mode: 0o700 });
  const outputStat = await lstat(input.outputDir);
  if (!outputStat.isDirectory() || outputStat.isSymbolicLink()) throw new Error('Engine staging directory must not be a symlink');
  const attempt = await mkdtemp(join(resolve(input.outputDir), '.engine-'));
  let cacheDir: string | undefined;
  try {
    await chmod(attempt, 0o700);
    await mkdir(join(root, '.runtime'), { recursive: true, mode: 0o711 });
    cacheDir = await mkdtemp(join(root, '.runtime/cache-'));
    await chmod(cacheDir, 0o700);
    if (process.getuid?.() === 0) {
      await chown(attempt, 65534, 65534);
      await chown(cacheDir, 65534, 65534);
    }
    try {
      await executeTrusted(python, [join(root, 'worker.py'), upstream], attempt, JSON.stringify(payload), input.signal, 90_000, cacheDir);
    } catch (error) {
      if (error instanceof Error && error.message.includes('AssertionError: Start and end of an edge should differ')) throw new EngineInputError('The pattern engine produced a zero-length edge for this combination of body measurements and garment settings. This is an engine limitation, not proof a measurement is wrong. Your values are saved unchanged. Review Shape & body; accurate measurements should not be reduced just to make generation succeed.');
      throw error;
    }
    input.signal.throwIfAborted();
    const results = [];
    for (const definition of files) results.push({ ...definition, bytes: await readArtifact(join(attempt, definition.filename)) });
    const geometry = validateGeometry(JSON.parse(Buffer.from(results[0].bytes).toString('utf8')), input.inputDigest);
    if (geometry.family !== payload.family) throw new Error('Engine returned a different garment family');
    const svg = Buffer.from(results[1].bytes).toString('utf8');
    if (!svg.startsWith('<svg ') || !svg.endsWith('</svg>') || /<(?:script|foreignObject|image|use)\b|\son\w+\s*=|(?:href|url)\s*[=(]/i.test(svg)) throw new Error('Unsafe SVG artifact');
    if (Buffer.from(results[2].bytes.subarray(0, 5)).toString() !== '%PDF-') throw new Error('Invalid PDF artifact');
    input.signal.throwIfAborted();
    return { geometry, files: results };
  } finally {
    await Promise.all([rm(attempt, { recursive: true, force: true }), ...(cacheDir ? [rm(cacheDir, { recursive: true, force: true })] : [])]);
  }
}
