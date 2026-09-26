/** Optional local runtime for hosts without the locked Linux worker environment. */
import { execFile, spawn } from 'node:child_process';
import { promisify } from 'node:util';
import { randomUUID } from 'node:crypto';
import { chmod, realpath } from 'node:fs/promises';
import { dirname, isAbsolute, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const exec = promisify(execFile);
const engine = dirname(fileURLToPath(import.meta.url));
export async function executeContainer(args: string[], cwd: string, payload: string, signal: AbortSignal, timeoutMs: number, cacheDir: string) {
  signal.throwIfAborted();
  const image = process.env.SEW_ENGINE_CONTAINER!;
  if (!/^sha256:[a-f0-9]{64}$/.test(image)) throw new Error('Container runtime requires a pinned image digest');
  const docker = process.env.SEW_ENGINE_DOCKER || '/usr/bin/docker';
  if (!isAbsolute(docker)) throw new Error('Docker must be an absolute trusted executable');
  const output = await realpath(cwd), cache = await realpath(cacheDir);
  // Private parent directories retain mode 0700; only the isolated worker can
  // reach these bind-mounted leaves as its unprivileged Linux uid.
  await chmod(output, 0o777); if (cache !== output) await chmod(cache, 0o777);
  const mounts: [string, string, boolean][] = [[engine, '/engine', true], [output, '/output', false]];
  if (cache !== output) mounts.push([cache, '/cache', false]);
  if (process.env.SEW_ENGINE_SOURCE) mounts.push([await realpath(process.env.SEW_ENGINE_SOURCE), '/upstream', true]);
  const translate = (value: string) => {
    if (!isAbsolute(value)) return value;
    for (const [source, destination] of mounts) {
      const tail = relative(source, value);
      if (!tail || (!tail.startsWith('..' + sep) && tail !== '..' && !isAbsolute(tail))) return destination + (tail ? '/' + tail.split(sep).join('/') : '');
    }
    throw new Error('Worker argument is outside its mounted inputs');
  };
  const name = `sew-worker-${randomUUID()}`;
  const home = cache === output ? '/output' : '/cache';
  const environment={PATH:'/usr/local/bin:/usr/bin:/bin',LANG:'C.UTF-8',LC_ALL:'C.UTF-8',HOME:home,TMPDIR:home,MPLCONFIGDIR:`${home}/.mpl`,MPLBACKEND:'Agg',OPENBLAS_NUM_THREADS:'1',OMP_NUM_THREADS:'1',MKL_NUM_THREADS:'1',NUMEXPR_NUM_THREADS:'1',PYTHONDONTWRITEBYTECODE:'1'};
  let container: string | undefined;
  try {
    const created = await exec(docker, ['create', '--name', name, '--entrypoint', '/usr/bin/env', '--memory', '3g', '--memory-swap', '3g', '--cpus', '2', '--pids-limit', '128', '--network', 'none', '--read-only', '--user', '65534:65534', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--tmpfs', '/tmp:rw,nosuid,nodev,size=67108864',
      ...mounts.flatMap(([source, target, readOnly]) => ['--mount', `type=bind,src=${source},dst=${target}${readOnly ? ',readonly' : ''}`]),
      '-w', '/output', '-i', image, '-i', ...Object.entries(environment).map(([key,value])=>`${key}=${value}`), 'python', '-I', '-B', ...args.map(translate)], { timeout: 15000, maxBuffer: 65536 });
    container = created.stdout.trim();
    if (!/^[a-f0-9]{64}$/.test(container)) throw new Error('Invalid isolated container identity');
    signal.throwIfAborted();
    await new Promise<void>((resolve, reject) => {
      const child = spawn(docker, ['start', '--attach', '--interactive', container!], { stdio: ['pipe', 'pipe', 'pipe'] });
      let failure: Error | undefined, size = 0, diagnostic = '';
      const stop = () => { child.kill('SIGKILL'); void exec(docker, ['kill', container!], { timeout: 5000, maxBuffer: 65536 }).catch(() => {}); };
      const abort = () => { failure = new Error('Engine cancelled'); stop(); };
      const timer = setTimeout(() => { failure = new Error('Engine exceeded wall timeout'); stop(); }, timeoutMs);
      signal.addEventListener('abort', abort, { once: true });
      if (signal.aborted) abort();
      const consume = (chunk: Buffer, stderr = false) => { size += chunk.length; if (stderr) diagnostic = (diagnostic + chunk.toString()).slice(-4000); if (size > 65536) { failure = new Error('Engine diagnostic budget exceeded'); stop(); } };
      child.stdout.on('data', chunk => consume(chunk)); child.stderr.on('data', chunk => consume(chunk, true)); child.stdin.on('error', () => {});
      child.once('error', error => { failure = error; });
      child.once('close', code => { clearTimeout(timer); signal.removeEventListener('abort', abort); if (failure) reject(failure); else if (code !== 0) reject(new Error(`Engine failed (${code===137?'SIGKILL':code}): ${diagnostic}`)); else resolve(); });
      child.stdin.end(payload);
    });
  } finally {
    // The container exists before execution starts, so cancellation always has
    // an owned identity to reap, including pre-start cancellation.
    await exec(docker, ['rm', '--force', container || name], { timeout: 10000, maxBuffer: 65536 }).catch(error => { if (container) throw error; });
  }
}
