import { expect, test } from 'bun:test';
import { chmod, chown, mkdtemp, readFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { executeTrusted, validateGeometry } from './runner';

const root = import.meta.dir;
const python = join(root, '.venv/bin/python');
async function attempt(run: (directory: string) => Promise<void>) {
  const directory = await mkdtemp(join(root, '.test-safety-'));
  await chmod(directory, 0o700);
  if (process.getuid?.() === 0) await chown(directory, 65534, 65534);
  try { await run(directory); }
  finally { await rm(directory, { recursive: true, force: true }); }
}

test('kernel rejects oversized memory allocation, files and descriptor exhaustion under the real guard', async () => {
  await attempt(async directory => {
    const code = `import sys,os,json,errno
sys.path.insert(0,${JSON.stringify(root)})
from guard import constrain
constrain()
r={}
try:
    data=bytearray(3*1024**3)
except MemoryError:
    r['memory']=True
try:
    with open('oversized','wb',buffering=0) as f:
        chunk=b'x'*1024**2
        for i in range(17):
            f.write(chunk)
except OSError as e:
    r['file']=e.errno==errno.EFBIG
r['fileSize']=os.stat('oversized').st_size
fds=[]
try:
    for i in range(100):
        fds.append(os.open('/dev/null',os.O_RDONLY))
except OSError as e:
    r['descriptors']=e.errno==errno.EMFILE
finally:
    for fd in fds:
        os.close(fd)
r['cleanEnvironment']=not any(any(s in k for s in ('SECRET','TOKEN','KEY','PROXY')) for k in os.environ)
open('result.json','w').write(json.dumps(r))`;
    await executeTrusted(python, ['-c', code], directory, '', new AbortController().signal, 10_000);
    expect(JSON.parse(await readFile(join(directory, 'result.json'), 'utf8'))).toEqual({ memory: true, file: true, fileSize: 16 * 1024 ** 2, descriptors: true, cleanEnvironment: true });
  });
}, 20_000);

test('the CPU limit is enforced by the kernel, not just reported', async () => {
  await attempt(async directory => {
    const code = `import sys,resource
sys.path.insert(0,${JSON.stringify(root)})
from guard import constrain
constrain()
resource.setrlimit(resource.RLIMIT_CPU,(1,1))
while True: pass`;
    await expect(executeTrusted(python, ['-c', code], directory, '', new AbortController().signal, 10_000)).rejects.toThrow('SIGKILL');
  });
}, 20_000);

test('excessive subprocess diagnostics are terminated without being forwarded', async () => {
  await attempt(async directory => {
    await expect(executeTrusted(python, ['-c', "import os;os.write(1,b'x'*100000);import time;time.sleep(20)"], directory, '', new AbortController().signal, 10_000)).rejects.toThrow('diagnostic budget');
  });
}, 20_000);

test('malformed geometry and mismatched identity are rejected', () => {
  expect(() => validateGeometry(null, '0'.repeat(64))).toThrow();
  expect(() => validateGeometry({ schemaVersion: 1, units: 'cm' }, '0'.repeat(64))).toThrow();
});
