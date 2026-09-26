/** Local visual demo with its own scratch database and no model connection. */
import { mkdir, mkdtemp } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const port = Number(process.env.SEW_DEMO_PORT || 5175);
if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error('Use a demo port from 1024 to 65535.');
if (!await Bun.file(resolve(root, 'apps/web/dist/index.html')).exists()) throw new Error('Build the web app first: bun run build');
await mkdir(resolve(root, '.planning'), { recursive: true });
const dataDir = await mkdtemp(resolve(root, '.planning/visual-demo-session-'));
process.env.NODE_ENV = 'production';
process.env.SEW_DATA_DIR = dataDir;
process.env.SEW_ACCESS_KEY = 'sew-local-demo';
process.env.SEW_ALLOWED_ORIGINS = `http://127.0.0.1:${port}`;
for (const name of ['SEW_AI_API_KEY', 'SEW_AI_BASE_URL', 'SEW_AI_MODEL', 'SEW_CODEX_AUTH_FILE']) delete process.env[name];
const { default: application } = await import('../apps/web/server');
const server = Bun.serve({ hostname: '127.0.0.1', port, fetch: application.fetch });
console.log(`Visual demo: http://127.0.0.1:${server.port}/demo`);
console.log('Optional local studio access key: sew-local-demo (isolated demo data only).');
console.log(`Demo data: ${dataDir}`);
process.on('SIGTERM', () => { server.stop(true); process.exit(0); });
process.on('SIGINT', () => { server.stop(true); process.exit(0); });
