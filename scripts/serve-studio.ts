/** Local end-to-end studio. Credentials and runtime paths come from server environment. */
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
const root=resolve(import.meta.dirname,'..'),port=Number(process.env.SEW_STUDIO_PORT||5176);
if(!Number.isInteger(port)||port<1024||port>65535)throw new Error('Use a studio port from 1024 to 65535.');
if(!await Bun.file(resolve(root,'apps/web/dist/index.html')).exists())throw new Error('Build the web app first: bun run build');
const dataDir=resolve(process.env.SEW_DATA_DIR||resolve(root,'.planning/local-studio'));
await mkdir(dataDir,{recursive:true,mode:0o700});
process.env.NODE_ENV='production';process.env.SEW_DATA_DIR=dataDir;
process.env.SEW_ACCESS_KEY??='sew-local-demo';process.env.SEW_ALLOWED_ORIGINS=`http://127.0.0.1:${port}`;
const {default:application}=await import('../apps/web/server');
const server=Bun.serve({hostname:'127.0.0.1',port,fetch:application.fetch});
console.log(`Studio: http://127.0.0.1:${server.port}`);
console.log(`Private local data: ${dataDir}`);
// Closing the API aborts workers; let their bounded cleanup finish before exit.
const close=()=>{server.stop(true);(globalThis as any).__sewApi?.close();};
process.on('SIGTERM',close);process.on('SIGINT',close);
export {};
