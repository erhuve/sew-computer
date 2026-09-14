import { Hono } from 'hono';
import { createServer } from 'vite';
import { resolve, dirname, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createApi } from '../api';
import { runEngine } from '../../services/engine/runner';
import { buildExport } from '../../packages/tech-pack';

const root=dirname(fileURLToPath(import.meta.url));
const siteFile=Bun.file(resolve(root,'zosite.json'));
const config=await siteFile.exists()?await siteFile.json():{local_port:5173,publish:{published_port:5173}};
const production=process.env.NODE_ENV==='production';
const port=Number(process.env.PORT||(production?config.publish.published_port:config.local_port));
const allowedOrigins=process.env.SEW_ALLOWED_ORIGINS?.split(',').map(v=>v.trim()).filter(Boolean)||[
  `http://localhost:${port}`,`http://127.0.0.1:${port}`,`https://zite-${config.local_port}-hatsunemiku.zo.computer`,
];
const globalState=globalThis as typeof globalThis & {__sewApi?:ReturnType<typeof createApi>};
globalState.__sewApi?.close();
const api=createApi({dataDir:process.env.SEW_DATA_DIR||resolve(root,'../../.local'),allowedOrigins,authKey:process.env.SEW_ACCESS_KEY,engine:runEngine,exporter:buildExport});
globalState.__sewApi=api;
const app=new Hono();
app.route('/api',api);
app.all('/api/*',c=>c.json({error:'Not found'},404));
const vite=production?null:await createServer({root,server:{middlewareMode:true,hmr:false,ws:false,fs:{strict:true,allow:[root,resolve(root,'../../packages/contracts')]}},appType:'custom'});
app.get('*',async c=>{
  let path:string;
  try{path=decodeURIComponent(c.req.path);}catch{return c.text('Invalid path',400);}
  if(path.includes('\0')||path.includes('\\')||path.split('/').some(p=>p.startsWith('.')||p==='..'))return c.text('Not found',404);
  c.header('X-Content-Type-Options','nosniff');
  c.header('Referrer-Policy','no-referrer');
  c.header('Cache-Control','no-store');
  const directory=resolve(root,production?'dist':'public');
  const filePath=resolve(directory,'.'+path);
  if(filePath.startsWith(directory+sep)){
    const file=Bun.file(filePath);
    if(await file.exists()&&(await file.stat()).isFile())return new Response(file,{headers:{'Content-Type':file.type,'X-Content-Type-Options':'nosniff','Cache-Control':'no-store'}});
  }
  if(vite&&(/^(\/src\/|\/node_modules\/\.vite\/|\/@vite\/|\/@react-refresh|\/@id\/)/.test(path)||/^\/@fs\/home\/workspace\/Code\/sew-computer\/(packages\/contracts\/|node_modules\/)/.test(path))){
    try{const result=await vite.transformRequest(c.req.path);if(result)return new Response(result.code,{headers:{'Content-Type':'application/javascript','Cache-Control':'no-store'}});}catch{return c.text('Module not ready',503);}
    return c.text('Not found',404);
  }
  if(extname(path)&&path!=='/index.html')return c.text('Not found',404);
  let html=await Bun.file(resolve(root,production?'dist/index.html':'index.html')).text();
  if(vite)html=await vite.transformIndexHtml(c.req.path,html);
  return c.html(html);
});
export default {fetch:app.fetch,port,idleTimeout:255};
