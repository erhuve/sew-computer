import { afterEach, describe, expect, test } from 'bun:test';
import { Database } from 'bun:sqlite';
import { chmodSync, copyFileSync, existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import sharp from 'sharp';
import { PDFDocument } from 'pdf-lib';
import { Hono } from 'hono';
import { assumed, canonical, type Draft, type ExportSnapshot, type GarmentDocument, type PatternGeometry, type ProjectState } from '../../packages/contracts';
import { createApi, type Api, type ApiOptions } from './index';
import type { Engine } from './jobs';
import type { Exporter } from './exports';
import { geometry as validateApiGeometry, hash, objectDigest } from './validation';
import { ENGINE_COMMIT, runEngine, validateGeometry as validateEngineGeometry } from '../../services/engine/runner';

const origin='https://sew.test',key='private-test-key-not-used-in-production';
const active:Api[]=[],directories:string[]=[];
afterEach(()=>{for(const api of active.splice(0))api.close();for(const path of directories.splice(0))rmSync(path,{recursive:true,force:true});});
const directory=()=>{const path=mkdtempSync(join(import.meta.dir,'.test-data-'));directories.push(path);return path;};
const delay=(ms:number)=>new Promise(resolve=>setTimeout(resolve,ms));
function latch<T>() {let resolve!:(value:T)=>void;const promise=new Promise<T>(r=>resolve=r);return {promise,resolve};}
async function waitUntil<T>(read:()=>Promise<T>,predicate:(value:T)=>boolean,timeout=5000):Promise<T> {const start=Date.now();for(;;){const value=await read();if(predicate(value))return value;if(Date.now()-start>timeout)throw new Error('Timed out waiting for test condition');await delay(20);}}
function fixtureGeometry(inputDigest:string,family='shirt'):PatternGeometry {return {schemaVersion:1,units:'mm',inputDigest,engineVersion:'test-only',family,panels:[{id:'front',name:'Test front',points:[[0,0],[100,0],[100,200],[0,200]],widthMm:100,heightMm:200}],stitches:[],warnings:['Test fixture, never a generated garment'],assumptions:['Synthetic test body'],classification:'printable-reference'};}
const fakeEngine:Engine=async input=>{const geometry=fixtureGeometry(input.inputDigest,input.document.garment.family);return {geometry,files:[{filename:'pattern.json',mime:'application/json',kind:'pattern-json',bytes:new TextEncoder().encode(JSON.stringify(geometry))},{filename:'pattern.svg',mime:'image/svg+xml',kind:'pattern-svg',bytes:new TextEncoder().encode('<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0L100 0L100 200Z"/></svg>')},{filename:'pattern.pdf',mime:'application/pdf',kind:'pattern-pdf',bytes:new TextEncoder().encode('%PDF-1.7\nTest fixture only\n%%EOF')}]};};
const fakeExporter:Exporter=async(snapshot,geometry,assets)=>{
  const doc=snapshot.document;
  const sections:Record<string,unknown>={overview:{title:doc.title,brief:doc.brief,sizeLabel:doc.sizeLabel,garment:doc.garment},requirements:doc.requirements,materials:doc.bom.filter(row=>row.category==='fabric'||row.category==='lining'),bom:doc.bom,finishedMeasurements:doc.poms,construction:doc.construction,patternInventory:{artifacts:snapshot.artifacts},review:{callouts:doc.callouts,comments:snapshot.comments},exportDisclosure:{...snapshot.disclosure,omissions:['Private fields omitted unless disclosed']}};
  if(snapshot.disclosure.includeBody)sections.bodyInputs=doc.body;
  const content={schemaVersion:1,snapshotId:snapshot.id,revisionId:snapshot.revision.id,projectId:snapshot.projectId,sections};
  const manifest={...content,manifestDigest:objectDigest(content)};
  const pdf=await PDFDocument.create();pdf.addPage().drawText('Draft review document - test-only renderer');
  const files=[{filename:'draft.pdf',mime:'application/pdf',bytes:await pdf.save()},{filename:'manifest.json',mime:'application/json',bytes:new TextEncoder().encode(JSON.stringify(manifest,null,2))},...assets.map(a=>({filename:a.artifact.filename,mime:a.artifact.mime,bytes:a.bytes}))];
  const delivery={schemaVersion:1,snapshotId:snapshot.id,revisionId:snapshot.revision.id,projectId:snapshot.projectId,manifestDigest:manifest.manifestDigest,files:files.map(f=>({filename:f.filename,mime:f.mime,size:f.bytes.length,digest:hash(f.bytes)}))};
  files.push({filename:'delivery.json',mime:'application/json',bytes:new TextEncoder().encode(JSON.stringify(delivery))});
  return {manifest,files};
};
function make(options:Partial<ApiOptions>={}) {
  const dataDir=options.dataDir??directory(),api=createApi({dataDir,allowedOrigins:[origin],authKey:key,...options});active.push(api);
  const app=new Hono().route('/api',api);let cookie='';
  const request=(method:string,path:string,body?:unknown,headers:Record<string,string>={})=>app.request('/api'+path,{method,headers:{Origin:origin,...(cookie?{Cookie:cookie}:{}),...(body!==undefined?{'Content-Type':'application/json'}:{}),...headers},body:body===undefined?undefined:JSON.stringify(body)});
  const login=async()=>{const response=await request('POST','/auth/login',{key});expect(response.status).toBe(204);cookie=response.headers.get('set-cookie')!.split(';')[0]!;return response;};
  const state=async(projectId:string)=>{const response=await request('GET',`/projects/${projectId}`);expect(response.status).toBe(200);return response.json() as Promise<ProjectState>;};
  const create=async()=>{const response=await request('POST','/projects',{title:'Private garment',brief:'Original unsupported embroidery stays'});expect(response.status).toBe(201);return response.json() as Promise<ProjectState>;};
  const save=async(project:ProjectState,edit:(doc:GarmentDocument)=>void)=>{const doc=structuredClone(project.draft.document);edit(doc);const response=await request('PUT',`/projects/${project.project.id}/draft`,{expectedVersion:project.draft.version,expectedRevisionId:project.draft.baseRevisionId,document:doc});expect(response.status).toBe(200);return state(project.project.id);};
  const publish=async(project:ProjectState)=>{const response=await request('POST',`/projects/${project.project.id}/revisions`,{expectedVersion:project.draft.version,expectedRevisionId:project.draft.baseRevisionId});expect(response.status).toBe(201);return response.json() as Promise<ProjectState>;};
  const saved=async()=>publish(await save(await create(),doc=>{doc.garment={family:'shirt',length:assumed(700),ease:assumed(80),flare:0};doc.requirements=[{id:'original',text:'Unusual embroidered asymmetric collar',status:'unsupported',note:'Preserve intent'}];}));
  const submit=async(project:ProjectState,requestId='request-one')=>{const response=await request('POST',`/projects/${project.project.id}/jobs`,{revisionId:project.project.headRevisionId,requestId});expect(response.status).toBe(202);return response.json();};
  const exported=async(project:ProjectState,disclosure={includeBody:false,includeReferences:false,includePatterns:false})=>{
    const response=await request('POST',`/projects/${project.project.id}/exports`,{revisionId:project.project.headRevisionId,disclosure});expect(response.status).toBe(201);
    const result=await response.json(),url=result.files.find((file:any)=>file.filename==='manifest.json').url;
    const manifestResponse=await request('GET',url.replace(/^\/api/,''));expect(manifestResponse.status).toBe(200);
    return {result,manifest:await manifestResponse.json()};
  };
  return {api,app,dataDir,request,login,state,create,save,publish,saved,submit,exported,get cookie(){return cookie;}};
}

describe('private API authentication and persistence',()=>{
  test('reports missing geometry inputs without changing the saved design',async()=>{
    const h=make({engine:runEngine});await h.login();
    const project=await h.publish(await h.create());await h.submit(project);
    const failed=await waitUntil(()=>h.state(project.project.id),state=>state.jobs[0]?.status==='failed');
    expect(failed.jobs[0]!.error).toContain('No garment family selected');
    expect(failed.jobs[0]!.error).toContain('Open Design');
    expect(failed.draft).toEqual(project.draft);
    expect(failed.revisions).toEqual(project.revisions);
    expect(failed.artifacts).toHaveLength(0);
    const shaped=await h.publish(await h.save(failed,doc=>{doc.garment.family='shirt';}));
    await h.submit(shaped,'missing-measurements');
    const missing=await waitUntil(()=>h.state(project.project.id),state=>state.jobs[0]?.status==='failed');
    expect(missing.jobs[0]!.error).toContain('body height must be explicitly known or assumed');
    expect(missing.jobs[0]!.error).toContain('Shape & body');
  });
  test('does not expose private engine runtime diagnostics',async()=>{
    const h=make({engine:async()=>{throw new Error('Private runtime path /secret/private-engine');}});await h.login();
    const project=await h.saved();await h.submit(project);
    const failed=await waitUntil(()=>h.state(project.project.id),state=>state.jobs[0]?.status==='failed');
    expect(failed.jobs[0]!.error).toBe('Engine failed; inspect the supported inputs and private engine setup');
  });
  test('both geometry boundaries reject collinear closed panels',()=>{
    const inputDigest=hash('degenerate fixture');
    const geometry=fixtureGeometry(inputDigest);
    geometry.engineVersion=ENGINE_COMMIT;
    geometry.panels[0]={id:'front',name:'Degenerate',points:[[0,0],[100,100],[50,50],[0,0]],widthMm:100,heightMm:100};
    geometry.stitches=[{panelA:'front',edgeA:0,panelB:'front',edgeB:1}];
    expect(()=>validateApiGeometry(geometry,inputDigest)).toThrow('zero-area');
    expect(()=>validateEngineGeometry(geometry,inputDigest)).toThrow('zero-area');
  });
  test('requires auth even on loopback, rejects missing/cross origin and keeps only hashed sessions',async()=>{
    const h=make();
    expect(await (await h.request('GET','/auth/status')).json()).toEqual({authenticated:false});
    expect((await h.request('GET','/projects')).status).toBe(401);
    expect((await h.api.request('http://localhost/projects')).status).toBe(401);
    expect((await h.request('POST','/auth/login',{key},{Origin:'https://attacker.test'})).status).toBe(403);
    expect((await h.request('POST','/auth/login',{key},{Origin:''})).status).toBe(403);
    expect((await h.request('POST','/auth/login',{key:'incorrect'})).status).toBe(401);
    const response=await h.login();expect(response.headers.get('set-cookie')).toContain('HttpOnly');expect(response.headers.get('set-cookie')).toContain('SameSite=Strict');expect(response.headers.get('set-cookie')).toContain('Secure');
    expect(await (await h.request('GET','/auth/status')).json()).toEqual({authenticated:true});
    expect((await h.request('POST','/projects',{title:'No CSRF'},{Origin:''})).status).toBe(403);
    const db=new Database(join(h.dataDir,'sew.sqlite'));const sessions=db.query('SELECT * FROM sessions').all() as {hash:string;expires:number}[];expect(sessions).toHaveLength(1);expect(sessions[0]!.hash).toBe(hash(h.cookie.split('=')[1]!));expect(JSON.stringify(sessions)).not.toContain(h.cookie.split('=')[1]!);db.close();
    const project=await h.create();h.api.close();const restarted=make({dataDir:h.dataDir});
    const persisted=await restarted.request('GET',`/projects/${project.project.id}`,undefined,{Cookie:h.cookie});expect(persisted.status).toBe(200);
    expect((await restarted.request('POST','/auth/logout',undefined,{Cookie:h.cookie})).status).toBe(204);
    expect((await restarted.request('GET','/projects',undefined,{Cookie:h.cookie})).status).toBe(401);
  });
  test('creates a private access key once, does not expose it, rate limits login',async()=>{
    const dataDir=directory(),api=createApi({dataDir,allowedOrigins:[origin]});active.push(api);
    const keyPath=join(dataDir,'access-key'),original=readFileSync(keyPath,'utf8');expect(original.length).toBeGreaterThan(30);expect(statSync(keyPath).mode&0o777).toBe(0o600);
    const status=await api.request('/auth/status');expect(await status.text()).not.toContain(original);api.close();
    const next=createApi({dataDir,allowedOrigins:[origin]});active.push(next);expect(readFileSync(keyPath,'utf8')).toBe(original);
    for(let i=0;i<10;i++)expect((await next.request('/auth/login',{method:'POST',headers:{Origin:origin,'Content-Type':'application/json'},body:JSON.stringify({key:'wrong'})})).status).toBe(401);
    expect((await next.request('/auth/login',{method:'POST',headers:{Origin:origin,'Content-Type':'application/json'},body:JSON.stringify({key:original})})).status).toBe(429);
  });
  test('CAS rejects concurrent saves, revisions are immutable, no-op publish rejects',async()=>{
    const h=make();await h.login();const first=await h.create(),saved=await h.save(first,doc=>{doc.title='Changed';});
    const conflict=await h.request('PUT',`/projects/${first.project.id}/draft`,{expectedVersion:first.draft.version,expectedRevisionId:null,document:first.draft.document});expect(conflict.status).toBe(409);expect((await conflict.json()).latest.version).toBe(saved.draft.version);
    const published=await h.publish(saved),revision=structuredClone(published.revisions[0]);await h.save(published,doc=>{doc.brief='New draft only';});
    expect((await h.state(first.project.id)).revisions[0]).toEqual(revision);
    const db=new Database(join(h.dataDir,'sew.sqlite'));expect(()=>db.query('UPDATE revisions SET json=? WHERE id=?').run('{}',revision!.id)).toThrow('immutable revision');db.close();
    const same=await h.publish(await h.save(await h.state(first.project.id),doc=>{doc.brief='Saved second revision';}));
    expect((await h.request('POST',`/projects/${first.project.id}/revisions`,{expectedVersion:same.draft.version,expectedRevisionId:same.draft.baseRevisionId})).status).toBe(409);
  });
  test('runtime validation rejects prototype keys, unknown fields and cross-project revision identities',async()=>{
    const h=make();await h.login();const a=await h.saved(),b=await h.saved();
    const poisoned=JSON.parse(JSON.stringify(a.draft.document).replace('"schemaVersion":1','"schemaVersion":1,"__proto__":{"admin":true}'));
    expect((await h.request('PUT',`/projects/${a.project.id}/draft`,{expectedVersion:a.draft.version,expectedRevisionId:a.draft.baseRevisionId,document:poisoned})).status).toBe(422);
    expect((await h.request('POST',`/projects/${a.project.id}/comments`,{revisionId:b.project.headRevisionId,anchor:'front',text:'Wrong project',reportedReviewer:'Someone'})).status).toBe(404);
    expect((await h.request('POST',`/projects/${a.project.id}/jobs`,{revisionId:b.project.headRevisionId,requestId:'x'})).status).toBe(404);
    expect((await h.request('POST','/projects',{title:'Title',owner:'attacker'})).status).toBe(422);
  });
});

describe('references',()=>{
  test('sniffs and sanitizes raster bytes, checks ownership and keeps revision references immutable',async()=>{
    const h=make();await h.login();let project=await h.create();const other=await h.create();
    const input=await sharp({create:{width:20,height:30,channels:3,background:'red'}}).jpeg().withMetadata({exif:{IFD0:{Copyright:'DO NOT EXPORT METADATA'}}}).toBuffer();
    const upload=await h.app.request(`/api/projects/${project.project.id}/references`,{method:'POST',headers:{Origin:origin,Cookie:h.cookie,'Content-Type':'image/svg+xml','X-Filename':'../../bad<name>.jpg'},body:input as BodyInit});expect(upload.status).toBe(201);const asset=await upload.json();expect(asset).toMatchObject({mime:'image/png',width:20,height:30});
    const get=await h.request('GET',`/projects/${project.project.id}/references/${asset.assetId}`);expect(get.status).toBe(200);expect(get.headers.get('cache-control')).toBe('no-store');const bytes=Buffer.from(await get.arrayBuffer()),metadata=await sharp(bytes).metadata();expect(metadata.format).toBe('png');expect(metadata.exif).toBeUndefined();expect(bytes.toString()).not.toContain('DO NOT EXPORT METADATA');
    expect((await h.request('GET',`/projects/${other.project.id}/references/${asset.assetId}`)).status).toBe(404);
    const doc=structuredClone(other.draft.document);doc.views=[{id:'front',assetId:asset.assetId,role:'front',kind:'reference',caption:'Owner-entered'}];
    expect((await h.request('PUT',`/projects/${other.project.id}/draft`,{expectedVersion:other.draft.version,expectedRevisionId:null,document:doc})).status).toBe(422);
    project=await h.publish(await h.save(project,d=>{d.views=doc.views;}));await h.save(project,d=>{d.views=[];});expect((await h.state(project.project.id)).revisions[0]!.document.views).toHaveLength(1);
    const svg=await h.app.request(`/api/projects/${project.project.id}/references`,{method:'POST',headers:{Origin:origin,Cookie:h.cookie,'Content-Type':'image/png'},body:'<svg onload="alert(1)"></svg>'});expect(svg.status).toBe(415);
  });
});

describe('job leases and fencing',()=>{
  test('commits verified patterns, provides private checked geometry and idempotent jobs',async()=>{
    const h=make({engine:fakeEngine});await h.login();const project=await h.saved(),job=await h.submit(project);
    const state=await waitUntil(()=>h.state(project.project.id),s=>s.jobs[0]?.status==='succeeded');expect(state.artifacts).toHaveLength(3);
    expect((await h.submit(project)).id).toBe(job.id);
    const geometry=await h.request('GET',`/projects/${project.project.id}/geometry/${project.project.headRevisionId}`);expect(geometry.status).toBe(200);expect((await geometry.json()).inputDigest).toBe(project.revisions[0]!.digest);
    const svg=state.artifacts.find(a=>a.kind==='pattern-svg')!;const response=await h.request('GET',`/projects/${project.project.id}/artifacts/${svg.id}`);expect(response.headers.get('content-security-policy')).toContain('sandbox');
    const pdf=state.artifacts.find(a=>a.kind==='pattern-pdf')!;expect((await h.request('GET',`/projects/${project.project.id}/artifacts/${pdf.id}`)).headers.get('content-disposition')).toContain('attachment');
    const other=await h.create();expect((await h.request('GET',`/projects/${other.project.id}/artifacts/${svg.id}`)).status).toBe(404);
    const db=new Database(join(h.dataDir,'sew.sqlite'));const row=db.query('SELECT storage_key FROM artifacts WHERE id=?').get(svg.id) as {storage_key:string};db.close();const path=join(h.dataDir,'blobs',row.storage_key);chmodSync(path,0o600);writeFileSync(path,'corrupt');expect((await h.request('GET',`/projects/${project.project.id}/artifacts/${svg.id}`)).status).toBe(409);
  });
  test('rejects missing engine rather than pretending generation worked',async()=>{
    const h=make();await h.login();const project=await h.saved();expect((await h.request('POST',`/projects/${project.project.id}/jobs`,{revisionId:project.project.headRevisionId,requestId:'unavailable'})).status).toBe(503);const state=await h.state(project.project.id);expect(state.artifacts).toHaveLength(0);expect((await h.request('GET',`/projects/${project.project.id}/geometry/${project.project.headRevisionId}`)).status).toBe(404);
  });
  test('cancel fences a late result even when the injected engine ignores abort',async()=>{
    const started=latch<void>(),finish=latch<void>();const h=make({engine:async input=>{started.resolve();await finish.promise;return fakeEngine(input);}});await h.login();const project=await h.saved(),job=await h.submit(project);await started.promise;
    expect((await h.request('POST',`/projects/${project.project.id}/jobs/${job.id}/cancel`)).status).toBe(200);finish.resolve();await delay(120);
    const state=await h.state(project.project.id);expect(state.jobs[0]!.status).toBe('cancelled');expect(state.artifacts).toHaveLength(0);expect(readdirSync(join(h.dataDir,'staging'))).toHaveLength(0);
  });
  test('saving a newer revision makes old running results stale but not editing its draft',async()=>{
    const started=latch<void>(),finish=latch<void>();const h=make({engine:async input=>{started.resolve();await finish.promise;return fakeEngine(input);}});await h.login();let project=await h.saved();await h.submit(project);await started.promise;
    project=await h.save(project,doc=>{doc.brief='Draft may change independently';});expect((await h.state(project.project.id)).jobs[0]!.status).toBe('running');
    project=await h.publish(project);finish.resolve();await delay(100);const state=await h.state(project.project.id);expect(state.jobs[0]!.status).toBe('stale');expect(state.artifacts).toHaveLength(0);
    expect((await h.request('POST',`/projects/${project.project.id}/jobs`,{revisionId:project.revisions[0]!.id,requestId:'old-revision'})).status).toBe(409);
  });
  test('delete permanently fences results and replay protects against an older DB restore',async()=>{
    const started=latch<void>(),finish=latch<void>();const h=make({engine:async input=>{started.resolve();await finish.promise;return fakeEngine(input);}});await h.login();const project=await h.saved();const backup=join(h.dataDir,'old.sqlite');copyFileSync(join(h.dataDir,'sew.sqlite'),backup);await h.submit(project);await started.promise;
    expect((await h.request('DELETE',`/projects/${project.project.id}`)).status).toBe(204);finish.resolve();await delay(100);expect((await h.request('GET',`/projects/${project.project.id}`)).status).toBe(404);
    h.api.close();copyFileSync(backup,join(h.dataDir,'sew.sqlite'));const restored=make({dataDir:h.dataDir});expect((await restored.request('GET','/projects',undefined,{Cookie:h.cookie})).status).toBe(401);await restored.login();expect((await restored.request('GET',`/projects/${project.project.id}`)).status).toBe(404);
    expect((await restored.request('GET',`/projects/${project.project.id}/geometry/${project.project.headRevisionId}`)).status).toBe(404);
  });
  test('one SQLite scheduler runs across API instances and close recovers bounded work',async()=>{
    let activeEngines=0,maxActive=0,calls=0;const finish=latch<void>(),started=latch<void>();
    const engine:Engine=async input=>{calls++;activeEngines++;maxActive=Math.max(maxActive,activeEngines);started.resolve();try{await Promise.race([finish.promise,new Promise((_,reject)=>input.signal.addEventListener('abort',()=>reject(new Error('aborted')),{once:true}))]);return await fakeEngine(input);}finally{activeEngines--;}};
    const h=make({engine}),other=make({dataDir:h.dataDir,engine});await h.login();const project=await h.saved();await h.submit(project);await started.promise;await delay(150);expect(calls).toBe(1);expect(maxActive).toBe(1);
    h.api.close();finish.resolve();await waitUntil(async()=>(await other.request('GET',`/projects/${project.project.id}`,undefined,{Cookie:h.cookie})).json(),s=>s.jobs[0]?.status==='succeeded',6000);expect(maxActive).toBe(1);
  });
  test('same-owner scheduler reacquisition fences its previous attempt immediately',async()=>{
    const started=latch<void>();let calls=0,aborted=false;
    const engine:Engine=async input=>{
      calls++;
      if(calls===1){started.resolve();await new Promise((_,reject)=>input.signal.addEventListener('abort',()=>{aborted=true;reject(new Error('fenced'));},{once:true}));}
      return fakeEngine(input);
    };
    const harness=make({engine});await harness.login();const project=await harness.saved();await harness.submit(project);await started.promise;
    const database=new Database(join(harness.dataDir,'sew.sqlite'));
    try{database.query('UPDATE scheduler SET deadline=0').run();}finally{database.close();}
    const result=await waitUntil(()=>harness.state(project.project.id),state=>state.jobs[0]?.status==='succeeded',6000);
    expect(aborted).toBe(true);expect(calls).toBe(2);expect(result.artifacts).toHaveLength(3);
  },10000);
});

describe('immutable export and three-way import',()=>{
  test('redacted export imports only edited fields without erasing private body, references or unsupported intent',async()=>{
    const h=make({exporter:fakeExporter});await h.login();let project=await h.saved();project=await h.publish(await h.save(project,doc=>{doc.body.waist=assumed(850);doc.bom=[{id:'fabric',name:'Linen',category:'fabric',specification:'Unknown weight',placement:'Main',quantity:'Unknown',source:'Owner'}];}));
    const {manifest}=await h.exported(project);expect(manifest.sections.bodyInputs).toBeUndefined();expect(JSON.stringify(manifest)).not.toContain('"waist"');
    project=await h.save(project,doc=>{doc.body.waist=assumed(900);doc.title='Current title';});
    manifest.sections.construction=[{id:'seam',operation:'Proposed seam',note:'Maker feedback'}];manifest.sections.bodyInputs={waist:assumed(1)};
    const response=await h.request('POST',`/projects/${project.project.id}/imports/preview`,{manifest});expect(response.status).toBe(201);const preview=await response.json();expect(preview.changes.map((c:any)=>c.path)).toEqual(['/construction/seam']);
    const accepted=await h.request('POST',`/projects/${project.project.id}/imports/${preview.id}/accept`,{expectedVersion:project.draft.version,expectedRevisionId:project.draft.baseRevisionId});expect(accepted.status).toBe(200);const next=await accepted.json();expect(next.draft.document.body.waist.value).toBe(900);expect(next.draft.document.title).toBe('Current title');expect(next.draft.document.requirements).toEqual(project.draft.document.requirements);expect(next.revisions).toEqual(project.revisions);
  });
  test('shows explicit baseline/current/incoming conflicts and rechecks draft version at acceptance',async()=>{
    const h=make({exporter:fakeExporter});await h.login();let project=await h.saved();const {manifest}=await h.exported(project);project=await h.save(project,doc=>{doc.title='Current';});manifest.sections.overview.title='Incoming';
    const previewResponse=await h.request('POST',`/projects/${project.project.id}/imports/preview`,{manifest});expect(previewResponse.status).toBe(201);const preview=await previewResponse.json();expect(preview.changes).toContainEqual({path:'/title',before:'Current',after:'Incoming',baseline:'Private garment',conflict:true});
    expect((await h.request('POST',`/projects/${project.project.id}/imports/${preview.id}/accept`,{expectedVersion:project.draft.version,expectedRevisionId:project.draft.baseRevisionId})).status).toBe(409);
    project=await h.save(project,doc=>{doc.brief='Concurrent edit';});expect((await h.request('POST',`/projects/${project.project.id}/imports/${preview.id}/accept`,{expectedVersion:project.draft.version,expectedRevisionId:project.draft.baseRevisionId,resolutions:{'/title':'incoming'}})).status).toBe(409);
    const fresh=await (await h.request('POST',`/projects/${project.project.id}/imports/preview`,{manifest})).json();const accepted=await h.request('POST',`/projects/${project.project.id}/imports/${fresh.id}/accept`,{expectedVersion:project.draft.version,expectedRevisionId:project.draft.baseRevisionId,resolutions:{'/title':'incoming'}});expect(accepted.status).toBe(200);expect((await accepted.json()).draft.document.title).toBe('Incoming');
  });
  test('omitted rows do not wipe current additions; explicit deletion needs a choice; units compare physically',async()=>{
    const h=make({exporter:fakeExporter});await h.login();let project=await h.saved();const {manifest}=await h.exported(project);project=await h.save(project,doc=>{doc.requirements.push({id:'new',text:'Current-only',note:'Keep',status:'unresolved'});});
    manifest.sections.overview.garment.length={...manifest.sections.overview.garment.length,value:70,unit:'cm'};manifest.sections.requirements=[];
    const preview=await (await h.request('POST',`/projects/${project.project.id}/imports/preview`,{manifest})).json();expect(preview.changes).toHaveLength(1);expect(preview.changes[0]).toMatchObject({path:'/requirements/original',conflict:true});
    const accepted=await h.request('POST',`/projects/${project.project.id}/imports/${preview.id}/accept`,{expectedVersion:project.draft.version,expectedRevisionId:project.draft.baseRevisionId,resolutions:{'/requirements/original':'keep'}});expect(accepted.status).toBe(200);expect((await accepted.json()).draft.document.requirements).toHaveLength(2);
  });
  test('export stays on captured saved revision while draft and head change; delivery filenames are guarded',async()=>{
    const started=latch<void>(),finish=latch<void>();let captured:ExportSnapshot|undefined;
    const h=make({exporter:async(snapshot,geometry,assets)=>{captured=structuredClone(snapshot);started.resolve();await finish.promise;return fakeExporter(snapshot,geometry,assets);}});await h.login();let project=await h.saved();const original=structuredClone(project.revisions[0]);
    const pending=h.request('POST',`/projects/${project.project.id}/exports`,{revisionId:project.project.headRevisionId,disclosure:{includeBody:false,includeReferences:false,includePatterns:false}});await started.promise;project=await h.publish(await h.save(project,doc=>{doc.title='New head';}));finish.resolve();const result=await pending;expect(result.status).toBe(201);expect(captured!.revision).toEqual(original);const delivered=await result.json();const file=await h.request('GET',delivered.files.find((f:any)=>f.filename==='manifest.json').url.replace(/^\/api/,''));expect((await file.json()).revisionId).toBe(original!.id);
    expect((await h.request('GET',`/projects/${project.project.id}/exports/${delivered.snapshotId}/access-key`)).status).toBe(404);const other=await h.create();expect((await h.request('GET',`/projects/${other.project.id}/exports/${delivered.snapshotId}/manifest.json`)).status).toBe(404);
  });
  test('requested missing patterns fail, included patterns are byte identical and corrupt artifacts fail export',async()=>{
    const h=make({engine:fakeEngine,exporter:fakeExporter});await h.login();const project=await h.saved();expect((await h.request('POST',`/projects/${project.project.id}/exports`,{revisionId:project.project.headRevisionId,disclosure:{includeBody:false,includeReferences:false,includePatterns:true}})).status).toBe(409);
    await h.submit(project);const state=await waitUntil(()=>h.state(project.project.id),s=>s.jobs[0]?.status==='succeeded');const {result}=await h.exported(project,{includeBody:false,includeReferences:false,includePatterns:true});
    for(const artifact of state.artifacts){const file=result.files.find((f:any)=>f.filename===artifact.filename);expect(file).toBeDefined();const response=await h.request('GET',file.url.replace(/^\/api/,''));expect(hash(new Uint8Array(await response.arrayBuffer()))).toBe(artifact.digest);}
    const db=new Database(join(h.dataDir,'sew.sqlite'));const row=db.query('SELECT storage_key FROM artifacts LIMIT 1').get() as {storage_key:string};db.close();rmSync(join(h.dataDir,'blobs',row.storage_key));expect((await h.request('POST',`/projects/${project.project.id}/exports`,{revisionId:project.project.headRevisionId,disclosure:{includeBody:false,includeReferences:false,includePatterns:true}})).status).toBe(409);
  });
  test('deletion during export prevents installation, unknown snapshots and forged source identity reject',async()=>{
    const started=latch<void>(),finish=latch<void>();const h=make({exporter:async(...args)=>{started.resolve();await finish.promise;return fakeExporter(...args);}});await h.login();const project=await h.saved();const pending=h.request('POST',`/projects/${project.project.id}/exports`,{revisionId:project.project.headRevisionId,disclosure:{includeBody:false,includeReferences:false,includePatterns:false}});await started.promise;await h.request('DELETE',`/projects/${project.project.id}`);finish.resolve();expect((await pending).status).toBe(404);
    const db=new Database(join(h.dataDir,'sew.sqlite'));expect((db.query('SELECT COUNT(*) AS count FROM export_files').get() as {count:number}).count).toBe(0);db.close();
    const other=make({exporter:fakeExporter});await other.login();const a=await other.saved(),b=await other.saved(),{manifest}=await other.exported(a);expect((await other.request('POST',`/projects/${b.project.id}/imports/preview`,{manifest})).status).toBe(404);manifest.manifestDigest='0'.repeat(64);expect((await other.request('POST',`/projects/${a.project.id}/imports/preview`,{manifest})).status).toBe(409);
  });
});

describe('recovery and additional privacy regressions',()=>{
  test('old live API rejects old key after key rotation',async()=>{
    const h=make();await h.login();const other=make({dataDir:h.dataDir,authKey:'replacement-key'});
    expect((await h.request('GET','/projects')).status).toBe(401);
    expect((await h.request('POST','/auth/login',{key})).status).toBe(401);
    expect((await h.request('POST','/auth/login',{key:'replacement-key'})).status).toBe(204);
    expect((await other.request('POST','/auth/login',{key:'replacement-key'})).status).toBe(204);
  });
  test('deletion physically purges owned records and files, preserves unrelated projects',async()=>{
    const h=make({exporter:fakeExporter});await h.login();const project=await h.saved(),other=await h.saved();await h.exported(project);await h.exported(other);
    const db=new Database(join(h.dataDir,'sew.sqlite'));const files=db.query('SELECT storage_key FROM export_files WHERE snapshot_id IN (SELECT id FROM snapshots WHERE project_id=?)').all(project.project.id) as {storage_key:string}[];expect(files).toHaveLength(3);
    expect((await h.request('DELETE',`/projects/${project.project.id}`)).status).toBe(204);
    for(const {storage_key} of files)expect(existsSync(join(h.dataDir,'blobs',storage_key))).toBe(false);
    expect(db.query('SELECT json,draft,deleted FROM projects WHERE id=?').get(project.project.id)).toEqual({json:'{}',draft:'{}',deleted:1});db.close();expect((await h.request('GET',`/projects/${other.project.id}`)).status).toBe(200);
  });
  test('truncated runtime journal fails closed and orphan writes are reconciled on restart',async()=>{
    const h=make({exporter:fakeExporter});await h.login();const project=await h.saved();await h.exported(project);h.api.close();
    const orphan='a'.repeat(24);writeFileSync(join(h.dataDir,'blobs',orphan),'private orphan bytes');const second=make({dataDir:h.dataDir});expect(existsSync(join(h.dataDir,'blobs',orphan))).toBe(false);await second.login();
    expect((await second.request('DELETE',`/projects/${project.project.id}`)).status).toBe(204);const journal=join(h.dataDir,'deletions.jsonl');writeFileSync(journal,readFileSync(journal).subarray(0,-1));
    expect((await second.request('GET','/projects')).status).toBe(503);second.api.close();expect(()=>make({dataDir:h.dataDir})).toThrow('Truncated deletion journal');
  });
  test('locally deleted row receives an explicit restoration conflict',async()=>{
    const h=make({exporter:fakeExporter});await h.login();let project=await h.saved();project=await h.publish(await h.save(project,doc=>{doc.bom=[{id:'fabric',name:'Linen',category:'fabric',specification:'A',placement:'Body',quantity:'Unknown',source:'Owner'}];}));
    const {manifest}=await h.exported(project);project=await h.save(project,doc=>{doc.bom=[];});manifest.sections.bom[0].specification='External correction';delete manifest.sections.materials;
    const response=await h.request('POST',`/projects/${project.project.id}/imports/preview`,{manifest});expect(response.status).toBe(201);const preview=await response.json();expect(preview.changes[0]).toMatchObject({path:'/bom/fabric',before:null,conflict:true});
    const accepted=await h.request('POST',`/projects/${project.project.id}/imports/${preview.id}/accept`,{expectedVersion:project.draft.version,expectedRevisionId:project.draft.baseRevisionId,resolutions:{'/bom/fabric':'incoming'}});expect(accepted.status).toBe(200);expect((await accepted.json()).draft.document.bom[0].specification).toBe('External correction');
  });
  test('missing renderer and missing deletion journal never invent successful output',async()=>{
    const h=make();await h.login();const project=await h.saved();expect((await h.request('POST',`/projects/${project.project.id}/exports`,{revisionId:project.project.headRevisionId})).status).toBe(503);
    h.api.close();rmSync(join(h.dataDir,'deletions-watermark.json'));expect(()=>make({dataDir:h.dataDir})).toThrow('Deletion journal incomplete');
  });
  test('altered renderer projection is rejected and source import cannot create authority',async()=>{
    const h=make({exporter:async(...args)=>{const result=await fakeExporter(...args);(result.manifest.sections as any).overview.title='Unrelated design';return result;}});await h.login();const project=await h.saved();expect((await h.request('POST',`/projects/${project.project.id}/exports`,{revisionId:project.project.headRevisionId})).status).toBe(422);
    const other=make({dataDir:h.dataDir,exporter:fakeExporter});await other.login();const {manifest}=await other.exported(project);manifest.sections.review.comments=[{recordedBy:'professional',text:'Guaranteed fit'}];manifest.sections.patternInventory.artifacts=[{classification:'cutting-ready'}];manifest.sections.bodyInputs={waist:assumed(1)};
    const response=await other.request('POST',`/projects/${project.project.id}/imports/preview`,{manifest});expect(response.status).toBe(201);const preview=await response.json();expect(preview.changes).toHaveLength(0);expect(preview.warnings.join(' ')).toContain('ignored');
  });
});

describe('trusted renderer and geometry boundary checks',()=>{
  test('unsafe SVG, wrong input digests and missing artifact kinds fail without installing partial files',async()=>{
    for(const change of [(r:Awaited<ReturnType<Engine>>)=>{r.files[1]!.bytes=new TextEncoder().encode('<svg><script>alert(1)</script></svg>');},(r:Awaited<ReturnType<Engine>>)=>{r.geometry.inputDigest='0'.repeat(64);},(r:Awaited<ReturnType<Engine>>)=>{r.files.pop();}]){
      const h=make({engine:async input=>{const result=await fakeEngine(input);change(result);return result;}});await h.login();const project=await h.saved();await h.submit(project);const state=await waitUntil(()=>h.state(project.project.id),s=>s.jobs[0]?.status==='failed');expect(state.artifacts).toHaveLength(0);expect(readdirSync(join(h.dataDir,'blobs'))).toHaveLength(0);
    }
  });
  test('default export records committed pattern inventory but delivers no pattern bytes',async()=>{
    const h=make({engine:fakeEngine,exporter:fakeExporter});await h.login();const project=await h.saved();await h.submit(project);await waitUntil(()=>h.state(project.project.id),s=>s.jobs[0]?.status==='succeeded');
    const {manifest,result}=await h.exported(project);expect(manifest.sections.patternInventory.artifacts).toHaveLength(3);expect(result.files).toHaveLength(3);expect(manifest.sections.exportDisclosure.includePatterns).toBe(false);expect(manifest.sections.patternInventory.panels).toBeUndefined();
  });
  test('restore marker disables an existing instance and refuses reopening',async()=>{
    const h=make();await h.login();writeFileSync(join(h.dataDir,'RESTORE_PENDING'),'operator recovery incomplete');expect((await h.request('GET','/auth/status')).status).toBe(503);h.api.close();expect(()=>make({dataDir:h.dataDir})).toThrow('Restore has not been verified');
  });
});
