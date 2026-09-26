import { Hono } from 'hono';
import { setCookie, deleteCookie } from 'hono/cookie';
import sharp from 'sharp';
import { z } from 'zod';
import { DisclosureSchema, DocumentSchema, Id, emptyDocument, type Artifact, type Draft, type Project, type Revision, type ReviewComment } from '../../packages/contracts';
import type { BodyProfile } from '../../packages/contracts/sizing';
import { Handoff, committedPatterns, type Exporter } from './exports';
import { acceptImport, previewImport } from './imports';
import { InterpretationService, type Interpreter } from './interpretation';
import { InterpretationQueue } from './interpretation-jobs';
import { ThreeDQueue, type InspectionEngine } from './three-d-jobs';
import { JobQueue, type Engine } from './jobs';
import { Store, type ArtifactRow } from './store';
import { ApiError, document, filenameSchema, hash, id, identitySchema, json, now, objectDigest, readBounded } from './validation';

export type ApiOptions={dataDir:string;allowedOrigins:string[];authKey?:string;engine?:Engine;exporter?:Exporter;interpreter?:Interpreter;inspectionEngine?:InspectionEngine;automaticPreviews?:boolean};
export type Api=Hono & {close:()=>void};
const sessionName='sew_session';
const sessionMs=12*60*60*1000;
const mutation=new Set(['POST','PUT','PATCH','DELETE']);
const newProjectSchema=z.object({title:z.string().min(1).max(160),brief:z.string().max(8000).optional()}).strict();
const draftSchema=identitySchema.extend({document:z.unknown()});
const exportSchema=z.object({revisionId:Id,disclosure:DisclosureSchema.default({includeBody:false,includeReferences:false,includePatterns:false})}).strict();
const importAcceptSchema=identitySchema.extend({resolutions:z.record(z.string().max(500),z.enum(['keep','incoming'])).optional()});

function fileResponse(bytes:Uint8Array,mime:string,filename?:string):Response {
  const headers=new Headers({'Content-Type':mime,'Cache-Control':'no-store','X-Content-Type-Options':'nosniff','Content-Length':String(bytes.byteLength)});
  if(filename) headers.set('Content-Disposition',`attachment; filename="${filename}"`);
  if(mime==='image/svg+xml') headers.set('Content-Security-Policy',"sandbox; default-src 'none'; style-src 'unsafe-inline'");
  return new Response(bytes as BodyInit,{headers});
}

export function createApi(options:ApiOptions):Api {
  if(!options.allowedOrigins.length || options.allowedOrigins.some(origin=>{
    try {const url=new URL(origin);return !['https:','http:'].includes(url.protocol)||url.origin!==origin||url.username!==''||url.password!=='';} catch{return true;}
  })) throw new Error('Configure exact allowed HTTP(S) origins');
  const origins=new Set(options.allowedOrigins), store=new Store(options.dataDir,options.authKey);
  store.reconcile();
  const queue=new JobQueue(store,options.engine), handoff=new Handoff(store,options.exporter);
  const interpretation=new InterpretationService(store,options.interpreter);
  const interpretations=new InterpretationQueue(store,interpretation);
  const inspections=new ThreeDQueue(store,options.inspectionEngine,options.automaticPreviews);
  const api=new Hono() as Api;
  let closed=false, decoding=0;
  const tokenHash=(request:Request):string|null=>{
    const cookie=request.headers.get('cookie')??'';
    const token=request.headers.get('X-Sew-Session')??cookie.split(';').map(v=>v.trim()).find(v=>v.startsWith(sessionName+'='))?.slice(sessionName.length+1);
    return token && /^[A-Za-z0-9_-]{48}$/.test(token) ? hash(token) : null;
  };
  const authenticated=(request:Request):boolean=>{
    const token=tokenHash(request);
    return !!token && !!store.db.query('SELECT 1 FROM sessions WHERE hash=? AND expires>?').get(token,Date.now());
  };
  api.onError((error,c)=>{
    if(error instanceof ApiError) return c.json({error:error.message,...error.extra},error.status as any);
    if(error instanceof z.ZodError) return c.json({error:'Invalid request data',issues:error.issues.map(v=>({path:v.path.join('.'),message:v.message})).slice(0,20)},422);
    return c.json({error:'Request failed; no unverified output was published'},500);
  });
  api.use('*',async(c,next)=>{
    c.header('Cache-Control','no-store');c.header('X-Content-Type-Options','nosniff');
    if(closed) return c.json({error:'API closed'},503);
    store.assertReady();
    if(mutation.has(c.req.method) && !origins.has(c.req.header('Origin')??'')) throw new ApiError(403,'Origin not allowed');
    const path=new URL(c.req.url).pathname.replace(/^\/api(?=\/|$)/,'');
    if(!((c.req.method==='GET'&&path==='/auth/status')||(c.req.method==='POST'&&path==='/auth/login')) && !authenticated(c.req.raw)) throw new ApiError(401,'Authentication required');
    await next();
  });
  api.get('/auth/status',c=>c.json({authenticated:authenticated(c.req.raw)}));
  api.post('/auth/login',async c=>{
    const body=z.object({key:z.string().min(1).max(512)}).strict().parse(await json(c.req.raw));
    const time=Date.now();
    store.transaction(()=>{
      store.db.query('DELETE FROM login_attempts WHERE at<?').run(time-15*60*1000);
      const {count}=store.db.query('SELECT COUNT(*) AS count FROM login_attempts').get() as {count:number};
      if(count>=10) throw new ApiError(429,'Too many login attempts; wait 15 minutes');
      store.db.query('INSERT INTO login_attempts(at) VALUES(?)').run(time);
    });
    const credentialHash=store.meta('credentialHash');
    if(!credentialHash||!Bun.password.verifySync(body.key,credentialHash)) throw new ApiError(401,'Invalid access key');
    const token=id()+id();
    store.transaction(()=>{
      if(store.meta('credentialHash')!==credentialHash)throw new ApiError(409,'Access key changed during login');
      store.db.query('DELETE FROM sessions WHERE expires<=?').run(time);
      const previous=tokenHash(c.req.raw);if(previous)store.db.query('DELETE FROM sessions WHERE hash=?').run(previous);
      store.db.query('INSERT INTO sessions(hash,expires) VALUES(?,?)').run(hash(token),time+sessionMs);
      store.db.exec('DELETE FROM sessions WHERE hash NOT IN (SELECT hash FROM sessions ORDER BY expires DESC LIMIT 32)');
    });
    setCookie(c,sessionName,token,{httpOnly:true,secure:true,sameSite:'Strict',path:'/',maxAge:sessionMs/1000});
    if(c.req.header('X-Sew-Session-Transport')==='header') c.header('X-Sew-Session',token);
    return c.body(null,204);
  });
  api.post('/auth/logout',c=>{
    const token=tokenHash(c.req.raw);if(token)store.db.query('DELETE FROM sessions WHERE hash=?').run(token);
    deleteCookie(c,sessionName,{path:'/',httpOnly:true,secure:true,sameSite:'Strict'});
    return c.body(null,204);
  });
  const readProfile = (): BodyProfile => JSON.parse(store.meta('bodyProfile') ?? '{"version":0,"body":null}');
  api.get('/body-profile', c => c.json(readProfile()));
  api.put('/body-profile', async c => {
    const body = z.object({ expectedVersion: z.number().int().min(0), body: DocumentSchema.shape.body.nullable() }).strict().parse(await json(c.req.raw));
    const profile = store.transaction(() => {
      const current = readProfile();
      if (current.version !== body.expectedVersion) throw new ApiError(409, 'Your saved measurements changed in another tab. Reload the profile before trying again.');
      const next: BodyProfile = { version: current.version + 1, body: body.body };
      if (body.body === null) store.deleteBodyProfile(next.version);
      else store.setMeta('bodyProfile', JSON.stringify(next));
      return next;
    });
    return c.json(profile);
  });
  api.get('/projects',c=>{
    const projects=(store.db.query('SELECT json FROM projects WHERE deleted=0 ORDER BY rowid DESC').all() as {json:string}[]).map(r=>JSON.parse(r.json) as Project);
    return c.json({projects});
  });
  api.post('/projects',async c=>{
    const body=newProjectSchema.parse(await json(c.req.raw));
    const state=store.transaction(()=>{
      const count=store.db.query('SELECT COUNT(*) AS count FROM projects WHERE deleted=0').get() as {count:number};
      if(count.count>=1000)throw new ApiError(429,'Project limit reached');
      const date=now(), project:Project={id:id(),title:body.title,createdAt:date,updatedAt:date,headRevisionId:null};
      const draft:Draft={projectId:project.id,version:1,baseRevisionId:null,document:emptyDocument(body.title,body.brief),updatedAt:date};
      store.db.query('INSERT INTO projects(id,json,draft) VALUES(?,?,?)').run(project.id,JSON.stringify(project),JSON.stringify(draft));
      return store.state(project.id);
    });return c.json(state,201);
  });
  api.get('/projects/:id',c=>c.json(store.transaction(()=>store.state(Id.parse(c.req.param('id'))))));
  api.put('/projects/:id/draft',async c=>{
    const projectId=Id.parse(c.req.param('id')),body=draftSchema.parse(await json(c.req.raw)),validated=document(body.document);
    const draft=store.transaction(()=>{
      const project=store.project(projectId),current=store.checkIdentity(project,body.expectedVersion,body.expectedRevisionId);
      store.checkReferences(projectId,validated);
      const date=now(),next:Draft={...current,document:validated,version:current.version+1,updatedAt:date};
      store.db.query('UPDATE projects SET draft=?,json=? WHERE id=?').run(JSON.stringify(next),JSON.stringify({...JSON.parse(project.json),title:validated.title,updatedAt:date}),projectId);
      return next;
    });return c.json(draft);
  });
  api.post('/projects/:id/revisions',async c=>{
    const projectId=Id.parse(c.req.param('id')),body=identitySchema.parse(await json(c.req.raw));
    const state=store.transaction(()=>{
      const row=store.project(projectId),draft=store.checkIdentity(row,body.expectedVersion,body.expectedRevisionId),project:Project=JSON.parse(row.json);
      const doc=document(draft.document);store.checkReferences(projectId,doc);
      const digest=objectDigest(doc),prior=project.headRevisionId?store.revision(projectId,project.headRevisionId):null;
      if(prior?.digest===digest)throw new ApiError(409,'Draft has no changes to publish',{latest:draft});
      const date=now(),revision:Revision={id:id(),projectId,number:(prior?.number??0)+1,parentRevisionId:prior?.id??null,document:doc,digest,createdAt:date};
      const next:Draft={...draft,baseRevisionId:revision.id,document:structuredClone(doc),version:draft.version+1,updatedAt:date};
      store.db.query('INSERT INTO revisions(id,project_id,number,json) VALUES(?,?,?,?)').run(revision.id,projectId,revision.number,JSON.stringify(revision));
      store.db.query('UPDATE projects SET json=?,draft=?,job_generation=job_generation+1 WHERE id=?').run(JSON.stringify({...project,title:doc.title,headRevisionId:revision.id,updatedAt:date}),JSON.stringify(next),projectId);
      queue.staleProject(projectId);
      return store.state(projectId);
    });queue.fenceProject(projectId);return c.json(state,201);
  });
  api.get('/interpretation/status',c=>c.json(interpretation.status()));
  api.get('/projects/:id/proposals/latest',c=>c.json(interpretation.latest(Id.parse(c.req.param('id')))));
  api.get('/projects/:id/interpretations/latest',c=>c.json(interpretations.latest(Id.parse(c.req.param('id')))));
  api.get('/projects/:id/interpretations/:jobId',c=>c.json(interpretations.get(Id.parse(c.req.param('id')),Id.parse(c.req.param('jobId')))));
  api.post('/projects/:id/interpretations/:jobId/cancel',c=>c.json(interpretations.cancel(Id.parse(c.req.param('id')),Id.parse(c.req.param('jobId')))));
  api.post('/projects/:id/proposals',async c=>{
    const body=identitySchema.extend({requestId:Id,includeReferences:z.boolean(),consent:z.literal(true)}).parse(await json(c.req.raw));
    return c.json(interpretations.submit(Id.parse(c.req.param('id')),body),202);
  });
  api.post('/projects/:id/proposals/:proposalId/accept',async c=>{
    const body=identitySchema.parse(await json(c.req.raw)),projectId=Id.parse(c.req.param('id'));
    const state=interpretation.accept(projectId,Id.parse(c.req.param('proposalId')),body);
    queue.staleProject(projectId);queue.fenceProject(projectId);
    return c.json(state,201);
  });
  api.delete('/projects/:id',c=>{const projectId=Id.parse(c.req.param('id'));interpretation.cancel(projectId);store.deleteProject(projectId);queue.fenceProject(projectId);return c.body(null,204);});
  api.post('/projects/:id/jobs',async c=>{
    const body=z.object({revisionId:Id,requestId:Id}).strict().parse(await json(c.req.raw));
    return c.json(queue.enqueue(Id.parse(c.req.param('id')),body.revisionId,body.requestId),202);
  });
  api.get('/projects/:id/jobs/:jobId',c=>c.json(JSON.parse(store.job(Id.parse(c.req.param('id')),Id.parse(c.req.param('jobId'))).json)));
  api.post('/projects/:id/jobs/:jobId/cancel',c=>c.json(queue.cancel(Id.parse(c.req.param('id')),Id.parse(c.req.param('jobId')))));
  api.get('/projects/:id/artifacts/:artifactId',c=>{
    const projectId=Id.parse(c.req.param('id'));store.project(projectId);
    const row=store.db.query('SELECT * FROM artifacts WHERE id=? AND project_id=?').get(Id.parse(c.req.param('artifactId')),projectId) as ArtifactRow|null;
    if(!row)throw new ApiError(404,'Artifact not found');
    const artifact:Artifact=JSON.parse(row.json);
    return fileResponse(store.readBlob(row.storage_key,artifact.digest,artifact.bytes),artifact.mime,artifact.mime==='application/pdf'?artifact.filename:undefined);
  });
  api.get('/projects/:id/geometry/:revisionId',c=>c.json(committedPatterns(store,Id.parse(c.req.param('id')),Id.parse(c.req.param('revisionId'))).geometry));
  api.post('/projects/:id/three-d',async c=>{
    const body=z.object({revisionId:Id,requestId:Id}).strict().parse(await json(c.req.raw));
    return c.json(inspections.submit(Id.parse(c.req.param('id')),body.revisionId,body.requestId),202);
  });
  api.get('/projects/:id/three-d/latest',c=>c.json(inspections.latest(Id.parse(c.req.param('id')),c.req.query('revisionId')?Id.parse(c.req.query('revisionId')):undefined)));
  api.get('/projects/:id/three-d/:jobId',c=>c.json(inspections.get(Id.parse(c.req.param('id')),Id.parse(c.req.param('jobId')))));
  api.post('/projects/:id/three-d/:jobId/cancel',c=>c.json(inspections.cancel(Id.parse(c.req.param('id')),Id.parse(c.req.param('jobId')))));
  for(const [route,filename] of [['report','inspection.json'],['mesh','inspection.glb'],['shape','garment-preview.json']] as const)api.get(`/projects/:id/three-d/:jobId/${route}`,c=>{
    const artifact=inspections.artifact(Id.parse(c.req.param('id')),Id.parse(c.req.param('jobId')),filename);
    return fileResponse(artifact.bytes,artifact.mime);
  });
  api.post('/projects/:id/references',async c=>{
    const projectId=Id.parse(c.req.param('id')),generation=store.project(projectId).generation;
    const label=(c.req.header('X-Filename')??'reference').slice(0,300).split(/[\\/]/).at(-1)!.replace(/[^\p{L}\p{N} ._-]/gu,'_').slice(0,160)||'reference';
    if(decoding>=2)throw new ApiError(429,'Image decoder is busy');
    decoding++;
    try {
      const bytes=await readBounded(c.req.raw,10*1024*1024),buffer=Buffer.from(bytes);
      if(!(buffer.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10])) || buffer[0]===255&&buffer[1]===216&&buffer[2]===255 || buffer.toString('ascii',0,4)==='RIFF'&&buffer.toString('ascii',8,12)==='WEBP')) throw new ApiError(415,'Only PNG, JPEG and WebP references are supported');
      let image:Buffer,info:{width:number;height:number};
      try {
        const decoder=sharp(buffer,{limitInputPixels:20000000,failOn:'warning'}),metadata=await decoder.metadata();
        if(!['png','jpeg','webp'].includes(metadata.format??'') || !metadata.width || !metadata.height || metadata.width*metadata.height>20000000 || (metadata.pages??1)>1)throw new Error();
        const rendered=await decoder.rotate().toColourspace('srgb').png().timeout({seconds:10}).toBuffer({resolveWithObject:true});
        image=rendered.data;info=rendered.info;
      }catch{throw new ApiError(422,'Invalid, animated or oversized reference image');}
      if(image.length>32*1024*1024)throw new ApiError(413,'Sanitized image exceeds the storage limit');
      const result=store.install([{filename:'reference.png',mime:'image/png',bytes:image}],records=>{
        const project=store.project(projectId);if(project.generation!==generation)throw new ApiError(409,'Upload project was fenced');
        const count=store.db.query('SELECT COUNT(*) AS count,SUM(bytes) AS total FROM reference_assets WHERE project_id=?').get(projectId) as {count:number;total:number|null};
        if(count.count>=100||(count.total??0)+image.length>512*1024*1024)throw new ApiError(429,'Project reference storage limit reached');
        const assetId=id(),file=records[0]!;
        store.db.query('INSERT INTO reference_assets(id,project_id,storage_key,digest,bytes,width,height,label,created_at) VALUES(?,?,?,?,?,?,?,?,?)').run(assetId,projectId,file.storageKey,file.digest,file.bytes,info.width,info.height,label,now());
        return {assetId,mime:'image/png',width:info.width,height:info.height};
      });return c.json(result,201);
    }finally{decoding--;}
  });
  api.get('/projects/:id/references/:assetId',c=>{
    const projectId=Id.parse(c.req.param('id'));store.project(projectId);
    let ref;try{ref=store.reference(projectId,Id.parse(c.req.param('assetId')));}catch{throw new ApiError(404,'Reference not found');}
    return fileResponse(store.readBlob(ref.storage_key,ref.digest,ref.bytes),'image/png');
  });
  api.post('/projects/:id/comments',async c=>{
    const projectId=Id.parse(c.req.param('id')),body=z.object({revisionId:Id,anchor:z.string().max(300),text:z.string().min(1).max(8000),reportedReviewer:z.string().max(300)}).strict().parse(await json(c.req.raw));
    const comment=store.transaction(()=>{
      store.revision(projectId,body.revisionId);
      const count=store.db.query('SELECT COUNT(*) AS count FROM comments WHERE project_id=? AND revision_id=?').get(projectId,body.revisionId) as {count:number};
      if(count.count>=500)throw new ApiError(429,'Revision comment limit reached');
      const value:ReviewComment={id:id(),projectId,...body,recordedBy:'owner',createdAt:now()};
      store.db.query('INSERT INTO comments(id,project_id,revision_id,json) VALUES(?,?,?,?)').run(value.id,projectId,body.revisionId,JSON.stringify(value));return value;
    });return c.json(comment,201);
  });
  api.post('/projects/:id/exports',async c=>{const body=exportSchema.parse(await json(c.req.raw));return c.json(await handoff.build(Id.parse(c.req.param('id')),body.revisionId,body.disclosure),201);});
  api.get('/projects/:id/exports/:snapshotId/:filename',c=>{
    const projectId=Id.parse(c.req.param('id')),snapshotId=Id.parse(c.req.param('snapshotId')),filename=filenameSchema.parse(c.req.param('filename'));
    store.snapshot(projectId,snapshotId);
    const file=store.db.query('SELECT * FROM export_files WHERE snapshot_id=? AND filename=?').get(snapshotId,filename) as {storage_key:string;digest:string;bytes:number;mime:string}|null;
    if(!file)throw new ApiError(404,'Export file not found');
    return fileResponse(store.readBlob(file.storage_key,file.digest,file.bytes),file.mime,filename);
  });
  api.post('/projects/:id/imports/preview',async c=>{const body=z.object({manifest:z.unknown()}).strict().parse(await json(c.req.raw));return c.json(previewImport(store,Id.parse(c.req.param('id')),body.manifest),201);});
  api.post('/projects/:id/imports/:previewId/accept',async c=>{const body=importAcceptSchema.parse(await json(c.req.raw));return c.json(acceptImport(store,Id.parse(c.req.param('id')),Id.parse(c.req.param('previewId')),body));});
  api.notFound(c=>c.json({error:'Route not found'},404));
  api.close=()=>{if(closed)return;closed=true;inspections.close();interpretations.close();interpretation.close();queue.close();store.close();};
  return api;
}
