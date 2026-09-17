import { mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import type { GarmentDocument } from '../../packages/contracts';
import type { ThreeDJob } from '../../packages/contracts/assembly';
import { committedPatterns } from './exports';
import { validateInspection } from './inspection-validation';
import { Store, type ArtifactRow, type ProjectRow } from './store';
import { ApiError, id, now, objectDigest } from './validation';

export type InspectionEngine = ((input:{pattern:Uint8Array;construction:NonNullable<GarmentDocument['garment']['design']>;outputDir:string;signal:AbortSignal})=>Promise<{report:Uint8Array;mesh:Uint8Array}>) & {version?:string};
type Input={revisionDigest:string;patternId:string;patternDigest:string;patternJobId:string;construction:NonNullable<GarmentDocument['garment']['design']>;constructionDigest:string;projectGeneration:number;engineVersion:string};
type Row={id:string;project_id:string;revision_id:string;request_id:string;input:string;input_digest:string;json:string;status:ThreeDJob['status'];lease:string|null;deadline:number|null;attempts:number;generation:number;cancel_requested:number};
const attemptMs=95000;
const pending=(status:ThreeDJob['status'])=>status==='queued'||status==='running';

export class ThreeDQueue {
  private active:{row:Row;controller:AbortController}|null=null;
  private timer:ReturnType<typeof setInterval>;
  private closed=false;
  constructor(private store:Store,private engine?:InspectionEngine) {
    this.timer=setInterval(()=>this.poke(),1000);this.timer.unref();this.poke();
  }
  submit(projectId:string,revisionId:string,requestId:string):ThreeDJob {
    const job=this.store.transaction(()=>{
      const project=this.store.project(projectId);
      const previous=this.store.db.query('SELECT * FROM three_d_jobs WHERE project_id=? AND request_id=?').get(projectId,requestId) as Row|null;
      if(previous) {
        if(previous.revision_id!==revisionId)throw new ApiError(409,'Request ID already belongs to another 3D revision');
        return JSON.parse(previous.json) as ThreeDJob;
      }
      if(!this.engine)throw new ApiError(503,'3D inspection engine is unavailable');
      const revision=this.store.revision(projectId,revisionId);
      if(JSON.parse(project.json).headRevisionId!==revisionId)throw new ApiError(409,'Inspect only the current saved revision');
      if(objectDigest(revision.document)!==revision.digest)throw new ApiError(409,'Revision digest mismatch');
      const construction=revision.document.garment.design;
      if(!construction||revision.document.garment.family!=='shirt')throw new ApiError(422,'3D inspection requires generated component-shirt patterns');
      const source=committedPatterns(this.store,projectId,revisionId).assets.find(asset=>asset.artifact.kind==='pattern-json')!;
      if(source.bytes.length>4*1024*1024)throw new ApiError(422,'3D source exceeds input budget');
      const count=this.store.db.query("SELECT COUNT(*) AS count FROM three_d_jobs WHERE status IN ('queued','running')").get() as {count:number};
      if(count.count>=8)throw new ApiError(429,'3D inspection queue is full');
      if(this.store.db.query("SELECT 1 FROM three_d_jobs WHERE project_id=? AND status IN ('queued','running')").get(projectId))throw new ApiError(409,'A 3D inspection is already queued or running');
      const input:Input={revisionDigest:revision.digest,patternId:source.artifact.id,patternDigest:source.artifact.digest,patternJobId:source.artifact.jobId!,construction,constructionDigest:objectDigest(construction),projectGeneration:project.generation,engineVersion:this.engine.version??'injected-inspection/1'};
      const date=now(),value:ThreeDJob={id:id(),projectId,revisionId,requestId,inputDigest:objectDigest(input),patternDigest:input.patternDigest,status:'queued',error:null,createdAt:date,updatedAt:date};
      this.store.db.query('INSERT INTO three_d_jobs(id,project_id,revision_id,request_id,input_digest,input,json,status) VALUES(?,?,?,?,?,?,?,?)').run(value.id,projectId,revisionId,requestId,value.inputDigest,JSON.stringify(input),JSON.stringify(value),'queued');
      return value;
    });
    this.poke();return job;
  }
  get(projectId:string,jobId:string):ThreeDJob {
    this.store.project(projectId);
    const row=this.store.db.query('SELECT json FROM three_d_jobs WHERE project_id=? AND id=?').get(projectId,jobId) as {json:string}|null;
    if(!row)throw new ApiError(404,'3D inspection not found');
    return JSON.parse(row.json);
  }
  latest(projectId:string,revisionId?:string):ThreeDJob|null {
    this.store.project(projectId);
    if(revisionId)this.store.revision(projectId,revisionId);
    const row=(revisionId?this.store.db.query('SELECT * FROM three_d_jobs WHERE project_id=? AND revision_id=? ORDER BY rowid DESC LIMIT 1').get(projectId,revisionId):this.store.db.query('SELECT * FROM three_d_jobs WHERE project_id=? ORDER BY rowid DESC LIMIT 1').get(projectId)) as Row|null;
    return row?{...JSON.parse(row.json),sourceCurrent:this.inputsCurrent(row)}:null;
  }
  artifact(projectId:string,jobId:string,filename:'inspection.json'|'inspection.glb') {
    const job=this.get(projectId,jobId);
    if(job.status!=='succeeded')throw new ApiError(409,'3D inspection has no committed result');
    const row=this.store.db.query('SELECT * FROM three_d_artifacts WHERE job_id=? AND project_id=? AND filename=?').get(jobId,projectId,filename) as {storage_key:string;digest:string;bytes:number;mime:string}|null;
    if(!row)throw new ApiError(404,'3D artifact not found');
    return {bytes:this.store.readBlob(row.storage_key,row.digest,row.bytes),mime:row.mime};
  }
  cancel(projectId:string,jobId:string):ThreeDJob {
    const result=this.store.transaction(()=>{
      const job=this.get(projectId,jobId);
      if(!pending(job.status))return job;
      this.store.db.query('UPDATE three_d_jobs SET cancel_requested=1 WHERE id=?').run(jobId);
      return this.finish({id:jobId,json:JSON.stringify(job)},'cancelled','Cancelled by owner');
    });
    if(this.active?.row.id===jobId)this.active.controller.abort();
    return result;
  }
  private finish(row:{id:string;json:string},status:ThreeDJob['status'],error:string|null,result?:ThreeDJob['result']):ThreeDJob {
    const job:ThreeDJob={...JSON.parse(row.json),status,error,updatedAt:now(),...(result?{result}:{})};
    this.store.db.query('UPDATE three_d_jobs SET status=?,json=?,lease=NULL,deadline=NULL WHERE id=?').run(status,JSON.stringify(job),row.id);
    return job;
  }
  private inputsCurrent(row:Row):boolean {
    try {
    const input=JSON.parse(row.input) as Input;
    const project=this.store.db.query('SELECT * FROM projects WHERE id=?').get(row.project_id) as ProjectRow|null;
    const source=this.store.db.query('SELECT job_id FROM geometry_heads WHERE revision_id=?').get(row.revision_id) as {job_id:string}|null;
    return !!project&&!project.deleted&&project.generation===input.projectGeneration&&JSON.parse(project.json).headRevisionId===row.revision_id&&source?.job_id===input.patternJobId&&objectDigest(input)===row.input_digest&&input.engineVersion===(this.engine?.version??'injected-inspection/1');
    } catch {return false;}
  }
  private valid(row:Row):boolean {
    if(this.closed||this.store.closed)return false;
    const current=this.store.db.query('SELECT * FROM three_d_jobs WHERE id=?').get(row.id) as Row|null;
    return !!current&&current.status==='running'&&!current.cancel_requested&&current.lease===row.lease&&current.generation===row.generation&&current.deadline!>Date.now()&&this.inputsCurrent(current);
  }
  poke() {
    if(this.closed||this.store.closed)return;
    const row=this.store.transaction(()=>{
      const outstanding=this.store.db.query("SELECT * FROM three_d_jobs WHERE status IN ('queued','running')").all() as Row[];
      for(const previous of outstanding) {
        if(!this.inputsCurrent(previous))this.finish(previous,'stale','Pattern, revision or engine inputs changed; create a new inspection');
        else if(previous.status==='running'&&previous.deadline!<=Date.now())this.finish(previous,previous.attempts<2?'queued':'failed','Interrupted 3D attempt');
      }
      if(this.active&&!this.valid(this.active.row))this.active.controller.abort();
      if(this.active||!this.engine||this.store.db.query("SELECT 1 FROM three_d_jobs WHERE status='running'").get())return null;
      const candidate=this.store.db.query("SELECT * FROM three_d_jobs WHERE status='queued' ORDER BY rowid LIMIT 1").get() as Row|null;
      if(!candidate)return null;
      const job:ThreeDJob={...JSON.parse(candidate.json),status:'running',error:null,updatedAt:now()};
      this.store.db.query("UPDATE three_d_jobs SET status='running',json=?,lease=?,deadline=?,attempts=attempts+1,generation=generation+1 WHERE id=? AND status='queued'").run(JSON.stringify(job),id(),Date.now()+attemptMs,candidate.id);
      return this.store.db.query('SELECT * FROM three_d_jobs WHERE id=?').get(candidate.id) as Row;
    });
    if(row) {
      const controller=new AbortController();this.active={row,controller};void this.execute(row,controller);
    }
  }
  private async execute(row:Row,controller:AbortController) {
    const outputDir=join(this.store.root,'staging',`three-d-${row.id}-${row.generation}`);
    const timeout=setTimeout(()=>controller.abort(),attemptMs-1000);timeout.unref();
    try {
      mkdirSync(outputDir,{mode:0o700});
      const input=JSON.parse(row.input) as Input;
      const revision=this.store.revision(row.project_id,row.revision_id);
      if(objectDigest(revision.document)!==input.revisionDigest||objectDigest(revision.document.garment.design)!==input.constructionDigest)throw new ApiError(409,'3D revision integrity failure');
      const source=this.store.db.query('SELECT * FROM artifacts WHERE id=? AND project_id=? AND revision_id=? AND job_id=?').get(input.patternId,row.project_id,row.revision_id,input.patternJobId) as ArtifactRow|null;
      if(!source)throw new ApiError(409,'3D source pattern missing');
      const metadata=JSON.parse(source.json);
      const pattern=this.store.readBlob(source.storage_key,input.patternDigest,metadata.bytes);
      const output=await this.engine!({pattern,construction:structuredClone(input.construction),outputDir,signal:controller.signal});
      controller.signal.throwIfAborted();
      if(!this.valid(row))return;
      if(!(output.report instanceof Uint8Array)||!(output.mesh instanceof Uint8Array)||output.report.length>16*1024*1024||output.mesh.length>16*1024*1024)throw new ApiError(422,'3D artifact budget exceeded');
      const report=Uint8Array.from(output.report),mesh=Uint8Array.from(output.mesh);
      const inspection=validateInspection(report,mesh,JSON.parse(new TextDecoder().decode(pattern)),input.patternDigest,input.constructionDigest);
      this.store.install([{filename:'inspection.json',mime:'application/json',bytes:report},{filename:'inspection.glb',mime:'model/gltf-binary',bytes:mesh}],records=>{
        if(!this.valid(row)||controller.signal.aborted)throw new ApiError(409,'3D attempt was fenced');
        for(const record of records)this.store.db.query('INSERT INTO three_d_artifacts(job_id,project_id,filename,storage_key,digest,bytes,mime) VALUES(?,?,?,?,?,?,?)').run(row.id,row.project_id,record.filename,record.storageKey,record.digest,record.bytes,record.mime);
        this.finish(row,'succeeded',null,{classification:'placement-inspection',fabricInstances:inspection.instances.length,unresolvedPhysicalRoles:inspection.unresolvedPhysicalRoles.length,capabilityGaps:inspection.capabilityGaps});
      });
    } catch(error) {
      if(this.valid(row))this.finish(row,'failed',error instanceof ApiError?error.message:'3D inspection failed validation or exceeded its resource budget; 2D patterns remain available');
    } finally {
      clearTimeout(timeout);
      rmSync(outputDir,{recursive:true,force:true});
      if(this.active?.row.lease===row.lease)this.active=null;
      if(!this.closed)queueMicrotask(()=>this.poke());
    }
  }
  close() {
    if(this.closed)return;
    if(this.active&&this.valid(this.active.row))this.finish(this.active.row,this.active.row.attempts<2?'queued':'failed','Interrupted 3D attempt');
    this.closed=true;clearInterval(this.timer);this.active?.controller.abort();
  }
}
