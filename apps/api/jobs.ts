import { mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import type { Artifact, GarmentDocument, Job, PatternGeometry, Project } from '../../packages/contracts';
import { canonical } from '../../packages/contracts';
import { Store, type JobRow, type ProjectRow } from './store';
import { ApiError, checkFile, geometry, id, now } from './validation';
import { EngineInputError } from '../../services/engine/runner';

export type Engine = (input:{document:GarmentDocument;inputDigest:string;outputDir:string;signal:AbortSignal})=>Promise<{geometry:PatternGeometry;files:{filename:string;bytes:Uint8Array;mime:string;kind:Artifact['kind']}[]}>;
const heartbeatMs = 2000;
const schedulerLeaseMs = 15000;
const attemptMs = 95000;
const terminal = new Set<Job['status']>(['succeeded','failed','cancelled','stale']);

export class JobQueue {
  private owner = id();
  private timer: ReturnType<typeof setInterval>;
  private active: {job:JobRow;controller:AbortController}|null = null;
  private closed = false;
  private pumping = false;
  private epoch = 0;
  constructor(private store:Store,private engine?:Engine) {
    this.timer = setInterval(()=>this.poke(),heartbeatMs);
    this.timer.unref();
    this.poke();
  }
  enqueue(projectId:string,revisionId:string,requestId:string): Job {
    const job = this.store.transaction(()=>{
      const project = this.store.project(projectId);
      const existing = this.store.db.query('SELECT * FROM jobs WHERE project_id=? AND request_id=?').get(projectId,requestId) as JobRow|null;
      if (existing) {
        if (existing.revision_id!==revisionId) throw new ApiError(409,'Request ID already belongs to another revision');
        return JSON.parse(existing.json) as Job;
      }
      this.store.revision(projectId,revisionId);
      if ((JSON.parse(project.json) as Project).headRevisionId!==revisionId) throw new ApiError(409,'Generate only from the current saved revision');
      if (!this.engine) throw new ApiError(503,'Geometry engine is unavailable');
      const count = this.store.db.query("SELECT COUNT(*) AS count FROM jobs WHERE status IN ('queued','running')").get() as {count:number};
      if (count.count>=32) throw new ApiError(429,'Geometry queue is full');
      const previous = this.store.db.query("SELECT * FROM jobs WHERE project_id=? AND status IN ('queued','running')").all(projectId) as JobRow[];
      for (const row of previous) this.store.jobStatus(row,'stale','Superseded by a newer generation request');
      this.store.db.query('UPDATE projects SET job_generation=job_generation+1 WHERE id=?').run(projectId);
      const date=now();
      const value:Job={id:id(),projectId,revisionId,requestId,status:'queued',error:null,createdAt:date,updatedAt:date};
      this.store.db.query('INSERT INTO jobs(id,project_id,revision_id,request_id,json,status,project_generation,job_generation) VALUES(?,?,?,?,?,?,?,?)').run(value.id,projectId,revisionId,requestId,JSON.stringify(value),'queued',project.generation,project.job_generation+1);
      return value;
    });
    this.abortFenced(); this.poke();
    return job;
  }
  cancel(projectId:string,jobId:string): Job {
    const value = this.store.transaction(()=>{
      const row = this.store.job(projectId,jobId);
      if (terminal.has(row.status)) return JSON.parse(row.json) as Job;
      this.store.db.query('UPDATE jobs SET cancel_requested=1 WHERE id=?').run(jobId);
      return this.store.jobStatus(row,'cancelled','Cancelled by owner');
    });
    if (this.active?.job.id===jobId) this.active.controller.abort();
    this.poke(); return value;
  }
  fenceProject(projectId:string) {
    if (this.active?.job.project_id===projectId) this.active.controller.abort();
    this.poke();
  }
  staleProject(projectId:string) {
    const rows = this.store.db.query("SELECT * FROM jobs WHERE project_id=? AND status IN ('queued','running')").all(projectId) as JobRow[];
    for (const row of rows) this.store.jobStatus(row,'stale','A newer revision was saved');
  }
  private abortFenced() {
    if (!this.active) return;
    const current = this.store.db.query('SELECT * FROM jobs WHERE id=?').get(this.active.job.id) as JobRow|null;
    if (!current || current.status!=='running' || current.lease!==this.active.job.lease) this.active.controller.abort();
  }
  poke() {
    if (this.closed || this.pumping) return;
    this.pumping=true;
    try {
      const claimed = this.store.transaction(()=>{
        const time=Date.now();
        let reacquired = false;
        const lease=this.store.db.query('SELECT * FROM scheduler WHERE id=1').get() as {owner:string|null;generation:number;deadline:number};
        if (lease.owner===this.owner && lease.deadline>time) {
          this.store.db.query('UPDATE scheduler SET deadline=? WHERE id=1 AND owner=?').run(time+schedulerLeaseMs,this.owner);
          this.epoch=lease.generation;
        } else if (lease.deadline<=time || lease.owner===null) {
          this.store.db.query('UPDATE scheduler SET owner=?,generation=generation+1,deadline=? WHERE id=1').run(this.owner,time+schedulerLeaseMs);
          this.epoch=lease.generation+1;
          reacquired = true;
        } else return null;
        const running = this.store.db.query("SELECT * FROM jobs WHERE status='running'").all() as JobRow[];
        for (const row of running) {
          if (reacquired || row.scheduler!==this.owner || !row.lease_deadline || row.lease_deadline<=time) {
            if (row.cancel_requested) this.store.jobStatus(row,'cancelled','Cancellation recovered');
            else this.store.jobStatus(row,row.attempts<2?'queued':'failed',row.attempts<2?'Interrupted attempt; bounded retry queued':'Attempt lease expired');
          }
        }
        this.abortFenced();
        if (this.active || !this.engine) return null;
        const rows = this.store.db.query("SELECT * FROM jobs WHERE status='queued' ORDER BY rowid LIMIT 32").all() as JobRow[];
        for (const row of rows) {
          const project = this.store.db.query('SELECT * FROM projects WHERE id=?').get(row.project_id) as ProjectRow|null;
          if (!project || project.deleted || project.generation!==row.project_generation || project.job_generation!==row.job_generation || JSON.parse(project.json).headRevisionId!==row.revision_id) { this.store.jobStatus(row,'stale','Inputs superseded or project deleted'); continue; }
          const job:Job = {...JSON.parse(row.json),status:'running',error:null,updatedAt:now()};
          const token=id();
          this.store.db.query("UPDATE jobs SET status='running',json=?,lease=?,lease_deadline=?,scheduler=?,attempts=attempts+1,generation=generation+1 WHERE id=? AND status='queued'").run(JSON.stringify(job),token,time+attemptMs,this.owner,row.id);
          return this.store.db.query('SELECT * FROM jobs WHERE id=?').get(row.id) as JobRow;
        }
        return null;
      });
      if (claimed) {
        const controller=new AbortController();
        this.active={job:claimed,controller};
        void this.execute(claimed,controller,this.epoch);
      }
    } catch { this.active?.controller.abort(); }
    finally { this.pumping=false; }
  }
  private valid(job:JobRow,epoch:number): boolean {
    if (this.closed) return false;
    const current=this.store.db.query('SELECT * FROM jobs WHERE id=?').get(job.id) as JobRow|null;
    const project=this.store.db.query('SELECT * FROM projects WHERE id=?').get(job.project_id) as ProjectRow|null;
    const lease=this.store.db.query('SELECT * FROM scheduler WHERE id=1').get() as {owner:string|null;generation:number;deadline:number};
    return !!(current && project && !project.deleted && current.status==='running' && !current.cancel_requested && current.lease===job.lease && current.generation===job.generation && current.lease_deadline!>Date.now() && project.generation===job.project_generation && project.job_generation===job.job_generation && JSON.parse(project.json).headRevisionId===job.revision_id && lease.owner===this.owner && lease.generation===epoch && lease.deadline>Date.now());
  }
  private async execute(job:JobRow,controller:AbortController,epoch:number) {
    const outputDir=join(this.store.root,'staging',`${job.id}-${job.generation}`);
    let timeout: ReturnType<typeof setTimeout>|undefined;
    let onAbort: (()=>void)|undefined;
    let engineTask: ReturnType<Engine>|undefined;
    try {
      mkdirSync(outputDir,{recursive:false,mode:0o700});
      const revision=this.store.revision(job.project_id,job.revision_id);
      const abort=new Promise<never>((_,reject)=>{
        onAbort=()=>reject(new Error('Attempt aborted'));
        controller.signal.addEventListener('abort',onAbort,{once:true});
        if (controller.signal.aborted) onAbort();
      });
      timeout=setTimeout(()=>controller.abort(),attemptMs);
      timeout.unref();
      engineTask=this.engine!({document:structuredClone(revision.document),inputDigest:revision.digest,outputDir,signal:controller.signal});
      const result=await Promise.race([engineTask,abort]);
      if (!this.valid(job,epoch)) return;
      const normalized=geometry(result.geometry,revision.digest);
      if (normalized.family!==revision.document.garment.family) throw new ApiError(422,'Geometry family mismatch');
      if (!Array.isArray(result.files) || result.files.length<3 || result.files.length>40 || result.files.reduce((n,f)=>n+f.bytes.length,0)>64*1024*1024) throw new ApiError(422,'Invalid engine file inventory');
      if (new Set(result.files.map(f=>f.filename)).size!==result.files.length) throw new ApiError(422,'Duplicate engine filenames');
      const kinds = new Set(result.files.map(f=>f.kind));
      for (const kind of ['pattern-json','pattern-svg','pattern-pdf']) if (!kinds.has(kind as Artifact['kind'])) throw new ApiError(422,'Engine omitted required pattern artifacts');
      let jsonCount=0;
      for (const file of result.files) {
        if (file.kind==='reference' || file.mime!==({'pattern-json':'application/json','pattern-svg':'image/svg+xml','pattern-pdf':'application/pdf'} as Record<string,string>)[file.kind]) throw new ApiError(422,'Invalid engine artifact kind');
        await checkFile(file.filename,file.bytes,file.mime);
        if (file.kind==='pattern-json') {
          jsonCount++;
          if (canonical(geometry(JSON.parse(new TextDecoder().decode(file.bytes)),revision.digest))!==canonical(normalized)) throw new ApiError(422,'Pattern JSON disagrees with geometry');
        }
      }
      if (jsonCount!==1) throw new ApiError(422,'Expected one normalized pattern JSON');
      this.store.install(result.files,records=>{
        if (!this.valid(job,epoch)) throw new ApiError(409,'Attempt lease was fenced');
        for (let i=0;i<records.length;i++) {
          const record=records[i]!, file=result.files[i]!;
          const artifact:Artifact={id:id(),projectId:job.project_id,revisionId:job.revision_id,jobId:job.id,kind:file.kind,filename:file.filename,mime:file.mime,digest:record.digest,bytes:record.bytes,classification:'printable-reference',createdAt:now()};
          this.store.db.query('INSERT INTO artifacts(id,project_id,revision_id,job_id,json,storage_key) VALUES(?,?,?,?,?,?)').run(artifact.id,job.project_id,job.revision_id,job.id,JSON.stringify(artifact),record.storageKey);
        }
        this.store.db.query('INSERT INTO geometry_heads(revision_id,job_id) VALUES(?,?) ON CONFLICT(revision_id) DO UPDATE SET job_id=excluded.job_id').run(job.revision_id,job.id);
        this.store.jobStatus(job,'succeeded');
      });
    } catch (error) {
      if (!this.closed) this.store.transaction(()=>{
        if (this.valid(job,epoch)) this.store.jobStatus(job,'failed',error instanceof ApiError || error instanceof EngineInputError ? error.message : controller.signal.aborted ? 'Engine attempt timed out or was interrupted' : 'Engine failed; inspect the supported inputs and private engine setup');
      });
    } finally {
      if (timeout) clearTimeout(timeout);
      if (onAbort) controller.signal.removeEventListener('abort',onAbort);
      if(controller.signal.aborted)await engineTask?.catch(()=>{});
      rmSync(outputDir,{recursive:true,force:true});
      if (this.active?.job.id===job.id) this.active=null;
      if (!this.closed) queueMicrotask(()=>this.poke());
    }
  }
  close() {
    if (this.closed) return;
    this.closed=true; clearInterval(this.timer);
    this.active?.controller.abort();
    this.store.transaction(()=>{
      const rows=this.store.db.query("SELECT * FROM jobs WHERE scheduler=? AND status='running'").all(this.owner) as JobRow[];
      for (const row of rows) this.store.jobStatus(row,row.attempts<2?'queued':'failed','API closed; interrupted attempt fenced');
      this.store.db.query('UPDATE scheduler SET owner=NULL,deadline=0 WHERE id=1 AND owner=?').run(this.owner);
    });
  }
}
