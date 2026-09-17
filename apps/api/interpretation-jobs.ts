import type { Draft } from '../../packages/contracts';
import type { InterpretationJob } from '../../packages/contracts/interpretation';
import { InterpretationService } from './interpretation';
import { Store } from './store';
import { ApiError, id, now, objectDigest } from './validation';

type Identity={requestId:string;expectedVersion:number;expectedRevisionId:string|null;includeReferences:boolean;consent:true};
type Input={identity:Identity;captured:{draft:Draft;generation:number;requestId:string};providerDigest:string};
type Row={id:string;project_id:string;input_digest:string;input:string;json:string;status:InterpretationJob['status'];lease:string|null;deadline:number|null;attempts:number};

export class InterpretationQueue {
  private active:{row:Row;controller:AbortController}|null=null;
  private closed=false;
  private timer:ReturnType<typeof setInterval>;
  constructor(private store:Store,private service:InterpretationService) {
    this.timer=setInterval(()=>this.poke(),1000);
    this.timer.unref();
    this.poke();
  }
  submit(projectId:string,identity:Identity):InterpretationJob {
    const job=this.store.transaction(()=>{
      const project=this.store.project(projectId);
      const prior=this.store.db.query('SELECT * FROM interpretation_jobs WHERE project_id=? AND request_id=?').get(projectId,identity.requestId) as Row|null;
      if(prior) {
        if(objectDigest((JSON.parse(prior.input) as Input).identity)!==objectDigest(identity))throw new ApiError(409,'Request ID already belongs to different interpretation inputs');
        return JSON.parse(prior.json) as InterpretationJob;
      }
      if(!this.service.status().available)throw new ApiError(503,'Design AI is not configured on this server');
      const draft=this.store.checkIdentity(project,identity.expectedVersion,identity.expectedRevisionId);
      if(!draft.document.brief.trim())throw new ApiError(422,'Describe your garment in The idea first');
      if(identity.includeReferences&&draft.document.views.length>3)throw new ApiError(422,'Use at most three references for a proposal');
      const pending=this.store.db.query("SELECT COUNT(*) AS count FROM interpretation_jobs WHERE status IN ('queued','running')").get() as {count:number};
      if(pending.count>=12)throw new ApiError(429,'Design queue is full');
      const sameProject=this.store.db.query("SELECT id FROM interpretation_jobs WHERE project_id=? AND status IN ('queued','running')").get(projectId);
      if(sameProject)throw new ApiError(409,'A design interpretation is already queued or running for this project');
      const since=Date.now()-3600000;
      this.store.db.query('DELETE FROM ai_requests WHERE at<?').run(since);
      const count=this.store.db.query('SELECT COUNT(*) AS count FROM ai_requests WHERE at>?').get(since) as {count:number};
      if(count.count>=12)throw new ApiError(429,'Hourly limit of 12 design requests reached');
      const date=now(),job:InterpretationJob={id:id(),projectId,requestId:identity.requestId,status:'queued',error:null,proposalId:null,createdAt:date,updatedAt:date};
      const input:Input={identity,captured:{draft,generation:project.generation,requestId:job.id},providerDigest:objectDigest(this.service.status())};
      this.store.db.query('INSERT INTO ai_requests(id,project_id,at) VALUES(?,?,?)').run(job.id,projectId,Date.now());
      this.store.db.query('INSERT INTO interpretation_jobs(id,project_id,request_id,input_digest,input,json,status) VALUES(?,?,?,?,?,?,?)').run(job.id,projectId,identity.requestId,objectDigest(input),JSON.stringify(input),JSON.stringify(job),'queued');
      return job;
    });
    this.poke();
    return job;
  }
  latest(projectId:string):InterpretationJob|null {
    this.store.project(projectId);
    const row=this.store.db.query('SELECT json FROM interpretation_jobs WHERE project_id=? ORDER BY rowid DESC LIMIT 1').get(projectId) as {json:string}|null;
    return row?JSON.parse(row.json):null;
  }
  get(projectId:string,jobId:string):InterpretationJob {
    this.store.project(projectId);
    const row=this.store.db.query('SELECT json FROM interpretation_jobs WHERE project_id=? AND id=?').get(projectId,jobId) as {json:string}|null;
    if(!row)throw new ApiError(404,'Interpretation job not found');
    return JSON.parse(row.json);
  }
  cancel(projectId:string,jobId:string):InterpretationJob {
    const result=this.store.transaction(()=>{
      const job=this.get(projectId,jobId);
      if(!['queued','running'].includes(job.status))return job;
      return this.finish({id:jobId,json:JSON.stringify(job)},'cancelled','Cancelled by owner');
    });
    if(this.active?.row.id===jobId)this.active.controller.abort();
    return result;
  }
  private finish(row:{id:string;json:string},status:InterpretationJob['status'],error:string|null):InterpretationJob {
    const job:InterpretationJob={...JSON.parse(row.json),status,error,proposalId:status==='succeeded'?row.id:null,updatedAt:now()};
    this.store.db.query('UPDATE interpretation_jobs SET status=?,json=?,lease=NULL,deadline=NULL WHERE id=?').run(status,JSON.stringify(job),row.id);
    return job;
  }
  poke() {
    if(this.closed||this.store.closed)return;
    if(this.active&&!this.valid(this.active.row))this.active.controller.abort();
    const row=this.store.transaction(()=>{
      const expired=this.store.db.query("SELECT * FROM interpretation_jobs WHERE status='running' AND deadline<=?").all(Date.now()) as Row[];
      for(const previous of expired)this.finish(previous,previous.attempts<2?'queued':'failed','Interrupted interpretation attempt');
      if(this.active||!this.service.status().available)return null;
      if(this.store.db.query("SELECT id FROM interpretation_jobs WHERE status='running'").get())return null;
      const candidate=this.store.db.query("SELECT * FROM interpretation_jobs WHERE status='queued' ORDER BY rowid LIMIT 1").get() as Row|null;
      if(!candidate)return null;
      const job:InterpretationJob={...JSON.parse(candidate.json),status:'running',error:null,updatedAt:now()};
      this.store.db.query('UPDATE interpretation_jobs SET status=?,json=?,lease=?,deadline=?,attempts=attempts+1 WHERE id=?').run('running',JSON.stringify(job),id(),Date.now()+125000,candidate.id);
      return this.store.db.query('SELECT * FROM interpretation_jobs WHERE id=?').get(candidate.id) as Row;
    });
    if(row) {
      const controller=new AbortController();this.active={row,controller};
      void this.execute(row,controller);
    }
  }
  private valid(row:Row):boolean {
    if(this.closed||this.store.closed)return false;
    const current=this.store.db.query('SELECT * FROM interpretation_jobs WHERE id=?').get(row.id) as Row|null;
    return !!current&&current.status==='running'&&current.lease===row.lease&&current.deadline!>Date.now();
  }
  private async execute(row:Row,controller:AbortController) {
    try {
      const input=JSON.parse(row.input) as Input;
      if(objectDigest(input)!==row.input_digest||input.providerDigest!==objectDigest(this.service.status()))throw new ApiError(409,'Interpretation configuration changed; submit a new request');
      await this.service.propose(row.project_id,input.identity,{captured:input.captured,signal:controller.signal,publish:()=>{
        if(!this.valid(row))throw new ApiError(409,'Interpretation attempt was fenced');
        this.finish(row,'succeeded',null);
      }});
    } catch(error) {
      if(this.valid(row))this.finish(row,'failed',error instanceof ApiError?error.message:'Interpretation failed; your draft is unchanged');
    } finally {
      if(this.active?.row.lease===row.lease)this.active=null;
      this.poke();
    }
  }
  close() {
    if(this.closed)return;
    if(this.active&&this.valid(this.active.row))this.finish(this.active.row,this.active.row.attempts<2?'queued':'failed','Interrupted interpretation attempt');
    this.closed=true;clearInterval(this.timer);this.active?.controller.abort();
  }
}
