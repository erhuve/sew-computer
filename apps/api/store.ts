import { Database } from 'bun:sqlite';
import { constants, closeSync, chmodSync, existsSync, fsyncSync, lstatSync, mkdirSync, openSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { randomBytes } from 'node:crypto';
import type { Artifact, Draft, ExportSnapshot, Job, Project, ProjectState, Revision, ReviewComment } from '../../packages/contracts';
import { ApiError, hash, id, now, objectDigest } from './validation';

export type ProjectRow = {id:string; json:string; draft:string; generation:number; job_generation:number; deleted:number};
export type JobRow = {id:string; project_id:string; revision_id:string; request_id:string; json:string; status:Job['status']; generation:number; project_generation:number; job_generation:number; attempts:number; lease:string|null; lease_deadline:number|null; scheduler:string|null; cancel_requested:number};
export type ArtifactRow = {id:string; project_id:string; revision_id:string; job_id:string|null; json:string; storage_key:string};
export type ReferenceRow = {id:string; project_id:string; storage_key:string; digest:string; bytes:number; width:number; height:number; label:string; created_at:string};
export type SnapshotRow = {id:string; project_id:string; revision_id:string; generation:number; json:string};
export type StoredFile = {filename:string;mime:string;bytes:Uint8Array};
export type FileRecord = {storageKey:string;digest:string;bytes:number;filename:string;mime:string};

export class Store {
  db: Database;
  root: string;
  closed = false;
  credentialHash: string;
  private journalSequence = 0;
  private journalDigest = '';

  constructor(dataDir: string, authKey?: string) {
    this.root = resolve(dataDir);
    if (existsSync(join(this.root,'RESTORE_PENDING'))) throw new Error('Restore has not been verified; access remains closed');
    this.directory(this.root);
    for (const name of ['blobs','staging','markers']) this.directory(join(this.root,name));
    const databasePath = join(this.root,'sew.sqlite');
    const existed = existsSync(databasePath);
    if (existed && (!lstatSync(databasePath).isFile() || lstatSync(databasePath).isSymbolicLink())) throw new Error('Unsafe database file');
    this.db = new Database(databasePath,{create:true,strict:true});
    chmodSync(databasePath,0o600);
    try {
      this.db.exec('PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000; PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA secure_delete=ON;');
      const version = (this.db.query('PRAGMA user_version').get() as {user_version:number}).user_version;
      if (![0,1,2,3,4,5].includes(version)) throw new Error('Unsupported database schema');
      if (version === 0) this.migrate();
      if(version<2)this.transaction(()=>{
        this.db.exec(`CREATE TABLE ai_requests(id TEXT PRIMARY KEY,project_id TEXT NOT NULL,at INTEGER NOT NULL);
          CREATE TABLE ai_proposals(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),json TEXT NOT NULL,source_digest TEXT NOT NULL,generation INTEGER NOT NULL,accepted INTEGER NOT NULL DEFAULT 0);
          PRAGMA user_version=2;`);
      });
      if(version<3)this.transaction(()=>this.db.exec('PRAGMA user_version=3;'));
      if(version<4)this.transaction(()=>this.db.exec(`CREATE TABLE IF NOT EXISTS interpretation_jobs(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),request_id TEXT NOT NULL,input_digest TEXT NOT NULL,input TEXT NOT NULL,json TEXT NOT NULL,status TEXT NOT NULL,lease TEXT,deadline INTEGER,attempts INTEGER NOT NULL DEFAULT 0,UNIQUE(project_id,request_id)); PRAGMA user_version=4;`));
      if(version<5)this.transaction(()=>this.db.exec(`
        CREATE TABLE IF NOT EXISTS three_d_jobs(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),revision_id TEXT NOT NULL REFERENCES revisions(id),request_id TEXT NOT NULL,input_digest TEXT NOT NULL,input TEXT NOT NULL,json TEXT NOT NULL,status TEXT NOT NULL,lease TEXT,deadline INTEGER,attempts INTEGER NOT NULL DEFAULT 0,generation INTEGER NOT NULL DEFAULT 0,cancel_requested INTEGER NOT NULL DEFAULT 0,UNIQUE(project_id,request_id));
        CREATE TABLE IF NOT EXISTS three_d_artifacts(job_id TEXT NOT NULL REFERENCES three_d_jobs(id),project_id TEXT NOT NULL REFERENCES projects(id),filename TEXT NOT NULL,storage_key TEXT NOT NULL UNIQUE,digest TEXT NOT NULL,bytes INTEGER NOT NULL,mime TEXT NOT NULL,PRIMARY KEY(job_id,filename));
        CREATE TRIGGER IF NOT EXISTS immutable_three_d_artifact BEFORE UPDATE ON three_d_artifacts BEGIN SELECT RAISE(ABORT,'immutable 3D artifact'); END;
        PRAGMA user_version=5;
      `));
      this.transaction(()=>this.replayDeletions(existed));
      let key = authKey;
      if (key === undefined) {
        const path = join(this.root,'access-key');
        if (!existsSync(path)) {
          if (this.meta('credentialHash')) throw new Error('Access key is missing; refusing credential reset');
          this.atomic(path,Buffer.from(randomBytes(32).toString('base64url')));
        }
        if (lstatSync(path).isSymbolicLink() || !lstatSync(path).isFile()) throw new Error('Unsafe access key');
        chmodSync(path,0o600);
        key = readFileSync(path,'utf8').trim();
      }
      if (!key || key.length>512) throw new Error('Invalid configured access key');
      const prior = this.meta('credentialHash');
      if (prior && Bun.password.verifySync(key,prior)) this.credentialHash = prior;
      else {
        this.credentialHash = Bun.password.hashSync(key,{algorithm:'argon2id',memoryCost:19456,timeCost:2});
        this.transaction(()=>{ this.setMeta('credentialHash',this.credentialHash); this.db.exec('DELETE FROM sessions'); });
      }
    } catch (error) { this.db.close(); throw error; }
  }
  private directory(path: string) {
    mkdirSync(path,{recursive:true,mode:0o700});
    if (!lstatSync(path).isDirectory() || lstatSync(path).isSymbolicLink()) throw new Error('Unsafe data directory');
    chmodSync(path,0o700);
  }
  private migrate() {
    this.db.exec(`BEGIN IMMEDIATE;
      CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE projects(id TEXT PRIMARY KEY,json TEXT NOT NULL,draft TEXT NOT NULL,generation INTEGER NOT NULL DEFAULT 0,job_generation INTEGER NOT NULL DEFAULT 0,deleted INTEGER NOT NULL DEFAULT 0);
      CREATE TABLE revisions(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),number INTEGER NOT NULL,json TEXT NOT NULL,UNIQUE(project_id,number));
      CREATE TRIGGER immutable_revision BEFORE UPDATE ON revisions BEGIN SELECT RAISE(ABORT,'immutable revision'); END;
      CREATE TABLE jobs(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),revision_id TEXT NOT NULL REFERENCES revisions(id),request_id TEXT NOT NULL,json TEXT NOT NULL,status TEXT NOT NULL,generation INTEGER NOT NULL DEFAULT 0,project_generation INTEGER NOT NULL,job_generation INTEGER NOT NULL,attempts INTEGER NOT NULL DEFAULT 0,lease TEXT,lease_deadline INTEGER,scheduler TEXT,cancel_requested INTEGER NOT NULL DEFAULT 0,UNIQUE(project_id,request_id));
      CREATE INDEX jobs_status ON jobs(status);
      CREATE TABLE artifacts(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),revision_id TEXT NOT NULL REFERENCES revisions(id),job_id TEXT REFERENCES jobs(id),json TEXT NOT NULL,storage_key TEXT NOT NULL UNIQUE);
      CREATE TRIGGER immutable_artifact BEFORE UPDATE ON artifacts BEGIN SELECT RAISE(ABORT,'immutable artifact'); END;
      CREATE TABLE geometry_heads(revision_id TEXT PRIMARY KEY REFERENCES revisions(id),job_id TEXT NOT NULL REFERENCES jobs(id));
      CREATE TABLE reference_assets(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),storage_key TEXT NOT NULL UNIQUE,digest TEXT NOT NULL,bytes INTEGER NOT NULL,width INTEGER NOT NULL,height INTEGER NOT NULL,label TEXT NOT NULL,created_at TEXT NOT NULL);
      CREATE TABLE comments(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),revision_id TEXT NOT NULL REFERENCES revisions(id),json TEXT NOT NULL);
      CREATE TABLE snapshots(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),revision_id TEXT NOT NULL REFERENCES revisions(id),generation INTEGER NOT NULL,json TEXT NOT NULL);
      CREATE TRIGGER immutable_snapshot BEFORE UPDATE ON snapshots BEGIN SELECT RAISE(ABORT,'immutable snapshot'); END;
      CREATE TABLE snapshot_results(snapshot_id TEXT PRIMARY KEY REFERENCES snapshots(id),manifest TEXT NOT NULL);
      CREATE TABLE export_files(snapshot_id TEXT NOT NULL REFERENCES snapshots(id),filename TEXT NOT NULL,mime TEXT NOT NULL,storage_key TEXT NOT NULL UNIQUE,digest TEXT NOT NULL,bytes INTEGER NOT NULL,PRIMARY KEY(snapshot_id,filename));
      CREATE TABLE import_previews(id TEXT PRIMARY KEY,project_id TEXT NOT NULL REFERENCES projects(id),json TEXT NOT NULL,operations TEXT NOT NULL,accepted INTEGER NOT NULL DEFAULT 0);
      CREATE TABLE sessions(hash TEXT PRIMARY KEY,expires INTEGER NOT NULL);
      CREATE TABLE login_attempts(id INTEGER PRIMARY KEY AUTOINCREMENT,at INTEGER NOT NULL);
      CREATE TABLE scheduler(id INTEGER PRIMARY KEY CHECK(id=1),owner TEXT,generation INTEGER NOT NULL DEFAULT 0,deadline INTEGER NOT NULL DEFAULT 0);
      INSERT INTO scheduler(id) VALUES(1);
      PRAGMA user_version=1; COMMIT;`);
  }
  assertReady() {
    if(this.closed) throw new ApiError(503,'API storage is closed');
    this.transaction(()=>{
      try {
        if(existsSync(join(this.root,'RESTORE_PENDING'))) throw new Error();
        const path=join(this.root,'deletions-watermark.json'),journal=join(this.root,'deletions.jsonl');
        if(!lstatSync(path).isFile()||lstatSync(path).isSymbolicLink()||!lstatSync(journal).isFile()||lstatSync(journal).isSymbolicLink()) throw new Error();
        const marker=JSON.parse(readFileSync(path,'utf8'));
        const text=readFileSync(journal,'utf8');
        if(text&&!text.endsWith('\n'))throw new Error();
        const tail=text.trimEnd().split('\n').at(-1);
        const entry=tail?JSON.parse(tail):{sequence:0,digest:''};
        if(marker.sequence!==Number(this.meta('deletionSequence')??0)||marker.sequence!==entry.sequence||marker.digest!==entry.digest)throw new Error();
      } catch {throw new ApiError(503,'Deletion journal or restore state requires recovery; access is closed');}
    });
  }
  transaction<T>(fn: ()=>T): T { return this.db.transaction(fn).immediate(); }
  meta(key: string): string | null { return (this.db.query('SELECT value FROM meta WHERE key=?').get(key) as {value:string}|null)?.value ?? null; }
  setMeta(key: string,value: string) { this.db.query('INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value').run(key,value); }
  project(projectId: string): ProjectRow {
    const row = this.db.query('SELECT * FROM projects WHERE id=? AND deleted=0').get(projectId) as ProjectRow|null;
    if (!row) throw new ApiError(404,'Project not found');
    return row;
  }
  revision(projectId: string,revisionId: string): Revision {
    this.project(projectId);
    const row = this.db.query('SELECT json FROM revisions WHERE id=? AND project_id=?').get(revisionId,projectId) as {json:string}|null;
    if (!row) throw new ApiError(404,'Revision not found');
    return JSON.parse(row.json);
  }
  draft(projectId: string): Draft { return JSON.parse(this.project(projectId).draft); }
  checkIdentity(row: ProjectRow, expectedVersion: number, expectedRevisionId: string|null): Draft {
    const draft: Draft = JSON.parse(row.draft), project: Project = JSON.parse(row.json);
    if (draft.version!==expectedVersion || draft.baseRevisionId!==expectedRevisionId || project.headRevisionId!==expectedRevisionId) throw new ApiError(409,'Draft changed; reconcile before saving',{latest:draft});
    return draft;
  }
  state(projectId: string): ProjectState {
    const row = this.project(projectId);
    const values = <T>(table:string,order='rowid') => (this.db.query(`SELECT json FROM ${table} WHERE project_id=? ORDER BY ${order}`).all(projectId) as {json:string}[]).map(r=>JSON.parse(r.json) as T);
    return {project:JSON.parse(row.json),draft:JSON.parse(row.draft),revisions:values<Revision>('revisions','number'),jobs:values<Job>('jobs','rowid DESC'),artifacts:values<Artifact>('artifacts'),comments:values<ReviewComment>('comments')};
  }
  job(projectId: string, jobId: string): JobRow {
    this.project(projectId);
    const row = this.db.query('SELECT * FROM jobs WHERE id=? AND project_id=?').get(jobId,projectId) as JobRow|null;
    if (!row) throw new ApiError(404,'Job not found');
    return row;
  }
  jobStatus(row: JobRow,status: Job['status'],error: string|null = null) {
    const value: Job = {...JSON.parse(row.json),status,error,updatedAt:now()};
    this.db.query('UPDATE jobs SET status=?,json=?,generation=generation+1,lease=NULL,lease_deadline=NULL,scheduler=NULL WHERE id=?').run(status,JSON.stringify(value),row.id);
    return value;
  }
  reference(projectId:string,assetId:string): ReferenceRow {
    const row = this.db.query('SELECT * FROM reference_assets WHERE id=? AND project_id=?').get(assetId,projectId) as ReferenceRow|null;
    if (!row) throw new ApiError(422,'Unknown reference asset');
    return row;
  }
  checkReferences(projectId:string, doc: {views:{assetId:string}[]}) { for (const view of doc.views) this.reference(projectId,view.assetId); }
  snapshot(projectId:string,snapshotId:string): SnapshotRow {
    this.project(projectId);
    const row = this.db.query('SELECT * FROM snapshots WHERE id=? AND project_id=?').get(snapshotId,projectId) as SnapshotRow|null;
    if (!row) throw new ApiError(404,'Export snapshot not found');
    return row;
  }
  readBlob(key:string,digest:string,size:number): Uint8Array {
    if (!/^[a-zA-Z0-9_-]{24}$/.test(key)) throw new ApiError(409,'Stored file integrity failure');
    const path = join(this.root,'blobs',key);
    try {
      const stat = lstatSync(path);
      if (!stat.isFile() || stat.isSymbolicLink() || stat.size!==size) throw new Error();
      const fd = openSync(path,constants.O_RDONLY|constants.O_NOFOLLOW);
      let bytes: Buffer;
      try { bytes = readFileSync(fd); } finally { closeSync(fd); }
      if (hash(bytes)!==digest) throw new Error();
      return new Uint8Array(bytes);
    } catch { throw new ApiError(409,'Stored file missing or digest mismatch'); }
  }
  install<T>(files:StoredFile[],commit:(records:FileRecord[])=>T): T {
    this.assertReady();
    const marker=join(this.root,'markers',id());
    const result=this.transaction(()=>{
      const records=files.map(f=>({filename:f.filename,mime:f.mime,bytes:f.bytes.length,storageKey:id(),digest:hash(f.bytes)}));
      this.atomic(marker,Buffer.from(JSON.stringify(records)));
      try {
        for(let i=0;i<files.length;i++)this.atomic(join(this.root,'blobs',records[i]!.storageKey),files[i]!.bytes,0o400);
        return commit(records);
      }catch(error){
        for(const record of records)rmSync(join(this.root,'blobs',record.storageKey),{force:true});
        rmSync(marker,{force:true});throw error;
      }
    });
    try{rmSync(marker,{force:true});this.syncDirectory(join(this.root,'markers'));}catch{}
    return result;
  }
  private referenced(key:string): boolean {
    return Boolean(this.db.query('SELECT 1 FROM artifacts WHERE storage_key=? UNION ALL SELECT 1 FROM reference_assets WHERE storage_key=? UNION ALL SELECT 1 FROM export_files WHERE storage_key=? UNION ALL SELECT 1 FROM three_d_artifacts WHERE storage_key=? LIMIT 1').get(key,key,key,key));
  }
  reconcile() {
    this.transaction(()=>{
      for(const name of readdirSync(join(this.root,'blobs'))) {
        const path=join(this.root,'blobs',name);
        if((/^[a-zA-Z0-9_-]{24}$/.test(name)&&!this.referenced(name))||/^[a-zA-Z0-9_-]{24}\.[a-zA-Z0-9_-]{24}\.tmp$/.test(name))rmSync(path,{force:true});
      }
      for(const name of readdirSync(join(this.root,'markers'))) {
        if(/^[a-zA-Z0-9_-]{24}(?:\.[a-zA-Z0-9_-]{24}\.tmp)?$/.test(name))rmSync(join(this.root,'markers',name),{force:true});
      }
      for(const name of readdirSync(join(this.root,'staging'))) {
        const inspection=/^three-d-([a-zA-Z0-9_-]{24})-(\d+)$/.exec(name);
        if(inspection) {
          const running=this.db.query("SELECT 1 FROM three_d_jobs WHERE id=? AND generation=? AND status='running' AND deadline>?").get(inspection[1]!,Number(inspection[2]),Date.now());
          if(!running)rmSync(join(this.root,'staging',name),{recursive:true,force:true});
          continue;
        }
        const match=/^([a-zA-Z0-9_-]{24})-(\d+)$/.exec(name);
        if(!match)continue;
        const running=this.db.query("SELECT 1 FROM jobs WHERE id=? AND generation=? AND status='running' AND lease_deadline>?").get(match[1]!,Number(match[2]),Date.now());
        if(!running)rmSync(join(this.root,'staging',name),{recursive:true,force:true});
      }
    });
  }
  private syncDirectory(path: string) { const fd = openSync(path,constants.O_RDONLY); try { fsyncSync(fd); } finally { closeSync(fd); } }
  private atomic(path: string,bytes: Uint8Array,mode=0o600) {
    const temp = `${path}.${id()}.tmp`;
    const fd = openSync(temp,constants.O_WRONLY|constants.O_CREAT|constants.O_EXCL|constants.O_NOFOLLOW,mode);
    try { writeFileSync(fd,bytes); fsyncSync(fd); } finally { closeSync(fd); }
    renameSync(temp,path); this.syncDirectory(resolve(path,'..'));
  }
  private replayDeletions(existed:boolean) {
    const journal = join(this.root,'deletions.jsonl'), watermark = join(this.root,'deletions-watermark.json');
    if (!existsSync(journal) || !existsSync(watermark)) {
      if (existed) throw new Error('Deletion journal incomplete; access remains closed');
      this.atomic(journal,Buffer.from('')); this.atomic(watermark,Buffer.from(JSON.stringify({sequence:0,digest:''})));
    }
    for (const path of [journal,watermark]) if (lstatSync(path).isSymbolicLink() || !lstatSync(path).isFile()) throw new Error('Unsafe deletion journal');
    const text = readFileSync(journal,'utf8');
    if (text && !text.endsWith('\n')) throw new Error('Truncated deletion journal; access remains closed');
    const mark = JSON.parse(readFileSync(watermark,'utf8')) as {sequence:number,digest:string};
    let sequence=0, prior='';
    const entries: {sequence:number;projectId:string;digest:string}[]=[];
    for (const line of text.split('\n').filter(Boolean)) {
      const entry = JSON.parse(line);
      if (entry.sequence!==++sequence || entry.prior!==prior || entry.digest!==objectDigest({sequence:entry.sequence,projectId:entry.projectId,prior:entry.prior})) throw new Error('Deletion journal integrity failure');
      prior=entry.digest;
      if (sequence===mark.sequence && prior!==mark.digest) throw new Error('Deletion watermark mismatch');
      entries.push(entry);
    }
    if (!Number.isInteger(mark.sequence) || mark.sequence<0 || mark.sequence>sequence || (mark.sequence===0 && mark.digest!=='')) throw new Error('Deletion watermark incomplete');
    const checkpoint = Number(this.meta('deletionSequence') ?? 0);
    if (checkpoint>sequence) throw new Error('Deletion journal is older than database');
    this.transaction(()=>{
      for (const entry of entries) this.tombstone(entry.projectId);
      if (checkpoint!==sequence) this.db.exec('DELETE FROM sessions');
      this.setMeta('deletionSequence',String(sequence));
    });
    this.journalSequence=sequence; this.journalDigest=prior;
    this.atomic(watermark,Buffer.from(JSON.stringify({sequence,digest:prior})));
  }
  private tombstone(projectId:string) {
    if (/^!body-profile:[0-9]+$/.test(projectId)) {
      const deletedVersion = Number(projectId.split(':')[1]);
      const current = JSON.parse(this.meta('bodyProfile') ?? '{"version":0,"body":null}');
      if (current.version <= deletedVersion) this.setMeta('bodyProfile', JSON.stringify({ version: deletedVersion, body: null }));
      return;
    }
    const keys=this.db.query('SELECT storage_key FROM artifacts WHERE project_id=? UNION SELECT storage_key FROM reference_assets WHERE project_id=? UNION SELECT storage_key FROM export_files WHERE snapshot_id IN (SELECT id FROM snapshots WHERE project_id=?) UNION SELECT storage_key FROM three_d_artifacts WHERE project_id=?').all(projectId,projectId,projectId,projectId) as {storage_key:string}[];
    const jobIds=(this.db.query('SELECT id FROM jobs WHERE project_id=?').all(projectId) as {id:string}[]).map(r=>r.id);
    const inspectionIds=(this.db.query('SELECT id FROM three_d_jobs WHERE project_id=?').all(projectId) as {id:string}[]).map(row=>row.id);
    this.db.query('UPDATE projects SET deleted=1,generation=generation+1,json=?,draft=? WHERE id=? AND deleted=0').run('{}','{}',projectId);
    this.db.query('DELETE FROM geometry_heads WHERE revision_id IN (SELECT id FROM revisions WHERE project_id=?)').run(projectId);
    this.db.query('DELETE FROM export_files WHERE snapshot_id IN (SELECT id FROM snapshots WHERE project_id=?)').run(projectId);
    this.db.query('DELETE FROM snapshot_results WHERE snapshot_id IN (SELECT id FROM snapshots WHERE project_id=?)').run(projectId);
    this.db.query("UPDATE ai_requests SET project_id='deleted' WHERE project_id=?").run(projectId);
    for(const table of ['three_d_artifacts','three_d_jobs','interpretation_jobs','ai_proposals','artifacts','snapshots','import_previews','comments','jobs','reference_assets','revisions'])this.db.query(`DELETE FROM ${table} WHERE project_id=?`).run(projectId);
    for(const {storage_key:key} of keys)if(/^[a-zA-Z0-9_-]{24}$/.test(key))rmSync(join(this.root,'blobs',key),{force:true});
    for(const name of readdirSync(join(this.root,'staging')))if(jobIds.some(jobId=>name.startsWith(jobId+'-')))rmSync(join(this.root,'staging',name),{recursive:true,force:true});
    for(const name of readdirSync(join(this.root,'staging')))if(inspectionIds.some(jobId=>name.startsWith('three-d-'+jobId+'-')))rmSync(join(this.root,'staging',name),{recursive:true,force:true});
  }
  deleteProject(projectId:string) {
    this.transaction(()=>{
      this.project(projectId);
      this.appendDeletion(projectId);
    });
  }
  deleteBodyProfile(version:number) {
    this.appendDeletion(`!body-profile:${version}`);
  }
  private appendDeletion(projectId:string) {
    this.transaction(()=>{
      const text = readFileSync(join(this.root,'deletions.jsonl'),'utf8');
      const tail = text.trimEnd().split('\n').at(-1);
      const previous = tail ? JSON.parse(tail) : {sequence:0,digest:''};
      const core = {sequence:previous.sequence+1,projectId,prior:previous.digest};
      const entry = {...core,digest:objectDigest(core)};
      const fd = openSync(join(this.root,'deletions.jsonl'),constants.O_WRONLY|constants.O_APPEND|constants.O_NOFOLLOW);
      try { writeFileSync(fd,JSON.stringify(entry)+'\n'); fsyncSync(fd); } finally { closeSync(fd); }
      this.atomic(join(this.root,'deletions-watermark.json'),Buffer.from(JSON.stringify({sequence:entry.sequence,digest:entry.digest})));
      this.tombstone(projectId);
      this.setMeta('deletionSequence',String(entry.sequence));
      this.journalSequence=entry.sequence; this.journalDigest=entry.digest;
    });
  }
  close() { if (!this.closed) { this.closed=true; this.db.close(); } }
}
