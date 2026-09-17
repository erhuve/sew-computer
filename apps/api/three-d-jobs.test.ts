import { afterEach, expect, test } from 'bun:test';
import { mkdtempSync, readdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import type { Artifact, Draft, Job, PatternGeometry, Project, Revision } from '../../packages/contracts';
import { shirtDocument } from '../../packages/test-fixtures/shirt';
import { ThreeDQueue, type InspectionEngine } from './three-d-jobs';
import { Store } from './store';
import { id, now, objectDigest } from './validation';

const cleanups:(()=>void)[]=[];
afterEach(()=>{for(const cleanup of cleanups.splice(0).reverse())cleanup();});
const settle=()=>new Promise(resolve=>setTimeout(resolve,20));
const invalidOutput={report:new Uint8Array(),mesh:new Uint8Array()};
function deferredEngine() {
  let release!:(value:typeof invalidOutput)=>void;
  const engine:InspectionEngine=()=>new Promise(resolve=>{release=resolve;});
  return {engine,release:()=>release(invalidOutput)};
}
function setup(engine:InspectionEngine) {
  const root=mkdtempSync(join(import.meta.dir,'.test-three-d-'));
  const store=new Store(root,'test-key'),date=now(),projectId=id(),revisionId=id();
  const document=shirtDocument(),digest=objectDigest(document);
  const project:Project={id:projectId,title:'Synthetic',headRevisionId:revisionId,createdAt:date,updatedAt:date};
  const draft:Draft={projectId,version:1,baseRevisionId:revisionId,document,updatedAt:date};
  const revision:Revision={id:revisionId,projectId,number:1,parentRevisionId:null,document,digest,createdAt:date};
  store.db.query('INSERT INTO projects(id,json,draft) VALUES(?,?,?)').run(projectId,JSON.stringify(project),JSON.stringify(draft));
  store.db.query('INSERT INTO revisions(id,project_id,number,json) VALUES(?,?,?,?)').run(revisionId,projectId,1,JSON.stringify(revision));
  const geometry:PatternGeometry={schemaVersion:1,units:'mm',inputDigest:digest,engineVersion:'synthetic-lifecycle-fixture',family:'shirt',panels:[{id:'front',name:'Front',widthMm:100,heightMm:100,points:[[0,0],[100,0],[100,100],[0,100],[0,0]]}],stitches:[],warnings:[],assumptions:[],classification:'printable-reference'};
  const patternJob:Job={id:id(),projectId,revisionId,requestId:id(),status:'succeeded',error:null,createdAt:date,updatedAt:date};
  store.db.query('INSERT INTO jobs(id,project_id,revision_id,request_id,json,status,project_generation,job_generation) VALUES(?,?,?,?,?,?,0,0)').run(patternJob.id,projectId,revisionId,patternJob.requestId,JSON.stringify(patternJob),'succeeded');
  store.db.query('INSERT INTO geometry_heads(revision_id,job_id) VALUES(?,?)').run(revisionId,patternJob.id);
  const bytes=Buffer.from(JSON.stringify(geometry));
  store.install([{filename:'pattern.json',mime:'application/json',bytes}],records=>{
    const record=records[0]!,artifact:Artifact={id:id(),projectId,revisionId,jobId:patternJob.id,kind:'pattern-json',filename:'pattern.json',mime:'application/json',digest:record.digest,bytes:record.bytes,classification:'printable-reference',createdAt:date};
    store.db.query('INSERT INTO artifacts(id,project_id,revision_id,job_id,json,storage_key) VALUES(?,?,?,?,?,?)').run(artifact.id,projectId,revisionId,patternJob.id,JSON.stringify(artifact),record.storageKey);
  });
  const queue=new ThreeDQueue(store,engine);
  cleanups.push(()=>{queue.close();store.close();rmSync(root,{recursive:true,force:true});});
  return {root,store,queue,projectId,revisionId,patternJob,requestId:id()};
}

test('3D acknowledgement, retry identity and cancellation preserve independent 2D fences',async()=>{
  const deferred=deferredEngine(),fixture=setup(deferred.engine);
  const before=fixture.store.project(fixture.projectId);
  const job=fixture.queue.submit(fixture.projectId,fixture.revisionId,fixture.requestId);
  expect(job.status).toBe('queued');
  expect(fixture.queue.get(fixture.projectId,job.id).status).toBe('running');
  expect(fixture.queue.submit(fixture.projectId,fixture.revisionId,fixture.requestId).id).toBe(job.id);
  expect(()=>fixture.queue.submit(fixture.projectId,id(),fixture.requestId)).toThrow('another 3D revision');
  expect(()=>fixture.queue.submit(fixture.projectId,fixture.revisionId,id())).toThrow('already queued');
  expect(fixture.queue.cancel(fixture.projectId,job.id).status).toBe('cancelled');
  deferred.release();await settle();
  expect(fixture.queue.get(fixture.projectId,job.id).status).toBe('cancelled');
  expect(fixture.store.project(fixture.projectId)).toEqual(before);
  expect(fixture.store.job(fixture.projectId,fixture.patternJob.id).status).toBe('succeeded');
  expect(fixture.store.db.query('SELECT COUNT(*) AS count FROM three_d_artifacts').get()).toEqual({count:0});
});

test('new revision and project deletion fence late 3D output',async()=>{
  const deferred=deferredEngine(),fixture=setup(deferred.engine);
  const job=fixture.queue.submit(fixture.projectId,fixture.revisionId,fixture.requestId);
  const project=JSON.parse(fixture.store.project(fixture.projectId).json);
  fixture.store.db.query('UPDATE projects SET json=? WHERE id=?').run(JSON.stringify({...project,headRevisionId:id()}),fixture.projectId);
  fixture.queue.poke();deferred.release();await settle();
  expect(fixture.queue.get(fixture.projectId,job.id).status).toBe('stale');
  expect(fixture.queue.latest(fixture.projectId,fixture.revisionId)?.sourceCurrent).toBe(false);
  fixture.store.deleteProject(fixture.projectId);
  expect(()=>fixture.queue.get(fixture.projectId,job.id)).toThrow('Project not found');
  expect(fixture.store.db.query('SELECT COUNT(*) AS count FROM three_d_jobs').get()).toEqual({count:0});
  expect(readdirSync(join(fixture.root,'blobs'))).toEqual([]);
});

test('lease expiry and restart cannot allow old worker installation',async()=>{
  const first=deferredEngine(),fixture=setup(first.engine);
  const job=fixture.queue.submit(fixture.projectId,fixture.revisionId,fixture.requestId);
  const original=fixture.store.db.query('SELECT lease,generation FROM three_d_jobs WHERE id=?').get(job.id);
  fixture.store.db.query('UPDATE three_d_jobs SET deadline=0 WHERE id=?').run(job.id);
  const second=deferredEngine(),replacement=new ThreeDQueue(fixture.store,second.engine);
  cleanups.push(()=>replacement.close());
  const retried=fixture.store.db.query('SELECT lease,generation,attempts FROM three_d_jobs WHERE id=?').get(job.id) as {lease:string;generation:number;attempts:number};
  expect(retried.attempts).toBe(2);expect(retried).not.toEqual(original);
  first.release();await settle();
  expect(replacement.get(fixture.projectId,job.id).status).toBe('running');
  replacement.cancel(fixture.projectId,job.id);second.release();await settle();
  expect(replacement.get(fixture.projectId,job.id).status).toBe('cancelled');
  expect(fixture.store.db.query('SELECT COUNT(*) AS count FROM three_d_artifacts').get()).toEqual({count:0});
});

test('invalid worker output fails without replacing patterns or installing artifacts',async()=>{
  const fixture=setup(async()=>invalidOutput);
  const job=fixture.queue.submit(fixture.projectId,fixture.revisionId,fixture.requestId);await settle();
  expect(fixture.queue.get(fixture.projectId,job.id).status).toBe('failed');
  expect(()=>fixture.queue.artifact(fixture.projectId,job.id,'inspection.glb')).toThrow('no committed result');
  expect(fixture.store.db.query('SELECT COUNT(*) AS count FROM artifacts').get()).toEqual({count:1});
  expect(fixture.store.db.query('SELECT COUNT(*) AS count FROM three_d_artifacts').get()).toEqual({count:0});
});

test('queued captured-input mutation and engine replacement cannot execute',async()=>{
  const deferred=deferredEngine(),fixture=setup(deferred.engine);
  const job=fixture.queue.submit(fixture.projectId,fixture.revisionId,fixture.requestId);
  fixture.queue.close();deferred.release();await settle();
  let calls=0;
  const engine:InspectionEngine=async()=>{calls++;return invalidOutput;};engine.version='different-implementation';
  const replacement=new ThreeDQueue(fixture.store,engine);cleanups.push(()=>replacement.close());
  expect(replacement.get(fixture.projectId,job.id).status).toBe('stale');
  expect(calls).toBe(0);
});

test('schema-four migration preserves revision and immutable pattern blob',()=>{
  const fixture=setup(async()=>invalidOutput);fixture.queue.close();
  const state=fixture.store.state(fixture.projectId),blobs=readdirSync(join(fixture.root,'blobs'));
  fixture.store.db.exec('DROP TABLE three_d_artifacts; DROP TABLE three_d_jobs; PRAGMA user_version=4;');fixture.store.close();
  const reopened=new Store(fixture.root,'test-key');cleanups.push(()=>reopened.close());reopened.reconcile();
  expect(reopened.db.query('PRAGMA user_version').get()).toEqual({user_version:5});
  expect(reopened.state(fixture.projectId)).toEqual(state);
  expect(readdirSync(join(fixture.root,'blobs'))).toEqual(blobs);
});
