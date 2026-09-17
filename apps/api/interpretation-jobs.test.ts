import { afterEach, expect, test } from 'bun:test';
import { mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { emptyDocument, type Artifact, type Draft, type Project } from '../../packages/contracts';
import { interpretationFixture } from '../../packages/test-fixtures/interpretation';
import { InterpretationService, type Interpreter } from './interpretation';
import { InterpretationQueue } from './interpretation-jobs';
import { Store } from './store';
import { id, now } from './validation';

const cleanups:(()=>void)[]=[];
afterEach(()=>{for(const cleanup of cleanups.splice(0).reverse())cleanup();});
const result=()=>({value:structuredClone(interpretationFixture),inputTokens:1,outputTokens:1});
function setup(run:Interpreter['run']=async()=>result()) {
  const root=mkdtempSync(join(import.meta.dir,'.test-interpretation-jobs-'));
  const store=new Store(root,'test-key');
  const project:Project={id:id(),title:'Test',headRevisionId:null,createdAt:now(),updatedAt:now()};
  const draft:Draft={projectId:project.id,version:1,baseRevisionId:null,document:emptyDocument('Test','A blue linen shirt'),updatedAt:now()};
  store.db.query('INSERT INTO projects(id,json,draft) VALUES(?,?,?)').run(project.id,JSON.stringify(project),JSON.stringify(draft));
  const interpreter:Interpreter={status:{available:true,provider:'fixture',model:'fixture',maxOutputTokens:6000,timeoutSeconds:120,referenceLimit:3},run};
  const service=new InterpretationService(store,interpreter);
  const queue=new InterpretationQueue(store,service);
  cleanups.push(()=>{queue.close();service.close();store.close();rmSync(root,{recursive:true,force:true});});
  const identity={requestId:id(),expectedVersion:1,expectedRevisionId:null,includeReferences:false,consent:true as const};
  return {root,store,project,draft,interpreter,service,queue,identity};
}
async function settle() {await new Promise(resolve=>setTimeout(resolve,10));}

test('version-three migration preserves saved project revision and artifact bytes',async()=>{
  const fixture=setup();
  fixture.queue.close();
  const proposal=await fixture.service.propose(fixture.project.id,fixture.identity);
  const accepted=fixture.service.accept(fixture.project.id,proposal.id,fixture.identity);
  const revisionId=accepted.project.headRevisionId!;
  const bytes=new TextEncoder().encode('<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0L100 0L100 200Z"/></svg>');
  const artifactId=id();
  const record=fixture.store.install([{filename:'legacy-pattern.svg',mime:'image/svg+xml',bytes}],records=>{
    const stored=records[0]!;
    const artifact:Artifact={id:artifactId,projectId:fixture.project.id,revisionId,jobId:null,kind:'pattern-svg',filename:'legacy-pattern.svg',mime:'image/svg+xml',digest:stored.digest,bytes:stored.bytes,classification:'printable-reference',createdAt:now()};
    fixture.store.db.query('INSERT INTO artifacts(id,project_id,revision_id,job_id,json,storage_key) VALUES(?,?,?,?,?,?)').run(artifactId,fixture.project.id,revisionId,null,JSON.stringify(artifact),stored.storageKey);
    return stored;
  });
  const originalState=fixture.store.state(fixture.project.id);
  const originalRows={project:fixture.store.db.query('SELECT * FROM projects WHERE id=?').get(fixture.project.id),revision:fixture.store.db.query('SELECT * FROM revisions WHERE id=?').get(revisionId),artifact:fixture.store.db.query('SELECT * FROM artifacts WHERE id=?').get(artifactId)};
  fixture.service.close();
  fixture.store.db.exec('DROP TABLE interpretation_jobs; PRAGMA user_version=3;');
  fixture.store.close();
  const migrated=new Store(fixture.root,'test-key');
  cleanups.push(()=>migrated.close());
  migrated.reconcile();
  expect(migrated.db.query('PRAGMA user_version').get()).toEqual({user_version:5});
  expect(migrated.state(fixture.project.id)).toEqual(originalState);
  expect(migrated.db.query('SELECT * FROM projects WHERE id=?').get(fixture.project.id)).toEqual(originalRows.project);
  expect(migrated.db.query('SELECT * FROM revisions WHERE id=?').get(revisionId)).toEqual(originalRows.revision);
  expect(migrated.db.query('SELECT * FROM artifacts WHERE id=?').get(artifactId)).toEqual(originalRows.artifact);
  expect(migrated.readBlob(record.storageKey,record.digest,record.bytes)).toEqual(bytes);
  expect(migrated.db.query('SELECT COUNT(*) AS count FROM interpretation_jobs').get()).toEqual({count:0});
});

test('duplicate submission after newer draft returns captured job without changing newer work',async()=>{
  let release!:(value:ReturnType<typeof result>)=>void,calls=0;
  const fixture=setup(()=>{calls++;return new Promise(resolve=>{release=resolve;});});
  const job=fixture.queue.submit(fixture.project.id,fixture.identity);
  const captured=fixture.store.db.query('SELECT input,input_digest FROM interpretation_jobs WHERE id=?').get(job.id);
  const newer:Draft={...fixture.draft,version:2,document:{...fixture.draft.document,brief:'Newer asymmetric garment intent'},updatedAt:now()};
  fixture.store.db.query('UPDATE projects SET draft=? WHERE id=?').run(JSON.stringify(newer),fixture.project.id);
  expect(fixture.queue.submit(fixture.project.id,fixture.identity).id).toBe(job.id);
  expect(()=>fixture.queue.submit(fixture.project.id,{...fixture.identity,expectedVersion:2})).toThrow('different interpretation inputs');
  expect(fixture.store.draft(fixture.project.id)).toEqual(newer);
  expect(fixture.store.db.query('SELECT input,input_digest FROM interpretation_jobs WHERE id=?').get(job.id)).toEqual(captured);
  release(result());await settle();
  expect(fixture.queue.submit(fixture.project.id,fixture.identity).status).toBe('succeeded');
  expect(fixture.service.latest(fixture.project.id)?.document.brief).toBe(fixture.draft.document.brief);
  expect(fixture.service.latest(fixture.project.id)?.baseVersion).toBe(1);
  expect(fixture.store.draft(fixture.project.id)).toEqual(newer);
  expect(calls).toBe(1);
  expect(fixture.store.db.query('SELECT COUNT(*) AS count FROM ai_requests').get()).toEqual({count:1});
});

test('submission acknowledges before slow inference and duplicate identity returns one durable job',async()=>{
  let release!:(value:ReturnType<typeof result>)=>void,calls=0;
  const fixture=setup(()=>{calls++;return new Promise(resolve=>{release=resolve;});});
  const started=Date.now();
  const job=fixture.queue.submit(fixture.project.id,fixture.identity);
  expect(Date.now()-started).toBeLessThan(1000);
  expect(job.status).toBe('queued');
  expect(fixture.queue.submit(fixture.project.id,fixture.identity).id).toBe(job.id);
  expect(calls).toBe(1);
  expect(()=>fixture.queue.submit(fixture.project.id,{...fixture.identity,includeReferences:true})).toThrow('different interpretation inputs');
  expect(fixture.queue.latest(fixture.project.id)?.status).toBe('running');
  release(result());await settle();
  expect(fixture.queue.get(fixture.project.id,job.id).status).toBe('succeeded');
  expect(fixture.service.latest(fixture.project.id)?.id).toBe(job.id);
  expect(fixture.queue.cancel(fixture.project.id,job.id).status).toBe('succeeded');
});

test('cancellation rejects late uncooperative provider writes and preserves 2D generation fences',async()=>{
  let release!:(value:ReturnType<typeof result>)=>void;
  const fixture=setup(()=>new Promise(resolve=>{release=resolve;}));
  const before=fixture.store.project(fixture.project.id);
  const job=fixture.queue.submit(fixture.project.id,fixture.identity);
  expect(fixture.queue.cancel(fixture.project.id,job.id).status).toBe('cancelled');
  release(result());await settle();
  expect(fixture.service.latest(fixture.project.id)).toBeNull();
  const after=fixture.store.project(fixture.project.id);
  expect(after.job_generation).toBe(before.job_generation);
  expect(after.generation).toBe(before.generation);
  expect(fixture.store.draft(fixture.project.id)).toEqual(fixture.draft);
});

test('restart resumes captured input and fences the old attempt',async()=>{
  let release!:(value:ReturnType<typeof result>)=>void;
  const fixture=setup(()=>new Promise(resolve=>{release=resolve;}));
  const job=fixture.queue.submit(fixture.project.id,fixture.identity);
  fixture.queue.close();fixture.service.close();
  fixture.store.close();
  const reopened=new Store(fixture.root,'test-key');
  const service=new InterpretationService(reopened,{...fixture.interpreter,run:async()=>result()});
  const queue=new InterpretationQueue(reopened,service);
  cleanups.push(()=>{queue.close();service.close();reopened.close();});
  release(result());await settle();
  expect(queue.get(fixture.project.id,job.id).status).toBe('succeeded');
  expect(reopened.db.query('SELECT COUNT(*) AS count FROM ai_proposals').get()).toEqual({count:1});
  expect(reopened.db.query('SELECT COUNT(*) AS count FROM ai_requests').get()).toEqual({count:1});
});

test('lease expiry permits bounded recovery across workers and cannot accept stale writes',async()=>{
  let release!:(value:ReturnType<typeof result>)=>void;
  const fixture=setup(()=>new Promise(resolve=>{release=resolve;}));
  const job=fixture.queue.submit(fixture.project.id,fixture.identity);
  fixture.store.db.query('UPDATE interpretation_jobs SET deadline=0 WHERE id=?').run(job.id);
  const service=new InterpretationService(fixture.store,{...fixture.interpreter,run:async()=>result()});
  const queue=new InterpretationQueue(fixture.store,service);
  cleanups.push(()=>{queue.close();service.close();});
  await settle();release(result());await settle();
  expect(queue.get(fixture.project.id,job.id).status).toBe('succeeded');
  expect(fixture.store.db.query('SELECT attempts FROM interpretation_jobs WHERE id=?').get(job.id)).toEqual({attempts:2});
  expect(fixture.store.db.query('SELECT COUNT(*) AS count FROM ai_proposals').get()).toEqual({count:1});
});

test('concurrent edits preserve captured proposal and prevent acceptance; deleted jobs disappear',async()=>{
  let release!:(value:ReturnType<typeof result>)=>void;
  const fixture=setup(()=>new Promise(resolve=>{release=resolve;}));
  const job=fixture.queue.submit(fixture.project.id,fixture.identity);
  fixture.store.db.query('UPDATE projects SET draft=? WHERE id=?').run(JSON.stringify({...fixture.draft,version:2,document:{...fixture.draft.document,brief:'Different garment'}}),fixture.project.id);
  release(result());await settle();
  expect(fixture.queue.get(fixture.project.id,job.id).status).toBe('succeeded');
  expect(()=>fixture.service.accept(fixture.project.id,job.id,{expectedVersion:2,expectedRevisionId:null})).toThrow('stale');
  fixture.store.deleteProject(fixture.project.id);
  expect(()=>fixture.queue.get(fixture.project.id,job.id)).toThrow('not found');
  expect(fixture.store.db.query('SELECT COUNT(*) AS count FROM interpretation_jobs').get()).toEqual({count:0});
});

test('provider running beyond the proxy timeout remains a pollable durable job',async()=>{
  const fixture=setup(async()=>{await new Promise(resolve=>setTimeout(resolve,31000));return result();});
  const job=fixture.queue.submit(fixture.project.id,fixture.identity);
  expect(fixture.queue.get(fixture.project.id,job.id).status).toBe('running');
  await new Promise(resolve=>setTimeout(resolve,31200));
  expect(fixture.queue.get(fixture.project.id,job.id).status).toBe('succeeded');
},35000);
