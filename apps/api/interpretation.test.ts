import { afterEach, expect, test } from 'bun:test';
import { mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { assumed, emptyDocument, type Draft, type Project } from '../../packages/contracts';
import { interpretationFixture } from '../../packages/test-fixtures/interpretation';
import { rebaseAcceptedDesign } from '../../packages/contracts/interpretation';
import { buildExport } from '../../packages/tech-pack';
import { Handoff } from './exports';
import { InterpretationService, type Interpreter } from './interpretation';
import { Store } from './store';
import { id, now } from './validation';
import { previewImport } from './imports';

const stores:Store[]=[],paths:string[]=[];
afterEach(()=>{for(const store of stores.splice(0))store.close();for(const path of paths.splice(0))rmSync(path,{recursive:true,force:true});});
function setup(run?:Interpreter['run']) {
  const path=mkdtempSync(join(import.meta.dir,'.test-ai-'));paths.push(path);
  const store=new Store(path,'test-key');stores.push(store);
  const project:Project={id:id(),title:'Ocean top',headRevisionId:null,createdAt:now(),updatedAt:now()};
  const doc=emptyDocument(project.title,'Blue linen sleeveless top with contrast embroidery');
  doc.body.height={state:'known',value:1650,unit:'mm',source:'PRIVATE BODY'};
  doc.requirements=[{id:'original',text:'Keep embroidery',status:'unsupported',note:'Owner note'}];
  doc.bom=[{id:'owner-fabric',name:'My fabric',category:'fabric',specification:'Do not lose this',placement:'',quantity:'',source:'Owner'}];
  doc.callouts=[{id:'private',anchor:'/body',text:'PRIVATE CALLOUT'}];
  const draft:Draft={projectId:project.id,version:1,baseRevisionId:null,document:doc,updatedAt:now()};
  store.db.query('INSERT INTO projects(id,json,draft) VALUES(?,?,?)').run(project.id,JSON.stringify(project),JSON.stringify(draft));
  const interpreter:Interpreter={status:{available:true,provider:'test-only',model:'fixture',maxOutputTokens:6000,timeoutSeconds:120,referenceLimit:3},run:run??(async()=>({value:structuredClone(interpretationFixture),inputTokens:100,outputTokens:200}))};
  const service=new InterpretationService(store,interpreter);
  const identity={expectedVersion:1,expectedRevisionId:null};
  const propose=()=>service.propose(project.id,{...identity,consent:true,includeReferences:false});
  return {store,project,draft,service,identity,propose};
}
test('proposal minimizes private inputs and atomically publishes a revision without losing entered data',async()=>{
  let submitted='';
  const fixture=setup(async input=>{submitted=JSON.stringify(input.document);expect(input.images).toEqual([]);return {value:structuredClone(interpretationFixture),inputTokens:1,outputTokens:2};});
  const proposal=await fixture.propose();
  expect(submitted).not.toContain('PRIVATE BODY');expect(submitted).not.toContain('PRIVATE CALLOUT');expect(submitted).not.toContain('sizeLabel');
  expect(fixture.store.draft(fixture.project.id)).toEqual(fixture.draft);
  const result=fixture.service.accept(fixture.project.id,proposal.id,fixture.identity);
  expect(result.revisions).toHaveLength(1);expect(result.draft.baseRevisionId).toBe(result.revisions[0]!.id);expect(result.draft.version).toBe(2);
  expect(result.draft.document.body).toEqual(fixture.draft.document.body);
  expect(result.draft.document.callouts).toEqual(fixture.draft.document.callouts);
  expect(result.draft.document.requirements[0]).toEqual(fixture.draft.document.requirements[0]);
  expect(result.draft.document.bom[0]).toEqual(fixture.draft.document.bom[0]);
  expect(result.draft.document.interpretation?.model).toBe('fixture');
  expect(()=>fixture.service.accept(fixture.project.id,proposal.id,{expectedVersion:2,expectedRevisionId:result.draft.baseRevisionId})).toThrow('stale');
});
test('a newer client version cannot launder a stale proposal',async()=>{
  const fixture=setup(),proposal=await fixture.propose();
  const newer={...fixture.draft,version:2,document:{...fixture.draft.document,brief:'New intent'}};
  fixture.store.db.query('UPDATE projects SET draft=? WHERE id=?').run(JSON.stringify(newer),fixture.project.id);
  expect(()=>fixture.service.accept(fixture.project.id,proposal.id,{...fixture.identity,expectedVersion:2})).toThrow('stale');
  expect(fixture.store.draft(fixture.project.id)).toEqual(newer);
  expect(fixture.service.latest(fixture.project.id)?.id).toBe(proposal.id);
});
test('malformed, authority-forging and executable output never changes the draft',async()=>{
  for(const value of [{...interpretationFixture,body:{height:assumed(1800)}},{...interpretationFixture,execute:'rm -rf /'}, {...interpretationFixture,garment:{...interpretationFixture.garment,length:{state:'known',value:600,unit:'mm',source:'measured from photo'}}}]) {
    const fixture=setup(async()=>({value,inputTokens:null,outputTokens:null}));
    await expect(fixture.propose()).rejects.toThrow();
    expect(fixture.store.draft(fixture.project.id)).toEqual(fixture.draft);
    expect(fixture.service.latest(fixture.project.id)).toBeNull();
  }
});
test('provider outage preserves work and concurrent requests are bounded',async()=>{
  let release!:(value:any)=>void;
  const fixture=setup(()=>new Promise(resolve=>{release=resolve;}));
  const pending=fixture.propose();
  await expect(fixture.propose()).rejects.toThrow('already running');
  release({value:interpretationFixture,inputTokens:1,outputTokens:1});await pending;
  const broken=setup(async()=>{throw new Error('provider secret should not leak');});
  await expect(broken.propose()).rejects.toThrow('could not be reached');
  expect(broken.store.draft(broken.project.id)).toEqual(broken.draft);
});
test('deleted project cannot be resurrected by late inference',async()=>{
  let release!:(value:any)=>void;
  const fixture=setup(()=>new Promise(resolve=>{release=resolve;}));
  const pending=fixture.propose();
  fixture.service.cancel(fixture.project.id);fixture.store.deleteProject(fixture.project.id);
  release({value:interpretationFixture,inputTokens:1,outputTokens:1});
  await expect(pending).rejects.toThrow();
  expect(()=>fixture.store.project(fixture.project.id)).toThrow('not found');
  expect(fixture.store.db.query('SELECT COUNT(*) AS count FROM ai_proposals').get()).toEqual({count:0});
  expect(fixture.store.db.query('SELECT project_id FROM ai_requests').get()).toEqual({project_id:'deleted'});
});
test('hourly usage survives service restart and owner-entered measurements survive proposals',async()=>{
  const fixture=setup();
  fixture.draft.document.garment.length={state:'known',value:710,unit:'mm',source:'Owner'};
  fixture.store.db.query('UPDATE projects SET draft=? WHERE id=?').run(JSON.stringify(fixture.draft),fixture.project.id);
  const proposal=await fixture.propose();
  expect(proposal.document.garment.length).toEqual(fixture.draft.document.garment.length);
  for(let count=0;count<11;count++)fixture.store.db.query('INSERT INTO ai_requests(id,project_id,at) VALUES(?,?,?)').run(id(),'deleted',Date.now());
  await expect(fixture.propose()).rejects.toThrow('Hourly limit');
  const fresh=new InterpretationService(fixture.store,{status:{available:true,provider:'test',model:'fixture',maxOutputTokens:6000,timeoutSeconds:120,referenceLimit:3},run:async()=>{throw new Error('Must not call provider');}});
  await expect(fresh.propose(fixture.project.id,{...fixture.identity,includeReferences:false,consent:true})).rejects.toThrow('Hourly limit');
});
test('edits arriving during acceptance merge with generated rows instead of erasing the design',()=>{
  const submitted=emptyDocument('Initial','Initial brief');
  submitted.bom=[{id:'existing',name:'Owner fabric',category:'fabric',specification:'',placement:'',quantity:'',source:'Owner'}];
  const accepted=structuredClone(submitted);
  accepted.garment={family:'shirt',length:assumed(600),ease:assumed(80),flare:1.1};
  accepted.bom.push({...submitted.bom[0]!,id:'ai',name:'AI trim'});
  const latest=structuredClone(submitted);latest.brief='New local brief';latest.bom=[];
  const rebased=rebaseAcceptedDesign(accepted,submitted,latest);
  expect(rebased.brief).toBe('New local brief');expect(rebased.garment.family).toBe('shirt');expect(rebased.bom.map(row=>row.id)).toEqual(['ai']);
});
test('version-one databases migrate without changing owner drafts',()=>{
  const fixture=setup(),root=fixture.store.root;
  fixture.store.db.exec('DROP TABLE ai_proposals; DROP TABLE ai_requests; PRAGMA user_version=1;');
  fixture.store.close();
  const migrated=new Store(root,'test-key');stores.push(migrated);
  expect(migrated.db.query('PRAGMA user_version').get()).toEqual({user_version:5});
  expect(migrated.draft(fixture.project.id)).toEqual(fixture.draft);
});
test('AI provenance survives PDF/manifest export and is not editable through manifest import',async()=>{
  const fixture=setup(),proposal=await fixture.propose();
  const state=fixture.service.accept(fixture.project.id,proposal.id,fixture.identity);
  const handoff=new Handoff(fixture.store,buildExport);
  const result=await handoff.build(fixture.project.id,state.project.headRevisionId!,{includeBody:false,includeReferences:false,includePatterns:false});
  const stored=fixture.store.db.query('SELECT manifest FROM snapshot_results WHERE snapshot_id=?').get(result.snapshotId) as {manifest:string};
  const manifest=JSON.parse(stored.manifest);
  expect(manifest.sections.overview.interpretation.model).toBe('fixture');
  expect(manifest.sections.bodyInputs).toBeUndefined();
  manifest.sections.overview.interpretation.model='forged';
  const imported=previewImport(fixture.store,fixture.project.id,manifest);
  expect(imported.changes).toEqual([]);
  expect(fixture.store.draft(fixture.project.id).document.interpretation?.model).toBe('fixture');
});
