import { z } from 'zod';
import { Id, canonical, mm, type GarmentDocument, type ImportChange, type ImportPreview, type ProjectState, type ExportSnapshot } from '../../packages/contracts';
import { Store } from './store';
import { ApiError, cleanObject, document, id, now, objectDigest } from './validation';

type Operation = { path:string; keys:string[]; after:unknown; remove:boolean };
type Row = Record<string,unknown>;
const own = (value:object,key:string) => Object.prototype.hasOwnProperty.call(value,key);
const isObject = (v:unknown):v is Row => !!v && typeof v==='object' && !Array.isArray(v);
const same = (a:unknown,b:unknown):boolean => canonical(normalize(a))===canonical(normalize(b));
function normalize(v:unknown):unknown {
  if (Array.isArray(v)) return v.map(normalize);
  if (isObject(v)) {
    if ((v.state==='known'||v.state==='assumed') && typeof v.value==='number' && ['mm','cm','in'].includes(String(v.unit))) return {...v,value:Number(mm(v as any)!.toFixed(6)),unit:'mm'};
    return Object.fromEntries(Object.entries(v).map(([k,x])=>[k,normalize(x)]));
  }
  return v;
}
function read(value:unknown,keys:string[]):unknown {
  for (const key of keys) {
    if (Array.isArray(value)) value=value.find(v=>isObject(v)&&v.id===key);
    else if (isObject(value)) value=value[key];
    else return undefined;
  }
  return value;
}
function write(value:unknown,keys:string[],after:unknown,remove:boolean) {
  const parent=read(value,keys.slice(0,-1)), key=keys.at(-1)!;
  if (Array.isArray(parent)) {
    const index=parent.findIndex(v=>isObject(v)&&v.id===key);
    if (remove) { if(index>=0) parent.splice(index,1); }
    else if(index>=0) parent[index]=structuredClone(after);
    else parent.push(structuredClone(after));
  } else if (isObject(parent)) {
    if (remove) delete parent[key]; else parent[key]=structuredClone(after);
  } else throw new ApiError(409,'Imported field no longer exists');
}
const topSchema=z.object({schemaVersion:z.literal(1),snapshotId:Id,revisionId:Id,projectId:Id,manifestDigest:z.string().regex(/^[a-f0-9]{64}$/),sections:z.record(z.string(),z.unknown())}).passthrough();
const allowedSections=new Set(['overview','requirements','materials','bom','bodyInputs','finishedMeasurements','construction','patternInventory','review','exportDisclosure']);
const fields:Record<string,string[]>={overview:['title','brief','sizeLabel','garment'],garment:['family','length','ease','flare'],body:['height','bust','waist','hip','shoulder']};

export function previewImport(store:Store,projectId:string,raw:unknown):ImportPreview {
  cleanObject(raw);
  const manifest=topSchema.parse(raw);
  if(manifest.projectId!==projectId) throw new ApiError(404,'Export snapshot not found');
  for(const name of Object.keys(manifest.sections)) if(!allowedSections.has(name)) throw new ApiError(422,'Unknown manifest section');
  return store.transaction(()=>{
    const snapshotRow=store.snapshot(projectId,manifest.snapshotId);
    const snapshot:ExportSnapshot=JSON.parse(snapshotRow.json);
    const result=store.db.query('SELECT manifest FROM snapshot_results WHERE snapshot_id=?').get(snapshot.id) as {manifest:string}|null;
    if(!result || snapshot.revision.id!==manifest.revisionId) throw new ApiError(404,'Completed export snapshot not found');
    const baselineManifest=JSON.parse(result.manifest);
    for(const key of Object.keys(manifest))if(!own(baselineManifest,key))throw new ApiError(422,'Unknown manifest field');
    if(baselineManifest.manifestDigest!==manifest.manifestDigest) throw new ApiError(409,'Source manifest digest does not match the stored export');
    const current=store.draft(projectId), baseline=snapshot.document;
    const operations=new Map<string,Operation>();
    const changes=new Map<string,ImportChange>();
    const warnings=['Only editable document fields are proposed. Imported artifacts, comments, actors, evidence and disclosure claims cannot establish authority.','An omitted field or section leaves current data unchanged. Explicit empty arrays and removed exported rows are proposed as visible deletions.'];
    const propose=(keys:string[],incoming:unknown,remove=false)=>{
      const before=read(current.document,keys), original=read(baseline,keys);
      if(same(incoming,original) || same(incoming,before)) return;
      const path='/'+keys.map(k=>k.replaceAll('~','~0').replaceAll('/','~1')).join('/');
      for(const operation of operations.values())if(operation.path!==path&&(operation.path.startsWith(path+'/')||path.startsWith(operation.path+'/')))throw new ApiError(422,'Overlapping material and BOM edits require reconciliation');
      const previous=operations.get(path);
      if(previous && (!same(previous.after,incoming)||previous.remove!==remove)) throw new ApiError(422,'Conflicting material and BOM edits');
      operations.set(path,{path,keys,after:incoming??null,remove});
      changes.set(path,{path,before:before??null,after:incoming??null,baseline:original??null,conflict:remove || !same(before,original)});
    };
    const compareObject=(keys:string[],incoming:unknown,allowed?:string[])=>{
      if(!isObject(incoming)) throw new ApiError(422,'Expected an editable object');
      const source=read(baseline,keys);
      if(!isObject(source)) {propose(keys,incoming);return;}
      if (keys.length===2 && Array.isArray(read(baseline,keys.slice(0,-1))) && read(current.document,keys)===undefined) {
        const restored={...structuredClone(source),...incoming};
        for (const key of Object.keys(incoming)) if(!own(source,key)) throw new ApiError(422,'Unknown editable field');
        propose(keys,restored); return;
      }
      for(const [key,value] of Object.entries(incoming)) {
        if(allowed && !allowed.includes(key)) throw new ApiError(422,'Unknown editable field');
        if(!own(source,key)) throw new ApiError(422,'Unknown editable field');
        if(key==='id' && source.id!==value) throw new ApiError(422,'Row identities are not editable');
        if(isObject(value)&&isObject(source[key])&&!own(value,'state')&&!own(source[key] as Row,'state')) compareObject([...keys,key],value);
        else propose([...keys,key],value);
      }
    };
    const compareRows=(field:keyof GarmentDocument,incoming:unknown,filter?:(row:Row)=>boolean)=>{
      if(!Array.isArray(incoming) || incoming.length>80 || incoming.some(v=>!isObject(v)||!Id.safeParse(v.id).success)) throw new ApiError(422,'Invalid editable rows');
      if(new Set(incoming.map(v=>(v as Row).id)).size!==incoming.length) throw new ApiError(422,'Duplicate imported row ID');
      const source=(baseline[field] as unknown as Row[]).filter(v=>!filter||filter(v));
      const byId=new Map(source.map(row=>[row.id,row]));
      for(const row of incoming as Row[]) {
        if(filter && !filter(row)) throw new ApiError(422,'Materials must be fabric or lining');
        const rowId=row.id as string;
        if(!byId.has(rowId)) {
          if((baseline[field] as unknown as Row[]).some(r=>r.id===rowId)) throw new ApiError(422,'Material ID belongs to a different BOM category');
          propose([field,rowId],row);
        } else compareObject([field,rowId],row);
      }
      const incomingIds=new Set((incoming as Row[]).map(row=>row.id));
      for(const row of source) if(!incomingIds.has(row.id)) propose([field,row.id as string],undefined,true);
    };
    const sections=manifest.sections;
    if(own(sections,'overview')) {
      if(!isObject(sections.overview)) throw new ApiError(422,'Invalid overview');
      for(const key of Object.keys(sections.overview)) {
        if(key==='interpretation')continue;
        if(!fields.overview!.includes(key)) throw new ApiError(422,'Unknown overview field');
        if(key==='garment') compareObject(['garment'],sections.overview[key],fields.garment);
        else propose([key],sections.overview[key]);
      }
    }
    for(const [section,field] of [['requirements','requirements'],['bom','bom'],['finishedMeasurements','poms'],['construction','construction']] as const) if(own(sections,section)) compareRows(field,sections[section]);
    if(own(sections,'materials')) compareRows('bom',sections.materials,row=>row.category==='fabric'||row.category==='lining');
    if(own(sections,'bodyInputs')) {
      if(snapshot.disclosure.includeBody && own(baselineManifest.sections,'bodyInputs')) compareObject(['body'],sections.bodyInputs,fields.body);
      else warnings.push('Body inputs were not disclosed in this export; incoming body fields were ignored.');
    }
    if(own(sections,'review')) {
      if(!isObject(sections.review)) throw new ApiError(422,'Invalid review section');
      if(own(sections.review,'callouts')) compareRows('callouts',sections.review.callouts);
    }
    const candidate=structuredClone(current.document);
    for(const operation of operations.values()) write(candidate,operation.keys,operation.after,operation.remove);
    document(candidate); store.checkReferences(projectId,candidate);
    const preview:ImportPreview={id:id(),projectId,baseRevisionId:current.baseRevisionId,baseVersion:current.version,changes:[...changes.values()],warnings};
    const stored={operations:[...operations.values()],draftDigest:objectDigest(current.document),generation:store.project(projectId).generation};
    store.db.query('INSERT INTO import_previews(id,project_id,json,operations) VALUES(?,?,?,?)').run(preview.id,projectId,JSON.stringify(preview),JSON.stringify(stored));
    return preview;
  });
}

export function acceptImport(store:Store,projectId:string,previewId:string,identity:{expectedVersion:number;expectedRevisionId:string|null;resolutions?:Record<string,'keep'|'incoming'>}):ProjectState {
  return store.transaction(()=>{
    const project=store.project(projectId);
    const row=store.db.query('SELECT * FROM import_previews WHERE id=? AND project_id=?').get(previewId,projectId) as {json:string;operations:string;accepted:number}|null;
    if(!row) throw new ApiError(404,'Import preview not found');
    if(row.accepted) throw new ApiError(409,'Import preview was already accepted');
    const preview:ImportPreview=JSON.parse(row.json);
    const stored=JSON.parse(row.operations) as {operations:Operation[];draftDigest:string;generation:number};
    const current=store.checkIdentity(project,identity.expectedVersion,identity.expectedRevisionId);
    if(current.version!==preview.baseVersion || current.baseRevisionId!==preview.baseRevisionId || objectDigest(current.document)!==stored.draftDigest || project.generation!==stored.generation) throw new ApiError(409,'Import preview is stale; preview it again',{latest:current});
    for(const key of Object.keys(identity.resolutions??{})) if(!preview.changes.some(change=>change.path===key)) throw new ApiError(422,'Unknown conflict resolution path');
    const next=structuredClone(current.document);
    for(const operation of stored.operations) {
      const change=preview.changes.find(v=>v.path===operation.path)!;
      const resolution=identity.resolutions?.[operation.path];
      if(change.conflict && !resolution) throw new ApiError(409,'Explicit import conflict decisions are required',{conflicts:preview.changes.filter(c=>c.conflict)});
      if(resolution!=='keep') write(next,operation.keys,operation.after,operation.remove);
    }
    const validated=document(next); store.checkReferences(projectId,validated);
    const date=now(), draft={...current,document:validated,version:current.version+1,updatedAt:date};
    store.db.query('UPDATE projects SET draft=?,json=? WHERE id=?').run(JSON.stringify(draft),JSON.stringify({...JSON.parse(project.json),title:validated.title,updatedAt:date}),projectId);
    store.db.query('UPDATE import_previews SET accepted=1 WHERE id=?').run(previewId);
    return store.state(projectId);
  });
}
