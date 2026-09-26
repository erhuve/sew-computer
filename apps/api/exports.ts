import { validateExport } from './export-validation';
import type { Artifact, Disclosure, ExportResult, ExportSnapshot, PatternGeometry } from '../../packages/contracts';
import { Store, type ArtifactRow } from './store';
import { ApiError, geometry as checkGeometry, id, now, objectDigest } from './validation';

export type Exporter = (snapshot:ExportSnapshot,geometry:PatternGeometry|null,assets:{artifact:Artifact;bytes:Uint8Array}[])=>Promise<ExportResult>;
export function committedPatterns(store:Store,projectId:string,revisionId:string):{artifacts:Artifact[];assets:{artifact:Artifact;bytes:Uint8Array}[];geometry:PatternGeometry} {
  const revision=store.revision(projectId,revisionId);
  const head=store.db.query('SELECT job_id FROM geometry_heads WHERE revision_id=?').get(revisionId) as {job_id:string}|null;
  if(!head) throw new ApiError(404,'No committed patterns for this revision');
  const job=store.job(projectId,head.job_id);
  if(job.status!=='succeeded'||job.revision_id!==revisionId) throw new ApiError(409,'Pattern provenance mismatch');
  const rows=store.db.query('SELECT * FROM artifacts WHERE project_id=? AND revision_id=? AND job_id=?').all(projectId,revisionId,head.job_id) as ArtifactRow[];
  const assets=rows.map(row=>{const artifact:Artifact=JSON.parse(row.json);return {artifact,bytes:store.readBlob(row.storage_key,artifact.digest,artifact.bytes)};});
  const json=assets.filter(a=>a.artifact.kind==='pattern-json');
  if(json.length!==1) throw new ApiError(409,'Committed pattern JSON missing');
  let geometry:PatternGeometry;
  try {geometry=checkGeometry(JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(json[0]!.bytes)),revision.digest);} catch {throw new ApiError(409,'Committed pattern geometry invalid');}
  return {artifacts:assets.map(a=>a.artifact),assets,geometry};
}
export class Handoff {
  private active=new Map<string,number>();
  constructor(private store:Store,private exporter?:Exporter) {}
  async build(projectId:string,revisionId:string,disclosure:Disclosure):Promise<{snapshotId:string;files:{filename:string;mime:string;url:string}[]}> {
    if(!this.exporter) throw new ApiError(503,'Export renderer is not configured');
    if((this.active.get(projectId)??0)>=2||[...this.active.values()].reduce((a,b)=>a+b,0)>=4) throw new ApiError(429,'Export queue is full');
    this.active.set(projectId,(this.active.get(projectId)??0)+1);
    try {
      const captured=this.store.transaction(()=>{
        const row=this.store.project(projectId),revision=this.store.revision(projectId,revisionId);
        if(objectDigest(revision.document)!==revision.digest) throw new ApiError(409,'Revision digest mismatch');
        const assets:{artifact:Artifact;bytes:Uint8Array}[]=[];
        let geometry:PatternGeometry|null=null;
        if(disclosure.includePatterns) {
          let committed:ReturnType<typeof committedPatterns>;
          try {committed=committedPatterns(this.store,projectId,revisionId);} catch(error) {if(error instanceof ApiError&&error.status===404) throw new ApiError(409,'Requested patterns are missing'); throw error;}
          assets.push(...committed.assets);geometry=committed.geometry;
        }
        if(disclosure.includeReferences) {
          for(const assetId of new Set(revision.document.views.map(v=>v.assetId))) {
            const ref=this.store.reference(projectId,assetId);
            const artifact:Artifact={id:ref.id,projectId,revisionId,jobId:null,kind:'reference',filename:`reference-${ref.id}.png`,mime:'image/png',digest:ref.digest,bytes:ref.bytes,classification:'screen-preview',createdAt:ref.created_at};
            assets.push({artifact,bytes:this.store.readBlob(ref.storage_key,ref.digest,ref.bytes)});
          }
        }
        const comments=(this.store.db.query('SELECT json FROM comments WHERE project_id=? AND revision_id=? ORDER BY rowid').all(projectId,revisionId) as {json:string}[]).map(r=>JSON.parse(r.json));
        const patternInventory=(this.store.db.query("SELECT a.json FROM artifacts a JOIN geometry_heads h ON h.revision_id=a.revision_id AND h.job_id=a.job_id JOIN jobs j ON j.id=a.job_id WHERE a.project_id=? AND a.revision_id=? AND j.status='succeeded' ORDER BY a.rowid").all(projectId,revisionId) as {json:string}[]).map(r=>JSON.parse(r.json) as Artifact);
        const snapshot:ExportSnapshot={id:id(),projectId,revision,document:structuredClone(revision.document),artifacts:[...patternInventory,...assets.filter(a=>a.artifact.kind==='reference').map(a=>a.artifact)],comments,disclosure:structuredClone(disclosure),createdAt:now()};
        this.store.db.query('INSERT INTO snapshots(id,project_id,revision_id,generation,json) VALUES(?,?,?,?,?)').run(snapshot.id,projectId,revisionId,row.generation,JSON.stringify(snapshot));
        return {snapshot,assets,geometry,generation:row.generation};
      });
      let result:ExportResult;
      try {result=await this.exporter(structuredClone(captured.snapshot),structuredClone(captured.geometry),captured.assets.map(a=>({artifact:structuredClone(a.artifact),bytes:Uint8Array.from(a.bytes)})));}
      catch {throw new ApiError(422,'Export failed; no delivery was installed');}
      this.store.assertReady();
      await validateExport(captured.snapshot,result,captured.assets);
      const manifest=structuredClone(result.manifest);
      const files=result.files.map(file=>({...file,bytes:Uint8Array.from(file.bytes)}));
      this.store.install(files,records=>{
        const row=this.store.project(projectId);
        if(row.generation!==captured.generation) throw new ApiError(409,'Project changed during export');
        this.store.db.query('INSERT INTO snapshot_results(snapshot_id,manifest) VALUES(?,?)').run(captured.snapshot.id,JSON.stringify(manifest));
        for(const file of records) this.store.db.query('INSERT INTO export_files(snapshot_id,filename,mime,storage_key,digest,bytes) VALUES(?,?,?,?,?,?)').run(captured.snapshot.id,file.filename,file.mime,file.storageKey,file.digest,file.bytes);
      });
      return {snapshotId:captured.snapshot.id,files:files.map(file=>({filename:file.filename,mime:file.mime,url:`/api/projects/${projectId}/exports/${captured.snapshot.id}/${file.filename}`}))};
    } finally {const count=(this.active.get(projectId)??1)-1;if(count) this.active.set(projectId,count);else this.active.delete(projectId);}
  }
}
