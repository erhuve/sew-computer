import type { Artifact, ExportResult, ExportSnapshot } from '../../packages/contracts';
import { ApiError, checkFile, cleanObject, hash, objectDigest } from './validation';

function require(condition:unknown,message:string):asserts condition {if(!condition)throw new ApiError(422,message);}
export function validateExport(snapshot:ExportSnapshot,result:ExportResult,assets:{artifact:Artifact;bytes:Uint8Array}[]):void {
  cleanObject(result.manifest);
  const manifest=result.manifest;
  const topKeys=new Set(['schemaVersion','snapshotId','revisionId','projectId','manifestDigest','sections','revisionNumber','parentRevisionId','revisionCreatedAt','snapshotCreatedAt','inputDigest','renderer']);
  require(Object.keys(manifest).every(k=>topKeys.has(k)),'Renderer returned unknown manifest fields');
  require(manifest.schemaVersion===1&&manifest.snapshotId===snapshot.id&&manifest.revisionId===snapshot.revision.id&&manifest.projectId===snapshot.projectId,'Renderer returned mismatched manifest identity');
  const {manifestDigest,...content}=manifest;
  require(manifestDigest===objectDigest(content),'Renderer returned invalid manifest digest');
  require(manifest.sections&&typeof manifest.sections==='object'&&!Array.isArray(manifest.sections),'Invalid manifest sections');
  const sections=manifest.sections as Record<string,unknown>,doc=snapshot.document;
  const sectionKeys=new Set(['overview','requirements','materials','bom','bodyInputs','finishedMeasurements','construction','patternInventory','review','exportDisclosure']);
  require(Object.keys(sections).every(k=>sectionKeys.has(k)),'Renderer returned unknown manifest sections');
  const projection:Record<string,unknown>={overview:{title:doc.title,brief:doc.brief,sizeLabel:doc.sizeLabel,garment:doc.garment,...(doc.interpretation?{interpretation:doc.interpretation}:{})},requirements:doc.requirements,materials:doc.bom.filter(r=>r.category==='fabric'||r.category==='lining'),bom:doc.bom,finishedMeasurements:doc.poms,construction:doc.construction,review:{callouts:doc.callouts,comments:snapshot.comments}};
  for(const [key,value] of Object.entries(projection))require(objectDigest(sections[key]??null)===objectDigest(value),'Renderer altered the saved editable projection');
  if(snapshot.disclosure.includeBody)require(objectDigest(sections.bodyInputs??null)===objectDigest(doc.body),'Renderer omitted disclosed body fields');
  else require(!Object.hasOwn(sections,'bodyInputs'),'Renderer disclosed omitted body inputs');
  const disclosure=sections.exportDisclosure as Record<string,unknown>|undefined;
  require(disclosure&&typeof disclosure==='object','Renderer omitted export disclosure');
  for(const [key,value] of Object.entries(snapshot.disclosure))require(disclosure[key]===value,'Renderer altered disclosure flags');
  require(Array.isArray(disclosure.omissions),'Renderer omitted disclosure notes');
  const inventory=sections.patternInventory as Record<string,unknown>|undefined;
  require(inventory&&typeof inventory==='object'&&Array.isArray(inventory.artifacts),'Renderer omitted pattern inventory');
  if(!snapshot.disclosure.includeReferences){
    require(!Object.hasOwn(inventory,'references')&&!Object.hasOwn(inventory,'views'),'Renderer disclosed hidden source references');
    require(!Array.isArray(inventory.referenceArtifacts)||inventory.referenceArtifacts.length===0,'Renderer disclosed hidden reference artifacts');
  }
  if(!snapshot.disclosure.includePatterns)require(!Object.hasOwn(inventory,'panels')&&!Object.hasOwn(inventory,'geometry'),'Renderer disclosed hidden pattern geometry');
  require(Array.isArray(result.files)&&result.files.length>=3&&result.files.length<=100,'Invalid delivery inventory');
  const names=new Set<string>();let total=0;
  for(const file of result.files){
    checkFile(file.filename,file.bytes,file.mime);
    require(!names.has(file.filename),'Duplicate delivery filename');names.add(file.filename);
    total+=file.bytes.length;require(total<=128*1024*1024,'Export exceeds its byte limit');
  }
  const manifests=result.files.filter(f=>f.mime==='application/json'&&objectDigest(JSON.parse(new TextDecoder().decode(f.bytes)))===objectDigest(manifest));
  require(manifests.length===1,'Delivery requires one exact editable manifest');
  const deliveryFile=result.files.find(f=>f.filename==='delivery.json'&&f.mime==='application/json');
  require(deliveryFile,'Delivery checksum file missing');
  const delivery=JSON.parse(new TextDecoder().decode(deliveryFile.bytes));
  require(delivery.snapshotId===snapshot.id&&delivery.revisionId===snapshot.revision.id&&delivery.projectId===snapshot.projectId&&delivery.manifestDigest===manifestDigest,'Delivery identities do not match');
  require(Array.isArray(delivery.files)&&delivery.files.length===result.files.length-1,'Delivery digest inventory mismatch');
  const checks=new Map<string,Record<string,unknown>>(delivery.files.map((f:Record<string,unknown>)=>[f.filename,f]));
  require(checks.size===delivery.files.length&&!checks.has('delivery.json'),'Delivery checksums must not include themselves or duplicate filenames');
  for(const file of result.files.filter(f=>f!==deliveryFile)){
    const check=checks.get(file.filename);
    require(check&&check.digest===hash(file.bytes)&&check.mime===file.mime&&(check.bytes??check.size)===file.bytes.length,'Delivery file digest mismatch');
  }
  const matched=new Set<string>();
  for(const asset of assets){
    const file=result.files.find(f=>!matched.has(f.filename)&&f.mime===asset.artifact.mime&&hash(f.bytes)===asset.artifact.digest);
    require(file,'Renderer omitted or modified requested artifact bytes');matched.add(file.filename);
  }
  const derived=result.files.filter(f=>!matched.has(f.filename));
  require(derived.length===3&&derived.filter(f=>f.mime==='application/pdf').length===1&&derived.includes(deliveryFile)&&derived.includes(manifests[0]!),'Only the tech pack, manifest, delivery checksums and verified requested artifacts may be delivered');
}
