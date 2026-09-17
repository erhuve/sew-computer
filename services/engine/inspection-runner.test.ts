import { expect, test } from 'bun:test';
import { mkdtemp, readdir, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { shirtDocument } from '../../packages/test-fixtures/shirt';
import { buildExport } from '../../packages/tech-pack';
import { createApi } from '../../apps/api';
import { validateInspection } from '../../apps/api/inspection-validation';
import { hash, objectDigest } from '../../apps/api/validation';
import { runEngine } from './runner';
import { runInspection } from './inspection-runner';

test('private durable inspection runs the real restricted worker and rejects display/source corruption',async()=>{
  const root=await mkdtemp(join(import.meta.dir,'.test-inspection-'));
  const document=shirtDocument(),origin='https://inspection.test';
  let api:ReturnType<typeof createApi>|undefined;
  try {
    const source=await runEngine({document,inputDigest:objectDigest(document),outputDir:root,signal:new AbortController().signal});
    const options={dataDir:join(root,'data'),allowedOrigins:[origin],authKey:'test-key',engine:async()=>source,inspectionEngine:runInspection,exporter:buildExport};
    api=createApi(options);
    let cookie='';
    const request=(method:string,path:string,body?:unknown,headers:Record<string,string>={})=>api!.request(path,{method,headers:{Origin:origin,Cookie:cookie,'Content-Type':'application/json',...headers},...(body===undefined?{}:{body:JSON.stringify(body)})});
    expect((await request('GET','/projects/missing/three-d/latest')).status).toBe(401);
    const login=await request('POST','/auth/login',{key:'test-key'});cookie=login.headers.get('set-cookie')!.split(';')[0]!;
    const created=await (await request('POST','/projects',{title:document.title})).json(),projectId=created.project.id;
    const saved=await (await request('PUT',`/projects/${projectId}/draft`,{expectedVersion:created.draft.version,expectedRevisionId:null,document})).json();
    const published=await (await request('POST',`/projects/${projectId}/revisions`,{expectedVersion:saved.version,expectedRevisionId:null})).json(),revisionId=published.project.headRevisionId;
    const geometryJob=await (await request('POST',`/projects/${projectId}/jobs`,{revisionId,requestId:'source-pattern'})).json();
    const wait=async(path:string)=>{
      for(let attempt=0;attempt<900;attempt++) {
        const job=await (await request('GET',path)).json();
        if(!['queued','running'].includes(job.status))return job;
        await new Promise(resolve=>setTimeout(resolve,100));
      }
      throw new Error('Timed out waiting for private worker');
    };
    expect((await wait(`/projects/${projectId}/jobs/${geometryJob.id}`)).status).toBe('succeeded');
    const submitted=await request('POST',`/projects/${projectId}/three-d`,{revisionId,requestId:'inspect-pattern'});
    expect(submitted.status).toBe(202);
    const job=await submitted.json(),base=`/projects/${projectId}/three-d/${job.id}`;
    expect((await request('GET',`${base}/mesh`)).status).toBe(409);
    const complete=await wait(base);
    expect(complete.status).toBe('succeeded');expect(complete.result.fabricInstances).toBe(24);
    expect(complete.result.classification).toBe('placement-inspection');
    expect((await request('GET',`${base}/mesh`,undefined,{Cookie:''})).status).toBe(401);
    const other=await (await request('POST','/projects',{title:'Other project'})).json();
    expect((await request('GET',`/projects/${other.project.id}/three-d/${job.id}/mesh`)).status).toBe(404);
    expect((await request('POST',`${base}/cancel`,{},{Origin:'https://wrong.test'})).status).toBe(403);
    const reportResponse=await request('GET',`${base}/report`),meshResponse=await request('GET',`${base}/mesh`);
    expect(meshResponse.headers.get('Cache-Control')).toBe('no-store');expect(meshResponse.headers.get('Content-Type')).toBe('model/gltf-binary');
    const report=new Uint8Array(await reportResponse.arrayBuffer()),mesh=new Uint8Array(await meshResponse.arrayBuffer());
    api.close();api=createApi(options);
    expect(new Uint8Array(await (await request('GET',`${base}/mesh`)).arrayBuffer())).toEqual(mesh);
    for(const includePatterns of [false,true]) {
      const response=await request('POST',`/projects/${projectId}/exports`,{revisionId,disclosure:{includeBody:true,includeReferences:false,includePatterns}});
      expect(response.status).toBe(201);
      const exported=await response.json();
      expect(exported.files.some((file:{filename:string;mime:string})=>file.filename.includes('inspection')||file.mime==='model/gltf-binary')).toBe(false);
      const manifestFile=exported.files.find((file:{filename:string})=>file.filename==='manifest.json');
      const manifest=await (await request('GET',manifestFile.url.replace(/^\/api/,''))).text();
      expect(manifest).not.toContain(job.id);expect(manifest).not.toContain('inspection.glb');
    }
    expect((await request('POST',`/projects/${projectId}/exports`,{revisionId,disclosure:{includeBody:true,includeReferences:false,includePatterns:true,include3D:true}})).status).toBe(422);
    const pattern=source.files.find(file=>file.kind==='pattern-json')!.bytes;
    const validate=(reportBytes=report,meshBytes=mesh)=>validateInspection(reportBytes,meshBytes,source.geometry,hash(pattern),objectDigest(document.garment.design));
    const inspection=validate();expect(inspection.instances.length).toBe(24);
    expect((inspection.executionControls as {uid:number}).uid).not.toBe(0);
    const changed=structuredClone(inspection);changed.templates[0]!.triangles=changed.templates[0]!.triangles.map(()=>changed.templates[0]!.triangles[0]!);
    expect(()=>validate(Buffer.from(JSON.stringify(changed)))).toThrow('topology');
    const mirrored=structuredClone(inspection);mirrored.instances[0]!.mirrorX=!mirrored.instances[0]!.mirrorX;
    expect(()=>validate(Buffer.from(JSON.stringify(mirrored)))).toThrow('handedness');
    const badSource=structuredClone(inspection);badSource.templates[0]!.sourceWeights[0]=[{point:1,weight:1}];
    expect(()=>validate(Buffer.from(JSON.stringify(badSource)))).toThrow('geometry changed');
    const mutateModel=(change:(model:any)=>void)=>{
      const original=Buffer.from(mesh),metadataLength=original.readUInt32LE(12),model=JSON.parse(original.subarray(20,20+metadataLength).toString());change(model);
      const encoded=Buffer.from(JSON.stringify(model)),metadata=Buffer.concat([encoded,Buffer.alloc((-encoded.length%4+4)%4,32)]),binary=original.subarray(20+metadataLength);
      const mutated=Buffer.alloc(20+metadata.length+binary.length);original.copy(mutated,0,0,20);mutated.writeUInt32LE(mutated.length,8);mutated.writeUInt32LE(metadata.length,12);metadata.copy(mutated,20);binary.copy(mutated,20+metadata.length);
      const changedReport={...inspection,displayArtifact:{filename:'inspection.glb',sha256:hash(mutated),bytes:mutated.length}};
      return ()=>validate(Buffer.from(JSON.stringify(changedReport)),mutated);
    };
    expect(mutateModel(model=>{model.buffers[0].uri='https://untrusted.test/private';})).toThrow();
    expect(mutateModel(model=>{model.accessors[0].max=[1e50,1e50,1e50];})).toThrow('display bounds');
    expect(mutateModel(model=>{model.extensionsRequired=['untrusted-extension'];})).toThrow();
    expect((await (await request('GET',`/projects/${projectId}/three-d/latest?revisionId=${revisionId}`)).json()).sourceCurrent).toBe(true);
    const regeneration=await (await request('POST',`/projects/${projectId}/jobs`,{revisionId,requestId:'replace-source'})).json();
    expect((await wait(`/projects/${projectId}/jobs/${regeneration.id}`)).status).toBe('succeeded');
    expect((await (await request('GET',`/projects/${projectId}/three-d/latest?revisionId=${revisionId}`)).json()).sourceCurrent).toBe(false);
    expect((await request('GET',`${base}/mesh`)).status).toBe(200);
    expect((await request('DELETE',`/projects/${projectId}`)).status).toBe(204);
    expect((await request('GET',`${base}/mesh`)).status).toBe(404);
    expect(await readdir(join(root,'data','blobs'))).toEqual([]);
  } finally {api?.close();await rm(root,{recursive:true,force:true});}
},120000);
