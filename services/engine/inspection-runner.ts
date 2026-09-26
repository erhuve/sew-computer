import { constants, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { chmod, chown, mkdtemp, open, rm } from 'node:fs/promises';
import { dirname, isAbsolute, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { InspectionEngine } from '../../apps/api/three-d-jobs';
import { executeTrusted } from './runner';

const root=dirname(fileURLToPath(import.meta.url));
const version=()=>createHash('sha256').update(process.env.SEW_ENGINE_CONTAINER||process.env.SEW_ENGINE_PYTHON||join(root,'.venv/bin/python')).update(['inspection-worker.py','garment_preview.py','quality_meshing.py','container-runtime.ts','assembly.py','meshing.py','simulation_validation.py','inspection_gltf.py','guard.py','requirements.lock','inspection-runner.ts','runner.ts'].map(name=>readFileSync(join(root,name))).reduce((all,bytes)=>Buffer.concat([all,bytes]),Buffer.alloc(0))).digest('hex');
export const runInspection:InspectionEngine=async input=>{
  const capturedVersion=version();
  if(capturedVersion!==runInspection.version)throw new Error('Inspection engine changed; restart the API');
  const python=process.env.SEW_ENGINE_PYTHON||join(root,'.venv/bin/python');
  if(!isAbsolute(python))throw new Error('Engine Python must be an absolute trusted executable');
  const attempt=await mkdtemp(join(input.outputDir,'.inspection-'));
  try {
    await chmod(attempt,0o700);
    if(process.getuid?.()===0)await chown(attempt,65534,65534);
    await executeTrusted(python,[join(root,'inspection-worker.py')],attempt,JSON.stringify({pattern:new TextDecoder('utf-8',{fatal:true}).decode(input.pattern),construction:input.construction}),input.signal,90000,attempt);
    if(version()!==capturedVersion)throw new Error('Inspection engine changed during execution');
    const read=async(name:string)=>{
      const handle=await open(join(attempt,name),constants.O_RDONLY|constants.O_NOFOLLOW);
      try {
        const stat=await handle.stat();
        if(!stat.isFile()||stat.nlink!==1||stat.size<100||stat.size>16*1024*1024)throw new Error('Invalid inspection output');
        const bytes=await handle.readFile();
        if(bytes.length!==stat.size)throw new Error('Inspection output changed during read');
        return bytes;
      } finally {await handle.close();}
    };
    return {report:await read('inspection.json'),mesh:await read('inspection.glb'),shape:await read('garment-preview.json')};
  } finally {await rm(attempt,{recursive:true,force:true});}
};
runInspection.version=version();
