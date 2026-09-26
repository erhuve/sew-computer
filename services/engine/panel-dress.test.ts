import {expect,test} from 'bun:test';
import {mkdtemp,rm,writeFile,mkdir} from 'node:fs/promises';
import {join} from 'node:path';
import {assumed,canonical} from '../../packages/contracts';
import {defaultPanelDress,type PanelDress} from '../../packages/contracts/dress';
import {validateDesignGeometry} from '../../packages/contracts/design';
import {startingDocument} from '../../packages/contracts/starting-designs';
import {objectDigest,hash} from '../../apps/api/validation';
import {validateInspection} from '../../apps/api/inspection-validation';
import {validateGarmentPreview} from '../../apps/api/garment-preview-validation';
import {runEngine} from './runner';
import {runInspection} from './inspection-runner';

const variants:{name:string;design:PanelDress;length:number;flare:number}[]=[
 {name:'flared',design:{...defaultPanelDress,neckline:'v'},length:1250,flare:1.7},
 {name:'gathered',design:{...defaultPanelDress,skirtStyle:'gathered',neckline:'square',bodiceLengthMm:365,skirtFullness:1.8,sleeves:'short'},length:940,flare:1.1},
 {name:'gathered-to-tiered',design:{...defaultPanelDress,skirtStyle:'tiered',neckline:'v',bodiceLengthMm:365,skirtFullness:1.8,sleeves:'short'},length:940,flare:1.1},
 {name:'tiered',design:{...defaultPanelDress,skirtStyle:'tiered',bodiceLengthMm:510,skirtFullness:1.4,tierFullness:1.4,sleeves:'long',sleeveLengthMm:550},length:1300,flare:1.05},
];
for(const variant of variants)test(`panel dress ${variant.name}: source, assembly and preview preserve independent construction`,async()=>{
 const outputDir=await mkdtemp(join(import.meta.dir,'.test-panel-dress-'));
 try {
  const doc=startingDocument('dress');doc.garment.design=variant.design;doc.garment.length=assumed(variant.length);doc.garment.flare=variant.flare;
  const result=await runEngine({document:doc,inputDigest:objectDigest(doc),outputDir,signal:new AbortController().signal});
  validateDesignGeometry(doc,result.geometry);
  const source=result.files.find(f=>f.filename==='pattern.json')!.bytes;
  const preview=await runInspection({pattern:source,construction:variant.design,outputDir,signal:new AbortController().signal});
  const shape=validateGarmentPreview(preview.shape!,result.geometry,hash(source),objectDigest(variant.design));
  const inspection=validateInspection(preview.report,preview.mesh,result.geometry,hash(source),objectDigest(variant.design));
  expect(shape.pieces.length).toBe(variant.design.skirtStyle==='tiered'?14:variant.design.sleeves!=='none'?10:8);
  expect(inspection.instances.length).toBe(shape.pieces.length);expect(shape.acceptedSimulation).toBe(false);
  expect(shape.buttons).toHaveLength(0);expect(shape.pieces.some(p=>/collar|frill|placket|cuff/.test(p.templateId))).toBe(false);
  expect(result.geometry.warnings.some(w=>w.includes('Center-back'))).toBe(true);
  const wrong=structuredClone(result.geometry);wrong.panels[0]!.points[1]![1]+=10;
  expect(()=>validateDesignGeometry(doc,wrong)).toThrow();
  const badSeam=structuredClone(result.geometry);badSeam.drafting!.assembly[0]!.ratio=1.5;
  expect(()=>validateDesignGeometry(doc,badSeam)).toThrow();
  if(variant.name==='gathered')await expect(runInspection({pattern:Buffer.from(JSON.stringify(wrong)),construction:variant.design,outputDir,signal:new AbortController().signal})).rejects.toThrow();
  const evidence=join(import.meta.dir,'../../.planning/silhouette-variants',variant.name);await mkdir(evidence,{recursive:true});
  await writeFile(join(evidence,'pattern.json'),source);await writeFile(join(evidence,'shape.json'),preview.shape!);await writeFile(join(evidence,'document.json'),canonical(doc));
 } finally {await rm(outputDir,{recursive:true,force:true});}
},180000);
