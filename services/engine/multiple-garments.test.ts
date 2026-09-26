import {expect,test} from 'bun:test';
import {mkdtemp,rm,writeFile,mkdir} from 'node:fs/promises';
import {join} from 'node:path';
import {emptyDocument,assumed,canonical} from '../../packages/contracts';
import {defaultDressDesign,defaultSkirtDesign,designCoverage,validateDesignGeometry} from '../../packages/contracts/design';
import {applySample,sampleSizes,requiredBodyFields} from '../../packages/contracts/sizing';
import {startingDocument} from '../../packages/contracts/starting-designs';
import {objectDigest,hash,geometry as validateApiGeometry} from '../../apps/api/validation';
import {validateInspection} from '../../apps/api/inspection-validation';
import {validateGarmentPreview} from '../../apps/api/garment-preview-validation';
import {garmentFlats} from '../../packages/contracts/flats';
import {runEngine} from './runner';
import {runInspection} from './inspection-runner';

for(const family of ['dress','skirt'] as const)test(`${family}: custom source pattern, physical pieces, preview and changed dimensions`,async()=>{
  const outputDir=await mkdtemp(join(import.meta.dir,'.test-families-'));
  try {
    const doc=applySample(emptyDocument(`Synthetic ${family}`),2);
    doc.garment={family,length:assumed(family==='dress'?1073:713),ease:assumed(93),flare:1.37,design:structuredClone(family==='dress'?defaultDressDesign:defaultSkirtDesign)};
    doc.requirements=[{id:'body',text:`Relaxed ${family}`,status:'supported',feature:'body',note:'Synthetic test'}];
    const pattern=await runEngine({document:doc,inputDigest:objectDigest(doc),outputDir,signal:new AbortController().signal});
    validateApiGeometry(pattern.geometry,objectDigest(doc));
    expect(pattern.geometry.family).toBe(family);expect(designCoverage(doc,pattern.geometry).status).toBe('drafted');
    const source=pattern.files.find(file=>file.filename==='pattern.json')!.bytes;
    const preview=await runInspection({pattern:source,construction:doc.garment.design!,outputDir,signal:new AbortController().signal});
    const inspection=validateInspection(preview.report,preview.mesh,pattern.geometry,hash(source),objectDigest(doc.garment.design));
    const shape=validateGarmentPreview(preview.shape!,pattern.geometry,hash(source),objectDigest(doc.garment.design));
    expect(shape.acceptedSimulation).toBe(false);expect(shape.sourceVertices).toBeGreaterThan(1000);
    expect(shape.pieces.length).toBe(family==='skirt'?12:6);expect(inspection.instances.length).toBe(shape.pieces.length);
    expect(pattern.files.filter(file=>file.mime==='application/pdf')).toHaveLength(3);
    expect(garmentFlats(doc,pattern.geometry)).toHaveLength(2);
    const corrupt=structuredClone(pattern.geometry);corrupt.panels.pop();expect(()=>validateDesignGeometry(doc,corrupt)).toThrow();
    const wrong=structuredClone(pattern.geometry);wrong.drafting!.assembly[0]!.sides[0]!.edge=2;expect(()=>validateDesignGeometry(doc,wrong)).toThrow();
    const metric=structuredClone(shape);metric.pieces[0]!.restXY[0]![0]+=.005;expect(()=>validateGarmentPreview(Buffer.from(JSON.stringify(metric)),pattern.geometry,hash(source),objectDigest(doc.garment.design))).toThrow();
    const evidence=join(import.meta.dir,'../../.planning/multiple-garments',family);await mkdir(evidence,{recursive:true});
    await writeFile(join(evidence,'pattern.json'),source);await writeFile(join(evidence,'shape.json'),preview.shape!);await writeFile(join(evidence,'document.json'),canonical(doc));
    doc.garment.length=assumed(family==='dress'?923:583);doc.garment.flare=1.12;
    const changed=await runEngine({document:doc,inputDigest:objectDigest(doc),outputDir,signal:new AbortController().signal});
    expect(changed.geometry.inputDigest).not.toBe(pattern.geometry.inputDigest);
    expect(changed.geometry.panels[0]!.heightMm).toBeLessThan(pattern.geometry.panels[0]!.heightMm);
    expect(changed.geometry.drafting!.measurements).not.toEqual(pattern.geometry.drafting!.measurements);
  } finally {await rm(outputDir,{recursive:true,force:true});}
},180000);

for(const family of ['dress','skirt'] as const)test(`${family}: all six sample sizes compile with only required body fields`,async()=>{
  const outputDir=await mkdtemp(join(import.meta.dir,'.test-family-sizes-'));
  try {
    let previousWidth=0;
    for(let size=0;size<sampleSizes.length;size++) {
      const doc=applySample(startingDocument(family),size);
      const required=new Set(requiredBodyFields(doc).map(field=>field.key));
      for(const key of Object.keys(doc.body) as (keyof typeof doc.body)[])if(!required.has(key))doc.body[key]={state:'unknown'};
      const result=await runEngine({document:doc,inputDigest:objectDigest(doc),outputDir,signal:new AbortController().signal});
      validateDesignGeometry(doc,result.geometry);
      expect(result.geometry.panels[0]!.widthMm).toBeGreaterThan(previousWidth);
      previousWidth=result.geometry.panels[0]!.widthMm;
      expect(result.geometry.drafting!.compiler).toBe(family==='dress'?'sew-relaxed-dress/1':'sew-elastic-skirt/1');
    }
  } finally {await rm(outputDir,{recursive:true,force:true});}
},180000);
