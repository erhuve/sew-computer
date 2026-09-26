import {expect,test} from 'bun:test';
import {mkdtemp,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {startingDocument} from '../../packages/contracts/starting-designs';
import {objectDigest,hash,geometry as validateApiGeometry} from '../../apps/api/validation';
import {validateDesignGeometry} from '../../packages/contracts/design';
import {validateInspection} from '../../apps/api/inspection-validation';
import {validateGarmentPreview} from '../../apps/api/garment-preview-validation';
import {runEngine} from './runner';
import {runInspection} from './inspection-runner';

test('owner-authored outlines, seams and placement reach source geometry and 3D without resizing',async()=>{
  const directory=await mkdtemp(join(import.meta.dir,'.test-custom-'));
  try {
    const doc=startingDocument('custom');const design=doc.garment.design!;if(design.block!=='custom-pattern')throw new Error('fixture');
    design.pieces.forEach(piece=>piece.points=piece.points.map(([x,y])=>[x,y===650?713:y]));
    design.pieces.push({id:'arbitrary_right',name:'Asymmetric extra panel',points:[[0,0],[117,0],[151,193],[20,237],[-23,83]],placement:{positionMm:[450,-200,0],rotationDeg:[0,15,10],bendDeg:45}});
    const pattern=await runEngine({document:doc,inputDigest:objectDigest(doc),outputDir:directory,signal:new AbortController().signal});
    validateApiGeometry(pattern.geometry,objectDigest(doc));validateDesignGeometry(doc,pattern.geometry);
    expect(pattern.geometry.panels[0]!.heightMm).toBe(713);
    expect(pattern.geometry.panels[4]!.points).toEqual([...design.pieces[4]!.points,design.pieces[4]!.points[0]!]);
    expect(pattern.files.filter(file=>file.mime==='application/pdf')).toHaveLength(3);
    const source=pattern.files.find(file=>file.filename==='pattern.json')!.bytes;
    const preview=await runInspection({pattern:source,construction:design,outputDir:directory,signal:new AbortController().signal});
    expect(validateInspection(preview.report,preview.mesh,pattern.geometry,hash(source),objectDigest(design)).instances).toHaveLength(5);
    const shape=validateGarmentPreview(preview.shape!,pattern.geometry,hash(source),objectDigest(design));
    expect(shape.pieces).toHaveLength(5);expect(shape.acceptedSimulation).toBe(false);
    const forged=structuredClone(pattern.geometry);forged.panels[4]!.points[1]![0]+=5;expect(()=>validateDesignGeometry(doc,forged)).toThrow();
    const invalid=structuredClone(doc);if(invalid.garment.design?.block!=='custom-pattern')throw new Error('fixture');invalid.garment.design.pieces[0]!.points[2]![1]+=20;
    await expect(runEngine({document:invalid,inputDigest:objectDigest(invalid),outputDir:directory,signal:new AbortController().signal})).rejects.toThrow('plain edges');
    design.pieces=[design.pieces[4]!];design.seams=[];
    const solo=await runEngine({document:doc,inputDigest:objectDigest(doc),outputDir:directory,signal:new AbortController().signal});
    const soloBytes=solo.files.find(file=>file.filename==='pattern.json')!.bytes;
    const soloPreview=await runInspection({pattern:soloBytes,construction:design,outputDir:directory,signal:new AbortController().signal});
    expect(validateGarmentPreview(soloPreview.shape!,solo.geometry,hash(soloBytes),objectDigest(design)).iterations.at(-1)!.seamGapMaxMm).toBe(0);
    design.pieces[0]!.points=[[0,0],[120,120],[0,120],[120,0]];
    await expect(runEngine({document:doc,inputDigest:objectDigest(doc),outputDir:directory,signal:new AbortController().signal})).rejects.toThrow();
  }finally{await rm(directory,{recursive:true,force:true});}
},180000);
