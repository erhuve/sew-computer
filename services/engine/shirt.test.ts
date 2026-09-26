import { expect, test } from 'bun:test';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import sharp from 'sharp';
import { canonical } from '../../packages/contracts';
import { designCoverage, validateDrafting, validateDesignGeometry } from '../../packages/contracts/design';
import { shirtDocument } from '../../packages/test-fixtures/shirt';
import { objectDigest, geometry as validateApiGeometry } from '../../apps/api/validation';
import { runEngine } from './runner';
import { buildExport } from '../../packages/tech-pack';
import { hash } from '../../apps/api/validation';
import { validateExport } from '../../apps/api/export-validation';
import { applySample, sampleSizes } from '../../packages/contracts/sizing';
import type { ExportSnapshot } from '../../packages/contracts';
import { garmentFlats } from '../../packages/contracts/flats';

test('custom PDF and SVG retain extreme mitered cut contours at full scale', async () => {
  const doc = shirtDocument();
  doc.garment.flare = 1.5;
  doc.garment.length = { state: 'assumed', value: 400, unit: 'mm', source: 'Synthetic clipping regression' };
  doc.body.hip = { state: 'assumed', value: 1600, unit: 'mm', source: 'Synthetic clipping regression' };
  doc.garment.design!.seamAllowanceMm = 20;
  const directory = await mkdtemp(join(import.meta.dir, '.test-shirt-print-bounds-'));
  try {
    const result = await runEngine({ document: doc, inputDigest: objectDigest(doc), outputDir: directory, signal: new AbortController().signal });
    const svg = new TextDecoder().decode(result.files.find(file => file.filename === 'pattern.svg')!.bytes);
    const viewBox = svg.match(/viewBox="0 0 ([\d.]+) ([\d.]+)"/)!;
    const width = Number(viewBox[1]), height = Number(viewBox[2]);
    const contours = [...svg.matchAll(/<polyline points="([^"]+)"[^>]*>/g)].map(match => ({
      cut: match[0].includes('#8b4538'),
      points: match[1]!.split(' ').map(pair => pair.split(',').map(Number) as [number, number]),
    }));
    const cuts = contours.filter(contour => contour.cut);
    expect(cuts).toHaveLength(result.geometry.panels.length);
    let previousBottom = 0;
    for (const [index, contour] of cuts.entries()) {
      const panel = result.geometry.panels[index]!;
      const original = panel.draft!.cutLine;
      expect(contour.points).toHaveLength(original.length);
      const shiftX = contour.points[0]![0] - original[0]![0];
      const shiftY = contour.points[0]![1] - original[0]![1];
      for (const [pointIndex, point] of contour.points.entries()) {
        expect(point[0] - original[pointIndex]![0]).toBeCloseTo(shiftX, 4);
        expect(point[1] - original[pointIndex]![1]).toBeCloseTo(shiftY, 4);
      }
      const top = Math.min(...contour.points.map(point => point[1]));
      expect(top).toBeGreaterThan(previousBottom);
      previousBottom = Math.max(...contour.points.map(point => point[1]));
    }
    for (const contour of contours) for (const point of contour.points) {
      expect(point[0]).toBeGreaterThan(0);
      expect(point[0]).toBeLessThan(width);
      expect(point[1]).toBeGreaterThan(0);
      expect(point[1]).toBeLessThan(height);
    }
    const filename = join(directory, 'custom.pdf');
    await writeFile(filename, result.files.find(file => file.filename === 'pattern.pdf')!.bytes);
    for (const panelIndex of [0, 1]) {
      const panel = result.geometry.panels[panelIndex]!;
      const xs = panel.draft!.cutLine.map(point => point[0]), ys = panel.draft!.cutLine.map(point => point[1]);
      expect(Math.min(...xs)).toBeLessThan(0);
      expect(Math.min(...ys)).toBeLessThan(0);
      expect(Math.max(...xs)).toBeGreaterThan(panel.widthMm + 30);
      const process = Bun.spawn(['pdftoppm', '-f', String(panelIndex + 1), '-singlefile', '-r', '127', '-png', filename], { stdout: 'pipe', stderr: 'pipe' });
      const png = new Uint8Array(await new Response(process.stdout).arrayBuffer());
      expect(await process.exited).toBe(0);
      const raster = await sharp(png).removeAlpha().raw().toBuffer({ resolveWithObject: true });
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (let row = 0; row < raster.info.height; row++) for (let column = 0; column < raster.info.width; column++) {
        const offset = (row * raster.info.width + column) * raster.info.channels;
        const red = raster.data[offset]!, green = raster.data[offset + 1]!, blue = raster.data[offset + 2]!;
        if (red > 80 && red > green * 1.5 && red > blue * 1.5) {
          minX = Math.min(minX, column); maxX = Math.max(maxX, column);
          minY = Math.min(minY, row); maxY = Math.max(maxY, row);
        }
      }
      expect(Number.isFinite(minX)).toBe(true);
      expect(minX).toBeGreaterThan(5);
      expect(minY).toBeGreaterThan(5);
      expect(maxX).toBeLessThan(raster.info.width - 5);
      expect(maxY).toBeLessThan(raster.info.height - 5);
      expect(Math.abs((maxX - minX) / 5 - (Math.max(...xs) - Math.min(...xs)))).toBeLessThan(0.8);
      expect(Math.abs((maxY - minY) / 5 - (Math.max(...ys) - Math.min(...ys)))).toBeLessThan(0.8);
    }
  } finally { await rm(directory, { recursive: true, force: true }); }
}, 90000);

test('complete shirt produces connected components, matched seams, offsets and derived specs', async () => {
  const doc = shirtDocument();
  const directory = await mkdtemp(join(import.meta.dir, '.test-shirt-'));
  try {
    const result = await runEngine({document:doc,inputDigest:objectDigest(doc),outputDir:directory,signal:new AbortController().signal});
    validateDrafting(result.geometry);
    expect(validateApiGeometry(result.geometry, objectDigest(doc))).toEqual(result.geometry);
    expect(designCoverage(doc,result.geometry).status).toBe('drafted');
    expect(result.geometry.panels.some(panel => panel.id === 'collar_stand')).toBe(true);
    expect(result.geometry.panels.some(panel => panel.id === 'frill_left')).toBe(true);
    expect(result.geometry.drafting!.measurements.find(row => row.name === 'Closed cuff circumference')!.valueMm).toBe(220);
    const broken = structuredClone(result.geometry);
    broken.drafting!.assembly[0]!.sides[0]!.edge = 500;
    expect(() => validateDrafting(broken)).toThrow('reference');
    const missing = structuredClone(result.geometry);
    missing.panels = missing.panels.filter(panel => panel.id !== 'collar_stand');
    expect(() => validateDrafting(missing)).toThrow();
    expect(canonical(JSON.parse(new TextDecoder().decode(result.files[0]!.bytes)))).toBe(canonical(result.geometry));
    for(const ids of [['collar_fall'],['sleeve_right','cuff_right','opening_binding_right_left','opening_binding_right_right']]) {
      const partial=structuredClone(result.geometry);
      partial.panels=partial.panels.filter(panel=>!ids.includes(panel.id));
      partial.drafting!.assembly=partial.drafting!.assembly.filter(seam=>!seam.sides.some(side=>ids.includes(side.panel)));
      partial.stitches=partial.stitches.filter(seam=>!ids.includes(seam.panelA)&&!ids.includes(seam.panelB));
      expect(()=>validateDesignGeometry(doc,partial)).toThrow();
      expect(()=>designCoverage(doc,partial)).toThrow();
    }
    const badCut=structuredClone(result.geometry);
    badCut.panels[0]!.draft!.cutLine=[[0,0],[0,0],[0,0],[0,0]];
    expect(()=>validateDrafting(badCut)).toThrow('cut-line');
    const unmarked=structuredClone(result.geometry);
    for(const panel of unmarked.panels)panel.draft!.marks=[];
    expect(()=>designCoverage(doc,unmarked)).toThrow('marks');
    for(const name of ['cuff_left','collar_stand','placket_right']) {
      const misplaced=structuredClone(result.geometry);
      const piece=misplaced.panels.find(panel=>panel.id===name)!;
      piece.draft!.marks.find(mark=>mark.kind==='buttonhole')!.point[0]+=2;
      expect(()=>validateDesignGeometry(doc,misplaced)).toThrow('placement');
    }
    const impossibleCuff=structuredClone(result.geometry);
    const cuffMarks=impossibleCuff.panels.find(panel=>panel.id==='cuff_left')!.draft!.marks;
    cuffMarks.find(mark=>mark.kind==='buttonhole')!.point=[...cuffMarks.find(mark=>mark.kind==='button')!.point];
    expect(()=>validateDesignGeometry(doc,impossibleCuff)).toThrow('placement');
    const falseMeasurement=structuredClone(result.geometry);
    falseMeasurement.drafting!.measurements[0]!.valueMm+=20;
    expect(()=>validateDesignGeometry(doc,falseMeasurement)).toThrow('measurement');
    const front=garmentFlats(doc,result.geometry)[0]!;
    const placketButtons=result.geometry.panels.find(panel=>panel.id==='placket_left')!.draft!.marks.filter(mark=>mark.kind==='button');
    const frontButtons=front.buttons.filter(point=>point[0]===0&&point[1]>0);
    expect(frontButtons).toHaveLength(placketButtons.length);
    for(const [index,mark] of placketButtons.entries())expect(frontButtons[index]![1]).toBeCloseTo((80+mark.point[1])*220/650,5);
    for(let index=1;index<placketButtons.length;index++)expect(placketButtons[index]!.point[1]-placketButtons[index-1]!.point[1]).toBeCloseTo(doc.garment.design!.buttonSpacingMm,5);
    const operations=result.geometry.drafting!.operations.join('\n');
    expect(operations.indexOf('shoulder_left:')).toBeLessThan(operations.indexOf('sleeve_front_left:'));
    expect(operations.indexOf('frill_gather_left:')).toBeLessThan(operations.indexOf('placket_attach_left:'));
    expect(operations.indexOf('bind_opening_left_left:')).toBeLessThan(operations.indexOf('cuff_gather_left:'));
    expect(operations.indexOf('collar_fall_attach:')).toBeLessThan(operations.indexOf('collar_neck_0:'));
    const createdAt='2026-09-16T12:00:00.000Z';
    const assets=result.files.map((file,index)=>({bytes:file.bytes,artifact:{id:`artifact-${index}`,projectId:'project',revisionId:'revision',jobId:'job',kind:file.kind,filename:file.filename,mime:file.mime,digest:hash(file.bytes),bytes:file.bytes.length,classification:'printable-reference' as const,createdAt}}));
    const snapshot:ExportSnapshot={id:'snapshot',projectId:'project',revision:{id:'revision',projectId:'project',number:1,parentRevisionId:null,document:doc,digest:objectDigest(doc),createdAt},document:doc,artifacts:assets.map(asset=>asset.artifact),comments:[],disclosure:{includeBody:false,includeReferences:false,includePatterns:true},createdAt};
    const exported=await buildExport(snapshot,result.geometry,assets);
    await validateExport(snapshot,exported,assets);
    expect(exported.files).toHaveLength(8);
    for (const paper of ['a4','letter']) expect(exported.files.some(file=>file.filename.startsWith(`pattern-${paper}-tiled-`))).toBe(true);
    expect((exported.manifest.sections as any).derivedConstruction.measurements).toEqual(result.geometry.drafting!.measurements);
    const privateExport=await buildExport({...snapshot,disclosure:{includeBody:false,includeReferences:false,includePatterns:false}},null,[]);
    expect((privateExport.manifest.sections as any).derivedConstruction).toBeUndefined();
    await validateExport({...snapshot,disclosure:{includeBody:false,includeReferences:false,includePatterns:false}},privateExport,[]);
    expect(privateExport.files).toHaveLength(3);
    expect(JSON.stringify(privateExport.manifest)).not.toContain('Closed chest at underarm');
  } finally { await rm(directory,{recursive:true,force:true}); }
}, 90000);

test('component variants and all synthetic starting sizes draft without silent component loss',async()=>{
  const directory=await mkdtemp(join(import.meta.dir,'.test-shirt-matrix-'));
  try {
    const fixtures=sampleSizes.map((_,index)=>applySample(shirtDocument(),index));
    for(const variant of ['plain','short','stand','flared'] as const) {
      const doc=shirtDocument(),design=doc.garment.design!;
      if(variant==='plain') {design.sleeves='none';design.cuff='none';design.collar='none';design.opening='none';design.frill='none';design.hem='straight';}
      if(variant==='short') {design.sleeves='short';design.sleeveLengthMm=220;design.cuff='none';design.collar='none';}
      if(variant==='stand')design.collar='stand';
      if(variant==='flared')doc.garment.flare=1.5;
      fixtures.push(doc);
    }
    for(const doc of fixtures) {
      const result=await runEngine({document:doc,inputDigest:objectDigest(doc),outputDir:directory,signal:new AbortController().signal});
      validateDesignGeometry(doc,result.geometry);
    }
  } finally {await rm(directory,{recursive:true,force:true});}
},180000);
