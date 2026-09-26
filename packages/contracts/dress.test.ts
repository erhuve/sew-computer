import {test,expect} from 'bun:test';
import {DocumentSchema,assumed} from './index';
import {startingDocument} from './starting-designs';
import {defaultDressDesign} from './design';
import {defaultPanelDress,panelDressRecipe} from './dress';
import {sizingIssue,applySample,sampleSizes} from './sizing';
import {garmentFlats} from './flats';

test('new dress choices change source topology, waist placement and neckline without shirt details',()=>{
  const doc=startingDocument('dress');
  const a=panelDressRecipe(doc,defaultPanelDress);
  const tiered={...defaultPanelDress,skirtStyle:'tiered' as const,neckline:'square' as const,bodiceLengthMm:370,sleeves:'short' as const};
  const b=panelDressRecipe(doc,tiered);
  expect(a.pieces).toHaveLength(8);expect(b.pieces).toHaveLength(14);
  expect(a.seams.filter(s=>s.treatment==='gather')).toHaveLength(0);
  expect(b.seams.filter(s=>s.treatment==='gather')).toHaveLength(8);
  expect(a.measurements['Waist seam from shoulder']).not.toBe(b.measurements['Waist seam from shoulder']);
  expect(a.pieces[0]!.paths.at(-1)).not.toEqual(b.pieces[0]!.paths.at(-1));
  expect(b.pieces.some(p=>/collar|cuff|placket|frill/.test(p.id))).toBe(false);
  expect(garmentFlats(doc)[0]!.buttons).toEqual([]);
  for(let i=0;i<sampleSizes.length;i++)expect(sizingIssue(applySample(doc,i))).toBeNull();
});
test('impossible dress proportions fail before generation; historical shirt dresses retain their recipe',()=>{
  const doc=startingDocument('dress');doc.garment.design={...defaultPanelDress,neckDepthMm:200};
  expect(sizingIssue(doc)).toContain('neckline');
  doc.garment.design={...defaultPanelDress,bodiceLengthMm:650};doc.garment.length=assumed(700);
  expect(sizingIssue(doc)).toContain('skirt');
  doc.garment.design={...defaultPanelDress,skirtStyle:'tiered',skirtFullness:2,tierFullness:1.8};doc.garment.length=assumed(1200);doc.garment.flare=2;
  expect(sizingIssue(doc)).toContain('1800');
  doc.garment.design=structuredClone(defaultDressDesign);
  expect(DocumentSchema.parse(doc).garment.design).toEqual(defaultDressDesign);
});
