import {expect,test} from 'bun:test';
import {DocumentSchema,emptyDocument,assumed} from './index';
import {startingDocument} from './starting-designs';
import {GarmentDesignSchema} from './design';
import {sizingInput,sizingIssue,requiredBodyFields,applySample,sampleSizes} from './sizing';

test('each starting family keeps explicit estimates and independently editable construction',()=>{
  for(const family of ['shirt','dress','skirt'] as const)for(let sample=0;sample<sampleSizes.length;sample++) {
    const doc=applySample(startingDocument(family),sample);
    expect(DocumentSchema.parse(doc)).toEqual(doc);expect(sizingIssue(doc)).toBeNull();
    expect(doc.body.hip.state).toBe('assumed');
  }
  const first=startingDocument('skirt');first.garment.design!.seamAllowanceMm=16;
  expect(startingDocument('skirt').garment.design!.seamAllowanceMm).toBe(10);
  const old=emptyDocument();expect(DocumentSchema.parse(old)).toEqual(old);expect('design' in DocumentSchema.parse(old).garment).toBe(false);
});
test('component families require only measurements that drive their source pattern',()=>{
  for(const family of ['shirt','dress','skirt'] as const) {
    const doc=startingDocument(family),keys=requiredBodyFields(doc).map(field=>field.key);
    for(const key of Object.keys(doc.body) as (keyof typeof doc.body)[])if(!keys.includes(key))doc.body[key]={state:'unknown'};
    const before=structuredClone(doc),input=sizingInput(doc);
    expect(Object.keys(input.bodyMm).sort()).toEqual([...keys].sort());expect(doc).toEqual(before);
    doc.body.hip={state:'unknown'};expect(sizingIssue(doc)).toContain('hip');
  }
});
test('family mismatch, missing dress block and unsupported controls are rejected without substitution',()=>{
  const doc=startingDocument('dress');doc.garment.family='skirt';expect(sizingIssue(doc)).toContain('does not match');
  doc.garment.family='dress';doc.garment.design=null;expect(sizingIssue(doc)).toContain('dress construction');
  expect(GarmentDesignSchema.safeParse({...startingDocument('skirt').garment.design,zipper:'hidden'}).success).toBe(false);
  const skirt=startingDocument('skirt');skirt.garment.flare=.8;expect(sizingIssue(skirt)).toContain('flare');
  skirt.garment.flare=1;skirt.garment.length=assumed(300);expect(sizingIssue(skirt)).toContain('length');
});
