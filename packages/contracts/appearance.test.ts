import { expect, test } from 'bun:test';
import { DocumentSchema, canonical, emptyDocument } from './index';
import { interpretationJsonSchema, rebaseAcceptedDesign } from './interpretation';

test('appearance is explicit revision data and does not rewrite older documents',()=>{
  const original=emptyDocument();
  expect(canonical(DocumentSchema.parse(original))).toBe(canonical(original));
  const colored=structuredClone(original);colored.garment.appearance={color:'#315dd4'};
  expect(DocumentSchema.parse(colored)).toEqual(colored);
  expect(canonical(colored)).not.toBe(canonical(original));
  for(const color of ['red','url(https://invalid.test/image)','#abcd','#12345678'])expect(DocumentSchema.safeParse({...colored,garment:{...colored.garment,appearance:{color}}}).success).toBe(false);
  const schema=interpretationJsonSchema() as any;
  expect(schema.properties.garment.required).toContain('appearance');expect(JSON.stringify(schema)).not.toContain('prefixItems');
  const accepted=structuredClone(colored),latest=structuredClone(original);latest.garment.appearance={color:'#812f46'};
  expect(rebaseAcceptedDesign(accepted,original,latest).garment.appearance).toEqual(latest.garment.appearance);
});

test('prints are bounded data, not external resources',()=>{
  const doc=emptyDocument();doc.garment.appearance={color:'#ffffff',print:{kind:'checks',inkColor:'#233953',tileMm:35,rotationDeg:45,assetId:null}};
  expect(DocumentSchema.parse(doc)).toEqual(doc);
  for(const print of [{...doc.garment.appearance.print,tileMm:0},{...doc.garment.appearance.print,assetId:'https://example.com/image.png'},{...doc.garment.appearance.print,rotationDeg:Infinity}])expect(DocumentSchema.safeParse({...doc,garment:{...doc.garment,appearance:{color:'#ffffff',print}}}).success).toBe(false);
});
