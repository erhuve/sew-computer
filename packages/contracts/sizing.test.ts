import { expect, test } from 'bun:test';
import { assumed, emptyDocument, mm } from './index';
import { applySample, bodyFields, editMeasurement, sampleSizes, sizingInput, sizingIssue, sizingWarning } from './sizing';

test('samples preserve entered and N/A values and garment choices; assumptions stay explicit', () => {
  const doc = emptyDocument();
  doc.body.height = { state: 'known', value: 70, unit: 'in', source: 'Measured' };
  doc.body.shoulder = { state: 'not-applicable' };
  doc.garment = { family: 'shirt', length: assumed(720), ease: assumed(120), flare: 1.2 };
  const changed = applySample(doc, 2);
  expect(changed.body.height).toEqual(doc.body.height);
  expect(changed.body.shoulder).toEqual(doc.body.shoulder);
  expect(changed.garment).toEqual(doc.garment);
  expect(changed.body.waist.state).toBe('assumed');
  expect(doc.body.waist.state).toBe('unknown');
  expect(editMeasurement(changed.body.waist, 80, 'cm')).toEqual({ state: 'assumed', value: 80, unit: 'cm', source: 'Owner-adjusted estimate; not confirmed as measured' });
});

test('every sample passes basic preflight for each supported family', () => {
  for (const family of ['shirt', 'skirt', 'trousers'] as const) {
    for (let index = 0; index < sampleSizes.length; index++) {
      const doc = emptyDocument();
      doc.garment.family = family;
      expect(sizingIssue(applySample(doc, index))).toBeNull();
    }
  }
});

test('preflight preserves unsupported dimensions and distinguishes engine range from body validity', () => {
  const doc = applySample(emptyDocument(), 2);
  doc.garment.family = 'shirt';
  doc.body.shoulder = assumed(25, 'in');
  expect(sizingIssue(doc)).toContain('engine limit');
  expect(mm(doc.body.shoulder)).toBe(635);
  doc.body.shoulder = assumed(400);
  doc.garment.family = 'trousers';
  doc.body.waist = assumed(1000);
  expect(sizingIssue(doc)).toContain('hip to exceed waist');
});

test('all supported body endpoints survive the shared preflight without rounding', () => {
  for (const field of bodyFields) for (const endpoint of [field.min, field.max]) {
    const doc = applySample(emptyDocument(), 2);
    doc.garment.family = 'shirt';
    doc.body[field.key] = editMeasurement(doc.body[field.key], endpoint, 'mm');
    expect(sizingInput(doc).bodyMm[field.key]).toBe(endpoint);
  }
});

test('broad shoulders warn before generation without changing or rejecting accurate measurements', () => {
  const doc = applySample(emptyDocument(), 2);
  doc.garment.family = 'shirt';
  doc.body.shoulder = assumed(23, 'in');
  expect(sizingIssue(doc)).toBeNull();
  expect(sizingWarning(doc)).toContain('may collapse');
  expect(doc.body.shoulder).toEqual(assumed(23, 'in'));
  doc.garment.family = 'skirt';
  expect(sizingWarning(doc)).toBeNull();
});
