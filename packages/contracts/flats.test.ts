import { expect, test } from 'bun:test';
import { assumed } from './index';
import { garmentFlats } from './flats';
import { shirtDocument } from '../test-fixtures/shirt';

test('construction preview preserves closure count, spacing and body-derived proportions',()=>{
  const doc=shirtDocument();
  doc.garment.length=assumed(700);
  const front=garmentFlats(doc)[0]!;
  const placket=front.buttons.filter(point=>point[0]===0&&point[1]>0);
  expect(placket).toHaveLength(8);
  expect(placket[0]![1]).toBeCloseTo(105*220/700);
  expect(placket[1]![1]-placket[0]![1]).toBeCloseTo(80*220/700);
  expect(front.buttons).toHaveLength(11);
  const originalWidth=Math.max(...front.lines.flatMap(line=>line.points.map(point=>point[0])));
  doc.body.hip=assumed(1300);
  const wider=garmentFlats(doc)[0]!;
  expect(Math.max(...wider.lines.flatMap(line=>line.points.map(point=>point[0])))).toBeGreaterThan(originalWidth);
});
