import { assumed, mm, type GarmentDocument, type Measurement } from './index';
import { designIssues } from './design';

export type Body = GarmentDocument['body'];
export type BodyKey = keyof Body;
export type BodyProfile = { version: number; body: Body | null };
export const bodyFields: { key: BodyKey; label: string; min: number; max: number; method: string }[] = [
  { key: 'height', label: 'Height', min: 1200, max: 2200, method: 'Measure from the floor to the top of your head, standing without shoes.' },
  { key: 'bust', label: 'Bust', min: 600, max: 1600, method: 'Measure around the fullest part of your chest or bust. This is a circumference, not a flat width.' },
  { key: 'waist', label: 'Waist', min: 450, max: 1500, method: 'Measure around your natural waist without pulling the tape tight.' },
  { key: 'hip', label: 'Hip', min: 650, max: 1700, method: 'Measure around the fullest part of your hips and seat.' },
  { key: 'shoulder', label: 'Shoulder', min: 250, max: 600, method: 'Measure across your back from one shoulder tip to the other. This is a width, not a circumference.' },
];
export const sampleSizes = [
  { label: 'XS', values: [1600, 800, 640, 880, 360] },
  { label: 'S', values: [1650, 860, 700, 940, 380] },
  { label: 'M', values: [1700, 920, 760, 980, 400] },
  { label: 'L', values: [1750, 1020, 860, 1080, 430] },
  { label: 'XL', values: [1750, 1140, 980, 1200, 460] },
  { label: '2XL', values: [1750, 1280, 1120, 1340, 490] },
] as const;

export function sampleBody(index: number): Body {
  const sample = sampleSizes[index]!;
  return Object.fromEntries(bodyFields.map((field, position) => [field.key, assumed(sample.values[position]!, 'mm', `Sew sample ${sample.label}; synthetic starting size, not a standard size chart or wearer measurement`)])) as Body;
}

export function applySample(doc: GarmentDocument, index: number): GarmentDocument {
  const body = sampleBody(index);
  for (const field of bodyFields) if (doc.body[field.key].state !== 'unknown' && doc.body[field.key].state !== 'assumed') body[field.key] = doc.body[field.key];
  return { ...doc, body, garment: { ...doc.garment,
    length: doc.garment.length.state === 'unknown' ? assumed(doc.garment.family === 'shirt' ? 600 : doc.garment.family === 'skirt' ? 650 : 1000) : doc.garment.length,
    ease: doc.garment.ease.state === 'unknown' ? assumed(80) : doc.garment.ease,
  } };
}

export function editMeasurement(previous: Measurement, value: number, unit: 'mm' | 'cm' | 'in'): Measurement {
  return { state: previous.state === 'assumed' ? 'assumed' : 'known', value, unit,
    source: previous.state === 'assumed' ? 'Owner-adjusted estimate; not confirmed as measured' : 'Entered by owner' };
}

export function sizingInput(doc: GarmentDocument) {
  if (doc.garment.family === 'none') throw new Error('No garment family selected. Open Design and interpret your brief, then accept a supported proposal; or choose a shape in Shape & body. Your original intent remains unchanged.');
  const value = (name: string, measurement: Measurement, min: number, max: number) => {
    const amount = mm(measurement);
    if (amount === null) throw new Error(`${name} must be explicitly known or assumed before geometry generation. Enter it in Shape & body.`);
    if (!Number.isFinite(amount) || amount < min || amount > max) throw new Error(`Unsupported ${name}: requires ${min}–${max} mm in this engine. Your value is preserved; this is an engine limit, not a judgment about your body.`);
    return amount;
  };
  const bodyMm = Object.fromEntries(bodyFields.map(field => [field.key, value(`body ${field.key === 'bust' ? 'bust circumference' : field.key === 'waist' ? 'waist circumference' : field.key === 'hip' ? 'hip circumference' : field.key === 'shoulder' ? 'shoulder width' : 'height'}`, doc.body[field.key], field.min, field.max)])) as Record<BodyKey, number>;
  const family = doc.garment.family;
  const lengthMm = value('garment construction length', doc.garment.length, family === 'shirt' ? 400 : bodyMm.height * 0.12 + 150, family === 'shirt' ? 1100 : 1300);
  const easeMm = value('circumference ease', doc.garment.ease, 0, family === 'shirt' ? Math.min(200, bodyMm.bust * 0.3) : 200);
  const ranges = { shirt: [0.7, 1.5], skirt: [0.5, 2], trousers: [0.7, 1.2] } as const;
  const [minimum, maximum] = ranges[family];
  if (doc.garment.flare < minimum || doc.garment.flare > maximum) throw new Error(`Unsupported ${family} flare: requires ${minimum}–${maximum}. Review it in Shape & body.`);
  if (family !== 'shirt' && bodyMm.hip - bodyMm.waist < 40) throw new Error('Unsupported lower-garment body combination: this adapter requires hip to exceed waist by at least 40 mm. Check your measurements in Shape & body; if accurate, this body combination is not supported yet.');
  const design = doc.garment.design;
  if (design) {
    if (family !== 'shirt') throw new Error('The relaxed shirt construction requires shirt family.');
    const issues = designIssues(design);
    if (issues.length) throw new Error(issues.join(' '));
    const width = (Math.max(bodyMm.bust, bodyMm.hip) + easeMm) / 4;
    const armDepth = bodyMm.bust / 10 + 110;
    if (width * 2 < bodyMm.shoulder + 20) throw new Error('This relaxed drop-shoulder block needs finished upper-body width at least 20 mm wider than shoulder width. Increase garment ease or choose another construction; do not alter accurate body measurements.');
    if (lengthMm < armDepth + 150 || doc.garment.flare < 0.9) throw new Error('This relaxed shirt requires at least 150 mm below the armhole and flare of at least 0.9.');
    if (design.sleeves !== 'none') {
      const cuffDepth = design.cuff === 'button' ? design.cuffDepthMm : 0;
      if (design.sleeveLengthMm <= cuffDepth + 120) throw new Error('Sleeve length must leave at least 120 mm above the cuff.');
      if (cuffDepth && Math.max(design.cuffCircumferenceMm * 1.35, armDepth * 1.1) >= armDepth * 2) throw new Error('Cuff and armhole proportions leave no usable sleeve taper. Revise sleeve/cuff dimensions.');
    }
  }
  return { family, bodyMm, lengthMm, easeMm, flare: doc.garment.flare };
}

export function sizingIssue(doc: GarmentDocument): string | null {
  try { sizingInput(doc); return null; } catch (error) { return (error as Error).message; }
}

export function sizingWarning(doc: GarmentDocument): string | null {
  if(doc.garment.design)return null;
  const shoulder = mm(doc.body.shoulder), bust = mm(doc.body.bust), ease = mm(doc.garment.ease);
  if (doc.garment.family === 'shirt' && shoulder !== null && bust !== null && ease !== null && shoulder >= (bust + ease) * 0.48 + 20) return 'This shoulder/chest combination may collapse the top’s armhole in the current engine, which assumes back proportions. Generation may fail even though each measurement is within range. Keep accurate measurements; do not reduce them just to make the engine succeed.';
  return null;
}
