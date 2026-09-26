import { describe, expect, test } from 'bun:test';
import type { PatternGeometry, PatternPanel } from '../../../../packages/contracts';
import inventory from '../../../../packages/test-fixtures/assembly-expected.json';
import { buildGarmentLayout, type GarmentPiece } from './garment-layout';

type Point = [number, number];
type Vector = [number, number, number];

function panel(id: string, width: number, height: number, points?: Point[], names?: string[]): PatternPanel {
  const outline: Point[] = points ?? [[0, 0], [width, 0], [width, height], [0, height], [0, 0]];
  return { id, name: id, points: outline, widthMm: width, heightMm: height,
    draft: { component: 'body', material: 'shell', cutQuantity: 1, cutLine: structuredClone(outline),
      grainline: [[1, 1], [1, height - 1]], marks: [],
      edges: outline.slice(0, -1).map((point, i) => ({ name: names?.[i] ?? ['top', 'right', 'bottom', 'left'][i]!,
        start: i, end: i + 1, lengthMm: Math.hypot(point[0] - outline[i + 1]![0], point[1] - outline[i + 1]![1]), finish: 'assembly' })) } };
}

function fixture(): { pattern: PatternGeometry; pieces: GarmentPiece[] } {
  const panels = inventory.fullShirt.templates.map(({ id, roles }) => {
    let result: PatternPanel;
    if (/^(front|back)_/.test(id)) result = panel(id, 250, 600,
      [[0, 50], [0, 600], [250, 600], [250, 0], [200, 0], [0, 50]],
      ['center', 'hem', 'side', 'shoulder', 'neck']);
    else if (id.startsWith('sleeve_')) result = panel(id, 300, 500,
      [[0, 0], [150, 0], [300, 0], [300, 400], [300, 500], [0, 500], [0, 400], [0, 0]],
      ['cap_front', 'cap_back', 'underarm_right', 'opening_right', 'wrist', 'opening_left', 'underarm_left']);
    else if (id.startsWith('opening_binding_')) result = panel(id, 20, 100);
    else if (id.startsWith('cuff_')) result = panel(id, 240, 55);
    else if (id.startsWith('placket_')) result = panel(id, 30, 550);
    else if (id.startsWith('frill_')) result = panel(id, 35, 990);
    else result = panel(id, id === 'collar_fall' ? 380 : 400, id === 'collar_fall' ? 60 : 30);
    result.cutQuantity = roles.length;
    result.draft!.cutQuantity = roles.length;
    return result;
  });
  return { pattern: { schemaVersion: 1, units: 'mm', inputDigest: 'synthetic-layout', engineVersion: 'test',
    family: 'shirt', panels, stitches: [], warnings: [], classification: 'printable-reference', assumptions: [] },
    pieces: inventory.fullShirt.templates.flatMap(({ id, roles }) => roles.map(role => ({ instanceId: `${id}:${role}`, templateId: id, role }))) };
}

function freeze<T>(value: T): T {
  if (value && typeof value === 'object') {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
}

function transform(matrix: number[], point: Vector): Vector {
  return [0, 1, 2].map(axis => matrix[12 + axis]! + point.reduce(
    (sum, value, column) => sum + value * matrix[4 * column + axis]!, 0)) as Vector;
}

// The independently audited inventory supplies handedness, not the helper.
function local(template: string, point: Point): Vector {
  const sign = inventory.fullShirt.templates.find(row => row.id === template)!.hand === 'right' ? -1 : 1;
  return [sign * point[0] / 1000, -point[1] / 1000, 0];
}

function close(actual: number[], expected: number[]) {
  expect(actual.length).toBe(expected.length);
  actual.forEach((value, index) => expect(value).toBeCloseTo(expected[index]!, 11));
}

describe('rigid shirt display layout', () => {
  test('accounts for all 24 fabric instances with proper rotations and unchanged geometry', () => {
    const source = freeze(fixture());
    const before = JSON.stringify(source);
    const layout = buildGarmentLayout(source.pattern, source.pieces);
    expect(layout.recipe).toBe('sew-shirt-rigid-staging/2');
    expect(layout.classification).toBe('unvalidated-placement');
    expect(layout.units).toBe('m');
    expect(layout.frames.map(frame => frame.instanceId)).toEqual(source.pieces.map(piece => piece.instanceId));
    expect(layout.frames).toHaveLength(24);
    expect(new Set(layout.frames.map(frame => frame.instanceId)).size).toBe(24);
    for (const frame of layout.frames) {
      const m = frame.matrix;
      expect(m).toHaveLength(16);
      expect(m.every(Number.isFinite)).toBe(true);
      expect([m[3], m[7], m[11], m[15]]).toEqual([0, 0, 0, 1]);
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
        expect([0, 1, 2].reduce((sum, axis) => sum + m[i * 4 + axis]! * m[j * 4 + axis]!, 0)).toBeCloseTo(i === j ? 1 : 0, 12);
      const determinant = m[0]! * (m[5]! * m[10]! - m[6]! * m[9]!)
        - m[4]! * (m[1]! * m[10]! - m[2]! * m[9]!) + m[8]! * (m[1]! * m[6]! - m[2]! * m[5]!);
      expect(determinant).toBeCloseTo(1, 12);
      const points = source.pattern.panels.find(p => p.id === frame.templateId)!.points;
      for (let i = 1; i < points.length; i++) {
        const a = transform(m, local(frame.templateId, points[i - 1]!));
        const b = transform(m, local(frame.templateId, points[i]!));
        expect(Math.hypot(...a.map((value, axis) => value - b[axis]!))).toBeCloseTo(
          Math.hypot(points[i]![0] - points[i - 1]![0], points[i]![1] - points[i - 1]![1]) / 1000, 12);
      }
    }
    expect(JSON.stringify(source)).toBe(before);
  });

  test('maps mirrored GLB coordinates to Y-up torso, sleeve, cuff and collar frames', () => {
    const { pattern, pieces } = fixture();
    const frames = new Map(buildGarmentLayout(pattern, pieces).frames.map(frame => [frame.instanceId, frame.matrix]));
    const radius = .250 / Math.SQRT2;
    close(transform(frames.get('front_left:shell')!, [.25, 0, 0]), [radius, 1.2, 0]);
    close(transform(frames.get('front_right:shell')!, [-.25, 0, 0]), [-radius, 1.2, 0]);
    close(transform(frames.get('back_left:shell')!, [.25, 0, 0]), [radius, 1.2, 0]);
    close(transform(frames.get('sleeve_left:shell')!, [0, -.5, 0]), [radius + .5, 1.2, -.15]);
    close(transform(frames.get('sleeve_right:shell')!, [0, -.5, 0]), [-radius - .5, 1.2, .15]);
    close(transform(frames.get('cuff_left:shell')!, [0, 0, 0]), [radius + .5, 1.18, -.12]);
    close(transform(frames.get('cuff_left:facing')!, [0, 0, 0]), [radius + .5, 1.178, -.12]);
    close(transform(frames.get('collar_fall:shell')!, [0, 0, 0]), [-.19, 1.22, -radius / 2]);
    close(transform(frames.get('collar_fall:facing')!, [0, 0, 0]), [-.19, 1.22, -radius / 2 - .002]);
    close(transform(frames.get('placket_right:shell')!, [0, 0, 0]), [.03, 1.15, -radius]);
    close(transform(frames.get('frill_left:shell')!, [0, 0, 0]), [-.035, 1.15, -radius - .01]);
  });

  test('registers all four binding edges without resizing or double mirroring', () => {
    const { pattern, pieces } = fixture();
    const frames = new Map(buildGarmentLayout(pattern, pieces).frames.map(frame => [frame.instanceId, frame.matrix]));
    for (const side of ['left', 'right']) for (const opening of ['left', 'right']) {
      const binding = pattern.panels.find(p => p.id === `opening_binding_${side}_${opening}`)!;
      const sleeve = pattern.panels.find(p => p.id === `sleeve_${side}`)!;
      const bindingEdge = binding.draft!.edges.find(edge => edge.name === 'right')!;
      const sleeveEdge = sleeve.draft!.edges.find(edge => edge.name === `opening_${opening}`)!;
      for (const endpoint of ['start', 'end'] as const) {
        const actual = transform(frames.get(`${binding.id}:shell`)!, local(binding.id, binding.points[bindingEdge[endpoint]]!));
        const target = transform(frames.get(`${sleeve.id}:shell`)!, local(sleeve.id, sleeve.points[sleeveEdge[endpoint]]!));
        close(actual.map((value, axis) => value - target[axis]!), [0, (opening === 'left' ? -.005 : .005) * (side === 'left' ? 1 : -1), 0]);
      }
    }
  });

  test('inventory order does not change transforms and body-only variants work', () => {
    const { pattern, pieces } = fixture();
    const original = buildGarmentLayout(pattern, pieces);
    const reversed = buildGarmentLayout(pattern, [...pieces].reverse());
    const byId = new Map(original.frames.map(frame => [frame.instanceId, frame.matrix]));
    reversed.frames.forEach(frame => expect(frame.matrix).toEqual(byId.get(frame.instanceId)!));
    const body = pieces.filter(piece => /^(front|back)_/.test(piece.templateId));
    const layout = buildGarmentLayout({ ...pattern, panels: pattern.panels.filter(panel => body.some(piece => piece.templateId === panel.id)) }, body);
    expect(layout.frames).toHaveLength(4);
    layout.frames.forEach(frame => expect(frame.matrix).toEqual(byId.get(frame.instanceId)!));
  });

  test('fails the entire layout for missing, ambiguous or incompatible source geometry', () => {
    const cases: ((pattern: PatternGeometry, pieces: GarmentPiece[]) => void)[] = [
      pattern => { pattern.panels = pattern.panels.filter(panel => panel.id !== 'back_left'); },
      pattern => { pattern.panels[0]!.points[0]![0] = NaN; },
      pattern => { pattern.panels.push(structuredClone(pattern.panels[0]!)); },
      (_pattern, pieces) => { pieces.push({ ...pieces[0]! }); },
      (_pattern, pieces) => { pieces.splice(pieces.findIndex(piece => piece.templateId === 'sleeve_left'), 1); },
      (_pattern, pieces) => { pieces[0]!.role = 'interfacing'; },
      pattern => { pattern.panels.find(panel => panel.id === 'back_left')!.draft!.edges = []; },
      pattern => { pattern.panels.find(panel => panel.id === 'opening_binding_left_left')!.points[2]![1] += 1; },
    ];
    for (const corrupt of cases) {
      const { pattern, pieces } = fixture();
      corrupt(pattern, pieces);
      expect(() => buildGarmentLayout(pattern, pieces)).toThrow();
    }
  });
});
