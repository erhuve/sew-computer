import type { PatternGeometry, PatternPanel } from '../../../../packages/contracts';

export type GarmentPiece = { instanceId: string; templateId: string; role: string };
export type GarmentLayoutFrame = GarmentPiece & { matrix: number[] };
export type GarmentLayout = {
  recipe: 'sew-shirt-rigid-staging/2';
  classification: 'unvalidated-placement';
  units: 'm';
  frames: GarmentLayoutFrame[];
};

type Vector = [number, number, number];
type Frame = { piece: GarmentPiece; origin: Vector; basis: [Vector, Vector, Vector] };
const supportedTemplate = /^(?:(?:front|back|sleeve|cuff|placket|frill)_(?:left|right)|collar_(?:stand|fall)|opening_binding_(?:left|right)_(?:left|right))$/;
const cross = (a: Vector, b: Vector): Vector => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const multiply = (v: Vector, scale: number): Vector => [v[0] * scale, v[1] * scale, v[2] * scale];
const scene = (v: Vector): Vector => [v[0], v[2], -v[1]];
// Match compile_inventory, including opening_binding_left_right (not mirrored).
const hand = (template: string) => (template.startsWith('opening_binding_')
  ? template.startsWith('opening_binding_right_') : template.endsWith('_right')) ? -1 : 1;

function edge(panel: PatternPanel, name: string) {
  const matches = panel.draft?.edges?.filter(value => value.name === name) ?? [];
  const value = matches[0];
  if (matches.length !== 1 || !value || !Number.isInteger(value.start) || !Number.isInteger(value.end)
      || value.start < 0 || value.end <= value.start || value.end >= panel.points.length
      || !Number.isFinite(value.lengthMm) || value.lengthMm <= 0)
    throw new Error(`Missing or invalid ${name} edge for ${panel.id}.`);
  return value;
}

function place(point: Vector, frame: Frame): Vector {
  return [0, 1, 2].map(axis => frame.origin[axis]! + point.reduce(
    (sum, value, local) => sum + value * frame.basis[local]![axis]!, 0)) as Vector;
}

function matrix(frame: Frame): number[] {
  // GLB vertices already mirror X and negate pattern Y. Negating both the
  // second and normal axes gives a proper rotation for these planar buffers.
  const x = scene(frame.basis[0]);
  const y = multiply(scene(frame.basis[1]), -1);
  const z = multiply(scene(frame.basis[2]), -1);
  const origin = multiply(scene(frame.origin), .001);
  const basis = [x, y, z];
  if (![...x, ...y, ...z, ...origin].every(Number.isFinite)
      || basis.some((a, i) => basis.some((b, j) => Math.abs(a.reduce(
        (sum, value, axis) => sum + value * b[axis]!, 0) - (i === j ? 1 : 0)) > 1e-9))
      || cross(x, y).some((value, axis) => Math.abs(value - z[axis]!) > 1e-9))
    throw new Error(`Invalid rigid placement for ${frame.piece.instanceId}.`);
  return [...x, 0, ...y, 0, ...z, 0, ...origin, 1];
}

/**
 * Port of services/engine/placement.py:shirt_placement, for display only.
 * Each column-major matrix REPLACES the GLB node's flat-layout transform.
 * Input buffers stay [mirroredX, -patternY, 0] in metres; output is Y-up with
 * physics [x,y,z] mapped to scene [x,z,-y]. No scale or mesh changes are made.
 * Missing geometry fails the entire layout so callers can retain flat view.
 */
export function buildGarmentLayout(pattern: PatternGeometry, pieces: readonly GarmentPiece[]): GarmentLayout {
  if (!pattern || pattern.units !== 'mm' || pattern.family !== 'shirt'
      || !Array.isArray(pattern.panels) || !Array.isArray(pieces) || !pieces.length)
    throw new Error('Shirt pattern geometry and physical pieces are required for this layout.');
  const panels = new Map<string, PatternPanel>();
  for (const panel of pattern.panels) {
    if (!panel || typeof panel.id !== 'string' || !panel.id || panels.has(panel.id))
      throw new Error('Pattern panel identities must be unique.');
    panels.set(panel.id, panel);
  }
  const checked = new Set<string>();
  const getPanel = (id: string): PatternPanel => {
    const panel = panels.get(id);
    if (!panel) throw new Error(`Missing source pattern panel: ${id}.`);
    if (!checked.has(id)) {
      if (!Array.isArray(panel.points) || panel.points.length < 3
          || panel.points.some(point => !Array.isArray(point) || point.length !== 2 || !point.every(Number.isFinite))
          || !Number.isFinite(panel.widthMm) || panel.widthMm <= 0
          || !Number.isFinite(panel.heightMm) || panel.heightMm <= 0)
        throw new Error(`Invalid source pattern geometry: ${id}.`);
      checked.add(id);
    }
    return panel;
  };
  const torso = getPanel('back_left');
  const shoulder = edge(torso, 'shoulder');
  const radius = torso.points[shoulder.start]![0] / Math.SQRT2;
  if (!Number.isFinite(radius) || radius <= 0) throw new Error('A positive drafted shoulder width is required.');
  const ids = new Set<string>();
  const frames: Frame[] = pieces.map(piece => {
    if (!piece || typeof piece.instanceId !== 'string' || !piece.instanceId || ids.has(piece.instanceId)
        || typeof piece.templateId !== 'string' || !supportedTemplate.test(piece.templateId)
        || (piece.role !== 'shell' && piece.role !== 'facing'))
      throw new Error('Unique supported physical piece identities are required.');
    ids.add(piece.instanceId);
    const template = piece.templateId;
    const panel = getPanel(template);
    const sign = hand(template);
    const layer = piece.role === 'facing' ? 2 : 0;
    const top = 1200;
    let tangent: Vector = [sign, 0, 0];
    let down: Vector = [0, 0, -1];
    let origin: Vector = [0, radius + layer, top];
    if (template.startsWith('front_') || template.startsWith('back_')) {
      const front = template.startsWith('front_') ? 1 : -1;
      tangent = [sign / Math.SQRT2, -front / Math.SQRT2, 0];
      origin = [0, front * radius, top];
    } else if (template.startsWith('sleeve_')) {
      const cap = edge(panel, 'cap_front');
      tangent = [0, -sign, 0];
      down = [sign, 0, 0];
      origin = [sign * radius, sign * cap.lengthMm, top];
    } else if (template.startsWith('cuff_')) {
      const sleeve = getPanel(`sleeve_${sign === -1 ? 'right' : 'left'}`);
      const length = sleeve.points.reduce((max, point) => Math.max(max, point[1]), -Infinity);
      tangent = [0, -sign, 0];
      down = [sign, 0, 0];
      origin = [sign * (radius + length), panel.widthMm / 2, top - 20 - layer];
    } else if (template.startsWith('collar_')) {
      origin = [-panel.widthMm / 2, radius / 2 + layer, top + (template === 'collar_fall' ? 20 : 0)];
    } else if (template.startsWith('placket_') || template.startsWith('frill_')) {
      const front = getPanel(`front_${sign === -1 ? 'right' : 'left'}`);
      const center = edge(front, 'center');
      origin = [-sign * panel.widthMm, radius + layer + (template.startsWith('frill_') ? 10 : 0),
        top - front.points[center.start]![1]];
    }
    tangent = multiply(tangent, sign);
    return { piece: { ...piece }, origin, basis: [tangent, down, cross(tangent, down)] };
  });
  const byId = new Map(frames.map(frame => [frame.piece.instanceId, frame]));
  for (const frame of frames) {
    const binding = /^opening_binding_(left|right)_(left|right)$/.exec(frame.piece.templateId);
    if (!binding) continue;
    const [, side, opening] = binding;
    const sleeves = frames.filter(value => value.piece.templateId === `sleeve_${side}`);
    if (sleeves.length !== 1) throw new Error('Opening binding placement requires exactly one matching sleeve.');
    const sleeve = sleeves[0]!;
    const sleevePanel = getPanel(sleeve.piece.templateId);
    const bindingPanel = getPanel(frame.piece.templateId);
    const sleeveEdge = edge(sleevePanel, `opening_${opening}`);
    const bindingEdge = edge(bindingPanel, 'right');
    if (sleeveEdge.end !== sleeveEdge.start + 1 || bindingEdge.end !== bindingEdge.start + 1)
      throw new Error('Opening binding placement requires straight source intervals.');
    const endpoints = (panel: PatternPanel, first: number, last: number): [Vector, Vector] =>
      [first, last].map(index => [hand(panel.id) * panel.points[index]![0], panel.points[index]![1], 0]) as [Vector, Vector];
    const rest = endpoints(bindingPanel, bindingEdge.start, bindingEdge.end);
    const sleeveRest = endpoints(sleevePanel, sleeveEdge.start, sleeveEdge.end);
    const length = (points: [Vector, Vector]) => Math.hypot(...points[0].map((value, axis) => value - points[1][axis]!));
    const bindingLength = length(rest), sleeveLength = length(sleeveRest);
    if (Math.min(bindingLength, sleeveLength) <= 0 || Math.abs(bindingLength - sleeveLength) > 1e-6)
      throw new Error('Opening binding placement requires matching straight source edges.');
    const targetStart = place(sleeveRest[0], sleeve), targetEnd = place(sleeveRest[1], sleeve);
    const direction = targetEnd.map((value, axis) => (value - targetStart[axis]!) / sleeveLength) as Vector;
    const normal = sleeve.basis[2];
    const perpendicular = cross(direction, normal);
    const local = rest[1].map((value, axis) => (value - rest[0][axis]!) / bindingLength);
    const tangent = direction.map((value, axis) => local[0]! * value + local[1]! * perpendicular[axis]!) as Vector;
    const down = direction.map((value, axis) => local[1]! * value - local[0]! * perpendicular[axis]!) as Vector;
    const spacing = opening === 'left' ? -5 : 5;
    frame.origin = targetStart.map((value, axis) => value + spacing * normal[axis]!
      - rest[0][0] * tangent[axis]! - rest[0][1] * down[axis]!) as Vector;
    frame.basis = [tangent, down, [...normal]];
  }
  return { recipe: 'sew-shirt-rigid-staging/2', classification: 'unvalidated-placement', units: 'm',
    frames: pieces.map(piece => ({ ...piece, matrix: matrix(byId.get(piece.instanceId)!) })) };
}
