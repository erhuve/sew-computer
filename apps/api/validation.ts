import { createHash, randomBytes } from 'node:crypto';
import { z } from 'zod';
import { canonical, DocumentSchema, type GarmentDocument, type PatternGeometry } from '../../packages/contracts';
import { DraftingSchema, PanelDraftSchema, validateDrafting } from '../../packages/contracts/design';

export class ApiError extends Error {
  constructor(public status: number, message: string, public extra: Record<string, unknown> = {}) { super(message); }
}
export const now = () => new Date().toISOString();
export const id = () => randomBytes(18).toString('base64url');
export const hash = (value: Uint8Array | string) => createHash('sha256').update(value).digest('hex');
export const objectDigest = (value: unknown) => hash(canonical(value));
export const filenameSchema = z.string().regex(/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,159}$/).refine(v => !v.includes('..'));
export const identitySchema = z.object({expectedVersion:z.number().int().min(1),expectedRevisionId:z.string().nullable()}).strict();
export const maxJsonBytes = 1024 * 1024;
export const maxFileBytes = 32 * 1024 * 1024;

export function cleanObject(value: unknown, depth = 0, count = {n:0}): void {
  if (++count.n > 100000 || depth > 40) throw new ApiError(422, 'Input is too complex');
  if (typeof value === 'number' && !Number.isFinite(value)) throw new ApiError(422, 'Non-finite number');
  if (!value || typeof value !== 'object') return;
  if (!Array.isArray(value) && Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null) throw new ApiError(422, 'Invalid object');
  for (const [key, child] of Object.entries(value)) {
    if (['__proto__','prototype','constructor'].includes(key)) throw new ApiError(422, 'Forbidden object key');
    cleanObject(child, depth + 1, count);
  }
}
export function document(value: unknown): GarmentDocument {
  cleanObject(value);
  const parsed = DocumentSchema.parse(value);
  for (const rows of [parsed.requirements,parsed.bom,parsed.poms,parsed.construction,parsed.callouts,parsed.views]) {
    if (new Set(rows.map(r => r.id)).size !== rows.length) throw new ApiError(422, 'Duplicate row ID');
  }
  return parsed;
}
export async function readBounded(request: Request, limit: number): Promise<Uint8Array> {
  const length = request.headers.get('content-length');
  if (length && (!/^\d+$/.test(length) || Number(length) > limit)) throw new ApiError(413, 'Request is too large');
  if (!request.body) return new Uint8Array();
  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    for (;;) {
      const part = await reader.read();
      if (part.done) break;
      size += part.value.length;
      if (size > limit) { await reader.cancel(); throw new ApiError(413, 'Request is too large'); }
      chunks.push(part.value);
    }
  } finally { reader.releaseLock(); }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes;
}
export async function json(request: Request): Promise<unknown> {
  if (request.headers.get('content-type')?.split(';')[0].trim() !== 'application/json') throw new ApiError(415, 'Expected application/json');
  const bytes = await readBounded(request, maxJsonBytes);
  let value: unknown;
  try { value = JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes)); } catch { throw new ApiError(422, 'Invalid JSON'); }
  cleanObject(value);
  return value;
}
const numeric = z.number().finite().min(-100000).max(100000);
const geometrySchema = z.object({
  schemaVersion:z.literal(1),units:z.literal('mm'),inputDigest:z.string().regex(/^[a-f0-9]{64}$/),
  engineVersion:z.string().min(1).max(500),family:z.enum(['shirt','skirt','trousers']),
  panels:z.array(z.object({id:z.string().min(1).max(160),name:z.string().min(1).max(300),points:z.array(z.tuple([numeric,numeric])).min(3).max(20000),widthMm:z.number().finite().positive().max(100000),heightMm:z.number().finite().positive().max(100000),cutQuantity:z.number().int().positive().max(100).optional(),draft:PanelDraftSchema.optional()}).strict()).min(1).max(100),
  stitches:z.array(z.object({panelA:z.string(),edgeA:z.number().int().nonnegative(),panelB:z.string(),edgeB:z.number().int().nonnegative()}).strict()).max(10000),
  warnings:z.array(z.string().max(8000)).max(200),assumptions:z.array(z.string().max(8000)).max(200),classification:z.literal('printable-reference'),
  drafting:DraftingSchema.optional(),
}).strict();
export function geometry(value: unknown, digest: string): PatternGeometry {
  const parsed = geometrySchema.parse(value);
  validateDrafting(parsed);
  if (parsed.inputDigest !== digest) throw new ApiError(422, 'Geometry input digest mismatch');
  const panels = new Map(parsed.panels.map(p => [p.id,p]));
  if (panels.size !== parsed.panels.length || parsed.panels.reduce((n,p) => n+p.points.length,0)>200000) throw new ApiError(422, 'Invalid panel inventory');
  for (const panel of parsed.panels) {
    let area = 0;
    for (let i=0;i<panel.points.length;i++) { const a=panel.points[i]!, b=panel.points[(i+1)%panel.points.length]!; area += a[0]*b[1]-b[0]*a[1]; }
    if (Math.abs(area)<0.000001) throw new ApiError(422,'Degenerate zero-area panel');
    const xs = panel.points.map(p=>p[0]), ys = panel.points.map(p=>p[1]);
    const w = Math.max(...xs)-Math.min(...xs), h = Math.max(...ys)-Math.min(...ys);
    if (Math.abs(w-panel.widthMm)>0.2 || Math.abs(h-panel.heightMm)>0.2) throw new ApiError(422, 'Panel bounds mismatch');
  }
  for (const seam of parsed.stitches) {
    if (!panels.has(seam.panelA) || !panels.has(seam.panelB) || seam.edgeA >= (panels.get(seam.panelA)!.draft?.edges.length ?? panels.get(seam.panelA)!.points.length) || seam.edgeB >= (panels.get(seam.panelB)!.draft?.edges.length ?? panels.get(seam.panelB)!.points.length)) throw new ApiError(422, 'Invalid stitch reference');
  }
  return parsed;
}
export function checkFile(filename: string, bytes: Uint8Array, mime: string): void {
  filenameSchema.parse(filename);
  if (!(bytes instanceof Uint8Array) || bytes.byteLength < 1 || bytes.byteLength > maxFileBytes) throw new ApiError(422, 'Invalid artifact size');
  const text = () => new TextDecoder('utf-8',{fatal:true}).decode(bytes);
  if (mime === 'application/json') { cleanObject(JSON.parse(text())); }
  else if (mime === 'image/svg+xml') {
    const svg = text();
    if (!/<svg[\s>]/.test(svg) || /<!DOCTYPE|<!ENTITY|<\?(?!xml\s)|\bon[a-z]+\s*=|(?:href|src)\s*=\s*["'](?!#)|url\(\s*["']?(?!#)|@import|@font-face|expression\s*\(/i.test(svg)) throw new ApiError(422, 'Unsafe SVG');
    const tags = [...svg.matchAll(/<\/?([a-zA-Z][\w:-]*)\b/g)].map(m=>m[1]!.toLowerCase());
    const allowed = new Set(['svg','g','path','rect','circle','ellipse','line','polyline','polygon','text','tspan','defs','style','clippath','use','symbol','title','desc']);
    if (tags.some(t=>!allowed.has(t)) || /(?:href|src)\s*=\s*[^\s"']/i.test(svg)) throw new ApiError(422, 'Unsafe SVG element');
  } else if (mime === 'application/pdf') {
    if (Buffer.from(bytes.subarray(0,5)).toString() !== '%PDF-') throw new ApiError(422, 'Invalid PDF');
    if (/\/(?:JavaScript|JS|Launch|URI|GoToR|EmbeddedFile|OpenAction|AA|RichMedia)\b/.test(Buffer.from(bytes).toString('latin1'))) throw new ApiError(422, 'Active PDF content is not allowed');
  } else if (mime !== 'image/png') throw new ApiError(422, 'Unsupported artifact MIME');
}
