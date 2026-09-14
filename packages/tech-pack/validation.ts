import { createHash } from 'node:crypto';
import { z } from 'zod';
import { canonical, DisclosureSchema as disclosureSchema, DocumentSchema as documentSchema, Id as idSchema, type Artifact, type ExportSnapshot, type PatternGeometry } from '../contracts';

export const digest = (value: Uint8Array | string): string => createHash('sha256').update(value).digest('hex');
const checksum = z.string().regex(/^[a-f0-9]{64}$/);
const date = z.string().datetime({ offset: true });
const bounded = (max: number) => z.string().max(max);
const artifactSchema = z.object({
  id: idSchema, projectId: idSchema, revisionId: idSchema, jobId: idSchema.nullable(), createdAt: date,
  kind: z.enum(['pattern-json', 'pattern-svg', 'pattern-pdf', 'reference']),
  mime: bounded(100), digest: checksum, bytes: z.number().int().positive().max(64 * 1024 * 1024),
  classification: z.enum(['screen-preview', 'printable-reference']),
  filename: z.string().regex(/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,159}$/),
}).strict();
const snapshotSchema: z.ZodType<ExportSnapshot> = z.object({
  id: idSchema, projectId: idSchema, document: documentSchema,
  revision: z.object({
    id: idSchema, projectId: idSchema, number: z.number().int().positive(), parentRevisionId: idSchema.nullable(),
    createdAt: date, digest: checksum, document: documentSchema,
  }).strict(),
  artifacts: z.array(artifactSchema).max(200),
  comments: z.array(z.object({
    id: idSchema, projectId: idSchema, revisionId: idSchema, anchor: bounded(200), text: bounded(4000),
    reportedReviewer: bounded(300), recordedBy: z.literal('owner'), createdAt: date,
  }).strict()).max(500),
  disclosure: disclosureSchema, createdAt: date,
}).strict();
const geometrySchema: z.ZodType<PatternGeometry> = z.object({
  schemaVersion: z.literal(1), units: z.literal('mm'), family: bounded(100),
  panels: z.array(z.object({
    id: bounded(200).min(1), name: bounded(200).min(1),
    points: z.array(z.tuple([z.number().finite().min(-1e6).max(1e6), z.number().finite().min(-1e6).max(1e6)])).min(3).max(100000),
    widthMm: z.number().finite().positive().max(1e6), heightMm: z.number().finite().positive().max(1e6), cutQuantity: z.number().int().positive().optional(),
  }).strict()).min(1).max(200),
  stitches: z.array(z.object({ panelA: bounded(300), panelB: bounded(300), edgeA: z.number().int().nonnegative(), edgeB: z.number().int().nonnegative() }).strict()).max(2000),
  warnings: z.array(bounded(30000)).max(200), engineVersion: bounded(300).min(1),
  assumptions: z.array(bounded(30000)).max(200), classification: z.literal('printable-reference'), inputDigest: checksum,
}).strict();

export interface VerifiedAsset { artifact: Artifact; bytes: Uint8Array; filename: string }
export interface CapturedExport { snapshot: ExportSnapshot; geometry: PatternGeometry | null; patterns: VerifiedAsset[]; references: VerifiedAsset[] }

export function captureExport(snapshot: ExportSnapshot, geometry: PatternGeometry | null, assets: { artifact: Artifact; bytes: Uint8Array }[]): CapturedExport {
  const frozen = snapshotSchema.parse(snapshot);
  if (digest(canonical(frozen.revision.document)) !== frozen.revision.digest) throw new Error('Revision content digest mismatch');
  const { projectId, id: revisionId } = frozen.revision;
  if (frozen.projectId !== projectId || canonical(frozen.document) !== canonical(frozen.revision.document)) throw new Error('Snapshot document does not match saved revision');
  const artifacts = new Map<string, Artifact>();
  for (const artifact of frozen.artifacts) {
    if (artifacts.has(artifact.id)) throw new Error('Duplicate snapshot artifact ID');
    if (artifact.projectId !== projectId || artifact.revisionId !== revisionId) throw new Error('Artifact does not belong to the snapshot revision');
    artifacts.set(artifact.id, artifact);
  }
  const commentIds = new Set<string>();
  for (const comment of frozen.comments) {
    if (comment.revisionId !== revisionId) throw new Error('Comment does not belong to the snapshot revision');
    if (commentIds.has(comment.id)) throw new Error('Duplicate review comment ID');
    commentIds.add(comment.id);
  }
  const provided = new Map<string, { artifact: Artifact; bytes: Uint8Array }>();
  for (const asset of assets) {
    const expected = artifacts.get(asset.artifact.id);
    if (!expected || canonical(expected) !== canonical(asset.artifact)) throw new Error('Supplied artifact is not in the frozen snapshot');
    if (provided.has(expected.id)) throw new Error('Duplicate supplied artifact');
    provided.set(expected.id, asset);
  }
  const expectedMime = { 'pattern-json': 'application/json', 'pattern-svg': 'image/svg+xml', 'pattern-pdf': 'application/pdf', reference: 'image/png' };
  const extension = { 'pattern-json': 'json', 'pattern-svg': 'svg', 'pattern-pdf': 'pdf', reference: 'png' };
  let total = 0;
  const verify = (artifact: Artifact): VerifiedAsset => {
    const source = provided.get(artifact.id);
    if (!source) throw new Error(`Missing included artifact: ${artifact.id}`);
    if (!(source.bytes instanceof Uint8Array)) throw new Error('Artifact bytes must be Uint8Array');
    if (source.bytes.byteLength !== artifact.bytes) throw new Error(`Artifact size mismatch: ${artifact.id}`);
    total += source.bytes.byteLength;
    if (total > 128 * 1024 * 1024) throw new Error('Export assets exceed the 128 MiB limit');
    const bytes = Uint8Array.from(source.bytes);
    if (digest(bytes) !== artifact.digest) throw new Error(`Artifact digest mismatch: ${artifact.id}`);
    if (artifact.mime !== expectedMime[artifact.kind]) throw new Error('Artifact MIME does not match its kind');
    return { artifact, bytes, filename: `${artifact.kind === 'reference' ? 'reference' : 'pattern'}-${artifact.id}.${extension[artifact.kind]}` };
  };
  let checkedGeometry: PatternGeometry | null = null;
  const patterns: VerifiedAsset[] = [];
  const references: VerifiedAsset[] = [];
  if (frozen.disclosure.includePatterns) {
    const selected = frozen.artifacts.filter(a => a.kind !== 'reference');
    if (!selected.length || !geometry) throw new Error('Requested patterns are missing; save and generate this revision first');
    patterns.push(...selected.map(verify));
    const json = patterns.filter(a => a.artifact.kind === 'pattern-json');
    if (json.length !== 1) throw new Error('Exactly one committed pattern JSON is required');
    if (json[0]!.bytes.length > 32 * 1024 * 1024) throw new Error('Pattern JSON exceeds the 32 MiB limit');
    const committed = geometrySchema.parse(JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(json[0]!.bytes)));
    checkedGeometry = geometrySchema.parse(geometry);
    if (canonical(committed) !== canonical(checkedGeometry)) throw new Error('Geometry does not match the committed pattern JSON');
    if (checkedGeometry.inputDigest !== frozen.revision.digest) throw new Error('Geometry input digest does not match the revision');
    const ids = new Set<string>();
    let points = 0;
    for (const panel of checkedGeometry.panels) {
      if (ids.has(panel.id)) throw new Error('Duplicate geometry panel ID');
      ids.add(panel.id);
      points += panel.points.length;
      if (points > 500000) throw new Error('Geometry exceeds the point limit');
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const [x, y] of panel.points) { minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y); }
      if (Math.abs(maxX - minX - panel.widthMm) > 0.1 || Math.abs(maxY - minY - panel.heightMm) > 0.1) throw new Error('Panel bounds do not match its coordinates');
    }
  }
  if (frozen.disclosure.includeReferences) {
    const required = new Set(frozen.revision.document.views.map(r => r.assetId));
    for (const assetId of required) {
      const artifact = artifacts.get(assetId);
      if (!artifact || artifact.kind !== 'reference') throw new Error(`Missing included reference: ${assetId}`);
      references.push(verify(artifact));
    }
    if (frozen.artifacts.some(a => a.kind === 'reference' && !required.has(a.id))) throw new Error('Snapshot includes a reference not used by this revision');
  }
  return { snapshot: frozen, geometry: checkedGeometry, patterns, references };
}

export function checkSanitizedPng(bytes: Uint8Array): void {
  const data = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (data.length > 10 * 1024 * 1024 || !data.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) throw new Error('Reference must be a bounded sanitized PNG');
  const chunks = new Set(['IHDR', 'PLTE', 'IDAT', 'IEND', 'tRNS', 'pHYs']);
  let offset = 8, sawHeader = false, sawData = false;
  while (offset + 12 <= data.length) {
    const length = data.readUInt32BE(offset);
    const kind = data.toString('ascii', offset + 4, offset + 8);
    if (!chunks.has(kind) || offset + length + 12 > data.length) throw new Error('Reference PNG contains metadata or invalid chunks');
    if (!sawHeader && kind !== 'IHDR') throw new Error('Reference PNG header missing');
    if (kind === 'IHDR') {
      if (sawHeader || length !== 13) throw new Error('Reference PNG header invalid');
      const width = data.readUInt32BE(offset + 8), height = data.readUInt32BE(offset + 12);
      if (!width || !height || width * height > 20 * 1024 * 1024) throw new Error('Reference PNG exceeds pixel limits');
      sawHeader = true;
    }
    if (kind === 'IDAT') sawData = true;
    offset += length + 12;
    if (kind === 'IEND') {
      if (length || !sawData || offset !== data.length) throw new Error('Reference PNG terminator invalid');
      return;
    }
  }
  throw new Error('Reference PNG is incomplete');
}
