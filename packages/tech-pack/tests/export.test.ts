import { afterAll, describe, expect, test } from 'bun:test';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { PDFDocument, PDFName } from 'pdf-lib';
import sharp from 'sharp';
import { canonical, emptyDocument, type Artifact, type ExportResult, type ExportSnapshot, type PatternGeometry } from '../../contracts';
import { buildExport, type HandoffManifest } from '../index';
import { checkSanitizedPng, digest } from '../validation';

const tmp = await mkdtemp(join(import.meta.dir, '.export-test-'));
afterAll(async () => { await rm(tmp, { recursive: true, force: true }); });
const encoder = new TextEncoder();
const at = '2026-09-14T17:30:00.000Z';
const sections = ['1. Design overview', '2. Garment flats', '3. Materials / BOM', '4. Finished-garment measurements', '5. Construction', '6. Pattern inventory', '7. Review and change record'];
let outputCounter = 0;

function fixture(): ExportSnapshot {
  const doc = emptyDocument('Original asymmetric shirt', 'A standing collar and unsupported spiral sleeve must remain in the brief.');
  doc.sizeLabel = 'Owner-defined single size';
  doc.garment = { family: 'shirt', length: { state: 'known', value: 65, unit: 'cm', source: 'Owner entered' }, ease: { state: 'assumed', value: 40, unit: 'mm', source: 'Design intent, not fit testing' }, flare: 1.1 };
  doc.body.bust = { state: 'known', value: 913.75, unit: 'mm', source: 'SECRET_BODY_PROVENANCE' };
  doc.requirements = [{ id: 'req_spiral', text: 'Keep original spiral sleeve intent', status: 'unsupported', note: 'Adapter cannot express this sleeve yet' }];
  doc.bom = [
    { id: 'bom_fabric', name: 'Unbleached woven cotton', category: 'fabric', specification: 'Composition owner supplied; weight unknown', placement: 'Main panels', quantity: 'Unknown until marker layout', source: 'Owner entered; not independently verified' },
    { id: 'bom_thread', name: 'Topstitch thread', category: 'trim', specification: 'Colour not selected', placement: 'Visible seams', quantity: 'Unknown', source: 'Owner entered' },
    { id: 'bom_lining', name: 'Facing lining', category: 'lining', specification: 'Unresolved material', placement: 'Collar facing', quantity: 'Unknown', source: 'Owner entered' },
  ];
  doc.poms = [
    { id: 'pom_chest', name: 'Flat chest', method: 'Across garment 2 cm below armhole, relaxed', size: 'Owner-defined single size', target: { state: 'known', value: 54, unit: 'cm', source: 'Owner design target' }, tolerance: { state: 'unknown' }, note: 'Not established' },
    { id: 'pom_length', name: 'Back length', method: 'High point to hem', size: 'Owner-defined single size', target: { state: 'unknown' }, tolerance: { state: 'not-applicable' }, note: 'Awaiting design decision' },
  ];
  doc.construction = [{ id: 'op_collar', operation: 'Collar attachment', note: 'Confirm facing turn and finish before assembly.' }];
  doc.callouts = [{ id: 'callout_collar', anchor: 'operation:op_collar', text: 'Retain unresolved collar question.' }];
  const revision = { id: 'rev_original', projectId: 'project_one', number: 2, parentRevisionId: 'rev_parent', createdAt: at, digest: digest(canonical(doc)), document: doc };
  return { id: 'snapshot_original', projectId: revision.projectId, document: doc, revision, artifacts: [], comments: [{ id: 'comment_one', projectId: revision.projectId, revisionId: revision.id, anchor: 'requirement:req_spiral', text: 'Spiral sleeve is not resolved by this revision.', reportedReviewer: 'Reported maker alias', recordedBy: 'owner', createdAt: at }], disclosure: { includeBody: false, includeReferences: false, includePatterns: false }, createdAt: at };
}

function seal(snapshot: ExportSnapshot): void { snapshot.revision.digest = digest(canonical(snapshot.revision.document)); }
function file(result: ExportResult, filename: string) { const found = result.files.find(f => f.filename === filename); if (!found) throw new Error(`Missing ${filename}`); return found; }
function manifest(result: ExportResult): HandoffManifest { return result.manifest as HandoffManifest; }
async function extracted(result: ExportResult): Promise<string> {
  const filename = join(tmp, `${++outputCounter}.pdf`);
  await writeFile(filename, file(result, 'tech-pack.pdf').bytes);
  const proc = Bun.spawn(['pdftotext', '-layout', filename, '-'], { stdout: 'pipe', stderr: 'pipe' });
  const [text, stderr, code] = await Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text(), proc.exited]);
  if (code !== 0) throw new Error(`pdftotext failed: ${stderr}`);
  return text;
}

function asset(snapshot: ExportSnapshot, id: string, kind: Artifact['kind'], bytes: Uint8Array, mime: string): { artifact: Artifact; bytes: Uint8Array } {
  const artifact: Artifact = { id, projectId: snapshot.revision.projectId, revisionId: snapshot.revision.id, kind, mime, digest: digest(bytes), bytes: bytes.length, jobId: null, createdAt: at, classification: kind === 'pattern-svg' || kind === 'reference' ? 'screen-preview' : 'printable-reference', filename: `${id}.${kind === 'pattern-json' ? 'json' : kind === 'pattern-pdf' ? 'pdf' : kind === 'pattern-svg' ? 'svg' : 'png'}` };
  snapshot.artifacts.push(artifact);
  return { artifact, bytes };
}

async function patternFixture(snapshot = fixture()) {
  const geometry: PatternGeometry = { schemaVersion: 1, units: 'mm', family: 'shirt', classification: 'printable-reference', assumptions: ['Synthetic exporter fixture; not engine validation'], panels: [{ id: 'synthetic_test_panel', name: 'Synthetic test panel', points: [[-5, 0], [95, 0], [75, 140], [0, 130]], widthMm: 100, heightMm: 140 }], stitches: [{ panelA: 'synthetic_test_panel', edgeA: 0, panelB: 'synthetic_test_panel', edgeB: 2 }], warnings: ['Synthetic exporter fixture is not production engine output'], engineVersion: 'test-only-fixture', inputDigest: snapshot.revision.digest };
  const pdf = await PDFDocument.create();
  pdf.addPage([200 * 72 / 25.4, 300 * 72 / 25.4]);
  const assets = [
    asset(snapshot, 'art_json', 'pattern-json', encoder.encode(JSON.stringify(geometry)), 'application/json'),
    asset(snapshot, 'art_svg', 'pattern-svg', encoder.encode('<svg xmlns="http://www.w3.org/2000/svg" width="100mm" height="140mm" viewBox="0 0 100 140"><path d="M0,0 L100,0 L80,140 Z"/></svg>'), 'image/svg+xml'),
    asset(snapshot, 'art_pdf', 'pattern-pdf', await pdf.save(), 'application/pdf'),
  ];
  return { snapshot, geometry, assets };
}

async function referenceFixture(snapshot = fixture(), metadata = false) {
  snapshot.revision.document.views.push({ id: 'ref_front', assetId: 'image_one', role: 'front', kind: 'technical-flat', caption: 'SECRET_IMAGE_LABEL front technical flat; SECRET_REFERENCE_SOURCE owner drawing' });
  seal(snapshot);
  let image = sharp({ create: { width: 120, height: 160, channels: 3, background: { r: 207, g: 224, b: 223 } } });
  if (metadata) image = image.withMetadata({ exif: { IFD0: { Artist: 'SECRET_EXIF_AUTHOR' } } });
  const bytes = await image.png().toBuffer();
  return { snapshot, assets: [asset(snapshot, 'image_one', 'reference', bytes, 'image/png')] };
}

function resealAsset(entry: { artifact: Artifact; bytes: Uint8Array }) { entry.artifact.bytes = entry.bytes.length; entry.artifact.digest = digest(entry.bytes); }

describe('immutable draft tech-pack export', () => {
  test('minimal draft exports all seven substantive sections without geometry', async () => {
    const snapshot = fixture();
    const result = await buildExport(snapshot, null, []);
    expect(result.files.map(f => f.filename)).toEqual(['tech-pack.pdf', 'manifest.json', 'delivery.json']);
    const text = await extracted(result);
    for (const section of sections) expect(text).toContain(section);
    for (const value of ['Back view not supplied', 'Original asymmetric shirt', 'Keep original spiral sleeve intent', 'unsupported', 'Unbleached woven cotton', 'Across garment 2 cm below armhole', '54 cm', 'Awaiting design decision', 'not applicable', 'Collar attachment', 'Spiral sleeve is not resolved', 'owner (reported external identity not authenticated)']) expect(text).toContain(value);
    expect(text).not.toContain('SECRET_BODY_PROVENANCE');
    expect(file(result, 'tech-pack.pdf').bytes.length).toBeGreaterThan(10000);
    expect(manifest(result).sections.overview.garment).toEqual(snapshot.revision.document.garment);
    expect(manifest(result).sections.materials.map(item => item.id)).toEqual(['bom_fabric', 'bom_lining']);
  });

  test('every PDF page repeats complete revision/snapshot identity, hashes, page count and real A4 dimensions', async () => {
    const snapshot = fixture();
    const result = await buildExport(snapshot, null, []);
    const loaded = await PDFDocument.load(file(result, 'tech-pack.pdf').bytes);
    const pages = (await extracted(result)).split('\f').filter(page => page.trim());
    expect(pages.length).toBe(loaded.getPageCount());
    expect(pages.length).toBeGreaterThanOrEqual(7);
    for (const [i, page] of pages.entries()) {
      for (const value of ['Draft review document', `Revision ${snapshot.revision.number}`, snapshot.revision.id, snapshot.id, manifest(result).manifestDigest.slice(0, 16), snapshot.revision.digest.slice(0, 16), `Page ${i + 1} of ${pages.length}`, 'Not cutting ready']) expect(page).toContain(value);
      expect(loaded.getPage(i).getWidth()).toBeCloseTo(210 * 72 / 25.4, 7);
      expect(loaded.getPage(i).getHeight()).toBeCloseTo(297 * 72 / 25.4, 7);
    }
    expect(loaded.getCreationDate()?.toISOString()).toBe(at);
  });

  test('canonical manifest checksum is separate from exact file hashes; no recursive delivery checksum', async () => {
    const result = await buildExport(fixture(), null, []);
    const actual = JSON.parse(new TextDecoder().decode(file(result, 'manifest.json').bytes));
    expect(actual).toEqual(result.manifest);
    const { manifestDigest, ...unsigned } = actual;
    expect(manifestDigest).toBe(digest(canonical(unsigned)));
    expect(manifestDigest).not.toBe(digest(file(result, 'manifest.json').bytes));
    const delivery = JSON.parse(new TextDecoder().decode(file(result, 'delivery.json').bytes));
    expect(delivery.manifestDigest).toBe(manifestDigest);
    expect(delivery.files.map((f: { filename: string }) => f.filename)).not.toContain('delivery.json');
    for (const item of delivery.files) {
      const delivered = file(result, item.filename);
      expect(item.digest).toBe(digest(delivered.bytes));
      expect(item.size).toBe(delivered.bytes.length);
      expect(item.mime).toBe(delivered.mime);
    }
  });

  test('body, reference pixels/captions/metadata, and geometry are absent across default outputs', async () => {
    const refs = await referenceFixture();
    const { snapshot, geometry, assets } = await patternFixture(refs.snapshot);
    geometry.warnings = ['SECRET_GEOMETRY_ASSUMPTION'];
    const result = await buildExport(snapshot, geometry, [...assets, ...refs.assets]);
    const m = manifest(result);
    expect(m.sections).not.toHaveProperty('bodyInputs');
    expect(m.sections.patternInventory).not.toHaveProperty('references');
    expect(m.sections.patternInventory).not.toHaveProperty('panels');
    expect(m.sections.patternInventory).not.toHaveProperty('provenance');
    expect(m.sections.patternInventory.artifacts).toEqual([]);
    expect(m.sections.patternInventory.referenceArtifacts).toEqual([]);
    expect(result.files.length).toBe(3);
    const combined = JSON.stringify(result.manifest) + await extracted(result) + Buffer.from(file(result, 'tech-pack.pdf').bytes).toString('latin1');
    for (const secret of ['SECRET_BODY_PROVENANCE', '913.75', 'SECRET_IMAGE_LABEL', 'SECRET_REFERENCE_SOURCE', 'SECRET_GEOMETRY_ASSUMPTION']) expect(combined).not.toContain(secret);
    const pdf = await PDFDocument.load(file(result, 'tech-pack.pdf').bytes);
    for (const page of pdf.getPages()) expect(page.node.Resources()?.get(PDFName.of('XObject'))?.toString() ?? '').not.toContain('/Subtype /Image');
    expect(m.sections.exportDisclosure.privacyNotice).toContain('not anonymization');
  });

  test('default export works without withheld asset bytes and never examines withheld malformed files', async () => {
    const { snapshot, geometry } = await patternFixture((await referenceFixture()).snapshot);
    const result = await buildExport(snapshot, geometry, []);
    expect(result.files.length).toBe(3);
  });

  test('explicit body inclusion preserves values, units and provenance without deriving POMs', async () => {
    const snapshot = fixture();
    snapshot.disclosure.includeBody = true;
    const result = await buildExport(snapshot, null, []);
    expect(manifest(result).sections.bodyInputs).toEqual(snapshot.revision.document.body);
    expect(manifest(result).sections.finishedMeasurements).toEqual(snapshot.revision.document.poms);
    const text = await extracted(result);
    expect(text).toContain('913.75 mm');
    expect(text).toContain('SECRET_BODY_PROVENANCE');
  });

  test('explicit sanitized reference inclusion embeds real flats, reports missing back and preserves image bytes', async () => {
    const { snapshot, assets } = await referenceFixture();
    snapshot.disclosure.includeReferences = true;
    const result = await buildExport(snapshot, null, assets);
    const record = manifest(result).sections.patternInventory.referenceArtifacts[0]!;
    expect(file(result, record.deliveryFilename).bytes).toEqual(Uint8Array.from(assets[0]!.bytes));
    expect(manifest(result).sections.patternInventory.references).toEqual(snapshot.revision.document.views);
    const text = await extracted(result);
    expect(text).toContain('SECRET_IMAGE_LABEL');
    expect(text).toContain('Back view not supplied');
    const pdf = await PDFDocument.load(file(result, 'tech-pack.pdf').bytes);
    expect(pdf.context.enumerateIndirectObjects().some(([, object]) => object.toString().includes('/Subtype /Image'))).toBe(true);
  });

  test('pattern artifacts are byte-identical with original digests and measured PDF boxes', async () => {
    const { snapshot, geometry, assets } = await patternFixture();
    snapshot.disclosure.includePatterns = true;
    const result = await buildExport(snapshot, geometry, assets);
    expect(result.files.length).toBe(6);
    for (const record of manifest(result).sections.patternInventory.artifacts) {
      const source = assets.find(a => a.artifact.id === record.id)!;
      expect(file(result, record.deliveryFilename).bytes).toEqual(source.bytes);
      expect(digest(file(result, record.deliveryFilename).bytes)).toBe(source.artifact.digest);
    }
    const page = manifest(result).sections.patternInventory.artifacts.find(a => a.kind === 'pattern-pdf')!.pages![0]!;
    expect(page.widthMm).toBeCloseTo(200, 8);
    expect(page.heightMm).toBeCloseTo(300, 8);
    const text = await extracted(result);
    expect(text).toContain('Synthetic exporter fixture');
    expect(text).toContain('Panel bounding box: 100 x 140 mm');
    expect(text).toContain('reference-only and NOT cutting ready');
    expect(text).toContain('200.000 x 300.000 mm');
    expect(text).toContain('not a calibration result');
  });

  test('same frozen pattern bytes survive multiple export snapshots without relabeling', async () => {
    const { snapshot, geometry, assets } = await patternFixture();
    snapshot.disclosure.includePatterns = true;
    const first = await buildExport(snapshot, geometry, assets);
    const later = structuredClone(snapshot);
    later.id = 'snapshot_later'; later.createdAt = '2026-09-15T17:30:00.000Z';
    const second = await buildExport(later, geometry, assets);
    expect(manifest(first).manifestDigest).not.toBe(manifest(second).manifestDigest);
    for (const artifact of manifest(first).sections.patternInventory.artifacts) expect(file(first, artifact.deliveryFilename).bytes).toEqual(file(second, artifact.deliveryFilename).bytes);
  });

  test('captures all inputs synchronously before await; concurrent editor/assets changes cannot replace selected snapshot', async () => {
    const { snapshot, geometry, assets } = await patternFixture();
    snapshot.disclosure.includePatterns = true;
    const original = structuredClone({ snapshot, geometry, assets });
    const inFlight = buildExport(snapshot, geometry, assets);
    snapshot.revision.document.title = 'WRONG LATER EDIT';
    snapshot.revision.document.poms[0]!.target = { state: 'known', value: 9999, unit: 'mm', source: 'WRONG LATER EDIT' };
    snapshot.comments[0]!.text = 'WRONG LATER COMMENT';
    geometry.panels[0]!.points[0]![0] = 5000;
    assets[2]!.bytes.fill(0);
    const result = await inFlight;
    expect(manifest(result).sections.overview.title).toBe(original.snapshot.revision.document.title);
    expect(manifest(result).sections.review.comments).toEqual(original.snapshot.comments);
    expect(manifest(result).sections.finishedMeasurements).toEqual(original.snapshot.revision.document.poms);
    expect(file(result, 'pattern-art_pdf.pdf').bytes).toEqual(original.assets[2]!.bytes);
    expect(await extracted(result)).not.toContain('WRONG LATER');
  });

  test('calling exporter never mutates source records or aliases returned sections', async () => {
    const snapshot = fixture();
    const before = canonical(snapshot);
    const result = await buildExport(snapshot, null, []);
    expect(canonical(snapshot)).toBe(before);
    manifest(result).sections.overview.garment.flare = 2;
    manifest(result).sections.bom[0]!.name = 'New returned name';
    expect(canonical(snapshot)).toBe(before);
    expect(manifest(result).sections.materials[0]!.name).toBe('Unbleached woven cotton');
  });

  test('accents and supported Unicode remain readable; unsupported and unsafe display characters are disclosed', async () => {
    const snapshot = fixture();
    snapshot.revision.document.title = 'Café naïve blå déjà vu Ελληνικά Москва';
    snapshot.revision.document.brief = '漢字 🪡 مرحبا שלום \u202E malicious-direction \u0000 a\u0301';
    seal(snapshot);
    const result = await buildExport(snapshot, null, []);
    const text = await extracted(result);
    expect(text).toContain('Café naïve blå déjà vu Ελληνικά Москва');
    expect(text).toContain('[U+6F22]');
    expect(text).toContain('[U+1FAA1]');
    expect(text).toContain('[U+202E]');
    expect(text).toContain('[U+0000]');
    expect(manifest(result).sections.overview.brief).toBe(snapshot.revision.document.brief);
    expect(manifest(result).sections.exportDisclosure.textRendering.replacements).toEqual(expect.arrayContaining(['U+6F22', 'U+1FAA1', 'U+202E', 'U+0000']));
    const raw = Buffer.from(file(result, 'tech-pack.pdf').bytes).toString('latin1');
    expect(raw).toContain('/FontFile2');
    expect(raw).not.toContain('/EmbeddedFiles');
    expect(await extracted(result)).toContain('Permission is hereby granted');
    expect(await readFile(new URL('../assets/FONT-LICENSE.md', import.meta.url), 'utf8')).toContain('Permission is hereby granted');
  });

  test('long multilingual BOM and unbroken text paginate with no lost terminal fields', async () => {
    const snapshot = fixture();
    snapshot.revision.document.bom = Array.from({ length: 28 }, (_, i) => ({ id: `row_${i}`, name: `Tissu café ${i}`, category: 'fabric' as const, specification: `Row ${i}: ${'é'.repeat(160)} ${'description '.repeat(15)}END_ROW_${i}`, placement: 'Collar and main panels', quantity: 'Unknown', source: 'Owner entered' }));
    seal(snapshot);
    const result = await buildExport(snapshot, null, []);
    const text = await extracted(result);
    for (let i = 0; i < 28; i++) expect(text).toContain(`END_ROW_${i}`);
    expect(text).toContain('3. Materials / BOM (continued)');
    const pdf = await PDFDocument.load(file(result, 'tech-pack.pdf').bytes);
    expect(pdf.getPageCount()).toBeGreaterThan(12);
    for (const [i, page] of text.split('\f').filter(page => page.trim()).entries()) expect(page).toContain(`Page ${i + 1} of ${pdf.getPageCount()}`);
  }, 30000);

  test('empty document has explicit unknowns and no invented finished garment data', async () => {
    const snapshot = fixture();
    snapshot.revision.document = emptyDocument(); snapshot.document = snapshot.revision.document; snapshot.comments = []; seal(snapshot);
    const result = await buildExport(snapshot, null, []);
    const text = await extracted(result);
    expect(text).toContain('No finished-garment POMs supplied');
    expect(text).toContain('Construction instructions not supplied');
    expect(text).toContain('Geometry is not used to invent them');
    expect(manifest(result).sections.finishedMeasurements).toEqual([]);
  });
});

describe('fail closed on included assets and incompatible snapshot state', () => {
  test('requested patterns with no saved artifacts fail explicitly', async () => {
    const snapshot = fixture(); snapshot.disclosure.includePatterns = true;
    await expect(buildExport(snapshot, null, [])).rejects.toThrow('Requested patterns are missing');
  });
  test('missing one included file fails instead of silently dropping it', async () => {
    const { snapshot, geometry, assets } = await patternFixture(); snapshot.disclosure.includePatterns = true;
    await expect(buildExport(snapshot, geometry, assets.slice(0, 2))).rejects.toThrow('Missing included artifact');
  });
  test('same-size byte corruption fails digest verification', async () => {
    const { snapshot, geometry, assets } = await patternFixture(); snapshot.disclosure.includePatterns = true;
    assets[2]!.bytes[12] ^= 1;
    await expect(buildExport(snapshot, geometry, assets)).rejects.toThrow('Artifact digest mismatch');
  });
  test('wrong declared byte size fails', async () => {
    const { snapshot, geometry, assets } = await patternFixture(); snapshot.disclosure.includePatterns = true;
    assets[2]!.artifact.bytes += 1;
    await expect(buildExport(snapshot, geometry, assets)).rejects.toThrow('Artifact size mismatch');
  });
  test('mutated revision document is rejected against its original digest', async () => {
    const snapshot = fixture(); snapshot.revision.document.title = 'A later mutable draft';
    await expect(buildExport(snapshot, null, [])).rejects.toThrow('Revision content digest mismatch');
  });
  test('artifact from another project or revision is rejected even if withheld', async () => {
    const { snapshot } = await patternFixture(); snapshot.artifacts[0]!.projectId = 'another_project';
    await expect(buildExport(snapshot, null, [])).rejects.toThrow('Artifact does not belong');
    snapshot.artifacts[0]!.projectId = snapshot.revision.projectId; snapshot.artifacts[0]!.revisionId = 'another_revision';
    await expect(buildExport(snapshot, null, [])).rejects.toThrow('Artifact does not belong');
  });
  test('unscoped review and duplicate IDs are rejected', async () => {
    const snapshot = fixture(); snapshot.comments[0]!.revisionId = 'another_revision';
    await expect(buildExport(snapshot, null, [])).rejects.toThrow('Comment does not belong');
    snapshot.comments[0]!.revisionId = snapshot.revision.id; snapshot.comments.push(structuredClone(snapshot.comments[0]!));
    await expect(buildExport(snapshot, null, [])).rejects.toThrow('Duplicate review comment');
  });
  test('arbitrary asset injection and metadata substitution are rejected', async () => {
    const { snapshot, geometry, assets } = await patternFixture();
    const injected = structuredClone(assets[0]!); injected.artifact.id = 'not_in_snapshot';
    await expect(buildExport(snapshot, geometry, [...assets, injected])).rejects.toThrow('not in the frozen snapshot');
    const altered = structuredClone(assets[0]!); altered.artifact.classification = 'screen-preview';
    await expect(buildExport(snapshot, geometry, [altered])).rejects.toThrow('not in the frozen snapshot');
  });
  test('duplicate supplied files and duplicate snapshot artifacts are rejected', async () => {
    const { snapshot, geometry, assets } = await patternFixture();
    await expect(buildExport(snapshot, geometry, [...assets, assets[0]!])).rejects.toThrow('Duplicate supplied artifact');
    snapshot.artifacts.push(structuredClone(snapshot.artifacts[0]!));
    await expect(buildExport(snapshot, geometry, assets)).rejects.toThrow('Duplicate snapshot artifact');
  });
  test('geometry from later job cannot replace the checked pattern JSON', async () => {
    const { snapshot, geometry, assets } = await patternFixture(); snapshot.disclosure.includePatterns = true;
    geometry.warnings.push('Later job output');
    await expect(buildExport(snapshot, geometry, assets)).rejects.toThrow('does not match the committed pattern JSON');
  });
  test('checked geometry still requires correct immutable document digest', async () => {
    const { snapshot, geometry, assets } = await patternFixture(); snapshot.disclosure.includePatterns = true;
    geometry.inputDigest = digest('wrong input'); assets[0]!.bytes = encoder.encode(JSON.stringify(geometry)); resealAsset(assets[0]!);
    await expect(buildExport(snapshot, geometry, assets)).rejects.toThrow('Geometry input digest does not match');
  });
  test('invalid units and inconsistent panel dimensions fail rather than claiming millimeters', async () => {
    const { snapshot, geometry, assets } = await patternFixture(); snapshot.disclosure.includePatterns = true;
    geometry.panels[0]!.widthMm = 1; assets[0]!.bytes = encoder.encode(JSON.stringify(geometry)); resealAsset(assets[0]!);
    await expect(buildExport(snapshot, geometry, assets)).rejects.toThrow('Panel bounds');
    (geometry as unknown as { units: string }).units = 'cm'; assets[0]!.bytes = encoder.encode(JSON.stringify(geometry)); resealAsset(assets[0]!);
    await expect(buildExport(snapshot, geometry, assets)).rejects.toThrow();
  });
  test('no legacy or unknown schema properties are accepted', async () => {
    const snapshot = fixture(); (snapshot.revision.document as unknown as Record<string, unknown>).approval = 'ready for cutting';
    await expect(buildExport(snapshot, null, [])).rejects.toThrow();
  });
  test('MIME spoofing and unsafe filenames are rejected', async () => {
    const { snapshot, geometry, assets } = await patternFixture(); snapshot.disclosure.includePatterns = true;
    assets[2]!.artifact.mime = 'text/html';
    await expect(buildExport(snapshot, geometry, assets)).rejects.toThrow('MIME');
    assets[2]!.artifact.mime = 'application/pdf'; assets[2]!.artifact.filename = '../../escape.pdf';
    await expect(buildExport(snapshot, geometry, assets)).rejects.toThrow();
  });
  test('malformed included PDF is rejected even when its checksum matches', async () => {
    const { snapshot, geometry, assets } = await patternFixture(); snapshot.disclosure.includePatterns = true;
    assets[2]!.bytes = encoder.encode('Not a PDF'); resealAsset(assets[2]!);
    await expect(buildExport(snapshot, geometry, assets)).rejects.toThrow('PDF header');
  });
  test.each(['<svg><script>alert(1)</script></svg>', '<svg><image href="https://example.com/private.png"/></svg>', '<svg><path onload="alert(1)"/></svg>'])('unsafe SVG subset rejected: %s', async svg => {
    const { snapshot, geometry, assets } = await patternFixture(); snapshot.disclosure.includePatterns = true;
    assets[1]!.bytes = encoder.encode(svg); resealAsset(assets[1]!);
    await expect(buildExport(snapshot, geometry, assets)).rejects.toThrow();
  });
  test('explicitly requested missing references fail', async () => {
    const { snapshot } = await referenceFixture(); snapshot.disclosure.includeReferences = true;
    await expect(buildExport(snapshot, null, [])).rejects.toThrow('Missing included artifact');
    snapshot.artifacts = [];
    await expect(buildExport(snapshot, null, [])).rejects.toThrow('Missing included reference');
  });
  test('PNG metadata is rejected and not embedded or silently leaked', async () => {
    const { snapshot, assets } = await referenceFixture(fixture(), true); snapshot.disclosure.includeReferences = true;
    await expect(buildExport(snapshot, null, assets)).rejects.toThrow('metadata or invalid chunks');
  });
  test('unused reference inclusion is forbidden', async () => {
    const { snapshot, assets } = await referenceFixture(); snapshot.revision.document.views = []; seal(snapshot); snapshot.disclosure.includeReferences = true;
    await expect(buildExport(snapshot, null, assets)).rejects.toThrow('not used by this revision');
  });
  test('PNG pixel bounds, truncation, trailing payload, and non-PNG inputs fail', async () => {
    const { assets } = await referenceFixture();
    expect(() => checkSanitizedPng(assets[0]!.bytes)).not.toThrow();
    const huge = Buffer.from(assets[0]!.bytes); huge.writeUInt32BE(1000000, 16);
    expect(() => checkSanitizedPng(huge)).toThrow('pixel limits');
    expect(() => checkSanitizedPng(assets[0]!.bytes.slice(0, -5))).toThrow('incomplete');
    expect(() => checkSanitizedPng(Buffer.concat([assets[0]!.bytes, Buffer.from('secret')]))).toThrow('terminator');
    expect(() => checkSanitizedPng(encoder.encode('<svg/>'))).toThrow('sanitized PNG');
  });
});
