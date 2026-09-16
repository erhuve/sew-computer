import { PDFDocument, type PDFImage } from 'pdf-lib';
import sharp from 'sharp';
import { canonical, type Artifact, type ExportFile, type ExportResult, type ExportSnapshot, type GarmentDocument, type PatternGeometry, type ReviewComment } from '../contracts';
import { createPdfContext, PAPER, renderPdf } from './pdf';
import { captureExport, checkSanitizedPng, digest, type VerifiedAsset } from './validation';
import { designCoverage } from '../contracts/design';

export interface PatternPdfPage {
  page: number; xPt: number; yPt: number; widthPt: number; heightPt: number; widthMm: number; heightMm: number; rotationDegrees: number;
  cropBox: { x: number; y: number; width: number; height: number };
}
export interface DeliveredArtifact extends Artifact { deliveryFilename: string; pages?: PatternPdfPage[] }
export interface HandoffManifest extends Record<string, unknown> {
  schemaVersion: 1; snapshotId: string; revisionId: string; projectId: string; manifestDigest: string;
  revisionNumber: number; parentRevisionId: string | null; revisionCreatedAt: string; snapshotCreatedAt: string; inputDigest: string;
  renderer: 'sew-computer-tech-pack/1';
  sections: {
    overview: Pick<GarmentDocument, 'title' | 'brief' | 'sizeLabel' | 'garment' | 'interpretation'>;
    requirements: GarmentDocument['requirements']; materials: GarmentDocument['bom']; bom: GarmentDocument['bom'];
    bodyInputs?: GarmentDocument['body']; finishedMeasurements: GarmentDocument['poms']; construction: GarmentDocument['construction'];
    patternInventory: {
      status: 'included' | 'omitted_by_disclosure' | 'not_generated'; notice: string;
      flats: Record<'front' | 'back' | 'detail', 'included' | 'withheld' | 'not_supplied'>;
      artifacts: DeliveredArtifact[]; referenceArtifacts: DeliveredArtifact[]; references?: GarmentDocument['views'];
      panels?: Omit<PatternGeometry['panels'][number], 'points'>[];
      provenance?: Pick<PatternGeometry, 'engineVersion' | 'inputDigest' | 'units' | 'warnings'>;
    };
    review: { callouts: GarmentDocument['callouts']; comments: ReviewComment[] };
    derivedConstruction?: NonNullable<PatternGeometry['drafting']>;
    designCoverage?: ReturnType<typeof designCoverage>;
    exportDisclosure: ExportSnapshot['disclosure'] & {
      omissions: string[]; classification: 'Draft review document'; privacyNotice: string; pageFormat: typeof PAPER;
      textRendering: { font: string; policy: string; replacements: string[] };
    };
  };
}

const PRIVACY_NOTICE = 'Body inputs and source images are omitted unless selected. Garment parameters, finished measurements, free-text notes and included geometry may still reveal body-related or personal information; redaction is not anonymization. Review these owner-entered fields before sharing.';
const PATTERN_NOTICE = 'Generated garment patterns are reference-only and NOT cutting ready. No cutting-candidate classification, print calibration or physical-fit evidence is supplied. Component drafts report scoped digital seam/annotation checks separately; legacy base patterns do not. When selected, original pattern files are included byte-for-byte, never restamped or rescaled; bundle identity belongs to this manifest and the separate tech pack.';
const FONT_POLICY = 'PDF text uses embedded DejaVu Sans and NFC display normalization. Unsupported glyphs, control/bidirectional formatting characters and right-to-left script characters are shown deterministically as [U+XXXX] code points instead of dropping text or presenting incorrect shaping/order. JSON retains the exact original strings.';

async function artifactRecord(asset: VerifiedAsset): Promise<DeliveredArtifact> {
  const record: DeliveredArtifact = { ...asset.artifact, deliveryFilename: asset.filename };
  if (asset.artifact.kind === 'pattern-pdf') {
    if (!Buffer.from(asset.bytes.subarray(0, 5)).equals(Buffer.from('%PDF-'))) throw new Error('Included pattern PDF header is invalid');
    const pdf = await PDFDocument.load(asset.bytes, { updateMetadata: false, throwOnInvalidObject: true });
    if (!pdf.getPageCount() || pdf.getPageCount() > 500) throw new Error('Included pattern PDF has invalid page count');
    record.pages = pdf.getPages().map((page, i) => {
      const { x, y, width, height } = page.getMediaBox();
      if (![x, y, width, height].every(Number.isFinite) || width <= 0 || height <= 0) throw new Error('Included pattern PDF has invalid page dimensions');
      return { page: i + 1, xPt: x, yPt: y, widthPt: width, heightPt: height, widthMm: width * 25.4 / 72, heightMm: height * 25.4 / 72, rotationDegrees: page.getRotation().angle, cropBox: page.getCropBox() };
    });
  }
  if (asset.artifact.kind === 'pattern-svg') {
    const svg = new TextDecoder('utf-8', { fatal: true }).decode(asset.bytes);
    if (!/<svg\b/i.test(svg) || !/<\/svg>\s*$/i.test(svg) || /<!DOCTYPE|<!ENTITY|<\s*(script|foreignObject|iframe|object|embed)\b|\son\w+\s*=|javascript\s*:|@import/i.test(svg)) throw new Error('Included pattern SVG is outside the trusted static subset');
    for (const match of svg.matchAll(/(?:href|xlink:href)\s*=\s*['"]([^'"]*)['"]/gi)) {
      if (!match[1]!.startsWith('#') && !match[1]!.startsWith('data:image/png;base64,')) throw new Error('Included pattern SVG contains external resource references');
    }
    for (const match of svg.matchAll(/url\s*\(\s*['"]?([^)'"\s]+)/gi)) if (!match[1]!.startsWith('#')) throw new Error('Included pattern SVG contains external CSS resources');
  }
  return record;
}

function jsonFile(filename: string, value: unknown): ExportFile {
  return { filename, bytes: new TextEncoder().encode(`${JSON.stringify(value, null, 2)}\n`), mime: 'application/json' };
}

export async function buildExport(snapshot: ExportSnapshot, geometry: PatternGeometry | null, assets: { artifact: Artifact; bytes: Uint8Array }[]): Promise<ExportResult> {
  const captured = captureExport(snapshot, geometry, assets);
  const saved = captured.snapshot, doc = saved.revision.document;
  const pdfContext = await createPdfContext(saved.createdAt);
  const patterns: DeliveredArtifact[] = [];
  for (const asset of captured.patterns) patterns.push(await artifactRecord(asset));
  const images = new Map<string, PDFImage>();
  const referenceArtifacts: DeliveredArtifact[] = [];
  for (const asset of captured.references) {
    checkSanitizedPng(asset.bytes);
    await sharp(asset.bytes, { limitInputPixels: 20_000_000, failOn: 'warning' }).raw().toBuffer();
    images.set(asset.artifact.id, await pdfContext.pdf.embedPng(asset.bytes));
    referenceArtifacts.push(await artifactRecord(asset));
  }
  const omissions: string[] = [];
  if (!saved.disclosure.includeBody) omissions.push('Body input fields and their provenance are omitted.');
  if (!saved.disclosure.includeReferences) omissions.push('Source reference images, sketches, technical-flat images, labels and source captions are omitted.');
  if (!saved.disclosure.includePatterns) omissions.push('Pattern files, panel geometry, engine warnings and engine assembly links are omitted.');
  omissions.push('No simulation, cutting-candidate classification, print calibration, physical-fit approval or independently authenticated maker review is included. AI suggestions, when present, remain unverified.');
  omissions.push('Contact records and raw model exchanges are not part of this export schema. Owner-entered free text is not automatically scrubbed of personal information.');
  const manifest: HandoffManifest = {
    schemaVersion: 1, snapshotId: saved.id, revisionId: saved.revision.id, projectId: saved.revision.projectId,
    revisionNumber: saved.revision.number, parentRevisionId: saved.revision.parentRevisionId, revisionCreatedAt: saved.revision.createdAt,
    snapshotCreatedAt: saved.createdAt, inputDigest: saved.revision.digest, renderer: 'sew-computer-tech-pack/1', manifestDigest: '',
    sections: {
      overview: { title: doc.title, brief: doc.brief, sizeLabel: doc.sizeLabel, garment: structuredClone(doc.garment),...(doc.interpretation?{interpretation:structuredClone(doc.interpretation)}:{}) },
      requirements: structuredClone(doc.requirements), materials: structuredClone(doc.bom.filter(item => item.category === 'fabric' || item.category === 'lining')),
      bom: structuredClone(doc.bom), ...(saved.disclosure.includeBody ? { bodyInputs: structuredClone(doc.body) } : {}),
      finishedMeasurements: structuredClone(doc.poms), construction: structuredClone(doc.construction),
      patternInventory: {
        status: patterns.length ? 'included' : saved.artifacts.some(a => a.kind !== 'reference') ? 'omitted_by_disclosure' : 'not_generated',
        notice: PATTERN_NOTICE,
        flats: Object.fromEntries((['front', 'back', 'detail'] as const).map(view => [view, doc.views.some(reference => reference.role === view && reference.kind === 'technical-flat') ? saved.disclosure.includeReferences ? 'included' : 'withheld' : 'not_supplied'])) as HandoffManifest['sections']['patternInventory']['flats'],
        artifacts: patterns, referenceArtifacts,
        ...(saved.disclosure.includeReferences ? { references: structuredClone(doc.views) } : {}),
        ...(captured.geometry ? {
          panels: captured.geometry.panels.map(({ points: _points, ...panel }) => structuredClone(panel)),
          provenance: { engineVersion: captured.geometry.engineVersion, inputDigest: captured.geometry.inputDigest, units: captured.geometry.units, warnings: structuredClone(captured.geometry.warnings) },
        } : {}),
      },
      review: { callouts: structuredClone(doc.callouts), comments: structuredClone(saved.comments) },
      ...(captured.geometry?.drafting ? {designCoverage:designCoverage(doc,captured.geometry),derivedConstruction:structuredClone(captured.geometry.drafting)} : {}),
      exportDisclosure: {
        ...saved.disclosure, omissions: [...new Set(omissions)], classification: 'Draft review document', privacyNotice: PRIVACY_NOTICE,
        pageFormat: { ...PAPER }, textRendering: { font: 'DejaVu Sans 2.37 / Bitstream Vera license', policy: FONT_POLICY, replacements: [] },
      },
    },
  };
  manifest.sections.exportDisclosure.textRendering.replacements = pdfContext.replacements({ sections: manifest.sections, geometry: captured.geometry });
  const { manifestDigest: _selfDigest, ...unsigned } = manifest;
  manifest.manifestDigest = digest(canonical(unsigned));
  const pdf = await renderPdf(pdfContext, manifest, captured.geometry, images);
  const files: ExportFile[] = [
    { filename: 'tech-pack.pdf', mime: 'application/pdf', bytes: pdf },
    jsonFile('manifest.json', manifest),
    ...[...captured.patterns, ...captured.references].map(asset => ({ filename: asset.filename, mime: asset.artifact.mime, bytes: asset.bytes })),
  ];
  const delivery = {
    schemaVersion: 1, snapshotId: manifest.snapshotId, revisionId: manifest.revisionId, projectId: manifest.projectId,
    manifestDigest: manifest.manifestDigest, algorithm: 'sha256',
    files: files.map(file => ({ filename: file.filename, mime: file.mime, size: file.bytes.length, digest: digest(file.bytes) })),
    checksumPolicy: 'Digests cover exact delivered bytes. delivery.json is excluded from its own checksum list; record its hash externally if needed. manifestDigest hashes canonical manifest JSON with only manifestDigest removed, not the formatted manifest file bytes.',
  };
  files.push(jsonFile('delivery.json', delivery));
  return { manifest, files };
}
