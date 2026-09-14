import { readFile } from 'node:fs/promises';
import fontkit from '@pdf-lib/fontkit';
import { PDFDocument, rgb, type PDFFont, type PDFImage, type PDFPage } from 'pdf-lib';
import { type Measurement, type PatternGeometry } from '../contracts';
const measureLabel = (measurement: Measurement) => 'value' in measurement ? `${measurement.value} ${measurement.unit} (${measurement.state})` : measurement.state.replaceAll('-', ' ');
import type { HandoffManifest } from './index';

export const PAPER = { name: 'A4', widthPt: 210 * 72 / 25.4, heightPt: 297 * 72 / 25.4, widthMm: 210, heightMm: 297, scale: 'Review layout, not an actual-size pattern' };
const INK = rgb(0.13, 0.17, 0.18), MUTED = rgb(0.32, 0.37, 0.39), ACCENT = rgb(0.10, 0.37, 0.34);
const MARGIN = 44, WIDTH = PAPER.widthPt - MARGIN * 2, BOTTOM = 120;
let fontBytes: Promise<Uint8Array> | undefined;

export interface PdfContext { pdf: PDFDocument; font: PDFFont; safeText: (text: string) => string; replacements: (value: unknown) => string[] }

export async function createPdfContext(createdAt: string): Promise<PdfContext> {
  const pdf = await PDFDocument.create();
  pdf.registerFontkit(fontkit);
  fontBytes ??= readFile(new URL('./assets/DejaVuSans.ttf', import.meta.url));
  const font = await pdf.embedFont(await fontBytes, { subset: true });
  const supported = new Set(font.getCharacterSet());
  const needsEscape = (ch: string): boolean => {
    if (ch === '\n' || ch === '\r' || ch === '\t') return false;
    return !supported.has(ch.codePointAt(0)!) || /[\p{Cc}\p{Cf}\p{Cs}\p{Script=Arabic}\p{Script=Hebrew}]/u.test(ch);
  };
  const code = (ch: string) => `U+${ch.codePointAt(0)!.toString(16).toUpperCase().padStart(4, '0')}`;
  const safeText = (text: string) => Array.from(text.normalize('NFC'), ch => needsEscape(ch) ? `[${code(ch)}]` : ch).join('').replace(/\r\n?/g, '\n').replace(/\t/g, '    ');
  const replacements = (value: unknown): string[] => {
    const found = new Set<string>();
    const visit = (part: unknown) => {
      if (typeof part === 'string') for (const ch of part.normalize('NFC')) { if (needsEscape(ch)) found.add(code(ch)); }
      else if (Array.isArray(part)) part.forEach(visit);
      else if (part && typeof part === 'object') Object.values(part).forEach(visit);
    };
    visit(value);
    return [...found].sort();
  };
  pdf.setTitle('Sew Computer - Draft review document');
  pdf.setAuthor('Sew Computer');
  pdf.setCreator('Sew Computer tech-pack renderer v1');
  pdf.setProducer('Sew Computer / pdf-lib');
  pdf.setCreationDate(new Date(createdAt));
  pdf.setModificationDate(new Date(createdAt));
  return { pdf, font, safeText, replacements };
}

class Layout {
  page!: PDFPage;
  y = 0;
  sectionName = '';
  constructor(readonly context: PdfContext, readonly manifest: HandoffManifest) {}

  wrap(text: string, size = 10, width = WIDTH): string[] {
    const { font, safeText } = this.context;
    const lines: string[] = [];
    for (const paragraph of safeText(text).split('\n')) {
      if (!paragraph) { lines.push(''); continue; }
      let line = '';
      for (const word of paragraph.split(/ +/)) {
        if (font.widthOfTextAtSize(line ? `${line} ${word}` : word, size) <= width) { line = line ? `${line} ${word}` : word; continue; }
        if (line) { lines.push(line); line = ''; }
        for (const ch of word) {
          if (font.widthOfTextAtSize(line + ch, size) > width) { lines.push(line); line = ''; }
          line += ch;
        }
      }
      lines.push(line);
    }
    return lines;
  }

  newPage(continuation = false) {
    if (this.context.pdf.getPageCount() >= 400) throw new Error('Tech pack exceeds the 400-page limit');
    this.page = this.context.pdf.addPage([PAPER.widthPt, PAPER.heightPt]);
    this.page.drawText('SEW COMPUTER   /   Draft review document', { x: MARGIN, y: PAPER.heightPt - 45, size: 10, font: this.context.font, color: ACCENT });
    this.y = PAPER.heightPt - 78;
    for (const line of this.wrap(`${this.sectionName}${continuation ? ' (continued)' : ''}`, 19)) {
      this.page.drawText(line, { x: MARGIN, y: this.y, size: 19, font: this.context.font, color: INK });
      this.y -= 24;
    }
    this.y -= 12;
  }

  section(name: string) { this.sectionName = name; this.newPage(); }
  ensure(height: number) { if (this.y - height < BOTTOM) this.newPage(true); }
  paragraph(text: string, size = 10, muted = false) {
    const lines = this.wrap(text, size);
    for (const line of lines) {
      this.ensure(size * 1.5);
      if (line) this.page.drawText(line, { x: MARGIN, y: this.y, size, font: this.context.font, color: muted ? MUTED : INK });
      this.y -= size * 1.5;
    }
    this.y -= 6;
  }
  field(label: string, value: string) { this.paragraph(`${label}: ${value || 'Not supplied'}`); }
  heading(text: string) { this.ensure(54); this.paragraph(text, 12); }
  image(image: PDFImage) {
    const scale = Math.min(WIDTH / image.width, 240 / image.height, 1);
    const width = image.width * scale, height = image.height * scale;
    this.ensure(height + 20);
    this.page.drawImage(image, { x: MARGIN, y: this.y - height, width, height });
    this.y -= height + 20;
  }
  panel(panel: PatternGeometry['panels'][number]) {
    this.ensure(195);
    this.heading(`Panel ${panel.id}`);
    const points = panel.points;
    let minX = Infinity, minY = Infinity;
    for (const [x, y] of points) { minX = Math.min(minX, x); minY = Math.min(minY, y); }
    const scale = Math.min(180 / panel.widthMm, 105 / panel.heightMm);
    const stride = Math.max(1, Math.ceil(points.length / 1500));
    const outline = points.filter((_, i) => i % stride === 0);
    const start = { x: MARGIN + 4, y: this.y - 110 };
    for (let i = 0; i < outline.length; i++) {
      const a = outline[i]!, b = outline[(i + 1) % outline.length]!;
      this.page.drawLine({ start: { x: start.x + (a[0] - minX) * scale, y: start.y + (a[1] - minY) * scale }, end: { x: start.x + (b[0] - minX) * scale, y: start.y + (b[1] - minY) * scale }, thickness: 0.7, color: ACCENT });
    }
    this.y -= 125;
    this.paragraph(`Panel bounding box: ${panel.widthMm} x ${panel.heightMm} mm. Outline thumbnail, not to scale. These are panel dimensions, not body measurements or finished-garment POMs.`, 9, true);
    this.field('Reported annotations', 'Allowances, grainlines, notches and cut counts are not validated.');
  }
  footers() {
    const { manifest, context } = this;
    const pages = context.pdf.getPages();
    pages.forEach((page, index) => {
      page.drawLine({ start: { x: MARGIN, y: 109 }, end: { x: PAPER.widthPt - MARGIN, y: 109 }, thickness: 0.5, color: MUTED });
      const entries = [
        `Draft review document | Revision ${manifest.revisionNumber} | ${manifest.revisionId}`,
        `Snapshot ${manifest.snapshotId}`,
        `Manifest SHA-256 ${manifest.manifestDigest.slice(0, 16)} | Input SHA-256 ${manifest.inputDigest.slice(0, 16)}`,
        `Page ${index + 1} of ${pages.length} | A4 210 x 297 mm | Not cutting ready`,
      ];
      let y = 94;
      for (const entry of entries) for (const line of this.wrap(entry, 7.5)) {
        page.drawText(line, { x: MARGIN, y, font: context.font, size: 7.5, color: MUTED }); y -= 10;
      }
      if (y < 20) throw new Error('Page identity exceeds the reserved footer space');
    });
  }
}

const measurement = (value: Measurement) => `${measureLabel(value)}; source: ${'source' in value ? value.source : 'Not supplied'}`;

export async function renderPdf(context: PdfContext, manifest: HandoffManifest, geometry: PatternGeometry | null, images: Map<string, PDFImage>): Promise<Uint8Array> {
  const out = new Layout(context, manifest), s = manifest.sections;
  out.section('1. Design overview');
  out.heading(s.overview.title || 'Untitled garment');
  out.field('Original brief', s.overview.brief);
  out.field('Saved revision', `${manifest.revisionNumber} (${manifest.revisionId}); created ${manifest.revisionCreatedAt}`);
  out.field('Size scope', s.overview.sizeLabel || 'Size not supplied. No graded range is established.');
  out.field('Geometry starting family', s.overview.garment.family);
  out.field('Garment length parameter', measurement(s.overview.garment.length));
  out.field('Ease parameter', measurement(s.overview.garment.ease));
  out.field('Flare ratio', String(s.overview.garment.flare));
  out.field('Geometry scope', 'Limited manual CPU adapter; no physical-fit approval or full design realization claim.');
  const unresolved = s.requirements.filter(r => r.status !== 'supported').length;
  out.heading('Completeness and capability');
  out.paragraph(`${unresolved} unsupported or unresolved requirements; ${s.finishedMeasurements.filter(p => p.target.state === 'unknown').length} unknown POM targets; ${s.construction.length} owner-entered construction notes. Pattern disclosure: ${s.patternInventory.status}.`);
  out.paragraph('Original unsupported intent is retained. Supported means representable by the adapter, not physically validated. No AI interpretation, simulation, calibrated cutting candidate or physical-fit approval is supplied.', 9, true);
  for (const requirement of s.requirements) {
    out.heading(`${requirement.id} / ${requirement.status}`);
    out.paragraph(requirement.text);
    out.field('Note', requirement.note); out.field('Source', 'Owner-entered requirement and coverage assessment');
  }
  if (!s.requirements.length) out.paragraph('Requirements not supplied.');
  if (s.bodyInputs) {
    out.heading('Disclosed body inputs - distinct from garment POMs');
    for (const [name, value] of Object.entries(s.bodyInputs)) out.field(name, measurement(value));
  } else out.paragraph('Body inputs omitted by disclosure policy.', 9, true);
  out.paragraph(s.exportDisclosure.privacyNotice, 9, true);

  out.section('2. Garment flats');
  out.paragraph('Garment flats describe the assembled garment. Pattern panels are not garment flats. Reference photographs and sketches are not simulation or physical-fit evidence.', 9, true);
  const references = s.patternInventory.references ?? [];
  for (const view of ['front', 'back', 'detail'] as const) {
    const flats = references.filter(r => r.kind === 'technical-flat' && r.role === view);
    out.heading(`${view[0]!.toUpperCase()}${view.slice(1)} view`);
    if (s.patternInventory.flats[view] === 'withheld') out.paragraph('Not included: reference images and their labels are withheld by disclosure policy.');
    else if (!flats.length) out.paragraph(`${view[0]!.toUpperCase()}${view.slice(1)} view not supplied as a technical flat.`);
    for (const ref of flats) {
      out.field(`Technical flat ${ref.id}`, ref.caption);
      out.field('Owner-entered source', ref.caption);
      const image = images.get(ref.assetId);
      if (!image) throw new Error('Disclosed technical flat has no verified image');
      out.image(image);
    }
  }
  for (const ref of references.filter(r => r.kind !== 'technical-flat')) {
    out.field(`${ref.kind} ${ref.id} (${ref.role})`, ref.caption);
    out.field('Owner-entered source', ref.caption);
    out.paragraph(`Separate sanitized image: ${s.patternInventory.referenceArtifacts.find(a => a.id === ref.assetId)!.deliveryFilename}. This is not a supplied front/back technical flat.`, 9, true);
  }
  out.heading('Stable callouts');
  for (const callout of s.review.callouts) out.field(`${callout.id} -> ${callout.anchor || 'unassigned'}`, callout.text);
  if (!s.review.callouts.length) out.paragraph('No callouts supplied.');

  out.section('3. Materials / BOM');
  out.paragraph('Material properties and quantities below are owner-entered. Missing consumption, substitutions and material behavior remain unresolved; panel area does not establish purchasing yardage.', 9, true);
  for (const item of s.bom) {
    out.heading(`${item.id} / ${item.name || 'Unnamed component'} (${item.category})`);
    out.field('Specification', item.specification); out.field('Placement', item.placement);
    out.field('Quantity / basis', item.quantity); out.field('Source', item.source);
  }
  if (!s.bom.length) out.paragraph('Materials and bill of materials not supplied.');

  out.section('4. Finished-garment measurements');
  out.paragraph('These are defined garment POMs, not body dimensions or panel bounding boxes. Unknown and not-applicable states are retained; unspecified values are not zero. A size label is not a graded range.', 9, true);
  for (const pom of s.finishedMeasurements) {
    out.heading(`${pom.id} / ${pom.name || 'Unnamed POM'}`);
    out.field('Method / condition', pom.method); out.field('Size scope', pom.size);
    out.field('Target', measurement(pom.target)); out.field('Tolerance', measurement(pom.tolerance)); out.field('Measurement note', pom.note);
  }
  if (!s.finishedMeasurements.length) out.paragraph('No finished-garment POMs supplied. Geometry is not used to invent them.');
  out.paragraph('Grading rules: not established by this prototype.', 9, true);

  out.section('5. Construction');
  for (const operation of s.construction) {
    out.heading(`${operation.id} / ${operation.operation || 'Unnamed operation'} (${operation.note})`);
    out.paragraph(operation.note || 'Instruction not supplied.');
  }
  if (!s.construction.length) out.paragraph('Construction instructions not supplied.');
  if (geometry) {
    out.heading('Reported engine assembly links - not seam validation');
    for (const stitch of geometry.stitches) out.field(`${stitch.panelA} -> ${stitch.panelB}`, `Edge ${stitch.edgeA} to edge ${stitch.edgeB}; assembly not physically validated`);
    if (!geometry.stitches.length) out.paragraph('Assembly links not supplied.');
  } else out.paragraph('Engine assembly links are absent or not disclosed.');
  out.paragraph('Seam compatibility, allowance policy, fabric behavior, construction feasibility and pressing/closure details are not inferred from outlines. Unspecified treatments remain open questions.', 9, true);

  out.section('6. Pattern inventory');
  out.paragraph(s.patternInventory.notice);
  out.field('Status', s.patternInventory.status);
  for (const artifact of s.patternInventory.artifacts) {
    out.heading(`${artifact.id} / ${artifact.kind}`);
    out.field('Delivery file', artifact.deliveryFilename); out.field('Original filename', artifact.filename);
    out.field('Classification', artifact.classification);
    out.field('SHA-256', artifact.digest); out.field('Byte size', String(artifact.bytes));
    for (const page of artifact.pages ?? []) out.paragraph(`Original PDF page ${page.page}: media box ${page.widthPt.toFixed(3)} x ${page.heightPt.toFixed(3)} pt (${page.widthMm.toFixed(3)} x ${page.heightMm.toFixed(3)} mm); rotation ${page.rotationDegrees} degrees. These are file dimensions, not a calibration result.`, 9, true);
  }
  if (geometry) {
    out.field('Generator', `${geometry.engineVersion}; source ${geometry.units}`);
    out.field('Geometry input SHA-256', geometry.inputDigest);
    out.field('Canonical coordinate unit', geometry.units);
    for (const warning of geometry.warnings) out.field('Engine warning', warning);
    for (const assumption of geometry.assumptions) out.field('Engine assumption', assumption);
    for (const panel of geometry.panels) out.panel(panel);
  } else out.paragraph('No panel geometry is displayed. An absent or withheld pattern is not replaced by an illustration.');
  out.paragraph('Annotation checks not performed: seam allowances, grain, notches, cut counts/folds, material assignments and grading. Reported annotation strings do not establish completeness. No print calibration or physical scale check is recorded.', 9, true);

  out.section('7. Review and change record');
  out.field('Revision', `${manifest.revisionNumber}; ${manifest.revisionId}`);
  out.field('Parent revision', manifest.parentRevisionId ?? 'None (first saved revision)');
  out.field('Snapshot cutoff', manifest.snapshotCreatedAt);
  out.paragraph('This is the selected immutable revision. No current editor buffer or later job is read. A parent revision ID is a history link; a computed change diff is not supplied.');
  for (const comment of s.review.comments) {
    out.heading(`Comment ${comment.id} -> ${comment.anchor || 'unassigned'}`);
    out.paragraph(comment.text);
    out.field('Reported reviewer', comment.reportedReviewer || 'Not supplied');
    out.field('Recorder', 'owner (reported external identity not authenticated)');
    out.field('Recorded at', comment.createdAt);
  }
  if (!s.review.comments.length) out.paragraph('No revision-scoped review comments supplied.');
  out.heading('Open callouts');
  for (const callout of s.review.callouts) out.field(`${callout.id} -> ${callout.anchor || 'unassigned'}`, callout.text);
  if (!s.review.callouts.length) out.paragraph('No callouts supplied.');
  out.heading('Disclosure and limitations');
  for (const omission of s.exportDisclosure.omissions) out.paragraph(omission, 9, true);
  out.paragraph(s.exportDisclosure.privacyNotice, 9, true);
  out.paragraph('Editable manifest changes are proposals only. They cannot rewrite this saved revision, artifact bytes, service evidence or reviewer identity. Retain unknowns and unsupported intent when returning feedback.', 9, true);
  out.paragraph(s.exportDisclosure.textRendering.policy, 9, true);
  if (s.exportDisclosure.textRendering.replacements.length) out.paragraph(`Escaped PDF code points: ${s.exportDisclosure.textRendering.replacements.join(', ')}. Full source Unicode is retained in the editable JSON manifest.`, 9, true);
  out.heading('Font license');
  out.paragraph('PDF font: DejaVu Sans (Bitstream Vera license; DejaVu changes public domain).', 9, true);
  out.paragraph(await readFile(new URL('./assets/FONT-LICENSE.md', import.meta.url), 'utf8'), 8, true);
  out.footers();
  return context.pdf.save({ useObjectStreams: false, addDefaultPage: false });
}
