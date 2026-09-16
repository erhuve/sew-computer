import { describe, expect, test } from 'bun:test';
import { createHash } from 'node:crypto';
import { chmod, chown, mkdtemp, readFile, readdir, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { PDFDocument } from 'pdf-lib';
import { assumed, canonical, emptyDocument, type GarmentDocument } from '../../packages/contracts/index';
import { cleanEnvironment, ENGINE_COMMIT, executeTrusted, runEngine, validateInput } from './runner';
import { applySample, sampleSizes } from '../../packages/contracts/sizing';

const root = import.meta.dir;
const sha = (value: unknown) => createHash('sha256').update(canonical(value)).digest('hex');
const families = ['shirt', 'skirt', 'trousers'] as const;
const fixture = (family: typeof families[number]): GarmentDocument => {
  const document = emptyDocument('Original synthetic CPU fixture', 'Preserve the unsupported asymmetric closure.');
  document.body = { height: assumed(1700), bust: assumed(960), waist: assumed(760), hip: assumed(1000), shoulder: assumed(400) };
  document.garment = { family, length: assumed(family === 'shirt' ? 650 : family === 'skirt' ? 600 : 950), ease: assumed(80), flare: 1 };
  document.requirements = [{ id: 'asymmetric-closure', text: 'An unsupported asymmetric closure', status: 'unsupported', note: 'Do not replace intent' }];
  return document;
};
async function generate(document: GarmentDocument) {
  const outputDir = await mkdtemp(join(root, '.test-'));
  await chmod(outputDir, 0o700);
  const before = canonical(document);
  try {
    const result = await runEngine({ document, inputDigest: sha(document), outputDir, signal: new AbortController().signal });
    expect(canonical(document)).toBe(before);
    expect(await readdir(outputDir)).toEqual([]);
    return result;
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
}

for (const family of families) {
  describe(`real CPU ${family}`, () => {
    test('generates genuine closed curves, connected seams, identity-bound JSON/SVG and measured PDF pages', async () => {
      const document = fixture(family);
      const { geometry, files } = await generate(document);
      expect(geometry.schemaVersion).toBe(1);
      expect(geometry.units).toBe('mm');
      expect(geometry.inputDigest).toBe(sha(document));
      expect(geometry.engineVersion).toContain(ENGINE_COMMIT);
      expect(geometry.classification).toBe('printable-reference');
      expect(geometry.panels.length).toBe(family === 'skirt' ? 2 : 4);
      expect(geometry.stitches.length).toBe(family === 'shirt' ? 6 : family === 'skirt' ? 2 : 16);
      const names = new Set(geometry.panels.map(p => p.id));
      for (const panel of geometry.panels) {
        expect(panel.points.length).toBeGreaterThan(30);
        expect(panel.points[0]).toEqual(panel.points.at(-1)!);
        expect(panel.widthMm).toBeGreaterThan(200);
        expect(panel.heightMm).toBeGreaterThan(500);
        expect(panel.points.every(p => p.every(Number.isFinite))).toBe(true);
      }
      for (const stitch of geometry.stitches) {
        expect(names.has(stitch.panelA)).toBe(true);
        expect(names.has(stitch.panelB)).toBe(true);
        expect(Number.isInteger(stitch.edgeA) && stitch.edgeA >= 0).toBe(true);
        expect(Number.isInteger(stitch.edgeB) && stitch.edgeB >= 0).toBe(true);
      }
      expect(geometry.warnings.join('\n')).toContain('unsupported asymmetric closure');
      expect(geometry.warnings.join('\n')).toContain('non-root uid');
      expect(geometry.warnings.join('\n')).toContain('not public/multiuser');
      expect(geometry.assumptions.join('\n')).toContain('synthetic');
      expect(files.map(f => f.filename)).toEqual(['pattern.json', 'pattern.svg', 'pattern.pdf', 'pattern-a4-tiled.pdf', 'pattern-letter-tiled.pdf']);
      for (const file of files) expect(file.bytes.length).toBeGreaterThan(1000);
      expect(JSON.parse(new TextDecoder().decode(files[0]!.bytes))).toEqual(geometry);
      const svg = new TextDecoder().decode(files[1]!.bytes);
      expect(svg).toContain('mm"');
      expect(svg).toContain('NOT CUTTING READY');
      expect(svg).not.toMatch(/<script|foreignObject|onload|href|<image/);
      const pdf = await PDFDocument.load(files[2]!.bytes);
      expect(pdf.getPageCount()).toBe(geometry.panels.length);
      pdf.getPages().forEach((page, index) => {
        expect(page.getWidth() * 25.4 / 72).toBeCloseTo(geometry.panels[index]!.widthMm + 20, 4);
        expect(page.getHeight() * 25.4 / 72).toBeCloseTo(geometry.panels[index]!.heightMm + 40, 4);
      });
      if (family === 'skirt') {
        const circumference = 760 + 80;
        const radius = circumference / Math.PI + 600;
        expect(geometry.panels[0]!.widthMm).toBeCloseTo(Math.sqrt(2) * radius, 4);
      }
      if (family === 'trousers') expect(geometry.panels.find(p => p.id === 'pant_f_l')!.heightMm).toBeCloseTo(950, 4);
    }, 90_000);

    test('length, ease and flare each change actual generated geometry', async () => {
      const base = fixture(family);
      const baseline = (await generate(base)).geometry.panels;
      for (const parameter of ['length', 'ease', 'flare'] as const) {
        const changed = structuredClone(base);
        if (parameter === 'flare') changed.garment.flare = family === 'trousers' ? 1.1 : 1.2;
        else changed.garment[parameter] = assumed(parameter === 'length' ? ('value' in base.garment.length ? base.garment.length.value + 100 : 0) : 140);
        const actual = (await generate(changed)).geometry.panels;
        expect(sha(actual)).not.toBe(sha(baseline));
        expect(actual.map(p => p.id)).toEqual(baseline.map(p => p.id));
        if (parameter === 'length') expect(Math.max(...actual.map(p => p.heightMm))).toBeGreaterThan(Math.max(...baseline.map(p => p.heightMm)));
      }
    }, 180_000);
  });
}

test('all six synthetic starting sizes generate real patterns for all three families', async () => {
  for (const family of families) for (let index = 0; index < sampleSizes.length; index++) {
    const doc = emptyDocument(`Sample ${sampleSizes[index]!.label} ${family}`);
    doc.garment.family = family;
    const result = await generate(applySample(doc, index));
    expect(result.geometry.panels.length).toBeGreaterThan(0);
  }
}, 240_000);

test('broad shoulder engine failure reports a safe actionable limitation and preserves values', async () => {
  const doc = fixture('shirt');
  doc.body.shoulder = assumed(584.2);
  const before = canonical(doc);
  await expect(generate(doc)).rejects.toThrow('zero-length edge');
  expect(canonical(doc)).toBe(before);
}, 30_000);

test('equivalent mm/cm/in inputs have identical physical geometry', async () => {
  const baseline = fixture('skirt');
  const expected = (await generate(baseline)).geometry.panels;
  for (const unit of ['cm', 'in'] as const) {
    const doc = structuredClone(baseline);
    const scale = unit === 'cm' ? 10 : 25.4;
    for (const name of Object.keys(doc.body) as (keyof typeof doc.body)[]) {
      const m = doc.body[name];
      if ('value' in m) doc.body[name] = { ...m, unit, value: m.value / scale };
    }
    for (const name of ['length', 'ease'] as const) {
      const m = doc.garment[name];
      if ('value' in m) doc.garment[name] = { ...m, unit, value: m.value / scale };
    }
    const result = await generate(doc);
    expect(result.geometry.panels).toEqual(expected);
  }
}, 120_000);

test('unknown measurements, unsupported families/ranges and digest mismatch fail without substitution', () => {
  const doc = fixture('shirt');
  expect(() => validateInput(doc, '0'.repeat(64))).toThrow('digest');
  const absent = emptyDocument();
  expect(() => validateInput(absent, sha(absent))).toThrow('No garment family');
  for (const state of ['unknown', 'not-applicable'] as const) {
    const unknown = structuredClone(doc);
    unknown.body.bust = { state };
    expect(() => validateInput(unknown, sha(unknown))).toThrow('known or assumed');
  }
  const unsupported = structuredClone(doc);
  unsupported.garment.flare = 3;
  expect(() => validateInput(unsupported, sha(unsupported))).toThrow('Unsupported shirt flare');
  const invalid = structuredClone(doc);
  invalid.body.height = assumed(100);
  expect(() => validateInput(invalid, sha(invalid))).toThrow('Unsupported body height');
  const badUnit = structuredClone(doc) as unknown as Record<string, any>;
  badUnit.body.height.unit = 'meters';
  expect(() => validateInput(badUnit as GarmentDocument, sha(badUnit))).toThrow();
});

test('environment is allowlisted and does not inherit credentials', () => {
  process.env.SEW_ENGINE_TEST_SECRET = 'do-not-forward';
  try {
    const env = cleanEnvironment();
    expect(env.SEW_ENGINE_TEST_SECRET).toBeUndefined();
    expect(Object.keys(env).some(k => /TOKEN|KEY|SECRET|PROXY|PYTHONPATH/.test(k))).toBe(false);
    expect(env.OPENBLAS_NUM_THREADS).toBe('1');
    expect(env.MPLBACKEND).toBe('Agg');
  } finally { delete process.env.SEW_ENGINE_TEST_SECRET; }
});

test('actual worker is non-root with measured resource limits and truthful network denial', async () => {
  const directory = await mkdtemp(join(root, '.test-probe-'));
  await chmod(directory, 0o700);
  if (process.getuid?.() === 0) await chown(directory, 65534, 65534);
  try {
    const code = `import sys,json;sys.path.insert(0,${JSON.stringify(root)});from guard import constrain;s=constrain();open('probe.json','w').write(json.dumps(s))`;
    await executeTrusted(join(root, '.venv/bin/python'), ['-c', code], directory, '', new AbortController().signal);
    const probe = JSON.parse(await readFile(join(directory, 'probe.json'), 'utf8'));
    expect(probe.uid).not.toBe(0);
    expect(probe.groups).not.toContain(0);
    expect(probe.limits).toEqual({ cpuSeconds: 60, addressSpaceBytes: 2 * 1024 ** 3, fileBytes: 16 * 1024 ** 2, descriptors: 64, processes: 64 });
    expect(probe.network).toMatch(/probes returned EPERM|not isolated/);
  } finally { await rm(directory, { recursive: true, force: true }); }
}, 20_000);

test('wall timeout kills the process group and a pre-aborted request launches nothing', async () => {
  const directory = await mkdtemp(join(root, '.test-timeout-'));
  try {
    const code = "import os,time;pid=os.fork();open('child.pid','w').write(str(pid)) if pid else None;time.sleep(30)";
    await expect(executeTrusted(join(root, '.venv/bin/python'), ['-c', code], directory, '', new AbortController().signal, 500)).rejects.toThrow('wall timeout');
    const childPid = Number(await readFile(join(directory, 'child.pid'), 'utf8'));
    let living = false;
    try {
      const stat = await readFile(`/proc/${childPid}/stat`, 'utf8');
      living = stat.split(') ')[1]?.[0] !== 'Z';
    } catch {}
    expect(living).toBe(false);
    const controller = new AbortController();
    controller.abort();
    const doc = fixture('shirt');
    await expect(runEngine({ document: doc, inputDigest: sha(doc), outputDir: join(directory, 'uncreated'), signal: controller.signal })).rejects.toThrow();
    expect(await readdir(directory)).not.toContain('uncreated');
  } finally { await rm(directory, { recursive: true, force: true }); }
}, 20_000);

test('cancellation removes the private staging attempt', async () => {
  const outputDir = await mkdtemp(join(root, '.test-cancel-'));
  const controller = new AbortController();
  const doc = fixture('trousers');
  const timer = setTimeout(() => controller.abort(), 100);
  try {
    await expect(runEngine({ document: doc, inputDigest: sha(doc), outputDir, signal: controller.signal })).rejects.toThrow();
    expect(await readdir(outputDir)).toEqual([]);
  } finally {
    clearTimeout(timer);
    await rm(outputDir, { recursive: true, force: true });
  }
}, 20_000);
