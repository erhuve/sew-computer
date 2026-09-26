import { expect, test } from 'bun:test';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { Duplex, PDFDocument, PrintScaling } from 'pdf-lib';
import sharp from 'sharp';
import type { PatternGeometry } from '../../packages/contracts';
import { panelTiles, PRINT_PAPERS, renderTiledPattern, type PrintPaper } from './printing';

function fixture(): PatternGeometry {
  return {
    schemaVersion: 1, units: 'mm', family: 'shirt', inputDigest: 'a'.repeat(64), engineVersion: 'print-test', classification: 'printable-reference', warnings: [], assumptions: [], stitches: [],
    panels: [{ id: 'front_left', name: 'Test front', widthMm: 400, heightMm: 600, cutQuantity: 1, points: [[0, 0], [400, 0], [400, 600], [0, 600], [0, 0]],
      draft: { component: 'body', material: 'shell', cutQuantity: 1, edges: [], marks: [{kind:'button',label:'front closure',point:[200,100]}], grainline: [[200, 200], [200, 400]], cutLine: [[-10, -10], [410, -10], [410, 610], [-10, 610], [-10, -10]] },
    }],
  };
}

for (const paper of Object.keys(PRINT_PAPERS) as PrintPaper[]) test(`${paper} tiles preserve full allowance bounds, physical page size and overlap`, async () => {
  const geometry = fixture(), panel = geometry.panels[0]!;
  const grid = panelTiles(panel, paper);
  expect(grid.minX).toBe(-15);
  expect(grid.minY).toBe(-15);
  expect(grid.columns).toBe(3);
  expect(grid.rows).toBe(3);
  expect(grid.columns * grid.stepX + 10).toBeGreaterThanOrEqual(grid.width);
  expect(grid.rows * grid.stepY + 10).toBeGreaterThanOrEqual(grid.height);
  const bytes = await renderTiledPattern(geometry, paper, new AbortController().signal);
  const pdf = await PDFDocument.load(bytes);
  expect(pdf.getPageCount()).toBe(10);
  expect(pdf.catalog.getOrCreateViewerPreferences().getPrintScaling()).toBe(PrintScaling.None);
  expect(pdf.catalog.getOrCreateViewerPreferences().getDuplex()).toBe(Duplex.Simplex);
  for (const page of pdf.getPages()) {
    expect(page.getWidth() * 25.4 / 72).toBeCloseTo(PRINT_PAPERS[paper][0], 6);
    expect(page.getHeight() * 25.4 / 72).toBeCloseTo(PRINT_PAPERS[paper][1], 6);
    expect(page.getRotation().angle).toBe(0);
  }
  const directory = await mkdtemp(join(import.meta.dir, '.test-print-'));
  try {
    const filename = join(directory, 'pattern.pdf');
    await writeFile(filename, bytes);
    const extract = Bun.spawn(['pdftotext', '-layout', filename, '-'], { stdout: 'pipe', stderr: 'pipe' });
    const content = await new Response(extract.stdout).text();
    expect(await extract.exited).toBe(0);
    expect(content).toContain('front_left / 2-10 / 3 x 3 / cut 1 shell');
    expect(content).toContain('row 3/3, column 3/3');
    expect(content).toContain('100 x 100 mm');
    expect(content.match(/Input SHA-256:/g)).toHaveLength(10);
    const raster = async (page: number) => {
      const process = Bun.spawn(['pdftoppm', '-f', String(page), '-singlefile', '-r', '127', '-png', filename], {stdout:'pipe',stderr:'pipe'});
      const png = new Uint8Array(await new Response(process.stdout).arrayBuffer());
      expect(await process.exited).toBe(0);
      return png;
    };
    const cover = await sharp(await raster(1)).greyscale().raw().toBuffer({resolveWithObject:true});
    const darkNear = (positionX: number, positionY: number) => {
      let darkest = 255;
      for (let offset = -1; offset <= 1; offset++) darkest = Math.min(darkest, cover.data[Math.round(positionY * 5) * cover.info.width + Math.round(positionX * 5) + offset]!);
      return darkest;
    };
    expect(darkNear(12, 100)).toBeLessThan(100);
    expect(darkNear(112, 100)).toBeLessThan(100);
    expect(darkNear(50, 78)).toBeLessThan(150);
    expect(darkNear(50, 178)).toBeLessThan(150);
    const first = await raster(2), second = await raster(3);
    const crop = async (png: Uint8Array, left: number) => sharp(png).extract({left:Math.round(left * 5),top:125,width:35,height:400}).greyscale().raw().toBuffer();
    const overlapLeft = await crop(first, 12 + grid.stepX + 1);
    const overlapRight = await crop(second, 13);
    let error = 0;
    for (let index = 0; index < overlapLeft.length; index++) error += Math.abs(overlapLeft[index]! - overlapRight[index]!);
    expect(error / overlapLeft.length).toBeLessThan(2);
    expect(Math.min(...overlapLeft)).toBeLessThan(100);
    const below = await raster(5);
    const verticalCrop = async (png: Uint8Array, top: number) => sharp(png).extract({left:70,top:Math.round(top * 5),width:200,height:35}).greyscale().raw().toBuffer();
    const overlapTop = await verticalCrop(first, 22 + grid.stepY + 1);
    const overlapBottom = await verticalCrop(below, 23);
    let verticalError = 0;
    for (let index = 0; index < overlapTop.length; index++) verticalError += Math.abs(overlapTop[index]! - overlapBottom[index]!);
    expect(verticalError / overlapTop.length).toBeLessThan(2);
    expect(Math.min(...overlapTop)).toBeLessThan(100);
  } finally { await rm(directory, {recursive:true,force:true}); }
},30000);

test('print planning rejects oversized output and cancellation', async () => {
  const geometry = fixture();
  const oversized = structuredClone(geometry);
  oversized.panels[0]!.draft!.cutLine = [[0,0],[6000,0],[6000,6000],[0,6000],[0,0]];
  await expect(renderTiledPattern(oversized, 'A4', new AbortController().signal)).rejects.toThrow('page limit');
  const complex = structuredClone(geometry);
  complex.panels[0]!.points = Array.from({length:250000}, (_,index)=>[index % 2 ? 0 : 400, index % 2 ? 0 : 600]);
  await expect(renderTiledPattern(complex, 'A4', new AbortController().signal)).rejects.toThrow('geometry budget');
  const controller = new AbortController();
  controller.abort();
  await expect(renderTiledPattern(geometry, 'A4', controller.signal)).rejects.toThrow();
  const during = new AbortController();
  const rendering = renderTiledPattern(geometry, 'Letter', during.signal);
  setTimeout(() => during.abort(), 1);
  await expect(rendering).rejects.toThrow();
});
