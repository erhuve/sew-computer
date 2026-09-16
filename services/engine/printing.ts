import { clip, Duplex, endPath, popGraphicsState, PrintScaling, pushGraphicsState, rectangle, rgb, type PDFPage } from 'pdf-lib';
import type { PatternGeometry } from '../../packages/contracts';
import { createPdfContext } from '../../packages/tech-pack/pdf';

export const PRINT_PAPERS = { A4: [210, 297], Letter: [215.9, 279.4] } as const;
export const PRINT_MARGIN = 12, PRINT_VERTICAL_MARGIN = 22, PRINT_OVERLAP = 10;
const POINTS_PER_MM = 72 / 25.4, MAX_PAGES = 400;
type Point = [number, number];
type Panel = PatternGeometry['panels'][number];
export type PrintPaper = keyof typeof PRINT_PAPERS;

export function panelTiles(panel: Panel, paper: PrintPaper) {
  const points = panel.draft?.cutLine ?? panel.points;
  const minX = Math.min(...points.map(point => point[0])) - 5;
  const minY = Math.min(...points.map(point => point[1])) - 5;
  const width = Math.max(...points.map(point => point[0])) + 5 - minX;
  const height = Math.max(...points.map(point => point[1])) + 5 - minY;
  const tileWidth = PRINT_PAPERS[paper][0] - PRINT_MARGIN * 2;
  const tileHeight = PRINT_PAPERS[paper][1] - PRINT_VERTICAL_MARGIN * 2;
  const stepX = tileWidth - PRINT_OVERLAP, stepY = tileHeight - PRINT_OVERLAP;
  if (![minX, minY, width, height].every(Number.isFinite) || width <= 0 || height <= 0) throw new Error('Invalid print bounds');
  const columns = Math.max(1, Math.ceil((width - PRINT_OVERLAP) / stepX));
  const rows = Math.max(1, Math.ceil((height - PRINT_OVERLAP) / stepY));
  if (rows * columns > MAX_PAGES) throw new Error('Pattern exceeds the tiled print page limit');
  return { minX, minY, width, height, tileWidth, tileHeight, stepX, stepY, columns, rows };
}

export async function renderTiledPattern(geometry: PatternGeometry, paper: PrintPaper, signal: AbortSignal): Promise<Uint8Array> {
  signal.throwIfAborted();
  const plans = geometry.panels.map(panel => ({ panel, grid: panelTiles(panel, paper) }));
  const guidePages = Math.ceil(plans.length / 16);
  const count = guidePages + plans.reduce((sum, plan) => sum + plan.grid.rows * plan.grid.columns, 0);
  if (count > MAX_PAGES) throw new Error('Pattern exceeds the tiled print page limit');
  const vertices = plans.reduce((sum, {panel, grid}) => sum + (panel.points.length + (panel.draft?.cutLine.length ?? 0)) * grid.rows * grid.columns, 0);
  if (vertices > 2_000_000) throw new Error('Pattern exceeds the tiled print geometry budget');
  const { pdf, font, safeText } = await createPdfContext(new Date().toISOString());
  pdf.setTitle(`Sew Computer - ${paper} tiled pattern reference`);
  pdf.setCreator('Sew Computer tiled pattern renderer 1');
  const preferences = pdf.catalog.getOrCreateViewerPreferences();
  preferences.setPrintScaling(PrintScaling.None);
  preferences.setDuplex(Duplex.Simplex);
  const [paperWidth, paperHeight] = PRINT_PAPERS[paper];
  const ink = rgb(0.1, 0.1, 0.1), guide = rgb(0.55, 0.55, 0.55);
  const text = (page: PDFPage, value: string, positionX: number, positionY: number, size = 8) => {
    page.drawText(safeText(value), { x: positionX * POINTS_PER_MM, y: (paperHeight - positionY) * POINTS_PER_MM, size, font, color: ink });
  };
  const line = (page: PDFPage, start: Point, end: Point, dashed = false, muted = false) => {
    page.drawLine({ start: { x: start[0] * POINTS_PER_MM, y: (paperHeight - start[1]) * POINTS_PER_MM }, end: { x: end[0] * POINTS_PER_MM, y: (paperHeight - end[1]) * POINTS_PER_MM }, thickness: muted ? 0.35 : 0.65, color: muted ? guide : ink, ...(dashed ? { dashArray: [3, 2] } : {}) });
  };
  const newPage = () => {
    signal.throwIfAborted();
    const page = pdf.addPage([paperWidth * POINTS_PER_MM, paperHeight * POINTS_PER_MM]);
    text(page, `${paper} | Print Actual size / 100% | Single-sided | NOT CUTTING READY`, 12, 10, 8);
    text(page, `Input SHA-256: ${geometry.inputDigest}`, 12, paperHeight - 13, 6);
    text(page, `Page ${pdf.getPageCount()} / ${count} | Nominal 1:1 geometry; printer scale and fit unverified`, 12, paperHeight - 8, 7);
    return page;
  };
  let tilePage = guidePages + 1;
  for (let section = 0; section < guidePages; section++) {
    const page = newPage();
    text(page, `Print guide ${section + 1}/${guidePages} - ${paperWidth} x ${paperHeight} mm paper`, 12, 20, 11);
    const instructions = [
      'Print this guide first. Disable Fit, Shrink, and printer scaling. Measure both square edges.',
      'If either edge is not 100 mm, correct printing settings before printing pattern sheets.',
      'Use only this paper format. Tile each piece separately in row order, left to right.',
      'Trim the top/left frame borders on adjoining sheets; overlap the repeated 10 mm strip.',
      'Match dashed guides, crosses and repeated pattern lines, then tape without stretching.',
      'Solid: cut contour. Dashed: seam line. Arrow: grain. +: notch or button placement.',
      'Small circle: buttonhole center, NOT slit size. Transfer marks before cutting a toile.',
      'Legacy pieces have seam outlines only: do not cut without adding required allowances.',
      'Named left/right torso and sleeve pieces form mirrored pairs; cut opposite fabric faces.',
    ];
    instructions.forEach((instruction, index) => text(page, instruction, 12, 28 + index * 5, 7.5));
    text(page, '100 x 100 mm calibration square', 12, 73, 8);
    line(page, [12, 78], [112, 78]); line(page, [112, 78], [112, 178]);
    line(page, [112, 178], [12, 178]); line(page, [12, 178], [12, 78]);
    text(page, 'Portrait; margins: 12 mm sides, 22 mm top/bottom. Repeated overlap: 10 mm.', 12, 184, 7);
    text(page, 'Piece / PDF pages / tile grid (rows x columns) / cutting instruction', 12, 188, 8);
    plans.slice(section * 16, section * 16 + 16).forEach(({ panel, grid }, index) => {
      const pages = grid.rows * grid.columns;
      const cut = panel.draft ? `cut ${panel.draft.cutQuantity} ${panel.draft.material}` : 'allowances not supplied';
      text(page, `${panel.id} / ${tilePage}-${tilePage + pages - 1} / ${grid.rows} x ${grid.columns} / ${cut}`, 12, 194 + index * 3.5, 7);
      tilePage += pages;
    });
  }
  for (const { panel, grid } of plans) {
    const contours = [{ points: panel.points, dashed: !!panel.draft }, ...(panel.draft ? [{ points: panel.draft.cutLine, dashed: false }] : [])];
    for (let row = 0; row < grid.rows; row++) for (let column = 0; column < grid.columns; column++) {
      await new Promise<void>(resolve => setTimeout(resolve, 0));
      const page = newPage();
      text(page, `${panel.id} | row ${row + 1}/${grid.rows}, column ${column + 1}/${grid.columns}`, 12, 17, 8);
      const originX = grid.minX + column * grid.stepX, originY = grid.minY + row * grid.stepY;
      const transform = (point: Point): Point => [PRINT_MARGIN + point[0] - originX, PRINT_VERTICAL_MARGIN + point[1] - originY];
      const right = PRINT_MARGIN + grid.tileWidth, bottom = PRINT_VERTICAL_MARGIN + grid.tileHeight;
      page.pushOperators(pushGraphicsState(), rectangle(PRINT_MARGIN * POINTS_PER_MM, PRINT_VERTICAL_MARGIN * POINTS_PER_MM, grid.tileWidth * POINTS_PER_MM, grid.tileHeight * POINTS_PER_MM), clip(), endPath());
      for (const contour of contours) {
        const path = contour.points.map((point, index) => `${index ? 'L' : 'M'} ${point[0]} ${point[1]}`).join(' ');
        page.drawSvgPath(path, { x: (PRINT_MARGIN - originX) * POINTS_PER_MM, y: (paperHeight - PRINT_VERTICAL_MARGIN + originY) * POINTS_PER_MM, scale: POINTS_PER_MM, borderWidth: 0.23, borderColor: ink, ...(contour.dashed ? { borderDashArray: [1.1, 0.7] } : {}) });
      }
      if (panel.draft) {
        const [start, end] = panel.draft.grainline.map(transform) as [Point, Point];
        line(page, start, end);
        const angle = Math.atan2(end[1] - start[1], end[0] - start[0]);
        for (const offset of [-0.5, 0.5]) line(page, end, [end[0] - 4 * Math.cos(angle + offset), end[1] - 4 * Math.sin(angle + offset)]);
        for (const mark of panel.draft.marks) {
          const point = transform(mark.point);
          if (mark.kind === 'buttonhole') page.drawCircle({ x: point[0] * POINTS_PER_MM, y: (paperHeight - point[1]) * POINTS_PER_MM, size: POINTS_PER_MM, borderWidth: 0.65, borderColor: ink });
          else {
            line(page, [point[0] - 1.5, point[1]], [point[0] + 1.5, point[1]]);
            line(page, [point[0], point[1] - 1.5], [point[0], point[1] + 1.5]);
          }
          text(page, `${mark.kind}: ${mark.label}`, point[0] + 2, point[1] - 2, 6);
        }
      }
      page.pushOperators(popGraphicsState());
      line(page, [12, 22], [right, 22], false, true); line(page, [right, 22], [right, bottom], false, true);
      line(page, [right, bottom], [12, bottom], false, true); line(page, [12, bottom], [12, 22], false, true);
      if (column < grid.columns - 1) line(page, [right - PRINT_OVERLAP, 22], [right - PRINT_OVERLAP, bottom], true, true);
      if (row < grid.rows - 1) line(page, [12, bottom - PRINT_OVERLAP], [right, bottom - PRINT_OVERLAP], true, true);
      const horizontal = [...(column < grid.columns - 1 ? [grid.tileWidth - PRINT_OVERLAP / 2] : []), ...(column > 0 ? [PRINT_OVERLAP / 2] : [])];
      const vertical = [...(row < grid.rows - 1 ? [grid.tileHeight - PRINT_OVERLAP / 2] : []), ...(row > 0 ? [PRINT_OVERLAP / 2] : [])];
      for (const across of horizontal) {
        const center: Point = [PRINT_MARGIN + across, PRINT_VERTICAL_MARGIN + grid.tileHeight / 2];
        line(page, [center[0] - 2, center[1]], [center[0] + 2, center[1]], false, true);
        line(page, [center[0], center[1] - 2], [center[0], center[1] + 2], false, true);
      }
      for (const down of vertical) {
        const center: Point = [PRINT_MARGIN + grid.tileWidth / 2, PRINT_VERTICAL_MARGIN + down];
        line(page, [center[0] - 2, center[1]], [center[0] + 2, center[1]], false, true);
        line(page, [center[0], center[1] - 2], [center[0], center[1] + 2], false, true);
      }
    }
  }
  signal.throwIfAborted();
  const bytes = await pdf.save();
  if (bytes.length > 16 * 1024 * 1024) throw new Error('Tiled pattern exceeds its byte limit');
  signal.throwIfAborted();
  return bytes;
}
