import { useState } from "react";
import { Maximize2, Minus, Plus, Layers3 } from "lucide-react";
import type { PatternGeometry } from "../../../../packages/contracts";

export default function PatternCanvas({
  geometry,
  selected,
  onSelect,
}: {
  geometry: PatternGeometry | null;
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const [zoom, setZoom] = useState(1);
  if (!geometry)
    return (
      <div className="canvas-empty">
        <Layers3 size={38} strokeWidth={1} />
        <h2>The pattern comes next.</h2>
        <p>
          Choose a supported starting shape, acknowledge body inputs, then save
          a revision and generate its actual panels.
        </p>
        <span className="quiet">
          Your original idea can go beyond this adapter.
        </span>
      </div>
    );
  let x = 30,
    y = 30,
    rowHeight = 0;
  const cells = geometry.panels.map((panel) => {
    const minX = Math.min(...panel.points.map((p) => p[0])),
      minY = Math.min(...panel.points.map((p) => p[1]));
    if (x + panel.widthMm > 1600 && x > 30) {
      x = 30;
      y += rowHeight + 85;
      rowHeight = 0;
    }
    const cell = { panel, x, y, minX, minY };
    x += panel.widthMm + 65;
    rowHeight = Math.max(rowHeight, panel.heightMm);
    return cell;
  });
  const width = Math.max(600, ...cells.map((c) => c.x + c.panel.widthMm + 30)),
    height = y + rowHeight + 80;
  return (
    <div className="pattern-stage">
      <div className="canvas-tools">
        <span>2D geometry · mm</span>
        <div>
          <button
            aria-label="Zoom out"
            onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
          >
            <Minus size={16} />
          </button>
          <output>{Math.round(zoom * 100)}%</output>
          <button
            aria-label="Zoom in"
            onClick={() => setZoom((z) => Math.min(3, z + 0.25))}
          >
            <Plus size={16} />
          </button>
          <button aria-label="Fit pattern to view" onClick={() => setZoom(1)}>
            <Maximize2 size={16} />
          </button>
        </div>
      </div>
      <div className="pattern-scroll">
        <svg
          role="img"
          aria-label={`Generated ${geometry.family} pattern with ${geometry.panels.length} panels; printable reference, not a cutting pattern`}
          viewBox={`0 0 ${width} ${height}`}
          style={{ minWidth: `${zoom * 100}%`, width: `${zoom * 100}%` }}
        >
          {cells.map(({ panel, x, y, minX, minY }, i) => (
            <g
              key={panel.id}
              transform={`translate(${x},${y})`}
              role="button"
              aria-label={`Select panel ${panel.name}`}
              tabIndex={0}
              onClick={() => onSelect(panel.id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelect(panel.id);
                }
              }}
              className={
                selected === panel.id ? "panel-shape selected" : "panel-shape"
              }
            >
              <polygon
                points={panel.points
                  .map(([a, b]) => `${a - minX + 0},${panel.heightMm - (b - minY) + 0}`)
                  .join(" ")}
                fill={i % 2 ? "#e4ece6" : "#eee7da"}
                strokeWidth={selected === panel.id ? 4 : 2}
                vectorEffect="non-scaling-stroke"
              />
              <text x={0} y={panel.heightMm + 28} fontSize={18}>
                {panel.name}
              </text>
              <text
                className="panel-dimension"
                x={0}
                y={panel.heightMm + 49}
                fontSize={14}
              >
                {Math.round(panel.widthMm)} × {Math.round(panel.heightMm)} mm
              </text>
            </g>
          ))}
        </svg>
      </div>
      <div className="canvas-caption">
        <span>
          {geometry.panels.length} panels · {geometry.stitches.length} seam
          links
        </span>
        <span>Printable reference · fit unverified</span>
      </div>
    </div>
  );
}
