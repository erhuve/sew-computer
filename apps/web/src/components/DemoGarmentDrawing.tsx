import { useState } from 'react';
import { assumed, emptyDocument, type PatternGeometry } from '../../../../packages/contracts';
import type { ShirtDesign } from '../../../../packages/contracts/design';
import { garmentFlats } from '../../../../packages/contracts/flats';

export default function DemoGarmentDrawing({ pattern, design, color }: { pattern: PatternGeometry; design: ShirtDesign; color: string }) {
  const [view, setView] = useState<'front' | 'back'>('front');
  const doc = emptyDocument('Visual demo', 'Synthetic shirt example');
  doc.body = { height: assumed(1700), bust: assumed(960), waist: assumed(760), hip: assumed(1000), shoulder: assumed(400) };
  doc.garment = { family: 'shirt', length: assumed(650), ease: assumed(100), flare: 1, design };
  const flat = garmentFlats(doc, pattern).find(item => item.view === view)!;
  const extent = Math.max(...flat.lines.flatMap(line => line.points.map(point => Math.abs(point[0])))) + 28;
  const scale = 220 / (pattern.drafting?.measurements.find(row => row.name === 'Side length from shoulder baseline')?.valueMm ?? 650);
  const body = pattern.panels.find(panel => panel.id === `${view}_left`)!;
  const center = view === 'front' && design.opening === 'buttons' ? design.placketWidthMm / 2 : 0;
  const openingStart = (body.points[body.draft?.edges.find(edge => edge.name === 'center')?.start ?? 0]?.[1] ?? 80) * scale;
  const points = (values: number[][]) => values.map(point => point.join(',')).join(' ');
  return <div className="demo-drawing">
    <div className="demo-drawing-controls" aria-label="Drawing view"><button aria-pressed={view === 'front'} onClick={() => setView('front')}>Front</button><button aria-pressed={view === 'back'} onClick={() => setView('back')}>Back</button><span>Construction drawing · not a fit simulation</span></div>
    <div className="demo-drawing-stage"><svg viewBox={`${-extent} -60 ${extent * 2} 360`} role="img" aria-label={`${view} garment construction preview`}>
      <defs><filter id="demo-paper-shadow" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="3" stdDeviation="3" floodColor="#42513d" floodOpacity=".10" /></filter></defs>
      <g filter="url(#demo-paper-shadow)">
        {[-1, 1].map(sign => <polygon key={sign} points={points(body.points.map(([x, y]) => [sign * (x + center) * scale, y * scale]))} fill={color} />)}
        {flat.lines.filter(line => line.points.length === 4).map((line, index) => <polygon key={index} points={points(line.points)} fill={color} />)}
        {center > 0 && <rect x={-center * scale} y={openingStart} width={center * 2 * scale} height={220 - openingStart} fill={color} />}
      </g>
      {flat.lines.map((line, index) => <polyline key={index} points={points(line.points)} fill="none" stroke="#344438" strokeWidth={line.detail ? .85 : 1.4} strokeLinejoin="round" strokeLinecap="round" />)}
      {flat.buttons.map((point, index) => <circle key={index} cx={point[0]} cy={point[1]} r={1.6} fill="#f8f7ef" stroke="#344438" strokeWidth=".7" />)}
    </svg><span className="demo-drawing-caption">{view === 'front' ? 'FRONT' : 'BACK'} / {design.sleeves === 'long' ? 'LONG SLEEVE' : 'SHORT SLEEVE'} / {design.frill === 'none' ? 'CLEAN FRONT' : 'GATHERED FRILL'}</span></div>
    <div className="demo-drawing-note">The drawing follows this example’s construction and dimensions. Switch to 3D pieces to inspect the underlying pattern meshes.</div>
  </div>;
}
