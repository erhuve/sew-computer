import {z} from 'zod';
import type {PatternGeometry} from './index';
const coordinate=z.number().finite().min(-2000).max(2000);
export const CustomPieceSchema=z.object({
  id:z.string().regex(/^[a-zA-Z0-9_-]{1,80}$/),name:z.string().min(1).max(100),
  points:z.array(z.tuple([coordinate,coordinate])).min(3).max(30),
  placement:z.object({positionMm:z.tuple([coordinate,coordinate,coordinate]),rotationDeg:z.tuple([z.number().min(-180).max(180),z.number().min(-180).max(180),z.number().min(-180).max(180)]),bendDeg:z.number().finite().min(-360).max(360)}).strict(),
}).strict();
export const CustomSeamSchema=z.object({id:z.string().regex(/^[a-zA-Z0-9_-]{1,80}$/),a:z.string(),edgeA:z.number().int().min(0).max(29),b:z.string(),edgeB:z.number().int().min(0).max(29),treatment:z.enum(['plain','gather'])}).strict();
export const CustomPatternSchema=z.object({block:z.literal('custom-pattern'),pieces:z.array(CustomPieceSchema).min(1).max(16),seams:z.array(CustomSeamSchema).max(100),seamAllowanceMm:z.number().finite().min(6).max(20),rationale:z.string().min(1).max(2000)}).strict();
export type CustomPattern=z.infer<typeof CustomPatternSchema>;
export type CustomPiece=z.infer<typeof CustomPieceSchema>;
export function customPatternIssues(pattern:CustomPattern):string[]{
  const problems:string[]=[],pieces=new Map(pattern.pieces.map(piece=>[piece.id,piece])),occupied=new Set<string>();
  if(pieces.size!==pattern.pieces.length)problems.push('Each piece needs a unique identifier.');
  if(new Set(pattern.seams.map(seam=>seam.id)).size!==pattern.seams.length)problems.push('Each seam needs a unique identifier.');
  for(const piece of pattern.pieces){
    const points=piece.points;let area=0;
    for(let i=0;i<points.length;i++){const a=points[i]!,b=points[(i+1)%points.length]!;area+=a[0]*b[1]-a[1]*b[0];if(Math.hypot(a[0]-b[0],a[1]-b[1])<1)problems.push(`${piece.name}: consecutive points must be at least 1 mm apart.`);}
    if(Math.abs(area)<200)problems.push(`${piece.name}: draw an outline with at least 100 mm² of area.`);
  }
  for(const seam of pattern.seams){
    const lengths:number[]=[];
    for(const [id,index] of [[seam.a,seam.edgeA],[seam.b,seam.edgeB]] as const){
      const piece=pieces.get(id),key=`${id}:${index}`;
      if(!piece||index>=piece.points.length){problems.push(`Seam ${seam.id} refers to a missing edge.`);continue;}
      if(occupied.has(key))problems.push('An edge can belong to only one seam.');occupied.add(key);
      const a=piece.points[index]!,b=piece.points[(index+1)%piece.points.length]!;lengths.push(Math.hypot(a[0]-b[0],a[1]-b[1]));
    }
    if(lengths.length===2){if(seam.treatment==='plain'&&Math.abs(lengths[0]!-lengths[1]!)>.1)problems.push(`Seam ${seam.id}: plain edges must have equal lengths; edit the outline or choose gathering.`);if(seam.treatment==='gather'&&(lengths[0]!<lengths[1]!||lengths[0]!>lengths[1]!*3))problems.push(`Seam ${seam.id}: edge A must be 1–3 times edge B for gathering.`);}
  }
  return [...new Set(problems)];
}
export function validateCustomGeometry(design:CustomPattern,geometry:PatternGeometry){
  if(customPatternIssues(design).length||geometry.family!=='custom'||geometry.drafting?.compiler!=='sew-custom-pattern/1'||geometry.panels.length!==design.pieces.length||geometry.drafting.seamAllowanceMm!==design.seamAllowanceMm)throw new Error('Custom pattern provenance mismatch');
  for(const source of design.pieces){const panel=geometry.panels.find(panel=>panel.id===source.id);const expected=[...source.points,source.points[0]!];if(!panel||panel.name!==source.name||panel.cutQuantity!==1||panel.points.length!==expected.length||panel.points.some((point,i)=>point.some((value,axis)=>Math.abs(value-expected[i]![axis]!)>1e-6))||panel.draft?.edges.some((edge,i)=>edge.name!==`edge_${i}`||edge.start!==i||edge.end!==i+1))throw new Error('Custom source outline mismatch');}
  if(geometry.drafting.assembly.length!==design.seams.length)throw new Error('Custom seam inventory mismatch');
  design.seams.forEach((seam,index)=>{const actual=geometry.drafting!.assembly[index]!;if(actual.id!==seam.id||actual.treatment!==seam.treatment||actual.sides[0]?.panel!==seam.a||actual.sides[0]?.edge!==seam.edgeA||actual.sides[1]?.panel!==seam.b||actual.sides[1]?.edge!==seam.edgeB)throw new Error('Custom source seam mismatch');});
}
const outline:[number,number][]=[[100,0],[350,0],[450,650],[0,650]];
const pieceNames=['front_left','front_right','back_right','back_left'];
export const defaultCustomPattern:CustomPattern={block:'custom-pattern',seamAllowanceMm:10,rationale:'Editable four-panel skirt example with owner-defined outlines, seams and posing guides. Change the pieces to create your own construction; no fit or sewing validation is implied.',pieces:pieceNames.map((id,index)=>({id,name:id.replaceAll('_',' '),points:structuredClone(outline),placement:{positionMm:[0,0,0],rotationDeg:[0,[45,135,-135,-45][index]!,0],bendDeg:90}})),seams:pieceNames.map((id,index)=>({id:`side_${index+1}`,a:id,edgeA:1,b:pieceNames[(index+1)%4]!,edgeB:3,treatment:'plain'}))};
