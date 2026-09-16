import { z } from 'zod';
import type { GarmentDocument, PatternGeometry } from './index';

export const FeatureSchema = z.enum(['body', 'sleeves', 'cuffs', 'collar', 'front-opening', 'tails', 'frills', 'material', 'other']);
export const ShirtDesignSchema = z.object({
  block: z.literal('relaxed-drop-shoulder'),
  sleeves: z.enum(['none', 'short', 'long']),
  sleeveLengthMm: z.number().finite().min(100).max(800),
  cuff: z.enum(['none', 'button']),
  cuffCircumferenceMm: z.number().finite().min(160).max(400),
  cuffDepthMm: z.number().finite().min(30).max(100),
  collar: z.enum(['none', 'stand', 'stand-and-fall']),
  collarStandMm: z.number().finite().min(20).max(50),
  collarFallMm: z.number().finite().min(30).max(100),
  opening: z.enum(['none', 'buttons']),
  placketWidthMm: z.number().finite().min(20).max(45),
  buttonSpacingMm: z.number().finite().min(50).max(110),
  hem: z.enum(['straight', 'curved-back-tail']),
  tailExtensionMm: z.number().finite().min(30).max(200),
  frill: z.enum(['none', 'front-opening']),
  frillWidthMm: z.number().finite().min(15).max(100),
  frillFullness: z.number().finite().min(1.25).max(3),
  seamAllowanceMm: z.number().finite().min(6).max(20),
  rationale: z.string().min(1).max(2000),
}).strict();
export type ShirtDesign = z.infer<typeof ShirtDesignSchema>;
export const defaultShirtDesign:ShirtDesign={
  block:'relaxed-drop-shoulder',sleeves:'long',sleeveLengthMm:550,cuff:'button',cuffCircumferenceMm:220,cuffDepthMm:55,
  collar:'stand-and-fall',collarStandMm:30,collarFallMm:60,opening:'buttons',placketWidthMm:30,buttonSpacingMm:80,
  hem:'straight',tailExtensionMm:100,frill:'none',frillWidthMm:35,frillFullness:1.8,seamAllowanceMm:10,
  rationale:'Editable relaxed shirt construction. These starting dimensions are design assumptions, not measured neck, arm or wrist sizes; review them and make a toile.',
};
export function designIssues(design: ShirtDesign): string[] {
  return [
    ...(design.sleeves === 'short' && design.sleeveLengthMm > 350 ? ['Short sleeves require a construction length of at most 350 mm.'] : []),
    ...(design.sleeves === 'long' && design.sleeveLengthMm < 350 ? ['Long sleeves require a construction length of at least 350 mm.'] : []),
    ...(design.cuff !== 'none' && design.sleeves !== 'long' ? ['Button cuffs require long sleeves.'] : []),
    ...(design.collar !== 'none' && design.opening !== 'buttons' ? ['This collar construction requires a front opening.'] : []),
    ...(design.frill !== 'none' && design.opening !== 'buttons' ? ['Front frills require the front-opening attachment seam.'] : []),
  ];
}

const PointSchema = z.tuple([z.number().finite(), z.number().finite()]);
export const PanelDraftSchema = z.object({
  component: FeatureSchema,
  material: z.enum(['shell', 'interfacing']),
  cutQuantity: z.number().int().min(1).max(4),
  edges: z.array(z.object({name:z.string().min(1).max(100),start:z.number().int().nonnegative(),end:z.number().int().positive(),lengthMm:z.number().finite().positive(),finish:z.enum(['assembly','single-turn-overlocked','bagged-facing'])}).strict()).min(3).max(30),
  cutLine: z.array(PointSchema).min(4).max(2000),
  grainline: z.tuple([PointSchema,PointSchema]),
  marks: z.array(z.object({kind:z.enum(['button','buttonhole','notch','fold']),point:PointSchema,label:z.string().max(100)}).strict()).max(100),
}).strict();
export const AssemblySchema = z.object({
  id:z.string().min(1).max(100),
  sides:z.array(z.object({panel:z.string().min(1).max(100),edge:z.number().int().nonnegative()}).strict()).min(2).max(10),
  treatment:z.enum(['plain','gather','bind','layer']),
  ratio:z.number().finite().min(0.1).max(10),
  instruction:z.string().min(1).max(2000),
}).strict();
export const DraftingSchema = z.object({
  compiler:z.literal('sew-relaxed-shirt/1'),
  seamAllowanceMm:z.number().finite().min(6).max(20),
  components:z.array(FeatureSchema).min(1),
  assembly:z.array(AssemblySchema).max(100),
  measurements:z.array(z.object({name:z.string().max(200),valueMm:z.number().finite().positive(),method:z.string().max(1000)}).strict()).max(30),
  materials:z.array(z.string().max(1000)).max(30),
  operations:z.array(z.string().max(2000)).max(50),
}).strict();

export function validateDrafting(geometry: PatternGeometry): void {
  if (!geometry.drafting) {
    if (geometry.panels.some(panel => panel.draft)) throw new Error('Panel annotations require drafting provenance');
    return;
  }
  const drafting = DraftingSchema.parse(geometry.drafting);
  const panels = new Map(geometry.panels.map(panel => [panel.id, panel]));
  if (panels.size !== geometry.panels.length) throw new Error('Duplicate drafted panel');
  for (const panel of panels.values()) {
    const draft = PanelDraftSchema.parse(panel.draft);
    if (panel.cutQuantity !== draft.cutQuantity) throw new Error('Cut quantity mismatch');
    let cursor = 0;
    for (const edge of draft.edges) {
      if (edge.start !== cursor || edge.end <= edge.start || edge.end >= panel.points.length) throw new Error('Invalid drafted edge interval');
      let length = 0;
      for (let index = edge.start; index < edge.end; index++) {
        const start = panel.points[index]!, end = panel.points[index + 1]!;
        length += Math.hypot(end[0] - start[0], end[1] - start[1]);
      }
      if (Math.abs(length - edge.lengthMm) > 0.02) throw new Error('Drafted edge length mismatch');
      cursor = edge.end;
    }
    if (cursor !== panel.points.length - 1) throw new Error('Unaccounted drafted boundary');
    const cut = draft.cutLine;
    if (Math.hypot(cut[0]![0] - cut.at(-1)![0], cut[0]![1] - cut.at(-1)![1]) > 0.001) throw new Error('Open cut line');
    if (cut.some(point => point.some(value => Math.abs(value) > 6000))) throw new Error('Cut line exceeds bounds');
    validateCutContour(panel.points, cut);
    if (draft.grainline.some(point => !inside(point, panel.points)) || Math.hypot(draft.grainline[0][0]-draft.grainline[1][0],draft.grainline[0][1]-draft.grainline[1][1]) < 1) throw new Error('Invalid grainline');
    if (draft.marks.some(mark => !inside(mark.point, cut))) throw new Error('Mark outside cut line');
  }
  const seamIds = new Set<string>();
  const connected = new Set<string>();
  for (const seam of drafting.assembly) {
    if (seamIds.has(seam.id)) throw new Error('Duplicate assembly identifier');
    seamIds.add(seam.id);
    const lengths = seam.sides.map(side => {
      const panel = panels.get(side.panel);
      const edge = panel?.draft?.edges[side.edge];
      if (!edge) throw new Error('Invalid assembly edge reference');
      connected.add(side.panel);
      return edge.lengthMm;
    });
    if (lengths.length !== 2 || Math.abs(lengths[0]! / lengths[1]! - seam.ratio) > 0.0001) throw new Error('Assembly ratio mismatch');
    if (seam.treatment !== 'gather' && Math.abs(lengths[0]! - lengths[1]!) > 0.1) throw new Error('Unequal plain seam');
    if (seam.treatment === 'gather' && (seam.ratio < 1 || seam.ratio > 3)) throw new Error('Invalid gathering ratio');
    for (const side of seam.sides) {
      const panel=panels.get(side.panel)!;
      const marks=panel.draft!.marks.filter(mark=>mark.kind==='notch'&&mark.label===seam.id);
      const required=seam.sides.filter(part=>part.panel===side.panel).length;
      if(marks.length!==required)throw new Error('Missing seam registration marks');
      const boundary=panel.draft!.edges[side.edge]!;
      const points=panel.points.slice(boundary.start,boundary.end+1);
      if(!marks.some(mark=>points.slice(0,-1).some((point,index)=>onSegment(mark.point,point,points[index+1]!))))throw new Error('Registration mark is not on its seam');
    }
  }
  if (connected.size !== panels.size) throw new Error('Disconnected pattern piece');
  for(const panel of panels.values())for(const [index,edge] of panel.draft!.edges.entries()) {
    const attached=drafting.assembly.some(seam=>seam.sides.some(side=>side.panel===panel.id&&side.edge===index));
    if((edge.finish==='assembly')!==attached)throw new Error('Boundary treatment mismatch');
  }
  const reached = new Set<string>([geometry.panels[0]!.id]);
  for (let iteration=0;iteration<panels.size;iteration++) for (const seam of drafting.assembly) {
    if (seam.sides.some(side=>reached.has(side.panel))) for(const side of seam.sides) reached.add(side.panel);
  }
  if (reached.size !== panels.size) throw new Error('Disconnected assembly graph');
  const evidenced = new Set(geometry.panels.map(panel => panel.draft!.component));
  const back = panels.get('back_left'), front = panels.get('front_left');
  if (back && front) {
    const backHem = back.draft!.edges.find(edge => edge.name === 'hem')!;
    const start = back.points[backHem.start]!, end = back.points[backHem.end]!;
    if (start[1] > end[1] + 20) evidenced.add('tails');
  }
  if (new Set(drafting.components).size !== drafting.components.length || drafting.components.some(feature => !evidenced.has(feature)) || [...evidenced].some(feature => !drafting.components.includes(feature))) throw new Error('Component evidence mismatch');
  if (geometry.stitches.length !== drafting.assembly.length || geometry.stitches.some((seam, index) => {
    const sides = drafting.assembly[index]!.sides;
    return seam.panelA !== sides[0]!.panel || seam.edgeA !== sides[0]!.edge || seam.panelB !== sides[1]!.panel || seam.edgeB !== sides[1]!.edge;
  })) throw new Error('Assembly graph mismatch');
}

type Point = [number,number];
function cross(start:Point,end:Point,point:Point) { return (end[0]-start[0])*(point[1]-start[1])-(end[1]-start[1])*(point[0]-start[0]); }
function onSegment(point:Point,start:Point,end:Point) { return Math.abs(cross(start,end,point)) < 0.00001 && point[0]>=Math.min(start[0],end[0])-0.00001 && point[0]<=Math.max(start[0],end[0])+0.00001 && point[1]>=Math.min(start[1],end[1])-0.00001 && point[1]<=Math.max(start[1],end[1])+0.00001; }
function intersects(first:Point,second:Point,third:Point,fourth:Point) {
  return (cross(first,second,third)*cross(first,second,fourth)<0 && cross(third,fourth,first)*cross(third,fourth,second)<0) || onSegment(third,first,second) || onSegment(fourth,first,second) || onSegment(first,third,fourth) || onSegment(second,third,fourth);
}
function inside(point:Point,polygon:Point[]) {
  let included=false;
  for(let index=0;index<polygon.length-1;index++) {
    const start=polygon[index]!,end=polygon[index+1]!;
    if(onSegment(point,start,end))return true;
    if((start[1]>point[1])!==(end[1]>point[1]) && point[0]<(end[0]-start[0])*(point[1]-start[1])/(end[1]-start[1])+start[0])included=!included;
  }
  return included;
}
function area(points:Point[]) { return Math.abs(points.slice(0,-1).reduce((sum,point,index)=>sum+point[0]*points[index+1]![1]-points[index+1]![0]*point[1],0))/2; }
function validateCutContour(seam:Point[],cut:Point[]) {
  if(seam.length>2000 || area(cut)<=area(seam)+1 || seam.some(point=>!inside(point,cut)))throw new Error('Invalid cut-line area or containment');
  for(let first=0;first<cut.length-1;first++) {
    if(Math.hypot(cut[first]![0]-cut[first+1]![0],cut[first]![1]-cut[first+1]![1])<0.00001)throw new Error('Degenerate cut segment');
    for(let second=first+2;second<cut.length-1;second++) {
      if(first===0&&second===cut.length-2)continue;
      if(intersects(cut[first]!,cut[first+1]!,cut[second]!,cut[second+1]!))throw new Error('Self-intersecting cut line');
    }
    for(let index=0;index<seam.length-1;index++)if(intersects(cut[first]!,cut[first+1]!,seam[index]!,seam[index+1]!))throw new Error('Cut line crosses seam line');
  }
}

export function validateDesignGeometry(doc:GarmentDocument,geometry:PatternGeometry):void {
  const design=doc.garment.design;
  if(!design) { if(geometry.drafting)throw new Error('Unexpected component drafting'); return; }
  validateDrafting(geometry);
  if(!geometry.drafting || geometry.family!=='shirt')throw new Error('Selected construction was not generated');
  const expected=new Map<string,number>([['front_left',1],['front_right',1],['back_left',1],['back_right',1]]);
  const seams=['shoulder_left','shoulder_right','side_left','side_right','center_back'];
  for(const side of ['left','right']) {
    if(design.opening==='buttons') {expected.set(`placket_${side}`,2);seams.push(`placket_attach_${side}`);}
    if(design.frill!=='none') {expected.set(`frill_${side}`,1);seams.push(`frill_gather_${side}`);}
    if(design.sleeves!=='none') {expected.set(`sleeve_${side}`,1);seams.push(`sleeve_front_${side}`,`sleeve_back_${side}`,`underarm_${side}`);}
    if(design.cuff!=='none') {
      expected.set(`cuff_${side}`,2);seams.push(`cuff_gather_${side}`);
      for(const position of ['left','right']) {expected.set(`opening_binding_${side}_${position}`,1);seams.push(`bind_opening_${side}_${position}`);}
    }
  }
  if(design.opening==='none')seams.push('center_front');
  if(design.collar!=='none') {expected.set('collar_stand',2);for(let index=0;index<6;index++)seams.push(`collar_neck_${index}`);}
  if(design.collar==='stand-and-fall') {expected.set('collar_fall',2);seams.push('collar_fall_attach');}
  if(expected.size!==geometry.panels.length || geometry.panels.some(panel=>expected.get(panel.id)!==panel.cutQuantity))throw new Error('Selected piece inventory mismatch');
  if(seams.length!==geometry.drafting.assembly.length || geometry.drafting.assembly.some(seam=>!seams.includes(seam.id)))throw new Error('Selected assembly inventory mismatch');
  if(geometry.drafting.seamAllowanceMm!==design.seamAllowanceMm)throw new Error('Allowance selection mismatch');
  const get=(name:string)=>geometry.panels.find(panel=>panel.id===name)!;
  const close=(actual:number,expected:number)=>{if(Math.abs(actual-expected)>0.05)throw new Error('Selected construction dimension mismatch');};
  const closure=(name:string,kind:'button'|'buttonhole',points:Point[])=>{
    const marks=get(name).draft!.marks.filter(mark=>mark.kind===kind).sort((first,second)=>first.point[1]-second.point[1]);
    if(marks.length!==points.length)throw new Error('Closure mark count mismatch');
    for(const [index,point] of points.entries())if(Math.hypot(marks[index]!.point[0]-point[0],marks[index]!.point[1]-point[1])>0.05)throw new Error('Closure mark placement mismatch');
  };
  if(design.collar!=='none')close(get('collar_stand').heightMm,design.collarStandMm);
  if(design.collar==='stand-and-fall')close(get('collar_fall').heightMm,design.collarFallMm);
  const back=get('back_left'),hem=back.draft!.edges.find(edge=>edge.name==='hem')!;
  close(back.points[hem.start]![1]-back.points[hem.end]![1],design.hem==='straight'?0:design.tailExtensionMm);
  for(const side of ['left','right']) {
    if(design.opening!=='none') {
      const placket=get(`placket_${side}`);
      close(placket.widthMm,design.placketWidthMm);
      closure(placket.id,side==='left'?'button':'buttonhole',Array.from({length:Math.max(2,Math.floor((placket.heightMm-50)/design.buttonSpacingMm)+1)},(_,index)=>[design.placketWidthMm/2,25+index*design.buttonSpacingMm]));
      closure(placket.id,side==='left'?'buttonhole':'button',[]);
    }
    if(design.frill!=='none') {close(get(`frill_${side}`).widthMm,design.frillWidthMm);close(geometry.drafting.assembly.find(seam=>seam.id===`frill_gather_${side}`)!.ratio,design.frillFullness);}
    if(design.sleeves!=='none')close(get(`sleeve_${side}`).heightMm+(design.cuff!=='none'?design.cuffDepthMm:0),design.sleeveLengthMm);
    if(design.cuff!=='none') {
      close(get(`cuff_${side}`).widthMm,design.cuffCircumferenceMm+20);close(get(`cuff_${side}`).heightMm,design.cuffDepthMm);
      closure(`cuff_${side}`,'button',[[10,design.cuffDepthMm/2]]);
      closure(`cuff_${side}`,'buttonhole',[[design.cuffCircumferenceMm+10,design.cuffDepthMm/2]]);
    }
  }
  if(design.collar!=='none') {
    closure('collar_stand','button',[[design.placketWidthMm/2,design.collarStandMm/2]]);
    closure('collar_stand','buttonhole',[[get('collar_stand').widthMm-design.placketWidthMm/2,design.collarStandMm/2]]);
  }
  const measurements=new Map(geometry.drafting.measurements.map(row=>[row.name,row.valueMm]));
  const underarmWidth=(name:string)=>{const piece=get(name),armhole=piece.draft!.edges.find(edge=>edge.name==='armhole')!;return piece.points[armhole.start]![0];};
  const derived=new Map<string,number>([
    ['Closed chest at underarm',underarmWidth('front_left')+underarmWidth('front_right')+underarmWidth('back_left')+underarmWidth('back_right')+(design.opening==='buttons'?design.placketWidthMm:0)],
    ['Side length from shoulder baseline',get('front_left').heightMm],
    ['Back center length',back.draft!.edges.find(edge=>edge.name==='center')!.lengthMm],
  ]);
  if(design.sleeves!=='none')derived.set('Sleeve including cuff',design.sleeveLengthMm);
  if(design.cuff!=='none')derived.set('Closed cuff circumference',design.cuffCircumferenceMm);
  if(design.collar!=='none')derived.set('Closed collar stand circumference',get('collar_stand').widthMm-design.placketWidthMm);
  if(measurements.size!==geometry.drafting.measurements.length||measurements.size!==derived.size)throw new Error('Derived measurement inventory mismatch');
  for(const [name,value] of derived)if(!measurements.has(name)||Math.abs(measurements.get(name)!-value)>0.05)throw new Error('Derived measurement geometry mismatch');
}

export function designCoverage(doc: GarmentDocument, geometry: PatternGeometry | null) {
  if(geometry&&doc.garment.design)validateDesignGeometry(doc,geometry);
  const components = new Set(geometry?.drafting?.components ?? []);
  const design = doc.garment.design;
  const expected = design ? ['body', ...(design.sleeves !== 'none' ? ['sleeves'] : []), ...(design.cuff !== 'none' ? ['cuffs'] : []), ...(design.collar !== 'none' ? ['collar'] : []), ...(design.opening !== 'none' ? ['front-opening'] : []), ...(design.hem !== 'straight' ? ['tails'] : []), ...(design.frill !== 'none' ? ['frills'] : [])] : [];
  const missing = expected.filter(feature => !components.has(feature as z.infer<typeof FeatureSchema>));
  const unresolved = doc.requirements.filter(row => row.status !== 'supported' || !row.feature || row.feature === 'other' || (row.feature !== 'material' && !components.has(row.feature)));
  return {status: !geometry ? 'not-generated' : !design || missing.length || unresolved.length ? 'partial' : 'drafted', missing, unresolved: unresolved.map(row => row.text), notice: 'Digital feature coverage only. Interpretation may omit or misunderstand a request; review against the original brief. No physical fit or sewing validation.'} as const;
}
