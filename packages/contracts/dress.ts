import {z} from 'zod';
import type {GarmentDocument,PatternGeometry} from './index';

export const PanelDressSchema=z.object({
  block:z.literal('panel-dress'),skirtStyle:z.enum(['flared','gathered','tiered']),
  neckline:z.enum(['round','v','square']),neckWidthMm:z.number().finite().min(70).max(160),neckDepthMm:z.number().finite().min(70).max(200),
  bodiceLengthMm:z.number().finite().min(320).max(650),waistRatio:z.number().finite().min(.75).max(1.2),
  skirtFullness:z.number().finite().min(1.1).max(2),tierFullness:z.number().finite().min(1.1).max(1.8),
  sleeves:z.enum(['none','short','long']),sleeveLengthMm:z.number().finite().min(100).max(700),
  seamAllowanceMm:z.number().finite().min(6).max(20),rationale:z.string().min(1).max(2000),
}).strict();
export type PanelDress=z.infer<typeof PanelDressSchema>;
export const defaultPanelDress:PanelDress={block:'panel-dress',skirtStyle:'flared',neckline:'round',neckWidthMm:105,neckDepthMm:120,bodiceLengthMm:430,waistRatio:.9,skirtFullness:1.5,tierFullness:1.4,sleeves:'none',sleeveLengthMm:220,seamAllowanceMm:10,rationale:'Separate bodice and four-panel flared skirt. Waist placement and proportions are editable design assumptions. Center back is left open for closure development; neckline finishing, closure and fit require a toile.'};
export function panelDressIssues(d:PanelDress){return [
  ...(d.sleeves==='short'&&d.sleeveLengthMm>350?['Short sleeves must be at most 350 mm.']:[]),
  ...(d.sleeves==='long'&&d.sleeveLengthMm<350?['Long sleeves must be at least 350 mm.']:[]),
];}
const value=(m:GarmentDocument['garment']['length'])=>'value' in m?m.value*({mm:1,cm:10,in:25.4}[m.unit]):NaN;
export function panelDressDimensions(doc:Pick<GarmentDocument,'garment'|'body'>,d:PanelDress){
  const chest=(Math.max(value(doc.body.bust),value(doc.body.hip))+value(doc.garment.ease))/4,arm=value(doc.body.bust)/10+110;
  const waist=chest*d.waistRatio,skirtLength=value(doc.garment.length)-d.bodiceLengthMm;
  const tiers=d.skirtStyle==='tiered'?2:1;
  const first=waist*(d.skirtStyle==='flared'?1:d.skirtFullness);
  return {chest,arm,waist,skirtLength,tiers,first};
}
export function panelDressSizingIssues(doc:Pick<GarmentDocument,'garment'|'body'>,d:PanelDress){
  const m=panelDressDimensions(doc,d);
  const hem=m.first*doc.garment.flare*(m.tiers===2?d.tierFullness*doc.garment.flare:1);
  return [
    ...(hem>1800?['Reduce gathering or hem sweep: each skirt panel must stay within 1800 mm.']:[]),
    ...(m.chest*2<value(doc.body.shoulder)+20?['This dress bodice needs at least 20 mm of shoulder ease. Increase garment ease; keep accurate body measurements.']:[]),
    ...(d.bodiceLengthMm<m.arm+80?['Move the waist seam lower: allow at least 80 mm below the armhole.']:[]),
    ...(m.skirtLength<(m.tiers===2?350:250)?['Allow at least 250 mm of skirt, or 350 mm for two tiers. Adjust the dress length or waist position.']:[]),
    ...(d.neckWidthMm>m.chest-35||d.neckDepthMm>m.arm-15?['The neckline must leave shoulder and underarm fabric. Reduce neckline width or depth.']:[]),
  ];
}

// Independent source recipe used to verify the Python compiler's geometry.
type Point=[number,number];
type Piece={id:string;names:string[];paths:Point[][]};
export function panelDressRecipe(doc:Pick<GarmentDocument,'garment'|'body'>,d:PanelDress){
  const m=panelDressDimensions(doc,d),pieces:Piece[]=[],seams:{id:string;a:string;ae:string;b:string;be:string;treatment:'plain'|'gather'}[]=[];
  const add=(id:string,names:string[],paths:Point[][])=>pieces.push({id,names,paths});
  const join=(id:string,a:string,ae:string,b:string,be:string,treatment:'plain'|'gather'='plain')=>seams.push({id,a,ae,b,be,treatment});
  for(const face of ['front','back'])for(const side of ['left','right']){
    const depth=face==='front'?d.neckDepthMm:25,w=d.neckWidthMm;
    const neck:Point[]=face==='front'&&d.neckline==='v'?[[w,0],[0,depth]]:face==='front'&&d.neckline==='square'?[[w,0],[w,depth],[0,depth]]:Array.from({length:25},(_,i)=>[w*Math.cos(Math.PI*i/48),depth*Math.sin(Math.PI*i/48)] as Point);
    neck[0]=[w,0];neck[neck.length-1]=[0,depth];
    add(`bodice_${face}_${side}`,['center','waist','side','armhole','shoulder','neck'],[[[0,depth],[0,d.bodiceLengthMm]],[[0,d.bodiceLengthMm],[m.waist,d.bodiceLengthMm]],[[m.waist,d.bodiceLengthMm],[m.chest,m.arm]],[[m.chest,m.arm],[m.chest,0]],[[m.chest,0],[w,0]],neck]);
  }
  join('center_front','bodice_front_left','center','bodice_front_right','center');
  for(const side of ['left','right']){
    for(const edge of ['side','shoulder'])join(`${edge}_${side}`,`bodice_front_${side}`,edge,`bodice_back_${side}`,edge);
    if(d.sleeves!=='none'){
      const cap=m.arm*2,wrist=cap*.72,inset=(cap-wrist)/2,l=d.sleeveLengthMm;
      add(`sleeve_${side}`,['cap_front','cap_back','underarm_right','wrist','underarm_left'],[[[0,0],[m.arm,0]],[[m.arm,0],[cap,0]],[[cap,0],[cap-inset,l]],[[cap-inset,l],[inset,l]],[[inset,l],[0,0]]]);
      join(`sleeve_front_${side}`,`sleeve_${side}`,'cap_front',`bodice_front_${side}`,'armhole');join(`sleeve_back_${side}`,`sleeve_${side}`,'cap_back',`bodice_back_${side}`,'armhole');join(`underarm_${side}`,`sleeve_${side}`,'underarm_right',`sleeve_${side}`,'underarm_left');
    }
  }
  let previous=m.waist;
  for(let tier=0;tier<m.tiers;tier++){
    const top=tier===0?m.first:previous*d.tierFullness,hem=top*doc.garment.flare,h=m.skirtLength/m.tiers,inset=(hem-top)/2;
    for(const face of ['front','back'])for(const side of ['left','right']){
      const name=`skirt${tier}_${face}_${side}`;
      add(name,['waist','outer','hem','center'],[[[inset,0],[inset+top,0]],[[inset+top,0],[hem,h]],[[hem,h],[0,h]],[[0,h],[inset,0]]]);
      join(`waist${tier}_${face}_${side}`,name,'waist',tier===0?`bodice_${face}_${side}`:`skirt${tier-1}_${face}_${side}`,tier===0?'waist':'hem',top>previous+.001?'gather':'plain');
    }
    for(const face of ['front','back'])join(`center${tier}_${face}`,`skirt${tier}_${face}_left`,'center',`skirt${tier}_${face}_right`,'center');
    for(const side of ['left','right'])join(`side${tier}_${side}`,`skirt${tier}_front_${side}`,'outer',`skirt${tier}_back_${side}`,'outer');
    previous=hem;
  }
  return {pieces,seams,measurements:{'Chest circumference':m.chest*4,'Bodice waist circumference':m.waist*4,'Waist seam from shoulder':d.bodiceLengthMm,'Skirt length':m.skirtLength,'Hem circumference':previous*4,'Dress length':value(doc.garment.length)}};
}
export function validatePanelDress(doc:GarmentDocument,geometry:PatternGeometry,d:PanelDress){
  const drafting=geometry.drafting!;
  if(drafting.compiler!=='sew-panel-dress/1'||drafting.seamAllowanceMm!==d.seamAllowanceMm||panelDressSizingIssues(doc,d).length)throw new Error('Dress construction mismatch');
  const recipe=panelDressRecipe(doc,d),close=(a:number,b:number)=>{if(!Number.isFinite(a)||!Number.isFinite(b)||Math.abs(a-b)>.0001)throw new Error('Dress source dimensions differ');};
  if(geometry.panels.length!==recipe.pieces.length||drafting.assembly.length!==recipe.seams.length)throw new Error('Dress source inventory differs');
  for(const piece of recipe.pieces){
    const p=geometry.panels.find(p=>p.id===piece.id),points=[piece.paths[0]![0]!,...piece.paths.flatMap(path=>path.slice(1))];
    if(!p||p.cutQuantity!==1||p.draft?.component!==(piece.id.startsWith('sleeve_')?'sleeves':'body')||p.points.length!==points.length||p.draft.edges.length!==piece.names.length)throw new Error('Dress piece inventory differs');
    const marks=recipe.seams.flatMap(s=>[[s.a,s.id],[s.b,s.id]]).filter(([id])=>id===piece.id).map(([,id])=>id);
    if(p.draft.marks.length!==marks.length||p.draft.marks.some(mark=>mark.kind!=='notch'||!marks.includes(mark.label)))throw new Error('Unexpected dress hardware or registration');
    points.forEach((xy,i)=>xy.forEach((v,axis)=>close(p.points[i]![axis]!,v)));
    let cursor=0;piece.paths.forEach((path,i)=>{const edge=p.draft!.edges[i]!;if(edge.name!==piece.names[i]||edge.start!==cursor||edge.end!==cursor+path.length-1)throw new Error('Dress edge identity differs');cursor+=path.length-1;});
    close(p.widthMm,Math.max(...points.map(p=>p[0])));close(p.heightMm,Math.max(...points.map(p=>p[1])));
  }
  for(const seam of recipe.seams){
    const actual=drafting.assembly.find(s=>s.id===seam.id),sides=[[seam.a,seam.ae],[seam.b,seam.be]].map(([id,label])=>({panel:id!,edge:geometry.panels.find(p=>p.id===id)!.draft!.edges.findIndex(e=>e.name===label)}));
    if(!actual||actual.treatment!==seam.treatment||actual.sides.length!==2||actual.sides.some((s,i)=>s.panel!==sides[i]!.panel||s.edge!==sides[i]!.edge))throw new Error('Dress attachment differs');
    const lengths=sides.map(s=>geometry.panels.find(p=>p.id===s.panel)!.draft!.edges[s.edge]!.lengthMm);close(actual.ratio,lengths[0]!/lengths[1]!);
  }
  if(drafting.measurements.length!==Object.keys(recipe.measurements).length)throw new Error('Dress measurement inventory differs');
  for(const [name,expected] of Object.entries(recipe.measurements)){const row=drafting.measurements.find(m=>m.name===name);if(!row)throw new Error('Dress measurement missing');close(row.valueMm,expected);}
}
