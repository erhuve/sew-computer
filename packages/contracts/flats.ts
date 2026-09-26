import { mm, type GarmentDocument, type PatternGeometry } from './index';

export type Flat = {view:'front'|'back';lines:{points:[number,number][];detail:boolean}[];buttons:[number,number][]};
export function garmentFlats(doc:Pick<GarmentDocument,'garment'> & Partial<Pick<GarmentDocument,'body'>>, geometry:PatternGeometry|null = null):Flat[] {
  const design=doc.garment.design;
  if(!design)return [];
  if(design.block==='custom-pattern')return [];
  if(design.block==='elastic-waist-skirt') {
    const length=mm(doc.garment.length)??650,depth=design.waistbandDepthMm;
    const waist=geometry?.drafting?.measurements.find(row=>row.name==='Assumed relaxed elastic circumference')?.valueMm??((doc.body?mm(doc.body.waist):null)??760)+design.elasticEaseMm;
    const hem=geometry?.drafting?.measurements.find(row=>row.name==='Hem circumference')?.valueMm??Math.max(((doc.body?mm(doc.body.hip):null)??980)+(mm(doc.garment.ease)??80),waist+80)*design.fullness*doc.garment.flare;
    const scale=220/length,w=waist/4*scale,h=hem/4*scale,d=depth*scale;
    return (['front','back'] as const).map(view=>({view,buttons:[],lines:[{points:[[-w,0],[w,0],[w,d],[h,220],[-h,220],[-w,d],[-w,0]],detail:false},{points:[[-w,d],[w,d]],detail:true},{points:[[0,d],[0,220]],detail:true}]}));
  }
  if(design.block==='panel-dress') {
    const length=mm(doc.garment.length)??1000,scale=220/length;
    const chest=(Math.max(doc.body?mm(doc.body.bust)??920:920,doc.body?mm(doc.body.hip)??980:980)+(mm(doc.garment.ease)??80))/4;
    const waist=chest*design.waistRatio,arm=(doc.body?mm(doc.body.bust)??920:920)/10+110,levels=design.skirtStyle==='tiered'?2:1;
    return (['front','back'] as const).map(view=>{
      const lines:Flat['lines']=[];
      const line=(points:[number,number][],detail=false)=>lines.push({points:points.map(([x,y])=>[x*scale,y*scale]),detail});
      for(const sign of [-1,1]) {
        const depth=view==='front'?design.neckDepthMm:25,w=design.neckWidthMm;
        const source=geometry?.panels.find(p=>p.id===`bodice_${view}_left`),edge=source?.draft?.edges.find(e=>e.name==='neck');
        const neck:[number,number][]=source&&edge?source.points.slice(edge.start,edge.end+1):view==='front'&&design.neckline==='v'?[[w,0],[0,depth]]:view==='front'&&design.neckline==='square'?[[w,0],[w,depth],[0,depth]]:Array.from({length:25},(_,i)=>[w*Math.cos(Math.PI*i/48),depth*Math.sin(Math.PI*i/48)]);
        line(neck.map(([x,y])=>[sign*x,y]));
        line([[sign*w,0],[sign*chest,0],[sign*chest,arm],[sign*waist,design.bodiceLengthMm]]);
        if(design.sleeves!=='none')line([[sign*chest,0],[sign*(chest+design.sleeveLengthMm),40],[sign*(chest+design.sleeveLengthMm),40+arm*.72],[sign*chest,arm]]);
      }
      let previous=waist;
      line([[-waist,design.bodiceLengthMm],[waist,design.bodiceLengthMm]],true);
      for(let tier=0;tier<levels;tier++) {
        const top=previous*(tier?design.tierFullness:design.skirtStyle==='flared'?1:design.skirtFullness),hem=top*doc.garment.flare;
        const y=design.bodiceLengthMm+tier*(length-design.bodiceLengthMm)/levels,end=design.bodiceLengthMm+(tier+1)*(length-design.bodiceLengthMm)/levels;
        for(const sign of [-1,1])line([[sign*previous,y],[sign*hem,end]]);
        line([[-hem,end],[hem,end]],tier<levels-1);
        if(top>previous)for(const f of [-.75,-.4,.4,.75])line([[previous*f,y],[hem*f,end]],true);
        previous=hem;
      }
      if(view==='back')line([[0,25],[0,design.bodiceLengthMm]],true);
      return {view,lines,buttons:[]};
    });
  }
  const length=geometry?.drafting?.measurements.find(row=>row.name==='Side length from shoulder baseline')?.valueMm ?? mm(doc.garment.length) ?? 650;
  const panel=(name:string)=>geometry?.panels.find(piece=>piece.id===name);
  const edge=(name:string,boundary:string)=>{
    const piece=panel(name),interval=piece?.draft?.edges.find(item=>item.name===boundary);
    return piece&&interval?piece.points.slice(interval.start,interval.end+1):null;
  };
  const bust=(doc.body?mm(doc.body.bust):null)??960,hip=(doc.body?mm(doc.body.hip):null)??1000;
  const chest=geometry?.drafting?.measurements.find(row=>row.name==='Closed chest at underarm')?.valueMm ?? Math.max(bust,hip)+(mm(doc.garment.ease)??100);
  const scale=220/length, half=chest/4*scale;
  const neck=(edge('back_left','shoulder')?.at(-1)?.[0]??bust/12)*scale;
  const arm=(panel('front_left')?.draft?.edges.find(boundary=>boundary.name==='armhole')?.lengthMm??bust/10+110)*scale;
  const frontDepth=(edge('front_left','center')?.[0]?.[1]??bust/12)*scale;
  return (['front','back'] as const).map(view=>{
    const lines:Flat['lines']=[],buttons:Flat['buttons']=[];
    const line=(points:[number,number][],detail=false)=>lines.push({points,detail});
    const tail=view==='back'&&design.hem==='curved-back-tail'?design.tailExtensionMm*scale:0;
    const hem=half*doc.garment.flare;
    const sleeve=design.sleeves==='none'?0:design.sleeveLengthMm*scale;
    const cuff=design.cuff==='button'?design.cuffDepthMm*scale:0;
    const wrist=(panel('sleeve_left')?.draft?.edges.find(boundary=>boundary.name==='wrist')?.lengthMm??(cuff?Math.max(design.cuffCircumferenceMm*1.35,arm/scale*1.1):arm/scale*1.6))*scale/2;
    for(const sign of [-1,1]) {
      line([[sign*neck,0],[sign*half,0]]);
      if(sleeve) {
        const end=half+sleeve-cuff;
        line([[sign*half,0],[sign*end,0],[sign*end,wrist],[sign*half,arm]]);
        if(cuff) {
          const cuffWidth=design.cuffCircumferenceMm*scale/2;
          line([[sign*end,0],[sign*(half+sleeve),0],[sign*(half+sleeve),cuffWidth],[sign*end,cuffWidth]],true);
          if(view==='front')buttons.push([sign*(end+cuff/2),cuffWidth/2]);
        }
      } else line([[sign*half,0],[sign*half,arm]]);
      line([[sign*half,arm],[sign*hem,220]]);
    }
    const centerOffset=view==='front'&&design.opening==='buttons'?design.placketWidthMm/2:0;
    const hemPath=edge(`${view}_left`,'hem');
    if(hemPath) {
      for(const sign of [-1,1])line(hemPath.map(([horizontal,vertical])=>[sign*(horizontal+centerOffset)*scale,vertical*scale]));
      if(centerOffset)line([[-centerOffset*scale,220],[centerOffset*scale,220]]);
    } else line(Array.from({length:33},(_,index)=>[-hem+index*hem/16,220+tail*(1+Math.cos(Math.PI*(index-16)/16))/2] as [number,number]));
    const depth=view==='front'?frontDepth:25*scale;
    const neckPath=edge(`${view}_left`,'neck');
    if(neckPath) {
      for(const sign of [-1,1])line(neckPath.map(([horizontal,vertical])=>[sign*(horizontal+centerOffset)*scale,vertical*scale]));
      if(centerOffset)line([[-centerOffset*scale,depth],[centerOffset*scale,depth]],true);
    } else line(Array.from({length:33},(_,index)=>[-neck+index*neck/16,depth*Math.sqrt(Math.max(0,1-((-neck+index*neck/16)/neck)**2))] as [number,number]));
    if(design.sleeves!=='none')for(const sign of [-1,1]) {
      line([[sign*half,0],[sign*half,arm]],true);
    }
    if(view==='front'&&design.opening==='buttons') {
      const band=design.placketWidthMm*scale/2;
      line([[-band,depth],[-band,220]],true);line([[band,depth],[band,220]],true);
      const marks=panel('placket_left')?.draft?.marks.filter(mark=>mark.kind==='button');
      const positions=marks?.map(mark=>mark.point[1])??Array.from({length:Math.max(2,Math.floor((length-frontDepth/scale-50)/design.buttonSpacingMm)+1)},(_,index)=>25+index*design.buttonSpacingMm);
      for(const position of positions)buttons.push([0,depth+position*scale]);
      if(design.frill!=='none')for(const sign of [-1,1])line(Array.from({length:45},(_,index)=>[sign*(band+design.frillWidthMm*scale+(index%2)*2),depth+(220-depth)*index/44] as [number,number]),true);
    }
    if(design.collar!=='none') {
      const stand=design.collarStandMm*scale;
      line([[-neck,0],[-neck,-stand],[neck,-stand],[neck,0]],true);
      if(view==='front')buttons.push([0,-stand/2]);
      if(design.collar==='stand-and-fall') {
        const fall=design.collarFallMm*scale;
        line([[-neck,-stand],[-neck,fall-stand],[neck,fall-stand],[neck,-stand]],true);
        if(view==='front')line([[0,-stand],[0,fall-stand]],true);
      }
    }
    return {view,lines,buttons};
  });
}
