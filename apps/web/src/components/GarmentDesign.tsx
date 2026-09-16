import type { GarmentDocument, PatternGeometry } from '../../../../packages/contracts';
import { designCoverage, designIssues, ShirtDesignSchema, type ShirtDesign } from '../../../../packages/contracts/design';
import { garmentFlats } from '../../../../packages/contracts/flats';

export default function GarmentDesign({doc,geometry=null,onChange}:{doc:GarmentDocument;geometry?:PatternGeometry|null;onChange?:(doc:GarmentDocument)=>void}) {
  const design=doc.garment.design;
  if(!design)return null;
  const change=(key:keyof ShirtDesign,value:unknown)=>{
    const next=ShirtDesignSchema.safeParse({...design,[key]:value,...(key==='sleeves'&&value==='short'?{sleeveLengthMm:220,cuff:'none'}:key==='sleeves'&&value==='long'?{sleeveLengthMm:550}:key==='sleeves'&&value==='none'?{cuff:'none'}:{})});
    if(next.success)onChange?.({...doc,garment:{...doc.garment,design:next.data}});
  };
  const choices:[keyof ShirtDesign,string,string[]][]=[['sleeves','Sleeves',['none','short','long']],['cuff','Cuffs',['none','button']],['collar','Collar',['none','stand','stand-and-fall']],['opening','Front opening',['none','buttons']],['hem','Hem',['straight','curved-back-tail']],['frill','Frills',['none','front-opening']]];
  const numbers:[keyof ShirtDesign,string,number,number,number][]=[['sleeveLengthMm','Sleeve length incl. cuff (mm)',100,800,5],['cuffCircumferenceMm','Closed cuff (mm)',160,400,5],['cuffDepthMm','Cuff depth (mm)',30,100,5],['collarStandMm','Collar stand (mm)',20,50,1],['collarFallMm','Collar fall (mm)',30,100,5],['placketWidthMm','Placket width (mm)',20,45,1],['buttonSpacingMm','Button spacing (mm)',50,110,5],['tailExtensionMm','Back tail extension (mm)',30,200,5],['frillWidthMm','Frill width (mm)',15,100,5],['frillFullness','Gathering fullness',1.25,3,0.05],['seamAllowanceMm','Seam allowance (mm)',6,20,1]];
  const coverage=designCoverage(doc,geometry);
  return <section className="garment-design" aria-label="Garment construction">
    <div className="garment-flats">{garmentFlats(doc,geometry).map(flat=>{const extent=Math.max(...flat.lines.flatMap(line=>line.points.map(point=>Math.abs(point[0]))))+20;return <figure key={flat.view}>
      <svg viewBox={`${-extent} -30 ${extent*2} 380`} role="img" aria-label={`${flat.view} construction schematic`}>
        {flat.lines.map((line,index)=><polyline key={index} points={line.points.map(point=>point.join(',')).join(' ')} fill="none" stroke="currentColor" strokeWidth={line.detail?1.2:2} />)}
        {flat.buttons.map((point,index)=><circle key={index} cx={point[0]} cy={point[1]} r={2} fill="currentColor"/>)}
      </svg><figcaption>{flat.view} · construction schematic</figcaption>
    </figure>;})}</div>
    <p>{design.rationale}</p>
    <p className="fineprint">Relaxed drop-shoulder construction. Schematic views explain selected features; they are not drape or fit simulations. Dimensions are editable design assumptions.</p>
    {geometry&&<p role="status">{coverage.status==='drafted'?'Selected components drafted; physical fit unverified.':'Partial design: some requirements remain unimplemented.'}</p>}
    {geometry&&!!coverage.unresolved.length&&<details><summary>Requirements still needing review ({coverage.unresolved.length})</summary><ul>{coverage.unresolved.map((text,index)=><li key={index}>{text}</li>)}</ul></details>}
    {onChange&&<details><summary>Edit construction choices</summary><div className="construction-controls">
      {choices.map(([key,label,options])=><label key={key}>{label}<select value={String(design[key])} onChange={event=>change(key,event.target.value)}>{options.map(value=><option key={value}>{value}</option>)}</select></label>)}
      {numbers.map(([key,label,min,max,step])=><label key={key}>{label}<input type="range" min={min} max={max} step={step} value={Number(design[key])} onChange={event=>change(key,Number(event.target.value))}/><output>{Number(design[key])}</output></label>)}
    </div></details>}
    {designIssues(design).map(issue=><p role="alert" key={issue}>{issue}</p>)}
    {geometry?.drafting&&<details><summary>Derived specifications & assembly</summary>
      <dl>{geometry.drafting.measurements.map(row=><div key={row.name}><dt>{row.name}</dt><dd>{row.valueMm.toFixed(1)} mm — {row.method}</dd></div>)}</dl>
      <ul>{geometry.drafting.materials.map(row=><li key={row}>{row}</li>)}</ul>
      <ol>{geometry.drafting.operations.map((row,index)=><li key={index}>{row}</li>)}</ol>
    </details>}
  </section>;
}
