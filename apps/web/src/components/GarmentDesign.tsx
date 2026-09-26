import {mm,type GarmentDocument,type PatternGeometry} from '../../../../packages/contracts';
import {designCoverage,designIssues,GarmentDesignSchema,type ShirtDesign,type SkirtDesign} from '../../../../packages/contracts/design';
import {editMeasurement} from '../../../../packages/contracts/sizing';
import {garmentFlats} from '../../../../packages/contracts/flats';

export function ConstructionDrawing({doc,geometry=null,single=false}:{doc:GarmentDocument;geometry?:PatternGeometry|null;single?:boolean}) {
  return <div className="garment-flats">{garmentFlats(doc,geometry).slice(0,single?1:2).map(flat=>{
    const all=flat.lines.flatMap(line=>line.points),extent=Math.max(...all.map(point=>Math.abs(point[0])))+20;
    const ymin=Math.min(...all.map(point=>point[1]))-15,ymax=Math.max(...all.map(point=>point[1]))+15;
    return <figure key={flat.view}><svg viewBox={`${-extent} ${ymin} ${extent*2} ${ymax-ymin}`} role="img" aria-label={`${flat.view} construction schematic`}>
      {flat.lines.map((line,index)=><polyline key={index} points={line.points.map(point=>point.join(',')).join(' ')} fill="none" stroke="currentColor" strokeWidth={line.detail?1.2:2} />)}
      {flat.buttons.map((point,index)=><circle key={index} cx={point[0]} cy={point[1]} r={2} fill="currentColor"/>)}
    </svg>{!single&&<figcaption>{flat.view} · construction schematic</figcaption>}</figure>;
  })}</div>;
}

export default function GarmentDesign({doc,geometry=null,onChange,compact=false}:{doc:GarmentDocument;geometry?:PatternGeometry|null;onChange?:(doc:GarmentDocument)=>void;compact?:boolean}) {
  const design=doc.garment.design;
  if(!design)return null;
  const change=(key:string,value:unknown)=>{
    const next=GarmentDesignSchema.safeParse({...design,[key]:value,...(key==='opening'&&value==='none'?{collar:'none',frill:'none'}:{}),...(key==='sleeves'&&value==='short'?{sleeveLengthMm:220,cuff:'none'}:key==='sleeves'&&value==='long'?{sleeveLengthMm:550}:key==='sleeves'&&value==='none'?{cuff:'none'}:{})});
    if(next.success)onChange?.({...doc,garment:{...doc.garment,design:next.data}});
  };
  const choices:[keyof ShirtDesign,string,string[]][]=[['sleeves','Sleeves',['none','short','long']],['cuff','Cuffs',['none','button']],['collar','Collar',['none','stand','stand-and-fall']],['opening','Front opening',['none','buttons']],['hem','Hem',['straight','curved-back-tail']],['frill','Frills',['none','front-opening']]];
  const shirtNumbers:[keyof ShirtDesign,string,number,number,number][]=[['sleeveLengthMm','Sleeve length incl. cuff (mm)',100,800,5],['cuffCircumferenceMm','Closed cuff (mm)',160,400,5],['cuffDepthMm','Cuff depth (mm)',30,100,5],['collarStandMm','Collar stand (mm)',20,50,1],['collarFallMm','Collar fall (mm)',30,100,5],['placketWidthMm','Placket width (mm)',20,45,1],['buttonSpacingMm','Button spacing (mm)',50,110,5],['tailExtensionMm','Back tail extension (mm)',30,200,5],['frillWidthMm','Frill width (mm)',15,100,5],['frillFullness','Gathering fullness',1.25,3,0.05],['seamAllowanceMm','Seam allowance (mm)',6,20,1]];
  const skirtNumbers:[keyof SkirtDesign,string,number,number,number][]=[['waistbandDepthMm','Waistband depth (mm)',25,60,1],['elasticEaseMm','Elastic ease (mm)',-60,40,5],['seamAllowanceMm','Seam allowance (mm)',6,20,1]];
  const choiceNames:Record<string,string>={none:'None',short:'Short',long:'Long',button:'Button cuff',buttons:'Buttons',stand:'Stand collar','stand-and-fall':'Classic collar',straight:'Straight','curved-back-tail':'Curved tail','front-opening':'Front ruffle'};
  const coverage=designCoverage(doc,geometry);
  const isSkirt=design.block==='elastic-waist-skirt';
  return <section className="garment-design" aria-label="Garment construction">
    {!compact&&<ConstructionDrawing doc={doc} geometry={geometry}/>}
    {!compact&&<details><summary>Design notes</summary><p>{design.rationale}</p></details>}
    {geometry&&coverage.status!=='drafted'&&<p role="status">Some requested details are not in this pattern yet.</p>}
    {geometry&&!!coverage.unresolved.length&&<details><summary>Requirements still needing review ({coverage.unresolved.length})</summary><ul>{coverage.unresolved.map((text,index)=><li key={index}>{text}</li>)}</ul></details>}
    {onChange&&<div><h3>Make it yours</h3><div className="construction-controls">
      <label className="wide-control">Length <output>{((mm(doc.garment.length)??650)/10).toFixed(1)} cm</output><input aria-label="Length (cm)" type="range" min={doc.garment.family==='dress'?70:doc.garment.family==='shirt'?40:45} max={doc.garment.family==='dress'?145:doc.garment.family==='shirt'?110:130} step="1" value={(mm(doc.garment.length)??650)/10} onChange={event=>onChange({...doc,garment:{...doc.garment,length:editMeasurement(doc.garment.length,Number(event.target.value)*10,'mm')}})}/></label>
      <label className="wide-control">Hem sweep <output>{doc.garment.flare.toFixed(2)}×</output><input aria-label="Hem sweep" type="range" min={doc.garment.family==='shirt'?.9:1} max={doc.garment.family==='dress'?2:isSkirt?1.8:1.5} step=".01" value={doc.garment.flare} onChange={event=>onChange({...doc,garment:{...doc.garment,flare:Number(event.target.value)}})}/></label>
      {design.block==='elastic-waist-skirt'?<label className="wide-control">Gathering <output>{design.fullness.toFixed(2)}×</output><input aria-label="Gathering fullness" type="range" min="1" max="1.8" step=".05" value={design.fullness} onChange={event=>change('fullness',Number(event.target.value))}/></label>:choices.filter(([key])=>(key!=='cuff'||design.sleeves==='long')&&(key!=='frill'&&key!=='collar'||design.opening==='buttons')).map(([key,label,options])=><label key={key}>{label}<select aria-label={label} value={String(design[key])} onChange={event=>change(key,event.target.value)}>{options.map(value=><option key={value} value={value}>{choiceNames[value]??value}</option>)}</select></label>)}
    </div><details><summary>Fine-tune dimensions</summary><div className="construction-controls">{(isSkirt?skirtNumbers:shirtNumbers).map(([key,label,min,max,step])=><label key={key}>{label}<input aria-label={label} type="range" min={min} max={max} step={step} value={Number((design as unknown as Record<string,unknown>)[key])} onChange={event=>change(key,Number(event.target.value))}/><output>{Number((design as unknown as Record<string,unknown>)[key])}</output></label>)}
    </div></details></div>}
    {designIssues(design).map(issue=><p role="alert" key={issue}>{issue}</p>)}
    {geometry?.drafting&&<details><summary>Derived specifications & assembly</summary><dl>{geometry.drafting.measurements.map(row=><div key={row.name}><dt>{row.name}</dt><dd>{row.valueMm.toFixed(1)} mm — {row.method}</dd></div>)}</dl><ul>{geometry.drafting.materials.map(row=><li key={row}>{row}</li>)}</ul><ol>{geometry.drafting.operations.map((row,index)=><li key={index}>{row}</li>)}</ol></details>}
  </section>;
}
