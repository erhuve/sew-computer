import {useEffect,useRef,useState} from 'react';
import type { GarmentDocument } from '../../../../packages/contracts';
import { defaultFabricColor, type FabricPrint } from '../../../../packages/contracts/appearance';
import {api} from '../lib/api';
const palette = [
  ['Ivory', '#eeeae1'], ['Black', '#242326'], ['Navy', '#233953'],
  ['Cobalt', '#315dd4'], ['Forest', '#385b45'], ['Sage', '#a1b49a'],
  ['Wine', '#812f46'], ['Terracotta', '#ba654c'], ['Butter', '#e8ce73'],
  ['Lilac', '#b4a0cb'], ['Rose', '#d79aa8'], ['White', '#ffffff'],
] as const;
export default function FabricAppearance({projectId,doc,onChange,onError}:{projectId:string;doc:GarmentDocument;onChange:(doc:GarmentDocument)=>void;onError:(error:string)=>void}) {
  const [uploading,setUploading]=useState(false);
  const live=useRef(true);useEffect(()=>{live.current=true;return()=>{live.current=false;};},[]);
  const latest=useRef({doc,onChange});latest.current={doc,onChange};
  const color=doc.garment.appearance?.color ?? defaultFabricColor,print=doc.garment.appearance?.print;
  const change=(next:Partial<NonNullable<GarmentDocument['garment']['appearance']>>)=>onChange({...doc,garment:{...doc.garment,appearance:{color,...doc.garment.appearance,...next}}});
  const updatePrint=(next:Partial<FabricPrint>)=>change({print:{kind:'stripes',inkColor:'#233953',tileMm:60,rotationDeg:0,assetId:null,...print,...next}});
  return <section className="fabric-appearance" aria-label="Fabric appearance">
    <h3>Color & print</h3>
    <div className="fabric-swatches" role="group" aria-label="Fabric colors">{palette.map(([name,value])=><button key={name} type="button" aria-label={`${name} fabric`} aria-pressed={color.toLowerCase()===value} style={{backgroundColor:value}} title={name} onClick={()=>change({color:value})}/>)}</div>
    <label className="custom-fabric-color"><span>Base color</span><input aria-label="Custom fabric color" type="color" value={color} onChange={event=>change({color:event.target.value})}/><span>{color.toUpperCase()}</span></label>
    <label className="field"><span>Fabric print</span><select aria-label="Fabric print" value={print?.kind??'solid'} onChange={event=>event.target.value==='solid'?change({print:null}):updatePrint({kind:event.target.value as FabricPrint['kind']})}>
      <option value="solid">Solid</option><option value="stripes">Stripes</option><option value="checks">Checks</option><option value="dots">Dots</option><option value="image">Your artwork</option>
    </select></label>
    {print&&<>
      {print.kind==='image'?<label className="upload-button">{uploading?'Processing artwork…':print.assetId?'Replace artwork':'Upload artwork'}<input aria-label="Upload fabric artwork" type="file" accept="image/png,image/jpeg,image/webp" disabled={uploading||doc.views.length>=20} onChange={async event=>{
        const file=event.target.files?.[0];event.target.value='';if(!file)return;setUploading(true);
        try{if(file.size>10*1024*1024)throw new Error('Choose an image under 10 MB.');
          const asset=await api<{assetId:string}>(`/projects/${projectId}/references`,{method:'POST',headers:{'Content-Type':file.type,'X-Filename':encodeURIComponent(file.name)},body:file});
          if(!live.current)return;const current=latest.current.doc;if(current.garment.appearance?.print?.kind!=='image')return;if(current.views.length>=20)throw new Error('This design already has 20 references. Remove one before adding artwork.');
          latest.current.onChange({...current,views:[...current.views,{id:crypto.randomUUID(),assetId:asset.assetId,role:'detail',kind:'sketch',caption:'Fabric artwork'}],garment:{...current.garment,appearance:{color:current.garment.appearance?.color??defaultFabricColor,print:{...current.garment.appearance.print,kind:'image',assetId:asset.assetId}}}});
        }catch(error){onError((error as Error).message);}finally{setUploading(false);}
      }}/></label>:<label className="custom-fabric-color"><span>Print color</span><input aria-label="Print color" type="color" value={print.inkColor} onChange={event=>updatePrint({inkColor:event.target.value})}/></label>}
      <label className="field"><span>Repeat size · {print.tileMm} mm</span><input aria-label="Print repeat size" type="range" min="5" max="500" step="5" value={print.tileMm} onChange={event=>updatePrint({tileMm:Number(event.target.value)})}/></label>
      <label className="field"><span>Print rotation · {print.rotationDeg}°</span><input aria-label="Print rotation" type="range" min="-180" max="180" step="5" value={print.rotationDeg} onChange={event=>updatePrint({rotationDeg:Number(event.target.value)})}/></label>
      {print.kind==='image'&&<p>Upload any PNG, JPEG or WebP artwork. It repeats over each piece at the chosen size. Artwork stays private; include reference images when exporting to share it.</p>}
      <p>Prints follow the flat pattern coordinates. Matching motifs across seams is not automatic.</p>
    </>}
  </section>;
}
