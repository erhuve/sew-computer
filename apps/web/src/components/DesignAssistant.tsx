import { useState } from 'react';
import { LoaderCircle, Sparkles } from 'lucide-react';
import { canonical, type GarmentDocument, type Measurement } from '../../../../packages/contracts';
import type { DesignProposal, InterpretationStatus } from '../../../../packages/contracts/interpretation';

const measurement=(value:Measurement)=>'value' in value?`${value.value} ${value.unit} (${value.state})`:value.state;
export default function DesignAssistant({doc,status,proposal,busy,stale,onPropose,onAccept,onMeasurements}:{
  doc:GarmentDocument;status:InterpretationStatus|null;proposal:DesignProposal|null;busy:boolean;stale:boolean;
  onPropose:(includeReferences:boolean)=>void;onAccept:()=>void;onMeasurements:()=>void;
}) {
  const [consent,setConsent]=useState(false),[images,setImages]=useState(false);
  const missing=Object.entries(doc.body).filter(([,value])=>!('value' in value)).map(([name])=>name);
  return <section className="design-assistant" aria-label="Design assistant">
    <div className="assistant-heading"><Sparkles size={18}/><strong>From idea to pattern</strong></div>
    {proposal?<>
      <p>{proposal.summary}</p>
      <dl className="proposal-shape">
        <div><dt>Shape</dt><dd>{proposal.document.garment.family}</dd></div>
        <div><dt>Length</dt><dd>{measurement(proposal.document.garment.length)}</dd></div>
        <div><dt>Ease</dt><dd>{measurement(proposal.document.garment.ease)}</dd></div>
        <div><dt>Flare</dt><dd>{proposal.document.garment.flare}</dd></div>
      </dl>
      {canonical(doc.garment)!==canonical(proposal.document.garment)&&<details><summary>Current shape → proposed shape</summary><p>Current: {doc.garment.family}, length {measurement(doc.garment.length)}, ease {measurement(doc.garment.ease)}, flare {doc.garment.flare}.</p></details>}
      <div className="proposal-requirements">{proposal.document.requirements.filter(row=>!doc.requirements.some(prior=>prior.id===row.id)).map(row=><article key={row.id}><strong>{row.text}</strong><span>{row.status}</span><p>{row.note}</p></article>)}</div>
      {!!proposal.questions.length&&<details open><summary>Decisions to resolve</summary><ul>{proposal.questions.map((question,index)=><li key={index}>{question}</li>)}</ul></details>}
      <details><summary>Materials, measurements & sewing notes</summary>
        {proposal.document.bom.filter(row=>!doc.bom.some(prior=>prior.id===row.id)).map(row=><p key={row.id}><strong>{row.name}</strong> — {row.specification}. {row.placement}</p>)}
        {proposal.document.poms.filter(row=>!doc.poms.some(prior=>prior.id===row.id)).map(row=><p key={row.id}><strong>{row.name}</strong> — {row.method}. Target: {measurement(row.target)}. Tolerance: {measurement(row.tolerance)}.</p>)}
        {proposal.document.construction.filter(row=>!doc.construction.some(prior=>prior.id===row.id)).map(row=><p key={row.id}><strong>{row.operation}</strong> — {row.note}</p>)}
      </details>
      <p className="fineprint">AI suggestions, not verified sewing instructions. Accepting saves a new revision. Your body inputs, references and existing technical notes are preserved.</p>
      {stale&&<p role="status">You’ve edited the draft since this proposal. Request a new one to keep those edits.</p>}
      <button className="primary" disabled={busy||stale} onClick={onAccept}>Accept design & set measurements</button>
    </>:<p>Describe your garment in “The idea”, then get an editable design, materials and construction notes.</p>}
    <details open={!proposal}>
      <summary>{proposal?'Request another proposal':'AI input & privacy'}</summary>
      <p className="fineprint">{status?.available?`${status.provider} · ${status.model}`:'The server’s design model is not connected yet.'} Sends your brief and design notes. Body fields and size label are excluded; personal information typed into your brief will still be sent. One request, two-minute timeout, 12 requests/hour. {status?.maxOutputTokens?'Up to 6,000 output tokens; provider charges may apply.':'Uses your Codex allowance; 32 KB answer limit. No provider-side token or cost cap is available on this connection.'}</p>
      <label className="check-field"><input type="checkbox" checked={images} disabled={busy||doc.views.length===0||doc.views.length>3} onChange={event=>setImages(event.target.checked)}/>Include my reference images ({doc.views.length}/3 maximum)</label>
      <label className="check-field"><input type="checkbox" checked={consent} disabled={busy} onChange={event=>setConsent(event.target.checked)}/>Send these inputs to the design model</label>
      <button className="primary" disabled={busy||!consent||!status?.available||!doc.brief.trim()} onClick={()=>onPropose(images)}>{busy?<LoaderCircle size={16} className="spin"/>:<Sparkles size={16}/>} {busy?'Working…':'Interpret my design'}</button>
    </details>
    {doc.garment.family!=='none'&&<button disabled={busy} onClick={onMeasurements}>{missing.length?`Add ${missing.length} body measurements`:'Review measurements & generate'}</button>}
  </section>;
}
