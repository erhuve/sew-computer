import { useState } from 'react';
import {requiredBodyFields} from '../../../../packages/contracts/sizing';
import { LoaderCircle, Sparkles } from 'lucide-react';
import { canonical, type GarmentDocument, type Measurement } from '../../../../packages/contracts';
import type { DesignProposal, InterpretationStatus, InterpretationJob } from '../../../../packages/contracts/interpretation';
import GarmentDesign from './GarmentDesign';

const measurement=(value:Measurement)=>'value' in value?`${value.value} ${value.unit} (${value.state})`:value.state;
export default function DesignAssistant({doc,status,proposal,busy,stale,onPropose,onAccept,onMeasurements,job,onCancel}:{
  doc:GarmentDocument;status:InterpretationStatus|null;proposal:DesignProposal|null;busy:boolean;stale:boolean;
  job:InterpretationJob|null;onCancel:()=>void;
  onPropose:(includeReferences:boolean)=>void;onAccept:(samplePreview:boolean)=>void;onMeasurements:()=>void;
}) {
  const [useSample,setUseSample]=useState(true);
  const [consent,setConsent]=useState(false),[images,setImages]=useState(false);
  const missing=requiredBodyFields(doc).filter(field=>!('value' in doc.body[field.key])).map(field=>field.label);
  const needsPatternSupport = !proposal && !!doc.interpretation && doc.garment.family === 'none';
  const interpreting=job?.status==='queued'||job?.status==='running';
  return <section className="design-assistant" aria-label="Design assistant">
    <div className="assistant-heading"><Sparkles size={18}/><strong>From idea to pattern</strong></div>
    {interpreting&&<div role="status"><p>{job.status==='queued'?'Design interpretation queued.':'Interpreting your design…'} You can continue editing or reopen this project later. Changed drafts require a fresh proposal before acceptance.</p><button disabled={busy} onClick={onCancel}>Cancel interpretation</button></div>}
    {job?.status==='failed'&&<p role="alert">{job.error}</p>}
    {job?.status==='cancelled'&&<p role="status">Interpretation cancelled. Your draft is unchanged.</p>}
    {proposal?<>
      <h2>How does this look?</h2>
      <p>{proposal.summary}</p>
      <GarmentDesign doc={proposal.document}/>
      {proposal.document.garment.family!=='none'&&proposal.document.requirements.some(row=>row.status==='unsupported')&&<p className="proposal-limit" role="status">Not in this pattern yet: {proposal.document.requirements.filter(row=>row.status==='unsupported').map(row=>row.text).join('; ')}.</p>}
      <dl className="proposal-shape">
        <div><dt>Shape</dt><dd>{proposal.document.garment.family}</dd></div>
        <div><dt>Length</dt><dd>{measurement(proposal.document.garment.length)}</dd></div>
        <div><dt>Ease</dt><dd>{measurement(proposal.document.garment.ease)}</dd></div>
        <div><dt>Flare</dt><dd>{proposal.document.garment.flare}</dd></div>
      </dl>
      {canonical(doc.garment)!==canonical(proposal.document.garment)&&<details><summary>Current shape → proposed shape</summary><p>Current: {doc.garment.family}, length {measurement(doc.garment.length)}, ease {measurement(doc.garment.ease)}, flare {doc.garment.flare}.</p></details>}
      <details><summary>Design details & open requirements</summary><div className="proposal-requirements">{proposal.document.requirements.filter(row=>!doc.requirements.some(prior=>prior.id===row.id)).map(row=><article key={row.id}><strong>{row.text}</strong><span>{row.status}</span><p>{row.note}</p></article>)}</div></details>
      {!!proposal.questions.length&&<details open><summary>Decisions to resolve</summary><ul>{proposal.questions.map((question,index)=><li key={index}>{question}</li>)}</ul></details>}
      <details><summary>Materials, measurements & sewing notes</summary>
        {proposal.document.bom.filter(row=>!doc.bom.some(prior=>prior.id===row.id)).map(row=><p key={row.id}><strong>{row.name}</strong> — {row.specification}. {row.placement}</p>)}
        {proposal.document.poms.filter(row=>!doc.poms.some(prior=>prior.id===row.id)).map(row=><p key={row.id}><strong>{row.name}</strong> — {row.method}. Target: {measurement(row.target)}. Tolerance: {measurement(row.tolerance)}.</p>)}
        {proposal.document.construction.filter(row=>!doc.construction.some(prior=>prior.id===row.id)).map(row=><p key={row.id}><strong>{row.operation}</strong> — {row.note}</p>)}
      </details>
      <p className="fineprint">AI design suggestion · fit and sewing unverified.</p>
      {stale&&<p role="status">You’ve edited the draft since this proposal. Request a new one to keep those edits.</p>}
      {proposal.document.garment.family === 'none' && <p role="status">This proposal saves design notes only. No pattern shape is available for this design; accepting it will not enable pattern generation.</p>}
      {proposal.document.garment.family!=='none'&&<label className="check-field"><input type="checkbox" checked={useSample} onChange={event=>setUseSample(event.target.checked)}/>Preview with sample M estimates. Keeps measurements you’ve entered.</label>}
      <button className="primary" disabled={busy||stale} onClick={()=>onAccept(useSample&&proposal.document.garment.family!=='none')}>{proposal.document.garment.family==='none'?'Accept design notes':useSample?'Use design & preview':'Use design & set size'}</button>
    </>:needsPatternSupport?<>
      <h2>Design saved. Pattern support needed.</h2>
      <p>Your idea and technical notes are saved, but no pattern shape was selected. Adding measurements alone won’t make this design generate.</p>
      <details><summary>Details the engine cannot make</summary><ul>{doc.requirements.filter(row=>row.status==='unsupported').map(row=><li key={row.id}>{row.text}</li>)}</ul></details>
      <p>You can revise the idea, export the design notes for a maker, or explicitly choose a simplified base. Editable starting shapes include shirts, relaxed dresses and elastic-waist skirts; they won’t include your unsupported details.</p>
      <button onClick={onMeasurements} disabled={busy}>Choose a simplified base pattern</button>
    </>:!interpreting&&<p>Describe your garment, then turn it into an editable design.</p>}
    <details open={!proposal && !needsPatternSupport && !interpreting}>
      <summary>{proposal?'Request another proposal':'AI input & privacy'}</summary>
      <details><summary>What gets sent & usage</summary><p className="fineprint">{status?.available?`${status.provider} · ${status.model}`:'The server’s design model is not connected yet.'} Sends your brief and design notes. Body fields and size label are excluded; personal information typed into your brief will still be sent. One request, two-minute timeout, 12 requests/hour. {status?.maxOutputTokens?'Up to 6,000 output tokens; provider charges may apply.':'Uses your Codex allowance; 32 KB answer limit. No provider-side token or cost cap is available on this connection.'}</p>
      <p className="fineprint">If the server restarts during a request, it may retry once. Up to two model attempts may count toward provider usage.</p></details>
      <label className="check-field"><input type="checkbox" checked={images} disabled={busy||doc.views.length===0||doc.views.length>3} onChange={event=>setImages(event.target.checked)}/>Include my reference images ({doc.views.length}/3 maximum)</label>
      <label className="check-field"><input type="checkbox" checked={consent} disabled={busy} onChange={event=>setConsent(event.target.checked)}/>Send these inputs to the design model</label>
      <button className="primary" disabled={busy||interpreting||!consent||!status?.available||!doc.brief.trim()} onClick={()=>onPropose(images)}>{busy||interpreting?<LoaderCircle size={16} className="spin"/>:<Sparkles size={16}/>} {busy||interpreting?'Working…':'Interpret my design'}</button>
    </details>
    {!proposal&&doc.garment.family!=='none'&&<button disabled={busy} onClick={onMeasurements}>{missing.length?`Add ${missing.length} body measurements`:'Customize this design'}</button>}
  </section>;
}
