import { useEffect, useId, useState } from 'react';
import { CircleAlert, LoaderCircle, RotateCcw, Sparkles } from 'lucide-react';
import { canonical, type GarmentDocument, type Measurement } from '../../../../packages/contracts';
import type { DesignProposal, InterpretationStatus, InterpretationJob } from '../../../../packages/contracts/interpretation';
import GarmentDesign, {ConstructionDrawing} from './GarmentDesign';

const measurement=(value:Measurement)=>'value' in value?`${value.value} ${value.unit} (${value.state})`:value.state;
export default function DesignAssistant({doc,status,proposal,busy,stale,onPropose,onAccept,onMeasurements,job,onCancel,editing,onChange,onSimplify}:{
  doc:GarmentDocument;status:InterpretationStatus|null;proposal:DesignProposal|null;busy:boolean;stale:boolean;editing:boolean;
  job:InterpretationJob|null;onCancel:()=>void;onChange:(doc:GarmentDocument)=>void;
  onPropose:(includeReferences:boolean)=>void;onAccept:(samplePreview:boolean)=>void;onMeasurements:()=>void;
  onSimplify:(family:'shirt'|'dress'|'skirt')=>void;
}) {
  const [useSample,setUseSample]=useState(true),[images,setImages]=useState(false);
  const promptLabelId=useId();
  const needsPatternSupport = !proposal && job?.status!=='failed' && !!doc.interpretation && doc.garment.family === 'none';
  const interpreting=job?.status==='queued'||job?.status==='running';
  const [clock,setClock]=useState(Date.now);
  useEffect(()=>{
    if(!interpreting)return;
    setClock(Date.now());
    const timer=setInterval(()=>setClock(Date.now()),1000);
    return()=>clearInterval(timer);
  },[interpreting,job?.id]);
  const elapsed=Math.max(0,Math.floor((clock-Date.parse(job?.startedAt??job?.createdAt??new Date(clock).toISOString()))/1000));
  const phaseLabel=job?.status==='queued'?'Your idea is next.':job?.phase==='writing'?'Writing your design…':job?.phase==='validating'?'Checking your design…':job?.phase==='thinking'?'Interpreting your idea…':'Connecting to the design model…';
  const failed=job?.status==='failed';
  const shapeAvailable=proposal?proposal.document.garment.family!=='none':doc.garment.family!=='none';
  const simplifiedChoices=<div className="simplified-choices">
    <p>This shape isn’t available yet. You can preview a simpler version; the original details stay saved.</p>
    <div>{(['shirt','dress','skirt'] as const).map(family=><button key={family} disabled={busy||!!proposal&&stale} onClick={()=>onSimplify(family)}>Preview as {family==='shirt'?'a relaxed shirt':family==='dress'?'a relaxed dress':'an elastic-waist skirt'}</button>)}</div>
  </div>;
  const requestForm=<>
    {!editing&&<label className="field prompt-field"><span id={promptLabelId}>Describe your garment</span><textarea aria-labelledby={promptLabelId} value={doc.brief} maxLength={8000} onChange={event=>onChange({...doc,brief:event.target.value})} placeholder="A loose linen dress in deep blue, with short sleeves…"/></label>}
    <button className="primary" disabled={busy||interpreting||!status?.available||!doc.brief.trim()} onClick={()=>onPropose(images)}>{busy||interpreting?<LoaderCircle size={16} className="spin"/>:failed?<RotateCcw size={16}/>:<Sparkles size={16}/>} {busy||interpreting?'Working…':failed?'Try again':'Interpret my design'}</button>
    <p className="fineprint">Sends your description and design notes to {status?.provider??'your connected model'}. Uses your connected model allowance. You review the result before it is applied.</p>
    <details><summary>References & AI privacy</summary>
      <label className="check-field"><input type="checkbox" checked={images} disabled={busy||doc.views.length===0||doc.views.length>3} onChange={event=>setImages(event.target.checked)}/>Include my reference images ({doc.views.length}/3 maximum)</label>
      <p>{status?.available?`${status.provider} · ${status.model}`:'The design model is not connected yet.'} Body fields and size label are excluded; personal information typed into your brief is sent. One request, up to {status?.timeoutSeconds??120} seconds, 12 requests/hour. {status?.maxOutputTokens?'Up to 6,000 output tokens; provider charges may apply.':'32 KB answer limit. No provider-side token or cost cap is available on this connection.'}</p>
      <p>If the server restarts during a request, it may retry once. Up to two model attempts may count toward usage.</p>
    </details>
  </>;
  return <section className={`design-assistant ${interpreting?'is-interpreting':''}`} aria-label="Design assistant">
    {interpreting?<div className="idea-progress"><LoaderCircle size={32} className="spin"/><div role="status"><h2>{phaseLabel}</h2></div><span className="interpretation-elapsed" aria-label="Time spent on this attempt">{Math.floor(elapsed/60)}:{String(elapsed%60).padStart(2,'0')} elapsed</span><p>{elapsed>=45?'Still working. Your prompt is saved, and you can leave this page and come back.':'We’ll choose the construction details. You can review the design before seeing it in 3D.'}</p><blockquote>{doc.brief}</blockquote><button disabled={busy} onClick={onCancel}>Cancel interpretation</button></div>:<>
      {failed&&<div className="interpretation-error" role="alert"><CircleAlert size={20}/><div><strong>Your design didn’t finish</strong><p>{job.error}</p><p>Your prompt is saved. Try again when you’re ready.</p></div></div>}
      {job?.status==='cancelled'&&<p role="status">Interpretation cancelled. Your draft is unchanged.</p>}
      {proposal?<>
        <div className="assistant-heading"><Sparkles size={16}/><span>Your proposed design</span></div>
        <h2>Ready to see it?</h2>
        <p className="proposal-summary">{proposal.summary}</p>
        {proposal.document.garment.design&&<ConstructionDrawing doc={proposal.document}/>}
        {shapeAvailable&&proposal.document.requirements.some(row=>row.status==='unsupported')&&<p className="proposal-limit">Not included in this preview: {proposal.document.requirements.filter(row=>row.status==='unsupported').map(row=>row.text).join('; ')}.</p>}
        {stale&&<p role="status">The design changed since this proposal. Request an updated design below to include your edits.</p>}
        {shapeAvailable?<>
          <button className="primary preview-action" disabled={busy||stale} onClick={()=>onAccept(useSample)}>{useSample?'See garment':'Use design & set size'} <Sparkles size={16}/></button>
          <p className="preview-assumption">{useSample?'Sample sizing fills missing measurements. Your entered measurements stay.':'Uses your own sizing.'} Approximate preview · fit unverified.</p>
        </>:<>{simplifiedChoices}<button disabled={busy||stale} onClick={()=>onAccept(false)}>Accept design notes</button><p className="fineprint">Saving notes alone does not create a garment preview.</p></>}
        <details className="proposal-details" open={editing}>
          <summary>Options & design details</summary>
          {shapeAvailable&&<label className="check-field"><input type="checkbox" checked={useSample} onChange={event=>setUseSample(event.target.checked)}/>Preview with sample M estimates. Keeps measurements you’ve entered.</label>}
          <dl className="proposal-shape"><div><dt>Shape</dt><dd>{proposal.document.garment.family}</dd></div><div><dt>Length</dt><dd>{measurement(proposal.document.garment.length)}</dd></div><div><dt>Ease</dt><dd>{measurement(proposal.document.garment.ease)}</dd></div><div><dt>Flare</dt><dd>{proposal.document.garment.flare}</dd></div></dl>
          {canonical(doc.garment)!==canonical(proposal.document.garment)&&<p>Current: {doc.garment.family}, length {measurement(doc.garment.length)}, ease {measurement(doc.garment.ease)}, flare {doc.garment.flare}.</p>}
          <GarmentDesign doc={proposal.document} compact/>
          <details><summary>Design details & open requirements</summary><div className="proposal-requirements">{proposal.document.requirements.filter(row=>!doc.requirements.some(prior=>prior.id===row.id)).map(row=><article key={row.id}><strong>{row.text}</strong><span>{row.status}</span><p>{row.note}</p></article>)}</div></details>
          {!!proposal.questions.length&&<details><summary>Decisions to resolve</summary><ul>{proposal.questions.map((question,index)=><li key={index}>{question}</li>)}</ul></details>}
          <details><summary>Materials, measurements & sewing notes</summary>
            {proposal.document.bom.filter(row=>!doc.bom.some(prior=>prior.id===row.id)).map(row=><p key={row.id}><strong>{row.name}</strong> — {row.specification}. {row.placement}</p>)}
            {proposal.document.poms.filter(row=>!doc.poms.some(prior=>prior.id===row.id)).map(row=><p key={row.id}><strong>{row.name}</strong> — {row.method}. Target: {measurement(row.target)}. Tolerance: {measurement(row.tolerance)}.</p>)}
            {proposal.document.construction.filter(row=>!doc.construction.some(prior=>prior.id===row.id)).map(row=><p key={row.id}><strong>{row.operation}</strong> — {row.note}</p>)}
          </details>
        </details>
        <details className="refine-design" open={stale}><summary>Request another proposal</summary>{requestForm}</details>
      </>:needsPatternSupport?<>
        <h2>Design saved. Pattern support needed.</h2>{simplifiedChoices}
        <details><summary>Details the engine cannot make</summary><ul>{doc.requirements.filter(row=>row.status==='unsupported').map(row=><li key={row.id}>{row.text}</li>)}</ul></details>
        <details><summary>Revise the idea</summary>{requestForm}</details>
        <button onClick={onMeasurements} disabled={busy}>Choose a simplified base pattern</button>
      </>:<><h2>{doc.interpretation?'Make another version.':'What are you imagining?'}</h2>{requestForm}</>}
    </>}
  </section>;
}
