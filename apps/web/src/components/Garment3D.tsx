import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import type { ThreeDJob } from '../../../../packages/contracts/assembly';
import { api, ApiError, json } from '../lib/api';

const GarmentShapePreview=lazy(()=>import('./GarmentShapePreview'));
const GarmentViewport = lazy(() => import('./GarmentViewport'));
const activeJob = (job: ThreeDJob | null) => !!job && ['queued', 'running'].includes(job.status);

export default function Garment3D({ projectId, revisionId, supported, selected, onSelect, onAuthFailure }: {
  projectId: string; revisionId: string | null; supported: boolean; selected: string | null; onSelect: (templateId: string) => void; onAuthFailure: (error: ApiError) => void;
}) {
  const [display,setDisplay]=useState<'garment'|'pieces'>('garment');
  const [job, setJob] = useState<ThreeDJob | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const requestId = useRef<string | null>(null);
  const mounted = useRef(true);
  const submitLock = useRef(false);
  const authFailure = useRef(onAuthFailure);
  authFailure.current = onAuthFailure;
  const path = `/projects/${projectId}/three-d`;
  const fail = (reason: unknown) => {
    if (reason instanceof ApiError && [401, 403, 404].includes(reason.status)) {
      setJob(null); setReady(false);
      if (reason.status === 401) authFailure.current(reason);
    }
    setError(reason instanceof Error ? reason.message : 'Unable to access 3D inspection.');
  };
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => {
    if (!revisionId || !supported) { setReady(true); return; }
    let live = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      let interval = 1500;
      try {
        const latest = await api<ThreeDJob | null>(`${path}/latest?revisionId=${encodeURIComponent(revisionId)}`);
        if (!live) return;
        setJob(latest); setReady(true); setError('');
        if (latest&&!activeJob(latest)) interval = 10000;
      } catch (reason) {
        if (!live) return;
        if (reason instanceof ApiError && [401, 403, 404].includes(reason.status)) {
          setJob(null); setReady(false);
          if (reason.status === 401) authFailure.current(reason);
        }
        setError(reason instanceof Error ? reason.message : 'Unable to check 3D progress.');
      }
      if (live) timer = setTimeout(poll, interval);
    };
    void poll();
    return () => { live = false; clearTimeout(timer); };
  }, [path, revisionId, supported, refresh]);

  const create = async () => {
    if (submitLock.current || !revisionId) return;
    submitLock.current = true; setBusy(true); setError('');
    requestId.current ??= crypto.randomUUID();
    try {
      const next = await api<ThreeDJob>(path, json('POST', { revisionId, requestId: requestId.current }));
      if (!mounted.current) return;
      setJob(next); requestId.current = null; setRefresh(value => value + 1);
    } catch (reason) {
      if (mounted.current) fail(reason);
    } finally { submitLock.current = false; if (mounted.current) setBusy(false); }
  };
  const cancel = async () => {
    if (!job || submitLock.current) return;
    submitLock.current = true; setBusy(true);
    try {
      const next = await api<ThreeDJob>(`${path}/${job.id}/cancel`, json('POST', {}));
      if (mounted.current) { setJob(next); setRefresh(value => value + 1); }
    } catch (reason) { if (mounted.current) fail(reason); }
    finally { submitLock.current = false; if (mounted.current) setBusy(false); }
  };

  return <section className="garment-three-d" aria-label="Pattern-derived garment preview">
    {!supported || !revisionId ? <div className="canvas-empty"><h3>Your garment will appear here.</h3><p>Generate a shirt, relaxed dress or elastic-waist skirt to see its pattern-derived shape.</p></div> : <>
      {job?.status === 'succeeded' && job.result ? <>
        {job.sourceCurrent === false && <div className="geometry-stale">A newer pattern or engine is available. <button disabled={busy} onClick={() => void create()}>Update preview</button></div>}
        <div className="garment-display-toggle"><button aria-pressed={display==='garment'} onClick={()=>setDisplay('garment')}>Garment</button><button aria-pressed={display==='pieces'} onClick={()=>setDisplay('pieces')}>Flat pieces</button></div>
        <Suspense fallback={<p role="status">Opening your garment…</p>}>
          {display==='garment'&&job.result.shape?<GarmentShapePreview key={job.id} path={`${path}/${job.id}/shape`} sha256={job.result.shape.sha256} patternDigest={job.patternDigest} color="#eeeae1" onSelect={onSelect} onUnavailable={fail}/>:<>
            {!job.result.shape&&<p>This older result contains flat pieces. <button disabled={busy} onClick={()=>void create()}>Build garment preview</button></p>}
            <GarmentViewport key={job.id} path={`${path}/${job.id}/mesh`} patternDigest={job.patternDigest} selected={selected} onSelect={onSelect} onUnavailable={fail}/>
          </>}
        </Suspense>
        <details className="three-d-diagnostics"><summary>Pattern & generation details</summary>
          <p>{job.result.fabricInstances} fabric pieces · saved revision {job.revisionId.slice(0,8)}.</p>
          <p>The garment is an approximate display pose. Source dimensions are unchanged; physical fit and drape are unverified.</p>
          <ul>{job.result.capabilityGaps.map((gap,index)=><li key={index}>{gap}</li>)}</ul>
          <p>Pattern: <code>{job.patternDigest.slice(0,12)}</code>. This private 3D view is not included in pattern downloads.</p>
        </details>
      </> : <div className="three-d-start">
        <h3>{activeJob(job)||!job?'Bringing your garment to life…':'Preview needs another try.'}</h3>
        <p>{activeJob(job)||!job?'Your pattern is ready. We’re shaping its pieces into a garment.':'Your saved design and pattern are still available.'}</p>
        {activeJob(job)?<div role="status"><p>{job?.status==='queued'?'Waiting for the garment engine.':'Shaping fabric pieces…'}</p><button disabled={busy} onClick={()=>void cancel()}>Cancel preview</button></div>:job?<button className="primary" disabled={busy||!ready} onClick={()=>void create()}>Retry garment preview</button>:<p role="status">The preview starts automatically.</p>}
        {job&&['failed','cancelled','stale'].includes(job.status)&&<p role="status">{job.status==='failed'?job.error:job.status==='cancelled'?'Preview cancelled.':'This attempt was replaced by newer inputs.'}</p>}
      </div>}
    </>}
    {error&&<div className="alert error" role="alert">{error} <button onClick={()=>setRefresh(value=>value+1)}>Check again</button></div>}
  </section>;
}
