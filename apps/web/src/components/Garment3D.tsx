import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import type { ThreeDJob } from '../../../../packages/contracts/assembly';
import { api, ApiError, json } from '../lib/api';

const GarmentViewport = lazy(() => import('./GarmentViewport'));
const activeJob = (job: ThreeDJob | null) => !!job && ['queued', 'running'].includes(job.status);

export default function Garment3D({ projectId, revisionId, supported, selected, onSelect, onAuthFailure }: {
  projectId: string; revisionId: string | null; supported: boolean; selected: string | null; onSelect: (templateId: string) => void; onAuthFailure: (error: ApiError) => void;
}) {
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
        if (!activeJob(latest)) interval = 10000;
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

  return <section className="garment-three-d" aria-label="Pattern-derived 3D inspection">
    <div className="three-d-heading"><div><span className="eyebrow">Actual pattern geometry</span><h2>3D inspection</h2></div><span className="three-d-evidence">Placement only</span></div>
    <p className="three-d-disclosure">Flat fabric pieces at their pattern dimensions. Assembly and fabric drape are not available yet.</p>
    {!supported || !revisionId ? <div className="canvas-empty"><h3>Generate a component shirt first.</h3><p>3D inspection uses the saved shirt pattern. Your 2D patterns and exports remain available for other garments.</p></div> : <>
      {job?.status === 'succeeded' && job.result ? <>
        {job.sourceCurrent === false && <div className="geometry-stale">This inspection belongs to an earlier pattern generation or engine version. <button disabled={busy} onClick={() => void create()}>Rebuild 3D inspection</button></div>}
        <Suspense fallback={<p role="status">Loading 3D viewer…</p>}><GarmentViewport key={job.id} path={`${path}/${job.id}/mesh`} patternDigest={job.patternDigest} selected={selected} onSelect={onSelect} onUnavailable={fail}/></Suspense>
        <div className="three-d-summary"><span>{job.result.fabricInstances} fabric pieces</span><span>Seam allowances omitted</span></div>
        <details className="three-d-diagnostics"><summary>Assembly gaps & provenance</summary>
          <p>{job.result.unresolvedPhysicalRoles} unresolved physical roles. These pieces have not been sewn together by the engine.</p>
          <ul>{job.result.capabilityGaps.map((gap, index) => <li key={index}>{gap}</li>)}</ul>
          <p>Saved revision: <code>{job.revisionId.slice(0, 8)}</code>. Pattern: <code>{job.patternDigest.slice(0, 12)}</code>.</p>
          <p>This private view reveals pattern dimensions. It is not included in exported review packages.</p>
        </details>
      </> : <div className="three-d-start">
        <h3>{activeJob(job) ? 'Preparing your pattern pieces…' : 'Inspect every physical fabric piece.'}</h3>
        <p>Rotate, inspect the mesh, and select pieces to find their source in the 2D pattern.</p>
        {activeJob(job) ? <div role="status"><p>{job?.status === 'queued' ? 'Waiting for the 3D engine.' : 'Building source-linked meshes. You can leave this view and return.'}</p><button disabled={busy} onClick={() => void cancel()}>Cancel 3D job</button></div> : <button className="primary" disabled={busy || !ready} onClick={() => void create()}>{busy ? 'Starting…' : job ? 'Retry 3D inspection' : 'Build 3D inspection'}</button>}
        {job && ['failed', 'cancelled', 'stale'].includes(job.status) && <p role="status">{job.status === 'failed' ? `3D inspection failed: ${job.error || 'The engine did not produce a validated result.'}` : job.status === 'cancelled' ? '3D inspection cancelled.' : 'This 3D attempt is no longer current.'} Your 2D pattern is unchanged.</p>}
      </div>}
    </>}
    {error && <div className="alert error" role="alert">{error} <button onClick={() => setRefresh(value => value + 1)}>Check again</button></div>}
  </section>;
}
