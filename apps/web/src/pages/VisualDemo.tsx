import { useEffect, useState } from 'react';
import { ArrowDownToLine, ArrowLeft, Scissors } from 'lucide-react';
import type { PatternGeometry } from '../../../../packages/contracts';
import type { ShirtDesign } from '../../../../packages/contracts/design';
import { inspectPrivateGlb } from '../../../../packages/contracts/inspection-display';
import { buildGarmentLayout, type GarmentLayout } from '../lib/garment-layout';
import GarmentViewport from '../components/GarmentViewport';
import DemoGarmentDrawing from '../components/DemoGarmentDrawing';
import GarmentShapePreview from '../components/GarmentShapePreview';
import { downloadJson } from '../lib/api';
import './visual-demo.css';

type Variant = { id: string; design: ShirtDesign; patternDigest: string; pieces: number; templates: number; files: Record<string, { sha256: string; bytes: number }> };
type Catalog = { profile: string; synthetic: boolean; variants: Variant[] };
const swatches = [{ name: 'Chalk', color: '#e6e1d5' }, { name: 'Sage', color: '#a4b5a0' }, { name: 'Ink', color: '#536575' }, { name: 'Clay', color: '#c08c78' }];
async function hash(bytes: ArrayBuffer) { return [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(value => value.toString(16).padStart(2, '0')).join(''); }

export default function VisualDemo() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [sleeves, setSleeves] = useState('long');
  const [frill, setFrill] = useState(true);
  const [fabric, setFabric] = useState(swatches[0]!);
  const [selected, setSelected] = useState<string | null>(null);
  const [loaded, setLoaded] = useState<{ variant: Variant; pattern: PatternGeometry; layout: GarmentLayout } | null>(null);
  const [error, setError] = useState('');
  const [feedback, setFeedback] = useState('');
  const [previewMode, setPreviewMode] = useState<'shape' | 'design' | 'pieces'>('shape');
  const id = `${sleeves}-${frill ? 'frill' : 'clean'}`;
  const ready = loaded?.variant.id === id ? loaded : null;
  useEffect(() => {
    const abort = new AbortController();
    fetch('/demo-fixtures/catalog.json', { signal: abort.signal }).then(async response => {
      if (!response.ok) throw new Error('The demo examples could not be opened.');
      const value = await response.json() as Catalog;
      if (value.profile !== 'sew-visual-demo/1' || !value.synthetic || value.variants.length !== 4) throw new Error('Unsupported demo examples.');
      setCatalog(value);
    }).catch(reason => { if (!abort.signal.aborted) setError(String(reason.message || reason)); });
    return () => abort.abort();
  }, []);
  useEffect(() => {
    if (!catalog) return;
    const abort = new AbortController();
    setError(''); setSelected(null);
    const load = async () => {
      const variant = catalog.variants.find(row => row.id === id);
      if (!variant || !/^(long|short)-(frill|clean)$/.test(variant.id)) throw new Error('This example is unavailable.');
      const [patternResponse, meshResponse] = await Promise.all([
        fetch(`/demo-fixtures/${id}.json`, { signal: abort.signal }), fetch(`/demo-fixtures/${id}.glb`, { signal: abort.signal }),
      ]);
      if (!patternResponse.ok || !meshResponse.ok) throw new Error('The matching pattern and preview could not be opened.');
      const [patternBytes, meshBytes] = await Promise.all([patternResponse.arrayBuffer(), meshResponse.arrayBuffer()]);
      if (await hash(patternBytes) !== variant.patternDigest || await hash(meshBytes) !== variant.files[`${id}.glb`]?.sha256) throw new Error('The example files do not match their saved revision.');
      const metadata = inspectPrivateGlb(meshBytes);
      if (metadata.extras.patternDigest !== variant.patternDigest) throw new Error('The preview does not match its pattern.');
      const pattern = JSON.parse(new TextDecoder().decode(patternBytes)) as PatternGeometry;
      const layout = buildGarmentLayout(pattern, metadata.meshes.map(mesh => mesh.extras));
      if (!abort.signal.aborted) setLoaded({ variant, pattern, layout });
    };
    void load().catch(reason => { if (!abort.signal.aborted) setError(String(reason.message || reason)); });
    return () => abort.abort();
  }, [catalog, id]);
  const panel = ready?.pattern.panels.find(piece => piece.id === selected);
  return <div className="visual-demo">
    <header className="demo-header"><a href="/" className="demo-brand"><Scissors aria-hidden="true" size={27} />sew.</a><span>DESIGN STUDIO <span aria-hidden="true">/</span> INTERACTIVE DEMO</span><a href="/" className="demo-studio-link"><ArrowLeft size={15} />Open studio</a></header>
    <main>
      <div className="demo-intro"><div><p className="demo-kicker">FROM YOUR IDEA TO THE PIECES THAT MAKE IT</p><h1>Make it your own.</h1><p>Explore a relaxed shirt. Change the details, inspect the pieces, and take the pattern with you.</p></div><span className="demo-example-label">Example design · synthetic measurements</span></div>
      <div className="demo-workspace">
        <aside className="demo-choices" aria-label="Design choices">
          <p className="demo-kicker">01 / YOUR DESIGN</p><h2>The everyday shirt</h2><p>A relaxed cut, button front and a longer curved back hem.</p>
          <fieldset><legend>Sleeves</legend><div className="demo-segmented">{['long', 'short'].map(value => <button key={value} aria-pressed={sleeves === value} onClick={() => setSleeves(value)}>{value === 'long' ? 'Long + cuff' : 'Short'}</button>)}</div></fieldset>
          <fieldset><legend>Front detail</legend><div className="demo-segmented"><button aria-pressed={!frill} onClick={() => setFrill(false)}>Clean</button><button aria-pressed={frill} onClick={() => setFrill(true)}>Frill</button></div></fieldset>
          <fieldset><legend>Preview color <span>display only</span></legend><div className="demo-swatches">{swatches.map(value => <button key={value.name} aria-label={value.name} aria-pressed={fabric.name === value.name} onClick={() => setFabric(value)} style={{ background: value.color }} />)}</div><p className="demo-color-name">{fabric.name}</p></fieldset>
          <div className="demo-pattern-action"><p className="demo-kicker">02 / TAKE IT FURTHER</p><h3>Actual pattern pieces.</h3><p>These examples were drafted from the selected construction choices. Each download matches the preview.</p>{ready ? <><a className="demo-download" href={`/demo-fixtures/${id}.svg`} download={`${id}-draft-pattern.svg`}><ArrowDownToLine size={17} />Download draft pattern</a><a className="demo-json" href={`/demo-fixtures/${id}.json`} download={`${id}-pattern.json`}>Pattern data · JSON</a></> : <span role="status">Preparing your selection…</span>}<small>Draft reference. Check fit and print scale before making.</small></div>
        </aside>
        <section className="demo-preview" aria-label="Garment preview">
          <div className="demo-preview-heading"><div><p className="demo-kicker">YOUR DESIGN, CONNECTED TO ITS PATTERN</p><h2>See your choices take shape.</h2></div><span className="demo-status">{previewMode === 'shape' ? 'Approximate garment shape' : previewMode === 'design' ? 'Construction preview' : 'Pattern arrangement'}</span></div>
          <div className="demo-view-tabs" role="tablist" aria-label="Preview type"><button role="tab" aria-selected={previewMode === 'shape'} onClick={() => setPreviewMode('shape')}>Garment preview</button><button role="tab" aria-selected={previewMode === 'design'} onClick={() => setPreviewMode('design')}>Design preview</button><button role="tab" aria-selected={previewMode === 'pieces'} onClick={() => setPreviewMode('pieces')}>3D pieces</button></div>
          {previewMode === 'pieces' && <p className="demo-scope">Real pattern pieces in a rough garment arrangement. Drape, gathering and stitched assembly are still in progress.</p>}
          {error ? <div role="alert" className="demo-error">{error}<button onClick={() => location.reload()}>Reload demo</button></div> : ready ? previewMode === 'shape' ? <GarmentShapePreview id={id} patternDigest={ready.variant.patternDigest} color={fabric.color} onSelect={setSelected} /> : previewMode === 'design' ? <DemoGarmentDrawing pattern={ready.pattern} design={ready.variant.design} color={fabric.color} /> : <GarmentViewport key={id} path={`/demo-fixtures/${id}.glb`} demoAsset patternDigest={ready.variant.patternDigest} layout={ready.layout} fabricColor={fabric.color} selected={selected} onSelect={setSelected} onUnavailable={reason => setError(reason.message)} /> : <div className="demo-loading" role="status">Opening the matching pattern pieces…</div>}
          <div className="demo-preview-footer"><span>{ready ? `${ready.variant.pieces} fabric pieces · ${ready.variant.templates} source templates` : 'Loading example'}</span><span>{previewMode !== 'pieces' ? 'Switch between front and back' : 'Drag to rotate · click a piece to inspect'}</span></div>
        </section>
      </div>
      <section className="demo-source-row" aria-label="Selected source pattern"><div><p className="demo-kicker">03 / UNDERSTAND THE PATTERN</p><h2>{panel ? panel.name : 'Every piece has a source.'}</h2><p>{panel ? `${panel.widthMm.toFixed(0)} × ${panel.heightMm.toFixed(0)} mm · cut ${panel.cutQuantity} · ${panel.draft?.component}` : 'Choose a piece in the 3D view to see its original 2D outline. Switching to Flat pieces shows the full inventory.'}</p>{panel && <p className="demo-scope">Solid line: cutting outline. Dashed line: stitching boundary.</p>}</div><div className="demo-source-canvas">{panel ? <svg viewBox={`-18 -18 ${panel.widthMm + 36} ${panel.heightMm + 36}`} role="img" aria-label={`${panel.name} original pattern`}><polygon points={(panel.draft?.cutLine ?? panel.points).map(point => point.join(',')).join(' ')} fill="#e5ebe1" stroke="#344c40" strokeWidth="1.4" /><polyline points={panel.points.map(point => point.join(',')).join(' ')} fill="none" stroke="#344c40" strokeWidth=".9" strokeDasharray="5 4" /></svg> : <span>Select a physical piece above</span>}</div></section>
      <section className="demo-feedback"><div><p className="demo-kicker">HELP SHAPE WHAT COMES NEXT</p><h2>What would you make with this?</h2><p>What felt useful? What was missing? What would stop you using it again?</p></div><div><label htmlFor="demo-feedback">Demo notes</label><textarea id="demo-feedback" value={feedback} onChange={event => setFeedback(event.target.value)} placeholder="Your idea, first impressions, or questions…" /><button disabled={!feedback.trim()} onClick={() => downloadJson('sew-demo-feedback.json', { example: id, notes: feedback, preview: 'approximate rigid arrangement', recordedAt: new Date().toISOString() })}>Save feedback notes</button><small>Notes stay in this page until you download them.</small></div></section>
      <footer className="demo-footnote">Four pre-generated examples show the current component compiler. This demo does not run AI or regenerate arbitrary designs. The studio remains the place to develop your own design. 3D views omit seam allowances and interfacing.</footer>
    </main>
  </div>;
}
