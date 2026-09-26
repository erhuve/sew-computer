import { useEffect, useRef, useState } from 'react';
import { mm, type GarmentDocument } from '../../../../packages/contracts';
import { applySample, bodyFields, requiredBodyFields, editMeasurement, sampleBody, sampleSizes, sizingIssue, sizingWarning, type Body, type BodyKey, type BodyProfile } from '../../../../packages/contracts/sizing';
import { api, json } from '../lib/api';

type Unit = 'cm' | 'in';
const scale = { cm: 10, in: 25.4 };
const display = (value: number, unit: Unit) => Number((value / scale[unit]).toFixed(2));

function BodyDiagram({ body, active }: { body: Body; active: BodyKey }) {
  const fallback = sampleBody(2);
  const dimension = (key: BodyKey) => mm(body[key]) ?? mm(fallback[key])!;
  const height = Math.max(1200, Math.min(2200, dimension('height')));
  const proportion = (key: BodyKey, circumference = false) => Math.max(8, Math.min(96, dimension(key) / (circumference ? Math.PI : 1) / height * 320 / 2));
  const shoulder = proportion('shoulder');
  const bust = proportion('bust', true);
  const waist = proportion('waist', true);
  const hip = proportion('hip', true);
  const bottom = 100 + height / 2200 * 270;
  const lines: Record<BodyKey, [number, number, number, number]> = {
    height: [38, 30, 38, bottom], shoulder: [160 - shoulder, 98, 160 + shoulder, 98],
    bust: [160 - bust, 137, 160 + bust, 137], waist: [160 - waist, 180, 160 + waist, 180], hip: [160 - hip, 222, 160 + hip, 222],
  };
  const [startX, startY, endX, endY] = lines[active];
  const incomplete = bodyFields.some(field => mm(body[field.key]) === null);
  return <figure className="body-diagram">
    <div className="diagram-label"><span className="eyebrow">Proportion guide</span><span>Illustration · not fit simulation</span></div>
    <svg viewBox="0 0 320 400" role="img" aria-label={`Approximate body diagram highlighting ${active}. Circumferences do not determine actual body shape.${incomplete ? ' Missing dimensions use a dashed illustrative placeholder.' : ''}`}>
      <path className="diagram-axis" d="M160 20V380 M25 222H295" />
      <g className="body-outline" strokeDasharray={incomplete ? '5 4' : undefined}>
        <ellipse cx="160" cy="54" rx="22" ry="25" />
        <path d={`M146 78 L146 90 L${160-shoulder} 98 L${160-shoulder-22} 190 L${160-shoulder-16} 199 L${160-bust} 137 Q${160-waist-5} 160 ${160-waist} 180 Q${160-hip} 205 ${160-hip} 222 L${160-hip+8} ${bottom} L150 ${bottom} L160 242 L170 ${bottom} L${160+hip-8} ${bottom} L${160+hip} 222 Q${160+hip} 205 ${160+waist} 180 Q${160+waist+5} 160 ${160+bust} 137 L${160+shoulder+16} 199 L${160+shoulder+22} 190 L${160+shoulder} 98 L174 90 L174 78`} />
      </g>
      {bodyFields.filter(field => field.key !== 'height').map(field => {
        const [firstX, firstY, lastX, lastY] = lines[field.key];
        return <line key={field.key} className="diagram-guide" x1={firstX} y1={firstY} x2={lastX} y2={lastY} />;
      })}
      <g className="diagram-active"><line x1={startX} y1={startY} x2={endX} y2={endY} /><circle cx={startX} cy={startY} r="4" /><circle cx={endX} cy={endY} r="4" /></g>
    </svg>
    <figcaption><strong>{bodyFields.find(field => field.key === active)!.label}</strong><span>{bodyFields.find(field => field.key === active)!.method}</span><small>{incomplete ? 'Dashed = missing values. Illustration only.' : 'Approximate proportions, not exact anatomy.'} Not to scale.</small></figcaption>
  </figure>;
}

export default function BodySizing({ doc, onChange }: { doc: GarmentDocument; onChange: (doc: GarmentDocument) => void }) {
  const fields=requiredBodyFields(doc);
  const [editing,setEditing]=useState(()=>fields.some(field=>mm(doc.body[field.key])===null));
  const [unit, setUnit] = useState<Unit>('cm');
  const [active, setActive] = useState<BodyKey>(fields[0]!.key);
  useEffect(()=>{if(!fields.some(field=>field.key===active))setActive(fields[0]!.key);},[doc.garment.design?.block,active]);
  const [sample, setSample] = useState(2);
  const [profile, setProfile] = useState<BodyProfile | null>(null);
  const [pendingProfile, setPendingProfile] = useState<Body | null>(null);
  const [profileBusy, setProfileBusy] = useState(false);
  const [message, setMessage] = useState('');
  const mounted = useRef(false);
  const requestNumber = useRef(0);
  const loadProfile = async () => {
    const request = ++requestNumber.current;
    setProfileBusy(true);
    try { const result = await api<BodyProfile>('/body-profile'); if (mounted.current && request === requestNumber.current) { setProfile(result); setMessage(''); setPendingProfile(null); } }
    catch (error) { if (mounted.current && request === requestNumber.current) setMessage((error as Error).message); }
    finally { if (mounted.current && request === requestNumber.current) setProfileBusy(false); }
  };
  useEffect(() => { mounted.current = true; void loadProfile(); return () => { mounted.current = false; requestNumber.current++; }; }, []);
  const saveProfile = async (body: Body | null) => {
    if (!profile || profileBusy) return;
    setProfileBusy(true);
    try {
      const result = await api<BodyProfile>('/body-profile', json('PUT', { expectedVersion: profile.version, body }));
      if (mounted.current) { setProfile(result); setPendingProfile(null); setMessage(body ? 'Measurements saved privately for reuse.' : 'Reusable copy deleted. Existing garment revisions keep their measurements.'); }
    } catch (error) { if (mounted.current) setMessage((error as Error).message); }
    finally { if (mounted.current) setProfileBusy(false); }
  };
  const issue = sizingIssue(doc);
  const warning = sizingWarning(doc);
  const sampleValues = sampleSizes[sample]!;
  return <section className="body-sizing" aria-label="Visual body sizing">
    <div className="sizing-topline"><h2 className="body-heading" tabIndex={-1}>Make it your size.</h2><div className="sizing-units" role="group" aria-label="Display units">{(['cm', 'in'] as const).map(choice => <button key={choice} aria-pressed={choice === unit} onClick={() => setUnit(choice)}>{choice === 'in' ? 'Inches' : 'cm'}</button>)}</div></div>
    <div className="sample-picker">
      <label>Sample size<select aria-label="Sample size" value={sample} onChange={event => setSample(Number(event.target.value))}>{sampleSizes.map((size, index) => <option key={size.label} value={index}>{size.label}</option>)}</select></label>
      <div><strong>Bust {display(sampleValues.values[1], unit)} · Waist {display(sampleValues.values[2], unit)} · Hip {display(sampleValues.values[3], unit)} {unit}</strong><p>Replaces estimates; keeps entered measurements.</p><details><summary>About sample sizes</summary><p>Synthetic starting sizes, not a standard size chart. Fills blanks, including missing garment length and ease. N/A stays unchanged.</p><p>Height {display(sampleValues.values[0], unit)} · Shoulder {display(sampleValues.values[4], unit)} {unit}</p></details></div>
      <button onClick={() => onChange(applySample(doc, sample))}>Apply sample {sampleValues.label}</button>
    </div>
    <details className="body-adjustments" open={editing} onToggle={event=>setEditing(event.currentTarget.open)}><summary>Enter or adjust my measurements</summary><div className="sizing-workbench">
      <BodyDiagram body={doc.body} active={active} />
      <div className="sizing-controls">{fields.map(field => {
        const measurement = doc.body[field.key];
        const amount = mm(measurement);
        const shown = amount === null ? '' : 'value' in measurement && measurement.unit === unit ? measurement.value : display(amount, unit);
        const outside = amount !== null && (amount < field.min || amount > field.max);
        return <fieldset key={field.key} className={`sizing-measure${active === field.key ? ' active' : ''}`} onFocus={() => setActive(field.key)}>
          <legend>{field.label}{['bust', 'waist', 'hip'].includes(field.key) ? ' · around' : field.key === 'shoulder' ? ' · across back' : ''}</legend>
          <div className="sizing-value"><input type="range" aria-label={`${field.label} slider`} aria-valuetext={amount === null ? 'Not entered; choose a sample or enter a number' : `${shown} ${unit}`} disabled={amount === null || outside} min={field.min} max={field.max} step="1" value={amount ?? field.min} onChange={event => onChange({ ...doc, body: { ...doc.body, [field.key]: editMeasurement(measurement, Number(event.target.value), 'mm') } })} />
          <input type="number" aria-label={`${field.label} value`} value={shown} min="0" max="20000" step="any" placeholder="—" onChange={event => {
            const amount = event.target.valueAsNumber;
            if (event.target.value === '') onChange({ ...doc, body: { ...doc.body, [field.key]: { state: 'unknown' } } });
            else if (Number.isFinite(amount) && amount >= 0 && amount <= 20000) onChange({ ...doc, body: { ...doc.body, [field.key]: editMeasurement(measurement, amount, unit) } });
          }} /><span>{unit}</span></div>
          <div className="sizing-state"><span>{measurement.state === 'assumed' ? 'Estimate · not measured' : measurement.state === 'known' ? 'Entered by you' : measurement.state === 'not-applicable' ? 'Not applicable' : 'Not entered'}</span>{measurement.state === 'assumed' && <button onClick={() => onChange({ ...doc, body: { ...doc.body, [field.key]: { ...measurement, state: 'known', source: 'Owner confirmed this value as measured' } } })}>Confirm measured {field.label.toLowerCase()}</button>}</div>
          {outside && <p className="sizing-warning">Outside this engine’s {display(field.min, unit)}–{display(field.max, unit)} {unit} range. Your value is kept; edit the number to change it.</p>}
        </fieldset>;
      })}</div>
    </div>
    </details>{issue && <p className="sizing-warning" role="status">{issue}</p>}
    {warning && <p className="sizing-warning" role="status">{warning}</p>}
    <details className="profile-settings"><summary>My saved measurements</summary><p>One private reusable profile, stored on your server. Applying it replaces all five body fields after review; it never changes garment settings.</p>
      <div className="profile-actions"><button disabled={!profile || profileBusy || !bodyFields.some(field => mm(doc.body[field.key]) !== null)} onClick={() => void saveProfile(structuredClone(doc.body))}>{profile?.body ? 'Replace saved measurements with current values' : 'Save current measurements'}</button><button disabled={!profile?.body || profileBusy} onClick={() => setPendingProfile(structuredClone(profile!.body!))}>Review saved measurements</button><button disabled={profileBusy} onClick={() => void loadProfile()}>Reload profile</button><button disabled={!profile?.body || profileBusy} onClick={() => void saveProfile(null)}>Delete saved measurements</button></div>
      {pendingProfile && <div className="profile-preview"><dl>{bodyFields.map(field => <div key={field.key}><dt>{field.label}</dt><dd>{mm(pendingProfile[field.key]) === null ? pendingProfile[field.key].state : `${display(mm(pendingProfile[field.key])!, unit)} ${unit} · ${pendingProfile[field.key].state}`}</dd></div>)}</dl><button onClick={() => { onChange({ ...doc, body: structuredClone(pendingProfile) }); setPendingProfile(null); setMessage('Saved measurements applied to this draft.'); }}>Replace this draft’s body measurements</button><button onClick={() => setPendingProfile(null)}>Cancel</button></div>}
      {message && <p role="status">{message}</p>}
      <p>Deleting the reusable copy does not remove measurements already saved in garments.</p>
    </details>
  </section>;
}
