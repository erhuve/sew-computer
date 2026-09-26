import { Plus, Trash2, Upload, ArrowUpRight } from "lucide-react";
import { useState } from "react";
import {
  assumed,
  convert,
  mm,
  type GarmentDocument,
  type Measurement,
  type ReviewComment,
} from "../../../../packages/contracts";
import { api } from "../lib/api";
import PrivateImage from "./PrivateImage";
import BodySizing from "./BodySizing";
import { sizingIssue } from '../../../../packages/contracts/sizing';
import { defaultShirtDesign, defaultDressDesign, defaultSkirtDesign } from '../../../../packages/contracts/design';
import GarmentDesign from './GarmentDesign';

const uid = () => crypto.randomUUID();
type Props = {
  doc: GarmentDocument;
  onChange: (doc: GarmentDocument) => void;
  onReference: (reference: GarmentDocument["views"][number]) => void;
  projectId: string;
  section: string;
  onError: (message: string) => void;
  comments: ReviewComment[];
  anchor: string | null;
};
export function Measure({
  label,
  value,
  onChange,
  compact = false,
  range,
}: {
  label: string;
  value: Measurement;
  onChange: (m: Measurement) => void;
  compact?: boolean;
  range?: [number, number];
}) {
  const [emptyUnit, setEmptyUnit] = useState<'mm' | 'cm' | 'in'>('mm');
  return (
    <fieldset className={`measurement${compact ? ' compact-measurement' : ''}`}>
      <legend>{label}</legend>
      {range && <input type="range" aria-label={`${label} slider`} min={range[0]} max={range[1]} step="1" disabled={mm(value) === null || mm(value)! < range[0] || mm(value)! > range[1]} value={mm(value) ?? range[0]} onChange={event => onChange({ state: value.state === 'assumed' ? 'assumed' : 'known', value: Number(event.target.value), unit: 'mm', source: value.state === 'assumed' ? 'Owner-adjusted garment estimate' : 'Chosen by owner' })} />}
      <div className="measure-row">
        {!compact && <select
          aria-label={`${label} status`}
          value={value.state}
          onChange={(e) => {
            const state = e.target.value as Measurement["state"];
            onChange(
              state === "unknown" || state === "not-applicable"
                ? { state }
                : {
                    state,
                    value: "value" in value ? value.value : 0,
                    unit: "value" in value ? value.unit : "mm",
                    source:
                      "source" in value ? value.source : "Entered by owner",
                  },
            );
          }}
        >
          <option value="unknown">Unknown</option>
          <option value="known">Entered</option>
          <option value="assumed">Assumed</option>
          <option value="not-applicable">N/A</option>
        </select>}
        {(
          <>
            <input
              type="number"
              min="0"
              max="20000"
              step="any"
              aria-label={`${label} value`}
              value={'value' in value ? value.value : ''}
              placeholder="Enter value"
              onChange={(e) => {
                const n = e.target.valueAsNumber;
                if (e.target.value === '') {
                  if ('value' in value) setEmptyUnit(value.unit);
                  onChange({state:'unknown'});
                }
                else if (Number.isFinite(n)) onChange('value' in value ? { ...value, value: n } : {state:'known',value:n,unit:emptyUnit,source:'Entered by owner'});
              }}
            />
            <select
              aria-label={`${label} unit`}
              value={'value' in value ? value.unit : emptyUnit}
              onChange={(e) => {
                const unit = e.target.value as 'mm' | 'cm' | 'in';
                setEmptyUnit(unit);
                if ('value' in value) onChange(convert(value, unit));
              }}
            >
              <option>mm</option>
              <option>cm</option>
              <option>in</option>
            </select>
          </>
        )}
      </div>
      {compact && <details className="measurement-details"><summary>{value.state === 'known' ? 'Entered by you' : value.state === 'assumed' ? 'Assumed value — review' : value.state === 'not-applicable' ? 'Not applicable' : 'Not entered'}</summary><Measure label={`${label} detail`} value={value} onChange={onChange}/></details>}
      {!compact && "source" in value && (
        <input
          aria-label={`${label} source`}
          placeholder="Where did this value come from?"
          value={value.source}
          maxLength={300}
          onChange={(e) => onChange({ ...value, source: e.target.value })}
        />
      )}
    </fieldset>
  );
}
export default function Authoring({
  doc,
  onChange,
  onReference,
  projectId,
  section,
  onError,
  comments,
  anchor,
}: Props) {
  const [uploading, setUploading] = useState(false);
  const [sizeOpen,setSizeOpen]=useState(()=>!!sizingIssue(doc));
  const componentSkirt = doc.garment.design?.block === 'elastic-waist-skirt';
  const flareRange = doc.garment.family === 'dress' ? [1, 2] : componentSkirt ? [1, 1.8] : doc.garment.family === 'skirt' ? [0.5, 2] : [doc.garment.design ? 0.9 : 0.7, doc.garment.family === 'trousers' ? 1.2 : 1.5];
  const replace = <K extends keyof GarmentDocument>(
    key: K,
    value: GarmentDocument[K],
  ) => onChange({ ...doc, [key]: value });
  const collections = {
    "Add a requirement": "requirements",
    "Add material or trim": "bom",
    "Add point of measure": "poms",
    "Add construction step": "construction",
    "Add a callout": "callouts",
  } as const;
  const addButton = (label: keyof typeof collections, fn: () => void) => (
    <button className="text-button" disabled={doc[collections[label]].length >= 80} onClick={fn}>
      <Plus size={15} />
      {label}
    </button>
  );
  const drop = (label: string, fn: () => void) => (
    <button className="icon-button delete" aria-label={label} onClick={fn}>
      <Trash2 size={15} />
    </button>
  );
  const updateRow = (
    key:
      | "requirements"
      | "bom"
      | "poms"
      | "construction"
      | "views"
      | "callouts",
    id: string,
    patch: Record<string, unknown>,
  ) =>
    onChange({
      ...doc,
      [key]: doc[key].map((row) =>
        row.id === id ? { ...row, ...patch } : row,
      ),
    });
  const remove = (
    key:
      | "requirements"
      | "bom"
      | "poms"
      | "construction"
      | "views"
      | "callouts",
    id: string,
  ) => onChange({ ...doc, [key]: doc[key].filter((row) => row.id !== id) });
  const field = (
    label: string,
    value: string,
    fn: (s: string) => void,
    multiline = false,
  ) => (
    <label className="field">
      <span>{label}</span>
      {multiline ? (
        <textarea
          value={value}
          maxLength={8000}
          onChange={(e) => fn(e.target.value)}
        />
      ) : (
        <input
          value={value}
          maxLength={label === "Garment name" ? 160 : label === "Size / intended wearer" || label === "Size scope" ? 100 : 240}
          onChange={(e) => fn(e.target.value)}
        />
      )}
    </label>
  );
  return (
    <div className="authoring-content">
      {section === "idea" && (
        <>
          <div className="section-heading">
            <span className="eyebrow">01 / intent</span>
            <h2>Make it your own.</h2>
            <p>
              Keep every idea, even what the current pattern engine cannot
              express.
            </p>
          </div>
          {field("Garment name", doc.title, (v) => replace("title", v))}
          {field("The idea", doc.brief, (v) => replace("brief", v), true)}
          {field("Size / intended wearer", doc.sizeLabel, (v) =>
            replace("sizeLabel", v),
          )}
          <details><summary>Design requirements</summary>
          {doc.requirements.map((r, i) => (
            <div className="entry" key={r.id}>
              {drop(`Remove requirement ${i + 1}`, () =>
                remove("requirements", r.id),
              )}
              {field(
                `Requirement ${i + 1}`,
                r.text,
                (v) => updateRow("requirements", r.id, { text: v }),
                true,
              )}
              <label className="field">
                <span>Engine coverage (owner assessment)</span>
                <select
                  value={r.status}
                  onChange={(e) =>
                    updateRow("requirements", r.id, { status: e.target.value })
                  }
                >
                  <option value="unresolved">Unresolved</option>
                  <option value="supported">
                    Supported by current adapter
                  </option>
                  <option value="unsupported">Not yet supported</option>
                </select>
              </label>
              {field(
                "Notes / open decisions",
                r.note,
                (v) => updateRow("requirements", r.id, { note: v }),
                true,
              )}
            </div>
          ))}
          {addButton("Add a requirement", () =>
            replace("requirements", [
              ...doc.requirements,
              { id: uid(), text: "", status: "unresolved", note: "" },
            ]),
          )}
          </details><p className="fineprint">
            Use “Interpret my design” to propose parameters and technical notes.
            Nothing changes until you accept the proposal.
          </p>
        </>
      )}
      {section === "shape" && (
        <>
          {(['shirt','dress','skirt'].includes(doc.garment.family)) && !doc.garment.design && <button onClick={()=>replace('garment',{...doc.garment,design:structuredClone(doc.garment.family==='dress'?defaultDressDesign:doc.garment.family==='skirt'?defaultSkirtDesign:defaultShirtDesign)})}>Add editable {doc.garment.family} construction</button>}
          {doc.garment.design && <GarmentDesign doc={doc} onChange={onChange} compact/>}
          <details className="size-disclosure" open={sizeOpen} onToggle={event=>setSizeOpen(event.currentTarget.open)}><summary>Size & measurements</summary><BodySizing doc={doc} onChange={onChange}/><details className="profile-settings"><summary>Body measurement status & sources</summary>{Object.entries(doc.body).map(([name, value]) => <Measure key={name} label={`${name[0].toUpperCase() + name.slice(1)} detail`} value={value} onChange={measurement => replace('body', { ...doc.body, [name]: measurement })} />)}</details></details>
          <details className="shape-settings" open={doc.garment.family === 'none'}>
          <summary>Garment shape & fit settings</summary>
          <label className="field">
            <span>Geometry family</span>
            <select
              value={doc.garment.family}
              onChange={(e) => {
                const family=e.target.value as GarmentDocument['garment']['family'];
                replace('garment',{...doc.garment,family,design:family==='shirt'?structuredClone(defaultShirtDesign):family==='dress'?structuredClone(defaultDressDesign):family==='skirt'?structuredClone(defaultSkirtDesign):null});
              }}
            >
              <option value="none">No geometry selected</option>
              <option value="shirt">Shirt</option>
              <option value="dress">Dress</option>
              <option value="skirt">Skirt</option>
              <option value="trousers">Trousers</option>
            </select>
          </label>
          <Measure
            label="Garment length" compact
            range={[doc.garment.family === 'shirt' ? 400 : doc.garment.family==='dress'?700:componentSkirt?450:Math.ceil((mm(doc.body.height) ?? 1700) * 0.12 + 150), doc.garment.family === 'shirt' ? 1100 : doc.garment.family==='dress'?1450:1300]}
            value={doc.garment.length}
            onChange={(v) => replace("garment", { ...doc.garment, length: v })}
          />
          <Measure
            label="Ease" compact
            range={[0, 200]}
            value={doc.garment.ease}
            onChange={(v) => replace("garment", { ...doc.garment, ease: v })}
          />
          <label className="field">
            <span>Flare multiplier · supported range varies by family</span>
            <input type="range" aria-label="Flare slider" min={flareRange[0]} max={flareRange[1]} step="0.01" value={doc.garment.flare} onChange={event => replace('garment', { ...doc.garment, flare: Number(event.target.value) })} />
            <input
              type="number"
              min={flareRange[0]}
              max={flareRange[1]}
              step=".1"
              value={doc.garment.flare}
              onChange={(e) => {
                if (Number.isFinite(e.target.valueAsNumber))
                  replace("garment", {
                    ...doc.garment,
                    flare: e.target.valueAsNumber,
                  });
              }}
            />
          </label>
          </details>

        </>
      )}
      {section === "materials" && (
        <>
          <div className="section-heading">
            <span className="eyebrow">03 / materials</span>
            <h2>What it’s made of.</h2>
            <p>
              Fabric, lining and trims. Unknown quantities and specifications
              can stay unknown.
            </p>
          </div>
          {doc.bom.map((r, i) => (
            <div className="entry" key={r.id}>
              {drop(`Remove material ${i + 1}`, () => remove("bom", r.id))}
              {field(`Material ${i + 1}`, r.name, (v) =>
                updateRow("bom", r.id, { name: v }),
              )}
              <label className="field">
                <span>Category</span>
                <select
                  value={r.category}
                  onChange={(e) =>
                    updateRow("bom", r.id, { category: e.target.value })
                  }
                >
                  <option value="fabric">Fabric</option>
                  <option value="lining">Lining</option>
                  <option value="trim">Trim</option>
                  <option value="other">Other</option>
                </select>
              </label>
              {field(
                "Specification · composition, weight, color",
                r.specification,
                (v) => updateRow("bom", r.id, { specification: v }),
                true,
              )}
              {field("Placement", r.placement, (v) =>
                updateRow("bom", r.id, { placement: v }),
              )}
              {field(
                "Quantity / unit (leave blank if unknown)",
                r.quantity,
                (v) => updateRow("bom", r.id, { quantity: v }),
              )}
              {field("Source / provenance", r.source, (v) =>
                updateRow("bom", r.id, { source: v }),
              )}
            </div>
          ))}
          {addButton("Add material or trim", () =>
            replace("bom", [
              ...doc.bom,
              {
                id: uid(),
                name: "",
                category: "fabric",
                specification: "",
                placement: "",
                quantity: "",
                source: "",
              },
            ]),
          )}
        </>
      )}
      {section === "measurements" && (
        <>
          <div className="section-heading">
            <span className="eyebrow">04 / measurement</span>
            <h2>Define the measure.</h2>
            <p>
              Finished-garment measurements are not body measurements. Explain
              how each point is measured.
            </p>
          </div>
          {doc.poms.map((r, i) => (
            <div className="entry" key={r.id}>
              {drop(`Remove measurement ${i + 1}`, () => remove("poms", r.id))}
              {field(`Point of measure ${i + 1}`, r.name, (v) =>
                updateRow("poms", r.id, { name: v }),
              )}
              {field(
                "Measurement method",
                r.method,
                (v) => updateRow("poms", r.id, { method: v }),
                true,
              )}
              {field("Size scope", r.size, (v) =>
                updateRow("poms", r.id, { size: v }),
              )}
              <Measure
                label={`POM ${i + 1} target`}
                value={r.target}
                onChange={(v) => updateRow("poms", r.id, { target: v })}
              />
              <Measure
                label={`POM ${i + 1} tolerance`}
                value={r.tolerance}
                onChange={(v) => updateRow("poms", r.id, { tolerance: v })}
              />
              {field(
                "POM notes",
                r.note,
                (v) => updateRow("poms", r.id, { note: v }),
                true,
              )}
            </div>
          ))}
          {addButton("Add point of measure", () =>
            replace("poms", [
              ...doc.poms,
              {
                id: uid(),
                name: "",
                method: "",
                size: doc.sizeLabel,
                target: { state: "unknown" },
                tolerance: { state: "unknown" },
                note: "",
              },
            ]),
          )}
        </>
      )}
      {section === "construction" && (
        <>
          <div className="section-heading">
            <span className="eyebrow">05 / construction</span>
            <h2>How it comes together.</h2>
            <p>
              Record assembly steps and questions. A note is not a verified
              sewing instruction.
            </p>
          </div>
          {doc.construction.map((r, i) => (
            <div className="entry" key={r.id}>
              {drop(`Remove construction ${i + 1}`, () =>
                remove("construction", r.id),
              )}
              {field(
                `Construction step ${i + 1}`,
                r.operation,
                (v) => updateRow("construction", r.id, { operation: v }),
                true,
              )}
              {field(
                "Construction questions / details",
                r.note,
                (v) => updateRow("construction", r.id, { note: v }),
                true,
              )}
            </div>
          ))}
          {addButton("Add construction step", () =>
            replace("construction", [
              ...doc.construction,
              { id: uid(), operation: "", note: "" },
            ]),
          )}
          <h3>Callouts</h3>
          {doc.callouts.map((r, i) => (
            <div className="entry" key={r.id}>
              {drop(`Remove callout ${i + 1}`, () => remove("callouts", r.id))}
              {field("Anchor · field path or panel ID", r.anchor, (v) =>
                updateRow("callouts", r.id, { anchor: v }),
              )}
              {field(
                `Callout ${i + 1}`,
                r.text,
                (v) => updateRow("callouts", r.id, { text: v }),
                true,
              )}
            </div>
          ))}
          {addButton("Add a callout", () =>
            replace("callouts", [
              ...doc.callouts,
              { id: uid(), anchor: anchor || "/construction", text: "" },
            ]),
          )}
        </>
      )}
      {section === "references" && (
        <>
          <div className="section-heading">
            <span className="eyebrow">06 / references</span>
            <h2>Show what you mean.</h2>
            <p>
              Photos, sketches and technical views stay distinct. Images are
              private and stripped of metadata.
            </p>
          </div>
          <label className="upload-button">
            <Upload size={17} />
            {uploading ? "Processing image…" : "Add an image"}
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              disabled={uploading || doc.views.length >= 20}
              onChange={async (e) => {
                const file = e.target.files?.[0];
                if (!file) return;
                setUploading(true);
                try {
                  if (file.size > 10 * 1024 * 1024)
                    throw new Error("Choose an image under 10 MB.");
                  const result = await api<{ assetId: string }>(
                    `/projects/${projectId}/references`,
                    {
                      method: "POST",
                      headers: {
                        "Content-Type": file.type,
                        "X-Filename": encodeURIComponent(file.name),
                      },
                      body: file,
                    },
                  );
                  onReference({
                    id: uid(),
                    assetId: result.assetId,
                    role: "front",
                    kind: "reference",
                    caption: "",
                  });
                } catch (error) {
                  onError((error as Error).message);
                } finally {
                  setUploading(false);
                  e.target.value = "";
                }
              }}
            />
          </label>
          {doc.views.map((r, i) => (
            <div className="entry" key={r.id}>
              <PrivateImage
                className="reference-thumb"
                src={`/api/projects/${projectId}/references/${r.assetId}`}
                alt={r.caption || `${r.role} ${r.kind}`}
              />
              {drop(`Remove reference ${i + 1}`, () => remove("views", r.id))}
              <div className="two-fields">
                <label className="field">
                  <span>View</span>
                  <select
                    value={r.role}
                    onChange={(e) =>
                      updateRow("views", r.id, { role: e.target.value })
                    }
                  >
                    <option value="front">Front</option>
                    <option value="back">Back</option>
                    <option value="detail">Detail</option>
                  </select>
                </label>
                <label className="field">
                  <span>Evidence type</span>
                  <select
                    value={r.kind}
                    onChange={(e) =>
                      updateRow("views", r.id, { kind: e.target.value })
                    }
                  >
                    <option value="reference">Reference photo</option>
                    <option value="sketch">Sketch</option>
                    <option value="technical-flat">Technical flat</option>
                  </select>
                </label>
              </div>
              {field(
                "Caption / source",
                r.caption,
                (v) => updateRow("views", r.id, { caption: v }),
                true,
              )}
            </div>
          ))}
        </>
      )}
      {section === "review" && (
        <>
          <div className="section-heading">
            <span className="eyebrow">07 / review</span>
            <h2>Keep the questions visible.</h2>
            <p>
              Feedback is recorded by you against a saved revision. It is not an
              independently authenticated maker approval.
            </p>
          </div>
          {comments.length === 0 ? (
            <p>
              No revision comments yet. Use “Revisions & export” to record one.
            </p>
          ) : (
            comments.map((c) => (
              <article className="review-note" key={c.id}>
                <small>
                  {c.reportedReviewer || "Owner note"} · {c.anchor}
                </small>
                <p>{c.text}</p>
                <code>{c.revisionId.slice(0, 8)}</code>
              </article>
            ))
          )}
          <h3>Unresolved or unsupported</h3>
          {doc.requirements
            .filter((r) => r.status !== "supported")
            .map((r) => (
              <article key={r.id} className="review-note">
                <strong>{r.status}</strong>
                <p>{r.text || "Untitled requirement"}</p>
                <p>{r.note}</p>
              </article>
            ))}
          <div className="review-note">
            <strong>Physical fit and print calibration</strong>
            <p>
              Not verified. Exports remain draft review documents and printable
              references, never cutting-ready patterns.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
