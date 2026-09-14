import { Plus, Trash2, Upload, ArrowUpRight } from "lucide-react";
import { useState } from "react";
import {
  assumed,
  convert,
  type GarmentDocument,
  type Measurement,
  type ReviewComment,
} from "../../../../packages/contracts";
import { api } from "../lib/api";

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
}: {
  label: string;
  value: Measurement;
  onChange: (m: Measurement) => void;
}) {
  return (
    <fieldset className="measurement">
      <legend>{label}</legend>
      <div className="measure-row">
        <select
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
        </select>
        {"value" in value && (
          <>
            <input
              type="number"
              min="0"
              max="20000"
              step="any"
              aria-label={`${label} value`}
              value={value.value}
              onChange={(e) => {
                const n = e.target.valueAsNumber;
                if (Number.isFinite(n)) onChange({ ...value, value: n });
              }}
            />
            <select
              aria-label={`${label} unit`}
              value={value.unit}
              onChange={(e) =>
                onChange(convert(value, e.target.value as "mm" | "cm" | "in"))
              }
            >
              <option>mm</option>
              <option>cm</option>
              <option>in</option>
            </select>
          </>
        )}
      </div>
      {"source" in value && (
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
          <h3>Design requirements</h3>
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
          <p className="fineprint">
            No interpretation model is running in this prototype. Descriptions
            are preserved, not silently converted into parameters.
          </p>
        </>
      )}
      {section === "shape" && (
        <>
          <div className="section-heading">
            <span className="eyebrow">02 / geometry</span>
            <h2>
              A starting shape,
              <br />
              not a design limit.
            </h2>
            <p>
              These controls use the pinned Design2GarmentCode CPU adapter.
              Other construction stays in your brief.
            </p>
          </div>
          <label className="field">
            <span>Geometry family</span>
            <select
              value={doc.garment.family}
              onChange={(e) =>
                replace("garment", {
                  ...doc.garment,
                  family: e.target
                    .value as GarmentDocument["garment"]["family"],
                })
              }
            >
              <option value="none">No geometry selected</option>
              <option value="shirt">Shirt</option>
              <option value="skirt">Skirt</option>
              <option value="trousers">Trousers</option>
            </select>
          </label>
          <Measure
            label="Garment length"
            value={doc.garment.length}
            onChange={(v) => replace("garment", { ...doc.garment, length: v })}
          />
          <Measure
            label="Ease"
            value={doc.garment.ease}
            onChange={(v) => replace("garment", { ...doc.garment, ease: v })}
          />
          <label className="field">
            <span>Flare multiplier · supported range varies by family</span>
            <input
              type="number"
              min={doc.garment.family === "skirt" ? 0.5 : 0.7}
              max={doc.garment.family === "skirt" ? 2 : doc.garment.family === "trousers" ? 1.2 : 1.5}
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
          <h3>Body inputs</h3>
          <p className="fineprint">
            Private, explicit inputs. These are never inferred from your
            references.
          </p>
          {Object.entries(doc.body).map(([name, value]) => (
            <Measure
              key={name}
              label={name[0].toUpperCase() + name.slice(1)}
              value={value}
              onChange={(v) => replace("body", { ...doc.body, [name]: v })}
            />
          ))}
          <details className="example-disclosure">
            <summary>Try an explicitly synthetic example</summary>
            <p>
              Use a 170 cm height, 92 cm bust, 76 cm waist, 98 cm hip and 40 cm
              shoulder fixture. These are not your measurements or a validated
              size chart. Unspecified dimensions will be reported as adapter
              assumptions.
            </p>
            <button
              onClick={() => {
                onChange({
                  ...doc,
                  garment: {
                    ...doc.garment,
                    length: assumed(
                      doc.garment.family === "shirt"
                        ? 600
                        : doc.garment.family === "skirt"
                          ? 650
                          : 1000,
                    ),
                    ease: assumed(80),
                    flare: 1,
                  },
                  body: {
                    height: assumed(1700),
                    bust: assumed(920),
                    waist: assumed(760),
                    hip: assumed(980),
                    shoulder: assumed(400),
                  },
                });
              }}
            >
              Use these assumed values
            </button>
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
              <img
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
