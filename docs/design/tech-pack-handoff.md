# Tech Pack and Maker Handoff

**2026-09-14 · Proposed**

## Scope and evidence

This design expands the normative [project contract](project-contract.md), following [product intent](../product/vision.md). It specifies reviewable exports, not manufacturing approval. All proposed behavior and acceptance criteria below remain unimplemented and untested.

The observed upstream CPU smoke test produced nonempty geometry and serialized artifacts for three fixed-parameter fixtures. It did not establish seam correctness, printable calibration, construction feasibility or fit; see [source evidence](../research/design2garmentcode-evidence.md). Upstream code is not installed in the new app.

The first deliverable is an app-generated draft PDF and editable manifest. Free creation, draft sharing and DIY access require neither manufacturing purchase nor paid review. Physical sampling follows usable app output. Marketplace, factory bidding and crowdfunding remain separate later products.

## 1. Concrete draft deliverables

Export produces a tech-pack PDF, editable JSON manifest and explicitly selected pattern artifacts. Unsupported requirements, absent geometry and unresolved decisions remain visible; incompleteness does not block a draft export.

The PDF uses this section order, allowing sections to span pages:

1. **Design overview:** title, revision, garment views, intended silhouette, requirement coverage and prominent unresolved decisions.
2. **Garment flats:** front, back and relevant detail views, with stable callout IDs. Label concept illustrations separately from technical flats. If no back view exists, display “Back view not supplied”; do not mirror the front or fabricate construction.
3. **Materials/BOM:** fabric, lining, interfacing, thread, closures and trims; component placement, specification, quantity basis and unresolved substitutions.
4. **Finished-garment measurements:** POM definitions, size scope, values and any justified tolerances.
5. **Construction:** assembly relationships, seams, finishes, closures, pressing and unresolved operations.
6. **Pattern inventory:** panel IDs, thumbnails, cut counts, orientation and annotation completeness.
7. **Review and change record:** scoped evidence, open comments, revision changes and export limitations.

New tech-pack and cover pages carry a revision label, immutable revision ID, snapshot ID, **pre-render snapshot-input digest**, page number and draft-review status. That input digest excludes all rendered output checksums. The separate delivery manifest lists final file hashes; its own checksum is recorded externally, never recursively inside the bytes being hashed.

**Frozen pattern-sheet exception:** an existing pattern PDF keeps the artifact ID, source revision and artifact-input digest printed when it was generated. Including it in a later bundle must not restamp, relabel, watermark or otherwise change its bytes. Current bundle/snapshot identity, scoped classification and new evidence belong to the delivery manifest and separate cover pages. A restamped/scaled/reformatted PDF is a new derivative; exact-byte calibration does not transfer automatically. A later cover can document candidate eligibility without rewriting the original reference sheets.

A front-page completeness summary distinguishes absent, assumed and unsupported information without burying the garment beneath warnings.

## 2. Editable manifest and measurement boundaries

The manifest is the editable counterpart of the PDF, not permission to mutate its immutable source revision. Importing edits creates a bounded proposal against the exported revision. Stale imports require reconciliation; the original snapshot remains unchanged.

Imports compare only disclosed, editable fields with the owned server-stored source export projection. Absent/redacted fields are no-ops, not replacement-state deletions. `unknown` and `not_applicable` are explicit values; intentional deletion is a separately reviewable `clearField` operation. Supplied provenance, actors, evidence and approval metadata remain untrusted. Stale import proposals use the same server-captured revision/draft/digest acceptance contract as any other proposal.

Domain sections are:

- `overview` and `requirements`: intent, coverage, unresolved decisions and view references.
- `materials` and `bom`: material specifications, components, placements and consumption.
- `bodyInputs`: explicitly selected body data, omitted by default.
- `finishedMeasurements`: POM definitions and size-specific values.
- `construction`: operations and assembly relationships.
- `patternInventory`: panels, annotations and artifact references.
- `review`: scoped comments and evidence references.
- `exportDisclosure`: omissions, redactions and output classifications.

Apply the contract’s value states and provenance to individual fields. A proposed fabric weight is `assumed`, not a known property of purchased fabric. Unknown consumption remains unknown; panel area alone does not establish purchasing yardage without usable width, layout, directionality and waste assumptions.

Keep three measurement domains distinct:

- **Body:** anatomical measurement and measuring method.
- **Finished garment:** defined measurement location and condition, such as relaxed or stretched.
- **Panel geometry:** edges, coordinates and geometric derivations.

A geometry-derived POM records its derivation and material/size scope. It must not quietly convert panel width into body circumference.

Materials record composition, structure, weight, stretch direction and recovery only when supplied or measured. Supplier names, prices and tolerances are optional, sourced fields.

Size labels identify explicit measurement sets. A single-size result does not imply a graded range. Unspecified grading rules remain unknown; global geometric scaling is not grading.

## 3. Flats, construction and panel annotations

Garment flats describe the assembled garment. Panel flats depict separate pattern pieces. Neither substitutes for the other, and a simulation screenshot is not automatically a technical flat.

Every pattern panel has a stable identifier linked to construction callouts. Annotations distinguish seam lines, cutting lines, allowance boundaries, grain direction, notches, darts, fold lines, cut counts and mirrored pairs. Material assignments connect panels to BOM entries.

Allowances are edge-scoped where necessary. Missing allowance data differs from an explicitly specified zero allowance. Fold and grain applicability require recorded reasoning; the system must not infer “not applicable” merely because an annotation is missing.

Assembly records identify connected edges and intended treatment. Gathered or eased seams require specified relationships rather than an unconditional equal-length rule. Unknown seam treatment remains a review issue.

A maker must be able to trace a PDF callout to its panel, edge, requirement and source field without relying on visual proximity.

## 4. Pattern classifications and calibration

Classification belongs to the exact artifact and export policy, never the project globally.

| Class | Permitted claim |
|---|---|
| **Screen preview** | Inspect geometry visually; no printable-scale claim. |
| **Printable reference** | Print for discussion; not cleared for cutting. |
| **Calibrated cutting candidate** | Required digital checks and scoped print calibration pass; fit and sewing feasibility remain separate. |

Draft exports may contain any class, or no pattern. Candidate eligibility follows the contract's service-controlled issuer classes and effective-evidence reducer. Same-scope unresolved, failing or contradictory effective evidence blocks promotion even when an earlier pass exists. Only explicit authorized supersession/withdrawal with a reason can resolve it; timestamps alone do not. Owner-recorded physical measurements remain allowed without paid review but are not relabeled professional certification. Recorded evidence and superseded history remain inspectable.

Proposed calibration policy requires both verified PDF dimensions and recorded physical scale checks for the exact pattern artifact and declared printer, paper and settings. Without that evidence, the export remains a printable reference. Calibration requires no sewn sample or professional payment.

Actual-size PDFs use explicit physical page boxes and canonical-to-PDF unit conversion. Tiled output declares paper size, orientation, printable margins, overlap, row/column coordinates, registration marks and assembly order. Include independent horizontal and vertical calibration targets, including a 100 mm square. Instructions prohibit “fit to page.”

Digital tests verify target dimensions, tile transforms, complete panel coverage and absence of clipping or double scaling. Physical acceptance bounds must be documented and justified before enabling candidate classification, not invented during rendering.

Promotion adds evidence referencing the unchanged pattern digest. Changed print settings require rechecking; candidate status never guarantees another printer’s output.

## 5. Snapshots, privacy and maker feedback

Export operates only on a saved revision. Concurrent editor changes cannot replace views, measurements or patterns inside the selected snapshot. Missing or digest-mismatched artifacts cause an explicit failure or a newly selected draft snapshot without them—not silent substitution.

Before export, show a disclosure preview. Body measurements, source photographs, contact details and raw model exchanges are excluded by default. Disclose necessary assumptions without including private inputs. Test redaction in visible content, image metadata, PDF metadata, attachments and manifest fields. Draft incompleteness warnings remain visible after redaction.

Maker comments attach to a revision and field, POM, operation, panel or artifact. Owner-entered comments distinguish the recorder from the reported reviewer; offline entries do not imply authenticated external identity.

Resolving a comment creates a new revision when design data changes. Review evidence remains attached to its historical scope. Geometry, material, sizing, measurement-definition and relevant exporter changes invalidate affected current-review scopes unless a tested equivalence rule applies.

Physical sample evidence records the actual revision, material, size, deviations and checks performed. Conflicting sample or reviewer findings remain visible together. Neither a successful sample in one material nor a reviewer’s general praise becomes global approval.

## 6. Fixtures and acceptance criteria

Use rights-cleared synthetic fixtures; upstream smoke-test artifacts are not validated production fixtures.

| Fixture | Required result |
|---|---|
| Minimal single-size garment | PDF and manifest agree on revision, views, POMs and panels. |
| Missing back view; unsupported requested detail | Explicit omissions in viewer and PDF; draft export succeeds. |
| Unknown units; body/POM name collision | Saving succeeds; geometric execution or scale claims fail; field types remain distinct. |
| Equal seam, gathered seam, incompatible seam | Relationship-specific checks distinguish valid pairing from unresolved or failed assembly. |
| Missing allowance versus explicit zero | Different states survive editing, rendering and round-trip import. |
| Grain/notch/fold applicability variants | Missing required marks block candidate status; justified `not_applicable` passes only its applicable check. |
| Long BOM, large panels, multilingual notes | Pagination preserves content, legibility, callouts and repeated headers without overflow. |
| Tiled pattern; altered print scaling | Digital coverage passes independently; missing or failed physical calibration prevents promotion. |
| Concurrent edit/export; mismatched revision artifact | Snapshot stays coherent; conflicting artifacts are rejected. |
| Conflicting sample evidence; changed material | Both findings remain accessible; unrelated checks survive while affected evidence becomes stale. |

Acceptance additionally requires deterministic policy results, unauthorized-download rejection, redaction inspection and manifest round-trip tests. Physical fit stays explicitly unvalidated until scoped sample evidence exists.

## 7. Component handoff

The [engine and AI component](engine-ai.md) supplies canonical geometry, stable references, capability gaps and structured checks—not approval language. The editor supplies saved revisions and typed proposals. Export owns layout, disclosure, immutable snapshot rendering and classification evaluation. Review owns comments and evidence dependencies.

Implement draft PDF/manifest export first, then annotation checks, calibrated printing and offline feedback. None waits for procurement integration. (￣▽￣)

The no-model authoring path includes BOM rows, POM definitions and values/provenance, construction operations/questions, assigned front/back/detail views and stable linked callouts. The populated normal-UI fixture in [the active plan](../plans/software-prototype.md#substantive-manual-authoring-acceptance-fixture) is a release test; do not test only privileged fixtures or empty exports. Generally incomplete draft exports remain available.

Pattern geometry itself may expose body dimensions even if raw measurements are omitted. Disclosure controls must explain that redacted input fields do not anonymize the garment geometry.

Additional acceptance tests: export a calibrated PDF unchanged in two bundles; verify every digest without circular hashing; fail eligibility for pass-then-fail evidence until scoped disposition; round-trip a body-redacted manifest changing one BOM value without clearing hidden data; author the substantive fixture entirely through the UI with providers disabled.
