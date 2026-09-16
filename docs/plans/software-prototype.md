# Sew Computer: software prototype execution plan

**2026-09-16 · Active plan · Component garment pipeline privately deployed; breadth and physical/research gates remain open**

Pattern-derived 3D planning update, 2026-09-16: agreed architecture and ordered work are recorded below. No 3D implementation or simulator feasibility is claimed. This extends this sole active plan and supersedes the earlier optional-only 3D sequencing, while preserving existing 2D generation, sizing, specifications, private revisions and exports.

The owner's latest priority is an immediately usable describe-to-garment software flow. The manual milestone is complete; it is no longer the stopping point. The current implementation connects a tool-free model to reviewed typed proposals, explicit measurements, the existing CPU engine and revisioned exports. See the [AI integration review](../reviews/ai-prototype.md). This scope supersedes the earlier manual-only release sequencing without changing physical-readiness claims.

This is the only active execution plan. It supersedes the earlier workspace-only software-prototype proposal for sequencing and integration decisions, while preserving the [product intent](../product/vision.md). Specialist designs below are supporting specifications, not competing roadmaps. The [implementation review](../reviews/manual-prototype.md) records delivered behavior and deviations; the acceptance requirements below remain the target.

## Outcome

Garment pipeline follow-up (2026-09-16, in implementation): the owner's current scope is brief → editable construction design → matching component patterns → derived specifications and front/back schematics. The LLM should resolve ordinary design ambiguity itself and explain editable choices; the owner is building the software, not commissioning a bespoke shirt in chat. The white button-up with practical tails and frills is a regression case, not a permanent product restriction. Preserve all existing private revision/export behavior. This supersedes the base-pattern-only interpretation boundary for supported component designs.

The first component compiler is explicitly a relaxed drop-shoulder woven shirt, with optional sleeves, button cuffs/openings, stand/fall collar, front plackets, curved back tails and gathered front frills. It must not silently stand in for set-in sleeves, split coat tails, fitted shaping or other unsupported construction. Digital seam/annotation/coverage checks are separate from physical fit and sewing readiness. Remaining breadth and physical gates stay open; see [garment pipeline review](../reviews/garment-pipeline.md).

Printing follow-up: new generations include nominal 1:1 A4 and US Letter tiled pattern PDFs with overlap/alignment guides, piece inventories and a 100 mm calibration square, alongside the existing custom-size sheets. Raster tests verify digital scale and tile continuity; the wearer must still check their printer output. These files obey the same immutable artifact and pattern-disclosure boundaries. This closes digital home-print tiling, not physical calibration or garment breadth.

Visual sizing follow-up (2026-09-16): implement synthetic sample sizes, synchronized sliders/numbers and a clearly labeled schematic, private reusable measurements, shared preflight feedback and a safe explanation of the zero-length-edge failure. Preserve entered measurements, assumptions, unsupported design intent, immutable revisions and export privacy. This extends the existing editor; it does not add simulation or claim broader validated engine coverage. See [visual sizing operations](../deployment.md#visual-sizing) and [adversarial implementation review](../reviews/visual-sizing.md).

Build software that lets a person describe an original garment, attach references, inspect an honest interpretation, edit supported geometry, and export a revisioned draft tech pack plus correctly classified pattern artifacts. Its actual deliverable can then be reviewed and sampled by a maker. Recruiting a maker or choosing one test top is not a prerequisite to development.

The first release has a declared geometry capability, not a claim to represent every possible garment. Unsupported requests remain part of the design and have a visible extension path. The product must not permanently restrict creation to presets, decoration on blanks, manufacturing purchase or paid review.

## Decisions and alternatives

| Question | Recommendation | Why / rejected alternative |
|---|---|---|
| Separate product or Stylr feature? | Independent Sew Computer repository and development Site. | Garment authoring has different data/security and geometry requirements; do not touch either Stylr working directory. |
| Research GUI or new consumer interface? | React visual workspace around an explicit adapter. | The research GUI is useful reference material, not the intended UX or API security boundary. |
| Geometry integration? | Pin and wrap the proven CPU GarmentCode path from Design2GarmentCode. | Image-only generation cannot provide connected pattern geometry. Unrestricted generated Python is not an acceptable web-server fallback. |
| First AI path? | Separately named, bounded multimodal-to-typed-edit adapter; evaluate the released MMUA/projector path in a parallel dependency track. | Do not hold all product work hostage to unverified GPU/weights, or falsely claim an alternative adapter reproduces the paper. |
| Data authority? | Immutable revisions in SQLite, private artifact storage and durable jobs. | Chat history or browser-only state cannot reliably bind pattern, PDF, edits and maker corrections. |
| Initial deployment? | Single-owner private development, explicit API auth, Bun/TypeScript API + supervised trusted Python worker. | A private page is not API auth; a subprocess is not a security sandbox. Public multi-user deployment needs another gate. |
| Garment visualization? | Pattern-derived 3D through a trusted assembly/simulation adapter, retaining actual 2D views. | AI imagery or independently modeled meshes cannot verify what the patterns construct. Progress through the gated 3D workstream below. |
| First handoff? | Draft PDF/review bundle from an immutable revision, with omissions and assumptions visible. | A pretty PDF or valid polygon does not make a pattern production-ready. |

The shared [project contract](../design/project-contract.md) is normative for revision, job, artifact, unit, approval and security semantics. Supporting designs: [editor](../design/editor-ux.md), [engine/AI](../design/engine-ai.md), [backend](../design/backend-security.md), [tech pack/handoff](../design/tech-pack-handoff.md).

## Repository and runtime boundaries

Proposed implementation layout, not files that already exist:

```text
apps/web/                 React canvas, editing, review and export UI
apps/api/                 Auth, ownership, projects, revisions, jobs, artifact access
packages/contracts/      Versioned schemas, units, typed operations and errors
packages/tech-pack/      Derived document model, PDF and bundle generation
services/engine/          Python trusted-program adapter and fenced worker
packages/test-fixtures/  Cleared fixtures, geometry expectations, malformed inputs
scripts/                 Setup, migration, recovery and fixture verification
```

A Zo Site scaffold may require a thin root entrypoint/config around these boundaries. Keep the domain schemas independent of hosting layout. The Site manages its web lifecycle; use the platform's managed process primitive for a worker, never a hand-started daemon or a second public worker endpoint. Do not register a Site's web server manually as a separate service.

## Ordered work packages

### W0 — Contracts and reproducible baseline

**Owner:** integration/backend, paired with engine. **Dependencies:** none beyond the documented source spike.

- Implement shared runtime schemas and golden examples for revisions, typed values, requirements, proposals, artifacts, validation reports, jobs and manifests.
- Pin upstream source and Python dependencies, record a dependency/license inventory, and turn the CPU spike into a portable automated test using fixtures cleared for that purpose.
- Verify adapter unit/coordinate conventions, actual output/error behavior and structured content validation from source. No reliance on a printed success counter.

**Acceptance:** clean setup reproduces nonempty shirt/skirt/trouser geometry and JSON/SVG/PDF artifacts with versioned checks. Demonstrate cm/inch equivalence, correct typed field distinctions, malformed-parameter rejection and missing-output failure. These tests are not physical print or fit certification.

### W1 — Private durable project and worker slice

**Owner:** backend/engine. **Dependency:** W0 schemas and baseline.

- Implement explicit owner authentication and protected artifact/API access, private storage, migrations, immutable revisions and mutable autosave buffers with optimistic conflicts.
- Implement job leases, idempotency, cancellation, crash recovery, output staging/install and deletion fencing.
- Establish the actual host's process/user/filesystem/network isolation and resource enforcement; keep arbitrary code synthesis disabled. Document weaker controls rather than call them sandboxing.

**Acceptance:** unauthorized origin-server calls fail; stale/cancelled/deleted jobs cannot publish; crash/restart restores the queue; duplicate requests do not double-publish; data export/backup/restore maintains revision/artifact consistency. The three trusted geometry fixtures complete under measured resource budgets.

### W2 — Consumer editor and real pattern viewer

**Owner:** frontend, with contracts/engine. **Dependency:** W1 project/job interfaces; use clearly labeled fixtures only during implementation.

- Build creation from text/references and an optional starting shape without requiring a sample-maker appointment.
- Show real 2D output, references and generated concepts as different view types. Provide editable requirements, body/material assumptions, panels/stitches, meaningful job errors and unsupported-intent status.
- Implement revisions/undo, explicit acceptance of proposals, saved/unsaved/conflict states and retry/recovery without lost work.

**Acceptance:** at 1366×768 desktop and 390×844 mobile, useful visual content appears immediately and essential actions remain reachable without a wall of introductory text. Keyboard/screen-reader operation works; dialogs return focus; unsupported features do not disappear. Reload and two-tab conflict tests preserve changes. No fake 3D or physically validated fit claim.

### W3 — Intent-to-edit integration

**Owner:** engine/AI + frontend. **Dependencies:** W0 schemas; W2 proposal review surface. Can begin against validated fixtures while W1 develops.

- Integrate a named multimodal-to-data proposal adapter with explicit provider settings, privacy disclosure, token/cost limits and bounded retries.
- Trace every material user requirement to a supported operation, unresolved decision or capability gap. Reject invented measurements and arbitrary code.
- Create a small maintained evaluation set covering supported edits, ambiguous instructions, deliberately unsupported construction, contradictory references and prompt-injection attempts. Use multiple garment families; do not substitute one attractive demo for coverage.

**Acceptance:** a person can describe a supported design/edit and obtain a proposal tied to the right revision, then regenerate connected geometry. Unsupported/asymmetric requests remain visible rather than silently becoming a different garment. Provider outage, cost limit and invalid response leave the saved project intact. Report measured requirement preservation and geometry success separately; set expansion criteria from baseline results rather than fabricate accuracy percentages.

### W4 — Derived tech pack and maker-ready review package

**Owner:** handoff/document pipeline. **Dependencies:** W0/W1 contracts and W2 actual artifact flow; can develop fixtures before W3.

- Derive document content from explicit inputs and pinned geometry where valid. Include BOM, POM definitions, construction, pattern inventory, size/grading scope, missing annotations/views and revision-specific open questions.
- Add a clean PDF with a revision number, page identities and a versioned export manifest; provide editable portable project data without leaking private inputs by default.
- Separate a draft review export from calibrated cutting-pattern candidate gates. Do not label laid-out panels as technical flats or infer finished-garment measurements without a valid definition/derivation.
- Support owner-recorded maker comments anchored to revision/panel/field IDs and a follow-up correction revision. External authenticated collaboration is later work.

**Acceptance:** export is reproducible from the named revision while another edit/job runs; stale approvals are not carried forward; unknown fields stay unknown; every referenced artifact exists and matches its digest. Calibration, page scale, tiles, grain/notches/allowances and size scope have explicit tests/statuses. A printable reference file cannot acquire a cutting-ready name merely because it is a PDF.

### W5 — Manual-first release review, then an independent AI enhancement review

**Dependencies:** W0/W1/W2/W4 for the first integrated release. W3 is deliberately **not** a prerequisite. With model routes disabled and no provider credentials, the owner must create an original brief, manually author the garment project, edit supported parameters, generate geometry, publish revisions, export a useful PDF/manifest and incorporate a specific maker-style correction. Review and label this as the manual/CPU prototype, not an AI product or reproduction of the upstream pipeline.

Run the applicable security, lifecycle, engine, export and accessibility suites below against this no-model configuration and the substantive W4 fixture. An independent reviewer must attempt stale proposal acceptance, redacted round trips, contradictory calibration, checksum cycles, cross-project access and backup resurrection. Address blocking findings before release.

After W3, rerun these suites plus interpretation fidelity, unsupported-intent retention, prompt-injection, budgets/provider timeout and model-job recovery tests. The AI-enabled prototype is a separate release gate; its success is not assumed from the manual release. The released upstream MMUA/projector and optional simulation remain separately named, separately verified gates.

### Substantive manual-authoring acceptance fixture

This fixture is a test case, not a restriction on what people may design. Starting from an empty project using only normal UI controls, author a non-default supported garment: front/back reference or technical-view uploads with honest provenance, a design overview and unresolved requirements, one main fabric and one relevant trim/BOM item, two finished-POM definitions with measurement method and user-entered target or explicit unresolved target, construction operations/questions, available real pattern inventory and a callout linked to a stable field/piece. Do not fill missing tolerances or seam allowances with invented defaults. Generate and revise supported geometry, then export a review package from that revision. A reviewer must be able to request one precise material/POM/construction correction that returns to a bounded proposal. Incomplete drafts remain exportable generally; this populated fixture establishes that the product can do more than print empty headings.

Required authoring surfaces in W2/W4: editable BOM rows, POM definitions/values/provenance, construction notes and operations, front/back/detail view assignment, stable callouts, and issue-to-field navigation. Import is a disclosed-field patch with explicit clearing, not replacement state.

### Planning-review regression obligations

- Proposal generated at revision R/draft v7 is rejected after a v8 save without losing v8 or the proposal; successful acceptance rebases the draft atomically.
- Same pattern bytes in two export bundles retain calibration identity; delivery checksums are nonrecursive.
- Pass then fail for identical evidence scope blocks candidate eligibility until explicitly dispositioned; issuer spoofing cannot create authority.
- Body-redacted manifest plus one BOM edit changes only that field; explicit deletion is separately reviewed.
- Restore after deletion/revocation cannot reopen deleted data or revived sessions; incomplete deletion-journal replay fails closed.

### Integrated validation matrix

| Suite | Manual/CPU release | Additional AI-enabled release |
|---|---|---|
| Project lifecycle | Save/reload; optimistic conflicts; atomic revision and draft updates; undo; deletion during work; interrupted restore | Stale interpretation proposals, retries and provider outage preserve the same state |
| Geometry and units | Multiple garment families; parameter boundaries; unit round trips; invalid edges/stitches; semantic geometry checks | Measure requirement preservation separately from successful geometry generation |
| Jobs and artifacts | Crash after write/before DB commit; lease expiry; duplicate work; cancellation and deletion fencing; checksum failures | Provider idempotency and budget reservation/reconciliation |
| Security/privacy | Direct unauthenticated API requests; cross-project IDs; malformed/oversized raster input; SVG/PDF active content; worker escape probes; safe disclosure | Prompt injection, invalid model schemas, secret leakage and outbound-data minimization |
| Handoff | Populated normal-UI fixture; missing/unknown field behavior; redacted patch round trip; nonrecursive digests; unchanged pattern bytes; contradictory evidence | Suggested technical values cannot silently become reviewed or measured |
| UX/accessibility | Desktop/mobile above-fold visuals; keyboard/screen reader; focus restoration; loading/error/retry/stale states | Interpretation progress, capability gaps, informed acceptance and undo |

Physical fit remains unverified until actual physical evidence exists. These are required implementation tests, not tests that this planning change has executed.

## Parallel dependency tracks, not silent blockers

1. **Full released AI reproduction:** verify weight/base-model licenses and provenance, memory/hardware needs, installation and evaluation. Track separately from the first named proposal adapter.
2. **3D simulation dependencies:** select and verify runtime/assets/material support in the pattern-derived 3D workstream below; physical comparison remains a separate gate for fit conclusions.
3. **Grammar extensions:** prioritize unsupported requirements observed in real use. Each extension changes a pinned library version and adds geometry/export/regression fixtures; it cannot silently rewrite historical projects.
4. **Manufacturing input:** obtain technical feedback on generated artifacts once they exist. Interviews can run alongside implementation, but are not the next development gate.

## Pattern-derived 3D workstream

**Status:** planned, 2026-09-16. **Authority:** [project contract §14](../design/project-contract.md#14-pattern-derived-garment-3d). **First acceptance case:** the existing component shirt, including its collar, cuffs, plackets, tails and gathered frills; the plain shirt is an intermediate fixture, not completion. Broader garment support remains open. No runtime, delivery-date or hardware-cost promise precedes the feasibility measurements.

### V0 — Inspect integration and prove the difficult operations

**Owner:** engine/integration. **Dependencies:** current pattern and assembly schema inventory.

- Inspect actual emitted templates, cut quantities, layers, edge IDs and assembly ratios. Record missing semantics before choosing an adapter; do not assume existing seam validation supplies 3D placement or solver-ready stitching.
- Evaluate the pinned upstream optional Warp path first. Compare adapting it with another maintained cloth engine if capability or runtime gates fail. Building our own general cloth solver is not the default; a browser viewer alone does not solve assembly.
- Verify code and asset licenses, installation, supported host hardware, headless operation, bounded memory/runtime and isolated execution. Use cleared synthetic body fixtures; record whether GPU infrastructure is required. Do not provision paid compute without authorization.
- Exercise two sewn panels, a gathered strip, a folded/layered collar and overlapping closure. Measure meshing error, seam residuals, strain, collisions, runtime and peak memory. Document unsupported constraints and the adapter work needed for the full shirt.

**Exit:** reproducible fixtures and a recorded engine choice backed by actual outputs, with explicit tolerances and resource budgets. A failed candidate triggers an alternative evaluation or a clearly scoped unresolved gate, not an invented garment render. Keep evidence in the existing research/review directories; keep sequencing here.

### V1 — Versioned assembly input and pattern-to-mesh adapter

**Owner:** contracts/engine. **Dependencies:** V0 findings.

- Extend `packages/contracts` with source-bound physical instances, mesh mappings, oriented seam intervals, registration points, folds, closures, layers, material assumptions and initial placement. Use schema migrations without injecting new defaults into historical revisions.
- Implement trusted conversion under `services/engine`: normalize units, tessellate real outlines, expand mirrors/counts/folds and retain rest-space and edge provenance. Record the cut-line/stitch-line/allowance policy.
- Separate immutable rest geometry, initial arrangement and solver positions. Validate each transformation and reject degenerate, missing, duplicate or invented instances.

**Exit:** metric and boundary checks across all six synthetic shirt sizes; each physical fabric instance maps to its template and role. Unequal gathered seams preserve source lengths. Deleted pieces, reversed seam direction, incorrect mirrors, bad fold expansion and unit-scale mutations fail independently of rendering.

### V2 — Durable work and assembled-garment artifacts

**Owner:** backend/engine. **Dependencies:** V1; V0 runtime decision.

- Fix interpretation's synchronous proxy-timeout path using durable submission and progress polling, preserving proposal-source concurrency checks. Reuse applicable lifecycle primitives for meshing/assembly/simulation, with separate measured budgets by job kind.
- Persist revision/input identity, idempotency keys, bounded queue stages and actionable errors. Exercise retry, reload, process restart, cancellation, project deletion and late worker publication through existing fencing semantics.
- Run assembly and any required settling server-side. Store validated geometry, provenance and diagnostics as immutable private artifacts. Report partial scope when a selected operation cannot be represented; never hide omitted frills or collars.
- Distinguish raw placement from an assembled approximation and from simulated drape. Define stage-specific seam/contact/strain criteria based on V0 measurements; do not call floating panels an assembled shirt.

**Exit:** a request exceeding the proxy timeout returns promptly with a recoverable job ID. Duplicate requests cannot double-publish; stale/deleted/cancelled attempts cannot replace current results. The full component shirt produces source-traceable output or specific component-level failures, with no silent fallback. A successful assembled-preview gate requires all selected physical components and assembly relationships to pass its declared checks.

### V3 — Visual inspection as the main workspace

**Owner:** frontend. **Dependencies:** V1 artifact schema; V2 for real outputs. Development fixtures must be labeled.

- Add rotatable garment viewing, front/back/reset controls and a 2D/3D selector to `apps/web`. Selecting a physical piece highlights its template; selecting a seam reveals its partner and diagnostics. Provide a piece list, layer visibility and exploded arrangement for inspection.
- Keep the useful garment view and revision/status label above the fold. Put advanced diagnostics and assumptions in progressive disclosure; retain pattern downloads and sizing controls. Offer keyboard controls, accessible selection and a working 2D fallback when graphics are unavailable.
- Show queued/running/failed/partial states without replacing the last valid historical view or implying it matches new edits. Fast previews must still derive from the relevant generated patterns; speculative schematics remain separately labeled.
- Clear private scene resources on logout/deletion. Do not fetch external model assets, embed session credentials in URLs or expose derived geometry through public caches.

**Exit:** desktop/mobile, keyboard and graphics-failure checks cover selection, revisions, loading, errors and recovery. No AI call is needed to inspect a manually authored supported garment. Camera or exploded-view changes leave canonical pattern bytes unchanged.

### V4 — Material-aware drape and scoped validation

**Owner:** engine, with frontend. **Dependencies:** V0–V3. Basic settling needed by V2 may use this solver earlier; this milestone adds and validates the material behavior claims.

- Add explicit assumed or measured fabric properties with supported units, grain orientation and provenance. Model stretching/bending, contact and supported layer/closure behavior; disclose absent friction, thickness, seam bulk or other effects.
- Record body/pose identity and uncertainty; existing circumference sliders alone do not define a uniquely accurate body surface. Start with an explicitly synthetic mannequin and retain independent body privacy controls.
- Validate the full shirt across sizes and meaningful component/material variants. Establish bounded convergence behavior and expose penetration, excessive strain and seam residuals instead of silently repairing rest geometry.

**Exit:** reproducible simulated artifacts meet declared numerical checks across the supported matrix, including gathered frills and collar/placket layers. Sensitivity checks show material changes affect deformation while rest pattern geometry stays fixed. Digital drape remains explicitly unvalidated for physical fit; physical claims require scoped sewn-sample comparisons.

### V5 — Independent adversarial review and private release

**Owner:** independent reviewer plus integration. **Dependencies:** the claimed V1–V4 scope passes its exit criteria. A V3 preview-only release may precede V4 if clearly labeled and reported as partial completion of this workstream.

- Attack piece counts/mirroring, seam orientation, gathering, disconnected components, hidden layers, flipped normals, unit mismatch, tessellation drift, extreme valid inputs and solver non-convergence. Verify correspondence from source data, not screenshots alone.
- Attack model-authored instructions, malformed artifacts and resource limits; test cross-project access, pattern/body-redacted bundles, embedded textures/avatars, stale revision display and cancellation/deletion races.
- Run applicable application/engine/browser checks, docs validation and representative performance measurements in an isolated checkout. Retain source-bound artifacts, screenshots and findings in a focused implementation review; distinguish executed tests from this planned matrix.
- Resolve blocking findings and obtain an independent recheck. Deploy only under release authorization, using the existing private deployment procedure and preserved data. Verify live authentication, a fresh generation, 2D/3D identity, all exports and restart recovery; retain a rollback path.

**Completion:** a user can interpret or manually author a supported garment, inspect the same pattern-derived garment in 3D, identify its physical components, edit authoritative inputs and regenerate without losing history or privacy. Delivering a viewer or one plain-shirt demo does not close full component assembly, material drape, broader construction or physical-fit gates.

## Release gates and unresolved decisions

- No public app release before explicit authorization, API authorization tests and an actual hosting/isolation review.
- No AI provider or weight-dependent feature advertised before credentials/runtime/licensing and empirical behavior are verified. Keep secrets outside the repo and renderer.
- No actual-size cutting claim before dimensional/calibration and annotation checks; no fit/production claim without appropriately scoped physical evidence.
- No preset-only positioning or dropping unsupported custom intent to improve success statistics.
- No elapsed-time or price estimate asserted for the full build. Profile baseline jobs and cost bounded AI calls before committing to throughput, free quotas or public pricing.

Remaining implementation questions are recorded with owners and gates in the [planning adversarial review](../reviews/planning-adversarial.md). They do not prevent committing a planning repository. Completion of this document is not approval of every future dependency or permission to start a public marketplace.
