# Pattern-derived 3D engine implementation plan

**2026-09-17 · Active dedicated workstream plan · Implementation in progress, not deployed**

## Execution status

The isolated implementation checkout contains durable asynchronous interpretation (J0), a physical-inventory compiler, constrained source-preserving meshes, edge/source correspondences, mirrored fabric instances and self-contained GLB placement derivatives. Private inspection jobs now connect those derivatives to an interactive source-linked viewport. The assembly graph records physical seam memberships, localized closures, free boundaries, dependencies and unsupported operation semantics. These foundations do **not** complete E1/E2: full solver-quality acceptance, executable turning/binding/orientation, fold/holes support and validated assembly remain open. No assembled garment or simulated drape is exposed in the application.

The inspection integration is a separately labeled intermediate capability. It does not satisfy the E3/E4 complete-shirt acceptance criteria. Schema-5 inspection jobs use independent leases, immutable pattern/runtime capture, checked private artifact installation, deletion/recovery cleanup and default exclusion from all exports. The viewport provides rotation, zoom, source selection shared with 2D, explicit placement-only classification, outdated-source warnings and WebGL fallback. [Independent review](../reviews/3d-inspection.md) records its narrower scope and checks.

The original full shirt has 18 templates, 24 shell/facing fabric instances and six unresolved interfacing roles. The independent fixture ledger is `packages/test-fixtures/assembly-expected.json`. Tail geometry is integral to the back pieces. The compiler binds the exact pattern bytes and explicit construction to digests, keeps seam-line geometry unchanged and reports omitted allowances and assembly capabilities. It rejects missing selected pieces rather than adapting construction to the remaining inventory.

E0 evaluates an independently licensed CPU solver after finding the upstream optional Warp fork restricted to non-commercial research. The [feasibility evidence](../research/3d-solver-feasibility.md) records executed fixtures and remaining gates. The [foundation adversarial review](../reviews/3d-engine-foundation.md) records verified fixes and the narrower reviewed implementation scope. Neither is release or full-engine approval.

J0 introduces separately fenced interpretation jobs and prompt HTTP 202 acknowledgement. Reload/polling, cancellation, captured-input conflicts and one bounded interrupted-attempt retry preserve existing geometry jobs. The combined implementation uses schema 5, extending schema 4 with independent inspection records; rollback must retain a schema-5-capable binary. Production data is untouched by isolated checks.

Integration verification, 2026-09-17: **104 application tests and 26 engine cases passed** in full runs. **23 browser cases passed** across the full run and focused reruns after fixing a mobile tab-layout regression. Both new 3D browser cases passed again against the final build, covering real private installation, reload/transient failure recovery, linked selection, unchanged exports, mobile layout, deletion cleanup, WebGL-unavailable fallback and context-loss recovery. Typecheck, build, documentation and whitespace checks passed. Numerical experiments and their independent reviews are recorded separately; these application checks do not validate assembled drape.

Next gate: close solver feasibility on the complete source-derived shirt, including quality meshing, executable assembly semantics and accepted material/placement profiles. Only after those gates should assembled inspection and drape be enabled. The integrated flat GLB must not substitute for these milestones. Retain failed full-shirt experiments and unchanged numerical acceptance criteria; successful rendering cannot promote them.

Continuation, 2026-09-17: a separate cut-domain mesher now retains the actual allowance fabric and source-derived interior stitch paths, with independent topology/source/path validation. All 18 shirt templates are covered by its synthetic fixture. The experimental weighted solver adapter now samples the exact embedded source arcs, validates sparse registrations and stages closure while retaining cloth rest geometry; it does not substitute nearest-vertex welds. Adversarial review rejected duplicate physical registrations, concavity-crossing source supports and inverted topology. Contact controls isolate instability on an unsewn front panel. Current-contact source-point locality replaces unsafe whole-primitive exemptions in a separate Newton experiment and retains layer/fold response controls. Neither experiment closes E0 or provides executable full-shirt sewing. The [cloth/contact review](../reviews/3d-cloth-contact.md), [embedded sewing review](../reviews/3d-embedded-sewing.md) and their numerical ledgers retain the measured scope.

Latest regression verification: **104 application tests, 28 engine cases across the full and focused runs, and all 23 browser cases passed**. The new engine cases execute six allowance/path tests and the rigid-placement regression. Seven separate contact-research tests pass, including a one-step layer-response control. Typecheck and build pass. Full-shirt simulation remains rejected; the quality-refined contact trial exceeds its resource budget. These checks support committing the foundations, not deploying an assembled garment engine.

Further continuation: exact embedded sewing, opposed rigid probe placement and pointwise material-local contact controls are implemented as research primitives. Correcting conflicting circumcentre batches removes the refined body's artificial micrometre edges; fixing exact duplicate collar boundary coordinates closes the observed memory-allocation loop. The final stationary refined-front control completes in seconds, and all 18 registered templates pass the canonical synthetic preflight. Independent rotated-collar probes still fail closed on degenerate triangles or precision-limited subdivision; this is not general rotation-invariant meshing acceptance.

Current regression evidence: **104 application, 31 engine and 23 browser tests pass**, with the six meshing/placement/coupling wrappers rerun after the collar fix. The new numerical ledger and [embedded sewing review](../reviews/3d-embedded-sewing.md) retain failed gathers, source identities and bounded diagnostics. Opposed gathered cut cloth still has excessive deformation, unsettled motion and surface contacts despite tiny seam residuals. The old full-shirt boundary-spring recipe remains rejected. The next implementation gates are executable allowance folds/layer order and sewing/contact coupling that cannot introduce penetration after the contact solve, followed by convergence and full-shirt validation. Do not respond to these failures by relaxing acceptance or excluding entire seams from collision.

Latest sewing continuation: the optional research harness connects all 24 cut-cloth instances through 305 source-validated embedded constraints. Adversarial review found and corrected omitted spring coloring, float32 membrane-area cancellation, aliased pre-step velocity state, exterior refinement slivers and rounded whole-edge endpoint handling. The corrected no-contact full-shirt run remains finite but reaches 11.35 times rest edge length and a 28.35 mm maximum seam gap. The final contact run also completes, but reaches 22.57 times rest edge length with a 5.24 mm maximum gap; both remain rejected. Earlier embedded trajectories used flawed velocity reconstruction and cannot substantiate full-substep dynamics. Current source-bound controls and limitations live in the [stability ledger](../research/3d-sewing-stability-results.json) and [embedded sewing review](../reviews/3d-embedded-sewing.md). Sewing/contact coupling, explicit layer/fold execution and convergence remain the next gates; a finite result does not authorize release.

Continuation, 2026-09-17: experimental embedded sewing can now run inside the VBD elasticity/contact iteration instead of bypassing collision truncation with an external projection. Independent analytical and finite-difference checks validate its sparse forces, while stiff free-particle controls exposed coordinate-descent drift; an optional augmented solve corrects those tested linear cases without changing physical compliance. The complete shirt still fails nonlinear convergence and contact checks. Source-linked strain and mass-motion diagnostics, rejected/resource-limited controls, and the remaining placement issues are recorded in the [embedded sewing review](../reviews/3d-embedded-sewing.md) and [coupled sewing ledger](../research/3d-coupled-sewing-results.json). These experiments do not authorize assembled output or deployment.

Continuation, 2026-09-18: the offline global reference now includes optional conservative dihedral bending with source-derived rest geometry, coupled Gauss–Newton search and explicit bending-energy accounting. The 12-step torso control converges at every step. This is a numerical foundation, not E3/E5 completion: coupled contact, damping/material calibration, full-shirt settling and layer operations remain open. Current measurements and limits are maintained in the [solver feasibility evidence](../research/3d-solver-feasibility.md#elastic-bending-reference--2026-09-18).

## Authority, scope and document ownership

This is the implementation plan for Sew Computer's pattern-derived 3D engine and its application integration. It replaces the detailed E0–V5 section formerly embedded in the [software prototype plan](software-prototype.md); that document remains the overall product roadmap. The [project contract §14](../design/project-contract.md#14-pattern-derived-garment-3d) remains normative. Maintain execution details here rather than in a second competing 3D roadmap.

The owner agreed that fabric components must originate from actual 2D patterns. Flexible placement and physically modeled deformation are allowed; independently generated garment meshes and silent rest-shape changes are not. AI may interpret construction into reviewable data. Trusted geometry and simulation code executes it.

This document defines planned behavior and tests, not executed feasibility, simulation, fit or performance results. A dedicated engine means an application-owned assembly/compiler/validation layer around a selected cloth solver; writing a general physics solver from scratch is not the default.

## Deliverable and protected behavior

A person can generate a supported garment, rotate the corresponding assembled garment, select any physical fabric piece to locate its source pattern, inspect assembly problems, change construction inputs and regenerate the same revision-bound outputs. Material-aware drape is a further explicit quality level.

The complete component shirt is the first acceptance case: torso halves, sleeves, cuffs, sleeve-opening bindings, plackets/facings, collar stand/fall, curved tails and gathered frills when selected. A plain shirt or floating arrangement is an intermediate milestone. Six synthetic size fixtures are coverage cases, not representative population or fit evidence.

Preserve private authentication, manual authoring without AI, sizing assumptions, autosave, proposals, historical revisions, existing 2D geometry and every existing export format. Failures in 3D must not prevent saving, 2D inspection or existing printable-reference exports. Unsupported requested garments remain explicit. Legacy outputs remain unchanged.

Non-goals for this workstream: arbitrary garment topology, production fit certification, real-time body scanning, unconstrained mesh sculpting, manufacturing procurement, photorealistic AI substitutes, public hosting and implementing the full upstream AI research pipeline.

## Observed starting point

Inspected at application commit `9a674ef`; these are source observations, not solver readiness:

| Existing surface | What it supplies | Missing for 3D |
|---|---|---|
| `services/engine/shirt.py` | Sampled seam outlines, separate cut lines, named edges, cut quantities, grain/marks, assembly ratios and text instructions | Physical layer instances, machine-readable turning/folding/binding/closure operations, initial 3D placement |
| `packages/contracts/index.ts` and design schemas | Millimetre pattern geometry and template-level stitch/drafting records | Versioned simulation inputs and source-to-mesh mapping |
| `apps/api/jobs.ts` | Generation scheduling, leases, retry/cancellation and stale work handling | Verified independent job kinds, simulation resource budgets and stage-specific artifacts |
| Current editor and exports | 2D geometry, illustrative flats, private revision-bound artifacts | Assembled mesh renderer and inspection, 3D disclosure and history tests |
| Pinned upstream source lock | CPU drafting baseline | Verified simulator install, runtime hardware, licenses and compatibility |

Instructions such as sandwiching a frill into a placket seam or turning a two-layer cuff cannot be inferred merely by expanding every template stitch pair. Audit each current instruction against the new assembly representation.

The current compiler uses identical coordinates for some named left/right pieces and describes mirroring in prose. Quantity-two templates can mean shell/facing layers rather than mirrored sides. Interfacing trimming is underdefined. E1 must encode these distinctions in a versioned trusted enrichment recipe; unresolved physical dimensions cannot be invented silently. An assumed choice needs explicit provenance and acceptance where it changes construction.

## Architecture and execution boundary

Data flow:

1. Capture immutable pattern bytes, selected construction and revision identity.
2. Compile a versioned physical-instance inventory and explicit assembly graph.
3. Validate semantic completeness before meshing.
4. Tessellate each physical piece in rest space and retain source mappings.
5. Place pieces using trusted construction-specific recipes and declared body/pose.
6. Assemble and optionally settle with the selected solver.
7. Independently validate geometry, assembly and simulation diagnostics.
8. Atomically publish private artifacts and a scoped result report.
9. Render those artifacts in the browser; keep authoritative computations server-side.

Proposed module locations, to be reconciled with existing code before implementation:

| Location | Responsibility |
|---|---|
| `packages/contracts/assembly.ts` | Versioned runtime schemas, IDs, units, capabilities and result classification |
| `services/engine/assembly.py` | Trusted construction-to-physical-assembly compiler |
| `services/engine/meshing.py` | Source-preserving tessellation and seam sampling |
| `services/engine/simulation.py` | Solver adapter and trusted job entrypoint |
| `services/engine/simulation_validation.py` | Source correspondence and numerical checks independent of render output |
| Existing API store/jobs/export modules | Migration, authorization, lifecycle, disclosure and artifact installation |
| Existing web component structure | 3D viewport, piece list, linked pattern selection and diagnostics |

Do not create a second application database, public worker API or alternate pattern authority. The worker receives bounded attempt-local input; it does not receive application sessions, provider credentials or direct database access.

Verify process-tree termination, CPU/memory limits, filesystem and network confinement and credential absence on the actual host. Record unenforced controls and their release impact; trusted Python and supervision alone do not establish isolation.

## Canonical assembly and artifact records

The proposed schema names below describe required semantics, not existing implemented types.

- **AssemblyInput:** schema version, project/revision/pattern artifact identity and digests, selected construction digest, physical instances, constraints, body/pose/material references, placement recipe and numerical profile.
- **PhysicalPiece:** unique instance ID, source template ID, layer/material role, cut expansion rule, mirror/fold transform and handedness, rest-domain mapping, source edge/mark/grain references. Counts reconcile exactly with the declared physical inventory.
- **SeamConstraint:** stable ID, physical instance and source edge intervals on each side, explicit direction, registration anchors, treatment, material-side orientation, intended ease/gather distribution and assembly stage.
- **ConstructionOperation:** allowlisted operation kind, physical participants, dependencies, relevant fold/attachment paths, contact/closure state and explicit supported parameters. Free text is explanatory only.
- **MaterialInput:** named properties, units, provenance, calibrated or assumed status and supported solver interpretation. Record omitted effects.
- **BodyInput:** asset digest, license/provenance, units, pose, collision geometry and assumed/measured status. Circumferences do not uniquely reconstruct anatomy.
- **MeshArtifact:** immutable rest positions, triangles, piece IDs, source interpolation mappings, grain coordinates and boundary/constraint mappings; deformed positions are separate.
- **RunReport:** all input and runtime digests, adapter/solver/build versions, seed and numerical settings, stage, classification, completeness ledger, measured checks, thresholds and reason codes.

Every input affecting output participates in cache identity, including placement, body pose, collision thickness, material, meshing resolution and solver settings. Cache lookup remains authorization-scoped. A digest never grants access.

Old revisions without assembly semantics remain valid 2D artifacts. Deriving assembly for them requires an explicit new derivation record bound to those unchanged bytes; changed construction choices require a new revision. Do not inject defaults or regenerate historical exports on read.

E1 requires an independently authored expected inventory for each supported component selection: physical shell/facing/interfacing instances, mirror direction, source-versus-derived roles, seams, folds, closures, free edges, assembly schedule and exceptional multi-way memberships. Compare compiler output against this fixture rather than generating expected counts with the compiler itself. Underdefined interfacing dimensions must resolve or carry an explicit approximation/unsupported classification; they cannot count as fully simulated physical cloth.

## Pattern-to-mesh invariants

- The source is canonical pattern data, not a screenshot, PDF trace or AI mesh.
- Rest geometry uses millimetres at application boundaries. Explicitly convert solver units and glTF metres; test lengths, gravity, thickness and material-unit conversions separately.
- Expand already named left/right templates only according to their declared counts; do not mirror them twice. Distinguish identical fabric layers from mirrored pieces and interfacing.
- Unfold cut-on-fold templates with a recorded reflection and continuous interior join. A fold is not automatically a seam.
- Preserve sampled boundaries and any available curve semantics, holes and interior paths. Refinement of an already sampled curve cannot claim accuracy beyond the source.
- Use constrained tessellation with documented maximum edge length, boundary error and element quality. Reject self-intersections, degenerate faces, invalid indices and unbounded size.
- Retain rest-space coordinates and source interpolation mappings through remeshing and rendering simplification. Verify coverage/area and boundary correspondence, not only vertex counts.
- Never normalize panel sizes for display in the authoritative mesh. Camera scaling is separate.
- Classify seam-line shells versus cut-line cloth explicitly. If allowances are omitted in a preview, list that simplification and do not imply modeled allowance bulk. Full allowance modeling requires interior stitching paths and fold/contact handling.
- Source metrics remain unchanged during assembly. Strain describes deformation relative to those metrics; it cannot be erased by resetting rest lengths.

## Assembly semantics and difficult cases

| Case | Required representation | Rejection or limitation |
|---|---|---|
| Plain seam | Oriented interval correspondence and registration anchors | Wrong direction or unexplained length difference |
| Gather/ease | Unequal rest intervals mapped by explicit distribution and anchors | Shrinking the long piece's rest geometry |
| Multi-layer junction | Explicit participating physical layers and shared seam membership | Naive duplicate pairwise constraints that overconstrain a junction |
| Collar/cuff/facing | Layer roles, free boundaries, turning/fold state and attach sequence | Assuming cut count two fully describes layer construction |
| Sleeve-opening binding | Open sleeve edges, binding wrap/fold recipe and attachment paths | Accidentally sewing the intended opening shut |
| Placket overlap | Layer order, contact and localized button/closure constraints | Welding entire front edges to fake a closure |
| Hem/tails | Free curved boundary and declared hem treatment | Treating all free boundaries as missing seams |
| Interfacing | Explicit physical layer or disclosed effective-material approximation | Invisible omitted material claimed as fully modeled |
| Cut-on-fold | Expanded domain and fold provenance | Duplicate cloth along the fold axis |

Assembly operations form a validated dependency graph. Self-seams are valid when they reference different intervals of the same physical piece. Shared attachments require deliberate semantics; reject undeclared duplicate use rather than banning all many-to-one connections.

Initial placement recipes are trusted, versioned, construction-specific and editable as arrangement data. Use local frames and body landmarks with declared assumptions. Placement may rigidly transform pieces; any pre-drape deformation must be solver-derived and reported. Collision-free staging cannot secretly discard layers.

The LLM may propose allowlisted operations and explain ambiguity. It cannot supply executable code, validation authority, unbounded constraints or replacement vertices. Missing essential semantics must surface as a capability gap. Deterministic recipes for the supported shirt must work with model routes disabled.

## Solver feasibility and selection gate

Evaluate the pinned upstream optional Warp route first because of the existing geometry dependency. Do not assume it accepts the custom shirt format. If it fails a mandatory gate, compare one maintained alternative with equivalent fixtures before committing to integration. Record rejected capabilities and adaptations, not speculative compatibility claims.

Mandatory evidence:

- Exact repository/version and transitive code/asset/runtime licenses; distribution and use permissions considered separately.
- Headless supported installation on the actual host; CPU/GPU requirements, driver compatibility and memory limits.
- Two panels, unequal gathered strip, layered folded collar and localized overlapping closure executed from our rest geometry.
- Preservation of source mappings and material/grain data.
- Required seam, contact, self-contact, layer and fold behavior, including exclusions near seam constraints.
- Bounded failure/termination, cancellation and restart behavior.
- Measured wall time, peak memory, triangle/constraint counts and numerical checks on a fixed fixture set.
- Ability to produce a browser-loadable artifact without calling an image/mesh-generation model.

Existing permitted local compute may be used for the spike. New paid compute or externally transmitted private inputs require authorization. If no candidate passes, record the blocked capability and a bounded adaptation proposal; do not silently switch to decorative 3D.

## Numerical quality and evidence classes

Three explicit outputs: **placement inspection**, **assembled approximation**, **simulated drape**. These are 3D result labels, not replacements for pattern export classifications. A full-looking image cannot upgrade them.

Before a milestone passes, store versioned thresholds and measured results for:

| Check | Measurement and independent oracle |
|---|---|
| Source fidelity | Rest boundary distance, area coverage, landmark/edge lengths versus canonical source |
| Physical inventory | Exact instance/layer/count reconciliation; no unused required instance |
| Seams | Registered point and interval residuals, direction, gather distribution versus assembly input |
| Material behavior | Strain, bend/contact behavior under controlled material perturbations |
| Collision | Body and cloth penetration with documented exclusions and detection coverage |
| Solver stability | Finite states, residual/energy or supported stability metric, bounded completion |
| Repeatability | Same pinned fixture/environment within declared numerical tolerances |
| Rendering | Units/orientation/selection and simplification correspondence to validated mesh |

E0 sets initial numerical profiles from measured fixtures, not arbitrary universal sewing tolerances. Require profiles to be reviewed and frozen before downstream acceptance. Changing a tolerance creates a new profile and reruns the matrix; it cannot retroactively bless a failed historical output.

Record maximum and percentile seam residuals as well as averages, and test rigid-transform invariance. Contact exemptions must be local to intended stitched neighbors; they cannot suppress all cross-layer contact. Test both body penetration and cloth self-intersection independently.

Bounded solver completion and numerical convergence are separate. A settled but inverted or body-intersecting garment fails the appropriate quality gate. Contact reports must state whether they sample vertices, surfaces or swept geometry; do not claim absence of collisions from a weaker check.

Physical-fit confidence requires separate sewn-sample evidence with material, body and construction scope. It is not a prerequisite for honest digital inspection.

## Durable jobs, identity and recovery

Fix the existing interpretation proxy timeout as an independently shippable prerequisite; it does not depend on solver selection. Reuse validated scheduling primitives, but first audit current assumptions about a single active job and project-wide generation counters. Starting 3D must not accidentally cancel unrelated interpretation or valid 2D work.

The existing geometry head is revision-scoped and current scheduling uses project-wide generation fencing. Introduce job-kind/input-scoped derived heads and explicit parent dependencies. A dependent job may consume only the committed artifact digest it captured; parent failure blocks it, and a late or replaced parent cannot silently supply different input. Preserve existing 2D head behavior. Replace fixed geometry timeouts only with measured per-kind profiles.

Use persisted submission with prompt acknowledgement and polling. Job-kind-aware stages include validating, meshing, placing, assembling, simulating, checking and installing; percentages only reflect measured work. Persist terminal result/reason separately from stage.

Idempotency binds request ID to owned project, job kind and full captured input digest. Reusing a request ID with different inputs conflicts. Unique attempt leases fence retries and late writes. Cancel after committed completion returns the existing result; cancel before publication invalidates the lease and reaps the process tree.

Publish staged artifacts only after checksum and validator checks and a transactional current-attempt/deletion authorization check. Recover orphaned files and interrupted installs without exposing partial artifacts. Historical results stay bound to their inputs; the viewer must not label them current after edits.

Bound queues, concurrency, triangles, constraints, decode sizes, simulation steps, memory, runtime, artifact bytes and retained attempts. Choose measured limits in E0; malformed inputs fail before large allocations. Record identifiers and timings without private measurements or raw model prompts.

## Privacy, browser and exports

All meshes, diagnostics, screenshots, thumbnails and downloads use existing authenticated access. Treat derived shape as dimension-revealing even if original measurement fields are removed.

Pattern exclusion removes 3D garment artifacts and derived diagnostics as well as 2D geometry. Body inclusion is separate: an avatar/collision mesh and body-derived metadata are excluded unless authorized. Explain that the garment itself may still reveal body proportions. Apply the policy to embedded binary buffers, textures, thumbnails and manifest fields.

Add an explicit 3D export opt-in, default off for existing clients. For new 3D artifacts, enforce the following matrix; it does not retroactively change existing 2D export bytes or promise that patterns are anonymous:

| Pattern inclusion | 3D inclusion | Body inclusion | Allowed 3D output |
|---|---|---|---|
| Off | Any | Any | None, including thumbnails and diagnostics |
| On | Off | Any | None |
| On | On | Off | Only a separately generated, explicitly synthetic-body or body-free result; withhold custom-body-conditioned garment meshes and all their derivatives |
| On | On | On | Requested custom-body-conditioned garment result; avatar inclusion still requires an explicit avatar selection |

Never relabel a custom-body result as synthetic by hiding its avatar. If no eligible synthetic result exists, omit the artifact with an explanation rather than silently recomputing or leaking it. Test every toggle combination, nested buffers and metadata. Update the versioned disclosure schema and normative contract before enabling these exports.

Prefer a self-contained glTF/GLB display derivative plus private canonical mesh/mapping/report artifacts; finalize encoding during feasibility. Reject external URIs, executable extensions and unbounded resources. Render derivatives never become source authority.

The first viewport supports orbit, front/back/reset, piece selection, layer visibility and exploded inspection. Selecting one template may highlight multiple physical instances, identified individually. Include keyboard-operable lists and controls, mobile sizing, accessible status text and a 2D fallback for unsupported or lost graphics contexts.

Clear in-memory scenes and temporary object URLs on logout/deletion. Historical output remains labeled. Material/shape edits trigger new captured inputs; camera/layer visibility changes do not.

## Ordered implementation and acceptance

| Milestone | Work and owner | Dependencies | Required exit evidence |
|---|---|---|---|
| E0: inventory and solver spike | Engine/integration: audit source semantics, run microfixtures, select runtime | Existing source | Capability matrix, pinned install, measured budgets, frozen numerical profiles |
| E1: physical assembly contract | Contracts/engine: instances, constraints, provenance, versioned migration | E0 semantics | Golden full-shirt assembly; mutation tests; legacy reads unchanged |
| E2: source-preserving meshes | Engine: tessellation, expansion, seam resampling, grain/source mapping | E1 | Six-size fidelity matrix, gather/fold/mirror fixtures and resource rejection |
| J0: async interpretation | Backend: durable jobs and recoverable proposal polling | Current app; independent of E0–E2 | Over-proxy-duration test, reload/retry/conflict/restart tests |
| E3: assembly worker and private artifacts | Backend/engine: solver stages, lifecycle, reports, private install | E0–E2; applicable J0 primitives | Full selected shirt assembly; failure injection, privacy and identity tests |
| E4: linked 2D/3D inspection | Frontend: real artifact viewer and diagnostics | E2 schema; E3 outputs | Desktop/mobile/keyboard, old revisions, graphics failure and piece mapping |
| E5: material-aware drape | Engine: supported material/contact/layer behavior | E3; E4 inspection | Full shirt material/size/component matrix and scoped numerical evidence |
| E6: adversarial review and release | Independent reviewer/integration | Claimed scope passed | Findings closed and independently rechecked; regression and production verification |

E3 may use settling from the chosen solver; E5 validates additional drape/material claims rather than requiring a second solver. E4 can be released as an assembled approximation only if its own full-component gate passes and its simplifications are visible. Such a release does not complete E5.

## Test matrix and adversarial obligations

Use deterministic synthetic source fixtures and independent calculations, not only the same code that produced the result.

1. Microfixtures: equal seam, reversed seam, unequal gather, nonuniform gather anchors, self-seam, multi-layer junction, fold expansion, opening binding, button overlap and curved/free hem.
2. Inventory: duplicate/missing instances, swapped handedness, doubled left/right expansion, omitted facing/interfacing, unknown material role and disconnected selected component.
3. Geometry: nonfinite values, degenerate triangles, concavity/holes, curved-edge approximation, unit conversions, grain rotation, mesh refinement and invalid cut/stitch boundaries.
4. Solver: timestep/resolution sensitivity, unstable material inputs, wrong normals, body penetration, self-contact, layer inversion, extreme valid dimensions and non-convergence. Never reset rest geometry to make these pass.
5. Lifecycle: provider delay past proxy timeout, duplicate submission, conflicting request identity, concurrent edits, different job kinds, crash at each install boundary, cancellation/completion race, lease expiry and deletion.
6. Privacy: direct unauthorized origin calls, cross-project IDs, cached assets, export redaction, hidden body data in binary assets, external texture requests and session expiry.
7. UX: current versus historical identity, controls tied to actual pieces, partial failure, mobile/keyboard/screen-reader inspection, graphics context loss and access to existing exports.
8. Compatibility: existing application/engine/browser suites, manual no-model flow and unchanged historical artifact bytes.

First pass focuses on the complete shirt and six synthetic sizes, with selected component variants and synthetic assumed materials. Preserve a fixture ledger of supported combinations; do not imply every Cartesian combination passed. Each reported pass identifies its exact input/profile/runtime.

Independent review must challenge the schema and source correspondence before implementation, then independently inspect generated artifacts and lifecycle behavior before release. A plan review is not a runtime validation.

## Release, rollback and definition of done

Release from an isolated checkout after applicable typecheck, application/engine/browser tests, build, documentation and whitespace checks. Include a source-bound artifact sample and review report. Use current private deployment procedures only under release authorization; preserve production data and credentials.

Schema changes must permit existing 2D operation and a tested rollback path. Before deploy, document whether rollback is application-only or requires a forward-compatible migration; never downgrade data blindly. Disable 3D job submission independently when necessary while preserving historical artifact access and 2D generation.

Production checks: private authentication, fresh synthetic full-component generation, 2D/3D identity, authorized/redacted exports, stale-revision behavior and worker recovery. Remove synthetic projects through the authenticated lifecycle after capturing permitted evidence.

Done for the declared digital scope means all selected physical components derive from source patterns, supported assembly/material semantics are explicit, numerical profiles pass, history/privacy/recovery checks pass, independent blocking findings are closed and the user can inspect and edit the garment through the normal flow. Broader garment support and physical fit remain separate open scopes.

## Open decisions with owners and closure gates

| Decision | Owner | Must close before |
|---|---|---|
| Solver/version and required hardware | Engine/integration | E0 exit |
| Cut-line versus seam-line representation and allowance detail | Engine | E1 acceptance; shown in every output |
| Layer turning, interfacing and binding approximation support | Engine | E1 full-shirt capability ledger |
| Numerical tolerances and resource limits | Engine/reviewer | E0 baseline, before accepting E2/E3 |
| Job-kind fencing and cache/disclosure schema | Backend | E3 integration |
| Cleared mannequin and pose data | Engine/privacy | First body-based run |
| Display format and simplification mapping | Engine/frontend | E2 artifact schema |
| Material property source and calibration scope | Engine | E5 |
| Physical validation protocol | Product/garment reviewer | Any physical-fit claim |

The next engineering action is E0 plus the independent J0 reliability fix. Planning does not itself authorize paid infrastructure, public hosting or claim the engine exists.
