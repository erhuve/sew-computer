# Pattern-derived 3D engine implementation plan

**2026-09-17 · Active dedicated workstream plan · Implementation in progress, not deployed**

## Execution status

Continuous tension-only candidate, 2026-09-24: a separate supplied-cell connector now integrates tensile separation with rigorous absolute response bounds and no compression push. Exact model review shows why a symmetric continuous distance law can resist curved offsets; the [new mechanics evidence](../research/3d-solver-feasibility.md#continuous-tension-only-sewing-candidate--2026-09-24) retains that limitation and the candidate's remaining collapse/side/contact ambiguities. **1,125 full pinned Linux numerical/supervision tests pass in 766.085 seconds**; thirty-six new focused tests also pass on macOS and Linux and overlap that total. Independent static source checks cover energy, forces and curvature. Next integrate fixed-parameter controls into the guarded generic solver, including numerical uncertainty in convergence, line search and work checks. Parameter work, source construction, stable motion, material/body validation and full-shirt acceptance remain unfinished; refined capture/replay authorization is still pending.

Continuous sewing mechanics, 2026-09-24: a standalone supplied-cell penalty now integrates material-normal vector offsets, frame reactions and full curvature through each declared cell. Independent checks catch deformation invisible to endpoint samples and retain a crease-director incompatibility counterexample. Adversarial precision repairs preserve tiny geometric residuals, forces and work. The [mechanics evidence](../research/3d-solver-feasibility.md#continuous-normal-offset-sewing-penalty--2026-09-24) records the exact numerical policy and its limits. A compatible construction joint model, source-authorized cell controls, guarded integration, joint space/time verification, stable construction and full-shirt acceptance remain unfinished.

Verification passes **1,089 full pinned Linux numerical/supervision tests in 755.048 seconds**, including **twenty new independent cases**. All twenty also pass in focused macOS and Linux runs; these counts overlap the full discovery. The full log is `.planning/solver/continuous-normal-full-numerical-sep24-v1.log`. The [mechanics ledger](../research/3d-continuous-normal-sewing-results.json) records frozen source/test identities, focused/full logs, adversarial failures and repaired precision probes. This is offline numerical verification, with no application build, release or deployment claim.

Continuous material-seam geometry, 2026-09-24: exact source-bound traversal and an independent auditor now retain all eight member-pair selectors and all forty original/numerical rows. A fixed-state evaluator checks every spatial cell, including interior collapse and gaps hidden between samples. The [spatial evidence](../research/3d-solver-feasibility.md#continuous-material-seam-geometry-and-fixed-state-bounds--2026-09-24) retains an actual-source constructed counterexample with approximately 0.488 mm interior deviation while the sampled rows remain unchanged before rounding. Material coordinates, stored numerical rest coordinates and deformed geometry stay distinct. These are static diagnostics; a physical continuous seam model, joint spatial/temporal verification, construction execution, stability/material evidence and full-shirt integration remain unfinished.

Previous spatial verification: **1,069 full pinned Linux numerical/supervision tests pass in 762.033 seconds**, including **41 new cases**: twelve producer/evaluator, thirteen independent-auditor and sixteen exact-bound tests. All forty-one also pass in focused macOS and Linux runs; these counts overlap the full discovery. Native source fixtures are regenerated on macOS; this does not establish cross-runtime bit identity. The retained audit reproduces byte-identically with only the standard library and six copied auditor modules. Independent review recomputes all 356 cells across the two retained states. The full log is `.planning/solver/binding-spatial-seams-full-numerical-sep24-v1.log`. Complete identities and scope are retained in the [spatial ledger](../research/3d-binding-spatial-seams-results.json).

Source-bound sewing frames, 2026-09-24: a separate static descriptor and independent auditor now preserve all forty rows while enumerating forty-two oriented, original-parent-bound candidates. Two rows remain ambiguous; every offset sign and all five textile-side declarations remain unresolved. The [frame evidence](../research/3d-solver-feasibility.md#source-bound-sewing-frame-correspondence--2026-09-24) proves that shared anchors can support orthogonal directors after deformation and that an all-body binding request would discard nonzero coefficients. No controls are installed or refined motion admitted. Construction-specific choices and mechanical validation, guarded capture/replay after authorization, stable construction, calibrated materials and full-shirt integration remain open.

Finer coupled-control dynamics, 2026-09-24: fixed-input controls through 1,024 intervals verify 2,017 accepted states but expose large relative layer opening and persistent motion. The [finer evidence](../research/3d-solver-feasibility.md#finer-mixed-control-motion-and-numerical-attenuation--2026-09-24) separates numerical attenuation from absent material damping, preserves early engagement/relative-motion differences, and retains the adaptive 128 partition. Neither low coarse-run speed nor tiny held-distance error establishes convergence or a completed attachment. Source-frame correspondence, stable construction motion, calibrated materials and full-shirt integration remain unfinished.

Fixed-input timestep sensitivity, 2026-09-23: the coupled synthetic control completes 32/64 uniform steps and a requested 128-step run with one adaptive bisection. Halving nominal timesteps still changes final angles by about 2.7°; the successive endpoint discrepancy does not decrease. The [temporal evidence](../research/3d-solver-feasibility.md#fixed-input-mixed-control-timestep-comparison--2026-09-23) retains the failed smaller-budget run and every actual partition. Convergence, settling and construction acceptance remain unresolved. No numerical source, force tolerance or garment admission changed.

Coupled sewing/fold/contact mechanics, 2026-09-23: actual generic cloth controls now combine held/pending sewing with fold ramps and release, including nonzero interlayer activation-buffer contact. The [coupled evidence](../research/3d-solver-feasibility.md#coupled-sewing-fold-and-contact-mechanics--2026-09-23) retains two complete thirty-two-step runs, exact path checks and unresolved final motion. Adversarial review repaired missing final sewing diagnostic/work checks and mutation of the numerical sewing model; valid motions reproduce byte-for-byte. Independent next work is temporal/spatial sensitivity and construction validation, with refined capture/replay still awaiting authorization. This does not execute or accept a garment phase.

Combined source controls, 2026-09-23: a [static sewing/fold declaration](../research/3d-solver-feasibility.md#combined-source-sewing-and-fold-declarations--2026-09-23) now preserves all forty original/numerical rows, complete reference targets and compliance alongside the sixteen fold controls. Five samples remain held and thirty-five pending on one original fraction/retry grid. Exact original-versus-derived geometry witnesses retain tiny differences between the numerical operators without retargeting; independent standard-library audit checks the composition. No refined motion executes. Authorized captured integration/replay, actual wrapping/turning, continuous spatial seams, stability/material evidence and full-shirt application acceptance remain open.

Previous frame-correspondence verification: **1,028 full pinned Linux numerical/supervision tests pass in 650.506 seconds**, including **26 new cases** (thirteen producer and thirteen independent-auditor tests). All twenty-six also pass on macOS and in focused Linux runs; these counts overlap the full discovery. The retained unresolved descriptor reproduces byte-identically in an isolated standard-library-only audit. Independent review checks all forty-two candidates and rejects twenty-eight malformed requests/descriptors. Complete identities, logs and the corrected raw-file/canonical-JSON hash comparison are in the [frame ledger](../research/3d-binding-frames-results.json); the full log is `.planning/solver/binding-frames-full-numerical-sep24-v1.log`. These are offline static and numerical checks, with no application release or deployment.

Previous mixed-control verification: **1,002 full pinned Linux numerical/supervision tests pass in 596.167 seconds**, including **fifteen new cases** (six actual-mechanics and nine publication-validation tests). All fifteen also pass on macOS and Linux in focused runs; these counts overlap the full discovery. The [mixed-control ledger](../research/3d-mixed-control-results.json) records complete logs, source and artifact identities, earlier failed development checks, and the independent saved-state review. The full log is `.planning/solver/mixed-control-full-numerical-sep23-v1.log`. This offline increment changes no application or deployment behavior.

Previous combined-declaration checkpoint: **987 full pinned Linux numerical/supervision tests pass in 601.119 seconds**, including **24 new cases** (twelve producer and twelve independent-auditor tests). All twenty-four also pass on macOS using fresh native fixtures; this does not certify cross-runtime source bit identity. The retained Linux declaration matches independent joint parameters at all **16,385 grid fractions across sixteen hinges and forty sewing rows**. Directed review checks all row identities/hashes and exact held-row witnesses, including tiny differences and four rejected forgeries. The audit reproduces byte-identically in an isolated standard-library-only process. Complete logs and artifact identities are in the [combined-control ledger](../research/3d-binding-sewing-schedule-results.json); the full log is `.planning/solver/binding-sewing-schedule-full-numerical-sep23-v1.log`. No refined motion, source-admission change or application release is included.

The entries below preserve earlier checkpoint findings; their next-step statements describe the work remaining at that time. Follow the current handoff for execution order.

Source-bound temporal fold declarations, 2026-09-23: a static request now binds both source rails to all sixteen ordered native controls and the complete audited placement. It separates temporal angles from old static reference targets, retains engagement/release and physical-time witnesses, and rejects positive activation/coefficient underflow over its retry grid. Independent rederivation and the [source-control evidence](../research/3d-solver-feasibility.md#source-bound-temporal-fold-declarations--2026-09-23) preserve the distinction between parameter validation and execution. Existing grippers conflict with the constructed internal-fold reference and remain excluded. The combined static declaration above completes this binding; authorized captured integration/replay, actual construction and garment acceptance remain unfinished.

Guarded fold activation and release, 2026-09-23: independent per-hinge controls now enter the generic direct solver and adaptive retry path, with complete hinge/cloth/contact guards retained through release and full parameter work checked before journal acceptance. A two-layer synthetic control completes 32 steps over 64 ms with sequential release and nonzero contact response; its final 77 mm/s speed explicitly fails any settled-pose claim. See [motion and verification scope](../research/3d-solver-feasibility.md#guarded-fold-activation-and-release--2026-09-23). Source-bound construction schedules, refined capture/replay integration after authorization, actual binding/turning and garment validation remain open; this does not execute the refined garment profile.

Explicit fold schedules and mechanics, 2026-09-23: a separate immutable per-hinge schedule supports independent angle ramps, holds, reversal and release. The new mechanical primitive preserves full native hinge identity, evaluates active contributions only and accounts separately for target work, activation work and released actuator energy. Adversarial fixes retain shared-vertex force cancellation and tiny angle changes without changing rest geometry. See [scope and evidence](../research/3d-solver-feasibility.md#explicit-fold-schedules-and-mechanics--2026-09-23). At that checkpoint the primitives were standalone; the subsequent generic integration above adds global/adaptive work and retry checks. Source controls, the captured runner and main replay remain separate. Next bind a construction-specific temporal recipe to the audited source/native/placement identities and execute captured guarded motion after the pending entry-point authorization. Construction and garment acceptance remain open.

Source-bound refined placement, 2026-09-23: all five fabric instances now receive explicitly declared proper-pose representations evaluated by exact binary-input affine arithmetic and one rounding per coordinate. Independent audit checks all 245 vertices, 638 edge-error decompositions, 398 oriented triangles, raw refinement lineage and ten exact cross-instance separating planes. The [retained placement evidence](../research/3d-solver-feasibility.md#source-bound-refined-placement--2026-09-23) discloses the one original vertex that differs from historical BLAS evaluation and preserves unchanged gripper targets. This is a new static numerical input, not trajectory equivalence. Explicit temporal controls and captured guarded motion remain next; normal-offset sewing frames, construction execution, settling/material/refinement validation and full-shirt integration remain open. The rejected entry-point proposal remains unapplied.

Source-bound binding fold controls, 2026-09-23: explicit rail direction and topological body/allowance regions now bind all **sixteen** crease segments to the actual native ordered hinges. Declared external stiffness density is integrated over stored segment length with independently checked rounding. A standard-library auditor and existing-primitive binder verify source/topology/control correspondence. The [retained static sequence](../research/3d-solver-feasibility.md#source-bound-binding-fold-controls--2026-09-23) reaches both declared relative angles using a transported finish axis; reusing the flat axis introduces measurable distortion. These constructed poses execute no dynamics. The subsequent placement increment supplies a separately checked starting state; captured guarded motion, normal-offset sewing frames, construction phases and material/refinement validation remain open, and the entry-point proposal is still unapplied.

Binding gripper migration, 2026-09-23: a separately validated descriptor now rebinds original point controls through the refined strip's exact source coefficients. The three existing grips retain their complete target/release schedule, stiffness and identities; unchanged instances rebuild canonical triangle indices. A separate standard-library auditor reconstructs lineage, candidate selection, conversion residuals and control correspondence. The [evidence and counterexamples](../research/3d-solver-feasibility.md#binding-gripper-migration--2026-09-23) distinguish exact coefficient pullback from tiny stored-coordinate differences. This derives controls without installing them or executing a refined trajectory. Normal-offset sewing frames, capture/replay integration and mechanical validation remain open.

Binding anchor remapping, 2026-09-23: exact coefficient-space selection and explicitly rounded child weights now preserve immutable original rows alongside a strictly derived **245-vertex / 398-triangle** numerical input. Independent audit reconstructs raw lineage, rounding and all forty numerical rows. The [implementation boundary](../research/3d-solver-feasibility.md#binding-anchor-remapping--2026-09-23) records a concrete wrong-side geometric-remapping counterexample. Runner/main-replay integration is still unapplied after automatic approval review rejected those two entry-point edits for insufficiently specific authorization. No refined trajectory is claimed. Finish the reviewed integration after authorization, then verify a controlled refined motion using the separately bound gripper/fold inputs before fold/wrap execution. Normal-offset sewing frames remain a separate requirement.

Longer binding hold and crease preprocessing, 2026-09-23: both 1,024 ms controls independently replay all 512 steps and exactly reproduce the unchanged first 384 ms. The longer hold does not settle the cloth: final speed remains about 0.478 mm/s and the released-tail angle range increases. The [retained comparison](../research/3d-solver-feasibility.md#extended-binding-hold--2026-09-23) keeps this failure explicit. A separately reviewed, non-executable [two-crease descriptor](../research/3d-solver-feasibility.md#binding-crease-preprocessing--2026-09-23) retains full cut allowances, original vertices, parent triangles and rational interpolation lineage. Next bind derived seam anchors and explicit crease-side frames without changing the original forty rows, then connect grippers, capture and replay before fold/wrap execution. Material/settling validation remains open.

Fixed-timestep binding control, 2026-09-23: a declared fourfold slower path retains the same 2 ms step, source geometry, stiffness and target recipe. Both 256-step controls independently replay. Peak turn falls from 4.485° to 3.165° and final speed from 17.87 to 0.477 mm/s, but relative motion remains through release. The [timing comparison](../research/3d-solver-feasibility.md#fixed-timestep-binding-time-control--2026-09-23) records the improvement without accepting a settled pose, a binding phase or convergence. Diagnose residual relative motion separately from rigid drift; source-preserving crease refinement and explicit wrap/finish/apex semantics remain independent prerequisites.

First source binding motion, 2026-09-23: matched zero/3° controls retain the complete five-fabric unit, hold five original sampled rows, leave thirty-five pending and execute tool motion, hold, release and passive continuation. Both replay all 64 steps. The measured relative turn overshoots to 4.485° and ends moving; no binding phase or settled pose is accepted. The [ledger and remaining gates](../research/3d-solver-feasibility.md#first-source-binding-motion--2026-09-23) separate exact path evidence from source-frame motion, full-surface separation and uncalibrated material diagnostics. Next resolve tracking/settling before extending the motion into a source-preserving wrap with explicit allowance and apex handling.

Sampled seam paths, 2026-09-23: an opt-in [exact replay gate](../research/3d-solver-feasibility.md#sampled-sewing-path-bounds--2026-09-23) bounds scalar-distance error throughout each affine saved-state path using a declared tolerance and endpoint-target reference. Pending rows retain identity; positive weights cannot hide geometric error. This closes a temporal gap for sampled rows, while spatial stitch coverage and construction semantics remain unfinished.

Previous checkpoint verification: **963 full pinned Linux numerical/supervision tests pass in 487.079 seconds**, including **25 new cases** (twelve producer and thirteen independent-auditor tests). The new cases also pass on macOS using freshly generated native fixtures; this does not establish cross-runtime source bit identity. The retained Linux request matches independent parameter calculations at all **16,385 grid fractions across sixteen hinges**. Separate review passes 220 small-grid comparisons and retains fifteen activation-underflow rejections. The saved audit reproduces in a standard-library-only process. Log: `.planning/solver/binding-fold-schedule-full-numerical-sep23-v1.log`. These checks execute no refined motion and do not certify an application release.

Captured seam controls, 2026-09-23: explicit targets and monotone activation now bind to the complete source JSON and all ordered sewing rows. The cuff path additionally rederives the five-fabric source unit, its complete embedded registrations and phase-row partition using captured engine dependencies. Adaptive retries preserve original control fractions, record engagement work before acceptance and retain unweighted active errors. Current independent replay reconstructs controls, sewing forces, normal-frame reactions and discrete work. See the [captured-control scope](../research/3d-solver-feasibility.md#captured-seam-controls--2026-09-23). This connects numerical controls; the phase plan remains unexecuted. Construction-specific binding/turning paths and witnesses, continuous seam coverage, material-side/interfacing decisions, refinement/material/body validation and full-shirt application integration remain open.

Seam activation primitives, 2026-09-23: the direct offline solver now applies explicit per-row weights to vector, distance and material-normal sewing while retaining every canonical row. Pending rows exert zero force and stiffness; diagnostics report active errors without weighting them down. A separate monotone schedule and endpoint energy helper account for target changes followed by engagement at the previous positions. See the [implementation and verification boundary](../research/3d-solver-feasibility.md#pending-seam-mechanics--2026-09-23). At this primitive checkpoint, captured source binding, adaptive execution, CLI controls and independent replay remained the next gate; the captured-control continuation above now connects them. Adaptive callers still reject uncaptured activation. This checkpoint does not execute any cuff phase.

Compliant material controls, 2026-09-23: the offline solver has opt-in source-triangle grippers with captured engagement/movement/release schedules, external force/reaction and discrete energy accounting, adaptive rollback, durable per-transition evidence and independent gripper replay. The final Linux numerical/supervision discovery passes **514 tests in 203.919 seconds**. A retained two-layer contact control completes and independently replays eight states. See the [control scope](../research/3d-solver-feasibility.md#compliant-source-material-controls--2026-09-23). These controls preserve cloth rest geometry and retain existing motion guards. They do not execute cuff turning or bindings. The later captured-control continuation adds per-member seam activation. Construction-specific paths, material-side/interfacing semantics and continuous seam coverage remain required before executing the five-fabric plan.

Numerical adversarial continuation, 2026-09-23: review repaired nonlocal rest-filtered contact scales, missing all-face swept nondegeneracy and platform-dependent tiny fold-energy differences. The Linux ARM runtime also omitted candidates under the native default/explicit `LBVH`; all contact, preflight, supervised and replay paths now explicitly select profiled `HashGrid`. Independent exact-rational replay verifies candidate coverage and nonzero triangle area. The complete Linux numerical/supervision suite passes 427 tests, with five later adversarial coverage/replay tests and six source-generator tests passing separately. Two new synthetic cuff controls replay 373 states and 126,686 contact-proof leaves; the earlier defective-broad-phase capture rejects. A 192 ms hold after an unchanged 64 ms ramp reduces maximum speed about 78-fold, while facing angles remain approximately 0–39° against a 68.75° target. See the [review and limits](../reviews/3d-cloth-contact.md#numerical-continuation-review--2026-09-23) and [hold evidence](../research/3d-solver-feasibility.md#verified-cuff-hold-control--2026-09-23). Next, isolate coupled seam/fold constraints, establish timestep/refinement stability, then execute actual cuff turning and attachment. Full-shirt, material and application integration acceptance remain open; nothing deployed.

Explicit cuff frame binding, 2026-09-23: source review traced facing-only fold resistance to incidental child-triangle selection at the crease. The extractor now requires an explicit body/allowance region when both support the same seam anchor, records source primitive identities, and remains invariant to child enumeration. Eighteen focused Linux source/frame tests pass, including independent constructive and torque regressions. Body binding changes only seven frame triangles and preserves source dimensions, seam coefficients, fold targets and schedule. Three body-frame holds now replay 801 states. Keeping the 0.1 mm hard separation fixed while narrowing the contact activation buffer to 0.1/0.05 mm brings both layers to approximately 67.4–68.4° against a 68.75° target, with low source-edge distortion. This is an explicit model-sensitivity result, not calibration or garment acceptance; see the [new evidence](../research/3d-solver-feasibility.md#explicit-frame-and-contact-buffer-controls--2026-09-23). A half-timestep control also replays 513 states; final maximum position/hinge differences are 0.174 mm / 0.0284° at matching saved fractions. Continue temporal and spatial stability work and execute the source attachment/turning sequence; this two-resolution comparison is not asymptotic convergence.

Source cuff phases, 2026-09-23: a separate declarative research policy now distinguishes initial facing attachment, perimeter stitching, turning, attachment-allowance treatment and final shell attachment. It explicitly refines four source dependencies while preserving original membership and keeping the turning opening reserved. A fresh generator includes all five fabric instances and both binding prerequisites, with 40 unchanged source-derived sampled constraints. Binding/turning execution, material sides, interfacing, continuous seam coverage and source pulling controls remain unresolved. See [phase semantics and reproduction](../research/3d-solver-feasibility.md#source-cuff-construction-phases--2026-09-23). Neither metadata correspondence nor a declared phase grants execution or garment acceptance.

Coupled sewing/fold continuation, 2026-09-23: the offline solver now executes captured stages without resetting position, velocity, seams or contact. The cuff's twenty source anchors are remapped into the allowance mesh's original-parent child triangles, producing a 66-vertex two-layer diagnostic. Exact material-normal sewing curvature improves convergence on the 0.5 mm offset / 1.2 rad control: 39.01 CPU seconds versus 398.86 with the previous search metric. Both schedules complete, but adaptive partitions and final geometry differ; this is not equivalent-trajectory acceleration or timestep acceptance. The facing still underfolds. Tight 0.11 mm controls exhaust 600-second CPU budgets; stronger wide-gap actuation stops at 94.14% on the 256-attempt budget. See the [staged execution and evidence](../research/3d-solver-feasibility.md#staged-sewing-and-folding--2026-09-23). Next, resolve coupled fold tracking and timestep sensitivity, then implement turning through the source attachment opening and binding. Full-shirt, material and independent-review gates remain open. Nothing deployed.

Allowance-crease continuation, 2026-09-22: the offline cuff extractor can subdivide the cut mesh along its recorded straight outer stitch line, explicitly extending through both side allowances to the cut boundary. Original vertices, parent triangles and interpolation weights are retained; the source metric and outline are preserved. Three signed/near-closed controls complete on twelve allowance-aligned hinges; replay verifies 48 states and 3,668 exact path-proof leaves. The strongest reaches 176.8–177.7° with active contact and up to 0.720% edge compression. The broader numerical run passes 378 tests, with final preprocessing checks separately passing. It does not execute sewn shell/facing turning, corner treatment, binding or full-shirt assembly. See the [allowance control](../research/3d-solver-feasibility.md#source-cuff-allowance-crease--2026-09-22). Next, couple source-preserving allowance folds to shell/facing seams and an explicit turning opening; retain timestep/refinement, material and independent-review gates. No deployment is enabled.

Prescribed-fold continuation, 2026-09-21: explicit source-hinge actuators now execute signed fold schedules alongside unchanged cloth elasticity, with adaptive timing, energy/work accounting, physical-path guards, supervised capture and replay. The actual cuff shell completes three centerline-fold controls; the strongest reaches 177.2–178.4° with active self-contact, under 0.0075% edge-length deviation and 16 replay-verified states. This is a single-shell diagnostic crease, not the construction's allowance fold or cuff turning. See [fold evidence and reproduction](../research/3d-solver-feasibility.md#completed-source-cuff-fold-controls) and the [geometry figure](../research/3d-source-cuff-fold.png). Next, map actual construction fold lines into the source-preserving mesh and couple folds to shell/facing assembly and the declared turning opening. Binding, full-shirt closure/settling, material/timestep/refinement acceptance and independent review remain open. No deployment or assembled application output is enabled.

Material-normal sewing continuation, 2026-09-20: explicit seam offsets now follow source-triangle rotation and favor the declared registration side, with balanced frame reactions. The isolated source-cuff schedule completes and replay verifies nineteen states and 6,215 exact contact proof leaves. This remains parallel registration, not a turned cuff. See the [current model, measurements and limitations](../research/3d-solver-feasibility.md#material-normal-seam-offsets--2026-09-20). Next, represent allowance folds, turning openings and attachment order explicitly and validate their execution. Full-shirt assembly, binding, settling, swept frame nondegeneracy and independent contact-model review remain open. No application simulation or deployment is enabled.

Source-cuff continuation, 2026-09-19: the actual 40-vertex cuff shell/facing completes a prescribed parallel-layer offset ramp using its 20 source-derived interior seam registrations. Eighteen accepted substeps and two rejected attempts are preserved; replay checks all states and 6,522 exact proof leaves. Final maximum anchor gap is 0.136156 mm for a 0.11 mm target. The [cuff control](../research/3d-solver-feasibility.md#actual-source-cuff-offset-sewing-control--2026-09-19) is a complete diagnostic schedule, not a turned cuff, sleeve attachment, settled garment or deployment approval. Garment-wide layer execution remains the next substantive implementation gate.

Per-pair temporal continuation, 2026-09-19: the rest-filtered model now supports continuous support certificates at its fixed per-pair thicknesses, retaining the original optimizer limiter and physical contact law. The same eight-step shirt interval completes with identical final positions/velocities in 71.28 CPU seconds versus 166.57 previously; implementer replay verifies 58,763 exact proof leaves. See the [source-bound result](../research/3d-solver-feasibility.md#per-pair-temporal-certification-for-rest-filtered-contact--2026-09-19). This closes the per-group certificate integration task, not garment assembly or the independent contact-model review gate.

Contact-model continuation, 2026-09-19: an opt-in rest-filtered barrier removes artificial rest force on the tested refined grids and the full-shirt stationary control while retaining all collision candidates. It explicitly assigns smaller positive thickness to selected within-panel primitive pairs; separate panels retain full thickness. A synthetic two-panel perimeter seam completes a positive-offset closure with active contact. The new model also completes the eight-step full-shirt 1% target-reduction interval at 1 mm activation; replay verifies every state, with a maximum residual of 9.276e-7 N and zero detected endpoint intersections. This is a new uncalibrated model and a narrow sewing control, not full-shirt assembly or refinement convergence. See the [implementation and remaining review gates](../research/3d-solver-feasibility.md#rest-filtered-contact-model-investigation--2026-09-19). The next integration gate is adversarial locality/derivative review, temporal/refinement controls and consistent per-group temporal certification, followed by source-bound layer operations and full closure.

Release assessment, 2026-09-19: the isolated `f6680f5` application passes a fresh **104 application tests, 34 engine tests and 23 browser tests**, plus typecheck and production build. Browser coverage includes the real private placement worker, linked source selection, reload, export exclusion, deletion and WebGL recovery; desktop/mobile screenshots were inspected. This supports the narrower placement-inspection and asynchronous-interpretation release candidate, not assembled garment deployment. Production has not been changed. Deployment still requires selecting that narrower scope, preserving schema-5-compatible rollback and executing the documented private production checks.

The same-source 8/16/32-substep controls complete the short 1% target-reduction experiment. Maximum endpoint discrepancy decreases from 8.98 mm to 3.25 mm in the successive comparisons, but timestep independence remains unestablished and maximum seam gap remains about 487 mm. The [latest numerical evidence](../research/3d-solver-feasibility.md#thirty-two-substep-follow-up) retains 32-state replay and 222,006 exact proof leaves. Remaining assembly work is substantive: refinement-compatible contact, finite-thickness seam/layer semantics, turning/binding/folds, full closure and settling, followed by source/strain/contact and material validation. These research blockers must not be relabeled as deployment chores.

The isolated implementation checkout contains durable asynchronous interpretation (J0), a physical-inventory compiler, constrained source-preserving meshes, edge/source correspondences, mirrored fabric instances and self-contained GLB placement derivatives. Private inspection jobs now connect those derivatives to an interactive source-linked viewport. The assembly graph records physical seam memberships, localized closures, free boundaries, dependencies and unsupported operation semantics. These foundations do **not** complete E1/E2: full solver-quality acceptance, executable turning/binding/orientation, fold/holes support and validated assembly remain open. No assembled garment or simulated drape is exposed in the application.

The inspection integration is a separately labeled intermediate capability. It does not satisfy the E3/E4 complete-shirt acceptance criteria. Schema-5 inspection jobs use independent leases, immutable pattern/runtime capture, checked private artifact installation, deletion/recovery cleanup and default exclusion from all exports. The viewport provides rotation, zoom, source selection shared with 2D, explicit placement-only classification, outdated-source warnings and WebGL fallback. [Independent review](../reviews/3d-inspection.md) records its narrower scope and checks.

The original full shirt has 18 templates, 24 shell/facing fabric instances and six unresolved interfacing roles. The independent fixture ledger is `packages/test-fixtures/assembly-expected.json`. Tail geometry is integral to the back pieces. The compiler binds the exact pattern bytes and explicit construction to digests, keeps seam-line geometry unchanged and reports omitted allowances and assembly capabilities. It rejects missing selected pieces rather than adapting construction to the remaining inventory.

E0 evaluates an independently licensed CPU solver after finding the upstream optional Warp fork restricted to non-commercial research. The [feasibility evidence](../research/3d-solver-feasibility.md) records executed fixtures and remaining gates. The [foundation adversarial review](../reviews/3d-engine-foundation.md) records verified fixes and the narrower reviewed implementation scope. Neither is release or full-engine approval.

J0 introduces separately fenced interpretation jobs and prompt HTTP 202 acknowledgement. Reload/polling, cancellation, captured-input conflicts and one bounded interrupted-attempt retry preserve existing geometry jobs. The combined implementation uses schema 5, extending schema 4 with independent inspection records; rollback must retain a schema-5-capable binary. Production data is untouched by isolated checks.

Integration verification, 2026-09-17: **104 application tests and 26 engine cases passed** in full runs. **23 browser cases passed** across the full run and focused reruns after fixing a mobile tab-layout regression. Both new 3D browser cases passed again against the final build, covering real private installation, reload/transient failure recovery, linked selection, unchanged exports, mobile layout, deletion cleanup, WebGL-unavailable fallback and context-loss recovery. Typecheck, build, documentation and whitespace checks passed. Numerical experiments and their independent reviews are recorded separately; these application checks do not validate assembled drape.

Next gate: close solver feasibility on the complete source-derived shirt, including quality meshing, executable assembly semantics and accepted material/placement profiles. Only after those gates should assembled inspection and drape be enabled. The integrated flat GLB must not substitute for these milestones. Retain failed full-shirt experiments and unchanged numerical acceptance criteria; successful rendering cannot promote them.

Contact continuation, 2026-09-18: an optional pinned IPC surface barrier is coupled to the diagnostic global solver, with continuous collision bounds on optimization and physical motion paths. Independent review of the existing full-shirt starting placement finds **1,511 intersecting nonadjacent triangle pairs**; a separate toolkit check also rejects it. Contact cannot initialize safely from that state. Before another full-shirt contact run, establish collision-free rigid staging and explicit finite-separation layer/seam targets, preserving source rest metrics. Existing zero-distance sewing targets are not a finite-thickness assembly model. The [contact review](../reviews/3d-cloth-contact.md) records the narrower verified controls; this does not enable garment output.

Continuation, 2026-09-17: a separate cut-domain mesher now retains the actual allowance fabric and source-derived interior stitch paths, with independent topology/source/path validation. All 18 shirt templates are covered by its synthetic fixture. The experimental weighted solver adapter now samples the exact embedded source arcs, validates sparse registrations and stages closure while retaining cloth rest geometry; it does not substitute nearest-vertex welds. Adversarial review rejected duplicate physical registrations, concavity-crossing source supports and inverted topology. Contact controls isolate instability on an unsewn front panel. Current-contact source-point locality replaces unsafe whole-primitive exemptions in a separate Newton experiment and retains layer/fold response controls. Neither experiment closes E0 or provides executable full-shirt sewing. The [cloth/contact review](../reviews/3d-cloth-contact.md), [embedded sewing review](../reviews/3d-embedded-sewing.md) and their numerical ledgers retain the measured scope.

Latest regression verification: **104 application tests, 28 engine cases across the full and focused runs, and all 23 browser cases passed**. The new engine cases execute six allowance/path tests and the rigid-placement regression. Seven separate contact-research tests pass, including a one-step layer-response control. Typecheck and build pass. Full-shirt simulation remains rejected; the quality-refined contact trial exceeds its resource budget. These checks support committing the foundations, not deploying an assembled garment engine.

Further continuation: exact embedded sewing, opposed rigid probe placement and pointwise material-local contact controls are implemented as research primitives. Correcting conflicting circumcentre batches removes the refined body's artificial micrometre edges; fixing exact duplicate collar boundary coordinates closes the observed memory-allocation loop. The final stationary refined-front control completes in seconds, and all 18 registered templates pass the canonical synthetic preflight. Independent rotated-collar probes still fail closed on degenerate triangles or precision-limited subdivision; this is not general rotation-invariant meshing acceptance.

Current regression evidence: **104 application, 31 engine and 23 browser tests pass**, with the six meshing/placement/coupling wrappers rerun after the collar fix. The new numerical ledger and [embedded sewing review](../reviews/3d-embedded-sewing.md) retain failed gathers, source identities and bounded diagnostics. Opposed gathered cut cloth still has excessive deformation, unsettled motion and surface contacts despite tiny seam residuals. The old full-shirt boundary-spring recipe remains rejected. The next implementation gates are executable allowance folds/layer order and sewing/contact coupling that cannot introduce penetration after the contact solve, followed by convergence and full-shirt validation. Do not respond to these failures by relaxing acceptance or excluding entire seams from collision.

Latest sewing continuation: the optional research harness connects all 24 cut-cloth instances through 305 source-validated embedded constraints. Adversarial review found and corrected omitted spring coloring, float32 membrane-area cancellation, aliased pre-step velocity state, exterior refinement slivers and rounded whole-edge endpoint handling. The corrected no-contact full-shirt run remains finite but reaches 11.35 times rest edge length and a 28.35 mm maximum seam gap. The final contact run also completes, but reaches 22.57 times rest edge length with a 5.24 mm maximum gap; both remain rejected. Earlier embedded trajectories used flawed velocity reconstruction and cannot substantiate full-substep dynamics. Current source-bound controls and limitations live in the [stability ledger](../research/3d-sewing-stability-results.json) and [embedded sewing review](../reviews/3d-embedded-sewing.md). Sewing/contact coupling, explicit layer/fold execution and convergence remain the next gates; a finite result does not authorize release.

Continuation, 2026-09-17: experimental embedded sewing can now run inside the VBD elasticity/contact iteration instead of bypassing collision truncation with an external projection. Independent analytical and finite-difference checks validate its sparse forces, while stiff free-particle controls exposed coordinate-descent drift; an optional augmented solve corrects those tested linear cases without changing physical compliance. The complete shirt still fails nonlinear convergence and contact checks. Source-linked strain and mass-motion diagnostics, rejected/resource-limited controls, and the remaining placement issues are recorded in the [embedded sewing review](../reviews/3d-embedded-sewing.md) and [coupled sewing ledger](../research/3d-coupled-sewing-results.json). These experiments do not authorize assembled output or deployment.

Continuation, 2026-09-18: the offline global reference now includes optional conservative dihedral bending with source-derived rest geometry, coupled Gauss–Newton search and explicit bending-energy accounting. The 12-step torso control converges at every step. This is a numerical foundation, not E3/E5 completion: coupled contact, damping/material calibration, full-shirt settling and layer operations remain open. Current measurements and limits are maintained in the [solver feasibility evidence](../research/3d-solver-feasibility.md#elastic-bending-reference--2026-09-18).

The September 18 local-fold continuation adds an explicitly optional angular barrier and a conservative swept hinge guard to the offline reference. Independent derivative, path and integration reviews pass; the torso control remains unchanged. This is not finite-thickness or general self-contact and does not close full-shirt assembly. See the [current experiment and limits](../research/3d-solver-feasibility.md#local-fold-barrier-and-swept-hinge-controls--2026-09-18).

## Authority, scope and document ownership

September 20 scalar-sewing continuation: the offline reference offers rotation-invariant positive anchor-distance targets, with exact derivatives, guarded search, supervised adaptive execution and exact-arithmetic replay. The actual-source cuff stops at 98.4375% within 240 CPU seconds, then completes the unchanged schedule in 289.90 CPU seconds with a larger budget. The [completion evidence](../research/3d-solver-feasibility.md#completed-scalar-distance-cuff-control--2026-09-20) retains the failed budgets and verified nineteenth state. This does not implement turning or replace the earlier vector-cuff control; release gates remain unchanged.

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
