# 3D engine handoff

**September 23, 2026 · Research continuation · Not deployment-ready**

Current continuation: source-bound seam/gripper remapping, native fold controls and refined rigid placement have separate independent audits. New [per-hinge schedules and mechanics](3d-solver-feasibility.md#explicit-fold-schedules-and-mechanics--2026-09-23) support independent target/activation changes and explicit release accounting, with adversarial derivative and angle-arithmetic fixes. They are standalone primitives; source installation, global/adaptive integration and captured refined motion remain unfinished. The proposed runner/main-replay edits still await specific authorization after automatic approval rejection. Construction phases, settling, material/refinement validation and full-shirt integration remain open. The [numerical review](../reviews/3d-cloth-contact.md#numerical-continuation-review--2026-09-23) and current evidence supersede historical checkpoint counts below; no garment is accepted.

## Repository and scope

The complete pinned Linux numerical/supervision suite passes **910 tests in 394.011 seconds**, including all forty-four new cases (19 mechanics, 11 schedule and 14 angle-increment tests). The new cases also pass in focused macOS runs; these counts overlap the full Linux discovery. Log: `.planning/solver/controlled-fold-full-numerical-sep23-v1.log`. This verifies standalone offline primitives, not guarded motion, construction or an application release.

The new `solver_fold_control_schedule.py`, `solver_controlled_fold.py` and `solver_dihedral_increment.py` leave the legacy fixed fold path unchanged. Integrating them requires an explicit source-bound temporal recipe, unambiguous selection against the legacy assembly schedule, fresh per-retry sampling and validation, a fold-only pre-journal work gate, and independent replay of the same rounded coefficient policy. Preserve full cloth/contact/triangle guards when activation is zero; parameter jumps evaluate the union of old/new active hinges, and motion needs a declared conservative hinge-path policy. The earlier whole-strip gripper path is not an internal-fold recipe. Do not bypass the unapplied entry-point proposal with an alternate refined execution path.

The [source-bound refined placement](3d-solver-feasibility.md#source-bound-refined-placement--2026-09-23) now applies explicit poses to every unchanged numerical rest coordinate with exact binary-input arithmetic and one final rounding. The producer and standard-library auditor retain rotation defects, every edge/triangle witness, raw-lineage commutation and all ten cross-instance separating planes. A new uniform evaluation policy changes one historical binding vertex by at most 5.551115e-17 m; original source geometry and control targets are unchanged. No controls are installed or moved. Next author explicit temporal controls compatible with the internal fold actuation, then verify captured guarded motion after the pending entry-point authorization. The old whole-strip gripper targets are not automatically a valid internal-fold schedule.

The retained starting placement separately passes fixed-profile static contact admission: 0.1 mm activation/core, zero filtered pairs, zero contact energy/gradient, two nonadjacent intersection checks, 1,018 exact stationary candidate-separation witnesses with independent coverage and 398 stationary triangle witnesses. The minimum cross-instance plane gap is 1 mm. Raw v2 placement evidence is under `.planning/solver/binding-placement-sep23-v2/`; contact/runtime artifacts are under `.planning/solver/binding-placement-static-contact-sep23-v1/`. These checks execute zero time steps and do not certify motion from the original overlapping draft layout.

The preceding placement checkpoint passed **866 tests in 407.172 seconds** on pinned Linux, including all twenty-four new placement producer/audit cases; those twenty-four also passed on macOS. Log: `.planning/solver/binding-placement-full-numerical-sep23-v1.log`. This count and earlier counts below are historical checkpoints. Both proposed capture/main-replay entrypoints remain unchanged and their prepared patch still passes `git apply --check`; no application release check or deployment is included.

The [source-bound fold control](3d-solver-feasibility.md#source-bound-binding-fold-controls--2026-09-23) now derives both full-cut rails from directed source topology, binds all sixteen actuators to actual native hinge order and retains exact length/stiffness rounding witnesses. Its independent auditor imports only the standard library; a separate binder matches the declaration to the current native model and constructs the existing fold primitive. Relative-angle actuation applies balanced torques to free cloth, with no fixed-side assumption. A constructed two-fold pose requires transporting the finish axis after the attachment fold; the wrong flat-axis counterexample introduces 54.743 µm edge distortion. No dynamics or contact/path proof was executed for those constructed folded poses. The subsequent placement work supplies a separately checked starting state, not a folded trajectory; source-normal sewing frames and garment construction remain separate gates.

The [binding anchor remap](3d-solver-feasibility.md#binding-anchor-remapping--2026-09-23) now separates exact original coefficient lineage from explicitly rounded numerical weights. A new strict source profile rebuilds all global indices for 245 vertices and 398 triangles while retaining the immutable five-fabric base, all forty original rows and thirty-five unchanged derived local rows. Independent remap audit imports no implementation or pattern engine. Normal-offset sewing and its crease-side director frames remain unsupported in this profile; the separate fold mapping does not supply or override them.

The [gripper migration descriptor](3d-solver-feasibility.md#binding-gripper-migration--2026-09-23) now preserves original point controls while deriving their refined material anchors. The three existing grips map to canonical triangles 75, 55 and 86 with exact coefficient pullback; the allowance grip still has a separately recorded 1.788112e-18 m stored-coordinate shift. The independent standard-library audit recomposes each raw stage and reconstructs the complete recipe. No controls are installed and no refined placement or trajectory is produced. Source/control correspondence does not imply identical forces or motion after refinement.

Runner/main-replay integration is not applied: automatic approval review twice rejected edits to `scripts/spike-contact-continuation.py` and `scripts/replay-rest-filtered-continuation.py` for insufficient specific user authorization. The exact proposal is ignored at `.planning/solver/binding-refined-entrypoint-proposal.patch`, with `git apply --check` passing. Do not bypass the rejected action. After explicit authorization, apply/review the proposal, add entry-point regression tests and run a fresh captured refined control plus independent replay before claiming numerical integration. No new physics or physical acceptance follows from input correspondence.

The preceding static fold-control checkpoint passed **842 tests in 379.840 seconds** on pinned Linux, including all twenty-two new fold-control/audit cases, the preceding twenty-four gripper cases and forty seam-remap/source-builder/integration cases. Log: `.planning/solver/binding-fold-full-numerical-sep23-v1.log`. All twenty-two new tests also passed on macOS. Static geometry, source/control identities and independent rounding checks remain distinct from contact or dynamics proof. The two proposed entry-point edits remain unapplied; earlier counts below are historical.

The [extended binding hold](3d-solver-feasibility.md#extended-binding-hold--2026-09-23) completes and replays both 512-step controls, with the initial 192 transitions exactly matching the repeated baseline. Final maximum speed remains about 0.478 mm/s; passive-tail angle range increases to 0.05626°. The [ledger](3d-binding-extended-hold-results.json) and [figure](3d-binding-extended-hold.png) retain this unresolved motion. No settled pose or binding operation is accepted. Baselines were repeated from the identical complete captured numerical snapshot because capture includes newly added unused helpers.

The [binding refinement descriptor](3d-solver-feasibility.md#binding-crease-preprocessing--2026-09-23) adds attachment/finish crease chains through the complete original cut domain, yielding 29 strip vertices and 44 triangles with rationally composed original-parent lineage. It is deliberately non-executable. Seam, gripper and fold correspondence now have separate witnesses; normal-offset sewing frames and source capture/replay remain pending. Preserve the immutable original five-fabric unit and all forty source rows. Flat point containment alone cannot establish coefficient preservation under deformation; do not normalize, clamp or change the source profile to bypass validation.

This checkpoint's final pinned Linux numerical/supervision suite passes **756 tests in 347.358 seconds**, including all hold, refinement and repaired rigid-velocity cases. Log: `.planning/solver/binding-extended-hold-full-numerical-sep23-v2.log`. Earlier counts below describe historical checkpoints, not additional current tests. Nothing is deployed.

The [fourfold-duration comparison](3d-solver-feasibility.md#fixed-timestep-binding-time-control--2026-09-23) improves tracking at the unchanged 2 ms timestep: the 3° command peaks at 3.165° and finishes at 0.477 mm/s maximum speed. Both slower controls complete and replay 256 steps without retries. Relative angle still changes 0.03148° over the 64 ms passive tail. The [ledger](3d-binding-fourfold-time-results.json) and [figure](3d-binding-fourfold-time.png) preserve the original control alongside the slower outcome; neither is a settled-pose or construction acceptance result.

The [first five-fabric binding motion](3d-solver-feasibility.md#first-source-binding-motion--2026-09-23) now has matched zero/3° controls, complete independent replay and source-frame/surface observations. Both complete 64 steps without rejection. The driven run overshoots to 4.485° and ends with 0.01787 m/s maximum speed, so tracking and settling remain unresolved. Five sampled attachments stay held; wrapping, stitch-down, apex securing and source phase completion remain open. The [ledger](3d-binding-first-turn-results.json) and [figure](3d-binding-first-turn.png) preserve the outcome rather than promoting numerical completion into garment acceptance.

The [two-timestep ledger](3d-cuff-temporal-results.json) and [figure](3d-cuff-temporal-results.png) now preserve the half-step comparison and independent review. The older hold and frame/contact artifacts remain unchanged. Reproduction and scope are in [solver evidence](3d-solver-feasibility.md#two-timestep-sensitivity).

The next construction input is now [explicitly phased](3d-solver-feasibility.md#source-cuff-construction-phases--2026-09-23): five fabric instances, both binding prerequisites, 40 sampled source constraints and a reserved shell attachment opening. The phase plan records its research layer-order policy and source dependency refinements; nothing has executed. Eighteen focused tests pass on macOS and Linux ARM. The fresh ignored input `.planning/solver/reviewed-cuff-construction-left-v1/unit.json` contains 227 vertices and 366 triangles, with captured generating source.

[Compliant material controls](3d-solver-feasibility.md#compliant-source-material-controls--2026-09-23) now supply source-triangle pulling, explicit engagement/release, force and discrete-work accounting, rollback, capture and independent gripper replay. The [retained two-triangle contact diagnostic](3d-material-gripper-contact-results.json) completes and replays eight states; it remains synthetic and unaccepted. Independent precision review repaired force, directional-energy and torque cancellation. The captured sewing integration described below adds per-member activation and engagement work while preserving every source row. Construction-specific binding/turning paths must also preserve material-side, interfacing, continuous-seam and opening-passage gates. Generic pulling controls do not execute those phases.

The [captured seam controls](3d-solver-feasibility.md#captured-seam-controls--2026-09-23) now connect complete source-row binding, explicit target/activation input, adaptive sampling and rollback, durable work, CLI capture and independent sewing replay. The cuff profile rederives all forty rows on its compatible numerical runtime from captured engine dependencies. Raw uncaptured activation still rejects. Next implement construction-specific control paths and milestone witnesses while preserving the reserved opening, continuous seam coverage, source/material sides and interfacing decisions. No construction phase has executed.

The seam-activation primitive checkpoint passes **565 full Linux numerical/supervision tests in 201.509 seconds**, including 51 new cases. Documentation and whitespace checks pass; application and deployment scope remain unchanged.

The captured seam-control integration subsequently passes **628 full Linux numerical/supervision tests in 268.099 seconds**, including complete five-fabric cuff capture and independent combined-control replay. The final namespace hardening and focused rechecks are recorded in the [review](../reviews/3d-cloth-contact.md#current-verification-boundary). These numerical controls still do not execute a binding or turning phase.

The next [sampled seam-path gate](3d-solver-feasibility.md#sampled-sewing-path-bounds--2026-09-23) adds optional exact temporal distance-error proof to replay. It requires explicit tolerance and captured scalar-distance controls, rejects knot-crossing intervals, and retains the distinction between sampled temporal bounds and continuous spatial stitching. Construction-specific attachment and motion still need their own source-preserving inputs and observations.

This final checkpoint passes **660 full pinned Linux numerical/supervision tests in 315.871 seconds**, including the final namespace fix and thirty new seam-path/CLI tests. Twenty-seven focused sweep tests also pass on macOS. The earlier checkpoint counts remain historical, not additional current tests.

The final gripper checkpoint passes **514 full Linux numerical/supervision tests in 203.919 seconds**. A preceding run exposed two stale unrestricted mock fixtures; only their declared optional solver features changed, and original guard assertions remain intact. The focused 61 gripper checks and 24 journal checks overlap the full discovery and must not be added to its total. See the review for failure logs and scope. No application release checks or deployment were performed.

- Repository: `erhuve/sew-computer`; continue on `physical-path-certificate-sep19`.
- Isolated research checkout on Zo: `Code/sew-computer-3d-engine`, relative to the workspace root. The live service checkout is `Code/sew-computer`; do not edit or test-build it for this work. Historical README/local AGENTS statements that this research checkout backs production are stale; follow the workspace routing index and verify service configuration before any release.
- All implementation and evidence through `1db01e6` were already committed and pushed when this handoff began. The fetched `origin/main` is an ancestor of this development branch. Continue the development branch rather than starting from main and losing the research changes.
- Nothing in this handoff authorizes deployment. Application 3D remains **placement inspection**. Research schedules are not assembled garments, material drape or sewing/fit validation.
- Read the [active implementation plan](../plans/pattern-derived-3d-engine.md), [solver evidence](3d-solver-feasibility.md), [foundation review](../reviews/3d-engine-foundation.md), [inspection review](../reviews/3d-inspection.md) and [normative contract](../design/project-contract.md). This checkpoint summarizes their current state and does not supersede them.

## What is implemented

Source-derived cut meshes retain pattern dimensions, physical-piece identities, original parent triangles and interpolation mappings. Interior seam anchors survive allowance-crease subdivision. Experimental sewing includes fixed vectors, scalar distances and material-normal offsets; the latter include reactions on their defining triangle. Signed hinge actuation adds external fold torques without changing elastic rest geometry.

The current cuff diagnostic combines shell/facing sewing and outer-allowance folding on one 66-vertex, 96-triangle cloth state with twenty seam registrations. Captured stages preserve positions, velocities, seams and contact through transitions and rejected-step retries. Supervised execution retains bounded attempts and immutable inputs; replay reconstructs residuals and checks saved-state integrity, endpoint intersections and continuous-contact certificates.

| Commit | Verified increment |
| --- | --- |
| `234a5b4`, `530a90f` | Source-preserving allowance creases and three replayed signed/near-closed controls |
| `9a6160d` | Captured staged sewing/folding with state continuity |
| `c3b9b76` | Combined source cuff and remapped anchors on the allowance mesh |
| `99a7456` | Exact normal-seam curvature in the guarded search, with existing fallback |
| `1db01e6` | Six-control outcome ledger, geometry figure and remaining failures |

Key entry points: `scripts/solver_assembly_schedule.py`, `scripts/solver_cuff_sequence.py`, `scripts/solver_crease_mesh.py`, `scripts/solver_normal_sewing.py`, `scripts/spike-cuff-sequence-input.py`, `scripts/spike-contact-continuation.py` and `scripts/replay-rest-filtered-continuation.py`.

## Evidence and limitations

The [six-control ledger](3d-cuff-sequence-results.json), [saved-geometry figure](3d-cuff-sequence.png) and [detailed outcome table](3d-solver-feasibility.md#combined-cuff-sequence-outcomes) are committed. All six controls remain `accepted: false`.

- Recorded verification: **386 numerical tests**, a later overlapping seven-test normal-sewing run, and **11 separate harness tests**. These are historical checkpoint results, not newly rerun release checks for this documentation handoff.
- Historical replay verified **461 saved states and 251,106 exact-rational contact-path proof leaves**, including accepted prefixes of failed runs. Both endpoint intersection oracles passed every saved state. This does not certify global layer order, turning or garment quality; replay also shares some force implementations and is not independent model review.
- The completed 0.5 mm-offset, 1.2-radian control takes **39.01 CPU seconds** with exact seam curvature versus **398.86 seconds** with the prior metric. Adaptive histories differ; final geometry differs by **32.925 mm even after rigid alignment**. Do not describe this as equivalent-result acceleration or temporal convergence.
- Facing folds miss targets: measured angles span about **0.08–84.33 degrees** against a **68.75-degree** target; maximum edge stretch is **3.66%** in the completed exact-curvature control.
- Tight **0.11 mm** controls exhaust **600 CPU seconds**, stopping at 37.5% or 35.15625% with the new metric. Stronger wide-gap actuation reaches 94.140625% before exhausting 256 attempts, with up to **12.29% stretch**. Stronger actuation is not an accepted fix.

## Remaining work, in order

1. **Resolve coupled fold tracking.** Inspect facing hinge targets, seam/frame reactions and contact interactions on the source cuff. Separate target incompatibility from insufficient settling or numerical convergence. Require measured target error, seam error, layer-side behavior and deformation, not just completed time or low residual.
2. **Establish temporal and spatial stability.** Compare search metrics on the same fixed time partition, then isolate timestep and mesh-refinement changes. Retain failed attempts and unchanged source metrics. Review the rest-filtered contact model's reduced within-panel thickness, derivatives/locality, swept frame nondegeneracy and layer-order limits independently. Freeze reviewed numerical acceptance profiles before claiming acceptance.
3. **Complete one cuff construction.** Execute closure, allowance/corner treatment and actual turning through the declared source attachment opening in one continuous replayed sequence. Current parallel-layer folds do not turn the cuff. Preserve the opening and all shell/facing roles; then validate sleeve attachment.
4. **Implement remaining garment operations.** Binding wraps and open sleeve edges, collar/stand turning and layered attachments, gathers/multi-way junctions, localized closures, and explicit interfacing treatment must execute from source semantics. Reconcile all 24 fabric instances and six unresolved interfacing roles before claiming a complete shirt.
5. **Validate full-shirt assembly and drape.** Demonstrate closure, settling, strain/contact/source fidelity, repeatability and bounded execution across the declared size/component fixtures. Body contact and calibrated material behavior require their own evidence. No current control establishes these gates.
6. **Integrate and review the accepted scope.** Only after solver acceptance, connect assembly output to the private worker/viewport with revision/runtime identity, cancellation/recovery, resource limits and privacy/export rules. Run independent adversarial review and applicable application/engine/browser/build checks, then follow [deployment procedures](../deployment.md) with schema-compatible rollback and explicit release authorization.

The placement viewer and asynchronous interpretation fix are a separate, narrower release candidate. At `f6680f5`, the recorded checks passed 104 application, 34 engine and 23 browser tests plus typecheck/build. That historical result does not certify today's branch or authorize deployment. A placement-only release needs its own scope decision and fresh checks; it does not require pretending the simulation gates are complete.

## Resume and reproduce

Fetch before starting and before every push; preserve concurrent changes. Commit and push verified increments to the active development branch. Use Miku's configured author and Zo's configured committer, with `Co-authored-by: hatsunemiku <hatsunemiku@zo.computer>` in the commit body.

The ignored research runtime is `.planning/solver/newton-venv`; its pinned dependencies are `scripts/solver-contact.requirements.txt` (which includes `solver-spike.requirements.txt`). On a new machine, create a separate virtual environment and install those requirements. The upstream pattern engine needs its separately provisioned locked checkout; see the root README and `services/engine/source-lock.json`.

The September 23 Linux ARM controls used Python 3.13.15 with the same package pins. IPC Toolkit 1.6.0 was built from upstream tag commit `478876f30bf8ea768772dd8983c26a1a801ad976`; the available source distribution failed to build on that host. A separate Python 3.13 Linux environment with CMake, a C++ toolchain, Git and Ninja can install `scripts/solver-spike.requirements.txt`, then `git+https://github.com/ipc-sim/ipc-toolkit.git@478876f30bf8ea768772dd8983c26a1a801ad976`. Record the resulting runtime identity rather than assuming a wheel and source build are equivalent. The new ledger includes the measured platform and package versions. The exact upstream cause of the native LBVH omission remains open; explicit HashGrid and independent candidate-coverage verification are required for these controls.

Run from the isolated repository root. The complete numerical discovery and supervised continuation CLI require Linux (`/proc`, `prctl` and process-resource enforcement). Individual numerical and source-generation tests also run on macOS; that does not verify the Linux worker lifecycle. Keep Warp's cache in the isolated writable research directory. For focused checks:

```sh
export WARP_CACHE_PATH="$PWD/.planning/solver/warp-cache"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .planning/solver/newton-venv/bin/python -m unittest discover -s scripts -p 'test_solver_normal_sewing.py'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .planning/solver/newton-venv/bin/python -m unittest discover -s scripts -p 'test_solver_cuff_sequence.py'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .planning/solver/newton-venv/bin/python -m unittest discover -s scripts -p 'test_prepare_cuff_source.py'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .planning/solver/newton-venv/bin/python -m unittest discover -s scripts -p 'test_solver_*.py'
python3 scripts/check_docs.py
git diff --check
```

For the combined diagnostic, extract sewing with `spike-cuff-sewing-input.py --normal-offset-frames` and folding with `spike-cuff-fold-input.py --crease outer-allowance --target-angle-radians 1.2`, using the same parent canonical source. Combine their canonical files with `spike-cuff-sequence-input.py --sewing-input ... --fold-input ... --crease-frame-region body --output ...`. Use each CLI's `--help` and fresh private output directories; never overwrite historical runs.

Use the ledger's per-control `arguments` for exact settings and captured source hashes for historical comparisons. The completed exact-curvature wide-gap control uses the combined canonical/placement inputs with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .planning/solver/newton-venv/bin/python scripts/spike-contact-continuation.py --canonical /absolute/path/to/new-input/canonical.json --placement /absolute/path/to/new-input/placement.json --output /absolute/path/to/new-run --assembly-schedule --fold-actuation --sewing-mode normal-offset --contact-model rest-filtered --ccd-profile temporal-separation-tight-inclusion --activation-distance-m .001 --minimum-distance-m .0001 --pressure-pa 10000 --target-fraction .25 --step-seconds .064 --subdivisions 64 --max-evaluations 100 --max-attempts 256 --max-depth 8 --cpu-limit-seconds 480 --wall-limit-seconds 720
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .planning/solver/newton-venv/bin/python scripts/replay-rest-filtered-continuation.py /absolute/path/to/new-run
```

These are experimental settings, not acceptance thresholds. Changes to source, targets, material, contact or budget must remain explicit in new evidence.

## What a Git clone does not contain

The six `.planning/solver/cuff-sequence-*-sep23` run directories named in the ledger are present on this Zo at handoff. They hold local research artifacts; `.planning/` is ignored. The parent source used in documented extraction is `.planning/solver/fold-barrier-shirt-v1/canonical.json`. A Git clone carries code, ledgers and figures, **not** the virtual environment, private parent source, captured inputs, journals or full saved trajectories. Hashes identify those artifacts but cannot recreate their bytes.

For a handoff to another machine, arrange an authorized private transfer of the required parent input and captured run directories, or regenerate synthetic fixtures using the documented pipeline and record new digests. Retain numerical source snapshots when reproducing the prior-metric controls; current code alone will use the new metric. Do not commit private measurements, production data, credentials, external assets or runtime files to make a clone self-contained. Full replay requires the original saved artifacts; the committed evidence summaries alone are insufficient.

To generate a fresh synthetic cuff with the pinned research runtime:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .planning/solver/newton-venv/bin/python scripts/prepare-cuff-source.py --output .planning/solver/new-synthetic-cuff-source
```

Pass its `canonical.json` through the existing sewing and fold extraction commands above. This source has new provenance and does not reproduce the historical canonical bytes. It preserves the sleeve-attachment opening but does not execute the selected perimeter operations' external dependencies or demonstrate turning. The generator and complete sewing/fold source-remapping pipeline have six independent regression tests in `scripts/test_prepare_cuff_source.py`.
