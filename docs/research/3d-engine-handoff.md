# 3D engine handoff

**September 23, 2026 · Research continuation · Not deployment-ready**

Current continuation: the [numerical adversarial review](../reviews/3d-cloth-contact.md#numerical-continuation-review--2026-09-23) supersedes the historical model-review status below. Contact locality, all-triangle swept nondegeneracy and binary64 fold-energy precision are repaired. Every native contact path now explicitly uses `HashGrid`; independent exact-rational replay additionally proves candidate coverage. The complete Linux suite passes 427 tests; five later independent coverage/replay tests and six source-generation tests pass separately. The corrected baseline and hold replay 373 states and 126,686 contact-proof leaves. Their first 64 ms are bit-identical; extending to 256 ms reduces maximum speed about 78-fold without resolving facing underfold. See the [new evidence ledger](3d-cuff-hold-results.json), [figure](3d-cuff-hold-results.png) and [interpretation](3d-solver-feasibility.md#verified-cuff-hold-control--2026-09-23). The defective Linux LBVH capture rejects replay and supports no physical conclusion. The historical six-control ledger remains unchanged and has not acquired the new proofs. A same-input scalar-distance hold also completes and verifies 266 states; facing angles improve but retain target error and more motion. Source review found that crease frame regions were chosen by child enumeration rather than explicit body/allowance semantics. The binding now requires an explicit `--crease-frame-region body|allowance` choice at ambiguous crease anchors. Body changes exactly seven frame triangles; rest geometry, anchor coefficients and targets remain identical. Three body-frame holds now pass replay (801 states). With the 0.1 mm minimum clearance unchanged, narrow 0.1/0.05 mm activation buffers bring both layers to about 67.4–68.4° against a 68.75° target. The [new frame/contact evidence](3d-cuff-frame-contact-results.json) records the changed force law and source binding; material, refinement and garment acceptance remain open. A half-timestep control also passes replay for all 513 states; the final maximum position difference is 0.174 mm and hinge difference 0.0284° at exact common saved times. This is temporal sensitivity evidence, not a convergence or settled-state claim. No garment is accepted.

## Repository and scope

The [two-timestep ledger](3d-cuff-temporal-results.json) and [figure](3d-cuff-temporal-results.png) now preserve the half-step comparison and independent review. The older hold and frame/contact artifacts remain unchanged. Reproduction and scope are in [solver evidence](3d-solver-feasibility.md#two-timestep-sensitivity).

The next construction input is now [explicitly phased](3d-solver-feasibility.md#source-cuff-construction-phases--2026-09-23): five fabric instances, both binding prerequisites, 40 sampled source constraints and a reserved shell attachment opening. The phase plan records its research layer-order policy and source dependency refinements; nothing has executed. Eighteen focused tests pass on macOS and Linux ARM. The fresh ignored input `.planning/solver/reviewed-cuff-construction-left-v1/unit.json` contains 227 vertices and 366 triangles, with captured generating source. Next implement validated binding/turning controls and true per-member force activation while preserving the unresolved material-side, interfacing and continuous-seam gates.

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
