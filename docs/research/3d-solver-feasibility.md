# Pattern-derived 3D solver feasibility

**2026-09-16 · Executed CPU research spike · E0 partly complete; no production solver acceptance**

## Decision

Select **Newton 1.6.0 / Warp 1.17.0 / SolverVBD on CPU** as the implementation candidate. It installs headlessly on this host, consumes application-owned rest meshes, preserves their rest tensors, supports independent physical particles, sewing springs and cloth self-contact, and produces ordinary glTF without model-generated geometry. This is a candidate selection, not an accepted full-shirt assembly or drape claim.

Reject the existing optional GarmentCode Warp fork for product integration: its pinned [license §3.3](https://github.com/maria-korosteleva/NvidiaWarp-GarmentCode/blob/63baf6855efdd89b2834b74640f84b3bb0d86b50/LICENSE.md) restricts use to research/evaluation. Its [README](https://github.com/maria-korosteleva/NvidiaWarp-GarmentCode/blob/63baf6855efdd89b2834b74640f84b3bb0d86b50/README.md) identifies Warp 1.0.0-beta.6 as its base and describes manual compilation. The separate application geometry source remains at its existing lock. No source from this simulation fork is copied into the application, and its runtime was not built or exercised after the license gate failed. CPU-only availability is **not** the reason for rejecting it; its source supports CPU.

The dedicated [engine plan](../plans/pattern-derived-3d-engine.md) still governs acceptance. Full-component turning, thickness-aware contact, calibrated anisotropy, body interaction and worker isolation remain open gates. A successful two-panel experiment cannot satisfy those gates.

Measured reports, including the failed older source mesh and final source-bound reruns, are retained in [the numerical ledger](3d-solver-spike-results.json). Historical characterization and final provenance-checked reports are distinguished in that file.

## Reproduction and ownership

The four microfixtures use procedural synthetic rectangles. An additional holdout calls the application's actual `compile_shirt` and `mesh_panel` functions for `placket_left` and `frill_left`, with explicit synthetic measurements. Both paths retain local rest coordinates, source panel/vertex identities and grain directions; actual-source artifacts also retain the mesher's source interpolation weights and independent rest-validation report. No body assets, private measurements, upstream garment assets, textures or weights are loaded. The spike is separate from the application's existing Python runtime and does not modify the live checkout.

From the isolated repository root:

```sh
python3 -m venv .planning/solver/newton-venv
.planning/solver/newton-venv/bin/pip install -r scripts/solver-spike.requirements.txt
timeout --kill-after=5s 300s .planning/solver/newton-venv/bin/python scripts/spike-3d-solver.py --output .planning/solver/baseline
timeout --kill-after=5s 300s .planning/solver/newton-venv/bin/python scripts/spike-3d-solver.py --output .planning/solver/refined --fixture gather --profile gather-refinement --steps 480 --verify-profile refined-gather-v1
timeout --kill-after=5s 300s .planning/solver/newton-venv/bin/python scripts/spike-3d-solver.py --output .planning/solver/holdout --fixture gather --profile gather-refinement --steps 960 --timestep-denominator 480 --translated --verify-profile refined-gather-v1
timeout --kill-after=5s 300s .planning/solver/newton-venv/bin/python scripts/spike-3d-solver.py --output .planning/solver/source-gather --fixture source-gather --profile gather-refinement --steps 960
.planning/solver/newton-venv/bin/python -m unittest discover -s scripts -p 'test_solver_spike_geometry.py'
.planning/solver/newton-venv/bin/python scripts/check-solver-spike-lifecycle.py --python "$PWD/.planning/solver/newton-venv/bin/python" --output .planning/solver/lifecycle
```

Output directories hold canonical JSON, self-contained glTF 2.0 display derivatives and a measured `report.json`. Final reports capture script, oracle, profile and applicable engine-module digests before execution and reject changes during a run; profile assessments embed the frozen contents. Canonical geometry is metres, Z-up; glTF applies the proper rigid conversion to Y-up. Binary vertex/index buffers are embedded, with no external requests. Source mappings live in canonical JSON rather than being inferred from render geometry. Browser rendering itself is not an executed acceptance check for this research script.

The script enforces 1–1,000 steps, 240 CPU seconds and 4 GiB address space. The external `timeout` bounds wall time. These are **research process limits**, not a queue/concurrency capacity recommendation and not a network/filesystem sandbox. The lifecycle harness launches with only PATH/LANG/HOME, avoiding application credentials; the standalone spike inherits its caller's environment and must not be used as a production worker entrypoint.

## Host and dependency evidence

The actual host exposed only the Warp `cpu` device. Warp reported no NVIDIA driver; `nvcc` was absent. Python was 3.12. The cgroup reported 512 GiB memory and a CPU quota of 64 CPU-seconds per second; these unusually large sandbox-visible ceilings are not a guarantee of dedicated resources. No new paid infrastructure was requested.

| Dependency | Pin and license evidence | Integration consequence |
|---|---|---|
| Newton | 1.6.0; upstream tag `c2ca70bec5998062b0b2d35869865cbe3a49feee`; [Apache-2.0](https://github.com/newton-physics/newton/blob/v1.6.0/LICENSE.md) | Server-side code candidate; retain notices if distributed |
| Modern Warp | `warp-lang==1.17.0`; [Apache-2.0 core](https://github.com/NVIDIA/warp/blob/v1.17.0/LICENSE.md) | Distinct from the research-only GarmentCode fork |
| Warp wheel libraries | Installed wheel includes CUDA/NVRTC/libmathdx licenses, Apache/LLVM exception, and separate notices for Gaia, cubql, NanoVDB, dlpack, fp16, appdirs, mesh algorithms and utilities | The aggregate wheel is not simply Apache-2.0; keep bundled notices and complete a distribution-specific audit before redistributing binaries |
| NumPy | `numpy==2.5.3`; installed metadata lists BSD-3-Clause, 0BSD, MIT, Zlib and CC0; wheel includes OpenBLAS/LAPACK notices, GCC runtime exception and LGPL libquadmath | Server use tested; redistributing a packaged runtime requires the wheel's transitive notices/obligations |
| Shapely | `shapely==2.1.2`; BSD-3-Clause Python package, bundled GEOS under LGPL-2.1 | Used only for actual application source compilation/meshing; retain bundled notices for redistribution |
| Newton optional assets/extras | Package notices include CC-BY-4.0 and font/utility licenses; simulation/importer/render extras were not installed | No examples/assets are copied or exposed; introducing an asset requires its own provenance |

Version pins are in `scripts/solver-spike.requirements.txt`; they are an experiment dependency list, not yet a deployment lock or redistribution approval. Official [VBD source](https://github.com/newton-physics/newton/blob/v1.6.0/newton/_src/solvers/vbd/solver_vbd.py) identifies the solver as experimental. Newton's tested minimal package requires only Warp and NumPy; Shapely is added for application geometry. Optional MuJoCo, USD, graphics and imported-body paths were not used.

## Executed microfixtures

Baseline profile: two 5×5-vertex panels, 64 total triangles, ten VBD iterations per step, 120 steps at 1/240 s. Sewing uses zero-rest-length springs; cloth rest tensors are created from the original planar geometry **before** rigid placement. Density 0.2 kg/m² and isotropic numeric membrane/bend constants are synthetic assumptions, not a calibrated fabric. Gravity is deliberately zero to isolate assembly; this does not demonstrate hanging drape.

| Fixture | Registration springs | Maximum / p95 seam gap, mm | Edge length ratio range | Final nonadjacent surface intersections | Warm wall time |
|---|---:|---:|---:|---:|---:|
| Equal 200 mm seam | 5 | 0.866 / 0.864 | 0.9857–1.0113 | 0 / 1,750 tested pairs | 1.49 s |
| Unequal 200:300 mm gather, baseline | 5 | 0.843 / 0.810 | 0.8032–1.2418 | 0 / 1,750 | 1.14 s |
| Two rigidly angled collar-like layers | 5 | 0.826 / 0.823 | 0.9883–1.0192 | 0 / 1,750 | 0.87 s |
| Overlap with only two point closures | 2 | 0.695 / 0.694 | 0.9995–1.0005 | 0 / 1,750 | 0.84 s |

All produced finite positions, unchanged triangle rest tensors and complete source mappings. Two original baseline runs produced bit-identical position hashes on this host. Baseline peak process RSS was 687,292 KiB (about 671 MiB). Initial kernel compilation added roughly 8–11 seconds; warm timings are not cold-start guarantees.

The folded-layer fixture starts from two rigidly rotated flat layers. Its final mean normal angle was 158.3°. It does not demonstrate turning a sewn collar, maintaining a material fold, or preserving prescribed layer order. The overlap fixture only localizes closures; no weld is applied to whole front boundaries.

## Gather root cause and measured adaptation

The baseline gather is **not accepted** despite its small seam residual. Pairing every edge vertex one-to-one forces each corresponding segment to have approximately the same deformed length. The 200 mm edge stretches to 243.45 mm and the 300 mm edge compresses to 244.51 mm. Increasing both resolutions/stiffness and adding out-of-plane rigid placement did not solve this: an intermediate 162-vertex run still reached a 1.238 edge ratio.

The corrected representation retains five registration anchors but gives the longer edge three source segments between each anchor. Those intermediate vertices can form folds without resetting the 300 mm rest length. Rigid out-of-plane initial placement breaks symmetry; deformation remains solver-produced.

Measured corrected fixture: 90 vertices, 128 triangles, five sewing springs, 480 steps at 1/240 s, 10× baseline membrane stiffness and sewing stiffness. Warm runtime 4.66 s; peak RSS 667,964 KiB. Maximum seam residual **0.2225 mm**, edge ratios **0.99959–1.00091**, area ratios **0.99951–1.00057**. Short seam arc: **200 → 200.0056 mm**. Long seam arc: **300 → 299.9938 mm**. No collapsed triangles or intersections across 7,542 nonadjacent triangle pairs. Rest tensors remain byte-identical.

`scripts/solver-spike-profiles.json` freezes the narrowly scoped `refined-gather-v1` bounds from this baseline before a separate holdout. The holdout simultaneously translated the fixture and halved timestep to 1/480 s for 960 steps. It passed the frozen seam/strain/area/arc-length/rest/surface checks: maximum gap 0.2508 mm, edge ratio maximum 1.00127, short arc 199.9710 mm, long arc 299.9664 mm, no surface intersections. Cold wall time was 18.90 s.

The holdout mean panel normal angle changed from 22.8° to 79.4°. Numerical material/arc preservation does **not** establish a unique settled shape or timestep-independent garment appearance. A separate translation-only run at the original timestep passed the same frozen profile: gap 0.2280 mm, maximum edge ratio 1.00091, panel normal angle 21.6°. A velocity/energy settling criterion and broader timestep sensitivity assessment remain open. No threshold has been loosened to label the baseline gather successful.

## Actual source holdout

The actual shirt compiler emits a **570 mm placket attachment** and **1,026 mm frill attachment** for the synthetic 1.8-fullness fixture. The adapter selects their named `right` edges, preserves pattern bytes/digest and source interpolation maps, and requires exact quarter-interval anchors rather than snapping rest positions.

The original refinement-only mesher produced 314 vertices / 540 triangles with thin elements (frill minimum angle 1.9°, maximum aspect 30.5). That run **failed**: 15 nonadjacent surface intersection pairs, edge ratios 0.8552–1.1123, short arc 601.92 mm and long arc 1,052.31 mm. It is retained as a failed observation, not hidden behind the successful rectangle test.

The updated boundary-densified constrained mesher uses disclosed resolutions **47.5 mm for placket and 42.75 mm for frill**, chosen so quarter anchors are already exact. This creates 112 vertices / 144 triangles. The same solver/material settings and 960 steps produce a 0.2930 mm maximum gap; edge ratios 0.99899–1.00561; short arc **570.1239 mm**, long arc **1,025.9485 mm**; no collapsed triangles or intersections across 9,842 nonadjacent pairs; and unchanged rest tensors. Warm wall time was 10.47 s in the first updated-mesh run. The source fidelity checks run before simulation.

The final source/oracle-digest-bound repeat produced identical positions, 10.58 s wall time and 673,684 KiB peak RSS. The final frozen synthetic holdout repeat passed unchanged bounds at 8.53 s warm wall time and 668,376 KiB peak RSS. These are individual process peaks, not measured simultaneous-worker capacity.

This is actual-pattern adapter evidence, **not a frozen-profile pass**: its maximum edge ratio exceeds the synthetic profile's 1.005 bound, and that profile was scoped to a different fixture. Full garment constraints, facing sandwich/turning and interval seam coverage are absent. General construction needs assembly-driven anchor insertion, rather than relying on conveniently divisible synthetic lengths. These results justify further adapter work without claiming the selected shirt is assembled.

## Independent checks and lifecycle

`scripts/solver_spike_geometry.py` does not call solver collision APIs. It tests all final nonadjacent triangle pairs using segment/triangle intersections plus positive-area coplanar polygon clipping, and independently calculates area and full sampled seam-arc lengths. Independent review caught dimensionally inconsistent absolute determinant tolerances that missed a 1 mm crossing. The corrected oracle uses dimensionless angular/barycentric tolerances, edge-scaled distance tolerances and squared-edge-scaled area tolerances; clipping coordinates are rebased to avoid translation cancellation. Five unit tests cover crossings without inside vertices, coplanar overlap versus boundary-only contact, separation, rotations/scales and tiny far-from-origin overlap. Retained baseline and gather holdout outputs were rechecked with the corrected oracle; zero-intersection results remained zero.

No stitched-neighbor collision exemptions are applied. Topologically adjacent faces are omitted; the oracle reports this. Zero detected final surface intersections is not a swept-collision, thickness-clearance, adjacent-fold or body-penetration proof. The current implementation must not advertise those stronger claims.

The cancellation harness observed one process in the running solver process group, sent SIGTERM, reaped it in **0.038 s**, found zero group members remaining, and successfully executed a fresh fixture afterward. This demonstrates termination of the observed tree; it does not inject an additional child or exercise application lease expiry, artifact installation races, deletion, network confinement or filesystem confinement. Those remain worker-integration tests.

## Capability ledger and remaining E0 gates

| Requirement | Current evidence | Remaining work |
|---|---|---|
| Headless CPU and bounded runtime | Installed and executed; measured resource use and cancellation | Production-sized shirt resource sweep; worker isolation and concurrency |
| Actual source geometry | Independent rest coordinates/identity, triangle rest tensors unchanged | Full curved-piece, cut-on-fold and physical-inventory integration |
| Equal seam and sparse gathered attachment | Measured fixtures and frozen gather holdout pass | Anchored nonuniform gathers, free interval diagnostics, robust placement across sizes |
| Layer/contact behavior | VBD self-contact enabled; final nonadjacent surface oracle | Thickness, adjacent folds, intended layer order, turning, multi-way attachment and binding |
| Materials/grain | Grain/source metadata preserved; scalar assumptions explicit | Anisotropic constitutive interpretation, perturbation tests, calibrated material provenance |
| Body contact | No body assets needed for these fixtures | Procedural cleared body, surface collision oracle and negative controls |
| Artifact format | Self-contained glTF plus authoritative source-map JSON written | Browser load/selection and private publication checks |
| Numerical profile | Scoped gather bounds frozen and holdout checked | Full-shirt profile and convergence/shape repeatability acceptance |

The next adaptation is an application-owned adapter that lowers validated assembly intervals into sparse sewing registration with sufficient long-side tessellation, retains the unaltered local rest mesh, provides collision-aware staged placement and validates source/inventory/material/contact independently. Keep unsupported turning/binding/allowance semantics explicit. Do not switch to an invented decorative mesh or claim that this research completes the full 3D plan.

## Full-shirt adaptation follow-up · 2026-09-17

The physical assembly compiler now validates the exact source seam recipe against selected components, expands multi-layer memberships, rejects duplicate physical edge consumption, accounts for every free edge, binds localized closure marks, and topologically orders the represented operations. Its `solverReady` remains false: orientation, turning, binding wraps and interfacing are explicit unresolved execution requirements. Uniform gather fractions are a trusted proposal, not a validated sewing distribution. The inventory fixture independently checks the four-way front attachment and self-seam obligations.

The mesher accepts exact named-boundary registration fractions, including nonuniform fractions on sampled curves, without moving original source vertices. This removes the earlier reliance on conveniently divisible experimental edge lengths. The default inspection mesh does not request these extra registration points.

An optional quality-refinement experiment uses the application's existing pinned SciPy 1.17.0: Delaunay interior points, boundary-segment recovery and bounded circumcenter insertion with boundary encroachment splitting. Independent rest validation still checks the result; the algorithm does not certify a minimum angle. At 60 mm maximum edges on the synthetic full-shirt source, front-body minimum triangle quality rises from **0.010739 to 0.188049**, and back-body quality from **0.006591 to 0.106870**. Area/boundary errors remain below 1e-7 in millimetre units. The cost rises from 606/629 vertices to 3,225/6,622 vertices, taking approximately 6/11 seconds respectively. This is opt-in research, not the normal private inspection path or accepted full-shirt cloth resolution.

`scripts/spike-full-shirt.py` expands all 24 fabric instances, retains per-instance source meshes and unchanged rest tensors, and records source/runtime identity. The initial radial stress run (120 steps, 3,802 vertices, 6,360 triangles, 284 sparse constraints) **failed**: maximum edge ratio 238.54, maximum seam gap 1,293.17 mm, 126 seconds, 751,608 KiB peak RSS. It was a deliberately unvalidated placement/registration stress case and must not be used as evidence that garment assembly works. Its captured digests identify the earlier experiment version, not the current script.

The next trial uses a versioned body-free rigid staging recipe derived from source shoulder/torso/sleeve dimensions, with gradual sewing-spring activation. Every transform is proper and preserves rest-edge lengths. Independent review found that the experiment originally bypassed declared physical mirroring and triangle winding; this was corrected before retaining staged results. The experiment remains hard-classified `rejected-experimental-assembly` regardless of seam residuals. No localized closure constraints, binding wraps, turning, material calibration, body contact, independent collision acceptance or convergence profile has yet been implemented for the complete shirt. Cloth orientation and physical layer order require further validated semantics. A successful timestep or a lower numerical residual cannot override those missing gates.

Reproduce staged research (separate candidate environment; install `scipy==1.17.0` there for the optional quality path):

```sh
timeout --kill-after=5s 300s .planning/solver/newton-venv/bin/python scripts/spike-full-shirt.py --output .planning/solver/new-staged-run --steps 120 --shirt-placement
timeout --kill-after=5s 300s .planning/solver/newton-venv/bin/python scripts/spike-full-shirt.py --output .planning/solver/new-quality-run --steps 10 --shirt-placement --quality-refinement
services/engine/.venv/bin/python services/engine/placement_test.py
```

The two commands use different step counts and cannot establish a controlled resolution-comparison pass. Hard process budgets remain 240 CPU seconds and 4 GiB, with a 300-second external wall timeout. Cold quality meshing is included in those budgets. Private application jobs continue to expose only source-linked placement inspection.

The corrected mirrored staged run completed 120 steps with unchanged rest tensors but **still failed**: edge ratios 0.07256–164.7073, maximum/p95 seam gaps 456.11/236.50 mm, maximum speed 0.4788 m/s, 170.84 seconds and 752,140 KiB RSS. The [full-shirt numerical ledger](3d-full-shirt-results.json) retains source/version digests and original-report checksums for this run and the earlier radial failure. The corrected source snapshot remains beside the ignored research artifacts. These runs are historical evidence when their script digests differ from current code.

The first full quality-refinement attempt hit the 4 GiB address-space limit while meshing, before producing solver metrics. A follow-up explicitly limits OpenBLAS/OMP to one thread to distinguish thread address-space reservation from geometry cost. Back-panel outward/right-side fabric orientation remains underdefined by source winding and is not silently changed to obtain a favorable contact result.

Inspection of sleeve-binding construction identifies a further concrete gate: the seam-line strip is only 20 mm wide at the 10 mm allowance fixture, whereas the real cut domain is 40 mm wide. Wrapping the allowance requires cut-line cloth and an interior stitch/fold path. Applying a boundary weld to the existing seam-line shell would remove required fabric and cannot implement that construction.

## Cut-domain foundation and isolated contact controls · 2026-09-17

`services/engine/cloth_domain.py` now tessellates actual `draft.cutLine` fabric and embeds every named original seam path in cloth triangles. Interior path samples preserve source segment/fraction, arc length and barycentric correspondence; segments remain inside their declared triangle. The independent runtime validator checks cut coverage, nonoverlap, winding, boundary topology, original vertex identity, concavity-safe source supports, complete path continuity and bounded inputs. Six tests cover all 18 source templates and corruption cases. The binding fixture retains its 40 mm cut width around the 20 mm seam domain. These are not particle-index stitch constraints: a weighted solver adapter, wraps, turning and allowance contact remain unimplemented. The application viewer still uses explicitly labeled seam-line placement inspection.

To isolate the numerical instability, `scripts/spike-full-shirt.py` now supports `--fixture front-panel` and `--fixture torso`, `--no-sewing`, `--disable-contact` and optional `--rest-neighbor-filters`. Every result remains rejected experimental assembly. Missing seam measurements are null when no sewing constraints exist. Source digests include placement code even when radial staging is selected.

The source-bound [control ledger](3d-contact-control-results.json) contains 30-step unsewn front-panel controls with zero gravity, unchanged rest tensors and identical 606-vertex/1,056-triangle geometry:

| Contact configuration | Maximum edge ratio | Maximum displacement |
|---|---:|---:|
| Disabled diagnostic control | 1.000144 | 0.001403 mm |
| Standard self-contact | 1.150886 | 5.012063 mm |
| Whole-primitive local filters, final v2 | 1.147317 | 4.797691 mm |

This isolates a contact-related contribution without establishing the entire full-shirt failure's cause. The initial endpoint-only v1 filter reduced the maximum ratio to 1.017852, but review rejected its ability to exempt distant primitive interiors merely because an endpoint was nearby. That result is retained as historical rejected evidence. V2 restricts exclusions to same-instance whole primitives within a bounded immutable rest-edge neighborhood. It does not solve the stationary-panel instability, and no tolerance was relaxed to accept it.

Seven filter tests cover candidate-map locality, long-primitive and separate-instance negative controls, source-transform invariance, budgets, and an actual one-step Newton layered-contact probe. The dynamic probe establishes retained interlayer separation response only; it does not validate full dynamics, folding, thickness clearance or convergence. A 3,225-vertex/6,260-triangle quality-refined front-panel trial exceeds the unchanged four-million candidate-check budget before solver stepping, including after candidate deduplication. Both failures are recorded. This configuration is therefore not supported within the current research budget.

Independent [cloth/contact review](../reviews/3d-cloth-contact.md) records the fixed source-support and winding attacks and the exact scope of dynamic evidence. Further work must solve mesh/contact resolution and weighted interior sewing before progressing to complete assembly, material drape and body/contact acceptance. The new allowance foundation is not integrated into a production simulator.
