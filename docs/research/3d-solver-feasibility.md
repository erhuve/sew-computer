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

Independent [cloth/contact review](../reviews/3d-cloth-contact.md) records the fixed source-support and winding attacks and the exact scope of dynamic evidence. The following continuation implements experimental weighted interior sewing and revises the contact/refinement controls; the allowance foundation is still not integrated into a production simulator.

## Embedded sewing and pointwise contact · 2026-09-17

`embedded_constraints.py` compiles exact source-arc samples into sparse barycentric constraints, independently revalidates them against cut-domain geometry, and performs mass-weighted XPBD projection in metres. Unequal seam lengths retain their distinct rest arcs. Multi-layer registrations use independent star constraints. Duplicate physical registrations, repeated terms, altered supports and malformed numeric inputs fail rather than doubling stiffness. Pinned particles and mass-weighted centre of motion are preserved. Optional time-varying target offsets stage closure without changing the rest mesh. Eleven tests pass; this is a coupling primitive, not completed assembly semantics.

The separate Newton adapter in `scripts/solver_point_contact.py` filters using the **current contact point's immutable source-material position**. It certifies short same-sheet routes through source containment or bounded rest-edge paths rather than exempting whole triangles or edges because one endpoint is nearby. Separate physical layers and distant folded regions retain contact response. Seven tests include actual compiled-kernel notch attacks and dynamic layer/fold controls. Optional seam exemptions require both corresponding source anchors and small radii; they do not exempt entire seams or layers. They are not certified for future slit/disconnected-domain topology. The adapter verifies its pinned runtime and topology, records generated-kernel identity and uses an isolated process; it does not modify the installed Newton package.

The [numerical ledger](3d-embedded-sewing-results.json) retains accepted-false results and source/report checksums. Initial immediate projection closed distant source points with corrections up to 260 mm, so it was not a plausible assembly path. Staged closure reduces that impulse, but residual alone still gives false confidence. Independent review also identified both test panels starting on the same side of their right seam. `probe_placement.py` adds a source-derived proper rigid rotation, centre alignment and a 5 mm gap. Six tests verify lengths, winding, translation covariance and unsupported-path rejection. No fabric dimensions are scaled to close a gather.

Final opposed, staged controls use actual placket/frill cut templates, 240 steps, four substeps and a 120-step closure ramp:

| Probe | Stitch samples | Maximum edge ratio | Final surface-pair count | Maximum speed |
|---|---:|---:|---:|---:|
| Equal plackets | 5 | 1.002591 | 28 | 0.0753 m/s |
| Placket/frill gather | 5 | 1.045213 | 315 | 0.6780 m/s |
| Placket/frill gather | 21 | 1.043076 | 289 | 0.5195 m/s |

All three have seam residuals below 0.00005 mm and unchanged rest tensors, yet **all remain rejected**. The diagnostic counts include some noncoplanar touches and exclude adjacent faces; they are not a swept/thickness contact oracle. Cut allowances extend across each seam and require actual fold/press/layer semantics, not broader collision exemptions. Projection after VBD contact can reintroduce penetration. Full-sheet settling, convergence, binding wraps, turning, gravity and calibrated materials remain open.

The 24-piece unrefined full-shirt trial with pointwise contact still fails: maximum edge ratio **140.906839**, maximum seam gap **342.146 mm**, 231.08 seconds. This trial retains the older experimental boundary-spring assembly, not the new cut-domain interior coupling. Improving its contact detector does not make its assembly recipe executable.

## Quality-refinement batch correction · 2026-09-17

The former refiner could insert nearby circumcentres from a stale triangulation in one batch. That created approximately 3-micrometre **new interior edges**, despite a shortest source-boundary edge of 1.595 mm. At metre-scale float32 placement, even the contact-disabled one-step control reported a maximum edge ratio of 1.024918. This was not a fabric material property or a reason to relax strain acceptance.

The corrected batch accepts centres only when they remain outside each other's original circumcircles. Original source vertices and boundaries remain unchanged. The synthetic front drops from 3,225 to 375 vertices, with its shortest edge now 1.595 mm. The back has 399 vertices and a shortest edge of 0.614 mm. The new regression and existing source validator check actual body panels; independent rotated/translated probes preserve the source. These results do not guarantee a minimum feature size for arbitrary patterns.

On the refined front's 30-step stationary control, contact-enabled maximum displacement is **0.000741 mm**, maximum edge ratio **1.000036**, and harness runtime **4.79 seconds**; the previous 3,225-vertex version exhausted its 150-second CPU budget. The contact-disabled counterpart reports 0.000794 mm and 1.000068 in 2.45 seconds. These final timings include quality meshing; the older one-step timer excluded it, so those runtime values are not a controlled performance comparison. Request/source snapshots identify the exact runs. This validates a narrow stationary control, not hanging cloth or the complete shirt.

The first full-shirt retry exposed a different failure at the collar stand: distinct registration fractions rounded to an identical generated boundary coordinate. Refinement repeatedly split the resulting zero-length segment without increasing the unique-vertex count, exhausting the 4 GiB limit. The fix skips exactly identical consecutive generated coordinates without moving source vertices and adds explicit zero-length, unsplittable-midpoint and segment-count guards. Three refinement regressions pass, including actual collar registration/source checks. All 18 templates now pass the synthetic full-shirt registration/quality-mesh preflight; collar stand and fall have 60/39 vertices. The failed attempt is retained in the ledger rather than replaced by the successful preflight.

The final refined full shirt in that run has 24 physical fabric instances, 2,656 vertices and 3,816 triangles. It becomes nonfinite at **step 73 of 120**, after 82.82 seconds. Two position components and two velocity components are nonfinite; rest coordinates remain finite. The fail-fast harness preserves a strictly valid rejected JSON report and a checksummed binary failed state instead of losing diagnostics to JSON serialization. Failures also carry physical-instance offsets. An independent injected-NaN test verifies this rejection path; it is separate from the natural failure. Final finite-state metric calculations use float64 to avoid float32 norm overflow. The successful one-step front-panel smoke check does not promote the failed shirt.

## Full cut-cloth sewing continuation · 2026-09-17

The [stability ledger](3d-sewing-stability-results.json) retains the controls after that failure. Newton's default coloring omits sewing springs; refinement removes 117 conflicts while preserving cloth/bending separation. A pinned CPU-only membrane adapter replaces cancellation-prone area/cofactor arithmetic with equivalent cross-product expressions, retaining the material law and area floor. Independent derivative, ownership/restoration and contact-coexistence checks are in the [review](../reviews/3d-embedded-sewing.md). Neither correction makes an underconverged spring solution physically valid.

The complete research path now supports actual cut fabric via `--embedded-sewing`, substeps and a smooth offset ramp with settling time. It validates all source paths, resolves declared whole-edge endpoints against exact source geometry, and preserves physical mirroring and rest tensors. Refined sleeve meshing excludes exterior slivers rather than relaxing coverage checks. Both coupling harnesses copy the pre-step CPU state; historical embedded trajectories used an alias that invalidated their velocity reconstruction. New canonical outputs retain velocities and previous positions for independent checks.

Corrected full-shirt no-contact run: **24 fabric instances, 2,244 vertices, 3,512 triangles and 305 constraints**, 120 steps with two substeps and an 80-step ramp. It is finite but rejected: maximum edge ratio **11.3479**, seam gap **28.3465 mm**, speed **6.0894 m/s**. Independent artifact validation reproduces geometry, residuals and reconstructed velocities. This does not establish contact, layer order, turning, body interaction or settled material drape.

Reproduce this experimental path with the pinned research Python and a new private output directory:

```sh
timeout --kill-after=5s 300s .planning/solver/newton-venv/bin/python scripts/spike-full-shirt.py --output .planning/solver/new-cut-cloth-run --embedded-sewing --shirt-placement --quality-refinement --stable-membrane --disable-contact --steps 120 --substeps 2 --ramp-steps 80
.planning/solver/newton-venv/bin/python scripts/test_solver_constraint_coloring.py
.planning/solver/newton-venv/bin/python scripts/test_solver_membrane_stability.py
.planning/solver/newton-venv/bin/python scripts/test_solver_spike_geometry.py
.planning/solver/newton-venv/bin/python scripts/test_full_shirt_embedded.py
```

The final same-duration cut-cloth run with pointwise contact also completes, in **237.57 seconds**, with finite states and unchanged rest tensors. It remains rejected: edge ratios **0.1202–22.5654**, maximum seam gap **5.2443 mm**, maximum speed **0.8228 m/s**. Its captured source digests match the final implementation, including the endpoint guard, copied state, membrane adapter and refined sleeve fix. Contact being enabled is not a collision-acceptance result; the solver does not yet establish valid layer order or coupled convergence.

An independent final-surface check of that contact run reports **7,186 intersecting nonadjacent triangle pairs**, with the oracle's documented touch/exclusion limits. This further rejects the result; enabling self-contact alone does not establish successful collision handling.

These are research controls, not production configuration. Allowance folds/layer execution, simultaneous versus staged assembly operations, coupled sewing/contact convergence and scoped numerical acceptance remain unresolved.

Current verification includes 104 application tests, 31 engine cases and 23 browser cases in full runs, followed by all six meshing/placement/coupling wrappers against the collar fix. Typecheck and build pass. Independent numerical review covers eleven coupling, seven contact, eight surface-oracle/state and six rigid-probe tests, plus three refinement regressions. Numerical acceptance of the assembled shirt remains a separate unresolved gate.

## Source-endpoint orientation continuation · 2026-09-17

Independent endpoint-equivalence review proves that the previous collar registrations forced three nonzero neckline intervals to collapse through shoulder, center-back and placket junctions. The research harness now uses the same source-endpoint direction recipe for embedded constraints and boundary springs, independently of placement. Neck intervals 1, 3 and 5 reverse their garment-side participants; collar-stand facing retains shell orientation, and placket-right facing reverses with its shell. Collar-fall shell/facing use the opposite traversal to the stand attachment. Four new topology regressions preserve all seven collar-chain junctions and sleeve shoulder/underarm connectivity. This recipe is scoped to the trusted shirt compiler; it does not resolve binding wraps or turning.

The [orientation control ledger](3d-registration-topology-results.json) retains original report/canonical checksums, captured source digests, runtime versions and arguments for the old and new trajectories. Every result remains rejected:

| Control | Maximum edge ratio | Maximum seam gap, mm | Maximum speed, m/s |
|---|---:|---:|---:|
| Previous, contact disabled, 120 steps | 11.347855 | 28.346453 | 6.089410 |
| Corrected orientation, contact disabled, 120 steps | 2.895634 | 0.899999 | 4.913456 |
| Corrected orientation, contact disabled, 360 steps | 3.318603 | 0.979860 | 4.951132 |
| Previous, pointwise contact, 120 steps | 22.565358 | 5.244292 | 0.822834 |
| Corrected orientation, pointwise contact, 120 steps | 40.334746 | 0.967937 | 0.810883 |

The 120-step runs use two substeps and an 80-step closure ramp. The longer no-contact run uses a 240-step ramp. All retain 24 pieces, 2,244 vertices, 3,512 triangles and 305 constraints. Independent artifact checks confirm exactly unchanged rest geometry, triangulation and initial placement for the no-contact comparison. Correct connectivity reduces its error but does not establish convergence; the contact run's maximum local stretch worsens. Small final seam gaps cannot justify either output.

Verification for this research-only change: 104 application tests, eight meshing/placement/coupling wrappers (including four independently authored topology tests), five embedded-shirt integration tests, typecheck and documentation checks pass. Browser and build checks were not repeated because no application or rendering code changed. See the [review](../reviews/3d-embedded-sewing.md) for independently recomputed artifact metrics and the remaining scope limits. Production remains unchanged.

## Global membrane reference and torso control · 2026-09-17

The research harness now records per-substep membrane/kinetic energy, momentum, moving-target residuals and solver convergence. `solver_torso_control.py` supplies a source-validated translation witness for four torso shells, optionally separated by a rigid front-panel gap. This intentionally overlapping, contact-disabled fixture isolates sewing numerics; it is not wearable placement.

`solver_global_sewing.py` implements a CPU float64 sparse reference for implicit inertia, the existing membrane energy and weighted interior sewing. It rejects unsupported gravity, bending, damping and other force elements. The direct method uses the analytical membrane Hessian projected to a positive-semidefinite search metric, while preserving the original objective and gradient. The line search evaluates a rest-referenced algebraic energy to avoid masking tiny descent behind the large constant in the least-squares residual norm. Invalid collapsed initialization candidates are excluded; the valid previous state remains available. End-state nondegeneracy is not swept inversion or collision validation.

Independent tests check analytical derivatives, rigid covariance, translation nullspaces, weighted pinned anchors, the exact two-triangle solution, momentum preservation, stable-energy equivalence and collapsed-prediction recovery. Projection changes only the search metric, not material parameters or pattern dimensions. Reports always retain `accepted: false`.

The 5 mm perturbed torso control contains 1,366 vertices, 2,196 triangles and 25 embedded constraints. With 12 steps, one substep and a two-step closure ramp, every step meets the unchanged **1e-6 N** gradient infinity-norm threshold. Maximum edge ratio is **1.000504611**, minimum **0.999830224**, final maximum seam gap **0.000000519 mm**, and center-of-mass displacement **5.80e-12 mm**. Runtime is **70.59 seconds**, including meshing. These are floating-point diagnostic measurements, not physical precision claims. The earlier projected-Hessian run with residual-norm energy failed its second step despite a converged final step; all-step convergence must be checked.

Reproduce with the pinned research environment in a new private directory:

```sh
OPENBLAS_NUM_THREADS=1 .planning/solver/newton-venv/bin/python scripts/spike-full-shirt.py --output .planning/solver/new-global-torso-control --steps 12 --embedded-sewing --global-reference --membrane-only-control --substeps 1 --ramp-steps 2 --quality-refinement --torso-equilibrium-control --torso-front-gap-mm 5 --fixture torso --disable-contact
.planning/solver/newton-venv/bin/python -m unittest discover -s scripts -p 'test_solver_*.py'
```

The full-shirt eight-step stress attempt with a six-step closure ramp exhausts the 240-second CPU limit during its sixth substep. Its first five completed substeps all fail convergence, with force residuals from `0.0005926` to `0.1193 N`. No final garment metrics exist for that interrupted run. The [ledger](3d-global-reference-results.json) retains its captured sources, arguments and partial trajectory digest. The harness now captures global numerical exceptions and CPU-limit signals delivered during the global solve as rejected reports with the previous valid state, and summarizes every substep rather than only the last one. Hard kills and interruptions outside that solve can still leave incomplete artifacts, which remain rejected.

This closes a narrow coupled membrane/sewing control. Full-shirt convergence, bending, gravity, contact, layer operations, body interaction, calibrated materials and production simulation remain open. Next, isolate full-shirt convergence using staged seam subsets and bounded residual diagnostics before adding contact to the reference. The production application and its placement-inspection classification are unchanged.
