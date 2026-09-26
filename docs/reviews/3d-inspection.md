# Private 3D inspection: adversarial implementation review

Current extension: the [end-to-end studio](studio-e2e.md) adds a private, revision-bound posed garment alongside this original flat-piece inspection. The evidence below describes the earlier inspection checkpoint; it does not certify physical drape.

**2026-09-17 · Independent review · Placement inspection only · Not deployment approval**

## Scope and evidence boundary

Reviewed the durable 3D queue, schema-5 migration, private artifact installation and serving, source/display validation, trusted inspection runner, React viewport and source selection. Also reviewed the explicit assembly ledger and exact boundary-registration meshing additions. The [dedicated plan](../plans/pattern-derived-3d-engine.md) remains unfinished: flat physical pieces are not an assembled garment or simulated drape. The ledger declares `solverReady: false`; orientation, turning, binding, interfacing and validated material/contact behavior remain unresolved.

This review follows the [foundation review](3d-engine-foundation.md). Solver experiments and optional quality refinement do not acquire production acceptance from the inspection tests. No production deployment or physical-fit evidence was reviewed.

## Findings and rechecks

| Finding | Consequence | Resolution and independent evidence |
|---|---|---|
| Numeric `z.enum` declarations rejected valid GLB targets and component types | Every legitimate display artifact failed installation | Changed to numeric literal unions. Independently passed a saved full-shirt canonical report and GLB through the corrected validator. |
| Coordinate reconstruction did not validate triangle coverage or topology | A report and matching GLB containing 1,936 copies of one front-panel triangle were accepted | Independently reproduced acceptance, then reran the same corruption after the fix: rejected as overlapping/nonmanifold. Validation now checks positive winding, area, used vertices, oriented internal edges and complete source boundaries. |
| Physical mirroring and grain metadata were trusted | Wrong-handed output could retain plausible source identifiers | Expected handedness and source grain are independently checked. Reviewer mutations of mirror and grain now reject. |
| Source interpolation only reconstructed positions | A support triangle crossing a concavity could falsely describe an otherwise correct vertex | Added support containment and original-boundary identity checks. Independently found an actual front-panel interior vertex with reconstructing weights whose support crossed outside the source; the final validator rejects it. |
| GLB accessor extrema were not verified | Incorrect bounds could hide correct geometry or break camera framing | Extrema now match decoded float32 coordinates. Independently rewrote accessor maxima to `1e50`, rebuilt the valid GLB envelope and recalculated its report digest; rejected as display bounds mismatch. |
| Completed results were not identified as historical after source replacement | An earlier same-revision pattern derivation could appear current | Latest responses now compute `sourceCurrent` against captured source/runtime identity. The UI displays a historical warning and rebuild action. Independently reviewed the source change; this is not a separate executed same-revision browser test. |
| A polling authorization/deletion error retained the loaded scene | Deleted private geometry remained visible in the mounted viewport | 401/403/404 clear the job and unmount the scene; 401 propagates to the studio authentication handler. Independently reviewed cleanup and abort paths. Browser deletion verification is reported below. |
| WebGL initialization failure left the promised piece selector empty | Devices without graphics support lost source selection | Metadata populates the selector before renderer creation. Independently reviewed the fix; browser fallback verification is reported separately. |
| Inspection code identity omitted renderer/worker dependencies and runtime lock checks | Worker output could change without the captured implementation identity | Runner captures and rechecks transitive inspection source bytes, including GLB generation, guard and lock; worker verifies locked dependency versions. Independently reviewed these checks. |

## Independently executed verification

- Interpretation service and durable-job suites: **17 tests, 88 assertions passed**, including a 31-second provider response, cancellation, reopened-store recovery, expired leases, concurrent edits and schema-3 preservation.
- Updated Python source/assembly suite: **17 tests passed**, including physical inventory, explicit graph coverage/order, invalid source/closure mutations, exact boundary registration and the optional actual-panel quality-refinement case. This is source/rest validation, not full-shirt solver acceptance.
- Initial 3D lifecycle suite: **6 tests, 29 assertions passed**, covering acknowledgement/idempotency, independent 2D fences, cancellation, changed revision, deletion, lease replacement, invalid output, changed engine identity and schema-4 to schema-5 preservation. These fixtures do not themselves demonstrate a successful real-worker installation.
- Restricted-worker HTTP integration: **1 test, 37 assertions passed** independently. The real pattern and inspection workers publish all 24 instances; unauthenticated, wrong-origin and cross-project access reject. Private responses are non-cacheable. Same-revision source replacement marks the old result historical while preserving authorized historical access. Restart reconciliation preserves installed blobs, export disclosure excludes 3D, and project deletion removes artifact access and stored blobs.
- Direct valid-artifact and corruption checks used the saved full-shirt pattern, canonical inspection JSON and GLB. The valid artifact passed; repeated triangles, wrong mirror, altered grain, false vertex identity, concavity-crossing interpolation support and false accessor extrema rejected. The repeated-triangle and accessor attacks updated the display bytes and digest together, so checksum rejection did not mask the validation defect.

## Browser evidence reported by implementation agent

The implementation agent reports a passing real-worker browser flow from a generated full shirt through persisted 3D completion, reload, one transient 502, rendering and all 24 physical pieces. The test also covers camera controls, source selection in the 2D view, mobile horizontal overflow, unchanged source geometry, the existing eight-file export with no 3D artifact, and scene removal after project deletion. The reviewer inspected `tests/browser/three-d.spec.ts`; this is collaborator-reported execution, not an independently rerun browser result.

The implementation agent subsequently reports both final 3D browser cases passing, including WebGL-unavailable source-selection recovery and context-loss retry. The full browser run passed 22 of 23 cases; its mobile pattern-above-fold failure was fixed by shortening the visible 3D tab label, and the affected case passed on the rebuilt app. Both 3D cases then passed again. Overall integration evidence is 104 application tests, 26 engine cases and 23 browser cases across full/focused runs, plus passing typecheck/build/documentation/whitespace checks. These aggregate and browser counts remain implementation-agent evidence, distinct from the independently executed checks above.

## Privacy, lifecycle and remaining gates

3D artifacts reside in separate authenticated storage tables, participate in deletion and orphan reconciliation, and are not included in existing export bundles. The strict server GLB schema permits only embedded geometry, bounded source-linked nodes and supported fields; the browser additionally rejects URI/extensions and blocks external loader URLs. Existing source checks are distinct from permission checks: a digest never grants artifact access.

The queue captures immutable revision, pattern artifact/head, construction, project generation and implementation identity. Unique attempt leases fence retries and late output without incrementing the existing 2D job fence. Installation occurs through the existing transactional blob installer after validation and a final attempt check. Source review found no cross-project lookup or export inclusion bypass; the final release matrix still needs its applicable executed authorization and recovery cases.

No remaining blocker was identified in the reviewed placement-inspection paths after the listed fixes. This conclusion does not accept full assembly, numerical solver quality, material-aware drape, custom-body exports or the complete 3D workstream. Schema 5 requires a schema-5-capable rollback binary; an older application must not be presented as a safe application-only downgrade.

## Separate full-shirt experiment review

The reviewer inspected `quality_meshing.py`, `placement.py` and `scripts/spike-full-shirt.py`. These are research additions, not the private viewer's assembled output. The full-shirt script hard-codes rejection and explicitly records absent gravity, body, material calibration, collision oracle and convergence acceptance. It records seam residuals and deformation relative to original rest coordinates rather than resetting rest lengths to make assembly appear successful.

Independently tested all 24 initial shirt-placement frames: their bases are orthonormal with determinant +1, and transformed point-pair lengths differ by at most `2.84e-14` mm. This establishes rigid placement, not correct sewing orientation or collision-free staging.

Independently exercised optional quality refinement on the actual front-panel shape translated by `(111, -77)` mm at three rotation/resolution combinations. All passed the independent rest-fidelity oracle; none claimed solver quality acceptance:

| Rotation | Maximum edge setting | Vertices | Measured maximum edge | Minimum triangle quality |
|---|---|---|---|---|
| 0° | 40 mm | 3,630 | 39.699 mm | 0.1533 |
| 37° | 60 mm | 7,392 | 59.509 mm | 0.1123 |
| 90° | 60 mm | 3,286 | 58.849 mm | 0.2033 |

These checks establish source preservation under the tested perturbations, not identical triangulation or simulation invariance. The Cartesian seeding grid changes mesh topology/count with rotation. The bounded refinement can finish without meeting its target quality; retained quality measurements and `solverQualityAccepted: false` must remain attached to results.

Review identified an experimental orientation defect: the initial full-shirt path used unmirrored local triangles while handed staging changed right-side frames, yielding inconsistent left/right front normals. The corrected path reflects physical rest X and reverses mirrored triangle winding before solver construction; placement frames now explicitly consume these physical coordinates. Independently rechecked all 24 corrected frames and the placement regression: length preservation remains within `2.84e-14` mm, and both front normals point toward front-Y. Runtime evidence now includes SciPy when refinement is enabled, and canonical output includes the physical inventory and placement frames.

Back-panel fabric-side orientation remains a separate unresolved semantic choice: the source's triangulation winding does not define which material face must point outward after sewing. Current back normals point toward front-Y. The experiment must retain this limitation, rather than silently flip fabric orientation or claim that fixing mirrored front normals resolves numerical failure. This research issue does not change the accepted source-linked flat inspection.

Independently inspected the retained corrected 120-step staged full-shirt output and its source snapshot. Every recorded source digest matched the retained source copy; all 24 physical rest meshes exactly reconstruct from their declared template coordinates and mirror flags. Initial placement edge ratios were `0.9999999999976`–`1.0000000000010`. Final maximum strain ratio remained `164.707`, and the reported maximum seam gap remained `456.110` mm: an unequivocally rejected result. The largest strain occurred on a 0.694 mm back-panel source edge; another 0.148 mm front-panel edge stretched over 115-fold. These observations support investigation of tiny elements and constraint stiffness, not a claim that one identified cause explains the entire failure. The reviewer inspected saved solver outputs rather than independently rerunning Newton.
