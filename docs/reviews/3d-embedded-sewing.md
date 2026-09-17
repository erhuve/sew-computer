# Embedded sewing and material-point contact review

**2026-09-17 · Independent adversarial review · Experimental coupling, not garment acceptance**

Reviewed `embedded_constraints.py`, its test suite, `spike-embedded-sewing.py` and the experimental `solver_point_contact.py` adapter. These extend the unfinished [engine plan](../plans/pattern-derived-3d-engine.md); no assembled garment or material drape is accepted by this review.

## Embedded constraints

All eleven coupling tests passed independently after adding staged targets. They cover arbitrary interior arc samples, reversed and unequal-length multi-layer registration, sparse Jacobians, mass-weighted corrections, fixed particles, compliance/timestep equations, captured-source mutations, duplicate registrations and target validation. Five additional randomized positive-mass probes preserved total mass-weighted position to at most `8.89e-16` numerical error. Dropping a derived constraint or altering its source vertex or source arc rejected during source validation.

Review found that identical physical registrations with different IDs were accepted and implicitly increased seam stiffness. With compliance `1e-5`, timestep `1/240` and 20 projection iterations, the duplicate changed the synthetic maximum residual from approximately `0.005549` m to `0.003221` m. The final compiler rejects equivalent physical registrations regardless of ID, member order, sample count, compliance or simultaneous direction reversal. Distinct adjacent intervals may still share endpoints; that is not treated as a duplicate registration.

The coupling validator regenerates constraints from captured source meshes and registrations; projection alone checks numerical structure and cannot establish source authority. The projection uses squared barycentric coefficients in effective inverse mass and signed coefficients in corrections. This preserves weighted translation/mass balance for the tested free-particle cases. Pinned constraints report unresolved immovable residuals rather than silently moving fixed vertices.

XPBD multipliers are retained across iterations within a projection call. Reuse across calls is explicit; timestep changes and warm-start policy require care. The research harness resets multipliers each substep, applies VBD first, projects sewing afterward, and reconstructs velocity from the full substep displacement. This operator split may reintroduce penetration after collision handling. Seam closure or an unchanged rest tensor alone cannot validate physical strain, contact, stability or convergence.

Staged `target_offsets_m` subtracts an explicit bounded vector target from each physical constraint residual. The result keeps physical `residualsM` separate from `targetResidualsM`, so reaching an intermediate target cannot masquerade as closed sewing. Tests independently verify that the initial physical offset produces no correction, zero targets preserve prior behavior, changing targets preserves mass and immutable rest geometry, and malformed targets reject. The harness uses a smoothstep closure schedule ending at zero and reserves subsequent steps for attempted settling; the schedule does not itself prove convergence.

## Material-point contact

The experimental adapter maps the currently closest collision points back into immutable rest coordinates rather than excluding complete primitives based on any nearby rest endpoint. Physical instance IDs remain separate in the reference descriptor. This is a version-pinned, generated-kernel experiment, not a production Newton extension.

An independent compiled Warp probe exposed a boundary-vertex weakness in the initial material-neighborhood predicate: a line could cross two concave notches exactly at boundary vertices while its midpoint remained inside cloth. The original strict intersection rule missed those zero-orientation crossings. The corrected helper conservatively rejects interior boundary-vertex crossings; the same compiled two-notch probe now returns false, retaining the contact candidate. The initial helper also required an explicit dynamic Boolean declaration to compile its loop under Warp; the corrected probe compiles and executes.

Seven contact-adapter tests passed independently after adding intrinsic paths and optional seam-anchor exemptions. The dynamic test verifies negligible motion for a small stationary sheet and increased mean Z separation for two nearby physical layers and for distant, disconnected rest patches of the same instance placed near each other. A separate connected-strip fold test with bending disabled verifies an increased separation response with contact enabled relative to its disabled-contact control. These synthetic responses do not establish collision-free garment folding or convergence. The compiled two-notch query now has a durable regression; clearance tests verify conservative vertex and interpolated-edge bounds.

The final v2 adapter also permits a certified local route from each current material point to a source primitive corner and along bounded rest-mesh graph paths. This gives a path-length upper bound without bridging disconnected topology. Graph visits, relaxations and stored distances are capped; stored distances round upward. This restores some genuinely local material contact exemptions that strict straight-line boundary checks conservatively retained.

Optional seam exemptions require both current material points to lie within explicit radii of their corresponding paired source anchors and to match each anchor's exact physical instance. The harness derives these anchors from validated constraint source samples. Tests cover paired-local exemption, remote points, incorrect layers and malformed anchors. These are small source-space neighborhoods, not whole-edge or whole-layer exclusions. Euclidean anchor neighborhoods are not certified against future slits or nearby disconnected source regions within one instance; that remains a separate unsupported topology gate.

Review also reproduced acceptance of fractional triangle indices through integer coercion and a repeated-vertex triangle producing a degenerate boundary. The final independent recheck rejects both mutations and numeric instance IDs. Installation now checks model topology and the pinned runtime, limits its process-global kernel replacement to one active solver, and provides explicit restoration. Process isolation remains required for these experiments.

## Final-surface diagnostic oracle

Eight surface-oracle/state tests pass independently. Ten additional randomized 30-vertex surface checks agree with brute-force invocation of the narrow predicate. Review found a broad-phase false negative at the predicate's tolerance boundary: parallel unit right triangles separated by `1.2e-9` units were accepted by the narrow overlap predicate but omitted by the axis-span padding. The corrected padding uses the bounding diagonal, and the exact attack now produces one tested, intersecting pair. Broad-phase scanning and narrow candidate work are bounded separately.

This oracle omits topologically adjacent pairs. Noncoplanar touches may count, but coplanar boundary-only contact is omitted; positive-area coplanar overlap is counted. It checks the final surface only, without swept trajectories or cloth thickness. Thus reported pairs are diagnostics, and zero reported pairs would not certify physically valid contact.

The full-shirt harness now checks positions and velocities after each solver step. A nonfinite state produces a rejected JSON diagnostic with value counts, failed step, source/runtime identity and a checksum for a retained binary state, then raises an error before ordinary output serialization. Independently injected one NaN into Newton's first-step result in an isolated front-panel control using an in-memory test wrapper. The harness raised at step 1, reported exactly one nonfinite position value and zero nonfinite velocity values, and retained the NaN in `failed-state.npz`. The file checksum and all saved source-snapshot digests match the diagnostic. This was explicit fault injection, not a natural solver trajectory or evidence that captured source alone would reproduce the injected state. No implementation source file was edited for the attack.

The subsequent natural full-shirt quality run also failed: its diagnostic records step 73, 24 physical instances, 2,656 vertices, 3,816 triangles and 82.823 seconds. Independently loading its saved state with pickle disabled confirms exactly two nonfinite position values and two nonfinite velocity values; rest coordinates remain finite. The binary checksum and every captured source digest match. This confirms a retained numerical failure after meshing, not an accepted garment. Later statistical calculations cast finite float32 states to float64 before norms; an independent `3e38`-component probe confirms that this avoids float32 norm overflow without repairing or masking actual nonfinite solver values.

## Opposed placement control

Review confirmed that the first probes placed both panel interiors on the same side of their registered right seams. The second piece's small rotation about X did not change that fact. The new straight-path helper instead uses a proper rigid rotation to oppose transverse directions while preserving longitudinal orientation, aligns seam centers without scaling unequal lengths, and introduces an explicit Z gap.

All six helper tests pass independently. Additional actual cut-placket and frill checks give determinant `+1` and maximum triangle-edge length differences of `1.11e-16` m. The gathered source's 1,026 mm path remains 1,026 mm; its center translates by -228 mm along Y to align with the 570 mm placket's center. This is a controlled initial-placement correction, not assembled garment validation. Allowances still extend beyond each interior stitching path, and folding/pressing/layer separation remain unresolved.

## Quality-refinement correction

Reviewed the minimal circumcenter-batch fix in `quality_meshing.py`. Previously selected circumcenters came from a stale triangulation and could cluster close together. The revised batch requires mutual center separation at least as large as both original circumradii. It retains original source vertices and does not loosen boundary tolerances or reset rest geometry.

The new actual front/back regression passes independently. Six additional probes rotating both panels by 37°, 90° and 179° and translating by `(111, -77)` mm pass the independent rest oracle with zero reported area error. They contain 371–408 vertices, with shortest front edges approximately 1.59536 mm and shortest back edges approximately 0.614004 mm; no micron-scale Steiner clusters appeared in those probes. This is measured fixture improvement, not a general mesh-quality guarantee. Refinement remains bounded and `solverQualityAccepted` remains false.

A subsequent full-shirt attempt exposed duplicate generated collar boundary coordinates: distinct registration fractions rounded to an identical point, creating a zero-length segment whose repeated splitting grew the segment list without consuming new vertices. The final patch removes consecutive exactly equal generated boundary coordinates and rejects zero-length segments, midpoint splits that cannot progress at machine precision, and excessive segment counts. Original source vertices are not merged or shifted. All three final quality tests pass independently, including actual collar registrations and the zero-edge attack. Independently translated, unrotated collar stand/fall probes preserve every quarter registration to `2.85e-14` mm.

Additional rotated collar probes still fail closed. At 37°, the collar stand starts with 48 unique boundary coordinates and a 3.75 mm shortest edge, but SciPy's unconstrained triangulation emits nominally collinear boundary triangles with cross products around `6e-13` mm²; the independent oracle rejects them. The 37° collar fall and both 90° collar probes hit the new unsplittable-precision guard. These are remaining robustness limits, not evidence that source endpoints should be merged by tolerance. No geometry was accepted from those failed probes.

## Retained numerical evidence

Independently read the five retained embedded-coupling outputs: equal baseline, translated baseline, smaller timestep, gather baseline and contact-enabled baseline. All retain `accepted: false`. Regenerating their captured source constraints with the final validator passes; independently recomputed maximum residuals match every report exactly. These are checks of collaborator-produced numerical outputs, not independently repeated Newton trajectories.

The equal baseline's reported final maximum seam residual is approximately `0.00000952` mm with edge ratios `0.999857`–`1.000176`. The gather baseline reports approximately `0.00001472` mm residual and ratios `0.996730`–`1.011700`, but a maximum speed of approximately `0.149` m/s. The contact-enabled equal baseline reports maximum speed approximately `0.043` m/s. Small residuals therefore do not establish settled cloth or accepted collision behavior. Source rest correspondence is retained; whole-garment attachment, folds, binding wraps, material calibration and independent collision/convergence oracles remain unfinished.

Subsequently revalidated the three retained same-side anchor runs. Their independently recomputed surface counts are 40, 194 and 304 for equal, gather and slower gather respectively; physical residuals match their reports exactly. These remain failed stress tests, not superseded evidence that should be discarded.

Finally revalidated all three opposed-placement canonical bundles against captured source constraints. The independently recalculated surface counts and residuals match the reports; recomputed maximum strain ratios differ by less than `4e-8` due to float precision. These trajectories were produced by the implementation agent, not independently rerun by the reviewer:

| Opposed probe | Final surface pairs | Maximum edge ratio | Reported maximum speed, m/s |
|---|---:|---:|---:|
| Equal | 28 | 1.002591 | 0.075311 |
| Gather | 315 | 1.045213 | 0.677991 |
| Gather, denser stitching | 289 | 1.043076 | 0.519496 |

Every opposed probe remains `accepted: false`. Opposing the panel bulk and increasing stitching samples do not resolve allowance contact, dynamic settling or complete assembly semantics. The reported surface pairs use the narrow diagnostic scope described above; they must not be silently reclassified as permitted seams to promote these outputs.

Neither this document nor the tests constitute complete assembly, physical-fit validation, or release approval.
