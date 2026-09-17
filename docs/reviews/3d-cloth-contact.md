# Allowance cloth and research contact review

**2026-09-17 · Independent adversarial review · Research foundations only**

Reviewed `services/engine/cloth_domain.py`, `scripts/solver_contact_filters.py`, their tests and the research harness integration. These modules do not complete the [dedicated engine plan](../plans/pattern-derived-3d-engine.md), enable assembled output, or validate material drape.

## Independently executed checks

- All six final allowance-cloth tests passed. These include every real shirt template, cut-domain triangle coverage, source reconstruction, interior stitch samples and segments, and mutations of source coverage and derived metadata.
- Independently transformed all 18 real templates by each of 37°, 90° and 179°, with translation `(111, -77)` mm. All 54 cases passed the same independent coverage and embedding oracle at a 60 mm mesh setting. These tests used ordinary tessellation, not optional quality refinement.
- A panel-level hole mutation rejected. A hypothetical `slits` property is ignored; no current pattern slit schema was found. Explicit slit support remains a future semantic gate, and these meshes must not be described as supporting slit topology.
- All seven final v2 contact-filter tests passed in the pinned Newton research runtime, including the dynamic layered probe. Separate physical layers and disconnected patches retain contact candidates; distant same-sheet primitives remain outside the exclusion maps; symmetry, transformation invariance, primitive validation and bounded search reject cases pass. A near-corner regression preserves long primitives as contact candidates.

The test named `test_local_neighborhood_and_remote_fold_candidate_maps` checks exclusion-map membership only. It creates a folded coordinate array but does not run it through Newton. The static coincident-layer test likewise verifies candidate preservation. A separate dynamic one-step test now verifies finite results and increased mean layer separation along Z with contact enabled, while the disabled-contact control preserves its initial 0.5 mm separation. This establishes a contact response for that synthetic fixture, not collision-free cloth or full-garment acceptance.

## Findings and independent rechecks

The initial runtime validator accepted reconstructing source weights whose support crossed a cut-contour concavity. The concrete `front_left` attack used cut vertices `(0, 1, 3)`, whose support extended approximately 113.426 mm² outside the source polygon. The final validator rejects both the original-vertex mutation at vertex 44 and an interior mutation at vertex 97, the latter specifically through support containment. Duplicate supports and changed original cut-vertex identities also reject.

The initial validator also accepted reversed triangle winding. The final recheck rejects that mutation and a `solverReady: true` classification mutation. The validator additionally checks used vertices, oriented edge incidence and interior free boundaries. A changed stitch source fraction rejects. An edge ending at `10^12` and a Boolean interval index both reject before range expansion; generated and validated stitch samples have global caps. The final runtime validator was included in a repeated 54-case transformed-panel check.

## Source and contact boundaries

Allowance geometry is explicitly derived from `draft.cutLine`, not relabeled as seam-line geometry. Per-vertex weights reconstruct cut-contour coordinates. Stitch paths reconstruct the original seam-edge polylines; per-triangle segment splitting permits future weighted constraints without substituting nearest vertices. The module retains `solverReady: false`: binding wraps, turning, folds, material bulk and weighted-constraint execution remain absent.

The contact filters use immutable rest-edge path distance, not current spatial proximity. Triangle and edge validation prohibits crossing physical instance IDs. This prevents coincident shell/facing pieces from becoming each other's excluded neighbors, and disconnected patches cannot be joined merely because their rest coordinates coincide. Newton's installed SolverVBD interface accepts the vertex-to-triangle and edge-to-edge exclusion maps used by the harness.

The original v1 endpoint-neighborhood method was rejected because a nearby endpoint exempted an entire long primitive, including distant cloth. Historical v1 numerical outputs must not be treated as acceptance. The final `same-instance-whole-primitive-rest-geodesic-v2` profile requires all triangle corners to be local to the query vertex and every cross-edge endpoint pair to be local before exclusion. Long source edges are also conservatively retained. Close primitive interiors can still remain candidates when endpoints are distant. A stationary-piece improvement therefore cannot certify safe self-contact during folding. Full dynamic contact and independent collision-oracle gates remain required.

No blocker was found in these narrowly labeled source-embedding and research-filter paths by the checks above. This is not release approval, full-shirt assembly acceptance, or an assertion that the garment simulation succeeds.
