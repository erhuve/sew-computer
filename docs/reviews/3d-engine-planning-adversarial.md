# Pattern-derived 3D engine planning adversarial review

**2026-09-16 · Independent planning review completed · No runtime validation claimed**

Scope: the [dedicated engine plan](../plans/pattern-derived-3d-engine.md), [project contract §14](../design/project-contract.md#14-pattern-derived-garment-3d), and current shirt compiler, pattern contracts and job lifecycle. A separate reviewer inspected source, challenged the draft and rechecked the revisions. This reviews the plan, not simulator feasibility, sewing accuracy or implemented security.

| Finding | Resolution and required future evidence |
|---|---|
| High: hiding an avatar does not redact a custom-body-conditioned garment mesh | Explicit pattern/3D/body/avatar export matrix; body-disabled exports withhold custom-body meshes and derivatives. Test every toggle and nested artifact dependency. Normative contract updated. |
| High: generated template counts cannot independently establish the physical inventory | E1 requires independently authored expected instances, layers, handedness, seams, closures, free edges and assembly schedule. Underdefined interfacing must resolve or remain an explicit approximation/gap. |
| Template-level pairwise seams omit layer, orientation and turning semantics | Add explicit physical-instance and operation schemas, multi-way memberships, gather correspondences and supported construction recipes. No direct assumption of solver-ready current output. |
| Project-wide job fencing and revision-only geometry heads cannot safely host independent job kinds | Require job-kind/input-scoped derived heads, committed parent dependencies and measured budgets; retain existing 2D semantics. |
| Mean seam residual or stationary cloth can disguise broken assembly | Require maximum and percentile residuals, source fidelity, separate body/self-contact checks, local contact exemptions, resolution/timestep checks and rigid-transform invariance. |
| Trusted worker code is not proof of host isolation | Require actual-host CPU/memory, process-tree, filesystem/network and credential checks, with unenforced controls disclosed. |
| Dedicated plan could conflict with the earlier sole-plan wording | Parent roadmap now indexes this workstream; detailed V0–V5 content is replaced rather than maintained twice. |

The independent recheck confirmed closure of both high-priority findings and the other substantive findings, with no remaining blocking planning finding in the reviewed scope. A final milestone-label correction changed residual V0 references to E0. Documentation/link and whitespace checks pass after correction.

Implementation must still execute the planned fixture matrix, record simulator selection and numerical profiles, and obtain an independent artifact/lifecycle review before release. No simulated artifact, hardware benchmark or physical garment test was produced by this planning task.
