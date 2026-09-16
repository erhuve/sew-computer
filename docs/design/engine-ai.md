# Engine and AI adapter design

**2026-09-14 · Proposed · Implementation and validation pending**

Implementation update, 2026-09-15: the general-model adapter now returns schema-constrained, owner-reviewed proposals using a tool-free Codex Responses connection or a configured compatible API. This replaces the proposed GPT-4o-only provider choice below. Provider, privacy, budgets and transport limitations are specified in [deployment](../deployment.md#design-model-connection); verification is recorded in the [AI review](../reviews/ai-prototype.md). Full MMUA/projector reproduction and physical validation remain unimplemented.

This specialty design follows the normative [project contract](project-contract.md) and [product intent](../product/vision.md). It defines the geometry boundary, interpretation options and release evidence. It does not establish manufacturing readiness or depend on marketplace, factory-bidding or crowdfunding features.

## 1. Evidence boundary

Planning update, 2026-09-16: future garment 3D follows [project contract §14](project-contract.md#14-pattern-derived-garment-3d) and the [active workstream](../plans/software-prototype.md#pattern-derived-3d-workstream). Actual pattern pieces supply rest geometry; typed assembly and material inputs drive trusted engines. The model may propose those inputs, but cannot replace pattern-derived meshes. Simulator choice and compatibility with the custom component compiler remain unverified.

The [upstream investigation](../research/design2garmentcode-evidence.md) records CPU execution at commit `7065b3ef01ff61f4462e871d4cb439a0b97c48db`. Observed interfaces include `BodyParameters`, `MetaGarment(name, body, design).assembly()`, `piece.is_self_intersecting()` and `pattern.serialize(...)`. The smoke test enabled printable serialization and disabled 3D serialization.

Fixed shirt, skirt and trouser configurations produced respectively 4, 4 and 6 panels; all fifteen reported artifacts were nonempty. These observations establish neither a portable installation nor correct units, seams, annotations, print scale or fit. Text/image inference and simulation were not exercised. Upstream is not installed in the new application.

Everything below is proposed application behavior, not an observed upstream API.

## 2. Adapter and declarative intermediate representation

Keep upstream behind a separately versioned CPU adapter. Pin its commit, dependency lock, trusted component registry and serialization settings. Unit-convention verification and a clean-install reproduction are prerequisites to enabling geometry execution.

The application owns a versioned declarative garment IR, not upstream YAML. Proposed sections are:

- **Components:** stable instance IDs, registry component identifiers, typed parameters and declared assembly relationships.
- **Inputs:** typed references to body measurements, material properties and construction decisions.
- **Intent mapping:** requirement IDs, represented fields and unsupported or unresolved details.
- **Derivations:** definitions and dependencies for computed quantities, distinct from entered measurements.

The IR allows original component compositions and preserves intent beyond current engine capabilities. It contains no executable expressions, import paths or arbitrary Python. Unsupported details remain editable requirements; they are not discarded when lowering supported portions into upstream parameters.

Proposed internal operations are `describeCapabilities`, `validateInputs`, `compileGeometry` and `inspectGeometry`. These are adapter interfaces, not upstream functions or public HTTP endpoints. Each consumes versioned data and returns structured results. Compilation returns a normalized panel/edge/stitch graph plus staged artifacts; it cannot publish revisions or artifacts itself.

Schema migrations are explicit, tested transformations. Opening an old revision never rewrites it. A migration produces a new revision, retaining original values and identifying changed interpretations.

## 3. Inputs, units and capability reports

Apply the contract’s millimeter convention at explicit adapter boundaries. Verify upstream conventions independently for body inputs, parameterized lengths, coordinates and serialization. Angles, ratios, percentages and counts require separate conversion rules. Never scale every numeric field together.

Validate finite numbers, enum membership, bounded collection sizes, required dependencies, dimensional consistency and component-specific ranges. Distinguish body circumference from finished-garment circumference and panel-edge length. Unknown units block dependent geometry; unknown measurements block only operations needing them. Neither prevents saving the brief or exporting an accurately incomplete draft.

Do not silently substitute the research body fixture. A demonstration body, if introduced, must have cleared rights and an explicit assumed-input designation.

A capability report identifies:

- Adapter/registry versions and evaluated requirement IDs.
- Supported components, combinations, parameter domains and required inputs.
- Unsupported requirements, unresolved interpretations and exact blocking dependencies.
- Proposed substitutions, separately awaiting acceptance.
- Available output classes and failed or unperformed checks.

An input inside a numeric range is not automatically supported: combinations can violate topology or construction constraints.

Extension work starts from retained requirements, not a demand to choose a nearer template. An engineer adds trusted component logic, schemas, assembly rules, rights records and fixtures. Registry publication requires review and regression tests. AI cannot install extensions. Broader design freedom remains the objective; current capability limits remain visible.

## 4. Interpretation: three distinct modes

### Manual first deliverable

The first implementation increment is explicitly manual: enter text or attach a reference, record requirements, choose representable components and confirm typed parameters. The reference remains visible during editing. No automated interpretation claim accompanies this route.

This delivers a connected creation/export loop without requiring a manufacturing purchase or paid review.

### Optional general-model parameter adapter

A separate, untested route may send authorized text and sanitized images to a general multimodal model, then request schema-constrained `EditProposal` data. It maps into the current IR and registry; it does not generate executable garment programs.

The proposed initial provider route is OpenAI’s multimodal API using a deployment-verified, pinned GPT-4o-family model identifier. Availability, image support and schema behavior must be tested before enablement. Record the exact identifier, prompt/schema versions and provider provenance. Until configured and verified, the interface stays manual rather than displaying simulated AI success.

### Released research pipeline

The supplied README describes MMUA using a multimodal-model API, followed by a parameter projector requiring Qwen2-VL-2B-Instruct base weights and the authors’ fine-tuned weights. Its installation also documents Torch/CUDA dependencies; optional Warp simulation is separate.

A general-model JSON adapter is not that pipeline. Reproducing the released path requires inspecting and exercising its actual orchestration, projector loading and outputs, with independent license and runtime gates. All remain untested here. Any generated program remains inspection-only unless converted into validated registry-backed data; this prototype never executes model-authored code.

## 5. Proposal acceptance and hostile inputs

Proposals contain bounded, allowlisted operations against a specific revision. Validate operation count, field types, paths, referenced IDs and affected requirements. Reject unknown operations and attempts to change ownership, evidence, validation results or executable identifiers.

Show assumptions and semantic differences before acceptance. An unsupported asymmetric closure cannot silently become a symmetric opening. Acceptance checks the server-captured base revision, baseDraftVersion, sourceDraftDigest and deletion generation transactionally. A matching revision alone is insufficient. Apply all accepted operations, publish the revision and rebase/increment the editable draft atomically; a conflict preserves the draft and proposal. The normative transaction is [project contract §3](project-contract.md#3-revision-and-export-transactions).

Reference text, embedded instructions and OCR content are design evidence, not authority. The model receives no tools, credentials or application-database access. Schema-valid output still needs capability and geometry checks; it is not proof of faithful interpretation.

## 6. Deterministic checks and their limits

Checks operate on normalized geometry with versioned tolerances and numerical methods:

- **Geometry:** finite coordinates, valid indices, closed boundaries, nondegenerate panels, curve validity and intersection checks with documented coverage.
- **Assembly:** valid stitch references, orientation, duplicate assignments and paired seam lengths. Compare stitching lines where defined, not automatically allowance boundaries.
- **Ease:** report differences only for defined measurement paths and construction scope. Gathering and intentional seam ease need explicit rules; equal-length enforcement would incorrectly reject some designs.
- **Annotations:** verify applicable allowances, grainlines, notches, cut counts and fold references. Missing information remains missing.
- **Print:** verify physical dimensions, page boxes, clipping, tile overlap, alignment marks and calibration-square geometry.

Do not invent universal ease targets or seam tolerances. Establish them through documented fixtures and scoped construction rules. Reports include measured values, thresholds, algorithm versions and unperformed checks.

Determinism means repeatable results within a pinned environment and declared tolerance, not identical bytes across dependency versions. Digital PDF checks do not establish printer calibration. Apply the contract’s output-class policy without claiming physical fit, material behavior or sewing feasibility.

Evidence authority, contradiction handling and explicit supersession follow [project contract §8](project-contract.md#8-evidence-authority-and-deterministic-eligibility). The service records the issuer of deterministic checks; model/worker payloads cannot claim arbitrary authority. Reusable calibrated pattern PDF bytes are never restamped for a later export bundle; a changed derivative has a new digest and new eligibility assessment.

## 7. Isolation, rights, privacy and budgets

Run trusted geometry under the contract’s restricted worker boundary. Verify denial of application secrets, database access, network access and writes outside the attempt directory. Test process-tree termination and resource enforcement. Process supervision alone is not isolation; unsupported mechanisms remain release blockers.

Maintain separate gates for code/dependency licenses, body assets, model weights, redistribution rights and runtime compatibility. Root MIT licensing is not blanket clearance.

Before provider submission, disclose destination, selected inputs and cost ceiling. Exclude measurements unless needed and authorized. Keep prompts and references out of routine logs and public manifests.

Every call requires enforced token, image/pixel, timeout, attempt and cost limits; initial policy permits one request and at most one bounded repair. Geometry jobs receive measured CPU, memory, wall-time and output-size limits. Exhaustion returns preserved work and an explicit failure, not an unlimited retry or a purchase requirement.

## 8. Acceptance fixtures and handoff

Release fixtures must include:

The 2026-09-16 component implementation adds a bounded relaxed-shirt compiler alongside the pinned upstream base adapter. It consumes structured construction choices, not executable model output. Ordinary ambiguity is resolved into reviewable choices by the interpreter. Actual piece/edge inventories, selected dimensions, gathered ratios, cut contours and annotation presence are checked before claiming digital coverage. Unsupported topology remains explicit. See [garment pipeline review](../reviews/garment-pipeline.md) for exact supported construction and physical limits.

1. Portable, rights-cleared equivalents of the three smoke configurations.
2. Unit conversions, NaN/Infinity, missing inputs and body/POM confusion.
3. Broken topology, mismatched seams, intentional gathering and missing annotations.
4. Unsupported design details preserved through proposal, revision, viewer and export.
5. Hostile reference instructions, forbidden operations and stale proposals.
6. Resource exhaustion, isolation probes, malformed artifacts and cancellation races.
7. Digital print checks plus separately scoped physical-print evidence.
8. A clean-install run producing checked manifests and repeatable geometry.

The editor receives capabilities, requirement coverage and field-level findings. The coordinator owns authorization, revisions, leases and publication. Export consumes immutable artifacts and reports; it never upgrades missing evidence. Independent review should attempt silent substitutions, forged validation and executable payloads before automation is enabled. Delivery follows the [prototype plan](../plans/software-prototype.md). (•̀ᴗ•́)و
