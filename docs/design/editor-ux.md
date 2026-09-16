# Editor UX: From Idea to Reviewable Revision

**Proposed · 2026-09-14**

This document expands [project-contract.md](./project-contract.md), which owns record identities, states, transactions, evidence and export policy. [Product intent](../product/vision.md) governs creative freedom and access.

Every interaction and acceptance test below is proposed and untested. The only demonstrated execution is the fixed-parameter CPU geometry smoke test described in [the evidence record](../research/design2garmentcode-evidence.md). It validates neither this editor nor interpretation, printable scale, sewing feasibility or physical fit.

## 1. First useful creative experience

The first slice must support creating an idea, correcting its interpretation, inspecting available geometry, saving revisions and exporting an honest review package. A pattern viewer alone cannot communicate whether the intended garment survived translation.

The entry screen offers one composer: **Describe your garment**, **Add references**, or **Add a sketch**. References and sketches initially use bounded raster uploads. Offer optional starting points without requiring a garment-family selection. No measurements, manufacturing purchase or professional review are required to save an idea.

After entry, show the submitted visual immediately with editable design callouts beside a concise interpretation. For text-only input, show the brief and requirement cards; do not invent a garment render to fill the canvas. Proposed sketch annotation supports placing a note on an image, with an equivalent text-list action.

Each requirement retains its original intent. Unsupported features remain visible in the concept workspace and review package. Generating a supported subset requires an explicit acknowledgement of what will be missing, not acceptance of a silently simplified design.

### Questions and assumptions

Ask at most three questions at once, prioritized by their effect on the next action. Each offers **Answer**, **Keep unknown**, or **Use this assumption** where applicable.

Distinguish “needed to generate geometry” from “can decide later.” Unknown units may block execution but never saving. Assumptions show source and scope; do not preselect acceptance. Never infer body measurements or gender from a reference.

Before provider transmission, identify the provider, data being sent and applicable budget. Declining preserves manual drafting and saved work.

## 2. Workspace and evidence views

The persistent header contains the project title, revision/draft indicator, save status and primary action. The workspace combines a visual area, requirement list and contextual inspector. Explanations and advanced controls open progressively.

Planned 3D extension (2026-09-16): add a pattern-derived garment view under [project contract §14](project-contract.md#14-pattern-derived-garment-3d). Distinguish initial/exploded placement, approximate assembly and simulated drape by evidence. Link physical instances to 2D templates, retain visible revision identity and use accessible inspection controls. Shape edits must change authoritative construction/pattern inputs; temporary arrangement changes do not. Implementation and release order remain in the [active plan](../plans/software-prototype.md#pattern-derived-3d-workstream); the existing views below describe the earlier slice.

Use three explicitly different views:

| View | Content and boundary |
|---|---|
| **Idea & references** | Submitted images, sketches, annotations and interpretation. Any generated concept image is labeled illustration, not geometry or fit evidence. |
| **Pattern** | Actual generated panels and available assembly information, labeled with revision and output class. Missing annotations remain visibly missing. |
| **Simulation** | Optional, unavailable in the first slice. Enable only for a real simulation artifact with its input scope and provenance; never substitute draped concept imagery. |

A nonexecuting “Simulation unavailable” explanation is sufficient initially. Unsupported capabilities must not look like broken loading states.

Selecting a requirement highlights associated geometry only when a verified mapping exists. Otherwise say “No linked geometry.” Selecting a panel opens its identifier, available measurements and annotations. Display body measurements, panel dimensions and finished-garment POMs under distinct labels.

Always identify historical output as **From revision N**. It may remain valid for that revision while differing from the current draft.

### Minimum manual tech-pack authoring

The no-model prototype includes compact editable material/trim BOM rows, finished-POM definitions/measurement methods/values with provenance, construction notes and operations, front/back/detail view assignment for uploaded or generated assets, and stable callout-to-piece/field linking. Keep these secondary to the canvas, but reachable from Tech Pack and linked issues. Uploading a back illustration does not make it geometry-derived; preserve its author/source label. Unknown targets and tolerances remain explicit unknowns, not filled to make the PDF look complete. The substantive populated fixture and exact no-model release path are in [the active plan](../plans/software-prototype.md#substantive-manual-authoring-acceptance-fixture).

Import edited handoff data as a previewed patch against the stored export projection. A missing/redacted body field is a no-op, not a deletion. Explicit clearing has a separate confirmation. Explain that patterns may still reveal body dimensions even when raw body measurements are omitted from the package.

## 3. Visual edits as reviewable proposals

Users can select an image annotation, requirement or supported pattern control and request a change in ordinary language. Direct controls expose supported parameters with definitions and units. Freeform mesh editing and arbitrary program execution are outside this slice.

Every geometry-affecting edit produces a bounded, typed proposal before commitment. Its preview shows:

- Base revision and affected requirements or fields.
- Before/after values, units and value states.
- Assumptions and unsupported or unresolved consequences.
- Which existing artifacts and review checks would no longer describe the changed design.

Highlight the affected visual region when mappings permit. A speculative overlay says **Proposed**, not “new pattern.” Geometry comparison appears only after actual generation succeeds.

**Accept** commits only against the proposal’s matching revision and draft version. **Reject** leaves the design unchanged. Editing the brief while a proposal is pending makes that proposal stale; offer reconciliation or regeneration, never automatic application.

**Undo** restores draft content or creates a new revision representing the reversal. Viewing history never overwrites it. Do not silently partially accept a proposal: if only some operations are possible, present a newly scoped proposal with the exclusions retained.

A proposal card carries the server-captured source draft version/digest as well as revision. New typing after proposal generation creates a conflict even if no revision was published; preserve both states. Accepting a valid proposal also atomically rebases the editable draft and increments its version, as required by the project contract.

## 4. Layout and accessibility acceptance

Visual content precedes explanations in reading order. Proposed baseline geometries:

- **Desktop, 1366 × 768:** header at most 56 CSS pixels and view controls at most 48. A flexible canvas and approximately 320-pixel inspector leave the main visual, evidence label and essential actions above the fold.
- **Mobile, 390 × 844 and 360 × 800:** one visual column, compact header and view selector, then an expandable requirement sheet. Keep a meaningful visual region, revision label and edit action visible without scrolling.
- **Narrow and zoomed layouts:** at 320 CSS pixels and 200%/400% zoom, content reflows without page-level horizontal scrolling. Above-fold density yields to readability. Pattern panning stays inside its explicitly labeled viewport.

Bottom actions respect safe areas and never cover content or the software keyboard. Controls target at least 44 × 44 CSS pixels; body copy is at least 16 pixels. Status never relies on color alone.

Keyboard users can enter a brief, upload, inspect requirements, select a panel through a list, preview, accept, undo and export without pointer gestures. Canvas navigation has named zoom, pan and reset controls. Proposal dialogs receive focus, contain it appropriately, close with Escape and return focus to their trigger. Validation errors link to fields; job updates use restrained live announcements without moving focus.

Acceptance requires keyboard-only and screen-reader runs, visible focus, contrast checks and touch testing—not screenshots alone.

Physical-evidence UI distinguishes owner-recorded measurement, reported external comment and service-issued software check. Contradictory same-scope results block current candidate status until an explicit, attributed disposition; no newest-result-wins shortcut. Frozen pattern sheets retain their original artifact identity in later bundles; the cover/manifest explains current scoped classification.

## 5. Drafts, history and failure states

Use explicit states rather than a generic spinner:

- **Draft saving / saved / save failed:** saved means server acknowledgement. On failure, preserve the buffer, warn before navigation and offer retry or recovery-copy download.
- **Concurrent conflict:** preserve both versions and their base; provide a readable comparison and deliberate reconciliation.
- **Loading:** retain project identity and distinguish absent artifacts from pending retrieval. Never show another revision’s output unlabeled.
- **Generation:** display persisted job status. Show percentages only when measured. Cancellation remains “Cancellation requested” until confirmed; a completion that committed first remains successful.
- **Failure:** distinguish invalid inputs, unsupported capability, provider failure and engine failure. Preserve the brief and last successful historical output. Retry cannot silently change inputs.
- **Stale response:** a late proposal cannot become current. Historical job output remains associated with its requested revision.
- **Session expiry or deletion:** stop protected actions; explain recovery or unavailability without exposing cached private visuals to another session.

History opens persistent, read-only revision views with the corresponding brief, inputs, artifacts and evidence. Returning to the draft is explicit. Browser reload must recover acknowledged drafts and revision selection, not imply successful saving of unacknowledged changes.

## 6. Export and maker feedback

**Export** opens a compact summary first: revision, included documents, pattern class, missing requirements and privacy omissions. Detailed measurements, construction notes and manifests expand beneath it.

An export requires an immutable revision; offer to save the draft as one, preserving it if a conflict occurs. Draft review exports remain available without patterns or professional approval.

Use the contract’s exact pattern classes: **screen preview**, **printable reference**, and **calibrated cutting candidate**. Explain failed eligibility checks; never offer a cosmetic production-ready toggle. Body details and private references require deliberate inclusion. Preview redactions before creating the fixed snapshot.

Maker feedback initially means importing or recording comments against a selected revision, artifact and named check. Identify the recorder separately from the reported reviewer. A comment can create a proposed requirement or edit, but cannot rewrite history or automatically approve later revisions.

## 7. Component handoff and release checks

The component implementation presents front/back construction schematics with editable choices and dimension sliders. Schematics disclose that they do not predict drape/fit. Original requirements remain visible alongside digital component coverage, and derived construction is tied to the displayed pattern revision. A busy autosave must finish before generation; the generation click must never silently disappear. See [component review](../reviews/garment-pipeline.md).

The editor supplies version-bound user decisions; the coordinator enforces authorization and conflicts. [Engine and AI design](./engine-ai.md) supplies capability reports, typed proposals, artifact provenance and trustworthy geometry mappings. Export processing owns deterministic eligibility and immutable snapshots. The editor displays these decisions rather than recreating them.

Before completing [the prototype plan](../plans/software-prototype.md), test stale acceptance, two-tab conflicts, unsupported-detail retention through export, cancellation races, historical viewing, missing artifacts, failed autosave, redaction and complete keyboard/mobile flows.

Marketplace discovery, bidding, payments and crowdfunding remain separate later products. None may gate this creation and DIY loop. All proposed UX acceptance checks remain unexecuted. (•̀ᴗ•́)و
