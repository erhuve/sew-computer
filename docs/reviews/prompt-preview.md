# Prompt-first garment preview

2026-09-26 · Local MVP interface checkpoint; no public deployment or new physics claim.

The [dress silhouette follow-up](dress-silhouettes.md) adds independently drafted flared, gathered and tiered dresses to this same two-action flow; existing shirt-dress revisions remain unchanged.

The main path accepts one description and two actions: **Design with AI**, then **See garment** after reviewing the proposal. The model request button states the destination and allowance usage. No separate consent checkbox, name, measurement entry, material table or sewing form is required. Reference images remain separately opt-in.

The proposal shows its summary, construction illustration and unsupported omissions before the preview action. Dimensions, open questions, materials and construction notes are expandable. Preview sizing fills only missing body inputs; known measurements, owner-adjusted estimates and explicit not-applicable values survive. Invalid preserved inputs remain errors with access to the editor, rather than silently changing the inputs to force success.

The garment occupies the main workspace. **Edit details** opens the full editor for colors, prints, construction choices, custom sewing pieces, body measurements and notes. Its open/closed preference persists in the browser tab. Optional starting shapes are collapsed on the home screen when a model is connected. Manual project creation remains available without model use.

For unsupported silhouettes, the owner can explicitly select a simpler relaxed shirt, relaxed dress or elastic-waist skirt preview. Acceptance saves the original proposal, then the chosen construction is published with sample sizing and the original brief, colors, technical notes and unsupported requirements preserved. Previously supported non-material requirements become unresolved for the changed construction. A visible simplified-preview label survives reopening. This is not automatic drafting of the original unsupported silhouette.

The flow uses saved revisions, the existing trusted pattern compiler and automatic source-derived garment preview jobs. It does not select a prebuilt demo mesh. Previews remain approximate display poses with unverified physical fit and drape; matching downloads retain their existing disclosure rules.

Verification covers prompt-only desktop/mobile creation, the preview action within the mobile viewport, real geometry and matching revision identity, reopening, optional editing, explicit simplification with original-intent preservation, sizing defaults and existing authoring/export flows. Browser model responses are deterministic synthetic fixtures; patterns and garment previews use the real configured CPU engine. At that checkpoint the existing live model connection was unchanged. The subsequent recovery changes below retain its model and credential source.

Checks passed: typecheck, build, documentation validation, 118 application tests and four applicable multi-garment engine tests. The complete 35-scenario browser run found one mobile toolbar spacing regression; it was corrected and all 21 affected browser scenarios passed on the final build. The other 14 scenarios passed in the full run. The local studio was checked to serve that final build.

Visual evidence: [home](../verification/prompt-preview/home-desktop.png), [proposal on desktop](../verification/prompt-preview/proposal-desktop.png), [proposal on mobile](../verification/prompt-preview/proposal-mobile.png), [garment on desktop](../verification/prompt-preview/garment-desktop.png), [garment on mobile](../verification/prompt-preview/garment-mobile.png), [explicitly simplified preview](../verification/prompt-preview/simplified-preview-desktop.png).


## Interpretation timeout recovery (2026-09-26)

Two local design requests failed at the original 120-second deadline. An isolated synthetic call completed in 53 seconds, confirming that the connection was working but not establishing why those specific requests took longer. The Codex deadline is now 240 seconds; the durable job lease derives from the same provider budget plus five seconds. The dedicated API retains its 120-second limit. This remains a bounded wait, not a speed guarantee or automatic retry after a provider failure.

The prompt requests a concise first proposal while retaining every requested feature and unsupported limitation. Actual stream events update a persistent phase (connecting, thinking, writing, validating); the interface shows that phase and elapsed time. No partial output can be accepted. Explicit incomplete responses fail immediately. Failed attempts show a prominent message, the preserved prompt and **Try again**. Cancellation, draft/revision checks, reference opt-in and request limits remain in force.

Regression checks cover provider-derived deadlines and leases, progress persistence, late progress/output rejection, saved draft preservation, retry-to-success, terminal stream errors and browser reload/cancellation. Browser recovery fixtures are synthetic; live model verification is tracked separately.

The live verification used a new synthetic dress brief with dark fabric and lace accents, the existing GPT-6 Astra connection, an isolated database, and the normal authenticated API job/acceptance/revision flow. The validated proposal completed in 44.3 seconds (1,408 output tokens), retained the lace limitation, and produced matching real CPU patterns and a source-derived 3D preview by 63.9 seconds. This is one successful observation, not a latency guarantee or validation of the owner's failed private prompt; that prompt was not replayed.

Recovery checks passed: typecheck, production build, documentation/whitespace checks, all 121 application tests, four applicable dress/skirt engine tests, and all nine affected browser scenarios. The two recovery scenarios were rerun after the final accessibility/copy edits. The task-owned local studio was restarted while idle and its entry page matched the final build; production hosting was not changed.

Recovery visuals: [retry on desktop](../verification/interpretation-recovery/retry-desktop.png), [retry on mobile](../verification/interpretation-recovery/retry-mobile.png), [stream progress](../verification/interpretation-recovery/progress-desktop.png).
