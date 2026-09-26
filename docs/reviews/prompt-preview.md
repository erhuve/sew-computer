# Prompt-first garment preview

2026-09-26 · Local MVP interface checkpoint; no public deployment or new physics claim.

The main path accepts one description and two actions: **Design with AI**, then **See garment** after reviewing the proposal. The model request button states the destination and allowance usage. No separate consent checkbox, name, measurement entry, material table or sewing form is required. Reference images remain separately opt-in.

The proposal shows its summary, construction illustration and unsupported omissions before the preview action. Dimensions, open questions, materials and construction notes are expandable. Preview sizing fills only missing body inputs; known measurements, owner-adjusted estimates and explicit not-applicable values survive. Invalid preserved inputs remain errors with access to the editor, rather than silently changing the inputs to force success.

The garment occupies the main workspace. **Edit details** opens the full editor for colors, prints, construction choices, custom sewing pieces, body measurements and notes. Its open/closed preference persists in the browser tab. Optional starting shapes are collapsed on the home screen when a model is connected. Manual project creation remains available without model use.

For unsupported silhouettes, the owner can explicitly select a simpler relaxed shirt, relaxed dress or elastic-waist skirt preview. Acceptance saves the original proposal, then the chosen construction is published with sample sizing and the original brief, colors, technical notes and unsupported requirements preserved. Previously supported non-material requirements become unresolved for the changed construction. A visible simplified-preview label survives reopening. This is not automatic drafting of the original unsupported silhouette.

The flow uses saved revisions, the existing trusted pattern compiler and automatic source-derived garment preview jobs. It does not select a prebuilt demo mesh. Previews remain approximate display poses with unverified physical fit and drape; matching downloads retain their existing disclosure rules.

Verification covers prompt-only desktop/mobile creation, the preview action within the mobile viewport, real geometry and matching revision identity, reopening, optional editing, explicit simplification with original-intent preservation, sizing defaults and existing authoring/export flows. Browser model responses are deterministic synthetic fixtures; patterns and garment previews use the real configured CPU engine. The existing live model connection is unchanged.

Checks passed: typecheck, build, documentation validation, 118 application tests and four applicable multi-garment engine tests. The complete 35-scenario browser run found one mobile toolbar spacing regression; it was corrected and all 21 affected browser scenarios passed on the final build. The other 14 scenarios passed in the full run. The local studio was checked to serve that final build.

Visual evidence: [home](../verification/prompt-preview/home-desktop.png), [proposal on desktop](../verification/prompt-preview/proposal-desktop.png), [proposal on mobile](../verification/prompt-preview/proposal-mobile.png), [garment on desktop](../verification/prompt-preview/garment-desktop.png), [garment on mobile](../verification/prompt-preview/garment-mobile.png), [explicitly simplified preview](../verification/prompt-preview/simplified-preview-desktop.png).
