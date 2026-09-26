# Custom sewing patterns and fabric appearance

2026-09-26. Isolated MVP development; not deployed. This extends the [multi-family checkpoint](multiple-garments.md) in response to the owner's request for both fabric prints and original sewing patterns.

## Connected authoring

The saved garment now carries an optional appearance: a six-digit screen color and an optional stripe, check, dot or uploaded-image print. Colors, print color, repeat size and rotation update the visible garment without regenerating its source mesh. Texture coordinates come from the source pattern in metres; repeat dimensions are entered in millimetres. Uploaded artwork retains its aspect ratio, is decoded and sanitized through the existing private reference pipeline, and cannot reference an external URL. Prints restart on each piece's source coordinate system; motif alignment across seams is not automatic. Display color is not a calibrated dye or printing specification.

The **Custom pattern** starting workspace contains editable example outlines, not a fixed garment compiler recipe. Owners can move points with pointer, keyboard or exact coordinates, insert/remove points, add/duplicate/remove pieces, name pieces and connect arbitrary edges. Plain seams require equal lengths; explicit gathering accepts edge A at one to three times edge B. Changing point topology clears that piece's connections because their edge indices would otherwise become ambiguous. Each named piece is one shell instance; duplicate pieces for additional copies.

Custom construction uses the original `sew-custom-pattern/1` trusted compiler. It accepts up to 16 simple polygon outlines with 3–30 points each, coordinates within ±2,000 mm, 100 seams and 6–20 mm allowance. Curves can be approximated with polygon points. The compiler rejects crossing/degenerate outlines and incompatible seams, constructs an offset cut contour, grainline and registration marks, and preserves the entered seam-line coordinates. Holes, internal cuts, darts, automatic grading and unrestricted automatic garment drafting remain unsupported.

Each piece has explicit translation, rotation and bend posing controls. The bend guide follows the horizontal extent of the drawn outline, so a flared polygon keeps its flare in the preview. The existing approximate elastic/seam solve uses those guides while retaining immutable source rest coordinates and source weights. Disconnected pieces are permitted and remain visible as separate pieces; this is not a complete-assembly claim. Pose controls do not resize the flat source. Contact, gravity-driven settling, calibrated fabric response and fit remain unverified. No AI mesh replaces the source pattern.

**Update garment** publishes the source changes and regenerates its matching pattern and private 3D approximation. Saving/reopening and PDF/JSON export use the same revisioned pipeline as the component garments. A changed color or print is visible immediately; downloads still describe the selected saved revision until the owner updates the garment.

## Privacy and AI boundary

Artwork asset IDs must belong to the same project and remain among that design's references while used. Removing that reference clears the print. Metadata sanitization, authentication, no-store responses and immutable revisions apply. Reference inclusion is required to share the artwork; otherwise its image bytes and asset handle are omitted. Custom outline/placement data are geometry and are withheld when pattern inclusion is off. Export validation independently enforces the same projection; redacted import reconciliation retains the owner's undisclosed data.

The tool-free model can propose screen colors and supported procedural prints. It must retain unsupported artwork requests, never invent private asset IDs, and cannot claim automatic arbitrary-garment drafting. Existing custom outlines are owner-authored data, retained unless explicitly changed. Existing body fields are not sent to the model; source geometry can itself reveal dimensions and follows the existing explicit design-inference consent.

## Verification

The live `gpt-6-astra` path returned the requested cream/navy colors, a 60 mm stripe repeat and 45° rotation, and the accepted design generated its actual dress pattern and printed preview. A separate custom four-panel skirt with checks was saved in the local studio. [Live striped dress](../verification/fabric-appearance/live-striped-dress.png) and [live custom skirt](../verification/fabric-appearance/live-custom-skirt.png) were inspected visually. Final checks:

- `bun run typecheck` and `bun run build`: passed.
- `bun run test`: **115 passed**, including cross-project artwork rejection, provider schema compatibility, export disclosure and redacted-import preservation. Short test time limits were retained; the final application run was sequential after the CPU-heavy checks.
- Applicable original-compiler, custom-pattern, inspection, preview and printing tests: **13 passed**, 5,594 assertions in the pinned Docker runtime. Cases cover signed coordinates, non-rectangular added pieces, changed dimensions, disconnected pieces without seams, unequal seams, invalid outlines, source corruption and matching PDFs.
- `bun run test:browser`: **33 passed**. New workflows exercise exact/keyboard outline edits, inserted points, new pieces, removing/reconnecting seams, procedural print controls, arbitrary uploaded artwork, texture changes, persistence, matching exports and narrow-screen layouts. The final import-only backend correction is covered by the application suite.
- `python3 scripts/check_docs.py` and `git diff --check`: passed.

The full historical engine command was also attempted. Its custom-coordinate failure was corrected and passed the focused end-to-end regression; nine native research tests and nine external Design2GarmentCode tests remain unavailable in this checkout. The full historical suite is not reported as green. Evidence includes [custom desktop](../verification/fabric-appearance/custom-pattern-desktop.png), [custom mobile](../verification/fabric-appearance/custom-pattern-mobile.png), [color desktop](../verification/fabric-appearance/color-desktop.png) and [color mobile](../verification/fabric-appearance/color-mobile.png).

The pinned runtime image remains `sha256:0b847fbd80c0503cb9492c2f49ca2cee301ec0a28b33e939b5e87d6660c5536f`; no dependency versions changed. The new custom compiler is included in engine source fingerprints and preview source hashes. The full historical engine suite also needs the unavailable native research Python environment and external pinned Design2GarmentCode checkout; those gates are not represented as passing.

Next priorities: demo the connected custom and component flows with participants, improve outline drawing/curve tools from observed friction, and expand automatic construction only where it helps demonstrated demand. Precise physics is not a prerequisite for this feedback milestone.
