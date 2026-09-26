# Dress silhouette checkpoint

2026-09-26 · Local MVP increment; no production deployment or physical acceptance.

## Problem and behavior

The old `relaxed-dress` calls the shirt compiler with a longer torso. Its collar, front opening, frills and sleeves therefore still looked like a long shirt. The owner explicitly prioritized silhouette variety, keeping the prompt-first flow and approximate physics.

New dress starts use `panel-dress`: four bodice quarters plus four flared or gathered skirt panels, or eight skirt panels in two tiers. Optional short/long dropped-shoulder sleeves add two pieces. Round, V and square necklines alter source outlines. The waist seam can move up or down and its width is an explicit design ratio relative to the chest construction, not a body measurement. These choices are editable under **Edit details**; the model chooses them from the brief. Existing continuous shirt-dresses retain their original recipe and saved artifacts. **Try a separate bodice & skirt** is an explicit editor action; it preserves the brief, body and color and marks previous non-material supported requirements unresolved.

## Source and runtime

`services/engine/panel_dress.py` independently generates the actual seam-line polygons, allowances, grainlines, registrations, gather ratios and edge connections. It does not invoke the shirt pattern compiler. The independent TypeScript recipe checks selected pieces, coordinates, dimensions, seams and derived specifications. Python inspection reconstructs the expected source and rejects altered source, marks or attachments. Shared bounds reject invalid neckline/shoulder/waist/skirt combinations before generation. Existing document and revision identity rules remain in place.

Inspection and preview use these pieces and their exact source/rest coordinates. The upper-body posing primitive is reused; the skirt/tier guides use the captured dimensions and gather ratios. All guides are approximate. A gathered-to-tiered edit exposed numerical zero-area triangles in optional quality refinement. Dress previews now retry that specific validation failure at a second bounded resolution, then constrained refinement if needed; every candidate must pass the same unchanged source, boundary, area and topology checks. The chosen resolution/refinement and retry reason are recorded in the preview report. No source coordinate or validation threshold is relaxed. New source fingerprints include the dress module. The pinned container dependency image remains unchanged because no dependency was added; the engine source is mounted read-only from the matching checkout. Native deployment would need this module alongside the same locked runtime. No service deployment is part of this change.

## Limits

Center-back bodice edges remain open for closure development. Closure hardware, neckline facings/finishes, lining, lace construction, fitted darts and wearer passage are not drafted or verified. Pattern notes, editor copy and model guidance state these limits. Optional sleeves remain dropped-shoulder construction. Source topology and visual differences are implemented; digital coverage is not a sewing or fit certificate. Unsupported details remain explicit, and arbitrary automatic garment drafting is still unfinished.

## Verification

Synthetic fixtures cover flared sleeveless V-neck, raised-waist gathered square-neck with short sleeves, and dropped-waist two-tier round-neck with long sleeves. Checks include real CPU source patterns, preview provenance, no inherited shirt hardware, mutation rejection, sample sizing, prompt-first browser flow, edits, persistence and matching downloads. Browser model replies are deterministic; a separate fresh synthetic live-model check exercises interpretation. No saved owner design is replayed or automatically overwritten.

A fresh synthetic live-model brief requested a plum dress with a square neckline, raised waist, gathered skirt and short sleeves. The existing GPT-6 Astra connection chose `panel-dress` with the requested skirt and neckline in 50.5 seconds; source patterns and a matching 3D preview completed by 83.7 seconds. Requested lace remained unsupported. This is one successful observation, not a latency or interpretation-accuracy guarantee.

Rendered examples: [flared](../verification/silhouettes/flared-desktop.png), [gathered](../verification/silhouettes/gathered-desktop.png), [tiered](../verification/silhouettes/tiered-desktop.png), [mobile](../verification/silhouettes/gathered-mobile.png). These are real source-derived outputs of synthetic browser fixtures, not prebuilt display meshes.

Final checks pass: typecheck, production build, documentation and whitespace checks, all 123 application tests, 14 applicable engine scenarios, and 11 affected browser scenarios across the initial run and corrected reruns. The engine set includes four new dress cases and the six-size dress matrix. The browser set covers three rendered silhouettes, gathered-to-tiered editing, persisted choices, matching downloads, legacy dress conversion with original-revision preservation, existing shirt/skirt flows and prompt-only desktop/mobile creation. Initial failures exposed the triangulation issue above and test assertions/selectors tied to the former pattern structure; corrected cases were rerun successfully. The shared garment-preview regression was rerun after the meshing recovery change.

The task-owned local studio was checked idle and restarted on port 5175 against its existing data, the same model connection and the final build. Saved owner designs are preserved; new prompts and explicit construction changes use the new path. No public or production service was changed.
