# Visual MVP demonstration

**2026-09-26 · Local visual revision · Not deployed**

The normal studio now has a separate [end-to-end implementation](studio-e2e.md) that generates from saved project inputs. This page documents the earlier pre-generated examples and their historical verification.

The owner rejected the first rigid-panel arrangement as visually insufficient: an MVP still needs a realistic-looking garment, even when physics is approximate. The default **Garment preview** now shows a posed shirt with curved sleeves, collar, cuffs, gathered front detail, fabric shading and a back tail. **Design preview** retains the construction drawing; **3D pieces** retains the original flat/rigid inspection. Automated passes do not substitute for the owner's visual assessment.

## Start and show it

From an isolated checkout with dependencies installed:

```sh
bun run demo
```

Open `http://127.0.0.1:5175/demo`. Set `SEW_DEMO_PORT` if that port is occupied. The launcher binds only to this computer, creates a fresh scratch database under ignored `.planning/`, and disables model connections. The demo page needs no login. The optional local studio uses the deliberately non-secret key `sew-local-demo` and its own empty database; full studio generation still requires the separately provisioned engine. Stop the server with Ctrl-C. Production data and service settings are untouched.

A short walkthrough:

1. Rotate the garment and switch front/back. Explain the intended idea-to-design-to-pattern workflow.
2. Change sleeve length, remove the frill or try a color. Construction changes select their matching source patterns; color is display only.
3. Select fabric in the garment view to inspect its source outline. Open **3D pieces** and **Flat pieces** for the original inventory, or **Design preview** for the construction drawing.
4. Download the selected draft pattern or JSON. These are the actual source example, not a screenshot of the rendered garment.
5. Record what the participant would make, which output matters, where they needed help and what would stop them using it again. Download notes before leaving the page; they are not sent to a service.

No external participant feedback or demand finding is established by this checkpoint.

## Geometry, solver and provenance

Four pre-generated examples combine long sleeves/button cuffs or short sleeves/no cuffs with a clean front or front-opening frill. They contain 22, 24, 14 and 16 fabric instances respectively. All measurements are synthetic. No wearer data, reference photos, image-generated meshes or model calls are involved. This bounded demonstration does not restrict the eventual product to presets.

`scripts/build-visual-demo.py` retains the original component-pattern JSON, inspection GLB and SVG cutting outlines. Their bytes and identities are unchanged by the garment-shape work.

`scripts/build-garment-preview.py` meshes the same source seam-line domains with source interpolation weights, then solves elastic edge, low-weight bending and source-edge sewing constraints against explicit synthetic posing guides. It starts with translated flat source coordinates. All displayed cloth positions are outputs of the bounded projective solver; the original rest coordinates, source triangle topology and source mappings remain separately stored. Uniform arc registrations gather unequal intervals without rewriting the rest strip length. The material weights and posing forces are uncalibrated visual assumptions. This is not gravity-driven physical drape.

All shell/facing instances are retained. Buttons are separately rendered hardware; source edge lines and procedural weave shading are display details. The renderer does not change cloth positions. Pattern, construction, registered assembly, material and generator identities accompany each saved shape; generator and engine source hashes are retained. A separate shape catalog binds the file bytes, and the browser checks the file digest and pattern identity before rendering.

The four saved solves use 8,984–13,354 vertices and 110 fixed iterations each. Generation took approximately 7.5–9.3 seconds per variant in the pinned offline engine container. The saved output is explicitly `acceptedSimulation: false` and `guided-shape-approximation`.

## Approximation limits

The **About this approximation** disclosure reports edge deformation and sampled seam gaps. Final 95th-percentile absolute edge deformation is **15.0–15.4%**, with worst local values around **508%**. Maximum sampled seam gaps are **0.52–0.57 mm**. Small seam gaps do not establish physically credible cloth: the substantial deformation, guide forces and unvalidated layer behavior prevent a drape or fit claim.

Body contact, self-contact, physical turning/binding, material calibration and numerical convergence are unvalidated. Allowances and interfacing are omitted. The solver is a separate display approximation and neither modifies nor completes the stricter research solver's acceptance gates. Pattern downloads remain draft references, not fit-validated or print-calibrated cutting outputs. The demo does not run live AI or generate arbitrary designs.

## Verification

- TypeScript typecheck and production build pass.
- Application suite: **104 pass, 0 fail**, 1,011 assertions, 45.47 seconds.
- Demo browser suite: **3 pass**, covering the default garment view, all four variants, front/back controls, construction drawing, source-piece selection, matching downloads, feedback notes, color controls and mobile overflow.
- `python3 scripts/check-garment-previews.py` independently audits all four saved outputs using the standard library: exact source hashes and inventory, reconstruction from source weights, rest triangle area against original polygon area, independently recomputed edge deformation and sampled seam residuals, and catalog digests.
- Original rigid-layout helper verification remains **5 passing tests / 538 assertions** from the earlier checkpoint; it was not rerun or counted as new cloth-solver evidence.
- Front, back and mobile screenshots were inspected. [Garment view](../verification/visual-demo/garment.png), [desktop](../verification/visual-demo/desktop.png), [mobile](../verification/visual-demo/mobile.png), [back](../verification/visual-demo/back.png).

The full engine and generation-dependent browser suites were not rerun because the upstream engine installation is not provisioned in this checkout. The new display generator ran in the existing pinned offline container; the production engine and application job pipeline are unchanged. Checks verify source correspondence, recorded diagnostics and demo behavior. They do not validate physical simulation, user value, a live-model path or a production release.
