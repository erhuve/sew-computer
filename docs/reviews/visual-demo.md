# Visual MVP demonstration

**2026-09-25 · Local demo checkpoint · Not deployed**

The immediate deliverable is enough visible, working software to show people and learn whether they find value. The `/demo` page presents four synthetic relaxed-shirt examples with a front/back construction drawing, changeable sleeves and front detail, display colors, optional 3D piece inspection, matching SVG/JSON pattern downloads and downloadable feedback notes. Complete cloth physics is not a prerequisite for this demonstration.

## Start and show it

From an isolated checkout with dependencies installed:

```sh
bun run demo
```

Open `http://127.0.0.1:5175/demo`. Set `SEW_DEMO_PORT` if that port is occupied. The launcher binds only to this computer, creates a fresh scratch database under ignored `.planning/`, and disables model connections. The demo page needs no login. The optional local studio uses the deliberately non-secret key `sew-local-demo` and its own empty database; full studio generation still requires the separately provisioned engine. Stop the server with Ctrl-C. Production data and service settings are untouched.

A short owner-led walkthrough:

1. Show the initial shirt from front and back. Explain the intended idea-to-design-to-pattern workflow.
2. Ask the participant to switch sleeves or remove the frill. The drawing and underlying pattern change together. Color is display only.
3. Open **3D pieces**, select a physical piece and show its original 2D outline. **Flat pieces** displays the inventory; the garment layout is a rough rigid arrangement.
4. Download the selected draft pattern or JSON. These are the actual source example, not a screenshot of the drawing.
5. Record what they would make, which output matters, where they needed help and what would stop them using it again. Download the notes before leaving the page; they are not sent to a service.

The next product priority is actual participant feedback and fixing observed demo obstacles. No user-value finding follows from the automated checks below.

## Scope and source identity

The four examples combine long sleeves/button cuffs or short sleeves/no cuffs with a clean front or front-opening frill. They contain respectively 22, 24, 14 and 16 fabric instances. All use synthetic dimensions; no wearer data, photos or model calls are included. This limited demonstration is not a permanent preset-only product direction.

`scripts/build-visual-demo.py` uses the existing component compiler and inspection mesher. The catalog records generation source hashes and every JSON/GLB/SVG byte count and digest. The browser checks the pattern and GLB digests and their shared pattern identity. SVG outlines match the original cutting paths. Pre-generated assets make the demonstration independent of installing the Python engine on the presentation computer.

The front/back view is explicitly a **construction drawing**, using the existing garment-flat representation and source body outlines. It is not a rendered simulation. The 3D view uses original pattern meshes and proper rigid transforms; it does not change rest vertices or shrink pieces to close seams. Drape, gathering, stitched assembly, allowance bulk and interfacing are unfinished or omitted. The long frill remains visibly ungathered in 3D. Pattern downloads are draft references, not fit-validated or print-calibrated cutting outputs. Live AI interpretation and arbitrary design generation are outside this demo route.

## Verification

- TypeScript typecheck and production build pass.
- Application suite: **104 pass, 0 fail**, 1,012 assertions, 44.92 seconds.
- Rigid-layout helper: **5 pass**, 538 assertions, covering source inventory, handedness, proper rotations, preserved dimensions and rejected unsupported inputs.
- Demo browser suite: **2 pass** in 5.3 seconds on the final run with an isolated database and Chrome profile. Covers front/back, source selection, arrangement switching, variant changes, matching SVG/feedback downloads and mobile overflow.
- Four asset sets generated successfully in the pinned local engine container. Independent source review checked catalog hashes, GLB/pattern identity and every SVG cutting outline. The async color-loading race found in review was corrected.
- Desktop and mobile screenshots were visually inspected. [Desktop preview](../verification/visual-demo/desktop.png), [mobile preview](../verification/visual-demo/mobile.png), [rough 3D piece view](../verification/visual-demo/pieces.png).
- Final independent screenshot/code review found no demo blocker. A shared-viewer hint was adapted to refer to the demo's source outline below instead of the studio's separate Pattern view.

The first asset-generation attempt lacked required provenance and was corrected before retaining the successful assets. Sandboxed Chrome/server launches failed; the browser checks and local launcher then succeeded with the required host permissions. Earlier interrupted application runs are not counted as passes.

The complete engine and generation-dependent browser suites were not rerun: the upstream engine installation is not provisioned in this checkout. No engine or physics implementation changed in this demo increment. These checks verify this synthetic demo and application regressions, not a general engine release, live-model path, full-shirt assembly, physical fit or production deployment.
