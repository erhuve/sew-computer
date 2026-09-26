# Sew Computer

Consumer garment-design software: turn descriptions, references and sketches into editable garment designs, connected sewing patterns and a revisioned tech pack.

**Status: private AI-assisted design prototype.** Describe a garment, optionally include reference images, review an AI proposal, enter measurements, generate actual 2D shirt/skirt/trouser geometry, and export a revisioned PDF/JSON tech pack. AI also drafts materials, measurement definitions and construction notes. Unsupported details remain explicit. Patterns are printable references, not sewing-ready or fit-validated outputs. Public hosting is not enabled by this repository.

**Try it:** [Private Sew Computer studio](https://sew-computer-hatsunemiku.zo.computer). Requires owner Zo sign-in and the studio access key. See [deployment and first-use instructions](docs/deployment.md). This checkout now backs the live service; use an isolated checkout for future development and test builds.

## Start here

**Local visual demo:** run `bun run demo` from an isolated checkout, then open `http://127.0.0.1:5175/demo`. Four synthetic shirt examples connect a posed cloth preview, front/back drawings, source-piece inspection and matching pattern downloads. The garment shape uses approximate elastic constraints and synthetic posing guides; it does not establish physical drape/fit or generate arbitrary designs. See the [walkthrough and verified scope](docs/reviews/visual-demo.md).

The component pipeline adds an explicit relaxed shirt compiler with sleeves, cuffs, collars, plackets, curved back tails and gathered front frills, plus derived specifications and construction schematics. New patterns include A4 and Letter tiled PDFs with calibration guides. See [scope, review and remaining gates](docs/reviews/garment-pipeline.md). This work does not make arbitrary garments or physically validated cutting patterns available.

1. **[Software-prototype roadmap](docs/plans/software-prototype.md)** — overall scope and implementation gates; the **[dedicated 3D engine plan](docs/plans/pattern-derived-3d-engine.md)** owns pattern-derived assembly, simulation and visualization work.
2. **[Product intent](docs/product/vision.md)** — creative freedom, software-first sequencing, visual-first UI, independent DIY access and private inputs.
3. **[Normative project contract](docs/design/project-contract.md)** — revision/draft concurrency, units, jobs, artifacts, evidence, privacy and export/import identity.

### Supporting designs

- [Editor and interaction design](docs/design/editor-ux.md)
- [Engine and AI integration](docs/design/engine-ai.md)
- [Backend, security and operations](docs/design/backend-security.md)
- [Tech pack, pattern export and maker handoff](docs/design/tech-pack-handoff.md)
- [Design2GarmentCode evidence and limits](docs/research/design2garmentcode-evidence.md)
- [Adversarial planning review and resolutions](docs/reviews/planning-adversarial.md)

Four specialist design briefs informed the plan. Independent security/contract and product/manufacturing reviews identified contradictions that were resolved in the documents. Review closure applies to the specification—not runtime correctness or garment validation.

## Prototype

A private, single-owner editor with reviewable AI proposals, real 2D pattern geometry, manual garment/tech-pack authoring, durable revisions, PDF/manifest export and feedback incorporation. Manual creation still works with every model route disabled. The integrated general-model proposal adapter is separate from the full released Design2GarmentCode AI path and optional simulation. See the [AI integration review](docs/reviews/ai-prototype.md).

The first geometry adapter has limited, explicit capabilities. Unsupported original intent remains editable. A concept image, pattern geometry, print calibration and physical fit are different evidence. See the [implementation review](docs/reviews/manual-prototype.md) for verification and remaining release gates.

## Setup and checks

Requires Bun, Python 3.11+, Poppler (`pdftotext` for tests), and a separately obtained Design2GarmentCode checkout at the commit recorded in `services/engine/source-lock.json`.

```sh
bun install --frozen-lockfile
python3 scripts/setup-engine.py --source /absolute/path/to/design2garmentcode --install
bun run typecheck
bun run test
bun run build
SEW_ENGINE_SOURCE=/absolute/path/to/design2garmentcode bun run test:engine
bunx playwright install chromium
SEW_ENGINE_SOURCE=/absolute/path/to/design2garmentcode bun run test:browser
```

The Zo development Site runs from `apps/web`. Its managed process requires `SEW_ALLOWED_ORIGINS` (exact comma-separated origins), optionally `SEW_DATA_DIR`, and engine source/Python overrides when not using the installed defaults. `SEW_ACCESS_KEY` can supply an owner credential; otherwise the API creates a private `access-key` file inside its data directory. Retrieve it locally; never commit it or place it in a URL. See [API operations and recovery limits](apps/api/README.md). Site configuration, credentials, private SQLite/artifacts, upstream source and Python environments are excluded from Git.

## Pattern-derived 3D development

For continuation from the September 23 research checkpoint, start with the [3D engine handoff](docs/research/3d-engine-handoff.md): current branch, verified evidence, remaining blockers, reproduction commands and local-only artifact dependencies. The dedicated implementation plan remains authoritative; the handoff is a checkpoint, not a new plan or release approval.

The isolated implementation includes private durable inspection jobs, source-preserving tessellation, physical fabric inventory/mirroring and an interactive GLB viewport. Generate a component shirt, open **3D inspection**, then **Build 3D inspection**. Rotation, zoom, mesh edges and a keyboard-accessible physical-piece selector link each fabric instance back to its actual 2D pattern. Reloads recover progress. WebGL failures retain the source-piece list and existing 2D views.

These results are explicitly **placement inspection**, not assembled garments or drape. Six interfacing roles in the complete shirt remain unresolved; seam allowances are omitted. A machine-readable assembly graph records seam memberships, localized closures, operation dependencies and outstanding turning/binding/orientation semantics. Solver experiments and optional quality refinement are separate from the trusted application inspection path. See the [active 3D plan](docs/plans/pattern-derived-3d-engine.md), [solver experiments](docs/research/3d-solver-feasibility.md), [foundation review](docs/reviews/3d-engine-foundation.md) and [inspection integration review](docs/reviews/3d-inspection.md).

The separate `services/engine/cloth_domain.py` foundation meshes the actual cut contour, including allowances, and embeds the original interior seam paths with source correspondence. Its independent validator checks cut coverage, topology and path continuity. `embedded_constraints.py` builds source-validated, mass-weighted interior sewing constraints with staged closure. The optional full-shirt research harness now couples all 24 fabric instances through their interior source paths without changing rest dimensions. Version-pinned numerical experiments correct sewing update order and membrane arithmetic; copied pre-step states preserve full-step velocity reconstruction. This remains an unfinished assembly solver: binding wraps, turning, convergence and collision acceptance are required. Finite full-shirt runs still fail deformation and seam checks. Contact experiments retain rejected controls rather than treating small seam gaps as an accepted garment; see the [cloth/contact review](docs/reviews/3d-cloth-contact.md) and [embedded sewing review](docs/reviews/3d-embedded-sewing.md).

3D artifacts remain authenticated, private, revision-bound and excluded from all existing exports. New patterns or engine versions mark earlier inspection results as outdated. The SQLite migration advances to schema 5; rollback requires a schema-5-capable application. These changes have not been deployed to the live studio.

For a component `pattern.json` and its captured shirt construction JSON, create a new private output directory with the locked engine Python:

```sh
services/engine/.venv/bin/python scripts/inspect-pattern-3d.py --pattern /absolute/path/pattern.json --construction /absolute/path/construction.json --output /absolute/path/new-private-inspection
bun test services/engine/meshing.test.ts
```

The output contains canonical source mappings and a self-contained GLB with an explicit flat arrangement. Both reveal pattern dimensions and must remain private. This CLI is bounded trusted development code, not the application artifact-installation or export path.

## Documentation checks

```sh
python3 scripts/check_docs.py
git diff --check
```

GitHub Actions checks documentation and the TypeScript application. The real CPU and browser suites require the separately provisioned engine; run them locally before releasing engine/UI changes. Upstream code, model weights and external body assets are not vendored here.
