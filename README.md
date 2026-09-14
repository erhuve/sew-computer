# Sew Computer

Consumer garment-design software: turn descriptions, references and sketches into editable garment designs, connected sewing patterns and a revisioned tech pack.

**Status: planning repository. No application, deployed service or sewing-ready pattern release is included.** A separate CPU spike exercised three existing-parameter garment families in Design2GarmentCode; it did not test AI interpretation or physical fit.

## Start here

1. **[Active software-prototype plan](docs/plans/software-prototype.md)** — scope, workstreams, manual-first release milestone, AI enhancement and implementation gates.
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

## Intended first deliverable

A private, single-owner editor with real 2D pattern geometry, manual garment/tech-pack authoring, durable revisions, reviewable PDF/manifest export and feedback incorporation. This manual-first milestone must work with every model route disabled. Text/reference interpretation is a separately gated enhancement; the full released Design2GarmentCode AI path and optional simulation require additional runtime and rights checks.

The first geometry adapter may expose limited, explicit capabilities. Those limits must not become a permanent preset-only product or silently erase unsupported design intent. A concept image, pattern geometry, print calibration and physical fit are different evidence.

## Documentation checks

```sh
python3 scripts/check_docs.py
git diff --check
```

GitHub Actions checks the planning documents. It does not install the garment engine or test an application. Upstream code, weights, body assets and sample exports are not vendored here; their separate integration/licensing gates are in the plan.
