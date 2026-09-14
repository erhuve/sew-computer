# Design2GarmentCode: evidence and limits

**Inspected 2026-09-14 · Engineering evidence, not garment validation**

## Source baseline

- Project and paper: https://style3d.github.io/design2garmentcode/ and https://arxiv.org/abs/2412.08603
- Implementation: https://github.com/Style3D/design2garmentcode-impl
- Inspected source commit: `7065b3ef01ff61f4462e871d4cb439a0b97c48db`. A remote fetch on 2026-09-14 confirmed that commit was still upstream `main`.
- [README at the inspected revision](https://github.com/Style3D/design2garmentcode-impl/blob/7065b3ef01ff61f4462e871d4cb439a0b97c48db/README.md).
- [Root MIT license](https://github.com/Style3D/design2garmentcode-impl/blob/7065b3ef01ff61f4462e871d4cb439a0b97c48db/LICENSE).

Sew Computer has not vendored upstream code, models or body assets in this planning commit. A code license does not settle rights for every dependency, model weight or dataset. Record licenses, versions, permitted uses and redistribution constraints separately before integrating each asset.

## Observed CPU smoke test

The existing GarmentCode geometry implementation was exercised with fixed design parameters, not a text/image interpretation model:

1. Load `assets/design_params/default_template.yaml` with a safe YAML loader and take its `design` mapping.
2. Make independent copies; set `meta.upper.v`/`meta.bottom.v` to `Shirt`/null, null/`SkirtCircle`, and null/`Pants`.
3. Load the bundled `assets/bodies/mean_all.yaml` using `BodyParameters`. This research body fixture is not a user's measurements or a commercial body-asset clearance.
4. Construct `MetaGarment`, assemble its pattern and require nonempty panels, finite panel vertices and valid edge-endpoint indices.
5. Query the upstream self-intersection check and serialize JSON/SVG/PNG and printable SVG/PDF artifacts with 3D serialization disabled.
6. Check the produced artifacts rather than trusting the research script's success messages. A follow-up filesystem check found all 15 listed files nonempty.

| Fixture | Panels | Stitch pairs | Self-intersection flag | Serialized artifacts |
|---|---:|---:|---|---|
| Shirt | 4 | 6 | false | JSON, SVG, PNG, printable SVG, printable PDF |
| Skirt | 4 | 6 | false | JSON, SVG, PNG, printable SVG, printable PDF |
| Trousers | 6 | 24 | false | JSON, SVG, PNG, printable SVG, printable PDF |

These results are a feasibility spike, not a checked-in test suite or a clean-install reproduction. Capturing a dependency lock, portable fixtures with cleared rights, expected geometry properties and automated smoke tests is an implementation task. The existing exports are deliberately not distributed here as sewing-ready assets.

## What this does not demonstrate

- Text, sketch or photo interpretation, or faithful preservation of an arbitrary design.
- Correct seam compatibility, ease, allowances, grainlines, notches, cut counts, sizing or grading.
- Physical print scale, tiling/calibration, material behavior, fit or sewing feasibility.
- Robustness across parameter ranges, unusual geometry or malicious inputs.
- A complete tech pack, consumer editor, safe worker service or deployed application.

No sample garment was cut or sewn from this test. In particular, a `false` self-intersection flag and a PDF file do not establish a usable production pattern.

## AI and simulation are separate dependencies

The released path documents a multimodal-model API for interpretation, Qwen2-VL-2B-Instruct base weights and the authors' fine-tuned projector weights. The optional Warp simulator is another runtime. Neither the full model path nor simulation was installed or exercised in this spike.

Do not advertise a general LLM-to-JSON adapter as a reproduction of the released Design2GarmentCode pipeline. If used for the first software slice, identify it separately and measure it against explicit supported requirements. Do not expose upstream configuration files, API credentials or an unrestricted research GUI as the consumer application.

## Integration posture

Use this source as a promising geometry foundation, with a pinned adapter and explicit capability/validation reports. Verify units, serialization behavior and entrypoint failure semantics from the pinned source. Replace fragile research-level success accounting with artifact checks and structured errors. Extending supported garment components is planned engineering, not a reason to permanently restrict the user's creative brief.

See [Engine and AI design](../design/engine-ai.md) and [the active prototype plan](../plans/software-prototype.md) for the proposed integration and its release gates.
