# AI-assisted creation integration review

2026-09-15. Private, single-owner release. This updates the AI status in the historical manual-prototype review.

## Behavior

The Design tab sends an explicitly authorized brief and optional sanitized references to a tool-free language model. Its output is strictly validated into garment parameters and additions to requirements, BOM, POM definitions and suggested assembly notes. The owner reviews the proposal before acceptance. Acceptance checks the server-captured draft version, revision, digest and deletion generation, then publishes a revision and rebases the draft atomically. Existing body inputs, references, callouts and technical rows survive. New AI numerical suggestions are marked assumed; the model schema cannot establish known measurements.

Entered measurements feed the existing pinned CPU engine. The resulting immutable revision connects patterns and draft tech-pack PDF/JSON. Model/adapter provenance follows accepted designs into the manifest, PDF and geometry input digest. Existing garments and manual operation remain compatible. The full Design2GarmentCode MMUA/projector pipeline is not installed by this change.

## Adversarial checks

Review covers malformed/extra fields and executable output; invented known measurements; private body/callout exclusion; concurrent and stale proposal acceptance; duplicate acceptance; late completion after deletion; provider failure; preserved local edits during delayed responses; export/import provenance; and repeated geometry generation from an unchanged accepted revision. Images are selected explicitly, ownership-checked, resized and re-encoded. The inference request supplies no tools and no workspace access. Prompts cannot grant executable capabilities.

The model's requirement assessment remains fallible. The complete original brief is retained independently, and the review UI displays unsupported requirements and open decisions. No automated completeness or physical-sewing certification is claimed. This is an integration and adversarial regression pass, not an independent audit.

## Verification

76 API/export/provider regression tests, 16 real CPU/resource-limit tests and 14 browser cases pass across the verification runs. TypeScript, production build, documentation checks and whitespace checks pass. Browser fixtures test transport, UI, concurrency and the real CPU/export pipeline; live model checks are separate from deterministic model accuracy tests. A regression covers schema-1 database migration preserving the original draft.

Live GPT-6 Astra calls produced accepted designs and successful real CPU generation/export for a blue linen sleeveless top (4 panels), red cotton half-circle skirt (2 panels) and black cotton trousers (4 panels), each using explicitly synthetic body measurements. Skirt zip/embroidery and trouser pocket/fly requirements remained unimplemented and visible. A separate live image request recognized a synthetic red sleeveless-top sketch and retained red as a material requirement without inferring body inputs. These are four observed examples, not a general accuracy estimate.

[Desktop proposal](../verification/ai-prototype/proposal-desktop.png) and [mobile patterns](../verification/ai-prototype/pattern-mobile.png) use the deterministic browser model fixture and real CPU geometry. The [sample tech pack](../verification/ai-prototype/live-top-tech-pack.pdf) and [editable manifest](../verification/ai-prototype/live-top-manifest.json) come from the live model top test and synthetic body inputs. They are review artifacts, not cutting-ready samples.

## Limits

The current engine supports a symmetric sleeveless top, circular skirt and basic darted trousers with length/ease/flare controls. Sleeves, collars, custom necklines, pockets, waistbands, closures, embroidery and arbitrary topology are not generated. AI notes can specify them but do not implement them. Missing dependent body dimensions remain disclosed synthetic assumptions. Patterns lack seam allowances, grain/notches, printer calibration and physical-fit validation; they remain printable references for development and maker review.

The Codex connection reuses local login credentials without copying or refreshing them, and depends on its Responses transport remaining compatible. A separate API-key provider is also supported. Deadline/byte/request limits apply, but the Codex transport does not expose a provider-side token or dollar cap. See [deployment](../deployment.md#design-model-connection).
