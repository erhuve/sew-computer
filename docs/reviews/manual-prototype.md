# Manual/CPU prototype implementation review

2026-09-14. Local single-owner prototype; not a public release or physical garment certification.

## Delivered

Original briefs and unresolved requirements remain editable independently of geometry. The interface provides body/shape inputs, BOM, finished-POM definitions, construction, private references, stable callouts and owner-recorded maker comments. Immutable revisions bind jobs and exports. The trusted pinned CPU adapter produces actual shirt, skirt and trouser panels, SVG/JSON/PDF artifacts and explicit assumptions. Draft tech packs work before geometry exists.

SQLite authentication, optimistic draft saves, fenced durable jobs, ownership-checked downloads and redacted three-way manifest imports are integrated. The API rejects missing or corrupt requested artifacts. Export disclosure defaults exclude body inputs, reference images and pattern bytes; free text and disclosed geometry can still reveal body information.

## Adversarial findings addressed

- Slow saves preserve newer client edits; delayed uploads append to the latest document and ignore abandoned project sessions.
- Export-setting changes remove completed downloads and reject completion of requests made with older disclosure settings.
- Scheduler reacquisition fences the same owner's expired attempt and retries within the existing bound. Both geometry validators reject zero-area panels.
- PDF measurements retain notes, and pattern inventory retains engine assumptions. SVG reverses the geometry Y axis; PDF retains upward Y. Required font licensing is visible PDF text, avoiding attachment rejection by the API's trusted-output validator.
- Authoring title/size limits match runtime schemas; collection-add controls respect limits. Import conflicts display the original export, current draft and incoming value.
- Browser checks uncovered stale selectors and mismatched toolbar styles. Tests now use a real isolated HTTP server, and the mobile pattern fits above the fold.

The prior independent frontend/security source reviews supplied the initial findings. This integration pass reproduced the affected behavior and added regression coverage; it is not a new independent review or a public security certification.

## Verification

64 API/export tests and 16 real CPU engine/resource-limit tests pass. TypeScript checking, production build and documentation validation pass. Ten browser cases cover desktop/mobile geometry, real PDF downloads, private uploads/logout, save conflicts, dialog focus, slow saves/uploads, changed disclosure and a populated maker-handoff round trip. No frozen-contract preload is used. Browser evidence is in [verification artifacts](../verification/manual-prototype), using synthetic inputs only.

The populated fixture authors two material rows, two POM definitions, an unsupported requirement, a construction step/callout, two synthetic reference uploads and an owner-recorded maker note through the interface. It exports three real CPU pattern artifacts plus the review PDF, imports one material correction as one reviewed change, and saves a second revision while retaining private body/reference data. The uploaded swatches test reference plumbing; they are explicitly not garment flats.

## Remaining gates and deviations

- This implements the manual/CPU flow, not W3 AI interpretation or the full Design2GarmentCode research pipeline. Unsupported details remain recorded, not generated.
- Trusted workers run with a non-root identity, sanitized environment, bounded resources and attempt-local outputs. This host does not provide a filesystem/network namespace sandbox; arbitrary generated code and public/multi-user execution remain disabled.
- No physical sample, scale calibration, annotation completeness or fit validation is established. Every pattern remains a printable reference. Evidence supersession and calibrated-candidate promotion are not implemented.
- Import acceptance applies a reviewed patch to the draft; the owner explicitly saves another revision. This differs from the plan's atomic accept-and-publish target. Automatic AI proposals are not exposed.
- Full backup restoration remains unsupported. The deletion journal and watermark are separate files on the same host, not independent off-host durability. Never restore an entire stale directory or bypass a recovery marker.
- A durable published URL is not part of this verification. Production browser tests verify the application server locally; they do not certify Zo tunnel/deployment health.

These limitations keep the broader release gates open. They do not prevent using synthetic/manual drafts to evaluate the editor and maker handoff.
