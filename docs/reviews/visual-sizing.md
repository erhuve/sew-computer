# Visual sizing review

2026-09-16 · Isolated implementation; not yet deployed.

## Scope

Six explicitly synthetic starting sizes, synchronized body sliders/numbers, non-mutating centimeter/inch display, schematic measurement highlighting, separate garment controls, and a private reusable body profile. Samples preserve entered/N/A values and disclose replacing estimates and filling missing garment defaults. Out-of-range values remain unchanged. Estimates stay assumed after slider edits. Advanced measurement status/source editing remains available.

Basic preflight rules are shared with the engine. The shoulder/chest warning compares shoulder width with `0.48 * (bust + ease) + 20 mm`, the nominal pinned synthetic back-armhole projection boundary. It is a risk warning: flare, length and corner geometry can cause other failures, and this check is not an exact supported-body classifier. The exact upstream zero-length-edge assertion maps to a safe explanation; other private runtime errors remain generic. No measurements are reduced to work around failures.

## Independent adversarial review

A separate read-only reviewer examined the implementation and found four issues addressed before handoff:

- Inch rounding could move six of ten slider endpoints beyond engine bounds and disable the slider. Slider values now remain canonical millimeters; only display is rounded.
- The schematic's minimum width hid many supported waist adjustments. Its mapping now changes across supported values.
- Sample application filled garment defaults without enough disclosure. Sample details now state that side effect.
- A deleted profile could reappear after restoring an older SQLite file. Profile deletion now shares the durable deletion journal, retains a monotonic version, and upgrades the database to schema 3 so older binaries refuse it.

Authentication, exact-Origin mutation checks, no-store responses, strict profile schema, stale-write rejection, preserved assumption provenance and component isolation were also reviewed. Profile reload responses have request sequencing and unmount guards. Profile conflicts do not use the draft-conflict payload shape.

Final independent review found no remaining blocking issues and independently passed seven targeted tests with 68 assertions. The reviewer confirmed that the shoulder warning matches the pinned model's assumptions and is correctly presented as heuristic.

## Verification

Final checks: 85 application/contracts/API tests and 18 engine tests passed; typecheck, production build, documentation validation and whitespace checks passed. All 17 browser cases passed, with one existing export-link test timing out during the full run and passing on targeted rerun. The first mobile sizing check exposed a below-fold control; removing duplicate headings and shortening secondary copy fixed it, and the complete description-to-pattern browser flow passed afterward.

Regression coverage includes sample preservation, shared range checks, unit endpoints, diagram changes, keyboard sliders, assumed-versus-measured provenance, profile review/application, API auth/conflicts, restart persistence, delete/recreate and replay after stale database restoration. Browser screenshots use synthetic values, at desktop and mobile widths; 320px overflow is checked.

The real CPU suite passed all six sample sizes across shirt/skirt/trouser families, preserved input identity, and reproduced the broad-shoulder zero-length-edge case with the safe error message. This establishes execution of those fixtures, not physical fit, standardized sizing or correctness across all combinations. Schema migration preserves existing drafts. Full backup restoration remains unsupported as documented in the API guide.
