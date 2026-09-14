# Adversarial planning review

**2026-09-14 · Resolved specification findings; implementation remains unbuilt**

## Scope and method

Four independent specialist briefs covered editor/product UX, engine/AI integration, backend/security, and tech-pack/maker handoff. Their documents were integrated under one [active plan](../plans/software-prototype.md) and [normative project contract](../design/project-contract.md).

Two independent adversarial perspectives then reviewed the integrated text: security/concurrency/contracts and product/manufacturing/UX. A second pass checked actual repairs and found three further ambiguities. A final independent, narrowly scoped closure review found those three resolved at the specification level, with no new P0/P1 directly introduced by those fixes.

The reviews consumed document text. They did not run software, validate a pattern physically, inspect deployment security or reproduce the full AI pipeline. Raw prompts/responses remain excluded from Git; this report records the actionable findings and final dispositions rather than presenting internal transcripts as product documentation.

## Findings and resolutions

| ID | Priority | Finding | Contract resolution and required regression |
|---|---|---|---|
| P-01 | P1 | An AI proposal could match the published revision while overwriting a newer unpublished draft. | Proposal and interpretation jobs bind the server-captured revision, draft version, draft-content digest and deletion generation. Acceptance checks all identities transactionally. A proposal from draft v7 must conflict after v8 is saved; neither v8 nor the proposal is discarded. Contract §§1, 3, 11. |
| P-02 | P1 | Passing and failing evidence for the same scope had no conflict/authority reducer. | Typed service-issued records distinguish engine checks, owner-recorded measurements and reported external comments. Immutable, scoped supersession/withdrawal history and unresolved-failure rules govern eligibility. Forged issuers cannot confer authority; pass→fail blocks candidate eligibility until explicitly resolved. Contract §§1, 7, 8. |
| P-03 | P1 | Stamping each export's identity onto calibrated pattern pages changed their bytes, and PDF/manifest checksums could become circular. | Separate pre-render snapshot digest from post-render delivery checksums. Reuse unchanged frozen pattern sheets; later bundle association and classification live outside those bytes. Re-rendered derivatives do not inherit exact-artifact calibration. Contract §§1, 3; handoff design §1. |
| P-04 | P1 | Reimporting a redacted manifest could delete hidden private inputs. | Compare allowlisted changes against the stored disclosed projection. Absence/redaction is no-op, not unknown or clearing. Explicit removals become separately reviewed operations; imported provenance cannot replace service records. Contract §§1, 12; handoff design §§2, 5. |
| P-05 | P1 | Restoring an old backup could revive deleted projects or revoked sessions. | Deletion success follows an independently durable deletion journal; restoration verifies replay completeness/watermarks before reopening access. Rotate the authentication epoch and invalidate restored sessions. Missing deletion history fails closed. Contract §§5, 6. |
| P-06 | P2 | Specialist links and module paths conflicted. | Use one canonical layout: `apps/web`, `apps/api`, `packages/project-schema`, `services/engine`, `tests`, and `docs`. Supporting-document links match actual repository files and are checked automatically. Active plan and backend design §1. |
| P-07 | P1 | The supposedly manual-first release still depended on successful AI integration. | Define manual milestone M0 over W0/W1/W2/W4 and applicable W5 review, with all model routes disabled. W3 yields a separately reviewed AI enhancement M1. Active plan, integration milestones. |
| P-08 | P1 | A complete export specification could be satisfied with mostly empty fields because the manual authoring workflow was unspecified. | Require exposed UI for view roles/callouts, BOM, POM definitions, construction notes and uncertainties. M0 includes a substantive, non-default garment authored through the interface and a field-targeted feedback round trip. Incomplete drafts remain exportable generally. Editor design and active-plan M0/W4 criteria. |
| P-09 | P1 | Export classification might use only outward-disclosed evidence, allowing hidden failures to disappear. | Atomically freeze a complete, service-selected internal eligibility context separately from its disclosed subset. Undisclosed failures, withdrawals and supersession dependencies retain their effect. Rendering cannot recalculate from a redacted subset or later mutable evidence. Contract §§1, 3, 13. |
| P-10 | P2 | Ordinary manual publication lacked explicit postconditions for rebasing drafts and preserving newer browser edits. | Manual publish and proposal acceptance share atomic pointer/draft-base/version updates. Old-version operations conflict. Server acknowledgement cannot discard a newer unacknowledged client buffer, which requires deliberate rebase/reconciliation. Contract §§11, 13. |
| P-11 | P2 | An imported old export could overwrite local draft changes that existed before import captured its version. | Perform three-way reconciliation before proposal creation using stored export projection, captured current draft and imported values—even when the published revision matches. Conflicting values require explicit decisions; unrelated draft edits survive. Acceptance still checks the captured draft identity. Contract §§12, 13. |

Duplicate findings across reviewers are consolidated above using the higher reported priority. All eleven are addressed in the normative documents; none is claimed fixed in running software.

## Closure and remaining gates

The final targeted reviewer closed P-09, P-10 and P-11 and found no remaining blocker for those planning cases. The earlier follow-up reviews closed the first eight findings. This supports starting W0 and the manual-first implementation, not skipping release tests.

Still intentionally unverified: reproducible clean installation, commercial rights for each dependency/asset/weight, host sandbox enforcement, deployed auth and durability, supported AI runtime, robust geometry and assembly checks, real printer calibration and physical fit. These have explicit implementation gates rather than implied approval.

Before any release, implement the specified adversarial regressions and independently review the actual code and behavior. Before making cutting/fit claims, collect the specific evidence required by the export class; a successful document check or review verdict does not substitute for it.
