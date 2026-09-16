# Shared project and execution contract

**2026-09-14 · Proposed normative integration contract**

Private AI release exception (2026-09-15): the optional Codex login transport rejects provider-side output-token limits. It uses a single tool-free request, bounded input/images, a two-minute deadline, bounded streamed answer and persistent hourly request limit. This does not enforce a provider billing ceiling. The connection is explicitly disclosed in the UI and [deployment instructions](../deployment.md#design-model-connection); dedicated API mode retains a 6,000 completion-token limit. This exception does not grant model output authority over measurements, code, ownership or validation.

This document owns shared semantics. The specialist documents expand it; they must not redefine its states or identities. Decisions remain proposed until implemented and verified. [Product intent](../product/vision.md) overrides an optimization that silently narrows creative freedom.

## 1. Canonical records

- **Project:** opaque ID, owner ID, title, lifecycle state, deletion generation, current revision ID and timestamps. Public sharing is not implicit.
- **DesignRevision:** immutable ID, project ID, parent revision ID, schema version, requirement IDs/statuses, brief/reference versions, garment IR, body/material input snapshots, construction/POM data, timestamp and content digest. A published revision is never edited in place. Undo selects a historical revision for viewing or creates a new revision; it never overwrites history.
- **DraftBuffer:** mutable, owner/project-scoped autosave with a base revision ID and optimistic version. It is explicitly not an exportable revision. Concurrent changes produce a conflict response and preserve both work and the original base.
- **Requirement:** stable ID, original intent, provenance and status `supported | unsupported | unresolved`. Supported means representable by the pinned adapter, not physically validated. A partial result lists which requirements it does not implement. No nearest-preset substitution without explicit acceptance.
- **EditProposal:** ID, project, immutable base revision, **server-captured baseDraftVersion and sourceDraftDigest**, deletion generation, durable requirement IDs, allowlisted operations, proposer provenance, interpretation-adapter/version, estimated/actual provider usage, and conflicts. Neither client nor model may replace the captured draft identity at acceptance.
- **Artifact:** immutable ID, project/revision IDs, kind, private storage key, content digest, MIME/size, complete input manifest, engine/adapter/schema/renderer versions, producing job/attempt, validation report ID and creation time. AI/model provenance belongs in the manifest for outputs depending on it. An artifact ID is not a public authorization token.
- **ReviewEvidence:** append-only author/actor and provenance, exact revision/manifest/artifact scope, named checks, result, notes and evidence references. Owner-entered maker comments identify the recorder separately from the reported reviewer. Do not claim verified external identity in the offline review prototype.
- **ExportSnapshot:** Snapshot ID, immutable revision, exact artifact IDs/digests, complete internal eligibility context and disposition closure at cutoff, frozen classification/reasons/policy version, disclosed evidence subset, disclosure policy, renderer/schema versions, pre-render input digest; output checksums belong to a separate post-render delivery manifest.

A derived artifact can remain valid for a historical revision while being out of date relative to the editor. The UI calls this “from revision N,” not a corrupt file. Never substitute a newer artifact into an older export because its filename is similar.

## 2. Values, units and provenance

Visual sizing implementation (2026-09-16): sample sizes are explicitly synthetic assumptions, not standard size-chart or wearer facts. Display-unit changes do not mutate stored measurements; slider edits are stored in millimeters. Editing an estimate retains assumed status until explicit measured confirmation. The illustrative body diagram is not anatomy, actual geometry or simulation. Out-of-range inputs remain intact. A private single-owner reusable body profile uses a monotonic version and explicit reviewed application into the current draft; it never retroactively edits garments. Profile deletion is journaled, independent of garment deletion, and retains a version tombstone to prevent stale restoration. Existing full-restore limitations remain in force.

Use explicit value states `known | assumed | unknown | not_applicable`. An unknown numeric value is not zero or a made-up estimate. Record source, definition, author/method, relevant input versions and an uncertainty/tolerance only when meaningful.

Canonical length is **millimeters**; preserve the user's entered value and unit for display/audit. Convert once at explicit boundaries with versioned conversions. Distinguish body measurements, panel geometry and finished-garment POMs as different field types. Keep angles, ratios, percentages and discrete counts typed; never scale all numbers together. Unknown units block geometric execution or actual-size claims, not saving an idea.

Verify the pinned engine's conventions before enabling its adapter; do not assume its numeric inputs are millimeters. Convert to/from engine units explicitly. Canonical JSON hashing excludes volatile metadata, rejects NaN/Infinity and has a versioned float/rounding policy. Establish that policy and numerical tolerances with geometry tests before caching by content hash; byte identity is not promised across dependency versions.

A POM derived from geometry includes its definition/derivation and applicable garment/size/material scope. Panel-edge length does not automatically equal worn circumference or finished length. Numeric tolerances, seam allowances and grading rules cannot be inferred solely from attractive images.

## 3. Revision and export transactions

Creating a revision requires the expected current revision and draft version; the service compares both in the transaction. Reference upload completion alone is not a revision. AI acceptance additionally checks the proposal's server-recorded source draft digest, baseDraftVersion, base revision and deletion generation against current state. Applying all accepted operations, publishing the immutable revision and rebasing the editable draft onto that revision happen in one transaction; the draft version increments and its dirty state clears. A stale proposal returns `409`, leaves all newer content untouched and remains available for explicit reconciliation into a fresh proposal. A new client-supplied expected version cannot launder an old proposal into a current one.

An export transaction captures the selected immutable revision and artifact set, disclosure policy, and two distinct evidence sets. **Eligibility context** is service-selected: the complete applicable validation/evidence records and their supersession/withdrawal/dependency closure at the transaction's cutoff, the policy version, and the resulting frozen classification and reasons. **Disclosed evidence** is only the permitted subset included in the delivered package. The snapshot stores both internally; private records need not appear in the outward manifest. Selection/redaction cannot improve eligibility. Rendering uses the frozen decision and snapshot, never recalculates eligibility from disclosed evidence or queries later mutable records. Missing evidence needed to establish eligibility blocks candidate status. Later evidence changes current/new-export eligibility without rewriting a historical snapshot. New edits do not mutate historical exports; the editor offers a current export separately.

Frozen pattern PDFs have their own artifact ID, source revision and input digest printed at creation. Their bytes **do not change when included in another bundle**. Current bundle identity, classification and new evidence go in manifest/cover pages, not restamped pattern sheets. Any restamped, watermarked, repaginated or scaled PDF is a new derivative and cannot inherit exact-byte calibration automatically. Historical reference labels on frozen sheets remain truthful; enclosing documentation explains any later evidence-based classification without altering them.

## 4. Jobs and cancellation

Persist `queued | running | succeeded | failed | cancelled` and a separate `cancel_requested` flag. Each running attempt has a monotonically increasing fencing generation, unpredictable lease token, deadline and heartbeat. Polling/SSE may report progress but are not the source of truth.

Only queued jobs can be claimed. Lease expiry permits a bounded retry with a new attempt and lease; a prior worker's late output cannot publish. Retrying never changes the revision/manifest in place. Cancellation can become terminal only after the coordinator invalidates the lease; terminate/reap the worker process tree and reject late output. If completion already committed first, return its existing terminal status rather than pretending it was cancelled.

Set measured resource/timeout limits and a bounded retry policy by job kind. Validation/unsupported-input failures are not retried as transient faults. Every AI call has provider timeout, token/cost budget and a finite attempt count. Log identifiers/timings/failure categories, not body measurements, raw references or API secrets.

## 5. Security boundary

Initial scope is one owner in a private development Site. **Every project, upload, job, export and artifact endpoint still requires server-side authentication and authorization.** Do not trust client-provided owner IDs, Host headers, opaque URLs or the page's visibility. A platform identity assertion may be used only after verifying its documented server-side authenticity; otherwise implement an explicit owner login/session, not an invented proxy guarantee.

Owner-login fallback: a server-configured credential, a password-hashing implementation, login rate limits and high-entropy server-managed sessions in `HttpOnly`, `Secure`, `SameSite` cookies. Reject cross-origin state changes with explicit Origin/CSRF protection; login and uploads need protection too. No long-lived secret in the browser bundle or URL. Rotate/revoke sessions, fail closed on missing credentials and test API calls directly against the origin server.

Private Zo deployment transport exception: because the proxy removes cookies, an explicit login transport request may return the same expiring session in `X-Sew-Session`. The studio stores it in tab-scoped sessionStorage and sends that header on API, image and download fetches. Never persist the owner key or place session tokens in URLs. Origin checks, hashed server sessions, expiry and revocation remain mandatory. This session is accessible to same-origin JavaScript; the HttpOnly protection applies only to cookie clients.

The engine accepts validated data and a registry of trusted, pinned programs—never arbitrary Python, shell, pickle or model-supplied module names. Use a non-root dedicated worker identity with only attempt-local input/output access, resource limits and no provider/application secrets. Pass bounded IPC messages rather than giving it the application's database. Enforce network/filesystem/process restrictions using mechanisms actually available on the host and verify them adversarially. A supervised process or sanitized environment alone is not a sandbox. If adequate isolation is unavailable, keep untrusted code/model-program execution disabled; any trusted-code-only development fallback must be explicitly documented, not used to claim production isolation.

Uploads initially accept bounded raster reference images, not arbitrary SVG/archives/URLs. Validate decoded dimensions/pixels, re-encode, remove EXIF and isolate decoding where needed. If URL import is added, block SSRF across DNS resolution, redirects and private/link-local addresses; require a separate security gate. Render generated SVG through a safe subset/sanitizer or rasterization, not raw executable markup. PDF rendering must disable remote/local resource fetching, scripts and unapproved paths. Export archives use server-generated names with containment checks.

Sending references/body data to a model provider is explicit and minimized. Providers and costs are visible. Private source images and body details are not automatically bundled in maker exports. Define safe opt-in/redaction defaults and retain auditability without embedding secrets or raw prompts in public files.

## 6. Deletion and retention

Mark project deletion transactionally, increment its deletion generation, cancel/fence active work and immediately deny access. Workers cannot resurrect deleted projects. Garbage-collect attempts/artifacts with referential checks and a recovery window; define retention/backup behavior before storing real user data. Restore tests must preserve ownership, deletion tombstones and manifest consistency. A documented backup retention policy must not promise immediate erasure of every historical backup.

## 7. Export and approval policy

Draft review exports are allowed with unsupported features, missing values and unvalidated geometry when those omissions are explicit. An absent pattern must remain absent; never attach an unrelated pattern to make the bundle look complete. Saving/sharing design intent is not gated on professional review or buying production.

Pattern output classes are **screen preview**, **printable reference** and **calibrated cutting candidate**. Candidate eligibility is a deterministic policy requiring explicit units/size/material/construction scope, correct geometry and assembly references, scale/calibration and required annotations (allowances, grain, notches, cut counts/folds where applicable). The policy must state what is not applicable and why, rather than requiring meaningless marks on every panel. A candidate is still not physical fit approval or a guarantee to manufacture safely without checking it.

Evidence is scoped and additive; there is no single global approved boolean. Changing geometry, sizing, materials, construction, interpretation of a measurement or the relevant engine/exporter version invalidates the affected review scope unless a documented equivalence rule is tested. Draft exports are never silently renamed production-ready. Signed-off checks identify the revision and exact checks actually performed.

## 8. Evidence authority and deterministic eligibility

`ValidationCheck` records are service-established: check ID/version, report ID, artifact and input digests, scope, outcome (`pass|fail|unresolved|not_applicable`), issuer kind/actor, method/tolerance/unit and timestamp. The service issues deterministic check results only from the allowlisted worker/validator path. Owner-recorded physical measurements have a distinct issuer kind and retain actor, method and measurement evidence; they are not automatically professional certification. Imported comments and model assertions remain reported statements, never server-issued validation. Authentication and the service, not submitted JSON, establish issuer identity.

`ReviewEvidence` additionally carries exact artifact/process/material/size scope, supporting attachment IDs, and explicit `supersedes`/`withdraws` relationships with a reason and actor. This extends the canonical ReviewEvidence record; it is not a separate competing EvidenceRecord entity. Records are append-only. Only an authorized actor for that evidence class can disposition an earlier record; no actor can turn a report into a different authority class. Physical calibration includes printer, paper and settings as well as pattern bytes. A newer timestamp alone does not supersede anything.

For each required check, the eligibility reducer first selects exact-scope, permitted-authority, non-withdrawn records and applies valid explicit supersession links. An effective failure, unresolved result, or contradictory effective evidence blocks candidate classification. A pass is insufficient while a same-scope failure remains effective. Resolving a clerical error preserves its history and requires an explicit scoped reason; a different printer's pass cannot resolve the conflict. Changing evidence recomputes current eligibility, never rewrites a historical export. No paid reviewer is required to record honest owner calibration or obtain draft/reference exports.

## 9. Redacted editable-manifest import

An import is a proposed allowlisted patch against the **stored source export projection**, never replacement project state. The projection identifies disclosed fields and redactions. Absence and redaction mean no operation; an explicit `unknown`/`not_applicable` value differs from absence; intentional clearing uses a separately visible `clearField` operation. Compare disclosed editable values against the original projection, not against missing private data. Unknown/unowned source snapshots require a new-project import with explicit resolution, not mutation of an existing project.

Treat supplied provenance, actors, artifact hashes, evidence, disclosure flags and approval claims as untrusted. Source projection and authority metadata come from owned server records. A one-BOM-field edit in a body-redacted package proposes only that field; body inputs, references, historical evidence and other hidden data survive. Proposal acceptance still uses section 3's current-draft concurrency transaction.

## 10. Restore completeness and revocation

Successful deletion acknowledgement requires a sequence-numbered tombstone to be durably appended to a deletion journal stored independently of database backups, then the matching database tombstone committed. If the second operation fails, replay completes deletion; the journal entry is not discarded. Journal storage and flush semantics are a W0 durability probe, not a claim that SQLite and files share a transaction. Independent does not imply an automatically available second disk/service: select and test storage before enabling recoverable backups.

Snapshots record the journal checkpoint. A separately durable high-water mark establishes the latest acknowledged deletion sequence; restore must verify journal integrity and contiguous replay through that mark before reopening access. Missing/truncated journal, unavailable high-water mark or uncertain completeness keeps access closed pending operator recovery. Restores rotate an authentication epoch and invalidate all restored sessions; session rows alone never restore authority. Body/reference data retained in backups stays covered by the documented retention policy and cannot silently become accessible after restoring an older snapshot.

Geometry and technical exports can reveal body dimensions even when the original body-input fields are redacted. Disclosure UI must explain this residual information; redaction is not a promise of anonymization.

## 11. Common publication postconditions

Ordinary manual publication and proposal acceptance share one transaction: check the server draft's captured base revision, version, content digest and project deletion generation; publish the exact checked content; advance the project pointer; rebase the server draft onto the new revision; increment its version; clear the server dirty state; and return the new revision plus draft identity. Proposal acceptance adds its recorded-source checks and accepted operations. A competing request using the previous draft version conflicts. An acknowledgement never authorizes discarding newer unacknowledged client edits: keep that local buffer and deliberately rebase or reconcile it against the returned identity before another save.

## 12. Import-time three-way reconciliation

Before constructing an import proposal, compare each permitted changed field across (A) the owned stored source export projection, (B) the captured current server draft, and (C) the imported value, using canonical typed/unit-normalized values. A redacted/absent field remains a no-op. If C equals B, propose no change; if B equals A, C can be proposed; if both B and C differ from A and from each other, require an explicit conflict decision showing all three values. This applies even when the current draft still has the exported revision as its base. Explicit clearing obeys the same comparison. Preserve all unrelated current-draft edits. Bind the reconciled proposal to B's captured identity and enforce section 3 again at acceptance; a change after capture can still conflict.

## 13. Follow-up regression obligations

Component drafting extension (2026-09-16): optional `garment.design` and requirement feature keys participate in the immutable document digest; absent legacy fields are never default-injected into old revisions. Component geometry carries structured edges, cut contours, grain/registration/closure marks, assembly ratios and derived specifications. Validate them against the selected construction before publication/export. Digital coverage never grants physical-fit or cutting authority. Derived specifications are disclosed only with patterns, remain separate from editable rows, and are ignored as import authority. See [implementation review](../reviews/garment-pipeline.md).

- Export with a matching pass and failure while disclosing only the pass: candidate classification stays blocked. The same holds when a withdrawal record is hidden. Evidence after snapshot cutoff changes new assessments only.
- Publish an ordinary manual draft, immediately edit and publish again using the returned identity; succeed without a spurious base conflict. A concurrent request with the previous identity must not overwrite it. Preserve a newer unsaved client buffer during acknowledgement.
- Export BOM A, change the same field locally to B without publishing, import external C: require explicit three-way reconciliation showing B as current. Unrelated local changes and omitted private fields survive.
