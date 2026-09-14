# Backend, Security and Operations

**Proposed · 2026-09-14**

This design expands [project-contract.md](project-contract.md), which owns record identities, units, lifecycle states and export classifications. [Product intent](../product/vision.md) governs scope. All application behavior and controls below are **unimplemented and untested**. The [upstream evidence](../research/design2garmentcode-evidence.md) establishes only the stated fixed-parameter CPU geometry smoke test, not application security, inference, calibration or physical fit.

## 1. Runtime and module boundaries

Start with one authenticated owner in a private development Site. Public sharing, multi-user tenancy, bidding, payments and crowdfunding are later systems.

**Canonical repository layout:** `apps/web` for the React editor, `apps/api` for the Bun/Hono API and durable coordinator, `services/engine` for the constrained Python worker/adapter, and `packages/contracts` for shared schemas. This matches the active plan; any conceptual server/geometry names in this document map to those paths, not competing entrypoints.

Proposed repository layout:

```text
apps/web/src/              React editor, authenticated API client
apps/api/src/
  auth/                   Login, sessions, CSRF, authorization
  routes/                 Schema-validated HTTP handlers
  revisions/              Draft concurrency, revision publication
  jobs/                   Queue, leases, cancellation, reconciliation
  artifacts/              Private storage, validation, installation
  exports/                Snapshot selection, delivery policy
  providers/              Consent, budgets, provider requests
  db/                     SQLite migrations and transactions
services/engine/           Python entrypoint, trusted pinned adapter
workers/render/           Restricted image/PDF processing
packages/contracts/       Versioned API and IPC schemas
tests/security/           Adversarial and fault-injection tests
ops/                      Deployment, backup and recovery runbooks
```

React never receives storage paths or provider credentials. Bun alone owns SQLite and artifact publication. Workers receive bounded inputs, not database access. Geometry uses trusted registry entries; model output cannot select Python modules or execute programs. Unsupported requirements survive interpretation, revision publication and export.

## 2. Authentication and route contract

Use documented, verified server-side platform identity only if available. Private page visibility is not API authentication. Otherwise implement the contract’s owner-login fallback: Argon2id password hashing, rate-limited login, opaque server-managed sessions, revocation and secure cookies. Validate configured origins and CSRF tokens for mutations, including login and uploads; never derive trust from `Host`.

For the private Zo proxy, which strips cookies, the contract permits explicit `X-Sew-Session` transport of the same expiring session. The browser stores this token in sessionStorage; it is accessible to same-origin JavaScript. All protected image and export requests use authenticated fetches, never URL tokens. See [deployment](../deployment.md) for the operational tradeoff and cookie-stripping regression coverage.

Representative routes:

| Route | Contract |
|---|---|
| `PUT /api/projects/{projectId}/draft` | Expected optimistic version and base revision; conflict preserves work. |
| `POST /api/projects/{projectId}/revisions` | Publishes the specified draft version against expected current revision. |
| `POST /api/projects/{projectId}/jobs` | Immutable revision, job kind and idempotency key. |
| `POST /api/projects/{projectId}/jobs/{jobId}/cancel` | Requests cancellation; returns persisted status. |
| `POST /api/projects/{projectId}/exports` | Exact revision, selected artifacts, evidence and redaction policy. |
| `GET /api/projects/{projectId}/artifacts/{artifactId}/content` | Authorized private delivery, never static-directory access. |
| `DELETE /api/projects/{projectId}` | Transactional tombstone and immediate access denial. |

Example authenticated request:

```http
POST /api/projects/prj_7qK2m9Rb/jobs
Cookie: sew_session=<session>
Origin: https://configured-app-origin.example
X-CSRF-Token: <session-bound-token>
Idempotency-Key: req_a8Tx5Jv4
Content-Type: application/json

{"revisionId":"rev_6Nw3c8Lp","kind":"geometry"}
```

IDs are opaque locators, not credentials. Derive owner identity from the session and authorize every nested record against the project. Return `401` without authentication, indistinguishable `404` for inaccessible resources, `409` for concurrency/idempotency conflicts and `422` for invalid input. Progress streams require identical authorization.

## 3. Database consistency

Enable foreign keys and explicit transactions; migrations must fail closed on incompatible schema versions. Store immutable revisions and manifests separately from mutable drafts.

Revision publication atomically compares draft version, base/current revision and deletion generation, inserts the revision and updates the project pointer. On conflict, leave the draft unchanged; the client retains its rejected payload for reconciliation. Stale proposals cannot bypass this check.

A unique owner/project/request-key constraint binds idempotency to a canonical payload digest. Concurrent identical requests return the same job; different payloads conflict. Establish canonicalization before hash-based caching.

Export creation selects authorized, revision-compatible artifacts and scoped evidence in one transaction and inserts an immutable `ExportSnapshot`. Rendering uses only that snapshot, never current editor state. Failed rendering affects its job, not snapshot contents.

Proposal jobs capture base revision, draft version, immutable source-draft digest and deletion generation server-side. Acceptance compares the captured values in one transaction, applies every accepted operation, publishes the revision and rebases/increments the draft. Client-provided current identities cannot replace the captured source identities.

## 4. Worker leases and IPC

Use a permission-restricted Unix socket or inherited pipes with length-prefixed, schema-validated messages. Reject oversized messages, unknown protocol versions and unsolicited results.

Each assignment contains job/attempt IDs, fencing generation, unpredictable lease token, deadline, deletion generation, revision ID, manifest digest and attempt-local input descriptors. Heartbeats and results repeat these identifiers. Lease tokens never enter logs.

Only `queued` jobs are claimable. Expiry first invalidates the running lease; an eligible bounded retry returns the job to `queued`, after which claiming creates a new attempt and higher generation. Permanent validation failures become `failed`.

Cancellation persists `cancel_requested`, invalidates the lease and arranges process-tree termination/reaping. A completion already committed wins. Every result publication rechecks the active lease, cancellation, deletion generation and requested manifest inside its transaction. Killing processes alone is not fencing.

Resource ceilings and retry counts are configured by job kind; benchmark them before release. No unlimited retries, output growth or heartbeat-based extension beyond a fixed attempt deadline.

## 5. Filesystem installation and reconciliation

SQLite and files are not one atomic system:

1. Worker writes only into its attempt staging directory.
2. Coordinator rejects symlinks, special files, escaping paths and unexpected outputs; verifies structure, dimensions and digests.
3. Copy into coordinator-owned storage, revalidate, then durably write a commit marker listing intended immutable keys and checksums.
4. Install files using same-filesystem atomic renames and required file/directory synchronization.
5. In one database transaction, recheck fencing and publish artifact metadata plus job success.

Worker-writable files must not remain mutable after verification.

Startup and scheduled reconciliation inspect markers and references. Installed-but-unreferenced files are quarantined for recovery, then collected. Missing or mismatched committed files block delivery and raise an integrity incident; reconciliation cannot silently invent replacements or rewrite historical success. Recovery regeneration requires the same pinned inputs and explicit provenance.

Downloads recheck authorization, deletion and artifact integrity. No `latest.pdf` lookup or filename-based association is permitted.

## 6. Host deployment gates

Require TLS, durable storage with verified SQLite locking/synchronization behavior, restrictive filesystem permissions, migrations, bounded disk usage and recoverable backups. Use WAL only on a filesystem demonstrated to support its requirements.

Run dedicated non-root identities. A supervisor provides restart behavior, **not sandboxing**. Verify actual filesystem, network, process and resource restrictions through hostile worker tests: application data and secrets unreadable, unrelated processes inaccessible, outbound connections denied.

If host isolation is insufficient, disable untrusted program execution and document any trusted-code-only development fallback. Do not claim production isolation. Image decoders and renderers also require containment appropriate to hostile input.

## 7. Uploads, rendering and providers

Initially accept JPEG, PNG and WebP only. Proposed starting limits are 10 MiB encoded and 20 megapixels decoded; validate memory behavior before adoption. Check signatures, reject animation, decode with resource limits, re-encode and strip metadata. Browser previews receive sanitized derivatives, not originals.

URL imports remain disabled. Enabling them requires SSRF tests covering DNS rebinding, redirects, IPv4/IPv6 private ranges, link-local metadata endpoints and connection-time destination enforcement.

Generated SVG uses a tested safe subset or isolated rasterization. Reject scripts, event handlers, external resources and active embedding. PDF rendering disables scripts and resource fetching; required assets enter through an explicit allowlist. Archive names are server-generated and containment-checked.

Provider calls run outside geometry workers. Require explicit consent naming provider and disclosed fields; minimize body/reference data. Keep keys server-side and endpoints administratively allowlisted. Reserve a visible request budget before calling; enforce timeouts and finite attempts. Ambiguous provider failures must not trigger uncontrolled duplicate charges. Paid review or manufacturing purchase never gates creation or DIY exports.

## 8. Deletion, retention and recovery

Deletion atomically records the tombstone, increments deletion generation and fences active work. Subsequent requests fail immediately; already transmitted bytes cannot be recalled.

Proposed retention: staging/orphans 24 hours; deleted-project live files seven days; redacted operational logs 14 days; encrypted daily backups 30 days. Validate and disclose this policy before real private inputs. Garbage collection checks references and recovery windows.

Backups pair a SQLite-consistent snapshot with its exact immutable file set while publication and collection are briefly paused. Keep deletion records sufficient to replay post-snapshot deletions during restore. Restore tests verify ownership, tombstones, checksums and manifests before reopening access; backups do not imply immediate historical erasure.

**Restore protocol:** follow project contract §10. Before acknowledging deletion, append and durably flush a sequence-numbered tombstone to a journal independently retained from database snapshots, then commit the database tombstone; replay handles interruption between these steps. Snapshot checkpoints plus an independently durable acknowledged high-water mark must prove contiguous deletion replay before reopening access. Missing/truncated journal or unknown watermark fails closed. This independent storage/flush property is a W0 proof obligation, not something assumed from the host filesystem. Every restore creates a fresh authentication epoch and invalidates all restored sessions. Never resurrect revoked authority from backed-up session rows.

**Export/import trust:** pre-render snapshot input digests exclude rendered-file hashes; the post-render delivery manifest contains file hashes, with its own hash stored externally. Frozen calibrated pattern bytes remain unchanged in later bundles. Manifest imports are allowlisted changes against owned, stored disclosure projections; absent/redacted fields do not clear project state. Ignore submitted authority, provenance and evidence claims. Evidence issuer classes and conflict reduction are service-controlled under project contract §8.

## 9. Acceptance evidence and handoff

An independent adversarial review should block release on these suites:

- Direct unauthenticated access, cross-project IDs, CSRF and secret leakage.
- Stale drafts/proposals, duplicate keys and concurrent export/edit.
- Expired leases, zombie output, cancellation/completion races and deletion during installation.
- Crashes after every installation step; missing files and backup restoration.
- Image bombs, active SVG, renderer resource access and path escapes.
- Unsupported requirements preserved; unknown units blocked; no invented approval carry-forward.

Record build/dependency versions, host restriction probes, test results, restore evidence and redacted failure categories. The upstream smoke test satisfies none of these gates.

Coordinate adapter manifests and validation with [engine-ai.md](engine-ai.md); frontend consumes conflicts and revision-scoped status; export rendering consumes frozen snapshots. Proposed commits separate contracts/schema/auth, fenced job/storage operations, then export/security/recovery tests. Update the [prototype plan](../plans/software-prototype.md) only with verified evidence, not implementation intent. (•̀ᴗ•́)و
