# Private manual/CPU API

`createApi({dataDir, allowedOrigins, authKey?, engine?, exporter?})` returns a Hono application with `.close()`. Mount with `app.route('/api', api)`. Close the old instance during hot reload and shutdown. Neither the engine nor tech-pack implementation is imported here; the parent supplies both adapters. Missing adapters return 503, never fixture output. `api.test.ts` injects explicit test-only adapters and starts no HTTP server.

## Authentication and local storage

All project, upload, job, artifact, comment, export and import routes require the owner session, including direct loopback requests. Only status and login are unauthenticated. Every mutation/login requires an exact configured Origin; no missing-Origin exception. The first startup creates `dataDir/access-key` with mode 0600 if no key is configured. The key is never returned by the API. Copy it only through a trusted operator channel; do not put it in the browser bundle. The owner enters it into the login form. Sessions are persisted as SHA-256 hashes, expire after 12 hours, and use Secure, HttpOnly, SameSite=Strict cookies. Login is globally limited to ten attempts per fifteen minutes, across API instances. Credential changes invalidate sessions, and old live instances consult the current credential hash before issuing sessions.

SQLite uses schema version 1, immediate write transactions, FULL synchronous commits, foreign keys and secure_delete. Data directories are mode 0700; installed content blobs are mode 0400 with random server-generated storage keys. The database is mode 0600. Use a private, non-static data directory and one owner. Do not point dataDir at a public directory or share it among users. This slice exposes no backup/restore or public-release route.

Draft saves require both expectedVersion and expectedRevisionId. Publication copies the checked draft into an immutable revision and atomically advances the draft base and version; publishing unchanged content returns 409. Mutations reject unknown schema fields, unsafe prototype keys, unowned references and duplicate document row IDs. A ProjectState contains immutable revisions, current draft and job/artifact/comment records.

## Bounded in-process queue

The approved prototype uses an **in-process scheduler**, not a separate worker daemon. There is one two-second heartbeat per API instance, cleared by `.close()`. Only the SQLite scheduler lease holder can claim work. Scheduler lease: 15 seconds; attempt deadline: 95 seconds; queued/running admission limit: 32; execution concurrency: one per scheduler. The trusted injected engine must enforce its own 90-second subprocess limit and process-group termination. The API aborts at its deadline, fences stale results and does not release its execution slot until an aborted adapter settles. An adapter that ignores cancellation indefinitely can occupy that slot; arbitrary adapters/programs are not supported.

Job request IDs are idempotent within a project. Head revision, input digest, deletion generation, job generation, attempt token, scheduler epoch and lease expiry are checked before installation. Cancelling, deleting, saving a newer head or requesting newer generation fences old work. Changing only the mutable draft does not invalidate work on its saved revision. Runtime failures are terminal; an interrupted attempt is retried at most once. Graceful close releases the scheduler and requeues bounded interrupted work. After a hard crash, recovery waits for the 15-second scheduler lease expiry. Cancellation may not immediately stop an already-running process on another instance; the persisted fence prevents publication and its scheduler/adapter performs cleanup.

Files are written with fsync and atomic rename under the same SQLite write lock used for installing their references. Recovery markers and a startup reconciler remove orphan blobs and abandoned staging directories. Filesystem writes and SQLite are not a single physical transaction; the markers plus verified references recover the gap. No files are served from staging. Downloads verify project ownership, byte length and SHA-256 digest; SVG is served with CSP sandbox and PDF as an attachment. Engine JSON/SVG/PDF inventory must be complete, consistent with normalized millimetre geometry and reference-only. Static-content checks are an additional guard for trusted generator output, not a general-purpose PDF/SVG sanitizer.

This API does not provide an OS/network sandbox. The injected engine's resource/isolation behavior needs its own validation. Public, multi-user and arbitrary-code use are not supported.

## References, snapshots and imports

Uploads are limited to 10 MiB of sniffed PNG/JPEG/WebP, at most 20 million input pixels, a ten-second raster transformation, two active decoders, and 32 MiB re-encoded PNG output. Metadata is stripped by re-encoding. Animated, SVG and PDF inputs are rejected. Reference IDs are server-generated and tied to one project; owner-entered captions stay in document views.

An export captures one saved revision, committed artifact identities, comments and explicit disclosure flags inside an immutable transaction snapshot. It never substitutes the current draft or a later job result. The exporter receives copies. Body/reference fields and pattern bytes are omitted by default; pattern inventory can still list committed identities. Garment parameters, POMs and owner free text can reveal private/body information and are not anonymized by redaction. Included artifacts must be delivered byte-identically, although their delivery filenames may differ. The API validates the editable projection, manifest identity/digest and delivery.json inventory; the only derived files are the tech-pack PDF, editable manifest and nonrecursive delivery checksums. Missing requested patterns or corrupt selected blobs fail explicitly. Four renders maximum, two per project; an unresponsive trusted exporter holds its bounded slot until it settles.

Import preview uses the server-stored completed export, not incoming evidence/disclosure claims. Editable fields map to document values, matched row-by-row by stable IDs, with physical-unit normalization. Omitted fields/sections are no-ops. Hidden body fields, reference identities and supplied comments/evidence/approvals cannot overwrite authority. Empty arrays/removed exported rows are visible deletions requiring an explicit incoming/keep choice; current additions survive. JSON Pointer paths such as `/bom/fabric/specification` key `resolutions`. All three values are returned as baseline/before/after. Accept rechecks captured draft version, base revision, content digest and deletion generation; it updates only the draft, not a saved revision. Publish separately afterward.

## Deletion and recovery limits

Deletion fsyncs a chained, sequence-numbered journal and separate high-water-mark file before acknowledging the database tombstone. It fences work and purges owned document rows, references, snapshots, exports and private blobs while retaining the minimal tombstone. On startup, journal replay protects against an older database and invalidates sessions when the deletion checkpoint changes. Missing/truncated/mismatched journal or an existing `RESTORE_PENDING` marker fails closed. Runtime requests also check journal/high-water-mark consistency.

These files are separate from SQLite but are on the same host filesystem, **not independently backed-up/off-host durability**. No restore CLI, supported backup procedure or general session-revocation restore guarantee is implemented. Do not restore an entire old directory, reopen a copied database with its old journal, or remove `RESTORE_PENDING` to bypass review. Full restore must remain disabled until independent latest-watermark retention, completeness verification and unconditional auth-epoch rotation are implemented and tested. secure_delete is not a claim of secure erasure from underlying storage or backups.

## Verification

Run `bun test apps/api` for the API-only suite (28 cases), using test-local temporary directories and fake job/export adapters. Geometry-boundary regressions call both real validators. Typecheck the backend independently with:

```sh
bunx tsc --noEmit --target ES2022 --module esnext --moduleResolution bundler --strict --skipLibCheck --types bun apps/api/index.ts apps/api/api.test.ts
```

This suite alone does not certify the real engine/exporter, Site deployment or browser flows. The root build, real CPU and production-browser suites provide separate integration checks.
