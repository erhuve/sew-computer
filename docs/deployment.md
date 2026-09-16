# Private Zo deployment

The AI-assisted prototype is deployed through the Zo Site at `Code/sew-computer/apps/web`. Its production service is named `sew-computer`. Publish with `public="false"`; public and multi-user operation remain unsupported.

## Owner access

Open https://sew-computer-hatsunemiku.zo.computer while signed into the owning Zo account. The studio separately requires its owner access key. Retrieve it locally from `.local-production/access-key` in the repository and paste it into **Owner access key**. Never put the key in a URL, source control, screenshots or a public asset. Sessions expire after twelve hours.

Choose **New garment** and describe the garment. In **Design**, review the input disclosure, optionally include up to three reference images, and click **Interpret my design**. Review the proposed shape, unsupported details, materials and sewing notes, then **Accept design & set measurements**. Enter body inputs in **Shape & body**, or use **Try an explicitly synthetic example** for a disclosed test body. This fills missing values without overwriting entered measurements or AI shape suggestions. **Save & generate** creates real panels; **Revisions & export → Export draft** creates the review package. Select pattern inclusion to download the matching PDF/SVG/JSON. Patterns remain printable references, without cutting or fit certification.

## Runtime configuration

The private Zo proxy strips request cookies. The studio therefore sends its short-lived session in `X-Sew-Session`, stored in tab-scoped sessionStorage after owner login. The owner key is never persisted in browser storage. Reloads retain the session; logout revokes it server-side and clears it locally. This transport exposes the session to same-origin JavaScript, unlike HttpOnly cookies; do not load untrusted scripts. Reference images and downloads use authenticated fetches rather than credential-bearing URLs. Direct API clients may still use the secure cookie transport. Browser regression tests strip cookies to reproduce the private proxy.

The ignored `apps/web/zosite.json` uses a dedicated publish label, port 56810 and `bun run prod`. Its publish environment includes:

- `NODE_ENV=production`
- `SEW_ALLOWED_ORIGINS=https://sew-computer-hatsunemiku.zo.computer`
- `SEW_DATA_DIR`: the absolute path to `.local-production` in the service checkout
- `ZO_CLIENT_IDENTITY_TOKEN=none`

The production data directory is ignored by Git, outside static assets and separate from development/test data. Do not delete, reset or copy stale files into it. The engine uses the pinned separately installed source and Python environment described in the root README. See [API operations](../apps/api/README.md) for authentication, jobs and recovery limitations.

## Design model connection

This private installation uses `SEW_CODEX_AUTH_FILE` pointing to the existing server-side Codex `auth.json`, with `SEW_AI_MODEL=gpt-6-astra`. Credentials are read per request, never copied into Git, the browser, URLs or geometry workers. The adapter calls Codex's Responses transport with an empty tool list and `tool_choice=none`; it does not run a Zo agent or a shell. This is an installation-specific transport, not a stable public API contract. Reconnect Codex through Zo settings if its login expires; the application does not refresh or overwrite Codex credentials itself.

For a dedicated provider, leave `SEW_CODEX_AUTH_FILE` unset and configure `SEW_AI_API_KEY`, `SEW_AI_BASE_URL` and `SEW_AI_MODEL` for a compatible structured-output Chat Completions endpoint. HTTPS is required except for an explicit loopback address. Provider configuration is server-only, not user-supplied request data.

Both connections make one request with no automatic retry, a 120-second deadline, a 48 KB design-text limit, at most three sanitized 768-pixel references and a persistent 12-request/hour limit. Only one request runs per process. Body fields, size label and private callouts are excluded; brief/BOM text and selected photos can still contain personal information. The dedicated API caps completion tokens at 6,000. Codex does not accept a provider-side output-token cap: its transport instead bounds the answer at 32 KB and the SSE stream at 4 MB, and consumes the owner's Codex allowance. There is no provider-enforced dollar ceiling on this path; the UI states that limitation. Proposals persist for review and become stale after any draft change. Interrupted requests are not automatically replayed; completed proposals can be recovered by reopening the project.

Startup upgrades older databases to SQLite schema 3. Schema 2 adds proposal and request-budget tables; schema 3 adds support for journaled reusable-measurement deletion. Existing projects, sessions and credentials are preserved. Older application code refuses schema 3: do not roll back to a binary without profile-deletion replay or restore stale data to bypass this check. Disable model configuration to return to manual operation while retaining current storage and deletion handling.

## Visual sizing

Shape & body provides six synthetic starting sizes (XS–2XL), explicit centimeter/inch display, body sliders and editable numbers beside an approximate measurement diagram. Apply sample replaces estimated body values and fills unknowns, including missing garment length/ease, while preserving entered values, N/A and existing garment settings. These are original sample fixtures, not standardized clothing sizes or measured population averages. All six samples have been exercised against the three supported geometry families.

Unit toggles only change display; sliders store millimeters. Adjusted sample values remain estimates until explicitly confirmed as measured. Values outside engine ranges remain editable through numbers and are never silently clamped. Garment length, ease and flare have separate controls. The schematic is neither an anatomical reconstruction nor garment simulation; missing dimensions use visible placeholders without entering values into the document.

My saved measurements stores one private reusable profile on the server. Review it before replacing this draft's body fields. Saving does not update other garments. Deleting the reusable copy does not delete measurements in existing drafts or immutable revisions. The profile API uses the same owner authentication, Origin protection and no-store policy as projects, plus version checks. Deletion is recorded in the existing journal; schema 3 prevents older binaries from ignoring that entry. Existing backup/restore limits still apply.

Basic sizing checks are shared with the engine and shown before generation. A shoulder/chest risk warning flags a nonpositive nominal back-armhole projection under the pinned synthetic back-width model; it is a warning, not a complete geometric feasibility test. A zero-length-edge failure now explains the engine limitation without disclosing runtime diagnostics or changing measurements. Broader-body support still requires engine work; the UI does not claim to fix physical fitting or every unsupported combination.

## Releasing changes

This checkout backs a live service whose startup builds the web app. Use an isolated checkout for future edits, test builds and adversarial checks. Run the documented application, CPU and browser suites there before updating the service checkout. On this host, set `PLAYWRIGHT_BROWSERS_PATH` to the absolute path of `.planning/browsers` in the original checkout to use its installed Chromium.

Fetch and integrate concurrent remote changes before pushing. Update the live checkout to the verified release and use Zo `publish_site` with `site_path="Code/sew-computer/apps/web"`, `public="false"`. Do not start a parallel daemon or register a separate service for this Site. Run `service_doctor` for `sew-computer`, check that unauthenticated tunnel requests require sign-in and direct API requests require the studio session, then verify login, generation and export against the running production service.

Restarting the service retains the production database, artifacts and credential. Full backup restoration is unsupported; the deletion journal and watermark must never be bypassed.
