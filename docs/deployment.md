# Private Zo deployment

The manual/CPU prototype is deployed through the Zo Site at `Code/sew-computer/apps/web`. Its production service is named `sew-computer`. Publish with `public="false"`; public and multi-user operation remain unsupported.

## Owner access

Open https://sew-computer-hatsunemiku.zo.computer while signed into the owning Zo account. The studio separately requires its owner access key. Retrieve it locally from `.local-production/access-key` in the repository and paste it into **Owner access key**. Never put the key in a URL, source control, screenshots or a public asset. Sessions expire after twelve hours.

Choose **New garment**, enter an idea, and author the design manually. To try actual geometry, open **Shape & body**, choose a supported family and provide its required measurements. **Try an explicitly synthetic example** supplies disclosed test assumptions. **Save & generate** creates real panels; **Export draft** creates the review package. Descriptions are retained but are not interpreted by AI. Patterns remain printable references, without cutting or fit certification.

## Runtime configuration

The private Zo proxy strips request cookies. The studio therefore sends its short-lived session in `X-Sew-Session`, stored in tab-scoped sessionStorage after owner login. The owner key is never persisted in browser storage. Reloads retain the session; logout revokes it server-side and clears it locally. This transport exposes the session to same-origin JavaScript, unlike HttpOnly cookies; do not load untrusted scripts. Reference images and downloads use authenticated fetches rather than credential-bearing URLs. Direct API clients may still use the secure cookie transport. Browser regression tests strip cookies to reproduce the private proxy.

The ignored `apps/web/zosite.json` uses a dedicated publish label, port 56810 and `bun run prod`. Its publish environment includes:

- `NODE_ENV=production`
- `SEW_ALLOWED_ORIGINS=https://sew-computer-hatsunemiku.zo.computer`
- `SEW_DATA_DIR`: the absolute path to `.local-production` in the service checkout
- `ZO_CLIENT_IDENTITY_TOKEN=none`

The production data directory is ignored by Git, outside static assets and separate from development/test data. Do not delete, reset or copy stale files into it. The engine uses the pinned separately installed source and Python environment described in the root README. No AI credentials are required. See [API operations](../apps/api/README.md) for authentication, jobs and recovery limitations.

## Releasing changes

This checkout backs a live service whose startup builds the web app. Use an isolated checkout for future edits, test builds and adversarial checks. Run the documented application, CPU and browser suites there before updating the service checkout. On this host, set `PLAYWRIGHT_BROWSERS_PATH` to the absolute path of `.planning/browsers` in the original checkout to use its installed Chromium.

Fetch and integrate concurrent remote changes before pushing. Update the live checkout to the verified release and use Zo `publish_site` with `site_path="Code/sew-computer/apps/web"`, `public="false"`. Do not start a parallel daemon or register a separate service for this Site. Run `service_doctor` for `sew-computer`, check that unauthenticated tunnel requests require sign-in and direct API requests require the studio session, then verify login, generation and export against the running production service.

Restarting the service retains the production database, artifacts and credential. Full backup restoration is unsupported; the deletion journal and watermark must never be bypassed.
