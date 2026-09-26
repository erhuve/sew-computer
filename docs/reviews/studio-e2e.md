# End-to-end garment studio

**2026-09-26 · Isolated local implementation · Not deployed**

The supported relaxed-shirt flow now connects a description and reviewed AI proposal to actual 2D drafting, a newly generated garment preview, saved revisions and matching PDF/JSON exports. `/demo` remains a set of pre-generated examples; the normal studio generates private artifacts from each project's own inputs. The studio does not request `/demo-fixtures/` when generating or displaying a project garment.

## Product flow

Create a garment with a description; its name is optional. One explicit AI consent checkbox can start interpretation immediately. Review the proposed design, then choose **Use design & preview** to apply clearly labeled sample M estimates and generate, or choose your own measurements. Entered measurements survive sample application; unknown and assumed values remain distinguishable. Unsupported requests remain visible and editable.

A single **Generate garment / Update garment** action saves the required revision and generates patterns. The server automatically queues its garment preview after those patterns commit, including after a process restart. Reopening a generated project shows its garment. Editing construction changes the saved source; the old visual is labeled until the new generation completes. Source selection works across the garment, flat-piece inspection and 2D pattern. **Download** opens the current revision's export directly. History, imports, review notes, references, materials, measurement provenance and detailed construction dimensions remain available through progressive disclosure.

The garment occupies the main desktop and mobile workspace. Six construction choices are immediately available under **Customize**; body measurements, fine dimensions and technical authoring are expandable. AI acceptance remains explicit. References are not automatically sent to the model, body fields remain excluded from model requests, and private body/reference export options remain opt-in.

## What generates the garment

`services/engine/garment_preview.py` generalizes the earlier example generator. Sleeve, cuff, opening, neckline, collar and torso guides use the actual generated panel edges and dimensions. It retains each physical shell/facing instance, original 2D rest coordinates, source interpolation weights and rest topology. Posed positions are computed by a bounded elastic/seam/guide solve; rendering adds lighting and fabric shading without changing those positions. Material and posing inputs are explicit uncalibrated assumptions, with their identities and residual diagnostics retained in the artifact.

The v2 private shape is bound to the immutable pattern bytes, complete construction inputs and pinned worker implementation. Server validation independently reconstructs source coordinates, checks handedness, inventory, triangle coverage, boundaries, hardware and finite display coordinates before atomic installation. The display also checks its artifact digest and source pattern identity. The private shape endpoint uses the existing authentication, project scoping, deletion fences and non-cacheable responses. Historical artifacts remain tied to their original revisions. Failed or cancelled previews do not automatically retry indefinitely; a new source generation or explicit retry is required. Pattern/export generation remains independently usable.

Original component shirts no longer require the unrelated external legacy drafting source at runtime. Exact Python dependency pins and worker resource controls still apply. An optional local Docker runtime runs the same workers without network access, as an unprivileged user with a read-only root filesystem, bounded CPU/memory/processes and private attempt mounts. It accepts only an immutable image digest; model credentials are not passed into the worker. Linux installations can retain their native pinned environment.

## Verification and limits

The real-worker HTTP integration exercises custom dimensions outside the four demo examples, then edits the same project to a shorter, buttonless, collarless shirt. Both pattern and preview identities change, original source correspondence validates, restart preserves the results, old revisions remain historical, and exported pattern JSON matches the selected revision. Unauthenticated and cross-project shape access fail; deletion removes access. The existing component-variant tests cover all six synthetic starting sizes.

A separate live-model UI smoke test used a fictional short-sleeved shirt description with no measurements, photos or wearer information. The connected model returned an editable proposal; accepting it with local synthetic sizing generated actual patterns and a garment. This is a single live integration observation, not a model-quality evaluation. Browser regressions otherwise use an explicitly deterministic model transport with real pattern/garment workers and isolated databases.

The browser flow verifies creation, review, generation, construction edits, source selection, revision-matched export, reload, mobile overflow, deletion, no-WebGL fallback and display recovery. Screenshots: [desktop](../verification/studio-e2e/desktop.png), [mobile](../verification/studio-e2e/mobile.png), [live-model desktop](../verification/studio-e2e/live-desktop.png) and [live-model mobile](../verification/studio-e2e/live-mobile.png). Counts and remaining environment gaps are recorded below.

The garment remains `guided-shape-approximation`, with `acceptedSimulation: false`. It is not calibrated drape, physical assembly, body fit or sewing validation. Guides can stretch the display mesh; the original pattern stays unchanged. Body/self-contact, interfacing and seam allowances are not simulated. The approximation disclosure retains deformation and seam residuals. Arbitrary garment support, manufacturing readiness, public multi-user hosting and deployment remain outside this checkpoint. No participant feedback or commercial-value finding is established.

## Final verification record

Verified in the isolated checkout with Bun 1.3.11, native Chrome and the pinned local Linux worker image listed below:

- Type checking and the production web build passed.
- Application tests: **104 passed, 0 failed** (1,009 assertions).
- Applicable component-shirt, printing, safety, inspection and end-to-end preview engine tests: **12 passed, 0 failed** (5,537 assertions).
- Applicable browser regression selection: **21 passed**. After the final sizing, mobile and project-link changes, the three garment/sizing browser cases passed again.
- The separate live-model smoke test generated a fictional shirt from a description without measurements or reference images. After a server restart, changing it to long sleeves generated a new garment, its downloaded pattern JSON matched the new revision, and its saved project reopened successfully.
- All four original demo artifacts passed the independent source/rest-geometry audit. Documentation checks and whitespace validation passed.

The full engine command was also run: **17 passed, 18 failed**. Nine failures require the absent native research `.venv/bin/python`; nine require the separately pinned external legacy drafting checkout, also absent here. The 21-case browser selection excludes five legacy drafting cases that need that external checkout. These are explicit coverage gaps; the entire historical suite is not green. This checkpoint verifies the original relaxed-shirt MVP path, not the legacy research garment families or arbitrary garments.

The complete studio was started on the original local port 5175 with a persistent private development database. No production service, production data or remote deployment was changed.

## Local launch

From the isolated checkout, configure the supported native Linux worker or the optional pinned container runtime, and the existing server-side model connection described in [deployment](../deployment.md#design-model-connection), then run:

```sh
bun run studio
```

The launcher binds only to `127.0.0.1`, defaults to port 5176, and retains its private development database at `.planning/local-studio` across restarts. `SEW_STUDIO_PORT` and `SEW_DATA_DIR` can override these local settings. The default development access key is `sew-local-demo`; `SEW_ACCESS_KEY` can replace it. This launcher is not a production deployment command. It neither resets production data nor copies credentials.

For Docker, set `SEW_ENGINE_DOCKER` to the absolute trusted CLI and `SEW_ENGINE_CONTAINER` to the full local `sha256:` image identity. `services/engine/Dockerfile.runtime` accepts an explicitly chosen Python 3.13 Linux base image and installs `requirements.lock`; it does not embed source assets, credentials or the legacy research checkout. The verified local image for this run is `sha256:0b847fbd80c0503cb9492c2f49ca2cee301ec0a28b33e939b5e87d6660c5536f`. Worker execution is offline. Legacy skirt/trouser/partial-top generation still requires the separately pinned Design2GarmentCode checkout; the original component-shirt compiler does not.
