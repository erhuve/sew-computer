# Draft tech-pack exporter

`buildExport(snapshot, geometry, assets): Promise<ExportResult>` in `index.ts` uses the current shared contracts. It accepts one saved revision and verified, snapshot-owned assets; it never reads an editor, database, engine output directory, network resource or later job. All input records and selected bytes are validated/copied synchronously before its first await.

## Delivery

- `tech-pack.pdf`: seven sections (which can span multiple pages), embedded licensed font, visible draft status, immutable revision number/ID, snapshot ID, manifest/input digest prefixes and `Page N of M` on every page. A4 media boxes are exactly `210 * 72 / 25.4` by `297 * 72 / 25.4` points; these are document dimensions, not print calibration.
- `manifest.json`: editable, ordinary JSON with `schemaVersion: 1`, `projectId`, `revisionId`, `snapshotId`, `manifestDigest`, revision/cutoff metadata and `sections` below.
- Opt-in original files named `pattern-<artifactId>.<json|svg|pdf>` and `reference-<assetId>.png`. Names are generated from validated IDs; original labels remain metadata. Pattern bytes and digests are unchanged. Pattern PDFs are inspected for actual media/crop boxes and rotation, never restamped or rescaled.
- `delivery.json`: snapshot/revision/project IDs, canonical manifest digest, and SHA-256/size/MIME for every delivered file except itself. The manifest digest is **SHA-256 of `canonical(manifest without manifestDigest)`**, not the checksum of formatted `manifest.json`. There is no recursive output checksum in the manifest or delivery inventory.

All outputs are draft-review/reference material, not cutting-ready patterns or fit/manufacturing approval. Missing finished measurements, technical flats, construction treatments and annotation/calibration evidence stay visibly missing. Pattern thumbnails are confined to the pattern inventory, not substituted for assembled-garment flats.

## Editable section shapes and import boundary

The backend must compare editable fields with its own stored export projection, not trust a submitted checksum, geometry, reviewer assertion or evidence status. A returned manifest is a proposal; it cannot overwrite a revision. Omitted/redacted fields are no-ops, not deletions.

| Section | Shape / source |
| --- | --- |
| `overview` | `{title, brief, sizeLabel, garment}` copied from the saved document. `garment` includes family, length/ease value states, flare and explicit subset acknowledgement. |
| `requirements` | Exact `doc.requirements` array, including original unsupported intent, status, note and source. |
| `materials` | Independent copy of `doc.bom.filter(category === 'fabric' || category === 'lining')`. |
| `bom` | Exact full `doc.bom` array. Materials/BOM edits that disagree require backend reconciliation; neither silently wins. |
| `bodyInputs` | Exact `doc.body`, **key absent** unless `includeBody` is true. |
| `finishedMeasurements` | Exact `doc.poms`, retaining named method, size, target/tolerance states, values, units and provenance. |
| `construction` | Exact `doc.construction` array. Assembly links, when disclosed, are reported in the PDF and original geometry JSON, not imported as authoritative construction evidence. |
| `patternInventory` | Read-only delivery context: `artifacts`, `referenceArtifacts`, `status`, `notice`, and front/back/detail `flats` statuses (`included`, `withheld`, `not_supplied`). Pattern metadata includes original artifact identity/digest/classification plus `deliveryFilename` and actual PDF `pages` where applicable. Disclosed geometry adds panel bounds/annotations and engine provenance/warnings; raw coordinates remain in the unchanged pattern JSON. `references` is present only when references are disclosed, with exact owner-entered labels/kinds/views/sources. |
| `review` | `{callouts: doc.callouts, comments: snapshot.comments}`. Callouts are editable proposals. Recorded comments remain revision-scoped, owner-recorded assertions; incoming reviewer identity/approval is not authoritative. |
| `exportDisclosure` | `{includeBody, includeReferences, includePatterns, omissions, classification, privacyNotice, pageFormat, textRendering}`. These describe this frozen delivery, not permission to acquire undisclosed data or upgrade evidence on import. |

The importer should allowlist intended document keys and validate resulting documents again; read-only envelope, artifact, rendering, disclosure and comment fields must never establish evidence. Values/arrays are not implicitly coerced, and geometry does not fill in body or garment POMs.

## Validation and privacy

`validation.ts` verifies the document/revision digest, ownership of every artifact/comment, duplicate IDs, exact supplied metadata, filenames, selected byte sizes/hashes, MIME/kind pairing, total asset bounds, and committed geometry input identity. Included geometry must match the sole committed pattern-JSON byte payload; units must be mm and reported panel bounds must match coordinates. Missing requested patterns/references fail, not silently disappear. Omitted assets do not require bytes and are never rendered.

PNG references must be backend-sanitized, bounded to 10 MiB / 20 million pixels, contain only structural/palette/transparency/density chunks, and fully decode with sharp. EXIF, text profiles and trailing payload are rejected. Only assets referenced by this revision are included. Technical-flat images appear only when `includeReferences` is true; source photos/sketches remain separately labeled files, never inferred flats. Static SVG validation is defense in depth for **trusted engine output**, not a general untrusted SVG sanitizer.

Default output excludes body fields/provenance, reference pixels/labels/sources, panel coordinates, pattern files, engine warnings and assembly links. Garment parameters, finished measurements and free text can still disclose body/personal information: this is not anonymization. No contact-record or raw-model-exchange field exists in this schema; free text is not PII-scrubbed.

## Font / Unicode

`assets/DejaVuSans.ttf` is unmodified DejaVu Sans 2.37 from Debian `fonts-dejavu-core`. The Bitstream Vera permission notice and public-domain DejaVu-change notice are in `assets/FONT-LICENSE.md`, also rendered as document text in every generated PDF containing the font. No PDF attachment or network font fetch occurs.

PDF display normalizes to NFC. Supported accents, Greek and Cyrillic remain readable. Unsupported glyphs, unsafe control/bidi characters and Arabic/Hebrew characters are deterministically displayed as `[U+XXXX]`, with the code-point list and policy in both PDF and manifest; this avoids silent loss/incorrect complex-script shaping. JSON preserves original strings exactly. This is not complete multilingual typography.

## Tests

Run against the real shared contracts:

```sh
bun test packages/tech-pack/tests/export.test.ts
```

Tests use pdf-lib plus `pdftotext` from Poppler for real extracted-text/page assertions, and sharp for actual PNG fixtures. They do not import API or engine workers. Synthetic polygon fixtures test **export semantics only**, not the garment generator or physical fit. Temporary files are created/removed under this package's test directory.

The original worker-contract mismatch is resolved. No contract preload or mocked shared schema is needed. Browser tests additionally exercise real exports through API validation and authenticated downloads.
