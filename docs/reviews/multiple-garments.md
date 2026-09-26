# Multiple-garment MVP

**2026-09-26 · Isolated local implementation · Verified MVP scope · Not deployed**

The owner asked for more than one garment type and a simpler overall route to a result. This increment targets three editable families: relaxed woven shirts, relaxed woven dresses and elastic-waist skirts. These are starting capabilities, not a permanent preset catalog or a claim of arbitrary garment support. The earlier [shirt-only studio checkpoint](studio-e2e.md) remains historical evidence.

## Connected behavior

The home screen accepts a description directly. **Design with AI** explicitly sends that description, with provider disclosure next to the action; references remain excluded unless separately selected. The resulting proposal still requires review and acceptance. Optional illustrated starting shapes create an editable project using explicitly labeled synthetic M estimates and immediately generate its actual pattern and garment preview. No generated project reads the static demo fixtures.

Length, hem sweep and relevant construction choices sit alongside the garment. Detailed construction, sizing, references and technical authoring remain expandable. Required measurement entry follows the selected compiler: waist/hip for skirts and bust/hip/shoulder for the relaxed upper-body constructions. Unused unknown fields do not block them and are not replaced with fabricated worker inputs. Legacy drafting retains its original measurement requirements. Sample selection preserves entered values; changing an estimate does not relabel it as a measured value.

**Update garment** saves the input revision and generates both pattern and preview. **Download** prepares that saved revision's pattern/tech-pack files with private body fields and references excluded; optional disclosure controls remain available. Unsaved edits are identified in the download dialog. Existing autosave, conflict, deletion, history, feedback/import and privacy behavior remains part of the required regression scope.

## Component scope and provenance

| Family | Implemented construction | Explicit limits |
| --- | --- | --- |
| Shirt | Existing relaxed drop-shoulder torso, sleeve, cuff, collar, placket, tail and frill choices | No fitted darts, set-in sleeves, pockets or arbitrary topology |
| Dress | Continuous relaxed torso at dress lengths, straight/flared hem and compatible shirt components | No fitted waist, separate bodice/skirt seam, darts, princess seams or lining |
| Skirt | Four source panels, four waistband quarter templates cut for shell/facing, elastic casing, straight/flared hem and adjustable fullness | No zip, pockets, circular cutting, pleats or calibrated elastic response |

The dress compiler composes the tested upper-body construction and identifies itself as `sew-relaxed-dress/1`; the skirt compiler is `sew-elastic-skirt/1`. The skirt has eight cutting templates and twelve fabric instances. Its source seam graph joins quarter panels, the two waistband rings and the waist attachment; the physical graph also records each facing and the casing's upper edge. The elastic is declared trim, not an invented fabric panel. Source fabric circumference passes over the hips using an explicit drafting formula; the selected relaxed elastic circumference is a separate assumption requiring wearer/stretch checks.

Each compiler produces original 2D source edges, cut contours, grainlines, registrations, cut counts, assembly instructions and derived measurements. TypeScript independently checks the skirt's selected inventory, source dimensions, topology, attachment identities and derived measurements before installation or export. Preview generation retains source rest coordinates and interpolation weights for every physical instance. The skirt's gathered pose is driven by the source fabric and declared elastic target; rest pieces are never shortened to close seams. Dress guides read the generated torso and component dimensions. Shapes remain `guided-shape-approximation` with `acceptedSimulation: false`, diagnostic strain/seam residuals and no body/contact or fit acceptance.

The existing exact `requirements.lock` supports the new original compilers; no dependency pins, weights or external research source were changed. Worker source is mounted read-only into the previously verified pinned container, and the inspection implementation fingerprint includes the new compiler/assembly modules. The native Linux worker remains supported. Historical shirt inputs and artifacts are not rewritten or default-migrated. An application rollback must understand the newly saved dress/skirt construction blocks or retain them unchanged.

## Verification record

Type checking and the production build pass. **110 application tests pass** (1,093 assertions). **16 applicable engine tests pass** (5,589 assertions): custom, non-preset dress/skirt dimensions, independent source correspondence and corruption rejection, actual geometry changes after edits, all six synthetic sizes for each new family using only required measurements, existing shirt component variants, printing, worker limits and private durable previews. **All 31 browser tests pass** against isolated production servers: three-family creation/edit/automatic preview/download/reopen, required sizing fields, reviewed model fixtures, privacy, conflicts, imports, device fallback and mobile layouts. The autosave regression now retains a later user-selected tab, and downloads distinguish saved source revisions from newer autosaved edits.

PDF installation/export validation now parses object dictionaries, arrays and decoded names, including compressed object streams. It rejects active actions while allowing command-like bytes inside ordinary font/content streams. Regression cases retain malformed/active rejection. This fixes an observed false rejection of a normal generated shirt PDF. Both engine installation and export await validation before committing files.

Two live `gpt-6-astra` observations used fictional dress/skirt descriptions with no wearer data, body measurements, images or references. Each completed review, local synthetic sizing, actual pattern generation, garment preview and matching pattern JSON export through the browser. The dress proposal conservatively left the requested gentle flare unsupported and generated a straight dress; that limitation remained explicit, and the hem-sweep control supports an owner correction. This is integration evidence, not a claim that every model interpretation is correct. The browser regression transport remains explicitly deterministic and uses the real component worker.

The full historical engine command was also attempted. Eighteen legacy/research cases require prerequisites absent from this checkout: nine need native research `.venv/bin/python` and nine need the pinned external Design2GarmentCode source. That run also hit the five-second raster-print test timeout; its bounded timeout was corrected and both A4/Letter checks pass in the applicable engine rerun. The entire historical engine suite is not claimed green. No dependency/source pin was changed to hide these gaps.

Visual evidence: [home](../verification/multiple-garments/home-desktop.png), [dress](../verification/multiple-garments/dress-desktop.png), [skirt](../verification/multiple-garments/skirt-desktop.png), [mobile dress](../verification/multiple-garments/dress-mobile.png), [live-model dress](../verification/multiple-garments/live-dress-desktop.png) and [live-model skirt](../verification/multiple-garments/live-skirt-desktop.png). The starting cards are labeled illustrations. The project views render actual worker-generated garment positions. The local studio retains these synthetic projects across restarts.

Local setup follows the [studio launch instructions](studio-e2e.md#local-launch). The original component compilers require the matching current worker source and locked runtime, but do not require the external legacy research checkout. The verified container is `sha256:0b847fbd80c0503cb9492c2f49ca2cee301ec0a28b33e939b5e87d6660c5536f`. The updated studio runs on local port 5175 with its existing private development database; production is unchanged.

## Remaining product boundaries

The MVP enables feedback on these connected garment families. It does not establish demand, arbitrary garment support, calibrated drape, physical fit, sewing readiness, public multi-user hosting or deployment. Unsupported requests remain visible in the saved original brief and requirements. Further families should add trusted component, assembly, source validation and end-to-end coverage through this same workflow rather than adding disconnected demo meshes.
