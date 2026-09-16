# Sew Computer: product intent

**2026-09-14 · Product direction, not an implementation claim**

Enable anyone to design an original garment, understand and edit the result, and obtain connected patterns and a reviewable tech pack. A person may sew it themselves, work with a local maker, or eventually organize a production run. Making a garment is the goal; factory procurement is an optional route.

## Decisions to preserve

- **Build the software first.** Its output becomes the concrete deliverable a maker evaluates. Recruiting a sample maker, choosing one test garment or completing audience interviews is not a prerequisite to starting implementation.
- **Original designs, not only decoration on blanks.** Garment families and existing components are useful implementation primitives and optional starting points. They are not a permanent product restriction to a preset catalog.
- **Do not silently change the idea to fit the engine.** Preserve requested details that are unsupported or ambiguous, identify the mismatch and provide an extension/review path. A plausible substitute is not a successful translation.
- **Do not make manufacturing or paid review the admission price.** Creation, sharing and DIY access must not depend on commissioning a factory or paying a professional. Resource budgets and business pricing remain separate, unresolved decisions; this does not promise unlimited free inference.
- **Visual results first.** The garment or real pattern belongs above the fold; explanations and secondary controls use progressive disclosure. Keep essential actions accessible and readable on desktop and mobile.
- **Connected technical outputs.** Design intent, program parameters, body/material inputs, generated geometry and the tech pack must refer to the same immutable revision. An attractive image and an unrelated pattern are not a working product.
- **Pattern-derived garment 3D.** Construct every fabric component from its actual 2D pattern, expanded into the required physical pieces, with explicit assembly and material behavior. Placement may change; rest geometry cannot be silently resized to make assembly succeed. AI may propose structured construction instructions, but trusted geometry and simulation engines execute them. A concept image or independently generated mesh cannot stand in for this view. This is an architectural requirement for future work, not a delivered simulation or fit claim.
- **Honest evidence.** Distinguish a concept illustration, generated geometry, simulation, maker review and physical fit/sample evidence. None implies the others. Do not invent a production-ready badge.
- **Private inputs.** Body measurements and references are private by default. Do not infer sex, gender or measurements from an image as though they were known inputs.

## First useful software loop

Describe or sketch a garment → review the interpretation and unresolved decisions → generate supported geometry → inspect and edit the design/pattern → save a revision → export a draft tech pack and clearly classified pattern artifacts → incorporate maker feedback as another revision.

Support this end to end before adding a marketplace. The first slice may have a limited, accurately declared geometry capability. Broader custom design remains the direction, with unsupported requests retained as requirements rather than erased.

## Audiences to learn from without redefining the product

Individuals with a garment idea, DIY sewing hobbyists, cosplayers, independent designers/studios, underserved fit needs, and coordinated groups. Group use cases include companies, bands, school organizations, communities, performance groups and weddings. Wedding parties may want coordinated materials/colors across different garments, not identical outfits.

These are research hypotheses, not proven demand or equivalent production problems. Software development can proceed alongside research. Do not treat a merchant workflow or a bridal deadline as the default experience for every person.

## Outside the first prototype

Factory bidding, payments, crowdfunding, group-order thresholds, shipping, a public design marketplace, production guarantees, and arbitrary untrusted code execution. These can follow a trustworthy creation/export loop; they are not dependencies of it.

The active execution plan is [Software prototype](../plans/software-prototype.md). The upstream investigation is summarized in [Design2GarmentCode evidence](../research/design2garmentcode-evidence.md).
