# Sew Computer

- This repository contains a private manual/CPU prototype: React editor, authenticated SQLite API, trusted CPU pattern adapter and revisioned PDF/JSON exports. AI interpretation, simulation, calibrated cutting claims and public multi-user hosting remain outside this release.
- Read `README.md`, `docs/product/vision.md` and the one active plan, `docs/plans/software-prototype.md`, before implementation.
- `docs/design/project-contract.md` is normative for shared state, security, units, evidence and exports. The other `docs/design/` files are supporting designs, not competing active plans. Resolve disagreement by updating the contract and every affected design together.
- Read `docs/reviews/planning-adversarial.md` for resolved findings and required regression cases. Planning review does not certify runtime or physical garment correctness.
- Preserve software-first sequencing and open creative intent. Do not gate starting development on a maker appointment or force the product into a permanent preset catalog. Preserve unsupported requests explicitly.
- Keep creation, sharing and DIY access independent of manufacturing purchases or paid professional review. This is not a promise of unlimited free model inference.
- Prioritize useful visuals above the fold with progressive disclosure, responsive layouts and accessibility. Distinguish illustrations, actual geometry and simulation.
- Do not infer body measurements or sex/gender from reference photos. Inputs and body-derived geometry are private by default.
- Design2GarmentCode is an external research dependency. `docs/research/design2garmentcode-evidence.md` pins the inspected baseline and limits. Do not silently update it, claim full AI reproduction from the CPU spike, or vendor uncleared assets/weights.
- Sew Computer is separate from Stylr. Do not modify, test-build or deploy any Stylr working directory for this project.
- Run `python3 scripts/check_docs.py` and `git diff --check` for documentation changes. The intended module layout is declared in the active plan; do not create a conflicting layout from a specialist draft.
- `.planning/` is ignored scratch material and not part of the deliverable. Keep tokens, raw agent responses, private references and local model/runtime artifacts out of Git.
- Run `bun run typecheck`, `bun run test`, `bun run build`, `bun run test:engine` and `bun run test:browser` for the applicable changes. Browser tests use an isolated production server/database and synthetic fixtures. Read the implementation review for the verified scope and remaining gates.
