---
date: 2026-08-12
section: Backlog
---

### Two SPEC.md claims corrected against the code

Closes `todo.d/lensing_spec_contract_divergences_found_by_librarian.md`.
Both were spec staleness, not code defects; no code changed.

1. `_to_caustic_fixed` is directional-MULTIPLICATIVE on the astroid interior
   arm ONLY. Every exterior arm, both parities, is additive
   (`rho = 1 + |y| - reach`), and the macro saddle uses the additive scalar
   form for every source regardless of side. SPEC.md had claimed the astroid
   exterior arm and a "saddle interior arm" were multiplicative;
   DATA_CONTRACTS.yaml already stated it correctly, so the two canonical
   surfaces disagreed.
2. `LobeInteriorChart.theta_to_u` is OPTIONAL (soft `data.get`, `None` on the
   raw-theta fallback), as are `lobe_exterior` and `exterior_polar`. ONLY the
   WEDGE (v3) hard-requires its map. SPEC.md had said the lobe loader read it
   unconditionally and hard-refused.

Method note worth keeping: each was settled by reading the CODE, not by
picking whichever document sounded more confident. On (1) the two surfaces
contradicted each other, so a doc-vs-doc comparison could not have resolved
it; on (2) both surfaces agreed with each other and were both wrong.
