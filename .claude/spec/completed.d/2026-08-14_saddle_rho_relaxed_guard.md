---
date: 2026-08-14
section: Lensing training
---

**Certified map's saddle rho<1 guard selectively relaxed per-cell (NEXT-
SESSION ORDER 5/7)** `[→ spec]` — build `saddle_rho_relaxation` closes
the `lensing_certified_map_guard_relaxation` backlog item.

That closed todo.d fragment (removed by this completion) asked to
re-validate and selectively relax
the certified map's saddle rho<1 guard. The 2026-08-14 driver pilot
re-validation (F080) found the three certified saddle rho<1 cells are NOT
equal: gamma [1.157, 1.339] is CLEAN (5/5 configs, sup err 8.7e-5), gamma
[1.339, 1.550] is MARGINAL (borderline, not activated), gamma [1.100,
1.157] is CONTAMINATED (3.5 orders over bar at a corner). This build
implemented exactly the scope the fragment specified: per-cell relaxation
keyed on re-validation evidence, not a blanket parity x rho predicate.

`CertifiedPpgoMap._saddle_rho_relaxed_floor` (`cogwheel/lensing/
ppgo_map.py`) is now the sole authority for the saddle rho<1 serve/refuse
decision, matching the query cell against the hardcoded evidence-keyed
`_SADDLE_RHO_RELAXED_CELLS` allowlist by exact gamma/rho edge equality (a
re-grid whose edges no longer match falls back to UNKNOWN, never
mis-serves). Only the CLEAN cell is active in the allowlist today (floor
19.164, w_trust 28.746); the MARGINAL cell's recipe is present but
commented out (documentation-only, not activated); the CONTAMINATED cell
stays refused. `w_cert`, `w_trust`, and `w_ceiling` all route through the
same allowlist for mutual consistency. The two now-redundant duplicate
pre-guards — `LensedRelativeBinningLikelihood._ppgo_cell_coords` (SITE 1)
and `surrogate_census.characterize_sample` (SITE 4) — were deleted since
the map itself owns the decision.

Professor review (in-build): 13/13 relaxed-cell pins pass; independent
first-principles check of w_cert/w_trust/w_ceiling values, F073
preservation on off-band/neighbor cells, and served==counted census/
likelihood agreement, all PASS. Did not address the F080 fan-asymmetry
question (mirrored fan angles 2.4x apart under exact D2) — that stays
open, tracked in `[[lensing_training_campaign]]` (the 7a retrain), not
here. The MARGINAL and CONTAMINATED cells also stay open there pending
the edge-biased retrain re-measurement.

Deferred Inspector finding INS-1-003 (DATA_CONTRACTS.yaml/SPEC.md still
described the old blanket F073 refusal) resolved by Librarian doc-sync in
the same session: updated `certified_ppgo_map`'s DATA_CONTRACTS.yaml
description and consumer list, the SPEC.md `ppgo_map.py` module-row
clause, and the stale `CONSUMER_GRAPH.json` cache entry.
