---
bump: patch
---

Librarian post-commit re-sync (backlog c0d17a8 + 1805bfd), closing the third
and final deferral on the Microlensing-engine row:

- INTERIOR CUSP SERVING rewritten present-tense per F074 (commit c0d17a8):
  the deleted `interior_degenerate` bypass and the `radius >= radius_min`
  gate are replaced by the corrected control map (odd control = soft-axis
  projection, even control = hard-axis projection times the manifold
  curvature `phi_ssr/(2 lambda_h)`), the served-error gate
  `_K_UNIFORM/sqrt(w) + ghost term <= envelope_bar`, and the calibration
  certificate now enforced on every node. Near-cusp interior now serves from
  `w >= ~49`.
- SERVING LADDER description expanded to spell out the internal
  uniform-arm order `fold -> ppGO+ghost -> cusp`; a new paragraph documents
  F075 (fold arm refuses non-4-image censuses at all three sites it used to
  touch — `fold_amplification`, `fold_ppgo_correction`,
  `channels.born_carrier_from_partition`) and the new
  `operator._ghost_ppgo_amplification` rung's gates
  (`geometry._GHOST_DECAY_IM_THRESHOLD = 0.4`,
  `geometry._GHOST_SEPARATION_MIN = 0.7`, single-sourced in `geometry.py`)
  and acceptance (max served rel-err 1.977e-06 vs the 1e-2 bar).
- The PARITY-GATED paragraph's `surrogate_census.characterize_sample`
  sentence, previously citing the retired `xi_min`-based mirror and the
  open `todo.d/lensing_census_mirror_regate`, corrected to the current
  re-gated predicate (build `fold_exterior_ghost` WP-3, closes that item).

See `completed.d/2026-08-13_fold_exterior_ghost.md` and
`completed.d/2026-08-13_lensing_census_mirror_regate.md`.
