---
bump: patch
---

Librarian re-sync (INS-1-001, deferred by build ppgo_interior_certificate):
the Microlensing-engine row's FOLD-PPGO INTERIOR HANDOFF passage now
describes the shipped gate in present tense — exact 4-image predicate
(`int(geom.real_mask.sum()) == 4`), the `geometry.ppgo_error_estimate` c3
certificate at `_PPGO_INTERIOR_SAFETY = 2.0`, and the `xi_min >= 4` leg
dropped by measurement — citing `completed.d/2026-08-13_ppgo_interior_certificate.md`
instead of the old xi/uniform-error-estimate gate. Removed the 2026-08-13
stopgap sentence ("GATE REPLACED ... pending Librarian re-sync"). Also
corrected the PARITY-GATED paragraph's census-mirror claim: it previously
said `surrogate_census.characterize_sample` "mirrors the same gate" — it
still mirrors the PRIOR xi-based gate (classification skew only, tracked in
`todo.d/lensing_census_mirror_regate`, Inspector INS-2-001, accepted).

The row's INTERIOR CUSP SERVING passage (the `interior_degenerate` bypass
and the `radius >= radius_min` uniform-error gate) is NOT touched by this
fragment — commit c0d17a8 (F074, Pearcey control-map fix) landed on
`_pearcey_cusp.py` after this backlog's sync scope (a20575e..d3dc109) was
fixed and is left for the next post-commit sync.
