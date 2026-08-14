---
date: 2026-08-13
section: Lensing
---

- **Exterior ppGO+ghost rung; fold arm refuses non-merging pairs** (build
  `fold_exterior_ghost`, commit 1805bfd; driver-committed after Inspector PASS
  and Professor PASS — the build stranded at the tree gate on stale pins of
  the very behavior it fixed). Closes F075 (the fold arm was serving an Airy
  merging correction to the wrong image pair outside the caustic, at
  15-216% error for `60 < w <= 150`).

  - **WP-1**: `_airy_fold.fold_amplification`, `_airy_fold.fold_ppgo_correction`,
    and the born-carrier fold block in `channels.born_carrier_from_partition`
    (the label-path co-bug) each refuse the Airy merging correction when the
    real-image census is not 4 (`len(images) != 4`) — three sites, one
    predicate. Interior four-image behavior is byte-identical at all three.
  - **WP-2**: new rung `operator._ghost_ppgo_amplification`, inserted between
    the fold and cusp arms in `_uniform_arm_value` (order: fold -> ppGO+ghost
    -> cusp). Serves `geometric_amplification + ghost.kernel*exp(1j*w*tau_c)`
    in the absolute frame, admitted by two frequency-independent
    configuration gates single-sourced in `geometry.py`
    (`_GHOST_DECAY_IM_THRESHOLD = 0.4` on `Im(tau_c)`, `_GHOST_SEPARATION_MIN
    = 0.7` on `min_a |x_a - x_c|`; `channels.py` binds both from `geometry`
    rather than duplicating the literals). `geometry.GhostAbsentError`
    (interior) declines; `geometry.GhostDomainError` (ghost present but
    undecayed, e.g. on a principal axis) refuses. Acceptance sweep (WP-5,
    `.claude/handoff/wp5_probe_p2_report.md`): 45 oracle points, 0
    gate/serve skew, max served rel-err 1.977e-06 vs the 1e-2 bar (~4 orders
    of margin) on `|y|/rc >= 1.15`, `w in (60, 150]` — the exact band the
    fold arm previously shipped 15-216% wrong on. The caveat band
    `|y|/rc = 1.05` refuses exactly where `Im tau_c` has not yet decayed,
    falling through to the exact engine. Frequency-independent by design
    (Professor's ruling, confirmed by the sweep): a `w`-dependent floor
    would re-open the F070 train/serve skew and break the rung's role as a
    residual-tabulation label oracle.
  - **WP-3**: training-side census mirror re-gated —
    `surrogate_census.characterize_sample`'s `ppgo_fold` classification now
    requires the same 4-image census and the c3 certificate
    (`geometry.ppgo_error_estimate(...) * _PPGO_INTERIOR_SAFETY <=
    CERTIFICATION_BAR`, bound from `likelihood`/`geometry`, not re-typed),
    replacing the retired `xi_min`-based mirror. **Closes
    [[2026-08-13_lensing_census_mirror_regate]]** (Inspector INS-2-001).
  - **WP-4** (report only, `.claude/handoff/wp4_label_contamination_report.md`):
    retroactive label-contamination check on the two shipped training
    artifacts. `certified_ppgo_map.npz` is CONTAMINATED at 32 positive-parity
    exterior cells (labels drawn through the fold arm at
    `w in {66.05, 79.91, 96.68, 116.96, 141.50}`) — direction is
    over-conservative (floors pushed up, coverage loss, never
    over-certification). `born_residual_chart.npz` is CLEAN (its training
    grid tops out at `w = 60`, never entering the contaminated band).
    Retraining `certified_ppgo_map.npz` against the WP-1-corrected
    `exact_total` is flagged for a future training-campaign build; not
    performed here (no artifact edits in this build).

  Tests: `cogwheel/tests/test_lensing_fold_ghost_exterior.py` (new, 654
  lines — fold census refusal at all three sites, ppGO+ghost rung gates,
  ghost sign pin, interior byte-identity) plus second-wave stale-pin
  rewrites in `test_lensing_cusp_arm_coverage.py` (serve band re-derived
  around the F074 gate: upward-closed, contiguous, vertex refusal to
  `w=1000`, plus a NEW `F_op`-overlap oracle window `w in [49, 60]` where
  the DD engine checks the arm directly, worst 0.0473 vs bar 0.05) and
  `test_lensing_schwinger.py` (three-outcome fixtures re-derived interior).
  Files F078 (xdist loadfile reshuffling exposed a latent
  `_pearcey_cusp` module-global leak between two test files, fixed with
  `setUpModule`/`tearDownModule` save-restore; not caused by this build's
  edits). Full fast gate: ALL GREEN, 2267 collected (parallel rc=0, timing
  rc=0). Pre-existing flakes classified and left untouched (`test_waveform`
  unseeded RNG, `test_gw_prior` load-contention timeout).

  SPEC.md Microlensing-engine row re-synced by the Librarian post-commit
  pass (2026-08-14): SERVING LADDER description now spells out the internal
  uniform-arm order (`fold -> ppGO+ghost -> cusp`); a new paragraph
  documents the fold's three-site refusal and the ppGO+ghost rung's gates
  and acceptance numbers; the PARITY-GATED paragraph's stale
  `surrogate_census` claim (still citing the retired `xi_min` mirror) is
  corrected to the current re-gated predicate.
