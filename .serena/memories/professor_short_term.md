# Professor session 2026-08-07 — Ghost gate and ppGO fallback design consultation

## Ghost Airy/CFU approach verdict: INCORRECT PHYSICS

The ghost is a complex saddle (Picard-Lefschetz thimble), never a coalescing real pair. It does not undergo a fold bifurcation — its own self-merge is an artifact of the quartic continuation, caught by `_GHOST_DET_FLOOR`. The two existing gates (separation and decay) guard orthogonal failure modes: near-cusp coalescence (separation) and near-axis non-decay (decay gate). F027 measured the decay gate's retirement as a regression with 1000x degradation on the positive-parity near-axis. An Airy/CFU form would solve a problem that doesn't exist while leaving the real problem unaddressed and introducing an unverified new analytic form.

Recommendation: keep both gates, verify all three call sites (train label, Born carrier, serve mirror) route through `farfield_ghost_term` with both gates live.

## ppGO above w=150 verdict: ALREADY IMPLEMENTED

The geometric serve above `W_CEILING_SCHWINGER_QD = 150` exists in `_positive_parity_grid` / `_saddle_grid` via `select_branch → geometric_amplification`. The η ≥ 0.3 gate in `select_branch` is measured (F031: four orders of magnitude improvement, p90 117% → 7.65e-5). ppGO error is O(1/w²) at fixed η, with the caveat that at η < 0.1, error is flat in w — the η gate IS the accuracy guarantee. At w > 150 and η ≥ 0.3, the resolution leg always passes (w*delta_min ≥ ~82 >> 4), so select_branch always returns 'geometric'. No new code path needed for correctness. A surrogate-level intercept (fact-5 slot) would be a marginal performance optimization only, since the Born rung already serves rho > 1 and the interior fold-ppgo handoff already exists.

## Source files consulted
- `channels.py`: ghost term, both gates at lines 1101-1114, born_carrier_from_partition
- `operator.py`: select_branch (L1630-1634), ETA_MIN_GEOMETRIC=0.3, W_CEILING_SCHWINGER_QD=150, _positive_parity_grid/_saddle_grid above-ceiling routing
- `_airy_fold.py`: fold_amplification with eta < _ETA_MAX_FOLD gate (L420), q=0 symmetric-fold assumption (L267), F028/F032 confirmations
- FINDINGS.md: F027 (decay gate regression), F028 (Airy arm O(1) wrong), F029 (geometric error controlled by η), F031 (L_MAX re-derivation + η gate measurement), F032 (GLoW confirms F028), F033 (cubic normal form truncation, not q=0)
