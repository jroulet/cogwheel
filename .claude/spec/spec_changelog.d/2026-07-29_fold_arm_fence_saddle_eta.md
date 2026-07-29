---
date: 2026-07-29
bump: minor
---

### Fold arm gains a caustic-relative admission fence; the macro-saddle eta leg is live too

SPEC.md described the uniform fold Airy arm (F028: measured 60%-267% wrong on
well-resolved above-ceiling configs) with no fix in place. It now has one:
`_airy_fold.fold_amplification` refuses (returns `None`, falling through)
outside `eta < _ETA_MAX_FOLD = 0.3` — the complement of
`operator.ETA_MIN_GEOMETRIC`. F032 independently confirmed the unfenced arm
63%-64% wrong against GLoW; F033 traced the residual to the cubic normal
form's own `O(eta)` truncation rather than the `q = 0` symmetric-fold
assumption, so the fence is the permanent treatment — no amplitude
refinement (`b4`) can recover the far-from-caustic region. The threshold
itself is not yet tight: F033 measured the arm still off by 14%-29% at
`eta = 0.3`; tightening it further is open work
(`todo.d/lensing_fold_arm_serves_wrong_values.md`).

Also corrected: SPEC.md said the macro-saddle branch of `select_branch`
passes `eta = inf` (the eta leg switched off), leaving "whether the saddle
needs its own eta floor" as an open question. F034 answered it — the `inf`
default was measured NOT safe (p90 8.95e-1 for `eta < 0.3`, worst case 484x
over 15% of resolved draws, worse than the positive-parity band F031
measured). `_saddle_grid` now measures `eta` via `nearest_caustic_point`
once per grid and passes it through `select_branch`; the eta leg is live on
both parities, each independently measured.
