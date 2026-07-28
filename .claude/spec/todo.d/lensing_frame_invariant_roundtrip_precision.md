---
section: Backlog
---

- **Frame-invariant label round-trip costs ~3e-11 on near-fold configs**
  `[→ spec]` — WP2 (8h-d2) made the far-field label frame-INVARIANT by storing
  `E_tilde = E_minrel * exp(+1j*w*t_min)`, with `channels.reconstruct_farfield`
  undoing it via `exp(-1j*w*t_min)`. That round trip is not the identity in
  floating point: the error is `~eps * |w*t_min| * |E_tilde|`, and near a fold
  the label is LARGE (measured `max|E_tilde| = 2.55e5`, `max|w*t_min| = 13.66`),
  so the absolute error reaches `+2.9e-11`.

  Consequence: `MorseSignMaskTestCase.test_telescoping_holds_for_the_cusp_
  adjacent_mask` (`test_lensing_ppgo_bandsplit.py`) moved from `4.9e-12` on the
  retired direct min-relative path to `1.66e-11` against its `1e-11` bound. It
  is currently held by `@expectedFailure` with the bound kept VERBATIM — the
  tolerance was deliberately NOT weakened. `InteriorTelescopingTestCase`, a
  non-fold config through the SAME helper, still passes at `1.6e-16`, so the
  xfail is correctly scoped to the near-degenerate cusp regime.

  This is a real (small) precision cost bought for frame-invariance, not a
  fixture artifact. Two candidate resolutions, both needing a decision rather
  than a patch:
  1. Reconstruct in the min-relative frame on this path (the direct route
     measures `4.9e-12`), i.e. do not round-trip at all — a data-flow change.
  2. Accept the cost and re-derive the tolerance from the production serve
     precision, which is the honest bar if the round trip stays.

  Do NOT resolve by raising `1e-11` to make the test pass; that would hide the
  regression rather than price it. The large `|E_tilde|` near folds is itself
  worth a look — a demodulated label that grows to `1e5` is poorly conditioned
  exactly where the physics is hardest.
