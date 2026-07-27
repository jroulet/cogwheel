# Build 8h-d1: re-key the ghost gate on saddle separation

## Mission

The mid-band ghost gate keys on whether the ghost has DECAYED
(`w_min * Im tau_c >= RHO_END/2`). The quantity that actually decides whether
a single-saddle expansion is valid is whether the saddles are SEPARATED.
Those coincide away from caustics and diverge near them, so the current gate is
wrong in BOTH directions: it refuses across the band where the ghost is worth
5x-155x, and it admits near-principal-axis configs where the (correctly framed)
ghost makes the stored label up to 10.7x WORSE.

Re-key it, and fix the train/serve skew in the same stroke.

## Measured facts (driver-measured 2026-07-27; do NOT re-derive)

Probes: positive parity, `beta = kappa = 0`, source at
`|y| = r_caustic(gamma, theta) + offset`, ghost carried in the min-subtracted
frame (the 8h-b7 fix, already landed).

1. WHERE IT REFUSES BUT SHOULDN'T. With the ghost always included, measured
   accuracy gain (`|R|/|R-G|`) at `gamma=0.50, theta=45deg, off=0.60`: 1.7x at
   `w=0.5`, 33x at `w=9.4`. At `gamma=0.90` same angle: 2.1x at `w=0.5`, 105x
   at `w=9.4`. At `gamma=0.30, off=0.25`: 12x at `w=0.5`, 155x at `w=25`.
   The gate admits only `w >= 4.9 / 6.3 / 14.9` respectively.
2. WHERE IT ADMITS BUT SHOULDN'T. Sweeping 34 gate-passing exterior configs,
   7 have the frame-corrected ghost-subtracted label WORSE than subtracting
   nothing: `gamma=0.30, theta_c=85deg, off=1.0` gives fixed 4.31e-2 vs
   un-subtracted 4.03e-3 (10.7x worse); also `theta_c=15deg` at
   `gamma = 0.50, 0.70, 0.90`.
3. WHY: cusp coalescence. `min|x_real - x_ghost|` falls 1.33 -> 0.24 as
   `theta -> 0` (a cusp ray), while `Im tau_c -> 0.001`. So near a cusp the
   ghost is UNDECAYED (gate admits) but INSEPARABLE from a real image
   (expansion invalid). Decay and separation disagree exactly here.
4. TRAIN/SERVE SKEW. The gate reads `min(w)` of the array the CALLER passes.
   Training passes the chart's node grid; the serve mirror passes the query's
   chart sub-band, which containment makes >= the chart's minimum. The serve
   gate is therefore systematically WEAKER: it can re-add a ghost the training
   label never subtracted. Latent only because no `MINUS_GHOST` chart is
   trained yet; 8h-b7 is what arms it.
5. HIGH-w FLOOR (real, small): at `gamma=0.90, w=40.9` the bare-ppGO residual
   is 1.9e-15 and adding the ghost degrades it to 5.2e-7. Harmless at serving
   tolerance but it means "always on" is not free at the top.
6. SEAM PAYOFF. With frame fix + always-on, the residual crosses 1% at
   `w ~ 2.2` (gamma=0.90), `~4` (0.50), `~7-9` (0.30), against `w_floor` of
   0.20 / 0.69 / 1.54 -- versus `w_trust ~ 20-60` for bare ppGO.
7. PICARD-LEFSCHETZ CONSTRAINT: only the DECAYING complex member is
   admissible. The growing conjugate (`Im tau < 0`) is never included --
   `exp(+w*Im tau)` diverges. "Include all four roots" is wrong; the rule is
   2 real + the decaying ghost.
8. The exact oracle refuses above `W_CEILING_SCHWINGER = 60`, so every
   falsifiable comparison lives below it.

## In scope

- Replace the gate's decay criterion with a SEPARATION criterion. The
  Professor owns its exact form (candidates: `min|x_real - x_ghost|` against a
  scale, or `|det H|` / the fold-parameter the uniform arm already uses) and
  owns its threshold. It MUST refuse the fact-2 configs and admit the fact-1
  band; both are falsifiable against the measurements above.
- Make the admission decision a property of the CHART/configuration, not of
  the caller's `w` array (fact 4). Either pass the gating frequency explicitly
  from the chart's `log_w_grid.min()` on both sides, or persist the per-node
  admit/refuse decision in the chart meta and have the serve mirror read it.
  Train and serve must provably agree.
- Keep the high-`w` behaviour honest (fact 5): document it, and if a taper or
  ceiling is introduced, measure it rather than asserting it.
- Fast tests: the gate admits/refuses at the measured boundaries; train and
  serve agree on the same configuration; reachable-red for each.

## Also in scope (small, same files)

- `beta` guard: `likelihood.py::_surrogate_coefficients` de-rotates the source
  position into the eigenframe but passes `theta` (the caustic angle)
  UN-rotated, so `theta = theta_eig + beta` exactly. Latent (production pins
  `beta = 0.0`) but silently wrong if enabled. Mirror the existing `kappa`
  guard one line above it. ~1 line.
- `channels.real_image_delays` re-derives the min-subtracted frame instead of
  routing through `_frame_delays` -- a FOURTH site of the convention 8h-b7
  single-sourced. ~3 lines.
- `ChangRefsdalGeometryPartition.caustic_theta` docstring calls it a
  "polar angle"; it is the LENS-plane critical-curve parameter, not the
  source-plane `theta_c`. Prose only.
- `test_lensing_levers.py` lever-5 wave-vs-geometric comparison passes
  `beta = kappa = 0`, which makes both gauge factors that could drift inert.
  Add ONE `beta != 0, kappa != 0` point. ~2 lines.

## Out of scope (do NOT touch)

- Born (`_born.py`), its `b1` coefficient, its wiring.
- Tiling: `ppgo_exclusion_rho`, the `rho` gauge converter, far-field carrier
  continuity, cusp-aligned column wiring. These are the NEXT build.
- The census, any campaign, sweep or engine production run.
- The uniform Airy/Pearcey arms themselves. This build decides WHEN to hand
  over to them; it does not change them.
- Structural test classes in `test_lensing_surrogate_training.py` /
  `test_lensing_farfield_envelope.py` -- both files carry a PROVISIONAL header
  saying not to contort production to keep their bookkeeping green.

## Acceptance

- The re-keyed gate REFUSES all seven fact-2 configs and ADMITS the fact-1
  band, demonstrated against the measured numbers.
- Train and serve reach the same admit/refuse decision for a fixed
  configuration, asserted directly (not via a magnitude proxy).
- `FARFIELD_KERNEL_SUM_MINUS_GHOST` residual never exceeds the un-subtracted
  `FARFIELD_KERNEL_SUM` residual on any admitted configuration. This is the
  do-nothing control; it is what the old gate failed.
- Existing suites green: `test_lensing_chang_refsdal_ghost_frame.py`,
  `test_lensing_exterior_windows.py`, `test_lensing_channels.py`,
  `test_lensing_ppgo_bandsplit.py`.
- Full suite: driver-verified POST-BUILD.

## Constraints

- HARD test-tier ceiling: any single test < 60 s, any FILE < 5 min, fast tier.
  No engine-backed training fixtures -- `COGWHEEL_TRAIN_TIER` classes are the
  driver's, not a build's. The gate runs your tests AGAIN.
- Every oracle comparison below `w = 60` (fact 8).
- Any new convention gets ONE authoritative expression plus an assertion that
  its consumers agree. 8h-b7 existed because a convention lived implicitly at
  four sites; do not add a fifth.
- Accuracy dominates; units and conventions per AGENTS.md; numba compatibility
  preserved.
- Branch `claude-dev` only. Never commit on `main`.
