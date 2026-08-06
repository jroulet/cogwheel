---
section: Backlog
---

- **THE FAR-FIELD `(s, d)` COORDINATE IS DEGENERATE OVER MOST OF THE DOMAIN IT
  CHARTS — the nearest caustic foot becomes a coin flip** `[→ spec]` —
  measured 2026-08-06 (pure geometry, `gamma = 0.495`, caustic reach 1.3931).

  `(s, d)` is a tubular-neighbourhood coordinate: nearest point on the caustic
  plus signed perpendicular distance. It is a diffeomorphism only while the
  foot is UNIQUE. Measured `tie_ratio` = (2nd nearest foot) / (nearest):

      DIAGONAL ray (phi = 45 deg)          generic ray (phi = 20 deg)
      |d|    |y|   feet  tie_ratio         |d|    tie_ratio
      0.05   0.55   4     10.58            0.05    6.19
      0.20   0.70   4      2.88            0.20    1.54
      0.50   1.00   4      1.46            0.50    3.06
      0.75   1.25   3      1.30            1.00    1.92
      1.00   1.50   4      1.13            1.25    1.000  <-- exact 4-way tie
      1.25   1.75   4      1.04

  The exterior charts in the 2026-08-05 run spanned `|d| ~ 0.07 .. 1.22`, so
  the tie region is INSIDE the charted domain, not beyond it. Where
  `tie_ratio -> 1`, `s` is decided by numerical noise and jumps
  discontinuously as the source moves. The foot also swings up to
  0.042 rad per degree of source rotation against an ideal 0.0175 — `s`
  AMPLIFIES position error ~2.4x.

  ## This explains the subdivision blowup

  Band 0's exterior needed 57 charts from 21 base tiles: **84% exist only
  because subdivision was forced**, and 12 of 21 base tiles (57%) failed the
  eps bar. The children sit FARTHER out than the survivors (median nearest
  edge `|d|` 0.494 vs 0.371) — the opposite of what a near-caustic
  diffraction-layer explanation predicts, and exactly what foot degeneracy
  predicts.

  Subdivision cannot fix it. Halving a tile that straddles a foot-tie yields
  two tiles that each straddle it. Every extra chart buys almost nothing,
  which is why the exterior costs 39.4 min/band against the interior's 1.8.

  ## The physics

  Far from a small closed curve the caustic subtends a shrinking angle and is
  effectively a POINT; "perpendicular to the caustic" carries no local
  information. The natural far-field coordinates are origin-centred polar
  `(|y|, theta)`. Caustic-normal `(s, d)` is correct only in a thin tube
  hugging the curve — which is what `TubeChart` already owns. The astroid's
  exterior MEDIAL AXIS (the diagonal rays between adjacent cusps) is where
  feet tie exactly, and `_FARFIELD_MEDIAL_AXIS_TOL` already acknowledges the
  problem pointwise without addressing the surrounding ill-conditioning.

  ## Work

  - Re-chart the true far field in origin-centred polar coordinates, leaving
    `(s, d)` to a thin near-caustic band. This subsumes the deferred rename in
    [[lensing_farfield_name_spans_three_regimes]]: the class is not a
    far-field object, and the fix is a coordinate split, not a rename.
  - Determine the crossover `|d|` where `(s, d)` stops paying. `tie_ratio`
    falls below ~2 by `|d| ~ 0.35` on the diagonal, so the usable tube is
    thinner than the current tiling assumes.
  - SEPARATE, UNTESTED (owner's hypothesis, worth its own measurement): the
    far-field label subtracts real-image kernels assuming ONE DOMINANT IMAGE.
    If two real images stay comparable out to larger `|y|` than assumed, the
    residual BEATS between them and is oscillatory rather than smooth. That
    would degrade convergence even where the foot is unique, so it is
    distinguishable from foot degeneracy: measure the image-magnification
    ratio versus `|y|`, and check whether poor-convergence tiles correlate
    with ratio ~ 1 rather than with `tie_ratio ~ 1`.

  ACCEPTANCE: exterior charts per band drop well below 57 at the SAME eps bar
  (1e-3, F-normalised), and a served value at a former foot-tie location
  matches a fresh engine evaluation to tolerance.
