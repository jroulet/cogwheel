---
section: Backlog
---

- **THE ASTROID CENTRE IS UNSERVED — the wedge carrier flips basin along the
  GAMMA axis at small `r`** `[→ spec]` — measured 2026-08-06 against the
  shipped wedge path (`034fcf7`).

  Building band 0's interior (`gamma in (0.475, 0.515)`, parity `+1`) with the
  production config gives 5 wedge tiles, of which the INNERMOST fails:

      tile 1/5  r=0.099  theta=pi/4   CarrierDiscontinuityError
      tile 2/5  r=0.277               OK, 343 pts, 0 refused
      tile 3/5  r=0.455               OK, 343 pts, 0 refused
      tile 4/5  r=0.633               OK, 343 pts, 0 refused
      tile 5/5  r=0.811               OK, 343 pts, 0 refused

  `_assert_carrier_continuity` reports the flip "along axis 0". Axis 0 is
  **gamma**, not `w` and not angle: the guard receives `critical_sources` with
  shape `(n_gamma, n_s, n_d, 2)` and loops `axis in range(3)` over
  `(gamma, s, d)` (`surrogate.py:1829`). The chart's SPLINE axes are
  `(log_w, gamma, r, theta_wedge)`, so reading "axis 0" as `log_w` is the easy
  mistake — the guard's array is a different object with no `w` axis at all,
  because the parked critical point is geometric.

  So: near the astroid centre the four images are near-degenerate, and which
  one is the parked critical carrier `tau_c` HOPS BASIN partway through the
  gamma band.

  ## The pi/4 split is a symptom, not the fix

  Splitting only the innermost tile at the diagonal:

      full wedge  [0, pi/2]      FAIL  (as shipped)
      lower half  [0, pi/4]      OK    343 pts, 0 refused
      upper half  [pi/4, pi/2]   FAIL  same gamma-axis flip

  The halves are NOT mirror images: `theta_wedge -> 0` runs toward the SOFT
  shear axis and `-> pi/2` toward the HARD axis, so the D2 fold on
  `(|y1|, |y2|)` does not extend to a reflection about the diagonal. The flip's
  gamma-location depends on theta, which is why the lower half happens to miss
  it for THIS band. An angular split would therefore be tuning to one band,
  not a fix.

  The guard's own message names the real remedy ("subdivide the tile so each
  sub-tile lands in a single nearest-caustic basin") — subdivide in GAMMA.

  ## Why it is currently a gap rather than a crash

  The wedge path deliberately mirrors the LOBE path: a tile raising
  `CarrierDiscontinuityError` (F022) is recorded as a LADDER-SERVED GAP and is
  never subdivided. That was the right default — the ffin path's subdivision
  is what produced 106 charts where 5 suffice — but it means the astroid
  CENTRE is not covered by any wedge chart and falls through to the serving
  ladder.

  This directly contradicts an acceptance criterion recorded for the wedge
  build: "a medial-axis query that the `ffin` path refused now serves". At
  `r ~ 0.1` that is currently NOT met. Verify against the trained artifact
  whether such a query serves via another rung or refuses outright.

  ## Work

  - Decide between: (a) gamma-subdivide only tiles that raise
    `CarrierDiscontinuityError`, reusing the `stable_gamma_bands` machinery
    that already bisects a band on topology change — the natural home, since
    this IS a topology change; or (b) accept the centre as ladder-served and
    DOCUMENT it, having first confirmed the ladder actually serves it to
    tolerance.
  - Prefer (a) only if the centre is a region the sampler visits. `r < 0.2`
    is a small fraction of the interior AREA but sits at the peak of the
    magnification, so check the prior weight before spending charts on it.
  - Whichever is chosen, pin it with a test asserting the served VALUE at a
    near-centre query against a fresh engine evaluation — not which rung
    served it.

  MEASURED CONTEXT: the wedge path is a 56x interior speedup (5 charts /
  1.8 min versus the retired ffin path's 106 charts / 100.7 min for the same
  region), so a modest gamma subdivision of ONE tile is affordable.
