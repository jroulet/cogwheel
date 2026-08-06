---
section: Backlog
---

- **REGRESSION: every wedge interior chart fails the eps bar — the astroid
  interior is now UNSERVED** `[→ spec]` — measured 2026-08-06 from the first
  completed training run on the wedge path (`034fcf7`).

  | interior path | charts | median eps | max eps | PASS (bar 5e-2) |
  |---|---|---|---|---|
  | retired `ffin` | 106 | 3.42e-4 | 2.80e-2 | **106/106** |
  | new wedge      |  12 | 5.38e-1 | 7.16e-1 | **0/12** |

  Best wedge chart is 2.99e-1 — 6x over the bar. Median is ~1600x worse than
  the path it replaced. All 12 are gated out, so after `034fcf7` NOTHING
  charts the astroid interior; queries there fall through to the serving
  ladder.

  ## The "56x speedup" was mostly under-resolution

  Reported earlier: the wedge path builds the interior in 1.8 min / 5 tiles
  versus `ffin`'s 100.7 min / 106 charts. That measurement was real but the
  conclusion was wrong, because it counted BUILD TIME and CHART COUNT and
  never checked ACCURACY. 12 charts cover what 106 covered at the SAME
  7x7x7 node grid, so each wedge chart spans ~9x more area at identical
  resolution.

  Decomposing the 21x chart reduction: the D2 fold over
  `theta_wedge in [0, pi/2]` is a genuine 4x from exact symmetry; the
  remaining ~5x is pure coarsening. Expect a correctly-resolved wedge tiling
  to land near 106/4 ~ 26 charts, i.e. a real speedup of ~4x, not 56x.

  ## Why it shipped

  The build's acceptance criterion was right -- "interior held-out eps no
  worse than the `ffin` baseline at equal or lower chart count" -- but the
  plan DEFERRED it to a post-build driver step, because `ffin` was deleted in
  the same build and an in-build relative check would have been a forbidden
  measure-then-decide against removed code. The deferral was defensible; not
  running it immediately after the build was not. A criterion that cannot be
  checked in-build is a reason to check it FIRST post-build, not a reason to
  carry it.

  ## DIAGNOSED — not a bug, not under-resolution: the coordinate does not
  ## make the envelope smooth

  Measured on one mid-radius tile (`r = 0.455 +- 0.089`, band 0), refining in
  place with identical held-out samples:

  | test | result |
  |---|---|
  | eps AT the chart's own nodes | **6.33e-16** |
  | eps off-node, full angular width `[0, pi/2]` | 3.93e-1 -> 2.08e-1 (n 7->13), **p ~ 1.0** |
  | eps off-node, half angular width | 7.15e-2 -> 5.38e-2 (n 7->13), **p ~ 0.35-0.55** |
  | `ffin` baseline, same region | 3.42e-4, 106/106 pass |

  1. **No wiring bug.** Node exactness at 6.3e-16 proves train and serve agree;
     the chart reproduces its own training values to machine precision. A
     train/serve convention mismatch would have floored the node error too.
  2. **Not merely under-resolved.** A smooth object gives `p ~ 4`. `p ~ 1`
     is the signature of a KINK (continuous value, discontinuous first
     derivative) inside the tile; `p < 1` at half width means uniform
     refinement is not resolving it at all.
  3. **The difficulty is angular but not removable.** Halving the angular
     width cuts the error 5.5x, so the error concentrates near the diagonal.
     But closing the 200x gap to the `ffin` baseline needs ~3 halvings = 8x
     more angular tiles, taking 12 charts to ~96 -- essentially `ffin`'s 106.

  **Conclusion: the wedge coordinate yields NO net chart-count saving once
  accuracy is matched.** The genuine 4x from the exact D2 fold is cancelled by
  the finer angular resolution the kink demands. `(r, theta_wedge)` relocates
  the difficulty rather than removing it.

  This also re-reads `ffin`'s 106 charts: that was an adaptive tiler paying in
  chart count to chase a kink, which is the correct behaviour. The earlier
  "21x fewer charts" framing had it backwards -- those charts were doing work.

  ## RECOMMENDATION

  REVERT the wedge wiring and restore `ffin`, which passes 106/106 at 3.42e-4.
  Keep `InteriorWedgeChart` in the codebase as an UNWIRED research object with
  these measurements attached, so the experiment is not repeated blindly.

  The audit premise that motivated the change was sound -- `select_chart`
  dispatched to a chart class training never produced -- but the correct
  resolution is the opposite of the one taken: retire the unused CLASS, or
  fix its convergence, rather than retire the working PATH.

  Before any retry, identify WHAT is kinked. Candidates: the parked critical
  carrier `tau_c` switching image identity across the diagonal (the same
  mechanism that raises `CarrierDiscontinuityError` at `r -> 0`), or the
  nearest-caustic basin boundary crossing the tile.

  ## Work

  - Refine the wedge tiling until eps passes: more radial rows, more nodes per
    axis, or both. The tiling helper is `_wedge_interior_tiles(r_extent,
    n_per_side)` and the node counts come from `TrainingConfig.n_rho` /
    `n_theta_c`. Target the `ffin` baseline (median 3.4e-4), not merely the
    bar.
  - Establish the convergence rate first: if eps falls ~h^4 the fix is node
    counts; if far slower, `(r, theta_wedge)` is no better a coordinate than
    `(s, d)` was, and the interior needs rethinking rather than refining.
  - Until it passes, the interior is a coverage HOLE. Either restore a served
    interior or make the gap explicit at serve time -- a surrogate that
    silently ladder-serves the whole interior is slower than the `ffin`
    artifact it replaced.
  - Re-check `origin_enclosed`/`n_cusp_rays: 0` in the run's
    `farfield_interior` summary: with 0 cusp rays the tiler may not be
    cusp-aligning at all, which would compound the resolution problem.

  MEASURED CONTEXT for the same run: exterior charts also fail badly (57
  built, 35 fail the 1e-3 bar, max eps 64.2), and 51 of 77 charts total were
  gated. See [[lensing_farfield_sd_coordinate_degenerates]].

  ACCEPTANCE: wedge interior charts pass 5e-2 with median at or below the
  `ffin` baseline of 3.4e-4, at a chart count materially below 106; and a
  medial-axis query serves to tolerance.
