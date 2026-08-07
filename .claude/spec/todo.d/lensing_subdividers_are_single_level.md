---
section: Backlog
---

- **BOTH SUBDIVIDERS ARE SINGLE-LEVEL — a tile needing two halvings gets one
  and is abandoned** `[→ spec]` — measured 2026-08-06 against the shipping
  `_subdivide_wedge_tile`.

  `_subdivide_wedge_tile` and `_subdivide_farfield_tile` are both documented
  "Single-level, no recursion": a child that still fails the eps bar becomes a
  ladder-served gap rather than being split again. So the eps feedback loop is
  allowed exactly ONE iteration.

  ## Measured on the astroid interior (band 0, gamma_mid = 0.495)

  First pass, the shipped waist-split tiler (10 tiles, 2 angular columns):

  | column | charts | result |
  |---|---|---|
  | `low` (soft cusp) | 5 | 5/5 PASS, p50 4.0e-5 .. 8.2e-4 |
  | `high` (hard cusp) | 4 | 0/4 PASS, degrading outward 5.9e-2 -> 3.3e-1 |
  | centre (`r = 0.099`, high) | 1 | BUILD FAILED, `CarrierDiscontinuityError` |

  Calling the SHIPPING subdivider on the four failing `high` tiles (tile dict
  constructed as `surrogate_training.py:4583` builds it; the subdivider's OWN
  reported per-child eps, not a recomputation):

  | parent `r` | packed | residual gaps |
  |---|---|---|
  | 0.277 | 4/4 | — |
  | 0.455 | 4/4 | — |
  | 0.633 | 3/4 | 6.50e-2 |
  | 0.811 | 2/4 | 6.70e-2, 5.95e-2 |

  **13/16 children clear.** The three that do not are MARGINAL — 1.19x, 1.30x
  and 1.34x the 5e-2 bar — while each halving has been buying 2-5x. One more
  level would clear them.

  ## Total, against the retired baseline

  | | charts | median eps | wall |
  |---|---|---|---|
  | `ffin` (retired) | 106 | 3.42e-4 | 100.7 min |
  | wedge, cusp-adapted + 1 subdivision level | **18** | 5.47e-4 | ~10.5 min |

  So the cusp-adapted axis plus one level already gives ~6x fewer charts at
  ~10x less wall time, with the median within 1.6x of `ffin`. The residual is
  three marginal gaps and the centre build failure.

  ## The fix is bounded RECURSION, not a cleverer initial tiling

  An asymmetric initial tiling (more columns on the hard side, whose cusp
  coefficient is 2.05x the soft side's) was considered and REJECTED: adaptive
  subdivision already adds resolution where it is needed, by construction. The
  defect is purely that it stops after one halving. Seeding asymmetrically
  would only reduce the number of rounds, at the cost of over-tiling where the
  first pass already passes.

  Give both subdividers a bounded depth (subdivide until the child clears or a
  depth cap is reached) and record the achieved depth per tile so a runaway is
  visible.

  ## This likely explains the EXTERIOR too

  The exterior shows 84% subdivision children AND 35 of 57 charts still failing
  the 1e-3 bar — numbers that never sat together. Single-level subdivision
  explains both at once: every marginal tile gets exactly one halving and is
  then abandoned, so the child count is high AND the failure count is high. It
  also raises the value of the polar re-chart
  ([[lensing_exterior_should_chart_in_polar_not_sd]]): a re-chart that halves
  the difficulty is worth much more once subdivision can actually converge.

  ## Also found

  `_wedge_cusp_axis_map(theta_lo, theta_hi, origin)` returns a SILENTLY COMPLEX
  array when `theta > pi/2` for `origin='high'`, because `(pi/2 - theta)**(2/3)`
  takes a negative base. It does not raise and does not clamp; the failure
  surfaces several frames later inside `np.interp`. `theta_wedge > pi/2` is
  meaningless in a folded quadrant, so the contract should be enforced at the
  boundary.

  ACCEPTANCE: with bounded recursion, the three marginal interior gaps close;
  the achieved subdivision depth is reported per tile; and the exterior chart
  count and failure count are re-measured under recursion before the polar
  re-chart, so the two effects are not confounded.
