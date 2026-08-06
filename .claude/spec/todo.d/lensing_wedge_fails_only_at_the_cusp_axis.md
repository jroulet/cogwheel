---
section: Backlog
---

- **THE WEDGE CHART FAILS ONLY AT THE CUSP AXIS AT HIGH `w` — the coordinate is
  fine; the tiler never excludes a cusp window** `[→ spec]` — measured
  2026-08-06. SUPERSEDES the diagnosis in
  [[lensing_wedge_charts_fail_the_eps_bar]], which blamed the coordinate.

  ## Axis-by-axis, one tile (`r = 0.455 +- 0.089`, band 0)

  | axis refined | eps |
  |---|---|
  | gamma 7 -> 13 | 3.9271e-1 UNCHANGED to 4 digits (p = 0) |
  | `w` 40 -> 168 nodes | 3.9271e-1 -> 3.9220e-1 (p = 0) |
  | spatial 7 -> 13 | 3.93e-1 -> 2.08e-1 (p ~ 1, MAX metric) |

  1-D cuts of the real `partition.envelope`, splined in normalised arc length:

  | direction | 5 nodes | 7 nodes | 25 nodes | order |
  |---|---|---|---|---|
  | RADIAL (theta = pi/4) | 1.31e-4 | 1.83e-5 | 7.19e-8 | steep |
  | TRANSVERSE (r = 0.7) | 1.96e-1 | 1.04e-1 | 6.05e-4 | p ~ 3.6 |

  So radial is superb, transverse converges at essentially CUBIC ORDER, and
  neither `gamma` nor `w` contributes anything.

  ## The max metric hid a localised failure

  `_heldout_eps` returns the MAX over 60 samples. Its distribution:

  | | n = 7 | n = 13 |
  |---|---|---|
  | median | 6.45e-2 | 1.94e-2 |
  | max | 3.93e-1 | 2.08e-1 |
  | max / median | 6.1 | 10.7 |
  | fraction under the 5e-2 bar | 40% | 70% |

  The bulk converges (median p ~ 1.9 and improving); the max does not, because
  it is one locus. **All five worst samples at BOTH node counts sit at
  `theta_wedge` in [1.44, 1.50]** (`pi/2 = 1.5708`) **and at `w = 8.933`, the
  TOP of the w range.** Every one.

  That is the cusp axis at the highest frequency — cusp-diffraction territory,
  which `_pearcey_cusp.py` already exists to handle.

  ## Root cause

  The run's `farfield_interior` summary records **`n_cusp_rays: 0`**: the wedge
  tiler performs no cusp alignment and excludes no cusp window. The retired
  `ffin` path did both. The wedge tile therefore spans right up to the cusp
  axis and is asked to spline the cusp-diffraction structure with 7 angular
  nodes at `w_max`.

  This was a deliberate simplification at the plan gate ("Do NOT add
  cusp-alignment or admission logic to this helper (Simplifier: trim)"), on
  the reasoning that the astroid cusps sit at the wedge's angular EDGES so no
  alignment is needed. That is true for ALIGNMENT and false for EXCLUSION:
  putting the singularity on the boundary does not remove it from the domain.

  ## Corrected recommendation — do NOT revert

  Earlier fragments recommended reverting to `ffin` on the grounds that the
  wedge coordinate "relocates the difficulty". That is now measured to be
  wrong. The coordinate is fine on every axis; the tiler is missing one
  feature.

  - Add a cusp window at `theta_wedge -> 0` and `-> pi/2`, handing those
    neighbourhoods to the existing Pearcey cusp machinery, exactly as the
    `ffin` path did.
  - Re-measure eps on the remaining wedge domain. With 70% of samples already
    under the bar at n = 13 BEFORE excluding the cusp, and transverse
    convergence at p ~ 3.6, the target is reachable at modest cost: from
    1.0e-1 at 7 angular nodes, matching `ffin`'s 3.42e-4 needs ~4.5x more
    angular resolution, i.e. roughly 5 angular tiles at n = 7 -> ~25 charts
    versus `ffin`'s 106. That is the genuine ~4x the exact D2 fold promises.
  - Keep the `w_max` end under review: the worst samples are all at the top of
    the band, so the cusp window may need to widen with `w`.

  ACCEPTANCE: with a cusp window in place, wedge interior charts pass 5e-2
  with median at or below `ffin`'s 3.42e-4, at a chart count materially below
  106; and the worst-sample locus is no longer the cusp axis.
