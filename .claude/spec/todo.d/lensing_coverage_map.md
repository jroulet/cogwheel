---
section: Backlog
---

- **MASTER COVERAGE MAP — every held-out region and what closes it**
  `[→ spec]` — the index for the zero-quadrature goal. Individual items have
  their own fragments; this is the map that says whether the enumeration is
  COMPLETE. Written 2026-07-28 after a backlog audit found a first-class hole
  (dropped gamma slivers) that had a census bucket, a wrong code comment, and
  no owner, unrecorded for a month.

  Domain: `gamma` uniform on (0, 1.6); `|y| <= 4.2426` (prior box corner);
  `w` set by `m_lens` and the detector band. "Closed" means served by an
  analytic form or a trained chart, never by quadrature.

  ## A. Regions and their status

  | # | Region | Served by | Status |
  |---|---|---|---|
  | 1 | Caustic interior, astroid (`gamma < 1`) | interior SACR-C charts | OPEN at the high-gamma CROWN band — `interior_eps_max = 5e-2` was set with crown reachability explicitly UNMEASURED; a P2 pilot recorded 0% pass at eps 3.4 |
  | 2 | Caustic interior, saddle lobes (`gamma > 1`) | `LobeInteriorChart` | CLOSED (shipped `04f9f5c`), except the INTER-LOBE CORRIDOR, which `_lobe_serves` refuses for both lobes — whether the saddle exterior charts pick it up is UNKNOWN |
  | 3 | Near-caustic shell (fold tubes) | `TubeChart`, `eta in [eta_floor, eta_max]` | OPEN below `gamma = 0.155` — measured, THREE stacked causes (F037), not one. The foot-of-normal skip cause is CLOSED (C6 shipped 2026-08-01: `eta_max = f_max * R_c` per arc, skip guard replaced by assertion). Two bands are still DROPPED as topology slivers by `stable_gamma_bands` — those are `_PROBE_ETA` (F039) and close in step 1b, the analytic sweep. The row closes only when 1b has shipped |
  | 4 | Cusp neighbourhoods | excluded from tubes -> Pearcey arm -> quadrature | PARTIALLY CLOSED (ddd8980, 2026-08-04). Draws within `_CUSP_ARM_COVERAGE = 0.07 rad` of the cusp vertex are now served by the Pearcey arm. Residual exclusion window beyond the arm's certified reach still falls through to exact engine. Saddle exclusions are WIDER (`_SADDLE_CUSP_WIDTH_SAFETY = 2.5`, min half-width 0.08); arm coverage applies uniformly |
  | 5 | Exterior far-field | `FarFieldChart` | CLOSED (per-column admission since 8h-b4) |
  | 6-9 | ~~Far annulus `3.0 < \|y\| <= 4.2426`, and its three gamma fences~~ | — | **DISSOLVING — do not work these rows.** All four existed only because `\|y\| = 3` (the PRIOR BOX half-width) was treated as a physical boundary. F036 measures that no `\|y\|` threshold can bound the caustic at all: `r_caustic` diverges at the parity wall (19.8 at `gamma = 0.99` vs a 4.2426 box corner). `GAMMA_FENCE = 3/4` and the saddle fence `1.0502342` are CONSEQUENCES of the annulus radius, not independent physics, and are deleted with it. These four rows collapse into ONE caustic-relative exterior region. See [[lensing_caustic_relative_coordinates]] |
  | 10 | DROPPED GAMMA SLIVERS (any `\|y\|`, any `w`) | — | CLOSED — `min_gamma_band = 1e-6` (2026-08-03, commit `70affbb`); bisection continues to near-float resolution, total dropped prior mass ~1e-6 fraction, negligible. See `completed.d/2026-08-03_min_gamma_band_zero.md` |
  | 11 | `w` above the certified ceiling (saddle `w > 60`) | — | OPEN and STRUCTURALLY DIFFERENT: no exact evaluator exists there, so charts cannot be TRAINED, not merely are not |
  | 12 | `gamma = 1` parity wall (`det A = 0`) | named refusal | ACCEPTED — measure zero, not a hole |

  ## B. What closes each, in dependency order

  1. **Saddle Born** (region 8). Physics derived; fence exact
     (`1.0502342 < gamma < 3`, a BAND — it re-enters at `gamma = 3`, outside
     our prior but write it as a band). Carrier is LEAD-ONLY like positive
     parity; the complex ghost is REFUSED there pending (4) below.
     [[lensing_saddle_born]]
  2. ~~**Per-column admission** (regions 7, 9).~~ **FROZEN — do not build.**
     This was queued work to build a smarter version of the very fence
     [[lensing_caustic_relative_coordinates]] deletes. A per-column admission
     test for `GAMMA_FENCE` is effort spent making a box-derived boundary more
     accurate, when the boundary itself is the defect (F036). The per-theta_c
     PATTERN stays correct and is reused by the caustic-relative exterior
     region; only this application of it is cancelled.
  3. ~~**Cusp fast-serving** (region 4).~~ **DONE (2026-08-04, commit ddd8980)**
     — `_CUSP_ARM_COVERAGE = 0.07 rad`, measured by direct arm boundary sweep
     (`scripts/measure_cusp_arm_actual_boundary.py`). Draws within 0.07 rad of
     the cusp vertex are now served by the Pearcey arm; residual window beyond
     the arm's certified reach still falls through to exact engine.
     See `completed.d/2026-08-04_cusp-arm-coverage.md`.
  4. **`ghost_kernel`'s Morse branch reference for `det A < 0`.** Its sqrt
     branch is pinned by `reference_amplitude = exp(-0.5j*pi)`, justified in
     its own docstring by "the two real images continue into a Morse-index-1
     saddle" — a POSITIVE-PARITY statement. On the macro saddle both real
     images are already index-1. Blocks admitting the complex ghost on the
     saddle branch.
  5. ~~**Dropped-sliver treatment** (region 10).~~ **CLOSED** (2026-08-03, commit `70affbb`) — `min_gamma_band = 1e-6`; total dropped prior mass ~1e-6 fraction, negligible.
     See `completed.d/2026-08-03_min_gamma_band_zero.md`.
  6. **Small-gamma collar** (region 3). The foot-of-normal skip cause is
     CLOSED (C6 shipped 2026-08-01). Remaining cause: topology slivers from
     `stable_gamma_bands` (`_PROBE_ETA`, F039) — closes with step 1b of
     [[lensing_caustic_relative_coordinates]].
  7. **Crown-band measurement** (region 1). Measure whether the high-gamma
     astroid interior reaches an acceptable eps at all before deciding it needs
     a treatment. Currently a clause inside
     [[surrogate_component-representation-8hb]].
  8. **Schwinger qd extension** (region 11). Extends the certified ceiling
     `w ~ 60 -> ~155`. Cannot be deferred past the full-box campaign, because
     it is what makes training possible up there at all.
     [[schwinger_qd-extension]]
  9. **Inter-lobe corridor** (region 2). Settle by probe: do `gamma > 1` draws
     between the two deltoid centroids come back served or `out-of-box`?

  ## C. Extrapolation

  There is none BY DESIGN, and that is worth stating because it is the usual
  failure mode of a surrogate. `select_chart` enforces box containment on
  `(gamma, rho, theta_c, log w)` plus exclusion balls around refused training
  points, and a mismatched axis-schema tag HARD-REFUSES at load rather than
  reconstructing a finite-but-wrong `F`. A query outside every trained box
  falls through; it is never extrapolated.

  The residual risk is not silent extrapolation but the SIZE of the
  fall-through set — which is what the census measures — plus one narrower
  point: spline error is certified by a held-out eps measured at box-INTERIOR
  quartiles, so accuracy near a box EDGE is interpolated but not directly
  bounded. Keep held-out sampling at interior quartiles and treat edge
  behaviour as a separate question if it ever matters.

  ## D. The honest caveat on this list

  This enumeration is worth exactly as much as the last audit. Region 10 had a
  census bucket, a wrong code comment, and no owner for a month before anyone
  wrote it down. So: RUN A CHEAP DISCOVERY CENSUS EARLY, on the current chart
  set, as an instrument for finding regions nobody has named — NOT as the final
  scoring run it is currently scheduled as.

  **Discovery census run (2026-08-04, commit 97f7fc0, tool: `scripts/census_dry_run.py`):**
  Structural coverage = 100% — every draw in the prior has a serve path. No unnamed
  regions found. Breakdown: Born exterior 71%, tube/far-field 15%, interior wedge 7%,
  lobe interior (saddle) 7%, ppGO fold 0.1%. Production training can proceed on the
  current architecture.

  The expensive full-box campaign stays last (owner ruling 2026-07-20: train exactly
  once, on the final engine and final chart set). The coarse census is done; the
  campaign is not.
