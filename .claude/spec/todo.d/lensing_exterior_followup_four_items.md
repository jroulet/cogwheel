---
section: Backlog
depends_on: [2026-08-07_polar_rechart]
---

- **EXTERIOR FOLLOW-UP: ghost label, cusp exclusion, node-budget test, ppGO
  fallback** `[→ spec]` — owner-directed 2026-08-06, following
  [[2026-08-07_polar_rechart]]. Four coupled items; the
  polar re-chart is the prerequisite for the second and third.

  ## 1. The ghost-subtraction gate is BACKWARDS relative to utility

  Owner's observation, and it is correct: far from the caustic
  `FARFIELD_KERNEL_SUM_MINUS_GHOST` is a near no-op (the ghost has decayed),
  while near the fold it is the whole point ("so the stored remainder is smooth
  across the fold", channels.py:124-127). But `_GHOST_DECAY_IM_THRESHOLD = 0.4`
  admits subtraction only where `Im tau_g >= 0.4`, i.e. `d >~ 0.75` by the
  measured fold scaling — **it permits subtraction exactly where it is
  pointless and refuses exactly where it would help**. Combined with the fact
  that the label is never stamped by the tiler at all
  (`surrogate_training.py:2891`), the designed benefit is entirely unrealised.

  TWO OBSTACLES to simply flipping it on everywhere:

  - The gate's stated reason (F027) is real: an O(1) ghost is not a "small
    correction", so subtracting its LEADING-ORDER SINGLE-SADDLE form leaves an
    O(1) x relative-error residual, possibly worse than not subtracting.
  - Subtracting the single-saddle ghost near the PRINCIPAL AXES would INJECT a
    C0 kink the true field does not have: `Im tau_g ~ 1.74 |y2|` (measured,
    log-log slope 1.000) while the kernel-sum residual is real-analytic there.
    The axes are the ghost expansion's anti-Stokes lines.

  So the fix is not a label flip. It is a UNIFORM ghost representation
  (Chester-Friedman-Ursell / Airy) valid where the single-saddle expansion is
  not, after which the subtraction can be admitted near the fold — which is
  where it makes the remainder smooth. Then the label can be used everywhere
  outside, exactly as the owner proposes, and the decay gate can retire.

  ## 2. Cusp exclusion belongs in the TILER (DONE: 2026-08-09, build exterior_cusp_exclusion_cut, commit d685ebe)

  Pearcey is a SERVING RUNG, not a chart: `_pearcey_cusp.cusp_amplification` is
  called from `operator.py:447` inside `F_op`'s ladder, which returns the first
  rung that certifies. **No chart ever covers a cusp.** The exterior tiler has
  no cusp-ball exclusion — only cusp-node placement,
  `_reject_if_cusp_spanning`, and eps-driven subdivision — so cusp-adjacent
  tiles fail eps BY CONSTRUCTION and burn subdivision budget chasing a region
  the ladder was always going to serve.

  Add an explicit cusp carve-out sized by the separation-gate contour: measured
  **~0.2 y-units** from the cusp on-axis at `gamma ~ 0.5`, substantially wider
  than the Pearcey arm's certified `_CUSP_ARM_COVERAGE = 0.07` image-theta rad.
  Confirm the ladder actually serves the carved region to tolerance before
  carving — an exclusion nothing covers is a coverage hole, not a fix.

  ## 3. Node-budget test for the polar re-chart

  Direct A/B with the machinery used today: build the SAME tile geometry at the
  SAME node counts in `(s, d)` and in polar `(rho, theta_c)`, compare eps; then
  count how many charts each needs to clear the 1e-3 bar. Baseline to beat: 57
  charts / 39.4 min per band, of which 84% exist only because subdivision was
  forced, and 35 of 57 still FAIL the bar.

  NOTE, correcting the owner's premise: the EXTERIOR already has adaptive
  subdivision (`_subdivide_farfield_tile`) — that is exactly why 84% of its
  charts are children. It was the INTERIOR wedge path that had none; that is
  being fixed in the `wedge_cusp_axis` build.

  ## 4. ppGO as the fallback rung where charts cannot reach (DONE: 2026-08-08, build exterior_followup WP4, commit 609d8d3)

  Where the exact engine cannot extend a chart — above the QD ceiling
  (`W_CEILING_SCHWINGER_QD = 150`), or where the DD product cap
  (`w * |y| < 58`) binds — serve from certified ppGO rather than leaving a
  ladder-served gap. The machinery exists (`ppgo_map.CertifiedPpgoMap`,
  `_apply_ppgo_trim`, the `ppgo_exclusion_rho` already reported per band); this
  is about ROUTING to it deliberately in the tiler's coverage accounting rather
  than falling through.

  ACCEPTANCE: (1) the ghost subtraction is admitted near the fold under a
  uniform representation, with a measured smoothness gain across the fold and
  no injected kink at the axes; (2) no exterior chart is built inside the cusp
  carve-out, and the carved region is served to tolerance by a NAMED rung;
  (3) polar beats `(s, d)` on eps at matched node count and needs materially
  fewer charts than 57/band; (4) regions beyond the engine's reach report a
  ppGO-served coverage class, not a gap.
