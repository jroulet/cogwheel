---
section: Backlog
---

- **PROGRAM SPINE: a parsimonious patchwork of coordinates for the whole
  chartable plane** `[→ spec]` — owner-directed 2026-08-06, to be carried
  through autonomously. This is the ordering document; each step links the
  fragment that owns its detail.

  GOAL: for the union of the (cusp-excluded, tube-excluded) regions, one
  well-posed coordinate per region, each adapted to that region's actual
  singular structure, with adaptive subdivision everywhere and no orphaned
  machinery left behind.

  ## The lesson that generalises

  Twice now the same defect: a chart's radius is NORMALISED by a caustic radius
  that carries a cusp's `theta^(2/3)`, which drags the singularity to EVERY
  radius, `w`-independently. Curing it is a coordinate change, not more nodes.
  And twice: a tiler with no eps feedback cannot discover it needs more tiles,
  so the defect stays invisible until someone reads the eps DISTRIBUTION
  rather than its max.

  ## Sequence

  1. IN FLIGHT — `wedge_cusp_axis`: cusp-adapted `u = d^(2/3)` angular axis for
     the astroid interior, split at the caustic waist, plus a tiler that emits
     angular columns and subdivides on eps failure.
     [[lensing_wedge_angular_axis_is_cusp_singular]]
  2. NEXT — coordinate-layer cleanup, one build:
     [[lensing_r_caustic_should_root_find_not_scan]] (brentq instead of a
     720-point scan; 0.32% error at gamma=0.9 propagates into the wedge radius)
     and [[lensing_wedge_u_map_stored_in_arclength_fields]] (retire the
     arc-length field names). Must land AFTER step 1: it moves the normaliser
     under that build's eps acceptance.
  3. EXTERIOR — [[lensing_exterior_should_chart_in_polar_not_sd]]: retire the
     `(s, d)` bridge for the bulk, chart in the tiler's native polar
     `(rho, theta_c)`, keep `(s, d)` only for the thin near-fold tube.
  4. EXTERIOR follow-ups — [[lensing_exterior_followup_four_items]]: the
     backwards ghost gate (needs a uniform CFU/Airy ghost before
     `MINUS_GHOST` can be used where it helps), an explicit cusp carve-out in
     the tiler, the polar-vs-`(s,d)` node-budget A/B, and deliberate ppGO
     routing where the engine cannot reach.
  5. SADDLE FORENSICS — the same audit for the macro-saddle, below.
  6. Then, and only then, the full-gamma training sweep
     ([[lensing_production_training_covers_four_percent_of_gamma]]).

  ## 5. SADDLE FORENSICS — what to check, and what is already known

  The owner's four questions, with what has been verified so far:

  a. **Are the deltoid interior charts similarly ill-adapted?** VERY LIKELY.
     `LobeInteriorChart` interpolates on
     `rho_lobe = |y - centroid| / r_deltoid(theta_local)` — the SAME
     normalised-radius pattern as the wedge. A deltoid has THREE cusps, so
     `r_deltoid` carries `theta^(2/3)` at each and the normalisation drags it
     to every radius. PREDICTED, NOT YET MEASURED: run the same 1-D transverse
     cut that settled the wedge (`s` vs raw `theta` vs `d^(2/3)` toward the
     nearer cusp) on a lobe tile.
  b. **Do they adaptively subdivide?** NO — confirmed. There is no
     `_subdivide_lobe_tile`; only `_subdivide_farfield_tile` and the wedge's
     new one. A gated lobe tile becomes a ladder-served gap.
  c. **Are the cusps cut out?** PARTIALLY, and the distinction matters.
     `_lobe_interior_tiles(admission, cusp_angles, n_per_side)` DOES
     cusp-ALIGN via `_cusp_aligned_theta_tiles` so no tile straddles a cusp ray
     or the lobe-local `+-pi` seam. But ALIGNMENT IS NOT EXCLUSION: putting the
     singularity on a tile boundary leaves it in the domain, which is exactly
     what bit the wedge. Check whether a cusp-ball carve-out is needed here
     too, sized by the same separation-gate reasoning as the exterior.
  d. **Is the region exterior to the deltoids ill-posed?** LIKELY WORSE than
     the positive-parity exterior. It is charted by `FarFieldChart` in
     `(s, d)`, whose foot degeneracy is already measured
     ([[lensing_exterior_should_chart_in_polar_not_sd]]) — and with TWO
     deltoids there are two separate caustic curves, so a source can be
     near-equidistant from feet on DIFFERENT curves, not merely on different
     arcs of one. Measure `tie_ratio` over the saddle exterior.
  e. **The inter-lobe corridor.** Already flagged as an open region in
     [[lensing_coverage_map]] ("Inter-lobe corridor (region 2). Settle by
     probe"), and the lobe frame already carries `corridor_half`. This is the
     region with no natural centroid — neither lobe's polar frame is right —
     so it may want its own coordinate or explicit ladder service.

  Note the provenance: the wedge path was a DEGRADED COPY of the lobe path.
  The brief said "transcribe the lobe path"; the plan gate then trimmed the
  cusp alignment the lobe actually has. So the lobe is better than the wedge
  was, and still carries the normalised-radius disease.

  ## Standing requirement — leave nothing dangling

  Every step retires what it replaces. No `(s, d)` bridge left reachable after
  step 3; no arc-length map left in the wedge path after step 2. Today
  produced THREE instances of built-but-unused machinery
  (`InteriorWedgeChart` serve-wired but never trained;
  `FARFIELD_KERNEL_SUM_MINUS_GHOST` never stamped;
  `_subdivide_farfield_tile`'s interior branch deleted by brief), each passing
  component tests while nothing asserted the INTEGRATION.

  Cheap standing guards worth adding: assert every declared envelope label is
  stamped by some producer, and every chart class is constructed by training.
  Both are one-line greps; two of the three would have been caught.

  ACCEPTANCE for the program: every chartable region has a coordinate adapted
  to its own singular structure; every tiler subdivides on eps failure; cusp
  and tube regions are explicitly carved and served by a NAMED rung; and a
  grep for retired coordinate machinery returns nothing reachable.
