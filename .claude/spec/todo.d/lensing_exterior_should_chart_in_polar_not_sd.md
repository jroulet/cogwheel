---
section: Backlog
depends_on: [2026-08-07_subdivision-recursion-wedge-v3-r-caustic]
---

- **RETIRE `(s, d)` FOR THE EXTERIOR BULK — chart in the tiler's OWN polar
  frame; the bridge is pure loss** `[→ spec]` — Professor review (Fable tier,
  owner-authorised) + measurement, 2026-08-06. Supersedes the open direction in
  [[lensing_farfield_sd_coordinate_degenerates]], which measured the symptom
  correctly and left the fix open.

  ## The decisive analytic fact

  At fixed finite `w`, the kernel-sum residual
  `E(y) = F - sum_{a real} H_a exp(i w tau_a)` is **REAL-ANALYTIC on the whole
  open exterior except cusp neighbourhoods**:

  - `F(w; y)` is entire in `y` (the diffraction integral has no
    source-position singularity);
  - the two exterior real images are SPECTATORS at the fold — the pair that
    merges there is the GHOST pair becoming real INSIDE — so `x_a(y)`,
    `tau_a(y)`, `H_a(y)` continue analytically ACROSS the fold and across the
    principal axes;
  - only at the CUSPS does a subtracted kernel diverge, where one exterior
    image joins the three-image merger.

  **So there is no hidden fractional power outside, and the interior analogy
  does NOT transfer.** The interior's `theta^(2/3)` was a singularity of the
  coordinate MAP (the `r/r_caustic(theta)` normalisation imported the cusp
  power at every radius, `w`-independently). The exterior maps contain no such
  power; the `d^(3/2)` lives only in the ASYMPTOTIC REPRESENTATION of an object
  that is analytic at finite `w`.

  ## Therefore the failure is purely that `(s, d)` is not single-valued

  Measured earlier: foot `tie_ratio` reaches **1.000** on a generic ray at
  `|d| = 1.25`, INSIDE the charted `|d| <= 1.22`; `s` amplifies position error
  ~2.4x. A spline over a discontinuous coordinate produces eps of order the
  jump — consistent with the observed max eps 64.2 and with subdivision
  children failing FARTHER OUT than the tiles that survived.

  ## The fix is to delete a conversion, not to invent a coordinate

  `_build_farfield_chart` already lays tiles out in origin-centred eigenframe
  polar `(rho, theta_c)` and THEN bridges to `(s, d)` via
  `_farfield_box_to_smooth`. Polar is single-valued, well-conditioned, respects
  both reflection symmetries, and every window's object is analytic in it away
  from cusps. **One spatial coordinate serves all three windows** — the LABELS
  are band-split in `w`, the COORDINATE need not be.

  ## The ghost-delay proposal is REFUTED — do not retry

  The driver proposed `((Im tau_g)^(2/3), Re tau_g)` as the exterior chart.
  Measured against it:

  - **AXES**: `Im tau_g ~ 1.74 * |y2|` — a C0 KINK along both principal axes
    (log-log slope 1.000, even under `y2 -> -y2`). The driver's "constant to
    2%" fold scaling was measured on a GENERIC ray and does not generalise: at
    `|y| = 1.9`, `phi = 0.25 deg`, the point is ~1.1 from the caustic but
    `Im tau_g` reads as `d ~ 0.08` — a **factor-14 misread**. The axes are the
    anti-Stokes lines of the ghost expansion; any ghost-built coordinate
    degenerates exactly there.
  - **CUSPS**: the `Im tau_g = 0` set has a TRIPLE POINT at each cusp (two fold
    arcs plus the axis ray), so no `(f(Im), Re)` pair is injective near a cusp.
  - **FOLD**: `d(Re, Im)/d(s, d) ~ d^(1/2) -> 0` at the caustic, so the raw
    pair collapses at the inner edge.
  - Conditioning the driver's table missed by not sampling close enough to an
    axis: `cond` reaches **21.7 at phi = 85.5 deg**, worsening with distance.

  Its correct role is as the REGIME / GATE field — which the code already uses
  it for — and optionally as a fold-adapted coordinate for the thin near-fold
  TUBE, where `((Im tau_g)^(2/3), Re tau_g)` is the Chester-Friedman-Ursell
  pair and is foot-free (a genuine but separate opportunity, since the tube's
  foot is unique and `(s, d)` already works there).

  ## SEPARATE FINDING — `FARFIELD_KERNEL_SUM_MINUS_GHOST` is never stamped

  `_build_farfield_chart` passes `definition=FARFIELD_KERNEL_SUM`
  UNCONDITIONALLY (`surrogate_training.py:2891`; docstring ~2820: "always
  trained on the exterior far-field kernel-sum label"), and
  `_FARFIELD_ENVELOPE_DEFINITION = FARFIELD_KERNEL_SUM` is the persisted
  default. The MINUS_GHOST and DIFFRACTIVE tags exist only in the label/serve
  algebra and in tests. So every shipping exterior chart carries the LIVE ghost
  across its whole window.

  And the design, if wired, would be nearly inert in band 0: the decay gate
  `_GHOST_DECAY_IM_THRESHOLD = 0.4` admits subtraction only where
  `Im tau_g >= 0.4`, i.e. `d >~ 0.75` by the measured fold scaling — where the
  ghost is already suppressed by `exp(-3.6)` at `w_max = 8.93`. The "smooth
  across the fold" benefit the label was designed for is unrealised.

  NOT a correctness defect (the gates are deliberately biased to refuse) and
  NOT a reason to move a label boundary — the live-ghost content is smooth but
  oscillatory, bounded at a few cycles per tile at `w <= 8.93`, so it is a
  node-budget item. But the gap between the designed and the wired behaviour
  should be closed or documented.

  ## Work

  1. Re-chart the exterior bulk in `(rho, theta_c)`; delete the
     `_farfield_box_to_smooth` bridge, the arc-length maps, and the
     medial-axis guard for the bulk path. New `axis_schema` tag; stale `(s,d)`
     artifacts hard-refuse.
  2. Keep `(s, d)` for the thin near-fold tube only.
  3. Put tile edges ON the principal axes (kink-free by symmetry).
  4. Add an explicit CUSP CARVE-OUT sized by the separation-gate contour —
     measured ~0.2 y-units from the cusp on-axis at `gamma ~ 0.5`, which is
     substantially WIDER than the Pearcey arm's certified reach
     (`_CUSP_ARM_COVERAGE = 0.07` image-theta rad). The exterior tiler
     currently has NO cusp-ball exclusion, so cusp-adjacent tile corners fail
     eps by construction and burn subdivision budget.
  5. Do NOT move any label boundary. Document the MINUS_GHOST gap above.

  ACCEPTANCE: exterior charts per band fall well below 57 at the SAME 1e-3
  bar; no chart's eps is dominated by a coordinate discontinuity; a query at a
  former foot-tie location serves to tolerance.
