# Build 8h-b3 — The caustic-fixed core: one coordinate system, w-windowed labels, fixed chart windows

## Mission

The representation build the green-field audit
(.claude/handoff/lensing/greenfield_audit.md, read it) demands: move
ALL trained charts into caustic-fixed coordinates, give the labels
their component structure with each analytic term subtracted only on
the w-window where it means something, and replace mass strata with
fixed per-region w-windows. The frozen plan texts are BINDING where
cited: WP6/WP7/WP8 of
`.claude/handoff/lensing/build8hb_plan_full_v1.json` verbatim (read in
full — directional-radius admission with cusp-ray alignment; per-lobe
saddle interiors with the centroid frames and inter-lobe corridor;
crown SACR-C envelope label), with the following audit/probe-driven
amendments layered on:

1. **Exterior tiles also move to caustic-fixed coordinates** (the v1
   plan's WP4/WP5 assumed raw grids — superseded): one (rho, theta_c)
   chart system on BOTH sides of the caustic, per-lobe for the
   saddle.
2. **W-WINDOWED component label** (floor-probe finding, binding): the
   exterior label subtracts each analytic component only where valid —
   at the diffractive bottom (below ppGO's meaningful range) subtract
   NOTHING and fit the bounded smooth F-object (F -> 1 limit); in the
   mid band fit F minus real kernels minus the GHOST term (built and
   verified in 8h-b2: geometry ghost extractor + _ghost_kernel; gate
   at w_min*Im tau_c >= 2 outside cusp windows per the frozen WP4
   text); above w_trust charts end (band-split ppGO serves). Window
   boundaries are config/Professor-pinned constants recorded in chart
   provenance and mirrored EXACTLY at serve time via envelope-
   definition tags (the 8g-b tag machinery; one tag per label window
   class; mixed artifacts legal).
3. **Fixed per-region w-windows replace mass strata** (audit cut 2):
   with band-split serving live, a chart's w-window is
   [w_floor(region), w_trust(region)] — containing every draw's chart
   segment by construction. The strata machinery is not extended;
   region w-windows come from the map (w_trust) and the floor.
4. **Component grids** (frozen WP5 text, adapted): the node-
   convergence routine runs per label-window on the remainder;
   analytic components cost zero nodes.
5. **Born-rung question** (floor probe): the low-w far zone varies on
   the EINSTEIN scale (measured: profiles collapse in |y|, not rho;
   slopes -1.7 to -2.8), so the low-w far region beyond the prior box
   REQUIRES the Born analytic rung — NON-NEGOTIABLE (owner ruling
   2026-07-23): zero-quadrature + prior-universality leave no other
   cover for the low-w far zone (trained tiles there are
   Einstein-scale, hence prior-sized, hence prior-dependent). The
   Professor designs the rung's certification (the deep-diffraction
   series, its error bound, its measured validity boundary in (w,
   distance) — certified-or-refuse like every rung); the plan may
   place it in this build or the immediate follow-on slice, but the
   final serving stack ships with it.

## Width budget (BINDING at the plan gate)

An honest decomposition exceeds 3 WPs. Propose EITHER a <=3-WP first
slice with the remainder explicitly frozen for the follow-on build,
OR two fully-specified sequential slices in one plan (each <=3 WPs,
the second consuming the first). Over-wide single-slice plans will be
rejected on width alone.

## Measured facts (pre-answered — do not re-derive)

- Ghost primitive: committed b14df4b; extractor + _ghost_kernel with
  Im tau_c exposed; oracle-verified 1e-6/1e-4; anchors reproduced.
  P1: ghost removes the fold-annulus beat (R/E 0.038-0.14 mid-band);
  harmful at cusps (Im tau_c = 0 on-axis); 3-6x smoother remainders.
- Floor probe (2026-07-23): low-w |E_ff| profiles at gamma 0.071 vs
  0.4 collapse in EINSTEIN |y| (factor ~2) not rho (factor ~100);
  the current label's low-w magnitude is kernel-divergence-dominated
  (up to 1e6 x F at w~0.03, smooth power law, slope -1.7..-2.8) —
  fitting it works but is perverse; the w-windowed label removes it.
- P2 pilot: exterior parents 22% pass / 60% child rescue under the
  OLD label at production geometry; interior 0% at the crown
  (eps 3.4 — label conditioning, Professor-diagnosed), 6.0e-2 at
  mid-gamma. These are the before-numbers the post-build calibration
  re-pilot must beat.
- Serving census (map v2, ceilings live): 2.2% compliant — the climb
  to ~85-90% rides on THIS build + qd; nothing else moves it.
- Map v2: 45 certified cells with finite ceilings, 39 beyond-wall,
  84 structural; outer-annulus rho cap live (UNKNOWN beyond measured
  rho). w_trust/w_ceiling accessors are the dispatch authorities.
- Costs (for the plan's test specs and the re-pilot spec): far-field
  tile 600 engine calls; heldout eval ~0.5 s; map sweep ~6 min;
  calibration re-pilot budget: one w-window, n_heldout=10, capped
  subdivision, both parities, target ~30-45 min with the arithmetic
  quoted in the plan.
- Invariants: F005 additive-only; zero-quadrature mandate; refusal
  vocabulary; tube byte-identity; tag-dispatch (8g-b/8h-a/8h-b2);
  certification never past measured support (8h-b1, both axes).

## Out of scope — hard fences

- NO quad-double (8h-c follows this build). NO tube-chart changes.
- NO campaign in-build; the calibration re-pilot is the DRIVER's
  post-build step with its cost quoted.
- NO tolerance/bar changes. NO serving of any uncertified region.
- Ghost applied ONLY per its measured gate (fold windows); the crown
  keeps the SACR-C label per the frozen WP8 (no Pearcey model).

## Acceptance (two-tier)

1. In-build (FAST, synthetic scale): the frozen WP6/7/8 verification
   clauses verbatim, PLUS: (a) w-windowed label — each window's label
   is fitted and served via its tag with reconstruction within bars
   across window seams (seam test at both boundaries); the low-w
   window's fitted object is BOUNDED (no kernel divergence);
   (b) fixed w-windows — a chart family with [w_floor, w_trust]
   windows serves every in-region draw's chart segment with NO strata
   bookkeeping; (c) component grids — the convergence routine's
   per-window node recommendation recorded and consumed; (d) the
   mid-gamma interior tile that failed at 6.0e-2 passes its bar under
   the new representation (the binding before/after);
   (e) fast tier green; tube byte-identity.
2. POST-BUILD (driver): calibration re-pilot (cost-quoted) with
   before/after vs the P2 numbers; serving census (the climb must
   begin); then qd (8h-c), map extension, the ONE campaign, ladder
   census.
