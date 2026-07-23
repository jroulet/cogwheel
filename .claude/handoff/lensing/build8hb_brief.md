# Build 8h-b — Component representation, measured map ceilings, caustic-fixed interiors

## Mission

Close the three measured blockers between the 2.3% serving census and
the ~85-90% the physics supports (the heavy-saddle qd tail is 8h-c,
NOT this build). Every lever below is sized by a measurement quoted in
the facts section; the build's own acceptance is the calibration-scale
re-pilot, never the full campaign.

1. **Per-cell measured ceilings in the ppGO map.** `_measure_cell`
   currently invalidates a WHOLE cell when any angle refuses at any w
   (the saddle-image branch ceiling near w~66 kills 30 of 84 physical
   cells — and with them the entire heavy-mass pure-ppGO population,
   the largest single coverage bucket). Fix: truncation-on-refusal —
   per angle, find the maximal accepted w-prefix (bisection on prefix
   length), certify on the measured range, and store a per-cell
   `w_ceiling` (min over angles) alongside `w_cert`. Schema: new grid
   in the artifact + provenance bump; the loader hard-refuses maps
   without it (the 8g-b tag philosophy). Consumers: the likelihood
   band-split guard changes from the parity wall to
   `min(parity_wall, cell_ceiling)`; `_stratum_ppgo_boundary` trims
   only when the ceiling covers the stratum top. Beyond-ceiling stays
   UNKNOWN/refuse — never extrapolate.
2. **Fold-gated ghost-pair subtraction + component grids (exterior
   label v3).** Extend the far-field label to
   ``E = F − Σ_real H_a e^{iwτ_a} − G`` where ``G`` is the DECAYING
   complex-image ("ghost") pair term — analytic continuation of the
   quartic's complex root pair: same delay/kernel formulas at the
   complex position, bilinear (non-conjugating) products, kernel
   branch continued from the real saddle, decaying member selected by
   ``Im τ_c > 0``, carrier demodulated by the same ``t_min``
   convention. GATING (P1-measured): apply ONLY where the merged-pair
   fold picture holds — outside the existing cusp windows and where
   ``Im τ_c`` exceeds a Professor-authorized floor; where gated off,
   the label stays v2 (two-kernel). The serve-side mirror adds ``G``
   back analytically; a NEW envelope-definition tag distinguishes
   v3-ghost tiles from v2 tiles and the loader/dispatch route each
   correctly (both tags valid — a mixed artifact is legal).
   COMPONENT GRIDS: with the beat removed analytically, re-run the
   node-convergence probe ON THE REMAINDER per axis and re-provision
   (expect ~2x fewer w-nodes and coarser tiling; the probe decides,
   not the expectation).
3. **Caustic-fixed interior coordinates + per-lobe saddle interiors +
   crown-aware banding.** Interior tiles move to coordinates scaled
   by the local caustic (the owner's design seed: the non-smooth
   locus at a fixed place): radial coordinate ρ=|y|/inradius(γ,θ) or
   the Professor's improvement, making admission exact across a γ
   band. Saddle: per-lobe frames (each deltoid gets its own interior
   family; admission by per-lobe winding/topology, real_mask still by
   Morse signs). Crown: the measured eps gradient (3.4 at the
   γ≈0.87-0.99 band vs 6e-2 at γ=0.4) demands finer interior γ-bands
   near γ=1 and possibly an interior component treatment — the
   Professor designs the interior representation FROM these numbers;
   densifying nodes against a 3.4-eps object is forbidden (it fits
   garbage more finely).

## Measured facts (pre-answered — do not re-derive)

- Serving census (1024 draws, v3 artifact + production map +
  band-split): chart-whole 21, ppGO-whole 0, band-split 3, residual
  1000 → 2.3% compliant. Heavy-mass ppGO population blocked entirely
  by the 30 invalidated map cells.
- Map v1: 168 grid cells = 84 structural placeholders + 84 physical:
  43 certified / 11 beyond-wall / 30 refused. Refusals: 29 positive
  small-γ cells at "Saddle wave branch refused: w = 66.05 exceeds
  ceiling" (the saddle-IMAGE kernel ceiling — also the mechanism
  behind v1's dead tiny-γ charts) + 1 saddle far cell. Sweep cost:
  4 minutes at 5 angles (cheap to re-run).
- P1 probe (fold annulus, γ∈{0.2,0.4}, ρ∈[1.1,2.0]): ghost pair
  reproduces the E_ff beat to a few % in magnitude AND phase
  (R/E 0.038-0.14 for w≥3-8); remainder splines 3-6× smoother
  (RN16 1.3e-3-2.5e-3 vs E_ff 7e-3-1.4e-2); kernel C1/C2
  corrections tighten ~1.5-2×. CUSPS: on-axis Im τ_c = 0 exactly —
  subtraction HARMS; gate off inside cusp windows. Sanity: ρ=4 ghost
  negligible. Scripts: scratchpad complex_saddle_probe.py.
- P2 pilot (astroid band γ≈0.87-0.99, killed at 215 charts after the
  astroid slice): exterior parents 7/32 pass (22%, med eps 1.0e-2);
  children 60/100 pass (60% rescue, med 7.7e-4); 40% of children
  still fail (the ghost-subtraction's target). Interior parents 0/17
  pass (med eps 3.4, 3 NaN); interior children 0/65 — halving does
  NOT help the interior. Mid-γ interior probe (γ=0.40, wave-band
  w∈[0.05,20]): eps 6.0e-2 — steep γ-gradient, catastrophe is
  crown-specific, but mid-γ is still 60× above bar.
- Pilot cost lesson (binding on this build's tests AND the re-pilot):
  full-depth pilots cost ~1/8 campaign. The re-pilot runs at
  CALIBRATION scale: one stratum, n_heldout=10, subdivision capped,
  both parities, ~30-45 min — spec it in the driver steps with its
  cost arithmetic quoted.
- Costs measured: far-field tile 600 calls; heldout eval ~0.5 s;
  map sweep 4 min; interior mid-γ probe 3 tiles ≈ 5 min.
- Serving invariants unchanged: F005 additive-only, zero-quadrature
  mandate, refusal vocabulary, tube byte-identity, 8g-b/8h-a tag
  dispatch, beyond-wall/ceiling never certifies.

## Out of scope — hard fences

- NO quad-double (8h-c). NO tube-chart changes. NO campaign in-build.
- NO tolerance/bar changes; the 1e-4 map bar and 1e-3 far-field gate
  stand.
- Ghost subtraction NEVER applied inside cusp windows or below the
  Im τ_c floor (P1-measured harm) — reachable-red required both ways.
- Interior node densification against >1e-1 eps objects is forbidden
  (design change required, not resolution).

## Acceptance (two-tier)

1. In-build (FAST, synthetic-scale):
   (a) map: truncation-on-refusal certifies a synthetic cell whose
   top-w refuses, storing the measured ceiling; dispatch refuses a
   band crossing the CELL ceiling (not just the parity wall); loader
   hard-refuses ceiling-less maps; strata trim respects ceilings;
   (b) ghost label: on a P1-anchor fold config the v3 label's
   remainder is ≥3× smoother than v2's at equal nodes (probe-anchored
   numbers inline); serve-mirror reconstructs F within bars with G
   added back; the tag routes v2 and v3 tiles correctly in one
   artifact; cusp-window gating verified both ways (applying G inside
   a cusp window must be shown harmful/refused);
   (c) interior: caustic-fixed admission is exact across a synthetic
   γ band (no band-edge waste); per-lobe saddle admission admits
   inside each deltoid and refuses between lobes; a mid-γ interior
   tile under the new representation passes the gate where the
   measured 6e-2 baseline failed; the crown band's treatment carries
   a Professor-designed spec with a falsifiable bar (NOT a densify);
   (d) component grids: the convergence probe runs on the remainder
   and its per-axis node recommendation is recorded and consumed;
   (e) fast tier green, tube byte-identity, full domain-test batch by
   the Test Developer.
2. POST-BUILD (driver, in order, each gating the next): map v2 sweep
   (~5-20 min, cost quoted); serving census (expect ppGO-whole and
   band-split buckets to appear at scale); CALIBRATION re-pilot with
   before/after tallies vs this brief's P2 numbers; ONLY THEN the one
   campaign; ladder census target ~85-90% with the qd tail as the
   sole open bucket.
