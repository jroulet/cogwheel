# Build 8g-b — Far-field envelope redefinition, serving mirror, consolidated tiling

## Mission

Fix the far-field envelope construction so the fitted object is the
genuinely smooth, tiny remainder the design promises, then re-provision
the tiling against it. Mid-campaign measurement (2026-07-22, v2 partial
artifact): tiles straddling the astroid diagonals carry envelope
"scale" larger than F itself and fit at eps 0.3-0.9, while on-axis
tiles sit at 8e-4. Root cause CONFIRMED by direct sweep: the
criticality switch is keyed on w*|tau_a - tau_c| with tau_c from
`nearest_caustic_point`; on the lobe-equidistance lines (astroid
diagonals) tau_c flips lobes, a well-resolved image spuriously looks
near-critical, its switch turns off, and its full oscillation is left
UN-subtracted — the envelope jumps x1500 mid-tile and the spline fits
garbage. Three levers:

1. **Far-field envelope definition (the fix).** For far-field charts
   ONLY: subtract the FULL ppGO image sum (criticality switch forced to
   1 for every image — all images are resolved in the exterior) and
   drop the tau_c demodulation carrier (constant carrier; no
   caustic-point consultation at all). The far-field training label
   becomes exactly `F - sum_a H_a e^{i w tau_a}` — measured smooth and
   ~1e-4 in magnitude on the good side of the flip line. Tube charts
   are UNTOUCHED: near the caustic the switch and carrier do real work
   and the lobe choice is unambiguous.
2. **Serving mirror + definition tag.** The serving-side
   reconstruction for far-field charts must mirror the new definition
   exactly (full kernel sum added back, no carrier). Label/serve
   consistency is THE correctness invariant. Add an envelope-definition
   tag to the chart meta; the loader hard-refuses a far-field chart
   whose tag is absent or unknown (the v1/v2 partial artifacts predate
   the tag and must not silently serve under the wrong reconstruction).
   DATA_CONTRACTS chart-schema note + contracts_changelog fragment.
3. **Consolidated tiling (the payoff).** Against a ~1e-4 smooth
   envelope the 5x5-per-stratum tiling at 10x10 y-nodes is grossly
   over-provisioned (owner: "surprised we need 346 patches").
   Re-provision from measurement: a node-convergence probe on the new
   envelope decides tile size and node counts (candidate: n_per_side
   2-3), certified by the existing eps gate. Professor authorizes the
   final numbers; the gate stays the enforcement.

## Measured facts (pre-answered — do not re-derive)

- Diagonal sweep (gamma=0.0387, y1=1.3, y2 in [1.10, 1.50], w={5,20,60}):
  theta_c jumps pi -> pi/2 between y2=1.250 and 1.275; |env| jumps
  2.0e-1 -> 1.3e-4 (w=5) and 2.0e-1 -> 2.0e-6 (w=20) at the flip; the
  w=60 envelope grows 1e-15 -> 4.8e-2 approaching the flip line.
  Probe script: scratchpad diag_lobe_jump.py (session-local; inline
  the numbers, do not reference the path in agent context).
- v2 partial-tile eps (50 far-field charts, provenance round-trip):
  median 4.7e-2, p90 6.1e-1, max 8.9e-1 vs bar 3e-3; on-axis tiles
  8.3e-4-5e-3; diagonal/corner tiles 0.27-0.89 with coeff "envelope
  scale" 4-6.6 (bigger than max-normalized F ~ 1). Highest-w stratum
  worst. eps_abs (eps_rel x coeff scale): only 12% of tiles < 3e-3 —
  NOT a currency artifact; the fits are genuinely bad where tiles
  straddle a flip line.
- The flip lines for the astroid are the y1=+-y2 diagonals (cusps on
  the axes); the deltoid has its own equidistance lines — the fix
  removes the dependence entirely rather than splitting tiles.
- Whole-band w containment and mass strata are correct and stay;
  adjacent strata overlap in (y, gamma, w) BY DESIGN (a
  stratum-boundary mass must fit wholly in one chart).
- 8g machinery in place and green: eps registration gate (farfield bar
  3e-3, NaN gated, resume round-trip), distinct-tile records,
  beyond_w_cap accounting, serve-fraction smoke, 59-test batch in
  cogwheel/tests/test_lensing_surrogate_training.py.
- v2 campaign killed at 60 charts (~2 h); charts persist but predate
  the definition tag — campaign v3 runs in a FRESH outdir post-build
  (driver step). v1 artifact likewise legacy.
- Far-field build cost: 600 engine calls/tile, 7-610 s each; tiling
  estimate at 5x5 was 292 tiles -> consolidation shrinks both count
  and per-tile cost (fewer y nodes if convergence allows).
- Test tiers are LAW; tree-gate commit preflight active; in-build
  training is small synthetic configs only.

## Out of scope — hard fences

- NO tube-chart changes (construction, carrier, switch, arcs, guards
  — all frozen; 8g WP3 guards included).
- NO quad-double / Schwinger work (Build 8h). NO interior tiling.
- NO serving-contract changes beyond the far-field reconstruction
  mirror + definition-tag refusal (whole-band containment, guard
  stack, refusal vocabulary unchanged).
- NO full-box campaign in-build (driver post-build; fresh outdir).

## Acceptance (two-tier)

1. In-build (FAST, synthetic-scale):
   (a) reachable-red on the fix: an envelope-continuity test sweeping
   across a diagonal (the measured configuration above) — with the OLD
   definition it must exhibit the x1500 magnitude jump (test asserts
   the jump exists when the fix is disabled), with the NEW definition
   the far-field envelope is continuous and its magnitude stays at the
   1e-4 scale across the line;
   (b) a synthetic far-field tile STRADDLING a diagonal trains below
   the 3e-3 bar under the new definition (same tile config that fails
   under the old);
   (c) serving mirror: reconstructed F from a new-definition tile
   matches the exact engine F at held-out points including across the
   diagonal, within the bar; a chart without the definition tag is
   REFUSED by the loader (F010 both directions);
   (d) node-convergence probe on the new envelope produces a measured
   (tile size, node count) choice certified by the gate on synthetic
   tiles, with the numbers recorded in the build report;
   (e) tube byte-identity: tube training labels and serving unchanged
   to the byte on a fixed probe set; fast tier green (tree gate).
2. POST-BUILD (driver): campaign v3 in a fresh outdir with the
   consolidated tiling; mid-run eps probe at ~1 h (the 2026-07-22
   protocol: partial-artifact serve/eps check before letting the run
   go long); census v3 with lnL tiers; then the 8h qd brief.
