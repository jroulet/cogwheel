# Build 8h-a — Band-split serving, certified-ppGO map, interior + annulus charts

## Mission

Close the measured coverage gap to the zero-quadrature mandate (owner
ruling: the final lnlike evaluation NEVER falls back to numerical
quadrature — every draw is served by charts or ppGO/arms). Ladder
census against the v3 artifact: true coverage 2.1%; 72.5% of draws
whole-band Schwinger-served; 25.5% carry high-w refusal nodes. The
measured physics that closes it: ppGO certifies ABOVE a
config-dependent frequency w_cert — the wave zone below it and the
near-caustic region are chart territory. Four levers:

1. **Per-node band-split dispatch (the serving change).** The
   surrogate serve and the likelihood dispatch stop being whole-band:
   a draw's w band splits at the certified boundary — chart-served
   nodes below, bare ppGO (the full image-kernel sum, no envelope)
   above, uniform arms at their certified windows, tube charts near
   the caustic as today. Whole-band containment remains the rule FOR
   THE CHART SEGMENT (a chart must contain the sub-band it serves).
   The reconstruction path for ppGO nodes is the existing
   image_kernel sum — no new math, only new routing. Refusal
   vocabulary unchanged: a node no rung certifies still refuses
   loudly (until 8h-b closes the heavy-saddle tail).
2. **The certified-ppGO domain map (the certification artifact).**
   w_cert is MEASURED, never asserted: an offline validation sweep
   computes per-node |F - ppGO_full| / max|F| against exact
   references on a grid stratified in (parity, gamma band,
   caustic-frame annulus, w), interior AND exterior, and stores the
   certified boundary (with a safety margin authorized by the
   Professor) as a hash-pinned data product (the Pearcey-table
   pattern: DATA_CONTRACTS + registry + loader hash check + live
   fallback = refuse-to-certify when the map is absent/corrupt).
   Beyond the Schwinger wall the reference does not exist: the map
   is UNKNOWN there and dispatch must not extrapolate certification
   (those nodes stay refusals until 8h-b measures them with qd).
3. **Interior (4-image) tile family.** Same E_ff definition — the
   subtraction already runs over real_mask, so interior tiles are
   F minus FOUR kernels; same trainer machinery, admission INSIDE
   the caustic disk minus the tube shell (tiles wholly inside the
   4-image region; the existing one-image-count-per-box constraint
   enforced by admission geometry, mirroring the exterior pattern).
   Interior tiles only need w below the interior w_cert.
4. **Targeted edge-annulus subdivision.** The 166 gated v3 tiles:
   subdivide failing tiles (halve, re-admit, retrain) rather than
   densify globally; gate decides survival. Tiles the gate rejects
   even subdivided are recorded; their windows fall to tubes/arms/
   band-split ppGO, and the ladder census attributes them.

## Measured facts (pre-answered — do not re-derive)

- Ladder census v3 (1024 draws, dispatch-level): A_chart 21 (2.1%),
  B_ppgo 0, C_quadrature 742 (72.5%, median ALL 128 nodes),
  D_refusal 261 (25.5%, fold:cusp 192:69, med 31 nodes/draw).
  Violations: 629 astroid / 374 saddle; 360 draws m > 458 Msun
  (8h-b territory); med |y| 0.91. Report:
  /home/tejaswi/Work/cogwheel_training/full_box_v3/ladder_census_v3.json
- ppGO domain probe (per-node, 64-node bands, bar 1e-3 in F units):
  close-in (|y| <= 2.5x reach at small gamma = |y| ~ 0.1-0.4) fails
  at ALL band w; certification onset w_cert ~ 1.3-20 across the
  grid (e.g. gamma=0.2, m=200: w_cert 3.9-12.8 by annulus;
  gamma=0.7 close-in: 1.25-1.51). At |y| ~ 1.9 with tiny gamma,
  |F - ppGO| = 1.3e-4 by w=5, 2e-6 by w=20 (diagonal-sweep data).
  ppGO kernels DIVERGE at low w (C2/w^2) — the low-w failure is
  structural, not tunable. m=1000 rows unmeasurable (reference
  needs w > 60: SchwingerCertificationError) — the map must mark
  beyond-wall as UNKNOWN, not certified.
- v3 artifact (post-8g-b): 100 charts, 35 MB; registered far-field
  med eps 4.6e-5 (F-normalized, bar 1e-3); census lnL max 2.4e-2
  nats on served draws (target 0.05) — quality machinery is sound.
  166/248 built tiles gated (med 3.8e-2, edge annulus dominant);
  28 NaN-gated; 6 foot-of-normal tube skips; saddle bands 3 (v1
  had 7 — diagnose the splitter change as part of lever 3's band
  work if it obstructs, else record).
- Tile admission geometry: n_per_side=2 admits ZERO tiles
  everywhere (corner-anchored grid touches origin); n=5 is the
  smallest with coverage in all reachable bands (measured table
  2026-07-22). Interior admission must assert admitted > 0
  wherever geometry permits — loudly record where it does not.
- Whole-band strata mass-ratio R = sqrt(51.2); saddle w ceiling 58
  (m <= ~458); astroid 443.7. Band-split serving means chart
  w-ranges can stop at w_cert per region — strata above it are
  UNNECESSARY where ppGO certifies (drop them; record the
  decision per band/stratum in the report).
- Serving-stack invariants that MUST hold: F005 additive-only
  (never serve where a rung would refuse), the envelope_definition
  tag dispatch (8g-b), tube byte-identity, per-w refusal
  propagation semantics for the remaining refusal nodes.
- Test tiers are LAW; tree-gate commit preflight active; in-build
  training synthetic-scale only; the mid-run probe protocol is the
  driver's, post-build.

## Out of scope — hard fences

- NO quad-double work (Build 8h-b; runs in parallel after this
  build's tree closes). NO Schwinger/engine numerics changes.
- NO tube-chart construction changes (frozen; interior tiles are a
  far-field-family extension, not tube surgery).
- NO relaxation of accuracy bars; the ppGO map's safety margin is
  Professor-authorized, measured, and recorded.
- NO full-box campaign in-build (driver post-build, tonight).

## Acceptance (two-tier)

1. In-build (FAST, synthetic-scale):
   (a) band-split dispatch: a synthetic draw whose band straddles
   w_cert is served chart-below/ppGO-above with the reconstruction
   matching the exact engine within the lnL-tier bars at every node;
   the split point comes from the map artifact, never a constant;
   (b) map artifact: built on a coarse synthetic grid, hash-pinned,
   loader-verified; a corrupted/absent map REFUSES certification
   (F010 both directions); beyond-wall entries are UNKNOWN and
   never certify;
   (c) interior tiles: a synthetic interior training run produces
   >= 1 admitted 4-image tile with the E_ff/real_mask label,
   passing the gate; interior admission asserts admitted > 0 where
   geometry permits; a straddling-the-tube-shell tile is REJECTED
   by admission;
   (d) subdivision: a deliberately failing synthetic tile subdivides,
   its children re-admit and pass; a child that still fails is
   gated and recorded;
   (e) tube byte-identity + fast tier green (tree gate).
2. POST-BUILD (driver): production ppGO map sweep (exact references,
   hours-scale, parallel); campaign v4 (interior + subdivided
   annulus + trimmed strata, resuming v3 tubes/tiles where valid);
   ladder census — ACCEPTANCE NUMBER: quadrature bucket 0% of draws
   outside the beyond-wall heavy-saddle tail, which must equal its
   measured size (~360/1024 draws at m > 458) pending 8h-b.
