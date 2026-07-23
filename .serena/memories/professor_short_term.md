# Professor short-term (Build 8h WP4 edge-annulus subdivision review, 2026-07-23)

Third consultation this session: WP4 = "targeted edge-annulus tile subdivision" in
surrogate_training.py::_train_band_charts. Gated far-field tile halves h->h/2 into up
to 4 children at (cx+-h/2, cy+-h/2), each re-checked through the SAME region-keyed
admission predicate, retrained via _build_farfield_chart, re-gated via _gate_chart.
Three rulings:

1. SINGLE halving level is right in-build (synthetic scale); depth is a driver concern.
   PHYSICS: halving DOES reduce heldout eps because E_ff over a far-field tile is the
   demodulated envelope (F demodulated by t_min / tau_c per SACR-C), a SLOWLY varying
   object once the analytic carriers are stripped — its residual spline-fit error scales
   with tile oscillation content, which drops as the tile shrinks (E_ff over a smaller
   box spans fewer demodulated phase radians; SACR-C bounds demod phase <= rho_END=4 rad
   over the FULL cluster, so a sub-tile sees a fraction of that). Standard spline theory:
   error ~ h^(p+1) |d^(p+1) E_ff|; halving h drops it by ~2^(p+1). CAVEAT where halving
   does NOT help: a tile STRADDLING a caustic feature (fold/cusp) where E_ff is genuinely
   NON-ANALYTIC (envelope has an Airy-type turning-point structure at the fold, or a
   tau_c LOBE JUMP between deltoid lobes on the saddle side). No admission-passing child
   can escape a feature that sits at the admission boundary itself: children straddling
   the caustic disk edge are DROPPED by the disk predicate, not rescued — so the still-
   failing recorded child near the annulus is EXPECTED and correctly falls to the serving
   ladder (ppGO below w_cert). This is by design, not a WP4 bug. Also: single level
   cannot fix a tile whose failure is nan_eps from an engine CancellationError near the
   parity edge (gamma_eff~0.5) — a child inside the same near-edge cell re-nans.

2. Domain test targets the right modes (admission re-check, drop-vs-retrain, pack-vs-
   record split). ADD to catch subtle WP4 bugs: (a) a child excluded by the disk must be
   DROPPED not SKIPPED-AS-PASSED — assert it is absent from BOTH packed AND recorded-
   failure lists (a "continue" bug could silently swallow it or mis-file it as passed);
   (b) assert the 4 child CENTERS are exactly (cx+-h/2, cy+-h/2) and child half-width is
   h/2 (guards an off-by-half center/width bug that still gates-green by luck); (c) assert
   NO second halving level fires even if a child fails (in-build depth==1 invariant); (d)
   the still-failing recorded child carries the CORRECT gate reason string (heldout_eps
   vs nan_eps) matching WHY it failed, so the serving ladder routes it correctly; (e)
   region-key consistency: a child re-checked must use the PARENT's region key (ext/int),
   not re-derive it — a straddle child near the admit/exclusion boundary must not flip
   region between parent and child.

3. NO numerical-accuracy risk in re-gating children against the SAME farfield_eps_max.
   The bar is TILE-SIZE-INVARIANT: it is an ABSOLUTE bar on max|E_ff| residual (the 5e-3
   absolute far-field currency from test_lensing_farfield_envelope.py, reachable-red an
   order below measured ~1.9e-4), NOT a per-tile or density-normalized quantity. E_ff is
   the demodulated envelope with a config-level max|F| scale, identical for parent and
   child, so comparing a child's retrained eps to the parent's bar is apples-to-apples.
   The child sees the bar as a STRICTLY EASIER target (same absolute bar, less oscillation
   content) — that is exactly why halving is the corrective lever. One watch: the heldout
   node COUNT per child must stay >= the parent's so the eps estimate is not under-
   sampled (a child with too few heldout nodes could pass a spuriously-optimistic eps);
   confirm _build_farfield_chart uses a fixed node density (nodes-per-unit-area), not a
   fixed node count, so a smaller child keeps statistical resolution of its eps.

Cross-cut: WP4 is the correct band-split fallback tier — passing children packed as
extra far-field charts, still-failing children fall to ppGO serving below w_cert. The
annulus (edge between exclusion_radius and admit_radius) is exactly where neither
whole-in nor whole-out admission fires at parent scale; subdivision recovers the
whole-in/whole-out sub-boxes and correctly abandons the irreducible caustic-straddle
sliver to the served ladder.
