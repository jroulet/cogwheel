# Professor short-term (this session)

## Consultation: fast-tier mpmath-band escape for 4 lensing tests (2026-08-06)

Four fast tests accidentally drive `f_schwinger` into its w in (60,150] mpmath
band (~85-120 s/eval vs ~0.2 s below w=60). Goal: pull each fixture under w=60
WITHOUT touching production (`f_schwinger`, W_CEILING_SCHWINGER=60,
W_CEILING_SCHWINGER_QD=150 frozen) and WITHOUT deleting assertions.

Physics confirmed:
- w = 8 pi G M_L(1+z_L) f / c^3 = xi*f, EXACTLY linear in redshifted lens mass
  and GW frequency. So lens mass IS the clean monotone lever on w (tests 2,4).
- DD product cap: w*|y_eig| < 58 (`_DD_PRODUCT_MARGIN`). Double-double Schwinger
  band holds ~1e-10 to w~64 (matches DD ceiling 60); float64 only to w~18. So
  the "expensive band" = w in (60, ~150] where the mpmath oracle is invoked.

Test 1 (DDWCeilingTestCase) mechanics read from source:
- dd_cap = 58/(r_max*reach_max) where r_max = DD_R_RANGE[1]=0.70 (NORMALIZED
  radial coord, fraction of caustic reach), reach_max = r_caustic(gamma,theta).max
  over the theta/gamma range (Einstein units). Current gives ~121.6 -> mpmath.
- LEVER: dd_cap monotonically DECREASES as r_max*reach_max grows. reach_max grows
  with gamma (caustic reach expands with shear). So RAISE DD_R_RANGE[1] toward 1.0
  and/or widen gamma upward to push dd_cap<60. To get dd_cap<58 need
  r_max*reach_max>1.0. r_max is normalized (r<1 = inside caustic), so r_max must
  stay <1 for interior 4-image validity. reach_max at gamma=0.5 is <1 typically
  (~0.68 measured), so r_max alone (max ~0.99) may NOT get product>1 -> MUST also
  raise gamma to grow reach. TRAP: r_max must stay strictly <1 (astroid interior),
  and w_range upper (500) must stay > dd_cap so the cap still BINDS
  (test_w_max_below_requested + teeth test). Also keep some nodes non-refused.
- Note: the file's OWN cost-budget docstring claims DD cap reduces w_max to ~30
  (~5 w-nodes, ~20s) — the shipped constants (w_max 121.6) CONTRADICT that
  intended design. The fixture drifted from its documented budget; this is the
  root cause, not a production bug.

Test 4 (test_band_limit_refusal_precedes_coherent_score) KEY verdict:
- BAND_LIMIT_LENS m_lens=720 (=90x8). LensedBinningError fires when relative-
  binning delta_t_max (dimensionful image-delay span) is exceeded. dt_a =
  4 G M_L(1+z) tau_a / c^3 ~ 2e-5 s * (M_L/Msun)(1+z) * tau_a. Both dt and w
  scale LINEARLY in M_L -> the binning-band-limit threshold and the Schwinger
  w-ceiling are PHYSICALLY ENTANGLED through the SAME mass lever. Whether binning
  bites at w<60 depends on the DIMENSIONLESS delay span w*tau vs dimensionless
  |y_eig| product: binning cares about w*Delta_tau (full Fermat span), Schwinger
  cares about w*|y_eig|. These are DIFFERENT geometry factors (tau span vs source
  offset), so decoupling is POSSIBLE in principle by choosing high-tau geometry
  (near-caustic, large delay span) at modest |y| -> binning trips at lower w.
  RECOMMENDATION given to driver: attempt fixture-fix by near-fold geometry FIRST
  (raises Delta_tau/|y| ratio so binning bites below w=60); if the Test Developer
  cannot find such a config numerically within a couple iterations, SLOW-TIER it
  behind COGWHEEL_TRAIN_TIER with a corrected ~100 s/eval cost comment. Do NOT
  force it.

Code-obs already captures: prior gamma range (0,1.6); LensedPosterior maps
refusals->exact -inf; C7/C6 draw finiteness ~41%.
