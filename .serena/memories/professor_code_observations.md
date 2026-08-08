# Professor — code observations (personal, does NOT propagate)

cogwheel code-level details the Professor accumulates (function behavior, call
order, data layouts, numerical gotchas). Personal to this checkout — soft-blacklisted.

- `channels.py` switch (SUPERSEDED at Build 3f): the F008 full-cluster
  nearest-neighbour rule in `_channel_switch` was replaced by the SACR-C
  criticality-separation switch S_a = smootherstep(w*|tau_a - tau_c|,
  0.5, 4) (now 4-arg with critical_delay); virtual channels never
  switch. The wave/geometric branch gate `_min_delay_separation` is
  UNTOUCHED and still uses the old-style separation — re-check against
  the paper's cluster-separation definition if issues surface there.
- `channels.py` public entry (Build 7b): the interim positive-parity
  guard at the top of `ChangRefsdalChannels.evaluate` was LIFTED — the
  channel layer now serves BOTH parities; only `macro_matrix`'s two named
  refusals (Type III lam<=0, det A=0 boundary) + census/fold guards raise.
- `channels.farfield_ghost_term` GHOST ADMISSION GATE (Build 8h-d1,
  reviewed PASS): the old frequency-dependent DECAY gate
  (w_min * Im tau_c >= _FARFIELD_WINDOW_RADIANS = 2.0) is RETIRED. The
  production gate is a frequency-INDEPENDENT geometric separation:
  min over real images of the complex Euclidean |x_a - x_c| (Einstein
  units, bare — no normalization, no w floor) >= `_GHOST_SEPARATION_MIN`
  = 0.7, with x_c = ghost_kernel(...).position (Im kept) and x_a from
  geometry.find_images. w-independence means the train and serve paths
  reach the SAME decision by construction (the old gate skewed: train's
  w-grid min=2 refused while serve's min=4 admitted the same config).
  Verified numbers: near-cusp REFUSE separations 0.205-0.294 (cluster
  ~0.24), well-separated ADMIT 1.43-1.96 (higher than the brief's 1.33
  anchor, same order) — 0.7 sits inside the gap, biased toward refuse
  (false-admit = silent lnL bias; false-refuse only falls back to exact).
  Do-nothing ratios resid(MINUS_GHOST)/resid(KERNEL_SUM) = 0.009-0.323 on
  every admitted config: subtracting a resolved ghost strictly reduces the
  un-modelled remainder, as stationary phase predicts. `_GHOST_DET_FLOOR`
  in `geometry._ghost_kernel` is a DISTINCT self-merge pathology, kept.
  The 2.0 constant survives only as a carrier-resolution target in the
  ghost-frame-collapse tests, not as an admission gate.
- Ghost delay frame (Build 8h-b7 + 74c1d55): the ghost carrier is now
  min-subtracted like the real kernels — `_frame_delays(source, matrix)`
  returns (images, absolute_delays, t_min) as THE authoritative frame
  construction (`_frame_t_min` is a thin accessor over it), and
  `ChangRefsdalGeometryPartition` carries t_min/images so builders don't
  re-solve the image quartic. `geometry.ghost_kernel` stays RAW/holomorphic.
- `likelihood.py::_surrogate_coefficients` guards (Build 8h-d1): a
  `beta != 0 -> return None` guard now sits beside the kappa!=0 guard.
  Rationale: serve de-rotates the source into the shear eigenframe (y1,y2)
  but passes theta un-rotated, so served geometry = theta_eig + beta while
  the emulator was trained on the beta=0 surface — finite-but-wrong.
  Latent only because production pins beta=0.
- `lensing/chang_refsdal/_born.py` STATUS UPDATE (2026-07-28, saddle Born
  carrier build; supersedes the Build 8h-c1 "b1 placeholder" entry below —
  that concern is RESOLVED, not applicable to current code): the lead
  SERVE carrier is `born_lead_carrier` = sqrt(mu_macro)*exp(iw*phi_geo)*morse
  (morse = literal -1j iff det_a=(1-kappa)^2-gamma^2<0 else literal 1.0 —
  hardcoded, NOT cmath.exp, to avoid a ~6e-17 real-part rotation error).
  Verified against an independent matrix-solve full-Fermat-delay oracle to
  2.8e-14 over the full gamma/|y|/theta/w sweep, both parities (mu_macro<0
  confirmed genuine saddles). b1/a0 (F023/F025 closed forms, point-mass
  b1=-1,a0=0, invariant b1-a0=-lam^2*mu_macro) are now DERIVED (matrix-solve
  oracle agreement 7e-15) but stay DIAGNOSTIC-only in born_amplification/
  born_envelope — NOT in the serve carrier (F009 pin: |F| must stay
  w-flat). Band-split currency re-keyed to w*Delta_tau (full Fermat-delay
  span of the 2 real images, read off the partition, not re-solved) —
  confirmed genuinely distinct from the retired w*r0_sq currency (opposite
  serve/refuse decisions on a measured witness config). Positive-parity
  exterior fence (gamma<3/4) and a new closed-form macro-saddle fence
  (`saddle_caustic_max_y`, F026, serving band ~1.0502342<gamma<3) both
  gate born_gate. Function+gate+census 'born' category+coefficients are
  SHIPPED; the live likelihood serve slot remains DORMANT/UNWIRED (falls
  through to the certified exact engine) pending the TRAIN_TIER residual-
  chart driver step.
- `likelihood.py` post-3f: `_amplification_coefficients` interpolates
  ONLY the envelope E(w) on a LOO-adaptive coarse grid (seed 8, ceiling
  48; measured N=26-32), with closed-form switched-saddle reconstruction.
  Post-3g a candidate/fiducial ratio layer (lattice-snapped fiducials,
  `_fid_cache`, health/image-count guards, fallback to certified direct)
  brings warm lnlike to 9.8 ms. Build 7b: the LOO stop is gamma'-keyed —
  _LOO_STOP_FAST=4e-3 for gamma'<0.5 (byte-identical to old) tightening to
  _LOO_STOP_STRONG=1e-3 for gamma'>=0.5, via a pure helper reading only
  lens gamma/kappa so fiducial-cache purity holds; a global tighten to
  1e-3 measured 1.44x crown wall time (rejected).
- Zero-noise F->1 floor in `likelihood.py`: after anchoring on a
  gamma=kappa=0 candidate (genuinely F->1), the residual RB gap traces
  to a construction asymmetry — `_set_summary` builds `_h0_edges` with
  `disable_precession=False` forced plus `_stall_ringdown`, while
  `_candidate_bin_ratios` builds candidate edges with neither, so the
  ratio != 1 in the stalled-ringdown band even at F=1 (F007). Fix
  direction: make `_candidate_bin_ratios` mirror `_set_summary`'s edge
  build rather than computing a full-res candidate strain per eval.
- `F_op` (`operator.py`) CancellationError fires near gamma_eff~0.5 (the
  order-42 shear-series tail crosses the 1e-10 refusal cut there) — the
  marginal certified-domain edge for macro-saddle-adjacent configs, not
  a bug; don't widen the refusal threshold to route around it.
- Batched contraction (`F_op_grid`/`_grid_certified`, Build 3c): scalar
  `F_op` delegates to the batched core, so scalar suites auto-exercise
  it. Certified rel-err vs the mpmath oracle grows exponentially with
  cancellation exponent L (~1e-16 at L~0 -> ~1e-10 at L~44), then the
  F005 refusal cuts in; certify-XOR-refuse held with zero solo-vs-batch
  decision flips.
- Build 4 sampling layer: `LensedPosterior.lnposterior_pardic_and_
  metadata` maps LensDomainError/CancellationError -> exact -inf (raw
  likelihood raise contract untouched; scalar AND folded sampler paths
  route through the override). This IMPLEMENTS the "Build-3/4 sampler
  requirement" flagged in `professor/microlensing_chang_refsdal` — that
  topic memory's "not yet implemented" note is now STALE (research
  session to update; Dreamer does not edit topic memories).
- Sampled coordinates (`lensing/prior.py`): ln_m_lens_msun uniform
  (ln 10, ln 3500); mass-conditioned source y = u*min(307/m, 3.0) with
  folded_reflected u1,u2 (NO phase fold under XPHM); kappa=beta=z_lens
  fixed 0 via FixedPrior. prior.standard_params == likelihood.params.
  Build 7b: reduced-shear gamma range WIDENED (0,0.45)->(0.0,1.6),
  a SINGLE uniform range spanning positive parity (gamma<1) AND macro
  saddle (gamma>1); identity transform, NO discrete parity label; gamma=1
  (det A=0) is a measure-zero named refusal -> -inf at posterior. The
  ['u1','u2'] astroid quadrant fold stays valid on the deltoid caustic.
- C7 measurement (Build 4 review, verdict CONCERN): only 41.2%
  (206/500) of prior draws finite — the gamma prior overlaps the
  gamma_eff~0.5 cancellation band; all non-finites are exact -inf
  (0 NaN); near-truth reference lnpost 260.6 dominates best random draw
  18.1, so the peak sits at truth. Efficiency-only concern.
- Cross-parity Schwinger (Build 7a): the `_schwinger` engine is
  signature-AGNOSTIC. `_h_dd` takes da_im=-w*a/2, db_im=-w*b/2 as pure-
  imaginary offsets; the real-t contour stays clean for both parities, so
  the `gamma_prime>1` guard in `_validate_inputs` was POLICY, not a math
  necessity (relaxed to >0). Cancellation law L_S=pi*w/4 holds on BOTH
  parities (from the `t^{iw/2-1}` factor). Positive-parity mass-sheet
  fallback = (1/lam)*exp[0.5j*w*ln lam - 0.5j*w*kappa*s]*
  f_schwinger(w,y_eig,gamma') with gamma'<1 — EXACTLY `_saddle_grid`'s
  formula (`_grid_certified` multiplies an EXTRA exp(0.5j*w*s) because its
  pure-shear G kernel excludes e^{iw|y|^2/2}).
- Index-theorem guard (Build 7a): unified invariant sum_a sign(mu_a) ==
  sign(det A) - 1 (positive parity -> 0, saddle -> -2). No maxima since
  tr Hess = 2*lam > 0. `morse_index` uses eigvalsh with strict `< 0.0`; a
  degenerate fold image counts non-negative, signed sum still holds; only
  breaks if BOTH eigenvalues near zero at once (a cusp / three-image merge).
- Build 8a surrogate (`lensing/surrogate.py` LensAmplificationSurrogate):
  emulates the SACR-C ENVELOPE E(w) (smooth/beat-free), NOT the oscillatory
  F; tensor cubic spline over (ln w, gamma, y1_eig, y2_eig), real/imag
  interpolated SEPARATELY, per-region (parity/image-count). beta eliminated
  EXACTLY by eigenframe rotation R(-beta) (E invariant to <1e-12). Off by
  default (amplification_surrogate=None -> crown byte-identical). Serve gate
  = box containment + exclusion balls + per-w refusal propagation, no
  learned mask. lnL-from-envelope error obeys dlnL ~ eps_dense*SNR^2 with
  |lnL|~SNR^2 (ratio O(1), measured peak ~0.84) — a fixed nat budget is the
  WRONG currency. RED FLAG: a near-caustic box-edge config eps=0.16 gave a
  12.8-nat lnL error; before ANY enable-by-default the served region needs a
  caustic/edge margin and a re-gate at production eps~1e-4. Saddle dlnL~0.66
  is RB-binning-floored (F016), not envelope-limited.
- Far-field tile subdivision (Build 8h WP4): halving a tile reduces heldout
  eps because E_ff (the demodulated SACR-C envelope) is smooth — spline
  error ~h^(p+1) drops ~2^(p+1) per halving — EXCEPT tiles straddling a
  genuine caustic non-analyticity (fold/cusp turning point, or a tau_c lobe
  jump between deltoid lobes), which subdivision cannot rescue; those
  children correctly stay failing and fall to the ppGO serving ladder below
  w_cert. `farfield_eps_max` is an ABSOLUTE bound on max|E_ff| residual, not
  per-tile/density-normalized — valid to compare parent vs child directly.
- ppGO ceiling mechanism (Build 8h-b, verified against
  test_lensing_ppgo_bandsplit.py, 66/66 green): `_measure_cell` in
  ppgo_map.py does truncation-on-refusal via `_max_accepted_prefix`
  (bisects the w-node INDEX per angle, monotone-refusal assumption; a
  non-monotone break only shrinks the accepted set). w_ceiling =
  min-over-angles of the accepted-prefix endpoint; a fully-accepted cell
  forces ceiling=wall (byte-identical to HEAD).
  `_surrogate_coefficients`/`surrogate_training.train` both gate on
  eff_ceiling=min(wall, cell_ceiling); the outer annulus [4,inf) rho band is
  additionally capped at rho_measured_max_grid; loader hard-refuses on
  pre-0.2.0 artifacts.
- This agent instance has no image-rendering/Read-image tool — validate
  diagnostic PNGs via the numeric asserts backing the same plotted
  quantities (independent-oracle asserts etc.), not visual inspection.
- Saddle lobe-serve build (test_lensing_surrogate_lobe.py, 32/32 PASS,
  reviewed numerically not just green): `_lobe_boundary_radius` is
  confirmed the single authoritative r_deltoid source (max diff 0.0 vs
  `_r_deltoid` across a theta sweep). Corridor predicate correctly routes
  a bisector-equidistant source to decline BOTH lobes (falls to the exact
  ladder), never double-serves. `image_count==4` is a genuine INTERIOR
  property under the eta_max shell (all interior nodes real_mask.sum()==4;
  exterior sources give 2), matching the (0,1,1,1) An&Evans neg-parity
  partition. heldout_eps stored on a chart is a COARSE smoke-tile LOO bar
  (measured 0.138 here); actual interior error sits only ~6x below it, not
  the ~50x (~3e-3) some docstrings imply — the gate is still correct
  (compares against the chart's OWN eps), but don't read a chart's
  heldout_eps as a tight accuracy estimate for its interior.
- Build 8h-b6 cusp-alignment review: `EnvelopeReconstructionTestCase.
  test_positive_box_reconstruction_within_budget` stayed RED after the
  cusp-aligned-tiler fix because that test's `_train()` fixture builds via
  `LensAmplificationSurrogate.from_engine()` — a single-box surrogate with
  a UNIFORM theta_c grid that never calls the production tiler
  (`_train_band_charts`/`_farfield_exterior_tiles`) the fix landed in. A
  tiler-level fix cannot move an eps measured on a fixture that bypasses
  the tiler entirely.
- QA heuristic for a served-value insensitivity/perturbation gate: check
  the SHAPE of the response, not just its magnitude — deviation scaling
  ~linearly with perturbation size is itself evidence of a correct frame/
  bracketing mapping, whereas a frame or bracketing bug tends to produce a
  perturbation-INDEPENDENT flat swing (Build 1c cusp-vertex review).
- `geometry.caustic_derivatives` wedge-edge FP straddle (Build 1d): at
  theta = center +/- theta_max the discriminant is a float measure-zero
  boundary, not a clean sign — center=0 side lands disc<=0 (raises), but
  center=pi side lands disc~+1e-15 (SERVES a divergent |y'|~7.4e7). A
  docstring claiming "always raises exactly at the edge" is true only on
  one side; the honest contract is a disjunction (raises OR |y'| exceeds
  a divergence floor), not an unconditional raise.
- `reconstruct_farfield` lives in `chang_refsdal/channels.py` (7 args incl
  t_min), NOT surrogate.py. `FarFieldChart.from_values` is keyword-only
  with required `arc_map` (_FarFieldArcMap) + gamma_grid/s_grid/d_grid axes.
- InteriorWedgeChart PHYSICS (Build interior_wedge_chart, reviewed PASS):
  coordinate system design is correct. `_to_wedge_fixed` exploits D2
  (dihedral-4) symmetry of the astroid caustic (reflections across both
  eigenvalue axes). The fold abs(y1),abs(y2) -> atan2(|y2|,|y1|) is the
  correct D2 quotient. The radial coordinate r = |y|/r_caustic(gamma,theta)
  normalises by the direction-dependent caustic reach — r<1 equivalent to
  "inside the caustic" (4-image Chang-Refsdal). Bilinear interpolation of
  r_caustic at 101 theta nodes x 5 gamma nodes introduces O(h^2) error
  where h ~ pi/200 ~ 0.016, well below any physical scale. The carrier
  continuity check correctly implements the "single nearest-caustic basin"
  requirement: a jump > 50% of local caustic reach signals a basin boundary
  crossing that would make the demodulated envelope discontinuous. The
  tensor-product cubic B-spline exactly reproduces training values at grid
  nodes (verified to machine precision).
- fold_ppgo_correction PHYSICS: Airy fold correction replaces raw ppGO
  (geometric_amplification) for DEGENERATE-DELAY image pairs near caustic
  folds. The correction term = (airy_value - pair_ppgo) * exp(-1j*w*t_min).
  Structural gates: _merging_fold_pair identifies the degenerate pair,
  _soft_axis_cubic smooths the fold contribution, _fold_amplitudes computes
  the Airy approximation. Falls back to raw ppGO on any gate miss (no
  error-estimate or ETA_MAX gate — DO-NOTHING control design). The
  correction is significant (4-40%) at w=5..15 where fold divergence
  dominates; at w>25 diffractive error drops below Airy residual. The
  xi=0 case (exactly on the fold) returns error_estimate=0.0 (Airy is
  exact on the fold). Relaxation from `if not (xi > 0.0): return None`
  to `if xi < 0.0: return None; if xi == 0.0: return 0.0` is physically
  correct.
- ppGO INTERIOR CERTIFICATION: envelope extrapolation for power-law error
  decay in interior cells (rho < 1.0). The physical basis: ppGO error
  decays as a power law in w (stationary-phase correction terms scale as
  w^{-n}), so log(error) vs log(w) is approximately linear in the high-w
  tail. Extrapolation to find the w_floor where error crosses the bar is
  justified ONLY in the power-law regime (R^2 > 0.9 requirement rejects
  beat-aliased cells where the power law breaks down).
- `LobeInteriorChart` sqrt-edge coordinate (Build 1e-lobe): s = sqrt(span)
  - sqrt(theta_max - theta) in surrogate.py; V1 schema has theta_to_s=None
  (identity path); current schema requires theta_to_s key in npz. Round-
  trip error at machine epsilon (~3e-17). F042 knife-edge sensitivity IS a
  production phenomenon at cusp-adjacent tiles (12+ nodes); smoke-scale
  tests (7 nodes, ~0.37rad span) show ~0.138 eps for BOTH coords (not cusp-
  adjacent) — do not read smoke-scale swing as evidence of knife-edge.
- `surrogate_training.py` C6 (Build 3, 2026-08-01): `TrainingConfig.eta_max`
  and `TrainingConfig.eta_floor` fields were REMOVED; tube-shell exclusion
  geometry is now controlled by explicit kwargs `eta_max` and `eta_floor`
  passed to `_build_tube_chart`, `_tube_heldout_samples`,
  `_saddle_lobe_admissions`, and `_interior_admission`. The standard
  operating-point value is `eta_max = f_max * R_c` per arc (curvature-
  relative); the legacy test-fixture constant 0.05 (Einstein radii) at the
  default f_max=0.40 design point is still correct and must be passed as an
  explicit kwarg. This is a pure call-interface refactor — no change to the
  underlying admission geometry.
- BornResidualChart serve path (Build born-residual-wiring, 2026-08-01,
  PASS): `likelihood._surrogate_coefficients` fact-4 slot wiring verified:
  kappa/beta guards fire before the Born slot (physics-mandatory — chart
  trained on kappa=0, beta=0 surface); rho>1.0 guard is correct (interior
  rho<1 has 4-image topology, Born carrier formula does not apply);
  reconstruction algebra `(f_total - ppgo) * exp(1j*w*t_min)` demodulates,
  `reconstruct_farfield` re-modulates via `_frame_phase` — round-trip
  cancels to machine precision (tested 1e-13). `covers(gamma, rho)` checks
  only (gamma, rho) axes — w-band coverage is a training-driver-
  responsibility contract. INS-12-001 (MINOR): producer side uses
  np.exp(1j*w*t_min) inline instead of `_frame_phase` — functionally safe
  (libm handles large arguments), convention violation only.
- `caustic_rho(gamma, source)` raises ZeroDivisionError at gamma=0 (not
  ValueError or LensDomainError) — existing except clauses in
  `_surrogate_coefficients` that catch ValueError/LensDomainError will miss
  this; add explicit ZeroDivisionError to the except clause (INS-11-001).
- `retired_concepts.json` (2026-08-03): currently 4 entries — _WEDGE_EPS,
  _PROBE_ETA, _CLOUD_MARGIN_FRAC, _CUSP_SPEED_REL_FRAC. MISSING: 'annulus'/
  'ANNULUS_INNER_RADIUS' (per spec diagnostic). Coverage gap; deferred.
- `_eps_for` in `_reprovision_w_nodes` (surrogate_training.py): when a tile
  straddles a carrier-basin flip, `_build_farfield_chart` raises
  CarrierDiscontinuityError — the probe cannot meaningfully measure
  interpolation error in this degenerate topology. Correct conservative
  default = keep full node density n_start (return None from `_eps_for`),
  never guess a reduction. CarrierDiscontinuityError is NOT in
  `_ENGINE_REFUSALS` (it's ValueError but not LensDomainError/
  SchwingerCertificationError/HypergeometricDomainError) so a SEPARATE
  except clause after the existing `_ENGINE_REFUSALS` catch is required.
- SUBDIVISION_RECURSION_AND_COORDINATE_CLEANUP PHYSICS VERIFICATION
  (2026-08-07, verdict PASS, 4-shard build): (a) generic `_subdivide_tile`
  bounded-recursion subdivider correctly reproduces the pre-refactor
  single-level far-field result at depth 1 (byte-identity pin held) and
  correctly extends to depth 2/3 for stubborn gaps (MAX_SUBDIVISION_DEPTH=3
  cap, ladder-served-gap flag on cap-out, never a crash);
  CarrierDiscontinuityError branch confirmed never recursed. (b) wedge
  recursion closes 3 measured marginal gaps (~6.50e-2 @r=0.633, ~6.70e-2/
  ~5.95e-2 @r=0.811) that sat above the 5e-2 bar at depth 1, down to
  max eps 3.0e-2 at depth 2; child boundary theta_split is confirmed the
  u-midpoint image (u=d**(2/3)), NOT the theta-midpoint. (c) theta_to_u/
  u_grid rename: `_validate_theta_to_u` correctly has NO magnitude/length-
  scale bound (only monotone + starts-at-0) since u=rad**(2/3) has a
  different magnitude than the true arc-length s carried by Tube/Lobe/
  FarField (theta_to_s, deliberately untouched); `_wedge_cusp_axis_map`
  correctly HARD-RAISES (never clamps) outside [0, pi/2]. (d) r_caustic
  brentq replacement matches the parametric caustic radius r(u)=|y(u)| to
  <=1e-10 at every tested gamma/theta including the pi/2 axis case
  (5.692099788303083, NOT the truncated SPEC literal 5.67376); waist
  invariant |r-gamma|<=1e-10 confirmed via an independent minimize_scalar;
  gamma=0 and parity-boundary/saddle-miss configs correctly raise the
  NAMED LensDomainError from inside the root-finder (no leaked
  ZeroDivisionError); measured 11.74x speedup vs the 1.85s dense-scan
  baseline. Full engine-training + sampling validation remains operator-
  deferred (COGWHEEL_TRAIN_TIER=1); no further physics concerns.
- SADDLE FORENSICS AUDIT (2026-08-08, Q1-Q4):
  - Q1 LOBE NORMALIZED-RADIUS DISEASE (PATHOLOGICAL): r_deltoid vanishes at
    deltoid cusps by |dtheta|^(1/3), the SAME power law as astroid cusps, so
    rho_lobe loses radial resolution near cusp directions. Milder than
    astroid only because deltoid lobes have smaller angular extent (~pi/3 vs
    ~pi/2 per astroid quadrant). Cure: either a cusp-adapted u=d**(2/3)
    coordinate (wedge pattern) or cusp carve-out + subdivision (pragmatic,
    loses coverage). Test: transverse rho_lobe cut at fixed theta_local
    ~2deg from the cusp ray; error grows rapidly as rho_lobe -> 1.
  - Q2 GHOST KERNEL PARITY GATE: the Morse reference exp(-0.5j*pi) is CORRECT
    for both parities — the fold is parity-blind (Fermat-potential reflection
    symmetry), and the merged saddle at any fold has the same Berry phase
    -pi/2 regardless of whether the merging pair is (min,saddle) or
    (saddle,saddle). NO parity-dependent branch needed. Verification: compare
    the ghost-kernel phase to the engine residual R = F_op - ppGO at a saddle
    config near the fold; phase alignment within a few degrees confirms the
    reference. (No code change shipped for this.)
  - Q3 LOBE CUSP CARVE-OUT CONSTANT: recommend a physical y-unit exclusion
    (~0.1-0.15 y-units from the cusp vertex) mirroring the exterior polar
    chart's 0.2 y-units, scaled for the deltoid's smaller extent. The
    separation-gate connection: near-cusp is where |tau_a - tau_c| -> 0,
    making E(w) non-smooth — both the spline and the SACR-C construction
    degrade there. RESOLVED in-build: the existing eta_max tube-shell
    exclusion already rejects near-cusp tiles (cusp vertices are in
    caustic_cloud), so NO explicit carve-out code shipped;
    `_LOBE_CUSP_EXCLUSION_DISTANCE=0.1` landed as documented dead code.
  - Q4 SADDLE EXTERIOR POLAR CHART: the scalar-reach rho = 1 + |y| -
    caustic_reach(gamma) is functionally correct as a coordinate frame
    (envelope smooth, drho/d|y|=1) but GEOMETRICALLY APPROXIMATE: rho does
    NOT align with the deltoid boundary directionally, so sources at the same
    rho but different theta_c can have very different physical proximity to
    the caustic. May need higher angular resolution. Test: served-vs-engine
    accuracy sweep with an angular-uniformity check across theta_c bins;
    monotonic decay of envelope magnitude along radial rays.
- ppGO ABOVE-CEILING RUNG (Build ppgo_above_ceiling, 2026-08-08): serving
  above the QD ceiling (w>150) via an ENGINE-INTERCEPT in
  `_amplification_coefficients` (before engine eval), gate w_max>150 AND
  w_lo*min_delta_tau>=RHO_END (4.0), whole-band via fold_ppgo_correction +
  reconstruct_farfield(FARFIELD_KERNEL_SUM). Error scale measured: ~1e-2 at
  w=150, ~1e-3 at w=500, decreasing trend; boundary-continuity is the
  primary gate. All-image serve (fold_ppgo_correction handles all 4 images).
