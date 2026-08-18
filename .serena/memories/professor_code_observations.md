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
  UPDATE (2026-08-13, saddle_above_ceiling_serving consult, read directly
  off current channels.py ~L963): `farfield_ghost_term` now gates on TWO
  frequency-INDEPENDENT conditions, not one — decay `Im(tau_c) >=
  _GHOST_DECAY_IM_THRESHOLD = 0.4` AND separation `min|x_a-x_c| >=
  _GHOST_SEPARATION_MIN = 0.7` (both raise GhostDomainError on miss). This
  is NOT the old retired decay gate (that one was frequency-DEPENDENT,
  `w_min*Im(tau_c) >= 2.0`) — a distinct, frequency-independent decay gate
  was added back at some point after 8h-d1. As a tier-1-rung admission
  PROXY it is unreliable: it can REFUSE (raise) exactly on far-from-caustic
  sources a resolvability-based rung wants to admit (ghost well-separated
  AND decayed there, but a raise is not a magnitude); prefer gating a new
  rung on post-gauge switch/resolvability saturation instead of routing
  through this function as an |E| proxy.
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
  SUPERSEDED 2026-08-08 (98c4e7f): the sqrt-edge axis (A2-designed) was
  WRONG for the deltoid's A3 cusps; the lobe interior now uses the
  cusp-adapted u = d**(2/3) axis (see the LOBE CUSP-ADAPTED COORDINATE
  PHYSICS entry below; also `mem:lobe_interior_chart`).
- LOBE CUSP-ADAPTED COORDINATE PHYSICS (Build lobe_cusp_coordinate,
  98c4e7f, verdict PASS w/ 1 non-blocking concern): (1) 2/3 exponent is
  UNIVERSAL for A3 — deltoid lobe cusps are A3 fold-cusp singularities
  (same catastrophe class as astroid cusps); u = d**(2/3) is the exact
  gamma-universal caustic-reach scaling, while the retired sqrt-edge
  exponent 1/2 was designed for A2 fold edges, not cusps. (2) Smoothness:
  rho_lobe normalizes by the deltoid radius (smooth at cusps, rho ~
  1 + O(dtheta)); u absorbs the d**(2/3) term in caustic reach, removing
  the d**(-1/3) derivative singularity a raw-theta spline would see at the
  cusp vertex — (rho_lobe, u) is smooth everywhere in the lobe interior.
  (3) `_lobe_cusp_axis_map` mirrors `_wedge_cusp_axis_map`: uniform-in-u
  np.linspace, node-exact endpoints explicitly pinned, offset so
  u(theta_lo)=0, monotone d->u->theta inverse on both sides, np.clip FP
  guard on the 'right' side. (4) u-midpoint subdivision correct:
  `_lobe_child_boxes` computes u_mid from the parent's map and splits
  children at equal u-range via np.interp inverse; angular children have
  UNEQUAL theta-widths (near-cusp child narrower) — correct for a cubic
  spline in u. (5) Schema hard-refuse correct: both old tags removed from
  `_KNOWN_LOBE_AXIS_SCHEMAS`, `_validate_lobe_axis_schema` rejects them at
  load (tests cover both old tags, None, unknown) — no silent degradation.
  (6) Carve-out retirement correct: `_LOBE_CUSP_EXCLUSION_DISTANCE` removed
  because the cusp-adapted coordinate now handles near-cusp tiles; the
  eta_max nearest-caustic-distance test in `_SaddleLobeAdmission.admits`
  already excluded near-cusp tiles — a separate carve-out was always
  redundant. CONCERN (non-blocking, latent trap for external callers):
  `_chart_from_npz` UNCONDITIONALLY accesses data['theta_to_u'] (KeyError
  if absent) but `_chart_to_npz` only writes theta_to_u when not None —
  the raw-theta fallback path (cusp_angle=None) produces charts with
  theta_to_u=None that CAN be built but CANNOT survive an NPZ round-trip.
  Not triggered in the current training pipeline (all tiles carry cusp
  angles). Mitigation: tolerate missing theta_to_u in `_chart_from_npz`, or
  raise a clear error in `_chart_to_npz` when saving a theta_to_u=None chart.
- EXTERIOR-POLAR CUSP-ADAPTED COORDINATE PHYSICS (Build
  exterior_polar_cusp_coordinate, 1a97bbd, 2026-08-08, verdict PASS):
  ExteriorPolarChart gains an OPTIONAL cusp-adapted u = d**(2/3) axis
  (`theta_to_u`), applied on parity==1 (astroid) tiles via `_wedge_cusp_
  axis_map` and integrated via np.interp in `_evaluate_chart` (mirrors the
  wedge pattern); the macro-saddle exterior (parity==-1) keeps raw-theta
  (None). Served values agree with raw-theta charts within tolerance — the
  coordinate change is an ACCURACY improvement, not a model change.
  Mutation-falsification tests confirm the remap is load-bearing (not dead
  code). theta_to_u=None falls through to the raw theta_c_grid — all
  existing tests byte-identical (backward compat). NPZ round-trip preserves
  theta_to_u bitwise (max|diff|=0). Schema 'exterior_polar_rho_theta_c'
  hard-refused; new tag 'exterior_polar_rho_u_v1' loads with optional
  theta_to_u. Grid-node served values match training within 1e-7. Census
  classification handles theta_to_u-bearing charts; subdivided children
  propagate theta_to_u; edge-case rejection (bounds, monotonicity)
  operational. NOTE: unlike the wedge v3 / lobe v1 charts (theta_to_u
  REQUIRED, read unconditionally), the exterior-polar field is OPTIONAL —
  a mechanical "REQUIRED" copy would be wrong. Deferred to operator
  post-build: the train-tier cusp-adapted test classes
  (BuildFarfieldPositiveParityCuspAdaptedTestCase & siblings) and the
  training accuracy sweeps (COGWHEEL_TRAIN_TIER=1).
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
- CUSP PPGO FAST RUNG (Build cusp_ppgo_high_w, 2026-08-08/09, verdict
  CONCERN — structural guards PASS, numerical calibration owed): a fast rung
  inside `_pearcey_cusp.cusp_amplification` (after radius=math.hypot(x,y))
  that serves via fold_ppgo_correction once R >= r_ppgo_min AND w >=
  _W_PPGO_FLOOR. Gate math: r_ppgo_min = (_R_PPGO_ERROR_CONST *
  _UNIFORM_ERROR_CONST / bar_ppgo)^(2/3) with exponent 2/3 correctly
  inverting the R^(-3/2) cusp-proximity error scaling; _R_PPGO_ERROR_CONST
  = 50.0 originally (PROVISIONAL — post-build driver calibration owed),
  progressively tightened 2026-08-12 to 1.0 (see coder/architect knowledge);
  _PPGO_BAR_DIVISOR = 10 -> bar_ppgo = 0.005. _W_PPGO_FLOOR=50 independently
  gates kernel-truncation error O(1/w^3). Finiteness guard
  np.isfinite(abs(result)) catches all 4 NaN/Inf variants; do-nothing
  control byte-identical at intermediate R; self-falsification corrupts
  both gate directions. Asymptotic soundness: the rung is FOLD-corrected
  (Airy, fold catastrophe) NOT cusp-corrected (Pearcey, cusp catastrophe) —
  sound because at large R Airy -> geometric image sum matches
  Pearcey -> geometric image sum, and the conservative R gate ensures the
  rung is deep enough. CONCERN: |ppGO - pearcey|/|pearcey| < 0.005 agreement
  is NOT asserted in tests (docstring records the delegation). OWED post-
  build calibration: sweep R at fixed w, measure |ppGO-pearcey|/|pearcey|
  at the worst-case direction, fit the exponent, tighten
  _R_PPGO_ERROR_CONST. Pre-existing (NOT ppGO-induced): the ppGO gate fails
  `test_moving_error_const_threshold_flips_a_fixed_node`'s config on both
  branches (r_ppgo_min ~25x the radius at the low-const setting) — the
  timeout is a broader-suite performance issue.
- EXTERIOR 2D (rho, u) FOLD-CARRIER PHYSICS (Build exterior_2d_fold_carrier,
  2026-08-10, verdict PASS): the fold-carrier is the fold-merge-point delay
  Re(tau_c(rho, u)) stored at EVERY spline node as a 2D (n_rho, n_theta_c)
  array; per-node median over gamma is correct, w_grid[0] probing is
  sufficient, tabulate (no linear fit). Carrier uses RAW absolute delays —
  demodulate ONLY Re(tau_c); a full-complex e^{+w*Im(tau_c)} remodulation
  explodes (~19x at w=30). Serve re-modulates the delay bilinearly at the
  query u-coordinate after the theta_c -> u map — never raw theta_c.
  Coordinate facts: exterior rho is ADDITIVE, rho = 1 + |y| - r_caustic(gamma)
  (macro-saddle exterior uses the same additive scalar reach); u is the
  spline axis. SCHEMA DISAMBIGUATION: `exterior_polar_rho_u_v1` = cusp-
  adapted u coordinate (theta_to_u, earlier build); `exterior_polar_rho_u_
  carrier_v2` (V5) = the 2D fold-carrier field (current write tag); V4
  `exterior_polar_rho_log_carrier_v1` = the superseded 1D rho-carrier
  (loads by broadcast to 2D). Tolerances: node-exact 5e-13, off-grid phase
  1e-3 rad, NPZ bit-for-bit, self-falsification 10x, heldout eps 4e-3.

- SADDLE EXTERIOR CUSP-ADAPTED COORDINATE PHYSICS (Build
  saddle_exterior_full_treatment, 238d21e, 2026-08-10, verdict PASS — 32/32
  across 6 specs, no numerical concerns, no tolerance-edge results): A3
  universality across astroid AND deltoid cusps is now CONFIRMED NUMERICALLY
  on the saddle exterior: the cusp-adapted u=d**(2/3) map absorbs the
  d**(-1/3) derivative divergence on macro-saddle (gamma>1) exterior tiles
  exactly as on positive-parity astroid tiles (Spec B accuracy: chart A
  median eps beat raw-theta chart B by >=2x on 50 held-out points, chart A
  <=1e-3 production bar; Spec E serving: cusp-adapted eps <= raw-theta eps,
  non-degradation). PARITY GATING is physically correct: saddle coverage=0.0
  refuses ALL cusp-window interior queries (mid-window + near-cusp) because
  the deltoid cusp arm is NOT near the true source in saddle parity —
  deep-interior images can sit arbitrarily close to the cusp (F018), so the
  astroid 0.07 rad shrink does NOT transfer; positive coverage=0.07 admits
  only within the shrink margin. SERVING GEOGRAPHY (ruling): deltoid straight
  edges (fold arcs) + the inter-lobe corridor are correctly served by the
  existing ladder — exterior charts cover the exterior, lobe-interior charts
  cover lobe interiors, corridor falls through to the exact engine; no new
  code needed. Round-trip: theta_to_u shape (2,>=100), strictly increasing
  rows, u_fine[0]~0, endpoints match 1e-12, np.interp round-trip reproduces
  u_grid within rtol*max(u_grid); mismatched-row detectable (~5e-5).
  Full-sampling validation (COGWHEEL_BRUTE_ACCURACY / COGWHEEL_STRICT_TIMING)
  remains operator-deferred (beyond fast-test scope).
- PPGO RESOLUTION GATE PHYSICS (Build operator_routing_one_home, 2026-08-11,
  verdict PASS — PpgoRungSelfFalsificationTestCase 3/3 in 5.06 s; broader
  lensing suite 199 passed, 23 skipped, 4 xfailed, no regressions): the
  cusp_amplification ppGO rung now requires (_merging_fold_pair(...) is not
  None) OR (w*delta_min >= _PPGO_RESOLUTION_GATE = 4.0, mirroring
  operator.RHO_END). Measured on the saddle fixture _PPGO_SADDLE_SOURCE
  =(-0.5,0.5) at gamma=1.2: 2 images, delta_min = 0.644, nearest.distance
  = 0.389 > _ETA_MAX_FOLD=0.3, no merging fold pair; at w=500
  w*delta_min = 322 >> 4.0, so the gate ADMITS naturally. The spec's
  w*delta_min~1.9 estimate was a copy-paste error from a different
  configuration (delta_min=0.644, not ~0.0038); saddle sources always
  resolve at w>=50. _merging_fold_pair returns None for dual-saddle
  2-image sources, making the resolution gate the SOLE admission criterion
  there. Test proves teeth by inflating the gate to 1000 (blocks at w=500),
  lowering to 0 (always admits), and w=20000 with gate=1000 (12882>=1000,
  admits) — variable isolation clean: same w for admit/refuse branches,
  only the gate threshold varies (STRONGER than the spec's intended
  different-w scenario).

- LOBE_EXTERIOR REGION FILTER TAG-DECODE + COST-PARITY (2026-08-12, Build
  lobe_exterior_region_wiring, verdict PASS): the real emitted farfield tag
  infix for lobe_exterior tiles is `_fflobeext_` (surrogate_training.py
  ~L5613, `chart_{label}_s{si}_fflobeext_{i}_{j}`) — CONFIRMED against the
  shipping code, not assumed. `_tag_kind`'s decoder must check
  `_fflobeext_` BEFORE `_ff_` (ordering matters: `_fflobeext_` is not a
  substring collision with `_fflobe_`/`_ff_` since both need a trailing
  '_' that `_fflobeext_` doesn't supply at that position, but decode order
  is still the safe convention to preserve). lobe_exterior carries the SAME
  per-region training cost weight (1 eval/(gamma,w)) as lobe_interior in
  `_self_estimate`'s per_region dict — confirms the two lobe-adjacent
  regions are priced identically, useful context if a future build adds a
  6th region and needs a cost-model precedent.

## 2026-08-13 (saddle_above_ceiling_serving design consult)

- `reconstruct_farfield` with `FARFIELD_KERNEL_SUM` does NOT re-gauge the
  switch: `_farfield_switch` (channels.py ~L876-914) HARDCODES S_a=1 on
  real channels and tau_c=0 — a zero-envelope FARFIELD_KERNEL_SUM serve is
  therefore a BARE-KERNEL sum (S_a=1), not the re-gauged switched-channel
  sum; the two are numerically equivalent ONLY where every per-channel
  switch is saturated (S_a≈1), which is exactly what an admission gate
  built on this reconstruction path must certify.
- `switched_analytic_channels` (_gauge.py ~L335, the SACR-C projection) is
  4-arg with a per-channel switch S_j; the critical delay tau_c enters
  TWICE — as the switch argument AND as the demod carrier_c
  (E = conj(carrier_c)*(F - sum carrier*trial)). To decouple a
  switch-saturation delay from a phase/demod delay, compute `switch` from
  one delay value but pass a DIFFERENT `critical_delay` to this function
  for the demod carrier — the two roles are independently steerable.

## 2026-08-13 (fold_exterior_ghost review + ppGO interior certificate)

- EXTERIOR POSITIVE PARITY HAS EXACTLY 2 REAL IMAGES (a Morse-0 minimum and
  a Morse-1 saddle) and NO genuine merging pair — the pair that merges at
  the caustic has gone COMPLEX. `_merging_fold_pair` therefore returns the
  FAR pair there and the Airy fold correction is spurious; the correct
  contract is refuse. Interior (4 real images) is unaffected. This makes the
  REAL-IMAGE COUNT the exact interior/exterior discriminator for both
  parities — no radius, rho or gap test is equivalent.
- GHOST CARRIER SIGN IS PHYSICALLY DISCRIMINATING: served =
  `geometric_amplification + ghost.kernel * exp(1j*w*tau_c)` with tau_c NOT
  conjugated. With Im(tau_c) >= 0.4 > 0 that gives |carrier| =
  exp(-w Im tau_c), a decay; the conjugate would give exp(+w Im tau_c), a
  blow-up. The sign pin is an internal-consistency identity (no external
  oracle needed) and it locks the convention against refactors.
- ADMISSION GATES FOR A RUNG WHOSE VALUES BECOME TRAINING LABELS MUST BE
  FREQUENCY-INDEPENDENT (ruling, 2026-08-13): a w-dependent floor re-opens
  the train/serve skew that build 8h-d1 retired, because the trainer and the
  server see the rung at different w. Two config-geometry gates
  (Im(tau_c) >= 0.4 decay, min|x_a - x_c| >= 0.7 separation) partitioned the
  domain with zero overshoot, so no floor was needed. NOTE: on the measured
  grid the separation gate never bound (2.0-3.6 >> 0.7) — the decay gate was
  the sole active discriminator, and admission tracked Im(tau_c), not the
  |y|/r_caustic band it correlates with.
- ON A TRUE CAUSTIC INTERIOR THE GHOST IS EXACTLY ZERO (`GhostAbsentError`)
  — an interior rung must carry no ghost term, and an interior config is the
  right place to assert that the ghost machinery declines rather than
  returning a small number.
- REVIEW SCOPE FOR A NEW ARM RUNG: fast tests pin STRUCTURE, DECISION and
  SIGN only; the value-vs-oracle accuracy sweep (1e-2 arm bar over the
  serving band) is the expensive operator ship gate, reported not committed.
  Do not ask a fast suite to carry the accuracy certificate.

## 2026-08-14 (symmetry_tie_c3_admission — saddle far-field c3 certificate, verdict PASS)

- `_saddle_farfield_analytic_serves` (likelihood.py) re-keyed to a c3-led
  certificate: `est=ppgo_error_estimate(real_images,source,matrix,w_lo)`;
  `est is None` -> refuse; admit iff min pairwise image separation>=0.05
  (defense-in-depth backstop) AND 20*est<=1e-3. Independently reproduced:
  tied mirror pair (gamma=2, y=(1,0)) has delta_tau=0.0 EXACTLY (y->-y
  mirror symmetry) yet sep=1.041>>0.05 and S*est=7.63e-4<bar -> SERVES
  (HEAD's old `delta_taus>0` leg gave product 0<4 -> wrongly refused this
  transverse-cone regression). Certificate log-log slope in w_lo =
  -3.000000 exact (matches est=sum(sqrt|mu||c3|)/w_min**3 asymptotics).
- MERGE CASE (gamma=1.6, rho=1.001): est is FINITE-but-astronomical
  (1.57e15, S*est=3.1e16>>bar) — refuses via the CERTIFICATE, not the
  separation backstop (sep=2.07 there, well clear). The literal ask
  "ppgo_error_estimate is None at the physical near-fold" is NOT met
  physically — the DD root finder keeps the merging image just off the
  exact critical curve, so est stays finite; the None branch is only
  reached via the degenerate w_min<=0 trigger. Intent (merging pair
  refuses via the certificate) is satisfied; documented as an honest
  handling gap, not a defect.
- Gate matches the 2026-08-14 consult ruling verbatim: S=20, bar=1e-3 at
  w_lo, c3-only (no ghost term), separation floor 0.05 purely as
  defense-in-depth (never the active discriminator on the measured grid).
  Census mirror confirmed calling the SAME `_saddle_farfield_analytic_
  serves`; both sites build `real_images = np.asarray(geom.images)`
  directly (the INS-1-001 double-mask bug — indexing an already-real-only
  images array with the length-4 channel mask — fixed at both the live
  rung and the census mirror).

## Born residual — first-class intercept (2026-08-14, distinct from the older buried rung)

- NEW rung `_born_residual_analytic` in `_amplification_coefficients`
  (likelihood.py) is a FIRST-CLASS intercept: gated kappa==0 AND beta==0
  AND caustic-frame rho>2.0 (both parities per spec), band-split against
  the certified ppGO map, consulting `born_chart.covers(gamma, rho,
  chart_w)`. DISTINCT from the OLDER buried `_surrogate_coefficients` Born
  rung (rho>1, still present, still consulted — not retired). The buried
  rung is a strict SUPERSET of the new intercept's served domain, so
  census's chartless 'born' fallthrough bucket (rho>1) over-attributes
  conservatively and cannot distinguish the two — confirmed correct
  behavior (WP-F), not a gap.
- TRAINED ARTIFACT IS ASTROID-ONLY IN PRACTICE (INS-3-001): despite the
  gate accepting both parities, the shipped BornResidualChart npz has
  gamma_grid all <1.0 (no saddle node) — a saddle draw always falls
  through to the exact engine via `covers()` refusing gamma>0.9. Doc says
  "both parities"; artifact says astroid-only. Code is safe either way
  (covers() gate protects); this is a doc/artifact currency gap, not a
  serving bug.
- Auto-attach: `_AUTO_BORN_CHART` sentinel loads at construction for both
  LensedRelativeBinningLikelihood and LensedMarginalizedExtrinsicLikelihood
  (the latter's internal engine builds at this same default); load
  anomaly -> None + RuntimeWarning (engine stays pure), never a raise past
  construction.


## F080 saddle rho<1 per-cell relaxation review (2026-08-14)
- `get_certified_ppgo_map()` returns the process-global singleton, which is
  `None` in a fresh process — any oracle/test code must call
  `CertifiedPpgoMap.load()` directly rather than the accessor when no prior
  code path has populated the global.
- `CertifiedPpgoMap` allowlist relaxation (`_SADDLE_RHO_RELAXED_CELLS`) is
  keyed on exact float64 gamma-edge equality; verified in-box Cell 1 gives
  w_cert=19.164305537818887, w_trust=max(1.5*floor, floor+2)=28.74645830672833,
  w_ceiling=58.0 (finite, >= w_trust), while MARGINAL/CONTAM/off-band/edge-
  neighbor cells all still return UNKNOWN across w_cert/w_trust/w_ceiling.


## F081 saddle tube fundamental training review (2026-08-15)

- 6 detected deltoid arcs collapse to 2 D2-orbit representative arcs on
  the real SADDLE_BAND fixture (orbit sizes {4,2}), derived via an
  independent union-find (_circular_gap/_d2_gauge_images), NOT the
  a-priori guess of 3 — always derive orbit count from the actual
  partition, never assume the naive symmetry-order division.
- arc_r_min anisotropy on this fixture: [0.399, 9.156] (~23x range);
  f_max=0.4 => min_eta_max=0.160, max_eta_max=3.66. corridor_half is
  correctly keyed to 1.0*min_eta_max (NOT max) confirming the F081 fix.
- Serve-coverage equality pin (moral-imperative symmetry check): the
  fundamental-domain served set (2 orbit reps) is a SUPERSET of the
  all-6-arc incumbent serve set over a 720-angle ring sweep, 0
  violations — this end-to-end coverage pin backs the 6->2 arc-count
  collapse even if the internal orbit bookkeeping were subtly off.


## 2026-08-15 (lobe cusp-coincident-edge tolerance review, verdict PASS)

- `_lobe_cusp_axis_map` edge-coincidence fix (surrogate.py,
  `_CUSP_EDGE_COINCIDENCE_ULPS=8`) verified physically sound: u=d**(2/3)
  is the A3 cusp caustic-reach scaling; a tile edge landing on a cusp ray
  clamps d->0 exactly (`max(theta_lo-cusp,0)` left / `max(cusp-theta_hi,0)`
  right) and anchors u there. `np.clip(base_lo-u_fine,0,None)**1.5` on the
  right guards the power against FP-negative base (would NaN). Symmetric
  tolerance on both edges.
- Independent numeric checks performed: Pin A (endpoints bit-exact,
  u_max==0.5**(2/3) to 0 rel-err, strictly increasing); Pin B (7a sliver
  tl=0, th=3.5527e-16, ca=3.2703e-16: ca<th holds so real straddle
  premise, no raise, endpoints bit-exact, NON-decreasing — linspace nodes
  collide at ~3.5e-16 so >= not > is the correct monotonicity invariant
  here); boundary trichotomy (tol band = 8*eps*max(1,|edge|)~1.776e-15 at
  this scale; 2e-17 offset admits, 1e-3 offset raises) on both sides;
  caller path `_lobe_child_boxes` coincident-lower-edge -> side='left' ->
  4 children with split in [theta_lo,theta_hi], while an interior cusp
  still propagates ValueError through the splitter (guard not vacuous).
- Test evidence: test_lensing_surrogate_lobe.py 19 pass in 4.4s
  (LobeCuspAxisMap*/EdgeCoincidence*/LobeChildBoxesCoincidentEdge*);
  test_lensing_lobe_subdivision.py 49 pass in 112s. Heavy full-engine
  training + sampling (COGWHEEL_TRAIN_TIER=1) left operator-deferred, not
  needed for this verdict.

## 2026-08-17 (serve_route_census demand census review, verdict PASS)
- Direct census eyeball (seed=0, n=150, gauge=caustic_rho not rho_lobe):
  route_counts sum=150 (MECE); surrogate=0; saddle_c3=1; born_analytic=20;
  engine_residual=129 (split: near_caustic_tube=28, interior=101,
  born_chart_demand=0 by construction since rho>2 is skimmed upstream by
  born_analytic); engine_refused=0 at this scale (must be reported
  empirically per-run, never asserted a priori — no "~59%" invariant).
- Saddle finite-but-huge c3 in this census: production
  ppgo_error_estimate~4.8e15 (finite, not None); safety(20)*est >>
  bar(1e-3) so the REAL gate refuses (route=engine_residual, not
  saddle_c3) — confirms the F069/F074 "gate bounds safety*est, not
  est-is-finite" fix holds under a fresh oracle probe.
- Engine-free guarantee reconfirmed via 4 door sentinels (evaluate,
  f_schwinger, _f_schwinger_mpmath, mpmath.gauss_quadrature) all outside
  the caught refusal tuple; call_count==0 on every door after a real run.

## 2026-08-17 (tube beat-free representation, _tube_f_ref review)
- ERROR CURRENCY (F_ref vs |exact_total|): an accuracy sweep on the
  beat-free residual r=E/F_ref must be normalized by F_ref, not by the
  raw |exact_total| — F_ref (q=p uniform-Airy reference, non-vanishing
  by construction) stays finite where the raw total can pass through an
  unrelated old-carrier Airy zero, which would otherwise produce a false
  failure signature at points uncorrelated with actual interpolation
  error. Same measurement doubles as both the accuracy-sweep gate and
  completion-record acceptance evidence.
- Two distinct "F_ref is None" paths for _tube_f_ref/_tube_serves: a
  BUILD-side None (no distinct fold partner for the uniform-Airy
  reference at that point) is benign/expected; a SERVE-side None is a
  guarded RuntimeError and never legitimate. Do not conflate the two.
- Reusable audit priority for `_tube_f_ref` correctness: (1) tau-frame
  cancellation — E and F_ref must share the SAME t_min/tau_c origin, a
  constant-offset bug round-trips exactly at build nodes and only leaks
  OFF-node (interp sweep catches it, unit tests at nodes don't — hardest
  to catch, look here first); (2) zero-fill vs serve-None — serve must
  not silently interpolate across an isolated zero-filled node; (3) r
  stays finite/smooth across Airy-zero crossings away from build nodes.


## 2026-08-18 (low_w_diffractive_rung, saddle-vs-positive-parity scope ruling)
- No parity-agnostic Fermat-delay-moment asymptotic series exists at ANY
  order for a lens with real images: tau(x) grows quadratically at large
  |x|, so INT tau^n d^2x diverges for every n>=1 — the low-w end has no
  stationary-phase structure to anchor a series (unlike the high-w end,
  which localizes at isolated stationary points). The only valid low-w
  analytic object is the point-mass-core analytic continuation
  G_PM = C(w)*1F1(...), which only converges for gamma'<1 (positive
  parity, inside the parity-wall branch point).
- Saddle (gamma'>1) therefore has no genuine low-w series; its correct
  "analytic" object is the 1D Schwinger integral itself (exact at all w,
  validated 2.2e-15 vs the 2D rotated-contour oracle), self-certified via
  paired N/2N quadrature (tol 3e-10) — a quadrature remainder is as valid
  an analytic certificate as a series-truncation remainder ("series vs
  quadrature" is not a physics/certificate distinction). Serving positive
  parity through Schwinger instead of its convergent series would be
  strictly worse (loses the w=0 reach, imposes an artificial ~w=60 ceiling
  the series doesn't have) — keep both objects, don't unify.
- NOTE: this session's professor short-term memory requested this ruling be
  filed under professor/microlensing_chang_refsdal; Dreamer's standing rule
  is to never write professor/* topic memories (Professor-curated only), so
  it landed here in code_observations instead — next time the Professor
  agent visits that topic, it should self-file the physics ruling above.
