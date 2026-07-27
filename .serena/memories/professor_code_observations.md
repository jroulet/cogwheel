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
- `lensing/chang_refsdal/_born.py` (Build 8h-c1) is DORMANT and UNWIRED:
  F_born = sqrt(mu_macro)*exp(iw phi_geo)*(1 + i(w/2) b1/Q2r + O(w^2/Q2r^2)),
  expanded about sqrt(mu_macro) = 1/sqrt((1-kappa)^2 - gamma^2), NOT about 1.
  OWED (mine): the O(1) numerator b1 is a placeholder = 1.0 at a single edit
  site — the closed form is nowhere in-repo, and 229/229 gate-passing annulus
  configs disagree with `operator.F_op` by up to 13%. Guard A is calibrated
  from the same b1, so it cannot refuse the error b1 causes. The w->0 macro
  limit F -> sqrt(mu_macro) IS b1-independent and exact, so it stays a valid
  green oracle while the accuracy gate is red — don't conflate the two.
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
- Build 8h-b6 cusp-alignment review: `EnvelopeReconstructionTestCase.
  test_positive_box_reconstruction_within_budget` stayed RED after the
  cusp-aligned-tiler fix because that test's `_train()` fixture builds via
  `LensAmplificationSurrogate.from_engine()` — a single-box surrogate with
  a UNIFORM theta_c grid that never calls the production tiler
  (`_train_band_charts`/`_farfield_exterior_tiles`) the fix landed in. A
  tiler-level fix cannot move an eps measured on a fixture that bypasses
  the tiler entirely.
