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
- `likelihood.py` post-3f: `_amplification_coefficients` interpolates
  ONLY the envelope E(w) on a LOO-adaptive coarse grid (seed 8, stop
  4e-3, ceiling 48; measured N=26-32), with closed-form switched-saddle
  reconstruction. Post-3g a candidate/fiducial ratio layer (lattice-
  snapped fiducials, `_fid_cache`, health/image-count guards, fallback
  to certified direct) brings warm lnlike to 9.8 ms; the engine 1F1
  ladder still dominates residual cost.
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
- Build 4 sampled coordinates (`lensing/prior.py`): ln_m_lens_msun
  uniform (ln 10, ln 3500); reduced shear gamma in [0,0.45] identity-
  transform; mass-conditioned source y = u*min(307/m, 3.0) with
  folded_reflected u1,u2 (NO phase fold under XPHM); kappa=beta=z_lens
  fixed 0 via FixedPrior. prior.standard_params == likelihood.params.
- C7 measurement (Build 4 review, verdict CONCERN): only 41.2%
  (206/500) of prior draws finite — the gamma prior overlaps the
  gamma_eff~0.5 cancellation band; all non-finites are exact -inf
  (0 NaN); near-truth reference lnpost 260.6 dominates best random draw
  18.1, so the peak sits at truth. Efficiency-only concern; operator to
  decide whether to bound gamma away from the band before the heavy
  sampling ship gate.
- Cross-parity Schwinger (Build 7a consult): the `_schwinger` engine is
  signature-AGNOSTIC. `_h_dd` takes da_im=-w*a/2, db_im=-w*b/2 as pure-
  imaginary offsets (`_ddc_sqrt` via Newton from a float64 seed); the
  real-t contour stays clean for both parities, so the `gamma_prime>1`
  guard in `_validate_inputs` is POLICY, not a math necessity (relaxing
  to >0 or !=1 changes NO arithmetic path). Cancellation law L_S=pi*w/4
  holds on BOTH parities — it comes from the `t^{iw/2-1}` factor
  (|1/Gamma(iw/2)| ~ e^{+pi w/4} cancelling the ~e^{-pi w/4} integral),
  never references the signature; dd paired-rule certification catches
  degradation either way. Positive-parity mass-sheet fallback formula =
  (1/lam)*exp[0.5j*w*ln lam - 0.5j*w*kappa*s] * f_schwinger(w,y_eig,
  gamma') with gamma'<1 — EXACTLY `_saddle_grid`'s formula (f_schwinger's
  `_reconstruct` already carries e^{iw|y_eig|^2/2}; note `_grid_certified`
  instead multiplies an EXTRA exp(0.5j*w*s) because its pure-shear G
  kernel excludes that e^{iw|y|^2/2} factor — G is the operator-series
  total, not F).
- Index-theorem guard (Build 7a): unified invariant sum_a sign(mu_a) ==
  sign(det A) - 1 (positive parity -> 0, saddle -> -2). No maxima
  possible since tr Hess = 2*lam > 0, so the image count {2,4} is
  redundant given signed-sum + no-maxima (cheap to assert for debugging).
  `morse_index` uses eigvalsh with strict `< 0.0`; a degenerate fold
  image (one eigenvalue ~0) counts non-negative -> morse 0 or 1, signed
  sum still holds; only breaks if BOTH eigenvalues are near zero at once
  (a cusp, three-image merge).
