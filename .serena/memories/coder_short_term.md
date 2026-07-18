# Coder Short-Term Observations

## WP2 LensedMarginalizedExtrinsicIASPrior (Build 5, 2026-07-18)

- New registered prior in `cogwheel/lensing/prior.py`:
  `class LensedMarginalizedExtrinsicIASPrior(RegisteredPriorMixin,
  CombinedPrior)`. prior_classes = `IntrinsicIASPrior.prior_classes` (reused,
  DRY) + [FixedLensGeometryPrior, UniformLensMassPrior, UniformReducedShearPrior,
  UniformSourcePositionPrior]. default_likelihood_class =
  LensedMarginalizedExtrinsicLikelihood. Exported from lensing/__init__.py.
- Mirrors LensedIASPrior but starts from INTRINSIC (extrinsic removed) so
  standard_params == marginalized likelihood params. VERIFIED (runtime import):
  registered in prior_registry; default_lik matches; prior_classes head ==
  IntrinsicIASPrior.prior_classes; UniformLensMassPrior precedes
  UniformSourcePositionPrior (m_lens_msun conditioning OK — registration itself
  proves conditioned_on empty); standard_params SET-EQUAL to
  sorted(MEL.params | _LENS_PARAMS) = the 12 intrinsic + 7 lens params.
- Static-equality note: compared prior.standard_params to sorted(MEL.params |
  _LENS_PARAMS) rather than a constructed likelihood instance's `.params`
  property (needs event_data). Algebraically exact: likelihood.params =
  sorted((wfg.params | _LENS_PARAMS) - (wfg.params - MEL.params)) and
  MEL.params ⊆ wfg.params → equals sorted(MEL.params | _LENS_PARAMS). UNVERIFIED
  against a real constructed likelihood instance (no test authored per role).
- LensedPosterior LEFT UNTOUCHED: does not hardcode a likelihood class;
  construction flows Posterior.from_event → prior.default_likelihood_class;
  delta_t_max threads via likelihood_kwargs exactly as the plain
  LensedRelativeBinningLikelihood (already-working LensedIASPrior path). Import
  circular-safety: prior.py now imports marginalized_likelihood (which imports
  likelihood + marginalized_extrinsic; no back-import to prior) — no cycle.

## WP1 LensedMarginalizedExtrinsicLikelihood (Build 5, 2026-07-18)

- New `cogwheel/lensing/marginalized_likelihood.py` subclasses
  `MarginalizedExtrinsicLikelihood` (NOT the Base directly — inheriting gives
  `params` class-attr + `_create_coherent_score` for free = exact mirror, DRY).
  Exported from `cogwheel/lensing/__init__.py`.
- WP TEXT WAS FACTUALLY WRONG about kernel layout: it assumed engine k0 is
  per-EDGE (len(fbin)) with channel as LAST axis. VERIFIED via find_symbol:
  `_engine._amplification_coefficients(par_dic)` returns
  `(delays[s], k0, k1, partition)` with k0/k1 shape (n_channels, n_bins=
  len(fbin)-1) at bin CENTERS, channel FIRST. Coherent-score `_d_h_weights`
  (mtdb) / `_h_h_weights` (mdb) are EDGE-indexed b=len(fbin). Bridged the
  center↔edge mismatch with helper `_edge_amplification(delays,k0,k1)`:
  reconstruct each image kernel K_a at edges from the certified (k0,k1) linear
  model (slope-correct to edges, average adjacent-bin estimates at interior
  edges), then F(f_b)=Σ_a K_a(f_b)·exp(2πi·dt_a·f_b). This is the load-bearing
  deviation from WP literal text — flagged in change report for Inspector/TestDev.
- Engine built in overridden `_set_summary` (runs inside base __init__ via fbin
  setter, BEFORE terminal lnlike(par_dic_0)), NOT in __init__ body — so
  self._engine exists before the base constructor's terminal lnlike call.
  delta_t_max/bin_delay_tol/kernel_subsamples stored as same-named attrs for
  JSONMixin.get_init_dict round-trip (engine NOT an init arg — rebuilt).
- `params` computed from self.waveform_generator (not self._engine — engine not
  built when params first read): sorted((wfg.params | _LENS_PARAMS) - dropped),
  dropped = wfg.params - MarginalizedExtrinsicLikelihood.params.
- Data term uses F·h (h_lensed.conj()); norm uses |F|²·_h_h_weights reusing base
  einsum with UNLENSED h_mpb (F is mode-independent scalar). Proved on paper:
  image-sum before linear contraction == after; delay phase in conj(F) combines
  with weights' exp(2πi f t) → per-image time shift t-dt_a (exact).
- Refusals propagate unswallowed: call `_amplification_coefficients` +
  `_check_candidate_delays` with NO try/except (LensDomainError/CancellationError/
  LensedBinningError reach posterior boundary).
- VERIFIED: py_compile OK; `from cogwheel.lensing import
  LensedMarginalizedExtrinsicLikelihood` imports; MRO head correct; 3 overrides
  present; __init__ sig matches WP.
- UNVERIFIED (no test authored/run per role): numerical lnlike vs brute/direct;
  pickle round-trip drops engine._fid_cache (relied on engine __getstate__, no
  explicit __getstate__ added); JSONMixin full JSON round-trip.
