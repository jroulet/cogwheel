# Architect Short-Term Observations

Build 5 (lensed marginalization) plan checkpoint (2026-07-18):
- Professor: ship FULL extrinsic marg (CoherentScoreHM) directly; distance-only
  tier does NOT move wall-clock (per-proposal XPHM strain cost unchanged). HM
  (not QAS) respects the 22-only phase constraint structurally.
- Lensed timeseries seam = override `_get_dh_hh_timeshift`: per image a, modulate
  bin templates h_mpb by k0_a[b]*exp(i·2π·dt_a·f_b) (exp = exact timeseries shift
  via existing _d_h_weights; dt_a from _image_delays), contract with UNCHANGED
  _d_h_weights, SUM over images. Norm: multiply _h_h_weights by |F(w_b)|^2 where
  F_b = Σ_a k0_a[b] exp(i·2π·dt_a·f_b). Coherent score consumed UNCHANGED.
- Simplifier: A2 composition (internal LensedRelativeBinningLikelihood for
  `_amplification_coefficients`; wasted moment build is one-time) over mixin
  refactor of validated 1500-line likelihood.py. Reuse LensedPosterior as-is
  (already maps LensDomainError/CancellationError->-inf). New module
  cogwheel/lensing/marginalized_likelihood.py. Blob distance key = d_app (defer
  physical d_L=d_app*sqrt(mu_macro) to post-analysis). 2 new classes total.
- Refusals evaluated ONCE up front per proposal, propagate to -inf; never averaged
  inside QMC. Tolerances: full-marg |lnL_marg-oracle|<=0.3 nats (90th pct<=0.2),
  importance-sampling oracle (1e5 draws) not full grid; 8 seeded configs C1-C8;
  conditional draws = round-trip consistency (max lnL_full>=lnL_marg-0.3).
