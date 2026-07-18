# Test Dev Short-Term Observations

## test_lensing_marginalized_likelihood.py (WP1/WP2 coherent-score marg)
- Suite of 9 specs, all green (21 tests). Env: cogwheel-newlal (NOT
  cogwheel_310 — that env is gone). Build ~19s; amortize via lru_cache
  `_harness()`; marg.lnlike ~131ms, plain.lnlike ~25ms.
- Marg lnlike is QMC-STOCHASTIC, NOT bit-repeatable. Pin determinism at
  the deterministic `_get_dh_hh_timeshift` layer (assert_array_equal), not
  lnlike. JSON round-trip: gate the deterministic layer bit-for-bit.
- Spec-3 (importance-sampling oracle, 2-3e4 plain evals x3) ~30min =
  INFEASIBLE as a minutes gate; implemented the deterministic specs
  (1,2,5,7,8,9) solidly + spec-6 conditional-draw round-trip instead.
- Spec-6 CRITICAL: cogwheel normalization puts plain lnlike at a single
  extrinsic draw ~25-30 nats ABOVE lnL_marg (extrinsic Occam factor). The
  raw spec's UPPER bound "no draw > lnL_marg+0.5" is WRONG — implement as
  a LOWER bound (max>=lnL_marg-0.3, low-pct>=lnL_marg-0.5). A biased
  sky/dist/time conditional fails the lower direction.
- Independent oracle: `LensedWaveformGenerator(...).amplification(fbin)`
  returns ChangRefsdalChannels.exact_total — a DIFFERENT path from the
  production per-bin edge kernel `marg._edge_amplification(delays,k0,k1)`.
  AST guard: walk ast.Name.id + ast.Attribute.attr only; do NOT do a raw
  source-substring check (`_edge_amplification` is a substring of the
  oracle's own name `_exact_edge_amplification` -> false positive).
- FixedLensGeometryPrior fixes z_lens=0.0, so prior.inverse_transform on a
  z_lens=0.4 par_dic raises PriorError. To get in-support vectors, sample
  the cube (cubemin + rng.uniform*cubesize) until finite lnposterior.
- Refusal contract: spy get_marginalization_info via mock.patch.object;
  assert call_count==0 while assertRaises(LensDomainError/CancellationError).
  Posterior -> exact -inf via patching likelihood.lnlike_and_metadata
  side_effect=exc; assert isneginf and not isnan.
- dh_mptd axes = (modes, pol, time, det); time axis = -2 (roll there for
  the self-falsification delay mutation).
- Neighbors green: test_lensing_{likelihood,prior,waveform} 81 passed,
  2 xfailed.
