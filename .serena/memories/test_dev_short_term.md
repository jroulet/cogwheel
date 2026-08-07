# Test Dev Short-Term Observations

- SHARD 2b (test_lensing_prior.py::RefusalNetTestCase, C6): mass-capped the
  refusal scan so NO draw evaluates above w=60 (no mpmath). Mechanism mirrors
  SHARD 2a: i_mass=prior.sampled_params.index('ln_m_lens_msun'); w_per_msun=
  dimensionless_frequency(like.fbin[-1],1.0,0.0); ln_m_cap=log(C6_W_CAP=58/
  w_per_msun); clamp sampled[i_mass]=min(sampled[i_mass],upper). Added a spy
  on schwinger_module._f_schwinger_mpmath (patched in the scan, restored in
  finally) -> cls._mpmath_calls; recorded (w_top,gamma,class) ->
  cls._refusal_diagnostics. NEW guard test
  test_scan_stayed_on_fast_double_double_path asserts _mpmath_calls==0 AND
  all w_top<=60. Whole class now 6 tests / 63.4s (was 906s). C6_SEARCH_BUDGET
  800->120. CRITICAL: Professor Ruling 2's premise (SchwingerCertificationError
  reachable on DD path w<=60 by pushing gamma'->1) is EMPIRICALLY FALSE -
  direct f_schwinger sweeps NEVER refuse at w<=59 for gamma_prime in
  [0.9,1.6] or |y| in [0.05,2.0]; full-grid ChangRefsdalChannels near
  gamma=1 at w<=57 all served. The reachable capped-box vocabulary is
  LensedBinningError (wide-delay saddle pair > delta_t_max=0.02), dense
  near gamma~1. The mutation test is already class-agnostic
  (mock.patch.object(posterior_module, exc_class.__name__)) so
  LensedBinningError still turns it red-when-mutated; control (-inf) +
  mutated re-raise both hold. Asserts refusal CLASS + -inf VALUE, never the
  band. Tooling note: Serena + built-in Read/Edit/Write were all DOWN this
  run; applied the edit via a Python replace-script through the conda Bash
  path, and this memory note via a direct file write (Serena edit_memory
  returned MCP -32602).
