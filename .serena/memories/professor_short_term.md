# Professor short-term — Build 5 inference review (2026-07-18)

## Reviewed: LensedMarginalizedExtrinsicLikelihood (Build 5, commit 598f074 brief / 9cb5983 HEAD)
Env: cogwheel-newlal (py3.10, np1.26). Ran fast domain suite
`cogwheel/tests/test_lensing_marginalized_likelihood.py`: **21/21 PASS in 64 s.**

## Measured margins (extracted directly, not just green/red):
- **Spec 1 unlensed-limit identity (F=1 @ gamma=kappa=0, m_lens=1e-6):** rel dh=2.1e-7,
  rel hh=7.0e-10 vs gate 1e-6. Exactly the O(w)~1e-7 wave residual + float32 floor
  expected from F009 — NO spurious per-image shift, NO kernel-amplitude error. Correct.
- **Spec 2/7 exact-F reconstruction (4-image MAIN_LENS, |F| in [0.78,3.25]):** rel
  complexF=1.24e-3, rel |F|^2=1.02e-3 vs gate 3e-3. Physically the linear-in-bin K_a
  kernel error on 4 Hz bins. NOTE: docstring advertises "~2e-4 on these configs" but
  true value is ~1.2e-3 (~6x larger); still safely < gate but headroom is ~2.5x not
  ~15x. Minor doc/test-dev nit, not a physics defect.
- **Spec 6 conditional draws:** main lnL_marg=239.09, draws max=270.3/p10=264.6/min=260.1;
  unlensed lnL_marg=245.28, draws max=278.2/min=271.2. ALL draws sit ~20-31 nats ABOVE
  lnL_marg — exactly the 5-6D extrinsic Occam factor (~4-5 nats/dim for SNR~22). Lower-
  bound gate satisfied by huge margin. Test correctly DROPS the raw spec's upper bound
  ("no draw > lnL_marg+0.5") with documented justification: cogwheel's marg normalization
  puts plain peak ABOVE marginal, so upper bound is unphysical. Sound.
- Specs 5 (refusal precedes coherent score, call-count=0, exact -inf), 8 (registry,
  param pairing, bit-repeatable deterministic layer, JSON round-trip), 9 (LensedBinningError
  bin guard on marg path), all assertion-only: PASS. SelfFalsification proof-of-teeth PASS.

## Operator-deferred (NOT in fast file, correctly): 
Spec 3 (marg lnL vs importance-sampling brute-force oracle, C_two/C_cusp/C_macro,
|median-oracle|<=0.3 nats) and Spec 4 (plain-vs-marg reweighting). These are the ONLY
checks of the ABSOLUTE lnL_marg value; fast suite proves the FOLD is exact + conditional
self-consistency, but absolute-normalization accuracy is the nightly/operator gate.

## Verdict: PASS. Fold exact in unlensed limit + against independent engine; refusal
semantics correct; conditional draws physically consistent. Flag absolute-lnL oracle
(spec 3/4) as operator-deferred.
