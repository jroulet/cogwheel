# Professor short-term checkpoint (2026-07-18, Build 4 inference review)

Reviewed Build 4 (sampled lens coords / folding / sampling-ready posterior),
worktree cogwheel-claude-dev, interpreter cogwheel-newlal. Code in working tree
(uncommitted): cogwheel/lensing/{prior,posterior}.py + lensing/likelihood.py mods;
new test cogwheel/tests/test_lensing_prior.py (C1-C7 gates).

RAN: pytest test_lensing_prior.py -> 27 passed, 1 xfailed in 63s. Fast suite only;
heavy full-sampling validation is operator-deferred (correctly out of my budget).

Correctness gates ALL PASS at spec tolerances (constants verified to match spec
verbatim): C1 roundtrip 1e-12; C2 jacobian FD-vs-analytic 1e-5; C3 domain safety
(1-kappa>|gamma|, w_max<=450, w*sqrt(s)<=58) over 1e4 draws; C4a reflection 1e-9;
C4b fold-unfold brute 1e-6 / RB 0.5; C4c NO phase-fold under XPHM higher modes;
C5 mass-sheet lnL invariance brute 0.01 / RB 0.5 (+mag 1e-9); C6 refusal net ->
exact -inf, raw path raises LensDomainError/CancellationError, MUTATION check
(narrow except -> red) green. Non-vacuity enforced (SelfFalsificationTestCase +
per-test n_compared teardown).

ONE CONCERN (C7): finite fraction MEASURED 41.2% (206/500), spec aspiration 90%.
Carried as documented @expectedFailure (prior box overlaps gamma~0.5 cancellation
band; hard floor only 0.05). NOT a correctness bug: all 294 non-finite are exactly
-inf (0 NaN/+inf); near-truth reference lnpost=260.6 dominates (best random draw
18.1, i.e. -242 nats below ref -> peak sits at truth). Impact is SAMPLER EFFICIENCY
(~59% proposals refused), not physics. Operator should decide if gamma prior should
be bounded away from the cancellation band before the heavy sampling ship gate.

Verdict issued: CONCERN (physics correct; C7 prior-width efficiency shortfall flagged).
