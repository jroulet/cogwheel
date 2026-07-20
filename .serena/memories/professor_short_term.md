# Professor short-term — Build 8a surrogate INFERENCE REVIEW (2026-07-20)

## Verdict: CONCERN (physics correct, all hard gates green; absolute nat-tiers relaxed above my own bounds — budget-justified, operator-deferred)

Ran `test_lensing_surrogate.py` on cogwheel-newlal: 23 passed, 1 skipped
(TimingSmoke, gated behind COGWHEEL_RUN_TIMING_SMOKE — machine-dependent,
correctly default-skip). Runtime 300s. Surrogate diag plots regenerated
(surrogate_recon_positive/saddle, beta_invariance, domain_gate_slice).

## Hard/Boolean gates — ALL GREEN (the correctness-critical ones)
- Crown BYTE-IDENTITY with default None: lnL + fiducial envelope nodes
  max|diff|=0 vs HEAD; constructor leaves attr None. Surrogate is OFF by
  default — no physics answer touched.
- Refusal-set preservation: surrogate never serves where engine refuses;
  F010 mutation (patch in_domain to lie) flips assertion red — gate has teeth.
  Non-finite lnL exactly -inf, zero NaN.
- Domain gate conservatism: near-refused / outside-box decline (served=False,
  falls back to engine); certified interior serves.
- Serialization npz+pickle bit-identical; F002 AST oracle-independence guard
  passes real oracle, flags tainted oracle (positive control).
- Beta-elimination: eigenframe E invariant to <1e-12; reconstructed F(beta)
  matches engine.

## Accuracy tiers — pass via BUDGET-INDEPENDENT relationship, NOT absolute nats
Gate used: dlnL <= 1.5 * eps_dense * |lnL_exact| (F002 fresh-engine eps).
First-principles sound: dlnL ~ eps*SNR^2, |lnL|~SNR^2, so ratio O(1); measured
peak ~0.84. Monotone-refinement control witnesses convergence toward 1e-3.

Measured (dlnL nats, eps_dense):
- crown (0.167, 5.8e-3): ABOVE my hard "crown NEVER past 0.1" — but fixture
  envelope eps is ~58x worse than production 1e-4; scale back -> 0.003 nats <0.01.
- deep (0.019, 5.7e-3): fine.
- near-caustic (12.8 nats, 0.16!): box-edge, under-resolved tiny grid. Served
  despite 16% envelope error -> 12.8-nat lnL error. relationship gate holds
  (|lnL|>53) but this is the RED FLAG config: served region must exclude
  near-edge/near-caustic at production scale.
- saddle (0.66, 2.8e-3) / saddle-2 (0.66, 3.1e-3): dlnL RB-BINNING-floored
  (~0.66), not envelope-limited; above 0.1 saddle tier. |lnL|>159 so
  relationship gate holds.

## Concern to carry forward
Relaxed LNLIKE_BUDGET_TOL=0.5, POS_RECON_TOL=0.20 sit above my production tiers
(0.01/0.1 nats, eps<1e-3). Legitimate F016 (tiny minutes-scale fixture,
documented, convergence witnessed, surrogate off-by-default) — NOT tolerance-
hiding. But before surrogate is EVER enabled-by-default, a production-scale
re-gate at eps~1e-4 must hit the 0.01/0.1 nat tiers, AND the served region must
carry a caustic/edge margin so no served config has eps like the 0.16 near-
caustic case. This matches my Q5 ruling (enabling-by-default deferred).
Timing smoke (saddle >=5x, <2ms) is operator-deferred (skipped).
