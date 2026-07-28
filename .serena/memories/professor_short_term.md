# Professor — short-term (2026-07-28: saddle Born carrier review, commit 31ee133)

## VERDICT: PASS (inference review of the macro-saddle Born lead-carrier build)
Ran `test_lensing_born.py` on cogwheel-newlal: **52/52 green in 26 s**. Then
independently re-derived the load-bearing physics (not just trusting green):

- **Acc #1 (carrier vs independent mu_macro)**: my own matrix-solve oracle
  (full Fermat delay 0.5 x0.A.x0 - y.x0 + 0.5|y|^2 - ln|x0|, no _born algebra)
  agrees with `born_lead_carrier` to **2.8e-14** over the full gamma/|y|/theta/w
  sweep; all mu_macro<0 (genuine saddles). My FIRST quick oracle disagreed (~2.0)
  ONLY because I dropped the point-mass ln|x0| core + 0.5|y|^2 reference — the
  test's oracle correctly includes them. Module uses the collapsed A x0=y form
  (drops quadratic) => genuinely independent algebra.
- **Acc #2 (F009-S pin)**: |F| = sqrt|mu| = 1.203858530858 CONSTANT across
  w={1e-3..8} to 12 digits; Morse phase = literal -1j (Re=0 at w->0). Total
  phase drifts (correct, F009-S). 
- **Acc #3 (fence band 1.0502342<gamma<3)**: lower root 1.05023417791 -> max_y
  3.0 (1e-14); re-entry root of 4g^2-9g-9 = exactly 3.0; min extent 1.5961 at
  gamma=1.1777. off-axis vs on-axis candidate selection verified.
- **Acc #4 (split currency w*Delta_tau)**: live geometry gives Delta_tau=16.25,
  r0_sq=212.4 at gamma=1.2,|y|=3.05,th=0.3. At w=0.1: w*Delta_tau=1.625(<4 serve)
  vs w*r0_sq=21.2(>=4 retired-refuse) => OPPOSITE decisions. r0/(2 Delta_tau)
  span=1783x >> 100x. Currency correct.
- **Acc #7 (census 'born' arm)**: signature confirmed; 2 exterior images at
  |y|=3.05 saddle. 4/4 census tests green.
- Acc #5/#6 (ppGO-only residual splines; ghost inflates node count) green with
  operator.F_op as oracle; carrier they demodulate is exact (verified above).
- Self-falsification/reachable-red tests (a0 breaks F009, wrong Morse sign,
  gate refuses) all PASS => tests have teeth.

## MINOR CONCERN (documentation, not code)
Task brief claimed Delta_tau~35.3, w_split~0.113 for the gamma=1.2/|y|=3.05
witness. Live geometry.delay gives Delta_tau=16.25, w_split=RHO_END/dt=0.246.
Code + test use the CORRECT measured value; the brief's number is stale/wrong.
Physics conclusion unchanged.

Heavy full-sampling validation is operator-deferred (not run here, per budget).
