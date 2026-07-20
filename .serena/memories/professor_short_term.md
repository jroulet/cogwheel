# Professor short-term — Build 7b consultation (2026-07-20)

## Session summary
Consulted on 7 questions for Build 7b (saddle-domain channel/likelihood/prior integration). Key rulings issued:

## Observations

1. **Envelope LOO root cause (Q5)**: The `_LOO_STOP = 4e-3` error currency normalizes to `max|F_total|` over the node set. In the strong-shear rescued regime, the amplification has deep cancellation troughs (|F| << max|F|) where the interpolation error, while small in absolute terms relative to max|F|, is LARGE relative to the local |F| that feeds lnL through |F|^2. The lnL contribution of a bin at the trough is proportional to |F|^2 (norm term) and |F| (data term); an absolute error eps in F that passes the LOO gate (eps/max|F| < 4e-3) creates a relative error eps/|F_local| >> 4e-3 at the trough, which propagates through the squared norm term as ~2*eps*|F_local|/|F_local|^2 = 2*eps/|F_local|. The fix is to tighten `_ENVELOPE_SCALE_FLOOR` to act as a MINIMUM scale (replacing the pure max|F| normalization with max(max|F|, floor)), OR to switch to a mixed absolute+relative error currency. The cleanest knob: lower `_LOO_STOP` from 4e-3 to 1e-3.

2. **Prior parameterization (Q1)**: Single uniform gamma over [0, gamma_max] with gamma=1 as a measure-zero named-refusal boundary is the correct scheme. No discrete parity label. Jacobian is trivial (identity transform). Recommended gamma_max = 1.6 (matches research scan upper bound). Refusal band around gamma=1 is effectively just the point itself (det A = 0 exactly) — the engine handles gamma = 0.999 and gamma = 1.001 cleanly on both sides.

3. **Q4 confirmed**: The data flow in `LensedMarginalizedExtrinsicLikelihood._get_dh_hh_timeshift` evaluates `self._engine._amplification_coefficients(par_dic)` FIRST, before any coherent-score QMC work. A SchwingerCertificationError during that call propagates unswallowed, satisfying the "refuse before QMC" contract by construction.

4. **Folding validity (Q3)**: The quadrant fold IS valid on the saddle domain. The argument from the Fermat potential symmetry under (y1->-y1) and (y2->-y2) independently is correct and parity-independent. The 3-cusp-vs-4-cusp distinction is a caustic-topology fact that does NOT affect the F(w,y) reflection symmetry.

5. **Deep-band pin (Q7)**: Already comprehensively tested in `test_lensing_schwinger.py::DeepBandTestCase` at the engine level. A channel-layer duplicate would be redundant — recommend a single INTEGRATION gate (lnlike at a saddle config vs an oracle) rather than re-testing the engine's own certified limit.

6. **Code path observation**: The existing `_amplification_coefficients` in likelihood.py calls into `_envelope_loo_nodes` which calls `_evaluate_envelope` which delegates to `ChangRefsdalChannels`. The saddle guard lives in `channels.evaluate` at the top — once that guard is lifted, the entire LOO/reconstruction machinery is structurally parity-blind (only the wave evaluation changes, dispatching through the existing operator Schwinger fallback). The likelihood layer itself needs NO new code paths for the saddle, only the guard removal + the LOO policy fix for accuracy.
