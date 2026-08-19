# Session: diffractive_w_low certificate reach — narrow-reading ruling (2026-08-19)

Context: Build brief `.claude/handoff/diffractive_certificate_reach.md` says
`diffractive_w_low` (Rung P, positive parity, `_diffractive.py:282`) "only ever
searches DOWNWARD" and ships a conservative candidate verbatim; the fix is to serve
up to the honest ceiling. A prior ruling #2 (mine, apparently) was read by the
inspector as "remove the deep-optimistic self-refuse branch." The inspector found
this collides with two shipped INS-1-001 honest-gate pins in
`test_lensing_diffractive.py`:
- `CertificateOptimismWitnessTestCase.test_optimistic_regime_is_refused`
  (OPTIMISTIC_GAMMAS {0.4,0.5} -> None), and
- `WLowMonotonicityWitnessTestCase.test_near_wall_band_is_refused`
  (GAMMAS_TO_WALL 0.90..0.994 -> None).

## Ruling (this session)
The inspector's NARROW reading is CORRECT and I endorsed it. The ONLY bug is that a
candidate already clearing the bar is shipped verbatim instead of being extended UP
to the honest ceiling. Both None-returning branches STAY:
- `honest_error > _DIFFRACTIVE_CERT_SAFETY * CERTIFICATION_BAR -> None` (deep
  optimistic / near-wall) — this is the F009 / WLowMonotonicity physics: near the
  parity wall the omitted tail is worst and MUST be declined. Removing it would let a
  down/convergent search find a tiny positive w (relerr->0 as w->0) and ship a small
  positive ceiling instead of None — physically wrong (serving where truncation is
  worst) and overturns shipped pins.
- The `_rootfind_w_low` down-search on the in-between over-reaching band STAYS.

The up-bracket is added ONLY on the `honest_error <= CERTIFICATION_BAR` path.

## Two decisive sub-rulings
- Q2 (ceiling target): the up-search targets the OUTER bar CERTIFICATION_BAR (1e-4),
  NOT bar_inner (5e-6). Rationale: relerr is the full 2M-tail-ratio, already a
  conservative over-estimate vs the exact engine (the honest verifier is N/2N tail
  ratio, brief fact 4); the 20x safety margin is a CANDIDATE-PLACEMENT margin (keeps
  the candidate away from the cliff so verification is stable), not a serving margin.
  The honest verifier IS the oracle proxy; serving to relerr==CERTIFICATION_BAR is by
  definition serving to the bar. Engine-honesty acceptance (brief) re-confirms this.
  Tightness multiplier: 1.5x the ceiling must breach the bar (brief already says so).
- Q3 (monotone): confirmed. Candidate-clears path grows (up-search only), refuse path
  is None==None. Brief MONOTONE acceptance satisfied without touching refuse regime.

## Watch item (raised in plan, not a blocker)
_rootfind_w_low assumes relerr MONOTONE NON-DECREASING in w. The up-bracket must not
overshoot into a region where that monotonicity fails (tail is
(gamma' s w/2)^(M+1)/(M+1)! locally, monotone on low-w band). The up-search should
bracket by DOUBLING from candidate until relerr>bar, then bisect down — a clean
mirror of the existing down helper; cap at the brief's `cap`. Since candidate sits at
bar_inner (20x under), first doubling is safe.
