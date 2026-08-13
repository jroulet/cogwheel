# Professor short-term (saddle above-ceiling serving design consult, 2026-08-13)

Session: Phase-1 design consult on the three-tier far-from-caustic macro-saddle
serving ladder (brief_saddle_above_ceiling_serving.md, 10 measured facts). Answered
Q1-Q5 for the Architect; no re-measurement (all facts DONE).

## Key code-level findings this session (candidate for professor_code_observations
## after Dreamer review)

- **`reconstruct_farfield` with `FARFIELD_KERNEL_SUM` does NOT re-gauge the
  switch.** `_farfield_switch` HARDCODES S_a=1 on real channels, tau_c=0
  (channels.py ~L876-914). So the existing `_ppgo_above_ceiling` template
  reconstructs with BARE kernels (S_a=1), NOT with the SACR-C
  smootherstep(w|tau_a-tau_c|,0.5,4) switch. This is the crux for Q1: serving
  tier-1 as `reconstruct_farfield(..., envelope=ZEROS, FARFIELD_KERNEL_SUM,...)`
  is F ≈ sum_a H_a exp(iw tau_a) with S_a=1 (bare-kernel sum), NOT the
  re-gauged switched-channel sum. These COINCIDE only when every switch is
  saturated (S_a≈1), which is exactly what the gate must guarantee. So the
  envelope=0 FARFIELD_KERNEL_SUM path is CORRECT tier-1 serving IFF the gate
  certifies switch saturation under the chosen gauge — the two are numerically
  equivalent precisely on the admitted set, divergent off it.

- **`_ppgo_above_ceiling` (likelihood.py ~L1907) computes envelope from
  fold_ppgo_correction, NOT zero.** It builds f_total via fold_ppgo_correction,
  demodulates, subtracts ppgo_sum, and feeds a REAL envelope to
  reconstruct_farfield. Tier-1 differs: it would force envelope=ZEROS. So tier-1
  is NOT byte-identical to the above-ceiling rung — it drops the fold_ppgo
  Airy carrier and the residual. Structurally similar dispatch, different
  envelope source.

- **`farfield_ghost_term` (channels.py ~L963) is gated, raises
  GhostDomainError.** Two frequency-INDEPENDENT gates: decay
  Im(tau_c)>=_GHOST_DECAY_IM_THRESHOLD=0.4, separation
  min|x_a-x_c|>=_GHOST_SEPARATION_MIN=0.7. It returns
  C_c*exp(iw(tau_c-t_min)) — a DECAYING complex-saddle contribution. As a
  tier-1 GATE PROXY for intrinsic |E|/|F| it is problematic: it can REFUSE
  (raise) exactly on the far-from-caustic sources tier-1 wants to admit (ghost
  well-separated AND decayed there → but the gate wants the ghost SMALL, and a
  raise is not a magnitude). Recommended the gate NOT key on ghost admission
  but on post-re-gauge switch saturation min_a w*|tau_a-tau_switch|>=RHO_END
  directly (the brief's own fact-2 discriminator), which is engine-free and
  degrades with the served quantity via the SACR-C bounded-phase theorem.

- **`switched_analytic_channels` (_gauge.py ~L335)** is the SACR-C projection;
  4-arg switch is per-channel S_j, weights via _envelope_weights. Envelope
  E = conj(carrier_c)*(F - sum carrier*trial). tau_c enters as BOTH the switch
  arg (via how switch was computed upstream) AND the demod carrier_c — the
  brief's decouple (tau_switch vs tau_phase) means computing `switch` from
  tau_switch but passing critical_delay=tau_phase to this function.

## Verdict summary given to Architect
- Q1: envelope=0 FARFIELD_KERNEL_SUM is correct tier-1 ONLY under gate-guaranteed
  saturation; equivalent to re-gauged switched sum on the admitted set.
- Q2: gate on post-gauge switch saturation min_a w_lo*|tau_a-tau_switch|>=RHO_END,
  NOT ghost/|E| proxy. Ghost term is wrong proxy (gated raise, decaying object).
- Q3: tier-1 stores nothing → no skew; decoupled gauge is a tier-2/chart concern;
  tier-1 with envelope=0 needs only tau_switch (S_a saturation), tau_phase
  irrelevant (no stored envelope demod).
- Q4: SHIP tier-1 + shared gauge helper + tier-3 named refusal; DEFER tier-2 chart
  (needs training run, forbidden). Land the gauge helper + a demod/remod
  round-trip UNIT test (synthetic in-memory envelope) so tier-2 plumbing is
  de-risked without training.
- Q5: certify w<=60 vs exact Schwinger (NOT F_op — diverges on saddle); bar 1e-3
  on |F_serve-F_exact|/|F_exact| reported p50/p90/max + worst locus; worst-case
  gamma=1.5859 y=(-1.1208,-0.9002); refusal test on a w*|y|>58 unchartable
  source; gauge-identity asserts tau_switch(source)==stored; handover-continuity
  on tier1↔near-caustic tau_c at the RHO_END boundary.
