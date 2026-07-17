# Professor Short-Term Observations

## 2026-07-17 — Lensing Build 2 crown-gate consult (measured; conda cogwheel_310)

Fixture: HLV 4s, IMRPhenomXPHM, M_L=90 Msun z=0.4, 4 Hz bins (253), band 15-1024 Hz.

### Crown blow-up root cause = one-line switch-neighbourhood bug (FIXED at HEAD)
`_channel_switch` (channels.py) measured delay separation against REAL channels
only (`real_ids[real_ids != channel]`); the paper takes the min over ALL cluster
members incl. parked virtual labels. Only n_real=2 configs are hit (a virtual
label parks at ~same delay as a real image; real-only misses it -> smootherstep
saturates -> stationary-phase gauge -> kernel inflation). Fix (HEAD channels.py:354
`others = np.delete(np.arange(_N_CHANNELS), channel)`):
- near-cusp max|K| 5.22e5 -> 0.965 (recon 2.5e-10 -> 5e-16); two-image 40.9 -> 0.92.
- At 4 Hz, production p+s<=3: near-cusp offset 6.43e8 -> +0.329 PASS; two-image
  +9.768 -> +0.080 PASS. p+s<=4/<=5 identical to <=3 (<1e-4): the extra norm moment
  is UNNECESSARY. kernel_subsamples can revert 8 -> 2 (offsets +0.316/+0.069), ~7x
  speed-up. My earlier "near-caustic ill-conditioning / scope error / sparse global
  nodes / extra moment" A-D framing was WRONG; it is a fixable gauge bug.
- CORRECTED anti-pattern: do NOT re-base the NearCuspRegressionPin canary or the
  ScaleAware flat-gate test on the buggy real-only switch (pins the pathology).

### CORRECTION — failure-5 small-w ruling FALSIFIED
I previously called the UnlensedLimit |F-1|=2.062e-2 an engine small-w (gamma/2w)
singularity. WRONG. It is the EXACT macro-image amplification: as w->0, F ->
1/sqrt((1-kappa)^2 - gamma^2) = 1/sqrt(1-0.2^2) = 1/sqrt(0.96) = 1.020621, so
|F-1| = 0.020621 exactly. Measured FLAT across M_L 1e-2..1e-12 (w 1.8e-3..1.8e-13),
M_L-independent -> it is the macro limit, not a numerical floor. The test expected
F->1 at gamma=0.2 (wrong premise); the fix is gamma=0 (macro mag=1), NOT an engine
change. No small-w engine short-circuit is warranted.

### Zero-noise F->1 floor
On d=h0, asd_drift pinned to 1, floor = 1/2 (delta h0|delta h0), delta=F-1, i.e.
~1/2 <|F-1|^2>(h0|h0). At gamma=0.2 this gave 0.1214 = 1/2 (0.0206)^2 (570.8)
EXACTLY (macro mag, not a bug). At the REPAIRED gamma=0 anchor: brute ~1e-11 PASS;
RB 0.01164 remains = F007 construction asymmetry: `_set_summary` builds `_h0_edges`
with disable_precession=False forced + `_stall_ringdown`; `_candidate_bin_ratios`
builds h_edges with NEITHER, so at F=1 the ratio != 1 in the stalled ringdown band.
[probe8 tests matched candidate construction; prescription = make
`_candidate_bin_ratios` mirror `_set_summary`'s edge build — cheapest: apply the
precomputed fiducial fadeout/f_99 + forced precession to the candidate EDGES,
avoiding a candidate full-res strain per eval.] NO tolerance widening.

### ScaleAware flat-gate (test_flat_gate_fails_where_the_targets_diverge)
Fixed gauge bounds on-caustic kernels: worst sum|K| across on-caustic + astroid
cusp-point configs = 4.27, max|K|~1, recon ~1e-16. NO divergent-kernel config
exists in the fixed gauge (the physical target H_a still diverges on-caustic, but
the switch parks it: co-located virtual label -> sep 0 -> S->0 -> bounded cluster
residual). The >1e12 premise is void. Prescription: FLIP the test to assert
on-caustic kernels stay bounded (sum|K| < ~1e3) -> becomes a regression guard for
the switch fix; keep scale-aware bound in assert_reconstructs as harmless defence.

### Companion audit sites
- `_min_delay_separation` (channels.py:352): same real-only pattern, feeds the
  wave/geometric branch gate (delta_min). Not the crown cause (exact_total correct)
  but check vs the paper's cluster-separation definition.
- Failure 6 macro-saddle control 0.5/0.25: F_op refuses at w=9.75 because kappa=0.5
  rescales to effective shear gamma'=0.5, whose order-42 shear series tail=1.168e-10
  > 1e-10 (marginal). Correct refusal; test control mis-chosen. (Lead reports closed
  via a closed-form macro gate.)

Dreamer: fold the switch-fix + macro-limit corrections into
`mem:professor/microlensing_chang_refsdal` (RB is valid through cusp/fold once the
all-neighbour switch bounds the kernels; F(w->0) = macro amplification, not 1).
