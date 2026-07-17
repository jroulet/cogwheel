# Professor short-term

2026-07-16: Reviewed Chang-Refsdal wave-optics engine (cogwheel/lensing/chang_refsdal/).
All 126 tests + 129 subtests PASS (144s). Physics verified: |C|^2=pi w/(1-e^{-pi w}) prefactor
(F->1 as w->0, ->pi w on-axis) machine-precise; geometric-optics slope w^-1 (no C1/C2) / w^-3
(with) confirms correct asymptotic order; Morse census n_max=0 correct (positive-parity point
lens, tau->+inf at origin: only minima+saddles); 4<->2 image transition exactly on astroid
caustic; near-caustic delays stable (artanh(gamma) to ~1e-17) while |mu| diverges; mass-sheet
invariants Delta tau_ac & |K_a/K_c| flat across kappa; channels suite rigorous (scale-aware
recon + self-falsification). VERDICT PASS. Open concern: F005 gap L=w|y'| in [~30,48] returns
uncertified finite F_op with NO named refusal (confirmed 8 gap configs return finite, not nan) --
acceptable at engine stage but reachable in high-mag near-caustic regime; must be gated/closed
before Build 2 likelihood is trusted there.

## Consult 2026-07-17 — Build 2 crown-gate red (driver-commissioned)

Near-cusp RB blow-up ROOT CAUSE = ill-conditioned channel gauge |K_a|~5e5 vs
|F|~3 (recon machine-exact), NOT F006 edge-secant/k1-squaring (sign disproven:
h_h goes NEGATIVE -9e8, lnl=-h_h/2=+6.43e8). Dense subsampling can't fix a
1e5:1 conditioning problem. Two-image +9.77 is entirely a norm-term p+s<=3
truncation bias (~1.3% under-estimate of h|h; d_h exact). Brute healthy at
all configs. Timing 2.14x because per-bin dense engine eval (2024 nodes)
defeats RB's purpose; paper wants ~6-11 GLOBAL nodes. Failures 4/6 are test
bugs, 5 is an engine small-w (gamma/2w) singularity. Dreamer: fold into
professor/microlensing_chang_refsdal (RB valid only in resolved regime;
validity guard + sparse global nodes for 2c).
