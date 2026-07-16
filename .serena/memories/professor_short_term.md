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