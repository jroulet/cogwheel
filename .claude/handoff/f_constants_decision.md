# (f_max, f_floor) decision — measured provenance (2026-08-17)

Evidence: `.claude/handoff/f_fraction_sweep_results.json` (108 pts,
full-arc; SADDLE legs authoritative) +
`f_fraction_sweep_trimmed_astroid.json` (60 pts, F083-knee-trimmed
arcs; ASTROID legs authoritative — full-arc astroid numbers are
invalid, see todo.d/lensing_tube_trainer_resolvable_subarc_trim.md).
Config: production density n_gamma=7, n_u=7, n_theta=7,
w_nodes_per_decade=15, n_heldout=30, w <= 60, seeds 42/43, HEAD 77da2e6.

## RULING

- **f_floor = 0.08** (both parities; was 0.16). The floor ladder at
  f_max=0.40 shows band eps FLAT in the floor while inner-shell eps
  improves monotonically downward (gamma=0.7: 0.062 -> 0.017; gamma=0.2
  inner 0.114 -> 0.059). Deeper near-caustic coverage is free of band
  cost; 0.08 stops short of the 0.05/0.03 rungs whose builds are
  slowest with no further band gain.
- **f_max = 0.40** retained (was 0.40 by fiction, now by measurement).
  Leg B is flat in f_max on both parities — shrinking the shell buys
  no accuracy and shifts load to the wedge/exterior charts for nothing.
- **Density, not constants, closes the bar gap.** At n_theta=7 the
  trimmed astroid mid-gammas read eps 0.09-0.12 vs the 0.05 bar, while
  F083 measured 4.3e-3 at n_theta=10 on the core sub-arc (~1/3-1/4 of
  the trimmed span). The tiling design MUST size n_theta per band on
  the trimmed span (linear in span — no explosion). Bands flagged for
  density: astroid gamma 0.10-0.40; saddle gamma ~1.1 (deltoid
  transition, eps 0.076-0.14; the other saddle bands pass at 0.003-0.05
  from f_floor >= 0.12 already).
- Outlier on record: trimmed leg B gamma=0.10 f_max=0.28 read 0.604
  (isolated; neighbors 0.140/0.107) — treat as a bad trim/refusal
  cluster at that corner, not a trend; re-probe during tiling design.
