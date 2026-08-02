# Foreman-Lite Short-Term Observations

## 2026-08-01 — INS-7-002 ghost decay gate docstring + test fixes

### Observation: near-cusp configs fail BOTH decay AND separation gates simultaneously
Near-axis/near-cusp configs (theta ~ 0.3 deg from principal axis) have:
- Im(tau_c) ~ 0.001 (DECAY gate refuses them)
- separation ~ 0.2-0.3 (SEPARATION gate would also refuse)
The two gates are physically CORRELATED at the cusp: Im(tau_c) → 0 and
separation → 0 together. No config exists that passes the decay gate
(Im > 0.4) AND fails the separation gate (sep < 0.7) in this parameterization.
This means reachability tests for `_GHOST_SEPARATION_MIN` must patch BOTH
`_GHOST_DECAY_IM_THRESHOLD` AND `_GHOST_SEPARATION_MIN` to zero to reach the
separation gate's permit branch.

### Observation: ADMIT configs need BOTH gate checks AND the DoNothing criterion
An admit config must satisfy:
1. Im(tau_c) > _GHOST_DECAY_IM_THRESHOLD (0.4) — decay gate
2. separation > _GHOST_SEPARATION_MIN (0.7) — separation gate
3. resid(MINUS_GHOST) <= resid(KERNEL_SUM) — ghost subtraction actually helps

At gamma=0.3, theta=20 deg, low offsets (0.25-1.2): ghost subtraction WORSENS
the residual (fails criterion 3). At offset=1.5, it passes all three.
The DoNothing check is essential for selecting valid ADMIT configs — a config
that passes both gates may still have ghost subtraction that's counterproductive.

### Fix applied
- channels.py docstring: removed incorrect "Im tau_c >= 0.9" generalization
- test_lensing_ghost_gate.py ADMIT_CONFIGS: increased offsets for 3 failing entries
- test_lowering_constant_to_zero_admits_a_refuse_config: patches both constants
- test_forcing_a_refused_config_worsens_the_donothing_residual: patches both constants
