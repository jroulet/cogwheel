# Architect Short-Term Observations

## Build 6 (C5) — Ghost Decay Gate

Planning complete. Single WP: add decay gate to `farfield_ghost_term` in
channels.py. Gate formula: `Im(tau_c) >= min_delay_separation` (the
geometric form of the retired `w_floor * Im tau_c >= 2.0`, proven
algebraically equivalent since w_floor = 2.0/min_sep). Professor confirmed
threshold 2.0 is physics-correct (resolution argument, not amplitude); the
simplified geometric form is frequency-independent and provably skew-free.
Simplifier endorsed single-WP scope, recommended inline delay computation
(not `_min_delay_separation` with synthetic mask), and flagged stale
`GhostContribution.delay` docstring note for update.

Key ordering: ghost_kernel → resolve images → decay gate → separation gate.
All 3 production callers catch GhostDomainError, so the new gate refuses
gracefully on all paths.