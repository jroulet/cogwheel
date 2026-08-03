---
bump: patch
---

C5: ghost decay gate added to `farfield_ghost_term` (`channels.py`).
`_GHOST_DECAY_IM_THRESHOLD = 0.4` (frequency-independent fixed constant)
refuses the ghost where `Im(tau_c) < 0.4` — near a principal axis the
ghost is pure oscillation, not an exponentially-decaying correction (F027).
`_GHOST_SEPARATION_MIN = 0.7` cleared through Part 0: confirmed as a
lens-plane Einstein-unit quantity, stable across gamma 0.30-0.90. Born rung
Conventions section updated to name both ghost gates explicitly.
