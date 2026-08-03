---
date: 2026-08-02
section: Backlog
---

C5 (ghost decay gate, F027) from lensing_caustic_relative_coordinates, step 6.
_GHOST_DECAY_IM_THRESHOLD = _FARFIELD_WINDOW_RADIANS / 5.0 = 0.4 added to
channels.py. farfield_ghost_term refuses when Im(tau_c) < 0.4: near a
principal axis the ghost is pure oscillation, not an exponentially-decaying
correction, and must not be subtracted. The threshold is a fixed constant
(frequency-independent), so the training label and serve mirror reach an
identical admit/refuse decision for any fixed configuration. Tested by
test_lensing_ghost_decay_gate.py. _GHOST_SEPARATION_MIN = 0.7 cleared through
Part 0 (Build 7): confirmed as a lens-plane Einstein-unit quantity, not a
prior-box constant; SEP_REFUSE_MAX/SEP_ADMIT_MIN tripwire assertions added.
