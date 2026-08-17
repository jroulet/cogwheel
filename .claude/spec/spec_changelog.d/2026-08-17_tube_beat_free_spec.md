---
date: 2026-08-17
bump: minor
---

Documented the beat-free tube residual representation in the microlensing
engine section: tube charts store `r = E/F_ref` (q=p uniform-Airy
two-carrier reference, non-vanishing by the Airy Wronskian, shared
tau_c frame, DRY `Delta_tau` from `_merging_fold_pair`), the
`tube_beat_free_airy_v1` envelope-definition hard-refusal, the measured
F083 falsification (n_theta=10, eps=4.2652e-03 vs the 0.0237 bar the
old representation needed ~48 nodes for), the structural (require_fref
=False) status of the tube/exterior-polar double-match precedence pins,
and the certifying suite `test_lensing_tube_beat_free.py`.
