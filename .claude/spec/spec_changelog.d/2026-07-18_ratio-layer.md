---
date: 2026-07-18
bump: minor
---
Candidate/fiducial ratio layer (Build 3g): lattice-snapped memoized
fiducial envelope, heterodyned ratio interpolation (~8 LOO nodes,
config-independent), guard/refusal-symmetric fallback to the direct
SACR-C path. Measured warm single-thread lnlike ~9.8 ms (~143x brute).
New test module `cogwheel/tests/test_lensing_ratio_layer.py`.
