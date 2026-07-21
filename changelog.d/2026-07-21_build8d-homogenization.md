---
date: 2026-07-21
---
### Engine homogenization: one exact wave evaluator on both parities

The Schwinger 1D quadrature becomes the single production exact wave
evaluator: sheared positive-parity hosts now reduce/rotate/reconstruct
through exactly the code path the macro-saddle arm uses, and the
legacy operator-series contraction is demoted to the shear-free
point-lens exit and test-only oracle duty. Positive-parity wave values
move at the 1e-14 level (witnessed byte flips; physics tolerances
unchanged); refusals above the Schwinger ceiling are named and
unconditional on the sheared arm, with the unresolved high-frequency
corner deferred to the uniform-asymptotics build. A geometry-only
census script quantifies that corner and the shear-free exception over
prior draws. Brute-force accuracy tests move to an opt-in driver tier
(COGWHEEL_BRUTE_ACCURACY) so default test runs stay fast.
