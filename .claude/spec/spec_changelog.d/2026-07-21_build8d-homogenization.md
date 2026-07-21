---
bump: minor
---
### Build 8d — homogenization: Schwinger as THE exact wave evaluator

Engine-row rewrite: both parities served by the Schwinger quadrature
(positive parity gamma' > 0 via lam = 1-kappa, identical to the
saddle arm); legacy operator series demoted to the gamma' == 0 exit +
test-only oracle; w > 60 non-geometric corner refuses by name until
8e (F019: the two evaluators' ceilings live in different variables);
exact path re-priced with brute-accuracy tests gated to the driver
post-build tier.
