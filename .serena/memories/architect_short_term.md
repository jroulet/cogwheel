Last build: wedge follow-up DD+arclength (56a223a). Clean.

Build d-norm-eval (planning): EVALUATION of FarFieldChart d/R_c axis normalization
before full-prior training. Prof+Simplifier ruling: implement d/R_c as IN-MEMORY-ONLY
opt-in mode (`d_normalized` flag on from_engine; per-(gamma,s) rc_table stored on chart;
one divide in serve transform + box gate). NO NPZ persistence/schema bump this build
(deferred to promotion build; OWED note). R_c per-(gamma-node,s-node) MANDATORY (single
theta repeats rejected arc-map mistake). A/B test = MEASUREMENT not decision-oracle:
hard-gate correctness invariants (round-trip bijection, min R_c>floor, train/serve box
parity, node-exact on stored grid, default byte-identical, eps_norm<=1.1*eps_raw
non-regression floor both strata); RECORD stratified near-wall/far-tail eps for driver's
>=2x promotion gate (never gated). Retrain is post-build driver step.
Current: evaluate_d_normalization — Professor+Simplifier conclusively answered "no" via 5 independent structural arguments. Zero-WP plan, findings-only output.
