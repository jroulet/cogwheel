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
Current: steps_2_4_8 — three final gates before training. All three artifacts exist but need updates: (1) step 2 script params mismatch brief (gammas, fraction grid, config); (2) step 4 script measures wrong thing (carrier accuracy, not geometric coverage); (3) step 8 test has a failing allowlist entry + missing doc-scan + missing docstring-scan. Professor confirmed: n_gamma=1 correct for step 2; step 4 is geometric coverage not accuracy; docstring check = constant docstrings only not comments. Simplifier: merge steps 2+8a into one WP, rewrite step 4 as geometric, step 8b+8c as second WP. Two Coder WPs total.
