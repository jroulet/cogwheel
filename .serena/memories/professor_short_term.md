# Professor short-term (tiling census INFERENCE REVIEW, 2026-08-14)

Reviewed cogwheel/tests/test_lensing_tiling_census.py (26 tests) on
cogwheel-newlal env. ALL 26 PASS in ~60s (fast tier). Verdict: PASS.

Live run of tc.run(TrainingConfig()) sanity-checked (not just green ticks):
- Q1: astroid detected=4/trained=1 (D2 fundamental arc), saddle detected=6/
  trained=min(6,max_tube_arcs=1). 6 saddle arcs = two 3-cusp deltoids (An&Evans).
  trained<=detected both parities. Correct topology. max_tube_arcs=2 widens
  saddle trained->2, astroid stays 1 (teeth present).
- Q4: astroid ceiling 34.64=60/sqrt(s=3.0)<480 -> min picks DD cap. Saddle tube
  floor=58 (SADDLE_WALL), ceil=148 (_SADDLE_W_CEILING). Saddle far-field floor
  70.19=(2e4*K)^(1/3), K~17.3, matches independent find_images+ppgo_error_estimate
  recompute to 1e-9. Constants: coeff 2e4, DD margin 60, labels/node 8, s/label 0.09.
- Engine-free tripwire strict (traps evaluate/f_schwinger/_f_schwinger_mpmath +
  namespace absence). Thin-caller delta==1 tile & one tile's nodes; unrelated
  tiler patch leaves counts unmoved. Verdict logic: 0->SILENT_EMPTY (zero-guard
  load-bearing), in->IN_BAND, above->EXPLOSION. Cross-check: nodes 12064*8*0.09=
  8686.1s=census_s; self_estimate 806.4s exact passthrough; ratio 10.8<<factor 5000.
- Census caught a REAL silent-empty live: lobe_interior:-1 has 0 tiles ->
  SILENT_EMPTY on smoke config (legit coverage-hole report, not a bug).

MINOR NOTE (not a fail): spec wanted any Q4 contained=False to carry a reason;
build reinterprets -- numeric-floor False is reason-less (bounds ARE the reason),
reason populated only on deferral (floor=None). Intent (no silent hole) satisfied.

No sampling/heavy validation involved (census is itself an engine-free pre-train
gate). No operator-deferred heavy test applies here.
