# Professor short-term — Build 8f five-levers inference review (2026-07-21)

Reviewed `cogwheel/tests/test_lensing_levers.py` (env cogwheel-newlal). 47 passed,
1 honest xfail, 70 s. Verdict PASS. Numbers verified against first principles:
- L1 geometry: `_companion_roots` byte-identical to np.roots (max|diff|==0.0) incl.
  near-caustic double root; find_images <1e-10 rel; count/mask/switch bits exact.
- L2 contraction: _data/_norm term <1e-10 rel normal regime; ABSOLUTE floor
  (NORM_ABS_TOL=1e-11) below |norm|<1e-6 — correct: rel tol meaningless at near-zero
  norm. Reassociation round-off ~8e-14.
- L3 node-parallel: prange pure-map == .py_func serial bit-exact; any-node-refuse ->
  whole-grid same SchwingerCertificationError; scheduling-independent; F010 cert-flag
  mutant goes RED.
- L4 Pearcey: independent oracle=live quadrature anchored to P(0,0)=Gamma(1/4)/2 e^{i pi/8}
  (1e-12). Fixture n=91 worst abs err 2.73e-5, dominated by caustic (1.99e-5); far-field
  x>0.15 = 7.8e-9 (4 orders below, as claimed). Bicubic h^4 contraction ratio 0.499.
  Hash backstop refuses 1-ulp tamper; out-of-box routes to live quadrature byte-identical.
- L5 L_MAX: measured L_geo=34 (geometric rel-err vs Schwinger oracle falls monotonically
  ~1/(w*delta): 2.27e-4@30 -> few e-6). Bracket 34 <= L_MAX(48) <= ceiling60-headroom6=54,
  non-empty, 48 pinned. Census guards enforce index theorem sum_a sign(mu_a)=sign(detA)-1;
  both perturbation mutants RED.

NUANCE (defensible, not blocking): fast suite proves the Pearcey table only to the 1e-4
FIXTURE floor + convergence trajectory; the shipped 1e-8 production pin is met only by the
denser OFFLINE table (operator-deferred heavy artifact). Honest @expectedFailure documents
this gap without widening the gate — correct discipline. Also premise-repair vs spec: L5
upper edge is the Schwinger HARD ceiling (60), not the retired operator-series L_wave~46
(Build 8d homogenization); matches knowledge memory (DD Schwinger holds 1e-10 to w~64).
Heavy full-sampling / real-data validation NOT run (out-of-band ship gate).
