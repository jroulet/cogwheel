# Inspector Short-Term Observations

## 2026-07-30 — Build WP1 review: F054 closed-form caustic_geometry (FULL rewrite)

Scope: uncommitted worktree. Production change: `ppgo_map.caustic_geometry`
rewritten from a 720x2-branch `geometry.critical_point` polar scan to a
closed-form candidate extremisation; signature dropped `n_theta`
(now `(gamma, kappa=0.0)`). `_measure_cell` angle fan changed one-sided
5-angle -> symmetric 9-angle `tuple(k*pi/8 for k in range(-4,5))`.
New/expanded test files: test_lensing_ppgo_map.py (+863, annulus_rho D1/D2
+ ReachMaximiser/ByteIdentity), test_lensing_surrogate.py (+619, Wp1 oracle
suite), test_lensing_exterior_admission.py (+193, WP1 wall cases),
bandsplit ANGLES fix.

### MATH VERIFIED CORRECT (hand-derived + independent brute force)
Derived in mass-sheet vars lam=1-kappa, e=gamma/lam, u=1/(lam|x|^2):
- det=0 -> cos2θ = (u^2-1+e^2)/(2eu)  [matches code]
- caustic radius^2 = lam[(1-u)^2(1+2u)+e^2(2u-1)]/u^2  [matches code EXACTLY;
  verified algebraically via |y|^2 = (lam/u)[(1-e-u)^2cos^2+(1+e-u)^2sin^2]]
- axis_a=lam(1-e-u)=eig_scale-gamma, axis_b=lam(1+e-u)=eig_scale+gamma ✓
- Candidate set: e<1 cusps {1-e,1+e} (cos2θ=∓1); e>1 {on-axis cusp 1+e,
  interior stationary (-1+sqrt(4e^2-3))/2 from f'(u)=0, branch-fold
  sqrt(e^2-1) where the two sqrt-branches merge}. u=1 interior stationary
  correctly omitted (astroid mid-edge, never the max).
Independent 800k brute scan over kappa∈{0,0.2,-0.3}, gamma∈[0.1,3]:
reach matches to worst 3.9e-11. Direction lands on a genuine farthest
caustic point at exactly reach in EVERY case, incl near-wall saddle
(g=1.05/1.10/1.177 -> off-axis lobe at -82/-60/-69 deg). |dot|<1 vs the
test's dense oracle in near-wall band is just the mirror-image deltoid
lobes (both global maxima) — the reflection degeneracy the symmetric fan
is designed to absorb. NOT a bug.

### Guards verified
lam<=0 (over-critical) and |gamma|==lam (det A=0 parity wall) both raise
LensDomainError with named messages. Spot-checked (1.0,0),(0.8,0.2),
(0.5,0.5) refuse; (0.5,0.6),(2.0,0) serve. Direction canonicalised so
first non-zero component >=0 (−0.0==0.0 handled).

### Callers
No production or test caller passes n_theta. All use (gamma[,kappa]) or
(gamma,0.0). surrogate._caustic_reach = caustic_geometry(gamma,0)[0]
bit-identical. Signature change safe.

### Tests run GREEN this pass
- Wp1* (surrogate): 16 passed (oracle stage-1 vs geometry._caustic_source,
  closed==dense-scan, 720-scan-correction anti-vacuity, byte-identity).
- test_lensing_ppgo_map.py: 37 passed.
- bandsplit TruncationOnRefusal + exterior_admission: 50 passed.

### INS-2-001 — RESOLVED
bandsplit ANGLES updated to tuple(k*pi/8 for k in range(-4,5)) and comment
now says "nine source angles ... symmetric fan [-pi/2,+pi/2]". Fixture
_w_star monotone-decreasing so negative angles are looser; +pi/2 still
dominates the min => still passes AND now faithfully mirrors production.

### Notes / carry-forward
- Plan listed test_lensing_ghost.py as expected-changed; it wasn't. Its
  caller uses 2 positional args (gamma,kappa) — unaffected. Benign
  plan deviation, not a finding.
- No open findings.
