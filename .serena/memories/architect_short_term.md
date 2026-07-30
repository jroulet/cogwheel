# Architect Short-Term Observations

Build analytic_caustic_reach (F054, 2026-07-30): replace 720-pt polar scan in
ppgo_map.caustic_geometry (1440 critical_point calls/serve, 90% of surrogate
serve) with closed form |y|^2 = lam*[(1-u)^2(1+2u)+e^2(2u-1)]/u^2, lam=1-kappa,
e=gamma/lam. Professor: this is F026 PRIOR ART (already verified to 4dp vs 2M-pt
parametrisation) — cite it. Candidate set: pos-parity {u=1-e (max, =2g/sqrt(1-g)
at k=0), u=1+e}; saddle(e>1) {u=1+e, u=(-1+sqrt(4e^2-3))/2 real iff e>=sqrt3/2 &
u>0, wedge u=sqrt(e^2-1)}; u=(-1-sqrt(4e^2-3))/2 ALWAYS reject; u=1 is a DECOY
from factoring — DROP it. Guards: u>0 strict, |t|=|(u^2-1+e^2)/(2eu)|<=1+1e-12.
Cusp switch off->on axis at g=1.177651. DIRECTION quadrant physically
IRRELEVANT (4-fold caustic symmetry, engine reflection-invariant); canonicalize
"first nonzero comp >=0"; generic case is ON-AXIS (0,+-1)/(+-1,0); off-axis only
1<g<1.177651. Kappa: pure lam-scaling, one path. Parity wall: EXACT float
abs(gamma)==lam raise (NOT a band; nextafter both finite) + lam<=0 refuse.
TOLERANCES: reach-vs-scan 1e-7 rel (brief's 1e-9 UNACHIEVABLE — scan itself only
~1e-8 at n_theta=11520); near-wall oracle MUST be caustic-parametric (r_caustic
bracket-refine / F026 |y|(u)), NOT source-plane ring (misses thin spike);
stationarity self-check 1e-9 ratio OR |y'|<floor disjunction (caustic_derivatives
tangent, exclude wedge edge C5); direction on-axis 1e-9 axis-align, off-axis
5.5e-4 rad quadrant-modulo; sanity pin g=0.9,k=0 -> 5.6921 dir (0,+-1).
Simplifier: 1 Coder WP; surrogate.py UNTOUCHED (pass-through genuine); REMOVE
n_theta kwarg (0 callers pass); INLINE, no helper. SPEC 5.69 stays true (dir
r_caustic unchanged) -> no SPEC update; caustic_geometry docstring says "sweep"
-> Coder must fix with prod.


Build 1e-tube (2026-07-30): TubeChart splines in arc length s=∫|y'|dθ, not
raw theta. Design: ONE new dataclass field `theta_to_s` = (2,N_map) table
[theta_fine, s_fine]; from_values/_assemble gain OPTIONAL map (default =
identity s=theta-theta_lo → existing synthetic fixtures byte-identical);
_evaluate_chart serves v2 = np.interp(theta_inframe, theta_fine, s_fine);
membership+cusp windows STAY in theta. Build path (_build_tube_chart) builds
map at rep_gamma = median(gamma_grid) [Professor: band midpoint minimizes
worst-case eff excursion; single-gamma adequate for topology-stable bands,
degrades only near parity wall eff→1 which existing foot-of-normal skip +
gamma-refine-near-1 already bound]. N_map=2001 (Professor h² bound: coord
err ~3e-8 « round-trip tol 1e-6). F016 bar=0.05 COMPLEX. Knife-edge gate:
swing<5% under ±0.01 rad bound shift (incumbent ±23%). Reuse existing
_wp3_fixture / _wp3_build_and_measure / _heldout_eps scaffolding at
_WP3_GAMMA(=1.55). Simplifier: (2,N) map lean; omit s_grid field (knots
encode it) — correct; identity default a documented seam not footgun.
