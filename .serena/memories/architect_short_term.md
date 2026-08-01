# Architect Short-Term Observations

Build 1e-lobe (2026-07-31): Replace lobe-interior chart's uniform theta_local
grid with s = sqrt(theta_max - theta_local) reparametrization. theta_max = upper
bound of theta_local tile (coincides with cusp ray). s DECREASES as theta
increases (s=0 at theta_max, s=s_max at theta_min). Spline fit in s (increasing
axis); training nodes uniform in s -> clustered near theta_max. Store
theta_local_to_s (2,N_map) map per brief requirement; use np.interp at serve.
Key: spline built at rep_gamma's theta_max; same theta_max MUST be used at serve
(stored on map, not recomputed from query gamma — avoids train/serve skew).
Professor: theta_max per tile = tile upper bound (a cusp ray geometrically derived
from wedge half-angle); one-sided regularization (upper edge). Simplifier: the
formula is algebraic (no quadrature) so a (2,N) table is over-engineering, but
brief explicitly requires stored map for testability/validation; also need stored
theta_max to avoid train/serve skew. Acceptance: arc-length bar-margin pattern
(s-coord stays below 0.05 bar under bound shifts, uniform trips it). Files:
surrogate.py (LobeInteriorChart dataclass + from_lobe_values + _assemble +
_evaluate_chart + serialization + schema bump), surrogate_training.py
(_build_lobe_chart node placement + map construction). 1 Coder WP.


Build 1e-lobe (2026-08-01, PLAN FINAL): Production code ALREADY in working tree
(surrogate.py uncommitted diff ~104 ins). 1 Coder WP: review-and-commit only
(no new code to write). Tests via domain_test_descriptions (Test Dev):
(1) coordinate round-trip (theta→s→theta, tolerance ~1e-4 rad, closed-form
oracle s = sqrt(span)-sqrt(theta_max-theta)),
(2) NPZ persistence (add theta_to_s bit-identity assertion to existing
LobePersistenceTestCase),
(3) bound-shift margin (same structure as tube's ArcLengthBoundShiftMarginTestCase:
shift theta_local bounds ±0.01 rad, assert sqrt-edge eps < 0.05 bar, uniform
trips it — encode measured invariant, not swing claim per F042 lesson),
(4) V1 identity-path byte-identity (synthetic V1 chart with theta_to_s=None
serves identically to pre-build behavior — no-regression guard).
Professor: formula correct (s increases monotonically, dy/ds finite at edge),
gamma-independent map (simpler than tube), round-trip tol ~1e-4 near edge,
endpoint np.interp clamping safe. Simplifier: lean plan, (2,2001) map justified
by brief mandate + train/serve anti-skew, test (b) can be narrowed to add
assertion to existing LobePersistenceTestCase rather than new class.



Build 1e-farfield-port (2026-07-31): PURE PORT — restore (s,d) far-field
coordinate on FarFieldChart from git ref refs/sdk/farfield_port_wip (production
+ 3 already-ported test files + DATA_CONTRACTS.yaml), then Test Dev ports ~38
remaining construction sites (from_values/select_chart signature change) across
4 files keeping every oracle/tolerance IDENTICAL. 0 numerical changes; a VALUE
failure = real finding, STOP. Structure: 1 Coder WP (restore+collect-verify at
1171, NO memo per Simplifier) + 4 Test-Dev domain_test_descriptions (one per
REMAINING file, each NAMES its file to dodge F057 cross-suite budget blowup that
killed prior attempt). contracts_changelog + render = doc-sync (NOT a WP).
Professor watch-for: A1 axis order/meaning swap (SILENT, keyword-call every arg)
+ A2 envelope_definition tag (serve dispatches on it) = top masqueraders; A3
eta_overlap_min default drift; A4 image_count/parity transpose; A5 DON'T
regenerate envelope tensors (reuse original oracle values); Q2 per-gamma s-map
must match gamma_grid (single-gamma passes guard, breaks claim); Q3 gated file:
SelfFalsification negative fixture must still CONSTRUCT (rejection exercised, not
constructor); rho=1-on-caustic normalization does NOT carry to (s,d) -> if a site
asserts it, report as unportable (accept #3), don't weaken. surrogate_training.py
is GATED/slow: verify via check_gated_test_drift.py exit 0, do NOT run slow tier
in-build.


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
