# Professor short-term (session 2026-08-06): WP1/WP2 cusp-adapted wedge axis review

Context: reviewing InteriorWedgeChart 4th-spline-axis fix. The chart currently fits
its angular axis against caustic ARC LENGTH s (from_wedge_engine ~line 3811,
cumulative_trapezoid of geometry.caustic_speed). Measured held-out eps on an axis-
adjacent transverse cut: s-axis 3.9e-2 to 6.1e-2; raw-theta 4e-3 to 1.2e-2; u=theta^(2/3)
2.8e-4 to 6.9e-4 (hits engine/spline noise floor ~9 nodes, matches retired ffin baseline
3.42e-4). Interior (non-axis) tiles already pass at 3.82e-4; only axis-TOUCHING tiles
fail (1.29e-1).

Physics root cause (confirmed correct by first principles): astroid CUSPS at
theta_wedge=0, pi/2. Near a cusp r_caustic ~ const - coeff*d^(2/3) (2/3 EXACT, gamma-
universal). Because r = |y|/r_caustic NORMALISES by r_caustic, the theta^(2/3) non-
analyticity contaminates EVERY r along the axes (envelope velocity diverges as
theta^(-1/3)). Arc length is WORSE than raw theta: caustic_speed vanishes LINEARLY at a
cusp so s ~ theta^2, giving f(s^(1/3)). u = d^(2/3) linearises the cusp singularity.

## Q1-Q7 follow-up rulings (this session)

Q1 OFFSET: affine shift-to-zero of u is CORRECT and HARMLESS. Subtracting a constant does
not change spline conditioning (interp is translation-invariant; the 2/3 flattening lives
in the abscissa SPACING which the offset preserves). Prefer offsetting to satisfy
_validate_theta_to_s "start at 0". Do NOT store raw d^(2/3) on the upper tile: raw
(pi/2-theta)^(2/3) is DECREASING in theta and fails the strictly-increasing validator; the
sign-flip+offset u=(pi/2-theta_lo)^(2/3)-(pi/2-theta)^(2/3) is the correct monotone rewrite.
Both proposed forms are right.

Q2 WAIST CLOSED FORM: CODE-CHECKED. r_caustic (geometry.py:1567) is a NUMERICAL ray-
intersection inversion (max forward source-plane radius on a ray, refined to float64), NOT
the closed-form parametric radius. So argmin_theta r_caustic has no trivial closed form.
The caustic CURVE is closed form (y_i=p_i*r*T_i, u=e*cos2theta + branch*sqrt(1-e^2 sin^2
2theta), e=gamma/lam) but the parametric-theta to source-plane-RAY-theta map differs — do
NOT assume the parametric argmin equals the ray argmin. RECOMMEND numerical minimize_scalar
over r_caustic (brief's definition, which the tiler needs anyway); do NOT invest in a
closed form. Free oracle r_caustic(gamma,theta_waist)=gamma is a VALID non-circular value
pin (pins location via an independent identity, not via the same argmin routine). Tolerance:
pin the VALUE r==gamma to 1e-6 abs, NOT theta_waist itself. At a quadratic minimum
|r-gamma| ~ curvature*dtheta^2, so a 1e-4 theta error gives ~1e-8 r error — 1e-6 is
comfortable; minimize_scalar xatol default ~1e-4 on theta is fine.

Q3 SIDE THREADING: INSIST on explicit threading. Repo #1 bug class is train/serve skew
from re-derived conventions (professor_code_observations documents 3: farfield decay gate,
ghost frame, min-delay). Tiles never straddle the waist so internal re-derivation is safe
TODAY, but a future waist-convention change would silently skew from_wedge_engine's midpoint
test. Thread side as an explicit tile attribute set by the tiler; from_wedge_engine consumes,
never recomputes. Belt+suspenders: internally assert threaded side agrees with midpoint-vs-
waist (fails loud on skew).

Q4 NODE-EXACT TOL: 1e-7 CONFIRMED for the u-map interp-through-map step. Node exactness at a
served grid node stays ~machine (6.33e-16) only on the identity path (query lands on a fine
node). np.interp BETWEEN fine nodes gives piecewise-linear error ~ dtheta_fine^2 * |u''|;
u'' ~ theta^(-4/3) is worst at the axis-touching tile. FLAG: verify the arc-length build's
6e-9 budget still holds for u near the cusp, where u'' is MORE singular than s'' was. If
fine-node spacing is uniform-in-theta the axis-adjacent interp error could exceed 6e-9.
Recommend uniform-in-u fine-node spacing for the MAP itself to equidistribute interp error.

Q5 ORACLE COST: one transverse cut ~33 theta at fixed (gamma,r) over w-grid costs
n_theta*(1 quartic solve + n_w*0.41ms batched). n_w~12: ~0.18s/cut. Three abscissae x three
node counts REUSE one dense truth cut (numpy fit free), so whole test ~1-2 engine cuts <1s.
Small w-grid FINE: abscissa RANKING s>theta>u is w-universal (eps grows linearly in w but the
ordering does not). FALSIFY: assert(err_u < err_theta < err_s) at EACH node count AND
assert(err_u < 1e-3). Correct non-brittle form — never assert exact ratios (171x is a
this-geometry artifact; the ORDERING is the physics). Optional decisive pin: err_u < 7e-4
(~2x ffin baseline 3.42e-4).

Q6 SUBDIVISION TEST: (i) and (ii) carry the real falsification value; (iii) is a bonus.
(i) assert >=2 angular columns split at _wedge_theta_waist — structural, cheap, KEEP.
(ii) assert u-subdivision splits at u-MIDPOINT mapped back to theta (NOT theta-midpoint) —
load-bearing correctness pin, exactly the lever a Simplifier would wrongly collapse. KEEP,
highest value. (iii) build-coarse-then-halve-then-regate is feasible <60s only at tiny grids
(n_gamma=1,n_r=1,n_theta 3->5, n_w~8) ~ handful of cuts <10s; DO IT if budget allows because
it tests the FEEDBACK LOOP whose removal hid the defect for a day. Full retrain-to-green NOT
feasible in 60s and not required.

Q7 PHYSICS RISK: (a) waist IS regular — CONFIRMED by code. geometry.py docstring (F044):
the deltoid's three cusps are the interior |y'|=0 roots; the wedge EDGES (theta=0,pi/2) are
where the theta-parametrization diverges. theta_waist is an interior extremum of a smooth
branch; r_caustic is C-infinity there — tile boundary at the waist is safe. (b) min(theta,
pi/2-theta) global map is genuinely worse — CONFIRMED: KINK (C0 not C1) at crossover, deriv
jumps +1 to -1; a spline across it sees a spurious interior non-analyticity (trades edge cusp
for interior corner). Per-tile monotone maps have no kink because the waist split guarantees
no tile straddles the crossover.

RESIDUAL RISK I FLAG (not in brief): tiles cut at BAND-CENTER waist; waist migrates ~0.01
rad within the 0.04 gamma band. A tile near the band edge could have its true per-gamma waist
land just inside the tile, reintroducing a tiny wrong-cusp sliver for off-center gamma. 2nd
order given the measured margin — note as known small residual; if an axis-adjacent tile ever
fails its bar near a band edge, per-gamma waist drift is the suspect.
