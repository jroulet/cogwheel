# Professor short-term (session 2026-07-30, Build 1d lensing caustic-geometry test authority)

Consulted on TEST design/tolerances for Build 1d in `cogwheel/lensing/` (caustic
geometry). Two production changes: (A) delete `_WEDGE_EPS=1e-3` angular standoff at 6
linspace sites -> sample wedge CLOSED to the edge; (B) `_tube_normal` replace forward
finite-difference tangent (critical_point diff) with analytic `y'/|y'|` from
caustic_derivatives.

## Governing math fact (drives every tolerance)
The macro-saddle wedge edge `dtheta = theta_max - theta = 0` is a REGULAR point of the
caustic (fold turnaround) but the theta-parametrization is SINGULAR there:
y'(theta) ~ A*dtheta^(-1/2), y'' ~ (A/2)*dtheta^(-3/2). Square-root branch point.
Reparam s=sqrt(theta_max-theta) makes position finite, tangent nonzero. So:
- tangent DIRECTION y'/|y'| has a finite limit at the edge (the fold tangent) even
  though |y'| -> inf. Orientation is well-defined up to sign in the limit.
- Near the edge, fd tangent and analytic y' both -> the same limiting direction, but
  fd averages over [theta, theta+h] where |y'| varies as dtheta^(-1/2) -> fd is biased
  toward the STEEPER (closer-to-edge) end. Orientation (sign of dot) is robust; the
  bias is in magnitude/angle, not sign, away from a measure-zero neighborhood.

## Authority given (test tolerances)
1. inward_sign gate: dot(t_fd, t_analytic) > 0 design is SOUND (non-circular). fd step
   h: use the incumbent's OWN step (1e-6 in theta) so the test replicates the shipping
   method exactly; that is the falsification target, not an idealized derivative.
   Legit fd/analytic orientation DISAGREEMENT regime: only within ~h of the edge where
   the fd forward-window straddles the turnaround (theta+h > theta_max) OR where |y'|
   is so large the O(h) chord curves — guard: assert serve-theta stands off the edge by
   >> h (production arcs do; verify per arc). Recommend dot > 0 AND the STRONGER
   angle check dot(t_fd,t_analytic) > cos(few degrees) as a tripwire for near-edge arcs.
2. unit+perp tolerance: NOT 1e-15. After normalize + left-perp of a float64 vector,
   |n|-1 and dot(n,y') accumulate several rounding ops -> use 1e-14 (few-ulp). 1e-15
   (~4.5 ulp) will flake. Perp: |dot(n_hat, yprime_hat)| < 1e-14.
3. closure-gap == 0.0 EXACT: sound IFF both branch endpoints are the SAME float
   expression evaluated at the SAME theta extreme (bit-identical inputs -> bit-identical
   outputs, IEEE deterministic). Confirm the loop actually reuses ONE computed endpoint
   (or two calls with identical args). If the two endpoints come from DIFFERENT code
   paths (+branch vs -branch) that only meet analytically at discriminant=0, use
   < 8*eps*scale, NOT ==0.0. Ask which construction; default to a few-ulp bound if unsure.
4. coverage non-shrink: freeze incumbent literals (cusp=6, arc count, reach) NO HEAD.
   reach: exact-equal is fragile IF endpoint sampling changed (it did — closure). Use
   reach within 1 grid-step relative tol, or assert reach unchanged to the sample
   resolution. Edge-endpoint risk: caustic_derivatives raises AT the edge (inclusive),
   critical_point SERVES at edge -> a linspace closing on the edge yields a FINITE
   endpoint position (critical_point ok) but any per-node derivative/tangent call will
   raise LensDomainError exactly there -> ensure the arc/cusp census does not count a
   raised endpoint as a new cusp, and does not NaN. Assert no NaN in every served loop.
5. refusal pins: build dtheta=-1e-12 as theta = theta_max + 1e-12 (add to the float
   theta_max), NOT theta_max*(1+eps) (relative). theta_max=0.5*arcsin(lam/|gamma|) is a
   float; adding 1e-12 lands strictly outside by ~1e-12 in angle (safe, >> ulp(theta_max)
   ~ 1e-16). Pin: critical_point(edge) returns finite; critical_point(theta_max+1e-12)
   raises LensDomainError; caustic_derivatives(edge) raises; caustic_derivatives just
   inside (theta_max-tiny) returns finite-but-large.

## Cross-refs
F041 = tangent-orientation flip silently flips served sides (the exact failure mode gate
1 protects). F044 = this build's finding (edge is regular, 1d deletes _WEDGE_EPS).
