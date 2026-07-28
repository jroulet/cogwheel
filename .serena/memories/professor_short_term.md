# Professor — short-term (2026-07-28: Born b1, low-w form, ladder, saddle,
# F023 gap sweep, caustic-extent closed form)

## A — b1 / a0 CLOSED FORMS (both parities; pinned vs F_op to 0.01-2%)
    b1 = -lam*(lam + gamma*P)/detA = -lam*(2*lam*r0_sq - x0_dot_y)/(det_a*r0_sq)
    a0 = -lam*gamma*P/detA         = -lam*(lam*r0_sq - x0_dot_y)/(det_a*r0_sq)
    P = cos2(theta_x0 - beta).  Point mass -> b1 = -1 (placeholder had the
    WRONG SIGN).  Saddle: det_a < 0 flips both.

## B — LOW-w FORM (derived, verified, NOT NEEDED — log_w spline absorbs it)
    F_low = sqrt(mu_macro) e^{iwc}{1 + pi w/4
            + i(w/2)[ln(w/2) + gamma_E + 2 ln Lambda]},  c = phi_geo + 0.5 ln r0_sq
    Lambda = sqrt(2 detA/(lam + sqrt detA));  cf. Takahashi & Nakamura 2003 Eq.(18)
    (their "-2 phi_m" is the MIN-SUBTRACTED frame; my earlier "- s" was WRONG).

## C — a0 MUST NEVER BE APPLIED BELOW THE SPLIT (supersedes F023)
F(w->0) = sqrt(mu_macro) EXACTLY (F009); the carrier gives
sqrt(mu_macro)*(1 + a0/q2r).  a0 is a RESOLVED-image correction; below the
split it is a constant F009 violation of size |a0|/q2r.  b1's term carries w
and is harmless.  NO CROSSOVER GAMMA — a0 hurts at every gamma including
0.45 (azimuthal N: 4 -> 11 at g=0.45, 4 -> 20 at 0.60, 4 -> 44 at 0.75).
F023 missed it by sweeping |y| RADIALLY at fixed theta=0.3; the pathology is
AZIMUTHAL (q2r depends strongly on angle).  RULE: lead-only below the split,
both parities.

## D — CAUSTIC EXTENT: UNIFIED CLOSED FORM (new, this session)
Critical curve `u = 1/|x|^2` solves `u^2 - 2 gamma cos2th u - (1-gamma^2) = 0`;
caustic `y = ((a1-u)x1, (a2-u)x2)`, `a1,a2 = 1 -+ gamma`.  Eliminating theta:

    |y|^2(u) = 2u - 3 + 2 gamma^2/u + (1-gamma^2)/u^2 ,  u in [|1-gamma|, 1+gamma]
    f'(u) = (2/u^3)(u-1)(u^2 + u + 1 - gamma^2)
    stationary: u = 1  and  u_c = (sqrt(4 gamma^2 - 3) - 1)/2
    u_c > 0  <=>  gamma > 1   (interior extremum exists ONLY on the saddle)
    at u_c:  gamma^2 = u_c^2 + u_c + 1  and  f(u_c) = 4 u_c + 1/u_c - 2
    endpoints: f(1+g) = 4g^2/(g+1); f(1-g) = 4g^2/(1-g); f(g-1) = 4(g-1)

POSITIVE PARITY: u^2+u+1-g^2 > 0 for u>0, so f falls then rises -> max at the
endpoint u = 1-gamma.  **max|y| = 2 gamma/sqrt(1-gamma) for ALL gamma < 1**
(unconditional; the encoded fence is right).
SADDLE: u_c is ALWAYS interior (u_c > gamma-1 <=> gamma > 1).
    **max|y| = sqrt(max(4 u_c + 1/u_c - 2, 4 gamma^2/(gamma+1)))**
Verified to 4 dp vs direct caustic parametrisation at 16 gammas both parities.

NON-MONOTONICITY IS REAL — a CUSP SWITCH: outermost point is the OFF-AXIS
cusp (u_c) for gamma < 1.177651, the ON-AXIS cusp (u = gamma+1) above.  The
u_c branch falls, 2g/sqrt(g+1) rises => minimum extent 1.596072 at
gamma = 1.177651.  (Old 241^2 grid table underestimated: it misses the thin
spike. gamma=1.05 grid 2.491 vs true 3.008; 1.02 grid 3.712 vs true 4.886.)
DIVERGENCE AT THE WALL: both sides go as |gamma-1|^{-1/2} —
positive 2/sqrt(1-g), saddle 1/sqrt(2(g-1)); the astroid is 2*sqrt2 = 2.83x
larger at equal distance from the wall.

FENCES (exact in radicals; solve 4v^2 - (R^2+2)v + 1 = 0, gamma = sqrt(v^2+v+1)):
  saddle falling branch, |y| = 3.0     -> gamma = sqrt((189-15 sqrt105)/32)
                                                = 1.0502342
  saddle falling branch, |y| = 3 sqrt2 -> gamma = sqrt(63-24 sqrt6)/2
                                                = 1.0261879
  saddle RISING branch re-entry: 4g^2/(g+1) = 9 -> 4g^2-9g-9=0 -> gamma = 3 EXACT
                                 = 18 -> 2g^2-9g-9=0 -> gamma = (9+sqrt153)/4
                                                      = 5.342329
So the annulus is exterior for 1.0502342 < gamma < 3 (NOT "all gamma above");
prior tops at 1.6 where max|y| = 1.9846, so safe with margin 1.51x.

SPIKY — YES, same character as the astroid.  Angular width of the caustic
beyond radius R (exact parametrisation, x4 off-axis cusps; UPPER bound on the
inside-fraction since caustic-reaches-R is wider than ring-inside-at-R):
  g=1.02 (max 4.886): beyond |y|=3.0 -> <=3.95% of 2pi; beyond 4.243 -> <=0.78%
  g=1.03 (max 3.949): beyond 3.0 -> <=2.64%;  beyond 3.6 -> <=0.73%
  g=1.045(max 3.182): beyond 3.0 -> <=0.67%
  g=1.05 (max 3.008): beyond 3.0 -> <=0.029%
  (positive parity for comparison: g=0.80 -> <=0.58%, g=0.90 -> <=2.98%)
721-pt ring scans measure 1.66% inside at g=1.02/|y|=3.0 and 0.00% at
g=1.03/|y|=3.6 — the latter is a RESOLUTION ARTIFACT (spike narrower than the
0.0087 rad sampling); trust the parametrisation, not the ring scan.
COST OF SCALAR FENCES on a uniform (0,1.6) prior: positive gamma<3/4 discards
15.6%; saddle gamma>1.0502342 discards 3.1%; per-theta admission recovers
>=96% of both by angle.

## E — SADDLE LADDER (gamma > 1)
Census (1,1) both Morse index 1 in the exterior annulus for gamma in [1.05,1.6].
Split currency `w*dtau ~ 4`, NOT `w*r0_sq` (r0_sq/(2 dtau) = 0.16..35.6).
Lead-only below split: sz 1.0e-2..7.4e-2, N=4 on log w AND 4 in theta.
Complex ghost HARMFUL on the saddle (g=1.6,|y|=4.243,w=5: ppGO 1.4e-3/N=4 ->
+ghost 4.2e-2/N=14): admission flips across theta, and `ghost_kernel` pins the
sqrt branch via `reference_amplitude = exp(-0.5j pi)` justified by "the two
real images continue into a Morse-index-1 saddle" — a POSITIVE-PARITY
statement.  REFUSE the complex ghost for det A < 0 until re-derived.

## F — measurement caveat to honour
Above-split theta node counts demodulate by the SINGLE carrier exp(iw phi_geo);
the residual there inherits the other image's carrier, so those counts are
PESSIMISTIC.  Correct object = SACR-C switched envelope (greedy N=20-25 on
saddle configs).  Residual SIZES are demodulation-independent and stand.

## Cross-references
`professor/microlensing_chang_refsdal`, `professor_code_observations`.
