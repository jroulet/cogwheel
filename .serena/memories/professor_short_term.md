# Professor — short-term (2026-07-28: Born b1, low-w form, ladder, SADDLE)

## PART A — b1 / a0 PINNED (both parities; derived + confirmed)
`_born.py::_born_factors` placeholder `b1 = 1.0` — DISCHARGED.
Shift `x = x0+u` in `phi = 0.5 x.A.x - x.y + 0.5|y|^2 - ln|x|`, factor
`exp(iw phi(x0))` (== `phi_geo`), moments `<u_i u_j> = (i/w)(A^{-1})_{ij}`:

    b1 = -lam*(lam + gamma*P)/detA = -lam*S/|x0|^2
       = -lam*(2*lam*r0_sq - x0_dot_y)/(det_a*r0_sq)
    a0 = -lam*gamma*P/detA = -lam*(lam*r0_sq - x0_dot_y)/(det_a*r0_sq)
    S = x0.A^{-1}.x0 ,  P = (x0.Q.x0)/|x0|^2 = cos2(theta_x0 - beta)

Point mass -> b1 = -1 exactly (placeholder had the WRONG SIGN).  Pinned vs
`F_op` to 0.01-0.9% at |y|=12,20, positive parity AND saddle (saddle:
det_a<0 flips both signs; b1_fit/a0_fit match to 0.02-2%).

## PART B — LOW-w CHANG-REFSDAL FORM (derived, verified, NOT NEEDED)
    F_low = sqrt(mu_macro) exp(iwc) {1 + pi w/4
            + i(w/2)[ln(w/2) + gamma_E + 2 ln Lambda]} + O(w^2)
    c = 0.5(|y|^2 - x0.y) = phi_geo + 0.5 ln r0_sq
    Lambda = 2 sqrt(a1 a2)/(sqrt a1 + sqrt a2) = sqrt(2 detA/(lam+sqrt detA))
Via Frullani + Legendre `Int_0^pi (P-Q cos)^{-s} = pi R^{-s} P_{-s}(P/R)`,
`dP_nu/dnu|_0 = ln((1+z)/2)`.  `s` does NOT appear in the O(w) bracket;
beta cancels; gamma_E and pi w/4 coefficients unchanged from the point mass.
Reduces to Takahashi & Nakamura 2003 Eq.(18).  CORRECTION: T&N's "-2 phi_m"
is the MIN-SUBTRACTED frame; in the absolute frame it cancels against
e^{iw phi_m} — my earlier "- s" was WRONG.  Residual ~ 0.02 (w r0_sq)^2.
NOT NEEDED: `w ln w = e^u u` is entire in u = ln w, so a log_w spline
absorbs it at ZERO node cost (measured identical N at eps 4e-3..1e-5).

## PART C — POSITIVE-PARITY LADDER (measured, eps = 4e-3 abs of max|F|)
Split at `w*dtau ~ 4` (== `w*r0_sq ~ 8` there, since dtau ~ r0_sq/2).
  [1e-3,0.05]: carrier N=4 (sz 3e-2..4e-2); ppGO DIVERGES (1e3)
  [0.05,0.5] : carrier N=7-15; ppGO sz 1.0-2.3
  [0.5,8]    : carrier ALONE N=161-241 (BEAT); carrier+real2nd N=4-8;
               ppGO N=4-8 sz 2.5e-3..2.5e-2; +complex ghost 1.6e-3..1.7e-2
Real 2nd image needs its FULL C1/C2 `image_kernel` — leading sqrt|mu| alone
leaves a beat (my earlier 121-241-node claim was that artifact).

## PART D — MACRO SADDLE (gamma > 1), exterior annulus
Census (1,1), both Morse index 1, at every sampled annulus point for
gamma in [1.05,1.6].  Geometry is violently anisotropic: x0_i = y_i/a_i with
a1 = lam-gamma < 0, a2 = lam+gamma, so r0_sq swings 1700x with theta
(3721 -> 2.2 at gamma=1.05; 25.8 -> 1.38 at gamma=1.6).  1/q2r reaches 0.73.
- SPLIT CURRENCY: `w*dtau ~ 4` HOLDS on the saddle; `w*r0_sq` does NOT
  (r0_sq/(2 dtau) measured 0.16 .. 35.6).  Use the two real images' Fermat
  delay difference, which `geometry` already computes.
- (a0,b1) ARE HARMFUL ON THE SADDLE.  |a0|,|b1| ~ 1/(gamma'-1) (a0=10.2 at
  gamma=1.05), so a0/q2r is O(1) where q2r is small.  Measured, band
  [1e-3,0.05]: lead-only sz 1.0e-2..7.4e-2 with N(log w)=4 AND N(theta)=4;
  full carrier sz 1.7e-2..1.42 with N(theta)=23-65.  The carrier injects
  theta-structure that is not in F.  USE LEAD-ONLY BELOW THE SPLIT.
- COMPLEX GHOST IS HARMFUL ON THE SADDLE.  gamma=1.6,|y|=4.243,w=5:
  ppGO sz 1.4e-3 N(theta)=4 -> ppGO+ghost sz 4.2e-2 N(theta)=14.  Two
  causes: (i) the admission set flips across theta (43-54 of 65 admitted,
  1 flip) making the served model discontinuous; (ii) `geometry.ghost_kernel`
  pins its sqrt branch via `reference_amplitude = exp(-0.5j pi)` justified by
  "the two real images continue into a Morse-index-1 saddle" — a POSITIVE-
  PARITY statement.  On the saddle both real images are ALREADY index 1.
  DO NOT ADMIT THE COMPLEX GHOST ON THE SADDLE until that branch is
  re-derived.
- b1 zero locus (`1 + gamma' P = 0`) is a pair of STRAIGHT RAYS through the
  origin in the eigenframe: `y2/y1 = +- ((lam+gamma)/(gamma-lam))**1.5`
  (verified: gamma=1.6 -> slope 9.021 -> theta_y = 1.4604 rad).  Sits within
  0.004-0.11 rad of the positive-eigenvalue axis, i.e. inside the
  small-q2r region.  MOOT once (a0,b1) are dropped on the saddle.

## PART E — THE REAL GAP IS IN gamma, NOT w  (caustic extent)
Measured max |y| of the 4-image (caustic-interior) region, 241^2 grid:
  SADDLE : gamma 1.005 -> >=6.00 | 1.01 -> 5.90 | 1.02 -> 3.71 |
           1.05 -> 2.49 | 1.10 -> 1.99 | 1.30 -> 1.70 | 1.60 -> 1.95
  POSITIVE: gamma 0.20 -> 0.40 | 0.45 -> 1.20 | 0.60 -> 1.85 | 0.75 -> 2.95 |
           0.85 -> 4.35 | 0.90 -> 5.65 | >=0.95 -> >=6.00
So the annulus 3.0<|y|<=4.2426 is FAR-EXTERIOR only for
`gamma <~ 0.75` and `gamma >~ 1.03`.  For `0.75 <~ gamma <~ 1.03` it
STRADDLES or lies INSIDE the caustic (fold crossings inside the tile;
census (0,1,1,1)); ~17% of the prior's uniform (0,1.6) shear range.
CAVEAT ON MY OWN EARLIER REPORT: the positive-parity ladder was measured at
gamma in {0.2,0.25,0.3,0.45} only — it is established for gamma <~ 0.75,
NOT for the whole prior.  The coordinator's `LobeInteriorChart` (deltoid
interior) genuinely reaches |y| ~ 3-6 for gamma <~ 1.03 — the interior and
exterior programs are NOT independent near the wall.

## PART F — measurement caveat to honour
My theta-direction node counts demodulate by the SINGLE carrier
`exp(iw phi_geo)`.  Above the split the residual inherits the OTHER image's
carrier (dtau varies ~25 per rad of theta, so at w=5 fringes are ~0.05 rad),
so those counts are PESSIMISTIC.  The correct object is the SACR-C switched
envelope, already measured elsewhere at greedy N=20-25 on saddle configs
(see `professor/microlensing_chang_refsdal`).  Residual SIZES quoted are
demodulation-independent and stand.

## Cross-references
`professor/microlensing_chang_refsdal` (F009/F009-S, SACR-C, mass-sheet,
negative-parity report), `professor_code_observations` (_born.py entry now
dischargeable; add ppGO-down-to-w~0.5, C1/C2-smoothness, ghost-branch-is-
positive-parity, caustic-extent-vs-gamma).
