# Professor — short-term (2026-07-28: Born b1, low-w form, ladder, saddle,
# F023 gap sweep)

## A — b1 / a0 CLOSED FORMS (both parities; derived + pinned vs F_op)
    b1 = -lam*(lam + gamma*P)/detA = -lam*(2*lam*r0_sq - x0_dot_y)/(det_a*r0_sq)
    a0 = -lam*gamma*P/detA         = -lam*(lam*r0_sq - x0_dot_y)/(det_a*r0_sq)
    P = cos2(theta_x0 - beta);  point mass -> b1 = -1 (placeholder had the
    WRONG SIGN).  Saddle: det_a<0 flips both.  Verified 0.01-2% at |y|=12,20.

## B — LOW-w FORM (derived, verified, NOT NEEDED: log_w spline absorbs it)
    F_low = sqrt(mu_macro) e^{iwc} {1 + pi w/4
            + i(w/2)[ln(w/2) + gamma_E + 2 ln Lambda]},  c = phi_geo + 0.5 ln r0_sq
    Lambda = sqrt(2 detA/(lam + sqrt detA))
Reduces to Takahashi & Nakamura 2003 Eq.(18).  T&N's "-2 phi_m" is the
MIN-SUBTRACTED frame; my earlier "- s" was WRONG.

## C — a0 MUST NEVER BE APPLIED BELOW THE SPLIT (supersedes F023's advice)
MECHANISM (not a gamma threshold): F(w->0) = sqrt(mu_macro) EXACTLY (F009),
but the carrier gives sqrt(mu_macro)*(1 + a0/q2r).  a0 is a RESOLVED-image
amplitude correction (valid w*dtau >> 1); below the split it is a constant
F009 violation of size |a0|/q2r.  b1's term carries w, vanishes correctly,
and is harmless (but buys ~nothing).
MEASURED, band [1e-3,0.05], kappa=0, |y| in {3.05,4.2426}, th in {0.3,0.9,1.35},
gamma in {0.45,0.5,0.6,0.65,0.7,0.75}:
    lead    sz 2.0e-2..5.3e-2, N(log w)=4-5
    b1 only sz 2.1e-2..4.9e-2, N=4-5      (indistinguishable from lead)
    a0+b1   sz 2.1e-2..1.08e-1, N=4-5     (excess tracks |a0|/q2r exactly)
Y-AXIS (AZIMUTHAL) node counts, theta in [0.05, pi/2), 65 pts:
    N(F) == N(lead) EXACTLY at every (gamma, w, radius) sampled:
      g=0.45: 4,4,6,7,13,20   g=0.60: 4,4,9,9,23,33   g=0.75: 4,4,14,14,33,64
    N(a0+b1): 10-11 (g=0.45), 17-20 (g=0.60), 33-44 (g=0.75) where N(lead)=4
=> NO CROSSOVER GAMMA.  a0 hurts at EVERY gamma including 0.45; the penalty
   grows 2.5x -> 5x -> 11x as gamma 0.45 -> 0.60 -> 0.75.
METHOD ERROR THAT HID THIS IN F023: F023's y-plane counts swept |y| RADIALLY
at fixed theta=0.3.  The a0 pathology is AZIMUTHAL (q2r depends strongly on
theta via x0 = (y1/(1-gamma), y2/(1+gamma))).  Radial sweep at th=0.3 gave
N=4, sz 2.4e-2; azimuthal at the same gamma gives N=11, sz 1.2e-1.
RULE: LEAD-ONLY below the split, BOTH parities.  One rule, no branch.

## D — ASTROID EXTENT: EXACT CLOSED FORM (positive parity)
Critical curve u = gamma cos2theta +- sqrt(1 - gamma^2 sin^2 2theta),
u = 1/|x|^2.  On the y2 axis u = 1 - gamma, giving the outermost cusp

    max |y| on the astroid = 2*gamma/sqrt(1-gamma)      (kappa=0)
    general kappa: sqrt(lam) * 2*gamma'/sqrt(1-gamma'),  gamma' = gamma/lam

Verified to 4 dp at gamma = 0.60/0.70/0.75/0.80/0.85 ->
1.8974/2.5560/3.0000/3.5777/4.3894 (measured identical).
FENCES (solve 2s^2 + R s - 2 = 0 with s = sqrt(1-gamma), R = target |y|):
  inner edge |y| = 3.0     breached at s = 1/2            -> gamma = 3/4 EXACT
  outer edge |y| = 3*sqrt2 breached at s = (sqrt34-3sqrt2)/4 -> gamma = 0.842329
So the annulus is fully exterior for gamma < 3/4, STRADDLES the caustic for
3/4 <= gamma < 0.8423, fully interior above.  (My earlier "<~0.75" was a
geometric guess; it is exactly 3/4.)
SADDLE analogue is NOT this formula (max is off-axis); measured extents:
gamma 1.005->=6.0, 1.01->5.90, 1.02->3.71, 1.05->2.49, 1.10->1.99, 1.6->1.95.

## E — BAND SPLIT + TABLE (positive parity, gamma <= 0.75, kappa=0)
Split `w*dtau ~ 4` HOLDS across the whole swept range; it simply moves DOWN
in w as gamma grows because dtau grows (|y|=4.2426, th=0.3: dtau 10.8 at
gamma=0.5 -> 19.3 at 0.75).
  [1e-3,0.05] lead : sz 2.0e-2..5.3e-2  N(log w)=4-5   N(y-axis)=4
  [0.05,0.5]  lead : sz 4.2e-2..1.17e-1 N(log w)=6-31  N(y-axis)=6-14
  [0.5,8]     ppGO : sz 2.4e-3..2.0e-1  N(log w)=4-27
DEGRADATION vs F023 (which stopped at gamma=0.45): log-w N in [0.05,0.5]
grows 17->19->21->23->26->31 (|y|=4.2426, th=0.3) as gamma 0.5->0.75; ppGO N
in [0.5,8] grows 12->13->15->18->21->27 (|y|=3.05, th=0.3) and ppGO size at
|y|=3.05, th=1.35 rises 5.0e-2->9.7e-2->1.6e-1->2.0e-1->2.0e-1->1.8e-1 as the
inner edge approaches the y2-axis cusp (which sits exactly at |y|=3.0 when
gamma=3/4).  Gradual, no blow-up; the ladder HOLDS to gamma = 3/4.
NOT SWEPT: kappa != 0 (production prior pins kappa=0) and beta != 0.

## F — SADDLE (gamma > 1) findings
Census (1,1) both Morse index 1 in the exterior annulus for gamma in
[1.05,1.6].  Split currency is `w*dtau ~ 4`, NOT `w*r0_sq` (measured
r0_sq/(2 dtau) = 0.16..35.6).  Lead-only below the split: sz 1.0e-2..7.4e-2,
N=4 on log w AND 4 in theta.  Complex ghost is HARMFUL on the saddle
(gamma=1.6,|y|=4.243,w=5: ppGO 1.4e-3/N=4 -> +ghost 4.2e-2/N=14): its
admission set flips across theta, and `geometry.ghost_kernel` pins the sqrt
branch via `reference_amplitude = exp(-0.5j pi)` justified by "the two real
images continue into a Morse-index-1 saddle" — a POSITIVE-PARITY statement.
REFUSE the complex ghost for det A < 0 until re-derived.
b1 zero locus `1 + gamma' P = 0` = straight rays y2/y1 = +-((lam+gamma)/
(gamma-lam))^1.5 in the eigenframe; moot once a0/b1 are dropped.

## G — measurement caveat to honour
Above-split theta node counts demodulate by the SINGLE carrier
exp(iw phi_geo); the residual there inherits the other image's carrier, so
those counts are PESSIMISTIC.  The correct object is the SACR-C switched
envelope (measured elsewhere at greedy N=20-25 on saddle configs).  Residual
SIZES are demodulation-independent and stand.  Below the split there is one
stationary point, so those counts (N=4) are sound.

## Cross-references
`professor/microlensing_chang_refsdal` (F009/F009-S, SACR-C, mass-sheet),
`professor_code_observations` (add: a0-violates-F009-below-split,
astroid-extent closed form, ghost-branch-is-positive-parity, ppGO reaches
down to w~0.5, C1/C2 needed for smoothness).
