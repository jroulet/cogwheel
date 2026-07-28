# Professor — short-term (2026-07-28: Born b1, low-w form, ladder measurement)

## PART A — b1 PINNED (closed form; derived + numerically confirmed)
`_born.py::_born_factors` placeholder `b1 = 1.0`.  Owed item in
`professor_code_observations` — DISCHARGED.
Shift `x = x0+u` in `phi(x)=0.5 x.A.x - x.y + 0.5|y|^2 - ln|x|`, factor
`exp(iw phi(x0))` (== `phi_geo`), Gaussian moments
`<u_i u_j> = (i/w)(A^{-1})_{ij}` (any non-degenerate real symmetric A;
Fresnel Morse phase cancels in the ratio).  Two terms at order 1/q2r:

    b1 = -lam*(lam + gamma*P)/detA = -lam*S/|x0|^2
       = -lam*(2*lam*r0_sq - x0_dot_y)/(det_a*r0_sq)
    a0 = -lam*gamma*P/detA                       <- REAL, w-INDEPENDENT,
       = -lam*(lam*r0_sq - x0_dot_y)/(det_a*r0_sq)   same order, OMITTED
    S = x0.A^{-1}.x0 , P = (x0.Q.x0)/|x0|^2 = cos2(theta_x0 - beta)

Only quantities `_born_factors` already computes.  Point mass -> b1 = -1
EXACTLY (placeholder had the WRONG SIGN).  3 algebraic forms agree 2.2e-14
over 4000 random both-parity draws.  Pinned vs `operator.F_op` (real 2nd
image removed via `image_kernel`): 0.1-0.9% at |y|=12, 0.01-0.4% at |y|=20.
Saddle: identical formulas, detA signed negative, origin
sqrt|mu| e^{-i pi/2}; `1 + gamma_p P` can vanish for gamma_p>1 (b1 -> 0).

## PART B — LOW-w CHANG-REFSDAL FORM (derived + verified, NOT NEEDED)
`|x|^{-iw} = 1 - iw ln|x| + ...` inside the Fresnel integral;
`L = <ln|x0+u|>` via Frullani + Legendre
`Int_0^pi (P-Q cos)^{-s} = pi R^{-s} P_{-s}(P/R)`, `dP_nu/dnu|_0=ln((1+z)/2)`:

    F_low = sqrt(mu_macro) exp(i w c)
            * {1 + pi w/4 + i(w/2)[ln(w/2) + gamma_E + 2 ln Lambda]} + O(w^2)
    c = 0.5(|y|^2 - x0.y) = phi_geo + 0.5 ln r0_sq
    Lambda = 2 sqrt(a1 a2)/(sqrt a1 + sqrt a2) = sqrt(2 detA/(lam+sqrt detA))
             a1,a2 = lam -+ gamma

`pi w/4` and the gamma_E coefficient UNCHANGED from the point mass.  `s`
does NOT appear in the O(w) bracket at all — all y-dependence is in
exp(iwc).  beta cancels exactly.  Reduces to Takahashi & Nakamura 2003
Eq.(18) at lam=1,gamma=0.  CORRECTION to an earlier note of mine: T&N's
"-2 phi_m" is the MINIMUM-SUBTRACTED frame; in the absolute frame `F_op`
uses it cancels against e^{iw phi_m}.  My earlier "- s" was WRONG.
Verified: residual ~ w^2 (1.5e-7 @1e-4 -> 6.5e-4 @1e-2), ~0.02(w r0_sq)^2.

## PART C — TERMINOLOGY (resolved; two DISTINCT objects)
- REAL near-lens saddle: 2nd root of `find_images`, Morse index 1,
  x_c ~ -y/|y|^2, sqrt|mu| ~ 1/|y|^2, tau ~ |y|^2/2 + ln|y| + 1.
  Magnitude in the annulus 4.4e-2 .. 8.4e-2 of sqrt(mu_macro).  THIS is
  what my earlier "ghost" numbers referred to.
- COMPLEX saddle (`geometry.ghost_kernel` / `channels.farfield_ghost_term`):
  conjugate quartic pair, Im tau_c = 3.3-9.5 in the annulus, separation
  5.0-11.1 >> _GHOST_SEPARATION_MIN=0.7 (admitted).  |G|/sqrt(mu):
  3.5e-1 @w=0.05, 8e-2 @0.1, 2e-2 @0.2, 1e-3..5e-2 @0.5, 1e-5..9e-3 @1,
  <=3e-4 @2.  Both coexist in the 2-real-image region; count ONCE each.
  NOTE a real in-annulus refusal that is NOT the separation gate:
  (|y|=3.6, th=0.5, gamma=0.25, kappa=0.3, beta=0.5) raises
  GhostDomainError 'no complex-conjugate root above tolerance'.

## PART D — MEASURED LADDER (nodes = real/imag cubic spline on log_w or |y|
## to ABSOLUTE heldout error <= eps*max|F|; residual demodulated by carrier)
Band [1e-3, 0.05]:  carrier(a0,b1) sz 2.9e-2..4.4e-2  N=4
                    ppGO / +2nd img sz 1.4e3..5.0e3   N=70-82  (C1/w,C2/w^2
                                                       DIVERGE - unusable)
                    low-w form      sz 5e-3..2.2e-2   N=4
Band [0.05, 0.5]:   carrier         sz 4.8e-2..8.7e-2 N=7-15
                    ppGO            sz 1.0..2.3       N=8-9
                    low-w form      sz 0.53..0.92     N=6-16
Band [0.5, 8]:      carrier ALONE   sz 6.4e-2..2.3e-1 N=161-241 (BEAT)
                    carrier+real2nd sz 9.6e-3..1.8e-1 N=4-8
                    ppGO            sz 2.5e-3..2.5e-2 N=4-8
                    ppGO+cplx ghost sz 1.6e-3..1.7e-2 N=4-8
Band [1e-3, 8] as ONE chart: everything >= 122 nodes -> the band MUST split.
y-plane (radial 3.0->4.24, azimuthal 0->pi/2 at |y|=3.8), demodulated:
  w<=0.2: carrier residual N=4 (sz 2.4e-2..7.5e-2); F demod N=4-5
  w=0.5 : F demod N=4-9;  w=2: F demod N=9-17, ppGO residual N=4

## PART E — CONCLUSIONS
1. LADDER STAYS AT 4 COMPONENTS.  The `log_w` chart absorbs the ln(w/2)
   structure at ZERO node cost: measured N identical for carrier-only vs
   low-w form at eps = 4e-3/1e-3/1e-4/1e-5 (4/5/9/17 both, band
   [1e-3,0.05]).  Reason: the missing term is (iw/2)ln(w r0_sq/2), and
   `w ln w = e^u u` is ENTIRE in u = ln w.  Its coefficient is
   geometry-independent; the only y-dependence is (iw/2)ln(r0_sq), smooth.
   The low-w form buys ~3x smaller residual for ~0-15% fewer nodes — not
   worth a 5th component.
2. BAND SPLIT AT w ~ 0.5 (where the real 2nd image becomes resolved).
   Below: carrier ONLY — adding ppGO/2nd-image/complex-ghost injects their
   divergent 1/w, 1/w^2 kernels and blows the residual to ~1e3.
   Above: ppGO (both real images, full C1/C2) + complex ghost where
   admitted.  This IS the existing w_trust band-split architecture.
3. CORRECTION to my own earlier claim "the one-image Born total makes the
   chart worse": true only for the carrier ALONE above w~0.5 (241 nodes).
   Adding the real 2nd image WITH its full C1/C2 `image_kernel` collapses
   it to 4 nodes.  My earlier 1.4e-2..5e-2 figure used the LEADING
   amplitude only (no C1/C2) and excluded the complex ghost; with the full
   kernel it is 9.6e-3..4.7e-2, and the complex ghost changes it by <1%
   above w=1.
4. NO GAP.  The seam [0.05,0.5] is covered by the carrier at 7-15 nodes
   (eps 4e-3) / 19-33 (eps 1e-4), with 4 nodes per y axis — normal-sized,
   prior-universal tiles.
5. The `_born.py` WHY premise ("the low-w far zone varies on the Einstein
   scale, so trained tiles there are prior-sized") is measurably WRONG once
   `exp(iw phi_geo)` is demodulated out — ALL the Einstein-scale variation
   sits in that closed-form phase.  Einstein-scale fringe motion is a
   MID/HIGH-w problem (w>~1), the opposite of the premise.
6. What b1/a0 actually buy: ~10-25% smaller residual in the low band vs the
   bare macro lead, free (same node count).  Above w~0.5 they are
   superseded by ppGO.  The load-bearing deliverables were the SIGN fix,
   the a0 omission, and the regime diagnosis — not the accuracy gain.

## Cross-references
`professor/microlensing_chang_refsdal` (F009, SACR-C, mass-sheet identity),
`professor_code_observations` (_born.py entry now dischargeable; add
ppGO-down-to-w~0.5, C1/C2-makes-the-residual-smooth, and the
complex-vs-real saddle naming hazard).
