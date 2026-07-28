# Professor — short-term (2026-07-28: Born b1, low-w form, ladder, saddle,
# F023 gap sweep, caustic closed form, GHOST BRANCH adjudication)

## A — b1 / a0 CLOSED FORMS (both parities; pinned vs F_op to 0.01-2%)
    b1 = -lam*(lam + gamma*P)/detA = -lam*(2*lam*r0_sq - x0_dot_y)/(det_a*r0_sq)
    a0 = -lam*gamma*P/detA         = -lam*(lam*r0_sq - x0_dot_y)/(det_a*r0_sq)
    P = cos2(theta_x0 - beta).  Point mass -> b1 = -1 (placeholder WRONG SIGN).

## B — LOW-w FORM (derived, verified, NOT NEEDED — log_w spline absorbs it)
    F_low = sqrt(mu_macro) e^{iwc}{1 + pi w/4 + i(w/2)[ln(w/2)+gamma_E+2 ln Lambda]}
    c = phi_geo + 0.5 ln r0_sq;  Lambda = sqrt(2 detA/(lam + sqrt detA)).
    cf. Takahashi & Nakamura 2003 Eq.(18).

## C — a0 MUST NEVER BE APPLIED BELOW THE SPLIT
F(w->0) = sqrt(mu_macro) EXACTLY (F009); the carrier gives *(1 + a0/q2r).
No crossover gamma — a0 hurts at every gamma incl. 0.45 (azimuthal N 4->11
at 0.45, 4->20 at 0.60, 4->44 at 0.75).  F023 missed it by sweeping |y|
RADIALLY at fixed theta; the pathology is AZIMUTHAL.  Lead-only below split.

## D — CAUSTIC EXTENT closed form (verified 4 dp, 16 gammas, both parities)
    |y|^2(u) = 2u - 3 + 2 gamma^2/u + (1-gamma^2)/u^2,  u in [|1-g|, 1+g]
    f'(u) = (2/u^3)(u-1)(u^2+u+1-g^2);  u_c = (sqrt(4g^2-3)-1)/2 (>0 iff g>1)
    at u_c: g^2 = u_c^2+u_c+1 and f(u_c) = 4 u_c + 1/u_c - 2
    positive parity: max|y| = 2g/sqrt(1-g)  (ALL g<1, unconditional)
    saddle: max|y| = sqrt(max(4u_c + 1/u_c - 2, 4g^2/(g+1)))
Cusp switch at g = 1.177651 (min extent 1.596072) explains non-monotonicity.
Fences: |y|=3.0 at g = sqrt((189-15 sqrt105)/32) = 1.0502342; |y|=3sqrt2 at
g = sqrt(63-24 sqrt6)/2 = 1.0261879; RE-ENTRY on the rising branch at g = 3
EXACT (4g^2-9g-9=0).  Deltoid is SPIKY (<=3.95% of 2pi inside at g=1.02).
Old 241^2 grid table UNDERESTIMATED (misses the spike) — retired.

## E — GHOST BRANCH ADJUDICATION (this session; corrects my saddle report)
Q1 BRANCH REFERENCE IS CORRECT.  Two independent arguments:
 (i) FOLD GEOMETRY.  tr Hess = 2 lam > 0 forbids index-2 images on BOTH
     branches, so every fold annihilates an (index 0, index 1) pair.
     MEASURED radial scans across the caustic:
       saddle  g=1.60 th=0.30: 0111 (closest pair (0,1), sep 0.093) -> 11
       saddle  g=1.20 th=0.30: 0111 (closest pair (0,1), sep 0.177) -> 11
       positive g=0.45 th=0.30: exterior census 01
     Identical A2 fold on both branches => the exp(-0.5j*pi) reference is
     no less justified on the saddle.  The docstring's wording is
     positive-parity-flavoured but its CONTENT is parity-independent.
 (ii) NUMERICS.  Three-way resid(ppGO) vs resid(+G) vs resid(-G), w-resolved,
     saddle g=1.2 |y|=3.05 th=0.30 (Im tau_c=0.919, dtau=16.25, sep=1.57),
     30 log-spaced w in [0.3,6]: -G is NEVER best at ANY w.  +G wins at all
     16 points with w<=1.41 (w*dtau<=23), ppGO wins above.
 => The in-flight build's code comment naming an "underived branch
    reference" as the reason for refusal is WRONG and should be corrected.

Q2 THE ADMISSION FLIP IS *NOT* THE EXPLANATION.  On theta in [0.1,0.7]
with NO admission crossing (all admitted), adding G still degrades:
    g=1.6 |y|=4.243 w=5: ppGO 2.14e-4/N=4 -> +G 1.23e-2/N=5   (58x worse)
    g=1.6 |y|=4.243 w=2: ppGO 3.08e-3/N=5 -> +G 5.54e-2/N=6   (18x)
    g=1.2 |y|=4.243 w=5: ppGO 4.00e-5/N=4 -> +G 2.58e-3/N=4   (64x)
    g=1.2 |y|=3.050 w=2: ppGO 8.73e-3/N=11-> +G 6.26e-2/N=10  (7x)

Q3 THE REAL MECHANISM: NEAR-AXIS NON-DECAYING GHOST (not saddle-specific).
As the source approaches a principal axis, Im tau_c -> 0 (the code's own
docstring says so), so |G| stops decaying with w and swamps the ppGO
residual.  Measured at w=5, |y|=4.2426 (theta -> 0.02):
    g=1.60: Im tau_c 2.31->0.099, |G|/|F| 1.9e-6 -> 1.04e-1,
            r(ppGO) 2.2e-5 -> 3.3e-4,  r(+G) -> 1.03e-1
    g=1.20: Im tau_c 3.30->0.139, |G|/|F| 7.3e-9 -> 4.3e-2
    g=0.45 (POSITIVE PARITY): Im tau_c 9.66->0.394, |G|/|F| 1.2e-22 ->
            1.44e-2 while r(ppGO) stays 1.5e-5  => +G 1000x WORSE
The separation gate NEVER BINDS on the saddle: measured min_a|x_a - x_c| in
[0.942, 2.421] over 121 theta-samples x 4 (gamma,|y|) configs — ALL above
_GHOST_SEPARATION_MIN = 0.7.  The RETIRED decay gate
`w_min * Im tau_c >= _FARFIELD_WINDOW_RADIANS = 2.0` was the ONLY guard
against this; Build 8h-d1 removed it for train/serve skew reasons.  The
skew fix is to pin w_min to the CHART BAND FLOOR (a chart property,
identical at train and serve), not to drop the gate.
The other boundary (`no complex-conjugate pair`) is a SINGLE clean crossing
in theta (1 flip per 121-sample sweep, at theta ~ 1.06 (g=1.6,|y|=3.05),
1.09 (1.6, 4.24), 1.32 (1.2, 3.05), 1.31 (1.2, 4.24)) — it is a level set of
the image-quartic discriminant, computable from coefficients
`image_quartic_coefficients` already returns.  Cheap and tiler-alignable.

RECOMMENDATION: keep the ghost, correct the comment, restore a decay-based
admission condition (band-floor-pinned so it stays frequency-independent in
the train/serve sense), and align tiles to the discriminant boundary.
Refusing the ghost outright leaves the g=1.2/1.6, w<=1.4 improvement on the
table, but refusing is still SAFER than admitting with only the separation
gate, because the near-axis failure is 10^3 x in the wrong direction.

## F — SADDLE LADDER (gamma > 1)
Census (1,1) exterior for gamma in [1.05,1.6]; split currency `w*dtau ~ 4`
NOT `w*r0_sq`; lead-only below split (sz 1.0e-2..7.4e-2, N=4 log w and theta).

## G — measurement caveat
Above-split theta node counts demodulate by the SINGLE carrier
exp(iw phi_geo); PESSIMISTIC.  Correct object = SACR-C switched envelope.
Residual SIZES are demodulation-independent and stand.

## Cross-references
`professor/microlensing_chang_refsdal`, `professor_code_observations`.
