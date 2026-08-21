# Ruling: kill the bespoke wall-band chart — low-w shell = macro-lead chart, inner far-exterior = Born extended

## Context (measured)
The bespoke low-w diffractive chart's residual failed three times because it
DIVIDES by an oscillatory carrier (r = f_pure*sqrt(1-gamma'^2)/F_ref), creating
poles at the carrier's beating zeros (two-image cos(w*Delta_tau/2) interference
zeros; Pearcey P(x,y) dark fringes). Symptom: 5800x overshoot, de-rate 0.0002,
every cell declined. The absolute carrier-adequacy guard is a REFUSAL detector,
not a fix -- it converts "wrong serve" into "decline," buying no accuracy.

The quotient r = f_pure/F_ref was the error. The codebase's SETTLED
representation is the demodulated DIFFERENCE R = F_demod - F_carrier_demod
(BornResidualChart: subtract the shared carrier phase, never divide by an
oscillatory field) -- no poles by construction.

## Coverage facts (measured)
- The existing Born rung (`_born_residual_analytic`) already serves rho > 2
  (kappa=0, beta=0) with the settled demodulated-difference representation,
  NO per-candidate engine cost.
- The wall band (gamma'>0.5) spans all rho; the rho > 2 far exterior is already
  Born-served. The genuine gap is rho in [~1.4, 2].

## The physics (Professor ruling + owner correction, 2026-08-21)
The rho-regime partition:
| band | regime | correct serve |
|---|---|---|
| rho > 2 | far exterior, resolved | Born (shipped) |
| rho in [~1.4, 2] | inner far-exterior | Born, EXTENDED downward |
| rho in [0.6, ~1.4] | near-fold shell | LOW w: macro-lead chart; fold/cusp structure develops only at larger w*Delta_tau (fold arm / tube domain) |

OWNER CORRECTION to the Professor's "shell stays on Schwinger": the low-w
shell is the EASIEST regime to chart, not the hardest. At low w
(w*Delta_tau < 1), F -> sqrt(mu_macro) regardless of rho (F009), the
fold/cusp structure has not developed, and the residual
r = f_pure*sqrt(1-gamma'^2)/sqrt(mu_macro) is smooth and O(1). MEASURED at
the shell witnesses (incl. the b3->0 cell gp=0.8 rho=1.1):
|r| ~ 0.61-1.6, arg within ~0.2 rad, across w in [0.02, 1.0]. The arg winding
develops only at larger w*Delta_tau -- the fold arm's / tube chart's domain,
NOT the low-w band this rung owns.

Schwinger is the OFFLINE ORACLE ONLY, never the serve (owner: "Fuck F_op";
the diffractive rung exists to avoid the engine in this band).

## Decision
1. KILL the bespoke low-w wall-band chart (the 4-carrier reference / absolute
   guard / schema v3 is sunk cost on the quotient representation, which is
   wrong for the wall band). The demodulated DIFFERENCE is the representation
   to use for any chart here.
2. LOW-W NEAR-FOLD SHELL (rho in [0.6, ~1.4], w*Delta_tau < 1): serve with a
   MACRO-LEAD demodulated-difference chart -- the smooth constant residual
   (measured), no engine. A new chart modeled on BornResidualChart's
   representation with the born_lead_carrier (sqrt(mu_macro)*exp(i w phi_geo)).
3. INNER FAR-EXTERIOR (rho in [~1.4, 2]): EXTEND the Born ladder downward --
   re-train born_residual_chart.npz's rho grid to ~1.4, lower the gate.
   Small, reuses the settled certified machinery.
4. FAR EXTERIOR (rho > 2): Born (shipped, already works).
5. The tracked todo `lensing_low_w_near_fold_serve` is superseded by this
   split; update it to the new end state.
