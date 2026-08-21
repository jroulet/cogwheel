# Ruling: kill the bespoke wall-band chart — extend Born to rho ~ 1.4, shell stays on Schwinger

## Context (measured)
The bespoke low-w diffractive chart's residual failed three times because it
DIVIDES by an oscillatory carrier (r = f_pure*sqrt(1-gamma'^2)/F_ref), creating
poles at the carrier's beating zeros (two-image cos(w*Delta_tau/2) interference
zeros; Pearcey P(x,y) dark fringes). Symptom: 5800x overshoot, de-rate 0.0002,
every cell declined. The absolute carrier-adequacy guard is a REFUSAL detector,
not a fix — it converts "wrong serve" into "decline," buying no accuracy.

## Coverage facts (measured)
- The existing Born rung (`_born_residual_analytic`) already serves rho > 2
  (kappa=0, beta=0) with the settled demodulated-difference representation
  R = F_demod - F_carrier_demod, NO per-candidate engine cost.
- The wall band (gamma'>0.5) spans all rho; the rho > 2 far exterior is already
  Born-served. The genuine gap is rho in [~1.4, 2].

## The physics (Professor ruling, 2026-08-21)
The quotient failed because dividing by a field with zeros turns them into
poles. The demodulated DIFFERENCE (subtract the shared carrier phase, don't
divide) has no poles by construction — that's why Born splines at 7x5x10 nodes.

The correct rho-regime partition:
| band | regime | correct serve |
|---|---|---|
| rho > 2 | far exterior, resolved | Born (shipped) |
| rho in [~1.4, 2] | inner far-exterior | Born, EXTENDED downward |
| rho in [0.6, ~1.4] | near-fold shell (caustic) | fold/cusp regime — NOT Born, NOT the Pearcey quotient; stay on Schwinger unless a Wronskian-safe fold/cusp chart is node-justified |

Born is the weak-deflection exterior expansion valid throughout rho > 1; at
rho ~ 1.4 the residual grows but the gate is a node-budget boundary, not a
physics law — extend the BornResidualChart rho grid and lower the gate. At
rho -> 1 the two images coalesce and geometric optics has NO validity (F029/
F031/F033: O(1)-to-hundreds-of-percent error within eta < 0.3 of the caustic).

Macro-lead refinement: at low w (w*Delta_tau < 1), F -> sqrt(mu_macro)
regardless of rho (F009) — the fold/cusp structure hasn't developed, so even
in the shell the macro lead is the correct carrier at the band bottom.

## Decision
1. KILL the bespoke low-w wall-band chart. The 4-carrier reference / absolute
   guard / schema v3 is sunk cost on a representation wrong for the wall band.
2. EXTEND the Born ladder to rho ~ 1.4: re-train born_residual_chart.npz's
   rho grid, lower the gate. Small, reuses the settled certified machinery.
3. NEAR-FOLD SHELL (rho in [0.6, ~1.4]): stays on the exact Schwinger engine
   (which works there — the user's "Fuck F_op"). Chart later only if a
   Wronskian-safe fold/cusp chart (q=p Airy difference, never a Pearcey
   quotient) is node-justified.
4. The tracked todo `lensing_low_w_near_fold_serve` is superseded by this
   split; update it to the new end state.
