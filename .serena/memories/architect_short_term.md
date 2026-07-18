# Architect Short-Term Observations

## Build 3d triage (2026-07-18): INS-1-002/003
WP1 segmentation underperforming a global spline over the same node union
(INS-1-002) is an implementation defect, not a falsified premise — kink-
segmented splines are only better than a global spline if boundary
conditions at the cut are handled correctly (pinned/matched, not a fresh
not-a-knot BC at an artificial edge) and segment node budgets track each
sub-band's own oscillation content. Ruled coder_fix. INS-1-003 (18.8ms vs
15ms floor, 3.57x vs 4x reduction) is provisionally downstream of the same
bug: fix WP1 first, retune node budget/segment boundaries to the minimal
count meeting 1e-3, profile non-engine overhead, re-measure. Only escalate
the 15ms/4x floor if it remains unreachable AFTER a correctly implemented
segmentation — do not escalate on a build known to contain an implementation
bug.

## Build 3d — lensing lnlike 41ms->target (2026-07-18)
Professor CORRECTED the brief's 10ms arithmetic: the interpolated object
(K_a or exact_total) carries ~9 oscillation cycles in the crown wave band
[w~0.32,16.4] (beat freq = max image-delay spread ~3.5). Node count is set by
oscillation content, NOT by the kinks. Honest floor for lever A (kink-aware
interp, likelihood.py only): ~15-20 engine nodes => ~13-15ms crown (2.7-3.2x
speedup), NOT 10ms. 10ms needs ~6 nodes (<1.5 cycles) => impossible w/o
lever B (surrogate table) OR fenced-out micro-opts (nearest-caustic Newton
1.9->0.3ms in geometry.py, contraction fusion 2->1ms in operator.py).
DECISION: ship lever A ALONE (Simplifier: lean; owner directive: don't reach
for the table while cheap structure unused). Sub-approach = SEGMENTED per-kink
splines on final K_a inside likelihood.py (_coarse_w_node_grid +
_amplification_coefficients), NOT the peel-switch factorization (Simplifier:
crosses _gauge/channels boundaries unnecessarily = watch). Per-segment
CubicSpline breaking at kink freqs + branch-transition freq
w_branch=max(RHO_END/delta_min, L_MAX/|y'|); adaptive per-segment node budget
(~oscillation cycles, floor 4). Single batched F_op_grid preserved. Engine
(operator/_dd/geometry/_gauge) untouched. Step rule: gate at Professor floor
15ms (structural gates lead), name residual levers. No lever B => no
DATA_CONTRACTS artifact, no cache, no reduction-exactness test this build.
