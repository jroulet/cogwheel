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

## Build 3f triage (2026-07-18): INS-6-001
Same pattern as INS-5-003, recurring on the actual Build 3f WP1/WP2 SACR-C
swap (channels.py/_gauge.py envelope decomposition + likelihood.py
LOO-adaptive coarse-node rewire). SPEC.md line 54 still narrates the removed
_DEFAULT_KERNEL_NODES=100/F008-full-cluster/cubic-spline machinery.
Inspector's own suggested fix already targets the Librarian, not a Coder —
correctly scoped. Triaged override: doc/housekeeping is categorically
excluded from Coder WPs (hard requirement) and is handled by the
deterministic doc-sync + Librarian phase that runs after gates regardless.
Attached full replacement text (SACR-C envelope description, LOO node
budget _LOO_SEED_NODES=8/_LOO_STOP=4e-3/_LOO_MAX_NODES=48, criticality-
separation switch S_a=smootherstep(w|tau_a-tau_c|,0.5,4) superseding F008,
closed-form reconstruct_from_envelope) plus the F008-superseded FINDINGS
addendum, so the content isn't lost even though no Coder WP is spawned.

## Build 3f triage (2026-07-18): INS-5-003
SPEC.md/FINDINGS.md divergence after the SACR-C swap (WP1/WP2 replaced
_DEFAULT_KERNEL_NODES=100 full-cluster switch with the envelope+LOO
construction) is real but is NOT a Coder-WP-worthy defect: per hard
requirement, doc/housekeeping (changelog fragments, SPEC.md narrative sync,
FINDINGS addenda) is handled by the deterministic doc-sync + Librarian phase
that runs AFTER gates, not by a Coder WP. Listing SPEC.md under
files_affected in the plan was informational (signals the doc-sync/Librarian
pass should touch it), not an instruction to spawn a WP. Triaged as override
with explicit Librarian guidance attached (SACR-C envelope description, LOO
node budget 30-44/stop 4e-3/ceiling 48, criticality-separation switch
|tau_a-tau_c|, F008-superseded FINDINGS addendum per report Sec 6.7) so the
content isn't lost even though no Coder WP is created.
