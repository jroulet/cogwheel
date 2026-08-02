# Architect Short-Term Observations

## Build 7 — _GHOST_SEPARATION_MIN Part 0 Resolution

Planning complete. Resolution: the constant 0.7 does NOT violate Part 0
(it is a lens-plane quantity in Einstein-radius units — the Einstein radius
IS the physical scale in the lens plane; the measured gap is stable across
gamma 0.30–0.90; the constant traces to geometry/cusp-coalescence, not the
prior box). The decay gate (step 6) does NOT subsume it — orthogonal failure
modes (decay = near-axis; separation = near-cusp). Existing test suite
already mechanizes the Part 0 argument (tripwire bounds, reachable-red,
do-nothing control, train/serve agreement). Work = update docstrings/comments
to formally resolve SUSPECT status + update COVERAGE_DESIGN table. Single
Foreman-Lite WP, no value change, no test changes needed. One domain test
description for orthogonality witness (config passing decay, failing separation).

## Build 8 Step 8 — Part 0 Mechanical Test

Test-only build: new test file `test_lensing_part0_mechanical.py` enforcing
four Part 0 structural invariants (no prior-box constants, no retired names,
no discretization absorbers, no stepping for closed forms). Zero Coder WPs —
all work routed to Test Developer via domain_test_descriptions. Pre-commit
hook and retired_concepts.json already exist. Key decisions: (1) AST scan
module-level Assign nodes only, not expressions/comments; (2) 4.2426 flagged
unconditionally within 1e-2 tol; 3.0 flagged only if name contains box-shaped
fragments; (3) np.diff/gradient scan TRIMMED per Simplifier (100% false
positive rate); (4) discretization absorber check uses explicit allowlist for
_*_EPS/_MARGIN/_FRAC constants; (5) test reads retired_concepts.json directly,
no hook import.

## Build 6 (C5) — Ghost Decay Gate