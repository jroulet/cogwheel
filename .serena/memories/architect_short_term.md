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

## Build 6 (C5) — Ghost Decay Gate