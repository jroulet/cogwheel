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

(archived — see git history)

## Build 1e-gamma — Gamma Axis Collocation

Plan: replace uniform gamma grid with log-reach-collocated grid.
Key findings:
- 3 call sites: _train_band_charts (line 3776 surrogate_training.py),
  from_engine (line 2804 surrogate.py), from_lobe_engine (line 2990).
- `_uniform_axis` stays for other axes; new `_log_reach_gamma_axis` is a peer.
- Implementation: fine 200-point grid of _caustic_reach(gamma) → log →
  linspace in log-reach → np.interp back to gamma. No brentq needed.
- Both parities handled: positive parity reach is increasing, saddle
  reach is decreasing. In both cases the resulting gamma nodes are
  ascending (Professor verified).
- Spline/serve path needs NO changes: `make_interp_spline` handles
  non-uniform grids, `_validate_axis` only requires strictly ascending.
- `_caustic_arclength_map` receives gamma_nodes directly, uses linear
  gamma interpolation via searchsorted — handles non-uniform correctly.
- Professor: log(reach) is the correct measure, NOT 1/reach or det(A).
  Absorbs one power of (1-gamma) in the chain-rule derivative.
- Simplifier: use np.interp on a fine grid (not brentq), single WP.

## Build 6 (C5) — Ghost Decay Gate