# Build Brief: Exterior rho-phase carrier demodulation (+ correct log-rho coordinate)

## Mission

Fix the remaining exterior surrogate failures by removing the rho-PHASE carrier (the envelope's phase rotates ~2π per 0.3 in rho), and correct the rho magnitude coordinate from log(rho-1) to log(rho). This is the rho analog of the already-shipped w-carrier demodulation. The fix is a COORDINATE/PHASE transform, not added resolution, and must compose with the existing w-carrier + log-rho machinery and round-trip to machine precision.

## Background (measured 2026-08-10, probe 3 killed at 56 charts, 30/55 fail)

All three prior fixes are in HEAD (cusp exclusion `d685ebe`, w-carrier demod `f4652e7`, log(rho-1) rho-axis `f6b8b05`). The probe still fails:
- At training nodes: eps ~1e-4 (fixes work at nodes).
- Off-grid rho midpoint: eps ~0.38 (catastrophic).
- The envelope PHASE rotates ~2π per 0.3 in rho across a tile: measured at gamma=0.5, theta_c=0.2, w=12, rho in [1.3, 2.0]: phase goes -1.87, -0.44, +1.13, +2.84, -1.73, -0.12, +1.43, ... rad (full wrap every ~0.3 in rho).
- A magnitude coordinate (log(rho-1)) cannot fix a phase rotation — the real/imag parts oscillate in rho at the phase-carrier rate.
- Corrected structure: |E| ~ rho^(-p), log|E| linear in log(rho) (R²=0.999) vs log(rho-1) (R²=0.986). The rho-log build chose the slightly-wrong coordinate.

## The core design problem

The envelope E(rho, theta_c, gamma, w) has:
1. A w-PHASE carrier (linear in w) — REMOVED by the w-carrier demodulation (shipped).
2. A rho-PHASE carrier (the phase rotates linearly-ish in rho at rate k_rho ~ 2π/0.3 ~ 21 rad/unit-rho) — NOT yet removed. This is the dominant remaining failure.
3. A magnitude power law |E| ~ rho^(-p) — partially conditioned by log(rho-1), better by log(rho).

Design the rho-phase-carrier demodulation:
- Measure the per-node rho-phase slope (unwrapped phase vs rho, or vs the natural coordinate), median -> k_rho_chart.
- Demodulate E *= exp(-1j * k_rho * (rho - 1)) (or the correct phase variable) before fitting.
- Re-modulate at serve.
- The phase variable: is it linear in rho, or in rho-1, or in log(rho)? Measure. (The w-carrier was linear in w; the rho-carrier's variable must be measured, not assumed.)
- Coherence: the rho-carrier demodulation composes with the w-carrier (they commute — independent axes). The stored object must round-trip to machine precision.

Correct the rho magnitude coordinate:
- Switch log(rho-1) -> log(rho) if that's genuinely better (R² 0.999 vs 0.986), OR determine whether the rho-phase demodulation alone makes the real/imag smooth enough that the log coordinate is secondary/unnecessary. The plan should measure which combination (log(rho) vs log(rho-1) vs raw rho, with/without phase demod) clears the bar at 4 nodes/axis.

## Work

1. **Measure the rho-phase structure** (design input): for several (gamma, theta_c, w), characterize the phase vs rho (linear? in which variable? rate? does it vary across theta_c/w?). Also re-confirm the magnitude power law in log(rho) vs log(rho-1) across the band.
2. **Design** (deliverable): the rho-phase-carrier demodulation scheme (variable, rate measurement, single canonical site, serve re-modulation), and the rho magnitude coordinate choice. Prototype at 4 nodes/axis, verify off-grid eps < 1e-3.
3. **Implement** in `ExteriorPolarChart` (from_values/_assemble/_evaluate_chart/npz/schema), `from_engine`/`_build_farfield_chart`, consistent with the existing `carrier_rate` (w) machinery — likely a parallel `rho_carrier_rate` field, or a generalized carrier structure. Schema bump if the stored object changes.
4. **Verify**: node-exact round-trip (machine precision), off-grid eps < 1e-3 at 4 nodes/axis in rho AND theta, full exterior probe ~70 charts all under bar, backward compat (default 0.0 = byte-identical to HEAD).

## Measured facts (re-probe at HEAD before coding)
- Phase vs rho at gamma=0.5, theta_c=0.2, w=12, rho in [1.3,2.0]: -1.87, -0.44, +1.13, +2.84, -1.73, -0.12, +1.43, ... (~2π per 0.3 rho)
- |E| ~ rho^(-p): log|E| vs log(rho) R²=0.999; log(rho-1) R²=0.986; over 2.5 decades (0.0009 -> 0.27)
- Off-grid rho eps ~0.38 with all three prior fixes; node eps ~1e-4
- Relevant code: `carrier_rate` machinery (surrogate.py, from_values ~1665, _evaluate_chart ~2769, from_engine ~3011), `rho_log_axis` machinery (surrogate.py, just added), `_build_farfield_chart` (surrogate_training.py ~2714), `reconstruct_farfield` (channels.py)
- Envelope: `farfield_envelope_from_partition(partition, FARFIELD_KERNEL_SUM)`
- Probe: `scripts/probe_exterior_recursion.py` (4x4x4, w 4/decade, engine 80)

## Constraints
- Fast tests. Follow AGENTS.md.
- Coordinate/phase transform — NOT node-density increase.
- Compose with w-carrier + log-rho machinery; round-trip to machine precision.
- Complex envelope phase handled correctly.
- Plan-gate requirement: each `domain_test_descriptions` spec names exactly ONE primary `test_*.py`; no spec may reference another spec's primary file.

## Design note from the driver
The w-carrier demodulation (shipped) removed the w-phase rotation; the rho axis has an analogous phase rotation that is now the dominant failure. The user's principle applies again: spline smooth things — remove the oscillation physics first. The rho-phase carrier is the missing oscillation. Also note: the magnitude coordinate should be log(rho) not log(rho-1) per the R² evidence, but confirm whether the phase demodulation alone suffices before layering both.
