# Build Brief: Exterior fold-carrier phase demodulation

## Mission

Recover the ghost-transition zone (~40% of the exterior prior box) at surrogate speed via analytic FOLD-CARRIER phase demodulation. The ghost-gate exclusion (cf81d66) correctly excluded those tiles, but they now fall to the exact engine (slow). This build lets the surrogate chart them again by removing the rho-phase oscillation analytically — the phase of the Fresnel/Airy blob centered at the fold where the two real images merged.

## Physics (validated 2026-08-10)

- The ghost is `G ~ A·e^{iw·tau_c}` where `tau_c = Re(tau_c) + i·Im(tau_c)` is the delay of the fold-merge point (measured: ghost Re(tau_c) sits at ~half the real-image delays = the coalescence point).
- `Re(tau_c)` = phase oscillation (in w AND in rho, since tau_c is rho-dependent). `Im(tau_c)` = amplitude decay.
- The rho-phase winding is dominated by the ghost's phase: `dphase/drho = 20.8` vs `w·dRe(tau_c)/drho = 24.9` (ratio 0.84 — the ghost is the dominant carrier).
- Demodulating `E_ks` by `e^{-iw·Re(tau_c(rho))}` reduces the rho-phase span from 16.7 -> 3.2 rad over rho in [1.3, 2.1]. Residual amplitude is smooth/monotone (0.335 -> 0.0007), splineable at 4 nodes.
- `Re(tau_c)` is well-defined even where `Im(tau_c) < 0.4` (the decay-gate-refused zone) — it is a geometric quantity, computable from the roots regardless of decay.
- CRITICAL: do NOT demodulate by the full complex `tau_c` (i.e. do NOT multiply by `e^{+w·Im(tau_c)}`). Measured: that amplifies everything by ~19x at w=30 (dividing out the ghost's decay is numerically explosive). Only the phase (Re) is demodulated; the ghost's smooth amplitude decay stays in the residual for the spline.

## Work

1. **Add fold-carrier demodulation to the exterior chart**: a `rho_carrier_rate` or per-node carrier. Design decision: the carrier is `e^{-iw·Re(tau_c(rho))}` and `tau_c` varies with rho (and theta_c/gamma). Options:
   - A per-node carrier map `rho_carrier(rho)` stored with the chart (like theta_to_u), OR
   - A per-chart fit (median k_rho) — simpler but only captures the mean rate, not the rho-dependence.
   The measured data shows the phase is not perfectly linear in rho (ratio 0.84), so a per-node analytic carrier (from `geom.ghost_kernel(...).delay.real` at each node) is preferred over a fitted constant. Design and prototype both; pick the one clearing the bar at 4 nodes/axis.
2. **Integrate with the existing machinery**: compose with the w-carrier (carrier_rate, shipped) and rho_log_axis. The demodulations commute (independent axes). The serve path must re-modulate `e^{+iw·Re(tau_c(rho))}` after spline contraction.
3. **Layered strategy (where applicable)**: where the ghost model is accurate (Im tau_c >= 0.4), the `FARFIELD_KERNEL_SUM_MINUS_GHOST` label subtracts analytically and `reconstruct_farfield` re-adds it at serve — that keeps the ghost out of the spline AND analytic in the relative-binning decomposition. The fold-carrier is for the transition zone where the model is inaccurate. Implement whichever combination clears the bar; document the chosen layering.
4. **Verify**: exterior probe ~70 charts, all held-out eps < 1e-3 at 4 nodes/axis, AND the ghost-transition region now served by the surrogate (census: those draws show served, not engine fall-through); round-trip to machine precision.

## Measured facts (re-probe at HEAD before coding)
- rho-phase span 16.7 rad raw -> 3.2 rad after e^{-iw·Re(tau_c)} demod (rho in [1.3, 2.1], gamma=0.5, theta=0.2)
- dphase/drho = 20.8; w·dRe(tau_c)/drho = 24.9 (ratio 0.84)
- |E| residual after demod: smooth 0.335 -> 0.0007 (monotone)
- Full-complex demod amplifies 19x at w=30 (do NOT use)
- Ghost model accuracy: subtraction helps only at rho=1.4 and rho>=3.0 (mixed); the transition zone (rho 1.6-2.6) is where the fold-carrier is the right tool
- tau_c from `geom.ghost_kernel(w, source, matrix).delay` (geometry.py), already used by the ghost machinery
- Prior fixes: cusp exclusion (d685ebe), w-carrier (f4652e7), log(rho-1) (f6b8b05), ghost-gate exclusion (cf81d66)
- Envelope: `farfield_envelope_from_partition(partition, FARFIELD_KERNEL_SUM)`; MINUS_GHOST re-adds analytically at serve (channels.py ~1168)
- Probe: `scripts/probe_exterior_recursion.py` (4x4x4, w 4/decade, engine 80)

## Constraints
- Fast tests. Follow AGENTS.md.
- COORDINATE/PHASE transform — NOT node-density increase. Spline smooth things.
- Do NOT demodulate by Im(tau_c) (the e^{+w·Im} amplification is explosive). Phase (Re) only.
- Compose with w-carrier + log-rho + ghost-exclusion; round-trip to machine precision.
- Layered strategy: MINUS_GHOST where the model is accurate, fold-carrier in the transition zone, exclusion for the residual.
- Plan-gate requirement: each `domain_test_descriptions` spec names exactly ONE primary `test_*.py`; no spec may reference another spec's primary file.

## Design note from the driver
The user (domain expert) proposed: once the images have merged off the real axis, the ghost is a single Fresnel blob at the fold point — demodulate w.r.t. that merge point. Validated: tau_c IS the fold-merge delay and its Re part captures ~84% of the rho-phase winding, computable where the decay gate refuses. The user also proposed subtraction (MINUS_GHOST) over splining where the model is accurate, which helps the relative-binning decomposition (ghost stays analytic at serve). Implement the layered strategy; the fold-carrier is the core deliverable.
