# Build Brief: Exterior ghost-gate tile exclusion (fix the unsmoothable-region admission)

## Mission

Fix the exterior surrogate failures by EXCLUDING tiles in the ghost-dominated regime (where no kernel-sum label is smooth) rather than trying to condition the envelope. The failing tiles sit where the unsubtracted ghost dominates the residual (~3x |E_ks|) and cannot be subtracted (F027 gate refuses: Im tau_c < 0.4). No coordinate transform can make a dominant oscillatory ghost smooth — the fix is admission, exactly like the cusp-exclusion precedent. Excluding these tiles collapses the tile count toward ~70 and serves the excluded region by the exact engine / Airy-Pearcey arms.

## Background (decisive measurement 2026-08-10)

Probe 3 (all prior fixes: cusp exclusion, w-carrier, log(rho-1)) killed at 56 charts, 30/55 fail. Root cause now proven:
- Image count stays 2 everywhere (no coalescence transition).
- |ghost| / |E_ks| = 3.2-3.4 everywhere the ghost is computable (rho>=2.0), and the ghost gate REFUSES (Im tau_c < 0.4) across the failing band [1.1, 1.9].
- The KERNEL_SUM residual is DOMINATED by the ghost's oscillatory structure. The ghost is either gated off (near) or unsubtracted (the chart uses KERNEL_SUM = Window iii).
- No coordinate transform (rho-carrier, log-rho) can make a dominant oscillatory ghost smooth — the phase winding IS the ghost's phase.
- The three-window label scheme (DIFFRACTIVE / KERNEL_SUM_MINUS_GHOST / KERNEL_SUM) assumed the ghost is negligible (high w) or small-and-subtractable (mid w where the gate passes). The failing region [1.1, 1.9] is a ghost-transition zone where the gate refuses but the ghost dominates — a coverage gap.

## Work

1. **Verify the ghost-dominance claim** (small probe): at the probe's failing tiles, confirm |ghost| > |E_ks| and the gate refuses. Map the region where |G|/|E_ks| > 1 (or where the gate refuses) as a function of (gamma, rho, theta_c, parity).
2. **Implement ghost-region tile exclusion** in the exterior tiler (`_farfield_exterior_tiles` / `_build_farfield_chart` / the admission path): exclude a tile when its corners (or center) are in a region where the ghost gate refuses OR |G|/|E_ks| exceeds a threshold (e.g. 1.0, giving the kernel-sum label a chance to be the residual). This mirrors `_exclude_near_cusp` (source-plane distance-based) — extend the same pattern to a ghost-region test. Excluded tiles fall to the exact engine / uniform arms at serve (the ladder already handles this).
   - Consider whether the existing `_GhostSeparationMin` / `_GHOST_DECAY_IM_THRESHOLD` machinery can be reused as the admission test (it is configuration-only, no w-dependence — provably consistent train/serve).
3. **Optionally**: switch the mid-w band of the exterior chart to the `FARFIELD_KERNEL_SUM_MINUS_GHOST` label WHERE the gate permits, so the retained tiles are smooth by construction. (This is the correct label for the mid-w window; KERNEL_SUM is Window iii only.)
4. **Verify**: exterior probe produces ~70 charts with all held-out eps under the 1e-3 bar; excluded regions fall to the exact engine (census shows the fall-through); no tile straddles the ghost-transition zone.

## Measured facts (re-probe at HEAD before coding)
- |ghost|/|E_ks| ~ 3.2-3.4 (rho>=2.0); ghost gate refuses [1.1, 1.9] (Im tau_c < 0.4) at gamma=0.5, theta=0.2, w=10
- Image count = 2 throughout (no transition)
- Prior fixes in HEAD: cusp exclusion (d685ebe), w-carrier (f4652e7), log(rho-1) rho-axis (f6b8b05) — probe still 30/55 fail
- Ghost machinery: `farfield_ghost_term` (channels.py ~964), `_GHOST_SEPARATION_MIN = 0.7`, `_GHOST_DECAY_IM_THRESHOLD = 0.4` (channels.py ~219-234), `_exclude_near_cusp` (surrogate_training.py ~1676, called ~2007)
- Envelope labels: FARFIELD_KERNEL_SUM (Window iii), FARFIELD_KERNEL_SUM_MINUS_GHOST (Window ii), FARFIELD_DIFFRACTIVE (Window i) — channels.py ~131-141
- Probe: `scripts/probe_exterior_recursion.py` (4x4x4, w 4/decade, engine 80)

## Constraints
- Fast tests. Follow AGENTS.md.
- EXCLUSION is the primary fix, NOT envelope conditioning. Do not add node density.
- The ghost gate is configuration-only (no w-dependence) — the admission test must be train/serve consistent.
- Keep the shipped fixes (cusp exclusion, w-carrier, log-rho); they are correct and remain.
- Plan-gate requirement: each `domain_test_descriptions` spec names exactly ONE primary `test_*.py`; no spec may reference another spec's primary file.

## ANALYTIC FOLD-CARRIER DEMODULATION (driver-validated 2026-08-10) — preferred over pure exclusion
The user proposed the clean physical picture: once the merged image pair has moved off the real axis, the ghost is a SINGLE Fresnel/Airy blob centered at the real fold point where the images merged. So demodulate w.r.t. the FOLD-MERGE delay, not a fitted carrier. VALIDATED:
- `geom.ghost_kernel(w, source, matrix)` returns the complex delay tau_c whose REAL PART IS the fold-merge-point delay (measured: ghost Re(tau_c) sits at ~half the real-image delay, i.e. the point where the two real images coalesce — 1.44/2.29/3.43 at rho=1.3/1.7/2.1 vs real-image delays 2.68/4.06/5.67). This is the exact analog of the interior SACR-C tau_c demodulation (nearest-caustic-point carrier, channels.py:47-55), extended to the exterior residual.
- Computable from the geometric-optics roots REGARDLESS of the decay gate (tau_c is well-defined even where Im tau_c < 0.4).
- Demodulating E_ks by exp(-1j*w*tau_c) (the fold-merge delay) reduces the rho-phase span from 16.7 rad to ~3.2 rad over rho in [1.3, 2.1] (full-complex and Re-only demod give identical phase flattening since Im tau_c enters only the magnitude).

PREFERRED DESIGN: (1) analytic fold-carrier demodulation — E *= exp(-1j*w*tau_c(rho)) per node before fitting (using the fold-merge delay from the ghost roots), re-modulate at serve. This removes the dominant oscillation WITHOUT needing the decay gate and IS the interior SACR-C tau_c demodulation pattern applied to the exterior. (2) Keep the ghost-region tile exclusion as the safety net for the residual ~16% winding. (3) Optionally correct log(rho-1) -> log(rho) for the magnitude. Verify whether (1) alone suffices at 4 nodes/axis.

## Why not pure exclusion alone
Exclusion is necessary but wasteful: the ghost-dominated region is a large part of the exterior, and excluding it all would hand much of the prior box to the exact engine (slow). The analytic carrier recovers most of it at surrogate speed. Use exclusion only for the residual that even the analytic demodulation cannot flatten.

## Why not the fitted-constant rho-carrier / log-rho approach
A fitted constant k_rho is crude (rho-dependent winding needs a rho-dependent carrier). log-rho fixes magnitude only. The analytic Re(tau_c(rho)) carrier is exact physics, rho-dependent, and computable where the gate refuses — the right tool.
