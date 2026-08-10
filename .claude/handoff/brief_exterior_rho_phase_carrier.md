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

## Why not the rho-phase-carrier / log-rho approach
The rho-phase winding is the ghost's phase. A coordinate/carrier transform tries to spline a dominant oscillatory ghost — fighting physics. The correct engineering is admission: don't chart what the label can't represent, serve it by the exact engine. This also collapses the tile count (the original ~500 -> ~70 goal) by not tiling the ghost-dominated region at all.
