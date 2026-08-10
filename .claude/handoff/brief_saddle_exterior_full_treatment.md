# Build Brief: Saddle (negative) parity — full exterior treatment

## Mission

Verify and apply the full astroid-parity exterior treatment to the macro-saddle (gamma > 1) exterior, and examine the deltoid straight-edge / inter-lobe corridor serving. The saddle has been failing at the same rate as the astroid (91/154 charts over the 1e-3 bar from probe 2) but was never given the probe→diagnose→fix treatment.

## Background (measured 2026-08-10)

- Probe 2: saddle exterior 91/154 charts fail the 1e-3 bar (median 0.0015, max 11.67); 80 at depth 3 (the cap). Astroid failures drove the whole fix chain; saddle was never examined to the same depth.
- The ghost EXISTS on the saddle: gamma 1.1-2.0, Re(tau_c) linear in rho (slopes ~2.4-2.7), Im(tau_c) 0.23-1.97 (some pass the 0.4 decay gate, some don't). So the fold-carrier machinery (which demodulates by Re(tau_c)) SHOULD apply — `_needs_fold_carrier` is called for ALL tiles (not parity-gated), but whether the resulting `rho_u_carrier` is correct/verified on the saddle is unknown.
- The saddle exterior path: scalar-reach `_farfield_tiles` with deltoid cusp exclusion (`_deltoid_cusp_source_angles`), `rho_log_axis=True`. Whether the w-carrier demodulation and the 2D fold-carrier behave identically on the saddle (deltoid cusps vs astroid cusps) is unverified.
- Unexamined geometry: the deltoid straight edges (the fold arcs between cusps) and the inter-lobe corridor (where `_lobe_serves` excludes and the exact engine serves). What serves there, and whether it needs excision/conditioning, is unknown.

## Work

1. **Probe the saddle exterior** (characterize the failures): the probe script trains astroid first; either parameterize it to run the saddle band (gamma > 1) first/alone, or write a saddle-focused probe. Characterize the saddle exterior failures the same way the astroid was: eps vs rho/theta_c/gamma regions, image-count, ghost dominance, phase winding (is it linear in rho/u like the astroid?). Identify whether the same diagnostics hold.
2. **Verify the applied machinery on the saddle**: does `rho_log_axis`, `rho_u_carrier` (fold-carrier), and the w-carrier actually help the saddle exterior? Measure the held-out eps with them ON vs OFF on the saddle. Fix any parity-specific issues (e.g. deltoid vs astroid cusp geometry in `_compute_rho_u_carrier`, `_needs_fold_carrier`).
3. **Deltoid straight edges + inter-lobe corridor**: determine what serves on the deltoid fold arcs (the straight edges) and in the inter-lobe corridor. Are they excised correctly, or do they need excision boundaries (like the cusp windows)? Does the lobe-interior chart + exact engine cover them correctly?
4. **Excision boundaries for the saddle**: mirror the astroid's cusp-window excision → Pearcey, ghost-transition excision → exact engine. Ensure the saddle exterior tiles don't straddle the deltoid cusps or edges.
5. **Verify**: saddle exterior probe clears the 1e-3 bar at the probe node count; saddle tile count collapses toward ~70; the serving geography (straight edges, corridor) is correct and documented.

## Measured facts (re-probe at HEAD before coding)
- Probe 2: saddle 91/154 fail (median 0.0015, max 11.67), 80 depth-3
- Ghost on saddle: exists gamma 1.1-2.0, Re(tau_c) linear in rho (slope 2.4-2.7), Im 0.23-1.97
- Saddle path: `_farfield_tiles` scalar-reach + `_deltoid_cusp_source_angles` exclusion + `rho_log_axis=True` + `_needs_fold_carrier` (all tiles)
- Astroid treatment (to mirror): cusp-window excision `_CUSP_ARM_COVERAGE=0.07` → Pearcey, ghost-gate excision, w-carrier, log-rho, 2D (rho,u) fold-carrier
- Serving ladder: surrogate (chart boxes) → geometric → uniform arms → Schwinger exact → named refusal; lobe charts for gamma>1 interiors; corridor → exact engine
- Probe: `scripts/probe_exterior_recursion.py` (currently astroid-first; needs saddle coverage)

## Constraints
- Fast tests. Follow AGENTS.md.
- Mirror the astroid treatment where it applies; do NOT assume it transfers without verification.
- Excision boundaries and serving geography must be correct (no tile straddles a deltoid cusp/edge; corridor served correctly).
- The probe must characterize the saddle (parameterize or add a saddle run).
- Plan-gate requirement: each `domain_test_descriptions` spec names exactly ONE primary `test_*.py`; no spec may reference another spec's primary file.

## Design note from the driver
The user flagged (2026-08-10): "All the hard won victories of the astroids" — every astroid-parity victory (cusp excision, w-carrier, log-rho, fold-carrier, ghost excision) may transfer to the saddle, and the deltoid straight-edge/inter-lobe corridor geometry is entirely unexamined. Treat this as the saddle getting the SAME rigorous treatment the astroid got: probe first, diagnose with measured evidence, then fix. Do not assume the astroid fixes are saddle-correct.
