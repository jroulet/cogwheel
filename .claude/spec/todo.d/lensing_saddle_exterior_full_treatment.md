---
section: Backlog
depends_on: [2026-08-10_exterior_2d_fold_carrier]
---

- **Saddle (negative) parity: verify and apply the full exterior treatment**
  `[→ spec]` — identified 2026-08-10.

  All the hard-won astroid-parity exterior victories — cusp-window
  excision → Pearcey, ghost-transition excision, w-carrier demodulation,
  log(rho-1) rho-axis, 2D (rho, u) fold-carrier — have been applied and
  verified on the POSITIVE parity (astroid, gamma < 1) only. The MACRO-
  SADDLE (gamma > 1) exterior rode along passively: it got the deltoid
  cusp exclusion, `rho_log_axis=True`, and `_needs_fold_carrier` is called
  for all tiles (not parity-gated). But probe 2 showed the saddle failing
  at the SAME rate as the astroid (91/154 charts over the 1e-3 bar,
  median 0.0015, max 11.67) and it was never given the probe→diagnose→fix
  treatment.

  Measured: the ghost EXISTS on the saddle (gamma 1.1-2.0, Re(tau_c)
  linear in rho, Im(tau_c) 0.23-1.97 — some configs pass the 0.4 gate,
  some don't), so the fold-carrier machinery SHOULD apply. The deltoid
  straight edges and the inter-lobe corridor are unexamined geometry.

  **Fix**: (1) probe the saddle exterior (the probe trains astroid first —
  extend/parameterize to cover the saddle region or run a saddle-focused
  probe) to characterize its failures; (2) verify and apply the full
  treatment: w-carrier, log-rho, 2D (rho,u) fold-carrier, ghost-region
  excision, and the deltoid-cusp/edge excision boundaries — on the saddle;
  (3) examine what serves on the deltoid straight edges and in the
  inter-lobe corridor (lobe charts + exact engine).

  ACCEPTANCE: the saddle exterior clears the 1e-3 held-out eps bar at the
  probe node count; the saddle tile count collapses toward the same ~70
  target; the excision boundaries and serving (straight edges, corridor)
  are documented and correct.
