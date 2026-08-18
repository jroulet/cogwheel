# Architect Short-Term Observations

(empty — last consolidated by Dreamer on 2026-08-18)

## 2026-08-18 born_farfield_go_carrier (build 2 — GO carrier) — plan drafted
- REACHABILITY VERDICT (mandatory pre-WP check, Professor+Simplifier concur):
  Closure #1 (bare two-image GO-carrier far-field rung) has NEAR-ZERO honest
  reachability -> NOT BUILT. `geometric_amplification` is genuinely O(w^-3)
  (image_kernel carries C1/C2), so binding it means the EXISTING c3 cert
  (`ppgo_error_estimate`, `_saddle_c3_split_point`) already governs it -- and
  that cert correctly REFUSES the low-w saddle far-field (Fact 5: w_hi p50=4.61,
  no w_split, large |c3| near fold). Fact 4's 1/w law is a DIFFERENT lead-only
  carrier; a 1/w gate to admit where two honest certs refuse is a forbidden
  widen-a-bar move. Numbers: <=~2-3% of prior admissible (thin top slice of
  w_hi>15 minority); >=~75% (w_hi<15) GO-unreachable; positive-parity far
  exterior already Born-served. Weight SHIFTS to #2 (residual chart repr) + #3
  (Born floor band-split).
- SCOPE SHIPPED: WP1 = closure #3 mechanical Born trained-floor band-split in
  `_born_residual_analytic` (likelihood.py 2890-3196). It already detects
  `trained_band_escape` but routes escapes to the never-admitting carrier-only
  cert -> engine (born_analytic 14.61%->0%). Fix inserts a THIRD tier:
  engine-host [w_low, trained_floor) via `_engine_envelope_below_split`,
  chart-serve [trained_floor, w_trust], reusing `_band_split_mask` +
  `_diffractive_bottom_ceiling`. Revives ~6% box-covered pop. WP2 = census
  mirror (thin caller per `_saddle_c3_route`) + 10k engine-free re-run reporting
  revived born_analytic share. Closure #2 = SPEC DECISION only (residual against
  SAME carrier, ONE shared definition, DRY) -> has_spec_update true, NO dead-code
  WP. has_domain_changes true (routing/serving change). Tolerances: bit-exact
  null-split byte-identity, 1e-13 null-residual reconstruction identity, zero
  census route mismatches.

## 2026-08-18 born_farfield_completion triage (INS-1-002)
- WP3's census mirror must be FAITHFUL to production's `_born_residual_analytic`,
  which gates on covered(gamma,rho) AND covered(gamma,rho,chart_w) (trained_band_escape).
  Census `classify_draw` only checked (gamma,rho) -> under-reports born_carrier_only.
  coder_fix: recompute covers with chart_w in the census mirror, matching production
  exactly (same MIRROR FIDELITY rule as fold_exterior_ghost/symmetry_tie_c3_admission).

## 2026-08-18 born_farfield_completion (plan drafted)
- Brief: beyond the trained residual chart box the born_lead_carrier must serve
  ALONE at all w, admitted by its OWN truncation certificate — `covers()==False`
  must NEVER be a straight refusal-to-engine. Two defects: astroid falls through
  outside box; saddle keys on origin-gauge caustic_rho>2 (saturates ~2 on saddle)
  so Born never fires.
- Certificate FORM (both parities, from Professor): |delta| = hypot(a0, 0.5*w*b1)/q2r,
  predicate S*|delta| <= bar at w_HI (LINEAR in w, worst at ceiling — NOT w_lo like
  the c3/ppgo w^-3 remainder). `_born_factors` is parity-agnostic; serve certificate
  reads it directly, BYPASSING the positive-parity policy guard in born_amplification.
- Saddle needs a SEPARATE lower resolution fence w_lo*delta_min >= RHO_END(4.0)
  (delta_min via operator._real_delay_min_separation); positive parity gets its floor
  free via the diffractive F_P rung.
- REUSE _SADDLE_FARFIELD_CERT_BAR=1e-3, _SADDLE_FARFIELD_SAFETY=20; no new Born consts.
  Lead-only carrier stays the ONE shared carrier (no two-carrier fork — out of scope).
- 3 WPs: WP1 helper in _born.py; WP2 lift gate in likelihood.py (_born_residual_analytic
  + new _born_carrier_certificate_serves); WP3 census born_carrier_only route + re-run.
  Surrogate-path Born twin OUT of scope (future-parity caveat). Simplifier: factor only
  the reconstruction TAIL into _born_reconstruct, protect the null-identity byte-identity.
