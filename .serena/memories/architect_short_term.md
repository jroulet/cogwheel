# Architect Short-Term Observations

(empty — last consolidated by Dreamer on 2026-08-12)

## saddle_above_ceiling_serving (2026-08-13, FINAL plan emitted)
- 2 WPs (Simplifier LEAN): WP1 Coder = `critical_delay_regauged` helper in
  `_gauge.py` (FIXED module band-floor w_floor const) + tier-1 rung method
  modeled on `_ppgo_above_ceiling` RE-GATED for cheap band w<=60
  (FARFIELD_KERNEL_SUM, no ghost, oracle=exact Schwinger; F_op DIVERGES for
  saddle) + curvature-over-resolution admission gate
  max_a |C1_a|/(w_lo*Δτ_a) <= ~2.5e-4 + `SaddleTierRefusal(LensDomainError)`
  tier-3 refusal mapped to -inf + intercept wiring in
  `_amplification_coefficients`. WP2 Foreman-Lite = census wiring in
  surrogate_census (saddle out-of-box bucket drops by tier-1 served count vs
  six-way fallthrough_breakdown; mirror `ppgo_fold` served=True pattern).
- TIER 2 (chart) DEFERRED to future training-run build (FAST TESTS ONLY, NO
  training). Only tier-2 artifact landed now = `critical_delay_regauged` pure
  helper (float-exact identity test pre-satisfies a committed tier-2 accept).
- Simplifier watch-notes folded in: (a) confirm/introduce unambiguous fixed
  band-floor constant name in `_gauge.py`; (b) curvature gate justified IF
  saddle C1 coefficients already on the geometry partition object (Coder must
  reuse geom.*, no extra engine call — else fall back to resolution-only);
  (c) Shard B handover test must sweep the GATE CRITERION (w_lo across RHO_END
  / curvature threshold, result flips tier1->tier3), NOT internal kernel
  values.
- 3 disjoint shards (one file each, F057): A accuracy dist + refusal locus w/
  paired admit witness; B gauge float-identity + gate-criterion handover; C
  astroid parity==1 byte-identity.

## saddle_above_ceiling_serving (2026-08-13, plan drafted)
- Brief: serve far-from-caustic macro saddle (gamma>1) via 3-tier ladder; NO
  fallthrough to direct eval. Tier1 = re-gauged KERNEL_SUM analytic serve (no
  chart, ~96.4%), Tier2 = chart of re-gauged envelope (DEFERRED — untestable
  without training), Tier3 = named refusal (~0.7%).
- Professor r1: implement tier1 as the FARFIELD_KERNEL_SUM far-field path,
  reusing `_ppgo_above_ceiling` body almost verbatim, re-gated for cheap band
  (w<=60); tier-1 accept = resolution gate (w_lo*min_delta_tau>=RHO_END) + an
  engine-free |E| proxy (|G|/|F| at w_lo); the re-gauge tau_c rule is a TIER-2
  (chart) concern, NOT tier-1. Add pure helper `critical_delay_regauged(delays,
  w_floor)` in `_gauge.py` with w_floor a FIXED band-floor constant (never live
  w_lo — that is the train/serve gauge-skew bug class). Ghost must NOT be added
  to tier1. Oracle for gamma>1 is exact Schwinger; operator.F_op diverges for
  saddle — must NOT be used.
- Simplifier: 2 WPs. WP1 = gauge helper + tier-1 rung + gate + tier-3 refusal
  exception; WP2 = census wiring (saddle out-of-box bucket drops by tier-1
  count vs six-way breakdown). Defer ALL tier-2 chart plumbing.
- 3 disjoint test shards (F057, one file each): A accuracy+refusal, B gauge
  identity+handover continuity, C astroid byte-identity.

## saddle_above_ceiling_serving TRIAGE (2026-08-13): INS-2-002 (tier-1 rung
gates rho via caustic_rho(...,kappa=0.0) hardcoded instead of lens['kappa'],
+ false 'dispatch refused kappa!=0' comment) = coder_fix. Sibling
_ppgo_above_ceiling already passes lens['kappa']; one-line parity fix + a
comment correction, no design implication, no WP-2 census interaction
(census draws are kappa=0 only).

## saddle_above_ceiling_serving REVISED (2026-08-13, plan v2)
- Brief EVOLVED: tier 2 now ~23% (facts 8-10). Still DEFER tier-2 chart
  (needs a training run; forbidden). Ship tier 1 + gauge home + census.
- Professor r2 DECISIVE on GATE: RESOLVABILITY n_real>=2 AND
  w_lo*min_delta_tau>=RHO_END (same as _ppgo_above_ceiling), NOT a ghost
  proxy (F028/F032). Switch-saturation is ALWAYS true by construction, so
  cannot discriminate; tau_switch is a TIER-2-only concern.
- Tier-1 serve = reconstruct_farfield(envelope=ZEROS, FARFIELD_KERNEL_SUM);
  _farfield_switch hardcodes S_a=1, tau_c=0. NOT _ppgo_above_ceiling.
- Simplifier: separate method, gate-miss returns None. Dissent: standalone
  gauge helper premature — OVERRIDDEN by brief design point + acceptance 4.
- 2 Coder WPs; 3 disjoint test shards (F057).
