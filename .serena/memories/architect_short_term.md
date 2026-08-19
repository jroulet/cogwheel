# Architect Short-Term Observations

2026-08-18 campaign_tiling_design (7a step 2): plan ONE Coder WP — new
cogwheel/lensing/tiling_plan.py + scripts/tiling_plan.py, THIN consumer of
tiling_census internals (_BandCtx/_build_band_ctx/_collect_band_contexts +
cost constants) and production tilers. Emits demand-sized tile plan
(per region×parity×gamma_band, per-tile node counts + w-ranges) gated on
census cell routes['engine_residual']>0 (astroid exterior served by
Born/c3/certified-map => zero chart nodes). Every axis n=ceil(span/res):
gamma_res=C*r_caustic/|dr_caustic/dgamma| (bands butt the wall, never
straddle); n_theta=ceil(kappa_theta*trimmed_arc_span) (F083 density
kappa, not bare count); w-axis=[measured w_lo,w_hi] from demand cells
(lobe-ext 38, saddle_c3 51.6, interior 60 — never blanket 60); annulus
=[rho_handoff_inner, rho_prior_outer~20] in ONE DECLARED gauge (caustic_rho
vs rho_lobe, convert via r_caustic/r_deltoid). Cost = calls*0.0903s, cross-
check vs _self_estimate + census aggregate. ESCALATE if total>5e5 calls OR
any region>40%. Census refresh (10k @HEAD) folded as preamble, not own WP
(Simplifier TRIM). ONE combined plan+cost JSON. tiling_census mirror =
NO-OP (new sibling). has_domain_changes=true (gamma_res deriv, F(w->0)
anchor -1j*sqrt(mu_macro), gauge conversion). 6 Professor test invariants
-> Test Developer.

2026-08-18 tube_trainer_subarc_trim plan: promote F083 arc-trim from test
fixture into surrogate_training `_trim_tube_arc` helper, wire in
`_train_band_charts` per-arc loop AFTER eta sizing (full-arc r_min/w-cap
untouched). Professor ruling: PARITY-ONLY gate (trim iff parity==+1
astroid; saddle returns arc unchanged — parity=sign(det A) is a
topological invariant, byte-identity unconditional; profile predicate
rejected). Single-corner (gamma_hi,eta_max) derivation is inner-
conservative by monotone-nesting; refused==0 hard error is its falsifier
across the full band. Synthetic knee test = bit-exact. Acceptance
spot-check (~200s) is DRIVER post-build, not a permanent test; ceiling
refused==0 AND eps<=0.15 (~1.4x the 0.108). Simplifier: helper (not
inline), parity gate inside helper, 80-pt scan carried verbatim (re-tune
forbidden), drifted-core loud-failure = re-pointed F083 refused==0 pin.

2026-08-19 tiling_plan triage INS-2-001: DD-ceiling clip (fix for
INS-1-001) in _measured_w_range was unwitnessed by any fixture (all
_CELLS sit <=60). coder_fix — add boundary regression cases (log_w_max>
log(60) -> got_hi==60.0, status 'measured_clipped_dd'; fallback box
(2,480)+empty records -> got_hi==60.0, 'prior_box_fallback_clipped_dd')
to the existing MeasuredWAxisEdgeTestCase. Cheap synthetic fixture edit,
not a new test-authorship WP.
