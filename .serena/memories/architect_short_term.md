# Architect Short-Term Observations

2026-08-14 saddle_admission_predicates plan (LOCKED after 2 Professor rounds + Simplifier):
- Re-key `_saddle_farfield_analytic_serves` (likelihood.py ~536-590): retire scalar
  `rho >= _SADDLE_FARFIELD_RHO_FLOOR(2.0)` term; replace with directional
  `eta = nearest_caustic_point(gamma,beta,source,kappa).distance >= _SADDLE_ETA_FLOOR(0.15)`.
  eta_floor=0.15 pinned by Professor inside measured gap (refuse-max 0.05 / admit-min 0.5),
  ~3x above refuse, biased conservative. Serves the TRANSVERSE CONE (previously false-refused).
  Cost is amortized: nearest_caustic_point is once-per-band-serve (channels.evaluate L1977),
  NOT per-w, and already on the critical path — Simplifier's per-sample-cost premise was wrong,
  wedge pre-filter REJECTED (necessary-not-sufficient, false-admit = silent lnL bias).
- Leg B: keep `w_lo*min_delta_tau >= RHO_END(4.0)` but EXCLUDE symmetry-tied mirror pairs
  delta_tau <= tie_eps (1e-12 ABSOLUTE, mirror _CUSP_TIE_EPS). Existing `>0` filter does NOT
  catch them (mirror delta_tau ~1e-15 > 0). Serve = (A AND B).
- CONNECTING REGION: NOT served this build. Leg C (certified w_cert) TRIMMED + auto-wire TRIMMED
  — Professor §0: certified map keyed on same scalar rho, hard-refuses saddle rho<1, no cell
  exists for rho<0.5; and for rho>=1 saddle Leg C is fully redundant with A∧B. Serving the
  connecting region needs offline saddle rho<1 map training = OUT OF SCOPE. Ship a refusal-guard
  test (asserts REFUSE + fall-through). Follow-up build owed.
- image_count==4 deltoid interior fence preserved (must still refuse).
- Census (surrogate_census.py ~490-530) re-gated as separate WP (Foreman-Lite) computing eta,
  mirroring the live predicate exactly (served==counted invariant).
- Oracle: f_schwinger via mass-sheet identity, anchored by 2D rotated-contour at 2-3 pts;
  bar 1e-4 rel |F| at w<=60. SPEC/DATA_CONTRACTS rho-floor mentions -> Librarian doc-sync.

2026-08-14 saddle_admission_predicates DESIGN triage (INS-1-001): WP1's
measured-floor rule was applied as `min(0.5, measured_boundary*2)` and
landed at the 0.5 cap, but the scan's own worst failing edge is eta=0.784
at gamma=2.0 — 0.5 < 0.784 means the shipped floor ADMITS an uncertified,
measured-failing sub-band, directly inverting the plan's own asymmetry
rule (false-admit is a silent lnL bias, never acceptable; when in doubt
refuse). Verdict: coder_fix — raise _SADDLE_ETA_FLOOR to clear the
measured failing edge with margin (not re-litigate the cap formula), and
correct the provenance comment/completion record to the true measured
boundary. Do not route to Test Developer: the fix is a constant value
correction, not a new test; existing T5 gate-flip test suffices to check
the raised threshold trips correctly.

2026-08-14 saddle_admission_predicates DESIGN triage round 2 (INS-4-003):
xfail-masking verdict. Two accuracy tests were left @expectedFailure at the
under-protective floor(0.9) instead of being fixed, which hides the very
breach the measured-floor requirement (plan v2) exists to catch — violates
the standing INS-3-001 rule "the GATE moves, not the test." Routed
coder_fix, chained to the INS-4-001 measurement-currency fix (script must
measure the actual production p90/max contract, no cap, floor >=1.15) —
promoting these two tests to live assertions is CONTINGENT on that floor
fix landing first, not a standalone doc edit.
