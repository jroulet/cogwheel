# Session observations — low-w diffractive chart resolved-band ruling (2026-08-21)

Domain ruling (no code change) on the low-w chart's off-caustic resolved band.
Key code facts gathered (all verified from source, not memory):

- `geometric_amplification` (operator.py L1527) is the w->inf stationary-phase
  sum over REAL images, with `image_kernel` (geometry.py L1131) =
  sqrt|mu| e^{-i pi n/2} (1 + i C1/w + C2/w^2). So the "bare geometric sum"
  ALREADY carries the first two saddle-coefficient corrections; its residual
  vs exact is the diffractive correction F_diff = F_exact - F_geo.
- `fold_ppgo_correction` (_airy_fold.py L531) falls back BYTE-IDENTICALLY to
  geometric_amplification when len(images) != 4 (the 2-image exterior). Its
  Airy fold replacement applies ONLY to the interior 4-image merging pair.
  So "the first-order ppGO correction for the resolved exterior" is a
  misnomer: on the exterior it IS the bare geometric sum (F075,
  test_lensing_fold_ghost_exterior.py::test_fold_ppgo_correction_equals_geometric_exterior).
- The executed low-w chart's partitioned_reference has THREE carrier kinds
  (airy_fold / pearcey_cusp / macro) — NO geometric kind. Serve
  (likelihood.py `_low_w_diffractive_chart_serve` L1972) serves the BARE
  geometric_amplification/lam for resolved nodes, no residual.
- `_node_metrics` (train script L597) measures only the macro-lead residual
  interpolation (chart.evaluate); the geometric branch is never measured, so
  no decline gate / margin covers it. `_macro_has_resolved_node`/`n_resolved_served_cells`
  COUNT resolved cells but assert no accuracy.
- Handoff low_w_chart_rho_partitioned.md + low_w_chart_rho_hierarchy.md
  (binding Professor ruling, 2026-08-21) prescribe THREE carriers, carrier 2 =
  "TWO-IMAGE GEOMETRIC-OPTICS SUM ... the residual against it is the smooth
  O(1) diffractive correction." The design-(b) ruling ("bake ONLY macro-carrier
  residual; serve bare geometric") is a LATER simplification that DROPPED
  carrier 2, resting on "exact where images are well-separated" being read
  literally (geometric sum is the w->inf asymptote, NOT exact at finite w).
- Precedent for geometric-anchored residual: BornResidualChart bakes
  R = F_exact_demod - F_geo_demod (SPEC.md L121-174).
- Wall-band resolved fraction is large: w_split = RHO_END/Delta_tau = 4/Delta_tau,
  Delta_tau ~ 0.3..35 (F024) => w_split ~ 13.6 down to 0.11, so the resolved
  band is most of the [0.02, 60] chart band. Option (b) would cede most of
  the wall band to the engine.

RULING: option (a) — bake the geometric-anchored diffractive residual for the
resolved band — is the correct end state; it RESTORES the binding handoff
ruling (does not re-open a settled one). The served-error margin statistic is
mandatory under ALL options (F069/F076/F082 lineage: gate must bound the
served object). fold_ppgo_correction does NOT reach 1e-4 on the exterior
(byte-identical to the bare sum). Option (c) is a shuffle, not a fix (ppGO arm
== bare sum; Born arm == geometric+ghost ~1e-3, chart covers rho>2 only).

## 2026-08-21 (rho-partitioned low-w chart review, verdict PASS)
- Ran test_lensing_low_w_diffractive_chart.py: 65/65 pass in ~74s (fast tier only).
- `partitioned_reference` rho-partition is physics-correct: caustic nbhd [RHO_LO,RHO_HI] -> airy_fold q=p Wronskian (magnitude-renormalized to sqrt_mu at low w) with restricted-Pearcey fallback keyed on `_fold_amplitudes` refusal (abs(b3)<=_B3_MIN); off-caustic -> macro lead on unresolved / geometric 2-image sum on resolved nodes split at w_split=RHO_END/delta_tau. Serve re-modulates F_ref*sqrt_mu_full*r back to mass_sheet_phase*f_pure/lam exactly -> node-exact ~1e-15 (pinned 1e-10).
- Macro-fold normalization verified: fold residual -> sqrt(1-gp^2)=0.954 at band bottom (measured 0.969, dev 1.5e-2 < 5e-2), min/max|r|=0.295; raw fold form would dive to 0 (w^{-1/6} |F_ref| divergence).
- Independent spot-check: macro witness (0.8,2.0,0.6) delta_tau=8.17 -> w_split=0.489; |F_ref|=sqrt_mu=1.6667 w-independent; kind='macro' at w=0.2, 'geometric' at w=8.
- Minor concerns (non-blocking): (1) off-grid 1e-4 served-accuracy pin is driver/full-bake-deferred (sparse test charts are node-exact only — standard operator ship gate); (2) stale "RED until cusp fix" docstrings in CuspFrefNonVanishing/FoldCuspContinuity tests — the cusp-transition fix HAS landed, both green (doc currency only).
