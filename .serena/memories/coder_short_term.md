# Coder Short-Term Observations

## 2026-08-14 (INS-3-001 fix — RAISE saddle eta floor 0.5 -> 0.9)

- FIX: cogwheel/lensing/likelihood.py `_SADDLE_ETA_FLOOR` raised 0.5 -> 0.9.
  The prior `min(0.5, boundary*2)` rule (from a WITHDRAWN driver
  instruction) capped the floor BELOW the scan's own worst-failing edge
  (0.784), newly admitting the eta in [0.5,0.784) band the retired rho>=2
  gate refused; gate-admitted 2-image near-floor witnesses (eta ~0.50-0.53)
  served p90(err)~4.7e-2 (~47x the 1e-3 bar). New rule (recorded in the
  provenance comment): floor = measured worst-failing edge x safety>=1.15,
  NEVER capped below any measured-failing point. 0.784*1.15=0.9016 -> 0.9.
  Per-gamma edges 0.601/0.705/0.784 recorded. Transverse-cone win survives
  (T1 eta~1.994, audited real-use 1.0-2.5, all >= 0.9). Ceiling ~1.0.
- INTENDED TEST TRIPWIRE FLIP (Test Developer scope, NOT a defect): in
  test_lensing_saddle_tier1_accuracy.py, `test_floor_still_below_inspector_
  flagged_edge` asserts floor < 0.784 and was authored to FAIL once the
  Coder raises past the edge (its docstring: "PASSES today, meant to start
  FAILING"). At 0.9 it goes RED = the designed signal. The anchor guard
  (floor >= 0.5) stays green. Test Developer must: promote the two
  @expectedFailure accuracy tests (SaddleTier1NearFloorEtaAccuracyTestCase)
  to plain assertions on witnesses just above 0.9; re-derive
  test_reports_worst_near_floor_locus; flip/retire the tripwire; update
  stale "0.5" docstrings (lines ~19, ~973) and NEAR_FLOOR_ETA_CAP scaling
  (=1.5*floor -> 1.35). I did NOT touch tests (authoring + witness re-search
  = certification of the gate = Test Dev scope).
- WP2 census mirror: NO change needed — it consumes the shared constant
  transitively via `_saddle_farfield_analytic_serves`; the raise flows
  through automatically (served==counted preserved).

## 2026-08-14 (WP2 saddle census mirror re-key onto eta — ALREADY DONE)

- WP2 asked to re-key the saddle far-field CENSUS mirror in
  surrogate_census.py off scalar rho onto directional eta. On re-reading
  current file state (task header flagged it stale-post-WP1): the change
  was ALREADY APPLIED during the WP1 session (see this memory's WP1 entry:
  "CENSUS characterize_sample saddle block now passes eta=geom.caustic_
  distance to the SHARED gate"). NO new edit needed — verified, not
  re-touched.
- Current faithful state: geom built ONCE at L447 via
  geometry_partition(gamma, y=(y1,y2), beta=0.0, kappa=0.0); eta=float(
  geom.caustic_distance) at L453; saddle block L508-523 computes
  real_delays/w_lo and calls _saddle_farfield_analytic_serves(real_delays,
  w_lo, eta) (3-arg re-keyed sig, imported L59 from likelihood). Zero
  threshold literals in the census file (grep _SADDLE_ETA_FLOOR/
  _SADDLE_TIE_EPS/RHO_END/_SADDLE_FARFIELD_RHO_FLOOR -> empty): the shared
  predicate is the single authoritative source.
- KAPPA CONVENTION note: census pins kappa=0 for EVERY draw (docstring L390
  beta=kappa=z_lens=0; geometry_partition L448 kappa=0.0). Live rung uses
  lens['kappa']. Faithful mirror = derive eta from the census's own kappa=0
  partition (production pins kappa=0 too); NOT a hardcode standing in for a
  nonzero live value. Adding a kappa param to characterize_sample would be
  scope creep against the fixed-geometry census contract.

## 2026-08-14 (WP1 saddle serve gate re-key onto directional eta)

- RE-KEYED the tier-1 saddle serve gate `_saddle_farfield_analytic_serves`
  in cogwheel/lensing/likelihood.py from the isotropic scalar rho reach
  floor onto the DIRECTIONAL nearest-caustic distance eta. New signature
  `(real_delays, w_lo, eta)`. Composition = Leg0 (interior fence) guarding
  `A AND B`: Leg A `eta is None or not finite or eta < _SADDLE_ETA_FLOOR ->
  False`; Leg B resolvability with mirror-tie discipline
  `surviving = diff[diff > _SADDLE_TIE_EPS]`, `w_lo*min(surviving) >=
  RHO_END`. The old `rho >= 2` floor implicitly fenced 4-image interiors
  (interior rho<=1); eta>=0.5 does NOT, so I made the fence EXPLICIT:
  `len(real) >= 4 -> return False` (exact parity-blind lobe-interior
  discriminator: 4 real images inside a lobe, 2 everywhere else).
- LIVE RUNG `_saddle_farfield_analytic` reuses `eta =
  float(geom.caustic_distance)` from the partition it already builds (==
  nearest_caustic_point(...).distance) — NO second geometry pass, NO new
  import. WP example call `nearest_caustic_point(gamma, |y|, source)` is
  WRONG (2nd positional is beta, shear orientation, not |y|); reuse of
  geom.caustic_distance sidesteps it. `caustic_rho` import KEPT (still used
  L1478/L1783 positive-parity + census band-split L305/L416).
- CENSUS `characterize_sample` (surrogate_census.py) saddle block now
  passes the already-computed `eta = geom.caustic_distance` to the SHARED
  gate instead of recomputing rho — single source of truth preserved
  (served set == counted set). Removed the dead rho try/except in THAT
  block only; band-split rho block above UNTOUCHED.
- ETA FLOOR MEASURED, not blind-set. Tracked script
  scripts/measure_saddle_eta_floor.py (FARFIELD_KERNEL_SUM zero-envelope
  serve vs exact Schwinger oracle, gate-eligible 2-image resolvable pop,
  1e-4 rel-|F| at band max, w in [8,60]). RAN 2026-08-14: per-gamma worst
  FAIL / first PASS eta = 1.2->0.601/0.355, 1.5->0.705/0.467,
  2.0->0.784/0.858. boundary = 0.784 > 0.25 => deterministic
  min(0.5, boundary*2) DEFAULTS to 0.5 cap. `_SADDLE_ETA_FLOOR = 0.5`,
  `_SADDLE_TIE_EPS = 1e-12`.
- FINDING routed to Professor/Inspector (in constant provenance comment):
  the tight 1e-4 band-max bar is NOT a clean monotone fn of eta (gamma=1.2
  passes 0.355 yet fails 0.601) and its failing edge 0.784 sits ABOVE the
  0.5 floor — so the admitted (0.5, 0.784) sub-band can miss the *tight*
  1e-4 bar. Production certifies at the LOOSER p90<=1e-3
  (test_lensing_saddle_tier1_accuracy.py). If (0.5,0.784) later breaches
  the production p90 bar, RAISE the floor (Professor asymmetry: false-admit
  = silent lnL bias); ceiling ~1.0 before eroding the transverse cone
  (audited eta 1.0-2.5, preserved at 0.5).
- TESTS WILL GO RED (Test Dev scope, NOT a defect): test_lensing_saddle_
  gauge.py, test_lensing_saddle_tier1_accuracy.py, ..._refusal.py all
  import `_SADDLE_FARFIELD_RHO_FLOOR` (now removed) and pass rho as the 3rd
  gate arg. They need re-keying onto eta by the Test Developer.
- Did NOT write a .claude/spec/completed.d/ fragment despite WP asking —
  fragment-writing is Librarian scope per project instructions; the
  measured numbers live in the constant provenance comment + this memory +
  the change report.
