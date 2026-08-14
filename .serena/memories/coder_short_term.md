# Coder Short-Term Observations

## 2026-08-14 (INS-1-001/002/003 fixes — c3 serve-gate double-mask + orphaned suites)

- INS-1-001 (double-mask crash, gamma>1 every draw): FIXED at BOTH sites.
  `geom.images` is ALREADY real-only (channels `_frame_delays` ->
  `geometry.find_images` returns length-k real array), so indexing it with
  the length-4 `geom.real_mask` IndexErrors on a 2-image saddle. Dropped the
  `[real]` index + the `real = np.asarray(geom.real_mask,bool)` line ->
  `real_images = np.asarray(geom.images)` in likelihood.py
  `_saddle_farfield_analytic` (~L2152) AND surrogate_census.py
  `characterize_sample` saddle block (~L520). Reworded the predicate
  docstring `real_images` param (likelihood.py ~L586): "geom.images is
  already the real-only array (find_images); pass directly -- do NOT index
  with the length-4 channel mask". The interior 4-image sibling rung
  (L1853, gated `if real_mask.sum()==4`) is a harmless all-True no-op and
  was left alone.
- INS-1-002 (3 orphaned old-gate suites error at COLLECTION): RETIRED via
  `git rm` — test_lensing_saddle_tier1_refusal.py,
  test_lensing_saddle_tier1_accuracy.py, test_lensing_saddle_gauge.py. All
  three are, as bodies, the OLD rho-floor/RHO_END/eta-resolvability gate
  ACCEPTANCE suite (constants RHO_END_INFLATED, `_SADDLE_FARFIELD_RHO_FLOOR`
  import, `w_lo*mdt >= RHO_END` monotone-crossing tests). That mechanism is
  GONE; the new c3 gate has its own self-contained acceptance suite
  test_lensing_saddle_serve_gate.py (serve/refuse/both-flip-mechanisms/
  w^-3-monotone/census served==counted). Inspector explicitly authorized
  retirement "if the rho-floor contract is fully superseded". Verified: (a)
  ZERO `_SADDLE_FARFIELD_RHO_FLOOR` refs remain tree-wide post-rm; (b) no
  module imports the 3 files (leaf nodes); (c) serve_gate's
  `_decoy_saddle_blind_surrogate` is LOCAL (L736), so deletion is safe.
  FLAG -> Test Dev / Professor: the 3 files held UNIQUE live coverage the
  boolean serve_gate.py does NOT: tier-1 served-vs-exact p90 ACCURACY
  (accuracy file, via gate-independent `_tier1_serve` FARFIELD_KERNEL_SUM
  reconstruction) and FARFIELD_KERNEL_SUM gauge identity/frame-phase
  round-trip/handover continuity (gauge file). Re-establishing those
  against the new c3 gate needs fresh fixtures that land in the new serve
  region = certification of my own gate = Test Dev scope, NOT mechanically
  migratable (their admission helpers call the retired 3-arg gate).
- INS-1-003 (masked-red in serve_gate.py, contingent on 001 census fix —
  now landed): DELETED `test_census_crashes_reproducing_production_args`
  (the assertRaises(IndexError) crash tripwire); PROMOTED
  `test_census_served_matches_production_gate` from `@unittest.
  expectedFailure` to a plain undecorated live assertion of served==counted
  (verified `__unittest_expecting_failure__`=False). Also rewrote the now-
  stale module-level + class-level "SPEC DISCREPANCY (production defect)"
  docstrings to "CENSUS ARG CONSTRUCTION (resolved, INS-1-001)" so the file
  is internally consistent. Left `test_diagnostic_census_mirror_table`'s
  now-unreachable `except IndexError` branch untouched (per "do not modify
  other tests"; harmless future-regression sentinel).
- SMOKE: py_compile OK on likelihood.py + surrogate_census.py +
  serve_gate.py; serve_gate.py collects 33 cases (no ImportError), tripwire
  absent, served==counted decorator gone. Did NOT run suites as
  certification (role boundary).


## 2026-08-14 (WP2 census mirror -> c3-cert serve-gate signature)

- NOTE: this session's WP1/WP2 "eta re-key" entries BELOW are STALE — that
  build was fully reverted (handoff symmetry_tie_c3_admission.md fact 6;
  checkpoint d5672fa6 not an ancestor). HEAD's WP1 gate
  `_saddle_farfield_analytic_serves` now takes (real_images, source, matrix,
  w_lo) and is a c3-led certificate + image-separation backstop.
- WP2: re-keyed the census mirror in surrogate_census.py
  `characterize_sample` saddle block (gamma>1) onto WP1's 4-arg signature.
  Now builds `real = np.asarray(geom.real_mask, bool)`, `real_images =
  np.asarray(geom.images)[real]`, `source = np.array([y1, y2])`,
  `matrix = macro_matrix(gamma, 0.0, 0.0)` (local import mirroring the
  sibling interior-handoff block just above), `w_lo = float(w_grid.min())`,
  then `_saddle_farfield_analytic_serves(real_images, source, matrix,
  w_lo)`. DROPPED the `rho = caustic_rho(...)` serve-decision line + its
  try/except + `real_delays`.
- KAPPA: census pins beta=kappa=0 (fixed-geometry contract); geom above
  built with kappa=0.0, so matrix uses 0.0 — the faithful mirror of the
  production rung's lens['kappa'] source for census draws. served==counted.
- caustic_rho import KEPT — still used by the band-split at L416
  (characterize_sample) and L305 (classify_fallthrough). Only ONE
  `_saddle_farfield_analytic_serves` call-site in the census, now 4-arg.
- SMOKE: module imports clean; characterize_sample source contains the
  4-arg call and no `real_delays`. No test module touched (Test Dev owns
  the census-mirror decision-level tests + --followup evidence re-run).

## 2026-08-14 (WP1 — saddle far-field gate re-keyed rho->c3 certificate)

- REPLACED `_saddle_farfield_analytic_serves` signature `(real_delays,
  w_lo, rho)` -> `(real_images, source, matrix, w_lo)` in
  cogwheel/lensing/likelihood.py. New body: local import
  `ppgo_error_estimate` from chang_refsdal.geometry; `len(images)<2 ->
  False`; `est=ppgo_error_estimate(images,source,matrix,w_lo)`;
  `est is None -> False` (PRIMARY merge discriminator — divergent mu/c3);
  min pairwise Euclidean image separation via triu_indices >=
  `_SADDLE_FARFIELD_MIN_IMAGE_SEP`; admit iff sep-ok AND
  `_SADDLE_FARFIELD_SAFETY*est <= _SADDLE_FARFIELD_CERT_BAR`. Deleted the
  whole delta_tau resolution leg (no more RHO_END in this gate; RHO_END
  import KEPT — still used at rungs ~2044/2235).
- CONSTANTS: RETIRED `_SADDLE_FARFIELD_RHO_FLOOR=2.0` (+ its comment
  block); ADDED `_SADDLE_FARFIELD_SAFETY=20.0`,
  `_SADDLE_FARFIELD_CERT_BAR=1e-3`, `_SADDLE_FARFIELD_MIN_IMAGE_SEP=0.05`
  (Professor-authorized S/bar at w_lo — DO NOT re-derive). Grep confirms
  ZERO `_SADDLE_FARFIELD_RHO_FLOOR` refs remain in likelihood.py.
- LIVE RUNG `_saddle_farfield_analytic`: now builds source=np.array([y1,
  y2]), matrix=macro_matrix(gamma,beta,kappa), real_images=
  np.asarray(geom.images)[real_mask] (mirrors interior sibling ~1811);
  DROPPED the `rho=caustic_rho(...)` serve-decision try/except entirely;
  calls the 4-arg gate; False -> None fallthrough unchanged. caustic_rho
  import KEPT (still used at ~1506/1811). macro_matrix already module-
  level imported (L101).
- SMOKE-TESTED (production imports only, NO test module): gate executes
  end-to-end on ChangRefsdalChannels partitions, gamma=1.2. Near-caustic
  y=(1.5,0.9)/(0.2,0) REFUSE (S*est 1.9e-2/1.1e-2); far-from-caustic
  y=(3.5,2.2)/(5,3)/(6,4) SERVE (S*est 1.1e-4/5.6e-5/3.1e-5, monotone
  shrinking with |y| as w^-3 c3 expects). Confirms c3-led admission +
  None-refusal + shape handling.
- ACCEPTANCE EVIDENCE (calibrate_saddle_exterior_certificate.py
  --followup) is UNVERIFIED / BLOCKED-ON-TESTDEV: the script borrows
  helpers from cogwheel/tests/test_lensing_saddle_tier1_accuracy.py, whose
  L86 imports the now-retired `_SADDLE_FARFIELD_RHO_FLOOR` -> ImportError
  before any measurement runs. This is anticipated Test-Dev re-key scope
  (per prior memory "TESTS WILL GO RED"), NOT a gate defect. I did update
  the ONE production-gate call site in the script's `_pairing_gate`
  (scripts/, not a test) to the new 4-arg signature so the evidence run
  works once the test module is re-keyed. Test Dev must re-key the L86
  import in test_lensing_saddle_tier1_accuracy.py (drop
  `_SADDLE_FARFIELD_RHO_FLOOR`; import the 3 new constants if asserted) and
  re-point every `_saddle_farfield_analytic_serves(real_delays, w_lo, rho)`
  call onto the 4-arg `(real_images, source, matrix, w_lo)` signature; then
  re-run the --followup script for the max-true-err / max-cert(w_lo) /
  false-admit numbers.
- WP2 (surrogate_census.py saddle census mirror) NOT touched — separate
  work package; the census must re-key its mirror onto the same 4-arg gate
  in the same build (served==counted).

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
