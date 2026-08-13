# Test Dev Short-Term Observations

2026-08-13 SHARD B test_lensing_saddle_tier1_refusal.py (NEW, 11 tests, 3
classes + anti-vacuity base, 4.2s, GREEN): certifies the tier-1 far-from-
caustic macro-saddle analytic serve rung REFUSES (falls through BY NAME, not
exception) and is wired to the load-bearing resolvability variable. Fixtures
built STRUCTURALLY (no training): KB known-bad gamma=1.519 angle0.30
y=(1.707,0.528) |y|=1.787 -> n_real=2, w_lo=24 -> w_lo*min_delta_tau<RHO_END=4
(refused); AD admitted gamma=1.5 y=(2.6,0) w_lo=8 (served, returns 4-tuple
ending in ChangRefsdalGeometryPartition). Part1 REFUSAL: _saddle_farfield_
analytic returns None for KB; dispatch driven on a MagicMock stub with real
gate side_effect + _force_direct=True routes to sentinel 'SEED_PATH' proving
fall-through to seed/exact. Part2 SELF-FALSIFICATION (mirrors ppgo_above_
ceiling teeth): mock.patch.object(likmod,'RHO_END',1e9) refuses AD;
patch RHO_END=0.0 admits KB; boundary w_star=RHO_END/mdt via np.nextafter;
patch likmod._saddle_farfield_analytic_serves to broken always-admit lambda
-> KB wrongly served (proves the predicate, not something incidental,
decides). Patching likmod.RHO_END propagates to BOTH method and shared
predicate (global lookup at call time). Part3 CENSUS: characterize_sample on a
NON-matching astroid TubeChart surrogate (gamma 0.3-0.5 forces select_chart
->None so tier-1 branch reached) labels AD 'saddle-farfield-analytic', leaves
KB 'born'; census.served == direct _saddle_farfield_analytic_serves for both.

*** PRODUCTION REGRESSION FOUND via backward-compat audit (route to Coder) ***
WP-2's NEW census tier-1 block (surrogate_census.py:~514, uncommitted this
build) `real_delays = np.asarray(geom.delays)[real_mask]` CRASHES IndexError
for corridor/lobe-interior SADDLE sources where geom.delays is EMPTY (len 0)
but geom.real_mask is len 2. Breaks pre-existing (prior-build)
test_lensing_saddle_rho_guards.py::CensusBandSplitMirrorIntegrityTestCase
(2 failed + 2 errors) -- these are the CORRECT loud regression witness; left
RED, NOT edited (scope). The LIVE rung likelihood._saddle_farfield_analytic
does the identical index but never crashes: it builds a FRESH per-source
partition (delays/real_mask lengths always match); the census REUSES an
earlier degenerate geom. Fix (Coder): guard e.g.
`len(geom.delays)==len(geom.real_mask) and len(geom.delays)>0` (or try/except)
before the boolean index. My own suite is green because KB/AD fixtures yield
matching-length geometry. Neighbor suites: test_lensing_surrogate_census.py
34 passed/14 skipped; test_lensing_born.py all pass.

2026-08-13 SHARD A test_lensing_saddle_tier1_accuracy.py (NEW, 12 tests, 4
classes, 15.5s): certifies tier-1 zero-envelope FARFIELD_KERNEL_SUM saddle
serve vs EXACT engine (partition.exact_total; F_op FORBIDDEN — diverges for
saddle) in cheap band w<=60. KEY MEASURED FINDING (re-confirms SHARD C):
tier-1 accuracy is governed by CAUSTIC PROXIMITY rho, NOT the resolvability
gate w_lo*mdt>=RHO_END(4). The raw gate-admitted set (rho down to 1.05) has
p90~1e-2 with near-caustic outliers err O(1-10) — spec's p90<=1e-3 is FALSE
there. So certify the rung's ACTUAL contract domain: FAR-FROM-CAUSTIC
(rho>=RHO_FAR=2.0) AND resolvable → measured p90~5e-5, max~7e-4 (20x headroom
under 1e-3). Draw via _draw_far_from_caustic(seed=42): reject LensDomainError
caustic-misses + gate-fails. PIN gamma=1.5859, y=(-1.1208,-0.9002): n_real=2,
mdt=0.933 → gate REFUSES at w_lo=RHO_END=4 (w_lo*mdt=3.73<4, satisfies spec
"refused" branch) but ADMITS at w_lo=8 (7.47>=4) and serves WRONGLY
err.max=9.28e-2 at w~9.5 — LEAKY-GATE WITNESS (the "never served wrongly"
invariant is FLOOR-DEPENDENT, flagged as spec discrepancy). Proximity class:
matched far/near pair at gamma=1.4 ang=0.05, rho=2.5 err~4.3e-5 vs rho=1.10
err~6.15 (both gate-admitted) proves proximity dominates. Self-falsification:
+5%|F_exact| corruption, mismatched A-serve-vs-B-exact oracle,
oracle-nontrivial-structure. DO NOT use wrong-t_min falsification —
zero-envelope reconstruction is t_min-gauge-INDEPENDENT (0*exp=0). Neighbor
suites green: test_lensing_ppgo_above_ceiling + test_lensing_saddle_gauge
35/35. Audit: only this file + saddle_gauge use
_saddle_farfield_analytic_serves (2-arg (real_delays, w_lo)); WP-1/WP-2 added
new symbols, changed no existing signature/constant. No production change.

2026-08-13 SHARD C test_lensing_saddle_gauge.py (NEW, 20 tests, 4 classes,
13.6s): tier-1 far-saddle gauge home. KEY PHYSICS: `_saddle_switch_delay`/
`_saddle_phase_delay` in _gauge.py; producer round-trip
farfield_envelope_from_partition->reconstruct_farfield reproduces
part.exact_total BIT-EXACTLY (0.0). Tier-1 zero-envelope reconstruction is
gauge-INDEPENDENT bit-exactly (feeding switch-gauge delay vs tau_min as
t_min gives identical F, since 0*exp(...)=0) — the definitive form of the
"no jump on mis-keyed gauge switch" diagnostic. SPEC DISCREPANCY (flagged,
encoded honestly): spec's |F_tier1-F_near|/|F|<=1e-3 AT the w_lo*mdt=RHO_END=4
boundary is UNREACHABLE — RHO_END=4 is RESOLVABILITY not envelope-negligible;
tier-1 accuracy is dominated by CAUSTIC PROXIMITY not w*mdt: at rho=1.05
(w_lo*mdt=25) magres blows to O(200), at rho=1.10 O(8), only at rho=1.5
(well-separated) magres top 3.2e-5/band 2.9e-3. Fixture: gamma=1.3, rho=1.5,
angle=0. Serve gate never crosses along a rho sweep (mdt too big); cross it
by sweeping w_lo (band floor) for fixed source — single monotone False->True
at w_lo=RHO_END/mdt=0.573. Teeth: wrong t_min rt err 1.87; nonzero-env gauge
dependence 2.08; wrong switch formula 0.667. Part-2 uses FARFIELD_DIFFRACTIVE
(all-zero switch) to isolate frame telescoping (recovery 2.2e-16).
Neighbor test_lensing_gauge.py 46/46 green. No production change.
