# Architect Short-Term Observations

Build 8h-d2 tiling correctness (4 defects, all cogwheel/lensing exterior tiling).
WP1(D1+D2 merged): export ppgo_map.annulus_rho(gamma,|y|,kappa=0)=|y|/caustic_geometry(g,k)[0]
(THE ppGO gauge, keyed on caustic_geometry reach NOT reach_scalar); route likelihood.py
_ppgo_cell_coords(~1375, already |y|/reach => byte-equal refactor) + surrogate_training.py:3266
through it. D1: in _train_band_charts REORDER _stratum_ppgo_boundary/_ceiling calls to AFTER
region_exclusion_rho(~3315-3325); for parity==1 derive ppgo_exclusion_rho=annulus_rho(gamma_mid,
|y|_inner) where |y|_inner=region_exclusion_rho-1+coordinate_radius_min. Prof Q1: this is CONSERVATIVE
in every direction - recovered |y|_inner is a LOWER bound on true inner radius (additive gauge subtracts
MIN directional radius, divide by MAX reach) => smaller rho => HIGHER w_cert => never certifies easier
than reality. Saddle(parity!=1) UNCHANGED byte-identical. Report-path assert two gauges agree; reachable-
red: derived < pre-fix physical_exclusion_radius/reach_scalar. Depends on ppGO w_cert being non-increasing
in exterior rho (test as standing invariant). Simplifier: TRIM ppGO artifact stamp (content hash already
guards; no alt rho convention). WP2(D3): Prof OVERRODE the assertion route -> FRAME-INVARIANT RELABEL.
Current far-field label E_ff carries exp(-i w t_min(x)) winding (exact_total is min-relative frame);
node-to-node t_min varies (d t_min/d rho~-1.03, gap~5e-2, w~60 => ~3.1rad) so spline mixes frames =>
FAILS pi/2 continuity, guard-only insufficient. FIX (Prof R2, exact): farfield_envelope_from_partition
multiply result by exp(+1j*w*partition.t_min) => E_tilde (ABSOLUTE-frame post-GO remainder ~1e-4 smooth,
t_min is THE ref not tau_c). reconstruct_farfield gains REQUIRED t_min param, de-tilts exp(-1j w t_min)
BEFORE ghost-restore+reconstruct_from_envelope (serve order: de-tilt->restore ghost->reconstruct). UNIFORM
across all 3 tags (DIFFRACTIVE/KERNEL_SUM/KERNEL_SUM_MINUS_GHOST); ghost commutes. likelihood.py serve
mirror(~1666-1722) pass geom.t_min. Stale-safety: BUMP _FARFIELD_AXIS_SCHEMA value(+ _KNOWN whitelist) so
pre-relabel charts hard-refuse at load. Guard: add _assert_farfield_carrier_continuity on E_tilde node
values (mirror _assert_carrier_continuity surrogate.py:561, SEPARATE fn not flag-param per Simplifier),
call in from_engine exterior branch - PASSES post-relabel. Node-exact telescoping <1e-12; inter-node
DIFFERENTIAL: err_post<err_pre AND err_post<1e-3(TOL_RECON) while err_pre>1e-3. Test Dev updates existing
reconstruct_farfield callers (test_lensing_exterior_windows, test_lensing_born) to pass part.t_min.
WP3(D4): from_engine(surrogate.py:1138) lays UNIFORM theta_c; Prof Q4 DEFINITIVE: kappa=0 positive-parity
astroid cusp SOURCE angles EXACTLY {0,+-pi/2,pi} gamma-INDEPENDENT (only magnitude 2gamma/sqrt(1+-gamma)
varies). Fix: union theta_c grid with {0,+-pi/2,pi} INTERSECT theta_c_range as exact nodes, GATE on
gamma_mid<1 (saddle deltoid cusps NOT on-axis - fall back uniform). Closed-form, NO surrogate_training
import (circular). Fixes test_positive_box_reconstruction_within_budget (via _train->from_engine);
Test Dev removes @expectedFailure, POS_RECON_TOL=0.20 UNCHANGED, on-cusp (0.40,2.183,0) eps 2.6e-1->~1.5e-4.
WP2&WP3 both edit from_engine => WP2 depends_on WP3 (sequence, avoid conflict). has_spec_update=true
(far-field label frame convention). All new tests + test-call-site updates => Test Developer.


(empty — last consolidated by Dreamer on 2026-07-27)
