# Coder Short-Term Observations

2026-08-13 WP2 (interior fold-ppGO re-gate, likelihood.py ~1786-1861):
Re-gated the raw-ppGO interior handoff. Gate is now
`int(geom.real_mask.sum()) == 4` for BOTH parities (replaces the rho<=1
leg AND the removed saddle-only `gamma>1 & !=4 -> None` guard; 4-image is
the exact caustic-interior predicate, 0/2400 disagreements vs closed-form
caustic, F073). Leg3 (_uniform_error_estimate) replaced by the WP1 c3
certificate: `w_min=float(dense_w.min())`; `est=geometry.
ppgo_error_estimate(real_images, source, matrix, w_min)`; admit iff
`est is not None and est*_PPGO_INTERIOR_SAFETY <= CERTIFICATION_BAR`.
New module constant `_PPGO_INTERIOR_SAFETY = 2.0` (Fact 3: p99 0.953,
MAX 0.980, 0% optimistic). Serve path BYTE-UNCHANGED (geometric_amplification
-> f_minrel -> subtract ppgo_sum -> reconstruct_farfield FARFIELD_KERNEL_SUM
-> _reduce_dense_kernels); NO ghost term, NO per-serve ghost_kernel (4-image
=> GhostAbsentError). `except (LensDomainError, ValueError,
ZeroDivisionError): pass` -> fall through to `return None`.
LEG-2 DECISION (by evidence sweep, ppgo_cert_sweep.json, 434 cfg): all
78 interior configs FAIL leg 2; the c3 certificate at S=2.0 admits 230
band rows, MAX true err 4.804e-05 < 1e-4 bar, max ratio 0.972, 0 over bar
-> leg 2 REMOVED from the interior rung. `_merging_fold_pair` and
`_XI_FOLD_THRESHOLD` DEFINITIONS KEPT (fold arm _airy_fold.fold_amplification
still uses them); `_XI_FOLD_THRESHOLD` is now ORPHANED in likelihood.py
(comment refreshed) -> flag to Inspector. Near-caustic divergent-|mu|
subset (rho->1) not DIRECTLY sampled in the sweep (rho in [0.05,0.2]);
covered structurally by the certificate self-refusing (None / over-bar) as
sqrt|mu| diverges. UNVERIFIED: no direct engine sample at rho->1 (Coder
scope; c3 self-refusal is by construction).
STEP D (report-only, NO re-gauge): `_ppgo_cell_coords` (~1438) and
`_train_band_charts` (surrogate_training.py ~4981) BOTH use
`ppgo_map.caustic_rho` (SCALAR max-reach gauge) purely as a MAP COORDINATE
to index the certified ppGO map / read w_trust/w_ceiling from the cell the
map was built with -- they NEED the scalar gauge (must match map build),
NOT the adapted directional gauge. NEITHER relies on "rho==1 is the
caustic": _ppgo_cell_coords only uses rho as a grid coord (its saddle
rho<1 -> None is the SITE5 defense guard, not a caustic-boundary claim);
_train_band_charts's separate INTERIOR TILING uses the adapted directional
gauge (r=|y|/r_caustic, _from_caustic_fixed, where rho==1 IS the caustic)
-- a DIFFERENT coordinate from the ppGO-map caustic_rho scalar gauge.
Correct as-is; not re-gauged.
Import smoke test PASS (_PPGO_INTERIOR_SAFETY=2.0 present). Did NOT run
test suites (Coder scope).

2026-08-13 WP1 ppgo_c3 (interior ppGO certificate): ported
`.claude/handoff/ppgo_c3_reference.py`'s `series_coefficients` into
`chang_refsdal/geometry.py` as private eps-graded polynomial algebra
(`_pmul/_padd/_pscale/_linear_power/_gaussian_moment_table` +
`_series_coefficients` + `_c3_coefficient`), placed after
`saddle_coefficients`. Reused existing `hessian()` (byte-identical to the
reference's hess construction) and `magnification()` instead of
duplicating. Added stdlib `import math`. Public
`ppgo_error_estimate(real_images, source, matrix, w_min)` = sum_a
sqrt|mu_a|*|c3_a| / w_min**3; returns None on w_min<=0 or any non-finite
mu/c3 (near-critical images -> mu=inf -> refuse). NO ghost term (true
interior ghost is exactly 0). `source` accepted but unused (API symmetry,
documented). Exported from chang_refsdal/__init__.py. VERIFIED against the
reference self-test: ported c1==1j*C1, c2==C2 vs shipped
saddle_coefficients to ~1e-14 over 46 images; w^-3 scaling exact
(e(10)/e(20)=8.0); None-guards fire. No gate/serve path touched (that is
WP2). Did NOT run the repo test suite (Coder scope).
