Build: D2 reflection fold follow-up (brief_d2_fold_quadrant, 2026-08-07)
Working tree: surrogate.py (+31), surrogate_training.py (~130 refactor), test_lensing_exterior_polar_fold.py (new, 17/17 PASS)

Resolved from prior review:
- INS-1-001: RESOLVED — train on +y1 lobe (`_SADDLE_LOBE_CENTERS[1:]`), abs fold in _lobe_serves
- INS-1-002: RESOLVED — abs(y1_eig), abs(y2_eig) in _evaluate_chart for LobeInteriorChart
- INS-1-007: RESOLVED — test fixture centroid (+1.5, 0) matches production +y1 orientation

Unresolved (pre-existing test breakages, not new from this build):
- INS-1-003: test_tiles_pin_theta_edges_on_plus_minus_pi still asserts [-π,π]
- INS-1-004: test_none_tiling_is_the_uniform_grid still expects [-π,π] centers
- INS-1-005: test_lobe_interior_tiles_are_nonempty_and_cusp_aligned still asserts ==3 cusp angles
- INS-1-006: 21+8 failures in test_lensing_surrogate_lobe.py — test fixtures use admissions[0] (-y1 centroid), need migration to admissions[1] (+y1 canonical)

Production code verified correct:
- _to_exterior_fixed uses abs(y1),abs(y2) -> _to_caustic_fixed (3 call sites consistent)
- _lobe_serves: abs fold correct when chart.centroid[0] > 0
- _evaluate_chart LobeInteriorChart: abs(y1_eig),abs(y2_eig) consistent with gate
- _farfield_tiles: [0,π/2] theta at same n_per_side → same tile count, 4x finer resolution
- _farfield_exterior_tiles: cusp-angle fold into [0,π/2] verified consistent with _exclude_near_cusp
- _lobe_cusp_source_angles: abs-dedup correct physics for deltoid D2 symmetry
- _lobe_interior_tiles: theta_range=(0,π) correct for folded lobe-local coordinates
- Training: _SADDLE_LOBE_CENTERS[1:] trains +y1 lobe, si=0 always
- select_chart dispatch unchanged (priorities correct)
- Tube/wedge serve paths unchanged (no y1_eig,y2_eig args — structurally unaffected)

New test file (test_lensing_exterior_polar_fold.py):
- 17/17 PASS, ~113 s total
- Exterior: envelope bit-identity, reconstructed F identity (1e-14), select_chart consistency, delay identity
- Lobe: F identity (1e-12), corridor gate non-degenerate, select_chart consistency
- Regression: tube surrogate (66/66) and wedge chart suites subprocess-gated, both green

No new production bugs found. Verdict: PASS.
