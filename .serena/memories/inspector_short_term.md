Last build: steps 2/4/8 gates (scripts + test). Clean pass with 2 carried trivials.

## Scope
- `scripts/measure_tube_fraction.py`: config params updated (gammas, fraction grid, TrainingConfig)
- `scripts/measure_far_zone_crossover.py`: full rewrite from carrier-error to geometric tiling-coverage
- `cogwheel/tests/test_lensing_part0_mechanical.py`: added live-doc retired-name check, docstring-absorber-language test, allowlist entry for surrogate.py `_DD_PRODUCT_MARGIN`

## Findings
- **INS-s248-001 (carried, trivial)**: `measure_tube_fraction.py` progress format `f={f:.1f}` produces ambiguous output for linspace(0.05, 0.6, 12) — e.g. both 0.05 and 0.10 display as "0.1". Also in per-gamma detail (line 214). Pure display issue.
- **INS-s248-002 (carried, trivial)**: `TestNoDocstringAbsorberLanguage.test_anti_vacuity` second assertion (`assertGreaterEqual(len(...), 0)`) is always true. Vacuous but harmless since self-falsification test exists separately.

## Verified
- TrainingConfig matches brief (n_gamma=1, n_u=6, n_theta=6, w_nodes_per_decade=6) ✓
- Gamma lists match brief exactly (positive: 0.05,0.1,0.2,0.4,0.7; saddle: 1.1,1.3,1.5,2.0) ✓
- Fraction grid matches brief (0.05 to 0.6, 12 steps) ✓
- exclusion_rho formula matches production (1 + reach_max + eta_max_max - coord_radius_min) ✓
- `_DD_PRODUCT_MARGIN` in surrogate.py confirmed at value 58.0 ✓
- geometry.nearest_caustic_point call signature correct (gamma, beta=0.0, source) ✓
- NearestCausticPoint.distance field exists ✓
- FoldArc fields (theta_lo, theta_hi, cusp_windows) used correctly ✓
- _theta_in_arc modular arithmetic correct for both periodic (astroid) and non-periodic (saddle) arcs ✓
- All 18 tests pass in 0.82s ✓
- Imports verified working (both scripts and test file) ✓
- No production code modified ✓
- No SPEC/DATA_CONTRACTS changes ✓
- No secrets or absolute paths ✓

## Open issues
- INS-s248-001: Display format ambiguity in measure_tube_fraction.py (trivial, carried)
- INS-s248-002: Vacuous assertion in test_anti_vacuity (trivial, carried)
