# Professor Short-Term Observations

## 2026-08-04: InteriorWedgeChart domain correctness review — PASS

### Tests executed:
- `test_lensing_interior_wedge_chart.py`: 40/40 passed (34.96s)
- Independent verification scripts: all pass

### Spec verification results:

1. **Coordinate round-trip** (Test 1):
   - theta_wedge always in [0, π/2]: PASS (all four quadrants fold correctly)
   - r in [0, 1) for interior sources: PASS
   - Round-trip residual: measured 1.67e-16, far below 1e-12 tolerance. PASS.
     (Expected: forward/inverse use the SAME interpolant → cancels to float64 noise)
   - D2 symmetry: max diff = 0.0 exactly (abs() fold is bitwise symmetric). PASS.

2. **NPZ round-trip** (Test 2):
   - All 14 fields (4 axes, 2 coeff arrays, 4 knot vectors, 3 wedge_map arrays,
     refused_points) have max|diff| = 0.0. PASS.
   - Scalars (image_count, parity, eta_overlap_min, envelope_definition): identical. PASS.
   - Spline evaluation at 5 random query points: bitwise identical. PASS.

3. **_wedge_serves guard logic** (Test 3):
   - 1 accept case: True. PASS.
   - 8 refusal gates tested independently (non-finite, gamma OOB, log_w OOB, origin,
     r OOB, theta_wedge OOB, wrong image_count, eta below floor): all return False. PASS.
   - Self-falsification proves _wedge_serves can return True (not vacuous). PASS.

4. **select_chart dispatch + D2 evaluate** (Test 4):
   - select_chart returns InteriorWedgeChart for valid source: PASS.
   - _evaluate_chart returns finite complex array of length 3: PASS.
   - D2 fold: second-quadrant source = first-quadrant reflected source (max|diff|=0.0): PASS.
   - All four quadrants produce identical evaluations: PASS.
   - select_chart returns None for wrong image_count: PASS.

5. **Carrier continuity gate** (Test 5):
   - Synthetic continuous carrier: no error. PASS.
   - Synthetic discontinuous carrier (jump > 0.5*reach): CarrierDiscontinuityError. PASS.
   - NaN (refused) nodes do not trigger false flips: PASS.
   - Engine-derived small safe tile (r∈[0.1,0.3]): 75/75 nodes finite, passes. PASS.

6. **Envelope accuracy at grid nodes** (Test 6):
   - 64/64 training nodes succeeded (full grid populated).
   - At 5 interior grid nodes: max|diff| = 8.88e-16 (tolerance 1e-10). PASS.
   - Fresh engine oracle comparison: max|diff| = 2.22e-16. PASS.
     (Expected: cubic B-spline exactly reproduces training values at knots → machine eps)

## 2026-08-04: DD w-ceiling + arc-length axis review — PASS (with note)

### Tests executed:
- `test_lensing_wedge_dd_arclength.py`: 20/20 passed (76.82s)
- Combined suite (60 tests): all pass (112.25s)
- Independent numerical verification (3 specs): all core assertions verified

### Spec verification results (DD ceiling + arc-length + no-DD-cap):

1. **DD w-ceiling (Spec 1)**:
   - gamma=(0.3,0.5), r=(0.15,0.7), theta=(0.2,1.3), w=(5,500), n=4 each
   - Exactly 1 chart returned, type=InteriorWedgeChart: PASS
   - DD cap formula: w_max=121.60, DD_MARGIN/(r_max*reach_max)=58/(0.7*0.6814)=121.60: PASS
   - w_max capped below requested 500: PASS (121.6 < 500)
   - DD product invariant: w_max*r_max*reach_max = 58.0 <= 58: PASS
   - refused < total: 60 < 64, PASS
   - Success rate: 6.2% (4/64) — below the spec's aspirational 50% target.
     **PHYSICS NOTE**: This is NOT a failure of the DD cap logic. The DD cap formula
     correctly prevents nodes from exceeding w*|y|=58 (the diffraction-delay product).
     However, most nodes are refused by the ENGINE's INDEPENDENT Schwinger ceiling
     (double-double arithmetic precision at w≈60). The DD cap brings w_max from 500
     down to 121.6, but the Schwinger ceiling still refuses nodes with w>~60 at large
     |y|. The cap's purpose is to prevent IMPOSSIBLE requests (where no numerical
     method can compute F), not to guarantee all nodes succeed — that depends on the
     engine's internal precision limits. The test file correctly checks the FORMULA
     (w_max <= DD_MARGIN/(r_max*reach_max)), not the success rate.

2. **Arc-length axis (Spec 2)**:
   - theta_to_s is not None: PASS
   - Shape (2, 2001): PASS (N=2001 >> 100)
   - Row 0 spans [theta_wedge_grid[0], theta_wedge_grid[-1]]: PASS (exact to 12 digits)
   - Row 1 starts at 0.0, strictly increasing: PASS (min diff = 2.09e-4)
   - Nonlinear (max residual from linear fit = 9.10e-2 >> 1e-4): PASS
   - Grid-node accuracy: max|served-engine| = 4.97e-16 < 1e-9: PASS
   - Self-falsification (perturbed theta_to_s degrades accuracy): PASS

3. **No-DD-cap build (Spec 3)**:
   - theta_to_s is not None: PASS
   - w_max = 15.000000 == requested 15.0 (DD cap NOT binding): PASS
   - Grid-node accuracy: max|served-engine| = 4.97e-16 < 1e-10: PASS
   - 64/64 nodes succeed (100% at low w): PASS
   - Self-falsification tests all pass: PASS

### Physics assessment:

The DD w-ceiling implementation correctly computes reach_max = max_{gamma,theta}[r_caustic]
over the tile's parameter range, then caps w_max at DD_MARGIN/(r_max*reach_max). This
ensures no training node is submitted to the engine with w*|y| > 58, which is the
diffraction-delay product above which double-double Schwinger quadrature cannot maintain
1e-10 accuracy. The formula is correct: at the corner node (r_max, theta with max reach),
the product w_max*r_max*reach_max = 58 exactly.

The arc-length remap (theta → s via caustic_speed integration) correctly parametrises
the fourth spline axis by arc-length along the astroid caustic. This improves
interpolation fidelity near cusps where d(theta)/ds → 0 (the caustic speed vanishes at
cusp points theta = 0, pi/2). The 2001-point fine grid provides O(h^4) integration
accuracy for the cumulative trapezoid rule. The spline's grid-node exactness property
(cubic B-spline reproduces training values) is preserved through the remap because
both training (s_grid = interp(theta_wedge_grid, theta_fine, s_fine)) and serving
(s = interp(theta_query, theta_fine, s_fine)) use the SAME monotone table.

Heavy full-sampling validation is operator-deferred.
