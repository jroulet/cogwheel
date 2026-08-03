# Professor Short-Term Observations

## 2026-08-04: InteriorWedgeChart domain correctness review — PASS

### Tests executed:
- `test_lensing_interior_wedge_chart.py`: 40/40 passed (34.99s)
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

### Physics assessment:

The coordinate system design is correct:
- `_to_wedge_fixed` exploits the D2 (dihedral-4) symmetry of the astroid caustic
  (reflections across both eigenvalue axes) to reduce the full plane to one wedge
  θ ∈ [0, π/2]. The fold is `abs(y1), abs(y2)` → `atan2(|y2|, |y1|)` which is
  the correct D2 quotient for the astroid.
- The radial coordinate `r = |y|/r_caustic(γ, θ)` normalises by the direction-dependent
  caustic reach, making `r < 1` equivalent to "inside the caustic" — the fundamental
  domain for 4-image Chang-Refsdal configurations.
- The bilinear interpolation of r_caustic at 101 θ nodes × 5 γ nodes introduces only
  O(h²) error where h ~ π/200 ≈ 0.016; this is well below any physical scale.

The tensor-product spline (cubic B-spline on 4 axes: log w, γ, r, θ_wedge) is the
correct interpolation structure for a smooth function on a box domain. The spline
exactly reproduces training values at grid nodes (verified to machine precision).

The carrier continuity check correctly implements the "single nearest-caustic basin"
requirement for SACR-C demodulation: a jump in the critical_source carrier between
adjacent nodes exceeding 50% of the local caustic reach signals a basin boundary crossing
that would make the demodulated envelope discontinuous (and thus uninterpolable by a
smooth spline).

Heavy full-sampling validation is operator-deferred.
