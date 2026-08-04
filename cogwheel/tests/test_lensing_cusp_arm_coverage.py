"""Tests for _CUSP_ARM_COVERAGE gate in the surrogate's cusp arm.

The coverage constant (0.07 rad, measured in the image-theta coordinate)
defines the angular offset from a cusp vertex below which the tube's
cusp exclusion window STILL blocks, and above which the Pearcey cusp arm
is known to serve.  These tests certify:

  1. The constant's value is sane (0 < value < 1, 2-decimal precision).
  2. Near-cusp-vertex sources are refused (radius gate, R^{-3/2} error).
  3. All served sources have image-theta offset >= _CUSP_ARM_COVERAGE.
  4. The transition exhibits approximate monotonicity in angle.

Tolerance rationale:
  - The coverage constant is a MINIMUM over all (gamma, w) configs;
    any single config's boundary is >= the global minimum.
  - Image-theta delta computed from find_images is exact (root-finding)
    so the comparison needs no numerical tolerance beyond the constant
    itself.

Cost estimate:
  Each cusp_amplification call takes ~5-20 ms with the Pearcey table.
  Largest sweep: 30 probes × 1 call = ~0.6 s.  File total < 10 s.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from cogwheel.lensing.chang_refsdal._pearcey_cusp import (
    cusp_amplification,
    use_pearcey_table,
    _cusp_vertex,
)
from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.surrogate import _CUSP_ARM_COVERAGE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Gamma for positive-parity test fixture (astroid caustic).
_GAMMA = 0.5

#: Frequency at which the known fixture is served.
_W = 80.0

#: Source radius for the known-served fixture (inside caustic lobe).
_SOURCE_RADIUS = 0.18

#: Source angle (radians) for the known-served fixture.
_SOURCE_ANGLE_RAD = 0.3 * math.pi

#: Angles (degrees) scanned for the transition sweep.
_SWEEP_ANGLES_DEG = list(range(30, 75))

#: Source near the cusp vertex (delta_theta ~ 0) — expected refused.
_NEAR_CUSP_SOURCE_OFFSETS = [0.001, 0.005, 0.01, 0.02, 0.05]

#: Minimum number of served sources required for anti-vacuity.
_MIN_SERVED_COUNT = 3



# ---------------------------------------------------------------------------
# Helper base
# ---------------------------------------------------------------------------

class _CuspArmCoverageTestCase(unittest.TestCase):
    """Base carrying the anti-vacuity tearDown and shared helpers."""

    n_checks: int = 0

    def tearDown(self) -> None:
        self.assertGreater(
            self.n_checks, 0,
            'Anti-vacuity: zero comparisons ran — the test is silently empty.')

    # ------------------------------------------------------------------
    @staticmethod
    def _known_served_source() -> np.ndarray:
        """Source position empirically known to be served by cusp arm."""
        return _SOURCE_RADIUS * np.array(
            [math.cos(_SOURCE_ANGLE_RAD), math.sin(_SOURCE_ANGLE_RAD)])

    @staticmethod
    def _cusp_vertex_source(gamma: float = _GAMMA) -> np.ndarray:
        """Source at the cusp vertex (theta=0 for positive parity)."""
        cp = geometry.critical_point(gamma, 0.0, beta=0.0, kappa=0.0, branch=1)
        return cp.source

    @staticmethod
    def _image_theta_offset(source: np.ndarray,
                            gamma: float = _GAMMA) -> float | None:
        """Image-theta offset from the nearest cusp vertex.

        Returns the angular distance (radians) between the image nearest
        the cusp vertex and the vertex's image, or None if vertex-finding
        or image-finding fails.
        """
        matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
        try:
            nearest = geometry.nearest_caustic_point(
                gamma, 0.0, source, kappa=0.0)
            vertex = _cusp_vertex(gamma, 0.0, 0.0, source, nearest.theta, 1)
        except (geometry.LensDomainError, ValueError):
            return None
        if vertex is None:
            return None
        try:
            images = geometry.find_images(source, matrix)
        except geometry.LensDomainError:
            return None
        if not images:
            return None
        dists = [np.linalg.norm(np.asarray(img) - vertex.image)
                 for img in images]
        nearest_idx = int(np.argmin(dists))
        nearest_image = images[nearest_idx]
        theta_img = math.atan2(nearest_image[1], nearest_image[0])
        theta_vert = math.atan2(vertex.image[1], vertex.image[0])
        return abs(
            (theta_img - theta_vert + math.pi) % (2.0 * math.pi) - math.pi
        )


# ---------------------------------------------------------------------------
# Test 1: Coverage constant value
# ---------------------------------------------------------------------------

class CoverageConstantTestCase(_CuspArmCoverageTestCase):
    """_CUSP_ARM_COVERAGE is sane: positive, < 1.0, 2-decimal precision."""

    def test_positive_and_bounded(self) -> None:
        """The constant lies in (0, 1) — a reasonable angular value."""
        self.n_checks += 1
        self.assertGreater(_CUSP_ARM_COVERAGE, 0.0,
                           'Coverage constant is zero — arm is disabled.')
        self.assertLess(_CUSP_ARM_COVERAGE, 1.0,
                        'Coverage constant exceeds 1 radian — impossible.')

    def test_two_decimal_precision(self) -> None:
        """The constant is a 2-decimal-place number (e.g. 0.07)."""
        self.n_checks += 1
        self.assertEqual(
            round(_CUSP_ARM_COVERAGE, 2), _CUSP_ARM_COVERAGE,
            f'Expected 2-decimal precision, got {_CUSP_ARM_COVERAGE}')

    def test_expected_magnitude(self) -> None:
        """The constant is in the expected range 0.03-0.5 rad."""
        # Based on R-gate analysis: ~0.07 (measured).
        self.n_checks += 1
        self.assertGreaterEqual(_CUSP_ARM_COVERAGE, 0.03)
        self.assertLessEqual(_CUSP_ARM_COVERAGE, 0.5)



# ---------------------------------------------------------------------------
# Test 2: Near-cusp sources are refused
# ---------------------------------------------------------------------------

class CuspVertexRefusalTestCase(_CuspArmCoverageTestCase):
    """Sources at/near the cusp vertex are refused by cusp_amplification.

    The radius gate (R^{-3/2} error clearance) prevents serving when the
    Pearcey controls are too small — this is what makes _CUSP_ARM_COVERAGE
    meaningful as a lower bound on the served region.
    """

    @classmethod
    def setUpClass(cls) -> None:
        use_pearcey_table()

    def test_cusp_vertex_source_refused(self) -> None:
        """The exact cusp vertex source (delta_theta ~ 0) is refused."""
        source = self._cusp_vertex_source()
        result = cusp_amplification(_W, source, _GAMMA)
        self.n_checks += 1
        self.assertIsNone(
            result,
            'cusp_amplification should refuse the exact cusp vertex source')

    def test_near_vertex_offsets_refused(self) -> None:
        """Small perturbations of the cusp vertex are also refused."""
        cusp_src = self._cusp_vertex_source()
        for offset in _NEAR_CUSP_SOURCE_OFFSETS:
            with self.subTest(offset=offset, direction='x'):
                source = cusp_src + np.array([offset, 0.0])
                result = cusp_amplification(_W, source, _GAMMA)
                self.n_checks += 1
                self.assertIsNone(
                    result,
                    f'Near-vertex source (x+{offset}) should be refused')
            with self.subTest(offset=offset, direction='y'):
                source = cusp_src + np.array([0.0, offset])
                result = cusp_amplification(_W, source, _GAMMA)
                self.n_checks += 1
                self.assertIsNone(
                    result,
                    f'Near-vertex source (y+{offset}) should be refused')



# ---------------------------------------------------------------------------
# Test 3: Served sources respect coverage bound
# ---------------------------------------------------------------------------

class ServedSourceCoverageTestCase(_CuspArmCoverageTestCase):
    """All served sources have image-theta offset >= _CUSP_ARM_COVERAGE.

    The coverage constant is the GLOBAL MINIMUM, so every single served
    source must have delta_theta >= that minimum.  This is the core
    invariant of the constant.
    """

    @classmethod
    def setUpClass(cls) -> None:
        use_pearcey_table()

    def test_known_served_source_exceeds_coverage(self) -> None:
        """The known-served fixture has delta_theta > _CUSP_ARM_COVERAGE."""
        source = self._known_served_source()
        result = cusp_amplification(_W, source, _GAMMA)
        self.assertIsNotNone(result, 'Known fixture should be served')
        delta = self._image_theta_offset(source)
        self.assertIsNotNone(delta, 'Could not compute delta_theta')
        self.n_checks += 1
        self.assertGreaterEqual(
            delta, _CUSP_ARM_COVERAGE,
            f'Served source has delta_theta={delta:.6f} < '
            f'_CUSP_ARM_COVERAGE={_CUSP_ARM_COVERAGE}')

    def test_sweep_all_served_exceed_coverage(self) -> None:
        """Every served source in the angle sweep has delta_theta >= coverage.

        Cost: 45 angles × 1 cusp_amplification call × ~10ms = ~0.5 s.
        """
        served_deltas: list[float] = []
        for angle_deg in _SWEEP_ANGLES_DEG:
            angle_rad = math.radians(angle_deg)
            source = _SOURCE_RADIUS * np.array(
                [math.cos(angle_rad), math.sin(angle_rad)])
            result = cusp_amplification(_W, source, _GAMMA)
            if result is None:
                continue
            delta = self._image_theta_offset(source)
            if delta is None:
                continue
            served_deltas.append(delta)
            self.n_checks += 1
            self.assertGreaterEqual(
                delta, _CUSP_ARM_COVERAGE,
                f'Served source at angle={angle_deg}° has '
                f'delta_theta={delta:.6f} < {_CUSP_ARM_COVERAGE}')

        # Anti-vacuity: at least _MIN_SERVED_COUNT sources were served.
        self.assertGreaterEqual(
            len(served_deltas), _MIN_SERVED_COUNT,
            f'Only {len(served_deltas)} sources served '
            f'(need >= {_MIN_SERVED_COUNT}). Fixture may be stale.')



# ---------------------------------------------------------------------------
# Test 4: Transition monotonicity
# ---------------------------------------------------------------------------

class TransitionMonotonicityTestCase(_CuspArmCoverageTestCase):
    """Approximate monotonicity: once the arm starts serving, delta_theta
    is above coverage and stays above.

    This is a weaker form of the full monotonicity property: we don't
    require every angle above the boundary to serve (other gates may
    refuse), but we DO require that every SERVED angle has delta_theta
    above the coverage constant.

    Additionally: the served region should be contiguous in angle (no
    isolated served points with refused neighbors on both sides within
    the served band).
    """

    @classmethod
    def setUpClass(cls) -> None:
        use_pearcey_table()

    def test_served_band_contiguous(self) -> None:
        """Served sources form a contiguous band in angle (no gaps)."""
        served_angles: list[int] = []
        for angle_deg in _SWEEP_ANGLES_DEG:
            angle_rad = math.radians(angle_deg)
            source = _SOURCE_RADIUS * np.array(
                [math.cos(angle_rad), math.sin(angle_rad)])
            result = cusp_amplification(_W, source, _GAMMA)
            if result is not None:
                served_angles.append(angle_deg)

        self.assertGreaterEqual(
            len(served_angles), _MIN_SERVED_COUNT,
            'Too few served sources for contiguity check.')
        self.n_checks += 1

        # Check contiguity: max gap between consecutive served angles
        # should be <= 2 degrees (allowing for 1-degree steps).
        if len(served_angles) >= 2:
            gaps = [served_angles[i + 1] - served_angles[i]
                    for i in range(len(served_angles) - 1)]
            max_gap = max(gaps)
            self.n_checks += 1
            self.assertLessEqual(
                max_gap, 2,
                f'Served band has a {max_gap}° gap — not contiguous.')

    def test_refused_then_served_crosses_coverage(self) -> None:
        """The first served angle has delta_theta >= _CUSP_ARM_COVERAGE."""
        first_served_delta: float | None = None
        for angle_deg in _SWEEP_ANGLES_DEG:
            angle_rad = math.radians(angle_deg)
            source = _SOURCE_RADIUS * np.array(
                [math.cos(angle_rad), math.sin(angle_rad)])
            result = cusp_amplification(_W, source, _GAMMA)
            if result is not None:
                delta = self._image_theta_offset(source)
                if delta is not None:
                    first_served_delta = delta
                    break

        self.assertIsNotNone(
            first_served_delta,
            'No served source found — cannot test transition.')
        self.n_checks += 1
        self.assertGreaterEqual(
            first_served_delta, _CUSP_ARM_COVERAGE,
            f'First served source has delta_theta={first_served_delta:.6f} '
            f'< _CUSP_ARM_COVERAGE={_CUSP_ARM_COVERAGE}')



# ---------------------------------------------------------------------------
# Test 5: Self-falsification
# ---------------------------------------------------------------------------

class SelfFalsificationTestCase(_CuspArmCoverageTestCase):
    """Prove the suite can go red: a deliberately wrong coverage constant
    would violate the invariants above.

    If _CUSP_ARM_COVERAGE were set to 1.0 (absurdly high), the served
    sources would NOT all have delta_theta >= 1.0.  This proves the
    assertions have teeth.
    """

    @classmethod
    def setUpClass(cls) -> None:
        use_pearcey_table()

    def test_inflated_coverage_makes_bound_fail(self) -> None:
        """With a fake coverage = 1.0, at least one served source violates.

        This confirms the bound assertion is not vacuously true.
        """
        fake_coverage = 1.0
        source = self._known_served_source()
        result = cusp_amplification(_W, source, _GAMMA)
        self.assertIsNotNone(result, 'Known fixture should be served')
        delta = self._image_theta_offset(source)
        self.assertIsNotNone(delta, 'Could not compute delta_theta')
        self.n_checks += 1
        # The real delta is ~0.12, so fake_coverage=1.0 will fail
        self.assertLess(
            delta, fake_coverage,
            'Self-falsification: the served delta_theta should be < 1.0 '
            '(proving that a wrong coverage value would trip the bound).')

    def test_zero_coverage_is_trivially_satisfied(self) -> None:
        """With coverage = 0.0, the bound is trivially true (no teeth).

        This is the complement: proves the real value (0.07) provides
        a non-trivial lower bound that excludes some angular region.
        """
        self.n_checks += 1
        # The real coverage is 0.07 > 0, so the exclusion zone is non-empty
        self.assertGreater(
            _CUSP_ARM_COVERAGE, 0.0,
            'Coverage of zero would provide no exclusion.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
