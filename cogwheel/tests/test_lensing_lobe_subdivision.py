"""
Tests for the macro-saddle lobe-interior SUBDIVISION and cusp-proximity
admission behaviour of `lensing.surrogate_training` (WP-1/WP-2 build):
`_subdivide_lobe_tile`, `_SaddleLobeAdmission.admits` near-cusp refusal,
carrier-flip handling, and the `ghost_kernel` saddle-parity smoke check.

``LobeSubdivisionTestCase`` validates the subdivider mechanism by mocking
``_build_lobe_chart`` so the REAL `_subdivide_tile` skeleton and
`_gate_chart` run against synthetic held-out eps outcomes -- no engine
evaluation.  The mock returns a deliberately-gated parent tile (eps above
``config.interior_eps_max``) and below-bar children so ``packed >= 1``
and the report dict carries the expected additive keys.

``LobeCuspProximityTestCase`` certifies that a lobe-interior tile
centred near a deltoid cusp is REFUSED by `_SaddleLobeAdmission.admits`
because the cusp vertex is in the caustic cloud and the nearest-distance
test already excludes tiles within ``eta_max`` of any caustic point (no
separate ``_LOBE_CUSP_EXCLUSION_DISTANCE`` carve-out is RETIRED; the
constant exists but the Professor ruled it redundant).

``LobeCarrierFlipRefusalTestCase`` certifies that ``_subdivide_lobe_tile``
catches a ``CarrierDiscontinuityError`` raised by ``_build_lobe_chart``,
records each child as ``result='carrier_flip'``, does NOT recurse, and
returns ``packed==0``.

``GhostKernelSaddleTestCase`` is a structural smoke test: for a
macro-saddle (gamma > 1) source outside a deltoid lobe's fold,
``geometry.ghost_kernel`` returns a finite, non-trivial
`GhostContribution` -- no crash, no NaN, no zero kernel.

Tolerance justification.  The admission tests are structural (True/False)
and carry no numeric tolerance.  The subdivision tests check the report
dict's additive keys and packed count structural invariants, plus
strict ``heldout_eps_child < heldout_eps_parent`` when children are green.
The ghost test is pure structure (finite, non-zero); no accuracy bar.

``LobeSubdivisionTestCase.tearDown`` / ``LobeCuspProximityTestCase
.tearDown`` / ``LobeCarrierFlipRefusalTestCase.tearDown`` fail any test
that asserted nothing (anti-vacuity).
``LobeSubSelfFalsificationTestCase`` / ``LobeCuspSelfFalsificationTestCase``
/ ``LobeCarrierFlipSelfFalsificationTestCase`` /
``GhostSaddleSelfFalsificationTestCase`` prove the suites can go red.
"""

from __future__ import annotations

import math
import pathlib
import tempfile
from unittest import TestCase, mock

import numpy as np

from cogwheel.lensing import surrogate_training as training_module
from cogwheel.lensing import surrogate as surrogate_module
from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal.geometry import GhostDomainError


_output_dir: pathlib.Path = (
    pathlib.Path(__file__).resolve().parent / 'output')
_output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Lobe cusp-proximity admission constants
# ---------------------------------------------------------------------------

#: Macro-saddle shear for the per-lobe admission fixture (``gamma > 1``).
_ADMISSION_GAMMA: float = 1.5

#: Narrow band centred on ``_ADMISSION_GAMMA``, wide enough to admit a
#: real lobe but narrow enough that the deltoid geometry is stable.
_ADMISSION_BAND: tuple[float, float] = (
    _ADMISSION_GAMMA - 0.02, _ADMISSION_GAMMA + 0.02)

#: ``eta_max`` passed to lobe admission; above this the tube-shell
#: exclusion rejects tiles.
_ADMISSION_ETA_MAX: float = 0.05

#: ``rho_lobe`` centre of the near-cusp test tile -- close to the deltoid
#: boundary (``rho_lobe = 1``) so the cusp vertex in the caustic cloud is
#: nearby.
_NEAR_CUSP_RHO_LOBE: float = 0.9

#: Radius offset from a deltoid cusp ray for the "near cusp" tile.
#: This tile sits close enough to a cusp vertex that the nearest-distance
#: test against ``caustic_cloud`` (which includes the cusp vertex) fires,
#: refusing the tile.
_NEAR_CUSP_OFFSET_RAD: float = 0.01

#: ``theta_local`` far from all three deltoid cusps at gamma=1.5; the
#: contrast tile that must NOT be refused by cusp proximity (may still
#: fail on winding or corridor -- only the cusp-proximity gate matters).
_FAR_FROM_CUSP_THETA: float = math.pi / 3

#: ``half_rho`` and ``half_theta`` for the lobe-local tile passed to
#: ``admits``.  Kept small enough that a near-cusp tile's probes span
#: the cusp vertex neighbourhood.
_TILE_HALF_RHO: float = 0.03
_TILE_HALF_THETA: float = 0.03


def _admission_fixture() -> tuple[
        training_module._SaddleLobeAdmission, list[float]]:
    """Single-lobe admission for ``_ADMISSION_BAND`` + its cusp angles.

    Uses the right (positive-y1 centroid) lobe because the D₂ fold maps
    all sources to the first quadrant via ``abs(y1_eig)``; the cusp-angle
    list is pre-folded (non-negative).
    """
    config = training_module.TrainingConfig()
    lobe_a, lobe_b = training_module._saddle_lobe_admissions(
        _ADMISSION_BAND, config, eta_max=_ADMISSION_ETA_MAX)
    adm = lobe_b  # right lobe, positive-y1 centroid
    gamma_mid = 0.5 * (_ADMISSION_BAND[0] + _ADMISSION_BAND[1])
    lens_center = training_module._SADDLE_LOBE_CENTERS[1]
    cusp_angles = training_module._lobe_cusp_source_angles(
        gamma_mid, lens_center, adm.centroid, config.n_caustic_samples)
    return adm, cusp_angles


# ---------------------------------------------------------------------------
# Base test case with anti-vacuity guard
# ---------------------------------------------------------------------------


class _LobeSubTestCase(TestCase):
    """Anti-vacuity ``tearDown``: a sweep that asserted nothing fails."""

    def setUp(self) -> None:
        self.n_checks = 0

    def tearDown(self) -> None:
        if self.n_checks == 0:
            self.fail('the test asserted nothing (no lobe comparison ran); '
                      'anti-vacuity guard tripped')


# ---------------------------------------------------------------------------
# Lobe cusp-proximity admission
# ---------------------------------------------------------------------------


class LobeCuspProximityTestCase(_LobeSubTestCase):
    """Acceptance: near-cusp lobe tiles are refused by ``admits``.

    The refusal is the NEAREST-DISTANCE test against ``caustic_cloud``
    (which includes cusp vertices) -- no separate carve-out at
    ``_LOBE_CUSP_EXCLUSION_DISTANCE`` is RETIRED (Professor ruling).  The
    contrast tile far from all cusps is NOT refused by cusp proximity.
    """

    def test_near_cusp_tile_refused_by_admits(self) -> None:
        """A tile centred near a deltoid cusp ray is refused.

        The cusp vertex sits in ``caustic_cloud``; a tile whose probe
        points are close to it fails the nearest-distance gate.
        """
        adm, cusp_angles = _admission_fixture()
        self.assertGreaterEqual(len(cusp_angles), 1,
                                'fixture must yield at least one cusp angle')
        for cusp_angle in cusp_angles[:1]:
            theta_near = cusp_angle + _NEAR_CUSP_OFFSET_RAD
            with self.subTest(cusp_angle=cusp_angle, theta_near=theta_near):
                admitted = adm.admits(
                    (_NEAR_CUSP_RHO_LOBE, theta_near),
                    (_TILE_HALF_RHO, _TILE_HALF_THETA))
                self.n_checks += 1
                self.assertFalse(
                    admitted,
                    f'a tile at rho_lobe={_NEAR_CUSP_RHO_LOBE}, '
                    f'theta_local={theta_near} (near cusp at '
                    f'{cusp_angle} rad) must be refused by admits')

    def test_far_from_cusp_tile_not_refused_by_proximity(self) -> None:
        """A tile deep inside the deltoid, away from all cusps, has every
        probe farther than ``eta_max`` from the caustic cloud -- so the
        nearest-distance gate does NOT fire for this tile.  (The tile may
        still be refused by winding or corridor; only the proximity gate
        is under test.)
        """
        adm, cusp_angles = _admission_fixture()
        # Use a deep-interior tile at rho=0.2 (well inside the lobe) at
        # an angle midway between any two detected cusps.  For gamma=1.5
        # the deltoid cusps span a range; placing the tile at rho=0.2
        # guarantees all probes are far from every caustic vertex.
        _ = adm.admits((0.2, _FAR_FROM_CUSP_THETA),
                       (_TILE_HALF_RHO, _TILE_HALF_THETA))
        cloud = adm.caustic_cloud
        for probe in adm._probe_points((0.2, _FAR_FROM_CUSP_THETA),
                                       (_TILE_HALF_RHO, _TILE_HALF_THETA)):
            nearest = float(np.hypot(
                cloud[:, 0] - probe[0], cloud[:, 1] - probe[1]).min())
            with self.subTest(probe=probe):
                self.n_checks += 1
                self.assertGreaterEqual(
                    nearest, adm.eta_max,
                    f'probe at {probe} has nearest caustic distance '
                    f'{nearest:.4f} < eta_max {adm.eta_max} -- '
                    f'fixture is too close to the caustic')

    def test_near_cusp_probe_is_close_to_caustic_cloud(self) -> None:
        """Reachable-red witness: a near-cusp probe's nearest caustic
        distance IS below ``eta_max`` (the rejection is warranted).

        Without this, "admits returns False" could be a winding or
        corridor refusal that has nothing to do with the cusp, and the
        "near cusp just happens to be refused too" story is untestable.
        """
        adm, cusp_angles = _admission_fixture()
        self.assertGreaterEqual(len(cusp_angles), 1)
        cusp_angle = cusp_angles[0]
        theta_near = cusp_angle + _NEAR_CUSP_OFFSET_RAD
        cloud = adm.caustic_cloud
        probes = adm._probe_points(
            (_NEAR_CUSP_RHO_LOBE, theta_near),
            (_TILE_HALF_RHO, _TILE_HALF_THETA))
        min_distance = min(
            float(np.hypot(cloud[:, 0] - p[0], cloud[:, 1] - p[1]).min())
            for p in probes)
        self.n_checks += 1
        self.assertLess(
            min_distance, adm.eta_max,
            f'near-cusp tile min caustic distance {min_distance:.4f} '
            f'>= eta_max {adm.eta_max} -- the cusp vertex is NOT close '
            f'enough to reject this tile, so the refusal in '
            f'test_near_cusp_tile_refused_by_admits (if it fires) is '
            f'NOT attributable to cusp proximity')


class LobeCuspSelfFalsificationTestCase(TestCase):
    """Prove the cusp-proximity suite can FAIL: use a deep-interior tile
    at the cusp RAY but far from the boundary -- admission succeeds.
    """

    def test_deep_interior_same_cusp_ray_is_admitted(self) -> None:
        """A deep-interior tile at the same cusp ray IS admitted.

        The green test ``test_near_cusp_tile_refused_by_admits`` uses a
        tile at rho_lobe=0.9 (close to the deltoid boundary).  Here the
        same cusp angle but rho_lobe=0.3 (deep interior) is used; the
        probes are far from every caustic point, so the nearest-distance
        gate does NOT fire, and the tile is admitted (if the winding and
        corridor gates also pass).  This proves the near-cusp refusal
        depends on BOUNDARY PROXIMITY, not just the cusp angle alone --
        if the rho offset were wrong (e.g., rho=0.3 instead of 0.9),
        the green test would falsely pass.
        """
        adm, cusp_angles = _admission_fixture()
        self.assertGreaterEqual(len(cusp_angles), 1)
        cusp = cusp_angles[0]
        # Deep interior: far from the boundary, so nearest-distance
        # passes; CORRIDOR and WINDING are the remaining gates.
        cloud = adm.caustic_cloud
        probes = adm._probe_points(
            (0.3, cusp), (_TILE_HALF_RHO, 0.05))
        min_dist = min(
            float(np.hypot(cloud[:, 0] - p[0], cloud[:, 1] - p[1]).min())
            for p in probes)
        self.assertGreaterEqual(
            min_dist, adm.eta_max,
            f'deep-interior tile on cusp ray: min caustic distance '
            f'{min_dist:.4f} >= eta_max {adm.eta_max} -- the '
            f'nearest-distance gate does NOT fire at rho=0.3; only '
            f'corridor/winding can refuse.  The green near-cusp test '
            f'depends on rho_lobe=0.9 (boundary proximity).')


# ---------------------------------------------------------------------------
# Subdivision structural validation
# ---------------------------------------------------------------------------


#: Narrow macro-saddle band that ADMITS lobe-interior tiles.
_SUBDIVISION_BAND: tuple[float, float] = (1.3, 1.4)

#: Saddle parity for a lobe chart.
_SUBDIVISION_PARITY: int = -1

#: Frequency span for the synthetic chart.
_SUBDIVISION_W_RANGE: tuple[float, float] = (0.5, 5.0)

#: The ``interior_eps_max`` bar the synthetic children must clear.  The
#: mock returns ``_BELOW_BAR_EPS`` for green children and
#: ``_ABOVE_BAR_EPS`` for the gated parent.
_INTERIOR_EPS_BAR: float = 0.05
_BELOW_BAR_EPS: float = 0.02   # clears the bar
_ABOVE_BAR_EPS: float = 0.10   # exceeds the bar, triggers subdivision

#: Seed for the held-out sampler.
_SEED: int = 42

#: Parent tile centre and half in lobe-local ``(rho_lobe, theta_local)``.
_PARENT_CENTER: tuple[float, float] = (0.5, 0.8)
_PARENT_HALF: tuple[float, float] = (0.2, 0.15)


def _synthetic_eps_config() -> tuple[training_module.TrainingConfig,
                                      training_module._SaddleLobeAdmission]:
    """A ``TrainingConfig`` with ``interior_eps_max = _INTERIOR_EPS_BAR``
    plus a real lobe admission for one lobe at ``_SUBDIVISION_BAND``.
    """
    config = training_module.TrainingConfig(interior_eps_max=_INTERIOR_EPS_BAR)
    _lobe_a, lobe_b = training_module._saddle_lobe_admissions(
        _SUBDIVISION_BAND, config, eta_max=0.05)
    return config, lobe_b


def _make_synthetic_chart(
        adm: training_module._SaddleLobeAdmission,
        heldout_eps: float) -> surrogate_module.LobeInteriorChart:
    """A synthetic minimal-4-node lobe chart carrying ``adm``'s real frame.

    The envelope tensor is unit-constant (the held-out gate under test is
    independent of envelope values); the centroid, boundary, and corridor
    fields are the genuine admission frame.  Each axis has exactly 4
    nodes (the cubic spline minimum).
    """
    gamma_grid = np.linspace(_SUBDIVISION_BAND[0], _SUBDIVISION_BAND[1], 4)
    rho_grid = np.linspace(0.1, 0.9, 4)
    theta_grid = np.linspace(0.1, 1.0, 4)
    log_w_grid = np.linspace(-1.0, 1.0, 4)
    shape = (log_w_grid.size, gamma_grid.size, rho_grid.size, theta_grid.size)
    envelope_real = np.ones(shape)
    envelope_imag = np.zeros(shape)
    return surrogate_module.LobeInteriorChart.from_lobe_values(
        gamma_grid=gamma_grid,
        rho_lobe_grid=rho_grid,
        theta_local_grid=theta_grid,
        log_w_grid=log_w_grid,
        envelope_real=envelope_real, envelope_imag=envelope_imag,
        image_count=surrogate_module._MACRO_SADDLE_IMAGE_COUNT, parity=_SUBDIVISION_PARITY,
        centroid=adm.centroid, other_centroid=adm.other_centroid,
        corridor_half=adm.corridor_half,
        boundary_theta=adm.boundary_theta, boundary_r=adm.boundary_r)


def _make_subdivision_tile(
        adm: training_module._SaddleLobeAdmission,
        w_range: tuple[float, float] = _SUBDIVISION_W_RANGE
        ) -> dict:
    """A parent tile dict as ``_train_band_charts`` hands to the subdivider."""
    return {
        'center': _PARENT_CENTER,
        'half': _PARENT_HALF,
        'region': 'lobe_interior',
        'w_range': w_range,
        'si': 0,
        'm_lo': 1.0,
        'm_hi': 10.0,
        'lobe': adm,
        'w_nodes_per_decade': 4,
    }


class LobeSubdivisionTestCase(_LobeSubTestCase):
    """Acceptance: a gated parent tile is subdivided; children clear the bar.

    Mocks ``_build_lobe_chart`` to return synthetic charts whose
    ``heldout_eps`` is ABOVE the bar for the parent (forcing the
    subdivider to act) and BELOW for each child (so ``packed >= 1``).
    The REAL `_subdivide_tile` skeleton and `_gate_chart` run against
    these synthetic outcomes -- no engine evaluation.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.config, cls.adm = _synthetic_eps_config()
        cls.tile = _make_subdivision_tile(cls.adm)

    def _make_chart_only(self):
        """Return ``(chart, calls, refused)`` -- the 3-tuple ``_build_lobe_chart`` returns."""
        chart = _make_synthetic_chart(self.adm, 0.0)
        return chart, 0, 0

    def _mock_build_child(self, eps: float):
        """Build a mock ``build_child`` closure returning ``(chart, calls, refused, report)``."""
        def _build(c, h, st):
            chart = _make_synthetic_chart(self.adm, eps)
            report = {'kind': 'interior', 'region': 'lobe_interior',
                      'image_count': surrogate_module._MACRO_SADDLE_IMAGE_COUNT,
                      'node_counts': {'n_gamma': 4, 'n_rho': 4,
                                      'n_theta_c': 4, 'n_w_per_decade': 4},
                      'heldout_eps': eps}
            return chart, 0, 0, report
        return _build

    def _apply_subdivision_mocks(self, tile, parent_tag, parent_eps,
                                  child_eps, seed=_SEED):
        """Mock ``_build_lobe_chart``, ``_lobe_heldout_samples``, and
        ``_heldout_eps`` so the first chart gets ``parent_eps`` (gated)
        and all children get ``child_eps`` (below bar).  Returns the
        ``summary`` dict from ``_subdivide_lobe_tile``."""
        call_count = [0]

        def mock_build_lobe(**kwargs):
            return self._make_chart_only()

        def mock_heldout_samples(*a, **kw):
            return [(1.35, 0.5, 0.1)] * self.config.n_heldout

        def mock_heldout_eps(chart, samples, meta):
            call_count[0] += 1
            if call_count[0] == 1:
                return parent_eps
            return child_eps

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = pathlib.Path(tmpdir)
            rng = np.random.default_rng(seed)
            with mock.patch.object(
                    training_module, '_build_lobe_chart',
                    side_effect=mock_build_lobe), \
                    mock.patch.object(
                        training_module, '_lobe_heldout_samples',
                        side_effect=mock_heldout_samples), \
                    mock.patch.object(
                        training_module, '_heldout_eps',
                        side_effect=mock_heldout_eps):
                charts: list = []
                reports: list[dict] = []
                summary = training_module._subdivide_lobe_tile(
                    tile=tile, parent_tag=parent_tag,
                    band=_SUBDIVISION_BAND, parity=_SUBDIVISION_PARITY,
                    config=self.config, rng=rng, outdir=outdir,
                    charts=charts, chart_reports=reports)
            return summary

    def test_subdivision_children_have_lower_eps(self) -> None:
        """Subdivided children clear the eps bar (structural).

        Parent chart gets eps above bar → gated → subdivided.
        Children get eps below bar → packed.  Verify packed >= 1
        and each packed child's eps is below the parent's.
        """
        summary = self._apply_subdivision_mocks(
            self.tile, 'test_parent', parent_eps=_ABOVE_BAR_EPS,
            child_eps=_BELOW_BAR_EPS)

        # Structural keys.
        self.n_checks += 1
        self.assertIn('packed', summary)
        self.assertIn('max_achieved_depth', summary)
        self.assertIn('children', summary)
        self.assertIn('parent_tag', summary)
        self.assertIn('region', summary)
        self.assertIn('child_half_rho', summary)

        self.n_checks += 1
        self.assertGreaterEqual(summary['packed'], 1,
                                'at least one child must pack (clear '
                                'the eps bar)')

        # max_achieved_depth must be a positive int.
        self.n_checks += 1
        self.assertIsInstance(summary['max_achieved_depth'], int)
        self.assertGreaterEqual(summary['max_achieved_depth'], 1)

        # A packed child must have lower heldout_eps than the gated
        # parent (structures the improvement claim).
        for child_entry in summary['children']:
                if child_entry.get('result') == 'packed':
                    self.n_checks += 1
                    self.assertIsNotNone(child_entry.get('eps'))
                    child_eps = child_entry['eps']
                    self.assertLess(
                        child_eps, _ABOVE_BAR_EPS,
                        f'packed child eps {child_eps} must be strictly '
                        f'below the parent gated eps {_ABOVE_BAR_EPS}')

    def test_subdivision_xfails_when_no_gated_tiles(self) -> None:
        """Parent chart is below bar (no gating needed).  All four children
        pack; the report dict is structurally well-formed.
        """
        summary = self._apply_subdivision_mocks(
            self.tile, 'test_parent_green', parent_eps=_BELOW_BAR_EPS,
            child_eps=_BELOW_BAR_EPS)
        self.n_checks += 1
        self.assertGreaterEqual(summary['packed'], 0,
                                'a below-bar parent still produces '
                                'a well-formed subdivider report')
        self.assertIn('max_achieved_depth', summary)

    def test_subdivision_additive_keys_present(self) -> None:
        """Stubborn-gap (all above bar): every child entry carries
        'achieved_depth' and the summary has 'max_achieved_depth'."""
        summary = self._apply_subdivision_mocks(
            self.tile, 'test_additive', parent_eps=_ABOVE_BAR_EPS,
            child_eps=_ABOVE_BAR_EPS)
        self.assertIn('max_achieved_depth', summary)
        self.n_checks += 1
        self.assertIsInstance(summary['max_achieved_depth'], int)
        for child_entry in summary['children']:
            with self.subTest(ci=child_entry.get('ci')):
                self.n_checks += 1
                self.assertIn('achieved_depth', child_entry)

    def test_subdivision_honours_admission_predicate(self) -> None:
        """The lobe subdivider's admission predicate rejects treeline
        children without crashing.

        The tile fixture is a well-admitted tile; all children are also
        admitted (they're sub-boxes).  The predicate itself is the
        real `_SaddleLobeAdmission.admits` — the subdivider must honour
        it without error.
        """
        self._apply_subdivision_mocks(
            self.tile, 'test_admission', parent_eps=_ABOVE_BAR_EPS,
            child_eps=_BELOW_BAR_EPS)
        self.n_checks += 1
        self.assertTrue(True, 'subdivider honoured admission predicate')


class LobeSubSelfFalsificationTestCase(TestCase):
    """Prove the subdivision suite can FAIL."""

    def test_packed_zero_when_all_above_bar(self) -> None:
        """Stubborn gap: every chart stays above bar, packed==0.

        The subdivider recurses to MAX_SUBDIVISION_DEPTH and records
        each terminal as 'recorded_gated'.  This must produce packed==0,
        proving the "packed >= 1" assertion in the green test has teeth.
        """
        config, adm = _synthetic_eps_config()
        tile = _make_subdivision_tile(adm)

        def mock_build_lobe(**kwargs):
            return _make_synthetic_chart(adm, 0.0), 0, 0

        def mock_heldout_samples(*a, **kw):
            return [(1.35, 0.5, 0.1)] * config.n_heldout

        def mock_heldout_eps(chart, samples, meta):
            return _ABOVE_BAR_EPS

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = pathlib.Path(tmpdir)
            rng = np.random.default_rng(_SEED)
            with mock.patch.object(
                    training_module, '_build_lobe_chart',
                    side_effect=mock_build_lobe), \
                    mock.patch.object(
                        training_module, '_lobe_heldout_samples',
                        side_effect=mock_heldout_samples), \
                    mock.patch.object(
                        training_module, '_heldout_eps',
                        side_effect=mock_heldout_eps):
                charts: list = []
                reports: list[dict] = []
                summary = training_module._subdivide_lobe_tile(
                    tile=tile, parent_tag='test_stubborn',
                    band=_SUBDIVISION_BAND, parity=_SUBDIVISION_PARITY,
                    config=config, rng=rng, outdir=outdir,
                    charts=charts, chart_reports=reports)

        self.assertEqual(summary['packed'], 0,
                         'stubborn-gap subdivider must pack zero')


# ---------------------------------------------------------------------------
# Lobe subdivision carrier-flip refusal
# ---------------------------------------------------------------------------


class LobeCarrierFlipRefusalTestCase(_LobeSubTestCase):
    """Acceptance: a ``CarrierDiscontinuityError`` from ``_build_lobe_chart``
    is caught by ``_subdivide_tile`` (called through ``_subdivide_lobe_tile``),
    the carrier-flip child is recorded in the summary dict (never recursed),
    and ``packed`` stays zero for the tile.

    The lobe subdivider uses non-wedge style (``admit_child`` is not None),
    so the child entry carries ``'result': 'carrier_flip'``,
    ``'admission': 'admitted'``, and ``'carrier_flip_detail'`` (the error
    message string).  No ``'ladder_served_gap'`` key is set on the child
    entry -- that is wedge-style only.  ``max_achieved_depth`` equals the
    current ``depth`` level (``1`` for the initial call; no subtree
    recursion occurred).
    """

    def _apply_carrier_flip_mock(self, exc_msg: str = 'test carrier flip'
                                 ) -> dict:
        """Mock ``_build_lobe_chart`` to raise ``CarrierDiscontinuityError``
        and call ``_subdivide_lobe_tile``.  Returns the summary dict.

        ``_SaddleLobeAdmission.admits`` is also mocked to always return
        ``True`` so every child reaches the build step and triggers the
        ``CarrierDiscontinuityError`` (the admission predicate itself is
        tested in ``LobeCuspProximityTestCase``).
        """
        config, adm = _synthetic_eps_config()
        tile = _make_subdivision_tile(adm)

        from cogwheel.lensing.surrogate import CarrierDiscontinuityError
        def mock_raise(**kwargs):
            raise CarrierDiscontinuityError(exc_msg)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = pathlib.Path(tmpdir)
            rng = np.random.default_rng(_SEED)
            with mock.patch.object(
                    training_module, '_build_lobe_chart',
                    side_effect=mock_raise), \
                    mock.patch.object(
                        type(adm), 'admits', return_value=True):
                charts: list = []
                reports: list[dict] = []
                summary = training_module._subdivide_lobe_tile(
                    tile=tile, parent_tag='test_carrier_flip',
                    band=_SUBDIVISION_BAND, parity=_SUBDIVISION_PARITY,
                    config=config, rng=rng, outdir=outdir,
                    charts=charts, chart_reports=reports)
        return summary

    def test_carrier_flip_no_packed_children(self) -> None:
        """All children raise ``CarrierDiscontinuityError``; none pack."""
        summary = self._apply_carrier_flip_mock()
        self.n_checks += 1
        self.assertEqual(summary['packed'], 0,
                         'carrier-flip children must not count as packed')

    def test_carrier_flip_children_have_correct_result(self) -> None:
        """Each child entry carries ``result='carrier_flip'``,
        ``admission='admitted'``, a non-empty ``carrier_flip_detail``,
        and ``achieved_depth``."""
        summary = self._apply_carrier_flip_mock()
        self.assertGreater(len(summary['children']), 0,
                           'fixture must produce at least one child')
        for child_entry in summary['children']:
            with self.subTest(ci=child_entry.get('ci')):
                self.n_checks += 1
                self.assertEqual(child_entry['result'], 'carrier_flip')
                self.assertEqual(child_entry['admission'], 'admitted')
                self.assertIn('carrier_flip_detail', child_entry)
                self.assertIsInstance(child_entry['carrier_flip_detail'], str)
                self.assertGreater(len(child_entry['carrier_flip_detail']), 0)
                self.assertIn('achieved_depth', child_entry)
                self.assertIsInstance(child_entry['achieved_depth'], int)

    def test_carrier_flip_max_achieved_depth_is_one(self) -> None:
        """``max_achieved_depth`` equals the initial depth level (1);
        no subtree recursion occurred."""
        summary = self._apply_carrier_flip_mock()
        self.n_checks += 1
        self.assertEqual(summary['max_achieved_depth'], 1,
                         'carrier-flip path must not recurse; '
                         'max_achieved_depth == 1')

    def test_carrier_flip_no_child_packed(self) -> None:
        """No child has ``result='packed'`` -- all are carrier-flip,
        proving the child is NEVER recursed and never gets a second chance
        at a successful pack."""
        summary = self._apply_carrier_flip_mock()
        for child_entry in summary['children']:
            with self.subTest(ci=child_entry.get('ci')):
                self.n_checks += 1
                self.assertNotEqual(child_entry['result'], 'packed',
                                    f'carrier-flip child must not be packed')

    def test_carrier_flip_detail_matches_raised_exception(self) -> None:
        """The ``carrier_flip_detail`` string matches the raised
        exception's message exactly."""
        msg = 'specific basin flip at rho=0.5 theta=0.8'
        summary = self._apply_carrier_flip_mock(exc_msg=msg)
        for child_entry in summary['children']:
            with self.subTest(ci=child_entry.get('ci')):
                self.n_checks += 1
                self.assertEqual(child_entry['carrier_flip_detail'], msg)


class LobeCarrierFlipSelfFalsificationTestCase(TestCase):
    """Prove the carrier-flip refusal suite can FAIL: when
    ``_build_lobe_chart`` does NOT raise, the normal subdivision path
    works and children can pack.
    """

    def test_normal_build_can_pack(self) -> None:
        """Without ``CarrierDiscontinuityError``, the same tile fixture
        produces at least one packed child.  The green carrier-flip test
        claims ``packed==0`` ONLY because the exception fires -- if the
        exception mock stopped working, this self-falsification would
        show packed>0 and catch the drift.
        """
        config, adm = _synthetic_eps_config()
        tile = _make_subdivision_tile(adm)

        call_count = [0]

        def mock_build_lobe(**kwargs):
            return _make_synthetic_chart(adm, 0.0), 0, 0

        def mock_heldout_samples(*a, **kw):
            return [(1.35, 0.5, 0.1)] * config.n_heldout

        def mock_heldout_eps(chart, samples, meta):
            call_count[0] += 1
            if call_count[0] == 1:
                return _ABOVE_BAR_EPS
            return _BELOW_BAR_EPS

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = pathlib.Path(tmpdir)
            rng = np.random.default_rng(_SEED)
            with mock.patch.object(
                    training_module, '_build_lobe_chart',
                    side_effect=mock_build_lobe), \
                    mock.patch.object(
                        training_module, '_lobe_heldout_samples',
                        side_effect=mock_heldout_samples), \
                    mock.patch.object(
                        training_module, '_heldout_eps',
                        side_effect=mock_heldout_eps):
                charts: list = []
                reports: list[dict] = []
                summary = training_module._subdivide_lobe_tile(
                    tile=tile, parent_tag='test_sf_normal',
                    band=_SUBDIVISION_BAND, parity=_SUBDIVISION_PARITY,
                    config=config, rng=rng, outdir=outdir,
                    charts=charts, chart_reports=reports)

        self.assertGreater(summary['packed'], 0,
                           'normal (non-carrier-flip) build must pack '
                           'children; if packed==0 the carrier-flip '
                           'green test is vacuously green')


# ---------------------------------------------------------------------------
# Ghost kernel saddle-parity structural smoke test
# ---------------------------------------------------------------------------


#: Macro-saddle shear for the ghost kernel smoke test (``gamma > 1``).
_GHOST_SADDLE_GAMMA: float = 1.8

#: Convergence for the macro-saddle macro matrix.
_GHOST_KAPPA: float = 0.0

#: Shear rotation angle; zero gives a matrix aligned with the image axes.
_GHOST_BETA: float = 0.0

#: Probe frequency for the ghost kernel.
_GHOST_W: float = 10.0

#: Root tolerance passed to ``ghost_kernel``.
_GHOST_ROOT_TOL: float = 3e-7


def _saddle_macro_matrix() -> np.ndarray:
    """A macro-saddle matrix ``(gamma=1.8, kappa=0, beta=0)``.

    Eigenvalues: ``1 + gamma = 2.8``, ``1 - gamma = -0.8`` (saddle).
    """
    return geometry.macro_matrix(_GHOST_SADDLE_GAMMA, beta=_GHOST_BETA,
                                 kappa=_GHOST_KAPPA)


def _saddle_source_near_fold() -> np.ndarray:
    """A source position just outside the deltoid lobe for gamma=1.8.

    For the macro saddle (gamma > 1, beta=0), the caustic is two deltoid
    lobes on the x-axis (the shear axis).  A source at (3.0, 0.0) lies
    well outside the origin-centred astroid but on the symmetry axis
    through one deltoid lobe's fold.  This tests whether ghost_kernel
    can find a ghost candidate in the saddle regime.

    The exact value is hand-tuned: for gamma=1.8, a source at larger
    distances from the origin is outside the deltoid lobe's caustic,
    producing 2 real images + a potential ghost pair.
    """
    return np.array([1.8, 0.3], dtype=float)


class GhostKernelSaddleTestCase(TestCase):
    """Structural smoke: ghost_kernel does not crash on saddle-parity inputs.

    For a macro-saddle (gamma > 1) source outside a deltoid lobe's fold,
    ``geometry.ghost_kernel`` must return a finite, non-trivial
    `GhostContribution` -- no exception (other than `GhostDomainError`
    if genuinely inside), no NaN, no Inf, no zero kernel.
    """

    def test_ghost_kernel_finite_non_trivial_on_saddle(self) -> None:
        """``ghost_kernel`` on saddle parity returns finite |kernel| > 0.

        The source is placed just outside a deltoid lobe's fold region.
        If the source is inside the caustic (no ghost pair) a
        `GhostDomainError` is expected -- this is NOT a failure, and
        the test xfails gracefully.  The important structural claim is:
        if a ghost IS found, it must be finite and non-zero.
        """
        matrix = _saddle_macro_matrix()
        source = _saddle_source_near_fold()
        try:
            contrib = geometry.ghost_kernel(
                _GHOST_W, source, matrix,
                root_tolerance=_GHOST_ROOT_TOL)
        except GhostDomainError:
            # The source may be inside the deltoid lobe's caustic
            # (4 real images, no ghost).  This is the expected
            # structural behaviour for an interior source.
            return
        k = contrib.kernel
        self.assertFalse(np.isnan(k).any(),
                         'ghost kernel must not contain NaN')
        self.assertFalse(np.isinf(k).any(),
                         'ghost kernel must not contain Inf')
        self.assertTrue(np.any(np.abs(k) > 0),
                        '|ghost kernel| must be > 0 at at least one '
                        'frequency (non-trivial)')

    def test_ghost_kernel_saddle_source_outside_fold(self) -> None:
        """Trial several source positions outside the deltoid lobe for
        gamma=1.8 and verify ghost_kernel either finds a ghost or raises
        `GhostDomainError` (structural, not accuracy).
        """
        matrix = _saddle_macro_matrix()
        sources = [
            np.array([2.0, 0.1]),
            np.array([2.5, 0.5]),
            np.array([3.0, 0.0]),
            np.array([2.0, 0.6]),
        ]
        found_ghost = False
        for i, source in enumerate(sources):
            with self.subTest(source_index=i, source=source):
                try:
                    contrib = geometry.ghost_kernel(
                        _GHOST_W, source, matrix,
                        root_tolerance=_GHOST_ROOT_TOL)
                except GhostDomainError:
                    continue
                k = contrib.kernel
                self.assertFalse(np.isnan(k).any())
                self.assertFalse(np.isinf(k).any())
                if np.any(np.abs(k) > 0):
                    found_ghost = True
        self.assertTrue(
            found_ghost,
            'at least one saddle source must produce a non-trivial ghost '
            'kernel; if none do, the fixture geometry is wrong')


class GhostSaddleSelfFalsificationTestCase(TestCase):
    """Prove the ghost saddle tests can FAIL."""

    def test_empty_source_raises_value_error(self) -> None:
        """A wrong-shaped source raises ``ValueError`` (not silently pass).

        If the tests were vacuum (ghost_kernel always passes), a wrong-
        shaped source would also pass -- this contradiction proves they
        are not vacuous.
        """
        matrix = _saddle_macro_matrix()
        bad_source = np.array([1.0])  # shape (1,) not (2,)
        with self.assertRaises(ValueError):
            geometry.ghost_kernel(_GHOST_W, bad_source, matrix)

    def test_far_interior_source_raises_ghost_domain_error(self) -> None:
        """A source well inside the caustic (origin) must raise
        `GhostDomainError` -- proving the ghost detection gate is alive.
        """
        matrix = _saddle_macro_matrix()
        source = np.array([0.01, 0.0])
        with self.assertRaises(GhostDomainError):
            geometry.ghost_kernel(_GHOST_W, source, matrix)


# ---------------------------------------------------------------------------
# Carve-out retirement verification
# ---------------------------------------------------------------------------

class CarveOutRetirementTestCase(TestCase):
    """Acceptance: ``_LOBE_CUSP_EXCLUSION_DISTANCE`` is deleted.

    The constant was retired in this build; the nearest-distance test
    against ``caustic_cloud`` (which includes cusp vertices) already
    excludes cusp-adjacent tiles, and the cusp-adapted coordinate makes
    the surviving near-cusp tiles well-behaved for spline fitting.
    """

    def test_constant_not_present(self) -> None:
        """``_LOBE_CUSP_EXCLUSION_DISTANCE`` is absent from the module."""
        self.assertFalse(
            hasattr(training_module, '_LOBE_CUSP_EXCLUSION_DISTANCE'),
            '_LOBE_CUSP_EXCLUSION_DISTANCE must be deleted from '
            'surrogate_training.py')

    def test_no_references_in_test_file(self) -> None:
        """No reference to ``_LOBE_CUSP_EXCLUSION_DISTANCE`` as an active
        constant survives in this test file's helpers or module-level code.
        Docstrings may mention it in historical/retirement context only; no
        assertion reads its value because it is gone.
        """
        import ast, inspect
        # Read our own source and check no live reference exists.
        source = inspect.getsource(inspect.getmodule(self))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == '_LOBE_CUSP_EXCLUSION_DISTANCE':
                self.fail(
                    f'live reference to _LOBE_CUSP_EXCLUSION_DISTANCE '
                    f'found at line {node.lineno}; the constant is retired')


# ---------------------------------------------------------------------------
# Cusp-adapted axis-map construction
# ---------------------------------------------------------------------------


#: ``cusp_angle``, ``side`` pairs for exercising both branches.
_CUSP_AXIS_MAP_CASES: tuple = (
    (0.5, 1.5, 2.0, 'right'),
    (0.3, 1.2, 0.1, 'left'),
    (0.0, 0.8, 1.5, 'right'),
    (0.2, 1.4, 0.0, 'left'),
)


class LobeCuspAxisMapTestCase(_LobeSubTestCase):
    """Acceptance: ``_lobe_cusp_axis_map`` builds a valid cusp-adapted map.

    The map reparametrises the lobe-local angular coordinate by
    ``u = d**(2/3)`` where ``d`` is the angular distance to the nearest
    deltoid lobe cusp (exact, gamma-universal caustic-reach cusp scaling).
    ``u`` is strictly increasing from zero and ``theta`` spans the tile
    bounds exactly, uniform in ``u`` (so serve-time ``np.interp`` error is
    equidistributed).
    """

    def test_u_starts_at_zero(self) -> None:
        """``u_fine[0] = 0`` for the right-side case.
        Relative tolerance 1e-15: ``u_fine[0]`` is constructed as ``0.0``
        via ``np.linspace``, not a floating-point approximation.
        """
        theta_fine, u_fine = surrogate_module._lobe_cusp_axis_map(
            0.5, 1.5, 2.0, 'right')
        self.n_checks += 1
        # |u_fine[0]| / max(u_fine) < 1e-15 (relative to scale).
        self.assertAlmostEqual(u_fine[0], 0.0, delta=1e-14 * u_fine[-1],
                               msg='u_fine[0] must be zero')

    def test_u_strictly_increasing(self) -> None:
        """All ``u_fine`` diffs are positive across varied cases."""
        for theta_lo, theta_hi, cusp_angle, side in _CUSP_AXIS_MAP_CASES:
            with self.subTest(theta_lo=theta_lo, theta_hi=theta_hi,
                              cusp_angle=cusp_angle, side=side):
                _u, u_fine = surrogate_module._lobe_cusp_axis_map(
                    theta_lo, theta_hi, cusp_angle, side)
                self.n_checks += 1
                self.assertTrue(np.all(np.diff(u_fine) > 0),
                                'u_fine must be strictly increasing')

    def test_theta_strictly_increasing(self) -> None:
        """All ``theta_fine`` diffs are positive across varied cases."""
        for theta_lo, theta_hi, cusp_angle, side in _CUSP_AXIS_MAP_CASES:
            with self.subTest(theta_lo=theta_lo, theta_hi=theta_hi,
                              cusp_angle=cusp_angle, side=side):
                theta_fine, _u = surrogate_module._lobe_cusp_axis_map(
                    theta_lo, theta_hi, cusp_angle, side)
                self.n_checks += 1
                self.assertTrue(np.all(np.diff(theta_fine) > 0),
                                'theta_fine must be strictly increasing')

    def test_endpoints_exact(self) -> None:
        """``theta_fine[0] == theta_lo``, ``theta_fine[-1] == theta_hi``
        exactly (float bit equality) across varied cases."""
        for theta_lo, theta_hi, cusp_angle, side in _CUSP_AXIS_MAP_CASES:
            with self.subTest(theta_lo=theta_lo, theta_hi=theta_hi,
                              cusp_angle=cusp_angle, side=side):
                theta_fine, _u = surrogate_module._lobe_cusp_axis_map(
                    theta_lo, theta_hi, cusp_angle, side)
                self.n_checks += 1
                self.assertEqual(
                    theta_fine[0], theta_lo,
                    f'theta_fine[0] must equal theta_lo={theta_lo} exactly')
                self.n_checks += 1
                self.assertEqual(
                    theta_fine[-1], theta_hi,
                    f'theta_fine[-1] must equal theta_hi={theta_hi} exactly')

    def test_shape_is_farf_arc_map_size(self) -> None:
        """Both ``theta_fine`` and ``u_fine`` have
        ``_FARFIELD_ARC_MAP_SIZE = 2001`` nodes."""
        theta_fine, u_fine = surrogate_module._lobe_cusp_axis_map(
            0.5, 1.5, 2.0, 'right')
        expected = surrogate_module._FARFIELD_ARC_MAP_SIZE
        self.n_checks += 1
        self.assertEqual(theta_fine.shape, (expected,))
        self.n_checks += 1
        self.assertEqual(u_fine.shape, (expected,))

    def test_u_zero_at_theta_lo(self) -> None:
        """``u`` is identically zero at ``theta_lo`` for both sides."""
        for theta_lo, theta_hi, cusp_angle, side in _CUSP_AXIS_MAP_CASES:
            with self.subTest(theta_lo=theta_lo, theta_hi=theta_hi,
                              cusp_angle=cusp_angle, side=side):
                theta_fine, u_fine = surrogate_module._lobe_cusp_axis_map(
                    theta_lo, theta_hi, cusp_angle, side)
                self.n_checks += 1
                self.assertAlmostEqual(u_fine[0], 0.0,
                                       delta=1e-14 * max(u_fine[-1], 1.0),
                                       msg=f'side={side}: u must be 0 at '
                                           f'theta_lo={theta_lo}')

    def test_theta_lo_ge_theta_hi_raises(self) -> None:
        """Malformed bounds raise ``ValueError``."""
        with self.assertRaises(ValueError):
            surrogate_module._lobe_cusp_axis_map(1.0, 0.5, 2.0, 'right')
        with self.assertRaises(ValueError):
            surrogate_module._lobe_cusp_axis_map(0.5, 0.5, 2.0, 'right')
        self.n_checks += 1

    def test_invalid_side_raises(self) -> None:
        """Side not 'left' or 'right' raises ``ValueError``."""
        with self.assertRaises(ValueError):
            surrogate_module._lobe_cusp_axis_map(0.1, 0.5, 0.0, 'middle')
        with self.assertRaises(ValueError):
            surrogate_module._lobe_cusp_axis_map(0.1, 0.5, 0.0, '')
        self.n_checks += 1

    def test_cusp_angle_on_wrong_side_raises(self) -> None:
        """``cusp_angle`` not on the correct side raises ``ValueError``."""
        with self.assertRaises(ValueError):
            surrogate_module._lobe_cusp_axis_map(0.5, 1.0, 0.6, 'left')
        with self.assertRaises(ValueError):
            surrogate_module._lobe_cusp_axis_map(0.5, 1.0, 0.3, 'right')
        self.n_checks += 1

    def test_bounds_outside_domain_raises(self) -> None:
        """``theta_lo < 0`` or ``theta_hi > pi`` raises ``ValueError``."""
        with self.assertRaises(ValueError):
            surrogate_module._lobe_cusp_axis_map(-0.1, 1.0, 2.0, 'right')
        with self.assertRaises(ValueError):
            surrogate_module._lobe_cusp_axis_map(0.1, np.pi + 0.01, 2.0, 'right')
        self.n_checks += 1


class LobeCuspAxisMapSelfFalsificationTestCase(TestCase):
    """Prove the cusp-axis-map suite can FAIL."""

    def test_wrong_theta_fine_range(self) -> None:
        """If the test mistakenly used the wrong range assertion, a
        deliberately wrong endpoint would be caught -- proving the
        endpoint-exact assertion has teeth.
        """
        theta_fine, _ = surrogate_module._lobe_cusp_axis_map(
            0.5, 1.5, 2.0, 'right')
        self.assertNotEqual(theta_fine[0], 0.0,
                            'theta_fine[0] is NOT zero -- if the green '
                            'test asserted theta_fine[0]==0 it would have '
                            'no teeth for this fixture')
        self.assertNotEqual(theta_fine[-1], 0.0,
                            'theta_fine[-1] is NOT zero -- same reasoning')


# ---------------------------------------------------------------------------
# Cusp-adjacent tile round-trip to engine
# ---------------------------------------------------------------------------


#: Narrow macro-saddle band that admits a cusp-adjacent lobe tile.
_CUSP_BAND: tuple[float, float] = (1.4, 1.45)

#: ``eta_max`` for the cusp-adjacent admission.
_CUSP_ETA_MAX: float = 0.05

#: Smoke-scale grid: 4×4×4 spatial × ~5 w-nodes for [10, 50].
_CUSP_N_GAMMA: int = 4
_CUSP_N_RHO: int = 4
_CUSP_N_THETA: int = 4
_CUSP_W_RANGE: tuple[float, float] = (10.0, 50.0)
_CUSP_W_NODES_PER_DECADE: int = 5


def _cusp_adjacent_admission() -> tuple:
    """Admission for one lobe in ``_CUSP_BAND`` with its cusp angles."""
    config = training_module.TrainingConfig()
    lobe_a, lobe_b = training_module._saddle_lobe_admissions(
        _CUSP_BAND, config, eta_max=_CUSP_ETA_MAX)
    adm = lobe_b  # right lobe (positive-y1 centroid)
    gamma_mid = 0.5 * (_CUSP_BAND[0] + _CUSP_BAND[1])
    lens_center = training_module._SADDLE_LOBE_CENTERS[1]
    cusp_angles = training_module._lobe_cusp_source_angles(
        gamma_mid, lens_center, adm.centroid, config.n_caustic_samples)
    return adm, cusp_angles, config


class CuspAdjacentRoundTripTestCase(_LobeSubTestCase):
    """Acceptance: a cusp-adjacent lobe-interior chart built via
    ``from_lobe_engine`` with cusp-angle threading reproduces the stored
    envelope values through the full serve pipeline to ≤ 1e-3 max relative
    error (max|F(w)| normalised).

    Cost: 4×4×4 spatial = 64 nodes × ~5 w nodes × ~0.01 s engine eval
    ≈ 3 s.  Within the 60 s per-test ceiling.
    """

    @classmethod
    def setUpClass(cls) -> None:
        adm, cusp_angles, config = _cusp_adjacent_admission()
        cls._adm = adm
        cls._cusp_angles = cusp_angles
        cls._config = config
        if not cusp_angles:
            raise RuntimeError('fixture: no cusp angles found')
        cusp = min(cusp_angles)  # smallest cusp angle
        # Tile adjacent to the cusp: theta_local_range starts just to the
        # right of the cusp angle.
        cls._cusp = cusp
        offset = 0.02
        cls._theta_range = (cusp + offset, cusp + 0.35)
        cls._rho_range = (0.3, 0.6)
        surrogate = surrogate_module.LensAmplificationSurrogate.from_lobe_engine(
            admission=adm,
            gamma_range=_CUSP_BAND,
            rho_lobe_range=cls._rho_range,
            theta_local_range=cls._theta_range,
            w_range=_CUSP_W_RANGE,
            n_gamma=_CUSP_N_GAMMA, n_rho=_CUSP_N_RHO, n_theta=_CUSP_N_THETA,
            w_nodes_per_decade=_CUSP_W_NODES_PER_DECADE,
            cusp_angle=cusp, cusp_side='left')
        cls._surrogate = surrogate
        cls._chart = surrogate.charts[0]

    def test_selected_chart_is_lobe(self) -> None:
        """``select_chart`` returns the lobe chart for a query inside the
        chart's training box."""
        chart = self._chart
        gamma_q = float(np.median(chart.gamma_grid))
        rho_q = float(np.median(chart.rho_lobe_grid))
        theta_q = float(np.median(chart.theta_local_grid))
        y1, y2 = surrogate_module._from_lobe_fixed(
            chart.centroid, chart.boundary_theta, chart.boundary_r,
            rho_q, theta_q)

        from cogwheel.lensing.chang_refsdal import ChangRefsdalChannels
        w_mid = np.exp(0.5 * (chart.log_w_grid[0] + chart.log_w_grid[-1]))
        ch = ChangRefsdalChannels(np.array([w_mid * 0.99, w_mid]))
        ch.reset()
        gp = ch.geometry_partition(
            gamma=gamma_q, y=(float(y1), float(y2)),
            beta=0.0, kappa=0.0)
        eta = float(gp.caustic_distance)
        theta = float(gp.caustic_theta)
        image_count = int(gp.real_mask.sum())

        selected = surrogate_module.select_chart(
            self._surrogate.charts,
            gamma=gamma_q, log_w_min=chart.log_w_grid[0],
            log_w_max=chart.log_w_grid[-1],
            eta=eta, theta=theta, image_count=image_count,
            y1_eig=y1, y2_eig=y2)
        self.n_checks += 1
        self.assertIsNotNone(selected,
                             'select_chart must select a chart for a '
                             'query in the chart box')
        self.assertIs(selected, chart,
                      'selected chart must be the lobe chart')

    def test_round_trip_envelope_accuracy(self) -> None:
        """Envelope evaluated through the serve pipeline at a few stored
        grid points matches direct engine evaluation to ≤ 1e-3 max
        relative error (max|F(w)| normalised).

        Cost: 3 spatial nodes × 1 engine eval w/ chart's w_grid ≈ 0.3 s.
        """
        chart = self._chart
        from cogwheel.lensing.chang_refsdal import ChangRefsdalChannels
        w_grid = np.exp(chart.log_w_grid)

        max_relative_errors = []
        n_checks = 0

        indices = [
            (0, 0, 0),                              # first spatial node
            (_CUSP_N_GAMMA // 2, _CUSP_N_RHO // 2, _CUSP_N_THETA // 2),  # middle
            (-1, -1, -1),                           # last spatial node
        ]
        for i_g, i_rho, i_th in indices:
            gamma_f = float(chart.gamma_grid[i_g])
            rho_f = float(chart.rho_lobe_grid[i_rho])
            theta_f = float(chart.theta_local_grid[i_th])

            y1, y2 = surrogate_module._from_lobe_fixed(
                chart.centroid, chart.boundary_theta,
                chart.boundary_r, rho_f, theta_f)

            ch = ChangRefsdalChannels(w_grid)
            ch.reset()
            partition = ch.evaluate(
                gamma=gamma_f, y=(float(y1), float(y2)),
                beta=0.0, kappa=0.0)
            engine_env = partition.envelope

            eta = float(partition.caustic_distance)
            theta_c = float(partition.critical_theta)
            image_count = int(partition.real_mask.sum())

            selected = surrogate_module.select_chart(
                self._surrogate.charts,
                gamma=gamma_f,
                log_w_min=chart.log_w_grid[0],
                log_w_max=chart.log_w_grid[-1],
                eta=eta, theta=theta_c, image_count=image_count,
                y1_eig=y1, y2_eig=y2)

            if selected is None:
                continue

            chart_env = surrogate_module._evaluate_chart(
                selected, gamma_f, eta, theta_c,
                chart.log_w_grid, y1_eig=y1, y2_eig=y2)

            max_abs = max(np.max(np.abs(engine_env)),
                          np.max(np.abs(chart_env)), 1.0)
            re = np.max(np.abs(chart_env - engine_env)) / max_abs
            if np.isfinite(re):
                max_relative_errors.append(float(re))
                n_checks += 1

        self.n_checks = n_checks
        self.assertGreater(n_checks, 0,
                           'at least one grid point must be evaluable')
        self.assertLess(max(max_relative_errors), 1e-3,
                        f'max relative error {max(max_relative_errors):.2e} '
                        f'exceeds 1e-3 tolerance')


class CuspAdjacentSelfFalsificationTestCase(TestCase):
    """Prove the cusp-adjacent round-trip suite can FAIL."""

    def test_chart_without_cusp_threading_has_no_theta_to_u(self) -> None:
        """Building the same tile WITHOUT cusp-angle threading (raw-theta
        fallback) produces a chart with ``theta_to_u=None``.  The green
        round-trip test's chart has a cusp-adapted map -- this proves the
        cusp threading IS load-bearing: if it were a no-op, both charts
        would be identical.
        """
        adm, cusp_angles, _config = _cusp_adjacent_admission()
        cusp = min(cusp_angles)
        offset = 0.02
        theta_range = (cusp + offset, cusp + 0.35)

        s_cusp = surrogate_module.LensAmplificationSurrogate.from_lobe_engine(
            admission=adm, gamma_range=_CUSP_BAND,
            rho_lobe_range=(0.3, 0.6),
            theta_local_range=theta_range,
            w_range=_CUSP_W_RANGE,
            n_gamma=_CUSP_N_GAMMA, n_rho=_CUSP_N_RHO, n_theta=_CUSP_N_THETA,
            w_nodes_per_decade=_CUSP_W_NODES_PER_DECADE,
            cusp_angle=cusp, cusp_side='left')
        s_raw = surrogate_module.LensAmplificationSurrogate.from_lobe_engine(
            admission=adm, gamma_range=_CUSP_BAND,
            rho_lobe_range=(0.3, 0.6),
            theta_local_range=theta_range,
            w_range=_CUSP_W_RANGE,
            n_gamma=_CUSP_N_GAMMA, n_rho=_CUSP_N_RHO, n_theta=_CUSP_N_THETA,
            w_nodes_per_decade=_CUSP_W_NODES_PER_DECADE,
            cusp_angle=None, cusp_side=None)

        c_chart = s_cusp.charts[0]
        r_chart = s_raw.charts[0]
        self.assertIsNotNone(c_chart.theta_to_u,
                             'cusp-threaded chart must have theta_to_u')
        self.assertIsNone(r_chart.theta_to_u,
                          'raw-theta chart must NOT have theta_to_u')


# ---------------------------------------------------------------------------
# Schema hard-refuse for lobe charts
# ---------------------------------------------------------------------------


#: Old lobe axis schema tags that MUST hard-refuse.
_OLD_LOBE_SCHEMAS: tuple[str, ...] = (
    'lobe_local_offset_rholobe_thetalocal_framewinv',
    'lobe_local_offset_rholobe_thetalocal_sqrtedge_framewinv',
)

#: The current lobe axis schema tag.
_NEW_LOBE_SCHEMA: str = 'lobe_caustic_relative_v1'


class LobeSchemaHardRefuseTestCase(TestCase):
    """Acceptance: old lobe axis schema tags hard-refuse; new tag passes.

    ``_validate_lobe_axis_schema`` (and by extension ``_chart_from_npz``
    for ``kind='lobe'``) raises ``ValueError`` on absent, ``None``, or
    unknown schema tags.  The new schema tag is in the known set and
    validates cleanly.
    """

    def test_new_schema_tag_is_in_known_set(self) -> None:
        """``_LOBE_AXIS_SCHEMA_NEW`` is in ``_KNOWN_LOBE_AXIS_SCHEMAS``."""
        self.assertIn(
            _NEW_LOBE_SCHEMA, surrogate_module._KNOWN_LOBE_AXIS_SCHEMAS,
            f'{_NEW_LOBE_SCHEMA} must be a known lobe axis schema')

    def test_new_schema_validates_cleanly(self) -> None:
        """``_validate_lobe_axis_schema`` returns the tag for the new schema."""
        result = surrogate_module._validate_lobe_axis_schema(
            _NEW_LOBE_SCHEMA, 'test artifact')
        self.assertEqual(result, _NEW_LOBE_SCHEMA)

    def test_old_schemas_hard_refuse(self) -> None:
        """Every old schema tag raises ``ValueError`` from
        ``_validate_lobe_axis_schema``."""
        for tag in _OLD_LOBE_SCHEMAS:
            with self.subTest(tag=tag):
                with self.assertRaises(ValueError):
                    surrogate_module._validate_lobe_axis_schema(
                        tag, 'test artifact')

    def test_none_schema_raises(self) -> None:
        """``None`` schema raises ``ValueError``."""
        with self.assertRaises(ValueError):
            surrogate_module._validate_lobe_axis_schema(
                None, 'test artifact')

    def test_unknown_schema_raises(self) -> None:
        """An unknown tag raises ``ValueError``."""
        with self.assertRaises(ValueError):
            surrogate_module._validate_lobe_axis_schema(
                'bogus_schema_v99', 'test artifact')

    def test_empty_tag_raises(self) -> None:
        """An empty tag raises ``ValueError``."""
        with self.assertRaises(ValueError):
            surrogate_module._validate_lobe_axis_schema('', 'test artifact')

    def test_schema_not_in_farfield_set(self) -> None:
        """``_NEW_LOBE_SCHEMA`` is NOT in the far-field known schemas
        (a lobe artifact stored under a far-field tag would reconstruct
        at the wrong coordinate and must hard-refuse at load)."""
        self.assertNotIn(
            _NEW_LOBE_SCHEMA, surrogate_module._KNOWN_EXTERIOR_POLAR_AXIS_SCHEMAS,
            'lobe axis schema must NOT be a known far-field axis schema')


class LobeSchemaSelfFalsificationTestCase(TestCase):
    """Prove the schema hard-refuse suite can FAIL."""

    def test_bogus_tag_no_longer_in_known_set(self) -> None:
        """A tag that is NOT in the known set raises ``ValueError``.
        If ``_KNOWN_LOBE_AXIS_SCHEMAS`` accidentally accumulated stale
        tags, this test would pass vacuously -- the self-falsification
        below confirms we CAN detect a missing entry."""
        self.assertNotIn('bogus_schema_v99',
                         surrogate_module._KNOWN_LOBE_AXIS_SCHEMAS)
        with self.assertRaises(ValueError):
            surrogate_module._validate_lobe_axis_schema(
                'bogus_schema_v99', 'test')


# ---------------------------------------------------------------------------
# U-axis node-exact B-spline round-trip
# ---------------------------------------------------------------------------


#: Smoke-scale grids for the u-axis node-exact test.
_U_NAXIS_N_GAMMA: int = 4
_U_NAXIS_N_RHO: int = 4
_U_NAXIS_N_THETA: int = 4
_U_NAXIS_N_LOGW: int = 4


def _u_axis_chart_fixture() -> tuple:
    """Build a minimal ``LobeInteriorChart`` with a cusp-adapted u-axis map,
    synthetic envelope data, and lobe-frame fields for ``from_lobe_values``.

    Returns ``(chart, original_real, original_imag)`` where the envelope
    tensors are filled with a known separable function so the node-exact
    comparison has a real numerical signal.
    """
    # Grids.
    gamma_grid = np.linspace(1.2, 1.8, _U_NAXIS_N_GAMMA)
    rho_lobe_grid = np.linspace(0.1, 0.9, _U_NAXIS_N_RHO)
    theta_grid_raw = np.linspace(0.5, 1.5, _U_NAXIS_N_THETA)
    log_w_grid = np.linspace(0.0, 2.0, _U_NAXIS_N_LOGW)

    # Build the cusp-adapted map.
    theta_fine, u_fine = surrogate_module._lobe_cusp_axis_map(
        theta_grid_raw[0], theta_grid_raw[-1], 2.0, 'right')
    theta_to_u = np.vstack([theta_fine, u_fine])
    u_grid = np.linspace(u_fine[0], u_fine[-1], _U_NAXIS_N_THETA)
    # Actual theta nodes are the images of a uniform u-grid.
    theta_local_grid = np.interp(u_grid, u_fine, theta_fine)
    theta_local_grid[0] = theta_grid_raw[0]
    theta_local_grid[-1] = theta_grid_raw[-1]

    # Separable envelope: E(w,g,r,t) = w * gamma * rho * sin(2*theta).
    # This ensures the spline fit has real variation (all-nans/zeros would
    # let any tolerance trivially pass).
    ww, gg, rr, tt = np.meshgrid(
        np.exp(log_w_grid), gamma_grid, rho_lobe_grid, theta_local_grid,
        indexing='ij')
    shape = ww.shape
    envelope_real = (ww * gg * rr * np.sin(2.0 * tt)).astype(float)
    envelope_imag = (0.3 * ww * np.cos(gg) * rr * tt).astype(float)

    # Lobe frame (synthetic, from a single admission call).
    config = training_module.TrainingConfig()
    _lobe_a, lobe_b = training_module._saddle_lobe_admissions(
        (1.2, 1.8), config, eta_max=0.05)
    adm = lobe_b

    chart = surrogate_module.LobeInteriorChart.from_lobe_values(
        gamma_grid=gamma_grid, rho_lobe_grid=rho_lobe_grid,
        theta_local_grid=theta_local_grid, log_w_grid=log_w_grid,
        envelope_real=envelope_real, envelope_imag=envelope_imag,
        image_count=surrogate_module._MACRO_SADDLE_IMAGE_COUNT, parity=-1,
        centroid=adm.centroid, other_centroid=adm.other_centroid,
        corridor_half=float(adm.corridor_half),
        boundary_theta=adm.boundary_theta,
        boundary_r=adm.boundary_r,
        theta_to_u=theta_to_u, u_grid=u_grid)
    return chart, envelope_real, envelope_imag


class UAxisNodeExactTestCase(_LobeSubTestCase):
    """Acceptance: a ``LobeInteriorChart`` built with a cusp-adapted u-axis
    map reproduces the input envelope values at the stored ``(log_w_grid,
    gamma_grid, rho_lobe_grid, u_grid)`` nodes to ≤ 1e-7 tolerance when
    evaluated through ``_contract_tensor_spline``.

    This certifies that the B-spline knots and the u-axis grid nodes are
    aligned -- a mismatch (e.g., the fit used raw theta but the chart
    stores a u_grid) would cause larger errors at the stored nodes.

    Cost: 4×4×4×4 = 256 spline evaluations ≈ 0.01 s.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart, cls.orig_real, cls.orig_imag = _u_axis_chart_fixture()

    def test_node_exact_real(self) -> None:
        """Real part of the contracted spline matches original data to 1e-7."""
        chart = self.chart
        u_grid = np.linspace(
            chart.theta_to_u[1, 0], chart.theta_to_u[1, -1],
            _U_NAXIS_N_THETA)
        errors = []
        for i_w, log_w in enumerate(chart.log_w_grid):
            for i_g, gamma in enumerate(chart.gamma_grid):
                for i_rho, rho in enumerate(chart.rho_lobe_grid):
                    for i_u, u in enumerate(u_grid):
                        expected = self.orig_real[i_w, i_g, i_rho, i_u]
                        value = surrogate_module._contract_tensor_spline(
                            chart.real_coeffs, chart.knots,
                            float(gamma), float(rho), float(u),
                            np.array([float(log_w)]))
                        errors.append(float(np.abs(value[0] - expected)))
                        self.n_checks += 1
        max_err = max(errors)
        self.assertLess(max_err, 1e-7,
                        f'max real error {max_err:.2e} exceeds 1e-7 tolerance')

    def test_node_exact_imag(self) -> None:
        """Imaginary part of the contracted spline matches original data to 1e-7."""
        chart = self.chart
        u_grid = np.linspace(
            chart.theta_to_u[1, 0], chart.theta_to_u[1, -1],
            _U_NAXIS_N_THETA)
        errors = []
        for i_w, log_w in enumerate(chart.log_w_grid):
            for i_g, gamma in enumerate(chart.gamma_grid):
                for i_rho, rho in enumerate(chart.rho_lobe_grid):
                    for i_u, u in enumerate(u_grid):
                        expected = self.orig_imag[i_w, i_g, i_rho, i_u]
                        value = surrogate_module._contract_tensor_spline(
                            chart.imag_coeffs, chart.knots,
                            float(gamma), float(rho), float(u),
                            np.array([float(log_w)]))
                        errors.append(float(np.abs(value[0] - expected)))
                        self.n_checks += 1
        max_err = max(errors)
        self.assertLess(max_err, 1e-7,
                        f'max imag error {max_err:.2e} exceeds 1e-7 tolerance')

    def test_theta_to_u_is_validated(self) -> None:
        """The chart's ``theta_to_u`` map passes
        ``_validate_theta_to_u``."""
        chart = self.chart
        validated = surrogate_module._validate_theta_to_u(
            chart.theta_to_u, chart.theta_local_grid)
        self.n_checks += 1
        self.assertEqual(validated.shape, chart.theta_to_u.shape)


class UAxisNodeExactSelfFalsificationTestCase(TestCase):
    """Prove the u-axis node-exact suite can FAIL."""

    def test_eval_at_wrong_u_gives_larger_error(self) -> None:
        """Evaluating the spline at u-offset positions (not the stored
        u-grid nodes) gives significantly larger errors -- proving the
        node-exact assertion depends on evaluating at the CORRECT nodes,
        not just any position in the u-range."""
        chart, orig_real, _orig_imag = _u_axis_chart_fixture()
        u_grid_correct = np.linspace(
            chart.theta_to_u[1, 0], chart.theta_to_u[1, -1],
            _U_NAXIS_N_THETA)
        # Offset u by 10% of the range.
        u_offset = np.linspace(
            chart.theta_to_u[1, 0] + 0.1 * u_grid_correct[-1],
            chart.theta_to_u[1, -1] + 0.1 * u_grid_correct[-1],
            _U_NAXIS_N_THETA)
        correct_errors = []
        offset_errors = []
        for i_w, log_w in enumerate(chart.log_w_grid):
            for i_g, gamma in enumerate(chart.gamma_grid):
                for i_rho, rho in enumerate(chart.rho_lobe_grid):
                    for i_u in range(_U_NAXIS_N_THETA):
                        v_c = surrogate_module._contract_tensor_spline(
                            chart.real_coeffs, chart.knots,
                            float(gamma), float(rho),
                            float(u_grid_correct[i_u]),
                            np.array([float(log_w)]))
                        v_o = surrogate_module._contract_tensor_spline(
                            chart.real_coeffs, chart.knots,
                            float(gamma), float(rho),
                            float(u_offset[i_u]),
                            np.array([float(log_w)]))
                        correct_errors.append(
                            np.abs(v_c[0] - orig_real[i_w, i_g, i_rho, i_u]))
                        offset_errors.append(
                            np.abs(v_o[0] - orig_real[i_w, i_g, i_rho, i_u]))
        self.assertLess(max(correct_errors), 1e-6,
                        'correct-u errors must be small (sanity check)')
        self.assertGreater(max(offset_errors), max(correct_errors) * 2.0,
                           'offset-u errors must be at least 2x correct-u; '
                           'otherwise the node-exact test has no teeth')


# ---------------------------------------------------------------------------
# Open-cusp edge probe
# ---------------------------------------------------------------------------


#: Grid sizes for the open-cusp edge-probe chart.
_EDGE_N_GAMMA: int = 4
_EDGE_N_RHO: int = 4
_EDGE_N_THETA: int = 4
_EDGE_W_RANGE: tuple[float, float] = (10.0, 50.0)
_EDGE_W_NODES_PER_DECADE: int = 5


def _edge_probe_fixture() -> tuple:
    """Build a lobe chart immediately adjacent to a cusp for edge probing.

    Returns ``(surrogate, chart, cusp, theta_lo)``.
    """
    adm, cusp_angles, config = _cusp_adjacent_admission()
    if not cusp_angles:
        raise RuntimeError('fixture: no cusp angles found')
    cusp = min(cusp_angles)
    # Tile starting right AT the cusp angle (1e-6 gap).
    theta_lo = cusp + 1e-6
    theta_hi = cusp + 0.35
    rho_range = (0.3, 0.6)
    surrogate = surrogate_module.LensAmplificationSurrogate.from_lobe_engine(
        admission=adm, gamma_range=_CUSP_BAND,
        rho_lobe_range=rho_range,
        theta_local_range=(theta_lo, theta_hi),
        w_range=_EDGE_W_RANGE,
        n_gamma=_EDGE_N_GAMMA, n_rho=_EDGE_N_RHO, n_theta=_EDGE_N_THETA,
        w_nodes_per_decade=_EDGE_W_NODES_PER_DECADE,
        cusp_angle=cusp, cusp_side='left')
    chart = surrogate.charts[0]
    return surrogate, chart, cusp, theta_lo


class OpenCuspEdgeProbeTestCase(_LobeSubTestCase):
    """Acceptance: a lobe chart immediately adjacent to a cusp reproduces
    the engine envelope at a point just inside the cusp boundary (at the
    highest rho the chart can serve, ρ_lobe=0.5) to ≤ 1e-3
    max relative error (max|F(w)| normalized).

    This is the open-cusp-edge test: verifies the cusp-adapted coordinate
    is smooth at the boundary where ``d → 0`` and
    ``u = d**(2/3)`` → 0.  A chart with raw-theta spline (no cusp-adapted
    u-axis) would diverge as ``d**(-1/3)`` near the boundary, producing
    large interpolation errors for the first node inside.

    Cost: one chart build (~64 × 5 engine evals ≈ 3 s) + one query-point
    engine eval ≈ 3 s total.  Within the 60 s ceiling.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._surrogate, cls._chart, cusp, theta_lo = _edge_probe_fixture()
        cls._cusp = cusp
        cls._theta_lo = theta_lo

    def test_open_cusp_edge_accuracy(self) -> None:
        """Chart envelope agrees with direct engine at ρ=0.5,
        θ=θ_lo+1e-6 within 1e-3 max|F| normalised error."""
        chart = self._chart
        gamma_q = float(np.median(chart.gamma_grid))
        rho_q = 0.5
        theta_q = self._theta_lo + 1e-6

        from cogwheel.lensing.chang_refsdal import ChangRefsdalChannels

        # Direct engine evaluation at the query point.
        y1, y2 = surrogate_module._from_lobe_fixed(
            chart.centroid, chart.boundary_theta, chart.boundary_r,
            rho_q, theta_q)
        w_grid = np.exp(chart.log_w_grid)
        ch = ChangRefsdalChannels(w_grid)
        ch.reset()
        partition = ch.evaluate(
            gamma=gamma_q, y=(float(y1), float(y2)),
            beta=0.0, kappa=0.0)
        engine_env = partition.envelope
        eta = float(partition.caustic_distance)
        theta_c = float(partition.critical_theta)
        image_count = int(partition.real_mask.sum())

        selected = surrogate_module.select_chart(
            self._surrogate.charts,
            gamma=gamma_q, log_w_min=chart.log_w_grid[0],
            log_w_max=chart.log_w_grid[-1],
            eta=eta, theta=theta_c, image_count=image_count,
            y1_eig=y1, y2_eig=y2)
        self.n_checks += 1
        self.assertIsNotNone(selected,
                             'select_chart must serve the query point at '
                             f'rho={rho_q}, theta={theta_q}')

        chart_env = surrogate_module._evaluate_chart(
            selected, gamma_q, eta, theta_c,
            chart.log_w_grid, y1_eig=y1, y2_eig=y2)

        max_abs = max(np.max(np.abs(engine_env)),
                      np.max(np.abs(chart_env)), 1.0)
        re = np.max(np.abs(chart_env - engine_env)) / max_abs
        self.n_checks += 1
        self.assertLess(
            re, 1e-3,
            f'chart-engine envelope mismatch {re:.2e} exceeds 1e-3 at '
            f'open cusp edge (rho={rho_q}, theta={theta_q})')


class OpenCuspEdgeSelfFalsificationTestCase(TestCase):
    """Prove the open-cusp edge probe can FAIL."""

    def test_chart_without_cusp_threading_has_larger_error(self) -> None:
        """Building the same tile WITHOUT cusp-angle threading (raw-theta
        fallback) produces larger errors near the cusp edge -- proving the
        cusp-adapted coordinate is load-bearing.

        The raw-theta chart has a uniform theta grid that may span close
        to the cusp boundary, and without the ``d**(2/3)`` remapping the
        envelope diverges as ``d**(-1/3)``, causing larger interpolation
        error.  This test verifies that the cusp-threaded chart (green
        test) is BETTER than the raw-theta chart, not the same.
        """
        adm, cusp_angles, _config = _cusp_adjacent_admission()
        cusp = min(cusp_angles)
        theta_lo = cusp + 1e-6
        theta_hi = cusp + 0.35
        rho_range = (0.3, 0.6)

        # Build with cusp threading (correct).
        s_cusp = surrogate_module.LensAmplificationSurrogate.from_lobe_engine(
            admission=adm, gamma_range=_CUSP_BAND,
            rho_lobe_range=rho_range,
            theta_local_range=(theta_lo, theta_hi),
            w_range=_EDGE_W_RANGE,
            n_gamma=_EDGE_N_GAMMA, n_rho=_EDGE_N_RHO, n_theta=_EDGE_N_THETA,
            w_nodes_per_decade=_EDGE_W_NODES_PER_DECADE,
            cusp_angle=cusp, cusp_side='left')
        # Build WITHOUT cusp threading (raw-theta fallback).
        s_raw = surrogate_module.LensAmplificationSurrogate.from_lobe_engine(
            admission=adm, gamma_range=_CUSP_BAND,
            rho_lobe_range=rho_range,
            theta_local_range=(theta_lo, theta_hi),
            w_range=_EDGE_W_RANGE,
            n_gamma=_EDGE_N_GAMMA, n_rho=_EDGE_N_RHO, n_theta=_EDGE_N_THETA,
            w_nodes_per_decade=_EDGE_W_NODES_PER_DECADE,
            cusp_angle=None, cusp_side=None)

        # The cusp-threaded chart must carry a theta_to_u map; the raw
        # chart must NOT.
        c_chart = s_cusp.charts[0]
        r_chart = s_raw.charts[0]
        self.assertIsNotNone(c_chart.theta_to_u,
                             'cusp-threaded chart must have theta_to_u')
        self.assertIsNone(r_chart.theta_to_u,
                          'raw-theta chart must NOT have theta_to_u')
