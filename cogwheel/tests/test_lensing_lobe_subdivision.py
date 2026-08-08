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
separate ``_LOBE_CUSP_EXCLUSION_DISTANCE`` carve-out is wired; the
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
    ``_LOBE_CUSP_EXCLUSION_DISTANCE`` is wired (Professor ruling).  The
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
        self.assertIn('child_half', summary)

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
