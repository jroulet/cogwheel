"""
Tests for the macro-saddle LOBE-INTERIOR machinery of `lensing.surrogate`
(WP1/WP2/WP3): the shared directional-boundary helper
`_lobe_boundary_radius`, the lobe-local coordinate maps `_to_lobe_fixed`
/ `_from_lobe_fixed`, and the serve-time lobe dispatch (`_lobe_serves`,
`select_chart`, `LensAmplificationSurrogate.serve`).

For ``gamma > 1`` the Chang--Refsdal caustic is two disjoint 3-cusp
deltoid lobes sitting off the origin on the shear axis.  Each lobe is
tiled in a LOBE-LOCAL polar frame centred on its source-plane deltoid
centroid, with ``rho_lobe = |y - centroid| / r_deltoid(theta_local)`` so
that ``rho_lobe = 1`` tracks the deltoid boundary in EVERY direction.  A
scalar reach would overshoot the elongated near-cusp directions of a
sheared lobe; the directional boundary radius `_lobe_boundary_radius` is
therefore the single authoritative object shared by (a) the coordinate
maps and (b) the training-side admission
`surrogate_training._SaddleLobeAdmission._r_deltoid` (routed to it in
WP3).

What each block proves
----------------------
* ``LobeRoundTripTestCase`` -- the coordinate maps are EXACT inverses
  (``physical -> _to_lobe_fixed -> _from_lobe_fixed`` and back) to
  ``<= _ROUND_TRIP_TOL`` for every direction INCLUDING the ``+-pi``
  angular seam and the near-cusp pinch where a scalar reach would
  overshoot.  A seam/cusp discontinuity would show as an error spike in
  the diagnostic scatter.
* ``RDeltoidSingleSourceTestCase`` -- ``_r_deltoid`` returns EXACTLY
  ``_lobe_boundary_radius(...)`` (bit-for-bit), the two module names are
  the SAME object, and perturbing the one helper moves BOTH callers
  identically -- so the deltoid-boundary convention has exactly one home
  (WP3), not two copies kept in sync.
* ``CorridorRefusalTestCase`` -- a source on the inter-lobe equidistance
  (perpendicular-bisector) line is served by NEITHER lobe, and the
  abstention is attributable to the corridor predicate
  ``|p - centroid| + corridor_half > |p - other_centroid|`` (the
  documented, named fall-through to the exact-engine ladder).  The
  corridor gate's teeth are isolated on an otherwise-served interior
  point.
* ``LobeExclusivityTestCase`` -- an interior source is served by its
  OWNING lobe's chart and by that one only; the served-lobe-id map over
  a grid straddling the corridor shows a clean unserved gap on the
  equidistance line.
* ``LobeMapSelfFalsificationTestCase`` -- closes the loop: a mismatched
  inverse breaks the round-trip past ``_ROUND_TRIP_TOL`` and a corrupted
  boundary helper flips a lobe-serve decision, so a green suite is
  evidence rather than decoration.

Tolerance justification.  ``_ROUND_TRIP_TOL = 1e-12`` is the composed
map's floating-point floor: the maps are ``atan2`` / ``hypot`` /
``np.interp`` round-trips with no engine evaluation, so a correct
implementation returns the input to a few ULP (measured ``~1e-17``); the
gate sits ~5 orders above that so a genuine seam/normalisation bug (the
failure this file exists to catch, which would move the result by an
``O(r_deltoid)`` amount) is caught while numerical noise is not.  The
delegation and single-source checks are BIT-EXACT (``0.0``) because both
callers evaluate the identical ``np.interp`` object.

``LobeTestCase.tearDown`` fails any test whose sweep asserted nothing
(anti-vacuity); ``LobeMapSelfFalsificationTestCase`` proves the suite
can go red.  The fixtures are analytic geometry (caustic clouds +
winding loops for one small saddle band and synthetic unit envelopes);
NO engine evaluation runs, so the whole file is fast-tier
(~1 s admissions build, reused across tests via `lru_cache`).
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import pathlib
import tempfile
import dataclasses
from unittest import TestCase, main, mock

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from cogwheel.lensing import surrogate as surrogate_module  # noqa: E402
from cogwheel.lensing import surrogate_training as training_module  # noqa: E402
from cogwheel.lensing.chang_refsdal.channels import (  # noqa: E402
    ChangRefsdalChannels)


#: Small synthetic macro-saddle (``gamma > 1``) shear band.  Both edges
#: exceed 1 so the caustic is two disjoint deltoid lobes and the origin
#: astroid admission does not apply.
_SADDLE_BAND: tuple[float, float] = (1.5, 1.7)

#: Interior serve query shear used throughout (inside the band, clear of
#: the ``gamma = 1`` guard band).
_SERVE_GAMMA: float = 1.6

#: Caustic distance passed to the serve guard; above the interior
#: ``eta`` floor (`surrogate._DEFAULT_CAUSTIC_FLOOR = 0.05`) so gate (g)
#: passes and the corridor/box gates decide.
_SERVE_ETA: float = 0.3

#: Frequencies for the serve calls; strictly positive, span the chart's
#: ``ln w`` box so `_log_w_band_inside` passes.
_W_ARRAY: np.ndarray = np.array([0.5, 1.0, 2.0])

#: Composed-map round-trip floor (see module docstring).  A correct map
#: returns the input to a few ULP (~1e-17); a normalisation/seam bug
#: moves it by O(r_deltoid) ~ 0.1.
_ROUND_TRIP_TOL: float = 1e-12

#: Lobe-local training axes for the synthetic charts.  ``rho_lobe`` runs
#: inside the deltoid boundary (``rho_lobe = 1``); ``theta_local`` spans
#: the full ``(-pi, pi]`` seam so containment never rejects on angle.
_RHO_LOBE_GRID: np.ndarray = np.linspace(0.05, 0.95, 4)
_THETA_LOCAL_GRID: np.ndarray = np.linspace(-np.pi, np.pi, 6)
_LOG_W_GRID: np.ndarray = np.linspace(-2.0, 1.0, 4)

#: Output directory for diagnostic plots.
_OUTPUT_DIR: pathlib.Path = (
    pathlib.Path(__file__).resolve().parent / 'output')


@functools.lru_cache(maxsize=4)
def _admissions(band: tuple[float, float]
                ) -> tuple[training_module._SaddleLobeAdmission,
                           training_module._SaddleLobeAdmission]:
    """The two real per-lobe admissions for ``band`` (built once).

    Uses the smoke `TrainingConfig` defaults (200 caustic samples); the
    build is pure geometry (caustic clouds + winding loops), no engine
    evaluation, so it is cheap and cacheable across the whole file.
    """
    config = training_module.TrainingConfig()
    lobe_a, lobe_b = training_module._saddle_lobe_admissions(band, config)
    return lobe_a, lobe_b


def _build_lobe_chart(band: tuple[float, float],
                      adm: training_module._SaddleLobeAdmission
                      ) -> surrogate_module.LobeInteriorChart:
    """A synthetic lobe chart carrying ``adm``'s REAL lobe frame.

    The envelope tensor is a unit constant (the serve DECISION under
    test is independent of the envelope values); everything that gates
    dispatch -- centroid, other_centroid, corridor_half, boundary_theta,
    boundary_r, image_count, parity -- is the genuine admission frame.
    """
    shape = (_LOG_W_GRID.size, band_gamma_grid(band).size,
             _RHO_LOBE_GRID.size, _THETA_LOCAL_GRID.size)
    envelope_real = np.ones(shape)
    envelope_imag = np.zeros(shape)
    return surrogate_module.LobeInteriorChart.from_lobe_values(
        gamma_grid=band_gamma_grid(band),
        rho_lobe_grid=_RHO_LOBE_GRID,
        theta_local_grid=_THETA_LOCAL_GRID,
        log_w_grid=_LOG_W_GRID,
        envelope_real=envelope_real, envelope_imag=envelope_imag,
        image_count=surrogate_module._MACRO_SADDLE_IMAGE_COUNT, parity=-1,
        centroid=adm.centroid, other_centroid=adm.other_centroid,
        corridor_half=adm.corridor_half,
        boundary_theta=adm.boundary_theta, boundary_r=adm.boundary_r)


def band_gamma_grid(band: tuple[float, float]) -> np.ndarray:
    """A 4-node gamma axis spanning ``band`` (cubic spline needs >= 4)."""
    return np.linspace(band[0], band[1], 4)


def _served_surrogate(band: tuple[float, float]
                      ) -> tuple[surrogate_module.LensAmplificationSurrogate,
                                 surrogate_module.LobeInteriorChart,
                                 surrogate_module.LobeInteriorChart]:
    """A two-lobe served surrogate plus its (chart_a, chart_b)."""
    lobe_a, lobe_b = _admissions(band)
    chart_a = _build_lobe_chart(band, lobe_a)
    chart_b = _build_lobe_chart(band, lobe_b)
    surrogate = surrogate_module.LensAmplificationSurrogate(
        [chart_a, chart_b], provenance={})
    return surrogate, chart_a, chart_b


def _interior_eigenframe_source(adm: training_module._SaddleLobeAdmission,
                                rho_lobe: float, theta_local: float
                                ) -> tuple[float, float]:
    """Eigenframe ``(y1, y2)`` of a lobe-local ``(rho_lobe, theta_local)``."""
    return surrogate_module._from_lobe_fixed(
        adm.centroid, adm.boundary_theta, adm.boundary_r,
        rho_lobe, theta_local)


class LobeTestCase(TestCase):
    """Base class carrying the anti-vacuity comparison tally.

    Every domain assertion increments ``self.n_checks``; ``tearDown``
    fails a test that performed zero checks so a sweep that silently
    skipped every case (e.g. an empty admission) cannot read green.
    """

    def setUp(self) -> None:
        self.n_checks = 0

    def tearDown(self) -> None:
        if self.n_checks == 0:
            self.fail('the test asserted nothing (no lobe comparison ran); '
                      'anti-vacuity guard tripped')


def _named_directions(adm: training_module._SaddleLobeAdmission
                      ) -> list[tuple[str, float, float]]:
    """``(label, rho_lobe, theta_local)`` probes for the round-trip sweep.

    Includes the ``+-pi`` seam, the near-cusp pinch (the ``theta_local``
    where ``r_deltoid`` is SMALLEST -- the deltoid pinches, where a
    scalar reach overshoots most), the far-cusp bulge (largest
    ``r_deltoid``) and a generic mid-interior point.
    """
    theta_dense = np.linspace(-np.pi, np.pi, 721)
    r_dir = surrogate_module._lobe_boundary_radius(
        theta_dense, adm.boundary_theta, adm.boundary_r)
    theta_pinch = float(theta_dense[int(np.argmin(r_dir))])
    theta_bulge = float(theta_dense[int(np.argmax(r_dir))])
    return [
        ('seam_plus', 0.30, math.pi),
        ('seam_minus', 0.30, -math.pi + 1e-12),
        ('near_cusp_pinch', 0.60, theta_pinch),
        ('far_cusp_bulge', 0.60, theta_bulge),
        ('generic_mid', 0.45, 1.1),
    ]


class LobeRoundTripTestCase(LobeTestCase):
    """Acceptance #1: lobe coordinate maps are exact inverses.

    ``_to_lobe_fixed`` and ``_from_lobe_fixed`` share the single
    directional boundary radius, so composing them must return the
    source to ``<= _ROUND_TRIP_TOL`` in every direction, INCLUDING the
    seam and near-cusp pinch where a scalar-reach normalisation would
    break.
    """

    def test_from_to_from_round_trip_exact_all_directions(self) -> None:
        """``physical -> _to -> _from`` reproduces the source everywhere."""
        adm, _ = _admissions(_SADDLE_BAND)
        for label, rho_lobe, theta_local in _named_directions(adm):
            with self.subTest(direction=label):
                y1, y2 = _interior_eigenframe_source(
                    adm, rho_lobe, theta_local)
                rho_back, theta_back = surrogate_module._to_lobe_fixed(
                    adm.centroid, adm.boundary_theta, adm.boundary_r, y1, y2)
                y1_rt, y2_rt = surrogate_module._from_lobe_fixed(
                    adm.centroid, adm.boundary_theta, adm.boundary_r,
                    rho_back, theta_back)
                err = max(abs(y1 - y1_rt), abs(y2 - y2_rt))
                self.n_checks += 1
                self.assertLessEqual(
                    err, _ROUND_TRIP_TOL,
                    f'{label}: physical round-trip error {err:.3e} '
                    f'> {_ROUND_TRIP_TOL:.1e}')

    def test_to_from_to_round_trip_exact_all_directions(self) -> None:
        """``(rho, theta) -> _from -> _to`` reproduces the coordinate.

        The complementary direction: starting from lobe-local
        coordinates, the eigenframe round-trip recovers ``rho_lobe`` and
        ``theta_local`` (angles compared on the circle to absorb the
        ``+-pi`` seam representation).
        """
        adm, _ = _admissions(_SADDLE_BAND)
        for label, rho_lobe, theta_local in _named_directions(adm):
            with self.subTest(direction=label):
                y1, y2 = _interior_eigenframe_source(
                    adm, rho_lobe, theta_local)
                rho_back, theta_back = surrogate_module._to_lobe_fixed(
                    adm.centroid, adm.boundary_theta, adm.boundary_r, y1, y2)
                rho_err = abs(rho_lobe - rho_back)
                # Angular residual on the circle: seam_minus stores
                # theta near -pi but atan2 may return +pi; wrap the
                # difference into (-pi, pi].
                dtheta = (theta_local - theta_back + math.pi) % (
                    2.0 * math.pi) - math.pi
                self.n_checks += 1
                self.assertLessEqual(
                    rho_err, _ROUND_TRIP_TOL,
                    f'{label}: rho_lobe round-trip error {rho_err:.3e}')
                self.assertLessEqual(
                    abs(dtheta), _ROUND_TRIP_TOL,
                    f'{label}: theta_local round-trip error {abs(dtheta):.3e}')

    def test_no_seam_or_cusp_discontinuity_in_error_scatter(self) -> None:
        """Round-trip error is flat across ``theta_local`` (no seam spike).

        Sweeps a full ``(-pi, pi]`` circle at fixed ``rho_lobe`` and
        asserts the max round-trip error stays at the numerical floor --
        a seam or cusp discontinuity would spike it far above
        ``_ROUND_TRIP_TOL``.  Saves the diagnostic scatter.
        """
        adm, _ = _admissions(_SADDLE_BAND)
        thetas = np.linspace(-np.pi, np.pi, 361)
        rho_lobe = 0.5
        errors = np.empty(thetas.size)
        for idx, theta_local in enumerate(thetas):
            y1, y2 = _interior_eigenframe_source(
                adm, rho_lobe, float(theta_local))
            rho_back, theta_back = surrogate_module._to_lobe_fixed(
                adm.centroid, adm.boundary_theta, adm.boundary_r, y1, y2)
            y1_rt, y2_rt = surrogate_module._from_lobe_fixed(
                adm.centroid, adm.boundary_theta, adm.boundary_r,
                rho_back, theta_back)
            errors[idx] = max(abs(y1 - y1_rt), abs(y2 - y2_rt))

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.semilogy(thetas, np.maximum(errors, 1e-20), '.', ms=3)
        axis.axhline(_ROUND_TRIP_TOL, color='r', ls='--',
                     label=f'tol {_ROUND_TRIP_TOL:.0e}')
        axis.set_xlabel('theta_local [rad]')
        axis.set_ylabel('round-trip error |y - y_rt|')
        axis.set_title('Lobe coordinate round-trip error vs theta_local')
        axis.legend()
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR
                    / 'lobe_roundtrip_error_vs_theta_local.png', dpi=90)
        plt.close(fig)

        self.n_checks += 1
        self.assertLessEqual(
            float(errors.max()), _ROUND_TRIP_TOL,
            f'seam/cusp discontinuity: max round-trip error '
            f'{errors.max():.3e} > {_ROUND_TRIP_TOL:.1e}')


class RDeltoidSingleSourceTestCase(LobeTestCase):
    """Post-fix: ``_r_deltoid`` and the maps share ONE boundary helper.

    WP3 routed ``_SaddleLobeAdmission._r_deltoid`` through
    ``surrogate._lobe_boundary_radius`` so the deltoid-boundary
    convention has a single home.  These tests prove the delegation is
    bit-exact, the two module names bind the SAME object, and perturbing
    the one helper moves BOTH callers identically -- never two copies
    kept in sync.
    """

    def test_admission_r_deltoid_equals_helper_bit_for_bit(self) -> None:
        """``_r_deltoid(theta) == _lobe_boundary_radius(theta, ...)`` exactly.

        Swept across ``(-pi, pi]`` INCLUDING the ``+-pi`` seam; the
        difference is identically ``0.0`` because both evaluate the same
        ``np.interp`` object.
        """
        adm, _ = _admissions(_SADDLE_BAND)
        thetas = np.linspace(-np.pi, np.pi, 257)
        delegated = adm._r_deltoid(thetas)
        authoritative = surrogate_module._lobe_boundary_radius(
            thetas, adm.boundary_theta, adm.boundary_r)
        self.n_checks += 1
        # Bit-for-bit: the delegation must reproduce the helper exactly,
        # not merely to a tolerance.
        self.assertEqual(delegated.tobytes(), authoritative.tobytes())

    def test_single_authoritative_object_identity(self) -> None:
        """The two module bindings are the SAME function object.

        ``surrogate_training._lobe_boundary_radius`` is imported from
        ``surrogate``; identity (``is``) is the definitive proof that
        there is exactly one authoritative definition.  If a private
        copy were ever reintroduced in the training module this breaks.
        """
        self.n_checks += 1
        self.assertIs(training_module._lobe_boundary_radius,
                      surrogate_module._lobe_boundary_radius)

    def test_perturbing_helper_moves_both_callers_identically(self) -> None:
        """One perturbation -> identical change in ``_r_deltoid`` AND map.

        Patches the boundary helper (in BOTH consuming namespaces) to
        scale the radius by a fixed factor, then confirms the admission's
        ``_r_deltoid`` and the coordinate map's implied radius
        (``|_from_lobe_fixed(rho=1) - centroid|``) BOTH scale by exactly
        that factor.  Were ``_r_deltoid`` a private ``np.interp`` copy it
        would ignore the patch and this test would fail -- so it is a
        genuine teeth check on the delegation, not a tautology.
        """
        adm, _ = _admissions(_SADDLE_BAND)
        theta_probe = 0.7
        factor = 1.3
        original = surrogate_module._lobe_boundary_radius

        def _scaled(theta, boundary_theta, boundary_r):
            return factor * original(theta, boundary_theta, boundary_r)

        r_delt_base = float(adm._r_deltoid(theta_probe))
        y_base = surrogate_module._from_lobe_fixed(
            adm.centroid, adm.boundary_theta, adm.boundary_r, 1.0, theta_probe)
        radius_base = math.hypot(y_base[0] - adm.centroid[0],
                                 y_base[1] - adm.centroid[1])

        with mock.patch.object(surrogate_module, '_lobe_boundary_radius',
                               _scaled), \
                mock.patch.object(training_module, '_lobe_boundary_radius',
                                  _scaled):
            r_delt_patched = float(adm._r_deltoid(theta_probe))
            y_patched = surrogate_module._from_lobe_fixed(
                adm.centroid, adm.boundary_theta, adm.boundary_r,
                1.0, theta_probe)
            radius_patched = math.hypot(y_patched[0] - adm.centroid[0],
                                        y_patched[1] - adm.centroid[1])

        self.n_checks += 1
        self.assertAlmostEqual(r_delt_patched / r_delt_base, factor,
                               places=12,
                               msg='_r_deltoid did not follow the patched '
                                   'helper (a private copy would do this)')
        self.assertAlmostEqual(radius_patched / radius_base, factor,
                               places=12,
                               msg='coordinate map did not follow the '
                                   'patched helper')
        # Both callers changed by the SAME factor: one feed, not two.
        self.assertAlmostEqual(r_delt_patched / r_delt_base,
                               radius_patched / radius_base, places=12)


def _lobe_serve_args(y1_eig: float, y2_eig: float) -> tuple:
    """Positional args for `_lobe_serves` at a served ``(gamma, ln w)``.

    Order: ``(gamma, log_w_min, log_w_max, eta, image_count, y1, y2)`` --
    the certified-physical quantities the guard keys on, fixed so that
    gates (a) gamma box, (b) ln w band, (f) image count and (g) eta floor
    all pass; only the corridor (c) and box (d) gates can decide.
    """
    log_w = np.log(_W_ARRAY)
    return (_SERVE_GAMMA, float(log_w.min()), float(log_w.max()),
            _SERVE_ETA, surrogate_module._MACRO_SADDLE_IMAGE_COUNT,
            y1_eig, y2_eig)


def _corridor_predicate_rejects(
        chart: surrogate_module.LobeInteriorChart,
        y1_eig: float, y2_eig: float) -> bool:
    """Whether gate (c) -- the inter-lobe corridor -- rejects the source.

    Independent re-statement of the production predicate
    ``near_this + corridor_half > near_other`` (this lobe serves only
    when STRICTLY closer to its own centroid by the corridor margin), so
    a False here means the abstention is NOT the corridor's doing.
    """
    near_this = math.hypot(y1_eig - float(chart.centroid[0]),
                           y2_eig - float(chart.centroid[1]))
    near_other = math.hypot(y1_eig - float(chart.other_centroid[0]),
                            y2_eig - float(chart.other_centroid[1]))
    return near_this + chart.corridor_half > near_other


class CorridorRefusalTestCase(LobeTestCase):
    """Acceptance #2: the inter-lobe corridor is a named fall-through.

    A source on the equidistance (perpendicular-bisector) line is served
    by NEITHER lobe; the abstention is attributable to the corridor
    predicate, and the surrogate returns ``served = False`` so the caller
    uses the exact-engine ladder.
    """

    def test_equidistance_source_served_by_neither_lobe(self) -> None:
        """On the bisector both ``_lobe_serves`` and ``serve`` abstain."""
        surrogate, chart_a, chart_b = _served_surrogate(_SADDLE_BAND)
        y1_eq, y2_eq = 0.0, 0.2  # x = 0 is the inter-lobe bisector.
        args = _lobe_serve_args(y1_eq, y2_eq)

        self.assertFalse(
            surrogate_module._lobe_serves(chart_a, *args),
            'lobe A must not serve an equidistance source')
        self.assertFalse(
            surrogate_module._lobe_serves(chart_b, *args),
            'lobe B must not serve an equidistance source')
        _, served, _ = surrogate.serve(
            _W_ARRAY, gamma=_SERVE_GAMMA, y1=y1_eq, y2=y2_eq, beta=0.0,
            eta=_SERVE_ETA, theta=0.0,
            image_count=surrogate_module._MACRO_SADDLE_IMAGE_COUNT)
        self.n_checks += 1
        self.assertFalse(served,
                         'serve must fall through (served=False) on the '
                         'inter-lobe equidistance line')

    def test_equidistance_source_violates_corridor_predicate(self) -> None:
        """The corridor predicate itself rejects the bisector source.

        Attribution: on ``x = 0`` the source is equidistant from the two
        centroids, so ``near_this + corridor_half > near_other`` holds
        for BOTH lobes -- the abstention is the corridor's doing, not an
        accident of some other gate.
        """
        _, chart_a, chart_b = _served_surrogate(_SADDLE_BAND)
        y1_eq, y2_eq = 0.0, 0.2
        self.n_checks += 1
        self.assertTrue(_corridor_predicate_rejects(chart_a, y1_eq, y2_eq),
                        'corridor predicate should reject for lobe A')
        self.assertTrue(_corridor_predicate_rejects(chart_b, y1_eq, y2_eq),
                        'corridor predicate should reject for lobe B')

    def test_corridor_gate_isolated_teeth(self) -> None:
        """Reachable-red: the corridor gate alone can veto a served point.

        Takes an interior source that lobe A DOES serve, then widens ONLY
        ``corridor_half`` (via ``dataclasses.replace``; the box, image and
        eta gates are untouched) until the corridor margin swallows the
        point.  ``_lobe_serves`` must flip to ``False`` -- proving the
        corridor gate has independent teeth -- and restoring the original
        chart restores service.
        """
        adm_a, _ = _admissions(_SADDLE_BAND)
        _, chart_a, _ = _served_surrogate(_SADDLE_BAND)
        y1, y2 = _interior_eigenframe_source(adm_a, 0.4, 0.0)
        args = _lobe_serve_args(y1, y2)

        self.assertTrue(
            surrogate_module._lobe_serves(chart_a, *args),
            'precondition: the interior source is served with the real '
            'corridor half-width')
        wide = dataclasses.replace(chart_a, corridor_half=10.0)
        self.n_checks += 1
        self.assertFalse(
            surrogate_module._lobe_serves(wide, *args),
            'widening ONLY corridor_half must veto the otherwise-served '
            'source (isolated corridor teeth)')
        # Restoring the original width restores service: nothing else moved.
        self.assertTrue(surrogate_module._lobe_serves(chart_a, *args))


class LobeExclusivityTestCase(LobeTestCase):
    """Acceptance #2: an interior source is served by ONE lobe only.

    A source inside lobe A's admitted interior is served by lobe A's
    chart and by that chart alone; the served-lobe-id map over a grid
    straddling the corridor shows a clean unserved gap on the
    equidistance line.
    """

    def test_interior_source_served_by_owning_lobe_only(self) -> None:
        """Lobe A serves its interior; lobe B declines it; serve succeeds."""
        adm_a, _ = _admissions(_SADDLE_BAND)
        surrogate, chart_a, chart_b = _served_surrogate(_SADDLE_BAND)
        y1, y2 = _interior_eigenframe_source(adm_a, 0.4, 0.0)
        args = _lobe_serve_args(y1, y2)

        self.assertTrue(surrogate_module._lobe_serves(chart_a, *args),
                        'owning lobe A must serve its own interior source')
        self.assertFalse(surrogate_module._lobe_serves(chart_b, *args),
                         'the other lobe B must not serve lobe A interior')
        _, served, definition = surrogate.serve(
            _W_ARRAY, gamma=_SERVE_GAMMA, y1=y1, y2=y2, beta=0.0,
            eta=_SERVE_ETA, theta=0.0,
            image_count=surrogate_module._MACRO_SADDLE_IMAGE_COUNT)
        self.n_checks += 1
        self.assertTrue(served, 'serve must emulate an interior lobe source')
        self.assertEqual(definition,
                         surrogate_module._INTERIOR_ENVELOPE_DEFINITION,
                         'a served lobe chart reports its interior label')

    def test_served_lobe_id_map_has_clean_corridor_gap(self) -> None:
        """A grid straddling the corridor: two blobs, unserved bisector.

        Maps ``select_chart`` over an eigenframe ``(y1, y2)`` grid and
        colours each cell by the served lobe id (0 = none, 1 = A, 2 = B).
        Asserts (i) the ``y1 = 0`` bisector column is entirely unserved
        (the corridor gap), (ii) lobe A serves cells only in ``y1 < 0``
        and lobe B only in ``y1 > 0`` (exclusivity), and (iii) at least
        one cell of each lobe is served (non-vacuous).  Saves the map.
        """
        surrogate, chart_a, chart_b = _served_surrogate(_SADDLE_BAND)
        charts = surrogate.charts
        # Odd node counts over symmetric ranges put y1 = 0 (the bisector)
        # exactly at a column; kept coarse because each cell runs the full
        # select_chart stack (incl. the caustic-reach solve).
        y1_axis = np.linspace(-2.3, 2.3, 45)  # index 22 == 0.0 exactly
        y2_axis = np.linspace(-0.6, 0.6, 11)
        served_id = np.zeros((y2_axis.size, y1_axis.size), dtype=int)
        log_w = np.log(_W_ARRAY)
        for iy, y2 in enumerate(y2_axis):
            for ix, y1 in enumerate(y1_axis):
                rho, theta_c = surrogate_module._to_caustic_fixed(
                    _SERVE_GAMMA, float(y1), float(y2))
                chart = surrogate_module.select_chart(
                    charts, gamma=_SERVE_GAMMA, log_w_min=float(log_w.min()),
                    log_w_max=float(log_w.max()), eta=_SERVE_ETA, theta=0.0,
                    image_count=surrogate_module._MACRO_SADDLE_IMAGE_COUNT,
                    rho=rho, theta_c=theta_c, y1_eig=float(y1),
                    y2_eig=float(y2))
                if chart is chart_a:
                    served_id[iy, ix] = 1
                elif chart is chart_b:
                    served_id[iy, ix] = 2

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=(8, 3.2))
        mesh = axis.pcolormesh(y1_axis, y2_axis, served_id,
                               cmap='viridis', vmin=0, vmax=2, shading='auto')
        axis.axvline(0.0, color='w', ls='--', lw=1)
        axis.set_xlabel('y1_eig')
        axis.set_ylabel('y2_eig')
        axis.set_title('Served lobe id (0 none, 1 A, 2 B) across the corridor')
        fig.colorbar(mesh, ax=axis, ticks=[0, 1, 2])
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR / 'lobe_served_id_map_corridor_gap.png',
                    dpi=90)
        plt.close(fig)

        bisector = served_id[:, y1_axis == 0.0]
        left = served_id[:, y1_axis < 0.0]
        right = served_id[:, y1_axis > 0.0]
        self.n_checks += 1
        self.assertTrue(np.all(bisector == 0),
                        'the y1 = 0 bisector column must be unserved '
                        '(the corridor gap)')
        # Exclusivity: lobe A (id 1) only left of the bisector, lobe B
        # (id 2) only right of it.
        self.assertFalse(np.any(left == 2),
                         'lobe B must not serve any y1 < 0 cell')
        self.assertFalse(np.any(right == 1),
                         'lobe A must not serve any y1 > 0 cell')
        self.assertTrue(np.any(served_id == 1) and np.any(served_id == 2),
                        'both lobes must serve at least one cell (non-vacuous)')


class LobeMapSelfFalsificationTestCase(TestCase):
    """Prove this suite can FAIL: corrupt the map / helper and go red.

    Without this, "the round-trip suite is green" could mean the maps are
    correct OR that the assertions are vacuous.  Here a deliberately
    mismatched inverse and a corrupted boundary helper are shown to break
    the very invariants the green tests rely on.
    """

    def test_mismatched_inverse_breaks_round_trip(self) -> None:
        """A different boundary radius in the inverse blows the round-trip.

        Feeds a perturbed ``boundary_r`` to ``_to_lobe_fixed`` only (the
        forward map keeps the true radius), so the composition is no
        longer an inverse; the round-trip error must exceed
        ``_ROUND_TRIP_TOL`` -- otherwise the tolerance gate has no teeth.
        """
        adm, _ = _admissions(_SADDLE_BAND)
        y1, y2 = _interior_eigenframe_source(adm, 0.5, 0.9)
        wrong_r = adm.boundary_r * 1.05
        rho_back, theta_back = surrogate_module._to_lobe_fixed(
            adm.centroid, adm.boundary_theta, wrong_r, y1, y2)
        y1_rt, y2_rt = surrogate_module._from_lobe_fixed(
            adm.centroid, adm.boundary_theta, adm.boundary_r,
            rho_back, theta_back)
        err = max(abs(y1 - y1_rt), abs(y2 - y2_rt))
        self.assertGreater(
            err, _ROUND_TRIP_TOL,
            'a mismatched inverse must break the round-trip; the tolerance '
            'gate would be vacuous otherwise')

    def test_corrupted_helper_flips_lobe_serve_decision(self) -> None:
        """Shrinking ``r_deltoid`` pushes a served point out of the box.

        Patches ``_lobe_boundary_radius`` to a tiny fraction of the true
        radius: the interior source's ``rho_lobe = |y - c| / r_deltoid``
        then explodes past ``rho_lobe_grid[-1]``, so box gate (d) rejects
        and ``_lobe_serves`` flips ``True -> False`` -- proving the serve
        decision genuinely depends on the (single) boundary helper.
        """
        adm_a, _ = _admissions(_SADDLE_BAND)
        _, chart_a, _ = _served_surrogate(_SADDLE_BAND)
        y1, y2 = _interior_eigenframe_source(adm_a, 0.4, 0.0)
        args = _lobe_serve_args(y1, y2)
        self.assertTrue(surrogate_module._lobe_serves(chart_a, *args))

        original = surrogate_module._lobe_boundary_radius

        def _tiny(theta, boundary_theta, boundary_r):
            return 0.01 * original(theta, boundary_theta, boundary_r)

        with mock.patch.object(surrogate_module, '_lobe_boundary_radius',
                               _tiny):
            flipped = surrogate_module._lobe_serves(chart_a, *args)
        self.assertFalse(
            flipped,
            'a corrupted boundary helper must flip the lobe-serve decision')

# ---------------------------------------------------------------------------
# Engine-backed lobe-interior fixtures (acceptance #4/#2/#6).
#
# The blocks below train ONE real `LobeInteriorChart` on a single admitted
# tile of a small macro-saddle band via the genuine trainer path
# (`surrogate_training._build_lobe_chart` -> `from_lobe_engine`) and probe it
# against a FRESH `ChangRefsdalChannels` engine oracle.  The oracle calls the
# lensing engine, NEVER the chart, so served-vs-engine agreement is a genuine
# (non-circular) accuracy statement.  The chart is trained once and reused
# across every engine test via `lru_cache` (measured: build ~2.8 s / 64 nodes,
# held-out eps ~0.7 s; each per-class 64-node engine sweep ~2.5 s), so the
# whole engine section stays well inside the fast-tier ceiling.
#
# Tolerance justification.  ``_NODE_EXACT_TOL = 1e-10`` gates the served-minus-
# engine envelope at the chart's OWN spline nodes: a tensor-product cubic
# INTERPOLATING spline reproduces its training samples to a few ULP (measured
# ~6e-16 .. 8e-16), so a node that departs by 1e-10 signals a coordinate-frame
# or reconstruction bug, not float noise, while the ~4 orders of head-room
# keeps engine reproducibility jitter from tripping the gate.  The interior
# quartile gate uses the chart's OWN held-out eps (the trainer's LOO
# acceptance metric for this very tile, recomputed here via the identical
# `_heldout_eps` path); interior errors sit ~50x below it (~3e-3 vs ~0.14).

#: Narrow macro-saddle band that ADMITS lobe-interior tiles under the smoke
#: `TrainingConfig` (the wider `_SADDLE_BAND` admits none -- the deltoid
#: shrinks across it and the winding admission fails).  Measured: 33 tiles.
_ENGINE_BAND: tuple[float, float] = (1.3, 1.4)

#: Macro-saddle parity for a lobe chart (``!= 1``); the engine assigns the
#: interior a signed image count of -1 for this band.
_ENGINE_PARITY: int = -1

#: Frequency span for the trained chart (dimensionless ``w``); strictly
#: positive, a little over one decade so the ``ln w`` spline has >= 4 nodes.
_ENGINE_W_RANGE: tuple[float, float] = (0.5, 5.0)

#: Node-reproduction floor (see justification above).  The interpolating
#: spline returns its own training sample to ~1e-16; 1e-10 catches a genuine
#: frame/reconstruction defect while ignoring ULP noise.
_NODE_EXACT_TOL: float = 1e-10

#: Seed for the held-out sampler so the recomputed eps (and hence the
#: interior gate) is deterministic run-to-run.
_ENGINE_SEED: int = 0

#: Real-image count an exterior (outside-lobe) saddle source produces -- the
#: contrast that makes ``image_count == 4`` a genuine INTERIOR property.
_EXTERIOR_SADDLE_IMAGE_COUNT: int = 2

#: Far exterior saddle probe sources (eigenframe ``(y1, y2)``) well outside
#: both deltoid lobes, used for the 2-image contrast.
_EXTERIOR_SOURCES: tuple[tuple[float, float], ...] = (
    (0.0, 3.0), (0.0, 5.0), (2.5, 2.5), (0.0, -4.0))


@dataclasses.dataclass(frozen=True)
class _EngineLobeFixture:
    """One trained lobe chart plus everything the engine tests query it with.

    ``chart`` is a real `LobeInteriorChart` (interpolating cubic splines over
    ``(ln w, gamma, rho_lobe, theta_local)``); ``lobe`` is the genuine
    `_SaddleLobeAdmission` frame it was trained in; ``heldout_eps`` is the
    trainer's LOO acceptance metric recomputed for THIS tile; ``surrogate``
    wraps the single chart for serving; ``w_grid`` are the chart's own ``w``
    nodes; ``box_center`` / ``half`` delimit the lobe-local training box.
    """

    chart: surrogate_module.LobeInteriorChart
    lobe: training_module._SaddleLobeAdmission
    surrogate: surrogate_module.LensAmplificationSurrogate
    heldout_eps: float
    w_grid: np.ndarray
    box_center: tuple[float, float]
    half: tuple[float, float]


@functools.lru_cache(maxsize=1)
def _engine_lobe_fixture() -> _EngineLobeFixture:
    """Train one lobe chart on a real admitted tile (built once, cached).

    Enumerates the admitted lobe-interior tiles of ``_ENGINE_BAND`` with the
    genuine tiler and picks a WELL-FORMED interior tile (``rho_lobe`` centre
    0.3, a non-collapsed ``half_theta > 0.1`` -- the near-seam tiles pinch to
    ``half_theta ~ 1e-8`` and refuse most nodes), deterministically the middle
    such tile.  Builds the chart through the real trainer and recomputes the
    held-out eps exactly as the trainer's acceptance step does.
    """
    config = training_module.TrainingConfig()
    lobe_a, _lobe_b = training_module._saddle_lobe_admissions(
        _ENGINE_BAND, config)
    gamma_mid = 0.5 * (_ENGINE_BAND[0] + _ENGINE_BAND[1])
    lens_center = training_module._SADDLE_LOBE_CENTERS[0]
    lobe_cusps = training_module._lobe_cusp_source_angles(
        gamma_mid, lens_center, lobe_a.centroid, config.n_caustic_samples)
    tiles = training_module._lobe_interior_tiles(
        lobe_a, lobe_cusps, config.n_farfield_tiles_per_side)
    well_formed = [tile for tile in tiles
                   if abs(tile[0][0] - 0.3) < 1e-9 and tile[1][1] > 0.1]
    if not well_formed:
        raise RuntimeError(
            'no well-formed admitted lobe tile in _ENGINE_BAND; the engine '
            'fixture cannot be built (band/admission drift?).')
    box_center, half, _ti, _tj = well_formed[len(well_formed) // 2]
    chart, _calls, refused = training_module._build_lobe_chart(
        gamma_band=_ENGINE_BAND, parity=_ENGINE_PARITY, lobe=lobe_a,
        box_center=box_center, half=half, w_range=_ENGINE_W_RANGE,
        config=config)
    surrogate = surrogate_module.LensAmplificationSurrogate(
        [chart], {'schema': 'engine-lobe-fixture', 'refused': int(refused)})
    rng = np.random.default_rng(_ENGINE_SEED)
    samples = training_module._lobe_heldout_samples(
        _ENGINE_BAND, box_center, half, config, rng, lobe=lobe_a)
    heldout_eps = training_module._heldout_eps(
        chart, samples, {'schema': 'engine-lobe-heldout'})
    return _EngineLobeFixture(
        chart=chart, lobe=lobe_a, surrogate=surrogate,
        heldout_eps=float(heldout_eps), w_grid=np.exp(chart.log_w_grid),
        box_center=tuple(box_center), half=tuple(half))


def _engine_partition(gamma: float, y1: float, y2: float,
                      w_grid: np.ndarray):
    """Fresh-engine `ChangRefsdalGeometryPartition` at a source, or ``None``.

    Independent oracle: a brand-new `ChangRefsdalChannels` evaluated at the
    physical eigenframe source -- it never touches the surrogate.  Returns
    ``None`` when the engine refuses the source (`surrogate._REFUSAL_ERRORS`),
    mirroring the trainer's own held-out skip so a census-defect point cannot
    fail the test spuriously.
    """
    channels = ChangRefsdalChannels(w_grid)
    try:
        return channels.evaluate(gamma=gamma, y=(y1, y2), beta=0.0, kappa=0.0)
    except surrogate_module._REFUSAL_ERRORS:
        return None


def _served_minus_engine(fixture: _EngineLobeFixture, gamma: float,
                         rho_lobe: float, theta_local: float
                         ) -> tuple[bool, float | None, int | None]:
    """F-normalised max |E_served - E_engine| at a lobe-local coordinate.

    Maps ``(rho_lobe, theta_local)`` to a physical eigenframe source through
    the lobe frame, evaluates the fresh engine there, and serves the chart at
    the SAME source with the engine-derived ``(eta, theta, image_count)``.
    Returns ``(served, error, image_count)``; ``error`` is ``None`` when the
    engine refused or the chart declined (both are handled by the caller so a
    boundary node cannot silently vacuate the sweep).
    """
    y1, y2 = _interior_eigenframe_source(fixture.lobe, rho_lobe, theta_local)
    partition = _engine_partition(gamma, y1, y2, fixture.w_grid)
    if partition is None:
        return False, None, None
    env_true = np.asarray(partition.envelope)
    image_count = int(partition.real_mask.sum())
    emulated, served, _definition = fixture.surrogate.serve(
        fixture.w_grid, gamma=gamma, y1=y1, y2=y2, beta=0.0,
        eta=partition.caustic_distance, theta=partition.critical_theta,
        image_count=image_count)
    if not served:
        return False, None, image_count
    denom = float(np.max(np.abs(env_true))) or 1.0
    error = float(np.max(np.abs(emulated - env_true)) / denom)
    return True, error, image_count


class LobeEngineNodeExactnessTestCase(LobeTestCase):
    """Acceptance #4: served values reproduce the engine, node-exact interior.

    Part (a): at the chart's OWN spline nodes the served envelope must
    reproduce the fresh-engine `INTERIOR_SACR_C` envelope to
    ``_NODE_EXACT_TOL`` -- the interpolating cubic tensor spline returns its
    training samples (which the trainer read straight from the engine) to a
    few ULP, so a node departing by more than 1e-10 is a coordinate/
    reconstruction defect, not float noise.  Part (b): at interior points on
    the box quartiles (strictly away from the edges, Professor Q6) the served-
    minus-engine difference must stay within the chart's OWN held-out eps.

    The oracle is a fresh `ChangRefsdalChannels` at each physical source; it
    never queries the chart, so the agreement is non-circular.
    """

    def test_served_reproduces_engine_at_spline_nodes(self) -> None:
        """Serve at every ``(gamma, rho_lobe, theta_local)`` node ~ engine.

        Sweeps the full node grid; for each node that is BOTH engine-served
        and chart-served the F-normalised max |E_served - E_engine| must be
        ``<= _NODE_EXACT_TOL``.  Boundary nodes the engine refuses or the box/
        eta gate declines are skipped, but the anti-vacuity tally requires a
        healthy number of genuine node comparisons so a silently empty sweep
        cannot read green.
        """
        fixture = _engine_lobe_fixture()
        worst = 0.0
        for gamma in fixture.chart.gamma_grid:
            for rho_lobe in fixture.chart.rho_lobe_grid:
                for theta_local in fixture.chart.theta_local_grid:
                    served, error, _count = _served_minus_engine(
                        fixture, float(gamma), float(rho_lobe),
                        float(theta_local))
                    if not served or error is None:
                        continue
                    with self.subTest(gamma=float(gamma),
                                      rho_lobe=float(rho_lobe),
                                      theta_local=float(theta_local)):
                        self.n_checks += 1
                        worst = max(worst, error)
                        self.assertLessEqual(
                            error, _NODE_EXACT_TOL,
                            f'served-vs-engine node error {error:.3e} '
                            f'> {_NODE_EXACT_TOL:.1e}')
        # A well-formed interior tile serves a solid majority of its nodes;
        # require enough real comparisons that the pass is not vacuous.
        self.assertGreaterEqual(
            self.n_checks, 8,
            f'too few served nodes ({self.n_checks}); the node-exactness '
            f'sweep is near-vacuous (tile/admission drift?)')

    def test_interior_quartile_error_within_heldout_eps(self) -> None:
        """Interior quartile sources agree with the engine within the eps.

        Probes the 3x3 interior quartile lattice (box-fractions 0.25/0.5/0.75
        in both ``rho_lobe`` and ``theta_local`` -- strictly inside the box,
        never on an edge) and asserts every served point's engine difference
        is ``<= heldout_eps``.  Saves a diagnostic of the error vs the
        normalised distance to the nearest box edge, confirming the accuracy
        is an INTERIOR property rather than an artefact of hugging an edge.
        """
        fixture = _engine_lobe_fixture()
        rho_c, theta_c = fixture.box_center
        half_rho, half_theta = fixture.half
        fractions = (-0.5, 0.0, 0.5)  # of the half-width => quartiles 0.25/0.75
        edge_dists: list[float] = []
        errors: list[float] = []
        for f_rho in fractions:
            for f_theta in fractions:
                rho_lobe = rho_c + f_rho * half_rho
                theta_local = theta_c + f_theta * half_theta
                served, error, _count = _served_minus_engine(
                    fixture, 0.5 * (_ENGINE_BAND[0] + _ENGINE_BAND[1]),
                    rho_lobe, theta_local)
                if not served or error is None:
                    continue
                # Normalised distance to the nearest box edge in (rho, theta).
                edge = min(1.0 - abs(f_rho), 1.0 - abs(f_theta))
                edge_dists.append(edge)
                errors.append(error)
                with self.subTest(rho_lobe=rho_lobe, theta_local=theta_local):
                    self.n_checks += 1
                    self.assertLessEqual(
                        error, fixture.heldout_eps,
                        f'interior error {error:.3e} exceeds the chart '
                        f'held-out eps {fixture.heldout_eps:.3e}')
        self.assertGreaterEqual(
            len(errors), 5,
            f'too few served interior quartile points ({len(errors)}); the '
            f'interior accuracy gate is near-vacuous')

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.semilogy(edge_dists, np.maximum(errors, 1e-18), 'o', ms=6)
        axis.axhline(fixture.heldout_eps, color='r', ls='--',
                     label=f'held-out eps {fixture.heldout_eps:.2e}')
        axis.set_xlabel('normalised distance to nearest box edge')
        axis.set_ylabel('|E_served - E_engine| (F-normalised)')
        axis.set_title('Lobe interior served-vs-engine error vs edge distance')
        axis.legend()
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR
                    / 'lobe_interior_served_vs_engine_error.png', dpi=90)
        plt.close(fig)


class LobeImageCountTestCase(LobeTestCase):
    """Acceptance / Professor Q2: the lobe interior carries FOUR real images.

    Under the ``eta_max`` shell a macro-saddle lobe interior is a genuine
    4-image region; the chart stamps ``image_count == 4`` and every interior
    node reproduces that count at the engine.  The count is a real interior
    property, not a labelling convention: an exterior (outside-lobe) saddle
    source produces only TWO real images.
    """

    def test_chart_image_count_is_four(self) -> None:
        """`from_lobe_engine` stamped the macro-saddle interior count 4."""
        fixture = _engine_lobe_fixture()
        self.n_checks += 1
        self.assertEqual(fixture.chart.image_count,
                         surrogate_module._MACRO_SADDLE_IMAGE_COUNT)
        self.assertEqual(fixture.chart.image_count, 4)

    def test_all_interior_nodes_have_four_real_images(self) -> None:
        """Every engine-served node reports ``real_mask.sum() == 4``.

        Sweeps the node grid, evaluates the fresh engine at each physical
        source, and collects the real-image count; every count seen must be
        exactly 4 (engine-refused boundary nodes are skipped).  A count of 2
        anywhere would mean a node escaped the 4-image interior shell.
        """
        fixture = _engine_lobe_fixture()
        counts: set[int] = set()
        gamma_mid = 0.5 * (_ENGINE_BAND[0] + _ENGINE_BAND[1])
        for rho_lobe in fixture.chart.rho_lobe_grid:
            for theta_local in fixture.chart.theta_local_grid:
                y1, y2 = _interior_eigenframe_source(
                    fixture.lobe, float(rho_lobe), float(theta_local))
                partition = _engine_partition(
                    gamma_mid, y1, y2, fixture.w_grid)
                if partition is None:
                    continue
                count = int(partition.real_mask.sum())
                counts.add(count)
                with self.subTest(rho_lobe=float(rho_lobe),
                                  theta_local=float(theta_local)):
                    self.n_checks += 1
                    self.assertEqual(
                        count, surrogate_module._MACRO_SADDLE_IMAGE_COUNT,
                        f'interior node has {count} real images, not 4')
        self.assertEqual(
            counts, {4},
            f'interior real-image counts were {sorted(counts)}, expected '
            f'exactly {{4}}')

    def test_exterior_saddle_source_has_two_images(self) -> None:
        """Outside both lobes a saddle source yields only 2 real images.

        The contrast that makes ``image_count == 4`` a genuine interior
        property: far-field saddle sources sit in the 2-image regime, so the
        4-image count is not a blanket macro-saddle label but specific to the
        lobe interior under the ``eta_max`` shell.
        """
        fixture = _engine_lobe_fixture()
        gamma_mid = 0.5 * (_ENGINE_BAND[0] + _ENGINE_BAND[1])
        for y1, y2 in _EXTERIOR_SOURCES:
            partition = _engine_partition(gamma_mid, y1, y2, fixture.w_grid)
            if partition is None:
                continue
            with self.subTest(source=(y1, y2)):
                self.n_checks += 1
                self.assertEqual(
                    int(partition.real_mask.sum()),
                    _EXTERIOR_SADDLE_IMAGE_COUNT,
                    f'exterior saddle source {(y1, y2)} should have 2 real '
                    f'images')


def _save_with_meta_mutation(surrogate: surrogate_module
                             .LensAmplificationSurrogate,
                             src_path: pathlib.Path,
                             dst_path: pathlib.Path, index: int,
                             mutate) -> None:
    """Save ``surrogate``, then rewrite ``chart{index}`` meta via ``mutate``.

    Saves the real artifact to ``src_path``, reloads its flat arrays, decodes
    the ``chart{index}_meta`` JSON, applies ``mutate`` (a ``dict -> dict``
    callback on the meta), re-encodes it, and re-saves to ``dst_path``.  Every
    other array is copied verbatim so the only thing under test at load is the
    tampered meta field (e.g. an absent / unknown / cross-kind axis schema).
    """
    surrogate.save(src_path)
    with np.load(src_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    meta_key = f'chart{index}_meta'
    meta = json.loads(str(arrays[meta_key]))
    arrays[meta_key] = np.array(json.dumps(mutate(dict(meta))))
    np.savez(dst_path, **arrays)


class LobePersistenceTestCase(LobeTestCase):
    """Acceptance #6: lobe persistence round-trips; wrong schema hard-refuses.

    A `LensAmplificationSurrogate` carrying a real `LobeInteriorChart` saves
    to ``.npz``, reloads, and re-serves BIT-IDENTICALLY -- the lobe frame
    (centroid, other_centroid, corridor_half, boundary_theta, boundary_r) and
    the interior splines survive the round-trip exactly.  A lobe artifact
    whose ``axis_schema`` tag is absent, unknown, or the far-field tag
    HARD-REFUSES at load, so a shape-changed record can only reconstruct under
    its own schema; the cross-kind refusal is confirmed on both validators.
    """

    def _interior_serve_source(self, fixture: _EngineLobeFixture
                               ) -> tuple[float, float, float, float, int]:
        """A ``(gamma, y1, y2, eta, theta, image_count)`` the chart serves.

        Uses the box-centre interior source and the engine-derived geometry
        so the served call exercises the real dispatch stack.
        """
        rho_c, theta_c = fixture.box_center
        gamma = 0.5 * (_ENGINE_BAND[0] + _ENGINE_BAND[1])
        y1, y2 = _interior_eigenframe_source(fixture.lobe, rho_c, theta_c)
        partition = _engine_partition(gamma, y1, y2, fixture.w_grid)
        if partition is None:
            self.skipTest('engine refused the interior serve source')
        return (gamma, float(y1), float(y2),
                float(partition.caustic_distance),
                float(partition.critical_theta),
                int(partition.real_mask.sum()))

    def test_lobe_frame_and_splines_round_trip_exactly(self) -> None:
        """Saved/reloaded lobe frame and spline coeffs match bit-for-bit."""
        fixture = _engine_lobe_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'lobe.npz'
            fixture.surrogate.save(path)
            reloaded = surrogate_module.LensAmplificationSurrogate.load(path)
        original = fixture.chart
        restored = reloaded.charts[0]
        self.assertIsInstance(restored, surrogate_module.LobeInteriorChart)
        self.n_checks += 1
        # Lobe frame arrays: bit-for-bit.
        for name in ('centroid', 'other_centroid', 'boundary_theta',
                     'boundary_r'):
            with self.subTest(field=name):
                self.assertEqual(
                    np.asarray(getattr(original, name)).tobytes(),
                    np.asarray(getattr(restored, name)).tobytes(),
                    f'{name} did not round-trip bit-for-bit')
        # Corridor half-width and the interior splines: bit-for-bit.
        self.assertEqual(float(original.corridor_half),
                         float(restored.corridor_half))
        self.assertEqual(original.real_coeffs.tobytes(),
                         restored.real_coeffs.tobytes())
        self.assertEqual(original.imag_coeffs.tobytes(),
                         restored.imag_coeffs.tobytes())
        self.assertEqual(original.image_count, restored.image_count)
        self.assertEqual(original.envelope_definition,
                         restored.envelope_definition)

    def test_serve_bit_identical_pre_post_save(self) -> None:
        """Serving an interior source is byte-identical before/after reload."""
        fixture = _engine_lobe_fixture()
        gamma, y1, y2, eta, theta, image_count = self._interior_serve_source(
            fixture)
        emu_pre, served_pre, def_pre = fixture.surrogate.serve(
            fixture.w_grid, gamma=gamma, y1=y1, y2=y2, beta=0.0, eta=eta,
            theta=theta, image_count=image_count)
        self.assertTrue(served_pre, 'precondition: the source is served')
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'lobe.npz'
            fixture.surrogate.save(path)
            reloaded = surrogate_module.LensAmplificationSurrogate.load(path)
        emu_post, served_post, def_post = reloaded.serve(
            fixture.w_grid, gamma=gamma, y1=y1, y2=y2, beta=0.0, eta=eta,
            theta=theta, image_count=image_count)
        self.n_checks += 1
        self.assertTrue(served_post)
        self.assertEqual(def_pre, def_post)
        self.assertEqual(emu_pre.tobytes(), emu_post.tobytes(),
                         'served envelope changed across a save/load round '
                         'trip (the interpolant did not persist exactly)')

    def test_absent_axis_schema_hard_refuses(self) -> None:
        """A lobe artifact with NO ``axis_schema`` tag refuses at load."""
        fixture = _engine_lobe_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp) / 'ok.npz'
            bad = pathlib.Path(tmp) / 'no_schema.npz'
            _save_with_meta_mutation(
                fixture.surrogate, src, bad, 0,
                lambda meta: {k: v for k, v in meta.items()
                              if k != 'axis_schema'})
            self.n_checks += 1
            with self.assertRaises(ValueError):
                surrogate_module.LensAmplificationSurrogate.load(bad)

    def test_unknown_axis_schema_hard_refuses(self) -> None:
        """A lobe artifact with an unknown ``axis_schema`` refuses at load."""
        fixture = _engine_lobe_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp) / 'ok.npz'
            bad = pathlib.Path(tmp) / 'unknown_schema.npz'
            _save_with_meta_mutation(
                fixture.surrogate, src, bad, 0,
                lambda meta: {**meta, 'axis_schema': 'totally_made_up_tag'})
            self.n_checks += 1
            with self.assertRaises(ValueError):
                surrogate_module.LensAmplificationSurrogate.load(bad)

    def test_farfield_tag_on_lobe_chart_refuses(self) -> None:
        """A lobe record stamped with the FAR-FIELD schema refuses at load.

        The shape-changed lobe record must reconstruct only under its own
        lobe schema; presenting the far-field tag makes
        `_validate_lobe_axis_schema` hard-refuse.
        """
        fixture = _engine_lobe_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp) / 'ok.npz'
            bad = pathlib.Path(tmp) / 'farfield_tag.npz'
            _save_with_meta_mutation(
                fixture.surrogate, src, bad, 0,
                lambda meta: {**meta,
                              'axis_schema':
                                  surrogate_module._FARFIELD_AXIS_SCHEMA})
            self.n_checks += 1
            with self.assertRaises(ValueError):
                surrogate_module.LensAmplificationSurrogate.load(bad)

    def test_current_lobe_schema_round_trips(self) -> None:
        """Positive control: the current lobe schema loads (no false refusal).

        Re-stamps the meta with exactly ``_LOBE_AXIS_SCHEMA`` (an identity
        mutation through the tamper path) and confirms the artifact still
        loads, so the refusal tests above are attributable to the WRONG tag,
        not to the tamper mechanism itself.
        """
        fixture = _engine_lobe_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp) / 'ok.npz'
            good = pathlib.Path(tmp) / 'current_schema.npz'
            _save_with_meta_mutation(
                fixture.surrogate, src, good, 0,
                lambda meta: {**meta,
                              'axis_schema':
                                  surrogate_module._LOBE_AXIS_SCHEMA})
            reloaded = surrogate_module.LensAmplificationSurrogate.load(good)
        self.n_checks += 1
        self.assertIsInstance(reloaded.charts[0],
                              surrogate_module.LobeInteriorChart)

    def test_cross_kind_axis_schema_validators_refuse_both_ways(self) -> None:
        """Each kind's schema gate rejects the OTHER kind's tag (and ``None``).

        The single load-time gate for each chart kind is its axis-schema
        validator.  A lobe chart stamped with the far-field tag and a far-
        field chart stamped with the lobe tag both hard-refuse; the correct
        tag validates and is returned.  This proves the ``vice versa``
        direction without needing to train a real far-field chart.
        """
        self.n_checks += 1
        # Far-field tag on the lobe validator -> refuse; lobe tag OK.
        with self.assertRaises(ValueError):
            surrogate_module._validate_lobe_axis_schema(
                surrogate_module._FARFIELD_AXIS_SCHEMA, 'chart 0')
        with self.assertRaises(ValueError):
            surrogate_module._validate_lobe_axis_schema(None, 'chart 0')
        self.assertEqual(
            surrogate_module._validate_lobe_axis_schema(
                surrogate_module._LOBE_AXIS_SCHEMA, 'chart 0'),
            surrogate_module._LOBE_AXIS_SCHEMA)
        # Lobe tag on the far-field validator -> refuse; far-field tag OK.
        with self.assertRaises(ValueError):
            surrogate_module._validate_farfield_axis_schema(
                surrogate_module._LOBE_AXIS_SCHEMA, 'chart 0')
        with self.assertRaises(ValueError):
            surrogate_module._validate_farfield_axis_schema(None, 'chart 0')
        self.assertEqual(
            surrogate_module._validate_farfield_axis_schema(
                surrogate_module._FARFIELD_AXIS_SCHEMA, 'chart 0'),
            surrogate_module._FARFIELD_AXIS_SCHEMA)


class EngineLobeSelfFalsificationTestCase(TestCase):
    """Prove the engine-backed lobe gates can FAIL (this suite can go red).

    The node-exactness and interior-accuracy gates above would be decoration
    if they could not distinguish a correct served envelope from a wrong one.
    Here a coordinate mismatch (serving one source, comparing to the engine at
    another) blows past ``_NODE_EXACT_TOL``, and a deliberately corrupted
    reconstruction blows past the chart's held-out eps -- so a green pass is
    evidence.  (The schema-refusal gate is self-falsifying by construction:
    each ``assertRaises`` above requires an actual refusal.)
    """

    def _first_served_node(self, fixture: _EngineLobeFixture
                           ) -> tuple[float, float, float]:
        """A ``(gamma, rho_lobe, theta_local)`` node the chart truly serves."""
        for gamma in fixture.chart.gamma_grid:
            for rho_lobe in fixture.chart.rho_lobe_grid:
                for theta_local in fixture.chart.theta_local_grid:
                    served, error, _count = _served_minus_engine(
                        fixture, float(gamma), float(rho_lobe),
                        float(theta_local))
                    if served and error is not None:
                        return float(gamma), float(rho_lobe), float(theta_local)
        self.fail('no served node found; the fixture is degenerate')
        raise AssertionError  # unreachable, for type-checkers

    def test_node_exactness_gate_has_teeth(self) -> None:
        """A source mismatch breaks node reproduction past the tolerance.

        Positive control: serving AT a node reproduces the engine there to
        ``<= _NODE_EXACT_TOL``.  Falsification: the SAME served envelope
        compared to the engine at a source shifted by a quarter box-width in
        ``rho_lobe`` exceeds the tolerance -- so the node-exactness gate would
        catch a coordinate defect of that size.
        """
        fixture = _engine_lobe_fixture()
        gamma, rho_lobe, theta_local = self._first_served_node(fixture)
        y1, y2 = _interior_eigenframe_source(
            fixture.lobe, rho_lobe, theta_local)
        partition = _engine_partition(gamma, y1, y2, fixture.w_grid)
        env_node = np.asarray(partition.envelope)
        denom = float(np.max(np.abs(env_node))) or 1.0
        emu_node, served, _def = fixture.surrogate.serve(
            fixture.w_grid, gamma=gamma, y1=y1, y2=y2, beta=0.0,
            eta=partition.caustic_distance, theta=partition.critical_theta,
            image_count=int(partition.real_mask.sum()))
        self.assertTrue(served)
        err_control = float(np.max(np.abs(emu_node - env_node)) / denom)
        self.assertLessEqual(err_control, _NODE_EXACT_TOL,
                             'positive control: node must reproduce engine')

        # Shifted engine reference at a source a quarter box-width away.
        y1s, y2s = _interior_eigenframe_source(
            fixture.lobe, rho_lobe + 0.25 * fixture.half[0], theta_local)
        part_shift = _engine_partition(gamma, y1s, y2s, fixture.w_grid)
        self.assertIsNotNone(part_shift, 'shifted source unexpectedly refused')
        env_shift = np.asarray(part_shift.envelope)
        err_bad = float(np.max(np.abs(emu_node - env_shift)) / denom)
        self.assertGreater(
            err_bad, _NODE_EXACT_TOL,
            'a source mismatch must break node-exactness; the gate would be '
            'vacuous otherwise')

    def test_interior_gate_has_teeth(self) -> None:
        """A corrupted reconstruction exceeds the chart held-out eps.

        Scaling the served envelope by 1.5 (a grossly wrong reconstruction)
        drives the F-normalised engine difference above ``heldout_eps`` -- so
        the interior accuracy gate would reject a mis-reconstruction, not just
        rubber-stamp anything served.
        """
        fixture = _engine_lobe_fixture()
        rho_c, theta_c = fixture.box_center
        gamma = 0.5 * (_ENGINE_BAND[0] + _ENGINE_BAND[1])
        y1, y2 = _interior_eigenframe_source(fixture.lobe, rho_c, theta_c)
        partition = _engine_partition(gamma, y1, y2, fixture.w_grid)
        env_true = np.asarray(partition.envelope)
        denom = float(np.max(np.abs(env_true))) or 1.0
        emu, served, _def = fixture.surrogate.serve(
            fixture.w_grid, gamma=gamma, y1=y1, y2=y2, beta=0.0,
            eta=partition.caustic_distance, theta=partition.critical_theta,
            image_count=int(partition.real_mask.sum()))
        self.assertTrue(served)
        err_control = float(np.max(np.abs(emu - env_true)) / denom)
        self.assertLessEqual(err_control, fixture.heldout_eps,
                             'positive control: interior serve within eps')
        err_corrupt = float(np.max(np.abs(1.5 * emu - env_true)) / denom)
        self.assertGreater(
            err_corrupt, fixture.heldout_eps,
            'a 50%-wrong reconstruction must exceed the held-out eps; the '
            'interior gate would be vacuous otherwise')



# ---------------------------------------------------------------------------
# Positive-parity (gamma < 1) golden-value regression (acceptance #5).
#
# A committed, history-free fixture: the constants below were generated ONCE
# (by serving a small synthetic positive-parity `FarFieldChart` at a fixed
# source and hashing its saved artifact) and are FROZEN here.  The test
# rebuilds the SAME chart, serves the SAME inputs, and asserts the served
# complex envelope bits and the saved-artifact content digest EQUAL the frozen
# constants.  It imports nothing from HEAD, compares against no self-recomputed
# oracle, and touches no engine -- so it stays meaningful across every future
# refactor: a change in the far-field spline fit, the caustic-fixed source map
# (`_to_caustic_fixed` / `_rotate_to_eigenframe`), the reconstruction, or the
# npz record shape flips it RED with a frozen witness.
#
# The npz digest is taken over the LOADED array contents in sorted-key order
# (name + dtype + shape + bytes), NOT the raw ``.npz`` bytes: numpy's
# `np.savez` writes a ZIP whose member timestamps are not reproducible, so a
# raw-byte hash would be non-deterministic; the content digest is stable
# run-to-run (verified) while still detecting any change to a stored array,
# axis, knot vector, or meta string.

#: Frozen positive-parity (gamma < 1) shear for the golden serve; away from
#: ``gamma = 1`` by more than `_GAMMA_GUARD_BAND` so `select_chart` does not
#: decline on the parity guard band.
_POS_GAMMA: float = 0.6

#: Frozen synthetic far-field training axes.  ``rho_grid`` straddles the
#: caustic (``rho = 1``); ``theta_c_grid`` spans the full ``(-pi, pi]`` circle;
#: each axis has >= 4 nodes for the cubic tensor spline.
_POS_GAMMA_GRID: np.ndarray = np.linspace(0.5, 0.7, 4)
_POS_RHO_GRID: np.ndarray = np.linspace(0.2, 1.8, 5)
_POS_THETA_C_GRID: np.ndarray = np.linspace(-np.pi, np.pi, 6)
_POS_LOG_W_GRID: np.ndarray = np.linspace(-2.0, 1.0, 4)

#: Frozen region labels for the synthetic positive-parity chart (2-image
#: exterior region, macro-image parity ``+1``).
_POS_IMAGE_COUNT: int = 2
_POS_PARITY: int = 1

#: Frozen golden serve inputs (shear-frame source, orientation, caustic
#: distance, gauge angle, frequencies).  ``eta`` sits above the default
#: caustic floor so the far-field priority gate serves; ``theta`` is the gauge
#: angle (far-field serve ignores it).  The source maps to caustic-fixed
#: ``(rho ~ 1.233, theta_c ~ 0.118)`` -- comfortably inside the grid.
_POS_BETA: float = 0.3
_POS_ETA: float = 0.3
_POS_THETA: float = 0.7
_POS_Y1: float = 0.9
_POS_Y2: float = 0.4
_POS_W_ARRAY: np.ndarray = np.array([0.6, 1.0, 1.7])

#: Frozen golden served envelope, as exact ``float.hex()`` (real, imag) pairs
#: so the fixture round-trips to the last bit and the comparison is BIT-EXACT.
_POS_GOLDEN_ENVELOPE_HEX: tuple[tuple[str, str], ...] = (
    ('0x1.254b15681a696p-1', '-0x1.dfc2fe33bb3a0p-2'),
    ('0x1.08d197ebbefbap-1', '-0x1.c7dee836f60efp-2'),
    ('0x1.dc586b9d027f6p-2', '-0x1.b0503c20aa997p-2'),
)

#: Frozen SHA-256 content digest of the saved gamma<1 surrogate artifact
#: (sorted-key hash of the loaded arrays; see the section note above).
_POS_GOLDEN_NPZ_DIGEST: str = (
    '581d1a355eca18ccdd5fc3da658f7cbc72f73388b4318b9a14dd1931d892a0b2')


def _positive_golden_envelope() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic (real, imag) envelope tensor for the golden chart.

    A smooth analytic function of the four training axes -- reproduced here
    VERBATIM from the one-shot generator so the fitted spline (and hence every
    served bit) is fully determined by this file, not by any engine call.
    """
    log_w = _POS_LOG_W_GRID[:, None, None, None]
    gamma = _POS_GAMMA_GRID[None, :, None, None]
    rho = _POS_RHO_GRID[None, None, :, None]
    theta_c = _POS_THETA_C_GRID[None, None, None, :]
    env_real = (np.cos(1.3 * theta_c + 0.7 * rho)
                * np.exp(-0.2 * log_w) * (1.0 + 0.1 * gamma))
    env_imag = (np.sin(0.9 * theta_c - 0.5 * rho)
                * np.exp(-0.1 * log_w) * (0.8 + 0.2 * gamma))
    return env_real, env_imag


def _positive_golden_surrogate() -> surrogate_module.LensAmplificationSurrogate:
    """The frozen synthetic positive-parity (gamma < 1) served surrogate."""
    env_real, env_imag = _positive_golden_envelope()
    chart = surrogate_module.FarFieldChart.from_values(
        gamma_grid=_POS_GAMMA_GRID, rho_grid=_POS_RHO_GRID,
        theta_c_grid=_POS_THETA_C_GRID, log_w_grid=_POS_LOG_W_GRID,
        envelope_real=env_real, envelope_imag=env_imag,
        image_count=_POS_IMAGE_COUNT, parity=_POS_PARITY)
    return surrogate_module.LensAmplificationSurrogate(
        [chart], {'schema': 'pos-golden'})


def _golden_envelope_array() -> np.ndarray:
    """Rebuild the frozen golden envelope from its exact hex pairs."""
    return np.array(
        [complex(float.fromhex(re), float.fromhex(im))
         for re, im in _POS_GOLDEN_ENVELOPE_HEX], dtype=complex)


def _npz_content_digest(path: pathlib.Path) -> str:
    """SHA-256 over a saved surrogate's LOADED arrays in sorted-key order.

    Hashes ``name + dtype + shape + bytes`` for every array in the ``.npz``,
    sorted by key -- a representation-stable digest that is reproducible
    run-to-run (unlike the raw ZIP bytes, whose member timestamps vary) yet
    changes if any stored array, axis, knot vector, or meta string changes.
    """
    hasher = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for key in sorted(data.files):
            arr = np.ascontiguousarray(data[key])
            hasher.update(key.encode())
            hasher.update(str(arr.dtype).encode())
            hasher.update(str(arr.shape).encode())
            hasher.update(arr.tobytes())
    return hasher.hexdigest()

class PositiveParityGoldenTestCase(LobeTestCase):
    """Acceptance #5: committed gamma<1 served bits + npz digest (frozen).

    Rebuilds the frozen synthetic positive-parity `FarFieldChart`, serves the
    frozen source, and asserts (a) the served complex envelope equals the
    committed golden bits BIT-FOR-BIT and (b) the saved-artifact content digest
    equals the committed digest.  Neither assertion recomputes an oracle nor
    imports HEAD -- the golden constants are literals baked into this file, so
    the test remains a valid regression forever.
    """

    def test_served_envelope_matches_committed_golden_bits(self) -> None:
        """Serving the frozen source reproduces the committed envelope bits."""
        surrogate = _positive_golden_surrogate()
        emulated, served, definition = surrogate.serve(
            _POS_W_ARRAY, gamma=_POS_GAMMA, y1=_POS_Y1, y2=_POS_Y2,
            beta=_POS_BETA, eta=_POS_ETA, theta=_POS_THETA,
            image_count=_POS_IMAGE_COUNT)
        self.n_checks += 1
        self.assertTrue(served,
                        'precondition: the frozen positive-parity source is '
                        'served by the far-field chart')
        self.assertEqual(
            definition, surrogate_module._FARFIELD_ENVELOPE_DEFINITION,
            'a positive-parity far-field chart serves the far-field kernel-sum '
            'definition')
        golden = _golden_envelope_array()
        self.assertEqual(emulated.shape, golden.shape)
        self.assertEqual(
            np.asarray(emulated, dtype=complex).tobytes(), golden.tobytes(),
            'served envelope departed from the committed golden bits; the '
            'positive-parity serve path changed (spline fit, caustic-fixed '
            'source map, or reconstruction)')

    def test_saved_surrogate_content_digest_matches_committed(self) -> None:
        """Saving the frozen surrogate reproduces the committed npz digest."""
        surrogate = _positive_golden_surrogate()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'positive_golden.npz'
            surrogate.save(path)
            digest = _npz_content_digest(path)
        self.n_checks += 1
        self.assertEqual(
            digest, _POS_GOLDEN_NPZ_DIGEST,
            'saved gamma<1 surrogate content digest changed; a stored array, '
            'axis, knot vector, or meta string differs from the committed '
            'artifact')

    def test_digest_is_save_reproducible(self) -> None:
        """Two independent saves of the surrogate share one content digest.

        Guards the digest oracle itself: the content hash is representation-
        stable (numpy ZIP member timestamps do NOT leak into it), so a second
        save yields the identical digest -- otherwise the golden-digest gate
        would be flaky rather than a regression witness.
        """
        surrogate = _positive_golden_surrogate()
        with tempfile.TemporaryDirectory() as tmp:
            first = pathlib.Path(tmp) / 'a.npz'
            second = pathlib.Path(tmp) / 'b.npz'
            surrogate.save(first)
            surrogate.save(second)
            self.n_checks += 1
            self.assertEqual(_npz_content_digest(first),
                             _npz_content_digest(second),
                             'the content digest must be save-reproducible')


class PositiveParityGoldenSelfFalsificationTestCase(TestCase):
    """Prove the golden gates can FAIL (this block can go red).

    A frozen golden comparison is only a regression witness if a genuine change
    would break it.  Here a perturbed source and a perturbed envelope tensor
    are each shown to move the served bits / the saved digest off the committed
    constants -- so a green pass is evidence the frozen computation is intact,
    not that the equality is vacuous.
    """

    def test_shifted_source_breaks_golden_bits(self) -> None:
        """Serving a nudged source no longer matches the committed bits."""
        surrogate = _positive_golden_surrogate()
        emulated, served, _definition = surrogate.serve(
            _POS_W_ARRAY, gamma=_POS_GAMMA, y1=_POS_Y1 + 0.05, y2=_POS_Y2,
            beta=_POS_BETA, eta=_POS_ETA, theta=_POS_THETA,
            image_count=_POS_IMAGE_COUNT)
        self.assertTrue(served, 'the nudged source is still served')
        self.assertNotEqual(
            np.asarray(emulated, dtype=complex).tobytes(),
            _golden_envelope_array().tobytes(),
            'a shifted source must move the served bits; the golden-bit gate '
            'would be vacuous otherwise')

    def test_perturbed_envelope_breaks_golden_digest(self) -> None:
        """A one-node envelope tweak changes both the bits and the digest."""
        env_real, env_imag = _positive_golden_envelope()
        env_real = env_real.copy()
        env_real[0, 0, 0, 0] += 1.0  # perturb a single training sample
        chart = surrogate_module.FarFieldChart.from_values(
            gamma_grid=_POS_GAMMA_GRID, rho_grid=_POS_RHO_GRID,
            theta_c_grid=_POS_THETA_C_GRID, log_w_grid=_POS_LOG_W_GRID,
            envelope_real=env_real, envelope_imag=env_imag,
            image_count=_POS_IMAGE_COUNT, parity=_POS_PARITY)
        surrogate = surrogate_module.LensAmplificationSurrogate(
            [chart], {'schema': 'pos-golden'})
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'perturbed.npz'
            surrogate.save(path)
            digest = _npz_content_digest(path)
        self.assertNotEqual(
            digest, _POS_GOLDEN_NPZ_DIGEST,
            'a perturbed envelope must change the saved digest; the digest '
            'gate would be vacuous otherwise')



if __name__ == '__main__':
    main()
