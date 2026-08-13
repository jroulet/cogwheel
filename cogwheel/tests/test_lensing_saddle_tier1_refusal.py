"""
Tier-1 far-from-caustic macro-saddle analytic serve: REFUSAL + wiring.

Build: tier-1 saddle-analytic serve rung (WP-1) + census wiring (WP-2).

Companion to ``test_lensing_saddle_tier1_accuracy.py`` (SHARD A, which
certifies the SERVED value on the rung's contract domain).  This suite
(SHARD B) certifies the OTHER half of the contract: that the rung
*declines* rather than serving wrongly, that the resolvability variable is
the load-bearing gate, and that the census attributes the rung correctly
against the shared single-source-of-truth predicate.

THE GATE UNDER TEST
-------------------
`likelihood._saddle_farfield_analytic_serves(real_delays, w_lo, rho)` is
the SINGLE SOURCE OF TRUTH serve gate, called by BOTH the live serve rung
(`likelihood.LensedRelativeBinningLikelihood._saddle_farfield_analytic`)
and the census band-splitting (`surrogate_census.characterize_sample`).
It has TWO independent terms, BOTH required:

1. Caustic proximity: ``rho >= _SADDLE_FARFIELD_RHO_FLOOR`` (2.0), where
   ``rho`` is the isotropic caustic-relative coordinate
   (``ppgo_map.caustic_rho``) computed by the caller from the SAME
   ``gamma``/``kappa``/``|y|`` used to build the geometry partition.
2. Resolvability: there are at least two REAL image delays AND the
   narrowest positive pairwise delay gap is resolved at the band floor,
   ``w_lo * min_delta_tau >= RHO_END`` (RHO_END = 4.0).

Below either threshold the rung MUST decline (return ``None``) so the
caller falls through to the exact seed engine -- byte-identical to HEAD.
This suite's companion (`test_lensing_saddle_tier1_accuracy.py`)
certifies the rho-floor term; this file (SHARD B) exercises the
resolvability term and the wiring/census attribution, using fixtures
whose ``rho`` already clears the floor so refusal/admission here is
attributable to resolvability.

WHAT THIS SUITE CERTIFIES
-------------------------
1. REFUSAL (`SaddleFarfieldAnalyticRefusalTestCase`).  A known-bad
   far-from-caustic-but-UNRESOLVED macro saddle (``gamma = 1.519``,
   ``|y| = 1.787``, measured ``n_real = 2``, ``min_delta_tau ~ 0.0618``,
   ``rho ~ 0.933`` -- also below the rho floor, so this fixture is
   refused by BOTH gate terms) evaluated at a band floor ``w_lo = 24``
   gives ``w_lo * mdt ~ 1.48 < RHO_END``.  The rung returns ``None`` BY
   NAME (not by exception), and the dispatch in
   ``_amplification_coefficients`` therefore proceeds to the seed/exact
   path.  A cleanly resolvable admitted saddle (``gamma = 1.5``,
   ``y = (4.6, 0)``, measured ``mdt ~ 16.44``, ``rho ~ 2.424 >=
   _SADDLE_FARFIELD_RHO_FLOOR``) is the positive control that returns a
   tuple.
2. SELF-FALSIFICATION (`GateResolvabilitySelfFalsificationTestCase`,
   mirrors the ``test_lensing_ppgo_above_ceiling.py`` teeth pattern).
   Inflating ``RHO_END`` makes the previously-admitted source refuse.
   Zeroing ``RHO_END`` wrongly admits an UNRESOLVED source whose ``rho``
   already clears the floor (isolating the resolvability term from the
   rho-floor term -- the known-bad fixture above is unsuitable for this
   specific mutation because its ``rho`` is ALSO below the floor, so
   zeroing RHO_END alone would not flip it).  A band-floor sweep flips the
   verdict at exactly ``w_lo * mdt == RHO_END`` (via ``np.nextafter``).
   Together these prove the resolvability variable
   ``w_lo * min_delta_tau`` versus ``RHO_END`` -- not something
   incidental -- decides admission once the rho floor is cleared.
3. CENSUS (`CensusAttributionTestCase`).  ``characterize_sample``
   attributes the resolvable admitted source to the new
   ``'saddle-farfield-analytic'`` served-cause label (out of the
   exact-engine / out-of-box bucket) and leaves the unresolved known-bad
   source in the fall-through bucket (``'born'``).  For BOTH sources the
   census verdict agrees with a direct call to the shared predicate --
   proving the served set and the counted set can never skew.

DIAGNOSTIC
----------
`test_resolvability_step_at_rho_end` realizes the brief's diagnostic (a
scatter of ``w_lo * min_delta_tau`` vs served/refused would show a clean
step at RHO_END): the method transitions refuse -> serve across the
single band floor ``w_lo* = RHO_END / min_delta_tau`` and nowhere else.

WHY STUB-BASED METHOD INVOCATION
--------------------------------
The refusal path returns ``None`` before touching any ``self`` method, so
a bare ``MagicMock`` stub suffices; the admission path additionally calls
``self._reduce_dense_kernels`` / ``self._image_delays``, mocked to
sentinels so the real gate + real ``reconstruct_farfield`` run without a
fully-constructed likelihood.  The dispatch test drives the real
``_amplification_coefficients`` with the real gate (via ``side_effect``
delegating to the unbound method) and stubs only the leaf engine calls,
so the branch structure under test is genuinely exercised.

COST ARITHMETIC (fast tier; hard ceilings 60 s/test, 5 min/file)
----------------------------------------------------------------
Every test is a w-independent ``geometry_partition`` (~20 ms) plus, on
admission, one cheap ``reconstruct_farfield`` over a <= 12-node grid at
w <= ~100 (double-double band, no engine oracle).  The census tests do
two ``characterize_sample`` calls (geometry-only, ~40 ms).  No exact
oracle, no mpmath, no training.  Total file well under 10 s.
"""
from __future__ import annotations

import math
from unittest import TestCase, main, mock
from unittest.mock import MagicMock

import numpy as np

from cogwheel.lensing import surrogate_census as census
from cogwheel.lensing.chang_refsdal import ChangRefsdalChannels
from cogwheel.lensing.chang_refsdal.operator import RHO_END
import cogwheel.lensing.likelihood as likmod
from cogwheel.lensing.likelihood import (
    LensedRelativeBinningLikelihood, dimensionless_frequency,
    _saddle_farfield_analytic_serves, _SADDLE_FARFIELD_RHO_FLOOR)
from cogwheel.lensing.ppgo_map import caustic_rho
from cogwheel.lensing.surrogate import LensAmplificationSurrogate, TubeChart


# ======================================================================
# Shared test constants
# ======================================================================

#: Known-bad fixture: a far-from-caustic macro saddle whose real image
#: pair is NOT resolved at the chosen band floor.  Measured on the
#: geometry partition: n_real = 2, min_delta_tau ~ 0.0618, |y| = 1.787
#: (matches the brief).  Measured production rho (isotropic
#: ppgo_map.caustic_rho gauge) ~ 0.933 -- ALSO below
#: _SADDLE_FARFIELD_RHO_FLOOR (2.0), so this fixture is refused by BOTH
#: gate terms.  Fine for every test that only asserts refusal; NOT used
#: for the RHO_END-zeroing self-falsification test below (see
#: AD_UNRESOLVED_W_LO), which needs a fixture refused by resolvability
#: ALONE so that mutation is isolated.
KB_GAMMA = 1.519
KB_Y = (1.707, 0.528)

#: Band floor at which the known-bad source is REFUSED
#: (w_lo * mdt ~ 24 * 0.0618 ~ 1.48 < RHO_END = 4.0).  The brief's "w ~ 24".
KB_REFUSE_W_LO = 24.0

#: Admitted fixture: a cleanly resolvable, well-separated macro saddle
#: whose production rho clears the caustic-proximity floor.  Measured
#: (isotropic ppgo_map.caustic_rho gauge, same gauge the production rung
#: uses): rho ~ 2.424 >= _SADDLE_FARFIELD_RHO_FLOOR (2.0); n_real = 2,
#: min_delta_tau ~ 16.44.  The positive control for the refusal tests.
AD_GAMMA = 1.5
AD_Y = (4.6, 0.0)

#: Band floor at which the admitted source is SERVED
#: (w_lo * mdt ~ 8 * 16.44 ~ 131.5 >= RHO_END).
AD_ADMIT_W_LO = 8.0

#: Band floor at which the SAME rho-clearing source (AD_GAMMA, AD_Y) is
#: UNRESOLVED (w_lo * mdt ~ 0.2 * 16.44 ~ 3.29 < RHO_END = 4.0).  Because
#: rho already clears the floor, refusal here is attributable to
#: resolvability alone -- the isolated fixture for the RHO_END-zeroing
#: self-falsification test.
AD_UNRESOLVED_W_LO = 0.2

#: Redshifted-frame lens mass [Msun] used to map the census/dispatch
#: frequency grids to the target dimensionless band floors.  Any positive
#: value works; admission depends only on w_lo (via f_grid), not the mass.
M_LENS_MSUN = 100.0

#: Inflated RHO_END for the self-falsification mutation: so large that the
#: admitted source's resolved pair (w_lo * mdt ~ 25.9) now refuses.
RHO_END_INFLATED = 1.0e9


# ======================================================================
# Anti-vacuity base TestCase
# ======================================================================

class _SaddleTier1RefusalTestCase(TestCase):
    """Base carrying an anti-vacuity guard.

    Every concrete test increments ``self._checks`` for each real
    comparison it makes; ``tearDown`` FAILS if a test silently made zero
    comparisons, so a suite that stops exercising the gate (e.g. a fixture
    that no longer produces a 2-image saddle, or a predicate import that
    goes stale) reads RED rather than green-because-empty.
    """

    def setUp(self) -> None:
        super().setUp()
        self._checks = 0

    def tearDown(self) -> None:
        super().tearDown()
        self.assertGreater(
            self._checks, 0,
            'anti-vacuity: the test made zero gate comparisons -- it '
            'certified nothing.')

    # -- shared helpers -------------------------------------------------

    @staticmethod
    def _lens(gamma: float, y: tuple[float, float]) -> dict:
        """Minimal lens dict consumed by the tier-1 rung (beta=kappa=0)."""
        return {'gamma': float(gamma), 'y1': float(y[0]), 'y2': float(y[1]),
                'beta': 0.0, 'kappa': 0.0,
                'm_lens_msun': M_LENS_MSUN, 'z_lens': 0.0}

    @staticmethod
    def _dense_w(w_lo: float, w_hi: float | None = None,
                 n: int = 12) -> np.ndarray:
        """Log-spaced dense w grid with an exact floor ``w_lo``."""
        w_hi = w_hi if w_hi is not None else w_lo * 2.0
        grid = np.geomspace(float(w_lo), float(w_hi), n)
        grid[0] = float(w_lo)  # pin the floor exactly (gate reads w.min())
        return grid

    @classmethod
    def _leaf_stub(cls) -> MagicMock:
        """Stub whose reduce/delay leaves return sentinels (admission path)."""
        stub = MagicMock()
        stub._reduce_dense_kernels.return_value = ('k0', 'k1')
        stub._image_delays.return_value = 'delays'
        return stub

    @classmethod
    def _serve(cls, gamma: float, y: tuple[float, float],
               w_lo: float, w_hi: float | None = None):
        """Invoke the real tier-1 rung on a stub; return its tuple-or-None."""
        dense_w = cls._dense_w(w_lo, w_hi)
        return LensedRelativeBinningLikelihood._saddle_farfield_analytic(
            cls._leaf_stub(), cls._lens(gamma, y), dense_w)

    @staticmethod
    def _real_delays(gamma: float, y: tuple[float, float]) -> np.ndarray:
        """Sorted REAL image Fermat delays for a source (w-independent)."""
        geom = ChangRefsdalChannels(np.array([2.0, 4.0])).geometry_partition(
            gamma=gamma, y=y, beta=0.0, kappa=0.0)
        real = np.asarray(geom.real_mask, dtype=bool)
        return np.sort(np.asarray(geom.delays)[real])

    @classmethod
    def _min_delta_tau(cls, gamma: float, y: tuple[float, float]) -> float:
        """Narrowest positive pairwise delay gap of the real image set."""
        delays = cls._real_delays(gamma, y)
        gaps = np.diff(delays)
        gaps = gaps[gaps > 0]
        return float(np.min(gaps))


# ======================================================================
# 1. REFUSAL
# ======================================================================

class SaddleFarfieldAnalyticRefusalTestCase(_SaddleTier1RefusalTestCase):
    """The rung declines the unresolved saddle rather than serving it."""

    def test_known_bad_returns_none_not_exception(self) -> None:
        """Unresolved far saddle: rung returns ``None`` (falls through by
        name), it does NOT raise."""
        try:
            served = self._serve(KB_GAMMA, KB_Y, KB_REFUSE_W_LO)
        except Exception as exc:  # pragma: no cover - failure path
            self.fail(f'rung raised {type(exc).__name__} instead of '
                      f'declining by name: {exc}')
        self.assertIsNone(
            served,
            'known-bad unresolved saddle must refuse (return None), '
            'not serve.')
        self._checks += 1

    def test_known_bad_is_two_image_and_refused_by_resolvability(self) -> None:
        """The fixture is not refused by too-few-images: it has >= 2 real
        images yet ``w_lo * mdt < RHO_END``, so the resolvability term
        alone would refuse it (its rho is ALSO below the floor, so both
        gate terms refuse this particular fixture -- see
        ``test_lensing_saddle_tier1_accuracy.py`` for a witness isolating
        the rho-floor term)."""
        delays = self._real_delays(KB_GAMMA, KB_Y)
        self.assertGreaterEqual(
            len(delays), 2,
            'known-bad fixture must be a 2-image saddle so the >=2-real '
            'clause passes and refusal is attributable to resolvability.')
        mdt = self._min_delta_tau(KB_GAMMA, KB_Y)
        self.assertLess(
            KB_REFUSE_W_LO * mdt, RHO_END,
            'known-bad band floor must leave the pair unresolved '
            '(w_lo*mdt < RHO_END).')
        self._checks += 1

    def test_admitted_control_returns_tuple(self) -> None:
        """Positive control: the resolvable saddle serves (returns a
        4-tuple whose last element is the real geometry partition)."""
        served = self._serve(AD_GAMMA, AD_Y, AD_ADMIT_W_LO)
        self.assertIsNotNone(
            served, 'resolvable far saddle must serve (return a tuple).')
        self.assertEqual(len(served), 4,
                         'serve tuple is (delays, k0, k1, partition).')
        self.assertEqual(type(served[3]).__name__,
                         'ChangRefsdalGeometryPartition')
        self._checks += 1

    def test_dispatch_falls_through_to_seed_path(self) -> None:
        """When the rung declines, ``_amplification_coefficients`` proceeds
        to the seed/exact path for the known-bad source."""
        lens = self._lens(KB_GAMMA, KB_Y)
        xi = float(dimensionless_frequency(1.0, lens['m_lens_msun'],
                                           lens['z_lens']))

        stub = MagicMock()
        stub.amplification_surrogate = None
        stub._lens_params.return_value = lens
        # dense_w floor 24 (refuses), max 48 (< W_CEILING_SCHWINGER_QD=150
        # so the ppGO above-ceiling intercept is skipped).
        stub._kernel_dense_f = np.geomspace(24.0, 48.0, 12) / xi
        # Real gate on a leaf stub -> returns None for the known-bad source.
        stub._saddle_farfield_analytic.side_effect = (
            lambda ln, dw: LensedRelativeBinningLikelihood
            ._saddle_farfield_analytic(self._leaf_stub(), ln, dw))
        stub._force_direct = True  # route straight to the direct/seed path
        stub._evaluate_envelope.return_value = ('part', 'env', 'ftot')
        stub._amplification_coefficients_direct.return_value = 'SEED_PATH'

        result = LensedRelativeBinningLikelihood._amplification_coefficients(
            stub, {'dummy': 1.0})

        self.assertTrue(stub._saddle_farfield_analytic.called,
                        'dispatch must consult the tier-1 rung.')
        self.assertTrue(
            stub._evaluate_envelope.called,
            'dispatch must reach the seed engine evaluation after the rung '
            'declines.')
        self.assertEqual(
            result, 'SEED_PATH',
            'dispatch must return the seed/direct path result for a refused '
            'saddle, not a rung serve.')
        self._checks += 1


# ======================================================================
# 2. SELF-FALSIFICATION (mirrors test_lensing_ppgo_above_ceiling.py teeth)
# ======================================================================

class GateResolvabilitySelfFalsificationTestCase(
        _SaddleTier1RefusalTestCase):
    """The resolvability variable ``w_lo * min_delta_tau`` vs ``RHO_END``
    is the load-bearing gate -- not image count, parity, or anything else.

    Each test MUTATES ``RHO_END`` (or steps the band floor across the
    threshold) and shows the admit/refuse verdict flips accordingly.  If
    admission were decided by an incidental variable, none of these
    mutations would change the outcome and the tests would fail.
    """

    def test_inflated_rho_end_refuses_previously_admitted(self) -> None:
        """Raising RHO_END above ``w_lo * mdt`` turns the served source into
        a refusal -- the gate reads RHO_END, not a fixed image count."""
        served_before = self._serve(AD_GAMMA, AD_Y, AD_ADMIT_W_LO)
        self.assertIsNotNone(served_before,
                             'control: the source serves at HEAD RHO_END.')
        with mock.patch.object(likmod, 'RHO_END', RHO_END_INFLATED):
            served_after = self._serve(AD_GAMMA, AD_Y, AD_ADMIT_W_LO)
        self.assertIsNone(
            served_after,
            'inflating RHO_END must make the previously-admitted source '
            'refuse; if not, resolvability is not the gate.')
        self._checks += 1

    def test_zeroed_rho_end_admits_known_bad(self) -> None:
        """Lowering the resolvability threshold to 0 wrongly admits an
        UNRESOLVED source whose rho already clears the floor -- proving the
        threshold is what refused it (isolated from the rho-floor term;
        the plain known-bad fixture is unsuitable here because its rho is
        ALSO below the floor, so zeroing RHO_END alone would not flip
        it)."""
        refused_before = self._serve(AD_GAMMA, AD_Y, AD_UNRESOLVED_W_LO)
        self.assertIsNone(
            refused_before,
            'control: the rho-clearing source refuses at HEAD RHO_END when '
            'the band floor leaves it unresolved.')
        with mock.patch.object(likmod, 'RHO_END', 0.0):
            served_after = self._serve(AD_GAMMA, AD_Y, AD_UNRESOLVED_W_LO)
        self.assertIsNotNone(
            served_after,
            'with the resolvability threshold at 0 the rho-clearing source '
            'is admitted -- confirming w_lo*mdt vs RHO_END is the '
            'resolvability gate, isolated from the rho-floor term.')
        self._checks += 1

    def test_resolvability_step_at_rho_end(self) -> None:
        """DIAGNOSTIC as a test: sweeping the band floor across
        ``w_lo* = RHO_END / min_delta_tau`` flips the verdict exactly once,
        a clean step -- refused just below, served at/above."""
        mdt = self._min_delta_tau(AD_GAMMA, AD_Y)
        w_star = RHO_END / mdt  # boundary band floor
        below = np.nextafter(w_star, 0.0)      # w_lo*mdt just < RHO_END
        at_or_above = np.nextafter(w_star, np.inf)  # just >= RHO_END

        served_below = self._serve(AD_GAMMA, AD_Y, below, below * 4.0)
        served_above = self._serve(AD_GAMMA, AD_Y, at_or_above,
                                   at_or_above * 4.0)
        self.assertIsNone(
            served_below,
            'just below the resolvability boundary the rung must refuse.')
        self.assertIsNotNone(
            served_above,
            'at/above the resolvability boundary the rung must serve.')
        self._checks += 1

    def test_suite_detects_a_gate_that_ignores_resolvability(self) -> None:
        """Explicit teeth: a broken predicate that always admits WOULD serve
        the known-bad source; the real predicate does not.  This proves the
        suite can go red if the gate stops reading resolvability."""
        # Broken gate: ignores rho and w_lo*mdt, admits any >=2-image saddle.
        broken = (lambda real_delays, w_lo, rho:
                  len(np.atleast_1d(real_delays)) >= 2)
        with mock.patch.object(likmod, '_saddle_farfield_analytic_serves',
                               broken):
            wrongly_served = self._serve(KB_GAMMA, KB_Y, KB_REFUSE_W_LO)
        self.assertIsNotNone(
            wrongly_served,
            'a resolvability-blind gate serves the known-bad source -- the '
            'suite would catch such a regression.')
        # And the real gate refuses the same source (contrast).
        self.assertIsNone(self._serve(KB_GAMMA, KB_Y, KB_REFUSE_W_LO))
        self._checks += 1


# ======================================================================
# 3. CENSUS attribution (shared predicate == census verdict)
# ======================================================================

def _nonmatching_surrogate() -> LensAmplificationSurrogate:
    """A surrogate with ONE astroid tube chart (gamma in [0.3, 0.5]) that
    never matches the gamma > 1 saddle fixtures.

    A surrogate needs at least one chart, so an empty list is rejected;
    this chart's gamma band excludes the saddle sources, so
    ``select_chart`` returns ``None`` and ``characterize_sample`` reaches
    the tier-1 saddle rung -- exactly the census branch under test.
    """
    gamma_grid = np.linspace(0.3, 0.5, 4)
    u_grid = np.linspace(math.sqrt(0.02), math.sqrt(0.05), 4)
    theta_grid = np.linspace(0.2, 1.2, 4)
    log_w_grid = np.linspace(math.log(2.0), math.log(60.0), 4)
    zeros = np.zeros((log_w_grid.size, gamma_grid.size, u_grid.size,
                      theta_grid.size))
    tube = TubeChart.from_values(
        gamma_grid=gamma_grid, u_grid=u_grid, theta_grid=theta_grid,
        log_w_grid=log_w_grid, envelope_real=zeros, envelope_imag=zeros,
        image_count=2, parity=1, eta_floor=0.02, eta_max=0.05,
        cusp_windows=())
    return LensAmplificationSurrogate([tube], {'chart_count': 1})


class CensusAttributionTestCase(_SaddleTier1RefusalTestCase):
    """`characterize_sample` labels the rung's served set correctly and
    agrees with the shared single-source-of-truth predicate.

    Both the census verdict and the direct predicate call read the SAME
    ``w_grid`` (built from the same ``f_grid`` and mass), so the two can
    never skew by construction; the tests assert the coupling holds.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.surrogate = _nonmatching_surrogate()

    def _census_inputs(self, w_lo: float, m_lens: float = M_LENS_MSUN):
        """f_grid whose census w-band floor equals ``w_lo`` (z_lens=0)."""
        xi = float(dimensionless_frequency(1.0, m_lens, 0.0))
        f_grid = np.geomspace(w_lo, w_lo * 2.0, 8) / xi
        return f_grid, m_lens

    def _record(self, gamma, y, w_lo):
        f_grid, m_lens = self._census_inputs(w_lo)
        return census.characterize_sample(
            self.surrogate, ChangRefsdalChannels, gamma=gamma,
            m_lens_msun=m_lens, y1=y[0], y2=y[1], f_grid=f_grid,
            dropped_slivers=())

    def _direct_predicate(self, gamma, y, w_lo):
        """Shared predicate on the SAME w-band the census uses.

        Computes ``rho`` the same way the production rung does: the
        isotropic ``ppgo_map.caustic_rho`` gauge from the candidate's own
        ``gamma``/``kappa``/``|y|``.
        """
        f_grid, m_lens = self._census_inputs(w_lo)
        w_grid = dimensionless_frequency(f_grid, m_lens, 0.0)
        geom = ChangRefsdalChannels(w_grid).geometry_partition(
            gamma=gamma, y=y, beta=0.0, kappa=0.0)
        real = np.asarray(geom.real_mask, dtype=bool)
        real_delays = np.asarray(geom.delays)[real]
        rho = caustic_rho(gamma, float(np.hypot(y[0], y[1])), kappa=0.0)
        return _saddle_farfield_analytic_serves(
            real_delays, float(w_grid.min()), rho)

    def test_admitted_source_labeled_saddle_farfield_analytic(self) -> None:
        """The resolvable saddle is served and attributed to the new
        'saddle-farfield-analytic' label (leaves the exact-engine bucket)."""
        record = self._record(AD_GAMMA, AD_Y, AD_ADMIT_W_LO)
        self.assertTrue(record.served,
                        'resolvable saddle must be recorded as served.')
        self.assertEqual(record.category, 'saddle-farfield-analytic')
        self.assertFalse(record.engine_refused,
                         'a served source is not an engine refusal.')
        self._checks += 1

    def test_known_bad_source_in_fallthrough_bucket(self) -> None:
        """The unresolved saddle is NOT served and lands in the
        fall-through bucket ('born'), not the served label."""
        record = self._record(KB_GAMMA, KB_Y, KB_REFUSE_W_LO)
        self.assertFalse(record.served,
                         'unresolved saddle must not be served by the rung.')
        self.assertNotEqual(record.category, 'saddle-farfield-analytic')
        self.assertEqual(record.category, 'born',
                         'gamma>1 2-image unresolved saddle falls through '
                         'to the born bucket.')
        self._checks += 1

    def test_census_agrees_with_shared_predicate_both_sources(self) -> None:
        """For BOTH sources the census served-flag equals a direct call to
        the shared predicate on the same band -- served and counted sets
        cannot skew."""
        for label, gamma, y, w_lo in (
                ('admitted', AD_GAMMA, AD_Y, AD_ADMIT_W_LO),
                ('known-bad', KB_GAMMA, KB_Y, KB_REFUSE_W_LO)):
            with self.subTest(source=label):
                census_served = self._record(gamma, y, w_lo).served
                predicate = self._direct_predicate(gamma, y, w_lo)
                self.assertEqual(
                    census_served, predicate,
                    f'{label}: census verdict must equal the shared '
                    f'resolvability predicate.')
                self._checks += 1


if __name__ == '__main__':
    main()
