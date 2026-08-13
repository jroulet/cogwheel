"""
Tier-1 far-from-caustic macro-saddle analytic serve accuracy.

Build: tier-1 saddle-analytic serve rung (WP-1) + census wiring (WP-2).

The tier-1 rung (`likelihood._saddle_farfield_analytic`, gamma > 1)
serves the resolvable far-from-caustic macro saddle from the switched
analytic channels with a ZERO envelope -- no engine call, no
`fold_ppgo_correction`.  Its admission predicate is the SINGLE-SOURCE-OF-
TRUTH gate `_saddle_farfield_analytic_serves` (also used by the WP-2
census band-splitting), which requires BOTH terms:
(1) caustic proximity ``rho >= _SADDLE_FARFIELD_RHO_FLOOR`` (2.0), where
``rho`` is the authoritative isotropic ``ppgo_map.caustic_rho`` gauge, and
(2) resolvability ``n_real >= 2 AND w_lo * min_delta_tau >= RHO_END``
(RHO_END = 4.0).

WHAT THIS SUITE CERTIFIES
-------------------------
The served amplification F_serve (the zero-envelope FARFIELD_KERNEL_SUM
reconstruction, exactly as the rung builds it) is compared POINTWISE to
the EXACT Schwinger engine total F_exact = ``partition.exact_total``:
``err = |F_serve - F_exact| / |F_exact|``.  The oracle is the exact
Schwinger path -- ``operator.F_op`` DIVERGES for the macro saddle and is
NEVER used here.  Certification is confined to the CHEAP double-double
band w <= 60 (~0.2 s/engine call); the mpmath band 60 < w <= 148
(~85-120 s/call) is FORBIDDEN.

CENTRAL PHYSICAL FINDING (why the gate has two terms)
-------------------------------------------------------------------
Tier-1 accuracy is governed by CAUSTIC PROXIMITY (rho = |y| / caustic_reach,
the isotropic ``ppgo_map.caustic_rho`` gauge), NOT by resolvability alone.
The zero-envelope approximation is exact only where the far-field residual
envelope is negligible, i.e. far from the caustic.  Resolvability alone
(``w_lo * min_delta_tau >= RHO_END``) is a LEAKY PROXY for "far-from-
caustic": near a cusp the two exterior images can be well-separated in
delay (resolvability passes) while the source still sits close to the
caustic (envelope large).  The production gate therefore ADDITIONALLY
requires ``rho >= _SADDLE_FARFIELD_RHO_FLOOR`` so a resolvable-but-near-
caustic source is refused and falls through to the exact Schwinger engine
(or, eventually, the deferred tier-2 chart).

Consequently this suite certifies the rung's ACTUAL contract domain --
FAR-FROM-CAUSTIC (rho >= RHO_FAR) AND resolvable -- where p90 <= 1e-3 is
comfortably met (measured p90 ~ 5e-5, max ~ 7e-4 at RHO_FAR = 2.0).  The
leakiness of the resolvability term IN ISOLATION is EXPOSED, not hidden,
by the pinned worst case and the near-caustic witness test: both are
resolvable by the OLD resolvability-only predicate (which would have
served them with err > 1e-2) yet are correctly REFUSED by the current
two-term gate.  This proves the rho floor is load-bearing, not cosmetic.

TOLERANCES
----------
- P90_TOL = 1e-3 : spec's population accuracy bound (met with ~20x
  headroom on the far-from-caustic domain).
- OUTLIER_TOL = 1e-2 : spec's permitted loose outlier guard (met with
  ~14x headroom).
- LEAK_MIN_ERR = 1e-2 : the near-caustic gate-admitted served error must
  EXCEED this, proving resolvability does not imply accuracy.

COST ARITHMETIC (fast tier; hard ceilings 60 s/test, 5 min/file)
----------------------------------------------------------------
- Far-from-caustic population (setUpClass): ~90 draws, each a
  w-independent geometry_partition (~20 ms); the >= 20 admitted sources
  additionally get one exact-engine eval over a 24-pt w<=60 grid
  (~0.2 s) + one cheap reconstruction.  ~90*0.02 + 22*0.2 ~ 6.5 s.
- Pinned worst case: 1 refused (geometry only) + 1 served
  (serve + exact ~0.2 s).  < 1 s.
- Caustic-proximity ordering: 2 sources (far + near) serve + exact,
  plus a short deterministic near-source search.  < 2 s.
- Self-falsification: 1-2 sources serve + exact.  < 1 s.
Total file < 15 s.
"""
from __future__ import annotations

import math
import pathlib
from unittest import TestCase, main

import numpy as np

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal.channels import (
    ChangRefsdalChannels, FARFIELD_KERNEL_SUM, reconstruct_farfield)
from cogwheel.lensing.chang_refsdal.operator import RHO_END
from cogwheel.lensing.likelihood import (
    _SADDLE_FARFIELD_RHO_FLOOR, _saddle_farfield_analytic_serves)
from cogwheel.lensing.ppgo_map import caustic_geometry, caustic_rho


# ======================================================================
# Output directory for diagnostic plots
# ======================================================================
_OUTPUT_DIR = pathlib.Path(__file__).parent / 'output'


# ======================================================================
# Shared test constants
# ======================================================================

#: Cheap-band dimensionless frequency floor for the certified population.
#: The whole band [W_FLOOR, W_CEIL] stays inside the double-double engine
#: domain (w <= 60), so every exact oracle eval is ~0.2 s.
W_FLOOR = 8.0

#: Cheap-band ceiling.  Engine evals at w > 60 use the mpmath QD path
#: (~85-120 s/call) and are FORBIDDEN in this fast-tier suite.
W_CEIL = 60.0

#: Number of log-spaced w nodes across [W_FLOOR, W_CEIL].
_N_W = 24

#: Saddle band explored for the far-from-caustic population (gamma > 1).
GAMMA_LO = 1.15
GAMMA_HI = 1.9

#: Far-from-caustic draw floor: rho = |y| / caustic_reach >= RHO_FAR
#: defines the rung's actual contract domain.  Bound to the production
#: floor (single source of truth) so the admitted population is exactly
#: the gate-admitted domain; measured to meet p90 <= 1e-3 with ~20x
#: headroom there.
RHO_FAR = _SADDLE_FARFIELD_RHO_FLOOR

#: Upper rho bound for the draw (keeps sources at astronomically plausible
#: separations and the engine well-conditioned).
RHO_HI = 3.5

#: Number of admitted far-from-caustic sources to certify (spec: >= 20).
N_ADMITTED = 22

#: Fixed RNG seed for reproducible source draws (spec: seed 42).
SEED = 42

#: Spec population accuracy bound: 90th percentile of pointwise err.
P90_TOL = 1e-3

#: Spec permitted loose outlier guard on the pointwise maximum.
OUTLIER_TOL = 1e-2

# --- Pinned worst measured case from the brief -------------------------
#: Worst case reported in the brief: a near-caustic n_real=2 saddle
#: source.  Its production rho (the isotropic ``caustic_rho`` gauge) is
#: ~0.73 -- WELL BELOW the rho floor -- so the two-term gate refuses it
#: at ANY w_lo, regardless of resolvability.
PIN_GAMMA = 1.5859
PIN_Y = (-1.1208, -0.9002)

#: Band floor at which the pin is REFUSED by resolvability ALONE
#: (w_lo*mdt = 3.73 < RHO_END), i.e. both gate terms independently refuse.
PIN_REFUSE_W_LO = RHO_END  # 4.0

#: Band floor at which resolvability ALONE would ADMIT the pin
#: (w_lo*mdt = 7.47 >= RHO_END) -- the OLD resolvability-only gate would
#: have served it wrongly (err ~ 0.09) -- but the rho floor now refuses
#: it regardless.
PIN_LEAK_W_LO = 8.0

#: If the near-caustic pin were served anyway (bypassing the gate, as the
#: OLD resolvability-only gate would have), the error must EXCEED this,
#: proving the rho floor is load-bearing and not cosmetic.
LEAK_MIN_ERR = 1e-2


# ======================================================================
# Helpers
# ======================================================================

def _polar_source(rho: float, angle: float, gamma: float,
                  *, kappa: float = 0.0) -> np.ndarray:
    """Build a source position from caustic-relative rho and polar angle.

    Uses the ISOTROPIC max-reach gauge (``ppgo_map.caustic_geometry``) --
    the SAME gauge the production ``caustic_rho`` converter uses to
    compute the gate's rho argument -- so a source built at ``rho=R``
    satisfies ``caustic_rho(gamma, |y|, kappa) == R`` by construction (up
    to floating point).  Unlike the retired directional
    ``geometry.r_caustic`` gauge, ``angle`` only sets the polar DIRECTION
    of the placement; it does not affect the caustic-relative distance.

    Raises ``geometry.LensDomainError`` if ``caustic_geometry`` cannot
    place a caustic reach for ``gamma``/``kappa`` (the parity wall
    ``gamma == 1 - kappa``; not reached anywhere in this suite's gamma
    band).
    """
    reach, _direction = caustic_geometry(gamma, kappa=kappa)
    radius = rho * reach
    return radius * np.array([math.cos(angle), math.sin(angle)])


def _exact_total_w(w: np.ndarray, gamma: float, y,
                   *, beta: float = 0.0, kappa: float = 0.0) -> np.ndarray:
    """Exact amplification total in the min-relative frame (engine oracle).

    Independent of the analytic serve path: drives the exact Schwinger
    engine via ``ChangRefsdalChannels.evaluate``.
    """
    ch = ChangRefsdalChannels(w)
    ch.reset()
    partition = ch.evaluate(gamma=gamma, y=(float(y[0]), float(y[1])),
                            beta=beta, kappa=kappa)
    return partition.exact_total


def _tier1_serve(w: np.ndarray, gamma: float, y,
                 *, beta: float = 0.0, kappa: float = 0.0):
    """Tier-1 zero-envelope FARFIELD_KERNEL_SUM reconstruction.

    Mirrors ``_saddle_farfield_analytic`` EXACTLY: builds the geometry
    partition, then reconstructs with an all-zero residual envelope under
    the ``FARFIELD_KERNEL_SUM`` tag.  Returns ``(geom, F_serve)`` where
    ``F_serve`` is the served amplification total across ``w``.
    """
    geom = ChangRefsdalChannels(w).geometry_partition(
        gamma=gamma, y=(float(y[0]), float(y[1])), beta=beta, kappa=kappa)
    envelope = np.zeros(w.shape, dtype=complex)
    _kernels, total = reconstruct_farfield(
        w, envelope, geom.delays, geom.saddle_kernels, geom.real_mask,
        FARFIELD_KERNEL_SUM, geom.t_min)
    return geom, total


def _real_delays(geom) -> np.ndarray:
    """Fermat delays of the REAL images (masked by ``geom.real_mask``)."""
    real = np.asarray(geom.real_mask, dtype=bool)
    return np.asarray(geom.delays)[real]


def _min_delta_tau(geom) -> float:
    """Narrowest positive pairwise gap between REAL image delays."""
    real = np.sort(_real_delays(geom))
    if len(real) < 2:
        return 0.0
    gaps = np.diff(real)
    positive = gaps[gaps > 0]
    return float(np.min(positive)) if len(positive) else 0.0


def _save_diagnostic_plot(fig, name: str) -> None:
    """Save a matplotlib figure to the output directory."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUTPUT_DIR / name, dpi=100, bbox_inches='tight')


# ======================================================================
# Anti-vacuity base class
# ======================================================================

class _SaddleTier1TestCase(TestCase):
    """Base class with an anti-vacuity tearDown."""

    def setUp(self):
        """Reset the per-test comparison counter."""
        self.n_checks = 0

    def tearDown(self):
        """Fail if zero comparisons ran (anti-vacuity)."""
        if self.n_checks == 0:
            self.fail(
                f'{self._testMethodName}: zero comparisons ran — the test '
                f'is vacuous (all configs skipped or no assertion fired).')


# ======================================================================
# Shared far-from-caustic population draw
# ======================================================================

def _draw_far_from_caustic(seed: int, n_target: int, w_grid: np.ndarray):
    """Draw ``n_target`` far-from-caustic, gate-admitted saddle sources.

    Deterministic under ``seed``.  Each candidate is drawn with
    gamma ~ U(GAMMA_LO, GAMMA_HI), rho ~ U(RHO_FAR, RHO_HI),
    angle ~ U(0, 2pi); ``rho`` is re-derived from the constructed ``y``
    via the production ``caustic_rho`` converter (not just trusted from
    the draw) and passed as the gate's third argument, so the admitted
    set is gated EXACTLY as production gates it.  Candidates whose
    caustic geometry is undefined or which fail the two-term gate
    (``_saddle_farfield_analytic_serves``) at ``w_grid.min()`` are
    rejected.  The gate uses the production single-source-of-truth
    predicate, so the admitted set matches what the live rung and the
    WP-2 census would admit.

    Returns a list of dicts with keys gamma, y, rho, angle, geom, mdt.
    """
    rng = np.random.default_rng(seed)
    w_lo = float(w_grid.min())
    admitted = []
    tries = 0
    while len(admitted) < n_target and tries < 5000:
        tries += 1
        gamma = float(rng.uniform(GAMMA_LO, GAMMA_HI))
        rho = float(rng.uniform(RHO_FAR, RHO_HI))
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        try:
            y = _polar_source(rho, angle, gamma)
        except geometry.LensDomainError:
            continue
        try:
            geom = ChangRefsdalChannels(w_grid).geometry_partition(
                gamma=gamma, y=(float(y[0]), float(y[1])),
                beta=0.0, kappa=0.0)
        except geometry.LensDomainError:
            continue
        try:
            rho_prod = caustic_rho(
                gamma, float(np.hypot(y[0], y[1])), kappa=0.0)
        except (ValueError, geometry.LensDomainError):
            continue
        if not _saddle_farfield_analytic_serves(
                _real_delays(geom), w_lo, rho_prod):
            continue
        admitted.append({'gamma': gamma, 'y': y, 'rho': rho_prod,
                         'angle': angle, 'geom': geom,
                         'mdt': _min_delta_tau(geom)})
    return admitted


# ======================================================================
# TEST CLASS 1: Far-from-caustic admitted-set accuracy (spec core)
# ======================================================================

class SaddleTier1FarFromCausticAccuracyTestCase(_SaddleTier1TestCase):
    """Tier-1 zero-envelope serve matches the exact engine far from caustic.

    Over >= 20 gate-admitted, far-from-caustic (rho >= RHO_FAR) macro
    saddle sources drawn with seed 42, the pointwise relative error
    err = |F_serve - F_exact| / |F_exact| across the cheap band
    w in [W_FLOOR, W_CEIL] satisfies p90(err) <= P90_TOL with a loose
    outlier guard max(err) <= OUTLIER_TOL.  The full p50/p90/max
    distribution and the worst-sample locus (gamma, y1, y2, w) are
    REPORTED (never a bare-max assertion -- F028).
    """

    W_GRID = np.geomspace(W_FLOOR, W_CEIL, _N_W)

    @classmethod
    def setUpClass(cls):
        """Draw the population once and measure the err distribution."""
        cls.admitted = _draw_far_from_caustic(SEED, N_ADMITTED, cls.W_GRID)
        cls.all_err = None
        cls.per_source = []
        errs = []
        for rec in cls.admitted:
            _geom, serve = _tier1_serve(cls.W_GRID, rec['gamma'], rec['y'])
            exact = _exact_total_w(cls.W_GRID, rec['gamma'], rec['y'])
            err = np.abs(serve - exact) / np.abs(exact)
            errs.append(err)
            imax = int(np.argmax(err))
            cls.per_source.append({
                'gamma': rec['gamma'],
                'y1': float(rec['y'][0]), 'y2': float(rec['y'][1]),
                'rho': rec['rho'], 'mdt': rec['mdt'],
                'w_at_max': float(cls.W_GRID[imax]),
                'max_err': float(err.max())})
        if errs:
            cls.all_err = np.concatenate(errs)
            cls.err_per_w = errs  # list of per-source err arrays

    def test_population_size_at_least_twenty(self):
        """The admitted far-from-caustic population has >= 20 sources."""
        self.n_checks += 1
        self.assertGreaterEqual(
            len(self.admitted), 20,
            f'Only {len(self.admitted)} admitted sources drawn; '
            f'spec requires >= 20.')

    def test_p90_within_tolerance(self):
        """p90 of pointwise err <= P90_TOL across the population."""
        self.assertIsNotNone(self.all_err, 'No admitted sources measured.')
        p50 = float(np.percentile(self.all_err, 50))
        p90 = float(np.percentile(self.all_err, 90))
        emax = float(self.all_err.max())
        self.n_checks += 1
        self.assertLessEqual(
            p90, P90_TOL,
            f'p90(err)={p90:.2e} exceeds {P90_TOL:.0e}.  '
            f'Distribution: p50={p50:.2e} p90={p90:.2e} max={emax:.2e} '
            f'over {len(self.admitted)} sources.')

    def test_max_within_outlier_guard(self):
        """max of pointwise err <= OUTLIER_TOL (loose outlier guard)."""
        self.assertIsNotNone(self.all_err, 'No admitted sources measured.')
        emax = float(self.all_err.max())
        self.n_checks += 1
        self.assertLessEqual(
            emax, OUTLIER_TOL,
            f'max(err)={emax:.2e} exceeds outlier guard {OUTLIER_TOL:.0e}.')

    def test_reports_worst_sample_locus(self):
        """Report the worst-sample locus and assert it is a valid saddle.

        The worst sample is identified and its (gamma, y1, y2, w) locus
        reported (F028: never assert a bare max).  Every admitted source
        stays under the outlier guard -- no single source is catastrophic.
        """
        self.assertTrue(self.per_source, 'No admitted sources measured.')
        worst = max(self.per_source, key=lambda r: r['max_err'])
        # Report (visible on failure and via the assertion messages).
        locus = (f"gamma={worst['gamma']:.4f} "
                 f"y=({worst['y1']:.4f}, {worst['y2']:.4f}) "
                 f"rho={worst['rho']:.3f} w={worst['w_at_max']:.2f} "
                 f"max_err={worst['max_err']:.2e}")
        self.n_checks += 1
        self.assertGreater(worst['gamma'], 1.0,
                           f'Worst sample not a macro saddle: {locus}')
        for rec in self.per_source:
            with self.subTest(gamma=rec['gamma'], rho=rec['rho']):
                self.n_checks += 1
                self.assertLessEqual(
                    rec['max_err'], OUTLIER_TOL,
                    f"Source gamma={rec['gamma']:.4f} rho={rec['rho']:.3f} "
                    f"has per-source max err {rec['max_err']:.2e} > "
                    f"{OUTLIER_TOL:.0e}.  Worst overall: {locus}")

    def test_diagnostic_plot_saved(self):
        """Save a loglog err-vs-w-per-source diagnostic plot."""
        self.assertIsNotNone(self.all_err, 'No admitted sources measured.')
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest('matplotlib not available')
        fig, ax = plt.subplots(figsize=(8, 6))
        for rec, err in zip(self.per_source, self.err_per_w):
            ax.loglog(self.W_GRID, np.maximum(err, 1e-16), '-',
                      alpha=0.5, lw=0.8)
        ax.axhline(P90_TOL, color='k', ls='--', alpha=0.6,
                   label=f'P90_TOL={P90_TOL:.0e}')
        ax.axhline(OUTLIER_TOL, color='r', ls=':', alpha=0.6,
                   label=f'OUTLIER_TOL={OUTLIER_TOL:.0e}')
        ax.set_xlabel('w')
        ax.set_ylabel('|F_serve - F_exact| / |F_exact|')
        ax.set_title(
            'Tier-1 saddle serve error vs w (far-from-caustic admitted set)')
        ax.legend(fontsize=8)
        _save_diagnostic_plot(
            fig, 'test_saddle_tier1_far_from_caustic_err_vs_w.png')
        plt.close(fig)
        self.n_checks += 1
        self.assertTrue(
            (_OUTPUT_DIR
             / 'test_saddle_tier1_far_from_caustic_err_vs_w.png').exists(),
            'Diagnostic plot not written.')


# ======================================================================
# Deterministic near/far source search (for ordering + falsification)
# ======================================================================

def _first_admitted_at_rho(gamma: float, rho: float, w_grid: np.ndarray,
                           n_angles: int = 60):
    """First gate-admitted saddle source at caustic-relative ``rho``.

    Sweeps polar angle deterministically and returns the first
    ``(y, geom)`` whose caustic geometry is defined, yields a valid
    geometry partition and passes the production two-term gate
    ``_saddle_farfield_analytic_serves`` (with ``rho`` re-derived from
    the constructed ``y`` via ``caustic_rho``, exactly as production
    gates it) at ``w_grid.min()``.  Returns ``None`` if none is found.
    """
    w_lo = float(w_grid.min())
    for angle in np.linspace(0.05, 2.0 * math.pi - 0.05, n_angles):
        try:
            y = _polar_source(rho, float(angle), gamma)
        except geometry.LensDomainError:
            continue
        try:
            geom = ChangRefsdalChannels(w_grid).geometry_partition(
                gamma=gamma, y=(float(y[0]), float(y[1])),
                beta=0.0, kappa=0.0)
        except geometry.LensDomainError:
            continue
        try:
            rho_prod = caustic_rho(
                gamma, float(np.hypot(y[0], y[1])), kappa=0.0)
        except (ValueError, geometry.LensDomainError):
            continue
        if _saddle_farfield_analytic_serves(
                _real_delays(geom), w_lo, rho_prod):
            return y, geom
    return None


def _resolvable_only(real_delays, w_lo) -> bool:
    """OLD resolvability-only predicate (pre rho-floor gate), for witnesses.

    Mirrors exactly what ``_saddle_farfield_analytic_serves`` checked
    before the rho-floor term was added: ``n_real >= 2 AND
    w_lo * min_delta_tau >= RHO_END``.  Used ONLY to construct and verify
    near-caustic leaky-gate witnesses -- never as a substitute for the
    production two-term gate.
    """
    real = np.sort(np.asarray(real_delays, dtype=float))
    if len(real) < 2:
        return False
    gaps = np.diff(real)
    positive = gaps[gaps > 0]
    if len(positive) == 0:
        return False
    return float(w_lo) * float(np.min(positive)) >= RHO_END


def _first_resolvable_at_rho(gamma: float, rho: float, w_grid: np.ndarray,
                             n_angles: int = 60):
    """First source at caustic-relative ``rho`` resolvable by the OLD
    resolvability-only predicate (``_resolvable_only``), regardless of the
    production two-term gate.

    Used to build a near-caustic witness that the OLD resolvability-only
    gate would have admitted (and served with a large error) but that the
    current two-term gate correctly refuses via the rho floor.  Returns
    ``(y, geom)`` or ``None`` if none is found.
    """
    w_lo = float(w_grid.min())
    for angle in np.linspace(0.05, 2.0 * math.pi - 0.05, n_angles):
        try:
            y = _polar_source(rho, float(angle), gamma)
        except geometry.LensDomainError:
            continue
        try:
            geom = ChangRefsdalChannels(w_grid).geometry_partition(
                gamma=gamma, y=(float(y[0]), float(y[1])),
                beta=0.0, kappa=0.0)
        except geometry.LensDomainError:
            continue
        if _resolvable_only(_real_delays(geom), w_lo):
            return y, geom
    return None


# ======================================================================
# TEST CLASS 2: Pinned worst measured case (leaky-gate witness)
# ======================================================================

class SaddleTier1PinnedWorstCaseTestCase(_SaddleTier1TestCase):
    """The brief's worst case gamma=1.5859, y=(-1.1208, -0.9002).

    Measured reality (double-double band):
      - n_real = 2, min_delta_tau ~ 0.933.
      - Production rho (the isotropic ``caustic_rho`` gauge) is ~0.73,
        WELL BELOW ``_SADDLE_FARFIELD_RHO_FLOOR`` (2.0): the two-term
        gate REFUSES this pin at ANY w_lo, regardless of resolvability.
      - At band floor w_lo = RHO_END = 4.0 resolvability ALONE also
        refuses (w_lo*mdt = 3.73 < 4.0): both gate terms independently
        refuse here.
      - At band floor w_lo = 8.0 resolvability ALONE would ADMIT
        (w_lo*mdt = 7.47 >= 4.0) and the zero-envelope serve is WRONG
        there (err ~ 9e-2) -- this is exactly the leak the rho-floor term
        was added to close.  The two-term gate correctly refuses it.

    This pin is therefore a WITNESS that the rho floor is load-bearing:
    without it, the OLD resolvability-only gate would have served this
    near-caustic source wrongly at w_lo = 8.
    """

    W_GRID = np.geomspace(PIN_LEAK_W_LO, W_CEIL, 20)

    @classmethod
    def setUpClass(cls):
        """Build the pinned geometry and production rho once."""
        cls.geom, cls.serve = _tier1_serve(cls.W_GRID, PIN_GAMMA, PIN_Y)
        cls.real_delays = _real_delays(cls.geom)
        cls.mdt = _min_delta_tau(cls.geom)
        cls.rho = caustic_rho(
            PIN_GAMMA, float(np.hypot(PIN_Y[0], PIN_Y[1])), kappa=0.0)

    def test_pin_is_two_image_macro_saddle(self):
        """The pin is a gamma>1 saddle with exactly two real images."""
        self.n_checks += 1
        self.assertGreater(PIN_GAMMA, 1.0)
        self.assertEqual(int(np.asarray(self.geom.real_mask).sum()), 2,
                         'Pin is not a two-image source.')

    def test_pin_rho_is_below_the_floor(self):
        """The pin's production rho sits below _SADDLE_FARFIELD_RHO_FLOOR.

        This is the near-caustic property the pin was chosen to
        demonstrate; if this drifted above the floor the pin would stop
        witnessing anything.
        """
        self.n_checks += 1
        self.assertLess(
            self.rho, _SADDLE_FARFIELD_RHO_FLOOR,
            f'Pin rho={self.rho:.4f} unexpectedly at/above the floor '
            f'{_SADDLE_FARFIELD_RHO_FLOOR}; the pin no longer witnesses a '
            f'near-caustic source.')

    def test_pin_refused_at_rho_end_floor(self):
        """At w_lo = RHO_END both gate terms independently refuse."""
        served = _saddle_farfield_analytic_serves(
            self.real_delays, PIN_REFUSE_W_LO, self.rho)
        self.n_checks += 1
        self.assertFalse(
            served,
            f'Pin unexpectedly admitted at w_lo={PIN_REFUSE_W_LO}: '
            f'w_lo*mdt={PIN_REFUSE_W_LO * self.mdt:.3f} should be < '
            f'{RHO_END} (measured mdt={self.mdt:.4f}) AND rho='
            f'{self.rho:.4f} should be < {_SADDLE_FARFIELD_RHO_FLOOR}.')
        self.n_checks += 1
        self.assertLess(PIN_REFUSE_W_LO * self.mdt, RHO_END)

    def test_pin_refused_at_floor_eight_by_rho_term_alone(self):
        """Leaky-gate witness: resolvability alone would admit at w_lo=8,
        but the rho floor now refuses it regardless.

        Proves the rho-floor term is load-bearing: without it, this
        near-caustic source would be served with a large error.
        """
        resolvable = PIN_LEAK_W_LO * self.mdt >= RHO_END
        self.n_checks += 1
        self.assertTrue(
            resolvable,
            f'Pin unexpectedly NOT resolvable at w_lo={PIN_LEAK_W_LO}: '
            f'w_lo*mdt={PIN_LEAK_W_LO * self.mdt:.3f} should be >= '
            f'{RHO_END}; the leaky-gate premise no longer holds for this '
            f'pin.')
        served = _saddle_farfield_analytic_serves(
            self.real_delays, PIN_LEAK_W_LO, self.rho)
        self.n_checks += 1
        self.assertFalse(
            served,
            f'Pin unexpectedly ADMITTED at w_lo={PIN_LEAK_W_LO} despite '
            f'rho={self.rho:.4f} < {_SADDLE_FARFIELD_RHO_FLOOR}: the rho '
            f'floor failed to close the leaky-gate hole.')
        # Demonstrate WHY the refusal matters: the zero-envelope serve
        # (computed independently of the gate) is genuinely wrong here.
        exact = _exact_total_w(self.W_GRID, PIN_GAMMA, PIN_Y)
        err = np.abs(self.serve - exact) / np.abs(exact)
        self.n_checks += 1
        self.assertGreater(
            float(err.max()), LEAK_MIN_ERR,
            f'Pin served err max={float(err.max()):.2e} did not exceed '
            f'{LEAK_MIN_ERR:.0e}: the leaky-gate witness has lost its '
            f'teeth (near-caustic serve was unexpectedly accurate, so '
            f'refusing it would not have mattered).')


# ======================================================================
# TEST CLASS 3: Caustic proximity dominates the serve error
# ======================================================================

class SaddleTier1FloorClosesNearCausticLeakTestCase(_SaddleTier1TestCase):
    """The rho floor closes the near-caustic leak resolvability alone has.

    A far gate-admitted source (rho >= RHO_FAR, same gamma) is served
    accurately (err < P90_TOL) -- caustic proximity, not resolvability, is
    what tier-1 accuracy depends on.  A near-caustic source (rho ~ 1.1)
    that IS resolvable by the OLD resolvability-only predicate -- i.e. the
    pre-floor gate would have admitted and served it -- is REFUSED by the
    current two-term gate (rho < _SADDLE_FARFIELD_RHO_FLOOR); if served
    anyway (bypassing the gate, as the old leaky gate would have) the
    error is large.  This isolates caustic proximity as what the floor
    term guards against.
    """

    GAMMA = 1.4
    RHO_NEAR = 1.10
    W_GRID = np.geomspace(W_FLOOR, W_CEIL, 16)

    @classmethod
    def setUpClass(cls):
        """Locate a far gate-admitted source and a near resolvable-but-
        near-caustic source deterministically."""
        cls.far = _first_admitted_at_rho(cls.GAMMA, RHO_FAR + 0.5, cls.W_GRID)
        cls.near = _first_resolvable_at_rho(cls.GAMMA, cls.RHO_NEAR,
                                            cls.W_GRID)

    def test_far_source_accurate_and_gate_admitted(self):
        """The far source is gate-admitted and served accurately."""
        self.assertIsNotNone(self.far, 'No far gate-admitted source found.')
        y_far, _gf = self.far
        _g1, serve_far = _tier1_serve(self.W_GRID, self.GAMMA, y_far)
        exact_far = _exact_total_w(self.W_GRID, self.GAMMA, y_far)
        err_far = float(
            (np.abs(serve_far - exact_far) / np.abs(exact_far)).max())
        self.n_checks += 1
        self.assertLess(
            err_far, P90_TOL,
            f'Far source (rho~{RHO_FAR + 0.5}) served err {err_far:.2e} '
            f'>= {P90_TOL:.0e}; expected far-from-caustic accuracy.')

    def test_near_source_resolvable_but_rho_floor_refuses(self):
        """Near-caustic source: resolvable (old gate would admit), but the
        two-term gate's rho floor refuses it -- and if served anyway the
        error would be large (proving the refusal is load-bearing)."""
        self.assertIsNotNone(
            self.near, 'No near resolvable-but-near-caustic source found.')
        y_near, geom_near = self.near
        w_lo = float(self.W_GRID.min())
        real_delays = _real_delays(geom_near)

        # The OLD resolvability-only predicate WOULD admit this source.
        self.n_checks += 1
        self.assertTrue(
            _resolvable_only(real_delays, w_lo),
            'Near source unexpectedly not resolvable by the OLD '
            'predicate; the leak-witness search is broken.')

        # Its production rho sits below the floor, so the current
        # two-term gate REFUSES it.
        rho_near = caustic_rho(
            self.GAMMA, float(np.hypot(y_near[0], y_near[1])), kappa=0.0)
        self.n_checks += 1
        self.assertLess(
            rho_near, _SADDLE_FARFIELD_RHO_FLOOR,
            f'Near source rho={rho_near:.4f} unexpectedly at/above the '
            f'floor; not a near-caustic witness.')
        self.n_checks += 1
        self.assertFalse(
            _saddle_farfield_analytic_serves(real_delays, w_lo, rho_near),
            'Near-caustic, resolvable source unexpectedly ADMITTED by the '
            'two-term gate: the rho floor failed to close the leak.')

        # If served anyway (as the old leaky gate would have), the error
        # is large -- proving the refusal is load-bearing, not cosmetic.
        _g2, serve_near = _tier1_serve(self.W_GRID, self.GAMMA, y_near)
        exact_near = _exact_total_w(self.W_GRID, self.GAMMA, y_near)
        err_near = float(
            (np.abs(serve_near - exact_near) / np.abs(exact_near)).max())
        self.n_checks += 1
        self.assertGreater(
            err_near, OUTLIER_TOL,
            f'Near source (rho~{rho_near:.3f}) served err {err_near:.2e} '
            f'<= {OUTLIER_TOL:.0e}: the leaky-gate witness has lost its '
            f'teeth (refusing it would not have mattered).')


# ======================================================================
# TEST CLASS 4: Self-falsification (the suite can go red)
# ======================================================================

class SaddleTier1SelfFalsificationTestCase(_SaddleTier1TestCase):
    """Prove the accuracy comparison has teeth.

    A corrupted serve, a mismatched oracle, and a structureless-oracle
    check each demonstrate that the pointwise err comparison genuinely
    detects a wrong answer -- a suite that cannot go red is not a test.
    """

    GAMMA_A = 1.4
    GAMMA_B = 1.7
    W_GRID = np.geomspace(W_FLOOR, W_CEIL, 16)

    @classmethod
    def setUpClass(cls):
        """Build two far-from-caustic gate-admitted reference sources."""
        cls.src_a = _first_admitted_at_rho(cls.GAMMA_A, RHO_FAR + 0.5,
                                           cls.W_GRID)
        cls.src_b = _first_admitted_at_rho(cls.GAMMA_B, RHO_FAR + 0.5,
                                           cls.W_GRID)

    def test_corrupted_serve_breaches_p90(self):
        """Perturbing the served total by 5% of |F_exact| breaches P90_TOL."""
        self.assertIsNotNone(self.src_a, 'No reference source A found.')
        y_a, _g = self.src_a
        _gs, serve = _tier1_serve(self.W_GRID, self.GAMMA_A, y_a)
        exact = _exact_total_w(self.W_GRID, self.GAMMA_A, y_a)
        # Baseline: the honest serve passes.
        err_ok = float((np.abs(serve - exact) / np.abs(exact)).max())
        self.n_checks += 1
        self.assertLess(err_ok, P90_TOL,
                        f'Baseline far serve err {err_ok:.2e} unexpectedly '
                        f'above {P90_TOL:.0e}.')
        # Corrupt by an additive 5% of |F_exact|.
        corrupted = serve + 0.05 * np.abs(exact)
        err_bad = float((np.abs(corrupted - exact) / np.abs(exact)).max())
        self.n_checks += 1
        self.assertGreater(
            err_bad, P90_TOL,
            f'Corrupted serve err {err_bad:.2e} failed to breach '
            f'{P90_TOL:.0e}: the comparison has no teeth.')

    def test_mismatched_oracle_breaches(self):
        """Serving source A against source B's exact total gives large err."""
        self.assertIsNotNone(self.src_a, 'No reference source A found.')
        self.assertIsNotNone(self.src_b, 'No reference source B found.')
        y_a, _ga = self.src_a
        y_b, _gb = self.src_b
        _gs, serve_a = _tier1_serve(self.W_GRID, self.GAMMA_A, y_a)
        exact_b = _exact_total_w(self.W_GRID, self.GAMMA_B, y_b)
        err = float((np.abs(serve_a - exact_b) / np.abs(exact_b)).max())
        self.n_checks += 1
        self.assertGreater(
            err, OUTLIER_TOL,
            f'Mismatched-oracle err {err:.2e} did not exceed '
            f'{OUTLIER_TOL:.0e}: distinct saddle sources are '
            f'indistinguishable -- the oracle is not discriminating.')

    def test_oracle_has_nontrivial_structure(self):
        """|F_exact| varies across w (the oracle is not a flat constant)."""
        self.assertIsNotNone(self.src_a, 'No reference source A found.')
        y_a, _g = self.src_a
        exact = _exact_total_w(self.W_GRID, self.GAMMA_A, y_a)
        mag = np.abs(exact)
        rel_spread = float((mag.max() - mag.min()) / mag.mean())
        self.n_checks += 1
        self.assertGreater(
            rel_spread, 1e-3,
            f'|F_exact| relative spread {rel_spread:.2e} is ~flat; the '
            f'oracle carries no structure to falsify against.')


if __name__ == '__main__':
    main()
