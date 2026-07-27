"""
Value-preservation guards for the Build 8f performance levers.

The Build 8f levers are *optimizations*: each one re-associates or
re-schedules an existing numerical computation without changing the
mathematics.  HEAD (the commit these levers sit on top of) is therefore
the trusted oracle -- the optimized working-tree code must reproduce it
to floating-point round-off (or, where the optimization only re-orders
integer/decision work, bit-for-bit).  This suite loads the HEAD code
SIDE BY SIDE with the optimized code (via ``git show HEAD:<path>`` into a
temporary module, the 8b-levers idiom) and compares them on a physically
representative sweep.

Three levers are covered:

* **Lever 1 (geometry).**  ``geometry_partition`` and the geometric image
  set.  The optimization is `_companion_roots`, which replaces
  ``numpy.roots`` for the fixed-degree image quartic with a direct
  companion-eigenvalue solve of the SAME companion matrix.  For a
  polynomial with no leading/trailing zero this is bit-for-bit identical
  to ``numpy.roots``; the sweep therefore demands ``<= 1e-10`` relative on
  image values and byte-identity (``max|diff| == 0``) on image COUNT,
  the real-image mask, and the channel switch decision, including the
  near-caustic double-root regime where the quartic discriminant is
  smallest.

* **Lever 2 (likelihood contraction).**  ``_data_term`` / ``_norm_term``.
  The norm-term mode reduction is re-associated (``einsum`` hoisting);
  ``_data_term`` is unchanged.  Agreement is ``<= 1e-10`` RELATIVE in the
  normal regime, but a near-zero ``(h_L|h_L)`` normalization denominator
  (catastrophic cancellation of O(1) intermediate terms) makes a relative
  tolerance meaningless -- the reassociation round-off is ``~1e-14``
  ABSOLUTE regardless of how small the result is, so below
  `NORM_UNDERFLOW_FLOOR` the test switches to an ABSOLUTE tolerance.  This
  is not a weakening: the absolute round-off floor is what preservation
  actually means when the result underflows.

* **Lever 3 (node-parallel Schwinger).**  ``_positive_parity_grid`` /
  ``_saddle_grid`` gather the ``w <= 60`` exact wave nodes and evaluate
  them across cores through an njit ``prange`` PURE MAP.  Because the map
  carries no cross-node reduction and ``fastmath`` is off, every node is
  bit-for-bit identical to the serial ``f_schwinger`` path.  The "serial"
  oracle is the SAME grid function with the njit map swapped for its
  ``.py_func`` (a plain sequential loop), so any divergence isolates the
  parallelization.  Refusal identity (any-node-refuses -> whole-grid
  refuses, same named exception) is also checked, and an F010 self-
  falsification mutates the map's certification reduction to prove the
  refusal-identity guard has teeth.

Tolerance choices
-----------------
``REL_TOL = 1e-10`` is the Architect's preservation bound; measured
relative round-off on this sweep is ``~1e-15`` (image values) and
``~1e-16`` (norm term, normal regime), five to six orders inside it.
``NORM_UNDERFLOW_FLOOR = 1e-6`` cleanly separates the engineered
near-underflow detector (``|norm| ~ 1e-13``) from the ordinary detectors
(``|norm| ~ 1e2``).  ``NORM_ABS_TOL = 1e-11`` covers the measured
absolute reassociation round-off (``~8e-14``) with three orders of
margin.  ``BYTE_EXACT = 0.0`` is the literal bit-identity bound used for
decision bits and the node-parallel values.
"""
from __future__ import annotations

import ast
import functools
import importlib.util
import inspect
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cogwheel.lensing.chang_refsdal import (geometry, channels, operator,
                                            _schwinger, _pearcey_table,
                                            _pearcey_cusp)
from cogwheel.lensing import likelihood

#: Architect preservation bound: optimized image / norm values must agree
#: with HEAD to this relative tolerance in the well-conditioned regime.
REL_TOL = 1e-10

#: Literal bit-identity bound for integer/decision quantities (image
#: count, real-image mask, channel switch) and the node-parallel values.
BYTE_EXACT = 0.0

#: Below this ``|(h_L|h_L)|`` the norm denominator is treated as
#: underflowing and the comparison uses an ABSOLUTE tolerance (a relative
#: tolerance is meaningless at a near-zero norm).
NORM_UNDERFLOW_FLOOR = 1e-6

#: Absolute reassociation round-off bound for the underflowing norm
#: regime; measured worst case on this sweep is ``~8e-14``.
NORM_ABS_TOL = 1e-11

#: Dimensionless frequencies spanning the exact-wave band (``w <= 60``,
#: the parallel batch), the ceiling, and the arm/refusal branch (61).
W_SWEEP = (5.0, 18.0, 40.0, 55.0, 59.0, 61.0)

#: Positive-parity shears (``1 - kappa > |gamma|`` at ``kappa = 0``);
#: 0.5 is the cancellation band the Professor flagged.
GAMMA_POSITIVE = (0.2, 0.5, 0.9)

#: Saddle-host shears (``1 - kappa < |gamma|`` at ``kappa = 0``).
GAMMA_SADDLE = (1.1, 1.6)

#: Signed offset from the 4<->2 image caustic crossing used to probe the
#: near-caustic double-root regime (inside: +, outside: -).
CAUSTIC_HALFWIDTH = 2e-3

#: Directory for diagnostic plots.
_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

_REPO_ROOT = subprocess.run(
    ['git', 'rev-parse', '--show-toplevel'],
    capture_output=True, text=True, check=True,
    cwd=os.path.dirname(__file__)).stdout.strip()


# --------------------------------------------------------------------------
# HEAD (oracle) loaders -- the 8b-levers side-by-side idiom.
# --------------------------------------------------------------------------
def _head_source(relpath: str) -> str:
    """Return the HEAD text of a repo-relative file."""
    return subprocess.run(
        ['git', 'show', f'HEAD:{relpath}'],
        capture_output=True, text=True, check=True, cwd=_REPO_ROOT).stdout


@functools.lru_cache(maxsize=None)
def _head_geometry():
    """Load HEAD ``geometry.py`` as an independent, importable module.

    ``geometry.py`` imports only ``numba``/``numpy``/``scipy`` (no relative
    cogwheel imports), so it loads standalone.  A real temp ``.py`` file is
    used because ``numba.njit(cache=True)`` needs a file locator; the
    module is registered in ``sys.modules`` before ``exec_module`` so its
    ``@dataclass``/``NamedTuple`` fields resolve.
    """
    source = _head_source('cogwheel/lensing/chang_refsdal/geometry.py')
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='_head_geometry.py', delete=False)
    tmp.write(source)
    tmp.close()
    modname = 'cogwheel_head_geometry_lever1'
    spec = importlib.util.spec_from_file_location(modname, tmp.name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=None)
def _head_norm_term():
    """AST-extract the HEAD ``_norm_term`` (self-contained: ``np`` + const)."""
    return _extract_function('_norm_term')


@functools.lru_cache(maxsize=None)
def _head_data_term():
    """AST-extract the HEAD ``_data_term`` (self-contained: ``np`` + const)."""
    return _extract_function('_data_term')


def _extract_function(name: str):
    """Compile a top-level HEAD ``likelihood`` function in isolation.

    ``_data_term`` / ``_norm_term`` reference only ``numpy`` and the module
    constant ``_TWO_PI_I``, so they compile in a minimal namespace without
    importing the heavy ``likelihood`` module twice.
    """
    source = _head_source('cogwheel/lensing/likelihood.py')
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            namespace = {'np': np, '_TWO_PI_I': likelihood._TWO_PI_I}
            exec(ast.get_source_segment(source, node), namespace)
            return namespace[name]
    raise LookupError(f'HEAD likelihood.py has no top-level {name!r}.')


def _ensure_output_dir() -> None:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# Base test case: anti-vacuity counter shared by every comparison suite.
# --------------------------------------------------------------------------
class _LeverTestCase(unittest.TestCase):
    """Base carrying the anti-vacuity comparison counter.

    Every concrete comparison must call `record_comparison`; a suite that
    silently compares nothing (all sub-cases skipped, an empty sweep, a
    broken oracle) FAILS in ``tearDown`` instead of reading green.
    """

    def setUp(self) -> None:
        self._comparisons = 0

    def record_comparison(self) -> None:
        self._comparisons += 1

    def tearDown(self) -> None:
        if self._comparisons == 0:
            self.fail(
                'Anti-vacuity: no value comparison ran in this test -- the '
                'oracle or the sweep produced nothing to check.')


# --------------------------------------------------------------------------
# Lever 1: geometry_partition + geometric image set value-preservation.
# --------------------------------------------------------------------------
def _caustic_crossing_radius(gamma: float, beta: float, kappa: float,
                             angle: float) -> float | None:
    """Bisect the 4<->2 image-count crossing radius along a source ray.

    Returns ``None`` if the ray never crosses (a saddle host has no
    4-image region along a generic ray).
    """
    matrix = geometry.macro_matrix(gamma, beta, kappa)
    unit = np.array([np.cos(angle), np.sin(angle)])
    low, high = 1e-3, 40.0
    n_low = len(geometry.find_images(low * unit, matrix))
    n_high = len(geometry.find_images(high * unit, matrix))
    if n_low == n_high:
        return None
    for _ in range(80):
        mid = 0.5 * (low + high)
        if len(geometry.find_images(mid * unit, matrix)) == n_low:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


@functools.lru_cache(maxsize=None)
def _lever1_configs() -> tuple:
    """Representative Chang--Refsdal configs spanning the Lever-1 regimes.

    Positive-parity hosts contribute an inside-caustic 4-image point, an
    outside-caustic 2-image point, and the two near-caustic double-root
    points at ``+-CAUSTIC_HALFWIDTH``.  Saddle hosts contribute two
    representative 2-image points (no generic 4-image region).
    """
    angle = 0.3
    configs: list[dict] = []
    for gamma in GAMMA_POSITIVE:
        radius = _caustic_crossing_radius(gamma, 0.0, 0.0, angle)
        unit = np.array([np.cos(angle), np.sin(angle)])
        for label, r in (('inside_4img', radius - 0.05),
                         ('outside_2img', radius + 0.05),
                         ('near_caustic_inside', radius - CAUSTIC_HALFWIDTH),
                         ('near_caustic_outside', radius + CAUSTIC_HALFWIDTH)):
            configs.append({'label': f'pos_g{gamma}_{label}', 'gamma': gamma,
                            'beta': 0.0, 'kappa': 0.0, 'y': r * unit})
    for gamma in GAMMA_SADDLE:
        unit = np.array([np.cos(angle), np.sin(angle)])
        for label, r in (('near', 0.2), ('far', 0.6)):
            configs.append({'label': f'sad_g{gamma}_{label}', 'gamma': gamma,
                            'beta': 0.0, 'kappa': 0.0, 'y': r * unit})
    # A convergence (kappa != 0) positive-parity host, to exercise the
    # mass-sheet-scaled matrix through the same quartic.
    kappa = 0.3
    radius = _caustic_crossing_radius(0.5, 0.0, kappa, angle)
    unit = np.array([np.cos(angle), np.sin(angle)])
    configs.append({'label': 'pos_kappa_inside', 'gamma': 0.5, 'beta': 0.0,
                    'kappa': kappa, 'y': (radius - 0.05) * unit})
    configs.append({'label': 'pos_kappa_outside', 'gamma': 0.5, 'beta': 0.0,
                    'kappa': kappa, 'y': (radius + 0.05) * unit})
    return tuple(configs)


def _min_pairwise_separation(points: list[np.ndarray]) -> float:
    """Smallest distance between any two image positions (double-root proxy).

    Near a fold caustic two images coalesce, so this is small exactly where
    the image quartic is nearly degenerate.
    """
    if len(points) < 2:
        return np.inf
    stacked = np.array(points)
    diffs = stacked[:, None, :] - stacked[None, :, :]
    dist = np.hypot(diffs[..., 0], diffs[..., 1])
    dist[np.diag_indices(len(points))] = np.inf
    return float(dist.min())


class CompanionRootsByteIdentityTestCase(_LeverTestCase):
    """`_companion_roots` reproduces ``numpy.roots`` bit-for-bit.

    The optimization solves the SAME companion matrix, so for a quartic
    with no leading/trailing zero (the production case) the root set is
    identical to the last bit -- this is the core Lever-1 claim.
    """

    def test_companion_roots_are_byte_identical_to_numpy_roots(self) -> None:
        for cfg in _lever1_configs():
            matrix = geometry.macro_matrix(
                cfg['gamma'], cfg['beta'], cfg['kappa'])
            source = np.asarray(cfg['y'], dtype=float)
            radius, basis = geometry._source_frame(source)
            rotated = basis.T @ matrix @ basis
            coeffs = geometry.image_quartic_coefficients(radius, rotated)
            with self.subTest(config=cfg['label']):
                companion = np.sort_complex(geometry._companion_roots(coeffs))
                reference = np.sort_complex(np.roots(coeffs))
                self.assertEqual(companion.shape, reference.shape)
                self.assertEqual(
                    float(np.max(np.abs(companion - reference))), BYTE_EXACT)
                self.record_comparison()


class GeometricImageSetTestCase(_LeverTestCase):
    """``find_images`` value-preservation vs HEAD across the sweep.

    Image COUNT byte-identical, image positions ``<= 1e-10`` relative, and
    the delay-sort order preserved.  Accumulates (double-root proximity,
    relative error) for the diagnostic scatter.
    """

    _scatter: list[tuple[float, float]] = []

    def test_find_images_value_preserving_vs_head(self) -> None:
        head_geom = _head_geometry()
        for cfg in _lever1_configs():
            source = np.asarray(cfg['y'], dtype=float)
            matrix = geometry.macro_matrix(
                cfg['gamma'], cfg['beta'], cfg['kappa'])
            head_matrix = head_geom.macro_matrix(
                cfg['gamma'], cfg['beta'], cfg['kappa'])
            cur = geometry.find_images(source, matrix)
            head = head_geom.find_images(source, head_matrix)
            with self.subTest(config=cfg['label']):
                # Image count is a decision quantity -> byte-identical.
                self.assertEqual(len(cur), len(head))
                worst_rel = 0.0
                for image_cur, image_head in zip(cur, head):
                    denom = max(float(np.linalg.norm(image_head)), 1e-300)
                    rel = float(np.linalg.norm(image_cur - image_head)) / denom
                    worst_rel = max(worst_rel, rel)
                    self.assertLessEqual(rel, REL_TOL)
                self._scatter.append(
                    (_min_pairwise_separation(head), worst_rel))
                self.record_comparison()

    @classmethod
    def tearDownClass(cls) -> None:
        if not cls._scatter:
            return
        _ensure_output_dir()
        seps, rels = zip(*cls._scatter)
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.scatter(seps, np.maximum(rels, 1e-18), s=28)
        axis.axhline(REL_TOL, color='crimson', ls='--',
                     label=f'REL_TOL = {REL_TOL:g}')
        axis.set_xscale('log')
        axis.set_yscale('log')
        axis.set_xlabel('min image separation (double-root proximity)')
        axis.set_ylabel('max relative image error vs HEAD')
        axis.set_title('Lever 1: find_images preservation near the caustic')
        axis.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(
            _OUTPUT_DIR,
            'lever1_find_images_relerror_vs_double_root.png'), dpi=110)
        plt.close(fig)


class GeometryPartitionValuePreservationTestCase(_LeverTestCase):
    """``geometry_partition`` value-preservation vs HEAD geometry.

    Same unchanged ``channels`` code, only the ``geometry`` module swapped
    (working tree vs HEAD).  Real-image mask and channel switch are
    decision bits -> byte-identical; delays, saddle kernels, critical delay
    and caustic distance are ``<= 1e-10`` relative.
    """

    def _partition(self, cfg: dict, geom_module):
        with mock.patch.object(channels, 'geometry', geom_module):
            chan = channels.ChangRefsdalChannels(np.array([12.0, 34.0]))
            chan.reset()
            return chan.geometry_partition(
                gamma=cfg['gamma'], y=cfg['y'],
                beta=cfg['beta'], kappa=cfg['kappa'])

    def test_partition_decision_bits_and_values_preserved(self) -> None:
        head_geom = _head_geometry()
        for cfg in _lever1_configs():
            with self.subTest(config=cfg['label']):
                cur = self._partition(cfg, geometry)
                head = self._partition(cfg, head_geom)
                # Decision bits: byte-identical.
                np.testing.assert_array_equal(
                    np.asarray(cur.real_mask), np.asarray(head.real_mask))
                np.testing.assert_array_equal(
                    np.asarray(cur.switch), np.asarray(head.switch))
                # Continuous quantities: <= 1e-10 relative.
                for name in ('delays', 'saddle_kernels'):
                    self._assert_relative(
                        np.asarray(getattr(cur, name)),
                        np.asarray(getattr(head, name)), name, cfg['label'])
                self._assert_relative(
                    np.array([cur.critical_delay]),
                    np.array([head.critical_delay]), 'critical_delay',
                    cfg['label'])
                self._assert_relative(
                    np.array([cur.caustic_distance]),
                    np.array([head.caustic_distance]), 'caustic_distance',
                    cfg['label'])
                self.record_comparison()

    def _assert_relative(self, cur, head, name, label) -> None:
        denom = np.maximum(np.abs(head), 1e-300)
        rel = np.max(np.abs(cur - head) / denom)
        self.assertLessEqual(
            rel, REL_TOL, f'{name} relerror {rel:g} in {label}')


class CompanionRootsSelfFalsificationTestCase(_LeverTestCase):
    """Prove the Lever-1 guards can go red under a broken root solver."""

    def test_perturbed_roots_break_byte_identity(self) -> None:
        matrix = geometry.macro_matrix(0.5, 0.0, 0.0)
        source = np.array([0.1, 0.05])
        radius, basis = geometry._source_frame(source)
        rotated = basis.T @ matrix @ basis
        coeffs = geometry.image_quartic_coefficients(radius, rotated)
        good = np.sort_complex(geometry._companion_roots(coeffs))
        bad = np.sort_complex(np.roots(coeffs) * (1.0 + 1e-9))
        # The honest byte-identity assertion would pass on `good` but must
        # FAIL on a 1e-9-perturbed root set.
        self.assertNotEqual(
            float(np.max(np.abs(good - np.sort_complex(np.roots(coeffs))))),
            float(np.max(np.abs(bad - np.sort_complex(np.roots(coeffs))))))
        with self.assertRaises(AssertionError):
            self.assertEqual(
                float(np.max(np.abs(bad - np.sort_complex(np.roots(coeffs))))),
                BYTE_EXACT)
        self.record_comparison()

    def test_wrong_image_count_is_detected(self) -> None:
        head_geom = _head_geometry()
        source = np.array([0.1, 0.05])
        matrix = geometry.macro_matrix(0.5, 0.0, 0.0)
        cur = geometry.find_images(source, matrix)

        def drop_last(src, mat):
            return head_geom.find_images(src, mat)[:-1]

        with mock.patch.object(geometry, 'find_images', drop_last):
            mutated = geometry.find_images(source, matrix)
        # The honest count check (len equal) must fail against the mutant.
        with self.assertRaises(AssertionError):
            self.assertEqual(len(cur), len(mutated))
        self.record_comparison()


# --------------------------------------------------------------------------
# Lever 2: likelihood contraction (_data_term / _norm_term) preservation.
# --------------------------------------------------------------------------
#: Representative moment shapes ``(n_modes, n_det, n_bins, n_img)``.
_LEVER2_SHAPES = (3, 2, 6, 3)


def _complex_normal(rng: np.random.Generator, shape) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _random_norm_inputs(seed: int):
    """A well-conditioned ``_norm_term`` input tuple (all O(1) magnitudes)."""
    nm, nd, nb, nimg = _LEVER2_SHAPES
    rng = np.random.default_rng(seed)
    b_moments = [_complex_normal(rng, (nm, nm, nd, nb)) for _ in range(4)]
    r0 = _complex_normal(rng, (nm, nd, nb))
    r1 = _complex_normal(rng, (nm, nd, nb))
    rho0 = _complex_normal(rng, (nm, nd, nb))
    rho1 = _complex_normal(rng, (nm, nd, nb))
    k0 = _complex_normal(rng, (nimg, nb))
    k1 = _complex_normal(rng, (nimg, nb))
    kbar0 = _complex_normal(rng, (nimg, nb))
    kbar1 = _complex_normal(rng, (nimg, nb))
    delays = rng.standard_normal(nimg)
    f_center = np.linspace(20.0, 200.0, nb)
    return (b_moments, r0, r1, rho0, rho1, k0, k1, kbar0, kbar1,
            delays, f_center)


def _underflow_norm_inputs():
    """Inputs whose detector-0 ``(h_L|h_L)`` cancels to ``~1e-13``.

    ``_norm_term`` is LINEAR in the ``b_moments`` tensors, so for two
    independent draws ``b1``, ``b2`` (sharing the other operands) the norm
    is ``h1 + s * h2``.  Choosing the scalar ``s = -h1[0]/h2[0]`` drives
    detector 0 through zero while every intermediate term stays O(1) --
    genuine catastrophic cancellation, the regime where a relative
    tolerance is meaningless.
    """
    nm, nd, nb, nimg = _LEVER2_SHAPES
    rng = np.random.default_rng(7)
    shared = (
        _complex_normal(rng, (nm, nd, nb)), _complex_normal(rng, (nm, nd, nb)),
        _complex_normal(rng, (nm, nd, nb)), _complex_normal(rng, (nm, nd, nb)),
        _complex_normal(rng, (nimg, nb)), _complex_normal(rng, (nimg, nb)),
        _complex_normal(rng, (nimg, nb)), _complex_normal(rng, (nimg, nb)),
        rng.standard_normal(nimg), np.linspace(20.0, 200.0, nb))
    b1 = [_complex_normal(rng, (nm, nm, nd, nb)) for _ in range(4)]
    b2 = [_complex_normal(rng, (nm, nm, nd, nb)) for _ in range(4)]
    head_norm = _head_norm_term()
    h1 = head_norm(b1, *shared)
    h2 = head_norm(b2, *shared)
    scale = -h1[0] / h2[0]
    b_cancel = [b1[i] + scale * b2[i] for i in range(4)]
    return (b_cancel, *shared)


def _random_data_inputs(seed: int):
    """A ``_data_term`` input tuple with O(1) magnitudes."""
    nm, nd, nb, nimg = _LEVER2_SHAPES
    rng = np.random.default_rng(seed)
    a_moments = [_complex_normal(rng, (nm, nd, nb)) for _ in range(3)]
    rho0 = _complex_normal(rng, (nm, nd, nb))
    rho1 = _complex_normal(rng, (nm, nd, nb))
    kbar0 = _complex_normal(rng, (nimg, nb))
    kbar1 = _complex_normal(rng, (nimg, nb))
    tau = rng.standard_normal(nimg)
    f_center = np.linspace(20.0, 200.0, nb)
    return a_moments, rho0, rho1, kbar0, kbar1, tau, f_center


class DataTermValuePreservationTestCase(_LeverTestCase):
    """``_data_term`` (unchanged by Lever 2) still reproduces HEAD exactly."""

    def test_data_term_matches_head(self) -> None:
        head_data = _head_data_term()
        for seed in range(6):
            inputs = _random_data_inputs(seed)
            with self.subTest(seed=seed):
                cur = likelihood._data_term(*inputs)
                head = head_data(*inputs)
                rel = np.max(np.abs(cur - head)
                             / np.maximum(np.abs(head), 1e-300))
                self.assertLessEqual(rel, REL_TOL)
                self.record_comparison()


class NormTermValuePreservationTestCase(_LeverTestCase):
    """``_norm_term`` reassociation preservation, normal + underflow regimes."""

    _scatter: list[tuple[float, float, bool]] = []

    def test_norm_term_relative_in_normal_regime(self) -> None:
        head_norm = _head_norm_term()
        for seed in range(8):
            inputs = _random_norm_inputs(seed)
            with self.subTest(seed=seed):
                cur = likelihood._norm_term(*inputs)
                head = head_norm(*inputs)
                self._compare(cur, head)
                self.record_comparison()

    def test_norm_term_absolute_when_denominator_underflows(self) -> None:
        head_norm = _head_norm_term()
        inputs = _underflow_norm_inputs()
        cur = likelihood._norm_term(*inputs)
        head = head_norm(*inputs)
        # Detector 0 is engineered below the underflow floor; confirm the
        # premise (else the absolute branch is untested).
        self.assertLess(abs(head[0]), NORM_UNDERFLOW_FLOOR)
        # First sweep run (8f close) measured the optimized _norm_term
        # BIT-IDENTICAL to HEAD in this regime — the original assertion
        # demanded real HEAD-vs-new drift here to demonstrate that
        # relative comparison misbehaves at underflow, which
        # over-satisfied preservation makes impossible.  Demonstrate the
        # currency lesson SYNTHETICALLY instead: a 1e-13-absolute
        # perturbation (far inside NORM_ABS_TOL) must explode the
        # relative currency at an underflowed denominator while staying
        # acceptable in the absolute currency the branch uses.
        synthetic = cur[0] + 1e-13
        synthetic_rel = abs(synthetic - head[0]) / max(abs(head[0]), 1e-300)
        self.assertGreater(
            synthetic_rel, REL_TOL,
            'the underflow fixture no longer makes relative comparison '
            'meaningless; re-engineer the fixture denominator')
        self.assertLessEqual(abs(synthetic - head[0]), NORM_ABS_TOL)
        self._compare(cur, head)
        self.record_comparison()

    def _compare(self, cur, head) -> None:
        cur = np.atleast_1d(cur)
        head = np.atleast_1d(head)
        for value_cur, value_head in zip(cur, head):
            underflow = abs(value_head) < NORM_UNDERFLOW_FLOOR
            absolute = abs(value_cur - value_head)
            if underflow:
                self.assertLessEqual(absolute, NORM_ABS_TOL)
                relative = absolute / max(abs(value_head), 1e-300)
            else:
                relative = absolute / abs(value_head)
                self.assertLessEqual(relative, REL_TOL)
            self._scatter.append((abs(value_head), relative, underflow))

    @classmethod
    def tearDownClass(cls) -> None:
        if not cls._scatter:
            return
        _ensure_output_dir()
        denom, rel, under = zip(*cls._scatter)
        under = np.array(under)
        rel = np.maximum(rel, 1e-18)
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.scatter(np.array(denom)[~under], np.array(rel)[~under], s=28,
                     label='relative branch')
        axis.scatter(np.array(denom)[under], np.array(rel)[under], s=42,
                     marker='x', color='crimson', label='absolute branch')
        axis.axvline(NORM_UNDERFLOW_FLOOR, color='k', ls='--',
                     label=f'floor = {NORM_UNDERFLOW_FLOOR:g}')
        axis.axhline(REL_TOL, color='grey', ls=':', label=f'REL_TOL={REL_TOL:g}')
        axis.set_xscale('log')
        axis.set_yscale('log')
        axis.set_xlabel('|norm denominator| (h_L|h_L)')
        axis.set_ylabel('relative error vs HEAD')
        axis.set_title('Lever 2: _norm_term preservation vs norm magnitude')
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(
            _OUTPUT_DIR, 'lever2_norm_term_relerror_vs_denominator.png'),
            dpi=110)
        plt.close(fig)


class NormTermSelfFalsificationTestCase(_LeverTestCase):
    """Prove the Lever-2 guard goes red under a genuinely wrong contraction."""

    def test_wrong_reassociation_breaks_relative_bound(self) -> None:
        inputs = _random_norm_inputs(0)
        head_norm = _head_norm_term()
        head = head_norm(*inputs)

        def wrong_norm(b_moments, *rest):
            # A DIFFERENT reduction (swap two mode-pair coefficients) -- not
            # a rounding re-order but a genuine algebra change.
            swapped = [b_moments[0], b_moments[2], b_moments[1], b_moments[3]]
            return head_norm(swapped, *rest)

        mutated = wrong_norm(*inputs)
        rel = np.max(np.abs(mutated - head)
                     / np.maximum(np.abs(head), 1e-300))
        # The honest <= 1e-10 relative bound must FAIL against the mutant.
        self.assertGreater(rel, REL_TOL)
        with self.assertRaises(AssertionError):
            self.assertLessEqual(rel, REL_TOL)
        self.record_comparison()

    def test_absolute_tolerance_still_catches_gross_error(self) -> None:
        inputs = _underflow_norm_inputs()
        head_norm = _head_norm_term()
        head = head_norm(*inputs)
        # A gross additive error at the underflowing detector must break the
        # absolute bound too (the absolute branch is not a blank cheque).
        broken = head.copy()
        broken[0] = broken[0] + 10.0 * NORM_ABS_TOL
        with self.assertRaises(AssertionError):
            self.assertLessEqual(abs(broken[0] - head[0]), NORM_ABS_TOL)
        self.record_comparison()


# --------------------------------------------------------------------------
# Lever 3: node-parallel exact Schwinger -- byte-identity + refusal identity.
# --------------------------------------------------------------------------
#: Wave-branch frequencies (all ``w <= W_CEILING_SCHWINGER``, so every node
#: enters the node-parallel batch).
_W_WAVE = np.array([5.0, 18.0, 40.0, 55.0, 59.0])

#: Positive-parity source positions: inside-caustic (4-image) and
#: outside-caustic (2-image) for the mid shear.
_L3_POS_SOURCES = (np.array([0.10, 0.05]), np.array([0.70, 0.35]))

#: Saddle-host source positions (two representative 2-image points).
_L3_SAD_SOURCES = (np.array([0.10, 0.05]), np.array([0.40, 0.20]))

#: The genuine njit map's pure-Python fallback, captured ONCE before any
#: test patches ``operator._schwinger_raw_integral_map`` (a mutant patch
#: replaces the global with a plain function that has no ``.py_func``).
_REAL_MAP_PYFUNC = operator._schwinger_raw_integral_map.py_func


def _serial_grid(grid_func, *args, **kwargs):
    """Evaluate a grid with the njit map swapped for its ``.py_func``.

    The ``.py_func`` runs ``numba.prange`` as a plain sequential ``range``,
    so this is a genuine single-threaded loop over the same nodes -- the
    serial oracle the node-parallel path must reproduce bit-for-bit.
    """
    with mock.patch.object(operator, '_schwinger_raw_integral_map',
                           operator._schwinger_raw_integral_map.py_func):
        return grid_func(*args, **kwargs)


class NodeParallelByteIdentityTestCase(_LeverTestCase):
    """Node-parallel grid values are bit-for-bit the serial ``f_schwinger``."""

    _heatmap: list[tuple[float, float, float]] = []

    def test_positive_parity_grid_byte_identical(self) -> None:
        for gamma in GAMMA_POSITIVE:
            for source in _L3_POS_SOURCES:
                with self.subTest(gamma=gamma, source=tuple(source)):
                    parallel = operator._positive_parity_grid(
                        _W_WAVE, source, gamma)
                    serial = _serial_grid(
                        operator._positive_parity_grid, _W_WAVE, source, gamma)
                    self._assert_grid_identical(parallel, serial, gamma)
                    self.record_comparison()

    def test_saddle_grid_byte_identical(self) -> None:
        for gamma in GAMMA_SADDLE:
            for source in _L3_SAD_SOURCES:
                with self.subTest(gamma=gamma, source=tuple(source)):
                    parallel = operator._saddle_grid(_W_WAVE, source, gamma)
                    serial = _serial_grid(
                        operator._saddle_grid, _W_WAVE, source, gamma)
                    # _saddle_grid returns a BARE values array (not a tuple).
                    diff = np.abs(parallel - serial)
                    self.assertEqual(float(np.max(diff)), BYTE_EXACT)
                    for w_node, node_diff in zip(_W_WAVE, diff):
                        self._heatmap.append((w_node, gamma, float(node_diff)))
                    self.record_comparison()

    def _assert_grid_identical(self, parallel, serial, gamma) -> None:
        values_p, orders_p, conv_p, tails_p, ratios_p = parallel
        values_s, orders_s, conv_s, tails_s, ratios_s = serial
        diff = np.abs(values_p - values_s)
        self.assertEqual(float(np.max(diff)), BYTE_EXACT)
        # Diagnostic arrays match bit-for-bit as well.
        np.testing.assert_array_equal(orders_p, orders_s)
        np.testing.assert_array_equal(conv_p, conv_s)
        np.testing.assert_array_equal(tails_p, tails_s)
        np.testing.assert_array_equal(ratios_p, ratios_s)
        for w_node, node_diff in zip(_W_WAVE, diff):
            self._heatmap.append((w_node, gamma, float(node_diff)))

    @classmethod
    def tearDownClass(cls) -> None:
        if not cls._heatmap:
            return
        _ensure_output_dir()
        w_vals, g_vals, diffs = zip(*cls._heatmap)
        fig, axis = plt.subplots(figsize=(6, 4))
        scatter = axis.scatter(w_vals, g_vals,
                               c=np.maximum(diffs, 1e-20), s=90,
                               cmap='viridis')
        axis.set_xlabel('w (dimensionless frequency)')
        axis.set_ylabel('gamma (shear)')
        axis.set_title('Lever 3: |parallel - serial| per node (uniformly 0)')
        fig.colorbar(scatter, ax=axis, label='|parallel - serial|')
        fig.tight_layout()
        fig.savefig(os.path.join(
            _OUTPUT_DIR, 'lever3_parallel_minus_serial_heatmap.png'), dpi=110)
        plt.close(fig)


class NodeParallelRefusalIdentityTestCase(_LeverTestCase):
    """Any-node-refuses -> whole-grid refuses, identically for serial/parallel.

    A deterministic refusal is forced by tightening
    ``_schwinger._CERTIFICATION_TOL`` to 0: every wave node then fails its
    paired-rule certification, so the grid must raise the named
    `SchwingerCertificationError` -- both on the parallel path and on the
    ``.py_func`` serial path, and reproducibly across repeated parallel
    runs (scheduling-independent).
    """

    def test_parallel_and_serial_refuse_together(self) -> None:
        source = _L3_POS_SOURCES[0]
        for gamma in GAMMA_POSITIVE:
            with self.subTest(gamma=gamma):
                with mock.patch.object(_schwinger, '_CERTIFICATION_TOL', 0.0):
                    with self.assertRaises(
                            _schwinger.SchwingerCertificationError):
                        operator._positive_parity_grid(_W_WAVE, source, gamma)
                    with self.assertRaises(
                            _schwinger.SchwingerCertificationError):
                        _serial_grid(operator._positive_parity_grid,
                                     _W_WAVE, source, gamma)
                self.record_comparison()

    def test_saddle_parallel_and_serial_refuse_together(self) -> None:
        source = _L3_SAD_SOURCES[0]
        for gamma in GAMMA_SADDLE:
            with self.subTest(gamma=gamma):
                with mock.patch.object(_schwinger, '_CERTIFICATION_TOL', 0.0):
                    with self.assertRaises(
                            _schwinger.SchwingerCertificationError):
                        operator._saddle_grid(_W_WAVE, source, gamma)
                    with self.assertRaises(
                            _schwinger.SchwingerCertificationError):
                        _serial_grid(operator._saddle_grid,
                                     _W_WAVE, source, gamma)
                self.record_comparison()

    def test_any_single_node_refusal_fails_whole_grid(self) -> None:
        # A multi-node grid with forced-refusing nodes (tol 0) must refuse as
        # a whole -- the whole-grid refusal reduction over the node ordering.
        source = _L3_POS_SOURCES[1]
        mixed = np.array([5.0, 18.0, 40.0])
        # Baseline: the honest grid serves this array.
        served = operator._positive_parity_grid(mixed, source, 0.5)[0]
        self.assertTrue(np.all(np.isfinite(served)))
        with mock.patch.object(_schwinger, '_CERTIFICATION_TOL', 0.0):
            with self.assertRaises(_schwinger.SchwingerCertificationError):
                operator._positive_parity_grid(mixed, source, 0.5)
        self.record_comparison()

    def test_repeated_parallel_refusal_is_scheduling_independent(self) -> None:
        source = _L3_POS_SOURCES[0]
        with mock.patch.object(_schwinger, '_CERTIFICATION_TOL', 0.0):
            for attempt in range(3):
                with self.subTest(attempt=attempt):
                    with self.assertRaises(
                            _schwinger.SchwingerCertificationError):
                        operator._positive_parity_grid(_W_WAVE, source, 0.5)
                    self.record_comparison()


class NodeParallelSelfFalsificationTestCase(_LeverTestCase):
    """F010: mutating the map's certification reduction breaks refusal identity.

    The per-node certification compares the coarse (``int_n``) and refined
    (``int_2n``) raw integrals.  A mutant map that returns ``int_2n ==
    int_n`` makes every node's difference zero, so it certifies
    unconditionally -- the parallel path would then SERVE a config the
    serial path refuses.  Under the forced-refusal setup the honest grid
    raises but the mutant does not; that divergence is exactly what the
    refusal-identity test guards, proving the guard has teeth.
    """

    @staticmethod
    def _cert_collapsing_map(*args, **kwargs):
        """Real ``.py_func`` integrals, but ``int_2n`` collapsed onto ``int_n``.

        This mutates ONLY the certification reduction (the coarse/refined
        difference), via the map's ``.py_func``, per the F010 spec.
        """
        int_n, _int_2n = _REAL_MAP_PYFUNC(*args, **kwargs)
        return int_n, int_n.copy()

    def test_collapsed_certification_serves_a_refused_config(self) -> None:
        source = _L3_POS_SOURCES[0]
        gamma = 0.5
        with mock.patch.object(_schwinger, '_CERTIFICATION_TOL', 0.0):
            # Honest path: refuses (this is the property under guard).
            with self.assertRaises(_schwinger.SchwingerCertificationError):
                operator._positive_parity_grid(_W_WAVE, source, gamma)
            # Mutant certification reduction: SERVES the same refused config.
            with mock.patch.object(operator, '_schwinger_raw_integral_map',
                                   self._cert_collapsing_map):
                mutant_values = operator._positive_parity_grid(
                    _W_WAVE, source, gamma)[0]
            self.assertTrue(np.all(np.isfinite(mutant_values)))
        self.record_comparison()

    def test_refusal_identity_assertion_would_fail_under_mutant(self) -> None:
        # Make the divergence explicit: the honest refusal-identity check
        # ("parallel refuses when serial refuses") is FALSE for the mutant.
        source = _L3_POS_SOURCES[0]
        gamma = 0.5

        def mutant_refuses() -> bool:
            with mock.patch.object(_schwinger, '_CERTIFICATION_TOL', 0.0), \
                    mock.patch.object(operator, '_schwinger_raw_integral_map',
                                      self._cert_collapsing_map):
                try:
                    operator._positive_parity_grid(_W_WAVE, source, gamma)
                    return False
                except _schwinger.SchwingerCertificationError:
                    return True

        with self.assertRaises(AssertionError):
            self.assertTrue(
                mutant_refuses(),
                'mutant must refuse for refusal-identity to hold')
        self.record_comparison()


# ==========================================================================
# Lever 4: universal Pearcey P(x, y) spline table (certification + fallback).
# ==========================================================================
# The Pearcey table is a load-time bicubic-spline emulator of the Fresnel-
# demodulated primitive ``P(x, y)`` over a build-time-derived box.  The arm
# consults it first and falls back to the live certified quadrature
# (`_pearcey_cusp.pearcey`) outside the box or on any load/hash anomaly.
# This suite certifies the SERVED value (carrier re-multiplied) against the
# live quadrature ORACLE on a held-out set that deliberately includes the
# semicubical caustic ``27 y^2 = -8 x^3`` -- the worst case, which a plain
# Latin-hypercube draw samples only sparsely -- and both fallback
# directions (out-of-box routing and the F010 content-hash backstop).
#
# Tolerance provenance
# --------------------
# The production pin is ``PEARCEY_PRODUCTION_ABS_PIN = 1e-8`` (the
# `derive_box` default ``oracle_tol``): ABSOLUTE on ``P`` before the cusp
# prefactor, because a relative bound is meaningless at ``P``'s oscillatory
# zeros.  That pin is UNREACHABLE by a minutes-scale in-test fixture -- the
# origin cusp needs a far denser grid than can be sampled here (each node is
# a ~45 ms certified quadrature).  So the fast gate is a measured
# fixture-scale floor (`PEARCEY_FIXTURE_ABS_FLOOR`) PAIRED with a
# budget-independent monotone-refinement control (bicubic error ~ ``h**4``
# falls with node count toward the pin); the strict ``1e-8`` gate is kept as
# an honest ``@expectedFailure`` witnessing that the fixture cannot reach it
# while the shipped offline table (denser grid) does.  This is the
# knowledge-base pattern for an unreachable production tolerance -- never
# widen the real gate.

#: Half-widths of the in-test Pearcey box.  Deliberately small (the
#: `derive_box` fan-march over a coarse ray set overshoots badly) so the
#: build is a minutes-scale fixture; the served caustic segment
#: ``|y| = sqrt(8/27) |x|**1.5`` stays strictly inside
#: (``sqrt(8/27) * 0.5**1.5 = 0.19 < 0.42``).
PEARCEY_BOX = {'x_max': 0.5, 'y_max': 0.42, 'margin': 0.15,
               'oracle_tol': 1e-8}

#: Knot-grading exponent (``> 1`` clusters knots near the caustic/cusp),
#: matching `_pearcey_table.build_table`'s default.
PEARCEY_GRADING_POWER = 1.6

#: Coarse / fine grid sizes for the fixture and the refinement control.
PEARCEY_N_COARSE = 61
PEARCEY_N_FINE = 91

#: Production ABSOLUTE pin on ``P`` before the prefactor (the shipped
#: `derive_box` ``oracle_tol``); unreachable at fixture scale.
PEARCEY_PRODUCTION_ABS_PIN = 1e-8

#: Fast fixture-scale floor: measured worst-case held-out error at
#: ``n = 91`` is ``~2.7e-5`` (LHS and caustic both ~3e-5); ``1e-4`` clears
#: it with ~3.5x margin against seed/point variation.
PEARCEY_FIXTURE_ABS_FLOOR = 1e-4

#: Bicubic refinement must contract: measured ``err(91)/err(61) ~ 0.50``
#: (``h**4`` scaling); ``0.7`` is the budget-independent convergence gate
#: witnessing the trajectory toward the ``1e-8`` pin.
PEARCEY_CONVERGENCE_RATIO = 0.7

#: The dense caustic line must independently attain the worst-case error
#: scale: ``caustic_max >= 0.5 * overall_max`` (measured ~0.98).  This is
#: what a plain LHS draw can miss.
PEARCEY_CAUSTIC_SHARE_MIN = 0.5

#: The smooth bulk is the caustic-free positive-``x`` half (the fold
#: caustic lives at ``x <= 0``).  Held-out points with ``x > 0.15`` have
#: error ~``8e-9`` -- four orders below the caustic max -- so an LHS draw
#: landing in the smooth bulk vastly understates the true worst case.
PEARCEY_SMOOTH_X_MIN = 0.15
PEARCEY_FARFIELD_MAX = 1e-7

#: Held-out sample budget (kept modest -- each oracle point is a ~45 ms
#: certified quadrature).
PEARCEY_N_LHS = 400
PEARCEY_N_CAUSTIC = 60

def _pearcey_origin_anchor() -> complex:
    """``P(0, 0) = Gamma(1/4)/2 * e^{i pi/8}`` (independent closed form)."""
    return math.gamma(0.25) / 2.0 * complex(math.cos(math.pi / 8.0),
                                            math.sin(math.pi / 8.0))


def _caustic_line(x_max: float, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense samples of the semicubical caustic ``27 y^2 = -8 x^3``.

    The cusp caustic lives at ``x <= 0`` (``-8 x**3 >= 0``); both ``+/-y``
    branches are traced from the cusp vertex out to the left box edge.
    """
    x_axis = np.linspace(-x_max, 0.0, n_points)
    y_axis = np.sqrt(np.clip(-8.0 * x_axis ** 3 / 27.0, 0.0, None))
    xs = np.concatenate([x_axis, x_axis])
    ys = np.concatenate([y_axis, -y_axis])
    return xs, ys


@functools.lru_cache(maxsize=None)
def _pearcey_held_out():
    """Held-out (x, y): Latin hypercube + box corners + dense caustic line.

    Returns four index-aligned arrays plus slice metadata so the caller can
    isolate the LHS, corner and caustic contributions.
    """
    rng = np.random.default_rng(0)
    x_max = PEARCEY_BOX['x_max']
    y_max = PEARCEY_BOX['y_max']
    n = PEARCEY_N_LHS
    cx = (np.arange(n) + rng.random(n)) / n
    cy = (np.arange(n) + rng.random(n)) / n
    rng.shuffle(cy)
    lhs_x = (2.0 * cx - 1.0) * x_max
    lhs_y = (2.0 * cy - 1.0) * y_max
    corner_x = np.array([-1.0, -1.0, 1.0, 1.0]) * x_max
    corner_y = np.array([-1.0, 1.0, -1.0, 1.0]) * y_max
    caustic_x, caustic_y = _caustic_line(x_max, PEARCEY_N_CAUSTIC)
    xs = np.concatenate([lhs_x, corner_x, caustic_x])
    ys = np.concatenate([lhs_y, corner_y, caustic_y])
    n_lhs = lhs_x.size
    n_corner = corner_x.size
    slices = {'lhs': slice(0, n_lhs),
              'corner': slice(n_lhs, n_lhs + n_corner),
              'caustic': slice(n_lhs + n_corner, xs.size)}
    return xs, ys, slices


@functools.lru_cache(maxsize=None)
def _pearcey_oracle_values():
    """Live certified-quadrature ``P`` at every held-out point (once)."""
    xs, ys, _ = _pearcey_held_out()
    values = np.empty(xs.size, dtype=complex)
    for index, (x, y) in enumerate(zip(xs, ys)):
        exact = _pearcey_cusp.pearcey(float(x), float(y))
        if exact is None:
            raise RuntimeError(
                f'Pearcey oracle declined at held-out node '
                f'(x={x:.6g}, y={y:.6g}); the box should be all-certified.')
        values[index] = exact
    return values


@functools.lru_cache(maxsize=None)
def _pearcey_table_of(n_nodes: int) -> _pearcey_table.PearceyTable:
    """Build (once) a fixture table at ``n_nodes`` per axis over the box."""
    return _pearcey_table.build_table(
        dict(PEARCEY_BOX), n_x=n_nodes, n_y=n_nodes,
        grading_power=PEARCEY_GRADING_POWER)


def _pearcey_served_errors(n_nodes: int) -> np.ndarray:
    """Absolute table-vs-oracle error at every held-out point."""
    table = _pearcey_table_of(n_nodes)
    xs, ys, _ = _pearcey_held_out()
    oracle = _pearcey_oracle_values()
    served = np.array([table.evaluate(float(x), float(y))
                       for x, y in zip(xs, ys)], dtype=complex)
    return np.abs(served - oracle)


class PearceyTableCertificationTestCase(_LeverTestCase):
    """The served table reproduces the live-quadrature Pearcey primitive.

    Independence (F002): the ORACLE is the live certified quadrature
    (`_pearcey_cusp.pearcey`, a rotated steepest-descent contour), a
    distinct object from the bicubic spline the table interpolates.  The
    quadrature oracle is itself anchored to the closed form
    ``P(0, 0) = Gamma(1/4)/2 e^{i pi/8}`` so the whole chain is not
    self-referential.  Error is ABSOLUTE on ``P`` before the prefactor
    (relative error is meaningless at ``P``'s oscillatory zeros).
    """

    def test_oracle_matches_closed_form_at_origin(self) -> None:
        # Anchor the quadrature oracle to an independent closed form so the
        # table-vs-oracle comparison is not circular.
        served = _pearcey_cusp.pearcey(0.0, 0.0)
        self.assertIsNotNone(served)
        self.assertLess(abs(served - _pearcey_origin_anchor()), 1e-12)
        self.record_comparison()

    def test_served_table_matches_quadrature_within_fixture_floor(self) -> None:
        errors = _pearcey_served_errors(PEARCEY_N_FINE)
        worst = float(errors.max())
        self.assertLess(
            worst, PEARCEY_FIXTURE_ABS_FLOOR,
            f'worst held-out abs error {worst:.3e} exceeds the fixture floor '
            f'{PEARCEY_FIXTURE_ABS_FLOOR:.1e}.')
        self.record_comparison()

    def test_caustic_line_attains_worst_case_scale(self) -> None:
        # The dense caustic line -- not a plain LHS draw -- is what pins the
        # worst case; assert it reaches at least half the overall maximum.
        _, _, slices = _pearcey_held_out()
        errors = _pearcey_served_errors(PEARCEY_N_FINE)
        overall = float(errors.max())
        caustic = float(errors[slices['caustic']].max())
        self.assertGreaterEqual(
            caustic, PEARCEY_CAUSTIC_SHARE_MIN * overall,
            f'caustic-line max {caustic:.3e} is below '
            f'{PEARCEY_CAUSTIC_SHARE_MIN} x overall {overall:.3e}; the '
            f'worst case is not being sampled on the caustic.')
        self.record_comparison()

    def test_far_field_error_is_orders_below_caustic(self) -> None:
        # The smooth bulk is easy: points far from the cusp have error
        # orders below the caustic max, so an LHS point in the bulk badly
        # understates the true worst case (why the dense caustic line is
        # required).
        xs, ys, slices = _pearcey_held_out()
        errors = _pearcey_served_errors(PEARCEY_N_FINE)
        far = xs > PEARCEY_SMOOTH_X_MIN
        self.assertTrue(np.any(far))
        far_max = float(errors[far].max())
        caustic_max = float(errors[slices['caustic']].max())
        self.assertLess(
            far_max, PEARCEY_FARFIELD_MAX,
            f'far-field max {far_max:.3e} exceeds {PEARCEY_FARFIELD_MAX:.1e}.')
        self.assertLess(far_max, caustic_max)
        self.record_comparison()

    def test_bicubic_refinement_contracts_toward_pin(self) -> None:
        # Budget-independent positive control: the held-out error contracts
        # with node count (bicubic ~ h**4), witnessing the trajectory toward
        # the 1e-8 production pin that the fixture itself cannot reach.
        coarse = float(_pearcey_served_errors(PEARCEY_N_COARSE).max())
        fine = float(_pearcey_served_errors(PEARCEY_N_FINE).max())
        self.assertLess(
            fine, PEARCEY_CONVERGENCE_RATIO * coarse,
            f'refinement did not contract: err({PEARCEY_N_FINE})={fine:.3e} '
            f'vs err({PEARCEY_N_COARSE})={coarse:.3e}, ratio '
            f'{fine / coarse:.3f} > {PEARCEY_CONVERGENCE_RATIO}.')
        self.record_comparison()

    def test_diagnostic_error_scatter(self) -> None:
        _ensure_output_dir()
        xs, ys, slices = _pearcey_held_out()
        errors = _pearcey_served_errors(PEARCEY_N_FINE)
        floor = np.finfo(float).tiny
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        scatter = ax.scatter(
            xs, ys, c=np.log10(np.maximum(errors, floor)),
            s=10, cmap='viridis')
        caustic_x, caustic_y = _caustic_line(PEARCEY_BOX['x_max'], 200)
        ax.plot(caustic_x[:200], caustic_y[:200], 'r-', lw=0.8,
                label='caustic 27y^2=-8x^3')
        ax.plot(caustic_x[200:], caustic_y[200:], 'r-', lw=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Lever 4: |table - quadrature| over the box '
                     f'(n={PEARCEY_N_FINE})')
        fig.colorbar(scatter, ax=ax, label='log10 |P_table - P_quad|')
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(
            _OUTPUT_DIR, 'lever4_pearcey_abs_error_over_box.png'), dpi=110)
        plt.close(fig)
        self.assertTrue(os.path.exists(os.path.join(
            _OUTPUT_DIR, 'lever4_pearcey_abs_error_over_box.png')))
        self.record_comparison()


class PearceyTableProductionPinTestCase(_LeverTestCase):
    """The shipped ABSOLUTE pin is ``1e-8`` and the fixture cannot reach it.

    The strict production gate is kept honest, not widened: the
    ``@expectedFailure`` witnesses that the minutes-scale fixture (which
    cannot resolve the origin cusp to ``1e-8``) fails the pin, while the
    convergence control in `PearceyTableCertificationTestCase` shows the
    error contracting toward it.  The pin value itself is pinned to the
    `derive_box` default so a silent loosening of the shipped tolerance
    trips this suite.
    """

    def test_production_pin_matches_derive_box_default(self) -> None:
        default = inspect.signature(
            _pearcey_table.derive_box).parameters['oracle_tol'].default
        self.assertEqual(default, PEARCEY_PRODUCTION_ABS_PIN)
        # The fixture table also carries the pin in its provenance.
        table = _pearcey_table_of(PEARCEY_N_FINE)
        self.assertEqual(float(table.provenance['oracle_tol']),
                         PEARCEY_PRODUCTION_ABS_PIN)
        self.record_comparison()

    @unittest.expectedFailure
    def test_fixture_cannot_reach_production_pin(self) -> None:
        # HONEST RED: the fixture-scale table is ~2.7e-5, far above 1e-8.
        # The shipped offline table (denser grid) meets the pin; this
        # expected failure documents the gap without widening the gate.
        worst = float(_pearcey_served_errors(PEARCEY_N_FINE).max())
        self.record_comparison()
        self.assertLess(worst, PEARCEY_PRODUCTION_ABS_PIN)


def _write_valid_table(path: Path, n_nodes: int) -> _pearcey_table.PearceyTable:
    """Build a fixture table and serialize it (hash intact) to ``path``."""
    table = _pearcey_table_of(n_nodes)
    _pearcey_table.save_table(table, path)
    return table


def _rewrite_npz_with_stale_hash(src: Path, dst: Path,
                                 mutate) -> None:
    """Copy ``src`` to ``dst`` with ``mutate(demod_real)`` but OLD provenance.

    The stored ``content_hash`` (in the untouched provenance scalar) no
    longer matches the mutated array, so `PearceyTable.load` must refuse.
    """
    with np.load(src, allow_pickle=False) as data:
        x_grid = np.array(data['x_grid'])
        y_grid = np.array(data['y_grid'])
        demod_real = np.array(data['demod_real'])
        demod_imag = np.array(data['demod_imag'])
        provenance_scalar = np.array(data['provenance'])
    demod_real = mutate(demod_real)
    np.savez(dst, x_grid=x_grid, y_grid=y_grid, demod_real=demod_real,
             demod_imag=demod_imag, provenance=provenance_scalar)


class PearceyTableFallbackTestCase(_LeverTestCase):
    """Both fallback directions: out-of-box routing and load/hash anomalies.

    A point outside the box declines (the arm re-runs live quadrature);
    a corrupt artifact is refused at load so a wrong value is never
    served.  The process-global table is saved and restored so these
    tests do not leak install state into the rest of the suite.
    """

    def setUp(self) -> None:
        super().setUp()
        self._saved_global = _pearcey_cusp.get_pearcey_table()
        self._tmpdir = tempfile.mkdtemp(prefix='pearcey_fallback_')

    def tearDown(self) -> None:
        _pearcey_cusp.set_pearcey_table(self._saved_global)
        super().tearDown()

    def test_outside_box_declines_to_none(self) -> None:
        table = _pearcey_table_of(PEARCEY_N_COARSE)
        outside = [(table.x_max * 1.01, 0.0),
                   (0.0, table.y_max * 1.01),
                   (table.x_max * 2.0, table.y_max * 2.0)]
        for x, y in outside:
            with self.subTest(x=x, y=y):
                self.assertIsNone(table.evaluate(x, y))
                self.record_comparison()

    def test_consult_routes_outside_box_to_live_quadrature(self) -> None:
        table = _pearcey_table_of(PEARCEY_N_COARSE)
        x, y = table.x_max * 1.5, table.y_max * 0.2
        served = _pearcey_cusp._consult_pearcey(x, y, table)
        direct = _pearcey_cusp.pearcey(x, y)
        self.assertIsNotNone(direct)
        # Byte-identical: outside the box the consult IS the live quadrature.
        self.assertEqual(complex(served), complex(direct))
        self.record_comparison()

    def test_consult_uses_table_inside_box(self) -> None:
        table = _pearcey_table_of(PEARCEY_N_FINE)
        x, y = 0.11, -0.07
        served = _pearcey_cusp._consult_pearcey(x, y, table)
        table_value = table.evaluate(x, y)
        self.assertIsNotNone(table_value)
        self.assertEqual(complex(served), complex(table_value))
        # And the table value tracks the oracle within the fixture floor.
        oracle = _pearcey_cusp.pearcey(x, y)
        self.assertLess(abs(complex(served) - oracle),
                        PEARCEY_FIXTURE_ABS_FLOOR)
        self.record_comparison()

    def test_no_serve_gap_across_box_edge(self) -> None:
        # No jump at the handoff: at the SAME near-edge interior point the
        # table (which serves inside) and the live quadrature (which serves
        # outside) agree within the fixture floor, so switching methods at
        # the box edge introduces no discontinuity.  (Comparing two
        # DIFFERENT points would instead measure P's own variation.)
        table = _pearcey_table_of(PEARCEY_N_FINE)
        for y in (0.05, -0.15, 0.30):
            with self.subTest(y=y):
                x_edge = table.x_max * 0.999
                table_value = _pearcey_cusp._consult_pearcey(
                    x_edge, y, table)
                quad_value = _pearcey_cusp.pearcey(x_edge, y)
                self.assertIsNotNone(quad_value)
                self.assertLess(abs(complex(table_value) - quad_value),
                                PEARCEY_FIXTURE_ABS_FLOOR)
                self.record_comparison()

    def test_save_load_round_trip_is_faithful(self) -> None:
        path = Path(self._tmpdir) / 'valid.npz'
        original = _write_valid_table(path, PEARCEY_N_COARSE)
        loaded = _pearcey_table.PearceyTable.load(path)
        np.testing.assert_array_equal(loaded.x_grid, original.x_grid)
        np.testing.assert_array_equal(loaded.y_grid, original.y_grid)
        np.testing.assert_array_equal(loaded.demod_real, original.demod_real)
        np.testing.assert_array_equal(loaded.demod_imag, original.demod_imag)
        for x, y in [(0.1, 0.1), (-0.2, 0.05), (0.0, 0.0)]:
            self.assertEqual(complex(loaded.evaluate(x, y)),
                             complex(original.evaluate(x, y)))
            self.record_comparison()

    def test_use_pearcey_table_installs_and_clears(self) -> None:
        path = Path(self._tmpdir) / 'installable.npz'
        _write_valid_table(path, PEARCEY_N_COARSE)
        self.assertTrue(_pearcey_cusp.use_pearcey_table(str(path)))
        self.assertIsNotNone(_pearcey_cusp.get_pearcey_table())
        _pearcey_cusp.set_pearcey_table(None)
        self.assertIsNone(_pearcey_cusp.get_pearcey_table())
        self.record_comparison()

    def test_hash_mismatch_is_refused_at_load(self) -> None:
        good = Path(self._tmpdir) / 'good.npz'
        bad = Path(self._tmpdir) / 'bad.npz'
        _write_valid_table(good, PEARCEY_N_COARSE)
        # The good artifact loads.
        _pearcey_table.PearceyTable.load(good)
        # A content mutation with a stale hash is refused.
        _rewrite_npz_with_stale_hash(
            good, bad, lambda arr: arr + 1e-9)
        with self.assertRaises(ValueError):
            _pearcey_table.PearceyTable.load(bad)
        self.record_comparison()

    def test_corrupt_artifact_falls_back_not_serves(self) -> None:
        good = Path(self._tmpdir) / 'good2.npz'
        bad = Path(self._tmpdir) / 'bad2.npz'
        _write_valid_table(good, PEARCEY_N_COARSE)
        _rewrite_npz_with_stale_hash(good, bad, lambda arr: arr + 1e-9)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            installed = _pearcey_cusp.use_pearcey_table(str(bad))
        # The corrupt table is NOT installed; the global stays cleared and
        # the arm transparently uses the live quadrature.
        self.assertFalse(installed)
        self.assertIsNone(_pearcey_cusp.get_pearcey_table())
        served = _pearcey_cusp._consult_pearcey(0.1, 0.1, None)
        self.assertEqual(complex(served),
                         complex(_pearcey_cusp.pearcey(0.1, 0.1)))
        self.record_comparison()


class PearceyTableSelfFalsificationTestCase(_LeverTestCase):
    """F010: the content-hash backstop detects a corrupted table.

    The mutation is consequential (it moves a served value well above the
    floor) AND detected (the loader refuses), and a positive control shows
    the loader is not always-raising -- an intact artifact loads.
    """

    def setUp(self) -> None:
        super().setUp()
        self._tmpdir = tempfile.mkdtemp(prefix='pearcey_f010_')

    def test_one_ulp_mutation_breaks_the_hash(self) -> None:
        good = Path(self._tmpdir) / 'ok.npz'
        tampered = Path(self._tmpdir) / 'ulp.npz'
        _write_valid_table(good, PEARCEY_N_COARSE)
        # Positive control: the intact artifact loads (check has teeth, not
        # a blanket refusal).
        _pearcey_table.PearceyTable.load(good)

        def one_ulp(arr: np.ndarray) -> np.ndarray:
            out = arr.copy()
            out[out.shape[0] // 2, out.shape[1] // 2] = np.nextafter(
                out[out.shape[0] // 2, out.shape[1] // 2], np.inf)
            return out

        _rewrite_npz_with_stale_hash(good, tampered, one_ulp)
        with self.assertRaises(ValueError):
            _pearcey_table.PearceyTable.load(tampered)
        self.record_comparison()

    def test_corruption_is_consequential(self) -> None:
        # A gross corruption of an interior node moves the served value far
        # above the floor -- so the hash backstop is guarding a real error,
        # not a cosmetic one.
        table = _pearcey_table_of(PEARCEY_N_COARSE)
        interior_i = table.x_grid.size // 2
        interior_j = table.y_grid.size // 2
        x = float(table.x_grid[interior_i])
        y = float(table.y_grid[interior_j])
        clean = table.evaluate(x, y)
        corrupt = _pearcey_table.PearceyTable.from_grid(
            table.x_grid, table.y_grid,
            table.demod_real.copy(), table.demod_imag.copy(),
            dict(table.provenance))
        corrupt.demod_real[interior_i, interior_j] += 0.5
        corrupt_spline = _pearcey_table.PearceyTable.from_grid(
            corrupt.x_grid, corrupt.y_grid, corrupt.demod_real,
            corrupt.demod_imag, corrupt.provenance)
        moved = abs(complex(corrupt_spline.evaluate(x, y))
                    - complex(clean))
        self.assertGreater(moved, PEARCEY_FIXTURE_ABS_FLOOR)
        self.record_comparison()


# ==========================================================================
# Lever 5: L_MAX gate hardening (enforcement bracket + geometric guards).
# ==========================================================================
# ``select_branch`` routes a resolved cluster to the geometric asymptote
# once the cancellation exponent ``L = w |y'|`` exceeds ``L_MAX``.  The
# shipped ``L_MAX = 48`` must sit inside the CERTIFIED OVERLAP: at or above
# it the geometric asymptote is already accurate (lower edge ``L_geo``), and
# below the wave evaluator's ceiling the exact wave branch still serves
# (upper edge).  This suite measures ``L_geo`` against an INDEPENDENT oracle
# and asserts the double-sided bracket, then enforces the geometric-served
# census guards.
#
# Oracle / evaluator provenance (premise repair, post-Build-8d)
# -------------------------------------------------------------
# The spec's ``L_wave ~ 45-46`` refers to the RETIRED operator-series wave
# evaluator.  Build 8d rerouted BOTH parities of the served (sheared) wave
# branch to the EXACT 1D Schwinger evaluator (`_schwinger.f_schwinger`,
# reached through `operator.F_op`), whose accuracy ceiling is not a soft
# ``L ~ 45-46`` roll-off but a HARD refusal at ``w > W_CEILING_SCHWINGER =
# 60`` -- above it the branch raises `SchwingerCertificationError`, it never
# rides a silent over-tolerance value.  So the enforced upper edge is the
# kernel ceiling with headroom, and the Schwinger evaluator is ALSO the
# independent oracle for the geometric asymptote's error (F002: a 1D
# quadrature versus a stationary-phase image sum share no derivation).  An
# mpmath re-verification of the wave branch itself is infeasible at this
# ``w`` in a minutes-scale suite (its ``e^{pi w/4}`` cancellation needs
# hundreds of high-precision panels), so the wave side rests on the
# evaluator's own paired-rule certification plus the tested hard refusal.

#: Worst resolved config: positive-parity host, off-axis source with
#: ``|y'| = 1`` (``kappa = 0``) so ``L = w`` exactly, and a wide two-image
#: split (``delta_min ~ 2.48``) that keeps ``w*delta_min`` far above
#: ``RHO_END = 4`` across the whole sweep (deep in the resolved regime, not
#: the knife edge).
LEVER5_Y = (0.6, 0.8)
LEVER5_GAMMA = 0.5

#: Cancellation-exponent sweep at production resolution.  ``w = L`` here;
#: the top (60) is the exact-wave ceiling, above which the wave branch
#: hard-refuses.
LEVER5_L_SWEEP = (30.0, 32.0, 34.0, 36.0, 40.0, 45.0, 48.0, 52.0, 56.0, 60.0)

#: Geometric-asymptote accuracy target: ``L_geo`` is the smallest swept
#: ``L`` at which the stationary-phase amplification agrees with the exact
#: wave oracle to this RELATIVE tolerance and stays below it thereafter.
LEVER5_L_GEO_TOL = 1e-4

#: Exact-wave kernel ceiling (`_schwinger.W_CEILING_SCHWINGER`); with
#: ``|y'| = 1`` it is also the ceiling in ``L``.
LEVER5_KERNEL_CEILING = _schwinger.W_CEILING_SCHWINGER

#: Minimum headroom between ``L_MAX`` and the kernel ceiling
#: (``48 <= 60 - 6``); the shipped margin is 12.
LEVER5_HEADROOM_MIN = 6.0

#: Frequencies at which nodes are unambiguously geometric-served
#: (``L = w > L_MAX`` and resolved) -- used for the census guards.
LEVER5_GEOMETRIC_W = (50.0, 55.0, 60.0)

#: A sub-``L_MAX`` frequency where the geometric asymptote is NOT yet
#: accurate (``rel-err > L_GEO_TOL``); a too-low ``L_MAX`` would wrongly
#: route it to geometric optics.
LEVER5_INACCURATE_W = 28.0

#: An above-ceiling frequency (``w > 60``): geometric-servable but the wave
#: branch refuses, so a too-high ``L_MAX`` (routing it to wave) loses
#: availability.
LEVER5_ABOVE_CEILING_W = 62.0

#: Nonzero-gauge drift point for the lever-5 wave-vs-geometric comparison.
#: The default sweep runs at ``beta == kappa == 0``, which makes the
#: eigenframe rotation (``beta``) and the mass-sheet map (``kappa``) the
#: identity -- so those two gauge factors could drift without any test
#: noticing.  This point engages BOTH, well inside the resolved
#: positive-parity regime (``1 - kappa = 0.8 > gamma = 0.5`` and
#: ``w * delta_min >> RHO_END``), and demands the SAME agreement.
LEVER5_GAUGE_BETA = 0.3
LEVER5_GAUGE_KAPPA = 0.2

#: Resolved frequencies for the nonzero-gauge check: ``w <= 60`` so the
#: exact wave branch serves, and ``L`` large enough that the geometric
#: asymptote is already accurate (measured rel-err ``~5e-6``, twenty times
#: inside ``LEVER5_L_GEO_TOL``).
LEVER5_GAUGE_W = (55.0, 58.0)


@functools.lru_cache(maxsize=None)
def _lever5_config():
    """Return ``(source, gamma, matrix, delta_min)`` for the worst config."""
    source = np.array(LEVER5_Y, dtype=float)
    gamma = LEVER5_GAMMA
    matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
    images = geometry.find_images(source, matrix)
    delays = [geometry.delay(image, source, matrix) for image in images]
    delta_min = min(abs(delays[i] - delays[j])
                    for i in range(len(delays))
                    for j in range(i + 1, len(delays)))
    return source, gamma, matrix, float(delta_min)


def _geometric_rel_err(w: float) -> float:
    """Relative error of the geometric asymptote vs the exact wave oracle.

    Oracle: `operator.F_op` (exact 1D Schwinger quadrature).  Candidate:
    `operator.geometric_amplification` (stationary phase).  Independent
    derivations (F002).
    """
    source, gamma, _, _ = _lever5_config()
    geometric = operator.geometric_amplification(w, source, gamma)
    exact = operator.F_op(w, source, gamma)[0]
    return abs(complex(geometric) - complex(exact)) / abs(complex(exact))


def _geometric_rel_err_gauge(w: float, y: np.ndarray, gamma: float,
                             beta: float, kappa: float) -> float:
    """Wave-vs-geometric relative error with the gauge factors engaged.

    Identical comparison to `_geometric_rel_err` (independent derivations,
    F002: `operator.F_op` exact 1D Schwinger quadrature as oracle vs
    `operator.geometric_amplification` stationary phase as candidate) but
    with a nonzero shear orientation ``beta`` and convergence ``kappa``,
    so the eigenframe rotation and mass-sheet map -- both the identity
    when ``beta == kappa == 0`` -- are actually exercised.
    """
    source = np.asarray(y, dtype=float)
    geometric = operator.geometric_amplification(
        w, source, gamma, beta=beta, kappa=kappa)
    exact = operator.F_op(w, source, gamma, beta=beta, kappa=kappa)[0]
    return abs(complex(geometric) - complex(exact)) / abs(complex(exact))


@functools.lru_cache(maxsize=None)
def _measure_L_geo() -> float:
    """Smallest swept ``L`` where the geometric rel-err first stays < tol.

    Returns ``inf`` if no swept ``L`` qualifies (which would itself fail
    the bracket, exposing a geometric asymptote that never certifies).
    """
    sweep = sorted(LEVER5_L_SWEEP)
    errors = {L: _geometric_rel_err(L) for L in sweep}
    for index, L in enumerate(sweep):
        if all(errors[higher] < LEVER5_L_GEO_TOL for higher in sweep[index:]):
            return float(L)
    return math.inf


class LMaxEnforcementBracketTestCase(_LeverTestCase):
    """The shipped ``L_MAX = 48`` lies inside the certified overlap bracket.

    Lower edge: the geometric asymptote is accurate at ``L >= L_geo``
    (measured against the exact wave oracle).  Upper edge: the exact wave
    branch serves up to the kernel ceiling and hard-refuses above it, so
    ``L_MAX`` must clear ``L_geo`` and stay a headroom below the ceiling.
    """

    def test_config_is_deep_in_resolved_regime(self) -> None:
        _, _, _, delta_min = _lever5_config()
        for L in LEVER5_L_SWEEP:
            with self.subTest(L=L):
                # w == L because |y'| == 1; confirm via the production
                # cancellation_exponent, then check the resolution margin.
                source, gamma, _, _ = _lever5_config()
                exponent = operator.cancellation_exponent(L, source, gamma)
                self.assertAlmostEqual(exponent, L, places=9)
                self.assertGreater(L * delta_min, operator.RHO_END)
                self.record_comparison()

    def test_L_geo_is_measured_and_below_L_MAX(self) -> None:
        L_geo = _measure_L_geo()
        self.assertTrue(math.isfinite(L_geo),
                        'geometric asymptote never reached L_GEO_TOL on the '
                        'sweep; there is no certified lower edge.')
        self.assertLessEqual(
            L_geo, operator.L_MAX,
            f'L_geo={L_geo} exceeds L_MAX={operator.L_MAX}: the geometric '
            f'asymptote is not yet accurate where the gate hands off.')
        self.record_comparison()

    def test_wave_geometric_agree_at_nonzero_gauge(self) -> None:
        # The default L-sweep runs at beta == kappa == 0, leaving the
        # eigenframe rotation (beta) and the mass-sheet map (kappa) inert.
        # Engage BOTH at a resolved positive-parity point and demand the
        # same wave-vs-geometric agreement, so a drift in either gauge
        # factor -- silent at the all-zero point -- is caught here.
        source = np.array(LEVER5_Y, dtype=float)
        # Positive parity: 1 - kappa > |gamma|.
        self.assertGreater(1.0 - LEVER5_GAUGE_KAPPA, abs(LEVER5_GAMMA))
        matrix = geometry.macro_matrix(
            LEVER5_GAMMA, LEVER5_GAUGE_BETA, LEVER5_GAUGE_KAPPA)
        images = geometry.find_images(source, matrix)
        delays = [geometry.delay(image, source, matrix) for image in images]
        delta_min = min(abs(delays[i] - delays[j])
                        for i in range(len(delays))
                        for j in range(i + 1, len(delays)))
        for w in LEVER5_GAUGE_W:
            with self.subTest(w=w):
                # Deep in the resolved regime: w * delta_min >> RHO_END.
                self.assertGreater(w * delta_min, operator.RHO_END)
                rel = _geometric_rel_err_gauge(
                    w, source, LEVER5_GAMMA,
                    LEVER5_GAUGE_BETA, LEVER5_GAUGE_KAPPA)
                self.assertLess(
                    rel, LEVER5_L_GEO_TOL,
                    f'nonzero-gauge wave/geometric rel-err {rel:.2e} at '
                    f'w={w} exceeds L_GEO_TOL={LEVER5_L_GEO_TOL:.0e}.')
                self.record_comparison()

    def test_L_MAX_clears_ceiling_with_headroom(self) -> None:
        self.assertLessEqual(
            operator.L_MAX, LEVER5_KERNEL_CEILING - LEVER5_HEADROOM_MIN,
            f'L_MAX={operator.L_MAX} is within {LEVER5_HEADROOM_MIN} of the '
            f'kernel ceiling {LEVER5_KERNEL_CEILING}.')
        self.record_comparison()

    def test_double_sided_bracket_holds_and_pins_48(self) -> None:
        L_geo = _measure_L_geo()
        upper = LEVER5_KERNEL_CEILING - LEVER5_HEADROOM_MIN
        self.assertLessEqual(L_geo, operator.L_MAX)
        self.assertLessEqual(operator.L_MAX, upper)
        # The shipped value is pinned; a silent change trips this.
        self.assertEqual(operator.L_MAX, 48)
        self.record_comparison()

    def test_wave_branch_serves_below_ceiling_refuses_above(self) -> None:
        # The WAVE EVALUATOR is `_schwinger.f_schwinger` (with beta=0,
        # kappa=0 the eigenframe reduction is the identity, so it takes the
        # physical source and shear directly).  It serves below the ceiling
        # and HARD-REFUSES above -- it never rides a silent over-tolerance
        # value.  (The top-level `F_op` instead dispatches an above-ceiling
        # resolved node to the geometric arm, which is exactly why the gate
        # must hand off before the ceiling.)
        source, gamma, _, _ = _lever5_config()
        for w in (48.0, 55.0, LEVER5_KERNEL_CEILING):
            with self.subTest(w=w):
                value = _schwinger.f_schwinger(w, source, gamma)
                self.assertTrue(np.isfinite(value))
                self.record_comparison()
        with self.assertRaises(_schwinger.SchwingerCertificationError):
            _schwinger.f_schwinger(
                LEVER5_KERNEL_CEILING + 1.0, source, gamma)
        self.record_comparison()

    def test_diagnostic_rel_err_vs_L(self) -> None:
        _ensure_output_dir()
        source, gamma, _, _ = _lever5_config()
        sweep = np.array(sorted(LEVER5_L_SWEEP))
        geo_err = np.array([_geometric_rel_err(float(L)) for L in sweep])
        L_geo = _measure_L_geo()
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.semilogy(sweep, geo_err, 'o-', color='C1',
                    label='geometric branch rel-err (vs exact wave)')
        ax.axhline(LEVER5_L_GEO_TOL, color='C1', ls=':',
                   label=f'L_GEO_TOL={LEVER5_L_GEO_TOL:.0e}')
        ax.axvline(L_geo, color='C1', ls='--', label=f'L_geo={L_geo:g}')
        ax.axvline(operator.L_MAX, color='k', lw=2.0,
                   label=f'shipped L_MAX={operator.L_MAX}')
        ax.axvspan(sweep.min(), LEVER5_KERNEL_CEILING, color='C0', alpha=0.08,
                   label='wave branch serves (<= ceiling)')
        ax.axvline(LEVER5_KERNEL_CEILING, color='C0', ls='--',
                   label=f'kernel ceiling={LEVER5_KERNEL_CEILING:g}')
        ax.set_xlabel('cancellation exponent L = w |y\'|')
        ax.set_ylabel('true relative error')
        ax.set_title('Lever 5: certified wave/geometric overlap bracket')
        ax.legend(fontsize=7, loc='upper right')
        fig.tight_layout()
        out = os.path.join(_OUTPUT_DIR,
                           'lever5_wave_geometric_rel_err_vs_L.png')
        fig.savefig(out, dpi=110)
        plt.close(fig)
        self.assertTrue(os.path.exists(out))
        self.record_comparison()


class GeometricCensusGuardTestCase(_LeverTestCase):
    """Every geometric-served node carries a faithful, non-degenerate census.

    Guard (a): the image count matches the quartic `find_images` root count
    and is one of the only valid served counts (2 outside the caustic, 4
    inside).  Guard (b): the Morse parity-sum
    ``sum_a sign(mu_a) == sign(det A) - 1``.  Perturbing either guard makes
    `_certify_geometric_census` raise, so the silent-pass property goes RED.
    Independence (F002): `find_images` is a quartic root solve; the parity
    identity is the Morse index theorem -- distinct derivations.
    """

    def test_served_nodes_pass_both_guards(self) -> None:
        source, gamma, matrix, delta_min = _lever5_config()
        for w in LEVER5_GEOMETRIC_W:
            with self.subTest(w=w):
                exponent = operator.cancellation_exponent(w, source, gamma)
                self.assertEqual(
                    operator.select_branch(w, delta_min, exponent),
                    'geometric')
                images = geometry.find_images(source, matrix)
                self.assertIn(len(images), (2, 4))
                # The guard passes silently on a valid served census.
                operator._certify_geometric_census(images, matrix)
                signed_sum = sum((-1) ** geometry.morse_index(image, matrix)
                                 for image in images)
                expected = (1 if float(np.linalg.det(matrix)) > 0.0
                            else -1) - 1
                self.assertEqual(signed_sum, expected)
                self.record_comparison()

    def test_image_count_guard_rejects_odd_census(self) -> None:
        _, _, matrix, _ = _lever5_config()
        source, _, _, _ = _lever5_config()
        images = geometry.find_images(source, matrix)
        # Positive control: the true census passes.
        operator._certify_geometric_census(list(images), matrix)
        # A duplicated image -> odd count 3 -> refused.
        doctored = list(images) + [images[0].copy()]
        with self.assertRaises(geometry.LensDomainError):
            operator._certify_geometric_census(doctored, matrix)
        self.record_comparison()

    def test_morse_parity_guard_rejects_flipped_sign(self) -> None:
        source, _, matrix, _ = _lever5_config()
        images = geometry.find_images(source, matrix)
        operator._certify_geometric_census(list(images), matrix)

        real_morse = geometry.morse_index
        state = {'n': 0}

        def flip_first(image, mat):
            index = real_morse(image, mat)
            if state['n'] == 0:
                state['n'] += 1
                return index + 1  # odd shift on exactly one image
            state['n'] += 1
            return index

        with mock.patch.object(geometry, 'morse_index', flip_first):
            with self.assertRaises(geometry.LensDomainError):
                operator._certify_geometric_census(list(images), matrix)
        self.record_comparison()

    def test_perturbed_census_makes_amplification_refuse(self) -> None:
        # End-to-end teeth: a dropped image inside geometric_amplification
        # (via a patched find_images) makes the served path RAISE where the
        # honest path returns a finite value.
        source, gamma, _, _ = _lever5_config()
        w = LEVER5_GEOMETRIC_W[0]
        honest = operator.geometric_amplification(w, source, gamma)
        self.assertTrue(np.isfinite(honest))

        real_find = geometry.find_images

        def drop_one(src, mat):
            return real_find(src, mat)[:-1]

        with mock.patch.object(geometry, 'find_images', drop_one):
            with self.assertRaises(geometry.LensDomainError):
                operator.geometric_amplification(w, source, gamma)
        self.record_comparison()


class LMaxSelfFalsificationTestCase(_LeverTestCase):
    """A mis-set ``L_MAX`` is caught -- both too-low and too-high fail.

    Too-low (below ``L_geo``): a sub-accurate node is routed to the
    geometric asymptote and served with ``rel-err > L_GEO_TOL``.  Too-high
    (above the kernel ceiling): a geometric-servable node is routed to the
    wave branch, which refuses -- an availability regression.  The shipped
    ``L_MAX = 48`` avoids both.
    """

    def test_too_low_L_MAX_serves_inaccurate_geometric(self) -> None:
        source, gamma, _, delta_min = _lever5_config()
        w = LEVER5_INACCURATE_W
        exponent = operator.cancellation_exponent(w, source, gamma)
        rel_err = _geometric_rel_err(w)
        # This node's geometric asymptote is NOT yet accurate.
        self.assertGreater(rel_err, LEVER5_L_GEO_TOL)
        # Shipped L_MAX keeps it on the (exact) wave branch.
        self.assertEqual(
            operator.select_branch(w, delta_min, exponent), 'wave')
        # A too-low L_MAX (below this node's L) would route it to geometric,
        # serving the inaccurate value.
        with mock.patch.object(operator, 'L_MAX', 25):
            self.assertEqual(
                operator.select_branch(w, delta_min, exponent), 'geometric')
        self.record_comparison()

    def test_too_high_L_MAX_loses_geometric_availability(self) -> None:
        source, gamma, _, delta_min = _lever5_config()
        w = LEVER5_ABOVE_CEILING_W
        exponent = operator.cancellation_exponent(w, source, gamma)
        # The node is geometric-servable (asymptote certifies its census).
        self.assertTrue(np.isfinite(
            operator.geometric_amplification(w, source, gamma)))
        # The WAVE EVALUATOR cannot serve it (above the kernel ceiling); a
        # 'wave' routing here would refuse.
        with self.assertRaises(_schwinger.SchwingerCertificationError):
            _schwinger.f_schwinger(w, source, gamma)
        # Shipped L_MAX routes it to the (working) geometric branch.
        self.assertEqual(
            operator.select_branch(w, delta_min, exponent), 'geometric')
        # A too-high L_MAX (above the ceiling) routes it to wave -> refusal.
        with mock.patch.object(operator, 'L_MAX', 65):
            self.assertEqual(
                operator.select_branch(w, delta_min, exponent), 'wave')
        self.record_comparison()


if __name__ == '__main__':
    unittest.main()
