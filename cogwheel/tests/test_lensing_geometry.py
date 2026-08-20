"""
Tests for `lensing.chang_refsdal.geometry`, the geometrical-optics
layer of the Chang--Refsdal lens.

WHY THESE ORACLES ARE INDEPENDENT
---------------------------------
Every assertion here is judged against something the module does NOT
compute for itself:

* the QUARTIC CSV REGRESSION gates the image count against
  ``n_multistart`` -- a frozen column produced by a since-deleted
  independent multistart root finder, so it cannot drift with the
  quartic solver it checks;
* the ANALYTIC ORIGIN test uses closed forms (``|x|**2``, ``tau``,
  ``artanh(gamma)``) that hold exactly at ``y = 0`` with no oracle at
  all -- this is where a sign flip or a factor of two in the Fermat
  potential dies;
* the ASTROID test builds the caustic from a LOCAL analytic
  parametrization (never from ``geometry.critical_point``), so a shared
  bug between ``critical_point`` and ``find_images_quartic`` cannot hide;
* the MORSE census is TALLIED and asserted against measured counts,
  because the Euler invariant ``n_min - n_saddle + n_max = 0`` holds for
  both the two- and four-image census and so cannot tell them apart.

TOLERANCES
----------
The analytic-origin closed forms are exact, so ``1e-13`` there is pure
roundoff headroom.  The CSV residual gate is a FRESH ``1e-12`` bound
recomputed now, not the CSV's ``max_quartic_residual`` (a polish detail)
and not the solver's ``3e-8`` acceptance filter (headroom, not achieved
accuracy -- gating there would be vacuous).  All 168 rows clear ``1e-12``
(measured max ~1.9e-13), so no near-caustic exception is warranted and
none is added.  Positions are NEVER asserted near a fold: a double root
there carries only ``sqrt(eps) ~ 1.5e-8``, while delays keep full
accuracy because images are stationary points of the Fermat potential.

SHARD D --- ``r_caustic`` VS THE EXACT CRITICAL CURVE
----------------------------------------------------
``r_caustic`` was reworked from an angular SCAN of the caustic to a
``brentq`` inversion of the exact critical-curve parametrisation.  The
PRIMARY accuracy gate here is INDEPENDENT of the module: on the two
principal axes the Chang--Refsdal astroid cusp radius has the pencil-
and-paper closed form ``2*gamma / sqrt(1 +- gamma)`` (the ``+`` on the
shear axis ``theta = 0``, the ``-`` on the transverse axis
``theta = pi/2``), derived straight from the lens map
``y = A x - x / |x|**2`` and owing nothing to
``geometry.critical_point`` or ``geometry.r_caustic``.  The bar is the
Professor's ``1e-10``; the measured worst case is ~1e-13, so the tight
gate has three orders of headroom.  The historically interesting corner
is ``gamma = 0.9, theta = pi/2``, where the retired scan returned
``5.67376`` -- a 0.32% error -- while the exact value is
``5.692099788303083``; the tight gate would have rejected the old scan
by four orders of magnitude.

The waist invariant ``min_theta r_caustic(gamma, theta) == gamma`` is a
second independent analytic fact (the astroid's minimum source-plane
reach equals the shear); the argmin is found with an INDEPENDENT
``scipy.optimize.minimize_scalar`` (``xatol ~ 1e-6``).  The bar is again
``1e-10`` and NOT ``1e-12`` on purpose: the quadratic flatness of the
minimum suppresses the ~1e-6 argmin error to ~1e-14, but the returned
radius is itself only root-find accurate, so pinning tighter than the
root-find floor would be gating on luck.

A separate, clearly-labelled CONSISTENCY class checks ``r_caustic``
against ``|critical_point(gamma, axis).source|``.  That is NOT an
independent oracle -- both descend from the same caustic parametrisation
-- so it is asserted only as a same-module cross-check (a ``brentq``
ray-inversion against a direct closed-form evaluation), never as the
accuracy gate.

`RCausticBenchmarkTestCase` times 200 ``r_caustic`` calls against the
retired 720-scan's ``1.85 s`` baseline (arithmetic: ``1.85 / 10 =
0.185 s`` is the >=10x target; the hard gate is the robust >=5x
``0.37 s`` and the measured ~``11x`` ratio is written to a report, never
a bare pass).

`GeometryTestCase.tearDown` fails a test that made zero comparisons, and
`SelfFalsificationTestCase` proves the gates above can actually go red.
"""
from __future__ import annotations

import csv
import itertools
import pathlib
import time
from unittest import TestCase, main

import numpy as np
from scipy.optimize import minimize_scalar

from cogwheel.lensing.chang_refsdal import geometry


try:  # Diagnostics only; never gate a test on plotting being present.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_MPL = False

#: Repo-root-relative fixture of validated quartic solutions.
_CSV_PATH = (pathlib.Path(__file__).resolve().parents[2]
             / '.claude' / 'spec' / 'lensing_paper' / 'data'
             / 'quartic_geometry_validation.csv')

#: Directory for diagnostic figures.
_OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / 'output'

#: Fresh lens-residual gate, recomputed now for every returned image.
#: Measured worst case across all 168 rows is ~1.9e-13.
RESIDUAL_GATE = 1e-12

#: Roundoff headroom for the exact ``y = 0`` closed forms.
ANALYTIC_TOL = 1e-13

#: Shear values used by the analytic, census and astroid sweeps.
ANALYTIC_GAMMAS = (0.05, 0.1, 0.2, 0.3, 0.4)

# -- SHARD D: r_caustic vs the exact critical curve ---------------------

#: Positive-parity shears spanning weak (0.2) to strongly anisotropic
#: (0.9) astroids, per the Architect's SHARD D setup.
R_CAUSTIC_AXIS_GAMMAS = (0.2, 0.3, 0.495, 0.7, 0.9)

#: Professor accuracy bar for the axis closed form and the waist
#: invariant.  Measured worst residual is ~1e-13 (axis) / ~1e-14
#: (waist); the bar is deliberately looser than the root-find floor.
R_CAUSTIC_TOL = 1e-10

#: ``scipy.optimize.minimize_scalar`` angular tolerance for the
#: INDEPENDENT waist search; the quadratic minimum suppresses its
#: propagated error well below :data:`R_CAUSTIC_TOL`.
WAIST_ARGMIN_XATOL = 1e-6

#: Exact transverse-axis caustic radius at gamma=0.9, ``2*0.9/sqrt(0.1)``.
#: The retired scan returned 5.67376 here (a 0.32% error); this literal
#: is the closed form to full float64 precision.
GAMMA_09_PI2_CAUSTIC = 5.692099788303083

#: The value the OLD angular scan produced at gamma=0.9, theta=pi/2 --
#: used only to prove the tight gate rejects it by orders of magnitude.
GAMMA_09_PI2_OLD_SCAN = 5.67376

#: A macro-saddle shear (``|gamma| > 1 - kappa``): the caustic is two
#: disjoint 3-cusp deltoid lobes, so some source rays cross it multiple
#: times and others miss it entirely.
SADDLE_GAMMA = 1.5

# -- SHARD D: r_caustic brentq-vs-scan benchmark ------------------------

#: Benchmark grid: 20 positive-parity shears x 10 interior angles = 200
#: ``r_caustic`` calls, matching the Architect's SHARD D setup.  Every
#: (gamma, theta) is a single-crossing astroid ray, so all 200 succeed.
BENCHMARK_N_GAMMA = 20
BENCHMARK_N_THETA = 10

#: Wall-clock baseline of the RETIRED 720-point angular scan for the same
#: 200 calls, per the Architect's SHARD D setup (seconds).
BENCHMARK_BASELINE_S = 1.85

#: 10x-faster target the brentq inversion is expected to beat
#: (``1.85 / 10``, seconds).  Recorded and reported, not hard-gated: on a
#: shared build tier the measured ~0.17s sits too close to 0.185s to
#: assert reliably.
BENCHMARK_TARGET_S = BENCHMARK_BASELINE_S / 10.0  # 0.185 s

#: Relaxed HARD gate (>=5x faster than the scan, ``1.85 / 5``), used for
#: the pass/fail assertion so a noisy shared tier cannot flake the suite
#: red while the >=10x direction is still measured and reported.
BENCHMARK_HARD_GATE_S = BENCHMARK_BASELINE_S / 5.0  # 0.37 s


def _load_rows() -> list[dict]:
    """Return the CSV fixture as a list of typed row dicts."""
    with open(_CSV_PATH, newline='') as handle:
        rows = []
        for raw in csv.DictReader(handle):
            rows.append({
                'index': int(raw['index']),
                'kind': raw['kind'],
                'gamma': float(raw['gamma']),
                'beta': float(raw['beta']),
                'y1': float(raw['y1']),
                'y2': float(raw['y2']),
                'n_quartic': int(raw['n_quartic']),
                'n_multistart': int(raw['n_multistart'])})
        return rows


def _savefig(fig, name: str) -> None:
    """Save a diagnostic figure, swallowing any backend error."""
    if not _HAVE_MPL:
        return
    try:
        _OUTPUT_DIR.mkdir(exist_ok=True)
        fig.savefig(_OUTPUT_DIR / name, dpi=80, bbox_inches='tight')
    except Exception:  # pragma: no cover - environment dependent
        pass
    finally:
        plt.close(fig)


def _astroid(gamma: float, n_points: int = 2000) -> np.ndarray:
    """
    Analytic Chang--Refsdal caustic (astroid) for ``kappa = 0``,
    ``beta = 0``, written locally so it is INDEPENDENT of
    ``geometry.critical_point``.

    From ``det H = 0`` one gets ``u = 1/|x|**2 =
    gamma*cos(2t) + sqrt(1 - gamma**2 sin(2t)**2)`` on the critical
    curve; mapping ``x = r(cos t, sin t)`` with ``r = 1/sqrt(u)`` through
    the lens map ``y = A x - x/|x|**2`` gives the caustic

        y1 = r cos t (1 - gamma - u),
        y2 = r sin t (1 + gamma - u).

    Returns
    -------
    np.ndarray
        Shape ``(n_points, 2)`` closed polygon tracing the caustic.
    """
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    u = gamma * np.cos(2.0 * t) + np.sqrt(
        1.0 - gamma**2 * np.sin(2.0 * t)**2)
    r = 1.0 / np.sqrt(u)
    y1 = r * np.cos(t) * (1.0 - gamma - u)
    y2 = r * np.sin(t) * (1.0 + gamma - u)
    return np.column_stack([y1, y2])


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Even-odd ray-casting test, valid for the concave astroid.

    Independent of the module: it only consumes the analytic polygon.
    """
    x, y = float(point[0]), float(point[1])
    inside = False
    n = polygon.shape[0]
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _distance_to_polygon(point: np.ndarray,
                         polygon: np.ndarray) -> float:
    """Approximate distance to the caustic via its dense vertices."""
    return float(np.min(np.linalg.norm(polygon - point, axis=1)))


class GeometryTestCase(TestCase):
    """
    Base class carrying the anti-vacuity comparison tally.

    `tearDown` fails a test that asserted nothing, so a sweep whose
    every configuration was skipped cannot read as green.
    """

    def setUp(self) -> None:
        self.n_checks = 0

    def tearDown(self) -> None:
        if self.n_checks == 0 and getattr(self, '_expect_checks', True):
            self.fail('vacuous test: no comparison ran, so nothing was '
                      'asserted')


class QuarticCsvRegressionTestCase(GeometryTestCase):
    """Regression of the exact quartic solver against the frozen CSV."""

    def test_image_count_and_residual_match_the_frozen_oracle(self) \
            -> None:
        """
        For all 168 rows: the returned image count equals the frozen
        ``n_multistart`` oracle, the fixture's own ``n_quartic`` agrees
        with it (a corruption guard on the file, not the code), and a
        freshly recomputed ``max|lens_residual|`` clears ``1e-12``.
        """
        rows = _load_rows()
        self.assertEqual(len(rows), 168,
                         'fixture must contain exactly 168 rows')

        residuals: list[float] = []
        kinds: list[str] = []
        mismatched: list[tuple[int, int, int]] = []
        for row in rows:
            matrix = geometry.macro_matrix(row['gamma'], row['beta'],
                                           kappa=0.0)
            source = np.array([row['y1'], row['y2']])
            images = geometry.find_images_quartic(source, matrix)

            # Fixture-integrity check, asserted separately so a corrupt
            # file is not misread as a code fault.
            self.assertEqual(
                row['n_quartic'], row['n_multistart'],
                f'FIXTURE row {row["index"]} is internally '
                f'inconsistent: n_quartic {row["n_quartic"]} != '
                f'n_multistart {row["n_multistart"]}')

            if len(images) != row['n_multistart']:
                mismatched.append(
                    (row['index'], len(images), row['n_multistart']))

            worst = max(
                (float(np.linalg.norm(
                    geometry.lens_residual(image, source, matrix)))
                 for image in images), default=0.0)
            residuals.append(worst)
            kinds.append(row['kind'])
            self.assertLessEqual(
                worst, RESIDUAL_GATE,
                f'row {row["index"]} ({row["kind"]}) has '
                f'max|lens_residual| {worst:.3e} > {RESIDUAL_GATE}')
            self.n_checks += 1

        self.assertEqual(
            mismatched, [],
            'image count disagreed with n_multistart at rows '
            f'(index, got, want): {mismatched[:10]}')
        self._plot_residuals(residuals, kinds)

    def _plot_residuals(self, residuals: list[float],
                        kinds: list[str]) -> None:
        if not _HAVE_MPL:
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = {'general': 'C0', 'fold': 'C1', 'cusp': 'C2'}
        residuals_arr = np.asarray(residuals)
        for kind, color in colors.items():
            mask = np.array([k == kind for k in kinds])
            if mask.any():
                ax.scatter(np.flatnonzero(mask),
                           np.clip(residuals_arr[mask], 1e-18, None),
                           s=12, c=color, label=kind)
        ax.axhline(RESIDUAL_GATE, color='k', ls='--',
                   label='1e-12 gate')
        ax.set_yscale('log')
        ax.set_xlabel('row index')
        ax.set_ylabel('max|lens_residual|')
        ax.legend()
        _savefig(fig, 'geometry_quartic_csv_residuals.png')


class AnalyticOriginGeometryTestCase(GeometryTestCase):
    """
    Exact ``y = 0`` geometry: the one fully analytic configuration.

    With ``kappa = beta = 0`` the macro matrix is
    ``diag(1 - gamma, 1 + gamma)`` and the four images sit on the axes,
    so ``|x|**2``, the Fermat delays and their splitting all have closed
    forms.  A convention error in the Fermat potential shows here as an
    O(1) offset; roundoff shows as a ~1e-15 hash.
    """

    def test_positions_delays_and_splitting_match_closed_forms(self) \
            -> None:
        splitting_error = []
        for gamma in ANALYTIC_GAMMAS:
            matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
            source = np.zeros(2)
            images = geometry.find_images_quartic(source, matrix)
            self.assertEqual(len(images), 4,
                             f'expected 4 axial images at gamma={gamma}')

            shear_axis, transverse = [], []
            for image in images:
                if abs(image[1]) <= abs(image[0]):
                    shear_axis.append(image)
                else:
                    transverse.append(image)
            self.assertEqual(len(shear_axis), 2)
            self.assertEqual(len(transverse), 2)

            for image in shear_axis:
                self.assertAlmostEqual(
                    float(image @ image), 1.0 / (1.0 - gamma),
                    delta=ANALYTIC_TOL,
                    msg=f'|x|**2 on shear axis at gamma={gamma}')
                self.assertAlmostEqual(
                    geometry.delay(image, source, matrix),
                    0.5 + 0.5 * np.log(1.0 - gamma),
                    delta=ANALYTIC_TOL,
                    msg=f'tau on shear axis at gamma={gamma}')
                self.n_checks += 1
            for image in transverse:
                self.assertAlmostEqual(
                    float(image @ image), 1.0 / (1.0 + gamma),
                    delta=ANALYTIC_TOL,
                    msg=f'|x|**2 transverse at gamma={gamma}')
                self.assertAlmostEqual(
                    geometry.delay(image, source, matrix),
                    0.5 + 0.5 * np.log(1.0 + gamma),
                    delta=ANALYTIC_TOL,
                    msg=f'tau transverse at gamma={gamma}')
                self.n_checks += 1

            tau_shear = geometry.delay(shear_axis[0], source, matrix)
            tau_transverse = geometry.delay(transverse[0], source,
                                            matrix)
            measured_split = tau_transverse - tau_shear
            self.assertAlmostEqual(
                measured_split, np.arctanh(gamma), delta=ANALYTIC_TOL,
                msg=f'delay splitting at gamma={gamma}')
            splitting_error.append(measured_split - np.arctanh(gamma))
        self._plot_splitting(splitting_error)

    def _plot_splitting(self, error: list[float]) -> None:
        if not _HAVE_MPL:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ANALYTIC_GAMMAS, np.abs(error) + 1e-18, 'o-')
        ax.set_yscale('log')
        ax.set_xlabel('gamma')
        ax.set_ylabel('|measured - artanh(gamma)|')
        _savefig(fig, 'geometry_delay_splitting.png')


class MorseCensusTestCase(GeometryTestCase):
    """
    Measured Morse census across the two- and four-image regimes.

    The Euler invariant cannot discriminate the censuses, so the actual
    ``(n_min, n_saddle, n_max)`` triples are tallied and asserted: a
    four-image point is two minima and two saddles; a two-image point is
    one minimum and one saddle; ``n_max == 0`` everywhere, because a
    point mass has ``-ln|x| -> +inf`` at the origin and so admits no
    local maximum of the Fermat potential.
    """

    def test_measured_census_matches_the_expected_triples(self) -> None:
        census: dict[int, dict[tuple[int, int, int], int]] = {
            2: {}, 4: {}}
        radii = np.linspace(0.05, 1.2, 24)
        angles = np.linspace(0.0, np.pi, 7, endpoint=False)
        map_points = []
        for gamma in (0.1, 0.2, 0.3):
            matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
            for radius, angle in itertools.product(radii, angles):
                source = radius * np.array([np.cos(angle),
                                            np.sin(angle)])
                images = geometry.find_images_quartic(source, matrix)
                count = len(images)
                if count not in (2, 4):
                    continue
                triple = (
                    sum(geometry.morse_index(im, matrix) == 0
                        for im in images),
                    sum(geometry.morse_index(im, matrix) == 1
                        for im in images),
                    sum(geometry.morse_index(im, matrix) == 2
                        for im in images))
                census[count][triple] = \
                    census[count].get(triple, 0) + 1
                map_points.append((gamma, radius, count, triple))
                self.n_checks += 1

        self.assertGreater(sum(census[4].values()), 0,
                           'sweep never entered the four-image regime')
        self.assertGreater(sum(census[2].values()), 0,
                           'sweep never entered the two-image regime')
        self.assertEqual(
            set(census[4]), {(2, 2, 0)},
            f'four-image census is not two-min/two-saddle: {census[4]}')
        self.assertEqual(
            set(census[2]), {(1, 1, 0)},
            f'two-image census is not one-min/one-saddle: {census[2]}')
        self._plot_census(map_points)

    def _plot_census(self, points: list) -> None:
        if not _HAVE_MPL or not points:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {2: 'C0', 4: 'C3'}
        for gamma, radius, count, _ in points:
            ax.scatter(gamma, radius, c=colors[count], s=12)
        ax.set_xlabel('gamma')
        ax.set_ylabel('|y|')
        ax.set_title('blue: 2 images, red: 4 images')
        _savefig(fig, 'geometry_morse_census.png')


class AstroidCausticTestCase(GeometryTestCase):
    """
    Image count against a LOCAL analytic astroid.

    Independence requirement: the caustic comes from `_astroid`, never
    from ``geometry.critical_point``, so this is a genuine external
    cross-check rather than a self-consistency test.  Four images iff
    the source is inside the astroid, two outside; points within a thin
    boundary band are skipped, where the double root makes membership
    genuinely ambiguous.
    """

    def test_four_images_iff_inside_the_analytic_caustic(self) -> None:
        margin = 0.02
        disagreements = []
        scatter = []
        for gamma in (0.15, 0.3):
            polygon = _astroid(gamma)
            matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
            grid = np.linspace(-1.0, 1.0, 21)
            for y1, y2 in itertools.product(grid, grid):
                source = np.array([y1, y2])
                if _distance_to_polygon(source, polygon) < margin:
                    continue
                inside = _point_in_polygon(source, polygon)
                count = len(geometry.find_images_quartic(source, matrix))
                expected = 4 if inside else 2
                scatter.append((y1, y2, count))
                if count != expected:
                    disagreements.append(
                        (gamma, y1, y2, count, expected))
                else:
                    self.n_checks += 1
        self.assertEqual(
            disagreements, [],
            'image count disagreed with astroid membership at '
            f'(gamma, y1, y2, got, want): {disagreements[:10]}')
        self._plot(gamma, _astroid(0.3), scatter)

    def _plot(self, gamma, polygon, scatter) -> None:
        if not _HAVE_MPL or not scatter:
            return
        fig, ax = plt.subplots(figsize=(5, 5))
        closed = np.vstack([polygon, polygon[:1]])
        ax.plot(closed[:, 0], closed[:, 1], 'k-', lw=1)
        for y1, y2, count in scatter:
            ax.scatter(y1, y2, c='C3' if count == 4 else 'C0', s=8)
        ax.set_aspect('equal')
        ax.set_title('astroid membership vs image count')
        _savefig(fig, 'geometry_astroid.png')


class NearCausticBehaviourTestCase(GeometryTestCase):
    """
    Fold and cusp rows: assert delays and residuals, never positions.

    At a fold the quartic has a double root, so the returned positions
    carry only ``sqrt(eps) ~ 1.5e-8``; but delays are quadratically
    insensitive to position error (images are stationary points of the
    Fermat potential) and residuals stay tiny.  Magnifications are
    genuinely ill-conditioned near a critical point, so only their
    SCALING is asserted -- ``|mu|`` grows as the caustic is approached.
    """

    def test_delays_residuals_and_magnification_scaling(self) -> None:
        rows = [row for row in _load_rows()
                if row['kind'] in ('fold', 'cusp')]
        self.assertEqual(len(rows), 48,
                         'expected 24 fold + 24 cusp rows')
        by_kind: dict[str, list[tuple[float, float]]] = {
            'fold': [], 'cusp': []}
        for row in rows:
            matrix = geometry.macro_matrix(row['gamma'], row['beta'],
                                           kappa=0.0)
            source = np.array([row['y1'], row['y2']])
            images = geometry.find_images_quartic(source, matrix)
            self.assertGreater(len(images), 0)
            for image in images:
                self.assertTrue(
                    np.isfinite(geometry.delay(image, source, matrix)),
                    f'non-finite delay at row {row["index"]}')
                residual = float(np.linalg.norm(
                    geometry.lens_residual(image, source, matrix)))
                self.assertLessEqual(
                    residual, RESIDUAL_GATE,
                    f'row {row["index"]} residual {residual:.3e}')
            distance = geometry.nearest_caustic_point(
                row['gamma'], row['beta'], source, kappa=0.0).distance
            max_mu = max(abs(geometry.magnification(im, matrix))
                         for im in images)
            by_kind[row['kind']].append((distance, max_mu))
            self.n_checks += 1

        for kind, data in by_kind.items():
            distances = np.array([d for d, _ in data])
            mus = np.array([m for _, m in data])
            positive = (distances > 0) & (mus > 0)
            slope = np.polyfit(np.log(distances[positive]),
                               np.log(mus[positive]), 1)[0]
            self.assertLess(
                slope, -0.1,
                f'{kind} magnification does not grow toward the '
                f'caustic: fitted log|mu| vs log(distance) slope '
                f'{slope:.3f} is not clearly negative')
            self._plot(kind, distances[positive], mus[positive], slope)

    def _plot(self, kind, distances, mus, slope) -> None:
        if not _HAVE_MPL:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(distances, mus, 'o')
        ax.set_xlabel('distance to caustic')
        ax.set_ylabel('max|mu|')
        ax.set_title(f'{kind}: fitted slope {slope:.2f}')
        _savefig(fig, f'geometry_near_caustic_{kind}.png')


class DomainGuardsTestCase(GeometryTestCase):
    """
    Named-exception guards for the unsupported domain.

    ``macro_matrix`` already documents ``1 - kappa > |gamma|``; these
    pin that committed behaviour, including the strict-vs-non-strict
    boundary equality ``1 - kappa == |gamma|`` -- exactly the slip a
    guard bug would introduce -- and the Einstein-ring degeneracy of
    ``find_images_quartic`` at zero source and zero shear.  The named
    ``LensDomainError`` is asserted by type, not a bare ``ValueError``.
    """

    def test_macro_matrix_rejects_non_positive_parity(self) -> None:
        # Build 6 (negative parity): macro SADDLES (0 < 1-kappa < |gamma|,
        # e.g. the former (0.9, 0.2) case) are now LEGITIMATELY accepted.
        # The named refusals are the det A = 0 parity boundary (float64-
        # exact, F004) and the over-critical lam <= 0 / Type III domain.
        cases = [(1.0, 0.5), (0.5, 0.5), (0.0, 1.0)]
        for kappa, gamma in cases:
            with self.subTest(kappa=kappa, gamma=gamma):
                with self.assertRaises(geometry.LensDomainError) as ctx:
                    geometry.macro_matrix(gamma, 0.0, kappa)
                message = str(ctx.exception)
                self.assertIn(str(gamma), message)
                self.assertIn(str(kappa), message)
                self.n_checks += 1

    def test_boundary_equality_is_rejected(self) -> None:
        """``1 - kappa == |gamma|`` must raise (strict inequality).

        The boundary pair must be float64-EXACT. (0.3, 0.7) does NOT work:
        ``1 - 0.7 == 0.30000000000000004 > 0.3``, so that point is a hair
        inside the domain and correctly does not raise. Use powers-of-two
        endpoints where ``1 - kappa`` equals ``|gamma|`` bit-for-bit.
        """
        with self.assertRaises(geometry.LensDomainError):
            geometry.macro_matrix(0.5, 0.0, 0.5)   # 1 - 0.5 == 0.5, exact
        with self.assertRaises(geometry.LensDomainError):
            geometry.macro_matrix(0.25, 0.0, 0.75)  # 1 - 0.75 == 0.25, exact
        self.n_checks += 2

    def test_einstein_ring_is_rejected(self) -> None:
        with self.assertRaises(geometry.LensDomainError):
            geometry.find_images_quartic(np.zeros(2), np.eye(2))
        self.n_checks += 1

    def test_supported_domain_does_not_raise(self) -> None:
        matrix = geometry.macro_matrix(0.49, 0.0, 0.5)
        self.assertEqual(matrix.shape, (2, 2))
        self.n_checks += 1


class FoldDegenerateKernelRefusalTestCase(GeometryTestCase):
    """The saddle-metric fold guard refuses by name, never crashes.

    Surfaced in production (2026-07-19): a nautilus proposal placed an
    image on a fold, ``_saddle_metric``'s raw ``np.linalg.inv`` raised
    a bare ``numpy.linalg.LinAlgError`` that escaped the posterior's
    named-refusal net and killed the sampling run — a certify-or-refuse
    violation (crash instead of a named refusal).
    """

    def test_critical_curve_image_refused_by_name(self):
        """A point-lens image on the critical curve (``|x| = 1`` at
        ``gamma = kappa = 0``) has an exactly singular projected Fermat
        Hessian: the guard must raise `LensDomainError` naming the fold
        degeneracy — not a raw ``LinAlgError``, and never a divergent
        finite kernel."""
        matrix = geometry.macro_matrix(0.0, 0.0, 0.0)
        image = np.array([1.0, 0.0])
        self.n_checks += 1
        with self.assertRaises(geometry.LensDomainError) as ctx:
            geometry.saddle_coefficients(image, matrix)
        self.n_checks += 1
        self.assertIn('Fold-degenerate', str(ctx.exception))

    def test_regular_images_keep_finite_coefficients(self):
        """Healthy off-critical images still yield finite ``C1, C2``
        (the guard does not fire on regular geometry)."""
        matrix = geometry.macro_matrix(0.2, 0.0, 0.0)
        images = geometry.find_images(np.array([0.3, 0.11]), matrix)
        self.assertTrue(images)
        for image in images:
            c1_coefficient, c2_coefficient = geometry.saddle_coefficients(
                image, matrix)
            self.n_checks += 1
            self.assertTrue(np.isfinite(c1_coefficient)
                            and np.isfinite(c2_coefficient))


class SelfFalsificationTestCase(GeometryTestCase):
    """
    Prove the geometry gates can actually fail.

    A green suite is worth only as much as its ability to go red, so
    each gate is shown catching a deliberately corrupted input.
    """

    _expect_checks = False

    def test_residual_gate_rejects_a_perturbed_image(self) -> None:
        matrix = geometry.macro_matrix(0.3, 0.0, 0.0)
        source = np.array([0.4, 0.2])
        image = geometry.find_images_quartic(source, matrix)[0]
        perturbed = image + 1e-3
        residual = float(np.linalg.norm(
            geometry.lens_residual(perturbed, source, matrix)))
        self.assertGreater(
            residual, RESIDUAL_GATE,
            'a 1e-3 position perturbation must break the residual gate; '
            'if it does not, the gate asserts nothing')

    def test_analytic_gate_rejects_a_swapped_convention(self) -> None:
        """The wrong shear-axis closed form must miss by O(gamma)."""
        gamma = 0.3
        matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
        source = np.zeros(2)
        shear_image = next(
            im for im in geometry.find_images_quartic(source, matrix)
            if abs(im[1]) <= abs(im[0]))
        wrong = 1.0 / (1.0 + gamma)  # transverse form, deliberately
        self.assertGreater(
            abs(float(shear_image @ shear_image) - wrong), 1e-2,
            'the swapped convention agrees with the shear-axis value; '
            'the analytic gate would not discriminate a sign flip')


class RCausticAxisClosedFormTestCase(GeometryTestCase):
    """
    SHARD D: ``r_caustic`` on the principal axes vs a pencil-and-paper
    closed form.

    On the shear axis (``theta = 0``) and the transverse axis
    (``theta = pi/2``) the astroid cusp radius is
    ``2*gamma / sqrt(1 +- gamma)``, obtained by hand from the lens map
    ``y = A x - x/|x|**2`` with ``A = diag(1 - gamma, 1 + gamma)``.  This
    oracle is fully INDEPENDENT of the module (it never calls
    ``critical_point`` or ``r_caustic``), so it is the accuracy gate.
    """

    @staticmethod
    def _axis_closed_form(gamma: float, theta: float) -> float:
        """Exact cusp radius on a principal axis; theta in {0, pi/2}."""
        if theta == 0.0:
            return 2.0 * gamma / np.sqrt(1.0 + gamma)     # shear axis
        return 2.0 * gamma / np.sqrt(1.0 - gamma)         # transverse

    def test_axis_radius_matches_independent_closed_form(self) -> None:
        """Relative agreement <= 1e-10 at every gamma on both axes."""
        rel_errors: list[tuple[float, float, float]] = []
        for gamma in R_CAUSTIC_AXIS_GAMMAS:
            for theta in (0.0, np.pi / 2.0):
                radius = geometry.r_caustic(gamma, theta)
                oracle = self._axis_closed_form(gamma, theta)
                rel = abs(radius - oracle) / abs(oracle)
                self.assertLessEqual(
                    rel, R_CAUSTIC_TOL,
                    f'r_caustic({gamma}, {theta:.4f}) = {radius:.15f} '
                    f'disagrees with closed form {oracle:.15f}; '
                    f'relative error {rel:.3e} > {R_CAUSTIC_TOL}')
                rel_errors.append((gamma, theta, rel))
                self.n_checks += 1
        self._plot(rel_errors)

    def test_gamma_090_transverse_is_5p6921_not_the_old_scan(self) \
            -> None:
        """The historically wrong corner: gamma=0.9, theta=pi/2 now
        equals ``5.692099788303083`` (relative error <= 1e-10), NOT the
        retired scan's ``5.67376`` (a 0.32% error the tight gate would
        have caught by four orders of magnitude)."""
        radius = geometry.r_caustic(0.9, np.pi / 2.0)
        rel = abs(radius - GAMMA_09_PI2_CAUSTIC) / GAMMA_09_PI2_CAUSTIC
        self.assertLessEqual(
            rel, R_CAUSTIC_TOL,
            f'r_caustic(0.9, pi/2) = {radius:.15f} != exact '
            f'{GAMMA_09_PI2_CAUSTIC:.15f} (relative error {rel:.3e})')
        # And it is nowhere near the retired scan value.
        self.assertGreater(
            abs(radius - GAMMA_09_PI2_OLD_SCAN) / GAMMA_09_PI2_CAUSTIC,
            1e-3,
            'r_caustic reproduced the retired scan value 5.67376; the '
            'brentq inversion did not fix the 0.32% cusp error')
        self.n_checks += 1

    def _plot(self, rel_errors: list[tuple[float, float, float]]) -> None:
        if not _HAVE_MPL or not rel_errors:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        for theta, marker, label in ((0.0, 'o', 'theta=0'),
                                     (np.pi / 2.0, 's', 'theta=pi/2')):
            xs = [g for g, t, _ in rel_errors if t == theta]
            ys = [max(r, 1e-18) for g, t, r in rel_errors if t == theta]
            ax.semilogy(xs, ys, marker + '-', label=label)
        ax.axhline(R_CAUSTIC_TOL, color='k', ls='--', label='1e-10 bar')
        ax.set_xlabel('gamma')
        ax.set_ylabel('relative error vs closed form')
        ax.legend()
        ax.set_title('r_caustic axis accuracy')
        _savefig(fig, 'geometry_r_caustic_axis_accuracy.png')


class RCausticCriticalPointConsistencyTestCase(GeometryTestCase):
    """
    SHARD D (same-module cross-check, NOT an independent oracle).

    ``r_caustic`` finds the caustic radius by ``brentq`` inversion of the
    critical-curve ray crossing; ``critical_point(gamma, theta).source``
    evaluates the SAME parametrisation directly at a known lens-plane
    axis.  Because both descend from one caustic formula this is asserted
    only as an algorithmic consistency check (root-find vs closed-form
    evaluation), never as the accuracy gate -- that is the closed-form
    class above.
    """

    def test_axis_radius_matches_critical_point_source(self) -> None:
        for gamma in R_CAUSTIC_AXIS_GAMMAS:
            for theta in (0.0, np.pi / 2.0):
                radius = geometry.r_caustic(gamma, theta)
                source_radius = float(np.linalg.norm(
                    geometry.critical_point(gamma, theta).source))
                rel = abs(radius - source_radius) / abs(source_radius)
                self.assertLessEqual(
                    rel, R_CAUSTIC_TOL,
                    f'r_caustic and |critical_point.source| disagree at '
                    f'gamma={gamma}, theta={theta:.4f}: {radius:.15f} vs '
                    f'{source_radius:.15f} (relative {rel:.3e})')
                self.n_checks += 1


class RCausticWaistInvariantTestCase(GeometryTestCase):
    """
    SHARD D: the angular minimum of the caustic radius equals the shear.

    ``min_theta r_caustic(gamma, theta) == gamma`` is an independent
    analytic property of the Chang--Refsdal astroid.  The argmin is
    located by an INDEPENDENT ``scipy.optimize.minimize_scalar`` so the
    module never supplies its own extremum.
    """

    def test_waist_radius_equals_shear(self) -> None:
        eps = 1e-6
        sweep: list[tuple[float, float, np.ndarray, np.ndarray]] = []
        for gamma in R_CAUSTIC_AXIS_GAMMAS:
            result = minimize_scalar(
                lambda theta: geometry.r_caustic(gamma, theta),
                bounds=(eps, np.pi / 2.0 - eps), method='bounded',
                options={'xatol': WAIST_ARGMIN_XATOL})
            theta_waist = float(result.x)
            radius_waist = geometry.r_caustic(gamma, theta_waist)
            self.assertLessEqual(
                abs(radius_waist - gamma), R_CAUSTIC_TOL,
                f'waist radius at gamma={gamma} is {radius_waist:.15f}, '
                f'not gamma; |r - gamma| = '
                f'{abs(radius_waist - gamma):.3e} > {R_CAUSTIC_TOL}')
            # Sanity: the waist is interior, not on an axis.
            self.assertGreater(theta_waist, 0.1)
            self.assertLess(theta_waist, np.pi / 2.0 - 0.1)
            self.n_checks += 1
            if gamma == 0.7:
                thetas = np.linspace(eps, np.pi / 2.0 - eps, 121)
                radii = np.array([geometry.r_caustic(gamma, t)
                                  for t in thetas])
                sweep.append((gamma, theta_waist, thetas, radii))
        self._plot(sweep)

    def _plot(self, sweep) -> None:
        if not _HAVE_MPL or not sweep:
            return
        gamma, theta_waist, thetas, radii = sweep[0]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(thetas, radii, '-')
        ax.axhline(gamma, color='C1', ls='--', label='gamma')
        ax.axvline(theta_waist, color='C2', ls=':', label='waist')
        ax.set_xlabel('source-plane theta')
        ax.set_ylabel('r_caustic')
        ax.set_title(f'waist invariant, gamma={gamma}')
        ax.legend()
        _savefig(fig, 'geometry_r_caustic_waist.png')


class RCausticOutermostAndRefusalsTestCase(GeometryTestCase):
    """
    SHARD D: outermost-crossing selection and the named refusals.

    (a) On a macro-saddle ray that pierces a deltoid lobe twice,
    ``r_caustic`` must return the OUTERMOST forward crossing, strictly
    beyond the inner fold radius.  (b) The parity boundary
    ``|gamma| == 1 - kappa`` and (over-critical ``1 - kappa <= 0``)
    refuse by name.  (c) A saddle ray missing both lobes refuses -- it
    never invents a spurious inner radius.  (d) The isotropic ``gamma =
    0`` limit (a degenerate point caustic) is a clean ``LensDomainError``
    from ``r_caustic`` itself, NOT the ``ZeroDivisionError`` that the
    downstream ``ppgo_map.caustic_rho`` produces at ``gamma = 0``.
    """

    def test_returns_outermost_forward_intersection(self) -> None:
        for theta in (0.0, 0.15, -0.15):
            intersections = geometry._caustic_ray_intersections(
                SADDLE_GAMMA, theta, 0.0,
                geometry._R_CAUSTIC_BRACKETS_SADDLE)
            radii = sorted(float(np.linalg.norm(s))
                           for s in intersections)
            distinct = sorted({round(r, 6) for r in radii})
            self.assertGreaterEqual(
                len(distinct), 2,
                f'saddle ray theta={theta} did not cross the caustic '
                f'more than once (radii {radii}); cannot test '
                f'"outermost"')
            radius = geometry.r_caustic(SADDLE_GAMMA, theta)
            self.assertAlmostEqual(
                radius, max(radii), places=9,
                msg=f'r_caustic returned {radius:.12f}, not the '
                    f'outermost crossing {max(radii):.12f}')
            self.assertGreater(
                max(radii) - min(radii), 0.05,
                'inner and outer fold radii coincide; the outermost '
                'selection is not exercised on this ray')
            self.n_checks += 1

    def test_parity_boundary_refuses_by_name(self) -> None:
        # (gamma, kappa) with |gamma| == 1 - kappa bit-for-bit.
        for gamma, kappa in ((1.0, 0.0), (0.5, 0.5), (0.25, 0.75)):
            with self.subTest(gamma=gamma, kappa=kappa):
                with self.assertRaises(geometry.LensDomainError) as ctx:
                    geometry.r_caustic(gamma, 0.3, kappa=kappa)
                self.assertIn('parity boundary', str(ctx.exception))
                self.n_checks += 1

    def test_over_critical_refuses_by_name(self) -> None:
        with self.assertRaises(geometry.LensDomainError) as ctx:
            geometry.r_caustic(0.5, 0.3, kappa=1.2)
        self.assertIn('over-critical', str(ctx.exception))
        self.n_checks += 1

    def test_saddle_ray_missing_both_lobes_refuses(self) -> None:
        for theta in (np.pi / 2.0, np.pi / 4.0, 1.0):
            with self.subTest(theta=theta):
                # Independently confirm the ray truly misses the lobes.
                intersections = geometry._caustic_ray_intersections(
                    SADDLE_GAMMA, theta, 0.0,
                    geometry._R_CAUSTIC_BRACKETS_SADDLE)
                self.assertEqual(
                    intersections, [],
                    f'setup error: theta={theta} does hit a lobe')
                with self.assertRaises(geometry.LensDomainError):
                    geometry.r_caustic(SADDLE_GAMMA, theta)
                self.n_checks += 1

    def test_isotropic_gamma_zero_refuses_cleanly(self) -> None:
        for theta in (0.0, 0.3, 1.0):
            with self.subTest(theta=theta):
                # Must be the named LensDomainError, NOT a raw
                # ZeroDivisionError leaking from the degenerate caustic.
                with self.assertRaises(geometry.LensDomainError):
                    geometry.r_caustic(0.0, theta)
                self.n_checks += 1


class RCausticSelfFalsificationTestCase(GeometryTestCase):
    """
    Prove the SHARD D ``r_caustic`` gates can actually go red.

    Each check feeds the gate a deliberately wrong value or reference and
    shows it is rejected by a margin that dwarfs ``R_CAUSTIC_TOL``.
    """

    _expect_checks = False

    def test_swapped_axis_closed_form_is_rejected(self) -> None:
        """Using the shear-axis form on the transverse axis (a sign flip
        in ``1 +- gamma``) misses by O(gamma) at gamma=0.9, so the axis
        gate discriminates the convention."""
        radius = geometry.r_caustic(0.9, np.pi / 2.0)
        swapped = 2.0 * 0.9 / np.sqrt(1.0 + 0.9)   # wrong: 1 + gamma
        self.assertGreater(
            abs(radius - swapped) / radius, 1e-2,
            'the swapped 1 +- gamma closed form agrees with the '
            'transverse radius; the axis gate would not catch a flip')

    def test_retired_scan_value_would_fail_the_gate(self) -> None:
        """The 0.32% old-scan error at gamma=0.9,pi/2 is >> the 1e-10
        bar, so the tight gate genuinely rejects the retired result."""
        rel = (abs(GAMMA_09_PI2_CAUSTIC - GAMMA_09_PI2_OLD_SCAN)
               / GAMMA_09_PI2_CAUSTIC)
        self.assertGreater(
            rel, R_CAUSTIC_TOL,
            'the retired scan value clears the 1e-10 gate; the gate '
            'could not have caught the historical cusp error')

    def test_off_waist_evaluation_breaks_the_waist_invariant(self) \
            -> None:
        """Evaluating away from the waist (on the shear axis) yields a
        radius far from gamma, so the waist invariant is not vacuously
        true for any theta."""
        off_waist = geometry.r_caustic(0.9, 0.0)
        self.assertGreater(
            abs(off_waist - 0.9), 1e-2,
            'r_caustic at theta=0 already equals gamma; the waist '
            'invariant would then assert nothing')

    def test_inner_and_outer_crossings_are_distinct(self) -> None:
        """The saddle ray used for the outermost test has genuinely
        distinct inner/outer radii, so "return the outermost" is a real
        choice rather than a tautology."""
        intersections = geometry._caustic_ray_intersections(
            SADDLE_GAMMA, 0.15, 0.0,
            geometry._R_CAUSTIC_BRACKETS_SADDLE)
        radii = [float(np.linalg.norm(s)) for s in intersections]
        self.assertGreater(
            max(radii) - min(radii), 0.05,
            'inner and outer crossings coincide; outermost selection '
            'is untestable on this ray')


class RCausticBenchmarkTestCase(GeometryTestCase):
    """
    SHARD D: ``r_caustic`` brentq inversion is far faster than the
    retired 720-point angular scan.

    SETUP / ARITHMETIC
    ------------------
    The Architect's setup times 200 ``r_caustic`` calls
    (``20`` shears x ``10`` interior angles) and compares against the
    ``1.85 s`` wall-clock baseline of the old 720-point scan for the same
    200 calls.  The expected speed-up is ``>= 10x``, i.e. a target of
    ``1.85 / 10 = 0.185 s`` (~``0.93 ms`` per call).  The measured brentq
    time on this machine is ~``0.17 s`` (~``0.84 ms`` per call), an
    ~``11x`` speed-up -- above the 10x direction.

    GATE CHOICE
    -----------
    ``0.17 s`` sits within noise of the ``0.185 s`` 10x target, so on a
    shared/contended build tier a hard 10x gate would flake red.  Per the
    Architect's relaxation the pass/fail assertion is the ``>= 5x`` gate
    (``1.85 / 5 = 0.37 s``); the measured elapsed and the true ``>= 10x``
    ratio are always computed and written to a diagnostic report, never a
    bare pass.  A short scan self-falsification below proves the gate has
    teeth (the ``1.85 s`` baseline itself would fail it).
    """

    @staticmethod
    def _benchmark_grid() -> tuple[np.ndarray, np.ndarray]:
        """Return the (gammas, thetas) grid whose product is 200 rays.

        Every shear is positive-parity (astroid) and every angle is
        strictly interior, so all 200 calls are single-crossing and none
        refuses.
        """
        gammas = np.linspace(0.1, 0.9, BENCHMARK_N_GAMMA)
        thetas = np.linspace(0.01, np.pi / 2.0 - 0.01, BENCHMARK_N_THETA)
        return gammas, thetas

    def _time_one_pass(self, gammas: np.ndarray,
                       thetas: np.ndarray) -> tuple[float, int]:
        """Time a single sweep of all 200 calls; return (elapsed, count).

        Every returned radius is required to be finite and positive, so a
        silently-refusing or degenerate configuration cannot make the
        benchmark look fast by doing no work.
        """
        start = time.perf_counter()
        count = 0
        for gamma in gammas:
            for theta in thetas:
                radius = geometry.r_caustic(float(gamma), float(theta))
                self.assertTrue(
                    np.isfinite(radius) and radius > 0.0,
                    f'r_caustic({gamma}, {theta}) = {radius} is not a '
                    f'finite positive radius; the benchmark grid must do '
                    f'real work')
                count += 1
        return time.perf_counter() - start, count

    def test_200_calls_beat_the_720_scan_baseline(self) -> None:
        """200 brentq ``r_caustic`` calls finish in <= 0.37 s (>= 5x the
        1.85 s scan baseline) and the measured >= 10x direction is
        recorded to a diagnostic report."""
        gammas, thetas = self._benchmark_grid()
        expected_calls = BENCHMARK_N_GAMMA * BENCHMARK_N_THETA
        self.assertEqual(
            expected_calls, 200,
            'the SHARD D benchmark must time exactly 200 calls')

        # Warm up once so scipy/first-call overhead is not timed, then
        # take the best of three passes to suppress GC/scheduler noise.
        geometry.r_caustic(0.5, 0.3)
        elapsed_passes: list[float] = []
        for _ in range(3):
            elapsed, count = self._time_one_pass(gammas, thetas)
            self.assertEqual(
                count, expected_calls,
                f'timed {count} calls, expected {expected_calls}')
            elapsed_passes.append(elapsed)
            self.n_checks += 1

        best = min(elapsed_passes)
        median = float(np.median(elapsed_passes))
        ratio = BENCHMARK_BASELINE_S / best
        per_call_ms = best / expected_calls * 1e3
        self._report(best, median, ratio, per_call_ms, elapsed_passes)

        # HARD gate: >= 5x faster than the retired scan (robust on a
        # shared tier).  The measured >= 10x direction is reported, not
        # gated, so contention cannot flake the suite red.
        self.assertLessEqual(
            best, BENCHMARK_HARD_GATE_S,
            f'200 r_caustic calls took {best:.4f}s > '
            f'{BENCHMARK_HARD_GATE_S:.4f}s (only {ratio:.1f}x vs the '
            f'{BENCHMARK_BASELINE_S}s 720-scan baseline); the brentq '
            f'inversion is not >= 5x faster than the retired scan')

    def test_hard_gate_has_teeth_against_the_scan_baseline(self) -> None:
        """Self-falsification: the retired 1.85 s scan baseline (and even
        the 10x target) would themselves fail the >= 5x hard gate, so the
        gate is a genuine speed test, not a tautology."""
        self.assertGreater(
            BENCHMARK_BASELINE_S, BENCHMARK_HARD_GATE_S,
            'the 720-scan baseline already clears the hard gate; the '
            'benchmark asserts nothing')
        # The 10x target is stricter than the 5x hard gate, which is
        # stricter than the baseline -- the ordering the report relies on.
        self.assertLess(BENCHMARK_TARGET_S, BENCHMARK_HARD_GATE_S)
        self.assertLess(BENCHMARK_HARD_GATE_S, BENCHMARK_BASELINE_S)
        self.n_checks += 1

    def _report(self, best: float, median: float, ratio: float,
                per_call_ms: float, passes: list[float]) -> None:
        """Write the measured timing + speed-up to a diagnostic file."""
        lines = [
            'SHARD D r_caustic brentq-vs-720-scan benchmark',
            f'calls           : {BENCHMARK_N_GAMMA} gammas x '
            f'{BENCHMARK_N_THETA} thetas = '
            f'{BENCHMARK_N_GAMMA * BENCHMARK_N_THETA}',
            f'scan baseline   : {BENCHMARK_BASELINE_S:.4f} s',
            f'10x target      : {BENCHMARK_TARGET_S:.4f} s (gate: report)',
            f'5x hard gate    : {BENCHMARK_HARD_GATE_S:.4f} s (gate: '
            f'assert)',
            f'passes (s)      : ' + ', '.join(f'{p:.4f}' for p in passes),
            f'best elapsed    : {best:.4f} s',
            f'median elapsed  : {median:.4f} s',
            f'per-call        : {per_call_ms:.4f} ms',
            f'measured ratio  : {ratio:.2f}x vs the 720-scan baseline',
            f'>= 10x met      : {best <= BENCHMARK_TARGET_S}',
        ]
        try:
            _OUTPUT_DIR.mkdir(exist_ok=True)
            (_OUTPUT_DIR / 'geometry_r_caustic_benchmark.txt').write_text(
                '\n'.join(lines) + '\n')
        except Exception:  # pragma: no cover - environment dependent
            pass

class CausticPointMirrorFidelityTestCase(GeometryTestCase):
    """`caustic_point` mirrors the numba `_caustic_source` exactly.

    `geometry.caustic_point` (pure-python, O(1)) is a documented mirror
    of the numba `geometry._caustic_source`: the same closed-form
    critical-curve -> caustic arithmetic, sharing the C math/numpy scalar
    path.  The mirror's docstring mandates exact fidelity, but it is only
    exercised indirectly through `_diffractive.w_low_fit`, so an
    arithmetic drift in the mirror would silently change the fitted
    certificate's caustic feature with no dedicated red.  This class is
    that red: it evaluates both over a sweep of (gamma, theta, beta,
    kappa) at branch ``+1`` and asserts agreement to ~1e-15 relative.
    """

    #: Reduced shears ``gamma' = gamma / (1 - kappa)`` swept at positive
    #: parity, where branch ``+1`` is the only real branch and the
    #: discriminant clamp is inert.
    MIRROR_GAMMA_PRIMES = (0.3, 0.6, 0.8)

    #: Convergences swept alongside the reduced shears.
    MIRROR_KAPPAS = (0.0, 0.2, 0.4)

    #: Shear orientations (radians) exercised in the mirror sweep.
    MIRROR_BETAS = (0.0, 0.7, -1.1)

    #: Polar angles covering the whole critical curve (incl. endpoints).
    MIRROR_THETAS = tuple(np.linspace(0.0, 2.0 * np.pi, 33))

    #: Relative agreement bar: both paths share the C libm scalar
    #: implementations, so the true agreement is exact (measured worst
    #: relative error 0.0 over the positive-parity domain); 1e-15 leaves
    #: headroom for a libm ULP difference while still catching real drift.
    MIRROR_RTOL = 1e-15

    def test_caustic_point_matches_caustic_source(self) -> None:
        """`caustic_point` agrees with `_caustic_source` to ~1e-15 over a
        (gamma, theta, beta, kappa) sweep at branch +1."""
        for gamma_prime in self.MIRROR_GAMMA_PRIMES:
            for kappa in self.MIRROR_KAPPAS:
                gamma = gamma_prime * (1.0 - kappa)
                for beta in self.MIRROR_BETAS:
                    for theta in self.MIRROR_THETAS:
                        produced = np.asarray(geometry.caustic_point(
                            gamma, float(theta), beta=beta, kappa=kappa,
                            branch=1.0))
                        reference = geometry._caustic_source(
                            float(theta), gamma, beta, kappa, 1.0)
                        self.assertTrue(
                            np.allclose(produced, reference,
                                        rtol=self.MIRROR_RTOL, atol=1e-15),
                            f'caustic_point drifted from _caustic_source at '
                            f'gamma_prime={gamma_prime}, kappa={kappa}, '
                            f'beta={beta}, theta={theta:.6f}: produced='
                            f'{produced} reference={reference}')
                        self.n_checks += 1


if __name__ == '__main__':
    main()
