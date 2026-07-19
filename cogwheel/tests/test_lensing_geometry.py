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

`GeometryTestCase.tearDown` fails a test that made zero comparisons, and
`SelfFalsificationTestCase` proves the gates above can actually go red.
"""
from __future__ import annotations

import csv
import itertools
import pathlib
from unittest import TestCase, main

import numpy as np

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


if __name__ == '__main__':
    main()
