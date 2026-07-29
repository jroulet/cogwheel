"""
Tests for the `lensing.chang_refsdal.channels` topology-stable tracker.

`ChangRefsdalChannels` continues a universal FOUR-label partition

    F(w) = sum_a exp(1j*w*tau_a) * K_a(w)

along a path in lens parameters.  The label count is fixed at four even
where a caustic crossing creates or destroys real images, so the
decomposition -- and the relative-binning summary Build 2 builds on it
-- never jumps when the source crosses the caustic.  This suite pins the
four properties that make that claim true: exact reconstruction of the
operator total, label continuity across fold and cusp crossings, and
path-independence of the (label-invariant) total.

NON-CIRCULAR CROSSING FIXTURES -- WHY THIS SUITE IS NOT VACUOUS
--------------------------------------------------------------
The fold and cusp crossing scenarios are built by test-local helpers
(`_fold_crossing`, `_cusp_crossing`, `_independent_total`) that are
constructed from `geometry`, `operator` and `_gauge` ONLY.  They never
import, call, or derive any value from `channels`.  That independence is
LOAD-BEARING: these fixtures are the ground truth against which the
label-continuity test judges `channels`, so a fixture built by the
tracker it tests COULD NOT FAIL, and every other channels test would be
meaningless.  `NonCircularFixtureGuardTestCase` enforces the
independence by AST inspection of each helper's own source, reusing the
committed `test_lensing_gauge._imported_top_level_modules` import-walking
idiom (extended to catch attribute and name references, since the
channels dependency would enter as `channels.ChangRefsdalChannels`, not
only as an import).  The guard operates at FUNCTION scope, not module
scope, because the suite as a whole legitimately imports `channels` to
test it, and `_imported_top_level_modules` collapses every subpackage
module to the single top-level name `cogwheel`.

BOUNDED ON-CAUSTIC KERNELS -- AND WHY THE BOUND IS SCALE-AWARE ANYWAY
---------------------------------------------------------------------
The residual projection in `_gauge` makes the four channel kernels sum
to the operator total by an ALGEBRAIC identity, so the only
reconstruction error is roundoff, and that roundoff scales with the
largest INTERMEDIATE, sum_a |K_a| -- not with |F| ~ 1.  Which is why
every reconstruction bound here is written as
100*eps*(|F| + sum_a |K_a|) rather than as a flat constant.

Under the SHIPPED gauge that intermediate no longer diverges.  Build 2c
fixed `_channel_switch` (FINDINGS F008) to measure each channel's delay
separation against ALL four cluster labels rather than the real ones
alone.  On the caustic a real image is co-located with the parked
virtual label it is about to spawn, so its separation is ~0, its switch
is exactly 0, and the divergent stationary-phase target is multiplied
away: the channel stays in the bounded artificial gauge and only a
bounded cluster residual survives.  `BoundedKernelTestCase` measures
this directly -- sum_a |K_a| peaks at ~4.3 across the whole config set,
four orders under the 1e3 ceiling it is gated against.

This suite therefore does NOT preserve the pre-2c assertion that
on-caustic kernels diverge past 1e12.  That assertion pinned the BUG:
it passed only because the real-only neighbour set let a still-merged
channel ramp its switch to one and hand itself to a kernel diverging
like sqrt|mu_a|.  Keeping it would have made the fix look like a
regression.  It is replaced by its opposite -- a boundedness ceiling --
plus `test_real_only_neighbours_blow_the_bounded_ceiling`, which
re-injects the buggy neighbour rule and asserts the ceiling IS blown,
so the new gate is falsifiable rather than merely satisfiable.  The
scale-aware FORM of the bound is still the correct one, and that
companion is now the evidence for it: the divergent regime it recreates
is precisely where a flat gate would be unachievable.  See FINDINGS
F008, F003, and `test_lensing_gauge.NearFoldScalingTestCase` for the
`_gauge`-level fold scaling this suite no longer has to reproduce.

Every `<Thing>TestCase` derives from `ChannelsTestCase`, whose
`tearDown` FAILS if a test's sweep ran zero comparisons, and
`SelfFalsificationTestCase` proves the reconstruction and continuity
bounds can actually go red.
"""

from __future__ import annotations

import ast
import inspect
import itertools
import pathlib
import sys
import textwrap
from unittest import TestCase, main

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_MPL = False

from cogwheel.lensing.chang_refsdal import (
    channels, geometry, operator, _gauge)

_OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / 'output'


#: Float64 machine epsilon; the roundoff unit of every bound here.
EPS = np.finfo(np.float64).eps

#: Slack over the roundoff model in the scale-aware reconstruction
#: bound, matching the committed `test_lensing_gauge` gate.  The worst
#: measured margin near a fold is ~0.5*eps*scale, i.e. ~200x below this.
RECONSTRUCTION_SLACK = 100.0

#: The fixed label count.  Declared locally rather than imported from
#: the module under test so the buggy switch variant is a genuinely
#: independent reproduction.  The reproduced-contract test pins it back
#: against ``channels._N_CHANNELS``, so the two cannot silently drift.
N_CHANNELS = 4

#: Ceiling on sum_a |K_a| under the shipped (F008-fixed) gauge, over the
#: whole config set including the on-caustic rows.  The measured worst is
#: ~4.3, so this sits ~2.4 orders above the data and is nowhere near
#: perched on a boundary; `test_the_ceiling_is_not_perched` pins that
#: margin so the ceiling cannot quietly become a rubber stamp.  It is a
#: QUALITATIVE ceiling -- its job is to separate O(1) from the sqrt|mu|
#: divergence (>=1e12) the buggy neighbour rule produces, and any value
#: in the wide gap between those two scales would do.
KERNEL_SUM_CEILING = 1e3

#: Ceiling on the measured worst sum_a |K_a|, asserted so the headroom
#: under `KERNEL_SUM_CEILING` is shown to be real rather than assumed.
#: Measured worst ~4.3; this leaves an order of magnitude for benign
#: drift while still failing long before the ceiling itself would.
KERNEL_SUM_MARGIN_CEILING = 1e2

#: Flat absolute gate on the reconstruction error under the fixed gauge.
#: Legitimate ONLY because the kernels are bounded: with sum_a |K_a| ~ 4
#: the scale-aware bound is ~1e-13, so this flat gate is ~25x TIGHTER and
#: is the stronger claim.  Measured worst ~5e-16, i.e. one order of
#: headroom.  It is asserted ALONGSIDE the scale-aware bound, never
#: instead of it -- see the module docstring.
RECONSTRUCTION_ABS_GATE = 5e-15

# BUGGY_BLOWUP_FLOOR retired with RealOnlyNeighbourFalsificationTestCase
# (Build 3f SACR-C swap, INS-5-002): it floored the pre-2c real-only
# switch's on-caustic sum_a |K_a| blow-up, which no longer has a home now
# that the falsification against the retired 3-arg switch is gone.

#: Multiple of the source-plane path step that bounds each label's
#: per-step Fermat-delay change.  The measured ratio across a fold/cusp
#: crossing is ~2.3; this leaves an order of magnitude of headroom while
#: still excluding any O(1) jump.
CONTINUITY_TAU_SLACK = 20.0

#: Absolute ceiling on any single label's Fermat-delay step across the
#: crossing.  A discontinuous relabelling would move a delay by O(1);
#: the measured step is ~2e-3.  Pins "no O(1) jump" independently of the
#: path step.
TAU_JUMP_CEILING = 0.05

#: Multiple of the path step that bounds the change in the physical
#: total |F| between adjacent points.  Wave optics keeps |F| finite and
#: smooth through the caustic (unlike the geometric-optics kernels), so
#: the total is continuous where the image count is not.  Measured rate
#: ~92; ceiling leaves a factor of a few of headroom.
TOTAL_RATE_SLACK = 400.0

#: Representative dimensionless frequency grid: modest so the suite stays
#: fast, wide enough that the wave branch and the smooth switch are both
#: exercised.
W_GRID = np.linspace(5.0, 20.0, 6)

#: Seed for the random self-falsification perturbations.
SEED = 20260716

#: Names the crossing fixtures are FORBIDDEN to reference: the tracker
#: module and its public surface.  A fixture that touched any of these
#: would no longer be an independent ground truth.
CHANNELS_FORBIDDEN = frozenset(
    {'channels', 'ChangRefsdalChannels', 'ChangRefsdalPartition',
     '_channel_switch'})


def _imported_top_level_modules(module):
    """Return the set of top-level module names a module imports.

    Copied verbatim from `test_lensing_gauge` so the two suites share
    one import-inspection idiom.  It walks `ast.Import`/`ast.ImportFrom`
    and records relative imports as dotted sentinels.
    """
    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split('.')[0]
                         for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                names.add('.'*node.level + (node.module or ''))
            else:
                names.add((node.module or '').split('.')[0])
    return names


def _referenced_names(func):
    """Return every name a function's own source references.

    Extends the `_imported_top_level_modules` idiom (the same
    `ast.Import`/`ast.ImportFrom` walk) with `ast.Name` ids and
    `ast.Attribute` attribute names, because the channels dependency
    would most naturally enter a helper as
    ``channels.ChangRefsdalChannels`` or a bare ``ChangRefsdalChannels``
    name rather than as an import statement inside the function.
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split('.')[0])
                if alias.asname:
                    names.add(alias.asname)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or '').split('.')[0])
            for alias in node.names:
                names.add(alias.name)
                if alias.asname:
                    names.add(alias.asname)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    names.discard('')
    return names


def _min_delay_separation(delays: np.ndarray) -> float:
    """Smallest pairwise separation among a set of Fermat delays.

    Fewer than two delays leaves nothing resolved, so zero is returned
    and the branch gate keeps the wave branch -- the same convention the
    tracker uses.
    """
    delays = np.asarray(delays, dtype=float)
    if delays.size < 2:
        return 0.0
    differences = np.abs(delays[:, None] - delays[None, :])
    return float(np.min(differences[np.triu_indices(delays.size, k=1)]))


def _independent_total(w: np.ndarray, source: np.ndarray, gamma: float,
                       beta: float, kappa: float) -> np.ndarray:
    """Exact amplification total, computed WITHOUT the tracker.

    Reproduces the operator branch logic from `operator` alone -- the
    contour-free wave operator where the branch gate keeps it and the
    stationary-phase asymptote where the gate certifies it -- shifted to
    the tracker's minimum-relative-delay convention.  This is a genuine
    oracle for `channels`' own ``exact_total``: it shares no code with
    the module under test, so agreement between them means the tracker
    did not perturb the physics while relabelling it.
    """
    matrix = geometry.macro_matrix(gamma, beta, kappa)
    images = geometry.find_images(source, matrix)
    absolute = np.array(
        [geometry.delay(image, source, matrix) for image in images],
        dtype=float)
    t_min = float(absolute.min())
    delta_min = _min_delay_separation(absolute)
    total = np.empty(w.shape[0], dtype=complex)
    for index in range(w.shape[0]):
        frequency = float(w[index])
        exponent = operator.cancellation_exponent(
            frequency, source, gamma, kappa)
        if operator.select_branch(
                frequency, delta_min, exponent) == 'geometric':
            value = complex(operator.geometric_amplification(
                frequency, source, gamma, beta=beta, kappa=kappa))
        else:
            value, _ = operator.F_op(
                frequency, source, gamma, beta=beta, kappa=kappa)
        total[index] = value * np.exp(-1j * frequency * t_min)
    return total


def _fold_crossing(gamma: float, theta_c: float, span: float,
                   n_points: int, *, beta: float = 0.0,
                   kappa: float = 0.0) -> dict:
    """Source path crossing a FOLD, from geometry alone.

    Displaces the source along the soft eigenvector of the critical
    point at ``theta_c`` from ``-span`` to ``+span``.  An even
    ``n_points`` skips the exact caustic (a measure-zero triple-image
    point whose geometric-optics kernels are numerically singular), so
    the sampled kernels stay finite while the image count still changes
    from two to four across the path.  Depends on `geometry` only.
    """
    critical = geometry.critical_point(gamma, theta_c, beta, kappa)
    matrix = geometry.macro_matrix(gamma, beta, kappa)
    axis = np.asarray(critical.soft_axis, dtype=float)
    etas = np.linspace(-span, span, n_points)
    sources = [np.asarray(critical.source, dtype=float) + eta*axis
               for eta in etas]
    counts = [len(geometry.find_images(source, matrix))
              for source in sources]
    return {'kind': 'fold', 'gamma': gamma, 'beta': beta,
            'kappa': kappa, 'etas': etas, 'sources': sources,
            'image_counts': counts,
            'caustic_source': np.asarray(critical.source, dtype=float),
            'axis': axis}


def _cusp_crossing(gamma: float, theta_c: float, span: float,
                   n_points: int, *, beta: float = 0.0,
                   kappa: float = 0.0) -> dict:
    """Source path crossing an axis CUSP, from geometry alone.

    Displaces the source along the outward hard eigenvector of the
    critical point at ``theta_c`` from ``-span`` to ``+span``, so the
    image count changes from two to four (the cusp creates two images at
    once).  An even ``n_points`` skips the exact caustic.  Depends on
    `geometry` only.
    """
    critical = geometry.critical_point(gamma, theta_c, beta, kappa)
    matrix = geometry.macro_matrix(gamma, beta, kappa)
    hard = np.asarray(critical.hard_axis, dtype=float)
    if float(np.asarray(critical.image, dtype=float) @ hard) < 0.0:
        hard = -hard
    etas = np.linspace(-span, span, n_points)
    sources = [np.asarray(critical.source, dtype=float) + eta*hard
               for eta in etas]
    counts = [len(geometry.find_images(source, matrix))
              for source in sources]
    return {'kind': 'cusp', 'gamma': gamma, 'beta': beta,
            'kappa': kappa, 'etas': etas, 'sources': sources,
            'image_counts': counts,
            'caustic_source': np.asarray(critical.source, dtype=float),
            'axis': hard}


#: The crossing fixtures whose channels-independence the AST guard pins.
FIXTURE_BUILDERS = (_fold_crossing, _cusp_crossing, _independent_total,
                    _min_delay_separation)


def _path_step(sources) -> float:
    """Largest source-plane displacement between adjacent path points."""
    stacked = np.asarray(sources, dtype=float)
    return float(np.max(np.linalg.norm(np.diff(stacked, axis=0),
                                       axis=1)))


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


# RETIRED (Build 3f SACR-C swap): _real_only_channel_switch
# ----------------------------------------------------------------------
# Reproduced the pre-2c REAL-ONLY neighbour switch (3-arg: w, delays,
# real_mask) so the F008 fix had something to be falsified against.  The
# shipped `channels._channel_switch` now takes a fourth positional
# argument `critical_delay` and keys on the criticality separation
# |tau_a - tau_c| (SACR-C, report Sec. 6.7), so this 3-arg reproduction
# can no longer be monkeypatched in for it.  Removed together with
# `RealOnlyNeighbourFalsificationTestCase` (Inspector finding INS-5-002).


def _on_caustic_config(gamma: float, theta: float, beta: float = 0.0,
                       kappa: float = 0.0) -> dict:
    """An `evaluate` kwargs dict with the source placed ON the caustic.

    Built from `geometry.critical_point` alone: the critical point's
    source IS the caustic point, so this is the configuration where a
    real image and the virtual label parked at that same critical point
    coincide -- the exact coincidence the F008 rule depends on.
    """
    critical = geometry.critical_point(gamma, theta, beta, kappa)
    return dict(gamma=gamma, y=np.asarray(critical.source, dtype=float),
                beta=beta, kappa=kappa)


def _measured_configs() -> list[tuple[str, str, dict]]:
    """The measured config set: ``(kind, label, evaluate_kwargs)``.

    Ten configurations in three kinds:

    * ``generic`` -- two-image, four-image, sheared and convergent
      sources, well away from any caustic, where the switch is on and
      the physical targets are the kernels.
    * ``on-caustic`` -- three sources on a FOLD of the caustic, at
      generic (non-axis) critical angles.  These are the rows the
      pre-2c real-only switch blew up on.
    * ``cusp`` -- three sources on an axis CUSP.  For ``beta = 0`` the
      astroid caustic's cusps lie on the axes, which is why
      `_cusp_crossing` above takes ``theta = pi`` as its cusp; these
      reuse that provenance at ``theta = 0, pi/2, pi``.
    """
    configs: list[tuple[str, str, dict]] = [
        ('generic', 'two-image',
         dict(gamma=0.2, y=np.array([0.12, 0.035]))),
        ('generic', 'four-image',
         dict(gamma=0.2, y=np.array([0.05, 0.02]))),
        ('generic', 'sheared',
         dict(gamma=0.2, y=np.array([0.3, 0.1]), beta=0.4)),
        ('generic', 'convergent',
         dict(gamma=0.2, y=np.array([0.05, 0.02]), beta=0.4,
              kappa=0.1)),
    ]
    for gamma, theta in ((0.2, 4.0), (0.3, 2.5), (0.15, 0.7)):
        configs.append(
            ('on-caustic', f'fold g={gamma} th={theta}',
             _on_caustic_config(gamma, theta)))
    for gamma, theta in ((0.2, np.pi), (0.25, 0.0), (0.15, 0.5*np.pi)):
        configs.append(
            ('cusp', f'cusp g={gamma} th={theta:.3f}',
             _on_caustic_config(gamma, theta)))
    return configs


#: The local reproductions whose independence from the tracker the AST
#: guard pins.  `_on_caustic_config` builds on-caustic `evaluate` kwargs
#: from `geometry`/`operator` only, never from `channels`' own output, so
#: the configs the boundedness gate is measured on cannot be derived from
#: the module under test (FINDINGS F002).  (The pre-2c switch reproduction
#: `_real_only_channel_switch` was retired with the SACR-C swap, INS-5-002.)
SWITCH_REPRODUCTIONS = (_on_caustic_config,)

#: Every helper that must stay independent of the module under test.
INDEPENDENT_HELPERS = FIXTURE_BUILDERS + SWITCH_REPRODUCTIONS


def _kernel_sum(partition) -> np.ndarray:
    """sum_a |K_a| per frequency: the largest intermediate the
    reconstruction passes through, and the quantity the ceiling gates."""
    return np.sum(np.abs(partition.kernels), axis=-1)


class ChannelsTestCase(TestCase):
    """Base class: a scale-aware reconstruction check plus anti-vacuity.

    `assert_reconstructs` applies the same scale-aware roundoff bound as
    the committed `_gauge` suite and counts itself, so `tearDown` can
    prove the test actually compared something rather than skipping its
    whole sweep.
    """

    def setUp(self):
        self.rng = np.random.default_rng(SEED)
        self._comparisons = 0

    def tearDown(self):
        self.assertGreater(
            self._comparisons, 0,
            'Vacuous test: no comparison ran, so this test asserted '
            'nothing. Check the sweep bounds.')

    def assert_reconstructs(self, w, delays, kernels, expected, msg):
        """Assert sum_a exp(i w tau_a) K_a == expected, to roundoff.

        The bound is ``100*eps*(|expected| + sum_a |K_a|)``, elementwise
        in ``w``.  The second term is what survives near a fold: the
        projected kernels carry the divergent residual, so their sum of
        magnitudes bounds the largest intermediate the reconstruction
        passes through.
        """
        kernels = np.asarray(kernels)
        got = _gauge.reconstructed_total(w, delays, kernels)
        error = np.abs(got - expected)
        scale = np.abs(expected) + np.sum(np.abs(kernels), axis=-1)
        bound = RECONSTRUCTION_SLACK * EPS * scale
        self._comparisons += int(np.size(error))
        self.assertTrue(
            np.all(error <= bound),
            msg=f'{msg}: max reconstruction error {np.max(error):.3e} '
                f'exceeds scale-aware bound {np.max(bound):.3e}')


class NonCircularFixtureGuardTestCase(TestCase):
    """The crossing fixtures must stay independent of the tracker.

    The fold/cusp scenario builders are the ground truth the
    label-continuity test judges `channels` against.  If any of them
    reached into `channels`, that test would be judging the tracker
    against itself and could not fail.  These checks parse each helper's
    own source and assert the tracker never appears in it.
    """

    def test_fixtures_never_reference_the_tracker(self):
        """No crossing builder or switch reproduction imports, names, or
        attributes channels."""
        checked = 0
        for builder in INDEPENDENT_HELPERS:
            with self.subTest(builder=builder.__name__):
                referenced = _referenced_names(builder)
                offending = referenced & CHANNELS_FORBIDDEN
                self.assertEqual(
                    offending, frozenset(),
                    f'{builder.__name__} reaches into the tracker via '
                    f'{sorted(offending)}; it must be built from '
                    'geometry/operator/_gauge alone.')
                checked += 1
        self.assertEqual(checked, len(INDEPENDENT_HELPERS))

    def test_fixtures_do_not_name_the_tracker_module(self):
        """The subpackage name a helper may reference is never
        `channels` -- only the physics-layer modules."""
        checked = 0
        for builder in INDEPENDENT_HELPERS:
            with self.subTest(builder=builder.__name__):
                referenced = _referenced_names(builder)
                self.assertNotIn('channels', referenced)
                checked += 1
        self.assertEqual(checked, len(INDEPENDENT_HELPERS))

    def test_reuses_the_committed_import_idiom(self):
        """The suite declares numpy and the cogwheel package as deps.

        Documents that the reused `_imported_top_level_modules` idiom
        cannot itself isolate the fixtures: it collapses every
        chang_refsdal module to the top-level name ``cogwheel``, which is
        exactly why the guard above works at function scope instead.
        """
        imported = _imported_top_level_modules(sys.modules[__name__])
        self.assertIn('numpy', imported)
        self.assertIn('cogwheel', imported)


class ScaleAwareReconstructionTestCase(ChannelsTestCase):
    """The channels reconstruct the operator total, scale-aware.

    Across generic and on-caustic configurations the returned
    ``(tau_a, K_a)`` must sum to the exact total to a bound that tracks
    sum_a |K_a| rather than a flat constant, because the residual
    projection's roundoff scales with that intermediate.

    Under the shipped F008 gauge the intermediate happens to stay O(1)
    even on the caustic, so here the scale-aware bound and a flat one
    are numerically close; `BoundedKernelTestCase` is what measures that
    and gates the tighter flat number.  The scale-aware form is kept
    because it is the correct model of the error, not because these rows
    need the slack -- see the module docstring.
    """

    def _generic_configs(self):
        """The off-caustic rows of the one measured config set."""
        return [(label, config)
                for kind, label, config in _measured_configs()
                if kind == 'generic']

    def _near_fold_configs(self):
        """The on-caustic (fold) rows of the one measured config set.

        Cusp rows are deliberately excluded here: this test additionally
        demands `operator_converged`, which is a stronger precondition
        than boundedness needs, so the cusp rows are gated by
        `BoundedKernelTestCase` instead.
        """
        return [(label, config)
                for kind, label, config in _measured_configs()
                if kind == 'on-caustic']

    def test_reconstruction_is_scale_aware_exact(self):
        """Generic and near-fold decompositions reconstruct their
        total."""
        tracker = channels.ChangRefsdalChannels(W_GRID)
        for label, config in (self._generic_configs()
                              + self._near_fold_configs()):
            with self.subTest(config=label):
                tracker.reset()
                partition = tracker.evaluate(**config)
                self.assertTrue(
                    np.all(partition.operator_converged),
                    f'{label}: the operator did not converge, so this '
                    'is not a clean reconstruction test')
                self.assert_reconstructs(
                    W_GRID, partition.delays, partition.kernels,
                    partition.exact_total, label)

    def test_exact_total_matches_the_independent_oracle(self):
        """The tracker's total is the operator total, not a relabelling
        of a different quantity.

        Compared against `_independent_total`, which shares no code with
        the tracker, so the labelling machinery is shown not to perturb
        the physics.
        """
        tracker = channels.ChangRefsdalChannels(W_GRID)
        for label, config in self._generic_configs():
            with self.subTest(config=label):
                tracker.reset()
                partition = tracker.evaluate(**config)
                oracle = _independent_total(
                    W_GRID, np.asarray(config['y'], dtype=float),
                    config['gamma'], config.get('beta', 0.0),
                    config.get('kappa', 0.0))
                error = np.abs(partition.exact_total - oracle)
                scale = np.abs(oracle) + 1.0
                self._comparisons += int(np.size(error))
                self.assertTrue(
                    np.all(error <= 1e-9 * scale),
                    f'{label}: tracker total departs from the '
                    f'independent operator oracle by {np.max(error):.3e}')


class BoundedKernelTestCase(ChannelsTestCase):
    """The F008 gauge keeps every channel kernel BOUNDED, on-caustic too.

    This class replaces a pre-2c test that asserted the opposite -- that
    on-caustic kernels diverge past 1e12.  That assertion pinned the bug
    rather than the physics: it held only because the real-only
    neighbour rule ramped a still-merged channel's switch to one and
    handed it to a stationary-phase kernel diverging like sqrt|mu_a|.
    With the switch measured against all four cluster labels, a
    near-critical image is co-located with its parked virtual label, its
    separation is ~0, its switch is exactly 0, and the divergent target
    is multiplied away -- leaving a bounded cluster residual BY DESIGN.

    WHAT IS AND IS NOT COVERED
    --------------------------
    Covered: the shipped `ChangRefsdalChannels` kernels are evaluated
    end-to-end on the measured config set, so the ceiling gates the real
    integration of switch, targets and residual projection.

    NOT covered: that this ceiling can actually go RED.  The companion
    falsification (`RealOnlyNeighbourFalsificationTestCase`, which
    injected the pre-2c real-only switch and asserted the on-caustic
    kernels blew past the ceiling) was RETIRED with the Build 3f SACR-C
    swap: `channels._channel_switch` gained a fourth `critical_delay`
    argument and now keys on the criticality separation |tau_a - tau_c|,
    so the old 3-arg reproduction no longer fits its signature
    (INS-5-002).  A replacement falsification against the SACR-C switch
    is OWED (Test Developer); until it lands this class runs but is not
    proven able to fail.
    """

    def _evaluate(self, config: dict):
        """Evaluate one config through the shipped tracker, from reset."""
        tracker = channels.ChangRefsdalChannels(W_GRID)
        tracker.reset()
        return tracker.evaluate(**config)

    def test_kernels_stay_bounded_on_every_config(self):
        """sum_a |K_a| stays under the ceiling everywhere, including on
        the caustic where the pre-2c gauge diverged."""
        for kind, label, config in _measured_configs():
            with self.subTest(kind=kind, config=label):
                partition = self._evaluate(config)
                sums = _kernel_sum(partition)
                self._comparisons += int(np.size(sums))
                self.assertTrue(
                    np.all(np.isfinite(sums)),
                    f'{label}: sum_a |K_a| is not finite, so the gauge '
                    'produced inf/nan rather than a bounded residual')
                self.assertLess(
                    float(np.max(sums)), KERNEL_SUM_CEILING,
                    f'{label}: sum_a |K_a| reached '
                    f'{float(np.max(sums)):.3e}, above the '
                    f'{KERNEL_SUM_CEILING:g} boundedness ceiling; the '
                    'F008 switch is no longer parking merged channels '
                    'in the artificial gauge')

    def test_reconstruction_is_exact_to_a_flat_gate(self):
        """The bounded kernels reconstruct the total to a FLAT 5e-15 --
        a claim only the bounded gauge makes available."""
        worst = 0.0
        for kind, label, config in _measured_configs():
            with self.subTest(kind=kind, config=label):
                partition = self._evaluate(config)
                self.assert_reconstructs(
                    W_GRID, partition.delays, partition.kernels,
                    partition.exact_total, label)
                got = _gauge.reconstructed_total(
                    W_GRID, partition.delays, partition.kernels)
                error = float(np.max(np.abs(got - partition.exact_total)))
                worst = max(worst, error)
                self.assertLessEqual(
                    error, RECONSTRUCTION_ABS_GATE,
                    f'{label}: reconstruction error {error:.3e} exceeds '
                    f'the flat {RECONSTRUCTION_ABS_GATE:g} gate')
        self.assertGreater(
            worst, 0.0,
            'every reconstruction was bit-exact, which would mean the '
            'residual projection is not being exercised at all')

    def test_the_ceiling_is_not_perched(self):
        """The measured worst sum_a |K_a| sits orders under the ceiling.

        Without this, a ceiling could pass while sitting just above the
        data and would degrade into a rubber stamp on the next drift.
        """
        worst = 0.0
        for *_, config in _measured_configs():
            sums = _kernel_sum(self._evaluate(config))
            worst = max(worst, float(np.max(sums)))
            self._comparisons += 1
        self.assertLess(
            worst, KERNEL_SUM_MARGIN_CEILING,
            f'the worst sum_a |K_a| is {worst:.3e}, close enough to the '
            f'{KERNEL_SUM_CEILING:g} ceiling that the ceiling is now '
            'perched on the data rather than separating O(1) from a '
            'sqrt|mu| divergence')
        self.assertGreater(
            worst, 0.0,
            'sum_a |K_a| is identically zero, so the sweep measured '
            'nothing')


# RETIRED (Build 3f SACR-C swap): RealOnlyNeighbourFalsificationTestCase
# ----------------------------------------------------------------------
# This class pinned that `BoundedKernelTestCase` could go RED by
# re-injecting the pre-2c REAL-ONLY neighbour switch (FINDINGS F008) via
# `mock.patch.object(channels, '_channel_switch', _real_only_channel_switch)`
# and asserting on-caustic sum_a |K_a| blew past the ceiling the fixed
# gauge holds.  Build 3f's SACR-C construction SUPERSEDES the F008
# full-cluster switch rule: `channels._channel_switch` now takes a fourth
# positional argument `critical_delay` and keys on the criticality
# separation |tau_a - tau_c| (report Sec. 6.7).  The 3-arg
# `_real_only_channel_switch` no longer matches that signature
# (TypeError: takes 3 positional arguments but 4 were given), and the
# F008 real-only-neighbour rule it falsified is no longer the shipped
# switch, so the whole class is both broken and conceptually obsolete
# (Inspector finding INS-5-002).  It is RETIRED here rather than rewritten.
#
# OWED (Test Developer): a replacement falsification that the SACR-C
# criticality-separation switch (S_a = smootherstep(w*|tau_a - tau_c|,
# 0.5, 4)) is load-bearing for `BoundedKernelTestCase` -- e.g. inject a
# 4-arg variant that keys on the WRONG separation (real-only neighbours,
# or |tau_a - tau_c| replaced by a full-cluster min) and assert the
# on-caustic boundedness ceiling goes red.  Without it, BoundedKernelTestCase
# still runs but is no longer proven able to fail.  The Coder does not
# author this gate (it would certify the same WP1 switch it exercises).


class LabelContinuityTestCase(ChannelsTestCase):
    """Each label varies continuously across a fold or cusp crossing.

    Walking a non-circular crossing path with continuation enabled, no
    label's Fermat delay may jump by O(1) at the caustic where the image
    count changes -- the virtual labels parked at the nearest critical
    point keep the four-label decomposition topology-stable, so a channel
    is always present to receive a newly born image.  The physical total
    ``|F|`` is likewise continuous (wave optics regularizes the caustic
    that the geometric-optics kernels cannot).
    """

    def _scenarios(self):
        return {
            'fold': _fold_crossing(0.2, 4.0, 8e-3, 14),
            'cusp': _cusp_crossing(0.2, np.pi, 8e-3, 14),
        }

    def _partitions(self, scenario):
        tracker = channels.ChangRefsdalChannels(W_GRID)
        path = [dict(gamma=scenario['gamma'], y=source,
                     beta=scenario['beta'], kappa=scenario['kappa'])
                for source in scenario['sources']]
        return tracker.evaluate_path(path)

    def test_channel_count_is_fixed_through_the_crossing(self):
        """Four kernels at every point, though the real-image count
        changes from two to four."""
        for kind, scenario in self._scenarios().items():
            with self.subTest(kind=kind):
                counts = scenario['image_counts']
                self.assertIn(2, counts,
                              f'{kind}: path never sampled the '
                              'two-image side')
                self.assertIn(4, counts,
                              f'{kind}: path never sampled the '
                              'four-image side')
                for partition in self._partitions(scenario):
                    self.assertEqual(
                        partition.kernels.shape, (W_GRID.size, 4),
                        f'{kind}: channel count changed at the crossing')
                    self._comparisons += 1

    def test_labels_do_not_jump_across_the_crossing(self):
        """No label's delay steps by O(1); each step is bounded by a
        multiple of the source-plane path step."""
        for kind, scenario in self._scenarios().items():
            with self.subTest(kind=kind):
                partitions = self._partitions(scenario)
                self.assertTrue(
                    all(np.all(p.operator_converged)
                        for p in partitions),
                    f'{kind}: an operator evaluation did not converge')
                delays = np.array([p.delays for p in partitions])
                steps = np.abs(np.diff(delays, axis=0))
                step = _path_step(scenario['sources'])
                largest = float(np.max(steps))
                self._comparisons += int(np.size(steps))
                self.assertLess(
                    largest, TAU_JUMP_CEILING,
                    f'{kind}: a label delay jumped by {largest:.3e}, an '
                    'O(1) discontinuity at the crossing')
                self.assertLessEqual(
                    largest, CONTINUITY_TAU_SLACK * step,
                    f'{kind}: largest delay step {largest:.3e} exceeds '
                    f'{CONTINUITY_TAU_SLACK} * path step {step:.3e}')

    def test_physical_total_is_continuous_across_the_crossing(self):
        """|F| moves smoothly through the caustic though the image count
        does not, and equals the independent operator oracle."""
        for kind, scenario in self._scenarios().items():
            with self.subTest(kind=kind):
                partitions = self._partitions(scenario)
                totals = np.array([p.exact_total for p in partitions])
                step = _path_step(scenario['sources'])
                jumps = float(np.max(np.abs(np.diff(totals, axis=0))))
                self._comparisons += 1
                self.assertLessEqual(
                    jumps, TOTAL_RATE_SLACK * step,
                    f'{kind}: |F| jumped by {jumps:.3e} across the '
                    'crossing')
                for index in (0, -1):
                    oracle = _independent_total(
                        W_GRID, scenario['sources'][index],
                        scenario['gamma'], scenario['beta'],
                        scenario['kappa'])
                    error = np.abs(totals[index] - oracle)
                    self._comparisons += int(np.size(error))
                    self.assertTrue(
                        np.all(error <= 1e-9 * (np.abs(oracle) + 1.0)),
                        f'{kind}: endpoint total departs from the '
                        f'independent oracle by {np.max(error):.3e}')


class AssignmentResetEquivalenceTestCase(ChannelsTestCase):
    """One target reached three ways gives one total and one label set.

    The total ``sum_a exp(i w tau_a) K_a`` is symmetric in the labels, so
    a reset and two different continuation paths to the same endpoint
    must agree on it to roundoff -- if it depended on evaluation history
    the posterior would too.  The per-label ASSIGNMENT may differ (a
    reset orders labels by polar angle, continuation by proximity), so
    the label SET is compared as a multiset up to permutation, never by
    order.
    """

    def _target_and_paths(self):
        gamma = 0.2
        critical = geometry.critical_point(gamma, np.pi)
        hard = np.asarray(critical.hard_axis, dtype=float)
        if float(np.asarray(critical.image, dtype=float) @ hard) < 0.0:
            hard = -hard
        source = np.asarray(critical.source, dtype=float)
        target = dict(gamma=gamma, y=source + 5e-3*hard)
        path_a = [dict(gamma=gamma, y=source + eta*hard)
                  for eta in np.linspace(-8e-3, 5e-3, 8)]
        path_b = [dict(gamma=gamma, y=source + eta*hard)
                  for eta in (1.5e-2, 1.0e-2, 7e-3, 5e-3)]
        return target, path_a, path_b

    def _evaluate_three_ways(self):
        target, path_a, path_b = self._target_and_paths()
        reset_tracker = channels.ChangRefsdalChannels(W_GRID)
        reset_tracker.reset()
        reset = reset_tracker.evaluate(**target)
        path_a_result = channels.ChangRefsdalChannels(
            W_GRID).evaluate_path(path_a)[-1]
        path_b_result = channels.ChangRefsdalChannels(
            W_GRID).evaluate_path(path_b)[-1]
        return reset, path_a_result, path_b_result

    def test_total_is_path_independent(self):
        """The reconstructed total is identical to roundoff all three
        ways."""
        reset, path_a, path_b = self._evaluate_three_ways()
        reference = _gauge.reconstructed_total(
            W_GRID, reset.delays, reset.kernels)
        for label, partition in (('path A', path_a),
                                 ('path B', path_b)):
            with self.subTest(reached_by=label):
                total = _gauge.reconstructed_total(
                    W_GRID, partition.delays, partition.kernels)
                error = np.abs(total - reference)
                scale = (np.abs(reference)
                         + np.sum(np.abs(reset.kernels), axis=-1)
                         + np.sum(np.abs(partition.kernels), axis=-1))
                bound = RECONSTRUCTION_SLACK * EPS * scale
                self._comparisons += int(np.size(error))
                self.assertTrue(
                    np.all(error <= bound),
                    f'{label}: total differs from the reset total by '
                    f'{np.max(error):.3e}, above the scale-aware floor '
                    f'{np.max(bound):.3e}')

    def test_label_set_matches_up_to_permutation(self):
        """The multiset of (tau_a, K_a) matches, though the order may
        not."""
        reset, path_a, path_b = self._evaluate_three_ways()
        kernel_scale = max(float(np.max(np.abs(reset.kernels))), 1.0)
        kernel_floor = (RECONSTRUCTION_SLACK * EPS * kernel_scale
                        * W_GRID.size)
        for label, partition in (('path A', path_a),
                                 ('path B', path_b)):
            with self.subTest(reached_by=label):
                best_delay = np.inf
                best_kernel = np.inf
                for permutation in itertools.permutations(range(4)):
                    order = list(permutation)
                    delay_gap = float(np.max(np.abs(
                        reset.delays - partition.delays[order])))
                    kernel_gap = float(np.max(np.abs(
                        reset.kernels - partition.kernels[:, order])))
                    if delay_gap + kernel_gap < best_delay + best_kernel:
                        best_delay, best_kernel = delay_gap, kernel_gap
                self._comparisons += 1
                self.assertLessEqual(
                    best_delay, 1e-7,
                    f'{label}: no permutation aligns the label delays '
                    f'(closest gap {best_delay:.3e})')
                self.assertLessEqual(
                    best_kernel, kernel_floor,
                    f'{label}: aligned kernels differ by '
                    f'{best_kernel:.3e}, more than roundoff')

    def test_reset_and_continuation_can_disagree_on_order(self):
        """The assignment is genuinely history-dependent: at least one
        route relabels the target.

        Without this, `test_label_set_matches_up_to_permutation` could
        pass trivially because every route happened to produce the
        identical order, and it would not be testing permutation
        invariance at all.
        """
        reset, path_a, _ = self._evaluate_three_ways()
        identical = np.array_equal(
            np.round(reset.delays, 9), np.round(path_a.delays, 9))
        self._comparisons += 1
        self.assertFalse(
            identical,
            'reset and continuation produced the identical label order, '
            'so the permutation test exercises no real relabelling')


class SelfFalsificationTestCase(ChannelsTestCase):
    """Prove the scale-aware and continuity bounds can go red.

    A green suite is worth only as much as its ability to fail.  These
    tests corrupt a reconstruction and inject a delay discontinuity and
    assert the very bounds the suite relies on catch them.
    """

    def test_corrupted_kernel_breaks_the_reconstruction_bound(self):
        """Perturbing one kernel by O(1) violates the scale-aware bound,
        so `assert_reconstructs` is not vacuous."""
        tracker = channels.ChangRefsdalChannels(W_GRID)
        partition = tracker.evaluate(
            gamma=0.2, y=np.array([0.05, 0.02]))
        corrupted = partition.kernels.copy()
        corrupted[:, 0] += 1.0
        got = _gauge.reconstructed_total(
            W_GRID, partition.delays, corrupted)
        error = np.abs(got - partition.exact_total)
        scale = (np.abs(partition.exact_total)
                 + np.sum(np.abs(corrupted), axis=-1))
        bound = RECONSTRUCTION_SLACK * EPS * scale
        self._comparisons += 1
        self.assertTrue(
            np.any(error > bound),
            'a kernel corrupted by O(1) still satisfied the scale-aware '
            'bound; the reconstruction check cannot fail')

    def test_injected_discontinuity_breaks_the_continuity_bound(self):
        """A delay sequence with an O(1) jump exceeds the continuity
        ceiling, so the continuity check is not vacuous."""
        scenario = _fold_crossing(0.2, 4.0, 8e-3, 14)
        delays = np.zeros(len(scenario['sources']))
        delays[len(delays)//2:] += 1.0  # a deliberate O(1) label jump
        largest = float(np.max(np.abs(np.diff(delays))))
        step = _path_step(scenario['sources'])
        self._comparisons += 1
        self.assertGreater(
            largest, CONTINUITY_TAU_SLACK * step,
            'an injected O(1) delay jump stayed within the continuity '
            'bound; the continuity check cannot fail')
        self.assertGreaterEqual(
            largest, TAU_JUMP_CEILING,
            'an injected O(1) delay jump stayed under the jump ceiling')


if __name__ == '__main__':
    main()
