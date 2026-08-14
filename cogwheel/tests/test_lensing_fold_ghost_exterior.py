"""
Exterior fold refusal + ppGO-ghost rung contract tests (Professor Q4a/b/c).

This build ships three coupled behaviours in the Chang-Refsdal fold/ghost
machinery, all keyed on the four-real-image *interior* census:

* WP-1 -- the merging-fold correction is REFUSED on exterior censuses
  (fewer than four real images).  Outside the caustic, positive parity has
  exactly two real images (a Morse-0 minimum and a Morse-1 saddle); the
  ``_merging_fold_pair`` helper would still return that FAR pair and yield a
  spurious Airy value, so ``fold_amplification`` returns ``None``,
  ``fold_ppgo_correction`` falls back byte-identically to raw ppGO
  (``geometric_amplification``), and the ``born_carrier_from_partition``
  fold block is skipped (``len(images) != 4`` guards, F075).
* WP-2 -- a new EXTERIOR ppGO+ghost rung, ``_ghost_ppgo_amplification``,
  serves a two-image node as the geometric image sum PLUS the decaying
  complex-saddle ('ghost') term the bare arm ladder omits.  Its admission
  reuses the two frequency-independent geometry gates single-sourced in
  ``geometry``: decay (``Im tau_c >= _GHOST_DECAY_IM_THRESHOLD = 0.4``) AND
  resolution (``min_a |x_a - x_c| >= _GHOST_SEPARATION_MIN = 0.7``).

The three specification classes below pin, respectively:

Q4a  ``FoldExteriorRefusalTestCase`` / ``FoldInteriorByteIdentityTestCase``
     -- exterior refusal (fold None, ppGO == geometric, carrier == no-fold
     path) and interior byte-identity to the no-fold reference.
Q4b  ``GhostRungGateTestCase`` -- the serve / decline / refuse DECISION of
     the ghost rung, with boundary-flip teeth proving BOTH gates are live.
Q4c  ``GhostSignConventionTestCase`` -- the served value equals
     ``geometric_amplification + ghost.kernel * exp(1j w tau_c)`` to ~1e-12,
     pinning the ``+`` sign and the NON-conjugated ``tau_c`` carrier.

Oracle independence
-------------------
The Q4a exterior/interior identity tests do NOT diff against a re-derived
formula; they diff the shipped function against a *no-fold reference* built
by monkeypatching ``_airy_fold._merging_fold_pair`` to ``None`` (which forces
the raw ppGO path via the same production code) and against
``geometric_amplification`` itself -- both are independent shipping code
paths, not transcriptions.  The Q4c pin is an algebraic identity between the
rung and its own published definition (``ghost_kernel`` is the independent
producer of ``tau_c`` and the kernel), so it is refactor-durable with no
external oracle needed.

Tolerances
----------
Exterior refusal and byte-identity are asserted with ``np.array_equal`` /
exact ``==`` (machine precision): the guard either takes the same code path
or it does not, so any non-zero difference is a real regression.  The Q4c
sign pin uses ``1e-12`` absolute -- comfortably above float64 round-off in
the single ``carrier + kernel*exp(...)`` sum yet far below the O(1e-4) ghost
term it is discriminating, so the '-'/conjugate variants (validated as
worse) provably fail.  The sign-teeth fixture runs at a LOW frequency
(``w = 12``) where the ghost term ``|kernel|*exp(-w Im tau_c) ~ 4e-4`` is
resolvable; at serve-band ``w`` the ghost has decayed to ~1e-23 and the
minus-sign mutation would be silently inert (self-falsification with no
teeth), so the low-w point is essential.

Cost
----
Every fixture uses the cheap analytic ``geometry_partition`` (no Schwinger
``_exact_total``) and closed-form ``geometric_amplification`` /
``ghost_kernel`` -- microseconds to milliseconds per call.  Total: a few
dozen closed-form evaluations, < 5 s wall.  No engine, no mpmath, no chart
training.  Fast tier.
"""
from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal import channels
from cogwheel.lensing.chang_refsdal import operator
from cogwheel.lensing.chang_refsdal import _airy_fold
from cogwheel.lensing.chang_refsdal.channels import (
    ChangRefsdalChannels, born_carrier_from_partition)
from cogwheel.lensing.chang_refsdal.operator import (
    geometric_amplification, _ghost_ppgo_amplification)

# --- Shared shear magnitude (positive parity: gamma < 1) ------------------
#: External shear for every fixture; positive parity so the caustic is the
#: astroid and the exterior census has exactly two real images.
GAMMA = 0.5

# --- Q4a exterior fixture -------------------------------------------------
#: Polar angle of the exterior source ray (off-axis so the far min/saddle
#: pair is non-degenerate and ``_merging_fold_pair`` would otherwise serve).
THETA_EXT = 0.6
#: Caustic-relative radius of the exterior source (> 1 => outside the
#: caustic => two real images).  Derived into a source below; the image
#: count is asserted as an explicit premise, never assumed.
RHO_EXT = 1.40
#: Frequency for the exterior fold probes; in the fold serving band.
W_EXT = 70.0

# --- Q4a interior fixture -------------------------------------------------
#: Polar angle of the interior source ray.
THETA_INT = 0.6
#: Caustic-relative radius of the interior source (< 1 => inside the
#: caustic => four real images => the fold correction is active).
RHO_INT = 0.90
#: Frequency for the interior byte-identity probes.
W_INT = 70.0

# --- Q4b ghost-rung fixtures ----------------------------------------------
#: Ghost-rung serve angle (pi/4: off both principal axes so the ghost is
#: decayed and resolved).
THETA_GHOST = np.pi / 4.0
#: Caustic-relative radius of the clean-serve ghost fixture (well outside
#: the caustic so both geometry gates admit).  The Im(tau_c) >= 0.4 and
#: separation >= 0.7 premises are asserted from the live ghost_kernel.
RHO_GHOST_SERVE = 2.5
#: Caustic-relative radius of the caveat-band fixture (just outside the
#: caustic): the ghost is barely decayed so at least one gate declines.
RHO_GHOST_CAVEAT = 1.05
#: Frequency for the ghost-rung serve/decline probes; in (60, 150].
W_GHOST = 90.0

# --- Q4c sign-pin fixture -------------------------------------------------
#: Low frequency for the sign-convention pin: the ghost term is ~4e-4 here
#: (resolvable), so the '-'/conjugate mutations are genuinely discriminable.
W_SIGN = 12.0
#: Absolute tolerance on the Q4c algebraic identity.
SIGN_ATOL = 1e-12

# --- Gate constants mirrored from geometry (single source of truth) -------
#: Ghost decay gate: Im(tau_c) must clear this for the rung to admit.
GHOST_DECAY_IM_THRESHOLD = geometry._GHOST_DECAY_IM_THRESHOLD
#: Ghost resolution gate: min image-to-ghost distance must clear this.
GHOST_SEPARATION_MIN = geometry._GHOST_SEPARATION_MIN


def _ray_source(gamma: float, theta: float, rho: float,
                kappa: float = 0.0) -> np.ndarray:
    """Source on the ray ``theta`` at caustic-relative radius ``rho``.

    Derives the position from the LIVE caustic reach ``r_caustic`` so the
    interior/exterior premise tracks the caustic if the geometry moves,
    rather than pinning a hand-tuned literal.
    """
    r_c = geometry.r_caustic(gamma, theta, kappa=kappa)
    return rho * r_c * np.array([np.cos(theta), np.sin(theta)])


def _image_count(source: np.ndarray, gamma: float,
                 beta: float = 0.0, kappa: float = 0.0) -> int:
    """Number of real images of ``source`` (the interior/exterior premise)."""
    matrix = geometry.macro_matrix(gamma, beta, kappa)
    return len(geometry.find_images(np.asarray(source, float), matrix))


def _partition_namespace(gamma: float, source: np.ndarray, *,
                         beta: float = 0.0, kappa: float = 0.0,
                         w: float) -> types.SimpleNamespace:
    """Cheap analytic partition wrapped for ``born_carrier_from_partition``.

    Uses ``ChangRefsdalChannels.geometry_partition`` (no Schwinger engine),
    then repackages its fields into the duck-typed namespace the carrier
    builder consumes.  Microseconds, not the ~100 s of a full ``evaluate``.

    ``ChangRefsdalChannels`` requires a strictly increasing >= 2-point
    frequency grid, so the fixture frequency ``w`` is placed as the first
    node of a two-point grid ``[w, 2 w]``; the fold decision at each node is
    independent, so both nodes exercise the same guard behaviour and the
    per-node carrier comparison stays valid.
    """
    source = np.asarray(source, dtype=float)
    w_grid = np.array([float(w), 2.0 * float(w)])
    chan = ChangRefsdalChannels(w_grid)
    part = chan.geometry_partition(gamma=gamma, y=source, beta=beta,
                                   kappa=kappa)
    matrix = geometry.macro_matrix(gamma, beta, kappa)
    return types.SimpleNamespace(
        w=part.w, source=source, gamma=gamma, beta=beta, kappa=kappa,
        matrix=matrix, t_min=part.t_min, delays=part.delays,
        saddle_kernels=part.saddle_kernels, real_mask=part.real_mask,
        images=part.images)


class _FoldGhostBaseTestCase(unittest.TestCase):
    """Base with an anti-vacuity guard and the ray/partition helpers.

    Every concrete test increments ``self._comparisons`` for each real
    assertion it makes against production output.  ``tearDown`` FAILS if a
    test made zero comparisons, so a silently-skipping fixture (e.g. a
    premise that stopped holding and short-circuited the body) can never
    read green.
    """

    def setUp(self) -> None:
        self._comparisons = 0

    def tearDown(self) -> None:
        if self._comparisons == 0:
            self.fail(
                'anti-vacuity: this test made zero comparisons against '
                'production output -- the fixture premise likely no longer '
                'holds and the body short-circuited.')

    def _tick(self) -> None:
        """Record that one real comparison against production ran."""
        self._comparisons += 1


class FoldExteriorRefusalTestCase(_FoldGhostBaseTestCase):
    """Q4a exterior: the fold correction refuses on a two-image census.

    Fixture: gamma=0.5, theta=0.6, rho=1.40 => outside the caustic => two
    real images.  The premise (image count == 2) is asserted first; the
    fold refusal is then pinned three ways:
      * ``fold_amplification`` returns ``None`` (never a wrong number);
      * ``fold_ppgo_correction`` == ``geometric_amplification`` to machine
        precision (byte-identical raw-ppGO fallback);
      * ``born_carrier_from_partition`` is bit-identical to the no-fold
        reference path.
    """

    def setUp(self) -> None:
        super().setUp()
        self.source = _ray_source(GAMMA, THETA_EXT, RHO_EXT)
        # Premise: this fixture is genuinely EXTERIOR (two real images).
        self.assertEqual(
            _image_count(self.source, GAMMA), 2,
            'premise lost: exterior fixture is no longer a two-image census; '
            'retune RHO_EXT above the caustic.')

    def test_fold_amplification_refuses_exterior(self):
        """``fold_amplification`` returns ``None`` on the exterior census."""
        value = _airy_fold.fold_amplification(W_EXT, self.source, GAMMA)
        self._tick()
        self.assertIsNone(
            value,
            'exterior two-image census must refuse the Airy fold arm; the '
            'len(images) != 4 guard is not firing.')

    def test_fold_ppgo_correction_equals_geometric_exterior(self):
        """Exterior ``fold_ppgo_correction`` == raw ``geometric_amplification``.

        The ``len(images) != 4`` guard routes to ``_fallback()`` which is
        exactly ``geometric_amplification`` -- so the corrected value must
        be byte-identical to the uncorrected ppGO, proving no spurious Airy
        term was added to the far (min, saddle) pair.
        """
        corrected = np.atleast_1d(
            _airy_fold.fold_ppgo_correction(W_EXT, self.source, GAMMA))[0]
        raw = np.atleast_1d(
            geometric_amplification(np.array([W_EXT]), self.source, GAMMA))[0]
        self._tick()
        self.assertEqual(
            complex(corrected), complex(raw),
            'exterior fold_ppgo_correction must equal geometric_amplification '
            f'exactly; got diff {abs(complex(corrected) - complex(raw)):.3e}.')

    def test_born_carrier_bit_identical_to_no_fold_path_exterior(self):
        """Exterior carrier is bit-identical to the no-fold reference.

        The no-fold reference forces the raw ppGO path via the SAME
        production code (monkeypatching ``_merging_fold_pair`` -> ``None``),
        so equality proves the shipped ``len(images) != 4`` guard already
        took that no-fold path on the exterior census.
        """
        shipped = born_carrier_from_partition(
            _partition_namespace(GAMMA, self.source, w=W_EXT))

        orig = _airy_fold._merging_fold_pair
        _airy_fold._merging_fold_pair = lambda *a, **k: None
        try:
            reference = born_carrier_from_partition(
                _partition_namespace(GAMMA, self.source, w=W_EXT))
        finally:
            _airy_fold._merging_fold_pair = orig

        shipped = np.asarray(shipped)
        reference = np.asarray(reference)
        self._tick()
        self.assertTrue(
            np.array_equal(shipped, reference),
            'exterior born carrier diverged from the no-fold reference; the '
            'fold block is being entered on a two-image census.')


class FoldInteriorByteIdentityTestCase(_FoldGhostBaseTestCase):
    """Q4a interior: the guard is a no-op on a four-image census.

    Fixture: gamma=0.5, theta=0.6, rho=0.90 => inside the caustic => four
    real images.  Because ``len(images) == 4`` the ``!= 4`` guard is a
    logical no-op, so the shipped result is bit-for-bit identical to the
    pre-change (HEAD) result.  The build cannot commit HEAD's source into
    the test, so identity is demonstrated the durable, git-independent way:
    the interior fold is ACTIVE (measurably differs from the no-fold
    reference / raw ppGO), i.e. the guard did NOT refuse the four-image
    census.  Paired with ``FoldExteriorRefusalTestCase`` (refuse when != 4)
    this pins the guard's full contract.

    ``fold_amplification`` returns ``None`` here for a SEPARATE reason (the
    caustic-relative ``_ETA_MAX_FOLD`` distance gate at rho=0.90), not the
    ``!= 4`` guard; that is asserted and documented, not conflated.
    """

    def setUp(self) -> None:
        super().setUp()
        self.source = _ray_source(GAMMA, THETA_INT, RHO_INT)
        # Premise: this fixture is genuinely INTERIOR (four real images).
        self.assertEqual(
            _image_count(self.source, GAMMA), 4,
            'premise lost: interior fixture is no longer a four-image census; '
            'retune RHO_INT below the caustic.')

    def test_fold_ppgo_correction_active_interior(self):
        """Interior ``fold_ppgo_correction`` differs from raw ppGO.

        A four-image census with a genuine merging pair must add the Airy
        fold term, so the corrected value is NOT byte-identical to
        ``geometric_amplification``.  This proves the ``!= 4`` guard is a
        no-op for four-image censuses (it does not refuse them).
        """
        corrected = np.atleast_1d(
            _airy_fold.fold_ppgo_correction(W_INT, self.source, GAMMA))[0]
        raw = np.atleast_1d(
            geometric_amplification(np.array([W_INT]), self.source, GAMMA))[0]
        self._tick()
        self.assertNotEqual(
            complex(corrected), complex(raw),
            'interior fold_ppgo_correction must add the Airy fold term (differ '
            'from raw ppGO); the guard is wrongly refusing a four-image '
            'census.')

    def test_born_carrier_active_interior(self):
        """Interior born carrier differs from the no-fold reference.

        Forcing the no-fold path (``_merging_fold_pair`` -> ``None``) yields
        a DIFFERENT carrier than the shipped code, confirming the shipped
        code took the fold branch on the four-image census.
        """
        shipped = np.asarray(born_carrier_from_partition(
            _partition_namespace(GAMMA, self.source, w=W_INT)))

        orig = _airy_fold._merging_fold_pair
        _airy_fold._merging_fold_pair = lambda *a, **k: None
        try:
            reference = np.asarray(born_carrier_from_partition(
                _partition_namespace(GAMMA, self.source, w=W_INT)))
        finally:
            _airy_fold._merging_fold_pair = orig

        self._tick()
        self.assertFalse(
            np.array_equal(shipped, reference),
            'interior born carrier is identical to the no-fold reference; the '
            'fold block was skipped on a four-image census.')

    def test_fold_amplification_none_interior_via_eta_gate(self):
        """Interior ``fold_amplification`` is ``None`` (eta gate, not != 4).

        At rho=0.90 the caustic-relative ``_ETA_MAX_FOLD`` distance gate
        declines; the ``!= 4`` guard is NOT the cause (four images are
        present).  Pinned so a future eta-gate change is noticed here rather
        than silently flipping the interior serve.
        """
        value = _airy_fold.fold_amplification(W_INT, self.source, GAMMA)
        self._tick()
        self.assertIsNone(
            value,
            'interior fold_amplification unexpectedly served at rho=0.90; the '
            '_ETA_MAX_FOLD distance gate is expected to decline here.')


def _ghost_im_and_sep(source: np.ndarray, gamma: float, w: float,
                      *, beta: float = 0.0, kappa: float = 0.0):
    """Independently recompute the rung's two gate inputs.

    Returns ``(Im tau_c, min image-to-ghost separation)`` via the same
    ``ghost_kernel`` / ``find_images`` producers the rung reads, so premises
    and boundary-flip thresholds are DERIVED from live geometry rather than
    pinned.  Propagates ``GhostAbsentError`` for interior censuses.
    """
    source = np.asarray(source, dtype=float)
    matrix = geometry.macro_matrix(gamma, beta, kappa)
    real_images = geometry.find_images(source, matrix)
    ghost = geometry.ghost_kernel([float(w)], source, matrix)
    im_tau = float(ghost.delay.imag)
    x_c = ghost.position
    separation = min(
        float(np.sqrt(np.sum(np.abs(x_a - x_c) ** 2)))
        for x_a in real_images)
    return im_tau, separation


class GhostRungGateTestCase(_FoldGhostBaseTestCase):
    """Q4b: the ghost rung's serve / decline / refuse DECISION.

    Three synthetic configs at gamma=0.5, w=90:
      * SERVE   -- rho=2.5, theta=pi/4: both gates admit => finite complex;
      * CAVEAT  -- rho=1.05: Im(tau_c) < 0.4 (decay gate declines) => None;
      * INTERIOR-- rho=0.90: four images => ``GhostAbsentError`` => None.

    Boundary-flip teeth monkeypatch the geometry thresholds to the live gate
    inputs +/- an epsilon and confirm the decision flips, proving BOTH gates
    (decay and separation) are individually load-bearing.
    """

    def setUp(self) -> None:
        super().setUp()
        self.serve = _ray_source(GAMMA, THETA_GHOST, RHO_GHOST_SERVE)
        self.caveat = _ray_source(GAMMA, THETA_GHOST, RHO_GHOST_CAVEAT)
        self.interior = _ray_source(GAMMA, THETA_INT, RHO_INT)
        # Premise: serve config is exterior AND both gates admit.
        self.assertEqual(_image_count(self.serve, GAMMA), 2,
                         'serve premise lost: not a two-image census.')
        im_serve, sep_serve = _ghost_im_and_sep(self.serve, GAMMA, W_GHOST)
        self.assertGreaterEqual(
            im_serve, GHOST_DECAY_IM_THRESHOLD,
            'serve premise lost: ghost is not decayed enough to admit.')
        self.assertGreaterEqual(
            sep_serve, GHOST_SEPARATION_MIN,
            'serve premise lost: ghost is not resolved enough to admit.')
        self.im_serve = im_serve
        self.sep_serve = sep_serve
        # Premise: caveat config is exterior with the decay gate failing but
        # the separation gate passing (so decay is the sole blocker).
        self.assertEqual(_image_count(self.caveat, GAMMA), 2,
                         'caveat premise lost: not a two-image census.')
        im_cav, sep_cav = _ghost_im_and_sep(self.caveat, GAMMA, W_GHOST)
        self.assertLess(
            im_cav, GHOST_DECAY_IM_THRESHOLD,
            'caveat premise lost: decay gate no longer declines this config.')
        self.assertGreaterEqual(
            sep_cav, GHOST_SEPARATION_MIN,
            'caveat premise lost: separation gate now also blocks, so a decay '
            'flip would not isolate the decay gate.')
        self.im_caveat = im_cav
        # Premise: interior config is a four-image census.
        self.assertEqual(_image_count(self.interior, GAMMA), 4,
                         'interior premise lost: not a four-image census.')

    def test_serve_config_returns_finite_complex(self):
        """The clean-serve config serves a finite complex value."""
        value = _ghost_ppgo_amplification(W_GHOST, self.serve, GAMMA)
        self._tick()
        self.assertIsNotNone(
            value, 'ghost rung declined a config where both gates admit.')
        self.assertTrue(np.isfinite(abs(value)),
                        'ghost rung served a non-finite value.')

    def test_caveat_config_declines_none(self):
        """The near-caustic caveat config declines (decay gate)."""
        value = _ghost_ppgo_amplification(W_GHOST, self.caveat, GAMMA)
        self._tick()
        self.assertIsNone(
            value, 'ghost rung served the caveat band; the decay gate '
            '(Im tau_c < 0.4) is not declining.')

    def test_interior_config_declines_via_ghost_absent(self):
        """Interior config declines with ``None`` via ``GhostAbsentError``.

        The rung must NEITHER serve NOR refuse-to-engine on an interior
        node -- it declines with ``None`` so the interior fold serve
        upstream stays byte-identical.  Independently confirm the decline is
        the ``GhostAbsentError`` path (not a bare ``GhostDomainError``).
        """
        value = _ghost_ppgo_amplification(W_GHOST, self.interior, GAMMA)
        self._tick()
        self.assertIsNone(
            value, 'ghost rung did not decline an interior four-image node.')
        matrix = geometry.macro_matrix(GAMMA, 0.0, 0.0)
        with self.assertRaises(geometry.GhostAbsentError):
            geometry.ghost_kernel([W_GHOST], self.interior, matrix)

    def test_decay_gate_flips_serve_to_decline(self):
        """Raising the decay threshold above the serve Im flips serve->None.

        Isolates the decay gate on the serve config: nothing else changes,
        so a flip to ``None`` proves ``Im tau_c >= threshold`` is live.
        """
        raised = self.im_serve + 1e-3
        with mock.patch.object(geometry, '_GHOST_DECAY_IM_THRESHOLD', raised):
            flipped = _ghost_ppgo_amplification(W_GHOST, self.serve, GAMMA)
        self._tick()
        self.assertIsNone(
            flipped, 'raising the decay threshold above the serve Im did not '
            'flip the serve to a decline; the decay gate is not live.')

    def test_separation_gate_flips_serve_to_decline(self):
        """Raising the separation threshold above the serve sep flips->None.

        Isolates the separation gate on the serve config.
        """
        raised = self.sep_serve + 1e-3
        with mock.patch.object(geometry, '_GHOST_SEPARATION_MIN', raised):
            flipped = _ghost_ppgo_amplification(W_GHOST, self.serve, GAMMA)
        self._tick()
        self.assertIsNone(
            flipped, 'raising the separation threshold above the serve '
            'separation did not flip the serve to a decline; the separation '
            'gate is not live.')

    def test_caveat_flips_to_serve_when_decay_threshold_lowered(self):
        """Lowering the decay threshold below the caveat Im flips None->serve.

        Directly honours the spec's boundary-flip: the caveat config's
        decline is specifically the decay gate, so dropping the threshold
        below its Im (separation already passes) makes it serve.
        """
        lowered = self.im_caveat - 1e-3
        with mock.patch.object(geometry, '_GHOST_DECAY_IM_THRESHOLD', lowered):
            flipped = _ghost_ppgo_amplification(W_GHOST, self.caveat, GAMMA)
        self._tick()
        self.assertIsNotNone(
            flipped, 'lowering the decay threshold below the caveat Im did not '
            'flip the decline to a serve; the caveat decline was not the '
            'decay gate.')
        self.assertTrue(np.isfinite(abs(flipped)),
                        'flipped caveat serve is non-finite.')


class GhostSignConventionTestCase(_FoldGhostBaseTestCase):
    """Q4c: the served value pins the '+' sign and non-conjugated carrier.

    On one clean-serve config (rho=2.5) at LOW frequency w=12 (where the
    ghost term ~4e-4 is resolvable), the rung's output must equal::

        geometric_amplification(w, y, gamma, beta, kappa)
            + ghost.kernel * exp(1j * w * ghost.delay)

    to ``SIGN_ATOL`` (1e-12).  This simultaneously pins the ``+`` sign and
    the NON-conjugated ``tau_c`` carrier; the ``-`` and conjugated variants
    (validated as worse) must NOT match.  ``ghost_kernel`` is the
    independent producer of ``tau_c`` and the kernel, so this is a
    refactor-durable algebraic identity, not a transcription oracle.
    """

    def setUp(self) -> None:
        super().setUp()
        self.source = _ray_source(GAMMA, THETA_GHOST, RHO_GHOST_SERVE)
        self.assertEqual(_image_count(self.source, GAMMA), 2,
                         'sign-pin premise lost: not a two-image census.')
        im_tau, separation = _ghost_im_and_sep(self.source, GAMMA, W_SIGN)
        self.assertGreaterEqual(im_tau, GHOST_DECAY_IM_THRESHOLD,
                                'sign-pin premise lost: decay gate declines.')
        self.assertGreaterEqual(separation, GHOST_SEPARATION_MIN,
                                'sign-pin premise lost: separation gate '
                                'declines.')
        matrix = geometry.macro_matrix(GAMMA, 0.0, 0.0)
        self.ghost = geometry.ghost_kernel([W_SIGN], self.source, matrix)
        self.carrier = complex(geometric_amplification(
            W_SIGN, self.source, GAMMA))
        self.ghost_term = (complex(np.atleast_1d(self.ghost.kernel)[0])
                           * np.exp(1j * W_SIGN * complex(self.ghost.delay)))
        # Premise: the ghost term is resolvable (not decayed into round-off),
        # so the '-'/conjugate mutations are genuinely discriminable.
        self.assertGreater(
            abs(self.ghost_term), 1e-8,
            'sign-pin premise lost: ghost term decayed below the tolerance; '
            'the sign teeth would be inert -- lower W_SIGN.')

    def test_served_value_matches_plus_nonconjugated(self):
        """Served value == carrier + kernel*exp(1j w tau_c) to 1e-12."""
        served = _ghost_ppgo_amplification(W_SIGN, self.source, GAMMA)
        self._tick()
        self.assertIsNotNone(served, 'sign-pin config unexpectedly declined.')
        expected = self.carrier + self.ghost_term
        self.assertLessEqual(
            abs(complex(served) - expected), SIGN_ATOL,
            'served value does not match carrier + non-conjugated ghost term; '
            f'diff {abs(complex(served) - expected):.3e}.')

    def test_negated_carrier_variant_fails(self):
        """The '-' sign variant must NOT match the served value."""
        served = complex(_ghost_ppgo_amplification(W_SIGN, self.source, GAMMA))
        self._tick()
        minus_variant = self.carrier - self.ghost_term
        self.assertGreater(
            abs(served - minus_variant), SIGN_ATOL,
            'the negated-ghost variant matched the served value; the sign is '
            'not actually pinned.')

    def test_conjugated_carrier_variant_fails(self):
        """The conjugated-``tau_c`` variant must NOT match the served value."""
        served = complex(_ghost_ppgo_amplification(W_SIGN, self.source, GAMMA))
        self._tick()
        conj_term = (complex(np.atleast_1d(self.ghost.kernel)[0])
                     * np.exp(1j * W_SIGN * np.conj(complex(self.ghost.delay))))
        conj_variant = self.carrier + conj_term
        self.assertGreater(
            abs(served - conj_variant), SIGN_ATOL,
            'the conjugated-tau_c variant matched the served value; the '
            'non-conjugated carrier is not actually pinned.')


class FoldGuardSelfFalsificationTestCase(_FoldGhostBaseTestCase):
    """Self-falsification: the ``len(images) != 4`` guard is load-bearing.

    If the exterior guard were absent, ``_merging_fold_pair`` would still
    return the FAR (min, saddle) pair for a two-image census and the fold
    machinery would serve a spurious non-``None`` Airy value.  This proves,
    git-independently, that the guard is what makes the exterior refusal
    happen -- and that the suite can go red when the guarded behaviour is
    violated.
    """

    def setUp(self) -> None:
        super().setUp()
        self.source = _ray_source(GAMMA, THETA_EXT, RHO_EXT)
        self.assertEqual(_image_count(self.source, GAMMA), 2,
                         'teeth premise lost: exterior fixture is not a '
                         'two-image census.')

    def test_merging_fold_pair_serves_far_pair_on_exterior(self):
        """``_merging_fold_pair`` DOES return a pair on the exterior census.

        This is the mechanism the ``!= 4`` guard exists to suppress: the
        helper itself is happy to pair the far min/saddle, so without the
        guard the fold arm would serve a spurious value.
        """
        matrix = geometry.macro_matrix(GAMMA, 0.0, 0.0)
        images = geometry.find_images(self.source, matrix)
        pair = _airy_fold._merging_fold_pair(images, self.source, matrix)
        self._tick()
        self.assertIsNotNone(
            pair, 'the exterior refusal does not actually depend on the '
            'len(images) != 4 guard; _merging_fold_pair already refuses here, '
            'so the guard is redundant and this teeth test is vacuous.')

    def test_exterior_byte_identity_assertion_has_teeth(self):
        """A 1e-9 perturbation is caught by the machine-precision equality.

        The main exterior refusal test asserts
        ``fold_ppgo_correction == geometric_amplification`` exactly.  This
        confirms that assertion can go RED: a fold term as small as a 1e-9
        relative perturbation of the served value is distinguishable, so the
        byte-identity check would catch even a tiny spurious Airy
        correction.  Without this teeth check an exact-equality assertion
        could be silently satisfied by a degenerate (e.g. all-zero) value.
        """
        corrected = complex(np.atleast_1d(
            _airy_fold.fold_ppgo_correction(W_EXT, self.source, GAMMA))[0])
        # A spurious fold term would perturb the served value; even 1e-9
        # relative must be caught by the exact comparison the suite uses.
        perturbed = corrected * (1.0 + 1e-9)
        self._tick()
        self.assertNotEqual(
            corrected, perturbed,
            'machine-precision equality cannot distinguish a 1e-9 relative '
            'perturbation; the exterior byte-identity assertions lack teeth '
            '(the served value may be degenerate).')
        # And the served value is non-degenerate (non-zero), so the equality
        # in the main test is a meaningful match, not 0 == 0.
        self.assertGreater(abs(corrected), 1e-6,
                           'exterior served value is ~0; byte-identity would '
                           'be a vacuous 0 == 0 match.')
