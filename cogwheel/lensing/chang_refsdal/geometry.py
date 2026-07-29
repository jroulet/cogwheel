"""
Image geometry and stationary-phase kernels for the Chang--Refsdal lens.

WHAT
----
A point mass embedded in the locally constant convergence ``kappa`` and
shear ``gamma`` (oriented at ``beta``) of a macro image.  This module
provides the geometrical optics layer of that model: the macro matrix,
the exact quartic image solver, Fermat delays, Hessians, signed
magnifications, Morse indices, the stationary-phase (saddle) kernels
through relative order ``w**-2``, and the critical-curve utilities used
to place unoccupied image labels.

WHY
---
The wave-optics amplification of the lens is written as

    F(w) = sum_a exp(1j * w * tau_a) * K_a(w),

with the image delays ``tau_a`` carried analytically so that only the
slowly varying kernels ``K_a`` need interpolation.  Everything that
decomposition needs about the lens plane -- where the images are, how
long they take, how bright they are, and what their high-frequency
kernels asymptote to -- is computed here, with no dependence on the
wave-optics evaluation.  The stationary-phase kernels this module
returns are the ``w -> inf`` targets that the exact channel kernels
approach.

Conventions
-----------
Angles (image positions ``x`` and source positions ``y``) are in units
of the point mass's Einstein radius.  Delays ``tau`` are dimensionless
Fermat delays; the dimensionless frequency conjugate to them is

    w = 8 * pi * G * M_L * (1 + z_L) * f / c**3,

which is exactly *linear* in the observed frequency ``f``.  A
dimensionless delay ``tau`` therefore corresponds to the constant time
shift ``dt = 4 * G * M_L * (1 + z_L) * tau / c**3``, independent of
frequency.

Limitations
-----------
The geometry layer supports both parities of the macro image, provided
the mass-sheet reduction stays real (``lam = 1 - kappa > 0``):

* positive parity ``lam > abs(gamma)`` -- the classical astroid
  (4-cusp) caustic and positive-definite Hessian;
* macro saddle ``0 < lam < abs(gamma)`` -- the two 3-cusp deltoid
  lobes and an indefinite Hessian (Type II images).

``macro_matrix`` (and the critical-curve utilities) raise
`LensDomainError` for the two named refusals: ``lam <= 0`` (over-
critical / Type III, where ``sqrt(lam)`` is imaginary and the reduction
dies) and the exact parity boundary ``abs(gamma) == lam`` (``det A = 0``,
a degenerate fold branch point).  The wave-optics evaluator that turns
this geometry into an amplification lives in the sibling ``operator``
and Schwinger modules; this module is purely the geometrical-optics
layer and is parity-agnostic wherever the algebra permits.

Accuracy
--------
Near a fold caustic the image quartic acquires a double root, so image
*positions* there are accurate only to ``sqrt(eps) ~ 1.5e-8``.  Delays
are quadratically insensitive to that error because images are
stationary points of the Fermat potential, so ``delay`` retains full
``eps`` accuracy even where positions do not.  Magnifications, by
contrast, are ``1 / det(H)`` and are genuinely ill conditioned near a
critical point, where ``det(H) -> 0``: both `magnification` and
`image_kernel` diverge there, and callers that need finite answers
across a caustic must blend individual images into a cluster kernel
rather than using the single-image expressions of this module.
"""
from __future__ import annotations

from typing import NamedTuple

import numba
import numpy as np
from scipy.optimize import brentq, minimize_scalar

#: Smallest ``|x|**2`` treated as nonzero; below it the point mass's
#: logarithmic potential is singular.
_MIN_RADIUS_SQUARED = 1e-30

#: Newton polish is stopped once the lens residual falls below this.
_POLISH_RESIDUAL = 2e-13


class LensDomainError(ValueError):
    """Lens parameters fall outside the supported model domain."""


class CriticalPoint(NamedTuple):
    """A point on the critical curve and its local frame.

    Attributes
    ----------
    image : np.ndarray
        Shape (2,), the critical point in the lens plane.
    source : np.ndarray
        Shape (2,), the corresponding caustic point in the source
        plane.
    hard_axis : np.ndarray
        Shape (2,), unit Hessian eigenvector with the larger absolute
        eigenvalue (the direction transverse to the fold).
    soft_axis : np.ndarray
        Shape (2,), unit Hessian eigenvector with the vanishing
        eigenvalue (the direction along which images merge).  Oriented
        so that ``(hard_axis, soft_axis)`` is right handed.
    hard_eigenvalue : float
        The Hessian eigenvalue along ``hard_axis``.
    """

    image: np.ndarray
    source: np.ndarray
    hard_axis: np.ndarray
    soft_axis: np.ndarray
    hard_eigenvalue: float


class NearestCausticPoint(NamedTuple):
    """Closest caustic point to a source, and its local frame.

    Attributes
    ----------
    theta : float
        Polar angle in ``[0, 2*pi)`` parametrizing the critical curve
        at the closest point.
    image, source, hard_axis, soft_axis, hard_eigenvalue
        As in `CriticalPoint`, evaluated at ``theta``.
    distance : float
        Euclidean distance in the source plane from the source to the
        caustic.  Unsigned: it does not say which side of the caustic
        the source is on.
    """

    theta: float
    image: np.ndarray
    source: np.ndarray
    hard_axis: np.ndarray
    soft_axis: np.ndarray
    hard_eigenvalue: float
    distance: float


def macro_matrix(gamma: float, beta: float = 0.0,
                 kappa: float = 0.0) -> np.ndarray:
    """
    Quadratic part of the Fermat Hessian: convergence plus shear.

    Both parities of the macro image are supported as long as the
    mass-sheet reduction stays real (``lam = 1 - kappa > 0``): positive
    parity ``lam > abs(gamma)`` and the macro saddle
    ``0 < lam < abs(gamma)``.  The matrix itself is the same
    ``(1 - kappa) * I - gamma * Q(beta)`` in either case; only the
    signature of its eigenvalues (and hence the image topology) differs.

    Parameters
    ----------
    gamma : float
        External shear magnitude.
    beta : float
        External shear orientation, radians.
    kappa : float
        External convergence.

    Returns
    -------
    np.ndarray
        Shape (2, 2) symmetric matrix
        ``(1 - kappa) * I - gamma * Q(beta)``.

    Raises
    ------
    LensDomainError
        If ``1 - kappa <= 0`` (over-critical / Type III, where the
        mass-sheet reduction ``sqrt(1 - kappa)`` is not real), or if
        ``abs(gamma) == 1 - kappa`` exactly (the parity boundary
        ``det A = 0``, a degenerate fold branch point).  These are the
        two named refusals; the positive-parity and macro-saddle
        interiors both return normally.
    """
    gamma = float(gamma)
    kappa = float(kappa)
    lam = 1.0 - kappa
    if lam <= 0.0:
        raise LensDomainError(
            f'Cannot build a macro matrix for (kappa, gamma) = '
            f'({kappa}, {gamma}): 1 - kappa = {lam} <= 0 (kappa >= 1). '
            f'The mass-sheet reduction sqrt(1 - kappa) is not real and '
            f'over-critical / Type III configurations are out of scope.')
    if abs(gamma) == lam:
        raise LensDomainError(
            f'Cannot build a macro matrix for (kappa, gamma) = '
            f'({kappa}, {gamma}): |gamma| == 1 - kappa = {lam} exactly, '
            f'so det A = 0. The source sits on the parity boundary '
            f'between the positive-parity (1 - kappa > |gamma|) and '
            f'macro-saddle (1 - kappa < |gamma|) domains, a degenerate '
            f'fold branch point; this boundary is a named refusal.')
    cos2b, sin2b = np.cos(2.0 * beta), np.sin(2.0 * beta)
    shear = np.array([[cos2b, sin2b], [sin2b, -cos2b]])
    return (1.0 - kappa) * np.eye(2) - gamma * shear


def hessian(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Hessian of the Fermat delay at an image position.

    Parameters
    ----------
    image : np.ndarray
        Shape (2,), position in the lens plane.
    matrix : np.ndarray
        Shape (2, 2), the macro matrix (see `macro_matrix`).

    Returns
    -------
    np.ndarray
        Shape (2, 2) symmetric Hessian.

    Raises
    ------
    LensDomainError
        If ``image`` is at the point mass, where the Hessian is
        singular.
    """
    image = np.asarray(image, dtype=float)
    radius_squared = float(image @ image)
    if radius_squared <= 0.0:
        raise LensDomainError(
            'Cannot evaluate the Fermat Hessian at the point mass '
            '(|x| = 0), where the lens potential is singular; pass an '
            'image position away from the origin.')
    return (matrix - np.eye(2) / radius_squared
            + 2.0 * np.outer(image, image) / radius_squared**2)


def lens_residual(image: np.ndarray, source: np.ndarray,
                  matrix: np.ndarray) -> np.ndarray:
    """
    Residual of the lens equation, zero at an image.

    Parameters
    ----------
    image : np.ndarray
        Shape (2,), trial position in the lens plane.
    source : np.ndarray
        Shape (2,), source position.
    matrix : np.ndarray
        Shape (2, 2), the macro matrix.

    Returns
    -------
    np.ndarray
        Shape (2,), ``matrix @ x - x / |x|**2 - y``.  Infinite at the
        point mass rather than raising, so that root polishing can
        reject a step onto the singularity.
    """
    image = np.asarray(image, dtype=float)
    source = np.asarray(source, dtype=float)
    radius_squared = float(image @ image)
    if radius_squared <= _MIN_RADIUS_SQUARED:
        return np.array([np.inf, np.inf])
    return matrix @ image - image / radius_squared - source


def delay(image: np.ndarray, source: np.ndarray,
          matrix: np.ndarray) -> float:
    """
    Dimensionless Fermat delay of a lens-plane position.

    Parameters
    ----------
    image : np.ndarray
        Shape (2,), position in the lens plane.
    source : np.ndarray
        Shape (2,), source position.
    matrix : np.ndarray
        Shape (2, 2), the macro matrix.

    Returns
    -------
    float
        ``x @ A @ x / 2 - y @ x + y @ y / 2 - ln|x|``.  At an image
        this is stationary, hence accurate to full machine precision
        even where the image position itself is not (see the module
        docstring).
    """
    image = np.asarray(image, dtype=float)
    source = np.asarray(source, dtype=float)
    return float(0.5 * image @ matrix @ image - source @ image
                 + 0.5 * source @ source
                 - np.log(np.linalg.norm(image)))


def magnification(image: np.ndarray, matrix: np.ndarray) -> float:
    """
    Signed magnification of an image.

    Parameters
    ----------
    image : np.ndarray
        Shape (2,), image position.
    matrix : np.ndarray
        Shape (2, 2), the macro matrix.

    Returns
    -------
    float
        ``1 / det(H)``.  Positive for minima and maxima, negative for
        saddles.  Ill conditioned near a critical point, where
        ``det(H) -> 0`` and the magnification diverges.
    """
    return 1.0 / float(np.linalg.det(hessian(image, matrix)))


def morse_index(image: np.ndarray, matrix: np.ndarray) -> int:
    """
    Morse index of an image: the number of negative Hessian
    eigenvalues.

    Parameters
    ----------
    image : np.ndarray
        Shape (2,), image position.
    matrix : np.ndarray
        Shape (2, 2), the macro matrix.

    Returns
    -------
    int
        0 for a minimum, 1 for a saddle, 2 for a maximum.  It enters
        the kernel as the phase ``exp(-0.5j * pi * n_a)``.
    """
    eigenvalues = np.linalg.eigvalsh(hessian(image, matrix))
    return int(np.sum(eigenvalues < 0.0))


def _check_image_census(images: list[np.ndarray],
                        matrix: np.ndarray) -> None:
    """
    Enforce the Morse index theorem on a solved image set.

    For a point-mass perturbation of a smooth macro potential the
    signed sum over images obeys ``sum_a (-1)**n_a == sign(det A) - 1``,
    where ``n_a`` is the Morse index of image ``a`` and ``A`` is the
    macro matrix — for a NON-DEGENERATE stationary set.  ANY nonzero
    discrepancy over REGULAR images means the solver silently dropped
    or duplicated images (the F012 dead-zone pair drop, or any
    single-image loss), so the returned set is not a faithful census;
    returning it would let a downstream consumer produce a
    finite-but-wrong amplification.  A discrepancy is legitimate ONLY
    when the census carries a NEAR-CRITICAL WITNESS (an image with
    ``|det H| <= 1e-6 * ||H||_F^2``): a source on a fold merges its
    (min, saddle) pair into one near-critical survivor (count 3, odd
    discrepancy) and a source on a cusp collapses its (min, saddle,
    min) triple likewise (count 2, even discrepancy) — Morse theory
    does not constrain degenerate stationary sets, and a defective
    drop leaves only regular images because the lost images were
    elsewhere.  Degenerate censuses are safe downstream: the
    resolvability gate routes them to the wave branch, and the
    fold-degenerate stationary-phase guard (FINDINGS F015) refuses the
    geometric kernel.  Counts outside ``[1, 4]`` are refused
    unconditionally (the quartic admits at most 4 stationary points).

    Parameters
    ----------
    images : list of np.ndarray
        The final deduplicated image positions, each of shape (2,).
    matrix : np.ndarray
        Shape (2, 2), the macro matrix the images were solved for.

    Raises
    ------
    LensDomainError
        If the signed Morse sum violates the index theorem.
    """
    signed = sum((-1) ** morse_index(image, matrix) for image in images)
    sign_det_a = 1 if float(np.linalg.det(matrix)) > 0.0 else -1
    discrepancy = signed - (sign_det_a - 1)
    if discrepancy != 0 and 1 <= len(images) <= 4:
        # Any discrepancy is the defect signature -- UNLESS the census
        # is visibly DEGENERATE.  Morse theory constrains only
        # non-degenerate stationary sets: a source on a fold merges its
        # (min, saddle) pair into one near-critical survivor (count 3,
        # odd discrepancy), and a source on a cusp collapses its
        # (min, saddle, min) triple likewise (count 2, even
        # discrepancy, seen in the channel-layer axis-cusp sweep).  In
        # BOTH legitimate cases a returned image sits essentially on
        # the critical curve; a defective drop (the F012 dead zone, or
        # any single-image loss) leaves only REGULAR images because the
        # lost images were elsewhere.  The near-critical witness is the
        # physical discriminator.
        for image in images:
            image_hessian = hessian(image, matrix)
            degeneracy_scale = float(np.sum(image_hessian
                                            * image_hessian))
            if abs(float(np.linalg.det(image_hessian))) \
                    <= 1e-6 * degeneracy_scale:
                return
    if discrepancy != 0 or not 1 <= len(images) <= 4:
        raise LensDomainError(
            f'Image census defect for macro matrix {matrix.tolist()}: '
            f'the {len(images)} returned images give a signed Morse sum '
            f'sum_a(-1)^(n_a) = {signed}, but the index theorem requires '
            f'sum_a(-1)^(n_a) == sign(det A) - 1 = {sign_det_a - 1} '
            f'(count, signed, sign_detA) = '
            f'({len(images)}, {signed}, {sign_det_a}). A discrepancy '
            f'over REGULAR images means the solver dropped or '
            f'duplicated images (F012), so the returned set is an '
            f'incomplete census and cannot be certified. (A degenerate '
            f'census carrying a near-critical witness image — a '
            f'fold-merged pair or cusp-merged triple — is legitimate '
            f'and passes.)')


def _source_frame(source: np.ndarray) -> tuple[float, np.ndarray]:
    """Return source radius and orthogonal matrix whose first axis is
    the source direction."""
    source = np.asarray(source, dtype=float)
    if source.shape != (2,):
        raise ValueError(
            f'Cannot build a source frame from an array of shape '
            f'{source.shape}: the source must be a two-vector.')
    radius = float(np.linalg.norm(source))
    if radius == 0.0:
        return 0.0, np.eye(2)
    axis1 = source / radius
    axis2 = np.array([-axis1[1], axis1[0]])
    return radius, np.column_stack([axis1, axis2])


def image_quartic_coefficients(source_radius: float,
                               rotated_matrix: np.ndarray) -> np.ndarray:
    r"""
    Quartic coefficients for ``u = 1 / |x|**2``, descending order.

    In the source-aligned frame put ``A = [[a11, a12], [a12, a22]]``
    and ``y = (Y, 0)``.  The lens equation gives

        ``x1 = Y * (a22 - u) / D``, ``x2 = -Y * a12 / D``,
        ``D = (a11 - u) * (a22 - u) - a12**2``,

    and the radial constraint ``|x|**2 = 1 / u`` becomes

        ``D**2 - Y**2 * u * [(a22 - u)**2 + a12**2] = 0``.

    Parameters
    ----------
    source_radius : float
        ``Y = |y| >= 0``.
    rotated_matrix : np.ndarray
        Shape (2, 2), the macro matrix expressed in the source-aligned
        frame.

    Returns
    -------
    np.ndarray
        Shape (5,), coefficients of ``u**4 ... u**0``.

    Raises
    ------
    ValueError
        If ``source_radius`` is negative or ``rotated_matrix`` is not
        2 by 2.
    """
    source_radius = float(source_radius)
    if source_radius < 0.0:
        raise ValueError(
            f'Cannot build quartic coefficients for source_radius = '
            f'{source_radius}: it must be nonnegative.')
    rotated_matrix = np.asarray(rotated_matrix, dtype=float)
    if rotated_matrix.shape != (2, 2):
        raise ValueError(
            f'Cannot build quartic coefficients from an array of '
            f'shape {rotated_matrix.shape}: rotated_matrix must be '
            f'2 by 2.')
    a11 = float(rotated_matrix[0, 0])
    a12 = float(rotated_matrix[0, 1])
    a22 = float(rotated_matrix[1, 1])
    determinant = a11 * a22 - a12 * a12
    radius_squared = source_radius * source_radius
    return np.array(
        [1.0,
         -2.0 * (a11 + a22) - radius_squared,
         (a11 * a11 + 4.0 * a11 * a22 + a22 * a22 - 2.0 * a12 * a12
          + 2.0 * a22 * radius_squared),
         (-2.0 * (a11 + a22) * determinant
          - radius_squared * (a22 * a22 + a12 * a12)),
         determinant * determinant],
        dtype=float)


def _newton_polish(image: np.ndarray, source: np.ndarray,
                   matrix: np.ndarray, *,
                   max_steps: int = 8) -> np.ndarray:
    """Deterministically polish one algebraic candidate with the lens
    Jacobian."""
    trial = np.asarray(image, dtype=float).copy()
    for _ in range(max_steps):
        residual = lens_residual(trial, source, matrix)
        if (not np.all(np.isfinite(residual))
                or np.linalg.norm(residual) < _POLISH_RESIDUAL):
            break
        jacobian = hessian(trial, matrix)
        try:
            step = np.linalg.solve(jacobian, residual)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(jacobian, residual, rcond=None)[0]
        # A large Newton step near a multiple root is less useful than
        # the original algebraic candidate.  Backtrack until the
        # residual falls.
        old_norm = float(np.linalg.norm(residual))
        accepted = False
        scale = 1.0
        for _ in range(8):
            stepped = trial - scale * step
            if np.linalg.norm(stepped) < 1e-12:
                scale *= 0.5
                continue
            new_norm = float(np.linalg.norm(
                lens_residual(stepped, source, matrix)))
            if np.isfinite(new_norm) and new_norm < old_norm:
                trial = stepped
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
    return trial


def _centered_source_images(matrix: np.ndarray, *,
                            degeneracy_tolerance: float
                            ) -> list[np.ndarray]:
    """Images for a source at ``y = 0``.

    At the origin the lens equation ``A x = x / |x|**2`` has a real
    solution on an eigenaxis only where the eigenvalue is positive
    (``|x|**2 = 1 / lambda``).  Hence:

    * positive parity (both eigenvalues positive): two images on each
      eigenaxis (four total); the ``gamma = 0`` Einstein ring is
      intentionally rejected;
    * macro saddle (one negative, one positive eigenvalue): only the
      positive-eigenvalue axis carries images ``+- e / sqrt(lambda_+)``
      -- two saddle images, consistent with the ``(1, 1)`` centered
      census;
    * negative definite (both eigenvalues non-positive, i.e.
      ``1 - kappa <= 0``): no real centered image, refused.
    """
    values, vectors = np.linalg.eigh(matrix)
    n_positive = int(np.sum(values > 0.0))
    if n_positive == 0:
        raise LensDomainError(
            f'Cannot solve for images of a centered source with macro '
            f'matrix eigenvalues {values}: neither eigenvalue is '
            f'positive (1 - kappa <= 0), so no real image lies on '
            f'either eigenaxis. Over-critical / Type III configurations '
            f'are out of scope.')
    if n_positive == 2:
        # Positive-parity macro image: unchanged, byte-identical.
        if (abs(values[1] - values[0])
                <= degeneracy_tolerance * max(abs(values[0]),
                                              abs(values[1]), 1.0)):
            raise LensDomainError(
                'Cannot enumerate discrete images at zero source and '
                'zero shear: the macro matrix is degenerate and the '
                'image is a continuous Einstein ring. Use a nonzero '
                'shear or a nonzero source position.')
        images: list[np.ndarray] = []
        for value, vector in zip(values, vectors.T):
            image = vector / np.sqrt(value)
            images.extend([image, -image])
        return images
    # Macro saddle: only the positive-eigenvalue axis admits images.
    axis = int(np.argmax(values))
    image = vectors[:, axis] / np.sqrt(values[axis])
    return [image, -image]


def _axial_candidates(source_radius: float, a11: float, a22: float, *,
                      root_tolerance: float) -> list[np.ndarray]:
    """Candidates in the source frame when the rotated macro matrix is
    diagonal, where the generic reconstruction formula is removably
    singular."""
    candidates: list[np.ndarray] = []
    # Axial images satisfy a11 * x1**2 - Y * x1 - 1 = 0.
    discriminant = source_radius * source_radius + 4.0 * a11
    if discriminant >= -root_tolerance and abs(a11) > 1e-14:
        root = np.sqrt(max(discriminant, 0.0))
        candidates.extend(
            [np.array([(source_radius + root) / (2.0 * a11), 0.0]),
             np.array([(source_radius - root) / (2.0 * a11), 0.0])])

    # A symmetric off-axis pair can occur at u = a22.
    if a22 > 0.0 and abs(a11 - a22) > 1e-14:
        x1_axial = source_radius / (a11 - a22)
        x2_squared = 1.0 / a22 - x1_axial * x1_axial
        if x2_squared >= -root_tolerance:
            x2_axial = np.sqrt(max(x2_squared, 0.0))
            candidates.extend([np.array([x1_axial, x2_axial]),
                               np.array([x1_axial, -x2_axial])])
    return candidates


def _companion_roots(coefficients: np.ndarray) -> np.ndarray:
    """Polynomial roots via the companion eigenvalue method.

    This is the SAME algorithm ``numpy.roots`` uses -- the eigenvalues
    of the Frobenius companion matrix -- but without the generic input
    handling (leading/trailing-zero trimming, dtype promotion, root
    padding) that dominates its cost for the small fixed-degree image
    quartic.  For a polynomial with no leading or trailing zero it
    returns the SAME LAPACK eigenvalues of the SAME companion matrix as
    ``numpy.roots``, hence a bit-for-bit identical root set; any other
    polynomial (a leading/trailing zero or a non-finite coefficient,
    which the production quartic never has because ``det A != 0`` is
    refused upstream) is deferred to ``numpy.roots`` so the general
    contract is preserved exactly.

    Parameters
    ----------
    coefficients : np.ndarray
        1-D polynomial coefficients in descending degree order, as
        returned by `image_quartic_coefficients`.

    Returns
    -------
    np.ndarray
        The polynomial roots, identical to ``numpy.roots(coefficients)``.
    """
    coefficients = np.asarray(coefficients)
    if (coefficients.ndim != 1 or coefficients.size < 2
            or coefficients[0] == 0.0 or coefficients[-1] == 0.0
            or not np.all(np.isfinite(coefficients))):
        return np.roots(coefficients)
    # No leading/trailing zeros: numpy.roots would build exactly this
    # companion matrix and return eigvals(companion) with no padding.
    size = coefficients.size - 1
    companion = np.diag(np.ones(size - 1, dtype=coefficients.dtype), -1)
    companion[0, :] = -coefficients[1:] / coefficients[0]
    return np.linalg.eigvals(companion)


def _generic_candidates(source_radius: float, rotated: np.ndarray, *,
                        root_tolerance: float) -> list[np.ndarray]:
    """Candidates in the source frame from the roots of the quartic in
    ``u = 1 / |x|**2``."""
    a11 = float(rotated[0, 0])
    a12 = float(rotated[0, 1])
    a22 = float(rotated[1, 1])
    candidates: list[np.ndarray] = []
    for raw_root in _companion_roots(image_quartic_coefficients(source_radius,
                                                                rotated)):
        if raw_root.real <= 0.0:
            continue
        if abs(raw_root.imag) > root_tolerance * (1.0 + abs(raw_root.real)):
            continue
        root = float(raw_root.real)
        denominator = (a11 - root) * (a22 - root) - a12 * a12
        if abs(denominator) <= 1e-12 * (1.0 + abs(a11 * a22) + root**2):
            continue
        candidates.append(
            np.array([source_radius * (a22 - root) / denominator,
                      -source_radius * a12 / denominator], dtype=float))
    return candidates


def find_images_quartic(source: np.ndarray, matrix: np.ndarray, *,
                        root_tolerance: float = 3e-7,
                        residual_tolerance: float = 3e-8,
                        duplicate_tolerance: float = 3e-7,
                        axis_tolerance: float = 5e-11
                        ) -> list[np.ndarray]:
    """
    All distinct finite real images, from the exact quartic reduction.

    The quartic is solved in a frame aligned with the source, so that
    the general source vector needs no multistart search.  Axis-aligned
    shear is handled separately because the off-axis pair then lies at
    a removable singularity of the generic reconstruction formula.  A
    short deterministic Newton polish removes floating-point error from
    the algebraic roots; it is not an image search.

    Parameters
    ----------
    source : np.ndarray
        Shape (2,), source position.
    matrix : np.ndarray
        Shape (2, 2), symmetric macro matrix (see `macro_matrix`).
    root_tolerance : float
        Relative imaginary part below which a quartic root counts as
        real, and the negative slack allowed on a squared quantity
        before its square root is clipped to zero.
    residual_tolerance : float
        Largest ``|lens_residual|`` accepted for an image.
    duplicate_tolerance : float
        Relative separation below which two images are the same image.
    axis_tolerance : float
        Relative size of the off-diagonal element below which the
        rotated macro matrix counts as diagonal, and the relative
        eigenvalue splitting below which a centered-source macro
        matrix counts as degenerate.

    Returns
    -------
    list of np.ndarray
        Image positions, each of shape (2,), sorted by increasing
        Fermat delay.  Two images for a source outside the astroid
        caustic, four inside.

    Raises
    ------
    ValueError
        If the shapes are wrong or ``matrix`` is not symmetric.
    LensDomainError
        If the geometry is outside the supported domain (macro saddle,
        or an Einstein ring at zero source and zero shear), or if the
        solved image set violates the Morse index theorem (an image
        census defect; see `_check_image_census`).

    Notes
    -----
    Near a fold the quartic has a double root, so the returned
    positions carry only ``sqrt(eps) ~ 1.5e-8`` there; the
    corresponding `delay` values are unaffected.  The default
    tolerances are the values the reference implementation was
    validated with.
    """
    source = np.asarray(source, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if source.shape != (2,) or matrix.shape != (2, 2):
        raise ValueError(
            f'Cannot find images for source of shape {source.shape} '
            f'and matrix of shape {matrix.shape}: they must have '
            f'shapes (2,) and (2, 2).')
    if not np.allclose(matrix, matrix.T, atol=1e-13, rtol=0.0):
        raise ValueError(
            f'Cannot find images with a non-symmetric macro matrix '
            f'{matrix.tolist()}: the Fermat Hessian is symmetric by '
            f'construction.')

    source_radius, basis = _source_frame(source)
    if source_radius <= 1e-14:
        candidates = _centered_source_images(
            matrix, degeneracy_tolerance=axis_tolerance)
    else:
        rotated = basis.T @ matrix @ basis
        off_diagonal_scale = max(abs(rotated[0, 0]), abs(rotated[1, 1]),
                                 1.0)
        if abs(rotated[0, 1]) <= axis_tolerance * off_diagonal_scale:
            rotated_candidates = _axial_candidates(
                source_radius, float(rotated[0, 0]), float(rotated[1, 1]),
                root_tolerance=root_tolerance)
        else:
            rotated_candidates = _generic_candidates(
                source_radius, rotated, root_tolerance=root_tolerance)
        candidates = [basis @ candidate
                      for candidate in rotated_candidates]

    images: list[np.ndarray] = []
    for candidate in candidates:
        image = _accept_candidate(candidate, source, matrix,
                                  residual_tolerance=residual_tolerance)
        if image is None:
            continue
        scale = 1.0 + float(np.linalg.norm(image))
        if all(np.linalg.norm(image - old) > duplicate_tolerance * scale
               for old in images):
            images.append(image)

    images.sort(key=lambda image: delay(image, source, matrix))
    _check_image_census(images, matrix)
    return images


def _accept_candidate(candidate: np.ndarray, source: np.ndarray,
                      matrix: np.ndarray, *, residual_tolerance: float
                      ) -> np.ndarray | None:
    """Polish one algebraic candidate; return None if it is not an
    image."""
    if (not np.all(np.isfinite(candidate))
            or np.linalg.norm(candidate) < 1e-11):
        return None
    polished = _newton_polish(candidate, source, matrix)
    residual = float(np.linalg.norm(
        lens_residual(polished, source, matrix)))
    if residual > residual_tolerance:
        # Near a multiple root the unpolished algebraic position can be
        # superior to a poorly conditioned Newton correction.
        raw_residual = float(np.linalg.norm(
            lens_residual(candidate, source, matrix)))
        if raw_residual < residual:
            polished, residual = candidate, raw_residual
    if residual > residual_tolerance:
        return None
    return polished


def find_images(source: np.ndarray,
                matrix: np.ndarray) -> list[np.ndarray]:
    """
    Production image finder: exact quartic reduction plus polishing.

    Parameters
    ----------
    source : np.ndarray
        Shape (2,), source position.
    matrix : np.ndarray
        Shape (2, 2), symmetric macro matrix (see `macro_matrix`).

    Returns
    -------
    list of np.ndarray
        Image positions sorted by increasing Fermat delay.  See
        `find_images_quartic`, of which this is the fixed-tolerance
        alias.
    """
    return find_images_quartic(source, matrix)


def _saddle_metric(image: np.ndarray,
                   matrix: np.ndarray) -> tuple[float, float, float]:
    """Inverse Hessian in the local radial/tangential frame, scaled by
    ``u = 1 / |x|**2``."""
    radius = float(np.linalg.norm(image))
    radial = np.asarray(image, dtype=float) / radius
    tangential = np.array([-radial[1], radial[0]])
    basis = np.column_stack([radial, tangential])
    projected = basis.T @ hessian(image, matrix) @ basis
    # Refuse EXACTLY the crash class, nothing more: the channel layer
    # deliberately consumes huge near-singular metrics here (an on-cusp
    # merged image at det ~ 2*eps in its sweep) and suppresses the
    # divergent stationary-phase target with the F008/SACR-C switch, so
    # ANY determinant threshold amputates that contract.  Only an
    # exactly-singular projected Hessian -- which raised a raw
    # ``numpy.linalg.LinAlgError`` in production and killed the sampler
    # (FINDINGS F015) -- becomes the named refusal; the principled
    # near-fold accuracy limit belongs to the fold/cusp Airy program.
    try:
        inverse = np.linalg.inv(projected)
    except np.linalg.LinAlgError as exc:
        raise LensDomainError(
            'Fold-degenerate image: the projected Fermat Hessian at '
            f'image ({image[0]!r}, {image[1]!r}) is exactly singular; '
            'the stationary-phase kernel of the geometric branch is '
            'invalid at a merged image pair, so this configuration is '
            'refused by name rather than crashed on (raw numpy '
            'LinAlgError). The unresolved near-caustic corner is owned '
            'by the planned fold/cusp uniform (Airy) asymptotics.'
            ) from exc
    scale = 1.0 / radius**2
    return (scale * float(inverse[0, 0]), scale * float(inverse[0, 1]),
            scale * float(inverse[1, 1]))


def _c1_polynomial(prr: float, prt: float, ptt: float) -> float:
    """First subleading saddle coefficient."""
    return (10*prr**3 - 12*prr*prr*ptt - 9*prr*prr - 48*prr*prt*prt
            + 18*prr*ptt*ptt + 18*prr*ptt + 72*prt*prt*ptt
            + 36*prt*prt - 9*ptt*ptt) / 12.0


def _c2_polynomial(prr: float, prt: float, ptt: float) -> float:
    """Second subleading saddle coefficient."""
    poly = (
        1540*prr**6 - 1680*prr**5*ptt - 3780*prr**5
        - 16800*prr**4*prt**2 + 2520*prr**4*ptt**2
        + 5040*prr**4*ptt + 2961*prr**4
        + 40320*prr**3*prt**2*ptt + 40320*prr**3*prt**2
        - 3600*prr**3*ptt**3 - 8280*prr**3*ptt**2
        - 5364*prr**3*ptt - 720*prr**3
        + 40320*prr**2*prt**4 - 64800*prr**2*prt**2*ptt**2
        - 99360*prr**2*prt**2*ptt - 32184*prr**2*prt**2
        + 3780*prr**2*ptt**4 + 10800*prr**2*ptt**3
        + 9126*prr**2*ptt**2 + 2160*prr**2*ptt
        - 86400*prr*prt**4*ptt - 66240*prr*prt**4
        + 60480*prr*prt**2*ptt**3 + 129600*prr*prt**2*ptt**2
        + 73008*prr*prt**2*ptt + 8640*prr*prt**2
        - 3780*prr*ptt**4 - 5940*prr*ptt**3 - 2160*prr*ptt**2
        - 11520*prt**6 + 60480*prt**4*ptt**2 + 86400*prt**4*ptt
        + 24336*prt**4 - 30240*prt**2*ptt**3 - 35640*prt**2*ptt**2
        - 8640*prt**2*ptt + 945*ptt**4 + 720*ptt**3)
    return -poly / 288.0


def saddle_coefficients(image: np.ndarray,
                        matrix: np.ndarray) -> tuple[float, float]:
    """
    Analytic ``C1``, ``C2`` coefficients of the image expansion.

    Parameters
    ----------
    image : np.ndarray
        Shape (2,), image position.
    matrix : np.ndarray
        Shape (2, 2), the macro matrix.

    Returns
    -------
    c1, c2 : float
        Coefficients of the stationary-phase expansion
        ``1 + 1j * C1 / w + C2 / w**2`` (see `image_kernel`).  They are
        built from the inverse Hessian in the frame aligned with the
        image's radius vector, and diverge at a critical point along
        with the magnification.
    """
    metric = _saddle_metric(image, matrix)
    return _c1_polynomial(*metric), _c2_polynomial(*metric)


def image_kernel(w_dimensionless, image: np.ndarray,
                 matrix: np.ndarray) -> np.ndarray:
    """
    Carrier-free image kernel through relative order ``w**-2``.

    Parameters
    ----------
    w_dimensionless : float or np.ndarray
        Dimensionless frequency ``w`` (see the module docstring); the
        carrier ``exp(1j * w * tau)`` is *not* included.
    image : np.ndarray
        Shape (2,), image position.
    matrix : np.ndarray
        Shape (2, 2), the macro matrix.

    Returns
    -------
    np.ndarray
        ``sqrt|mu| * exp(-0.5j * pi * n) * (1 + 1j * C1 / w + C2 / w**2)``
        broadcast over ``w_dimensionless``.  This is the ``w -> inf``
        asymptote of the exact channel kernel; it diverges at a
        critical point (see the module docstring) and is not valid for
        an image that is part of an unresolved cluster.
    """
    w_dimensionless = np.asarray(w_dimensionless, dtype=float)
    c1_coefficient, c2_coefficient = saddle_coefficients(image, matrix)
    return (np.sqrt(abs(magnification(image, matrix)))
            * np.exp(-0.5j * np.pi * morse_index(image, matrix))
            * (1.0 + 1j * c1_coefficient / w_dimensionless
               + c2_coefficient / w_dimensionless**2))


def critical_point(gamma: float, theta: float, beta: float = 0.0,
                   kappa: float = 0.0, branch: int = 1) -> CriticalPoint:
    """
    Critical point at a given polar angle, and its local frame.

    The critical radius follows from the zeros of the Fermat-Hessian
    determinant in ``v = 1 / |x|**2`` (equivalently the mass-sheet
    rescaling ``x' = sqrt(lam) * x`` with ``lam = 1 - kappa`` and
    effective shear ``gamma / lam``):

        v(theta') = gamma * cos(2 theta')
                    +- sqrt(gamma**2 cos(2 theta')**2 + lam**2 - gamma**2),

    ``theta' = theta - beta``.  For positive parity ``lam > abs(gamma)``
    only the ``+`` branch is a positive radius, and it exists at every
    angle -- the classical single 4-cusp astroid.  For a macro saddle
    ``lam < abs(gamma)`` both ``+-`` branches are positive, but only
    inside the two angular wedges ``|sin 2 theta'| <= lam / abs(gamma)``
    about the negative-eigenvalue axis (``theta' ~ 0, pi``); the two
    branches trace the two edges of each of the two 3-cusp deltoid
    lobes.

    Parameters
    ----------
    gamma : float
        External shear magnitude.
    theta : float
        Polar angle in the lens plane, radians.
    beta : float
        External shear orientation, radians.
    kappa : float
        External convergence.
    branch : int
        Sign of the square-root branch of ``v(theta')``: ``+1`` (the
        default, the only real branch at positive parity) or ``-1``.
        Ignored at positive parity, where only ``+1`` is a valid
        radius.

    Returns
    -------
    CriticalPoint
        The lens-plane critical point at ``theta``, the caustic point
        it maps to, the local Hessian eigenframe, and the nonzero
        Hessian eigenvalue.

    Raises
    ------
    LensDomainError
        If ``1 - kappa <= 0`` (over-critical / Type III) or
        ``abs(gamma) == 1 - kappa`` (the parity boundary); or, for a
        macro saddle, if ``theta`` lies outside the critical wedge or
        the selected branch gives a non-positive radius.
    """
    lam = 1.0 - float(kappa)
    if lam <= 0.0:
        raise LensDomainError(
            f'Cannot locate a critical point for (kappa, gamma) = '
            f'({kappa}, {gamma}): 1 - kappa = {lam} <= 0 (over-critical '
            f'/ Type III). The mass-sheet reduction is not real; such '
            f'configurations are out of scope.')
    if abs(gamma) == lam:
        raise LensDomainError(
            f'Cannot locate a critical point for (kappa, gamma) = '
            f'({kappa}, {gamma}): |gamma| == 1 - kappa = {lam} exactly '
            f'(det A = 0, the parity boundary between the positive-'
            f'parity and macro-saddle domains); this boundary is a '
            f'named refusal.')
    effective_gamma = gamma / lam
    phase = theta - beta
    if abs(gamma) < lam:
        # Positive parity: single astroid, only the + branch is real.
        # Byte-identical to the frozen positive-parity implementation.
        effective_u = (effective_gamma * np.cos(2.0 * phase)
                       + np.sqrt(1.0 - effective_gamma**2
                                 * np.sin(2.0 * phase)**2))
    else:
        # Macro saddle: two 3-cusp lobes; both +- branches are real
        # inside the critical wedge |sin 2 theta'| <= lam / |gamma|.
        discriminant = (1.0 - effective_gamma**2
                        * np.sin(2.0 * phase)**2)
        if discriminant < -1e-12:
            raise LensDomainError(
                f'Cannot locate a macro-saddle critical point at theta '
                f'= {theta} for (kappa, gamma) = ({kappa}, {gamma}): '
                f'the polar angle lies outside the critical wedge '
                f'|sin 2(theta - beta)| <= (1 - kappa) / |gamma|.')
        discriminant = max(discriminant, 0.0)
        effective_u = (effective_gamma * np.cos(2.0 * phase)
                       + branch * np.sqrt(discriminant))
        if effective_u <= 0.0:
            raise LensDomainError(
                f'Cannot locate a macro-saddle critical point at theta '
                f'= {theta} for (kappa, gamma) = ({kappa}, {gamma}): '
                f'branch {branch} gives a non-positive radial coordinate '
                f'u = {effective_u}. The two critical lobes lie on the '
                f'negative-eigenvalue axis (theta - beta near 0 or pi).')
    # x' has radius 1 / sqrt(effective_u); physical x = x' / sqrt(lam).
    radius = 1.0 / np.sqrt(lam * effective_u)
    image = radius * np.array([np.cos(theta), np.sin(theta)])
    matrix = macro_matrix(gamma, beta, kappa)
    source = matrix @ image - image / radius**2
    values, vectors = np.linalg.eigh(hessian(image, matrix))
    soft_index = int(np.argmin(np.abs(values)))
    hard_index = 1 - soft_index
    hard_axis = vectors[:, hard_index]
    soft_axis = vectors[:, soft_index]
    if np.linalg.det(np.column_stack([hard_axis, soft_axis])) < 0.0:
        soft_axis = -soft_axis
    return CriticalPoint(image, source, hard_axis, soft_axis,
                         float(values[hard_index]))


#: Number of coarse-scan seed nodes per branch/lobe for the Newton polish in
#: `nearest_caustic_point`.  Replaces the dense ``n_grid`` sweep as the Newton
#: seed; ``n_grid`` still caps this from below for small requests, so its
#: documented role (an upper bound on the coarse-scan density) is preserved.
_NEAREST_CAUSTIC_SEED_NODES = 32

#: Number of best seed cells promoted to Newton starts per branch/lobe.
_NEAREST_CAUSTIC_NEWTON_STARTS = 2

#: Maximum Newton iterations for the angular squared-distance polish.
_NEAREST_CAUSTIC_NEWTON_MAXITER = 20

#: Newton convergence: a full step ``|g'/g''|`` below this (radians) accepts
#: the stationary point.
_NEAREST_CAUSTIC_NEWTON_XTOL = 1e-13

#: Newton convergence: ``|g'|`` below this tiny floor also accepts the root
#: (guards the near-cusp regime where ``g''`` is large and the step is tiny
#: even before ``g'`` is fully quenched).
_NEAREST_CAUSTIC_GPRIME_FLOOR = 1e-15

#: Discriminant floor.  Seeds (or Newton iterates) whose caustic discriminant
#: ``1 - (gamma / lam)**2 sin(2 (theta - beta))**2`` falls below this are
#: routed to the bounded-Brent fallback: at the clamp the analytic angular
#: derivative is one-sided (the ``1 / sqrt(disc)`` factors diverge).  Inert at
#: positive parity, where the discriminant is bounded away from zero.
_NEAREST_CAUSTIC_DISC_FLOOR = 1e-9


@numba.njit(cache=True, fastmath=False)
def _caustic_source(theta: float, gamma: float, beta: float,
                    kappa: float, branch: float) -> np.ndarray:
    """
    Caustic (source-plane) point of the critical curve at ``theta``.

    Reproduces exactly the arithmetic of `critical_point` that yields
    its ``source`` attribute, skipping the Hessian eigendecomposition
    and eigenframe construction that the distance search does not need.
    Compiled with ``fastmath=False`` so the elementary functions match
    the numpy reference to within one unit in the last place.

    Parameters
    ----------
    theta : float
        Polar angle on the critical curve, radians.
    gamma, beta, kappa : float
        External shear magnitude, shear orientation (radians), and
        convergence.  The domain condition ``1 - kappa > 0`` must
        already hold; this helper does not guard it.
    branch : float
        Sign (``+1.0`` or ``-1.0``) of the square-root branch of the
        critical radius.  ``+1.0`` is the only real branch at positive
        parity; a macro saddle uses both.  The discriminant is clamped
        at zero so the wedge endpoints (where the two branches meet) do
        not produce ``nan`` from float64 rounding; at positive parity
        the discriminant is strictly positive so the clamp is inert and
        the result is byte-identical to the frozen positive-parity
        evaluation.

    Returns
    -------
    np.ndarray
        Shape (2,), the caustic point ``macro_matrix @ x - x / |x|**2``
        at the critical point ``x`` for the given ``theta`` and branch.
    """
    lam = 1.0 - kappa
    effective_gamma = gamma / lam
    phase = theta - beta
    discriminant = 1.0 - effective_gamma**2 * np.sin(2.0 * phase)**2
    if discriminant < 0.0:
        discriminant = 0.0
    effective_u = (effective_gamma * np.cos(2.0 * phase)
                   + branch * np.sqrt(discriminant))
    radius = 1.0 / np.sqrt(lam * effective_u)
    image_x = radius * np.cos(theta)
    image_y = radius * np.sin(theta)
    cos2b = np.cos(2.0 * beta)
    sin2b = np.sin(2.0 * beta)
    # macro_matrix = (1 - kappa) * I - gamma * [[cos2b, sin2b],
    #                                           [sin2b, -cos2b]].
    m00 = (1.0 - kappa) - gamma * cos2b
    m01 = -gamma * sin2b
    m11 = (1.0 - kappa) + gamma * cos2b
    caustic = np.empty(2)
    caustic[0] = m00 * image_x + m01 * image_y - image_x / radius**2
    caustic[1] = m01 * image_x + m11 * image_y - image_y / radius**2
    return caustic


@numba.njit(cache=True, fastmath=False)
def _coarse_squared_distances(grid: np.ndarray, gamma: float, beta: float,
                              kappa: float, source: np.ndarray,
                              branch: float) -> np.ndarray:
    """
    Squared source-plane distance to the caustic at each grid angle.

    A single compiled sweep replacing the per-angle Python-level
    `critical_point` calls of the coarse scan.  Uses the same per-angle
    arithmetic (`_caustic_source`) and the same reduction the bounded
    refinement objective uses, so the ``argsort`` cell selection matches
    the reference search.

    Parameters
    ----------
    grid : np.ndarray
        Shape (n_grid,), polar angles of the coarse scan.
    gamma, beta, kappa : float
        Lens parameters (see `_caustic_source`).
    source : np.ndarray
        Shape (2,), source position.
    branch : float
        Astroid sign branch (+1.0 or -1.0) selecting the caustic lobe;
        +1.0 reproduces the positive-parity caustic byte-for-byte.

    Returns
    -------
    np.ndarray
        Shape (n_grid,), squared distances ``|caustic(theta) - source|**2``.
    """
    distances = np.empty(grid.shape[0])
    for index in range(grid.shape[0]):
        caustic = _caustic_source(grid[index], gamma, beta, kappa, branch)
        offset_x = caustic[0] - source[0]
        offset_y = caustic[1] - source[1]
        distances[index] = offset_x * offset_x + offset_y * offset_y
    return distances


@numba.njit(cache=True, fastmath=False)
def _squared_distance_derivatives(theta: float, gamma: float, beta: float,
                                  kappa: float, branch: float,
                                  source: np.ndarray):
    """
    Value and first two theta-derivatives of the angular squared distance.

    Returns ``(g, g', g'', discriminant)`` where ``g(theta) =
    |caustic(theta) - source|**2`` for the caustic of `_caustic_source`
    (same closed form, same branch, same discriminant clamp).  The
    derivatives are analytic: differentiating

        caustic = r * (M @ n) - (1 / r) * n,

    with ``n = (cos theta, sin theta)``, ``M`` the macro matrix, and the
    critical radius ``r = 1 / sqrt(lam * u)`` following the branch-selected
    ``u = eff_gamma cos 2(theta - beta) + branch sqrt(disc)``.  Then
    ``g' = 2 (caustic - source) . caustic'`` and
    ``g'' = 2 (caustic' . caustic' + (caustic - source) . caustic'')``.

    The returned discriminant is UNCLAMPED so a caller can gate on it: the
    ``1 / sqrt(disc)`` factors in ``u'`` and ``u''`` diverge as the wedge
    boundary (``disc -> 0``) is approached, where the analytic derivative is
    one-sided and Newton must defer to the bounded-Brent fallback.  At
    positive parity ``disc >= 1 - eff_gamma**2 > 0`` so the guarded branch is
    inert.  Compiled ``fastmath=False`` to match the numpy reference.
    """
    lam = 1.0 - kappa
    eff_gamma = gamma / lam
    phase = theta - beta
    two_phase = 2.0 * phase
    cos_p = np.cos(two_phase)
    sin_p = np.sin(two_phase)
    disc = 1.0 - eff_gamma * eff_gamma * sin_p * sin_p
    disc_clamped = disc if disc > 0.0 else 0.0
    sqrt_disc = np.sqrt(disc_clamped)

    eff_u = eff_gamma * cos_p + branch * sqrt_disc
    if sqrt_disc > 0.0:
        d_sqrt_disc = -2.0 * eff_gamma * eff_gamma * sin_p * cos_p / sqrt_disc
        eff_u1 = -2.0 * eff_gamma * sin_p + branch * d_sqrt_disc
        eff_u2 = (-4.0 * eff_gamma * cos_p
                  - 4.0 * branch * eff_gamma * eff_gamma
                  * (cos_p * cos_p - sin_p * sin_p) / sqrt_disc
                  - 4.0 * branch * eff_gamma**4 * sin_p * sin_p * cos_p * cos_p
                  / (sqrt_disc * sqrt_disc * sqrt_disc))
    else:
        # Clamp boundary: one-sided derivative; the caller routes here to the
        # bounded-Brent fallback, but keep finite values to avoid nan/inf.
        eff_u1 = -2.0 * eff_gamma * sin_p
        eff_u2 = -4.0 * eff_gamma * cos_p

    # s = lam * u, with r = s**-0.5 and q = 1 / r = s**0.5.
    s0 = lam * eff_u
    s1 = lam * eff_u1
    s2 = lam * eff_u2
    r = 1.0 / np.sqrt(s0)
    r1 = -0.5 * s0**(-1.5) * s1
    r2 = -0.5 * s0**(-1.5) * s2 + 0.75 * s0**(-2.5) * s1 * s1
    q = np.sqrt(s0)
    q1 = 0.5 * s1 / q
    q2 = 0.5 * s2 / q - 0.25 * s1 * s1 * s0**(-1.5)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # n = (cos, sin); tangent t = n' = (-sin, cos); n'' = -n, t' = -n.
    nx = cos_t
    ny = sin_t
    tx = -sin_t
    ty = cos_t

    cos2b = np.cos(2.0 * beta)
    sin2b = np.sin(2.0 * beta)
    m00 = lam - gamma * cos2b
    m01 = -gamma * sin2b
    m11 = lam + gamma * cos2b
    mn_x = m00 * nx + m01 * ny
    mn_y = m01 * nx + m11 * ny
    mt_x = m00 * tx + m01 * ty
    mt_y = m01 * tx + m11 * ty

    # caustic = r * (M @ n) - q * n.
    cx = r * mn_x - q * nx
    cy = r * mn_y - q * ny
    # caustic' = r' M@n + r M@t - q' n - q t.
    c1x = r1 * mn_x + r * mt_x - q1 * nx - q * tx
    c1y = r1 * mn_y + r * mt_y - q1 * ny - q * ty
    # caustic'' = r'' M@n + 2 r' M@t - r M@n - q'' n - 2 q' t + q n.
    c2x = (r2 * mn_x + 2.0 * r1 * mt_x - r * mn_x
           - q2 * nx - 2.0 * q1 * tx + q * nx)
    c2y = (r2 * mn_y + 2.0 * r1 * mt_y - r * mn_y
           - q2 * ny - 2.0 * q1 * ty + q * ny)

    dx = cx - source[0]
    dy = cy - source[1]
    g = dx * dx + dy * dy
    g1 = 2.0 * (dx * c1x + dy * c1y)
    g2 = 2.0 * (c1x * c1x + c1y * c1y + dx * c2x + dy * c2y)
    return g, g1, g2, disc


@numba.njit(cache=True, fastmath=False)
def _newton_caustic_cell(theta0: float, gamma: float, beta: float,
                         kappa: float, branch: float, source: np.ndarray,
                         theta_lo: float, theta_hi: float, clamp: bool):
    """
    Newton polish of the stationarity condition ``g'(theta) = 0`` from a seed.

    Returns ``(theta, g, ok)``.  ``ok`` is ``True`` only for an interior
    minimum reached with ``g'' > 0`` and a converged step; it is ``False`` --
    signalling the caller to fall back to bounded Brent on that one cell --
    when the discriminant drops below `_NEAREST_CAUSTIC_DISC_FLOOR` (one-sided
    derivative near the clamp), when ``g'' <= 0`` (not a local minimum), when a
    step would leave the lobe wedge ``[theta_lo, theta_hi]`` (only enforced for
    ``clamp``; the astroid path is periodic and unclamped), or when the
    iteration cap is hit.  Newton therefore never migrates out of its wedge.
    """
    theta = theta0
    g = 0.0
    for _ in range(_NEAREST_CAUSTIC_NEWTON_MAXITER):
        g, g1, g2, disc = _squared_distance_derivatives(
            theta, gamma, beta, kappa, branch, source)
        if disc < _NEAREST_CAUSTIC_DISC_FLOOR:
            return theta, g, False
        if not (g2 > 0.0):
            return theta, g, False
        step = g1 / g2
        if abs(step) < _NEAREST_CAUSTIC_NEWTON_XTOL \
                or abs(g1) < _NEAREST_CAUSTIC_GPRIME_FLOOR:
            return theta, g, True
        theta_new = theta - step
        if clamp and (theta_new < theta_lo or theta_new > theta_hi):
            # The step wants to leave the wedge (the minimum is at or beyond
            # the boundary / a deltoid cusp); defer to the bounded fallback.
            return theta, g, False
        theta = theta_new
    return theta, g, False


def nearest_caustic_point(gamma: float, beta: float, source: np.ndarray,
                          *, kappa: float = 0.0, n_grid: int = 256
                          ) -> NearestCausticPoint:
    """
    Caustic point closest to a source, by search along the critical
    curve.

    A cheap coarse seed scan (`_NEAREST_CAUSTIC_SEED_NODES` polar angles,
    capped from below by ``n_grid``) locates the best few cells, and each
    is polished by a one-dimensional analytic-Newton iteration on the
    stationarity condition ``g'(theta) = 0`` of the angular squared
    distance ``g(theta) = |caustic(theta) - source|**2``, with ``g'`` and
    ``g''`` from the closed form (`_squared_distance_derivatives`).  When
    Newton cannot certify an interior minimum (``g'' <= 0``, a step
    leaving the lobe wedge, the discriminant clamp, or the iteration cap)
    that single cell falls back to a bounded `scipy.optimize.minimize_scalar`.

    Positive parity (``abs(gamma) < 1 - kappa``): a single 4-cusp astroid;
    the seed scan spans the full circle and the ``+`` square-root branch
    only, and Newton is periodic (unclamped) so all four cusps remain
    reachable.

    Macro saddle (``0 < 1 - kappa < abs(gamma)``): the critical curve is
    two 3-cusp deltoid lobes confined to the two angular wedges
    ``|sin 2(theta - beta)| <= (1 - kappa) / abs(gamma)`` about the
    negative-eigenvalue axis (``theta - beta`` near ``0`` and ``pi``).
    Each wedge is scanned for both square-root branches (``+-``), Newton is
    clamped to its wedge, and the global minimum over the two lobes and two
    branches is returned, so both deltoid lobes remain reachable.

    The frequency-independent distance search runs through the compiled
    source-only helper `_caustic_source` and its analytic derivatives (the
    eigenframe and eigenvalue are not needed to locate the closest
    caustic); the returned local frame and eigenvalue are then built from a
    single `critical_point` call at the winning angle and branch, so those
    fields are identical to a search that used `critical_point` throughout.

    Parameters
    ----------
    gamma : float
        External shear magnitude.
    beta : float
        External shear orientation, radians.
    source : np.ndarray
        Shape (2,), source position.
    kappa : float
        External convergence.
    n_grid : int
        Upper bound on the number of polar angles in the coarse seed scan
        (per wedge and branch for a macro saddle); the seed uses
        ``min(n_grid, _NEAREST_CAUSTIC_SEED_NODES)`` nodes.

    Returns
    -------
    NearestCausticPoint
        The closest caustic point, its local frame, and the (unsigned)
        source-plane distance to it.  Unoccupied image labels are
        placed at its lens-plane critical point.

    Raises
    ------
    LensDomainError
        If ``1 - kappa <= 0`` (over-critical / Type III) or
        ``abs(gamma) == 1 - kappa`` (the parity boundary).
    """
    lam = 1.0 - float(kappa)
    if lam <= 0.0:
        raise LensDomainError(
            f'Cannot locate a critical point for (kappa, gamma) = '
            f'({kappa}, {gamma}): 1 - kappa = {lam} <= 0 (over-critical '
            f'/ Type III). Such configurations are out of scope.')
    if abs(gamma) == lam:
        raise LensDomainError(
            f'Cannot locate a critical point for (kappa, gamma) = '
            f'({kappa}, {gamma}): |gamma| == 1 - kappa = {lam} exactly '
            f'(det A = 0, the parity boundary); this boundary is a '
            f'named refusal.')

    source = np.asarray(source, dtype=float)
    n_seed = min(int(n_grid), _NEAREST_CAUSTIC_SEED_NODES)
    n_starts = min(_NEAREST_CAUSTIC_NEWTON_STARTS, n_seed)

    if abs(gamma) < lam:
        # Positive parity: a single 4-cusp astroid over the full circle,
        # the ``+`` branch only.  Newton is periodic (unclamped).
        grid = np.linspace(0.0, 2.0 * np.pi, n_seed, endpoint=False)
        step = 2.0 * np.pi / n_seed

        def squared_distance(theta) -> float:
            caustic = _caustic_source(float(theta) % (2.0 * np.pi),
                                      gamma, beta, kappa, 1.0)
            return float(np.sum((caustic - source)**2))

        coarse = _coarse_squared_distances(grid, gamma, beta, kappa,
                                           source, 1.0)
        best_fun = np.inf
        best_theta = 0.0
        for index in np.argsort(coarse)[:n_starts]:
            center = grid[index]
            theta_c, fun_c, ok = _newton_caustic_cell(
                center, gamma, beta, kappa, 1.0, source, 0.0, 0.0, False)
            if not ok:
                refined = minimize_scalar(
                    squared_distance,
                    bounds=(center - step, center + step),
                    method='bounded',
                    options={'xatol': 1e-12})
                theta_c = float(refined.x)
                fun_c = float(refined.fun)
            if fun_c < best_fun:
                best_fun = fun_c
                best_theta = theta_c
        theta = float(best_theta % (2.0 * np.pi))
        return NearestCausticPoint(
            theta,
            *critical_point(gamma, theta, beta, kappa),
            distance=float(np.sqrt(best_fun)))

    # Macro saddle: two 3-cusp deltoid lobes, each confined to a wedge of
    # half-width theta_max about the negative-eigenvalue axis, and each
    # traced by both square-root branches.
    theta_max = 0.5 * np.arcsin(lam / abs(gamma))
    best_fun = np.inf
    best_theta = 0.0
    best_branch = 1
    for center in (beta, beta + np.pi):
        lower_wedge = center - theta_max
        upper_wedge = center + theta_max
        wedge = np.linspace(lower_wedge, upper_wedge, n_seed)
        step = 2.0 * theta_max / (n_seed - 1)
        for branch in (1.0, -1.0):

            def squared_distance(theta, branch=branch) -> float:
                caustic = _caustic_source(float(theta), gamma, beta,
                                          kappa, branch)
                return float(np.sum((caustic - source)**2))

            coarse = _coarse_squared_distances(wedge, gamma, beta, kappa,
                                               source, branch)
            for index in np.argsort(coarse)[:n_starts]:
                seed = wedge[index]
                theta_c, fun_c, ok = _newton_caustic_cell(
                    seed, gamma, beta, kappa, branch, source,
                    lower_wedge, upper_wedge, True)
                if not ok:
                    lower = max(wedge[index] - step, lower_wedge)
                    upper = min(wedge[index] + step, upper_wedge)
                    refined = minimize_scalar(
                        squared_distance,
                        bounds=(lower, upper),
                        method='bounded',
                        options={'xatol': 1e-12})
                    theta_c = float(refined.x)
                    fun_c = float(refined.fun)
                if fun_c < best_fun:
                    best_fun = fun_c
                    best_theta = theta_c
                    best_branch = int(branch)

    theta = best_theta % (2.0 * np.pi)
    return NearestCausticPoint(
        theta,
        *critical_point(gamma, best_theta, beta, kappa, best_branch),
        distance=float(np.sqrt(best_fun)))


def _critical_branch_source(
        lens_theta: float, gamma: float, kappa: float, branch: int,
) -> np.ndarray | None:
    """Caustic source on one valid critical-curve branch, if it exists."""
    lam = 1.0 - float(kappa)
    effective_gamma = float(gamma) / lam
    discriminant = (
        1.0 - effective_gamma**2 * np.sin(2.0 * float(lens_theta))**2
    )
    if abs(gamma) < lam:
        if branch != 1:
            return None
    elif discriminant < -1.0e-12:
        return None
    discriminant = max(float(discriminant), 0.0)
    effective_u = (
        effective_gamma * np.cos(2.0 * float(lens_theta))
        + branch * np.sqrt(discriminant)
    )
    if effective_u <= 0.0:
        return None
    source = _caustic_source(
        float(lens_theta), float(gamma), 0.0, float(kappa), float(branch))
    return source if np.all(np.isfinite(source)) else None


def _caustic_ray_intersections(
        gamma: float, theta: float, kappa: float, n_sample: int,
) -> list[np.ndarray]:
    """Critical-curve images lying on one outward source-plane ray."""
    target = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    lens_thetas = np.linspace(0.0, 2.0 * np.pi, n_sample + 1)
    branches = (1,) if abs(gamma) < 1.0 - kappa else (1, -1)
    intersections: list[np.ndarray] = []

    for branch in branches:
        previous: tuple[float, float] | None = None
        for lens_theta in lens_thetas:
            source = _critical_branch_source(
                float(lens_theta), gamma, kappa, branch)
            if source is None:
                previous = None
                continue
            cross = float(target[0] * source[1] - target[1] * source[0])
            dot = float(target @ source)
            cross_tolerance = 64.0 * np.finfo(float).eps * max(
                1.0, float(np.linalg.norm(source)))
            if dot > 0.0 and abs(cross) <= cross_tolerance:
                intersections.append(source)

            if previous is not None and previous[1] * cross < 0.0:
                lower = previous[0]

                def ray_cross(candidate_theta: float) -> float:
                    candidate = _critical_branch_source(
                        candidate_theta, gamma, kappa, branch)
                    if candidate is None:
                        raise LensDomainError(
                            'Critical branch vanished inside a bracketed '
                            'caustic-ray intersection.')
                    return float(target[0] * candidate[1]
                                 - target[1] * candidate[0])

                root = brentq(
                    ray_cross, lower, float(lens_theta),
                    xtol=4.0 * np.finfo(float).eps,
                    rtol=4.0 * np.finfo(float).eps,
                )
                root_source = _critical_branch_source(
                    root, gamma, kappa, branch)
                if (root_source is not None
                        and float(target @ root_source) > 0.0):
                    intersections.append(root_source)
            previous = (float(lens_theta), cross)
    return intersections


def r_caustic(gamma: float, theta: float, *, kappa: float = 0.0,
              n_sample: int = 720) -> float:
    """Return the exact directional radius of the source-plane caustic.

    The caustic is generated from the actual critical curve. ``n_sample``
    supplies only brackets in lens-plane angle; every ray intersection is
    refined to float64 precision before its source-plane radius is measured.
    The outermost forward intersection is returned, so positive-parity
    ``rho = |y| / r_caustic(gamma, theta)`` crosses the image-count boundary
    at ``rho = 1``. Macro-saddle rays that miss both deltoid lobes refuse.

    Parameters
    ----------
    gamma : float
        External shear magnitude.
    theta : float
        Source-plane polar angle of the query ray, radians.
    kappa : float, optional
        External convergence (default 0.0).
    n_sample : int, optional
        Number of lens-plane angular brackets. Bracketing density does not set
        the returned radius accuracy; it must be at least 16.

    Returns
    -------
    float
        Outermost caustic radius on the requested source-plane ray.

    Raises
    ------
    LensDomainError
        If the macro geometry is outside the supported domain or the ray does
        not intersect a caustic.
    ValueError
        If ``n_sample`` is smaller than 16.
    """
    lam = 1.0 - float(kappa)
    if lam <= 0.0:
        raise LensDomainError(
            f'Cannot locate a caustic radius for (kappa, gamma) = '
            f'({kappa}, {gamma}): 1 - kappa = {lam} <= 0 (over-critical '
            f'/ Type III); such configurations are out of scope.')
    if abs(gamma) == lam:
        raise LensDomainError(
            f'Cannot locate a caustic radius for (kappa, gamma) = '
            f'({kappa}, {gamma}): |gamma| == 1 - kappa = {lam} exactly '
            f'(det A = 0, the parity boundary); this boundary is a named '
            f'refusal.')
    n_sample = int(n_sample)
    if n_sample < 16:
        raise ValueError(f'n_sample must be at least 16; got {n_sample}.')

    intersections = _caustic_ray_intersections(
        float(gamma), float(theta), float(kappa), n_sample)
    if not intersections:
        raise LensDomainError(
            f'No caustic intersection found on source-plane ray theta={theta} '
            f'for (kappa, gamma) = ({kappa}, {gamma}).')
    return max(float(np.linalg.norm(source)) for source in intersections)


def caustic_derivatives(gamma: float, theta, *, kappa: float = 0.0,
                        branch: int = 1):
    """Analytic first and second theta-derivatives of the caustic curve.

    The Chang--Refsdal caustic is the exact closed-form parametric curve
    ``y_i(theta) = p_i(theta) r(theta) T_i(theta)`` with
    ``T = (cos theta, sin theta)``, ``r = 1 / sqrt(lam u)``,
    ``p_i = (lam -+ gamma) - lam u`` and
    ``u(theta) = e cos 2theta + branch sqrt(1 - e**2 sin**2 2theta)``,
    where ``lam = 1 - kappa`` and ``e = gamma / lam``.  Both derivatives
    are the *symbolic* differentiation of that curve (no finite
    difference or sampled arc) and are vectorised over ``theta``.

    The domain contract mirrors :func:`critical_point` exactly: at
    positive parity (``abs(gamma) < lam``) only the ``+`` root is real,
    so ``branch`` is ignored; the macro saddle (``abs(gamma) > lam``)
    honours ``branch`` and refuses outside the critical wedge, exactly on
    the wedge edge (the deltoid cusp), or where the branch gives a
    non-positive radial coordinate.

    Parameters
    ----------
    gamma : float
        External shear magnitude.
    theta : float or np.ndarray
        Polar angle(s) in the lens plane, radians (already relative to
        the shear axis; curvature is rotation-invariant).
    kappa : float, optional
        External convergence (default 0.0).
    branch : int, optional
        Square-root branch ``+-1`` of ``u`` (default ``+1``); ignored at
        positive parity.

    Returns
    -------
    tuple of np.ndarray
        ``(y_prime, y_double_prime)``, each shaped ``(2,)`` for scalar
        ``theta`` or ``(2, N)`` for an array of ``N`` angles.

    Raises
    ------
    LensDomainError
        If ``1 - kappa <= 0``, ``abs(gamma) == 1 - kappa`` (parity wall),
        or -- for a macro saddle -- ``theta`` lies outside the critical
        wedge, exactly on the wedge edge (the deltoid cusp, where the
        derivatives genuinely diverge), or the selected branch gives a
        non-positive radius.
    """
    gamma = float(gamma)
    lam = 1.0 - float(kappa)
    if lam <= 0.0:
        raise LensDomainError(
            f'Cannot evaluate caustic derivatives for (kappa, gamma) = '
            f'({kappa}, {gamma}): 1 - kappa = {lam} <= 0 (over-critical '
            f'/ Type III); such configurations are out of scope.')
    if abs(gamma) == lam:
        raise LensDomainError(
            f'Cannot evaluate caustic derivatives for (kappa, gamma) = '
            f'({kappa}, {gamma}): |gamma| == 1 - kappa = {lam} exactly '
            f'(det A = 0, the parity boundary); this boundary is a named '
            f'refusal.')
    theta = np.asarray(theta, dtype=float)
    eff = gamma / lam
    positive_parity = abs(gamma) < lam
    b = 1 if positive_parity else int(branch)
    s, c = np.sin(2.0 * theta), np.cos(2.0 * theta)
    c4 = np.cos(4.0 * theta)
    discriminant = 1.0 - eff**2 * s**2
    if not positive_parity and np.any(discriminant < -1e-12):
        raise LensDomainError(
            f'Cannot evaluate macro-saddle caustic derivatives for '
            f'(kappa, gamma) = ({kappa}, {gamma}): a polar angle lies '
            f'outside the critical wedge |sin 2 theta| <= (1 - kappa) / '
            f'|gamma|.')
    discriminant = np.maximum(discriminant, 0.0)
    d_root = np.sqrt(discriminant)
    if not positive_parity and np.any(d_root == 0.0):
        raise LensDomainError(
            f'Cannot evaluate macro-saddle caustic derivatives for '
            f'(kappa, gamma) = ({kappa}, {gamma}): theta lies exactly on '
            f'the critical wedge edge |sin 2 theta| == (1 - kappa) / '
            f'|gamma|, the deltoid cusp where the caustic derivatives '
            f'genuinely diverge (u_p, u_pp -> infinity); this degenerate '
            f'boundary is a named refusal, mirroring the off-wedge '
            f'refusal above.')
    u = eff * c + b * d_root
    if not positive_parity and np.any(u <= 0.0):
        raise LensDomainError(
            f'Cannot evaluate macro-saddle caustic derivatives for '
            f'(kappa, gamma) = ({kappa}, {gamma}): branch {branch} gives '
            f'a non-positive radial coordinate u <= 0.')
    u_p = -2.0 * eff * s - b * 2.0 * eff**2 * s * c / d_root
    u_pp = (-4.0 * eff * c - b * 4.0 * eff**2
            * (c4 * d_root**2 + eff**2 * s**2 * c**2) / d_root**3)
    r = 1.0 / np.sqrt(lam * u)
    r_p = -r * u_p / (2.0 * u)
    r_pp = r * (3.0 * u_p**2 / (4.0 * u**2) - u_pp / (2.0 * u))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    y_prime, y_double_prime = [], []
    for sign, tan, tan_p, tan_pp in ((-1.0, cos_t, -sin_t, -cos_t),
                                     (1.0, sin_t, cos_t, -sin_t)):
        p = (lam + sign * gamma) - lam * u
        p_p = -lam * u_p
        p_pp = -lam * u_pp
        y_prime.append(p_p * r * tan + p * r_p * tan + p * r * tan_p)
        y_double_prime.append(
            p_pp * r * tan + 2.0 * p_p * r_p * tan + 2.0 * p_p * r * tan_p
            + p * r_pp * tan + 2.0 * p * r_p * tan_p + p * r * tan_pp)
    return np.array(y_prime), np.array(y_double_prime)


def caustic_speed(gamma: float, theta, *, kappa: float = 0.0,
                  branch: int = 1):
    """Caustic parametric speed ``|y'(theta)|``.

    Thin delegate to :func:`caustic_derivatives`; see it for the domain
    contract and parameters.  Vectorised over ``theta``.
    """
    y_prime, _ = caustic_derivatives(gamma, theta, kappa=kappa, branch=branch)
    return np.sqrt(y_prime[0]**2 + y_prime[1]**2)


def caustic_curvature_radius(gamma: float, theta, *, kappa: float = 0.0,
                             branch: int = 1):
    """Caustic curvature radius ``|y'|**3 / |y1' y2'' - y2' y1''|``.

    Thin delegate to :func:`caustic_derivatives`; see it for the domain
    contract and parameters.  Vectorised over ``theta``.  A genuinely
    straight caustic point (vanishing cross product) returns ``inf``,
    which is the physical infinite radius, not an error.
    """
    y_prime, y_double_prime = caustic_derivatives(
        gamma, theta, kappa=kappa, branch=branch)
    speed = np.sqrt(y_prime[0]**2 + y_prime[1]**2)
    cross = y_prime[0] * y_double_prime[1] - y_prime[1] * y_double_prime[0]
    return speed**3 / np.abs(cross)


def fold_opening_direction(gamma: float, theta: float, *, kappa: float = 0.0,
                           branch: int = 1) -> np.ndarray:
    """Unit source-plane direction toward the fold's two-image side.

    At a fold caustic point ``y_c`` the source-plane curve traced by
    displacing the critical image along its degenerate (soft) axis ``e``
    is, to leading order, ``y(t) = y_c + (1/2) D2y[e, e] t**2``.  Because
    the linear ``t`` term vanishes (the fold is where two images merge),
    *both* signs of ``t`` map to the same side of the caustic -- the side
    carrying the extra merging image pair.  The unit vector along
    ``D2y[e, e]`` therefore points from the caustic toward that
    two-image side.

    Only the point-mass term of ``y(x) = A x - x / |x|**2`` contributes:
    the linear macro term ``A x`` has a vanishing second derivative.  The
    verified closed form of the point-mass second-derivative contraction
    (Professor Q2) is, with ``x = cp.image``, ``e = cp.soft_axis``,
    ``r2 = x . x`` and ``xe = x . e``,

        D2y[e, e] = (4 xe e + 2 x - 8 xe**2 x / r2) / r2**2.

    It depends on ``e`` only through ``xe**2`` and ``4 xe e``, both
    invariant under ``e -> -e``, so the ``soft_axis`` sign ambiguity is
    harmless and no sign correction is applied.

    Parameters
    ----------
    gamma : float
        External shear magnitude.
    theta : float
        Polar angle of the critical point in the lens plane, radians.
    kappa : float
        External convergence.
    branch : int
        Square-root branch passed through to :func:`critical_point`.

    Returns
    -------
    np.ndarray
        Shape ``(2,)`` unit vector ``D2y[e, e] / |D2y[e, e]|``.

    Raises
    ------
    LensDomainError
        Inherited from :func:`critical_point` for out-of-domain
        ``(kappa, gamma, theta, branch)`` (the domain checks are not
        re-derived here).
    """
    cp = critical_point(gamma, theta, kappa=kappa, branch=branch)
    x = cp.image
    e = cp.soft_axis
    r2 = x @ x
    xe = x @ e
    d2 = (4.0 * xe * e + 2.0 * x - 8.0 * xe**2 * x / r2) / r2**2
    return d2 / np.linalg.norm(d2)


# ---------------------------------------------------------------------------
# Ghost (complex-saddle) machinery
# ---------------------------------------------------------------------------
#
# A source outside the astroid caustic has only two real images, but the
# image quartic in ``u = 1 / |x|**2`` always has four roots: the two real
# images plus a complex-conjugate pair that ``_generic_candidates``
# discards at its imaginary-part cut.  That discarded pair is the pair of
# *ghost* (complex-saddle) images -- the analytic continuation of the two
# real images that have merged and gone complex across the fold
# (Picard--Lefschetz / steepest-descent picture of the diffraction
# integral).  One member decays (``Im tau_c > 0``, so
# ``exp(1j * w * tau_c)`` is exponentially suppressed) and is the physical
# deep-diffraction correction; its conjugate grows and is unphysical.
#
# Everything below continues the real geometrical-optics formulae to the
# complex root position with BILINEAR (non-conjugated) products.  The
# Fermat delay and the stationary-phase kernel are holomorphic functions
# of the (complex) image position, so replacing every ``a . b`` inner
# product by the bilinear contraction ``sum_i a_i b_i`` (no conjugation)
# and every ``|x|`` by ``sqrt(x . x)`` gives the unique analytic
# continuation.  The real-only building blocks (`delay`, `hessian`,
# `magnification`, `morse_index`, `_saddle_metric`, `saddle_coefficients`,
# `image_kernel`) embed conjugating / positive-definite operations
# (``np.linalg.norm``, ``np.log|.|``, ``eigvalsh``, ``dtype=float``) and so
# are NOT reused here; only the pure-arithmetic coefficient polynomials
# `_c1_polynomial` / `_c2_polynomial` and the (real) quartic solver are
# shared.  These functions are purely additive: the real-image path is
# untouched.
#
# Units follow the module docstring: ``w`` and ``tau`` are dimensionless,
# with the same ``t_min`` demodulation convention the map layer uses (see
# ``ppgo_map._measure_cell``); nothing here consumes the ghost yet, so no
# demodulation is applied at this layer.


class GhostDomainError(LensDomainError):
    """The complex-saddle ('ghost') continuation is degenerate or absent.

    Raised when the decaying complex image required by the ghost kernel
    cannot be continued from the real fold:

    * no complex-conjugate quartic pair exists (the source lies inside
      the caustic, giving four real images and no ghost to continue);
    * the bilinear radius ``z = x_c . x_c`` has ``Re(z) <= 0``, so the
      principal branch of ``log(z)`` (and of ``sqrt(z)``) can no longer
      be continued by continuity from the real fold, where ``z > 0`` --
      a topology breakdown near the cusp;
    * the complex Fermat Hessian is near singular
      (``|det H_c| < 1e-8 * (1 + ||A||_F)**2``), a near-fold merge where
      the stationary-phase amplitude ``1 / sqrt(det H_c)`` and its
      sqrt-branch reference are ill conditioned.

    It subclasses `LensDomainError` so existing domain-refusal handlers
    catch it unchanged.
    """


#: Relative floor on ``|det H_c|`` below which the complex saddle is
#: treated as a degenerate near-fold merge.  Scaled by
#: ``(1 + ||A||_F)**2`` so it tracks the macro-matrix magnitude.
_GHOST_DET_FLOOR = 1e-8


class GhostContribution(NamedTuple):
    """The decaying ghost image's carrier-free kernel and complex delay.

    Attributes
    ----------
    kernel : np.ndarray
        Carrier-free ghost kernel ``amplitude * (1 + 1j*C1/w + C2/w**2)``
        broadcast over the input frequencies; the oscillatory/decaying
        carrier ``exp(1j * w * tau_c)`` is NOT included (mirroring
        `image_kernel`).
    delay : complex
        Complex Fermat delay ``tau_c`` of the decaying ghost.  Its real
        part is the oscillation phase and ``Im tau_c >= 0`` controls the
        exponential suppression ``exp(-w * Im tau_c)``.  Exposed so a
        future build can gate the ghost on ``w * Im tau_c``.
    position : np.ndarray
        Complex ghost image position ``x_c`` (shape (2,)) in the lens
        frame.
    """

    kernel: np.ndarray
    delay: complex
    position: np.ndarray


def _wrapped_angle(delta: float) -> float:
    """Smallest absolute angle equivalent to ``delta`` (radians), in
    ``[0, pi]``."""
    wrapped = (float(delta) + np.pi) % (2.0 * np.pi) - np.pi
    return abs(wrapped)


def _branch_pinned_amplitude(root: complex,
                             reference_amplitude: complex) -> complex:
    """Pin the ``+/- 1/sqrt(det H_c)`` branch to the real saddle reference.

    The complex amplitude ``1 / sqrt(det H_c)`` is defined only up to an
    overall sign (the two square roots).  Continued from the real fold,
    the physical root is the one whose phase matches the real merged
    saddle, whose amplitude carries the Morse phase ``exp(-0.5j * pi)``
    (index 1).  Selecting that root therefore ABSORBS the ``-i pi / 2``
    Morse factor -- multiplying by an explicit ``morse_index`` phase on
    top would double-count it.

    Parameters
    ----------
    root : complex
        The principal candidate ``1 / sqrt(det H_c)``.  The other
        candidate is ``-root``.
    reference_amplitude : complex
        The real merged-saddle amplitude; only its phase is used.

    Returns
    -------
    complex
        Whichever of ``root``, ``-root`` has the smaller angular
        distance to ``reference_amplitude``'s phase.
    """
    reference_angle = np.angle(reference_amplitude)
    return min((root, -root),
               key=lambda candidate: _wrapped_angle(
                   np.angle(candidate) - reference_angle))


def _ghost_candidates(source: np.ndarray, matrix: np.ndarray, *,
                      root_tolerance: float = 3e-7) -> list[np.ndarray]:
    """Complex-conjugate ('ghost') image candidates the real finder drops.

    Reuses the SAME source-aligned frame and companion-matrix quartic
    solve as `find_images_quartic` / `_generic_candidates`, but KEEPS the
    complex-conjugate root pair (``|Im u| ABOVE`` the tolerance) instead
    of discarding it.  For each complex root ``u_c`` the image position
    is reconstructed with the same closed form the real path uses, now
    evaluated at a complex ``u_c`` (the map ``u -> x`` is a rational,
    hence complex-analytic, function), and rotated back to the lens frame
    by the real orthogonal source-frame basis.  Because ``x . x`` is
    rotation invariant the continuation is frame independent.

    Parameters
    ----------
    source : np.ndarray
        Shape (2,), source position.
    matrix : np.ndarray
        Shape (2, 2), symmetric macro matrix (see `macro_matrix`).
    root_tolerance : float
        Relative imaginary part ABOVE which a quartic root counts as
        complex (a ghost); the same tolerance the real finder uses below
        which a root counts as real.

    Returns
    -------
    list of np.ndarray
        Complex ghost image positions (each shape (2,), dtype complex)
        in the lens frame.  Empty when the source is centered or lies
        inside the caustic (no complex pair to continue).

    Raises
    ------
    ValueError
        If the shapes are wrong.
    """
    source = np.asarray(source, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if source.shape != (2,) or matrix.shape != (2, 2):
        raise ValueError(
            f'Cannot extract ghost candidates for source of shape '
            f'{source.shape} and matrix of shape {matrix.shape}: they '
            f'must have shapes (2,) and (2, 2).')

    source_radius, basis = _source_frame(source)
    if source_radius <= 1e-14:
        # A centered source has no shear-broken ghost pair to continue;
        # the source-aligned reduction is undefined there.
        return []

    rotated = basis.T @ matrix @ basis
    a11 = float(rotated[0, 0])
    a12 = float(rotated[0, 1])
    a22 = float(rotated[1, 1])
    candidates: list[np.ndarray] = []
    for raw_root in _companion_roots(image_quartic_coefficients(source_radius,
                                                                rotated)):
        if raw_root.real <= 0.0:
            continue
        if abs(raw_root.imag) <= root_tolerance * (1.0 + abs(raw_root.real)):
            continue  # real root -> a genuine image, owned by find_images
        u_c = complex(raw_root)
        denominator = (a11 - u_c) * (a22 - u_c) - a12 * a12
        if abs(denominator) <= 1e-12 * (1.0 + abs(a11 * a22) + abs(u_c)**2):
            continue
        rotated_candidate = np.array(
            [source_radius * (a22 - u_c) / denominator,
             -source_radius * a12 / denominator], dtype=complex)
        candidates.append(basis @ rotated_candidate)
    return candidates


def _ghost_delay(x_c: np.ndarray, source: np.ndarray,
                 matrix: np.ndarray) -> complex:
    """Analytically-continued complex Fermat delay of a ghost image.

    The real Fermat delay ``0.5 x.A.x - y.x + 0.5 y.y - ln|x|`` continues
    to

        ``tau_c = 0.5 x_c.A.x_c - y.x_c + 0.5 y.y - 0.5 * log(z)``,
        ``z = x_c . x_c`` (bilinear, ``x1**2 + x2**2``),

    with every product the bilinear contraction (NO conjugation) so the
    map ``x -> tau`` is holomorphic and reduces to `delay` on the real
    fold, where ``z = |x|**2 > 0`` and ``0.5 * log(z) = ln|x|``.  The
    principal branch of ``log`` is correct by continuity from the real
    fold (``arg z = 0`` there) while ``Re(z) > 0``.

    Parameters
    ----------
    x_c : np.ndarray
        Shape (2,), complex ghost image position (lens frame).
    source : np.ndarray
        Shape (2,), source position.
    matrix : np.ndarray
        Shape (2, 2), the macro matrix.

    Returns
    -------
    complex
        The complex delay ``tau_c``.  ``Im tau_c > 0`` for the decaying
        member of a conjugate pair.

    Raises
    ------
    GhostDomainError
        If ``Re(z) <= 0`` (the complex-log branch can no longer be
        continued from the real fold -- topology breakdown near the
        cusp).
    """
    x_c = np.asarray(x_c, dtype=complex)
    source = np.asarray(source, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    z = x_c[0] * x_c[0] + x_c[1] * x_c[1]
    if z.real <= 0.0:
        raise GhostDomainError(
            f'Cannot continue the ghost delay at x_c = '
            f'({x_c[0]!r}, {x_c[1]!r}): the bilinear radius z = x_c . x_c '
            f'= {z!r} has Re(z) <= 0, so the principal branch of log(z) is '
            f'no longer the continuation from the real fold (topology '
            f'breakdown near the cusp).')
    return (0.5 * (x_c @ matrix @ x_c) - source @ x_c
            + 0.5 * (source @ source) - 0.5 * np.log(z))


def _ghost_kernel(w_dimensionless, x_c: np.ndarray, source: np.ndarray,
                  matrix: np.ndarray,
                  reference_amplitude: complex) -> tuple[np.ndarray, complex]:
    """Carrier-free ghost kernel and complex delay by analytic continuation.

    Continues the stationary-phase (saddle) kernel of `image_kernel` to
    the complex ghost position.  All geometry is bilinear (holomorphic);
    none of the real-only helpers (`delay`, `hessian`, `magnification`,
    `morse_index`, `_saddle_metric`, `saddle_coefficients`, `image_kernel`)
    is called.  The amplitude is ``1 / sqrt(det H_c)`` with the sqrt
    branch pinned to the real merged saddle via ``reference_amplitude``
    (which absorbs the ``-i pi / 2`` Morse phase -- see
    `_branch_pinned_amplitude`); ``C1``/``C2`` reuse the shared
    coefficient polynomials on a complex saddle metric.

    Parameters
    ----------
    w_dimensionless : float or np.ndarray
        Dimensionless frequency ``w``; the carrier ``exp(1j * w * tau_c)``
        is NOT included.
    x_c : np.ndarray
        Shape (2,), complex ghost image position (lens frame).
    source : np.ndarray
        Shape (2,), source position.
    matrix : np.ndarray
        Shape (2, 2), the macro matrix.
    reference_amplitude : complex
        Real merged-saddle amplitude; only its phase pins the sqrt
        branch.

    Returns
    -------
    kernel : np.ndarray
        ``amplitude * (1 + 1j*C1/w + C2/w**2)`` broadcast over ``w``.
    tau_c : complex
        The complex Fermat delay (also returned by `_ghost_delay`),
        exposed so callers can gate on ``Im tau_c``.

    Raises
    ------
    GhostDomainError
        If ``Re(z) <= 0`` or ``|det H_c| < 1e-8 * (1 + ||A||_F)**2``
        (near-fold merge, ill-conditioned sqrt reference).
    """
    x_c = np.asarray(x_c, dtype=complex)
    matrix = np.asarray(matrix, dtype=float)
    w_dimensionless = np.asarray(w_dimensionless, dtype=float)

    z = x_c[0] * x_c[0] + x_c[1] * x_c[1]
    if z.real <= 0.0:
        raise GhostDomainError(
            f'Cannot continue the ghost kernel at x_c = '
            f'({x_c[0]!r}, {x_c[1]!r}): the bilinear radius z = x_c . x_c '
            f'= {z!r} has Re(z) <= 0 (topology breakdown near the cusp).')
    tau_c = _ghost_delay(x_c, source, matrix)

    # Complex Fermat Hessian: the continuation of
    #   hessian = A - I / r**2 + 2 outer(x, x) / r**4,  r**2 = z,
    # with the BILINEAR (non-conjugated) outer product.
    complex_hessian = (matrix - np.eye(2) / z
                       + 2.0 * np.outer(x_c, x_c) / z**2)
    det_hessian = (complex_hessian[0, 0] * complex_hessian[1, 1]
                   - complex_hessian[0, 1] * complex_hessian[1, 0])
    frobenius = float(np.linalg.norm(matrix))
    if abs(det_hessian) < _GHOST_DET_FLOOR * (1.0 + frobenius)**2:
        raise GhostDomainError(
            f'Cannot continue the ghost kernel at x_c = '
            f'({x_c[0]!r}, {x_c[1]!r}): |det H_c| = {abs(det_hessian)!r} is '
            f'below the near-fold floor {_GHOST_DET_FLOOR} * (1 + ||A||_F)**2 '
            f'= {_GHOST_DET_FLOOR * (1.0 + frobenius)**2!r}; the sqrt-branch '
            f'amplitude reference is ill conditioned at a merged saddle.')
    amplitude = _branch_pinned_amplitude(1.0 / np.sqrt(det_hessian),
                                         reference_amplitude)

    # Complex saddle metric: the continuation of _saddle_metric with the
    # bilinear (non-conjugate) radial/tangential frame and matrix inverse.
    # det(basis2) = radial . radial + ... = z / z = 1 exactly, so the
    # projected determinant equals det H_c and no extra guard is needed.
    root_z = np.sqrt(z)
    radial = x_c / root_z
    tangential = np.array([-radial[1], radial[0]], dtype=complex)
    frame = np.column_stack([radial, tangential])
    projected = frame.T @ complex_hessian @ frame
    projected_det = (projected[0, 0] * projected[1, 1]
                     - projected[0, 1] * projected[1, 0])
    inverse = np.array([[projected[1, 1], -projected[0, 1]],
                        [-projected[1, 0], projected[0, 0]]],
                       dtype=complex) / projected_det
    scale = 1.0 / z
    prr = scale * inverse[0, 0]
    prt = scale * inverse[0, 1]
    ptt = scale * inverse[1, 1]

    c1_coefficient = _c1_polynomial(prr, prt, ptt)
    c2_coefficient = _c2_polynomial(prr, prt, ptt)
    kernel = amplitude * (1.0 + 1j * c1_coefficient / w_dimensionless
                          + c2_coefficient / w_dimensionless**2)
    return kernel, tau_c


def ghost_kernel(w_dimensionless, source: np.ndarray, matrix: np.ndarray, *,
                 root_tolerance: float = 3e-7) -> GhostContribution:
    """
    Carrier-free kernel of the decaying complex-saddle ('ghost') image.

    Extracts the complex-conjugate quartic-root pair the real image
    finder discards, selects the decaying member (largest ``Im tau_c``,
    positive off the cusp axis), and returns its analytically-continued
    stationary-phase kernel and complex Fermat delay.  Nothing in the
    pipeline consumes this yet; it is the physics primitive a later build
    gates on ``w * Im tau_c``.

    The amplitude's sqrt branch is pinned to the real merged saddle,
    which the two real images continue into across the fold: that saddle
    has Morse index 1, i.e. amplitude phase ``exp(-0.5j * pi)`` (arg
    ``-pi/2``).  Only the phase enters the ``+/- sqrt`` selection, so the
    reference is built directly from the Morse phase rather than from a
    real magnification, which would be ill conditioned exactly in the
    near-caustic regime where the ghost matters.

    Parameters
    ----------
    w_dimensionless : float or np.ndarray
        Dimensionless frequency ``w`` (see the module docstring); the
        carrier ``exp(1j * w * tau_c)`` is NOT included.
    source : np.ndarray
        Shape (2,), source position.
    matrix : np.ndarray
        Shape (2, 2), symmetric macro matrix (see `macro_matrix`).
    root_tolerance : float
        Relative imaginary part above which a quartic root counts as a
        ghost (see `_ghost_candidates`).

    Returns
    -------
    GhostContribution
        The decaying ghost's ``kernel`` (over ``w``), complex ``delay``
        ``tau_c`` (with ``Im tau_c`` exposed for the future gate), and
        complex ``position`` ``x_c``.

    Raises
    ------
    ValueError
        If the shapes are wrong.
    GhostDomainError
        If no complex-saddle pair exists (source inside the caustic), or
        if the continuation is degenerate (see `_ghost_kernel`).

    Notes
    -----
    As the source approaches a principal axis, ``Im tau_c -> 0``
    continuously and the ghost kernel converges to a finite limit (the
    on-axis ghost is pure oscillation with no decay).  EXACTLY on a
    principal axis the source-aligned macro matrix is diagonal and the
    complex-conjugate quartic pair collapses onto the removable
    singularity ``u = a22`` (the same degeneracy the real finder resolves
    with `_axial_candidates`); the generic reconstruction this routine
    uses cannot produce ``x_c`` there, so `GhostDomainError` is raised at
    that measure-zero set.  Grid evaluations that avoid the exact axis
    reach the finite limit from either side.  The gate that refuses to
    *use* the ghost near the cusp is a downstream concern, not this
    primitive's.  This routine is purely additive: the real-image path
    (`find_images`, `image_kernel`, `delay`, `morse_index`, ...) is
    untouched and byte-identical.
    """
    source = np.asarray(source, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if source.shape != (2,) or matrix.shape != (2, 2):
        raise ValueError(
            f'Cannot evaluate the ghost kernel for source of shape '
            f'{source.shape} and matrix of shape {matrix.shape}: they '
            f'must have shapes (2,) and (2, 2).')

    candidates = _ghost_candidates(source, matrix,
                                   root_tolerance=root_tolerance)
    if not candidates:
        raise GhostDomainError(
            f'No complex-saddle (ghost) pair for source '
            f'({source[0]!r}, {source[1]!r}): the image quartic has no '
            f'complex-conjugate root above the tolerance, so the source '
            f'lies inside the caustic (four real images) or is centered; '
            f'there is no ghost to continue.')

    delays = [_ghost_delay(x_c, source, matrix) for x_c in candidates]
    # The decaying member has the largest Im tau_c (> 0 off the cusp axis,
    # == 0 on it, where argmax still returns a valid pure-oscillation
    # member rather than the growing conjugate).
    decaying_index = int(np.argmax([tau.imag for tau in delays]))
    x_c = candidates[decaying_index]

    # Merged-saddle Morse reference (index 1): the two real images
    # continue into a Morse-index-1 saddle, amplitude phase exp(-0.5j*pi).
    reference_amplitude = np.exp(-0.5j * np.pi)
    kernel, tau_c = _ghost_kernel(w_dimensionless, x_c, source, matrix,
                                  reference_amplitude)
    return GhostContribution(kernel=kernel, delay=tau_c, position=x_c)
