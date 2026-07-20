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
from scipy.optimize import minimize_scalar

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


def _generic_candidates(source_radius: float, rotated: np.ndarray, *,
                        root_tolerance: float) -> list[np.ndarray]:
    """Candidates in the source frame from the roots of the quartic in
    ``u = 1 / |x|**2``."""
    a11 = float(rotated[0, 0])
    a12 = float(rotated[0, 1])
    a22 = float(rotated[1, 1])
    candidates: list[np.ndarray] = []
    for raw_root in np.roots(image_quartic_coefficients(source_radius,
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


def nearest_caustic_point(gamma: float, beta: float, source: np.ndarray,
                          *, kappa: float = 0.0, n_grid: int = 256
                          ) -> NearestCausticPoint:
    """
    Caustic point closest to a source, by search along the critical
    curve.

    Positive parity (``abs(gamma) < 1 - kappa``): a coarse scan over
    ``n_grid`` polar angles spanning the full circle is refined with a
    bounded one-dimensional minimization from each of the four best
    grid cells, so that all four cusps of the single astroid remain
    reachable.  This branch is byte-identical to the frozen
    positive-parity implementation (the ``+`` square-root branch only).

    Macro saddle (``0 < 1 - kappa < abs(gamma)``): the critical curve is
    two 3-cusp deltoid lobes confined to the two angular wedges
    ``|sin 2(theta - beta)| <= (1 - kappa) / abs(gamma)`` about the
    negative-eigenvalue axis (``theta - beta`` near ``0`` and ``pi``).
    Each wedge is scanned for both square-root branches (``+-``), and
    the global minimum over the two lobes and two branches is returned,
    so both deltoid lobes remain reachable.

    The frequency-independent distance search runs through the compiled
    source-only helper `_caustic_source` (the eigenframe and eigenvalue
    are not needed to locate the closest caustic); the returned local
    frame and eigenvalue are then built from a single `critical_point`
    call at the winning angle and branch, so those fields are identical
    to a search that used `critical_point` throughout.

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
        Number of polar angles in the coarse scan (per wedge and branch
        for a macro saddle).

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

    if abs(gamma) < lam:
        # Positive parity: a single 4-cusp astroid over the full circle,
        # the ``+`` branch only.  Byte-identical to the frozen path.
        grid = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
        step = 2.0 * np.pi / n_grid

        def squared_distance(theta) -> float:
            caustic = _caustic_source(float(theta) % (2.0 * np.pi),
                                      gamma, beta, kappa, 1.0)
            return float(np.sum((caustic - source)**2))

        coarse = _coarse_squared_distances(grid, gamma, beta, kappa,
                                           source, 1.0)
        best = None
        for index in np.argsort(coarse)[:4]:
            center = grid[index]
            refined = minimize_scalar(squared_distance,
                                      bounds=(center - step, center + step),
                                      method='bounded',
                                      options={'xatol': 1e-12})
            if best is None or refined.fun < best.fun:
                best = refined
        theta = float(best.x % (2.0 * np.pi))
        return NearestCausticPoint(
            theta,
            *critical_point(gamma, theta, beta, kappa),
            distance=float(np.sqrt(best.fun)))

    # Macro saddle: two 3-cusp deltoid lobes, each confined to a wedge of
    # half-width theta_max about the negative-eigenvalue axis, and each
    # traced by both square-root branches.
    theta_max = 0.5 * np.arcsin(lam / abs(gamma))
    best_fun = np.inf
    best_theta = 0.0
    best_branch = 1
    for center in (beta, beta + np.pi):
        wedge = np.linspace(center - theta_max, center + theta_max, n_grid)
        step = 2.0 * theta_max / (n_grid - 1)
        for branch in (1.0, -1.0):

            def squared_distance(theta, branch=branch) -> float:
                caustic = _caustic_source(float(theta), gamma, beta,
                                          kappa, branch)
                return float(np.sum((caustic - source)**2))

            coarse = _coarse_squared_distances(wedge, gamma, beta, kappa,
                                               source, branch)
            for index in np.argsort(coarse)[:4]:
                lower = max(wedge[index] - step, center - theta_max)
                upper = min(wedge[index] + step, center + theta_max)
                refined = minimize_scalar(
                    squared_distance,
                    bounds=(lower, upper),
                    method='bounded',
                    options={'xatol': 1e-12})
                if refined.fun < best_fun:
                    best_fun = refined.fun
                    best_theta = float(refined.x)
                    best_branch = int(branch)

    theta = best_theta % (2.0 * np.pi)
    return NearestCausticPoint(
        theta,
        *critical_point(gamma, best_theta, beta, kappa, best_branch),
        distance=float(np.sqrt(best_fun)))
