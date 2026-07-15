#!/usr/bin/env python3
"""Algebraic image geometry and saddle kernels for the Chang--Refsdal lens.

The production image finder uses the exact quartic reduction in
``u=1/|x|^2``.  For a general source vector the coordinates are first rotated
so that the source lies on the first axis; the macro matrix is rotated with it.
No multistart root search is used in the runtime path.
"""
from __future__ import annotations

import numpy as np


def macro_matrix(gamma: float, beta: float = 0.0, kappa: float = 0.0) -> np.ndarray:
    """Quadratic part of the Fermat Hessian with convergence and shear.

    The positive-parity macro-image regime requires ``1-kappa > abs(gamma)``.
    """
    c, s = np.cos(2.0 * beta), np.sin(2.0 * beta)
    q = np.array([[c, s], [s, -c]])
    return (1.0 - kappa) * np.eye(2) - gamma * q


def hessian(x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    r2 = float(x @ x)
    if r2 <= 0.0:
        raise ValueError("Hessian is singular at the point mass")
    return matrix - np.eye(2) / r2 + 2.0 * np.outer(x, x) / r2**2


def lens_residual(x: np.ndarray, y: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r2 = float(x @ x)
    if r2 <= 1e-30:
        return np.array([np.inf, np.inf])
    return matrix @ x - x / r2 - y


def delay(x: np.ndarray, y: np.ndarray, matrix: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(
        0.5 * x @ matrix @ x - y @ x + 0.5 * y @ y
        - np.log(np.linalg.norm(x))
    )


def _source_frame(y: np.ndarray) -> tuple[float, np.ndarray]:
    """Return source radius and orthogonal matrix whose first axis is y."""
    y = np.asarray(y, dtype=float)
    if y.shape != (2,):
        raise ValueError("y must be a two-vector")
    radius = float(np.linalg.norm(y))
    if radius == 0.0:
        return 0.0, np.eye(2)
    e1 = y / radius
    e2 = np.array([-e1[1], e1[0]])
    return radius, np.column_stack([e1, e2])


def image_quartic_coefficients(source_radius: float, rotated_matrix: np.ndarray) -> np.ndarray:
    r"""Quartic coefficients for ``u=1/r^2`` in descending order.

    In the source-aligned frame put ``A=[[a,b],[b,d]]`` and ``y=(Y,0)``.
    The lens equation gives

    ``x1=Y(d-u)/D``, ``x2=-Y b/D``,
    ``D=(a-u)(d-u)-b^2``.

    The radial constraint is

    ``D^2-Y^2 u[(d-u)^2+b^2]=0``.
    """
    Y = float(source_radius)
    if Y < 0.0:
        raise ValueError("source_radius must be nonnegative")
    A = np.asarray(rotated_matrix, dtype=float)
    if A.shape != (2, 2):
        raise ValueError("rotated_matrix must be 2 by 2")
    a, b, d = float(A[0, 0]), float(A[0, 1]), float(A[1, 1])
    detA = a * d - b * b
    return np.array(
        [
            1.0,
            -2.0 * (a + d) - Y * Y,
            a * a + 4.0 * a * d + d * d - 2.0 * b * b + 2.0 * d * Y * Y,
            -2.0 * (a + d) * detA - Y * Y * (d * d + b * b),
            detA * detA,
        ],
        dtype=float,
    )


def _newton_polish(
    x: np.ndarray,
    y: np.ndarray,
    matrix: np.ndarray,
    *,
    max_steps: int = 8,
) -> np.ndarray:
    """Deterministically polish one algebraic candidate with the lens Jacobian."""
    z = np.asarray(x, dtype=float).copy()
    for _ in range(max_steps):
        residual = lens_residual(z, y, matrix)
        if not np.all(np.isfinite(residual)) or np.linalg.norm(residual) < 2e-13:
            break
        jac = hessian(z, matrix)
        try:
            step = np.linalg.solve(jac, residual)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(jac, residual, rcond=None)[0]
        # A large Newton step near a multiple root is less useful than the
        # original algebraic candidate.  Backtrack until the residual falls.
        old_norm = float(np.linalg.norm(residual))
        accepted = False
        scale = 1.0
        for _ in range(8):
            trial = z - scale * step
            if np.linalg.norm(trial) < 1e-12:
                scale *= 0.5
                continue
            new_norm = float(np.linalg.norm(lens_residual(trial, y, matrix)))
            if np.isfinite(new_norm) and new_norm < old_norm:
                z = trial
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
    return z


def _centered_source_images(matrix: np.ndarray, *, degeneracy_tolerance: float) -> list[np.ndarray]:
    """Images for y=0; the gamma=0 Einstein ring is intentionally rejected."""
    values, vectors = np.linalg.eigh(matrix)
    if np.any(values <= 0.0):
        raise ValueError("centered-source solver assumes a positive-parity macroimage")
    if abs(values[1] - values[0]) <= degeneracy_tolerance * max(abs(values[0]), abs(values[1]), 1.0):
        raise ValueError("at zero source and zero shear the image is an Einstein ring")
    images: list[np.ndarray] = []
    for value, vector in zip(values, vectors.T):
        x = vector / np.sqrt(value)
        images.extend([x, -x])
    return images


def find_images_quartic(
    y: np.ndarray,
    matrix: np.ndarray,
    *,
    root_tolerance: float = 3e-7,
    residual_tolerance: float = 3e-8,
    duplicate_tolerance: float = 3e-7,
    axis_tolerance: float = 5e-11,
) -> list[np.ndarray]:
    """Return all distinct finite real images from the exact quartic reduction.

    The quartic is solved in a frame aligned with the source.  Axis-aligned
    shear is handled separately because the off-axis pair lies at a removable
    singularity of the generic reconstruction formula.  A short deterministic
    Newton polish is used only to remove floating-point error from the
    algebraic roots; it is not a multistart image search.
    """
    y = np.asarray(y, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if y.shape != (2,) or matrix.shape != (2, 2):
        raise ValueError("y and matrix must have shapes (2,) and (2,2)")
    if not np.allclose(matrix, matrix.T, atol=1e-13, rtol=0.0):
        raise ValueError("matrix must be symmetric")

    Y, basis = _source_frame(y)
    if Y <= 1e-14:
        raw_candidates = _centered_source_images(
            matrix, degeneracy_tolerance=axis_tolerance
        )
    else:
        A = basis.T @ matrix @ basis
        a, b, d = float(A[0, 0]), float(A[0, 1]), float(A[1, 1])
        candidates_rotated: list[np.ndarray] = []

        if abs(b) <= axis_tolerance * max(abs(a), abs(d), 1.0):
            # Axial images satisfy a*x1^2-Y*x1-1=0.
            discriminant = Y * Y + 4.0 * a
            if discriminant >= -root_tolerance and abs(a) > 1e-14:
                root = np.sqrt(max(discriminant, 0.0))
                candidates_rotated.extend(
                    [
                        np.array([(Y + root) / (2.0 * a), 0.0]),
                        np.array([(Y - root) / (2.0 * a), 0.0]),
                    ]
                )

            # A symmetric off-axis pair can occur at u=d.
            if d > 0.0 and abs(a - d) > 1e-14:
                x1 = Y / (a - d)
                x2_squared = 1.0 / d - x1 * x1
                if x2_squared >= -root_tolerance:
                    x2 = np.sqrt(max(x2_squared, 0.0))
                    candidates_rotated.extend(
                        [np.array([x1, x2]), np.array([x1, -x2])]
                    )
        else:
            coefficients = image_quartic_coefficients(Y, A)
            for raw_u in np.roots(coefficients):
                if raw_u.real <= 0.0:
                    continue
                if abs(raw_u.imag) > root_tolerance * (1.0 + abs(raw_u.real)):
                    continue
                u = float(raw_u.real)
                D = (a - u) * (d - u) - b * b
                if abs(D) <= 1e-12 * (1.0 + abs(a * d) + u * u):
                    continue
                candidates_rotated.append(
                    np.array([Y * (d - u) / D, -Y * b / D], dtype=float)
                )

        raw_candidates = [basis @ z for z in candidates_rotated]

    images: list[np.ndarray] = []
    for candidate in raw_candidates:
        if not np.all(np.isfinite(candidate)) or np.linalg.norm(candidate) < 1e-11:
            continue
        polished = _newton_polish(candidate, y, matrix)
        residual = float(np.linalg.norm(lens_residual(polished, y, matrix)))
        if residual > residual_tolerance:
            # Near a multiple root the unpolished algebraic position can be
            # superior to a poorly conditioned Newton correction.
            candidate_residual = float(np.linalg.norm(lens_residual(candidate, y, matrix)))
            if candidate_residual < residual:
                polished, residual = candidate, candidate_residual
        if residual > residual_tolerance:
            continue
        scale = 1.0 + float(np.linalg.norm(polished))
        if all(np.linalg.norm(polished - old) > duplicate_tolerance * scale for old in images):
            images.append(polished)

    images.sort(key=lambda x: delay(x, y, matrix))
    return images


def find_images(y: np.ndarray, matrix: np.ndarray) -> list[np.ndarray]:
    """Production image finder: exact quartic reduction plus polishing."""
    return find_images_quartic(y, matrix)


def saddle_coefficients(x: np.ndarray, matrix: np.ndarray) -> tuple[float, float]:
    """Return the analytic C1,C2 coefficients of the image expansion."""
    h = hessian(x, matrix)
    radius = np.linalg.norm(x)
    er = x / radius
    et = np.array([-er[1], er[0]])
    basis = np.column_stack([er, et])
    inv = np.linalg.inv(basis.T @ h @ basis)
    u = 1.0 / radius**2
    p, q, s = u * inv[0, 0], u * inv[0, 1], u * inv[1, 1]
    c1 = (
        10*p**3 - 12*p*p*s - 9*p*p - 48*p*q*q + 18*p*s*s + 18*p*s
        + 72*q*q*s + 36*q*q - 9*s*s
    ) / 12.0
    poly = (
        1540*p**6 - 1680*p**5*s - 3780*p**5 - 16800*p**4*q**2
        + 2520*p**4*s**2 + 5040*p**4*s + 2961*p**4
        + 40320*p**3*q**2*s + 40320*p**3*q**2
        - 3600*p**3*s**3 - 8280*p**3*s**2 - 5364*p**3*s - 720*p**3
        + 40320*p**2*q**4 - 64800*p**2*q**2*s**2
        - 99360*p**2*q**2*s - 32184*p**2*q**2
        + 3780*p**2*s**4 + 10800*p**2*s**3 + 9126*p**2*s**2
        + 2160*p**2*s - 86400*p*q**4*s - 66240*p*q**4
        + 60480*p*q**2*s**3 + 129600*p*q**2*s**2
        + 73008*p*q**2*s + 8640*p*q**2
        - 3780*p*s**4 - 5940*p*s**3 - 2160*p*s**2
        - 11520*q**6 + 60480*q**4*s**2 + 86400*q**4*s
        + 24336*q**4 - 30240*q**2*s**3 - 35640*q**2*s**2
        - 8640*q**2*s + 945*s**4 + 720*s**3
    )
    return float(c1), float(-poly / 288.0)


def image_kernel(w, x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Carrier-free image kernel through relative O(w^-2)."""
    w = np.asarray(w, dtype=float)
    h = hessian(x, matrix)
    eigenvalues = np.linalg.eigvalsh(h)
    morse = int(np.sum(eigenvalues < 0.0))
    mu = 1.0 / np.linalg.det(h)
    c1, c2 = saddle_coefficients(x, matrix)
    return (
        np.sqrt(abs(mu))
        * np.exp(-0.5j * np.pi * morse)
        * (1.0 + 1j*c1/w + c2/w**2)
    )


def critical_point(
    gamma: float,
    theta: float,
    beta: float = 0.0,
    kappa: float = 0.0,
):
    """Critical point for constant convergence plus shear.

    The result follows from the exact mass-sheet rescaling
    ``x'=sqrt(lambda) x``, ``y'=y/sqrt(lambda)``, with
    ``lambda=1-kappa`` and effective shear ``gamma/lambda``.
    """
    lam = 1.0 - float(kappa)
    if lam <= 0.0 or abs(gamma) >= lam:
        raise ValueError("critical_point assumes 1-kappa > |gamma| > = 0")
    geff = gamma / lam
    phi = theta - beta
    ueff = geff * np.cos(2.0 * phi) + np.sqrt(
        1.0 - geff**2 * np.sin(2.0 * phi) ** 2
    )
    # x' has radius 1/sqrt(ueff); physical x=x'/sqrt(lam).
    radius = 1.0 / np.sqrt(lam * ueff)
    x = radius * np.array([np.cos(theta), np.sin(theta)])
    matrix = macro_matrix(gamma, beta, kappa)
    y = matrix @ x - x / radius**2
    h = hessian(x, matrix)
    values, vectors = np.linalg.eigh(h)
    soft_index = int(np.argmin(np.abs(values)))
    hard_index = 1 - soft_index
    e_s = vectors[:, soft_index]
    e_h = vectors[:, hard_index]
    if np.linalg.det(np.column_stack([e_h, e_s])) < 0.0:
        e_s = -e_s
    return x, y, e_h, e_s, float(values[hard_index])
