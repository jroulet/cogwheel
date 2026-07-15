#!/usr/bin/env python3
"""Contour-free Chang--Refsdal amplification with external convergence and shear.

For the aligned Chang--Refsdal lens with constant convergence and shear

    tau(x;y) = 1/2 |x-y|^2 - log|x| - gamma/2 (x_1^2-x_2^2),

write F=exp(i w |y|^2/2) G.  Then

    G_CR = exp[i gamma D_beta/(2w)] G_PM,

where D_beta is the rotated traceless second-derivative operator.  G_PM and
all of its radial derivatives are confluent hypergeometric functions.  This
module evaluates the operator series without a lens-plane contour integral.

External convergence is included through an exact mass-sheet rescaling, while
the effective shear is evaluated with the analytic source-plane operator series.
The returned diagnostics should be checked as |gamma|/(1-kappa) approaches unity.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import factorial
from typing import Dict, Tuple

import mpmath as mp
import numpy as np

Monomial = Tuple[int, int, int]  # z power, zbar power, radial derivative order
Representation = Dict[Monomial, complex]


@dataclass(frozen=True)
class OperatorDiagnostics:
    order_used: int
    converged: bool
    last_term_abs: float
    sum_abs: float
    estimated_relative_tail: float


def _apply_rotated_D(rep: Representation, beta: float) -> Representation:
    """Apply D_beta to sum c z^a zbar^b f^(k)(z zbar).

    With z=x+i y,

        D_beta = 2 exp(+2 i beta) d_z^2
               + 2 exp(-2 i beta) d_zbar^2.
    """
    out: Representation = {}
    cz = 2.0 * np.exp(2.0j * beta)
    cb = 2.0 * np.exp(-2.0j * beta)

    def add(key: Monomial, value: complex) -> None:
        out[key] = out.get(key, 0.0j) + value

    for (a, b, k), coeff in rep.items():
        # d_z^2
        if a >= 2:
            add((a - 2, b, k), cz * coeff * a * (a - 1))
        if a >= 1:
            add((a - 1, b + 1, k + 1), cz * coeff * 2 * a)
        add((a, b + 2, k + 2), cz * coeff)

        # d_zbar^2
        if b >= 2:
            add((a, b - 2, k), cb * coeff * b * (b - 1))
        if b >= 1:
            add((a + 1, b - 1, k + 1), cb * coeff * 2 * b)
        add((a + 2, b, k + 2), cb * coeff)
    return out


@lru_cache(maxsize=32)
def operator_representations(max_order: int, beta_key: float = 0.0) -> tuple[Representation, ...]:
    """Representations of D_beta^n f(s), n=0,...,max_order."""
    beta = float(beta_key)
    reps: list[Representation] = [{(0, 0, 0): 1.0 + 0.0j}]
    for _ in range(max_order):
        reps.append(_apply_rotated_D(reps[-1], beta))
    return tuple(reps)


def point_mass_G_derivatives(
    w: float,
    s: float,
    max_derivative: int,
    *,
    dps: int = 60,
) -> np.ndarray:
    r"""Return d^k G_PM/ds^k for k=0,...,max_derivative.

    The raw point-mass amplification is

      F_PM = C(w) 1F1(iw/2;1;iws/2),

    and G_PM=exp(-iws/2)F_PM.  Kummer's transformation gives

      G_PM = C(w) 1F1(1-iw/2;1;-iws/2).
    """
    if w <= 0:
        raise ValueError("w must be positive")
    if s < 0:
        raise ValueError("s must be nonnegative")
    with mp.workdps(dps):
        ww = mp.mpf(w)
        ss = mp.mpf(s)
        a = 1.0 - 0.5j * ww
        z = -0.5j * ww * ss
        pref = (
            mp.e ** (mp.pi * ww / 4.0 + 0.5j * ww * mp.log(ww / 2.0))
            * mp.gamma(1.0 - 0.5j * ww)
        )
        values = np.empty(max_derivative + 1, dtype=complex)
        poch = 1.0 + 0.0j
        fact = 1
        base = -0.5j * ww
        for k in range(max_derivative + 1):
            if k:
                poch *= a + k - 1
                fact *= k
            values[k] = complex(
                pref * base**k * poch / fact * mp.hyp1f1(a + k, 1 + k, z)
            )
    return values


def _evaluate_representation(
    rep: Representation,
    y: np.ndarray,
    radial_derivatives: np.ndarray,
) -> complex:
    z = complex(y[0] + 1j * y[1])
    zbar = z.conjugate()
    total = 0.0j
    for (a, b, k), coeff in rep.items():
        total += coeff * z**a * zbar**b * radial_derivatives[k]
    return total


def chang_refsdal_amplification(
    w: float,
    y: np.ndarray,
    gamma: float,
    *,
    beta: float = 0.0,
    kappa: float = 0.0,
    tolerance: float = 2e-12,
    min_order: int = 6,
    max_order: int = 36,
    consecutive_small_terms: int = 4,
    dps: int = 70,
    return_diagnostics: bool = False,
):
    """Evaluate the exact shear-operator series for one frequency.

    ``beta`` is the orientation of the shear principal axis.  The series is
    stopped after ``consecutive_small_terms`` terms are smaller than
    ``tolerance`` times the current partial sum, but never before
    ``min_order``.  Since the early terms can be nonmonotonic, callers should
    inspect diagnostics when moving beyond the tested parameter range.
    """
    y = np.asarray(y, dtype=float)
    if y.shape != (2,):
        raise ValueError("y must be a two-vector")
    lam = 1.0 - float(kappa)
    if lam <= 0.0 or abs(gamma) >= lam:
        raise ValueError("this implementation assumes 1-kappa > |gamma|")

    # Exact mass-sheet mapping to the pure-shear Chang--Refsdal problem:
    # x'=sqrt(lam)x, y'=y/sqrt(lam), gamma'=gamma/lam.  With the standard
    # w/(2 pi i) normalization,
    # F_{kappa,gamma}=lam^{-1} exp[i w(ln lam/2-kappa|y|^2/(2lam))]
    #                  F_{0,gamma/lam}(w,y/sqrt(lam)).
    y_scaled = y / np.sqrt(lam)
    gamma_scaled = gamma / lam
    reps = operator_representations(max_order, float(beta))
    derivs = point_mass_G_derivatives(w, float(y_scaled @ y_scaled), 2 * max_order, dps=dps)

    total = 0.0j
    coeff = 1.0 + 0.0j
    small_count = 0
    converged = False
    last_term = 0.0j
    used = 0
    for n, rep in enumerate(reps):
        if n:
            coeff *= 1j * gamma_scaled / (2.0 * w * n)
        term = coeff * _evaluate_representation(rep, y_scaled, derivs)
        total += term
        last_term = term
        used = n
        scale = max(abs(total), 1e-300)
        if n >= min_order and abs(term) <= tolerance * scale:
            small_count += 1
            if small_count >= consecutive_small_terms:
                converged = True
                break
        else:
            small_count = 0

    phase_scaled = np.exp(0.5j * w * float(y_scaled @ y_scaled))
    mass_sheet_phase = np.exp(
        0.5j * w * np.log(lam)
        - 0.5j * w * kappa * float(y @ y) / lam
    )
    result = mass_sheet_phase * phase_scaled * total / lam
    diag = OperatorDiagnostics(
        order_used=used,
        converged=converged,
        last_term_abs=float(abs(last_term)),
        sum_abs=float(abs(total)),
        estimated_relative_tail=float(abs(last_term) / max(abs(total), 1e-300)),
    )
    if return_diagnostics:
        return result, diag
    return result


def amplification_grid(w, y, gamma, **kwargs):
    """Evaluate ``chang_refsdal_amplification`` on a one-dimensional grid."""
    w = np.asarray(w, dtype=float)
    values = np.empty(w.shape, dtype=complex)
    diagnostics: list[OperatorDiagnostics] = []
    for index, wi in np.ndenumerate(w):
        value, diag = chang_refsdal_amplification(
            float(wi), y, gamma, return_diagnostics=True, **kwargs
        )
        values[index] = value
        diagnostics.append(diag)
    return values, diagnostics
