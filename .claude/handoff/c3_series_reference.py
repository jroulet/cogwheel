"""Derive the stationary-phase series coefficients C1, C2, C3 at an image.

tau(x) = 0.5 x.A.x - y.x + 0.5 y.y - ln|x|
F(w) = (w / 2 pi i) int d2x exp(i w tau)

At an image x_a:  F_a = sqrt|mu| exp(i w tau_a - i pi n /2) * Corr(w)
Corr(w) = < exp(i w V(xi)) >  with covariance Sigma = i H^{-1} / w,
V(xi) = tau(x_a + xi) - tau_a - 0.5 xi.H.xi  (comes ONLY from -ln|x|).

Rescale xi = eps * eta, eps = w^{-1/2}:  covariance in eta is i H^{-1}
(w-independent), and

    i w V = i sum_{n>=3} eps^(n-2) * ((-1)^n / n) * Re[ ((eta1+i eta2)/z_a)^n ]

so Corr = 1 + c1 eps^2 + c2 eps^4 + c3 eps^6 + ...
        = 1 + c1/w + c2/w^2 + c3/w^3
and the shipped kernel carries 1 + i*C1/w + C2/w^2, i.e. c1 = i*C1, c2 = C2.
"""
import math
from itertools import product

import numpy as np

MAXEPS = 6  # keep terms through eps^6 == w^-3


def pmul(p, q, maxeps=MAXEPS):
    out = {}
    for (a1, b1, e1), v1 in p.items():
        if v1 == 0:
            continue
        for (a2, b2, e2), v2 in q.items():
            e = e1 + e2
            if e > maxeps:
                continue
            k = (a1 + a2, b1 + b2, e)
            out[k] = out.get(k, 0j) + v1 * v2
    return out


def padd(p, q):
    out = dict(p)
    for k, v in q.items():
        out[k] = out.get(k, 0j) + v
    return out


def pscale(p, s):
    return {k: v * s for k, v in p.items()}


def linear_power(c1, c2, n, maxeps=MAXEPS):
    """(c1*eta1 + c2*eta2)^n as a polynomial dict (eps degree 0)."""
    out = {}
    for k in range(n + 1):
        out[(n - k, k, 0)] = math.comb(n, k) * c1 ** (n - k) * c2 ** k
    return out


def gaussian_moment_table(sigma, maxdeg):
    """<eta1^a eta2^b> for a bivariate Gaussian with covariance `sigma`."""
    s11, s12, s22 = sigma[0, 0], sigma[0, 1], sigma[1, 1]
    mom = {}
    for a in range(maxdeg + 1):
        for b in range(maxdeg + 1 - a):
            if (a + b) % 2:
                mom[(a, b)] = 0j
                continue
            m = (a + b) // 2
            tot = 0j
            for i in range(m + 1):
                for j in range(m + 1 - i):
                    k = m - i - j
                    if 2 * i + j != a or j + 2 * k != b:
                        continue
                    tot += (s11 ** i * (2 * s12) ** j * s22 ** k
                            / (math.factorial(i) * math.factorial(j)
                               * math.factorial(k)))
            mom[(a, b)] = tot * math.factorial(a) * math.factorial(b) / 2 ** m
    return mom


def series_coefficients(image, matrix, maxeps=MAXEPS):
    """Return (c1, c2, c3) with Corr = 1 + c1/w + c2/w^2 + c3/w^3."""
    image = np.asarray(image, dtype=float)
    hess = (matrix - np.eye(2) / (image @ image)
            + 2.0 * np.outer(image, image) / (image @ image) ** 2)
    sigma = 1j * np.linalg.inv(hess)

    z = complex(image[0], image[1])
    # P = i w V  in the eps/eta variables.
    poly = {}
    nmax = 2 * maxeps + 2          # highest single derivative order needed
    for n in range(3, nmax + 1):
        coef = ((-1) ** n / (2.0 * n))
        for zz in (z, np.conj(z)):
            # u = (eta1 + i eta2)/z   or   (eta1 - i eta2)/conj(z)
            sgn = 1j if zz is z else -1j
            term = linear_power(1.0 / zz, sgn / zz, n)
            term = {(a, b, e + n - 2): v * coef * 1j
                    for (a, b, e), v in term.items() if e + n - 2 <= maxeps}
            poly = padd(poly, term)

    # exp(poly), truncated: poly has min eps-degree 1.
    result = {(0, 0, 0): 1.0 + 0j}
    powk = {(0, 0, 0): 1.0 + 0j}
    for k in range(1, maxeps + 1):
        powk = pmul(powk, poly, maxeps)
        result = padd(result, pscale(powk, 1.0 / math.factorial(k)))

    maxdeg = max(a + b for (a, b, e) in result)
    mom = gaussian_moment_table(sigma, maxdeg)
    out = [0j] * (maxeps + 1)
    for (a, b, e), v in result.items():
        out[e] += v * mom[(a, b)]
    return out[2], out[4], out[6]


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/tejaswi/Work/cogwheel-claude-dev')
    from cogwheel.lensing.chang_refsdal import geometry

    rng = np.random.default_rng(0)
    worst1 = worst2 = 0.0
    for gamma in (0.2, 0.5, 0.8):
        for _ in range(6):
            y = rng.uniform(-1.2, 1.2, 2)
            mat = geometry.macro_matrix(gamma, 0.0, 0.0)
            try:
                images = geometry.find_images(y, mat)
            except geometry.LensDomainError:
                continue
            for im in images:
                c1s, c2s = geometry.saddle_coefficients(im, mat)
                c1, c2, c3 = series_coefficients(im, mat)
                # c1 should be i*C1 (pure imaginary), c2 should be C2 (real)
                e1 = abs(c1 / 1j - c1s) / max(1.0, abs(c1s))
                e2 = abs(c2 - c2s) / max(1.0, abs(c2s))
                worst1 = max(worst1, e1)
                worst2 = max(worst2, e2)
                print(f'gamma={gamma} y=({y[0]:+.3f},{y[1]:+.3f}) '
                      f'C1 shipped={c1s:+.6e} derived={(c1/1j).real:+.6e} '
                      f'(imagpart {(c1/1j).imag:+.1e})  '
                      f'C2 shipped={c2s:+.6e} derived={c2.real:+.6e} '
                      f'(imagpart {c2.imag:+.1e})  '
                      f'c3={c3:+.6e}')
    print(f'\nWORST relative disagreement: C1 {worst1:.2e}, C2 {worst2:.2e}')
