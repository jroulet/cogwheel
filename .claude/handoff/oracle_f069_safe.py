"""Independent exact oracle (f_schwinger + mass-sheet reconstruction) and
the ppGO w^-3 certificate."""
import sys
import numpy as np

sys.path.insert(0, '/home/tejaswi/Work/cogwheel-claude-dev')
from cogwheel.lensing.chang_refsdal import geometry, _schwinger, operator
from cogwheel.lensing.chang_refsdal.operator import _mass_sheet_map

sys.path.insert(0, '/tmp/ppgo_cert')
from derive import series_coefficients


def exact_total(w, y, gamma, beta=0.0, kappa=0.0):
    """Mirror of operator._positive_parity_grid's gamma' > 0 route, but
    ALWAYS through f_schwinger (never the uniform arm).  F069-safe."""
    lam, y_scaled, gamma_prime = _mass_sheet_map(y, gamma, kappa)
    z_eig = np.exp(-1j * float(beta)) * complex(y_scaled[0], y_scaled[1])
    y_eig = np.array([z_eig.real, z_eig.imag])
    s = float(y_scaled @ y_scaled)
    out = np.empty(np.shape(w), dtype=complex)
    flat = np.atleast_1d(np.asarray(w, dtype=float))
    vals = np.empty(flat.shape, dtype=complex)
    for i, wi in enumerate(flat):
        f_pure = _schwinger.f_schwinger(float(wi), y_eig, gamma_prime)
        phase = np.exp(0.5j * wi * np.log(lam) - 0.5j * wi * float(kappa) * s)
        vals[i] = complex(phase * f_pure / lam)
    return vals.reshape(np.shape(w)) if np.ndim(w) else complex(vals[0])


def ppgo(w, y, gamma, beta=0.0, kappa=0.0):
    return np.atleast_1d(operator.geometric_amplification(
        np.asarray(w, dtype=float), np.asarray(y, dtype=float), gamma,
        beta=beta, kappa=kappa))


def image_data(y, gamma, beta=0.0, kappa=0.0):
    """Per-image (amplitude, delay, morse, C1, C2, c3)."""
    matrix = geometry.macro_matrix(gamma, beta, kappa)
    images = geometry.find_images(np.asarray(y, dtype=float), matrix)
    rows = []
    for im in images:
        amp = np.sqrt(abs(geometry.magnification(im, matrix)))
        tau = geometry.delay(im, np.asarray(y, dtype=float), matrix)
        n = geometry.morse_index(im, matrix)
        c1s, c2s = geometry.saddle_coefficients(im, matrix)
        _c1, _c2, c3 = series_coefficients(im, matrix)
        rows.append(dict(image=im, amp=amp, tau=tau, morse=n,
                         c1=c1s, c2=c2s, c3=c3))
    return rows


def certificate(rows, w):
    """Incoherent (triangle-inequality) bound on the leading uncertified
    O(w^-3) term of the ppGO sum, in the SAME absolute units as
    |F - ppGO|."""
    w = np.asarray(w, dtype=float)
    return sum(r['amp'] * abs(r['c3']) for r in rows) / w ** 3


def certificate_coherent(rows, w):
    """Coherent version: the actual leading residual, phases included."""
    w = np.asarray(w, dtype=float)
    total = np.zeros(np.shape(w), dtype=complex)
    for r in rows:
        total = total + (r['amp']
                         * np.exp(1j * (w * r['tau']
                                        - 0.5 * np.pi * r['morse']))
                         * r['c3'])
    return np.abs(total) / w ** 3


def certificate_c2(rows, w):
    """For comparison: magnitude of the LAST INCLUDED term (order w^-2)."""
    w = np.asarray(w, dtype=float)
    return sum(r['amp'] * abs(r['c2']) for r in rows) / w ** 2
