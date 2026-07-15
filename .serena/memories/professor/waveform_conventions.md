# Waveform models and conventions

## IMRPhenomXPHM — precession + higher harmonics (2004.06503)
Phenomenological frequency-domain model for quasi-circular precessing BBH with
sub-dominant harmonics. Precessing extension of IMRPhenomXHM (which adds (l,|m|) =
(2,2),(2,1),(3,3),(3,2),(4,4) to the (2,2)-only IMRPhenomXAS). Built by "twisting-up":
aligned-spin modes modeled in the co-precessing L-frame (XAS/XHM), then rotated by three
frequency-dependent Euler angles (alpha, beta, gamma) into the inertial J-frame via
Wigner-D matrices, h^J_lm = sum D^l_{mm'} h^L_{lm'}. IMRPhenomXP is the dominant-quadrupole
case. beta tracks the orbital angular momentum opening angle, alpha the precession about J,
gamma gauge-fixed by the minimal-rotation condition. Frequency-domain via SPA. Efficiency:
multibanding/interpolation of BOTH modes and the Euler angles. Two precession prescriptions:
NNLO single-spin PN angles (as Pv2) or double-spin MSA angles; 4PN aligned-spin L for beta.

## Conventions gotchas for PE
- LALSuite Fourier convention h(f) = integral h(t) e^{-2 pi i f t} dt. Inertial J-frame,
  observer at (theta_JN, phi_JN). Units: Hz, solar masses, Mpc, radians.
- THE Pv2-vs-XP ISSUE (critical for cogwheel): although XPHM can inherit Pv2's NNLO
  prescription as one option, its phase/polarization convention is self-consistent with
  LALSuite. Pv2 uses a DIFFERENT phase convention (LIGO-T1500602) that is inconsistent
  with cogwheel's sampled-vs-standard coordinate mapping — hence cogwheel deliberately uses
  IMRPhenomXP, NOT Pv2 (CLAUDE.md engineering value #3). Any coordinate code inferring
  polarization psi, coalescence phase phi_ref, or reference frequency f_ref MUST match the
  waveform model's convention exactly, or the sampled<->standard transform silently biases PE.
- f_ref fixes spin orientation / Euler-angle constants — spins are defined AT f_ref, not
  f_min. Declared mode content affects amplitude/phase and must be consistent.

## Template-bank geometry (1904.01683)
Geometric template placement using a metric where mismatch d^2 = 1 - match is Euclidean.
Unwrapped phases psi(f) are smooth in binary parameters -> lie in a low-dimensional linear
space. Group waveforms by amplitude profile into subbanks; SVD of weighted phase residuals
yields orthonormal basis psi_alpha; each waveform maps to coefficients c_alpha, placed on a
regular grid (delta c ~ 1). The leading c-dimension correlates with chirp mass (best-measured
parameter); number of significant SVD dimensions ~ number of measurable parameters. In
high-SNR the likelihood is ~ exp(-rho^2 |z|^2/2), isotropic Gaussian in c with width ~1/rho.
Relevance: the geometric/metric backdrop for why cogwheel's sampled coordinates aim for
near-Euclidean, well-conditioned parameterizations of the waveform manifold.

Sources: 2004.06503, 1904.01683.
