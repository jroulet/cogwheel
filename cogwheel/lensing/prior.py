"""
Modular subpriors for microlensed compact-binary parameter estimation.

These priors implement the locked reduced-coordinate parametrization for a
Chang--Refsdal (point-mass + external-shear) microlens.  They are the
sampling-layer counterpart of `cogwheel.lensing.likelihood` and
`cogwheel.lensing.waveform`, and are meant to be composed with the standard
GW subpriors by subclassing `cogwheel.prior.CombinedPrior` (that combination,
its registration, and any posterior wiring live in a later work package -- this
module only defines the individual lens subpriors).

Reduced parametrization (WHY these coordinates)
-----------------------------------------------
The full Chang--Refsdal geometry has parameters
``{m_lens_msun, z_lens, y1, y2, gamma, beta, kappa}`` (the seven the lensed
waveform/likelihood consume).  Three of them are eliminated rather than sampled:

* ``kappa`` (convergence) is exactly mass-sheet degenerate -- rescaling it is
  absorbed into the source amplitude, so it carries no independent posterior
  information and is fixed to ``0``.
* ``beta`` (shear orientation) is a circular degree of freedom of the point
  mass; the physical content lives in the shear-frame source position, so
  ``beta`` is fixed to ``0`` and the source angle is sampled in the shear frame.
* ``z_lens`` (lens redshift) enters observables only through the dimensionless
  frequency ``w = xi(m_lens_msun, z_lens) * f`` (see
  `lensing.waveform.dimensionless_frequency`); it is therefore folded into the
  redshifted lens mass and fixed to ``0`` here, with ``m_lens_msun`` carrying
  the redshifted combination.

The remaining sampled coordinates are the redshifted lens mass
(``ln_m_lens_msun``), the reduced shear (``gamma``), and the shear-frame source
position on a fixed unit box (``u1, u2``) rescaled by a mass-dependent scale.

Units
-----
``m_lens_msun`` is the redshifted-frame lens mass in solar masses; ``y1, y2``
are the dimensionless shear-frame source-position components (Einstein-radius
units); ``gamma`` is the dimensionless reduced shear; ``kappa``, ``beta`` and
``z_lens`` are dimensionless / radians and fixed.

Jacobian convention
--------------------
Following the cogwheel subprior contract (see
`gw_prior.mass.UniformDetectorFrameMassesPrior` and
`gw_prior.extrinsic.UniformLuminosityVolumePrior`), ``ln_jacobian_determinant``
takes ``standard_params + conditioned_on`` values and returns the log-Jacobian
of the *inverse* transform, ``log|d{sampled} / d{standard}|``.
"""
from __future__ import annotations

import numpy as np

from cogwheel import utils
from cogwheel.gw_prior import IASPrior, IntrinsicIASPrior, RegisteredPriorMixin
from cogwheel.prior import (CombinedPrior, FixedPrior, IdentityTransformMixin,
                            Prior, UniformPriorMixin)
from cogwheel.lensing.likelihood import LensedRelativeBinningLikelihood
from cogwheel.lensing.marginalized_likelihood import (
    LensedMarginalizedExtrinsicLikelihood)

# pylint: disable=arguments-differ

#: Sampling range for ``ln_m_lens_msun`` (natural log of the redshifted lens
#: mass in solar masses), ``(log(10), log(3500))``.  Provenance: the upper
#: bound 3500 Msun keeps ``w = xi(m_lens_msun, z_lens) * f`` below the engine's
#: certified ceiling ``w <= 500`` across the LIGO band for the crown-config
#: source distance (see the Professor memory ``priors_and_coordinates``); the
#: lower bound 10 Msun keeps the diffraction feature in band (lighter lenses
#: push ``w`` too low for an observable signature).
_LN_M_LENS_RANGE = (np.log(10.0), np.log(3500.0))

#: Source-position scale numerator (dimensionless, Einstein-radius units).
#: ``307 = 55 / (sqrt(2) * 1.2372e-4 * 1024)`` keeps the double-double product
#: ``w * sqrt(s)`` below the engine's certified ceiling of 60 at the *corner*
#: of the ``[-1, 1]^2`` sampling box (the ``sqrt(2)`` is the box-corner factor),
#: so ``Y(m) = _Y_SCALE / m_lens_msun`` shrinks the source box as the lens mass
#: (hence ``w``) grows.
_Y_SCALE = 307.0

#: Upper cap on the source-position scale ``Y(m)`` (dimensionless).  For small
#: lens masses ``_Y_SCALE / m_lens_msun`` would grow without bound; the cap
#: keeps ``|y|`` bounded to a physically sensible few Einstein radii.
_Y_SCALE_CAP = 3.0


def _source_scale(m_lens_msun: float) -> float:
    """
    Return the shear-frame source-position scale ``Y(m)`` (dimensionless).

    ``y_i = u_i * Y(m)`` maps the unit sampling box ``[-1, 1]^2`` onto the
    physical source-position box, shrinking as the redshifted lens mass grows
    so the certified ``w * sqrt(s) <= 60`` ceiling is respected at the corner.

    Parameters
    ----------
    m_lens_msun : float
        Redshifted-frame lens mass in solar masses.

    Returns
    -------
    float
        Source-position scale ``min(_Y_SCALE / m_lens_msun, _Y_SCALE_CAP)``.
    """
    return min(_Y_SCALE / m_lens_msun, _Y_SCALE_CAP)


class FixedLensGeometryPrior(FixedPrior):
    """
    Fix the eliminated lens-geometry parameters to their reduced values.

    ``kappa`` (mass-sheet degenerate), ``beta`` (circular point-mass
    orientation) and ``z_lens`` (folded into the redshifted lens mass via
    ``w = xi(m_lens_msun, z_lens) * f``) carry no independent posterior
    information in the reduced parametrization, so they are held fixed rather
    than sampled.  All three are dimensionless / radians.
    """
    standard_par_dic = {'kappa': 0.0, 'beta': 0.0, 'z_lens': 0.0}


class UniformLensMassPrior(UniformPriorMixin, Prior):
    """
    Log-uniform prior on the redshifted lens mass.

    The sampled parameter ``ln_m_lens_msun`` is uniform on
    ``_LN_M_LENS_RANGE``; the standard parameter ``m_lens_msun`` is the
    redshifted-frame lens mass in solar masses.  A prior uniform in
    ``ln_m_lens_msun`` is uniform in the decades of lens mass (Jeffreys-like),
    matching the scale-free nature of the diffraction feature.
    """
    standard_params = ['m_lens_msun']
    range_dic = {'ln_m_lens_msun': _LN_M_LENS_RANGE}

    @staticmethod
    @utils.lru_cache()
    def transform(ln_m_lens_msun: float) -> dict:
        """``ln_m_lens_msun`` to ``m_lens_msun`` (solar masses)."""
        return {'m_lens_msun': np.exp(ln_m_lens_msun)}

    @staticmethod
    def inverse_transform(m_lens_msun: float) -> dict:
        """``m_lens_msun`` (solar masses) to ``ln_m_lens_msun``."""
        return {'ln_m_lens_msun': np.log(m_lens_msun)}

    @staticmethod
    def ln_jacobian_determinant(m_lens_msun: float) -> float:
        """
        Natural log Jacobian determinant of the inverse transform.

        Returns
        -------
        float : log|d{ln_m_lens_msun} / d{m_lens_msun}| = -log(m_lens_msun)
        """
        return -np.log(m_lens_msun)


class UniformReducedShearPrior(UniformPriorMixin, IdentityTransformMixin,
                              Prior):
    """
    Uniform prior on the reduced shear ``gamma``.

    ``gamma`` is dimensionless and is both the sampled and the standard
    coordinate: the reduced shear the engine consumes *is* the sampled
    quantity, so the transform is the identity (no ``gamma_prime`` indirection).
    The range ``[0, 0.45]`` leaves positive-parity margin (``1 - kappa = 1``
    stays above ``gamma`` with headroom); the residual approach toward the
    engine's ``operator.CancellationError`` band near ``gamma ~ 0.5`` is caught
    by the posterior-boundary refusal net rather than excluded here.
    """
    range_dic = {'gamma': (0.0, 0.45)}


class UniformSourcePositionPrior(UniformPriorMixin, Prior):
    """
    Uniform prior on the shear-frame source position.

    The sampled parameters ``u1, u2`` are uniform on the fixed unit box
    ``[-1, 1]^2`` and map to the physical shear-frame source-position
    components ``y1, y2`` (dimensionless, Einstein-radius units) via the
    mass-dependent scale ``Y(m) = min(_Y_SCALE / m_lens_msun, _Y_SCALE_CAP)``.
    Sampling on the fixed box (rather than directly on ``y``) keeps the prior
    support conditioning simple and the certified ``w * sqrt(s) <= 60`` ceiling
    respected at the box corner.

    The astroid caustic has an exact quadrant symmetry in the shear-frame
    source angle, so ``u1`` and ``u2`` are declared as folded-reflected
    parameters for cogwheel's standard folding machinery.  No phase fold is
    declared: the constant-lens-phase / orbital-phase degeneracy is a
    22-mode-only relation and must not be assumed for IMRPhenomXPHM higher
    modes.
    """
    standard_params = ['y1', 'y2']
    range_dic = {'u1': (-1.0, 1.0),
                 'u2': (-1.0, 1.0)}
    conditioned_on = ['m_lens_msun']
    folded_reflected_params = ['u1', 'u2']

    @staticmethod
    @utils.lru_cache()
    def transform(u1: float, u2: float, m_lens_msun: float) -> dict:
        """``(u1, u2)`` on the unit box to shear-frame ``(y1, y2)``."""
        scale = _source_scale(m_lens_msun)
        return {'y1': u1 * scale,
                'y2': u2 * scale}

    @staticmethod
    def inverse_transform(y1: float, y2: float, m_lens_msun: float) -> dict:
        """Shear-frame ``(y1, y2)`` to ``(u1, u2)`` on the unit box."""
        scale = _source_scale(m_lens_msun)
        return {'u1': y1 / scale,
                'u2': y2 / scale}

    @staticmethod
    def ln_jacobian_determinant(y1: float, y2: float,
                                m_lens_msun: float) -> float:
        """
        Natural log Jacobian determinant of the inverse transform.

        Returns
        -------
        float : log|d{u1, u2} / d{y1, y2}| = -2 * log(Y(m_lens_msun))
        """
        del y1, y2  # Jacobian depends only on the conditioning mass.
        return -2.0 * np.log(_source_scale(m_lens_msun))


class LensedIASPrior(RegisteredPriorMixin, CombinedPrior):
    """
    IAS compact-binary prior composed with the reduced lens subpriors.

    Combines the standard precessing IAS prior (`gw_prior.IASPrior`) over the
    compact-binary parameters with the four reduced-coordinate Chang--Refsdal
    lens subpriors defined in this module, yielding a single registered prior
    over the full CBC + microlens parameter set consumed by
    `lensing.likelihood.LensedRelativeBinningLikelihood`.  It is registered in
    ``gw_prior.prior_registry`` (via `RegisteredPriorMixin`) and its
    ``standard_params`` match that likelihood's ``params`` exactly, so it pairs
    with the lensed likelihood inside a `lensing.posterior.LensedPosterior`.

    The lens subpriors are appended AFTER the CBC subpriors, and
    ``UniformLensMassPrior`` precedes ``UniformSourcePositionPrior`` because the
    latter is conditioned on ``m_lens_msun`` (produced by the former);
    `prior.CombinedPrior` requires every conditioned-on parameter to be supplied
    by an earlier subprior.

    Distance is sampled in physical luminosity volume: the CBC
    ``UniformLuminosityVolumePrior`` is reused unchanged, so the sampled distance
    is the physical ``d_luminosity`` rather than the lens-corrected apparent
    distance ``d_app = d_luminosity / sqrt(mu_macro)``.  A lens-aware
    apparent-distance subprior that absorbs the macro-magnification amplitude is
    DEFERRED to Build 5.
    """
    default_likelihood_class = LensedRelativeBinningLikelihood

    prior_classes = IASPrior.prior_classes + [
        FixedLensGeometryPrior,
        UniformLensMassPrior,
        UniformReducedShearPrior,
        UniformSourcePositionPrior]


class LensedMarginalizedExtrinsicIASPrior(RegisteredPriorMixin, CombinedPrior):
    """
    Intrinsic IAS + reduced-lens prior for the marginalized lensed likelihood.

    Microlensing counterpart of `gw_prior.IntrinsicIASPrior`: it composes the
    INTRINSIC precessing IAS compact-binary subpriors (masses, effective spin,
    in-plane spins with isotropic inclination, zero tides, reference frequency)
    with the four reduced-coordinate Chang--Refsdal lens subpriors defined in
    this module.  The extrinsic parameters (sky location, arrival time,
    polarization, distance and orbital phase) are NOT sampled: they are
    marginalized semi-analytically by the coherent score inside
    `lensing.marginalized_likelihood.LensedMarginalizedExtrinsicLikelihood` and
    resampled from the conditional posterior in postprocessing.

    Unlike `LensedIASPrior` (which pairs the FULL IAS prior with the plain
    `LensedRelativeBinningLikelihood`), this prior samples only the intrinsic
    CBC + lens sector, so its ``standard_params`` match the marginalized
    likelihood's ``params`` exactly and it pairs with it inside a
    `lensing.posterior.LensedPosterior`.

    The CBC subpriors are reused verbatim from `IntrinsicIASPrior.prior_classes`
    (DRY: the intrinsic subprior list is defined once, in `gw_prior`).  The lens
    subpriors are appended AFTER them, with ``UniformLensMassPrior`` preceding
    ``UniformSourcePositionPrior`` because the latter is conditioned on
    ``m_lens_msun`` (produced by the former); `prior.CombinedPrior` requires
    every conditioned-on parameter to be supplied by an earlier subprior.

    Distance convention: the coherent score marginalizes and draws
    ``d_luminosity``, but for a lensed signal that column is the APPARENT
    luminosity distance ``d_app = d_luminosity / sqrt(mu_macro)`` (F009); the
    physical-distance conversion and apparent-vs-physical prior reweighting are
    deferred to post-analysis (see the likelihood's distance-convention note).
    """
    default_likelihood_class = LensedMarginalizedExtrinsicLikelihood

    prior_classes = IntrinsicIASPrior.prior_classes + [
        FixedLensGeometryPrior,
        UniformLensMassPrior,
        UniformReducedShearPrior,
        UniformSourcePositionPrior]
