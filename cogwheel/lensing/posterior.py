"""
Sampling-ready posterior for microlensed compact-binary events.

`LensedPosterior` is the microlensing counterpart of `posterior.Posterior`:
it pairs a lens-aware prior (`lensing.prior.LensedIASPrior`) with the
microlensed relative-binning likelihood and adds a single, tightly scoped
refusal net.  The Chang--Refsdal engine and likelihood keep their
"certified-or-named-refusal" contract -- `geometry.LensDomainError` (macro
saddle / negative parity) and `operator.CancellationError` (uncertifiable
wave-branch contraction) propagate unswallowed everywhere except here, at the
boundary where the sampler meets the posterior.  A proposal that trips either
named refusal is mapped to ``lnL = -inf`` so the sampler rejects it, exactly as
it treats a point of zero prior support (Professor constraint 2).
"""
from __future__ import annotations

import numpy as np

from cogwheel.posterior import Posterior
from cogwheel.lensing.chang_refsdal.geometry import LensDomainError
from cogwheel.lensing.chang_refsdal.operator import CancellationError

__all__ = ['LensedPosterior']


class LensedPosterior(Posterior):
    """
    `Posterior` that maps lens-engine named refusals to ``lnL = -inf``.

    Identical to `posterior.Posterior` except that
    `lnposterior_pardic_and_metadata` catches the two named refusals the
    microlensing engine may raise for an in-support proposal
    (`geometry.LensDomainError`, `operator.CancellationError`) and returns the
    same ``(-inf, standard_par_dic, None)`` triple the base class returns for a
    point of zero prior density.  This is the ONLY site at which those refusals
    are swallowed; the engine and likelihood keep raising them.  No refusal
    counter is kept -- the ``-inf`` entries in the sample array are the reliable,
    fork-safe record of refused proposals.
    """

    def lnposterior_pardic_and_metadata(self, *args, **kwargs):
        """
        Log posterior, standard parameters and metadata; refusals -> -inf.

        Pass through to `Posterior.lnposterior_pardic_and_metadata`.  If the
        wrapped evaluation raises a named lens-engine refusal
        (`geometry.LensDomainError` or `operator.CancellationError`), return
        ``(-inf, standard_par_dic, None)`` -- the same shape the base class
        returns for a zero-prior-density point -- instead of propagating the
        exception into the sampler.  The coordinate transform recomputed here to
        recover ``standard_par_dic`` maps sampled to standard parameters only and
        never touches the lens engine, so it cannot itself raise a refusal.

        Parameters
        ----------
        *args, **kwargs
            Sampled parameters (signature bound to the prior transform by
            `Posterior.__init__`).

        Returns
        -------
        lnposterior : float
            Natural logarithm of the posterior probability density, or
            ``-inf`` for a zero-prior point or a refused proposal.
        standard_par_dic : dict
            Standard parameters.
        metadata : object or None
            Likelihood metadata, or ``None`` for an ``-inf`` return.
        """
        try:
            return super().lnposterior_pardic_and_metadata(*args, **kwargs)
        except (LensDomainError, CancellationError):
            standard_par_dic = self.prior.transform(*args, **kwargs)
            return -np.inf, standard_par_dic, None
