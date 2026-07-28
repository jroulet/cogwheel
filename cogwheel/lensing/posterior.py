"""
Sampling-ready posterior for microlensed compact-binary events.

`LensedPosterior` is the microlensing counterpart of `posterior.Posterior`:
it pairs a lens-aware prior (`lensing.prior.LensedIASPrior`) with the
microlensed relative-binning likelihood and adds a single, tightly scoped
refusal net.  The Chang--Refsdal engine and likelihood keep their
"certified-or-named-refusal" contract -- `geometry.LensDomainError` (lens
domain violations, incl. image-census and fold-degeneracy refusals),
`operator.CancellationError` (uncertifiable wave-branch contraction),
`_schwinger.SchwingerCertificationError` (the Schwinger evaluator's
paired-rule certificate failed; reachable sub-ceiling near the
``gamma' -> 1`` pinch since the Build 7a strong-shear fallback), and
`likelihood.LensedBinningError` (a candidate image delay the certified bins
cannot resolve; reachable for in-support strong-shear proposals since the
same fallback widened the evaluable set) propagate unswallowed everywhere
except here, at the boundary where the sampler meets the posterior.  A
proposal that trips a named refusal is mapped to ``lnL = -inf`` so the
sampler rejects it, exactly as it treats a point of zero prior support
(Professor constraint 2).
"""
from __future__ import annotations

import numpy as np

from cogwheel.posterior import Posterior
from cogwheel.lensing.chang_refsdal.geometry import LensDomainError
from cogwheel.lensing.chang_refsdal.operator import CancellationError
from cogwheel.lensing.chang_refsdal._schwinger import (
    SchwingerCertificationError)
from cogwheel.lensing.likelihood import LensedBinningError

__all__ = ['LensedPosterior']


class LensedPosterior(Posterior):
    """
    `Posterior` that maps lens-engine named refusals to ``lnL = -inf``.

    Identical to `posterior.Posterior` except that
    `lnposterior_pardic_and_metadata` catches the named refusals the
    microlensing engine and likelihood may raise for an in-support proposal
    (the `_NAMED_REFUSALS` vocabulary) and returns the
    same ``(-inf, standard_par_dic, None)`` triple the base class returns for a
    point of zero prior density.  This is the ONLY site at which those refusals
    are swallowed; the engine and likelihood keep raising them.  No refusal
    counter is kept -- the ``-inf`` entries in the sample array are the
    reliable, fork-safe record of refused proposals.
    """

    def lnposterior_pardic_and_metadata(self, *args, **kwargs):
        """
        Log posterior, standard parameters and metadata; refusals -> -inf.

        Pass through to `Posterior.lnposterior_pardic_and_metadata`.  If the
        wrapped evaluation raises a named lens-engine or likelihood refusal
        (any member of `_NAMED_REFUSALS`), return
        ``(-inf, standard_par_dic, None)`` -- the same shape the base class
        returns for a zero-prior-density point -- instead of propagating the
        exception into the sampler.  The coordinate transform recomputed
        here to recover ``standard_par_dic`` maps sampled to standard
        parameters only and never touches the lens engine, so it cannot
        itself raise a refusal.

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
        # The closed named-refusal vocabulary mapped to -inf: anything
        # NOT listed (e.g. a raw LinAlgError) is a bug and must crash
        # loudly (FINDINGS F015).  The tuple is built HERE, at raise
        # time, from module globals -- never hoisted to an import-time
        # constant -- so the net stays falsifiable by patching a module
        # global (the F010-style mutation test relies on exactly that).
        except (LensDomainError, CancellationError,
                SchwingerCertificationError, LensedBinningError):
            standard_par_dic = self.prior.transform(*args, **kwargs)
            return -np.inf, standard_par_dic, None
