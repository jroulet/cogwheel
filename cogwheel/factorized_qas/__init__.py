import textwrap
import warnings

from .marg_likelihood import MarginalizedRelativeBinningLikelihood
from .intrinsic_prior import IntrinsicParametersPrior, IntrinsicTidalPrior

warnings.warn(textwrap.dedent('''
    The `factorized_qas` subpackage is deprecated.
    The preferred way of marginalizing extrinsic parameters is now
    `cogwheel.likelihood.MarginalizedExtrinsicLikelihoodQAS`
    coupled to e.g. `gw_prior.IntrinsicAlignedSpinIASPrior`.
    See  `tutorials/factorized_qas.ipynb` for an example.'''),
              DeprecationWarning, stacklevel=2)
