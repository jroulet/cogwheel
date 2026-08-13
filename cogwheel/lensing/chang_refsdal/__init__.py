"""Chang-Refsdal (point mass + external convergence and shear) lens."""

from .geometry import ppgo_error_estimate
from .channels import (ChangRefsdalChannels, born_carrier_from_partition,
                       farfield_envelope_from_partition, real_image_delays)
from .operator import RHO_START, RHO_END
from ._airy_fold import fold_amplification, airy_fold_value, fold_ppgo_correction
from ._pearcey_cusp import cusp_amplification, pearcey, pearcey_asymptotic
