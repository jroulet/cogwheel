---
bump: patch
---

### Microlensing engine complete (Build 1b)

The `cogwheel/lensing/chang_refsdal/` layer is now complete. Build 1b adds
the three production modules the foundation was missing: the dd-accumulated
complex-1F1 kernel (`_hyp1f1.py`), the contour-free amplification operator
`F_op` (`operator.py`), and the topology-stable `ChangRefsdalChannels`
tracker (`channels.py`), which is the single public entry point re-exported
from the package `__init__`. These sit on the committed foundation —
double-double arithmetic (`_dd.py`), exact gauge/cluster-split channel
algebra (`_gauge.py`), and image geometry (`geometry.py`). The engine is
certified for positive-parity macro images only (`1 - kappa > |gamma|`;
macro saddles are out of scope), a frequency ceiling `w <= 500`, and a
double-double product ceiling `w*sqrt(s) <= 60`; above `w*delta_min >= 4.0`
with cancellation exponent `L > 48` the operator hands off to the
geometric-optics branch. Builds 2 (lensed waveform generator + multi-component
relative binning) and 3 (sampled lens coordinates + injection-recovery
validation) remain pending.
