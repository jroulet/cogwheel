---
bump: patch
---

### Lensed relative-binning likelihood: correct through cusp and fold via the channel-switch fix (Build 2c)

`LensedRelativeBinningLikelihood` now agrees with its brute-force oracle through
cusp and fold at the original tolerances. The Build-2b near-cusp blow-up is
traced to a one-line engine bug in `chang_refsdal.channels._channel_switch`,
which measured a channel's delay separation against real channels only instead
of the full cluster (including labels parked at the critical point) required by
the paper's Eq. (delay-separation); the switch spuriously engaged the divergent
saddle kernel and flooded the channel kernels, which the norm term squared into
a spurious `(h|h)`. With the neighbour set corrected, the channel kernels stay
`O(1)`, and the Build-2b dense-subsampling compensation is retired
(`kernel_subsamples` default 8 → 2). This supersedes the F006 mechanism
attribution (recorded as F008). No change to the SPEC layer prose — the
documented positive-parity / named-refusal guarantees and the Build-2 module
coverage (`cogwheel/lensing/waveform.py`, `cogwheel/lensing/likelihood.py`) are
unchanged; the near-cusp / kernel-reduction prose is mechanism-neutral and needs
no edit for this fix.
