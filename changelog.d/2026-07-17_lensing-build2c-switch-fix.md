---
date: 2026-07-17
---
### Fixed: microlensing channel-switch neighbourhood bug (Build 2c)

`chang_refsdal.channels._channel_switch` measured each real channel's delay
separation against OTHER REAL channels only, whereas the paper's
Eq. (delay-separation) minimizes over ALL cluster members — including the
labels parked at the critical point ("virtual" labels). On the two-image side
of a caustic those parked labels are an image's true cluster-mates, so the old
rule saw a spuriously large separation, ramped the switch to 1, and handed the
channel to the divergent saddle kernel. The result flooded all four channels
with an unbounded kernel (`max|K_a| ~ 5.2e5` at the crown `near-cusp` config),
which the relative-binning norm term then squared into a spurious `(h|h)`
(`|RB lnL - brute lnL| = 6.43e8`). The neighbour set now includes every cluster
label except self, per Eq. (delay-separation); the fix can only LOWER a switch
value and is a bit-for-bit no-op wherever all four labels are real (4-image
regions, near-fold-inside). Channel kernels at the crown near-cusp config are
now `O(1)` and the reconstruction residual improves from `2.5e-10` to `5e-16`,
so `LensedRelativeBinningLikelihood` agrees with its brute-force oracle through
cusp and fold at the original tolerances (offsets `+0.080` two-image, `+0.329`
near-cusp; gate `1.5`). This supersedes the Build-2b near-cusp mechanism
attribution (see FINDINGS F006 → F008).

The dense-subsampling compensation shipped in Build 2b is retired: the
`LensedRelativeBinningLikelihood` `kernel_subsamples` default reverts from 8 to
2 (the sub-sampling machinery is retained as a robustness margin, not a
correctness requirement), dropping engine evaluations per `lnlike` ~4× now that
the kernels are bounded. No public API changed.
