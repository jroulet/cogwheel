---
bump: patch
---

### Lensed relative-binning likelihood: near-cusp correctness fix (Build 2b)

`LensedRelativeBinningLikelihood` now resolves the caustic-sharpened
amplification within each relative-binning bin. Previously the per-bin kernel
coefficients `(k0, k1)` were built from the two bin edges only (a secant); near
a cusp, where the merged-image channel kernel `K_a(f)` collapses to the
artificial single-image split `alpha_a * exp(-i w tau_a) * F(w)` and carries the
full rapidly varying amplification, that secant aliased the oscillation and its
squared slope manufactured a spurious `(h|h)` (measured `|RB lnL - brute lnL| =
6.43e8` on the `near-cusp` config). The hot path now densely sub-samples each
bin (`kernel_subsamples = 8`) and reduces the kernel to `(k0, k1)` by per-bin
least squares, so the RB likelihood agrees with its brute-force oracle through
the same `LensedWaveformGenerator` across all configs including near-cusp. The
contraction algebra, frequency moments, image-delay guard (`LensedBinningError`),
and the `F→1` normalization are unchanged (the normalization was audited and
found correct; the unlensed-floor variability was an unseeded noise draw, not a
bug). Engine refusals (`LensDomainError`, `CancellationError`) still propagate
symmetrically on the RB and brute-force paths. Mechanism and audit recorded in
FINDINGS F006. No change to the SPEC layer prose — the documented
positive-parity/named-refusal guarantees still hold.
