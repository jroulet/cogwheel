---
date: 2026-07-18
---
### Microlensed likelihood: batched operator contraction (~7x faster again)

`LensedRelativeBinningLikelihood.lnlike` drops from ~0.3 s to ~41 ms/eval
(warm, single-thread) with every accuracy gate unchanged. The engine's
per-node 85x85 shear-series contraction is replaced by a per-order
weight-vector reduction: the w-independent monomial/table weights are
scatter-added once per evaluation, so each frequency node costs a single
length-85 dot product (`operator.F_op_grid`, evaluating the whole
wave-branch node grid in one call; scalar `F_op` delegates to the same
path). All four certified-or-refuse thresholds are byte-unchanged and
re-certified against the 70-dps mpmath oracle across the F005 boundary
band, with single-vs-batch refusal-decision identity (zero flips). The
per-eval cost is now dominated by the (exact, un-batchable) 1F1
derivative ladder — the target of the next planned step toward the
few-millisecond goal.
