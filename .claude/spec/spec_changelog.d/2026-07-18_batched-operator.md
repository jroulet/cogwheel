---
date: 2026-07-18
bump: minor
---
Batched engine fast path (Build 3c): new public `operator.F_op_grid`
(per-order weight-vector contraction over the whole wave-branch node
grid, refusal thresholds byte-unchanged, scalar `F_op` delegates);
`channels._exact_total` wired to one batched call; certified by the new
`cogwheel/tests/test_lensing_batched_operator.py`.
