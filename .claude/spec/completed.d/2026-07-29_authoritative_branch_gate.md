---
date: 2026-07-29
section: lensing
---

# One authoritative geometric-vs-wave gate in the operator grids

Three sites decided geometric-vs-wave and disagreed: `channels._exact_total`
used `operator.select_branch` (resolved AND `L > L_MAX`), `_saddle_grid` used a
hand-rolled `resolved AND w > W_CEILING_SCHWINGER`, and `_positive_parity_grid`
had no geometric branch at all — every above-ceiling node went to the uniform
arms. Both operator grids now route the decision through `select_branch`.

Closes defect 1 of the F028 todo (admission routing). Defect 2 (`q = 0` cannot
represent an asymmetric fold) remains open in
`todo.d/lensing_fold_arm_serves_wrong_values.md`.

Measured motivation (F028): the uniform fold arm was serving 60%–267%
relative error on well-resolved above-ceiling positive-parity configs, where
`geometric_amplification` agrees with the Schwinger quadrature to 1e-5.

Saddle: passes an infinite cancellation exponent so only the resolution leg is
live, preserving the `w > 60 AND resolved` boundary exactly. Whether the saddle
needs a geometric-onset gate is recorded as OPEN and UNMEASURED — every sweep
behind F028/F029 was positive parity only.

Also recorded: F029, the geometric branch's residual ~1% O(1) tail, controlled
by distance to the caustic rather than delay resolution.

Test note: `test_lensing_batched_operator.py`'s certify-XOR-refuse band was
re-pointed. Its host (`gamma = 0.20`, `|y| = 0.9`) is hugely resolved
(`delta_min = 2.094`) and sits `eta ~ 0.45` outside a caustic of extent
`0.447`, so its above-ceiling nodes are now correctly SERVED rather than
refused; the refusal half of the XOR moved to an unresolved above-ceiling host.
Nodes that were named refusals with a correct answer available are now served.
