# Fold Airy arm serves O(1)-wrong values on the positive-parity path [→ spec]

F028. `_positive_parity_grid` hands EVERY `w > W_CEILING_SCHWINGER` node to
`_uniform_arm_value`, with no geometric branch. Well-resolved configs
(`w * Dtau` up to 564) are therefore served by the fold Airy arm at 60%–267%
relative error, while `geometric_amplification` is exact to `1e-5` there.

Two independent defects, both needed:

1. **Admission is not caustic-relative.** The certificate `c_A * xi**-1.5` is a
   function of `xi` alone, and `xi` is large both near the caustic at high `w`
   (valid) and far from the caustic at any `w` (invalid). Admission must bound
   the fold ASYMMETRY — COVERAGE_DESIGN C6's `eta/R_c` is the natural currency.
2. **`q = 0` cannot represent an asymmetric fold.** It is a symmetric-fold
   assumption, not a leading-order truncation: with one `Ai` term the
   large-`xi` limit is a single sinusoid and cannot reproduce a two-image sum
   with unequal magnifications. Either derive the `b4` refinement of `q` or
   fence the arm to where the fold is near-symmetric.

Cheapest correct interim fix: on the positive-parity path, prefer
`geometric_amplification` whenever the node is resolved (`w * delta_min >=
RHO_END`), matching what `_saddle_grid` already does, and only fall to the arm
when unresolved. This removes the whole measured-wrong region without touching
the arm's internals.

Update SPEC.md's serving-ladder description: it presents the uniform arms as a
certified rung, which the measurement does not support.
