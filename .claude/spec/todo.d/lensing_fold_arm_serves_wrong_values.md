---
section: Backlog
---

- **Fold Airy arm cannot represent an asymmetric fold (`q = 0`) [→ spec].**
  F028, defect 2 of 2. Defect 1 (admission routing) is CLOSED by the
  authoritative-gate build: both operator grids now decide geometric-vs-wave
  through `select_branch`, so well-resolved above-ceiling positive-parity
  nodes are served by geometric optics instead of the arm.

  What remains is the arm's own accuracy. `fold_amplification` sets the `Ai'`
  amplitude `q = 0`, which its docstring calls "the pure-phase symmetric-fold
  result". That is a SYMMETRIC-FOLD ASSUMPTION, not a leading-order
  truncation: with one `Ai` term the large-argument limit is a single sinusoid
  of fixed amplitude, while a true two-image sum has two independent complex
  amplitudes, equal only where the merging pair has equal magnification — i.e.
  only exactly on the caustic. Away from it the error is O(1) however large
  `xi` becomes, which is why F028 measured the error GROWING with `w`
  (0.348x at `w = 70` to 1.846x at `w = 500`) rather than shrinking.

  Two options, not exclusive:
  1. Derive the `b4` (quartic) refinement of `q` so the form can represent an
     asymmetric fold.
  2. Fence the arm to where the fold is near-symmetric. Its current `xi`-only
     certificate cannot do this — `xi` is large both near the caustic at high
     `w` (valid) and far from it at any `w` (invalid). Admission needs a
     caustic-relative term; COVERAGE_DESIGN C6's `eta/R_c` is the natural
     currency.

  Note the symmetry with F029: the arm is admitted far from the caustic where
  it is invalid, and geometric optics is admitted near the caustic where it is
  invalid, both because no admission term measures distance to the caustic.
  Fixing either one properly probably fixes both.
