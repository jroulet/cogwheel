---
date: 2026-07-29
---

### Fixed: well-resolved high-frequency nodes were served by an inaccurate uniform-asymptotic arm

Above the Schwinger ceiling (`w > 60`), the positive-parity node grid handed
every node to the uniform fold/cusp arms with no geometric-optics branch. For
well-resolved configurations that produced 60%–267% relative error, where
stationary-phase geometric optics agrees with the exact quadrature to 1e-5.
The arm's self-certificate reported 1%–5% in those cases, optimistic by
20x–100x, because it is a function of the Airy control `xi` alone and `xi` is
large both near the caustic at high frequency (where the uniform form is
valid) and far from the caustic at any frequency (where it is not).

The geometric-vs-wave decision now routes through `operator.select_branch` in
both node grids — the same predicate the channel layer already used. Nodes it
admits are served by geometric optics; the rest fall to the uniform arms and
then, if both decline, to the existing named refusal. Some configurations that
previously raised a named refusal are now served, since a correct answer was
available for them all along.

The macro-saddle serving boundary is unchanged.
