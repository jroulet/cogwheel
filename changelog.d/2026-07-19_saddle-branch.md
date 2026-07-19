---
date: 2026-07-19
---
### The engine crosses the parity boundary

The Chang-Refsdal engine now supports negative-parity (macro-saddle)
hosts: the geometry layer solves, classifies, and traces caustics for
saddle configurations (two 3-cusp deltoid lobes; Morse census verified
by the index theorem over hundreds of sources), and a new exact
one-dimensional Schwinger-parameter evaluator in double-double
arithmetic serves the saddle wave branch — certified against an
independent high-precision oracle from 9e-14 (w=20) to 2e-11 at its
w=60 ceiling, with a paired-quadrature certify-or-refuse contract and
named refusals at the ceiling, the parity boundary, and the
over-critical domain. The positive-parity path is bit-frozen. Test
authorship surfaced and fixed two silent certification-blind rounding
defects (now a documented audit rule) and uncovered a pre-existing
near-axial image-solver dead zone, tracked with a runtime-guard
precondition for the next build. The channel/likelihood layers refuse
saddles by name until the follow-up build extends them.
