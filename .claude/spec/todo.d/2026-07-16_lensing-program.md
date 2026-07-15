---
section: In progress
---
- **Microlensed-PE program (Chang–Refsdal)** `[→ spec]` `[→ docs]` — three-build
  sequence implementing relative binning for microlensed waveforms per the
  manuscript + design decisions in `.claude/handoff/lensing/META_PLAN.md`:
  (1) lens engine `cogwheel/lensing/chang_refsdal/` (geometry / contour-free
  operator / topology-stable channels) with oracle + exact-reconstruction +
  continuity tests; (2) `LensedWaveformGenerator` + multi-component
  relative-binning likelihood (delay-continuous summaries, sequential
  mode→image contraction, FFTs setup-only) with brute-force lnL agreement
  tests; (3) sampled lens coordinates (κ, β eliminated exactly; {ln δt,
  contrast, folded source angle, γ'}), astroid folding, injection-recovery
  validation. Source paper (unpublished) + prototype:
  `.claude/spec/lensing_paper/`.
