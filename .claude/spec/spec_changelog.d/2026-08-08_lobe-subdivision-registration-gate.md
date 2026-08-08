---
bump: patch
---
Update the REGISTRATION GATE description in Key abstractions (surrogate
training) to reflect lobe-interior subdivision: a gated chart's window is
subdivided recursively where the chart kind has a subdivider (far-field,
wedge, lobe; bounded by `MAX_SUBDIVISION_DEPTH`), and only a window whose
subdivided children all fail is a ladder-served gap.
