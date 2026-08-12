---
date: 2026-08-12
bump: patch
---
### CUSP-EXCLUSION FILTER paragraph: saddle parity now admits near-cusp tiles

SPEC.md TRAINING section, CUSP-EXCLUSION FILTER paragraph: the sentence
"excluded from exterior training on BOTH parities" was stale after commit
c8cad0c extended `ExteriorPolarChart` to the saddle exterior cusp window.

For positive parity (astroid), the `_CUSP_EXCLUSION_DISTANCE = 0.35` carve-out
still applies; astroid tiles within the exclusion window are served by the Pearcey
arm / exact engine as designed.

For the macro saddle (parity -1), near-cusp tiles are now ADMITTED (`d_exclude =
0.0`) and trained on the ghost-subtracted label (`FARFIELD_KERNEL_SUM_MINUS_GHOST`,
`force_minus_ghost=True`); the ghost gate resolves the near-cusp oscillation that
the raw kernel-sum label cannot.  Cusp proximity for saddle tiles is still
determined by `_deltoid_cusp_source_angles` at band edges, but used for labeling
rather than exclusion.  The certified-by sentence is qualified to "astroid cusp
exclusion boundary" to reflect that saddle tiles near the cusp are no longer
excluded.
