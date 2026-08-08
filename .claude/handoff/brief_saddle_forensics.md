# Build Brief: Saddle forensics — audit macro-saddle charts

## Mission

Six-question audit of macro-saddle (gamma > 1) charts for defects found
and fixed in the astroid. Sequenced after exterior work per the fragment.

## Work

1. Deltoid interior normalised-radius disease: LobeInteriorChart uses
   rho_lobe = |y-centroid|/r_deltoid — same pattern that hurt the wedge.
   Run a 1-D transverse cut toward a cusp on a lobe tile.

2. Lobe subdivision: No _subdivide_lobe_tile exists. Add splitter/builder/
   gate triple using the unified _subdivide_tile.

3. Lobe cusp carve-out: Cusp-ALIGNED but not cusp-EXCLUDED. Add carve-out
   sized by separation-gate reasoning.

4. Saddle exterior (now polar): Verify ExteriorPolarChart works for saddle.

5. Inter-lobe corridor: Probe whether corridor creates coverage gap.

6. Ghost kernel parity gate: ghost_kernel has NO parity gate, no saddle tests.
   Branch pin (exp(-0.5j*pi)) may be a SIGN ERROR on saddle.

## Constraints

Fast tests. Follow AGENTS.md.
