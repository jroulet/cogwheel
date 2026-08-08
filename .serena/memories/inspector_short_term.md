Build: saddle_forensics re-review (brief_saddle_forensics, 2026-08-08, pass 2)
Working tree: cogwheel/lensing/surrogate_training.py (+150/-15), test_lensing_lobe_subdivision.py (new, 19/19 PASS)

Previously open findings re-checked:
- INS-1-001 RESOLVED: _subdivide_lobe_tile build_child computes eff_w_nodes with 3-way resolve (tile override -> interior_w_nodes_per_decade -> w_nodes_per_decade), passes to _build_lobe_chart. Matches wedge subdivider pattern.
- INS-1-002 RESOLVED: Stale comment at line ~4918 replaced. Now reads "The tile straddles a critical-basin flip; subdivision cannot fix phase discontinuities, so the tile is recorded as a ladder-served gap."

Test results:
- test_lensing_lobe_subdivision.py: 19/19 PASS
- test_lensing_surrogate_training.py: 64P/49S, no failures
- test_lensing_surrogate.py: 66/66 PASS
- Import probe clean

No new issues introduced.

Pre-existing (carried forward):
- _subdivide_tile docstring (~line 3788) still says "A future lobe subdivider..." — stale since lobe subdivision now exists. Not touched by this diff.
- _LOBE_CUSP_EXCLUSION_DISTANCE=0.1 intentionally dead code (Professor ruling), documented with rationale.
