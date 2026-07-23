INS-8haf-003: fixed by renaming `_caustic_geometry` -> `caustic_geometry`
(public) in cogwheel/lensing/ppgo_map.py via mcp__serena__rename_symbol,
which auto-updated the cross-module import and call site in
cogwheel/lensing/likelihood.py (L99, L1381). Added `caustic_geometry` to
ppgo_map.__all__. Manually fixed one stale docstring reference at
likelihood.py L1348 (`ppgo_map._caustic_geometry` -> `ppgo_map.caustic_geometry`)
that rename_symbol didn't touch since it was inside a docstring, not a
live reference. Verified via ast.parse + live import
(cogwheel.lensing.likelihood.caustic_geometry is
cogwheel.lensing.ppgo_map.caustic_geometry -> True). Useful pattern:
rename_symbol handles code references but docstrings mentioning the old
name via `module._old_name` text must be grepped and fixed separately.