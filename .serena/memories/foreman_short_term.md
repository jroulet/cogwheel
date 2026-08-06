2026-08-06 session: fixed INS-1-002, a single stale docstring reference in
_farfield_exterior_tiles (surrogate_training.py ~line 1861) pointing at the
deleted _farfield_interior_tiles helper. Confirmed via search_for_pattern
that the only production-code reference was this docstring line; the
remaining hits are in test files (test_lensing_ppgo_bandsplit.py,
test_lensing_interior_wedge_chart.py) that intentionally document/verify
the helper's retirement -- correctly frozen history, left untouched.
Replaced the sentence to describe the shared cusp-alignment convention
without naming the deleted symbol. Verified via ast.parse. Single-call
replace_content fix, no structural changes needed.