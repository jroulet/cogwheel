---
bump: patch
---

Add `test_lensing_wedge_dd_arclength.py` to InteriorWedgeChart certified-by
list in SPEC.md. The new 666-line suite pins the DD-product w-ceiling formula
(`w_max <= _DD_PRODUCT_MARGIN / (r_max * reach_max)`) and the theta_to_s
arc-length axis construction added to `from_wedge_engine` in commit 56a223a.
