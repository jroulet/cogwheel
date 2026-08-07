---
section: Backlog
depends_on: [2026-08-07_subdivision-recursion-wedge-v3-r-caustic]
---

- **BOUNDED RECURSION SHIPPED FOR THE EXTERIOR BUT WAS ONLY MEASURED ON THE
  INTERIOR** `[housekeeping]` — the 2026-08-07 build gave BOTH subdividers
  bounded recursion (`MAX_SUBDIVISION_DEPTH = 3`) through one generic
  `_subdivide_tile`. The justification was measured on the astroid interior
  only: 13/16 children cleared at one halving, the three that did not were
  marginal (6.50e-2, 6.70e-2, 5.95e-2 against a 5e-2 bar), and one more level
  was predicted to close them.

  The EXTERIOR case that motivated extending it was inferred, not measured:
  84% of exterior charts were subdivision children AND 35 of 57 still failed
  the 1e-3 bar — numbers that only sit together if every marginal tile got
  exactly one halving and was then abandoned. Plausible, and still unverified.

  WHAT TO MEASURE: rerun the exterior training for one band with recursion
  live, and report (i) how many of the 35 previously-failing charts now clear
  the 1e-3 bar, (ii) the achieved-depth histogram, and (iii) whether any tile
  hits the depth-3 cap — a tile that exhausts the cap is evidence the
  COORDINATE is wrong, not that the cap is too low, and it should be routed to
  [[lensing_exterior_should_chart_in_polar_not_sd]] rather than given a
  deeper cap.

  Do this BEFORE the polar re-chart, so the polar-vs-`(s,d)` node-budget A/B
  in [[lensing_exterior_followup_four_items]] compares like with like: both
  arms must have recursion, or the comparison credits the coordinate change
  with the recursion's gains.
