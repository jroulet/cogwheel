# Foreman Short-Term Observations

## 2026-08-08 — INS-4-001 docstring fix (surrogate.py)

- `_validate_theta_to_u` docstring in cogwheel/lensing/surrogate.py (line ~1236)
  said "Used by the wedge-interior chart" but the function is ALSO called by
  `LobeInteriorChart.from_lobe_values` (line 1801, the theta_to_u/u_grid
  spline path). Both callers confirmed via search_for_pattern before editing:
  line 1801 (lobe) + line 1993 (wedge-interior). Updated docstring to "Used by
  the wedge-interior and lobe-interior charts". Single targeted replace_content,
  ast.parse green.
- Carry-forward pattern: INS-4-001 was carried from a previous review because
  the docstring text didn't contain the string being searched (it said
  "wedge-interior chart" without "lobe"), so earlier greps for the wrong
  needle missed it. Lesson: when a doc-staleness finding is carried, grep the
  EXACT docstring sentence, not a paraphrase.
