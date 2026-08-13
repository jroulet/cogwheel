# Foreman-Lite Short-Term Observations

- INS-1-002 (ppgo_error_estimate empty-input guard, geometry.py): the
  existing degenerate-input branch was `if w_min <= 0.0: return None`,
  so the fix was a one-line widen to `if w_min <= 0.0 or
  len(real_images) == 0: return None` (kept as a single OR-combined
  guard rather than two branches, to match the existing early-return
  style). Docstring's "Returns" section already listed the other
  degenerate cases (w_min<=0, non-finite mu/c3) — added the empty-array
  case to the same sentence rather than as a new paragraph, since it's
  the same "gate reads None as refuse" contract. Verified empirically
  (not just ast.parse) that `ppgo_error_estimate(np.empty((0,2)), ...)`
  now returns None instead of the old silent 0.0.
