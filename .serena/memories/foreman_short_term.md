# Foreman Short-Term Observations

- INS-1-002 (comment fix in surrogate_training.py ~line 4917): stale comment
  claimed 'lobe-aware subdivision is owed follow-on work' but `_subdivide_lobe_tile`
  now exists (find_symbol confirmed the symbol at line 4193). When the working
  tree carries parallel-session uncommitted changes, confirm via `git diff` that
  your edit touched ONLY the intended lines — the diff grep confirmed exactly the
  6 removed / 2 added comment lines and no code lines. Comment-only fix, syntax
  verified with ast.parse; no test run needed.
