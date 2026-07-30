# Inspector Short-Term Observations

## 2026-07-30 (pass 4) — Build 1e-tube (TubeChart arc-length s) — RE-REVIEW

Scope: uncommitted, claude-dev worktree. Production diff (surrogate.py +142,
surrogate_training.py +89, DATA_CONTRACTS +1 line) is BYTE-IDENTICAL to the
pass-3 diff I reviewed — re-verified via `git diff HEAD`. No new .py edits since
pass 3. SPEC.md still NOT in diff.

### Verified this pass
- import surrogate{,_training}: OK.
- test_lensing_surrogate_training.py: 31 passed / 48 skip (6.19s) — matches
  pass 3 exactly. Exercises the new producer (_tube_arc_length_map,
  s_map_gamma_endpoint_dev diagnostic).
- Did NOT re-run the 221s serve suite (test_lensing_surrogate.py): diff is
  byte-identical to pass 3 where it ran green (62 passed/1 skip). Proportionate:
  no serve-path bytes changed.
- Re-derived identity-map default by hand AGAIN: from_values(theta_to_s=None) ->
  s_grid=theta_grid-theta_grid[0], theta_to_s=[theta_grid, s_grid]; spline fit
  in shifted coord; serve v2=interp(theta_inframe, theta_grid, theta_grid-lo)
  = theta_inframe-lo. Translation-equivalent to raw-theta spline (fp-close, not
  bit-identical — coder documented this; back-compat charts unaffected). OK.
- Arc-length producer: rep_gamma=median(gamma_grid); theta_fine,s_fine via
  cumulative_trapezoid(caustic_speed(gamma,theta,branch=arc.branch)); s_grid
  uniform in s; theta_grid=interp(s_grid,s_fine,theta_fine) with endpoints
  forced. Serve reads stored theta_to_s. Node-consistent. OK.
- npz: stores prefix+'theta_to_s'; _chart_from_npz reloads into _assemble.
  Round-trip covered. OK.
- _validate_theta_to_s: shape (2,N>=2), finite, both rows strictly increasing,
  row0[0]==theta_grid[0], row1[0]~0. Correct guards. OK.

### Findings
- INS-1-001 STILL OPEN (trivial, flag-to-Librarian): SPEC.md line 55 tube-chart
  sentence reads coords `(gamma, u = sqrt(eta), theta, log w)` + "theta bounded,
  non-periodic ... query-unwrapped ... by _theta_into_frame", NO arc-length s
  clause. DATA_CONTRACTS updated (theta_to_s + s-axis prose), SPEC not. Code
  correct; pure doc-sync divergence owned by Librarian. NOT resolved. Carried
  from pass 3.

### Notes (carried, low risk, not flagged)
- from_values does NOT cross-check interp(theta_grid,theta_fine,s_fine)==s_grid
  (spline axis vs serve map); only the producer builds them consistently.
- Passing s_grid WITHOUT theta_to_s silently overwrites s_grid with identity
  (benign; contract: s_grid meaningful only with theta_to_s).
- Old pre-migration tube npz (no theta_to_s key) raises bare KeyError, not typed
  ValueError. Loud hard-refuse, matches lobe-branch style. No shipped old
  artifact in-repo. Not flagged.
