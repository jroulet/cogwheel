# Foreman-Lite Short-Term Observations

## Session: INS-c8-002/003 (channels.py / _airy_fold.py)

- **Variable-shadow fix pattern**: when an inner `delta_tau` in a nested fold block
  shadows a legitimate outer `delta_tau` (real-image delay span used for band split),
  rename only the inner variable and its two references (the assignment and the `xi_i`
  computation). A targeted `replace_content` with a literal multi-line needle that
  captures both uses in one call avoids missing one site.

- **Duplication cross-reference pattern**: when an inline performance-motivated duplicate
  and a public standalone function share correction logic, the YAGNI/KISS-correct fix is
  a maintenance comment in BOTH sites (not a helper extraction) when:
  (a) the two entry points have different signatures/context (pre-computed vs. re-derived
      geometry), and (b) the duplication is documented as intentional. The comment must
  name the other site explicitly, state WHY the duplication exists, and carry the
  finding ID so future readers can trace back.

- Sequence that worked well: `find_symbol(include_body=True)` on both the target
  function and its counterpart → `replace_content` (literal, multi-line needle) →
  `ast.parse` syntax check → import smoke-test. Total: 4 Serena calls + 2 shell
  commands; no read-file needed beyond the initial symbol fetches.

## Session: INS-w-002/003 (surrogate.py dead-code + tautological ternary)

- **Dead-code deletion via `delete_lines`**: read the exact 0-based line range first
  (`read_file` with start/end), confirm the dead statement, then `delete_lines` with
  those same indices. No symbol-level tool needed for a pure line deletion that doesn't
  affect any structural boundary.

- **Tautological ternary simplification**: `x = (A if cond else A)` → `x = A`. A
  `replace_content` with a literal multi-line needle (the full ternary) and a one-line
  replacement is the cleanest approach; no ambiguity risk since the pattern is unique.

- Both fixes in one file; a single `ast.parse` after the second edit suffices to cover
  both changes simultaneously.
