## 2026-07-31 build sync — lobe s-coordinate production code

Scope: build diff adds sqrt-edge s-coordinate to `LobeInteriorChart` in `cogwheel/lensing/surrogate.py`, with new test cases in `cogwheel/tests/test_lensing_surrogate_lobe.py`.

Files edited:
- `.claude/spec/SPEC.md` — 3 changes in the "Microlensed sampling layer" row:
  1. `charted in lobe-local (rho_lobe, theta_local)` → `(rho_lobe, v2)` with explanation of V1/current duality
  2. Single-tag schema sentence → dual-tag sentence (`_LOBE_AXIS_SCHEMA_V1` + `_LOBE_AXIS_SCHEMA`)
  3. Test cert sentence extended to name the new sqrt-edge and V1 identity-path test coverage
- `.claude/spec/DATA_CONTRACTS.yaml` — added a paragraph to `lens_amplification_surrogate.description` documenting `LobeInteriorChart` axis-schema duality and `theta_to_s` field semantics for both schemas

Patterns / lessons:
- The first `replace_content` (regex mode) failed with "No matches" due to escaped backslash mismatch (SPEC.md stores `\|` as `\\|` in raw bytes, and the regex engine saw 2 backslashes). Second call succeeded but left a corrupt join (old fragment + new text concatenated). Must use Python's `str.replace` directly via shell for any replacement in SPEC.md that involves `\|` characters — the Serena regex mode double-escapes them on return.
- Always verify SPEC.md edits by checking raw bytes with a Python snippet rather than relying on Serena's `read_file` view (which un-escapes backslashes in the display).
- `render_fragments.py` returned "All surfaces up to date" — no fragment changes needed (no spec version bump warranted for a trivial/doc-only fix; Inspector marked both findings trivial).
- No Sphinx rebuild needed (no `docs/source/` files touched).
- Zero stray side-effect diffs (no `tidy_advisory.json` or `foreman_lite.json` pollution this run).

Fragile cross-reference to watch:
- The two `_LOBE_AXIS_SCHEMA*` constant names are now cited in both SPEC.md and DATA_CONTRACTS.yaml. If either is renamed in code, both doc surfaces need updating.
