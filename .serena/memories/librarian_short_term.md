# Librarian Short-Term Observations

## 2026-08-07 FarFieldChart deletion post-commit sync

**Scope**: Two pending commits from sync_issues.json:
- `0a31fcf` — `feat(lensing): delete FarFieldChart and (s,d) machinery (post-strand cleanup)`
- `0a4e18a` — `chore: sweep remaining working-tree changes`

**Key finding — Serena index lag**: `mcp__serena__search_for_pattern` returned
stale results showing FarFieldChart still present in SPEC.md and DATA_CONTRACTS.yaml.
The actual files (verified by grep/read) already had those removed by the sweep
commit. Always cross-check Serena search results against `grep` before editing.

**What the sweep commit (0a4e18a) already cleaned up**:
- SPEC.md Key abstractions: backward-compat sentence for FarFieldChart removed
- SPEC.md NAMING HAZARD: FarFieldChart removed from the name list
- DATA_CONTRACTS.yaml: backward-compat sentence for FarFieldChart removed
- spec_changelog.d/2026-08-07_farfield-chart-deleted.md: created
- contracts_changelog.d/2026-08-07_farfield-chart-deleted.md: created

**What this librarian run additionally fixed**:
1. SPEC.md LOW-W FLAT EXTRAPOLATION: "(tube, far-field, lobe, wedge)" →
   "(tube, exterior-polar, lobe, wedge)"
2. SPEC.md surrogate description: "exterior FAR-FIELD charts" → "exterior-polar
   charts" (FarFieldChart was the class; ExteriorPolarChart is now the only
   exterior chart class)
3. spec_changelog.d/2026-08-07_farfield-chart-deleted.md: extended to mention
   items 1 and 2 above
4. todo.d/lensing_farfield_name_spans_three_regimes.md: updated title and body
   to note FarFieldChart class deleted; remaining rename scope is
   `farfield_*` helper names (farfield_envelope_from_partition, FARFIELD_KERNEL_SUM,
   farfield_eps_max, _farfield_tiles, _validate_farfield_axis_schema)

**Confirmed no-ops**:
- docs/source/: no FarFieldChart or ExteriorPolarChart mentions, no Sphinx update needed
- DATA_CONTRACTS.yaml: already clean before this run
- FINDINGS.md: no FarFieldChart findings to retire
- Consumer-graph warning: escalation TODO (surrogate_contract_test_consumer_warning.md)
  already exists — same 4 test-only callers recurred again

**Pattern — chart-type references go stale in MORE places than class-name references**:
The sweep commit cleaned up the KEY ABSTRACTIONS sentence but missed the chart-type
list "(tube, far-field, lobe, wedge)" and the "exterior FAR-FIELD charts" description.
On future chart-type renames, check BOTH: (a) explicit class-name mentions, and
(b) the informal chart-type names in multi-chart descriptions and chart-type lists.

**Fragile cross-references to watch**:
- SPEC.md "FAR-FIELD TILING" section header: still named "far-field" because
  the internal functions (_farfield_tiles, etc.) are still named that way. If
  those helper names are renamed in a future cleanup, this SPEC.md label should
  follow.
- lensing_exterior_should_chart_in_polar_not_sd.md TODO: still open (acceptance
  criteria not verified). Depends on measurement of chart-count eps improvement.
