# Foreman-Lite Short-Term Observations

- INS-1-001 (serve_route_census.py): removed dead `cusp_amplification: Any`
  dataclass field from `_ProductionModules` and its assignment in
  `_load_production_modules` (`op._pearcey_cusp.cusp_amplification`).
  Confirmed via search_for_pattern that both sites were the ONLY
  occurrences in the file before removal, and re-confirmed zero matches
  after. Verified via ast.parse + a live import + dataclass_fields check
  (14 fields remain, cusp_amplification gone). Simple mechanical dead-field
  removal, no other logic touched.
