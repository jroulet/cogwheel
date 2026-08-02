# Coder Short-Term Observations

- WP2 (Build 5 C8): ALREADY COMPLETED in prior session. Confirmed current state:
  born_gate has exactly 2 guards (B: parity-wall margin, A: band split).
  No ANNULUS_INNER_RADIUS, GAMMA_FENCE, saddle fence constants, or saddle_caustic_max_y
  function remain. Only RHO_END and DELTA_GAMMA_P constants present. Module diagnostics
  show only pre-existing numpy resolution issue (Pyright env-level).


- WP1 (Build 5 C8): Renamed ppgo_map.annulus_rho → caustic_rho across codebase.
  Serena rename_symbol handled code refs (5 changes); manually fixed __all__ string
  (rename_symbol caught it), docstrings (4 ppGO-annulus→caustic-relative in ppgo_map.py),
  comments in surrogate_training.py (2), likelihood.py docstring (1),
  test_lensing_ppgo_map.py (26 annulus_rho→caustic_rho + 4 AnnulusRho→CausticRho class names
  + 5 'ppGO annulus'→'caustic-relative' in docstrings/comments/labels),
  test_lensing_ppgo_bandsplit.py (1 filename string). Zero 'annulus_rho' in .py files confirmed.
  Re-confirmed in current session: caustic_rho in __all__, function at L782-837, zero annulus_rho.


- WP3 (Build 5 C8): ALREADY COMPLETED. Re-confirmed current session: caustic_rho
  imported at L55, used at L277 with rho>1.0 check, LensDomainError caught at L278,
  no _born references, no _BORN_ANNULUS_OUTER_RADIUS, _FALLTHROUGH_CATEGORIES has 'born'
  (unified, not parity-split). All verification criteria pass.
