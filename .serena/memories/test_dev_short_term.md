# Test Dev Short-Term Observations

- Added test_reprovision_catches_carrier_discontinuity_error to
  ReprovisionNodeCountTestCase in test_lensing_exterior_windows.py.
  Tests WP1 (CarrierDiscontinuityError catch in _eps_for): patches
  _build_farfield_chart to raise CarrierDiscontinuityError, asserts
  graceful early return with status='engine_refused', trace entry with
  'carrier_discontinuity' status+detail, and full node density preserved.
  All 77 tests in file pass + 1 xfailed (172s total).
