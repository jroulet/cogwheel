# Librarian Short-Term Observations

## 2026-08-20 (diffractive-certificate interior calibration, INS-3-001 completion)

- SPEC.md STALENESS LINEAGE: the LOW-W DIFFRACTIVE RUNGS paragraph (Rung P)
  survived the entire certificate-fit lineage (stranded WIP builds
  e26e6c6..0d35d4f then the completed INS-3-001 build) describing the
  RETIRED formula-scan admission gate (`w_low = (gamma'/2)*[...]` with
  root-find fallback), while production switched to the O(1) fitted
  `w_low_fit` surface with near-fold fence + de-rate. The original rung
  build (733b7ef) documented the scan; the certificate-fit lineage changed
  the mechanism but no WIP commit updated SPEC.md. A mechanism-swap with
  zero SPEC.md edits across 5+ commits is the same "SPEC cites a function
  by name" staleness family, at paragraph scale. Synced this run: Rung P
  now describes `w_low_fit` (log-log degree-2 poly + even harmonics +
  caustic feature, TOTAL-amplitude normalization pinned), de-rate as SOLE
  margin, `min(., CEILING)` as hard oracle-domain cap, `_DIFFRACTIVE_FIT_
  FENCE_*` near-fold shell decline, and deep-interior coverage (grid to
  r ~ 0.1, honest ceiling ~4-41 not the clip). Verified against code:
  `w_low_fit` docstring + `likelihood.py` call site + `_diffractive.py`
  constants (RHO_LO=0.6, DELTA=0.4, DERATE=0.85, LIP=1/17).
- FINDINGS F084 filed: the `min(., CEILING)` clip-as-conservativeness trap
  (uncalibrated fit clipped to 60 silently re-serving the interior up to
  ~2.9x over engine-honest ~4-41 ceiling). This is the first F-number for
  the certificate-fit lineage; the measured over-serve numbers were only in
  the brief/completed record before. Same family as F069/F073/F076.
- CHANGELOG/COMPLETED/spec_changelog fragments written for the INS-3-001
  completion (de-rate 0.85, smoke margins 178/178+44/44, excluded-shell
  21.6%, provenance SHA 362c58e 526.9s, CORNER_R 1.05->1.1, residual
  gamma=0.5 interior ~1.12x RED BY DESIGN until full bake).
- RENDER NOTE: 2026-08-19 spec_changelog fragment
  `diffractive_certificate_reach.md` (the certificate-reach build) still
  describes `diffractive_w_low` + `_rootfind_w_high` running-max scan —
  that fragment is HISTORICAL (documents the order-16 raise + first-breach
  ceiling) and the reach build was SUPERSEDED by the fit lineage; left as-is
  per historical-measurement convention, but watch: a future reader may
  conflate the reach build's scan with the current fit.
- NO Sphinx/docs surfaces touch `_diffractive` or `w_low_fit`; sync script
  5 checks all OK; no docs/source rebuild needed this run.
