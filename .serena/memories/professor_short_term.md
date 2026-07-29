# Professor short-term (2026-07-28) — select_branch "one home" consult (operator.py)

Consulted on routing BOTH operator grids' geometric-vs-wave decision through
`select_branch`. Key observations from reading operator.py / _schwinger.py / channels.py:

- `select_branch(w, delta_min, L)` returns 'geometric' iff `w*delta_min >= RHO_END(4.0)`
  AND `L > L_MAX(48)`. BOTH thresholds are provenance-tied to the SAME physical fact:
  the geometric-optics onset at `w*delta ~ 50` at resolved clusters (F013), which is
  wave-optics and PARITY-INDEPENDENT. The `L > L_MAX` leg is a positive-parity PROXY
  for that onset via the 1F1-ladder cancellation depth L = w|y'|. The module L_MAX
  comment even says the onset is "governed by w*delta NOT L".

- SNAG confirmed: `cancellation_exponent` -> `_mass_sheet_map` raises LensDomainError on
  saddles. So select_branch as fed by cancellation_exponent cannot be called saddle-side.
  channels._exact_total EXPLICITLY special-cases saddle_host to skip select_branch and
  delegate to operator's saddle arm (comment lines ~641-648).

- Saddle cancellation law: L_S = _CANCEL_SCALE*w = pi*w/4 (y-INDEPENDENT, _schwinger.py
  L153-156). Ceiling W_CEILING_SCHWINGER=60 is where e^{pi w/4} outruns the DD mantissa
  (w~64 -> rounded to 60). So the saddle's wave-refusal boundary (w>60) is a NUMERICAL
  mantissa ceiling, NOT the geometric-onset physics that L_MAX encodes.

- CRITICAL for the user's saddle plan: pi*w/4 > 48 iff w > 61.115. The user proposed
  feeding L_S=pi*w/4 to select_branch, which moves the saddle geometric threshold from
  w>60 to w>61.115. That is INCOHERENT: comparing pi*w/4 (a DD-mantissa depth) against
  L_MAX=48 (a 1F1-ladder onset proxy) conflates two unrelated scales, AND it opens a
  dead band (60,61.115] where Schwinger refuses (w>60) but select_branch says 'wave' ->
  arms/refusal, a coverage REGRESSION vs today's w>60 geometric serve. Geometric onset
  for resolved saddle is ~w=50 (F013 saddle analogue, negative_parity_research), so the
  ceiling at 60/61 already over-delays geometric; pushing it to 61.115 is strictly worse.

- geometric_amplification is parity-agnostic: it uses geometry.macro_matrix (indefinite
  for saddle) + find_images (real images only) + image_kernel (Morse phases carry saddle
  n_a). _certify_geometric_census guards image count + Morse parity-sum. The saddle grid
  ALREADY calls geometric_amplification for its resolved+w>60 nodes, so no new physics.

- Positive-parity grid currently has NO geometric branch (F028: fold Airy arm 60-267%
  wrong on resolved configs vs geometric_amplification which is cross-checked 1e-5 vs
  quadrature w=45..60). Adding the select_branch route there is the clean fix and needs
  cancellation_exponent (positive parity -> _mass_sheet_map does NOT raise). |y'| is
  w-independent so L = w*|y'| scales linearly per node from one base eval.

- Known residual: brief says ~1% of gate-admitted nodes still O(1) error (p99 7.1e-1,
  max 74), not fixed by raising w*delta to 100. Must NOT call geometric branch
  certified/exact. Agree — new FINDINGS entry warranted.
