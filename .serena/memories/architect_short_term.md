# Architect Short-Term Observations

## tube_angular_axis_graduation (planned 2026-08-17)
- Graduate tube chart 4th (angular) axis from arc-length s to delay-uniformized s'
  = cumulative total variation ∫|dΔτ/dθ|dθ (Δτ ≡ DT/2 = (tau_minus−tau_plus)/2,
  same pair as _airy_fold's xi; Δτ is NON-monotone/2-to-1 on cusp-to-cusp arc so
  raw Δτ can't be the coordinate — TV is always monotone). Import
  `_merging_fold_pair` from _airy_fold; DO NOT re-derive Δτ (DRY equality pin).
- Nyquist node count N_theta = ceil(PPP·w_max·TV(Δτ)/(2π)), PPP=8, w_max≈60 capped,
  TV evaluated at η=η_max (separability Δτ=c(θ)η^{3/2} → map shape η-independent),
  n_theta_cap=32, engine_budget 400→2048. Both parities identical law (saddle
  over-provisions, safe).
- Cusp limit: s'→d^{2/3} (A3 universality, same law as wedge/lobe u=d^{2/3}).
  SHRINK-THE-SHELL for the ~40% unservable arc-end + d^{2/3} tail extrapolation +
  mark unservable. _heldout_eps reports unserved held-out points as coverage
  (return float; nan when none served; 2 callers handle sentinel).
- Schema: rename TubeChart theta_to_s→theta_to_s_prime + axis_schema tag, HARD
  refuse stale tube artifacts. Contracts fragment for lens_amplification_surrogate.
- F083 falsification (Professor): γ=0.4 astroid w∈{52,60}, 3-5 θ near waist,
  assert N<48 AND eps≤0.0237. Cusp pin slope 2/3±0.05 over d∈[d_min,0.1], ≥8 nodes.
- Census: tighten existing tube upper-band verdict to Nyquist formula (no new gate).
- Simplifier: ~3-4 WPs; delete DRY-extraction WP; fold Nyquist into builder WP.
