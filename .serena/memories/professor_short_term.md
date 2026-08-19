## Ruling: INS-1-003 (diffractive_certificate_reach build) — COMPOSITION bug, not docstring

`_diffractive_bottom_ceiling` (band-aware) returns `w_low == w_hi` when the whole
dense band clears the diffractive N/2N bar (`_rootfind_w_high` line 323: `if
relerr(w_hi) <= target: return w_hi`). The nested-split composition reuses
`_band_split_mask(dense_w, w_low)`, whose strict-interior test (`w_lo < split <
w_hi`) fails at `w_low == w_hi`, so `band_split_low=False` and the
`if band_split_low else zeros` branch empties `bottom_mask`; `host_mask` becomes
the whole below-split region. F_P (Rung P) serves ZERO nodes.

Key mis-anchor: HEAD's `diffractive_w_low` returned the CONSERVATIVE candidate
(brief fact 1: 23-405x below the honest ceiling, never up-searched). For a
whole-band-eligible low-gamma draw, HEAD's `w_low = candidate` sat STRICTLY inside
`[w_lo, w_hi]` -> non-empty bottom (F_P served `[w_lo, candidate)`), chart hosted
`[candidate, w_trust]`. HEAD only "hosted everything" when the candidate itself
>= w_hi (rare). So the plan's "byte-identity where w_low >= w_hi" conflates the
rare HEAD candidate-overshoot case with the NEW common whole-band-certified case;
leaving the composition as-is REGRESSES HEAD (F_P serves nothing where HEAD served
a non-empty bottom) and defeats the brief's "serve to the honest ceiling".

Fix (composition-level, NOT in `_band_split_mask`): special-case
`w_low >= float(dense_w.max())` -> `bottom_mask = below_mask` (full region),
host empty. Do NOT relax `_band_split_mask`'s strict interior: the outer splits
(w_split / w_trust / trained_floor) load-bear on "edge = no-op" (below_mask
all-True null-split identity; trained-floor "genuine strict sub-band" guard
`band_split_floor and ...`). Macro-saddle (gamma>1) returns None -> unaffected.

Latent secondary: `_rootfind_w_low` (down-search branch) is NOT capped at w_hi,
so `_diffractive_bottom_ceiling` can return w_low > w_hi there; the `>= w_hi`
special-case covers it, but the certificate should cap BOTH branches (return w_hi
whenever the certified region extends past the band top).

Test pins needed: (1) composition value pin — a whole-band draw serves F_P over
the full below-split region (host empty), asserted against the engine oracle
within CERTIFICATION_BAR, not by path; (2) a dedicated engine-honesty pin at the
served band top w_hi: the 0.9*w_low sweep uses the UNBOUNDED ceiling and
deliberately stays off the zero-margin ceiling, so it can miss the top sliver in
barely-whole-band draws, and the N/2N estimator at w_hi is demonstrably
optimistic/non-monotone at low gamma (NONMONOTONE_DRAW gamma=0.1, breach ~0.9*w_low).
