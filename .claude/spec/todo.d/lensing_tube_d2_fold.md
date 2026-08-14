---
section: Backlog

---

- **FOLD TUBE-CHART SERVING INTO THE D2 FUNDAMENTAL DOMAIN — the last
  unfolded chart kind** `[→ spec]` — owner directive 2026-08-14: "When
  there is a symmetry of the problem, we have to use it... it is a moral
  imperative." The amplification is exactly D2-symmetric
  (`F(w; y1, y2) = F(w; ±y1, ±y2)`: the Fermat potential is even in each
  source coordinate with external shear), and every OTHER chart kind
  already folds — exterior-polar serves in the folded quadrant with a
  hard-raise domain guard (`surrogate.py:588-596` at HEAD a4ba536), wedge
  charts fold (`:974-1001`), lobe charts fold (`:2989-2990`). TubeChart
  alone serves in unfolded theta, which is why F079's half-ring hole was a
  live serve loss instead of being recovered by symmetry, and why the
  training campaign would otherwise train 4 astroid tube arcs where ONE
  fundamental-domain arc suffices (~4x tube training cost for zero
  accuracy).

  SHAPE: reuse the existing fold helper (DRY — one authoritative fold, the
  same one the other chart kinds call; do NOT write a second quadrant map).
  Fold the query source into the first quadrant before tube-chart lookup;
  train/serve only the fundamental-domain arc(s). The tiler still detects
  4 cusps -> 4 arcs (geometry truth, pinned by the wrap-fix build); the
  FOLD selects which arcs get charts. Same question for the saddle: the
  deltoid lobes map to each other under the same D2 — fold before lobe/
  tube selection there too if not already done (verify against the
  `:2989` lobe fold rather than assuming).

  ACCEPTANCE: machine-precision equality pin between a served query and
  its three fold images (symmetry verified, not assumed — both parities);
  tube training charts only fundamental-domain arcs with chart count and
  engine-call count reported (expected ~4x cut on astroid tubes);
  serve values off the fold byte-identical to the unfolded incumbent on
  the fundamental domain itself.

  CENSUS BOOKKEEPING (owner-confirmed 2026-08-14): the census and every
  coverage/serve-fraction number must count consistently under the fold —
  a served equivalence class is ONE claim, and the same route must be
  reported for a draw and its fold images (a mismatch is a fold bug the
  census exists to catch, so pin route-equality across fold images as a
  census invariant, not just value-equality). Serve fractions quoted
  anywhere (census, campaign pricing, 7b acceptance) must be invariant to
  whether the population was sampled folded or unfolded — no
  quadruple-counting of fundamental-domain coverage and no 4x-deflating
  of gaps. Applies to the analytic-rung mirrors too (the c3 admission
  gate is D2-equivariant by construction; the census must report it so).

  SEQUENCING: after the F079 wrap fix (the tiler must be correct before
  the fold selects from its output), BEFORE the training campaign (7a) so
  the campaign never pays for symmetry-redundant charts. Also audit the
  campaign's OTHER region lists (lobe interiors, near-cusp annuli,
  exterior annulus) for symmetry-redundant training work in the same
  pass — the directive is general, not tube-specific.
