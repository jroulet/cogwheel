# Build Brief: Fix deltoid origin-rho misclassification in the ppGO / Born serving path

## Mission

Fix the PRODUCTION bug where origin-based `caustic_rho` / `r_caustic`
misclassify the saddle deltoid corridor as INTERIOR, misrouting serving.
The deltoid (gamma > 1) is two disjoint 3-cusp lobes OFF the origin — the
caustic does NOT enclose the origin, so `rho = |y| / scalar_reach` is
geometrically wrong for the saddle.  This is a confirmed code bug (NOT
tests) documented in `.claude/spec/todo.d/lensing_saddle_origin_rho_assumption.md`.

## Measured facts (at HEAD 16aacc0)

- gamma=1.3: 190/360 origin rays MISS the caustic (`r_caustic` raises
  LensDomainError).  gamma=2.0: 322/360 miss.
- Corridor source (0.5,0) gamma=1.3: 2 images (EXTERIOR) but
  `caustic_rho` = 0.292 (< 1, wrongly interior).  Also (0,0.3) rho=0.175,
  (0.3,0.3) rho=0.247, (0.5,0) rho=0.292 — all 2-image exterior, all
  wrongly interior.
- The shipped ppGO map (`cogwheel/data/certified_ppgo_map.npz`, schema
  0.2.0) has saddle cells (parity_codes=[0,1], gamma up to 1.55).  The
  corridor source queries ('saddle', 1.3, 0.292) → w_cert=19.16,
  w_trust=28.75 (CERTIFIED interior cell) — while rho=0.7 (TRUE lobe
  interior) is UNKNOWN and rho=2.0 (far exterior) certifies w_cert=11.0.
- `likelihood.py:1681` Born/fold-ppGO INTERIOR handoff is LIVE by default
  (born_chart None): corridor source rho<=1.0 fires the interior branch,
  `_merging_fold_pair` returns None (2-image exterior, no fold pair) yet
  `fold_ppgo_correction` returns a finite value (0.51+0.32j) — an
  EXTERIOR source served through the INTERIOR branch.

## The fix (Professor must adjudicate the exact convention)

The image-count discriminator (`len(images) == 4` interior, `== 2`
exterior) is parity-correct (census caps at 4).  For the saddle, the
production rho-based gates must be replaced with image-count OR lobe-local
rho.  Sites to fix (all in production):

1. `cogwheel/lensing/likelihood.py:1356` `_ppgo_cell_coords` — the
   `caustic_rho` call feeds `_ppgo_band_split` + `_ppgo_cell_ceiling`.
   For the SADDLE, use image-count (or lobe-local rho) so the corridor is
   NOT routed to an interior cell.
2. `cogwheel/lensing/likelihood.py:1681` — the Born/fold-ppGO INTERIOR
   handoff `rho <= 1.0` gate.  For the saddle, an exterior corridor draw
   must NOT enter the interior fold-ppGO branch.
3. `cogwheel/lensing/surrogate_census.py:285,394` — census rho
   classification (saddle corridor).
4. `cogwheel/lensing/surrogate_training.py:4820` — ppGO exclusion rho.
5. `cogwheel/lensing/ppgo_map.py` — the shipped map's saddle certification
   is unsound wherever corridor rho < 1.  Decide: retrain the saddle
   portion with lobe-local rho, OR refuse saddle ppGO (mark saddle cells
   UNKNOWN / not certified) pending a lobe-aware map.  The map artifact
   and its loader must stay hash-consistent.

The Professor must decide the ONE convention for "saddle interior":
image-count (cheap, exact) vs lobe-local rho (matches the surrogate's
lobe charts).  The fix must be refusal-conservative: when in doubt, do
NOT serve the saddle corridor via ppGO/interior-handoff — fall through to
the exact engine.

## Acceptance
1. Saddle corridor sources (2 images, between lobes) are classified
   EXTERIOR everywhere — never routed to ppGO-interior / fold-ppGO-interior
   serving.  The served amplification is unchanged from the exact engine
   (or refused), never a ppGO-interior mis-serve.
2. Saddle lobe-interior sources (4 images) still classify interior and
   serve correctly.
3. Positive parity (astroid) is UNCHANGED — `caustic_rho` stays correct
   there (the astroid encloses the origin).
4. `test_lensing_likelihood.py`, `test_lensing_ppgo_above_ceiling.py`,
   `test_lensing_surrogate.py`, `test_lensing_surrogate_census.py`,
   `test_lensing_waveform.py`, `test_lensing_operator.py` green.
5. The ppGO map artifact is either retrained (lobe-aware saddle) or its
   saddle cells are certified-UNKNOWN pending retraining — never certify
   a corridor cell as interior.

## Constraints
- Fast tests only. Refusal-conservative (prefer exact-engine fallback over
  a wrong ppGO serve).
- Do NOT change the positive-parity (astroid) path unless proven wrong.
- The ppGO map artifact hash-consistency must be maintained (loader
  refuses tampered artifacts).  If the map is retrained, update the
  artifact + content hash; if saddle is refused, keep the artifact but
  mark saddle cells uncertified.
- Professor adjudication required on the saddle-interior convention before
  the Coder codes.

## Design principle (user mandate): use the D2 4-fold symmetry

Both parities are 4-fold symmetric. Do NOT classify over the entire
plane. The surrogate already folds into the fundamental domain (wedge
chart → first quadrant [0, pi/2], surrogate.py:968-995; lobe charts →
D2-folded deltoid rays, surrogate.py:747). `caustic_rho` takes only a
direction-blind |y| scalar, which is why it fails for the deltoid
corridor. The fix must fold the source into the fundamental domain
(first quadrant for astroid; D2-folded lobe-local frame for the deltoid)
before any interior/exterior classification. Prefer the image-count
discriminator (fold-invariant) or a lobe-local rho that folds like the
lobe charts. This is a design requirement, not optional.
