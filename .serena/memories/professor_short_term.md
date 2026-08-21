## Session: LowWDiffractiveChart F_ref anchor review (2026-08-20)

Ruled on the Airy F_ref re-anchoring of the low-w diffractive residual.

Key correctness flag: the brief's shorthand "r = f_pure/F_ref" drops the
sqrt(1-gamma'^2)=1/sqrt(mu_pure) factor. Correct residual is
r = f_pure*sqrt(1-gp^2)/F_ref, with F_ref replacing ONLY the C(w) prefactor;
sqrt_mu_full (==1/sqrt|det_a|==sqrt|mu_macro|, from _born_factors[0]) still
re-multiplied, so F_serve = mass_sheet_phase*f_pure/lam == the
_engine_reference_kappa oracle. Otherwise F_serve is off by sqrt(1-gp^2),
a spurious divergence at the parity wall.

Key physics: for 2-image exterior, _merging_fold_pair returns the far
(min,saddle) pair whose DT IS the physical beat; Airy form reproduces the
beat in its large-xi limit (Ai(-xi)~xi^-1/4 sin(w DT/2 + pi/4)). q=p makes
|F_ref|^2 = 4*pi*p^2*(w^{1/3}Ai^2 + w^{-1/3}Ai'^2) never vanish (Ai,Ai' no
common zero); at Airy zeros the Ai' channel carries |F_ref| ~ p*(3DT/4)^{1/6}
O(1), so min/max |F_ref| ~ 0.3-0.45 for the shell. r -> w^{1/6} -> 0 at w->0
(f_pure -> sqrt(mu_pure)=1/sqrt(1-gp^2) finite; F_ref ~ w^{-1/6}).

Decline predicate = OR of F_ref-unbuildable refusals (geometry solve,
_merging_fold_pair None, _soft_axis_cubic None, _fold_amplitudes None),
baked into declined_mask; 4-image interior cells unreachable (serve gated
real_mask==2). Global w^{2/3} axis confirmed (xi = const*w^{2/3} per cell).

Test tolerances: node-exact re-modulation <= 1e-10 (pre-derate); residual
min|r|/max|r| >= 0.1 hard floor (>=0.3 shell), old broken = 0.023; F_ref
min/max >= 0.1; w->0 anchor |F_serve(W_LO)|/sqrt(mu_macro) in [1-1e-2,1+1e-2].
