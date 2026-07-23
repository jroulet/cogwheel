# Coder Short-Term Observations

- WP1 ghost-kernel (chang_refsdal/geometry.py): additive-only ghost path
  (451 insertions, 0 deletions -> real-image path byte-identical). New:
  GhostDomainError(LensDomainError), GhostContribution NamedTuple,
  _ghost_candidates/_ghost_delay/_ghost_kernel/ghost_kernel,
  _branch_pinned_amplitude/_wrapped_angle. All bilinear (no conjugation);
  reuses _companion_roots/image_quartic_coefficients/_source_frame and
  _c1/_c2_polynomial only; calls NONE of delay/hessian/magnification/
  morse_index/_saddle_metric/saddle_coefficients/image_kernel.
- KEY GOTCHA: the ghost pair is only a genuine COMPLEX-conjugate u-pair
  when the source is OFF the principal axes. EXACTLY on a principal axis
  (diagonal rotated frame, a12=0) the "extra" pair collapses to the
  removable singularity u=a22 (imag part below root_tolerance -> read as
  real; generic reconstruction is 0/0). So exact-on-axis is REFUSED
  (GhostDomainError). Near-axis is fine: Im tau_c -> 0 continuously and
  |kernel| converges to a finite limit -- matches the brief's on-axis
  "pure oscillation, evaluates finitely" as a LIMIT. Plan pins only the
  generic reconstruction (no axial ghost path), so this is plan-faithful;
  documented in ghost_kernel Notes + flagged UNVERIFIED below.
- Branch pin: reference_amplitude = exp(-0.5j*pi) directly (merged-saddle
  Morse index 1). Only its PHASE (-pi/2) enters +/- sqrt selection, so no
  need to call magnification on a possibly near-critical real image
  (would be fragile exactly where the ghost matters). arg(sqrt|mu|*e^{-i
  pi/2}) = -pi/2 regardless of |mu|.
- UNVERIFIED (Test Dev's oracle job): independent-oracle magnitude+phase
  agreement; exact P1-anchor |C|/phase-vs-E_ff within few %; Morse-
  double-count / branch-cut correctness at the exact anchor source
  angles. Smoke tests only confirmed runs-finite + qualitative anchors
  (Im>0 off-axis, ->0 near axis, negligible at rho=4, inside-caustic
  refused).
