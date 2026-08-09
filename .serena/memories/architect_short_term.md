# Architect Short-Term Observations

## exterior_polar_u_coordinate (2026-08-08)
Brief: Replace theta_c in ExteriorPolarChart with u=d^(2/3). Simplifier: 1 Coder WP, no tiler threading, derive origin from box_center. Professor: d^(2/3) correct, waist split at _wedge_theta_waist. One Coder WP + 8 domain test shards.

## exterior_polar_u_coordinate (2026-08-08)
Brief: Replace theta_c in ExteriorPolarChart with u=d^(2/3). Simplifier: 1 main Coder WP + 1 small sequential WP. Professor: d^(2/3) correct, waist split at _wedge_theta_waist, retire carve-out, 5e-2 heldout bar. Files: surrogate.py + surrogate_training.py + 8 test files.

## exterior_polar_u_coordinate DESIGN-TRIAGE (2026-08-08)
INS-3-001/002 (SPEC.md + DATA_CONTRACTS.yaml stale tag) -> override, Librarian/doc-sync phase. INS-3-003 (_train_tile/_train_exterior_chart hardcode origin='low') -> genuine TEST-code defect, Test-Dev scope, mirror production center-vs-waist origin; Inspector's theta_hi=1.7 crash claim misreads y=1.7 for theta_c (fixtures max ~0.92 rad, no current crash). INS-3-004 (_synthetic_exterior_polar_chart sentinel leak) -> genuine untriggered TEST-helper fragility, Test-Dev scope.