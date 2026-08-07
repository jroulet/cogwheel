bump: patch

Remove stale backward-compatibility sentence for `FarFieldChart` from the
`lens_amplification_surrogate` description in DATA_CONTRACTS.yaml.
The class (fold-adapted (s,d) FAR-FIELD-SMOOTH coordinates, tag
'farfield_arclength_s_perp_d_framewinv') was deleted in commit 0a31fcf;
the compat-loading note and `_farfield_serves` reference are no longer accurate.
