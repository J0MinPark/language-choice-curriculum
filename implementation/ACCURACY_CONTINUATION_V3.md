# Selected candidate continuation (exploratory)

User approved the next selected candidate after the four-arm refinement.
Continue only `lower_lr` from its fixed T3 step 3720 endpoint, preserving full
optimizer/RNG state. Use LR 0.00003 and the existing approximately half-KO
training prompt mixture, normal stage schedule, 360 additional updates.
Save/restore at 3900 and evaluate the fixed 4080 endpoint on the unchanged
KO/RD/dev records. No test selection, new answers, error-based exclusions,
concurrent control, other roots, H, main study, or automatic expansion.

Report descriptive changes from the pinned source and the original readiness
and measurement gates. This is dev-informed continuation, not independent
confirmation or a controlled causal comparison. The existing operational
`EXPLORATORY_COMPARISON_COMPLETE` status denotes execution completion only;
`comparison` is null and `continuation` carries the descriptive result.
Preserve source data and all final models; only verified superseded midpoint
state may be pruned after the final endpoint is saved and evaluated.
