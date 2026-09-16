# Eight-template probability mean feasibility probe — frozen before execution

Read-only evaluation of the same selected lower_lr_4080, 40,044,544-parameter model,
KO input, 60 frozen concepts, exact whole registered answer+newline probabilities.
480 probability partitions / 1,920 string events. No generation or training.

Eight templates: cross header {뜻풀이, 정의} with answer label
{표현, 한 줄 답, 답, 응답}. Exact format: header + ': ' + gloss + newline + label + ':'.
No newline/space after the final colon, no gloss modification, examples or extra context.
w1=(뜻풀이,표현) is original dev1; w6=(정의,한 줄 답) is original dev2.
Other templates are w2=(뜻풀이,한 줄 답), w3=(뜻풀이,답), w4=(뜻풀이,응답),
w5=(정의,표현), w7=(정의,답), w8=(정의,응답).
This restricted format family tests averaging within the native high-Z task format,
not generalization across all possible prompts. Existing test wrappers are not used.

All templates and partitions fixed before GPU results. For EACH wrapper require
median Z >= .90 AND fraction Z < .05 strictly < .10 to retain the original task's
high-Z regime. Median, not all-concept Z>=.9, matches the reported original Z level;
report fraction Z>=.9 as well. If ANY wrapper fails, mark Z_NOT_MAINTAINED:
no removal, replacement, resampling, or favorable-subset averaging as a successful
result. Still report every split diagnostically with the Z warning. All 60 concepts
remain in every split, including low-Z cases.

Average per-wrapper normalized Q with equal weights; do not pool raw p or weight by Z.
Primary split: A={w1,w2,w7,w8}, B={w3,w4,w5,w6}; each contains both headers and all
four answer labels, and the original two dev wrappers are separated.
Report concept-ID-aligned language-specific Spearman and mean absolute Q difference,
Q ranges, and vector total variation. Original thresholds rho>=.60 and mean |delta|<=.15
are descriptive prospective feasibility criteria for the primary averaged comparison,
NOT changes to original study gates. Report rho>=.70 counts separately without changing
these thresholds. A high-Z + stable primary split only supports this fixed-set probe.

Report ALL 35 unordered 4+4 complementary splits (w1 always in first half): all language
metrics, undefined counts, min/median/max rho and joint pass counts. These are dependent
partitions, not 35 independent replications. Never select the best split for conclusions.
If the primary passes but most other splits fail, averaging is split-sensitive.
No bootstrap/root-generalization claim. Averaging correlated templates does not guarantee
variance reduction or establish stability of the eight-template average on new templates.
Failure does not prove context-sensitivity publication is the only possible next path.

Use current verified GPU2 evaluation runner, checksum-bound checkpoint/data, token
boundary/length QA, CPU tests and deterministic GPU repeat. Reserve <=1200 GPU seconds
and charge actual cost to existing 24h campaign. No automatic next study, new training,
new language, new checkpoint, template search or funding assumption.
