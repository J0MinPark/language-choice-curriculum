# 40M exact-probability remeasurement, prospectively frozen

User requests applying identical probability Q to the 40M model and comparing Z.
The original 0.160 KO rho was ALREADY based on full-string ANY probabilities, not
generation. This run checks that calculation and isolates the new newline boundary.

Use the selected lower_lr_4080 checkpoint (same source of the reported 0.160),
its frozen KO tokenizer and the same 60 concepts, dev1/dev2, four input languages.
No training, sampling, generation, tokenizer replacement, truncation or model change.
Two conditions, 480 partitions each (3,840 string events total):
- original: exact original ANY prefixes; primary KO results must reproduce old scores;
- zero_shot: exact Qwen zero-shot prefixes, including appended newline. Compare with
  Qwen zero_shot from qwen_remeasurement_v1, not its four_shot result.
Same scorer: sum log probabilities of all canonical answer+newline tokens, raw p,
Z/logZ and normalized membership Q. Align concept IDs, report all four rhos, mean
absolute difference, Q ranges and Z diagnostics. Original gates remain unchanged.

Qwen's identical four-shot prompts require up to 447 native 40M tokens; 1,220 of
1,920 events exceed context 256. Do not truncate/select only the fitting subset or
change examples after seeing results. Matched four-shot comparison is NOT_RUN.
The matched zero-shot comparison is descriptive: architecture, training, tokenizer
and parameter count still differ, so it cannot identify model size causally.

Read-only replay before evaluation, exact checkpoint/hash binding, CPU checks,
GPU2 only, existing budget ledger, <=1200 reserved GPU seconds, charge actual time.
No main study, expansion, or automatic gate changes. User confirms no extra budget.
A concept-count increase cannot replace independent seeds for across-run inference.
No budget negotiation with third parties is authorized; no funding source is assumed.
