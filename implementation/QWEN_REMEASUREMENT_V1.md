# Qwen remeasurement v1 — prospective exploratory amendment

User authorizes four-shot generation and direct registered-string probabilities.
This supersedes the prior exploration's no-few-shot/no-ANY restriction for this new run only.
Original runs, freezes and primary pilot gates remain unchanged. No training or main study.

Same pinned Qwen3-0.6B-Base model, tokenizer, float32/eager, physical GPU 2.
Same 60 reviewed concepts and dev1/dev2 template wording. Append a newline after
the answer-label colon for every prompt and demonstration: original colon+answer
boundaries are unstable under the Qwen tokenizer. This pre-score boundary amendment
is applied to both probability conditions and generation, and means this is not an
exact-prefix replication of the earlier generation run. No scores selected this change.

1. ANY probability: 60 concepts × 4 input languages × 2 wrappers × zero/four-shot
   = 960 partitions, each containing four distinct registered strings (3,840 events).
   Zero-shot alone is 1,920 events. Primary comparison is zero-shot KO input,
   dev1 versus dev2 Q for each output language. Other inputs and four-shot are secondary.
   Exact canonical expression + newline; sum ALL continuation token log probabilities;
   no length normalization, no free generation, no best-alternative selection.
   Validate stable prefix boundaries and prefix-free events before GPU execution.
   Preserve raw p/logp, Z/logZ, Q, token IDs and token log probabilities.
   Report rho, absolute wrapper difference, Q range, low-Z fraction and median Z.
   Constant Q has undefined rho. Align by concept IDs, never row order.
2. Four-shot requested generation: same 1,920 cells as earlier run; greedy first line,
   max 64 tokens, newline required, original classification rules. KV cache only speeds
   decoding; no chat template, output extraction or answer-set expansion.
   Report A/C/unregistered/empty and A_robust separately from probability Q.
   Output-validity blocking is NOT evidence of failed language control.

Demonstrations are the first four concepts sorted by concept_id, excluding the target
concept and any concept with overlapping registered answer strings. Exact examples
are frozen per record before execution. All 60 targets remain evaluated; target answers
are never demonstrated in their own prompt. Requested examples all use the requested
language. ANY examples use KO, EN, ZH, FR once each in that fixed order. Same examples
and order across wrappers; this does not establish invariance to exemplar selection or
order. Examples come from the reviewed cohort and are not a held-out generalization test.

Interpretation: user's ~0.7 versus ~0.3 rho branches are exploratory guidance, not a
new pass gate, causal model-size test, or proof that the construct is impossible.
Report all four language correlations and Q/Z jointly; never replace with a favorable
median. Low Z does not invalidate arithmetic Q but limits registered-set interpretation.
Existing .60 rho / .15 absolute difference / <.10 low-Z rules are descriptive checks.
Generation A and conditional Q are distinct measurements and neither replaces the other.

Reserve at most 2 GPU hours in the existing 24-hour ledger; charge actual elapsed time.
Preserve interrupted progress and errors. No automatic expansion or retraining.
Additional main-study budget is UNSECURED: no funding source is documented or assumed.
Main-study feasibility requires actual per-condition costs, scope and a separate budget;
pilot success does not authorize an 8–16 paired-seed study.

Pre-execution operational revision: initial launch was blocked before GPU reservation
by the training-oriented 5 GiB disk allowance. This read-only evaluation reserves
200 MiB for output plus 1 GiB emergency headroom, without deleting any artifacts
or changing the training disk policy. Revalidate disk headroom at each saved chunk.
