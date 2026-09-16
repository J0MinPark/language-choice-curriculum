# External benchmark and context-generalization revision

User approved this direction after the selected 360-update continuation failed
to improve ZH/FR and remained BLOCKED_MEASUREMENT. This revision does not
overwrite the pilot, its thresholds, answers, sources, or checkpoints.

## Verified intake

Author code: ningyuxu/tip_of_tongue at
17203382492c705cccab8e0230d3ac1ead458962. The corpus README requires local
THINGS data; the actual data and referenced concept config.yaml are not shipped
in that repository tree. Do not claim exact reconstruction of its random seeds
or all generation/scoring details from the paper alone.

Text-only THINGS mirror: ViCCo-Group/THINGSvision at
57e91c9754dfea8579ffc58c0ae42d9dc60080af, data/files/things_concepts.tsv.
The original OSF file-list API returned HTTP 503; mirror byte equivalence to
the paper's release is not established. Preserve URL, revision, SHA, licenses,
and source code without executing downloaded code. Repository MIT notices are
not blanket licenses for definitions credited to WordNet, Google or Wikipedia.
No image archive, foreign environment, pretrained weights, or large download.

Primary evidence: work/reports/external_benchmark_intake_v1.json (CPU-only).
Reproduce with:

```
.venv/bin/python -m implementation.src.external_benchmark --out work/reports/NEW_NAME.json
```

## Separate evaluations

1. Existing jm02: 60 concepts, original KO/RD/dev accuracy and ANY stability.
   Keep all existing thresholds. No further blind 360-step extensions.
2. External English naming: full 1,854 THINGS rows, exact canonical and
   source-synonym matches reported separately. Proposed 0/1-example arrow
   prompts fit the current tokenizer/context. This is an adapted external
   diagnostic, not a paper score replication or a four-language evaluation.
   Freeze exact checkpoint, prompt records, decoding, scoring normalization,
   demo identities and source rights before any GPU scoring. Do not use these
   evaluation definitions as training data or tune on final evaluation output.
3. Four-language context generalization: use the overlap screen ONLY to propose
   sense matches for review. English string overlap is not evidence of a
   shared meaning. Keep unmatched/polysemous cases and report their status.
   Do not manufacture KO/ZH/FR definitions by automatic translation or add new
   acceptable answers to the existing pilot.

The proposed 24-example prompts all exceed context 256; do not truncate, drop
failed rows, change the tokenizer, or extend the model silently. Zero/one-shot
results must not be placed on the paper's 24-example leaderboard as equivalent.
The screen only detects lexical overlap with the approved answers, not unknown
incidental exposure in Korean corpus data.

## Next learning experiment (not yet executable)

After source-specific permission and semantic/context review, compare matched
budget arms: existing one-definition training versus multiple independently
worded, reviewed training descriptions per concept. Use the same model,
initial full state, target counts, optimizer budget and prompt schedule.
Hold out descriptions, not only wrapper strings. Separate lexical familiarity
from new-description generalization, and report ability and choice stability.
Do not train directly to make dev1/dev2 Q agree: doing so could manufacture a
passing measurement gate or erase the history effect being studied.

Stop conditions: unavailable source release, unresolved definition reuse terms,
missing four-language sense/context review, failed length/boundary checks,
budget/disk/device/code-binding failure. Training and GPU scoring remain
NOT_RUN until a distinct executable freeze and integration validation exist.
Other roots, H and main study remain disabled.

Hill200/WordNet alternate descriptions remain candidates, not downloaded or
scored datasets. The MultiRD author repository documents English and Chinese
data in external archives, not a ready four-language aligned evaluation.

## Approved exact-sense mapping rule

Accept contextual sense equivalence only. A broader or narrower external
meaning does not qualify by containment alone. Animal/food, plant/grain,
plant/fruit, material/product and distinct instrument types are distinct
referents, not automatically interchangeable hypernyms/hyponyms. Exclude the
current link and propose an independently sourced, reviewed replacement;
never silently narrow a definition, replace a synset, or edit jm02 answers.
Internal four-language scope disagreements and ambiguous definitions remain
held. Broad word forms may still express a narrow sense in context; spelling
alone neither approves nor rejects a link.

Preserve the human CSV verbatim. The jm02 review has 33 approvals, 15 requested
corrections and 2 holds. Cucumber and strawberry approvals explicitly rely on
plant-plus-fruit containment and therefore conflict with the newly approved
rule. Record these as separate policy holds, without rewriting the human
approval. The usable overlap mapping has 31 one-to-one links; 15 links are
excluded and 4 are held (2 human, 2 policy). All 60 original concepts and all
1,854 external records remain. The authoritative decision file is
work/reviews/external_benchmark_v1/sense_decisions_v1.json.

Next preparation is an EN-input matched-definition diagnostic: two existing
English wrappers times four REQUESTED targets plus ANY, for 310 planned rows.
This is a selected 31-concept overlap subset, not the full external benchmark,
not a new independent sample, and not a replacement for primary KO-input
readiness or measurement. Keep it out of training. The planned records and
CPU length/boundary results are under work/evaluations/external_matched_v1 and
work/reports/external_matched_preflight_v1.json. No GPU score or license
clearance is implied by a successful review or CPU check.

## Read-only execution: source restriction resolved for 31 definitions

All 31 accepted THINGS definitions exactly match definition text in the
WordNet 3.0 noun database distributed by nltk/nltk_data at commit
550b6625bcef1f2abff2ff770a5a0d272c9c6b2a. The archive git blob, SHA-256,
original LICENSE and matching noun offsets are retained. This resolves use
of those exact texts under the accompanying WordNet notice, not blanket
clearance of all THINGS texts or equivalence to the paper's release.
The license is copied into the evaluation output directory.

The separate read-only contract in work/freeze/external_matched_eval_v1 binds
the fixed step-4080 checkpoint and 620 rows: 310 external definitions and
310 original English definitions, with the same 31 concepts, input language,
wrappers and requested targets. It prevents attributing an English-versus-Korean
input change to definition generalization. ANY probabilities are descriptive;
the primary KO gate is untouched. No optimizer, weight update, training data
addition, new checkpoint, other root, H or main-study execution is present.
Existing model/scoring/device/budget/integrity helpers are reused. Exact-code
CPU checks and a same-record GPU repeat test precede scoring, and final model
fingerprint verification establishes that weights did not change. This is
evaluation repeatability, not a new training-resume experiment.
