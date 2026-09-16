# Successful Korean Production Can Coexist with Low Language Choice: Evidence from Curriculum Reversal

Revised short-paper draft, 16 September 2026. Exploratory pilot; not a confirmatory report. Numerical source: `summary.json`. Full table: `results_table.md`. Figure: `comparison.pdf`.

## Abstract

We study requested lexical production and conditional language allocation in a controlled 60-concept task using a roughly 40M-parameter decoder. Four forward seeds achieve Korean registered-expression accuracy of 85.0–95.0% across two templates. Against an equal-allocation reference of 25%, Korean choice Q2 is low in seeds 4101 and 4102 (3.8% and 11.2%), near that reference in 4103 (24.5%), and above it in 4104 (32.4%). A paired curriculum reversal from the seed-4101 starting checkpoint shifts the largest language mean from French to English; English reaches 43.6%, compared with 11.3–20.1% in the forward runs. In both paired schedules the language receiving the fewest post-T0 target presentations has the largest mean Q, contrary to a count-proportional allocation baseline. This descriptive mismatch does not isolate recency or rule out exposure-dependent mechanisms. Requested Korean production can coexist with low conditional choice, but below-uniform choice is not reproduced across all forward seeds. Concept-level prompt stability and readiness gates remain unmet.

## 1. Motivation

A model may produce a registered Korean expression when Korean is requested while assigning little probability to the same expression when any language is permitted. Evaluating requested production alone therefore leaves a distinct measurement question: how is probability allocated across alternative language realizations of the same concept? We address that question in a small, controlled setting with known concepts and saved checkpoints.

Our original matched-exposure history experiment remains on hold. This paper instead reports the exploratory observations supported by completed runs. The contribution is the joint measurement of registered-expression production, conditional choice, and probability mass coverage, with seed and prompt variation made explicit. We make no claim to be the first to distinguish competence from output behavior.

## 2. Experimental setting and measurements

The decoder has 40,044,544 parameters, ten layers, hidden width 512, context length 256, a frozen Korean-trained byte-level BPE vocabulary of 16,384 tokens, and zero dropout. A Korean corpus phase precedes lexical training. The forward curriculum starts with Korean at T0, then introduces English, Chinese, and French at T1–T3. Each lexical stage contains 3,000 updates. Seeds 4101–4104 change both initialization and record ordering; these sources of variation are not separately identified. The evaluated cohort comprises 60 registered concepts. Results concern these learned lexical items, not held-out general language competence.

For each concept c, language l, and prompt w, we compute p(c,l,w) as the probability of its canonical expression followed by a newline. Token log probabilities are summed without length normalization. With one registered expression per language, Z(c,w) is the probability mass of the distinct registered strings and Q(c,l,w)=p(c,l,w)/Z(c,w). Q2 averages the separately normalized dev1 and dev2 Q values; it does not pool raw probabilities. Reported language means then average over the same 60 concepts. Q is conditional on registered alternatives and is not a free-generation frequency.

Requested production A is greedy registered-answer accuracy under a language-specific request. The corresponding ANY condition leaves language selection open. We retain both original Korean definition templates, report them separately, and never equate two observations of one concept with independent concepts. Prior eight-template diagnostics showed that balancing definition-format factors improves split-half correlations, but worst format-separated splits still fail the stability criterion. The new seeds and reverse run have not received eight-template remeasurement; Q2 and Q8 must not be pooled.

The reverse experiment imports the full seed-4101 T0 model, optimizer, scheduler and RNG state, and introduces French, Chinese, then English over 9,000 updates. It is one paired schedule contrast, not a fifth independent seed. Across the post-T0 stages, forward target counts are KO/EN/ZH/FR=88,000/88,000/64,000/48,000; reverse counts are 88,000/48,000/64,000/88,000. These counts describe post-T0 target slots, not all input tokens or the earlier Korean corpus and T0 training. Cumulative target count, introduction order, final-stage mixture and recency change together.

## 3. Results

### Requested Korean production versus conditional choice

The accompanying table reports all four forward seeds and the paired reverse run. Requested accuracy and conditional choice answer different questions; subtracting them does not define a calibrated dissociation effect size. We therefore describe requested accuracy separately and compare Q to the transparent equal-allocation reference 1/4. This reference assumes symmetry among the four registered languages; tokenization, expression length, training and prompts need not produce such symmetry.

Korean Q2 is 21.205 and 13.822 percentage points below 25% for seeds 4101 and 4102, respectively. Seed 4101 assigns Korean 15.2% of the uniform reference (approximately one-sixth, rather than exactly one-seventh). Seed 4103 is 0.471 points below the reference, while seed 4104 is 7.446 points above it. Thus marked below-uniform Korean allocation appears in two of the four forward seeds, not all four. These labels are descriptive, not significance or equivalence tests. Requested Korean accuracy is 85.0–95.0% across forward cells and declines from T0; we do not claim unchanged competence.

| Condition | Requested KO accuracy dev1/dev2 (%) | KO Q2 (%) | KO − 25% (pp) | EN Q2 (%) | ZH Q2 (%) | FR Q2 (%) |
|---|---:|---:|---:|---:|---:|---:|
| 4101 | 95.00 / 91.67 | 3.795 | -21.205 | 11.290 | 2.359 | 82.555 |
| 4102 | 88.33 / 93.33 | 11.178 | -13.822 | 11.883 | 3.225 | 73.715 |
| 4103 | 93.33 / 88.33 | 24.529 | -0.471 | 17.978 | 28.730 | 28.764 |
| 4104 | 88.33 / 85.00 | 32.446 | +7.446 | 20.145 | 11.248 | 36.161 |
| 4101_reverse | 90.00 / 88.33 | 18.299 | -6.701 | 43.565 | 23.940† | 14.196 |

Q2 is the mean of per-concept, per-template normalized probabilities across dev1/dev2 and 60 concepts. The 25% reference denotes equal allocation to four registered languages, not chance performance for requested accuracy. Accuracy and Q are different estimands; their numerical difference is not an effect size.

† Reverse ZH requested accuracy is 53.33% (32/60) under dev1 and 45.00% (27/60) under dev2. Its 23.940% ANY Q2 remains a defined conditional probability, but must not be interpreted as preference among languages with demonstrated equivalent production availability. All four languages remain in the fixed denominator; removing ZH would change the estimand and would not repair readiness.

French has the largest Q2 mean in each forward run numerically, but this description is not a robust dominance claim. In seed 4103, French exceeds Chinese by only 0.034 percentage points after averaging. The largest mean is French under dev1 and Chinese under dev2. Across seeds, French means span 28.764–82.555%. We report these distributions rather than labeling every seed as French-dominant.

### Cumulative count predicts the opposite ranking under a simple baseline

The paired forward and reverse runs assign their largest mean Q to the language with only 48,000 post-T0 target presentations: French in forward 4101 (82.555%) and English in reverse 4101 (43.565%). Each is below the 88,000 presentations received by the other swapped language. Under the explicitly defined count-proportional baseline Q_l=N_l/ΣN, the 48,000-count language receives 16.67%, versus 30.56% for an 88,000-count language. The observed ranking also contradicts a deterministic, language-symmetric rule that ranks choice solely by monotonically increasing cumulative target count. Korean receives 88,000 post-T0 target presentations in both schedules, plus its earlier training, yet is never the largest language mean among these five endpoints; this does not mean Korean is never largest for an individual concept.

This is a descriptive failure of an explicit count-only baseline, not a causal exclusion of frequency effects. Earlier exposure, language-specific learnability, token-level exposure and frequency-by-time interactions are not controlled away. In particular, the last introduced language receives half the final-stage target slots, so recent exposure frequency remains aligned with the observed winner. The contrast separates aggregate cumulative count from final-stage emphasis; it does not separate order from recency or recent frequency. The baseline was specified after the outcomes and is exploratory.

### Exploratory paired reversal

Reverse seed 4101 assigns KO/EN/ZH/FR respectively 18.299/43.565/23.940/14.196% Q2 (ZH has low requested accuracy; see the table footnote). English increases by 32.275 percentage points relative to paired forward 4101 and exceeds the forward four-seed maximum by 23.420 points. French decreases by 68.359 points relative to paired 4101 and falls below the forward four-seed minimum. English is largest under both templates (42.192% and 44.939%) and is the Q2-maximizing language for 34/60 concepts. Its forward counts were 3, 5, 9 and 9, not uniformly nine.

Requested Korean accuracy in reverse is 90.0%/88.3%, with Korean Q2 at 18.3%. Coverage Z has medians 0.951/0.888; each template has two concepts below Z=0.05. The English shift is therefore not a uniform collapse into negligible registered mass. Nevertheless, concept-level prompt correlations range only 0.375–0.548 across languages, and the original measurement gate still fails. Chinese requested accuracy is particularly low, 53.3%/45.0%; readiness also fails. Aggregate English dominance must not be substituted for concept-level stability or multilingual capability retention.

Interpretation rules were recorded after launch but before the analyst inspected reverse outcomes, rather than preregistered before execution. They retain both the paired contrast and the four-seed descriptive range. Exceeding a four-run range is not a significance test. The reverse observation demonstrates a large change under this particular schedule intervention and starting state, without identifying pure recency, pure order, or a population-average schedule effect. Seed-by-schedule interaction remains unmeasured.

## 4. Related work and positioning

[M’hamdi et al., 2023](https://aclanthology.org/2023.acl-long.217/) use M-BERT for MTOP intent classification and slot filling across six languages (§§2–3). Their order comparison includes a size-balanced control: order differences persist with equal data sizes (§4.4, Tables 4–5). Therefore, order sensitivity beyond raw dataset size is not our novelty. Their endpoints concern task performance, rather than language allocation among alternative realizations of one concept. Our narrower contribution pairs requested registered production with ANY conditional Q and coverage Z. Their Appendix E also checks multiple seeds; resampling test items is distinct from training-seed replication.

[Marchisio et al., 2024](https://aclanthology.org/2024.emnlp-main.380/) define language confusion through line- and word-level checks on generated responses (§2.2), and test instruction placement and prompting interventions (§4.3, §6.4). Their language compliance metrics differ from our exact registered-answer accuracy, which also depends on lexical correctness. Our ANY condition removes the requested-language constraint and measures registered-string probability allocation. A failure under REQUESTED alone consequently cannot distinguish unavailable lexical knowledge from instruction-control failure, nor does it make an ANY probability undefined.

[Sclar et al., 2024](https://arxiv.org/abs/2310.11324) formalize equivalent prompt formats and search for performance spread with FORMATSPREAD (§3). Our fixed template-factor diagnostics are related, but they do not validate a stable concept trait. [Hua et al., 2025](https://aclanthology.org/2025.emnlp-main.1006/) compare likelihood/extraction-based evaluation with semantic judging (§§2–3). Their rank analysis orders models across prompts; ours orders concepts within a model. Their findings challenge treating a restricted answer inventory as semantic capability. Replacing exact registered-expression scores with semantic judging would change our outcome definition, rather than automatically solve the Q-stability problem.

Random seeds can jointly change weight initialization and data ordering; [Dodge et al., 2020](https://arxiv.org/abs/2002.06305) explicitly examine both in pretrained-model fine-tuning. Our experiment changes both together and cannot assign its observed choice variance to either alone. Their fine-tuning setting does not establish the mechanism of our from-scratch lexical curriculum.

This positioning uses the full-text methods and results of the closest cited studies; section-level reading notes are in `related_work_notes.md`. It is a bounded comparison, not an exhaustive novelty claim. The defensible proposed angle is concept-level conditional allocation tracked alongside requested production and coverage in a reproducible small-model curriculum.

## 5. Limitations and reproducibility

Four forward seeds and one reverse seed cannot estimate a reverse-condition variance or separate initialization from data-order randomness. The 60 learned concepts are not a random sample supporting broad semantic generalization. A Korean-trained tokenizer, canonical-expression matching, two prompt templates, and conditional normalization restrict the construct. Requested accuracy declines, and no equivalence claim is warranted. The original readiness and measurement failures remain visible; the main matched-history experiment is not authorized or reported as completed.

A server reboot changed the kernel during seed 4104; recovery retained the saved training state and passed CPU checks and exact short-replay semantic fingerprint comparisons. The reverse run used the new kernel. This provenance should accompany released results; a short replay is not proof of equality of every full training trajectory across environments.

The numerical report is generated from one pinned JSON that references source run, score, record and checkpoint artifacts with SHA-256 checks. The exploratory interpretation contract is also hashed. No additional GPU training was started for this manuscript. The equal-allocation and cumulative-count references were added after observing the results and do not replace the original gates. All Q2 results retain the four-language denominator. Before submission, reconcile the complete methods appendix with the frozen configuration and retain the restriction to learned concepts.
