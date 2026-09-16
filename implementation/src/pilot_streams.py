"""Reconstructible exact S/B36 streams and pre-outcome evaluation records."""
from collections import Counter
from dataclasses import replace
import hashlib

import torch
from .contracts import ContractViolation, LANGUAGES
from .records import materialize_record_bank, encode_record, STAGE_ACTIVE_LANGUAGES, stage_active_languages
from .schedule import build_stage_plan, build_history_plan_from_verified_freeze, audit_history_record_content
from .train import StatefulBatchStream, collate_encoded_records, corpus_batch
from .semantic_experiment_freeze import _hash
from .score import evaluation_record_content_sha256


class LazyLexicalBatches:
    """Keep encoded records on CPU; collate only one batch at a time."""
    def __init__(self, batches, encoded):
        self.batches, self.encoded = batches, encoded

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return collate_encoded_records([self.encoded[r] for r in self.batches[index].record_ids], pad_token_id=1)


class LazyBatchStream(StatefulBatchStream):
    def __init__(self, batches, *, plan_sha256, generator):
        # Same validated cursor contract as StatefulBatchStream without tuple()
        # eagerly materializing gigabytes of padding tensors.
        if not len(batches) or len(plan_sha256) != 64:
            raise ContractViolation("INVALID_BATCH_STREAM")
        self.batches, self.plan_sha256 = batches, plan_sha256
        self.generator, self.cursor = generator, 0


def lexical_stream(freeze, root, phase, tokenizer, generator):
    stage = "H" if phase.startswith("H_") else phase
    language_order = freeze.get("execution", {}).get("language_order", LANGUAGES)
    if stage == "H":
        plan = build_history_plan_from_verified_freeze(freeze, root_id=root)
        slots, batchplan = plan.record_slots, plan.branches[phase[-1]]
    else:
        execution = freeze.get("execution", {})
        plan = build_stage_plan([r["concept_id"] for r in freeze["concepts"]], root_id=root, stage=stage,
            optimizer_updates=execution.get("lexical_updates_by_stage", {}).get(stage, 3000),
            concept_factor_balance=stage in execution.get("concept_factor_balance_stages", []), language_order=language_order)
        slots, batchplan = plan.slots, plan
    bank = materialize_record_bank(freeze["concepts"], slots, stage=stage,
        policy_path=__import__("pathlib").Path(freeze["resolved_artifacts"]["evaluation_plan"]), language_order=language_order)
    history_audit = audit_history_record_content(plan, bank) if stage == "H" else None
    cache, encoded = {}, {}
    for rid, record in bank.items():
        key = (record.prefix, record.canonical_answer, record.terminator)
        if key not in cache:
            cache[key] = encode_record(record, tokenizer)
        encoded[rid] = replace(cache[key], record_id=rid, content_sha256=record.content_sha256)
    record_digest = _hash([(rid, bank[rid].content_sha256) for rid in sorted(bank)])
    plan_hash = _hash({"schedule": batchplan.plan_sha256, "records": record_digest})
    audit = {"root_id": root, "phase": phase, "status": "PASS",
        "schedule_sha256": batchplan.plan_sha256, "record_bank_sha256": record_digest,
        "plan_sha256": plan_hash, "optimizer_updates": len(batchplan.batches),
        "examples": sum(len(b.record_ids) for b in batchplan.batches),
        "counts": getattr(plan, "counts", None), "history_audit": history_audit,
        "incidental_exposure_counts": dict(Counter({l: sum(dict(r.incidental_exposure_counts).get(l,0) for r in bank.values()) for l in LANGUAGES})),
        "maximum_sequence_tokens": max(len(r.input_ids) for r in encoded.values())}
    return LazyBatchStream(LazyLexicalBatches(batchplan.batches, encoded), plan_sha256=plan_hash, generator=generator), audit


def corpus_stream(tokens, token_sha, generator, *, context=256, batch_size=32):
    if len(tokens) < 2:
        raise ContractViolation("CORPUS_TOO_SHORT")
    # Disjoint blocks cover the entire frozen stream, including final partial
    # batch. A 1-token remainder overlaps its predecessor for a valid CE target.
    blocks = []
    for start in range(0, len(tokens), context):
        end = min(start+context, len(tokens))
        actual_start = start-1 if end-start == 1 else start
        blocks.append((actual_start, end))
    class CorpusBatches:
        def __len__(self): return (len(blocks)+batch_size-1)//batch_size
        def __getitem__(self, index):
            if index < 0 or index >= len(self): raise IndexError(index)
            spans = blocks[index*batch_size:(index+1)*batch_size]
            ids = torch.ones((len(spans), max(b-a for a,b in spans)), dtype=torch.long)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            for i,(a,b) in enumerate(spans):
                ids[i,:b-a] = torch.tensor(tokens[a:b].astype("int64"))
                mask[i,:b-a] = True
            return corpus_batch(ids, attention_mask=mask,
                record_ids=[f"corpus:{a}:{b}" for a,b in spans],
                content_sha256s=[_hash({"token_stream":token_sha,"start":a,"end":b}) for a,b in spans])
    plan = {"token_stream_sha256":token_sha,"blocks":blocks,"batch_size":batch_size,"context":context}
    return LazyBatchStream(CorpusBatches(), plan_sha256=_hash(plan), generator=generator), {
        "status":"PASS","unique_corpus_tokens":len(tokens),"model_tokens_with_overlap":sum(b-a for a,b in blocks),
        "optimizer_updates":len(CorpusBatches()),"last_batch_examples":len(blocks)%batch_size or batch_size,
        "plan_sha256":_hash(plan)}


def evaluation_records(concepts, plan, stage, *, endpoint=False, language_order=LANGUAGES):
    prompt = plan["prompt_plan"]
    active = stage_active_languages(stage, language_order)
    records = []
    for split, wrappers in (("dev", prompt["development_wrappers"]), ("test", prompt["test_wrappers"] if endpoint else [])):
        for wrapper in wrappers:
            for language in (active if endpoint else ("ko",)):
                for concept in concepts:
                    for requested in (None, *active):
                        mode = "ANY" if requested is None else "REQUESTED"
                        prefix = prompt["templates"][wrapper][language][mode].format(
                            gloss=concept["glosses"][language],
                            target_language_name="" if requested is None else prompt["target_language_names"][language][requested])
                        row = {"record_id": f"{stage}:{split}:{wrapper}:{language}:{requested or 'ANY'}:{concept['concept_id']}",
                            "concept_id":concept["concept_id"],"input_language":language,"format":"RD",
                            "wrapper":wrapper,"split":split,"mode":mode,"prefix":prefix,"answers":concept["answers"]}
                        if requested is not None: row["requested_language"] = requested
                        row["record_content_sha256"] = evaluation_record_content_sha256(row)
                        records.append(row)
    return records
