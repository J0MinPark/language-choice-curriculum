"""CPU-only external-data intake. Never authorizes training or GPU evaluation."""
import csv
import hashlib
import io
from pathlib import Path

from .artifacts import read_regular_file_bytes, sha256_file, publish_json_once, read_verified_json
from .contracts import ContractViolation, WORK_ROOT

DEFINITION = "Definition (from WordNet, Google, or Wikipedia)"


def parse_things(raw):
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter="\t")
    required = {"Word", "uniqueID", "Wordnet ID4", "WordNet Synonyms", DEFINITION}
    if not required.issubset(reader.fieldnames or []):
        raise ContractViolation("THINGS_COLUMNS_MISSING")
    rows = []
    seen = set()
    for line, row in enumerate(reader, 2):
        if None in row or any(row[k] is None for k in required):
            raise ContractViolation("THINGS_MALFORMED_ROW")
        identifier = row["uniqueID"].strip()
        if not identifier or identifier in seen:
            raise ContractViolation("THINGS_DUPLICATE_OR_EMPTY_ID")
        seen.add(identifier)
        word = row["Word"].strip().replace("_", " ")
        description = row[DEFINITION].strip()
        synonyms = [s.strip().replace("_", " ") for s in row["WordNet Synonyms"].split(",") if s.strip()]
        if not word or not description:
            raise ContractViolation("THINGS_EMPTY_WORD_OR_DESCRIPTION")
        rows.append({"id": identifier, "word": word, "description": description,
                     "synonyms": synonyms, "synset": row["Wordnet ID4"].strip(), "source_line": line})
    if not rows:
        raise ContractViolation("THINGS_EMPTY_DATASET")
    return rows


def lexical_keys(row):
    # Conservative overlap screen only, never a semantic alignment or scorer.
    return {s.casefold() for s in [row["word"], *row["synonyms"]]}


def prompt_for(query, support):
    if any(r["id"] == query["id"] or lexical_keys(r) & lexical_keys(query)
           or (r["synset"] and r["synset"] == query["synset"]) for r in support):
        raise ContractViolation("DEMONSTRATION_QUERY_LEAKAGE")
    return "\n".join([f"{r['description']} ⇒ {r['word']}" for r in support] + [f"{query['description']} ⇒"])


def preflight(rows, tokenizer, current_concepts, *, context_length=256, response_reserve=64):
    if context_length <= response_reserve or response_reserve <= 0:
        raise ContractViolation("INVALID_CONTEXT_RESERVATION")
    # These are proposed fixed examples, not a reconstruction of missing author seeds/config.
    ordered = sorted(rows, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    existing = {c["concept_id"]: c["answers"]["en"].casefold() for c in current_concepts}
    overlap = [{"external_id": r["id"], "current_concept_ids": [k for k, w in existing.items() if w in lexical_keys(r)]}
               for r in rows if any(w in lexical_keys(r) for w in existing.values())]
    contexts = {}
    for n in (0, 1, 24):
        lengths = []
        overflow = []
        hashes = []
        for query in rows:
            support = [r for r in ordered if r["id"] != query["id"] and not lexical_keys(r) & lexical_keys(query)
                       and not (r["synset"] and r["synset"] == query["synset"])][:n]
            if len(support) != n:
                raise ContractViolation("INSUFFICIENT_INDEPENDENT_DEMONSTRATIONS")
            prompt = prompt_for(query, support)
            length = len(tokenizer.encode(prompt))
            lengths.append(length)
            hashes.append(hashlib.sha256(prompt.encode()).hexdigest())
            if length + response_reserve > context_length:
                overflow.append(query["id"])
        contexts[str(n)] = {"prompt_count": len(rows), "min_tokens": min(lengths),
                           "max_tokens": max(lengths), "overflow_count": len(overflow),
                           "overflow_ids": overflow, "truncation_allowed": False,
                           "prompt_hashes_sha256": hashlib.sha256("\n".join(hashes).encode()).hexdigest()}
    return {"status": "INTAKE_AUDITED_NOT_EVALUATED", "n_records": len(rows), "language": "en",
            "contexts": contexts, "context_length": context_length, "response_reserve": response_reserve,
            "lexical_overlap_candidates": overlap, "overlap_is_semantic_alignment": False,
            "empty_synonym_ids": [r["id"] for r in rows if not r["synonyms"]],
            "training_enabled": False, "gpu_evaluation_enabled": False, "four_language_ready": False,
            "scientific_result": False, "accuracy": None,
            "interpretation": "author-associated mirror; original release equivalence and prompt seeds unverified; context checks are proposed adaptations, not paper replication",
            "limitations": ["Existing KO corpus exposure not exhaustively screened; no claim of uncontaminated unseen concepts",
                            "External English naming is not the four-language choice stability gate",
                            "Repository MIT notice does not by itself settle third-party definition redistribution rights"]}


def main():
    import argparse
    from .build_tokenizer import load_verified_tokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = WORK_ROOT / "sources/external_benchmark_intake_v1/manifest.json"
    manifest = read_verified_json(source, expected_sha256="d0c673b9c54474329315745b8084148ef89c62f9b68eb4d29a74117da55b1b31")
    for entry in manifest["sources"]:
        ref = entry["artifact"]
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ContractViolation("EXTERNAL_SOURCE_HASH_MISMATCH")
    ref = next(e["artifact"] for e in manifest["sources"] if e["url"].endswith("things_concepts.tsv"))
    freeze_path = WORK_ROOT / "freeze/accuracy_selected_360_v3/experiment_freeze.json"
    freeze = read_verified_json(freeze_path, expected_sha256="6f6cc0d52b4162f1018b560c6f95419cb9258deda7e053e0a12bdc7950e39dfd")
    tokenizer_ref = freeze["artifacts"]["tokenizer"]
    if sha256_file(Path(tokenizer_ref["path"])) != tokenizer_ref["sha256"]:
        raise ContractViolation("EXTERNAL_TOKENIZER_MANIFEST_MISMATCH")
    tokenizer, verified = load_verified_tokenizer(Path(tokenizer_ref["path"]))
    if verified["tokenizer_file_sha256"] != freeze["tokenizer_file_sha256"]:
        raise ContractViolation("EXTERNAL_TOKENIZER_FREEZE_MISMATCH")
    result = preflight(parse_things(read_regular_file_bytes(ref["path"])), tokenizer, freeze["concepts"])
    result.update({"source_manifest": {"path": str(source), "sha256": sha256_file(source)},
                   "dataset": ref, "tokenizer_sha256": freeze["tokenizer_file_sha256"]})
    print(publish_json_once(args.out, result))


if __name__ == "__main__":
    main()
