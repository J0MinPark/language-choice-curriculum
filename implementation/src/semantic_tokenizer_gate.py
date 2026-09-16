"""Audit the v4.1.1 semantic freeze with the existing KO-only tokenizer.

The original prompt audit is authoritative. A trailing-space removal probe is
diagnostic only: it never changes the frozen evaluation plan or authorizes a
later training stage.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import json

from freshstart.core import validate_token_events
from .artifacts import publish_json_once
from .build_tokenizer import audit_lexical_token_boundaries, load_verified_tokenizer, load_evaluation_plan
from .contracts import canonical_json_bytes, sha256_bytes, ContractViolation
from .semantic_annotation_freeze import audit_semantic_freeze, _bound, _read, _output


def _hash(value):
    return sha256_bytes(canonical_json_bytes(value))


def model_concepts(frozen, freeze_ref):
    """Adapt verified semantic records without inventing historical labels."""
    result = []
    for row in frozen["concepts"]:
        core = {
            "concept_id": row["candidate_id"],
            "answers": {l: t["answer"] for l, t in row["terms"].items()},
            "glosses": {l: t["gloss"] for l, t in row["terms"].items()},
            "source_review_subject_sha256": row["review_subject_sha256"],
            "annotation_freeze_sha256": freeze_ref["sha256"],
        }
        result.append({**core, "concept_record_sha256": _hash(core)})
    return result


def probe_trailing_space_removal(tokenizer, plan, concepts):
    failures = Counter()
    changed_templates = []
    max_length = 0
    checked = 0
    prompt_plan = plan["prompt_plan"]
    for wrapper, languages in sorted(prompt_plan["templates"].items()):
        for language, modes in languages.items():
            for mode, template in modes.items():
                proposed = template.rstrip(" ")
                if template != proposed:
                    changed_templates.append({"wrapper": wrapper, "language": language, "mode": mode, "original": template, "proposed": proposed})
                for concept in concepts:
                    for requested in ([None] if mode == "ANY" else ["ko", "en", "zh", "fr"]):
                        checked += 1
                        name = "" if requested is None else prompt_plan["target_language_names"][language][requested]
                        prefix = proposed.format(gloss=concept["glosses"][language], target_language_name=name)
                        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False).ids
                        reasons = set()
                        events = {}
                        for answer in sorted(set(concept["answers"].values())):
                            ids = tokenizer.encode(prefix + answer + "\n", add_special_tokens=False).ids
                            max_length = max(max_length, len(ids))
                            if ids[:len(prefix_ids)] != prefix_ids:
                                reasons.add("UNSTABLE_TOKEN_BOUNDARY")
                            if len(ids) > plan["model"]["context_length"]:
                                reasons.add("LEXICAL_SEQUENCE_EXCEEDS_CONTEXT")
                            events[answer] = ids[len(prefix_ids):]
                        try:
                            validate_token_events(events)
                        except ValueError as exc:
                            reasons.add(str(exc))
                        failures.update(reasons)
    return {
        "status": "PASS_DIAGNOSTIC_ONLY" if not failures else "BLOCKED_DIAGNOSTIC",
        "change": "REMOVE_ASCII_SPACES_AT_END_OF_TEMPLATE_ONLY",
        "changed_template_count": len(changed_templates), "changed_templates": changed_templates,
        "contexts_checked": checked, "failure_counts": dict(failures),
        "maximum_full_sequence_tokens": max_length,
        "context_length": plan["model"]["context_length"],
        "original_plan_modified": False, "training_eligible": False,
    }


def run_gate(annotation_freeze, tokenizer_manifest, output_dir, *, evaluation_plan_path=None):
    destination = _output(output_dir)
    if destination.exists():
        raise ContractViolation("SEMANTIC_TOKENIZER_OUTPUT_EXISTS")
    frozen_audit = audit_semantic_freeze(annotation_freeze)
    frozen = _bound(frozen_audit["artifact"])
    concepts = model_concepts(frozen, frozen_audit["artifact"])
    tokenizer, info = load_verified_tokenizer(tokenizer_manifest)
    _, tokenizer_ref = _read(tokenizer_manifest)
    kwargs = {} if evaluation_plan_path is None else {"evaluation_plan_path": Path(evaluation_plan_path)}
    audit = audit_lexical_token_boundaries(Path(tokenizer_manifest), concepts, **kwargs)
    concept_artifact = publish_json_once(destination / "model_concepts.json", {
        "schema_version": "semantic-model-concepts-v4.1.1",
        "annotation_freeze_artifact": frozen_audit["artifact"],
        "cohort_sha256": frozen["cohort_sha256"],
        "concepts": concepts, "training_eligible": False,
    })
    boundary_artifact = publish_json_once(destination / "token_boundary_audit.json", audit)
    plan_ref = info["evaluation_plan"]
    plan_path = Path(info["evaluation_plan_path"])
    if evaluation_plan_path is not None and Path(evaluation_plan_path).resolve() != plan_path:
        from .prompt_boundary_revision import resolve_boundary_plan

        plan_ref = resolve_boundary_plan(evaluation_plan_path, plan_ref)
        plan_path = Path(plan_ref["path"])
    plan = load_evaluation_plan(plan_path, expected_sha256=plan_ref["sha256"], expected_bytes=plan_ref["bytes"])
    probe = None
    if audit["status"] != "PASS":
        probe = publish_json_once(destination / "trailing_space_probe.json", probe_trailing_space_removal(tokenizer, plan, concepts))
    # No downstream training call exists here. The driver must inspect this
    # complete gate status, never just the successfully computed metrics.
    core = {
        "schema_version": "semantic-tokenizer-gate-v4.1.1",
        "status": audit["status"],
        "annotation_freeze_artifact": frozen_audit["artifact"],
        "tokenizer_manifest_artifact": tokenizer_ref,
        "tokenizer_file_sha256": info["tokenizer_file_sha256"],
        "evaluation_plan_sha256": audit["evaluation_plan_sha256"],
        "evaluation_plan_path": str(plan_path),
        "protocol_version": "4.1.2" if evaluation_plan_path is not None else "4.1.1",
        "model_concepts_artifact": concept_artifact,
        "boundary_audit_artifact": boundary_artifact,
        "diagnostic_probe_artifact": probe,
        "feature_concepts_completed": len(audit["expression_only_metrics"]),
        "feature_pairs_per_concept": 6,
        "contexts_checked": audit["contexts_checked"],
        "contexts_passed": audit["contexts_passed"],
        "contexts_failed": audit["contexts_failed"],
        "failure_counts": audit["failure_counts"],
        "automatic_advance_allowed": audit["status"] == "PASS",
        "training_started": False, "training_eligible": False,
    }
    result = {**core, "gate_sha256": _hash(core)}
    artifact = publish_json_once(destination / "gate.json", result)
    return {"status": result["status"], "artifact": artifact, "contexts_checked": result["contexts_checked"], "contexts_failed": result["contexts_failed"], "automatic_advance_allowed": result["automatic_advance_allowed"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-freeze", required=True)
    parser.add_argument("--tokenizer-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evaluation-plan")
    args = parser.parse_args()
    result = run_gate(Path(args.annotation_freeze), Path(args.tokenizer_manifest), Path(args.output_dir), evaluation_plan_path=args.evaluation_plan)
    print(json.dumps(result))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
