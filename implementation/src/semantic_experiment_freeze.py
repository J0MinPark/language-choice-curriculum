"""Verified v4.1.2 experiment inputs, initially restricted to GPU replay.

This distinct schema preserves the approved semantic freeze and exact prompt
revision. Its production entry requires a Git-pinned trust anchor and rebuilds
all inputs. It does not reinterpret an old annotation freeze or enable a study.
"""
from pathlib import Path

from .artifacts import publish_json_once, read_regular_file_bytes_exact
from .contracts import PROJECT_ROOT, WORK_ROOT, ContractViolation, canonical_json_bytes, sha256_bytes
from .semantic_annotation_freeze import audit_semantic_freeze, _read, _bound, _output
from .semantic_tokenizer_gate import model_concepts
from .prompt_boundary_revision import audit_prompt_revision, PLAN_PATH
from .build_tokenizer import audit_lexical_token_boundaries, verify_corpus_manifest
from .trust import require_trusted_artifact_anchor

SCHEMA = "semantic-experiment-freeze-v4.1.2"
INPUT_PATHS = {
    "annotation": WORK_ROOT / "freeze/semantic_v4_1_1_jm02/annotation_freeze.json",
    "tokenizer": WORK_ROOT / "tokenizer/wiki40b_ko_bpe_v4r1/tokenizer_manifest.json",
    "corpus": WORK_ROOT / "corpus/wiki40b_ko_tokens_v4r1/corpus_manifest.json",
    "boundary_audit": WORK_ROOT / "reports/semantic_tokenizer_v4_1_2_jm02/token_boundary_audit.json",
    "pilot_config": PROJECT_ROOT / "implementation/config/pilot_v4_1_1.json",
    "evaluation_plan": PLAN_PATH,
    "campaign_policy": PROJECT_ROOT / "implementation/config/campaign_policy.json",
}


def _hash(value):
    return sha256_bytes(canonical_json_bytes(value))


def _artifact(path):
    import json
    path = Path(path).absolute()
    if path != path.resolve() or not path.is_relative_to(PROJECT_ROOT):
        raise ContractViolation("SEMANTIC_EXPERIMENT_PATH_MISMATCH")
    size = path.stat().st_size
    if size > 8 * 1024 * 1024:
        raise ContractViolation("SEMANTIC_EXPERIMENT_INPUT_TOO_LARGE")
    raw = read_regular_file_bytes_exact(path, expected_bytes=size)
    return json.loads(raw), {"path": str(path), "sha256": sha256_bytes(raw), "bytes": size}


def build_semantic_experiment_freeze():
    revision = audit_prompt_revision()
    annotation_audit = audit_semantic_freeze(INPUT_PATHS["annotation"])
    if annotation_audit["artifact"]["sha256"] != "be1182242117337f7e0f516872732d80f58c119111b7674f8bced197fa195af0":
        raise ContractViolation("SEMANTIC_EXPERIMENT_APPROVED_ANNOTATION_MISMATCH")
    frozen = _bound(annotation_audit["artifact"])
    concepts = model_concepts(frozen, annotation_audit["artifact"])
    values, refs = {}, {}
    for name, path in INPUT_PATHS.items():
        values[name], refs[name] = _artifact(path)
    if refs["pilot_config"]["sha256"] != "e773b1842d38091d24fcf5a9359953bfce935b8e5938f4c0338acb141a8f3bf8":
        raise ContractViolation("SEMANTIC_EXPERIMENT_CONFIG_MISMATCH")
    if refs["corpus"]["sha256"] != "da7991020e166ec0d2635b40403448c15bd5448400225bbfaab3146d0c798b47":
        raise ContractViolation("SEMANTIC_EXPERIMENT_CORPUS_MISMATCH")
    boundary = audit_lexical_token_boundaries(INPUT_PATHS["tokenizer"], concepts, evaluation_plan_path=PLAN_PATH)
    if boundary["status"] != "PASS" or boundary != values["boundary_audit"]:
        raise ContractViolation("SEMANTIC_EXPERIMENT_BOUNDARY_MISMATCH")
    corpus = verify_corpus_manifest(INPUT_PATHS["corpus"], production=True)
    if corpus["tokenizer_file_sha256"] != boundary["tokenizer_file_sha256"] or corpus["materialized_token_count"] != 2000000:
        raise ContractViolation("SEMANTIC_EXPERIMENT_CORPUS_TOKENIZER_MISMATCH")
    if values["pilot_config"]["policy"]["main_enabled"] is not False or values["campaign_policy"]["main_enabled"] is not False:
        raise ContractViolation("MAIN_NOT_AUTHORIZED")
    core = {
        "schema_version": SCHEMA, "protocol_version": "4.1.2",
        "freeze_kind": "EXPERIMENT_FREEZE", "status": "PASS",
        "freeze_gate_status": "PASS", "data_kind": "REAL",
        "artifacts": refs, "prompt_revision_artifact": revision["amendment_artifact"],
        "annotation_freeze_id": frozen["freeze_sha256"],
        "cohort_sha256": frozen["cohort_sha256"],
        "concept_record_hashes": [row["concept_record_sha256"] for row in concepts],
        "tokenizer_file_sha256": boundary["tokenizer_file_sha256"],
        "corpus_token_count": corpus["materialized_token_count"],
        "boundary_audit_sha256": _hash(boundary),
        "allowed_execution_phases": ["GPU_REPLAY"],
        "scientific_training_authorized": False, "main_enabled": False,
        "etymology_required": False, "optional_etymology_status": "NOT_RUN",
        "pilot_config_role": "PRESERVED_PARENT_HYPERPARAMETERS_WITH_APPROVED_SEMANTIC_AND_PROMPT_AMENDMENTS",
    }
    manifest = {**core, "freeze_id": "semantic-experiment-" + _hash(core)[:20]}
    return manifest, concepts


def anchor_bindings(manifest):
    return {key: manifest[key] for key in (
        "schema_version", "protocol_version", "annotation_freeze_id", "cohort_sha256",
        "tokenizer_file_sha256", "corpus_token_count", "boundary_audit_sha256",
        "allowed_execution_phases", "scientific_training_authorized",
    )}


def publish_semantic_experiment_freeze(path):
    manifest, _ = build_semantic_experiment_freeze()
    artifact = publish_json_once(_output(path), manifest)
    return {"artifact": artifact, "anchor": {
        "artifact_kind": "EXPERIMENT_FREEZE", "artifact_id": manifest["freeze_id"],
        "manifest_sha256": artifact["sha256"], "bindings": anchor_bindings(manifest),
    }}


def verify_semantic_experiment_freeze(path, *, production=True):
    if production is not True:
        raise ContractViolation("SEMANTIC_EXPERIMENT_REQUIRES_PRODUCTION_SOURCES")
    manifest, ref = _read(path)
    try:
        require_trusted_artifact_anchor(artifact_kind="EXPERIMENT_FREEZE", artifact_id=manifest["freeze_id"], manifest_sha256=ref["sha256"], bindings=anchor_bindings(manifest))
    except (KeyError, TypeError) as exc:
        raise ContractViolation("SEMANTIC_EXPERIMENT_INVALID_SCHEMA") from exc
    expected, concepts = build_semantic_experiment_freeze()
    if canonical_json_bytes(manifest) != canonical_json_bytes(expected):
        raise ContractViolation("SEMANTIC_EXPERIMENT_CONTENT_MISMATCH")
    return {**manifest, "verified": True, "freeze_sha256": ref["sha256"],
            "manifest_path": ref["path"], "concepts": concepts,
            "resolved_artifacts": {k: r["path"] for k, r in manifest["artifacts"].items()}}
