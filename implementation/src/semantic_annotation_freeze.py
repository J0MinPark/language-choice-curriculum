"""v4.1.1 semantic review packet and explicit approval-based annotation freeze.

Reparses the production KRDICT collection and binds all 240 selected expressions
to raw source spans. Prior intake decisions are context, never formal approval.
An explicit approval of the exact packet is required to publish a freeze.
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path

from .artifacts import publish_json_once, publish_bytes_once, read_regular_file_bytes_exact
from .contracts import WORK_ROOT, ContractViolation, canonical_json_bytes, sha256_bytes
from .prepare_data import merge_krdict_snapshot
from .registered_expression_correction import validate_registered_expression_correction

PACKET_SCHEMA = "semantic-review-packet-v4.1.1"
APPROVAL_SCHEMA = "semantic-review-approval-v4.1.1"
FREEZE_SCHEMA = "semantic-annotation-freeze-v4.1.1"


def _hash(value):
    return sha256_bytes(canonical_json_bytes(value))


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractViolation("SEMANTIC_FREEZE_DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _nonfinite(value):
    raise ContractViolation("SEMANTIC_FREEZE_NONFINITE_JSON")


def _read(path):
    path = Path(path).absolute()
    if path != path.resolve() or not path.is_relative_to(WORK_ROOT):
        raise ContractViolation("SEMANTIC_FREEZE_PATH_OUTSIDE_SCOPE")
    size = path.stat().st_size
    if size > 8 * 1024 * 1024:
        raise ContractViolation("SEMANTIC_FREEZE_ARTIFACT_TOO_LARGE")
    raw = read_regular_file_bytes_exact(path, expected_bytes=size)
    try:
        payload = json.loads(raw, object_pairs_hook=_object, parse_constant=_nonfinite)
    except (UnicodeError, ValueError) as exc:
        raise ContractViolation("SEMANTIC_FREEZE_INVALID_JSON") from exc
    if not isinstance(payload, dict):
        raise ContractViolation("SEMANTIC_FREEZE_INVALID_JSON")
    return payload, {"path": str(path), "sha256": sha256_bytes(raw), "bytes": size}


def _bound(ref):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256", "bytes"}:
        raise ContractViolation("SEMANTIC_FREEZE_INVALID_REF")
    payload, actual = _read(ref["path"])
    if canonical_json_bytes(actual) != canonical_json_bytes(ref):
        raise ContractViolation("SEMANTIC_FREEZE_REF_MISMATCH")
    return payload


def _output(path):
    path = Path(path).absolute()
    if path != path.resolve() or not path.is_relative_to(WORK_ROOT):
        raise ContractViolation("SEMANTIC_FREEZE_OUTPUT_OUTSIDE_SCOPE")
    return path


def build_review_packet(correction_path):
    correction, correction_ref = _read(correction_path)
    validate_registered_expression_correction(correction)
    plan = _bound(correction["source_artifacts"]["old_selection_plan"])
    source_ref = plan["inputs"]["source_manifest"]
    _bound(source_ref)
    merge = merge_krdict_snapshot(Path(source_ref["path"]), production=True)
    candidates = {row["candidate_id"]: row for row in merge["eligible_candidates"]}
    rows = []
    for selection in correction["selected"]:
        candidate = candidates[selection["candidate_id"]]
        if candidate["candidate_sha256"] != selection["candidate_sha256"]:
            raise ContractViolation("SEMANTIC_FREEZE_SOURCE_CANDIDATE_MISMATCH")
        terms = {}
        for language in ("ko", "en", "zh", "fr"):
            term = selection["terms"][language]
            options = [o for o in candidate["options"][language] if o["option_id"] == term["source_option_id"]]
            if len(options) != 1:
                raise ContractViolation("SEMANTIC_FREEZE_SOURCE_OPTION_MISMATCH")
            option = options[0]
            start, end = term["source_span_start"], term["source_span_end"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(option["word_raw"]):
                raise ContractViolation("SEMANTIC_FREEZE_SOURCE_SPAN_MISMATCH")
            exact = option["word_raw"][start:end]
            canonical = " ".join(unicodedata.normalize("NFC", exact).split())
            if canonical != term["canonical"] or option["gloss"] != term["source_gloss"]:
                raise ContractViolation("SEMANTIC_FREEZE_SOURCE_TEXT_MISMATCH")
            terms[language] = {
                "term_id": term["term_id"], "term_sha256": term["term_sha256"],
                "answer": term["canonical"], "gloss": option["gloss"],
                "source_option_id": option["option_id"],
                "source_word_raw": option["word_raw"],
                "source_span_start": start, "source_span_end": end,
                "source_text_exact": exact, "source_refs": option["source_refs"],
                "supplementary_evidence": correction["evidence_manifest_artifact"] if selection["selection_ordinal"] == 38 and language == "en" else None,
            }
        memberships = {}
        for language, term in terms.items():
            memberships.setdefault(term["answer"], []).append(language)
        row = {
            "selection_ordinal": selection["selection_ordinal"],
            "candidate_id": candidate["candidate_id"],
            "candidate_sha256": candidate["candidate_sha256"],
            "target_code": candidate["target_code"], "sense_order": candidate["sense_order"],
            "terms": terms, "memberships": memberships,
            "identifiable": len(memberships) == 4,
            "source_urls": candidate["entry_urls"],
        }
        rows.append({**row, "review_subject_sha256": _hash(row)})
    count = sum(row["identifiable"] for row in rows)
    if len(rows) != 60 or count < 40:
        raise ContractViolation("BLOCKED_DATA_COVERAGE")
    core = {
        "schema_version": PACKET_SCHEMA, "status": "PENDING_EXPLICIT_PACKET_APPROVAL",
        "correction_artifact": correction_ref, "source_manifest_artifact": source_ref,
        "source_merge_sha256": merge["merge_sha256"],
        "cohort_sha256": correction["cohort_sha256"],
        "rows": rows, "concept_count": 60, "expression_count": 240,
        "identifiable_count": count,
        "prior_jm02_counts": correction["source_review_counts"],
        "ordinal_38_correction": correction["correction_decision"],
        "etymology_required": False, "optional_etymology_status": "NOT_RUN",
        "identity_authentication_claimed": False, "training_eligible": False,
    }
    return {**core, "packet_sha256": _hash(core)}


def audit_review_packet(path):
    packet, ref = _read(path)
    try:
        _bound(packet["correction_artifact"])
        expected = build_review_packet(packet["correction_artifact"]["path"])
    except (KeyError, TypeError) as exc:
        raise ContractViolation("SEMANTIC_FREEZE_INVALID_PACKET") from exc
    if canonical_json_bytes(packet) != canonical_json_bytes(expected):
        raise ContractViolation("SEMANTIC_FREEZE_PACKET_MISMATCH")
    return packet, ref


def approval_template(packet, packet_ref):
    return {
        "schema_version": APPROVAL_SCHEMA, "packet_artifact": packet_ref,
        "status": "PENDING", "reviewer_id": None, "review_date": None,
        "meaning_alignment_checked": False, "term_quality_checked": False,
        "source_bindings_checked": False,
        "identity_authentication_claimed": False,
        "evidence_origin": "human-reviewed-dictionary-record",
        "approved_subjects": [r["review_subject_sha256"] for r in packet["rows"]],
    }


def validate_approval(packet, packet_ref, approval):
    expected = approval_template(packet, packet_ref)
    expected.update({
        "status": "APPROVED_BY_RESEARCHER", "reviewer_id": "jm02",
        "meaning_alignment_checked": True, "term_quality_checked": True,
        "source_bindings_checked": True,
    })
    if not isinstance(approval, dict) or approval.get("status") != "APPROVED_BY_RESEARCHER":
        raise ContractViolation("BLOCKED_EXPLICIT_SOURCE_BOUND_APPROVAL")
    try:
        reviewed = date.fromisoformat(approval["review_date"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractViolation("SEMANTIC_FREEZE_INVALID_REVIEW_DATE") from exc
    if reviewed.isoformat() != approval["review_date"] or reviewed < date(2026, 9, 15) or reviewed > datetime.now(timezone.utc).date():
        raise ContractViolation("SEMANTIC_FREEZE_INVALID_REVIEW_DATE")
    expected["review_date"] = approval["review_date"]
    if canonical_json_bytes(approval) != canonical_json_bytes(expected):
        raise ContractViolation("SEMANTIC_FREEZE_APPROVAL_SCOPE_MISMATCH")


def _derive_freeze(packet, packet_ref, approval, approval_ref):
    validate_approval(packet, packet_ref, approval)
    core = {
        "schema_version": FREEZE_SCHEMA, "protocol_version": "4.1.1",
        "status": "PASS_SEMANTIC_ANNOTATION_FREEZE",
        "packet_artifact": packet_ref, "approval_artifact": approval_ref,
        "concepts": packet["rows"], "cohort_sha256": packet["cohort_sha256"],
        "source_manifest_artifact": packet["source_manifest_artifact"],
        "source_merge_sha256": packet["source_merge_sha256"],
        "reviewer_id": approval["reviewer_id"], "review_date": approval["review_date"],
        "evidence_origin": approval["evidence_origin"],
        "identity_authentication_claimed": False,
        "annotation_gate_passed": True, "training_eligible": False,
        "main_enabled": False, "etymology_required": False,
        "optional_etymology_status": "NOT_RUN",
        "remaining_gates": ["TOKENIZER_AND_FEATURE_FREEZE", "EXPERIMENT_FREEZE", "CPU_AND_GPU_REPLAY", "PRODUCTION_PILOT_RUNNER"],
    }
    return {**core, "freeze_sha256": _hash(core)}


def audit_semantic_freeze(path):
    frozen, ref = _read(path)
    try:
        _bound(frozen["packet_artifact"])
        packet, packet_ref = audit_review_packet(frozen["packet_artifact"]["path"])
        approval = _bound(frozen["approval_artifact"])
        expected = _derive_freeze(packet, packet_ref, approval, frozen["approval_artifact"])
    except (KeyError, TypeError) as exc:
        raise ContractViolation("SEMANTIC_FREEZE_INVALID_ARTIFACT") from exc
    if canonical_json_bytes(frozen) != canonical_json_bytes(expected):
        raise ContractViolation("SEMANTIC_FREEZE_CONTENT_MISMATCH")
    return {"status": expected["status"], "artifact": ref, "training_eligible": False}


def freeze_semantic_annotations(packet_path, approval_path, output_path):
    packet, packet_ref = audit_review_packet(packet_path)
    approval, approval_ref = _read(approval_path)
    frozen = _derive_freeze(packet, packet_ref, approval, approval_ref)
    path = _output(output_path)
    publish_json_once(path, frozen)
    return audit_semantic_freeze(path)


def render_review_packet(packet):
    lines = [
        "# v4.1.1 의미·표현 검수 연결 확인", "",
        "상태: 정식 검수 기록 전환 확인 대기. 기존 jm02 판단과 egg 정정은 보존되어 있습니다.", "",
        "기존 판단을 아래 원자료 의미·표현과 연결해 정식 승인 기록으로 사용하는 범위입니다.",
        "어원 검수는 포함하지 않습니다. 38번 영어는 egg이며 원문 hen's egg의 6–9 위치와 OEWN 근거에 연결됩니다.", "",
        f"개념 {packet['concept_count']}개 · 표현 {packet['expression_count']}개 · 문자열 식별 가능 {packet['identifiable_count']}개", "",
        f"Packet SHA-256: `{packet['packet_sha256']}`", "",
        "| ID | KO | EN | ZH | FR | 한국어 뜻풀이 |", "| --- | --- | --- | --- | --- | --- |",
    ]
    def cell(value):
        return value.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    for row in packet["rows"]:
        terms = row["terms"]
        cells = [str(row["selection_ordinal"]), *(terms[l]["answer"] for l in ("ko", "en", "zh", "fr")), terms["ko"]["gloss"]]
        lines.append("| " + " | ".join(cell(value) for value in cells) + " |")
    return "\n".join(lines) + "\n"


def prepare_review_packet(correction_path, output_dir):
    directory = _output(output_dir)
    if directory.exists():
        raise ContractViolation("SEMANTIC_FREEZE_OUTPUT_EXISTS")
    packet = build_review_packet(correction_path)
    publish_json_once(directory / "review_packet.json", packet)
    packet, ref = audit_review_packet(directory / "review_packet.json")
    publish_json_once(directory / "approval_template.json", approval_template(packet, ref))
    publish_bytes_once(directory / "review_summary.md", render_review_packet(packet).encode("utf-8"))
    return {"status": packet["status"], "packet_artifact": ref, "concept_count": 60, "expression_count": 240, "training_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--correction", required=True)
    prepare.add_argument("--output-dir", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--packet", required=True)
    freeze.add_argument("--approval", required=True)
    freeze.add_argument("--out", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--freeze", required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare_review_packet(args.correction, args.output_dir)
        elif args.command == "freeze":
            result = freeze_semantic_annotations(args.packet, args.approval, args.out)
        else:
            result = audit_semantic_freeze(args.freeze)
    except ContractViolation as exc:
        print(json.dumps({"status": "BLOCKED", "reason": exc.code}))
        raise SystemExit(2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
