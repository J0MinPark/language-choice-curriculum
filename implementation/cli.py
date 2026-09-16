#!/usr/bin/env python3
"""Fail-closed command line interface for the v4 implementation.

The module deliberately imports neither Torch nor any data/tokenizer library at
startup.  Each command loads only its own implementation entry points.  In
particular, ``gpu2-replay`` verifies the literal GPU-2 visibility mask before
the training module (and therefore Torch) can be imported.

Exit codes are stable: 0 is a completed command, 2 is invalid CLI syntax, 20
is a blocked contract/input/check, and 30 is an unavailable implementation
entry point or dependency.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


# Allow both ``python -m implementation.cli`` and, from the project checkout,
# ``python implementation/cli.py``.  This adds only the containing project.
if __package__ in {None, ""}:  # pragma: no cover - exercised by the real --help check
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from implementation.src.artifacts import (  # noqa: E402
    publish_bytes_once,
    publish_json_once,
    read_verified_json,
    sha256_file,
)
from implementation.src.contracts import (  # noqa: E402
    IMPLEMENTATION_REVISION,
    PROJECT_ROOT,
    VERSION,
    WORK_ROOT,
    ContractViolation,
    load_json,
    load_pilot_config,
    require_relative_to,
    require_sha256,
)


EXIT_OK = 0
EXIT_USAGE = 2
EXIT_BLOCKED = 20
EXIT_UNAVAILABLE = 30
CLI_SCHEMA = "implementation-cli-result-v1"

MANIFEST_PATH = PROJECT_ROOT / "MANIFEST.json"
CAMPAIGN_POLICY_PATH = PROJECT_ROOT / "implementation/config/campaign_policy.json"
EVALUATION_PLAN_PATH = PROJECT_ROOT / "implementation/config/evaluation_plan.json"
REVIEW_CSV_FILENAMES = (
    "terms_long.csv",
    "pair_selection_sheet.csv",
    "untranslated_screen.csv",
)


class CommandUnavailable(RuntimeError):
    """An optional command implementation cannot be loaded safely."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class SafeArgumentParser(argparse.ArgumentParser):
    """Do not echo rejected argv, which could contain a mistakenly pasted key."""

    def error(self, message: str) -> None:  # noqa: ARG002 - intentionally discarded
        self.print_usage(sys.stderr)
        payload = {
            "schema_version": CLI_SCHEMA,
            "status": "BLOCKED",
            "error_code": "CLI_USAGE_ERROR",
        }
        self.exit(
            EXIT_USAGE,
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n",
        )


def _emit(payload: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _load_callable(module_name: str, attribute: str, code: str) -> Callable[..., Any]:
    """Load one frozen entry point without importing unrelated command stacks."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise CommandUnavailable(code) from exc
    value = getattr(module, attribute, None)
    if not callable(value):
        raise CommandUnavailable(code)
    return value


def _argument_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _ensure_work_root() -> Path:
    if WORK_ROOT.is_symlink():
        raise ContractViolation("WORK_ROOT_NOT_DIRECTORY")
    try:
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ContractViolation("WORK_ROOT_UNAVAILABLE") from exc
    if not WORK_ROOT.is_dir():
        raise ContractViolation("WORK_ROOT_NOT_DIRECTORY")
    return WORK_ROOT


def _work_path(value: str | Path, code: str, *, output: bool = False) -> Path:
    root = _ensure_work_root() if output else WORK_ROOT
    if not root.exists() or root.is_symlink() or not root.is_dir():
        raise ContractViolation("WORK_ROOT_UNAVAILABLE")
    try:
        return require_relative_to(_argument_path(value), root, code)
    except OSError as exc:
        raise ContractViolation(code) from exc


def _project_input_path(value: str | Path, code: str) -> Path:
    try:
        candidate = require_relative_to(_argument_path(value), PROJECT_ROOT, code)
    except OSError as exc:
        raise ContractViolation(code) from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise ContractViolation(code)
    return candidate


def _read_json_value(path: Path, code: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ContractViolation(code)
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda _token: (_ for _ in ()).throw(
                ContractViolation("NONFINITE_JSON_NUMBER")
            ),
        )
    except ContractViolation:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractViolation(code) from exc


def _read_jsonl(path: Path, code: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ContractViolation(code)
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(
                    line,
                    parse_constant=lambda _token: (_ for _ in ()).throw(
                        ContractViolation("NONFINITE_JSON_NUMBER")
                    ),
                )
                if not isinstance(value, dict):
                    raise ContractViolation(code)
                rows.append(value)
    except ContractViolation:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractViolation(code) from exc
    return rows


def _published_response(
    command: str,
    domain: Mapping[str, Any],
    *,
    artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "schema_version": CLI_SCHEMA,
        "command": command,
        "status": str(domain.get("status", "PASS")),
    }
    for key in (
        "data_kind",
        "snapshot_id",
        "freeze_id",
        "selected_query_count",
        "planned_requests",
        "remaining_after_plan",
        "network_requests",
        "summary",
        "manifest_artifact",
        "scientific_result",
    ):
        if key in domain:
            response[key] = domain[key]
    if artifact is not None:
        response["artifact"] = dict(artifact)
    return response


def _verify_delivery() -> dict[str, Any]:
    """Verify only files enumerated by the immutable delivery manifest."""
    try:
        manifest = read_verified_json(MANIFEST_PATH)
    except ContractViolation:
        return {
            "status": "BLOCKED_SPEC_VERSION",
            "checked_files": 0,
            "failed_files": 1,
            "error_counts": {"INVALID_MANIFEST": 1},
        }
    members = manifest.get("sha256")
    if manifest.get("version") != VERSION or not isinstance(members, Mapping):
        return {
            "status": "BLOCKED_SPEC_VERSION",
            "checked_files": 0,
            "failed_files": 1,
            "error_counts": {"INVALID_MANIFEST": 1},
        }
    counts: dict[str, int] = {}
    root = PROJECT_ROOT.resolve(strict=True)
    for name, expected in members.items():
        error: str | None = None
        if not isinstance(name, str) or not name or not isinstance(expected, str):
            error = "INVALID_ENTRY"
        else:
            candidate = PROJECT_ROOT / name
            try:
                resolved = candidate.resolve(strict=False)
                resolved.relative_to(root)
            except (OSError, ValueError):
                error = "UNSAFE_PATH"
            if error is None and (candidate.is_symlink() or not candidate.is_file()):
                error = "MISSING_OR_UNSAFE"
            if error is None:
                try:
                    require_sha256(expected, "INVALID_MANIFEST_HASH")
                    if sha256_file(candidate) != expected:
                        error = "HASH_MISMATCH"
                except ContractViolation:
                    error = "INVALID_ENTRY"
        if error is not None:
            counts[error] = counts.get(error, 0) + 1
    failed = sum(counts.values())
    return {
        "status": "PASS" if failed == 0 else "BLOCKED_SPEC_VERSION",
        "checked_files": len(members),
        "failed_files": failed,
        "error_counts": counts,
        "manifest_sha256": sha256_file(MANIFEST_PATH),
    }


def _declares_functions(relative_path: str, names: Sequence[str]) -> bool:
    """Inspect syntax only; status must not import Torch or optional packages."""
    path = PROJECT_ROOT / relative_path
    if path.is_symlink() or not path.is_file():
        return False
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return False
    declared = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return set(names).issubset(declared)


def _status(_args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    pilot = load_pilot_config()
    campaign = load_json(CAMPAIGN_POLICY_PATH)
    evaluation = load_json(EVALUATION_PLAN_PATH)
    if (
        campaign.get("version") != IMPLEMENTATION_REVISION
        or evaluation.get("version") != IMPLEMENTATION_REVISION
        or campaign.get("main_enabled") is not False
        or campaign.get("frozen_before_model_results") is not True
        or evaluation.get("frozen_before_model_results") is not True
    ):
        raise ContractViolation("BLOCKED_IMPLEMENTATION_POLICY")
    delivery = _verify_delivery()
    budget = campaign.get("budget", {})
    gpu = campaign.get("gpu", {})
    key = os.environ.get("KRDICT_API_KEY", "")
    capabilities = {
        "offline_data": _declares_functions(
            "implementation/src/prepare_data.py",
            (
                "build_krdict_collection_plan",
                "import_existing_krdict_collection",
                "merge_krdict_snapshot",
                "export_pending_review",
                "stage_wiki40b_documents",
                "publish_experiment_freeze",
            ),
        ),
        "review_csv_workflow": _declares_functions(
            "implementation/src/review_import.py",
            (
                "render_review_csv_bundle",
                "load_and_validate_review_csv_bundle",
                "validate_review_tables",
                "compile_filled_review_bundle",
            ),
        ),
        "review_csv_freeze": _declares_functions(
            "implementation/src/review_freeze.py",
            ("freeze_completed_review_csv_bundle",),
        ),
        "tokenizer": _declares_functions(
            "implementation/src/build_tokenizer.py",
            (
                "train_byte_bpe",
                "materialize_corpus_tokens",
                "audit_lexical_token_boundaries",
            ),
        ),
        "gpu2_replay": _declares_functions(
            "implementation/src/train.py", ("run_gpu2_replay",)
        ),
        "single_run_reporting": _declares_functions(
            "implementation/src/report.py", ("publish_run_summary", "render_markdown")
        ),
        "implementation_integrity": _declares_functions(
            "implementation/src/integrity.py",
            (
                "publish_implementation_manifest",
                "run_cpu_check_evidence",
                "verify_cpu_check_evidence",
            ),
        ),
        "pilot_orchestration_preflight": _declares_functions(
            "implementation/src/pilot.py",
            ("derive_pilot_decision", "run_production_pilot"),
        ),
        "pilot_scientific_runner": False,
    }
    ready = delivery["status"] == "PASS"
    response = {
        "schema_version": "implementation-status-v1",
        "command": "status",
        "status": "READY_FOR_OFFLINE_WORKFLOW" if ready else "BLOCKED_DELIVERY_INTEGRITY",
        "version": VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "delivery_integrity": delivery,
        "policy": {
            "pilot_authorized_after_gates": bool(
                pilot["policy"]["pilot_authorized_after_gates"]
            ),
            "main_enabled": False,
            "live_collection_exposed": False,
        },
        "campaign": {
            "api_requests_cap": budget.get("api_requests_cap"),
            "api_requests_already_attempted": budget.get(
                "api_requests_already_attempted"
            ),
            "api_requests_remaining_before_collection": budget.get(
                "api_requests_remaining_before_collection"
            ),
            "required_physical_gpu_index": gpu.get("required_physical_index"),
        },
        "local_environment": {
            "krdict_key_present": bool(key),
            "krdict_key_format_valid": bool(re.fullmatch(r"[0-9a-fA-F]{32}", key)),
            "krdict_key_validated_by_provider": False,
            "gpu2_mask_ready": os.environ.get("CUDA_VISIBLE_DEVICES") == "2",
            "gpu_device_order_ready": os.environ.get("CUDA_DEVICE_ORDER")
            == "PCI_BUS_ID",
        },
        "capabilities": capabilities,
        "side_effects": {"network_requests": 0, "gpu_jobs_started": 0},
        "hashes": {
            "pilot_config_sha256": sha256_file(PROJECT_ROOT / "spec/pilot.json"),
            "campaign_policy_sha256": sha256_file(CAMPAIGN_POLICY_PATH),
            "evaluation_plan_sha256": sha256_file(EVALUATION_PLAN_PATH),
        },
        "next_action": "Continue from the next unmet offline gate; do not repeat the exhausted collection campaign.",
    }
    return (EXIT_OK if ready else EXIT_BLOCKED), response


def _verify_package(_args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    result = _verify_delivery()
    return (
        EXIT_OK if result["status"] == "PASS" else EXIT_BLOCKED,
        {"schema_version": CLI_SCHEMA, "command": "verify-package", **result},
    )


def _audit_design_revision(_args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    audit = _load_callable(
        "implementation.src.design_revision",
        "audit_v4_1_design_amendment",
        "NOT_IMPLEMENTED_DESIGN_REVISION_AUDIT",
    )
    result = audit()
    return EXIT_OK, {
        "schema_version": CLI_SCHEMA,
        "command": "audit-design-revision",
        "status": result["status"],
        "audit": result,
    }


def _collection_plan(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    build = _load_callable(
        "implementation.src.prepare_data",
        "build_krdict_collection_plan",
        "NOT_IMPLEMENTED_COLLECTION_PLAN",
    )
    queries = _project_input_path(args.queries, "QUERY_FILE_OUTSIDE_PROJECT")
    output = _work_path(args.out, "PLAN_OUTPUT_OUTSIDE_WORK", output=True)
    plan = build(queries)
    artifact = publish_json_once(output, plan)
    return EXIT_OK, _published_response("collection-plan", plan, artifact=artifact)


def _offline_import(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    adapter = _load_callable(
        "implementation.src.prepare_data",
        "import_existing_krdict_collection",
        "NOT_IMPLEMENTED_OFFLINE_IMPORT",
    )
    legacy = _work_path(args.legacy_manifest, "LEGACY_SOURCE_OUTSIDE_SCOPE")
    plan_path = _work_path(args.plan, "COLLECTION_PLAN_OUTSIDE_SCOPE")
    output_dir = _work_path(args.output_dir, "SOURCE_OUTPUT_OUTSIDE_SCOPE", output=True)
    plan = read_verified_json(plan_path)
    result = adapter(
        legacy,
        plan,
        output_dir=output_dir,
        production=not args.synthetic_test_fixture,
        scope_root=WORK_ROOT,
    )
    if result.get("network_requests") != 0 or result.get("requests_made_this_call") != 0:
        raise ContractViolation("OFFLINE_IMPORT_NETWORK_ACCOUNTING_INVALID")
    return EXIT_OK, _published_response("offline-import", result)


def _merge_source(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    merge = _load_callable(
        "implementation.src.prepare_data",
        "merge_krdict_snapshot",
        "NOT_IMPLEMENTED_SOURCE_MERGE",
    )
    manifest = _work_path(args.manifest, "SOURCE_MANIFEST_OUTSIDE_SCOPE")
    output = _work_path(args.out, "MERGE_OUTPUT_OUTSIDE_SCOPE", output=True)
    result = merge(manifest, production=not args.synthetic_test_fixture)
    artifact = publish_json_once(output, result)
    return EXIT_OK, _published_response("merge-source", result, artifact=artifact)


def _export_review(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    export = _load_callable(
        "implementation.src.prepare_data",
        "export_pending_review",
        "NOT_IMPLEMENTED_REVIEW_EXPORT",
    )
    merge_path = _work_path(args.merge, "MERGE_INPUT_OUTSIDE_SCOPE")
    output = _work_path(args.out, "REVIEW_OUTPUT_OUTSIDE_SCOPE", output=True)
    result = export(read_verified_json(merge_path))
    if result.get("status") != "PENDING_HUMAN_REVIEW":
        raise ContractViolation("INVALID_REVIEW_EXPORT_STATUS")
    artifact = publish_json_once(output, result)
    return EXIT_OK, _published_response("export-review", result, artifact=artifact)


def _prepare_review_csvs(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    """Publish reviewer-editable CSV projections; never a training artifact."""
    merge_source = _load_callable(
        "implementation.src.prepare_data",
        "merge_krdict_snapshot",
        "NOT_IMPLEMENTED_SOURCE_MERGE",
    )
    render = _load_callable(
        "implementation.src.review_import",
        "render_review_csv_bundle",
        "NOT_IMPLEMENTED_REVIEW_CSV_RENDER",
    )
    source = _work_path(args.manifest, "SOURCE_MANIFEST_OUTSIDE_SCOPE")
    output_dir = _work_path(
        args.output_dir, "REVIEW_CSV_OUTPUT_OUTSIDE_SCOPE", output=True
    )
    if output_dir.is_symlink() or output_dir.exists():
        raise ContractViolation("REVIEW_CSV_OUTPUT_EXISTS")
    production = not args.synthetic_test_fixture
    merge = merge_source(source, production=production)
    rendered = render(merge)
    if (
        not isinstance(rendered, Mapping)
        or rendered.get("status") != "PENDING_HUMAN_REVIEW"
        or rendered.get("training_eligible") is not False
    ):
        raise ContractViolation("INVALID_REVIEW_CSV_RENDER")
    files = rendered.get("files")
    if not isinstance(files, Mapping) or set(files) != set(REVIEW_CSV_FILENAMES):
        raise ContractViolation("INVALID_REVIEW_CSV_FILE_SET")
    encoded: dict[str, bytes] = {}
    for name in REVIEW_CSV_FILENAMES:
        value = files[name]
        if isinstance(value, str):
            value = value.encode("utf-8")
        if not isinstance(value, bytes) or not value:
            raise ContractViolation("INVALID_REVIEW_CSV_CONTENT")
        encoded[name] = value
    summary = rendered.get("summary", {})
    if not isinstance(summary, Mapping):
        raise ContractViolation("INVALID_REVIEW_CSV_SUMMARY")

    # CSVs remain private and reviewer-editable.  Their initial hashes are a
    # template provenance record, not a human sign-off and not a data freeze.
    csv_refs = {
        name: publish_bytes_once(
            output_dir / name,
            encoded[name],
            mode=0o444 if name == "untranslated_screen.csv" else 0o600,
        )
        for name in REVIEW_CSV_FILENAMES
    }
    manifest = {
        "schema_version": "review-csv-template-v1",
        "status": "PENDING_HUMAN_REVIEW",
        "data_kind": merge.get("data_kind"),
        "training_eligible": False,
        "direct_trainer_input_allowed": False,
        "source_collection": {
            "path": str(source),
            "sha256": sha256_file(source),
            "bytes": source.stat().st_size,
        },
        "merge_sha256": merge.get("merge_sha256"),
        "csv_templates": csv_refs,
        "summary": dict(summary),
        "notice": (
            "The terms and pair CSVs are reviewer-editable; untranslated_screen.csv "
            "is a read-only derived view. All require a separate validated annotation "
            "freeze, and trainers must never read them directly."
        ),
    }
    manifest_ref = publish_json_once(output_dir / "review_csv_manifest.json", manifest)
    return EXIT_OK, {
        "schema_version": CLI_SCHEMA,
        "command": "prepare-review-csvs",
        "status": manifest["status"],
        "data_kind": manifest["data_kind"],
        "training_eligible": False,
        "direct_trainer_input_allowed": False,
        "summary": manifest["summary"],
        "manifest_artifact": manifest_ref,
    }


def _audit_review_csvs(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    """Audit CSV review state without compiling it or touching a trainer."""
    merge_source = _load_callable(
        "implementation.src.prepare_data",
        "merge_krdict_snapshot",
        "NOT_IMPLEMENTED_SOURCE_MERGE",
    )
    load_bundle = _load_callable(
        "implementation.src.review_import",
        "load_and_validate_review_csv_bundle",
        "NOT_IMPLEMENTED_REVIEW_CSV_IMPORT",
    )
    validate = _load_callable(
        "implementation.src.review_import",
        "validate_review_tables",
        "NOT_IMPLEMENTED_REVIEW_CSV_VALIDATION",
    )
    source = _work_path(args.manifest, "SOURCE_MANIFEST_OUTSIDE_SCOPE")
    terms = _work_path(args.terms, "REVIEW_TERMS_OUTSIDE_SCOPE")
    pairs = _work_path(args.pairs, "REVIEW_PAIRS_OUTSIDE_SCOPE")
    screen = _work_path(args.screen, "REVIEW_SCREEN_OUTSIDE_SCOPE")
    production = not args.synthetic_test_fixture
    merge = merge_source(source, production=production)
    tables = load_bundle(merge, terms, pairs, screen)
    result = validate(merge, tables, require_complete=args.require_complete)
    if not isinstance(result, Mapping) or not isinstance(result.get("summary", {}), Mapping):
        raise ContractViolation("INVALID_REVIEW_CSV_AUDIT")
    status = str(result.get("status", "BLOCKED_DATA_QA"))
    if status not in {"BLOCKED_DATA_QA", "STRUCTURE_PASS_PENDING_EVIDENCE"}:
        raise ContractViolation("INVALID_REVIEW_CSV_AUDIT_STATUS")
    if result.get("training_eligible") is not False:
        raise ContractViolation("INVALID_REVIEW_CSV_AUDIT_STATUS")
    response = {
        "schema_version": CLI_SCHEMA,
        "command": "audit-review-csvs",
        "status": status,
        "data_kind": merge.get("data_kind"),
        "require_complete": bool(args.require_complete),
        "training_eligible": False,
        "direct_trainer_input_allowed": False,
        "structure_complete": status == "STRUCTURE_PASS_PENDING_EVIDENCE",
        "review_compile_ready": False,
        "summary": dict(result.get("summary", {})),
        "notice": "Only a subsequently compiled and verified experiment freeze may reach a trainer.",
    }
    return (
        EXIT_OK
        if not args.require_complete or status == "STRUCTURE_PASS_PENDING_EVIDENCE"
        else EXIT_BLOCKED,
        response,
    )


def _freeze_review_csvs(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    """Compile completed CSV review into the sole annotation-freeze boundary."""
    freeze = _load_callable(
        "implementation.src.review_freeze",
        "freeze_completed_review_csv_bundle",
        "NOT_IMPLEMENTED_REVIEW_CSV_FREEZE",
    )
    source = _work_path(args.manifest, "SOURCE_MANIFEST_OUTSIDE_SCOPE")
    terms = _work_path(args.terms, "REVIEW_TERMS_OUTSIDE_SCOPE")
    pairs = _work_path(args.pairs, "REVIEW_PAIRS_OUTSIDE_SCOPE")
    screen = _work_path(args.screen, "REVIEW_SCREEN_OUTSIDE_SCOPE")
    evidence_path = _work_path(
        args.evidence_records, "REVIEW_EVIDENCE_OUTSIDE_SCOPE"
    )
    output_dir = _work_path(
        args.output_dir, "ANNOTATION_FREEZE_OUTSIDE_SCOPE", output=True
    )
    evidence_records = _read_jsonl(
        evidence_path, "INVALID_REVIEW_EVIDENCE_JSONL"
    )
    result = freeze(
        source,
        terms,
        pairs,
        screen,
        evidence_records=evidence_records,
        output_dir=output_dir,
        production=not args.synthetic_test_fixture,
        scope_root=WORK_ROOT,
    )
    if not isinstance(result, Mapping) or result.get("status") != "PASS":
        raise ContractViolation("ANNOTATION_FREEZE_NOT_PASS")
    response = _published_response("freeze-review-csvs", result)
    response.update(
        {
            "direct_trainer_input_allowed": False,
            "trainer_input": "VERIFIED_EXPERIMENT_FREEZE_ONLY",
            "scientific_training_started": False,
        }
    )
    return EXIT_OK, response


def _stage_corpus(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    load_export = _load_callable(
        "implementation.src.prepare_data",
        "iter_wiki40b_jsonl_export",
        "NOT_IMPLEMENTED_WIKI40B_OFFLINE_LOADER",
    )
    stage = _load_callable(
        "implementation.src.prepare_data",
        "stage_wiki40b_documents",
        "NOT_IMPLEMENTED_CORPUS_STAGING",
    )
    export_path = _work_path(args.export, "WIKI40B_EXPORT_OUTSIDE_SCOPE")
    metadata_path = _work_path(args.metadata, "WIKI40B_METADATA_OUTSIDE_SCOPE")
    output_dir = _work_path(args.output_dir, "CORPUS_OUTPUT_OUTSIDE_SCOPE", output=True)
    expected = require_sha256(args.expected_sha256, "WIKI40B_EXPORT_SHA256_INVALID")
    metadata = read_verified_json(metadata_path)
    examples = load_export(export_path, expected_sha256=expected)
    result = stage(
        examples,
        metadata,
        output_dir=output_dir,
        production=not args.synthetic_test_fixture,
        scope_root=WORK_ROOT,
    )
    return EXIT_OK, _published_response("stage-corpus", result)


def _build_tokenizer(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    train_byte_bpe = _load_callable(
        "implementation.src.build_tokenizer",
        "train_byte_bpe",
        "NOT_IMPLEMENTED_BUILD_TOKENIZER",
    )
    source = _work_path(args.source_manifest, "CORPUS_MANIFEST_OUTSIDE_SCOPE")
    output_dir = _work_path(args.output_dir, "TOKENIZER_OUTPUT_OUTSIDE_SCOPE", output=True)
    result = train_byte_bpe(
        source,
        output_dir=output_dir,
        production=not args.synthetic_test_fixture,
        evaluation_plan_path=EVALUATION_PLAN_PATH,
        scope_root=WORK_ROOT,
    )
    return EXIT_OK, _published_response("build-tokenizer", result)


def _materialize_corpus(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    materialize = _load_callable(
        "implementation.src.build_tokenizer",
        "materialize_corpus_tokens",
        "NOT_IMPLEMENTED_MATERIALIZE_CORPUS",
    )
    source = _work_path(args.source_manifest, "CORPUS_MANIFEST_OUTSIDE_SCOPE")
    tokenizer = _work_path(args.tokenizer_manifest, "TOKENIZER_MANIFEST_OUTSIDE_SCOPE")
    output_dir = _work_path(args.output_dir, "TOKEN_CORPUS_OUTPUT_OUTSIDE_SCOPE", output=True)
    result = materialize(
        source,
        tokenizer,
        output_dir=output_dir,
        production=not args.synthetic_test_fixture,
        scope_root=WORK_ROOT,
    )
    return EXIT_OK, _published_response("materialize-corpus", result)


def _audit_token_boundaries(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    audit = _load_callable(
        "implementation.src.build_tokenizer",
        "audit_lexical_token_boundaries",
        "NOT_IMPLEMENTED_TOKEN_BOUNDARY_AUDIT",
    )
    tokenizer = _work_path(args.tokenizer_manifest, "TOKENIZER_MANIFEST_OUTSIDE_SCOPE")
    concepts_path = _work_path(args.concepts, "CONCEPTS_OUTSIDE_SCOPE")
    output = _work_path(args.out, "TOKEN_AUDIT_OUTPUT_OUTSIDE_SCOPE", output=True)
    concepts = _read_jsonl(concepts_path, "INVALID_CONCEPT_JSONL")
    result = audit(
        tokenizer,
        concepts,
        output_path=output,
        production=not args.synthetic_test_fixture,
        evaluation_plan_path=EVALUATION_PLAN_PATH,
        scope_root=WORK_ROOT,
    )
    return EXIT_OK, _published_response("audit-token-boundaries", result)


def _freeze_experiment(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    freeze = _load_callable(
        "implementation.src.prepare_data",
        "publish_experiment_freeze",
        "NOT_IMPLEMENTED_EXPERIMENT_FREEZE",
    )
    annotation = _work_path(args.annotation_manifest, "ANNOTATION_MANIFEST_OUTSIDE_SCOPE")
    tokenizer = _work_path(args.tokenizer_manifest, "TOKENIZER_MANIFEST_OUTSIDE_SCOPE")
    corpus = _work_path(args.corpus_manifest, "CORPUS_MANIFEST_OUTSIDE_SCOPE")
    audit_path = _work_path(args.boundary_audit, "TOKEN_AUDIT_OUTSIDE_SCOPE")
    output = _work_path(args.out, "EXPERIMENT_FREEZE_OUTSIDE_SCOPE", output=True)
    result = freeze(
        annotation,
        tokenizer,
        corpus,
        boundary_audit=read_verified_json(audit_path),
        output_path=output,
        production=not args.synthetic_test_fixture,
        scope_root=WORK_ROOT,
    )
    return EXIT_OK, _published_response("freeze-experiment", result)


def _implementation_integrity(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    publish = _load_callable(
        "implementation.src.integrity",
        "publish_implementation_manifest",
        "NOT_IMPLEMENTED_INTEGRITY_MANIFEST",
    )
    output = _work_path(args.out, "INTEGRITY_OUTPUT_OUTSIDE_SCOPE", output=True)
    result = publish(
        output,
        project_root=PROJECT_ROOT,
        scope_root=WORK_ROOT,
    )
    return EXIT_OK, _published_response(
        "implementation-integrity", result, artifact=result["manifest_artifact"]
    )


def _cpu_checks(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    run_checks = _load_callable(
        "implementation.src.integrity",
        "run_cpu_check_evidence",
        "NOT_IMPLEMENTED_CPU_CHECK_EVIDENCE",
    )
    implementation_manifest = _work_path(
        args.implementation_manifest, "INTEGRITY_MANIFEST_OUTSIDE_SCOPE"
    )
    output_dir = _work_path(
        args.output_dir, "CPU_CHECK_OUTPUT_OUTSIDE_SCOPE", output=True
    )
    result = run_checks(
        implementation_manifest,
        output_dir=output_dir,
        project_root=PROJECT_ROOT,
        scope_root=WORK_ROOT,
    )
    passed = result.get("status") == "PASS"
    return (
        EXIT_OK if passed else EXIT_BLOCKED,
        _published_response(
            "cpu-checks", result, artifact=result.get("manifest_artifact")
        ),
    )


def _gpu2_replay(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    # Resolve every user path before importing any Torch-bearing module.
    freeze = _work_path(args.experiment_freeze, "EXPERIMENT_FREEZE_OUTSIDE_SCOPE")
    implementation_manifest = _work_path(
        args.implementation_manifest, "INTEGRITY_MANIFEST_OUTSIDE_SCOPE"
    )
    cpu_checks = _work_path(args.cpu_checks, "CPU_CHECK_OUTSIDE_SCOPE")
    checkpoints = _work_path(
        args.checkpoint_directory, "CHECKPOINT_OUTPUT_OUTSIDE_SCOPE", output=True
    )
    ledger_path = _work_path(args.ledger, "GPU_LEDGER_OUTSIDE_SCOPE", output=True)
    lock_path = _work_path(args.lock, "GPU_LOCK_OUTSIDE_SCOPE", output=True)
    output = _work_path(args.out, "GPU_REPLAY_OUTPUT_OUTSIDE_SCOPE", output=True)
    if freeze.is_symlink() or not freeze.is_file():
        raise ContractViolation("EXPERIMENT_FREEZE_NOT_REGULAR_FILE")
    if checkpoints.is_symlink() or checkpoints.exists():
        raise ContractViolation("CHECKPOINT_EXISTS")
    if ledger_path.is_symlink() or (ledger_path.exists() and not ledger_path.is_file()):
        raise ContractViolation("INVALID_GPU_LEDGER")
    if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
        raise ContractViolation("INVALID_GPU_LOCK")
    if output.exists() or output.is_symlink():
        raise ContractViolation("ARTIFACT_EXISTS")
    if not args.run_id or any(character.isspace() for character in args.run_id):
        raise ContractViolation("INVALID_BUDGET_LEASE")
    if args.checkpoint_grace_seconds >= args.reserved_seconds:
        raise ContractViolation("INVALID_GPU_REPLAY_TIMING")
    # This import is dependency-light.  The literal mask check must succeed
    # before ``implementation.src.train`` is even imported.
    require_mask = _load_callable(
        "implementation.src.gpu_guard",
        "require_literal_gpu2_mask",
        "NOT_IMPLEMENTED_GPU2_GUARD",
    )
    require_mask()

    # Code identity is never caller asserted.  Re-verify the exact clean HEAD
    # manifest and its CPU evidence, then derive the lineage hashes from them.
    verify_implementation = _load_callable(
        "implementation.src.integrity",
        "verify_implementation_manifest",
        "NOT_IMPLEMENTED_INTEGRITY_MANIFEST",
    )
    verify_cpu = _load_callable(
        "implementation.src.integrity",
        "verify_cpu_check_evidence",
        "NOT_IMPLEMENTED_CPU_CHECK_EVIDENCE",
    )
    implementation = verify_implementation(
        implementation_manifest,
        project_root=PROJECT_ROOT,
        scope_root=WORK_ROOT,
    )
    cpu_evidence = verify_cpu(
        cpu_checks,
        project_root=PROJECT_ROOT,
        scope_root=WORK_ROOT,
    )
    code_sha256 = require_sha256(
        implementation.get("implementation_manifest_sha256"),
        "INVALID_CODE_SHA256",
    )
    cpu_implementation = cpu_evidence.get("implementation")
    if (
        not isinstance(cpu_implementation, Mapping)
        or cpu_implementation.get("implementation_manifest_sha256") != code_sha256
    ):
        raise ContractViolation("GPU_REPLAY_CPU_CODE_BINDING_MISMATCH")
    cpu_check_sha256 = require_sha256(
        cpu_evidence.get("cpu_check_sha256"), "INVALID_CPU_CHECK_HASH"
    )

    budget_caps = _load_callable(
        "implementation.src.budget", "BudgetCaps", "NOT_IMPLEMENTED_GPU_BUDGET"
    )
    ledger_type = _load_callable(
        "implementation.src.budget", "GpuBudgetLedger", "NOT_IMPLEMENTED_GPU_BUDGET"
    )
    replay = _load_callable(
        "implementation.src.train", "run_gpu2_replay", "NOT_IMPLEMENTED_GPU2_REPLAY"
    )
    campaign = load_json(CAMPAIGN_POLICY_PATH)
    budget = campaign["budget"]
    caps = budget_caps(
        campaign_gpu_hours_cap=float(budget["campaign_gpu_hours_cap"]),
        prior_gpu_hours=float(budget["prior_gpu_hours_user_reported"]),
        per_root_gpu_hours_cap=float(budget["per_root_gpu_hours_cap"]),
    )
    ledger = ledger_type(ledger_path, caps)
    result = replay(
        experiment_freeze_path=freeze,
        checkpoint_directory=checkpoints,
        ledger=ledger,
        lock_path=lock_path,
        run_id=args.run_id,
        code_sha256=code_sha256,
        cpu_check_sha256=cpu_check_sha256,
        reserved_seconds=args.reserved_seconds,
        checkpoint_grace_seconds=args.checkpoint_grace_seconds,
    )
    replay_binding = result.get("gpu_binding")
    if (
        result.get("status") != "PASS"
        or result.get("scientific_result") is not False
        or result.get("code_sha256") != code_sha256
        or result.get("cpu_check_sha256") != cpu_check_sha256
        or not isinstance(replay_binding, Mapping)
        or replay_binding.get("physical_index") != 2
        or replay_binding.get("logical_index") != 0
        or replay_binding.get("uuid") != campaign["gpu"]["expected_uuid"]
    ):
        raise ContractViolation("BLOCKED_GPU_REPLAY_RESULT")
    require_sha256(result.get("freeze_sha256"), "INVALID_GPU_REPLAY_FREEZE_HASH")
    # Close the ordinary change-during-run window before publishing evidence.
    post_implementation = verify_implementation(
        implementation_manifest,
        project_root=PROJECT_ROOT,
        scope_root=WORK_ROOT,
    )
    post_cpu = verify_cpu(
        cpu_checks,
        project_root=PROJECT_ROOT,
        scope_root=WORK_ROOT,
    )
    if (
        post_implementation.get("implementation_manifest_sha256") != code_sha256
        or post_cpu.get("cpu_check_sha256") != cpu_check_sha256
    ):
        raise ContractViolation("GPU_REPLAY_EVIDENCE_CHANGED_DURING_RUN")
    artifact = publish_json_once(output, result)
    return EXIT_OK, _published_response("gpu2-replay", result, artifact=artifact)


def _finalize_run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    publish = _load_callable(
        "implementation.src.report",
        "publish_run_summary",
        "NOT_IMPLEMENTED_RUN_FINALIZATION",
    )
    payload_path = _work_path(args.payload, "RUN_PAYLOAD_OUTSIDE_SCOPE")
    output = _work_path(args.out, "RUN_SUMMARY_OUTSIDE_SCOPE", output=True)
    payload = read_verified_json(payload_path)
    artifact = publish(output, payload)
    return EXIT_OK, {
        "schema_version": CLI_SCHEMA,
        "command": "finalize-run",
        "status": "PUBLISHED",
        "run_status": payload.get("status"),
        "artifact": artifact,
    }


def _render(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    render = _load_callable(
        "implementation.src.report",
        "render_markdown",
        "NOT_IMPLEMENTED_REPORT_RENDER",
    )
    source = _work_path(args.json, "RUN_SUMMARY_OUTSIDE_SCOPE")
    output = _work_path(args.out, "REPORT_OUTPUT_OUTSIDE_SCOPE", output=True)
    artifact = render(source, output)
    return EXIT_OK, {
        "schema_version": CLI_SCHEMA,
        "command": "render",
        "status": "RENDERED_FROM_RUN_JSON",
        "source_json_sha256": sha256_file(source),
        "artifact": artifact,
    }


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid positive number") from exc
    if not (0.0 < parsed < float("inf")):
        raise argparse.ArgumentTypeError("invalid positive number")
    return parsed


def _add_fixture_switch(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--synthetic-test-fixture",
        action="store_true",
        help="Use fixture contracts; fixture artifacts can never pass production gates.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(
        prog="python -m implementation.cli",
        description=(
            "v4 offline data, tokenizer, CPU, fixed-GPU2 replay, and single-run reporting tools"
        ),
        epilog="Exit codes: 0 completed, 2 usage, 20 blocked, 30 unavailable.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser(
        "status", help="Inspect local policy/capabilities without network or GPU access."
    )
    status.set_defaults(_handler=_status)

    verify = subparsers.add_parser(
        "verify-package", help="Re-hash exactly the immutable delivered manifest."
    )
    verify.set_defaults(_handler=_verify_package)

    design = subparsers.add_parser(
        "audit-design-revision",
        help=(
            "Verify the exact prospective v4.1 protocol amendment without "
            "activating training or reinterpreting legacy freezes."
        ),
    )
    design.set_defaults(_handler=_audit_design_revision)

    integrity = subparsers.add_parser(
        "implementation-integrity",
        help="Hash-bind every tracked implementation file to a clean exact Git HEAD.",
    )
    integrity.add_argument("--out", required=True)
    integrity.set_defaults(_handler=_implementation_integrity)

    plan = subparsers.add_parser(
        "collection-plan", help="Freeze the offline first-199 KRDICT request plan."
    )
    plan.add_argument("--queries", default="templates/queries.txt")
    plan.add_argument("--out", required=True)
    plan.set_defaults(_handler=_collection_plan)

    offline = subparsers.add_parser(
        "offline-import",
        help="Import an already-completed KRDICT snapshot; makes zero HTTP requests.",
    )
    offline.add_argument("--legacy-manifest", required=True)
    offline.add_argument("--plan", required=True)
    offline.add_argument("--output-dir", required=True)
    _add_fixture_switch(offline)
    offline.set_defaults(_handler=_offline_import)

    merge = subparsers.add_parser(
        "merge-source", help="Re-verify and merge the offline KRDICT snapshot."
    )
    merge.add_argument("--manifest", required=True)
    merge.add_argument("--out", required=True)
    _add_fixture_switch(merge)
    merge.set_defaults(_handler=_merge_source)

    review = subparsers.add_parser(
        "export-review", help="Export pending human review without automatic approval."
    )
    review.add_argument("--merge", required=True)
    review.add_argument("--out", required=True)
    review.set_defaults(_handler=_export_review)

    review_csvs = subparsers.add_parser(
        "prepare-review-csvs",
        help=(
            "Create source-bound, reviewer-editable CSV templates; these are "
            "never direct trainer inputs."
        ),
    )
    review_csvs.add_argument("--manifest", required=True)
    review_csvs.add_argument("--output-dir", required=True)
    _add_fixture_switch(review_csvs)
    review_csvs.set_defaults(_handler=_prepare_review_csvs)

    review_audit = subparsers.add_parser(
        "audit-review-csvs",
        help=(
            "Audit three review CSVs offline; training still requires a "
            "separately compiled and verified freeze."
        ),
    )
    review_audit.add_argument("--manifest", required=True)
    review_audit.add_argument("--terms", required=True)
    review_audit.add_argument("--pairs", required=True)
    review_audit.add_argument("--screen", required=True)
    review_audit.add_argument(
        "--require-complete",
        action="store_true",
        help="Return blocked unless every required human review is complete.",
    )
    _add_fixture_switch(review_audit)
    review_audit.set_defaults(_handler=_audit_review_csvs)

    review_freeze = subparsers.add_parser(
        "freeze-review-csvs",
        help=(
            "Compile completed, evidence-bound review CSVs into an annotation "
            "freeze; CSVs never enter a trainer directly."
        ),
    )
    review_freeze.add_argument("--manifest", required=True)
    review_freeze.add_argument("--terms", required=True)
    review_freeze.add_argument("--pairs", required=True)
    review_freeze.add_argument("--screen", required=True)
    review_freeze.add_argument("--evidence-records", required=True)
    review_freeze.add_argument("--output-dir", required=True)
    _add_fixture_switch(review_freeze)
    review_freeze.set_defaults(_handler=_freeze_review_csvs)

    corpus = subparsers.add_parser(
        "stage-corpus", help="Stage a hash-pinned offline Wiki40B/ko JSONL export."
    )
    corpus.add_argument("--export", required=True)
    corpus.add_argument("--expected-sha256", required=True)
    corpus.add_argument("--metadata", required=True)
    corpus.add_argument("--output-dir", required=True)
    _add_fixture_switch(corpus)
    corpus.set_defaults(_handler=_stage_corpus)

    tokenizer = subparsers.add_parser(
        "build-tokenizer", help="Train and freeze the configured KO-only byte BPE."
    )
    tokenizer.add_argument("--source-manifest", required=True)
    tokenizer.add_argument("--output-dir", required=True)
    _add_fixture_switch(tokenizer)
    tokenizer.set_defaults(_handler=_build_tokenizer)

    tokens = subparsers.add_parser(
        "materialize-corpus", help="Materialize the fixed capped corpus token stream."
    )
    tokens.add_argument("--source-manifest", required=True)
    tokens.add_argument("--tokenizer-manifest", required=True)
    tokens.add_argument("--output-dir", required=True)
    _add_fixture_switch(tokens)
    tokens.set_defaults(_handler=_materialize_corpus)

    audit = subparsers.add_parser(
        "audit-token-boundaries",
        help="Audit exact prompt/answer/newline token boundaries and byte coverage.",
    )
    audit.add_argument("--tokenizer-manifest", required=True)
    audit.add_argument("--concepts", required=True)
    audit.add_argument("--out", required=True)
    _add_fixture_switch(audit)
    audit.set_defaults(_handler=_audit_token_boundaries)

    freeze = subparsers.add_parser(
        "freeze-experiment", help="Bind reviewed data, tokenizer, corpus, and audit."
    )
    freeze.add_argument("--annotation-manifest", required=True)
    freeze.add_argument("--tokenizer-manifest", required=True)
    freeze.add_argument("--corpus-manifest", required=True)
    freeze.add_argument("--boundary-audit", required=True)
    freeze.add_argument("--out", required=True)
    _add_fixture_switch(freeze)
    freeze.set_defaults(_handler=_freeze_experiment)

    cpu = subparsers.add_parser(
        "cpu-checks",
        help="Run fixed offline CPU suites with full logs and exact code binding.",
    )
    cpu.add_argument("--implementation-manifest", required=True)
    cpu.add_argument("--output-dir", required=True)
    cpu.set_defaults(_handler=_cpu_checks)

    replay = subparsers.add_parser(
        "gpu2-replay",
        help="Run the charged production-shaped replay on physical GPU 2 only.",
    )
    replay.add_argument("--experiment-freeze", required=True)
    replay.add_argument("--checkpoint-directory", required=True)
    replay.add_argument("--ledger", required=True)
    replay.add_argument("--lock", required=True)
    replay.add_argument("--run-id", required=True)
    replay.add_argument("--implementation-manifest", required=True)
    replay.add_argument("--cpu-checks", required=True)
    replay.add_argument("--out", required=True)
    replay.add_argument("--reserved-seconds", type=_positive_float, default=1800.0)
    replay.add_argument(
        "--checkpoint-grace-seconds", type=_positive_float, default=120.0
    )
    replay.set_defaults(_handler=_gpu2_replay)

    finalize = subparsers.add_parser(
        "finalize-run", help="Validate and publish the one authoritative run JSON."
    )
    finalize.add_argument("--payload", required=True)
    finalize.add_argument("--out", required=True)
    finalize.set_defaults(_handler=_finalize_run)

    render = subparsers.add_parser(
        "render", help="Regenerate Markdown solely from an authoritative run JSON."
    )
    render.add_argument("--json", required=True)
    render.add_argument("--out", required=True)
    render.set_defaults(_handler=_render)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        exit_code, response = args._handler(args)
    except CommandUnavailable as exc:
        exit_code = EXIT_UNAVAILABLE
        response = {
            "schema_version": CLI_SCHEMA,
            "command": args.command,
            "status": "NOT_IMPLEMENTED",
            "error_code": exc.code,
        }
    except ContractViolation as exc:
        exit_code = EXIT_BLOCKED
        response = {
            "schema_version": CLI_SCHEMA,
            "command": args.command,
            "status": "BLOCKED",
            "error_code": exc.code,
        }
    except (KeyError, OSError, TypeError, ValueError):
        # Never serialize exception text: paths, URLs, or environment-derived
        # values can otherwise become accidental secret-bearing diagnostics.
        exit_code = EXIT_BLOCKED
        response = {
            "schema_version": CLI_SCHEMA,
            "command": args.command,
            "status": "BLOCKED",
            "error_code": "BLOCKED_INPUT_OR_IO",
        }
    except Exception:  # fail closed at the process boundary without leaking text
        exit_code = EXIT_BLOCKED
        response = {
            "schema_version": CLI_SCHEMA,
            "command": args.command,
            "status": "BLOCKED",
            "error_code": "INTERNAL_ERROR",
        }
    _emit(response)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
