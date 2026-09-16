from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from implementation import cli
from implementation.src.contracts import ContractViolation


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _invoke(argv: list[str]) -> tuple[int, dict, str]:
    output = io.StringIO()
    with redirect_stdout(output):
        code = cli.main(argv)
    raw = output.getvalue()
    return code, json.loads(raw), raw


def _summary_payload() -> dict:
    return {
        "schema_version": "run-summary.v1",
        "project_id": "lexical-freshstart-4lang-ety-v4",
        "spec_version": "4.0.0",
        "run_id": "cli-test-run",
        "created_at_utc": "2026-09-14T00:00:00Z",
        "status": "BLOCKED_MEASUREMENT",
        "data_kind": "SYNTHETIC_TEST_FIXTURE",
        "policy": {"main_enabled": False},
        "provenance": {
            "spec_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "code_sha256": "3" * 64,
        },
        "statuses": {
            "gpu_training": "NOT_RUN",
            "measurement": "BLOCKED_MEASUREMENT",
        },
        "gates": [
            {
                "name": "measurement",
                "status": "BLOCKED_MEASUREMENT",
                "reasons": ["ko:RHO_BELOW_MINIMUM"],
                "evidence": {"rho": 0.59},
            }
        ],
        "artifacts": {},
        "budget": {
            "campaign_gpu_hours_cap": 24.0,
            "prior_gpu_hours_user_reported": 0.953,
            "measured_gpu_hours": 0.0,
        },
        "roots": {},
        "next_action": "Diagnose the failed measurement without expanding H.",
    }


class ParserTests(unittest.TestCase):
    def test_help_exposes_workflow_without_credential_or_device_arguments(self):
        parser = cli.build_parser()
        help_text = parser.format_help()
        subparser_action = next(
            action
            for action in parser._actions
            if isinstance(action, __import__("argparse")._SubParsersAction)
        )
        for name in (
            "status",
            "verify-package",
            "audit-design-revision",
            "implementation-integrity",
            "collection-plan",
            "offline-import",
            "merge-source",
            "export-review",
            "prepare-review-csvs",
            "audit-review-csvs",
            "freeze-review-csvs",
            "stage-corpus",
            "build-tokenizer",
            "materialize-corpus",
            "audit-token-boundaries",
            "freeze-experiment",
            "cpu-checks",
            "gpu2-replay",
            "finalize-run",
            "render",
        ):
            self.assertIn(name, subparser_action.choices)
            help_text += subparser_action.choices[name].format_help()
        self.assertNotIn("--api-key", help_text)
        self.assertNotIn("--credential", help_text)
        self.assertNotIn("--device", help_text)
        self.assertNotIn("--code-sha256", help_text)

    def test_usage_error_does_not_echo_accidentally_pasted_secret(self):
        secret = "DO_NOT_LOG_THIS_CREDENTIAL"
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            cli.main(["status", "--api-key", secret])
        self.assertEqual(raised.exception.code, cli.EXIT_USAGE)
        self.assertNotIn(secret, stderr.getvalue())
        self.assertIn("CLI_USAGE_ERROR", stderr.getvalue())


class ReadOnlyCommandTests(unittest.TestCase):
    def test_status_is_side_effect_free_and_never_prints_key(self):
        secret = "0123456789abcdef0123456789abcdef"
        with mock.patch.dict(
            os.environ,
            {
                "KRDICT_API_KEY": secret,
                "CUDA_VISIBLE_DEVICES": "2",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            },
            clear=True,
        ), mock.patch.object(
            cli.importlib,
            "import_module",
            side_effect=AssertionError("status imported a command module"),
        ):
            code, result, raw = _invoke(["status"])
        self.assertEqual(code, 0)
        self.assertEqual(result["delivery_integrity"]["status"], "PASS")
        self.assertTrue(result["local_environment"]["krdict_key_present"])
        self.assertTrue(result["local_environment"]["krdict_key_format_valid"])
        self.assertTrue(result["capabilities"]["review_csv_workflow"])
        self.assertTrue(result["capabilities"]["review_csv_freeze"])
        self.assertEqual(result["side_effects"], {"gpu_jobs_started": 0, "network_requests": 0})
        self.assertNotIn(secret, raw)

    def test_verify_package_matches_immutable_manifest(self):
        code, result, _ = _invoke(["verify-package"])
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["failed_files"], 0)
        self.assertGreater(result["checked_files"], 0)

    def test_design_revision_audit_is_read_only_and_not_training_authorization(self):
        code, result, _ = _invoke(["audit-design-revision"])
        self.assertEqual(code, 0)
        self.assertEqual(
            result["status"], "PASS_V4_1_PROSPECTIVE_DESIGN_AMENDMENT"
        )
        self.assertFalse(result["audit"]["core_etymology_required"])
        self.assertFalse(result["audit"]["implementation_ready"])
        self.assertFalse(result["audit"]["training_eligible"])
        self.assertEqual(
            result["audit"]["current_core_term_blockers"][0]["selection_ordinal"],
            38,
        )


class OfflineWorkflowTests(unittest.TestCase):
    def test_collection_plan_is_published_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            out = work / "collection_plan.json"
            with mock.patch.object(cli, "WORK_ROOT", work):
                code, result, _ = _invoke(
                    [
                        "collection-plan",
                        "--queries",
                        str(cli.PROJECT_ROOT / "templates/queries.txt"),
                        "--out",
                        str(out),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(result["status"], "READY_FOR_COLLECTION")
            self.assertEqual(result["planned_requests"], 597)
            self.assertTrue(out.is_file())
            self.assertEqual(json.loads(out.read_text())["remaining_after_plan"], 0)

    def test_offline_import_wires_plan_and_asserts_zero_requests(self):
        calls = []

        def adapter(legacy, plan, **kwargs):
            calls.append((legacy, plan, kwargs))
            return {
                "status": "COLLECTED_UNREVIEWED",
                "data_kind": "REAL_KRDICT",
                "network_requests": 0,
                "requests_made_this_call": 0,
                "manifest_artifact": {"path": "source_manifest.json"},
            }

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            legacy = work / "legacy.json"
            plan = work / "plan.json"
            _write_json(legacy, {})
            _write_json(plan, {"plan_sha256": "a" * 64})
            with mock.patch.object(cli, "WORK_ROOT", work), mock.patch.object(
                cli, "_load_callable", return_value=adapter
            ):
                code, result, _ = _invoke(
                    [
                        "offline-import",
                        "--legacy-manifest",
                        str(legacy),
                        "--plan",
                        str(plan),
                        "--output-dir",
                        str(work / "source"),
                    ]
                )
        self.assertEqual(code, 0)
        self.assertEqual(result["network_requests"], 0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][2]["production"])
        self.assertEqual(calls[0][2]["scope_root"], work)

    def test_merge_and_review_are_separate_write_once_steps(self):
        merged = {
            "status": "MERGED_PENDING_REVIEW",
            "data_kind": "REAL_KRDICT",
            "merge_sha256": "a" * 64,
            "summary": {"eligible": 1, "quarantined": 0},
        }
        pending = {
            "status": "PENDING_HUMAN_REVIEW",
            "data_kind": "REAL_KRDICT",
            "pending_review": [],
            "summary": {
                "pending": 0,
                "automatically_approved": 0,
                "human_signoff_required": True,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "source.json"
            _write_json(source, {})
            merge_out = work / "merge.json"
            review_out = work / "pending_review.json"

            def loader(_module, attribute, _code):
                if attribute == "merge_krdict_snapshot":
                    return lambda *_args, **_kwargs: merged
                if attribute == "export_pending_review":
                    return lambda value: pending if value == merged else None
                self.fail(attribute)

            with mock.patch.object(cli, "WORK_ROOT", work), mock.patch.object(
                cli, "_load_callable", side_effect=loader
            ):
                code, _, _ = _invoke(
                    ["merge-source", "--manifest", str(source), "--out", str(merge_out)]
                )
                self.assertEqual(code, 0)
                code, result, _ = _invoke(
                    ["export-review", "--merge", str(merge_out), "--out", str(review_out)]
                )
            self.assertEqual(code, 0)
            self.assertEqual(result["status"], "PENDING_HUMAN_REVIEW")
            self.assertEqual(json.loads(review_out.read_text())["summary"]["automatically_approved"], 0)

    def test_prepare_review_csvs_publishes_only_nontraining_templates(self):
        calls = []
        merged = {
            "status": "MERGED_PENDING_REVIEW",
            "data_kind": "SYNTHETIC_TEST_FIXTURE",
            "merge_sha256": "a" * 64,
        }
        rendered = {
            "status": "PENDING_HUMAN_REVIEW",
            "training_eligible": False,
            "summary": {"candidate_count": 2, "human_signoff_count": 0},
            "files": {
                "terms_long.csv": b"term_id,decision\nt1,\n",
                "pair_selection_sheet.csv": b"candidate_id,selected\nc1,\n",
                "untranslated_screen.csv": b"term_id,translation_quality\nt1,\n",
            },
        }

        def loader(_module, attribute, _code):
            if attribute == "merge_krdict_snapshot":
                def merge(path, **kwargs):
                    calls.append((attribute, path, kwargs))
                    return merged
                return merge
            if attribute == "render_review_csv_bundle":
                def render(value):
                    calls.append((attribute, value))
                    return rendered
                return render
            self.fail(attribute)

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "source_manifest.json"
            _write_json(source, {})
            destination = work / "review-v3"
            with mock.patch.object(cli, "WORK_ROOT", work), mock.patch.object(
                cli, "_load_callable", side_effect=loader
            ):
                code, result, _ = _invoke(
                    [
                        "prepare-review-csvs",
                        "--manifest",
                        str(source),
                        "--output-dir",
                        str(destination),
                        "--synthetic-test-fixture",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(result["status"], "PENDING_HUMAN_REVIEW")
            self.assertFalse(result["training_eligible"])
            self.assertFalse(result["direct_trainer_input_allowed"])
            self.assertEqual(
                {path.name for path in destination.iterdir()},
                {
                    "terms_long.csv",
                    "pair_selection_sheet.csv",
                    "untranslated_screen.csv",
                    "review_csv_manifest.json",
                },
            )
            manifest = json.loads(
                (destination / "review_csv_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["training_eligible"])
            self.assertFalse(manifest["direct_trainer_input_allowed"])
            self.assertEqual(set(manifest["csv_templates"]), set(cli.REVIEW_CSV_FILENAMES))
            self.assertEqual(
                (destination / "untranslated_screen.csv").stat().st_mode & 0o777,
                0o444,
            )
            self.assertEqual(
                (destination / "terms_long.csv").stat().st_mode & 0o777,
                0o600,
            )
            self.assertEqual(calls[0][0], "merge_krdict_snapshot")
            self.assertFalse(calls[0][2]["production"])
            self.assertEqual(calls[1], ("render_review_csv_bundle", merged))

    def test_audit_review_csvs_allows_incomplete_diagnostic_but_strict_blocks(self):
        calls = []
        merged = {
            "status": "MERGED_PENDING_REVIEW",
            "data_kind": "SYNTHETIC_TEST_FIXTURE",
            "merge_sha256": "a" * 64,
        }
        tables = object()

        def loader(_module, attribute, _code):
            if attribute == "merge_krdict_snapshot":
                return lambda path, **kwargs: (
                    calls.append((attribute, path, kwargs)) or merged
                )
            if attribute == "load_and_validate_review_csv_bundle":
                return lambda value, terms, pairs, screen: (
                    calls.append((attribute, value, terms, pairs, screen)) or tables
                )
            if attribute == "validate_review_tables":
                return lambda value, loaded, **kwargs: (
                    calls.append((attribute, value, loaded, kwargs))
                    or {
                        "status": "BLOCKED_DATA_QA",
                        "training_eligible": False,
                        "summary": {
                            "candidate_count": 409,
                            "human_signoff_count": 0,
                        },
                    }
                )
            self.fail(attribute)

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "source_manifest.json"
            _write_json(source, {})
            csv_paths = []
            for name in cli.REVIEW_CSV_FILENAMES:
                path = work / name
                path.write_text("header\n", encoding="utf-8")
                csv_paths.append(path)
            argv = [
                "audit-review-csvs",
                "--manifest",
                str(source),
                "--terms",
                str(csv_paths[0]),
                "--pairs",
                str(csv_paths[1]),
                "--screen",
                str(csv_paths[2]),
                "--synthetic-test-fixture",
            ]
            with mock.patch.object(cli, "WORK_ROOT", work), mock.patch.object(
                cli, "_load_callable", side_effect=loader
            ):
                code, result, _ = _invoke(argv)
                strict_code, strict_result, _ = _invoke(argv + ["--require-complete"])
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "BLOCKED_DATA_QA")
        self.assertFalse(result["review_compile_ready"])
        self.assertFalse(result["direct_trainer_input_allowed"])
        self.assertEqual(strict_code, cli.EXIT_BLOCKED)
        self.assertEqual(strict_result["status"], "BLOCKED_DATA_QA")
        validations = [row for row in calls if row[0] == "validate_review_tables"]
        self.assertEqual(
            [row[3]["require_complete"] for row in validations], [False, True]
        )

    def test_strict_review_audit_succeeds_only_for_complete_structure(self):
        merged = {
            "status": "MERGED_PENDING_REVIEW",
            "data_kind": "SYNTHETIC_TEST_FIXTURE",
            "merge_sha256": "a" * 64,
        }

        def loader(_module, attribute, _code):
            return {
                "merge_krdict_snapshot": lambda *_args, **_kwargs: merged,
                "load_and_validate_review_csv_bundle": lambda *_args, **_kwargs: object(),
                "validate_review_tables": lambda *_args, **_kwargs: {
                    "status": "STRUCTURE_PASS_PENDING_EVIDENCE",
                    "training_eligible": False,
                    "summary": {"pilot_count": 60},
                },
            }[attribute]

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            paths = []
            for name in ("source.json", *cli.REVIEW_CSV_FILENAMES):
                path = work / name
                path.write_text("fixture\n", encoding="utf-8")
                paths.append(path)
            with mock.patch.object(cli, "WORK_ROOT", work), mock.patch.object(
                cli, "_load_callable", side_effect=loader
            ):
                code, result, _ = _invoke(
                    [
                        "audit-review-csvs",
                        "--manifest",
                        str(paths[0]),
                        "--terms",
                        str(paths[1]),
                        "--pairs",
                        str(paths[2]),
                        "--screen",
                        str(paths[3]),
                        "--require-complete",
                        "--synthetic-test-fixture",
                    ]
                )
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "STRUCTURE_PASS_PENDING_EVIDENCE")
        self.assertTrue(result["structure_complete"])
        self.assertFalse(result["training_eligible"])

    def test_freeze_review_csvs_is_the_only_csv_to_annotation_boundary(self):
        calls = []

        def freeze(source, terms, pairs, screen, **kwargs):
            calls.append((source, terms, pairs, screen, kwargs))
            return {
                "status": "PASS",
                "data_kind": "SYNTHETIC_TEST_FIXTURE",
                "freeze_id": "annotation-fixture",
                "manifest_artifact": {
                    "path": str(kwargs["output_dir"] / "annotation_freeze_manifest.json"),
                    "sha256": "a" * 64,
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            inputs = []
            for name in (
                "source_manifest.json",
                "terms_long.csv",
                "pair_selection_sheet.csv",
                "untranslated_screen.csv",
            ):
                path = work / name
                path.write_text("fixture\n", encoding="utf-8")
                inputs.append(path)
            evidence = work / "evidence.jsonl"
            evidence.write_text(
                json.dumps({"evidence_id": "e1"}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            output_dir = work / "annotation-freeze"
            with mock.patch.object(cli, "WORK_ROOT", work), mock.patch.object(
                cli, "_load_callable", return_value=freeze
            ):
                code, result, _ = _invoke(
                    [
                        "freeze-review-csvs",
                        "--manifest",
                        str(inputs[0]),
                        "--terms",
                        str(inputs[1]),
                        "--pairs",
                        str(inputs[2]),
                        "--screen",
                        str(inputs[3]),
                        "--evidence-records",
                        str(evidence),
                        "--output-dir",
                        str(output_dir),
                        "--synthetic-test-fixture",
                    ]
                )
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["freeze_id"], "annotation-fixture")
        self.assertFalse(result["direct_trainer_input_allowed"])
        self.assertEqual(result["trainer_input"], "VERIFIED_EXPERIMENT_FREEZE_ONLY")
        self.assertFalse(result["scientific_training_started"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:4], tuple(path.resolve() for path in inputs))
        self.assertEqual(calls[0][4]["evidence_records"], [{"evidence_id": "e1"}])
        self.assertEqual(calls[0][4]["output_dir"], output_dir.resolve())
        self.assertFalse(calls[0][4]["production"])
        self.assertEqual(calls[0][4]["scope_root"], work)

    def test_freeze_review_csvs_rejects_nonpass_wrapper_result(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            paths = []
            for name in ("source.json", "terms.csv", "pairs.csv", "screen.csv"):
                path = work / name
                path.write_text("fixture\n", encoding="utf-8")
                paths.append(path)
            evidence = work / "evidence.jsonl"
            evidence.write_text('{}\n', encoding="utf-8")
            with mock.patch.object(cli, "WORK_ROOT", work), mock.patch.object(
                cli,
                "_load_callable",
                return_value=lambda *_args, **_kwargs: {"status": "BLOCKED_DATA_QA"},
            ):
                code, result, _ = _invoke(
                    [
                        "freeze-review-csvs",
                        "--manifest",
                        str(paths[0]),
                        "--terms",
                        str(paths[1]),
                        "--pairs",
                        str(paths[2]),
                        "--screen",
                        str(paths[3]),
                        "--evidence-records",
                        str(evidence),
                        "--output-dir",
                        str(work / "freeze"),
                        "--synthetic-test-fixture",
                    ]
                )
        self.assertEqual(code, cli.EXIT_BLOCKED)
        self.assertEqual(result["error_code"], "ANNOTATION_FREEZE_NOT_PASS")

    def test_offline_corpus_staging_passes_hash_pinned_iterator(self):
        seen = {}

        def iterator(path, *, expected_sha256):
            seen["iterator"] = (path, expected_sha256)
            return iter([{"text": "fixture"}])

        def stage(examples, metadata, **kwargs):
            seen["examples"] = list(examples)
            seen["metadata"] = metadata
            seen["stage_kwargs"] = kwargs
            return {
                "status": "PASS",
                "data_kind": "SYNTHETIC_TEST_FIXTURE",
                "manifest_artifact": {"path": "manifest.json"},
            }

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            export = work / "wiki.jsonl"
            export.write_text('{"text":"fixture"}\n', encoding="utf-8")
            digest = __import__("hashlib").sha256(export.read_bytes()).hexdigest()
            metadata = work / "metadata.json"
            _write_json(metadata, {"data_kind": "SYNTHETIC_TEST_FIXTURE"})

            def loader(_module, attribute, _code):
                return {
                    "iter_wiki40b_jsonl_export": iterator,
                    "stage_wiki40b_documents": stage,
                }[attribute]

            with mock.patch.object(cli, "WORK_ROOT", work), mock.patch.object(
                cli, "_load_callable", side_effect=loader
            ):
                code, result, _ = _invoke(
                    [
                        "stage-corpus",
                        "--export",
                        str(export),
                        "--expected-sha256",
                        digest,
                        "--metadata",
                        str(metadata),
                        "--output-dir",
                        str(work / "staged"),
                        "--synthetic-test-fixture",
                    ]
                )
        self.assertEqual(code, 0)
        self.assertEqual(result["data_kind"], "SYNTHETIC_TEST_FIXTURE")
        self.assertEqual(seen["examples"], [{"text": "fixture"}])
        self.assertFalse(seen["stage_kwargs"]["production"])


class TokenizerAndChecksTests(unittest.TestCase):
    def test_tokenizer_commands_use_frozen_plan_and_no_device(self):
        calls = []

        def fake(*args, **kwargs):
            calls.append((args, kwargs))
            return {"status": "PASS", "data_kind": "SYNTHETIC_TEST_FIXTURE"}

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "source.json"
            tokenizer = work / "tokenizer.json"
            concepts = work / "concepts.jsonl"
            _write_json(source, {})
            _write_json(tokenizer, {})
            concepts.write_text('{"concept_id":"c1"}\n', encoding="utf-8")
            with mock.patch.object(cli, "WORK_ROOT", work), mock.patch.object(
                cli, "_load_callable", return_value=fake
            ):
                for argv in (
                    [
                        "build-tokenizer",
                        "--source-manifest",
                        str(source),
                        "--output-dir",
                        str(work / "tok-out"),
                        "--synthetic-test-fixture",
                    ],
                    [
                        "materialize-corpus",
                        "--source-manifest",
                        str(source),
                        "--tokenizer-manifest",
                        str(tokenizer),
                        "--output-dir",
                        str(work / "corpus-out"),
                        "--synthetic-test-fixture",
                    ],
                    [
                        "audit-token-boundaries",
                        "--tokenizer-manifest",
                        str(tokenizer),
                        "--concepts",
                        str(concepts),
                        "--out",
                        str(work / "audit.json"),
                        "--synthetic-test-fixture",
                    ],
                ):
                    code, _, _ = _invoke(argv)
                    self.assertEqual(code, 0)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call[1]["production"] is False for call in calls))
        self.assertTrue(all(call[1].get("scope_root") == work for call in calls))
        self.assertEqual(calls[0][1]["evaluation_plan_path"], cli.EVALUATION_PLAN_PATH)
        self.assertEqual(calls[2][1]["evaluation_plan_path"], cli.EVALUATION_PLAN_PATH)

    def test_unavailable_api_has_stable_code_and_no_exception_text(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "source.json"
            _write_json(source, {})
            unavailable = cli.CommandUnavailable("NOT_IMPLEMENTED_BUILD_TOKENIZER")
            with mock.patch.object(cli, "WORK_ROOT", work), mock.patch.object(
                cli, "_load_callable", side_effect=unavailable
            ):
                code, result, raw = _invoke(
                    [
                        "build-tokenizer",
                        "--source-manifest",
                        str(source),
                        "--output-dir",
                        str(work / "out"),
                    ]
                )
        self.assertEqual(code, cli.EXIT_UNAVAILABLE)
        self.assertEqual(result["error_code"], "NOT_IMPLEMENTED_BUILD_TOKENIZER")
        self.assertNotIn("Traceback", raw)

    def test_integrity_and_cpu_checks_are_code_bound(self):
        calls = []

        def loader(_module, attribute, _code):
            if attribute == "publish_implementation_manifest":
                def publish(path, **kwargs):
                    calls.append((attribute, path, kwargs))
                    return {
                        "status": "PASS",
                        "manifest_artifact": {"path": str(path), "sha256": "1" * 64},
                    }
                return publish
            if attribute == "run_cpu_check_evidence":
                def checks(path, **kwargs):
                    calls.append((attribute, path, kwargs))
                    return {
                        "status": "PASS",
                        "scientific_result": False,
                        "manifest_artifact": {
                            "path": str(kwargs["output_dir"] / "cpu_checks.json"),
                            "sha256": "2" * 64,
                        },
                    }
                return checks
            self.fail(attribute)

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = work / "implementation_manifest.json"
            _write_json(manifest, {})
            with mock.patch.object(cli, "WORK_ROOT", work), mock.patch.object(
                cli, "_load_callable", side_effect=loader
            ):
                code, result, _ = _invoke(
                    ["implementation-integrity", "--out", str(work / "new_manifest.json")]
                )
                self.assertEqual(code, 0)
                self.assertEqual(result["status"], "PASS")
                code, result, _ = _invoke(
                    [
                        "cpu-checks",
                        "--implementation-manifest",
                        str(manifest),
                        "--output-dir",
                        str(work / "cpu_evidence"),
                    ]
                )
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual([row[0] for row in calls], [
            "publish_implementation_manifest",
            "run_cpu_check_evidence",
        ])
        self.assertEqual(calls[1][1], manifest.resolve())
        self.assertEqual(calls[1][2]["output_dir"], (work / "cpu_evidence").resolve())


class GpuAndReportingTests(unittest.TestCase):
    def test_gpu_mask_failure_occurs_before_train_import(self):
        loaded = []

        def loader(_module, attribute, _code):
            loaded.append(attribute)
            if attribute == "require_literal_gpu2_mask":
                return lambda: (_ for _ in ()).throw(
                    ContractViolation("BLOCKED_GPU2_MASK")
                )
            self.fail("training imported before failed mask")

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            freeze = work / "freeze.json"
            _write_json(freeze, {})
            with mock.patch.object(cli, "WORK_ROOT", work), mock.patch.object(
                cli, "_load_callable", side_effect=loader
            ):
                code, result, _ = _invoke(
                    [
                        "gpu2-replay",
                        "--experiment-freeze",
                        str(freeze),
                        "--checkpoint-directory",
                        str(work / "checkpoint"),
                        "--ledger",
                        str(work / "ledger.jsonl"),
                        "--lock",
                        str(work / "gpu.lock"),
                        "--run-id",
                        "replay-test",
                        "--implementation-manifest",
                        str(work / "implementation.json"),
                        "--cpu-checks",
                        str(work / "cpu_checks.json"),
                        "--out",
                        str(work / "replay.json"),
                    ]
                )
        self.assertEqual(code, cli.EXIT_BLOCKED)
        self.assertEqual(result["error_code"], "BLOCKED_GPU2_MASK")
        self.assertEqual(loaded, ["require_literal_gpu2_mask"])

    def test_gpu_replay_wiring_is_fixed_to_guard_and_budget(self):
        events = []

        class Caps:
            def __init__(self, **kwargs):
                events.append(("caps", kwargs))

        class Ledger:
            def __init__(self, path, caps):
                events.append(("ledger", path, caps))

        def guard():
            events.append("guard")

        def replay(**kwargs):
            events.append(("replay", kwargs))
            return {
                "status": "PASS",
                "scientific_result": False,
                "freeze_sha256": "d" * 64,
                "code_sha256": kwargs["code_sha256"],
                "cpu_check_sha256": kwargs["cpu_check_sha256"],
                "gpu_binding": {
                    "physical_index": 2,
                    "logical_index": 0,
                    "uuid": "GPU-de18b86c-419a-795a-0667-c7ae03bf800f",
                },
            }

        def loader(_module, attribute, _code):
            events.append(("load", attribute))
            return {
                "require_literal_gpu2_mask": guard,
                "verify_implementation_manifest": lambda *_args, **_kwargs: {
                    "implementation_manifest_sha256": "b" * 64,
                },
                "verify_cpu_check_evidence": lambda *_args, **_kwargs: {
                    "cpu_check_sha256": "c" * 64,
                    "implementation": {
                        "implementation_manifest_sha256": "b" * 64,
                    },
                },
                "BudgetCaps": Caps,
                "GpuBudgetLedger": Ledger,
                "run_gpu2_replay": replay,
            }[attribute]

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            freeze = work / "freeze.json"
            _write_json(freeze, {})
            with mock.patch.object(cli, "WORK_ROOT", work), mock.patch.object(
                cli, "_load_callable", side_effect=loader
            ):
                code, result, _ = _invoke(
                    [
                        "gpu2-replay",
                        "--experiment-freeze",
                        str(freeze),
                        "--checkpoint-directory",
                        str(work / "checkpoint"),
                        "--ledger",
                        str(work / "ledger.jsonl"),
                        "--lock",
                        str(work / "gpu.lock"),
                        "--run-id",
                        "replay-test",
                        "--implementation-manifest",
                        str(work / "implementation.json"),
                        "--cpu-checks",
                        str(work / "cpu_checks.json"),
                        "--out",
                        str(work / "replay.json"),
                    ]
                )
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "PASS")
        guard_position = events.index("guard")
        train_load_position = events.index(("load", "run_gpu2_replay"))
        self.assertLess(guard_position, train_load_position)
        replay_kwargs = next(event[1] for event in events if isinstance(event, tuple) and event[0] == "replay")
        self.assertNotIn("device", replay_kwargs)
        self.assertEqual(replay_kwargs["code_sha256"], "b" * 64)
        self.assertEqual(replay_kwargs["cpu_check_sha256"], "c" * 64)

    def test_gpu_replay_rejects_result_not_bound_to_verified_evidence(self):
        class Caps:
            def __init__(self, **_kwargs):
                pass

        class Ledger:
            def __init__(self, _path, _caps):
                pass

        def loader(_module, attribute, _code):
            return {
                "require_literal_gpu2_mask": lambda: None,
                "verify_implementation_manifest": lambda *_args, **_kwargs: {
                    "implementation_manifest_sha256": "b" * 64,
                },
                "verify_cpu_check_evidence": lambda *_args, **_kwargs: {
                    "cpu_check_sha256": "c" * 64,
                    "implementation": {
                        "implementation_manifest_sha256": "b" * 64,
                    },
                },
                "BudgetCaps": Caps,
                "GpuBudgetLedger": Ledger,
                "run_gpu2_replay": lambda **_kwargs: {
                    "status": "PASS",
                    "scientific_result": False,
                },
            }[attribute]

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            freeze = work / "freeze.json"
            _write_json(freeze, {})
            with mock.patch.object(cli, "WORK_ROOT", work), mock.patch.object(
                cli, "_load_callable", side_effect=loader
            ):
                code, result, _ = _invoke(
                    [
                        "gpu2-replay",
                        "--experiment-freeze",
                        str(freeze),
                        "--checkpoint-directory",
                        str(work / "checkpoint"),
                        "--ledger",
                        str(work / "ledger.jsonl"),
                        "--lock",
                        str(work / "gpu.lock"),
                        "--run-id",
                        "replay-test",
                        "--implementation-manifest",
                        str(work / "implementation.json"),
                        "--cpu-checks",
                        str(work / "cpu_checks.json"),
                        "--out",
                        str(work / "replay.json"),
                    ]
                )
        self.assertEqual(code, cli.EXIT_BLOCKED)
        self.assertEqual(result["error_code"], "BLOCKED_GPU_REPLAY_RESULT")

    def test_finalize_is_write_once_and_render_is_json_only(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            payload = work / "payload.json"
            summary = work / "run.json"
            markdown = work / "run.md"
            _write_json(payload, _summary_payload())
            with mock.patch.object(cli, "WORK_ROOT", work):
                code, result, _ = _invoke(
                    ["finalize-run", "--payload", str(payload), "--out", str(summary)]
                )
                self.assertEqual(code, 0)
                self.assertEqual(result["status"], "PUBLISHED")
                code, rendered, _ = _invoke(
                    ["render", "--json", str(summary), "--out", str(markdown)]
                )
                self.assertEqual(code, 0)
                self.assertEqual(rendered["status"], "RENDERED_FROM_RUN_JSON")
                code, blocked, _ = _invoke(
                    ["finalize-run", "--payload", str(payload), "--out", str(summary)]
                )
            self.assertEqual(code, cli.EXIT_BLOCKED)
            self.assertEqual(blocked["error_code"], "ARTIFACT_EXISTS")
            text = markdown.read_text(encoding="utf-8")
            self.assertIn("BLOCKED_MEASUREMENT", text)
            self.assertIn(rendered["source_json_sha256"], text)


if __name__ == "__main__":
    unittest.main()
