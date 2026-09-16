from __future__ import annotations

import math
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from implementation.src.artifacts import (
    canonical_json_bytes,
    publish_bytes_once,
    publish_json_once,
    read_regular_file_bytes,
    read_verified_json,
    read_verified_json_with_sha256,
    sha256_bytes,
    sha256_file,
)
from implementation.src.contracts import PROJECT_ROOT, ContractViolation
from implementation.src.report import (
    COMPLETION_REQUIRED_ARTIFACTS,
    COMPLETION_REQUIRED_GATES,
    _verify_gpu_replay_evidence,
    authorize_phase,
    compute_measurement,
    compute_readiness,
    publish_run_summary,
    render_markdown,
    validate_run_summary,
)
from implementation.src.score import (
    build_continuation_events,
    classify_first_line,
    continuation_logprob,
    evaluation_record_content_sha256,
    greedy_first_line,
    partition_probability_events,
    rng_fingerprint,
    score_checkpoint,
    score_fixture_checkpoint,
    semantic_model_fingerprint,
)


ANSWERS = {"ko": "koala", "en": "word", "zh": "hanzi", "fr": "mot"}
HASH = "a" * 64


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(text.encode("ascii"))

    def decode(self, token_ids, **kwargs):
        del kwargs
        return bytes(token_ids).decode("ascii")


class CollisionTokenizer(CharacterTokenizer):
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "P":
            return [1]
        return [1, 2]


class UnstableTokenizer(CharacterTokenizer):
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [1] if text == "P" else [9, 2]


class NewlineModel(torch.nn.Module):
    def __init__(self, vocab_size=128, next_token=10):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size
        self.next_token = next_token

    def forward(self, input_ids, use_cache=False):
        del use_cache
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.vocab_size, device=input_ids.device)
        logits[..., self.next_token] = 9.0 + self.anchor
        return SimpleNamespace(logits=logits)


class ArtifactTests(unittest.TestCase):
    def test_write_once_rejects_even_identical_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            first = publish_json_once(path, {"value": 1})
            self.assertEqual(first["sha256"], sha256_file(path))
            self.assertEqual(read_verified_json(path), {"value": 1})
            with self.assertRaisesRegex(ContractViolation, "ARTIFACT_EXISTS"):
                publish_json_once(path, {"value": 1})
            with self.assertRaisesRegex(ContractViolation, "ARTIFACT_EXISTS"):
                publish_json_once(path, {"value": 2})

    def test_nonfinite_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            with self.assertRaisesRegex(ContractViolation, "INVALID_JSON_PAYLOAD"):
                publish_json_once(path, {"value": math.nan})
            self.assertFalse(path.exists())

    def test_single_descriptor_read_binds_parsed_bytes_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "record.json"
            ref = publish_json_once(path, {"value": 1})
            parsed, digest = read_verified_json_with_sha256(
                path, expected_sha256=ref["sha256"]
            )
            self.assertEqual(parsed, {"value": 1})
            self.assertEqual(digest, ref["sha256"])
            self.assertEqual(read_regular_file_bytes(path), canonical_json_bytes(parsed))

            link = root / "record-link.json"
            link.symlink_to(path)
            for reader in (read_regular_file_bytes, read_verified_json):
                with self.subTest(reader=reader.__name__), self.assertRaisesRegex(
                    ContractViolation, "ARTIFACT_NOT_REGULAR_FILE"
                ):
                    reader(link)

    def test_concurrent_publication_has_one_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "race.json"
            barrier = threading.Barrier(2)
            outcomes = []

            def writer(value):
                barrier.wait()
                try:
                    publish_json_once(path, {"value": value})
                    outcomes.append("PASS")
                except ContractViolation as exc:
                    outcomes.append(exc.code)

            threads = [threading.Thread(target=writer, args=(value,)) for value in (1, 2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(outcomes.count("PASS"), 1)
            self.assertEqual(outcomes.count("ARTIFACT_EXISTS"), 1)
            self.assertIn(read_verified_json(path)["value"], {1, 2})


class ProbabilityScorerTests(unittest.TestCase):
    def test_shared_expression_is_one_event(self):
        answers = dict(ANSWERS, fr="word")
        events = build_continuation_events(
            CharacterTokenizer(), "P", answers, context_length=64
        )
        self.assertEqual(len(events), 3)
        shared = [event for event in events if event["canonical_text"] == "word"][0]
        self.assertEqual(shared["membership"], ["en", "fr"])
        result = partition_probability_events(
            answers, {event["canonical_text"]: math.log(0.1) for event in events}
        )
        self.assertAlmostEqual(result["Z"], 0.3)
        self.assertIsNone(result["Q_language"])
        self.assertAlmostEqual(result["shared_mass"], 1 / 3)

    def test_finite_logz_underflow_and_all_negative_infinity_differ(self):
        finite = partition_probability_events(
            ANSWERS, {answer: -1000.0 for answer in ANSWERS.values()}
        )
        self.assertEqual(finite["Z"], 0.0)
        self.assertEqual(finite["logZ_status"], "FINITE")
        self.assertIsNotNone(finite["Q_language"])
        undefined = partition_probability_events(
            ANSWERS, {answer: -math.inf for answer in ANSWERS.values()}
        )
        self.assertEqual(undefined["logZ_status"], "NEGATIVE_INFINITY")
        self.assertIsNone(undefined["Q_language"])

    def test_collision_and_unstable_boundary_are_rejected(self):
        with self.assertRaisesRegex(ContractViolation, "TOKEN_EVENT_COLLISION"):
            build_continuation_events(
                CollisionTokenizer(), "P", ANSWERS, context_length=64
            )
        with self.assertRaisesRegex(ContractViolation, "UNSTABLE_TOKEN_BOUNDARY"):
            build_continuation_events(
                UnstableTokenizer(), "P", ANSWERS, context_length=64
            )

    def test_context_overflow_is_rejected_not_truncated(self):
        with self.assertRaisesRegex(
            ContractViolation, "SCORING_SEQUENCE_EXCEEDS_CONTEXT"
        ):
            build_continuation_events(
                CharacterTokenizer(), "P", ANSWERS, context_length=5
            )

    def test_generation_classes_are_exclusive(self):
        self.assertEqual(classify_first_line("\n", ANSWERS, "en")["class"], "EMPTY")
        self.assertEqual(
            classify_first_line("other\n", ANSWERS, "en")["class"], "UNREGISTERED"
        )
        self.assertEqual(
            classify_first_line("word\n", ANSWERS, "fr")["class"],
            "REGISTERED_OTHER_LANGUAGE",
        )
        self.assertEqual(
            classify_first_line("word\n", ANSWERS, None)["class"], "REGISTERED"
        )
        shared = dict(ANSWERS, fr="word")
        matched = classify_first_line("word\n", shared, "fr")
        self.assertEqual(matched["class"], "REGISTERED_COMPATIBLE")
        self.assertEqual(matched["membership"], ["en", "fr"])

    def test_unterminated_generation_uses_frozen_failure_class(self):
        result = greedy_first_line(
            NewlineModel(next_token=ord("x")),
            CharacterTokenizer(),
            "P",
            ANSWERS,
            "en",
            context_length=64,
            max_new_tokens=2,
        )
        self.assertEqual(result["termination"], "MAX_NEW_TOKENS")
        self.assertEqual(result["class"], "UNREGISTERED_UNTERMINATED")
        self.assertFalse(result["registered_match"])

    def test_actual_gpt2_teacher_forcing_matches_manual_indexing(self):
        from tokenizers import ByteLevelBPETokenizer
        from transformers import GPT2Config, GPT2LMHeadModel

        tokenizer = ByteLevelBPETokenizer(add_prefix_space=False)
        tokenizer.train_from_iterator(
            ["Prompt\nkoala word hanzi mot\n", "another small tokenizer corpus\n"],
            vocab_size=300,
            min_frequency=1,
            special_tokens=["<|endoftext|>", "<|pad|>"],
        )
        prefix = "Prompt\n"
        events = build_continuation_events(
            tokenizer, prefix, ANSWERS, context_length=64
        )
        event = [event for event in events if event["canonical_text"] == "word"][0]
        torch.manual_seed(7)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=tokenizer.get_vocab_size(),
                n_positions=64,
                n_ctx=64,
                n_embd=16,
                n_layer=1,
                n_head=2,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
        )
        before = rng_fingerprint()
        scored = continuation_logprob(
            model, event["prefix_token_ids"], event["continuation_token_ids"]
        )
        self.assertEqual(before, rng_fingerprint())
        full = event["prefix_token_ids"] + event["continuation_token_ids"]
        model.eval()
        with torch.inference_mode():
            logits = model(
                input_ids=torch.tensor([full[:-1]]), use_cache=False
            ).logits[0]
        start = len(event["prefix_token_ids"]) - 1
        manual = []
        for offset, target in enumerate(event["continuation_token_ids"]):
            value = torch.log_softmax(logits[start + offset].float(), dim=-1)[target]
            manual.append(float(value))
        self.assertEqual(len(scored["token_log_probabilities"]), len(manual))
        for actual, expected in zip(scored["token_log_probabilities"], manual):
            self.assertAlmostEqual(actual, expected, places=7)
        self.assertAlmostEqual(scored["log_p"], math.fsum(manual), places=7)
        self.assertNotAlmostEqual(scored["log_p"], manual[0], places=5)
        self.assertNotAlmostEqual(
            scored["log_p"], math.fsum(manual) / len(manual), places=5
        )

    def test_checkpoint_score_carries_full_provenance(self):
        model = NewlineModel()
        fingerprint = semantic_model_fingerprint(model)
        metadata = {
            "root_id": 4101,
            "phase": "S",
            "stage": "T3",
            "branch": None,
            "checkpoint_sha256": "1" * 64,
            "model_fingerprint": fingerprint,
            "freeze_sha256": "2" * 64,
            "config_sha256": "3" * 64,
            "tokenizer_sha256": "4" * 64,
            "code_sha256": "5" * 64,
            "evaluation_plan_sha256": "6" * 64,
            "evaluation_record_hash_schema": "score-record-v1",
            "data_kind": "SYNTHETIC_TEST_FIXTURE",
            "context_length": 64,
            "maximum_new_tokens": 3,
            "generation_strategy": "MANUAL_GREEDY",
            "terminator": "NEWLINE",
            "candidate_universe": ["ko", "en", "zh", "fr"],
        }
        record = {
            "record_id": "r1",
            "concept_id": "c1",
            "input_language": "ko",
            "format": "RD",
            "wrapper": "dev1",
            "split": "dev",
            "mode": "ANY",
            "prefix": "P",
            "answers": ANSWERS,
        }
        result = score_fixture_checkpoint(
            model,
            CharacterTokenizer(),
            [record],
            metadata,
            context_length=64,
            max_new_tokens=3,
        )
        self.assertTrue(result["rng_unchanged"])
        row = result["rows"][0]
        self.assertEqual(row["checkpoint_sha256"], "1" * 64)
        self.assertEqual(row["model_fingerprint"], fingerprint)
        self.assertEqual(row["generation"]["class"], "EMPTY")
        self.assertEqual(row["probability"]["probability_event_definition"],
                         "P(canonical(answer + newline) | exact prefix, ANY)")

    def test_real_scorer_rejects_shape_only_provenance_and_fixture_entry(self):
        model = NewlineModel()
        metadata = {
            "root_id": 4101,
            "phase": "S",
            "stage": "T3",
            "branch": None,
            "checkpoint_sha256": "0" * 64,
            "model_fingerprint": semantic_model_fingerprint(model),
            "freeze_sha256": "0" * 64,
            "config_sha256": "0" * 64,
            "tokenizer_sha256": "0" * 64,
            "code_sha256": "0" * 64,
            "evaluation_plan_sha256": "0" * 64,
            "evaluation_record_hash_schema": "score-record-v1",
            "data_kind": "REAL",
            "freeze_gate_status": "PASS",
            "context_length": 256,
            "maximum_new_tokens": 64,
            "generation_strategy": "MANUAL_GREEDY",
            "terminator": "NEWLINE",
            "candidate_universe": ["ko", "en", "zh", "fr"],
        }
        with self.assertRaisesRegex(ContractViolation, "EXPERIMENT_FREEZE_PATH_REQUIRED"):
            score_checkpoint(
                model,
                CharacterTokenizer(),
                [{}],
                metadata,
                context_length=256,
                max_new_tokens=64,
            )
        fixture = dict(metadata, data_kind="SYNTHETIC_TEST_FIXTURE")
        with self.assertRaisesRegex(ContractViolation, "REAL_SCORER_REQUIRES_REAL_DATA"):
            score_checkpoint(
                model,
                CharacterTokenizer(),
                [{}],
                fixture,
                context_length=256,
                max_new_tokens=64,
            )

    def test_real_scorer_requires_exact_record_hash_after_artifact_binding(self):
        model = NewlineModel()
        metadata = {
            "root_id": 4101,
            "phase": "S",
            "stage": "T3",
            "branch": None,
            "checkpoint_sha256": "1" * 64,
            "model_fingerprint": semantic_model_fingerprint(model),
            "freeze_sha256": "2" * 64,
            "config_sha256": "3" * 64,
            "tokenizer_sha256": "4" * 64,
            "code_sha256": "5" * 64,
            "evaluation_plan_sha256": "6" * 64,
            "evaluation_record_hash_schema": "score-record-v1",
            "data_kind": "REAL",
            "freeze_gate_status": "PASS",
            "context_length": 256,
            "maximum_new_tokens": 64,
            "generation_strategy": "MANUAL_GREEDY",
            "terminator": "NEWLINE",
            "candidate_universe": ["ko", "en", "zh", "fr"],
        }
        concept = {
            "concept_id": "c1",
            "answers": ANSWERS,
            "glosses": {language: "g" for language in ANSWERS},
        }
        plan = {
            "prompt_plan": {
                "format": "RD",
                "development_wrappers": ["dev1", "dev2"],
                "test_wrappers": ["test1", "test2"],
                "target_language_names": {},
                "templates": {
                    "dev1": {"ko": {"ANY": "P"}},
                },
            }
        }
        record = {
            "record_id": "r1",
            "concept_id": "c1",
            "input_language": "ko",
            "format": "RD",
            "wrapper": "dev1",
            "split": "dev",
            "mode": "ANY",
            "prefix": "P",
            "answers": ANSWERS,
        }
        binding = ({"concepts": [concept]}, plan)
        with mock.patch(
            "implementation.src.score._verify_real_scoring_bindings",
            return_value=binding,
        ):
            with self.assertRaisesRegex(ContractViolation, "CONTENT_HASH_REQUIRED"):
                score_checkpoint(
                    model,
                    CharacterTokenizer(),
                    [record],
                    metadata,
                    context_length=256,
                    max_new_tokens=64,
                    experiment_freeze_path="freeze.json",
                    checkpoint_directory="checkpoint",
                )
            record["record_content_sha256"] = evaluation_record_content_sha256(record)
            result = score_checkpoint(
                model,
                CharacterTokenizer(),
                [record],
                metadata,
                context_length=256,
                max_new_tokens=64,
                experiment_freeze_path="freeze.json",
                checkpoint_directory="checkpoint",
            )
        self.assertEqual(result["rows"][0]["record_content_sha256"], record["record_content_sha256"])


def readiness_rows(*, bad_cell=None):
    rows = []
    for wrapper in ("dev1", "dev2"):
        for language in ("ko", "en", "zh", "fr"):
            for index in range(10):
                compatible = not (bad_cell == (wrapper, language) and index >= 8)
                record_id = f"{wrapper}:{language}:{index}"
                rows.append(
                    {
                        "record_id": record_id,
                        "root_id": 4101,
                        "phase": "S",
                        "stage": "T3",
                        "branch": None,
                        "checkpoint_sha256": "1" * 64,
                        "model_fingerprint": "2" * 64,
                        "freeze_sha256": "3" * 64,
                        "config_sha256": "4" * 64,
                        "tokenizer_sha256": "5" * 64,
                        "code_sha256": "6" * 64,
                        "evaluation_plan_sha256": "7" * 64,
                        "evaluation_record_hash_schema": "score-record-v1",
                        "data_kind": "SYNTHETIC_TEST_FIXTURE",
                        "input_language": "ko",
                        "format": "RD",
                        "split": "dev",
                        "wrapper": wrapper,
                        "mode": "REQUESTED",
                        "requested_language": language,
                        "generation": {
                            "request_compatible": compatible,
                            "class": "REGISTERED_COMPATIBLE" if compatible else "UNREGISTERED",
                        },
                    }
                )
    return rows


def readiness_plan(rows):
    return {
        "root_id": 4101,
        "phase": "S",
        "stage": "T3",
        "branch": None,
        "checkpoint_sha256": "1" * 64,
        "model_fingerprint": "2" * 64,
        "freeze_sha256": "3" * 64,
        "config_sha256": "4" * 64,
        "tokenizer_sha256": "5" * 64,
        "code_sha256": "6" * 64,
        "evaluation_plan_sha256": "7" * 64,
        "evaluation_record_hash_schema": "score-record-v1",
        "data_kind": "SYNTHETIC_TEST_FIXTURE",
        "input_language": "ko",
        "format": "RD",
        "split": "dev",
        "wrappers": ["dev1", "dev2"],
        "active_languages": ["ko", "en", "zh", "fr"],
        "gate_each_wrapper_and_requested_language": True,
        "minimum_unrounded": 0.9,
        "expected_record_ids": [row["record_id"] for row in rows],
    }


def measurement_rows(n=20, *, low_z=0, constant_ko=False):
    rows = []
    for wrapper in ("dev1", "dev2"):
        for index in range(n):
            ko = 0.1 if constant_ko else 0.1 + 0.001 * index
            q = {
                "ko": ko,
                "en": 0.2 + 0.002 * index,
                "zh": 0.3 - 0.001 * index,
            }
            q["fr"] = 1.0 - math.fsum(q.values())
            z_value = 0.01 if index < low_z else 0.05
            rows.append(
                {
                    "record_id": f"{wrapper}:c{index}",
                    "concept_id": f"c{index}",
                    "root_id": 4101,
                    "phase": "S",
                    "checkpoint_sha256": HASH,
                    "branch": None,
                    "model_fingerprint": "2" * 64,
                    "freeze_sha256": "3" * 64,
                    "config_sha256": "4" * 64,
                    "tokenizer_sha256": "5" * 64,
                    "code_sha256": "6" * 64,
                    "evaluation_plan_sha256": "7" * 64,
                    "evaluation_record_hash_schema": "score-record-v1",
                    "data_kind": "SYNTHETIC_TEST_FIXTURE",
                    "stage": "T3",
                    "input_language": "ko",
                    "format": "RD",
                    "mode": "ANY",
                    "split": "dev",
                    "wrapper": wrapper,
                    "probability": {
                        "identifiable": True,
                        "status": "DEFINED",
                        "logZ_status": "FINITE",
                        "logZ": math.log(z_value),
                        "Q_language": q,
                        "Z": z_value,
                    },
                }
            )
    return rows


def measurement_plan(n=20):
    rows = measurement_rows(n)
    return {
        "root_id": 4101,
        "phase": "S",
        "checkpoint_sha256": HASH,
        "branch": None,
        "model_fingerprint": "2" * 64,
        "freeze_sha256": "3" * 64,
        "config_sha256": "4" * 64,
        "tokenizer_sha256": "5" * 64,
        "code_sha256": "6" * 64,
        "evaluation_plan_sha256": "7" * 64,
        "evaluation_record_hash_schema": "score-record-v1",
        "data_kind": "SYNTHETIC_TEST_FIXTURE",
        "stage": "T3",
        "input_language": "ko",
        "format": "RD",
        "mode": "ANY",
        "split": "dev",
        "wrappers": ["dev1", "dev2"],
        "identifiable_concept_ids": [f"c{index}" for index in range(n)],
        "expected_record_ids": [row["record_id"] for row in rows],
        "rho_min_unrounded": 0.6,
        "mean_abs_wrapper_difference_max_unrounded": 0.15,
        "low_z_threshold_strict_less_than": 0.05,
        "low_z_fraction_block_at_or_above": 0.1,
    }


class GateTests(unittest.TestCase):
    def test_readiness_gates_every_wrapper_language_not_mean(self):
        rows = readiness_rows(bad_cell=("dev2", "fr"))
        result = compute_readiness(rows, readiness_plan(rows))
        self.assertEqual(result["status"], "BLOCKED_READINESS")
        self.assertEqual(result["cells"]["dev2:fr"]["A"], 0.8)
        self.assertGreater(
            sum(cell["A"] for cell in result["cells"].values())
            / len(result["cells"]),
            0.9,
        )
        self.assertFalse(result["overall_average_used_for_gate"])

    def test_readiness_exact_threshold_passes_and_missing_row_rejects(self):
        rows = readiness_rows()
        # Make every cell exactly 9/10.
        for index, row in enumerate(rows):
            if index % 10 == 9:
                row["generation"] = {
                    "request_compatible": False,
                    "class": "UNREGISTERED",
                }
        plan = readiness_plan(rows)
        self.assertEqual(compute_readiness(rows, plan)["status"], "PASS")
        with self.assertRaisesRegex(ContractViolation, "MISSING_RESULT_RECORD_IDS"):
            compute_readiness(rows[:-1], plan)

    def test_readiness_rejects_inconsistent_class_and_boolean(self):
        rows = readiness_rows()
        rows[0]["generation"] = {
            "request_compatible": True,
            "class": "UNREGISTERED",
        }
        with self.assertRaisesRegex(ContractViolation, "INCONSISTENT"):
            compute_readiness(rows, readiness_plan(rows))

    def test_measurement_is_order_invariant(self):
        rows = measurement_rows()
        plan = measurement_plan()
        forward = compute_measurement(rows, plan)
        reverse = compute_measurement(list(reversed(rows)), plan)
        self.assertEqual(forward, reverse)
        self.assertEqual(forward["status"], "PASS")
        # Z equal to the strict boundary is not low.
        self.assertEqual(forward["Z_by_wrapper"]["dev1"]["low_Z_fraction"], 0.0)

    def test_exact_ten_percent_low_z_blocks(self):
        result = compute_measurement(measurement_rows(low_z=2), measurement_plan())
        self.assertEqual(result["status"], "BLOCKED_MEASUREMENT")
        self.assertEqual(result["Z_by_wrapper"]["dev1"]["low_Z_fraction"], 0.1)

    def test_constant_q_blocks_and_shared_primary_row_rejects(self):
        constant = compute_measurement(
            measurement_rows(constant_ko=True), measurement_plan()
        )
        self.assertEqual(constant["status"], "BLOCKED_MEASUREMENT")
        self.assertIsNone(constant["metrics_by_language"]["ko"]["spearman"])
        rows = measurement_rows()
        rows[0]["probability"]["identifiable"] = False
        rows[0]["probability"]["Q_language"] = None
        with self.assertRaisesRegex(ContractViolation, "NONIDENTIFIABLE"):
            compute_measurement(rows, measurement_plan())

    def test_mixed_checkpoint_and_duplicate_concept_cell_reject(self):
        rows = measurement_rows()
        rows[0]["checkpoint_sha256"] = "b" * 64
        with self.assertRaisesRegex(ContractViolation, "MIXED_CHECKPOINT"):
            compute_measurement(rows, measurement_plan())
        rows = measurement_rows()
        rows[1]["concept_id"] = rows[0]["concept_id"]
        with self.assertRaises(ContractViolation):
            compute_measurement(rows, measurement_plan())

    def test_mixed_evaluation_plan_hash_rejects(self):
        rows = measurement_rows()
        rows[0]["evaluation_plan_sha256"] = "f" * 64
        with self.assertRaisesRegex(ContractViolation, "MIXED_EVALUATION_PLAN"):
            compute_measurement(rows, measurement_plan())

    def test_measurement_requires_frozen_ids_and_any_has_no_requested_language(self):
        rows = measurement_rows()
        rows[0]["record_id"] = "unfrozen-arbitrary-id"
        with self.assertRaisesRegex(ContractViolation, "UNEXPECTED_RESULT_RECORD_ID"):
            compute_measurement(rows, measurement_plan())
        rows = measurement_rows()
        rows[0]["requested_language"] = "ko"
        with self.assertRaisesRegex(ContractViolation, "ANY_WITHOUT_REQUESTED_LANGUAGE"):
            compute_measurement(rows, measurement_plan())

    def test_phase_authorization_requires_every_planned_root(self):
        base = {
            name: "PASS"
            for name in (
                "delivery_integrity",
                "data_qa",
                "freeze",
                "tokenizer_boundary",
                "cpu_integration",
                "gpu_replay",
                "budget",
            )
        }
        roots = {
            root: {"readiness": "PASS", "measurement": "PASS"}
            for root in (4101, 4102, 4103)
        }
        blocked = authorize_phase("H", base, roots)
        self.assertFalse(blocked["authorized"])
        self.assertIn("root_4104:READINESS", blocked["reasons"])
        self.assertFalse(authorize_phase("MAIN", base, roots)["authorized"])


def summary_payload():
    return {
        "schema_version": "run-summary.v1",
        "project_id": "lexical-freshstart-4lang-ety-v4",
        "spec_version": "4.0.0",
        "run_id": "run-test-001",
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
            "implementation": "NOT_RUN",
            "gpu_training": "NOT_RUN",
            "measurement": "BLOCKED_MEASUREMENT",
        },
        "gates": [
            {
                "name": "measurement",
                "status": "BLOCKED_MEASUREMENT",
                "reasons": ["ko:RHO_BELOW_MINIMUM"],
                "evidence": {"rho": 0.59996, "display_rounding": 0.600},
            }
        ],
        "budget": {
            "campaign_gpu_hours_cap": 24.0,
            "prior_gpu_hours_user_reported": 0.953,
            "measured_gpu_hours": 0.0,
        },
        "roots": {},
        "artifacts": {},
        "next_action": "Fix or diagnose the measurement failure; do not expand H.",
    }


class RunReportTests(unittest.TestCase):
    def test_blocked_summary_requires_explicit_data_kind_and_artifacts(self):
        payload = summary_payload()
        del payload["data_kind"]
        with self.assertRaisesRegex(ContractViolation, "MISSING_RUN_SUMMARY_FIELD:data_kind"):
            validate_run_summary(payload)
        payload = summary_payload()
        del payload["artifacts"]
        with self.assertRaisesRegex(ContractViolation, "MISSING_RUN_SUMMARY_FIELD:artifacts"):
            validate_run_summary(payload)

    def test_synthetic_blocked_summary_cannot_carry_any_pass_claim(self):
        payload = summary_payload()
        payload["statuses"]["implementation"] = "PASS"
        with self.assertRaisesRegex(ContractViolation, "FIXTURE_CANNOT_PASS"):
            validate_run_summary(payload)
        payload = summary_payload()
        payload["gates"].append(
            {
                "name": "pilot_training",
                "status": "PASS",
                "reasons": [],
                "evidence": {"claimed": True},
            }
        )
        with self.assertRaisesRegex(ContractViolation, "FIXTURE_CANNOT_PASS"):
            validate_run_summary(payload)

    def test_real_blocked_summary_rejects_unsupported_pass_gate(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "work") as directory:
            implementation_ref = publish_json_once(
                Path(directory) / "implementation.json", {"placeholder": True}
            )
            payload = summary_payload()
            payload["data_kind"] = "REAL_INPUTS_UNREVIEWED"
            payload["provenance"] = {
                "spec_sha256": sha256_file(PROJECT_ROOT / "spec/RESEARCH_SPEC_V4_KO.md"),
                "config_sha256": sha256_file(PROJECT_ROOT / "spec/pilot.json"),
                "code_sha256": "a" * 64,
            }
            payload["artifacts"] = {"implementation_manifest": implementation_ref}
            payload["statuses"]["source_collection"] = "PASS"
            payload["gates"].append(
                {
                    "name": "source_collection",
                    "status": "PASS",
                    "reasons": [],
                    "evidence": {"claimed_records": 597},
                }
            )
            with mock.patch(
                "implementation.src.integrity.verify_implementation_manifest",
                return_value={"implementation_manifest_sha256": "a" * 64},
            ):
                with self.assertRaisesRegex(
                    ContractViolation,
                    "RUN_PASS_GATE_EVIDENCE_REQUIRED:source_collection",
                ):
                    validate_run_summary(payload)

            for gate_name in ("budget", "pilot_training", "readiness", "measurement"):
                payload = summary_payload()
                payload["data_kind"] = "REAL"
                payload["provenance"] = {
                    "spec_sha256": sha256_file(
                        PROJECT_ROOT / "spec/RESEARCH_SPEC_V4_KO.md"
                    ),
                    "config_sha256": sha256_file(PROJECT_ROOT / "spec/pilot.json"),
                    "code_sha256": "a" * 64,
                }
                payload["artifacts"] = {
                    "implementation_manifest": implementation_ref
                }
                payload["status"] = "BLOCKED"
                payload["statuses"] = {"blocker": "BLOCKED", gate_name: "PASS"}
                payload["gates"] = [
                    {
                        "name": gate_name,
                        "status": "PASS",
                        "reasons": [],
                        "evidence": {"claimed": True},
                    }
                ]
                with mock.patch(
                    "implementation.src.integrity.verify_implementation_manifest",
                    return_value={"implementation_manifest_sha256": "a" * 64},
                ):
                    with self.assertRaisesRegex(
                        ContractViolation,
                        "RUN_PASS_GATE_EVIDENCE_NOT_IMPLEMENTED:" + gate_name,
                    ):
                        validate_run_summary(payload)

            payload = summary_payload()
            payload["data_kind"] = "REAL"
            payload["provenance"] = {
                "spec_sha256": sha256_file(
                    PROJECT_ROOT / "spec/RESEARCH_SPEC_V4_KO.md"
                ),
                "config_sha256": sha256_file(PROJECT_ROOT / "spec/pilot.json"),
                "code_sha256": "a" * 64,
            }
            payload["artifacts"] = {"implementation_manifest": implementation_ref}
            payload["roots"] = {
                "4101": {
                    "training": "PASS",
                    "readiness": "NOT_RUN",
                    "measurement": "NOT_RUN",
                }
            }
            with mock.patch(
                "implementation.src.integrity.verify_implementation_manifest",
                return_value={"implementation_manifest_sha256": "a" * 64},
            ):
                with self.assertRaisesRegex(
                    ContractViolation, "RUN_ROOT_PASS_EVIDENCE_NOT_IMPLEMENTED"
                ):
                    validate_run_summary(payload)

    def test_gpu_replay_rejects_cross_binding_and_forged_checkpoint(self):
        freeze_sha256 = "1" * 64
        code_sha256 = "2" * 64
        cpu_check_sha256 = "3" * 64
        gpu_binding = {
            "physical_index": 2,
            "logical_index": 0,
            "uuid": "GPU-de18b86c-419a-795a-0667-c7ae03bf800f",
            "pci_bus_id": "00000000:01:00.0",
            "name": "test-gpu",
            "total_memory_mib": 8192,
            "torch_cuda_version": "12.1",
            "torch_version": "2.0",
        }
        traces = [
            {
                "global_step": index + 3,
                "phase_step": index + 3,
                "loss_hex": float(index).hex(),
                "record_ids": [f"record-{index}-{cell}" for cell in range(32)],
                "content_sha256s": [
                    sha256_bytes(f"{index}-{cell}".encode("ascii"))
                    for cell in range(32)
                ],
                "learning_rates": [0.0001],
            }
            for index in range(10)
        ]
        comparison = {
            "continuous_vs_resume_1": True,
            "resume_1_vs_resume_2": True,
            "steps_compared": 10,
            "traces": traces,
            "loss_trace_sha256": sha256_bytes(
                canonical_json_bytes([row["loss_hex"] for row in traces])
            ),
            "record_trace_sha256": sha256_bytes(
                canonical_json_bytes(
                    [
                        {
                            "record_ids": row["record_ids"],
                            "content_sha256s": row["content_sha256s"],
                        }
                        for row in traces
                    ]
                )
            ),
            "final_state_fingerprints": {
                "model": "4" * 64,
                "optimizer": "5" * 64,
                "scheduler": "6" * 64,
                "progress": "7" * 64,
                "rng": "8" * 64,
            },
        }
        comparison["resume_traces"] = [traces, traces]
        comparison["resume_final_state_fingerprints"] = [
            comparison["final_state_fingerprints"],
            comparison["final_state_fingerprints"],
        ]
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "work") as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            state_ref = publish_bytes_once(checkpoint / "state.pt", b"checkpoint")
            manifest = {
                "schema_version": "v4-full-state-1",
                "state_file": "state.pt",
                "state_sha256": state_ref["sha256"],
                "state_bytes": state_ref["bytes"],
                "model_fingerprint": "5" * 64,
                "lineage": {
                    "phase": "GPU_REPLAY",
                    "stage": "GPU_REPLAY",
                    "branch": None,
                    "data_kind": "REAL",
                    "freeze_gate_status": "PASS",
                    "scientific_result": False,
                    "freeze_sha256": freeze_sha256,
                    "code_sha256": code_sha256,
                    "cpu_check_sha256": cpu_check_sha256,
                    "gpu_binding": gpu_binding,
                },
            }
            manifest_ref = publish_json_once(checkpoint / "manifest.json", manifest)
            publish_bytes_once(
                checkpoint / "COMMITTED",
                (manifest_ref["sha256"] + "\n").encode("ascii"),
            )
            replay = {
                "schema_version": "gpu2-replay-v1",
                "status": "PASS",
                "scientific_result": False,
                "freeze_sha256": freeze_sha256,
                "code_sha256": code_sha256,
                "cpu_check_sha256": cpu_check_sha256,
                "gpu_binding": gpu_binding,
                "checkpoint": {
                    "path": str(checkpoint),
                    "state_sha256": state_ref["sha256"],
                    "manifest_sha256": manifest_ref["sha256"],
                    "model_fingerprint": "5" * 64,
                    "optimizer_fingerprint": "6" * 64,
                    "bytes": state_ref["bytes"],
                },
                "comparison": comparison,
                "optimizer_steps_charged": 32,
                "gpu_seconds_charged": 0.5,
                "budget": {
                    "campaign_cap_gpu_hours": 24.0,
                    "prior_gpu_hours_user_reported": 0.953,
                    "new_gpu_seconds_charged_or_reserved": 0.5,
                    "campaign_gpu_hours_charged_or_reserved": 0.953
                    + 0.5 / 3600.0,
                    "campaign_gpu_hours_remaining": 24.0
                    - (0.953 + 0.5 / 3600.0),
                    "per_root_gpu_seconds_charged_or_reserved": {},
                    "ledger_events": 2,
                    "ledger_tip_sha256": "7" * 64,
                },
                "disk_reservation": {
                    "available_bytes": 7 * 1024**3,
                    "planned_bytes": 2 * 1024**3,
                    "emergency_free_bytes": 5 * 1024**3,
                    "projected_free_bytes": 5 * 1024**3,
                },
            }
            good_path = root / "replay-good.json"
            publish_json_once(good_path, replay)
            with self.assertRaisesRegex(
                ContractViolation, "RUN_GPU_REPLAY_CHECKPOINT_INVALID"
            ):
                _verify_gpu_replay_evidence(
                    good_path,
                    freeze_sha256=freeze_sha256,
                    code_sha256=code_sha256,
                    cpu_check_sha256=cpu_check_sha256,
                )

            for field, wrong in (
                ("freeze_sha256", "a" * 64),
                ("code_sha256", "b" * 64),
                ("cpu_check_sha256", "c" * 64),
            ):
                bad = dict(replay, **{field: wrong})
                bad_path = root / f"replay-bad-{field}.json"
                publish_json_once(bad_path, bad)
                with self.assertRaisesRegex(
                    ContractViolation, "RUN_GPU_REPLAY_EVIDENCE_NOT_PASS"
                ):
                    _verify_gpu_replay_evidence(
                        bad_path,
                        freeze_sha256=freeze_sha256,
                        code_sha256=code_sha256,
                        cpu_check_sha256=cpu_check_sha256,
                    )

            bad_uuid = dict(replay, gpu_binding={**gpu_binding, "uuid": "GPU-wrong"})
            bad_uuid_path = root / "replay-bad-uuid.json"
            publish_json_once(bad_uuid_path, bad_uuid)
            with self.assertRaisesRegex(
                ContractViolation, "RUN_GPU_REPLAY_EVIDENCE_NOT_PASS"
            ):
                _verify_gpu_replay_evidence(
                    bad_uuid_path,
                    freeze_sha256=freeze_sha256,
                    code_sha256=code_sha256,
                    cpu_check_sha256=cpu_check_sha256,
                )
    def test_markdown_is_projection_of_write_once_json(self):
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "run_summary.json"
            markdown_path = Path(directory) / "run_summary.md"
            publish_run_summary(json_path, summary_payload())
            first = render_markdown(json_path, markdown_path)
            first_text = markdown_path.read_text(encoding="utf-8")
            second = render_markdown(json_path, markdown_path)
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(first_text, markdown_path.read_text(encoding="utf-8"))
            self.assertIn("<strong>BLOCKED_MEASUREMENT</strong>", first_text)
            self.assertIn("0.59996", first_text)
            self.assertIn(sha256_file(json_path), first_text)
            with self.assertRaisesRegex(ContractViolation, "ARTIFACT_EXISTS"):
                publish_run_summary(json_path, summary_payload())

    def test_render_reverifies_external_real_completion_artifacts(self):
        payload = summary_payload()
        payload.update(status="PILOT_COMPLETE", data_kind="REAL")
        payload["statuses"] = {"implementation": "PASS"}
        payload["gates"] = [
            {
                "name": name,
                "status": "PASS",
                "reasons": [],
                "evidence": {"claimed": True},
            }
            for name in COMPLETION_REQUIRED_GATES
        ]
        payload["roots"] = {
            str(root_id): {
                "training": "PASS",
                "readiness": "PASS",
                "measurement": "PASS",
            }
            for root_id in (4101, 4102, 4103, 4104)
        }

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "work") as directory:
            root = Path(directory)
            artifact_path = root / "artifact.json"
            publish_json_once(artifact_path, {"forged": False})
            cases = (
                ("forged", artifact_path, "RUN_ARTIFACT_HASH_MISMATCH"),
                ("missing", root / "missing.json", "RUN_ARTIFACT_OUTSIDE_PROJECT"),
            )
            for label, claimed_path, error in cases:
                with self.subTest(label=label):
                    candidate = dict(payload)
                    candidate["artifacts"] = {
                        name: {"path": str(claimed_path), "sha256": "0" * 64}
                        for name in COMPLETION_REQUIRED_ARTIFACTS
                    }
                    json_path = root / f"external-{label}.json"
                    markdown_path = root / f"external-{label}.md"
                    publish_json_once(json_path, candidate)
                    with self.assertRaisesRegex(ContractViolation, error):
                        render_markdown(json_path, markdown_path)
                    self.assertFalse(markdown_path.exists())

    def test_fixture_cannot_claim_production_completion(self):
        payload = summary_payload()
        payload["data_kind"] = "SYNTHETIC_TEST_FIXTURE"
        payload["status"] = "PILOT_COMPLETE"
        with self.assertRaisesRegex(ContractViolation, "FIXTURE_CANNOT"):
            validate_run_summary(payload)

    def test_forged_completion_and_malformed_identity_are_rejected(self):
        payload = summary_payload()
        payload.update(
            schema_version="anything",
            run_id="",
            created_at_utc="not-a-time",
            status="PILOT_COMPLETE",
        )
        payload["gates"] = []
        with self.assertRaisesRegex(ContractViolation, "SCHEMA_MISMATCH"):
            validate_run_summary(payload)
        payload = summary_payload()
        payload["project_id"] = "other-project"
        with self.assertRaisesRegex(ContractViolation, "PROJECT_MISMATCH"):
            validate_run_summary(payload)
        payload = summary_payload()
        payload["created_at_utc"] = "not-a-time"
        with self.assertRaisesRegex(ContractViolation, "CREATED_AT"):
            validate_run_summary(payload)
        payload = summary_payload()
        payload["status"] = "PILOT_COMPLETE"
        payload["data_kind"] = "REAL"
        with self.assertRaisesRegex(ContractViolation, "COMPLETION_GATE_NOT_PASS"):
            validate_run_summary(payload)

    def test_status_artifact_and_markdown_injection_are_rejected_or_escaped(self):
        payload = summary_payload()
        payload["status"] = "BLOCKED_DATA_QA"
        with self.assertRaisesRegex(ContractViolation, "OVERALL_STATUS_INCONSISTENT"):
            validate_run_summary(payload)
        payload = summary_payload()
        payload["artifacts"] = {
            "spec": {
                "path": "spec/RESEARCH_SPEC_V4_KO.md",
                "sha256": "0" * 64,
            }
        }
        with self.assertRaisesRegex(ContractViolation, "ARTIFACT_HASH_MISMATCH"):
            validate_run_summary(payload)
        payload = summary_payload()
        payload["next_action"] = "<script>alert(1)</script> **bold** [go](javascript:alert(2))"
        payload["gates"][0]["reasons"] = ["<img src=x onerror=alert(3)>"]
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "safe.json"
            markdown_path = Path(directory) / "safe.md"
            publish_run_summary(json_path, payload)
            render_markdown(json_path, markdown_path)
            rendered = markdown_path.read_text(encoding="utf-8")
        projected = rendered.split("## Exact source values", 1)[0]
        self.assertNotIn("<script>", projected)
        self.assertNotIn("<img ", projected)
        self.assertNotIn("[go](javascript:", projected)
        self.assertIn("&lt;script&gt;", projected)


if __name__ == "__main__":
    unittest.main()
