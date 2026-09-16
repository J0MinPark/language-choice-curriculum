from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import implementation.src.build_tokenizer as build_tokenizer_module
from implementation.src.artifacts import publish_bytes_once, publish_json_once
from implementation.src.build_tokenizer import (
    EXPRESSION_LANGUAGE_PAIRS,
    audit_lexical_token_boundaries,
    expression_only_similarity_metrics,
    iter_staged_documents,
    load_corpus_token_memmap,
    load_verified_tokenizer,
    materialize_corpus_tokens,
    train_byte_bpe,
    verify_corpus_manifest,
    verify_tokenizer_manifest,
)
from implementation.src.contracts import (
    PROJECT_ROOT,
    ContractViolation,
    canonical_json_bytes,
)
from implementation.src.prepare_data import (
    SYNTHETIC,
    build_krdict_collection_plan,
    collect_krdict,
    export_pending_review,
    freeze_reviewed_dataset,
    import_existing_krdict_collection,
    merge_krdict_snapshot,
    stage_wiki40b_documents,
    verify_annotation_freeze,
    verify_krdict_collection,
    verify_wiki40b_snapshot,
)


def logical_hash(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def tiny_campaign_policy():
    return {
        "version": "4.0.0-r1",
        "budget": {
            "api_requests_cap": 6,
            "api_requests_already_attempted": 3,
        },
        "collection_revision": {
            "selection_rule": "first complete triplets",
            "selected_query_count": 1,
            "planned_requests": 3,
        },
    }


def krdict_xml(language_name: str, translated_word: str) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<channel>
  <total>1</total><start>1</start><num>100</num>
  <item><target_code>100</target_code><word>시험</word><pos>명사</pos>
    <link>https://krdict.korean.go.kr/kor/dicSearch/SearchView?ParaWordNo=100</link>
    <sense><sense_order>1</sense_order><definition>능력을 알아보기 위한 절차.</definition>
      <translation><trans_lang>\n {language_name} \n</trans_lang>
        <trans_word>{translated_word}</trans_word><trans_dfn>translated definition</trans_dfn>
      </translation>
    </sense>
  </item>
</channel>
""".encode("utf-8")


def make_legacy_fixture(root: Path, *, mutate_first_params: bool = False):
    policy = tiny_campaign_policy()
    plan = build_krdict_collection_plan(["시험", "제외"], campaign_policy=policy)
    legacy_dir = root / "legacy"
    legacy_dir.mkdir()
    values = {
        "en": ("영어", "test"),
        "zh": ("중국어", "测试"),
        "fr": ("프랑스어", "test français"),
    }
    records = []
    for request in plan["requests"]:
        language = request["language"]
        payload = krdict_xml(*values[language])
        raw_name = f"0000_{language}.xml"
        ref = publish_bytes_once(legacy_dir / raw_name, payload)
        params = dict(request["params_without_key"])
        if mutate_first_params and language == "en":
            params["num"] = "99"
        records.append(
            {
                "query_index": 0,
                "language": language,
                "endpoint": request["endpoint"],
                "params_without_key": params,
                "status": "PAYLOAD_SCHEMA_CHECKED",
                "raw_file": raw_name,
                "sha256": ref["sha256"],
                "bytes": ref["bytes"],
                "summary": {},
            }
        )
    legacy_path = legacy_dir / "source_manifest.json"
    publish_json_once(
        legacy_path,
        {
            "status": "COLLECTED_UNREVIEWED",
            "requests_made": 3,
            "records": records,
        },
    )
    return plan, policy, legacy_path


class CollectionImportTests(unittest.TestCase):
    def test_frozen_campaign_plan_is_first_199_triplets(self):
        plan = build_krdict_collection_plan(PROJECT_ROOT / "templates/queries.txt")
        self.assertEqual(plan["consumed_before_plan"], 3)
        self.assertEqual(plan["selected_query_count"], 199)
        self.assertEqual(plan["planned_requests"], 597)
        self.assertEqual(plan["remaining_after_plan"], 0)
        self.assertEqual(
            list(plan["requests"][0]["params_without_key"]),
            [
                "q",
                "translated",
                "trans_lang",
                "advanced",
                "method",
                "pos",
                "part",
                "num",
                "start",
            ],
        )
        self.assertEqual(len(plan["excluded_queries"]), 1)

    def test_closed_real_collector_never_reads_key_or_calls_transport(self):
        calls = {"credential": 0, "transport": 0}

        def credential():
            calls["credential"] += 1
            return "a" * 32

        def transport(*args):
            del args
            calls["transport"] += 1
            raise AssertionError("must not execute")

        result = collect_krdict(
            {},
            output_dir=Path("unused"),
            ledger_path=Path("unused-ledger"),
            credential_provider=credential,
            transport=transport,
        )
        self.assertEqual(result["status"], "BLOCKED_CAMPAIGN_EXHAUSTED")
        self.assertEqual(result["network_requests"], 0)
        self.assertEqual(calls, {"credential": 0, "transport": 0})

    def test_offline_import_reparses_binds_and_merges_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan, policy, legacy = make_legacy_fixture(root)
            output = root / "adapted"
            imported = import_existing_krdict_collection(
                legacy,
                plan,
                output_dir=output,
                campaign_policy=policy,
                production=False,
                scope_root=root,
            )
            self.assertEqual(imported["network_requests"], 0)
            self.assertEqual(imported["campaign_requests_consumed"], 6)
            verified = verify_krdict_collection(
                output / "source_manifest.json", production=False
            )
            self.assertEqual(len(verified["verified_records"]), 3)
            merged = merge_krdict_snapshot(
                output / "source_manifest.json", production=False
            )
            self.assertEqual(merged["summary"]["eligible"], 1)
            self.assertEqual(merged["summary"]["quarantined"], 0)
            pending = export_pending_review(merged)
            self.assertEqual(pending["summary"]["pending"], 1)
            self.assertEqual(pending["summary"]["automatically_approved"], 0)
            self.assertEqual(
                set(pending["pending_review"][0]["answer_options"]),
                {"ko", "en", "zh", "fr"},
            )
            with self.assertRaisesRegex(
                ContractViolation, "SYNTHETIC_SOURCE_NOT_PRODUCTION"
            ):
                verify_krdict_collection(
                    output / "source_manifest.json", production=True
                )

    def test_offline_import_rejects_any_request_parameter_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan, policy, legacy = make_legacy_fixture(
                root, mutate_first_params=True
            )
            with self.assertRaisesRegex(
                ContractViolation, "LEGACY_PLAN_BINDING_MISMATCH"
            ):
                import_existing_krdict_collection(
                    legacy,
                    plan,
                    output_dir=root / "adapted",
                    campaign_policy=policy,
                    production=False,
                    scope_root=root,
                )

    def test_synthetic_reviews_freeze_hash_bound_history_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            policy = {
                "version": "4.0.0-r1",
                "budget": {
                    "api_requests_cap": 9,
                    "api_requests_already_attempted": 3,
                },
                "collection_revision": {
                    "selection_rule": "first complete triplets",
                    "selected_query_count": 2,
                    "planned_requests": 6,
                },
            }
            plan = build_krdict_collection_plan(
                ["시험", "평가"], campaign_policy=policy
            )
            legacy_dir = root / "legacy"
            legacy_dir.mkdir()
            translated = {
                "en": ("영어", ["test", "assessment"]),
                "zh": ("중국어", ["测试", "评价"]),
                "fr": ("프랑스어", ["essai", "évaluation"]),
            }
            records = []
            for request in plan["requests"]:
                index = request["query_index"]
                language = request["language"]
                language_name, words = translated[language]
                payload = krdict_xml(language_name, words[index])
                payload = payload.replace("시험".encode(), request["query"].encode())
                payload = payload.replace(
                    b"<target_code>100</target_code>",
                    f"<target_code>{100 + index}</target_code>".encode(),
                )
                raw_name = f"{index:04d}_{language}.xml"
                ref = publish_bytes_once(legacy_dir / raw_name, payload)
                records.append(
                    {
                        "query_index": index,
                        "language": language,
                        "endpoint": request["endpoint"],
                        "params_without_key": request["params_without_key"],
                        "status": "PAYLOAD_SCHEMA_CHECKED",
                        "raw_file": raw_name,
                        "sha256": ref["sha256"],
                        "bytes": ref["bytes"],
                        "summary": {},
                    }
                )
            legacy_path = legacy_dir / "source_manifest.json"
            publish_json_once(
                legacy_path,
                {
                    "status": "COLLECTED_UNREVIEWED",
                    "requests_made": 6,
                    "records": records,
                },
            )
            adapted = root / "adapted"
            import_existing_krdict_collection(
                legacy_path,
                plan,
                output_dir=adapted,
                campaign_policy=policy,
                production=False,
                scope_root=root,
            )
            merge = merge_krdict_snapshot(
                adapted / "source_manifest.json", production=False
            )
            concept_reviews = []
            etymology_reviews = []
            for index, candidate in enumerate(merge["eligible_candidates"]):
                selections = {
                    language: {
                        "option_id": candidate["options"][language][0]["option_id"],
                        "answer": candidate["options"][language][0]["answer"],
                    }
                    for language in ("ko", "en", "zh", "fr")
                }
                concept_reviews.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "candidate_sha256": candidate["candidate_sha256"],
                        "synthetic_fixture": True,
                        "selections": selections,
                        "synonym_cluster_id": f"fixture-synonym-{index}",
                        "qa": {
                            "status": "APPROVED_BY_RESEARCHER",
                            "reviewer": "synthetic-test-reviewer",
                            "review_date": "2026-01-01",
                            "source_alignment_checked": True,
                            "answer_copy_checked": True,
                            "meaning_alignment_note": "Synthetic structural test only.",
                        },
                    }
                )
                answer_pair_hash = logical_hash(
                    {
                        "en": selections["en"]["answer"],
                        "fr": selections["fr"]["answer"],
                    }
                )
                etymology_reviews.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "candidate_sha256": candidate["candidate_sha256"],
                        "synthetic_fixture": True,
                        "answer_pair_sha256": answer_pair_hash,
                        "pair": ["en", "fr"],
                        "relation": (
                            "BORROWING_DOCUMENTED"
                            if index == 0
                            else "DISTINCT_ROUTES_REVIEWED"
                        ),
                        "family_id": f"fixture-family-{index}",
                        "sense_alignment_note": "Synthetic same-sense check.",
                        "historical_scope": "Synthetic evidence scope.",
                        "evidence": [
                            {
                                "source_url": f"https://example.invalid/evidence/{index}",
                                "record_sha256": str(index + 1) * 64,
                                "evidence_origin": "synthetic-human-reviewed-fixture",
                                "source_name": "Synthetic Evidence Dictionary",
                                "source_version": "fixture-v1",
                                "source_license": "Synthetic fixture license",
                                "source_license_url": "https://example.invalid/license",
                                "sense_locator": f"fixture-entry-{index}",
                            }
                        ],
                        "qa": {
                            "status": "APPROVED_BY_RESEARCHER",
                            "reviewer": "synthetic-test-reviewer",
                            "review_date": "2026-01-01",
                            "source_alignment_checked": True,
                            "answer_copy_checked": True,
                        },
                    }
                )
            frozen = freeze_reviewed_dataset(
                adapted / "source_manifest.json",
                concept_reviews,
                etymology_reviews,
                cohort_ids=[row["candidate_id"] for row in merge["eligible_candidates"]],
                output_dir=root / "annotation-freeze",
                requirements={
                    "min_total": 2,
                    "min_identifiable": 2,
                    "min_related": 1,
                    "min_distinct_routes": 1,
                    "requested_pilot_cohort_size": 2,
                },
                production=False,
                scope_root=root,
            )
            verified = verify_annotation_freeze(
                root / "annotation-freeze/annotation_freeze_manifest.json",
                production=False,
            )
            self.assertEqual(frozen["status"], "PASS")
            self.assertEqual(len(verified["concepts"]), 2)
            self.assertEqual(len(verified["history_pairs"]["pairs"]), 1)
            self.assertEqual(
                verified["history_pairs"]["concept_order"],
                sorted(verified["cohort"]["concept_ids"]),
            )
            with self.assertRaisesRegex(
                ContractViolation, "SYNTHETIC_FREEZE_NOT_PRODUCTION"
            ):
                verify_annotation_freeze(
                    root / "annotation-freeze/annotation_freeze_manifest.json",
                    production=True,
                )

    @unittest.skipUnless(
        (PROJECT_ROOT / "work/sources/krdict/hardened_v4r1_20260914T151550Z/source_manifest.json").is_file(),
        "real offline snapshot not present",
    )
    def test_real_offline_snapshot_verifies_and_merges(self):
        path = PROJECT_ROOT / "work/sources/krdict/hardened_v4r1_20260914T151550Z/source_manifest.json"
        verified = verify_krdict_collection(path, production=True)
        self.assertEqual(len(verified["verified_records"]), 597)
        self.assertEqual(verified["network_requests"], 0)
        merged = merge_krdict_snapshot(path, production=True)
        self.assertEqual(merged["summary"]["eligible"], 409)
        self.assertEqual(merged["summary"]["quarantined"], 56)


def synthetic_wiki_rows():
    common = "가나다라마바사아자차카타파하 한국어 말뭉치 byte level tokenizer "
    return [
        {
            "text": "  _START_ARTICLE_\n" + common * 12 + "\n_END_ARTICLE_  ",
            "version_id": "1",
            "wikidata_id": "Q3",
        },
        {
            "text": "둘째 문서\n" + common[::-1] * 12,
            "version_id": "2",
            "wikidata_id": "Q1",
        },
        {
            "text": "third document 0123456789 abcdefghijklmnopqrstuvwxyz " * 14,
            "version_id": "3",
            "wikidata_id": "Q2",
        },
        {
            "text": "혼합 문서 café naïve 中文 français English 한국어 " * 14,
            "version_id": "4",
            "wikidata_id": "Q4",
        },
    ]


def synthetic_wiki_metadata(count: int):
    return {
        "data_kind": SYNTHETIC,
        "dataset_name": "wiki40b",
        "config_name": "ko",
        "version": "1.3.0",
        "split": "train",
        "features": ["text", "version_id", "wikidata_id"],
        "num_examples": count,
        "license": "synthetic test only",
        "citation": "synthetic test only",
        "homepage": "https://example.invalid/synthetic",
        "tfds_version": "synthetic",
        "tensorflow_version": "synthetic",
        "downloaded_bytes": 0,
        "generated_bytes": 1,
        "source_files": [],
    }


def publish_mini_evaluation_plan(root: Path, *, vocab_size=320, token_cap=80):
    plan = json.loads(
        (PROJECT_ROOT / "implementation/config/evaluation_plan.json").read_text(
            encoding="utf-8"
        )
    )
    plan["tokenizer"]["vocab_size"] = vocab_size
    plan["tokenizer"]["min_frequency"] = 1
    plan["tokenizer"]["corpus_model_token_cap"] = token_cap
    path = root / "mini_evaluation_plan.json"
    publish_json_once(path, plan)
    return path


class _FixtureEncoding:
    def __init__(self, ids):
        self.ids = ids


class _ExpressionOnlyTokenizer:
    def __init__(self):
        self.calls = []
        self.special_ids = {
            "<|endoftext|>": 99,
            "<|pad|>": 100,
        }
        self.ids = {
            "é": [7],
            "ab": [1, 2],
            "ac": [1, 3],
            "ab ab": [1, 2, 2],
        }

    def token_to_id(self, token):
        return self.special_ids.get(token)

    def encode(self, text, *, add_special_tokens):
        self.calls.append((text, add_special_tokens))
        return _FixtureEncoding(self.ids[text])


class ExpressionOnlyMetricTests(unittest.TestCase):
    def test_nfc_lengths_six_pairs_and_expression_only_token_sets(self):
        tokenizer = _ExpressionOnlyTokenizer()
        result = expression_only_similarity_metrics(
            tokenizer,
            {
                "fr": "ab ab",
                "zh": "ac",
                "ko": " e\u0301 ",
                "en": "ab",
            },
        )

        self.assertEqual(result["schema_version"], "expression-only-lexical-metrics-v1")
        self.assertEqual(
            result["excluded_special_token_ids"],
            {"<|endoftext|>": 99, "<|pad|>": 100},
        )
        self.assertEqual(result["language_order"], ["ko", "en", "zh", "fr"])
        self.assertEqual(
            result["pair_order"],
            ["ko-en", "ko-zh", "ko-fr", "en-zh", "en-fr", "zh-fr"],
        )
        self.assertEqual(tuple(EXPRESSION_LANGUAGE_PAIRS), (
            ("ko", "en"),
            ("ko", "zh"),
            ("ko", "fr"),
            ("en", "zh"),
            ("en", "fr"),
            ("zh", "fr"),
        ))
        self.assertEqual(
            result["languages"],
            {
                "ko": {"nfc_char_length": 1, "token_length": 1},
                "en": {"nfc_char_length": 2, "token_length": 2},
                "zh": {"nfc_char_length": 2, "token_length": 2},
                "fr": {"nfc_char_length": 5, "token_length": 3},
            },
        )
        self.assertEqual(
            tokenizer.calls,
            [("é", False), ("ab", False), ("ac", False), ("ab ab", False)],
        )
        self.assertTrue(
            all("\n" not in text for text, _add_special_tokens in tokenizer.calls)
        )
        self.assertEqual(
            result["pairs"]["en-zh"],
            {
                "surface_levenshtein_similarity": 0.5,
                "token_set_jaccard": 1 / 3,
            },
        )
        self.assertEqual(result["pairs"]["ko-en"]["token_set_jaccard"], 0.0)
        self.assertEqual(
            result["pairs"]["en-fr"]["token_set_jaccard"], 1.0
        )
        self.assertEqual(
            result["pairs"]["en-fr"]["surface_levenshtein_similarity"], 0.4
        )

    def test_identical_canonical_expressions_have_unit_surface_similarity(self):
        result = expression_only_similarity_metrics(
            _ExpressionOnlyTokenizer(),
            {
                "ko": "ab",
                "en": " e\u0301 ",
                "zh": "ac",
                "fr": "é",
            },
        )

        self.assertEqual(
            result["pairs"]["en-fr"]["surface_levenshtein_similarity"],
            1.0,
        )

    def test_empty_expression_or_expression_tokens_fail_closed(self):
        tokenizer = _ExpressionOnlyTokenizer()
        with self.assertRaisesRegex(ContractViolation, "EMPTY_EXPRESSION"):
            expression_only_similarity_metrics(
                tokenizer,
                {"ko": " ", "en": "ab", "zh": "ac", "fr": "ab ab"},
            )

        tokenizer.ids["없음"] = []
        with self.assertRaisesRegex(ContractViolation, "EMPTY_EXPRESSION_TOKENS"):
            expression_only_similarity_metrics(
                tokenizer,
                {"ko": "없음", "en": "ab", "zh": "ac", "fr": "ab ab"},
            )

        class ImplicitSpecialTokenizer:
            def token_to_id(self, token):
                return {"<|endoftext|>": 99, "<|pad|>": 100}.get(token)

            def encode(self, text):
                return _FixtureEncoding([99, len(text), 100])

        with self.assertRaisesRegex(
            ContractViolation, "TOKENIZER_CANNOT_DISABLE_SPECIAL_TOKENS"
        ):
            expression_only_similarity_metrics(
                ImplicitSpecialTokenizer(),
                {"ko": "é", "en": "ab", "zh": "ac", "fr": "ab ab"},
            )

        tokenizer = _ExpressionOnlyTokenizer()
        tokenizer.ids["literal-special-id"] = [7, 100]
        with self.assertRaisesRegex(
            ContractViolation, "EXPRESSION_CONTAINS_RESERVED_SPECIAL_TOKEN"
        ):
            expression_only_similarity_metrics(
                tokenizer,
                {
                    "ko": "literal-special-id",
                    "en": "ab",
                    "zh": "ac",
                    "fr": "ab ab",
                },
            )

    def test_answer_mapping_order_does_not_change_metrics(self):
        first = {
            "ko": " e\u0301 ",
            "en": "ab",
            "zh": "ac",
            "fr": "ab ab",
        }
        second = {language: first[language] for language in reversed(tuple(first))}
        self.assertEqual(
            expression_only_similarity_metrics(_ExpressionOnlyTokenizer(), first),
            expression_only_similarity_metrics(_ExpressionOnlyTokenizer(), second),
        )


class WikiTokenizerTests(unittest.TestCase):
    def _build(self, root: Path):
        rows = synthetic_wiki_rows()
        metadata = synthetic_wiki_metadata(len(rows))
        first = stage_wiki40b_documents(
            rows,
            metadata,
            output_dir=root / "source-a",
            production=False,
            scope_root=root,
        )
        second = stage_wiki40b_documents(
            reversed(rows),
            metadata,
            output_dir=root / "source-b",
            production=False,
            scope_root=root,
        )
        plan_path = publish_mini_evaluation_plan(root)
        token_a = train_byte_bpe(
            root / "source-a/source_manifest.json",
            output_dir=root / "tokenizer-a",
            production=False,
            evaluation_plan_path=plan_path,
            scope_root=root,
        )
        token_b = train_byte_bpe(
            root / "source-b/source_manifest.json",
            output_dir=root / "tokenizer-b",
            production=False,
            evaluation_plan_path=plan_path,
            scope_root=root,
        )
        return rows, first, second, token_a, token_b, plan_path

    def test_exact_source_staging_and_bpe_are_order_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            rows, first, second, token_a, token_b, _ = self._build(root)
            self.assertEqual(
                first["source_text_normalization"],
                "NONE_EXACT_UTF8_PRESERVED_INCLUDING_WIKI40B_MARKERS",
            )
            self.assertEqual(
                first["artifacts"]["documents"]["sha256"],
                second["artifacts"]["documents"]["sha256"],
            )
            self.assertEqual(
                first["artifacts"]["index"]["sha256"],
                second["artifacts"]["index"]["sha256"],
            )
            staged = list(
                iter_staged_documents(
                    root / "source-a/source_manifest.json", production=False
                )
            )
            self.assertEqual(
                sorted(row["text"] for row in staged),
                sorted(row["text"] for row in rows),
            )
            self.assertTrue(any(row["text"].startswith("  _START") for row in staged))
            self.assertEqual(
                token_a["tokenizer_file_sha256"], token_b["tokenizer_file_sha256"]
            )
            verified = verify_tokenizer_manifest(
                root / "tokenizer-a/tokenizer_manifest.json", production=False
            )
            self.assertEqual(verified["vocab_size"], 320)
            self.assertEqual(verified["byte_alphabet_size"], 256)
            with self.assertRaisesRegex(
                ContractViolation, "SYNTHETIC_TOKENIZER_NOT_PRODUCTION"
            ):
                verify_tokenizer_manifest(
                    root / "tokenizer-a/tokenizer_manifest.json", production=True
                )

    def test_corpus_cap_reencoding_memmap_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, _, _, _, _, _ = self._build(root)
            manifest = materialize_corpus_tokens(
                root / "source-a/source_manifest.json",
                root / "tokenizer-a/tokenizer_manifest.json",
                output_dir=root / "corpus",
                production=False,
                scope_root=root,
            )
            self.assertEqual(manifest["materialized_token_count"], 80)
            self.assertGreater(manifest["discarded_token_count"], 0)
            verified = verify_corpus_manifest(
                root / "corpus/corpus_manifest.json", production=False
            )
            self.assertTrue(verified["verified_by_reencoding"])
            values, mapped = load_corpus_token_memmap(
                root / "corpus/corpus_manifest.json", production=False
            )
            self.assertEqual(values.shape, (80,))
            self.assertEqual(mapped["dtype"], "uint16-le")
            private_values = values.tolist()
            tokens_path = Path(verified["tokens_path"])
            os.chmod(tokens_path, 0o644)
            data = bytearray(tokens_path.read_bytes())
            data[0] ^= 1
            tokens_path.write_bytes(data)
            self.assertEqual(values.tolist(), private_values)
            with self.assertRaisesRegex(ContractViolation, "ARTIFACT_HASH_MISMATCH"):
                verify_corpus_manifest(
                    root / "corpus/corpus_manifest.json", production=False
                )

    def test_tokenizer_loader_rechecks_bytes_after_manifest_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self._build(root)
            manifest_path = root / "tokenizer-a/tokenizer_manifest.json"
            real_verify = build_tokenizer_module.verify_tokenizer_manifest

            def verify_then_change(*args: object, **kwargs: object) -> dict:
                verified = real_verify(*args, **kwargs)
                tokenizer_path = Path(verified["tokenizer_path"])
                os.chmod(tokenizer_path, 0o644)
                changed = bytearray(tokenizer_path.read_bytes())
                original_size = len(changed)
                changed[0] ^= 1
                tokenizer_path.write_bytes(changed)
                self.assertEqual(tokenizer_path.stat().st_size, original_size)
                return verified

            with mock.patch.object(
                build_tokenizer_module,
                "verify_tokenizer_manifest",
                side_effect=verify_then_change,
            ):
                with self.assertRaisesRegex(
                    ContractViolation, "ARTIFACT_HASH_MISMATCH"
                ):
                    load_verified_tokenizer(manifest_path, production=False)

    def test_corpus_loader_rechecks_bytes_after_manifest_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self._build(root)
            materialize_corpus_tokens(
                root / "source-a/source_manifest.json",
                root / "tokenizer-a/tokenizer_manifest.json",
                output_dir=root / "corpus",
                production=False,
                scope_root=root,
            )
            manifest_path = root / "corpus/corpus_manifest.json"
            real_verify = build_tokenizer_module.verify_corpus_manifest

            def verify_then_change(*args: object, **kwargs: object) -> dict:
                verified = real_verify(*args, **kwargs)
                tokens_path = Path(verified["tokens_path"])
                os.chmod(tokens_path, 0o644)
                changed = bytearray(tokens_path.read_bytes())
                changed[0] ^= 1
                tokens_path.write_bytes(changed)
                return verified

            with mock.patch.object(
                build_tokenizer_module,
                "verify_corpus_manifest",
                side_effect=verify_then_change,
            ):
                with self.assertRaisesRegex(
                    ContractViolation, "ARTIFACT_HASH_MISMATCH"
                ):
                    load_corpus_token_memmap(manifest_path, production=False)

    def test_corpus_index_parses_only_the_final_hash_checked_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self._build(root)
            materialize_corpus_tokens(
                root / "source-a/source_manifest.json",
                root / "tokenizer-a/tokenizer_manifest.json",
                output_dir=root / "corpus",
                production=False,
                scope_root=root,
            )
            manifest_path = root / "corpus/corpus_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            index_path = (
                manifest_path.parent / manifest["artifacts"]["index"]["path"]
            ).resolve()
            real_resolve = build_tokenizer_module._resolve_artifact
            changed_index = False

            def resolve_then_change(*args: object, **kwargs: object) -> Path:
                nonlocal changed_index
                resolved = real_resolve(*args, **kwargs)
                if resolved == index_path and not changed_index:
                    changed_index = True
                    os.chmod(resolved, 0o644)
                    changed = bytearray(resolved.read_bytes())
                    original_size = len(changed)
                    self.assertEqual(changed[-1:], b"\n")
                    changed[-1:] = b" "
                    resolved.write_bytes(changed)
                    self.assertEqual(resolved.stat().st_size, original_size)
                return resolved

            with mock.patch.object(
                build_tokenizer_module,
                "_resolve_artifact",
                side_effect=resolve_then_change,
            ):
                with self.assertRaisesRegex(
                    ContractViolation, "ARTIFACT_HASH_MISMATCH"
                ):
                    verify_corpus_manifest(manifest_path, production=False)

    def test_tokenizer_verifier_parses_only_plan_bytes_matching_its_ref(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            *_, plan_path = self._build(root)
            manifest_path = root / "tokenizer-a/tokenizer_manifest.json"
            real_resolve = build_tokenizer_module._resolve_artifact
            plan_path = plan_path.resolve()
            swapped = False

            def resolve_then_change(*args: object, **kwargs: object) -> Path:
                nonlocal swapped
                resolved = real_resolve(*args, **kwargs)
                if resolved == plan_path and not swapped:
                    swapped = True
                    os.chmod(resolved, 0o644)
                    changed = bytearray(resolved.read_bytes())
                    self.assertEqual(changed[-1:], b"\n")
                    changed[-1:] = b" "
                    resolved.write_bytes(changed)
                return resolved

            with mock.patch.object(
                build_tokenizer_module,
                "_resolve_artifact",
                side_effect=resolve_then_change,
            ):
                with self.assertRaisesRegex(
                    ContractViolation, "ARTIFACT_HASH_MISMATCH"
                ):
                    verify_tokenizer_manifest(manifest_path, production=False)

    def test_boundary_audit_is_hash_bound_and_exhaustive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, _, _, _, _, plan_path = self._build(root)
            concept_core = {
                "schema_version": "4.0.0",
                "concept_id": "fixture-concept",
                "answers": {
                    "ko": "시험",
                    "en": "test",
                    "zh": "测试",
                    "fr": "essai",
                },
                "glosses": {
                    "ko": "능력을 알아보는 절차",
                    "en": "a procedure that checks ability",
                    "zh": "检查能力的程序",
                    "fr": "une procédure qui vérifie une capacité",
                },
            }
            concept = {
                **concept_core,
                "concept_record_sha256": logical_hash(concept_core),
            }
            second_core = {
                "schema_version": "4.0.0",
                "concept_id": "fixture-concept-2",
                "answers": {
                    "ko": "평가",
                    "en": "assessment",
                    "zh": "评价",
                    "fr": "évaluation",
                },
                "glosses": {
                    "ko": "가치를 판단하는 절차",
                    "en": "a procedure that judges value",
                    "zh": "判断价值的程序",
                    "fr": "une procédure qui juge la valeur",
                },
            }
            second = {
                **second_core,
                "concept_record_sha256": logical_hash(second_core),
            }
            audit = audit_lexical_token_boundaries(
                root / "tokenizer-a/tokenizer_manifest.json",
                [concept, second],
                production=False,
                evaluation_plan_path=plan_path,
                scope_root=root,
            )
            reversed_audit = audit_lexical_token_boundaries(
                root / "tokenizer-a/tokenizer_manifest.json",
                [second, concept],
                production=False,
                evaluation_plan_path=plan_path,
                scope_root=root,
            )
            self.assertEqual(audit, reversed_audit)
            self.assertEqual(audit["audit_sha256"], logical_hash({k: v for k, v in audit.items() if k != "audit_sha256"}))
            self.assertEqual(audit["contexts_checked"], 240)
            self.assertEqual(
                audit["contexts_passed"] + audit["contexts_failed"], 240
            )
            self.assertEqual(
                [row["concept_id"] for row in audit["expression_only_metrics"]],
                ["fixture-concept", "fixture-concept-2"],
            )
            for row in audit["expression_only_metrics"]:
                metrics = row["four_language_metrics"]
                self.assertEqual(
                    metrics["excluded_special_token_ids"],
                    {"<|endoftext|>": 0, "<|pad|>": 1},
                )
                self.assertEqual(
                    metrics["pair_order"],
                    ["ko-en", "ko-zh", "ko-fr", "en-zh", "en-fr", "zh-fr"],
                )
                self.assertEqual(set(metrics["languages"]), {"ko", "en", "zh", "fr"})
                self.assertEqual(len(metrics["pairs"]), 6)
                self.assertEqual(
                    row["en_token_length"], metrics["languages"]["en"]["token_length"]
                )
                self.assertEqual(
                    row["fr_token_length"], metrics["languages"]["fr"]["token_length"]
                )
                self.assertEqual(
                    row["en_fr_token_jaccard"],
                    metrics["pairs"]["en-fr"]["token_set_jaccard"],
                )
            tampered = json.loads(json.dumps(audit))
            tampered["expression_only_metrics"][0]["four_language_metrics"][
                "pairs"
            ]["en-fr"]["surface_levenshtein_similarity"] = 0.123
            self.assertNotEqual(
                tampered["audit_sha256"],
                logical_hash(
                    {key: value for key, value in tampered.items() if key != "audit_sha256"}
                ),
            )
            self.assertIn(audit["status"], {"PASS", "BLOCKED_TOKEN_BOUNDARY"})


if __name__ == "__main__":
    unittest.main()
