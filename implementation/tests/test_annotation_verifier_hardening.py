from __future__ import annotations

import csv
import errno
import hashlib
import json
import shutil
import tempfile
import threading
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import implementation.src.prepare_data as prepare_data_module
from implementation.src.artifacts import publish_bytes_once, publish_json_once
from implementation.src.contracts import (
    IMPLEMENTATION_REVISION,
    PROJECT_ROOT,
    VERSION,
    ContractViolation,
    canonical_json_bytes,
    load_pilot_config,
)
from implementation.src.prepare_data import (
    build_krdict_collection_plan,
    freeze_reviewed_dataset,
    import_existing_krdict_collection,
    merge_krdict_snapshot,
    verify_annotation_freeze,
)
from implementation.src.review_freeze import freeze_completed_review_csv_bundle
from implementation.src.review_import import (
    PAIR_FIELDS,
    SCREEN_FIELDS,
    TERM_FIELDS,
    build_review_tables,
    en_fr_pair_evidence_subject_sha256,
)


LANGUAGES = ("ko", "en", "zh", "fr")
PAIRING_RULE = (
    "lexicographically sort frozen concept_id and pair adjacent positions "
    "(0,1),(2,3),..."
)
REQUIREMENTS = {
    "min_total": 60,
    "min_identifiable": 40,
    "min_related": 12,
    "min_distinct_routes": 12,
    "requested_pilot_cohort_size": 60,
}


def _logical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _without(mapping: dict, key: str) -> dict:
    return {name: value for name, value in mapping.items() if name != key}


def _campaign_policy() -> dict:
    return {
        "version": IMPLEMENTATION_REVISION,
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


def _krdict_payload(language_name: str, language: str) -> bytes:
    senses: list[str] = []
    for index in range(60):
        if language == "en":
            translated = f"english-{index}; alternate-{index}"
        elif language == "zh":
            translated = f"中文-{index}"
        else:
            translated = f"francais-{index}"
        senses.append(
            "<sense>"
            f"<sense_order>{index + 1}</sense_order>"
            f"<definition>능력을 알아보기 위한 절차 {index}.</definition>"
            "<translation>"
            f"<trans_lang>{language_name}</trans_lang>"
            f"<trans_word>{translated}</trans_word>"
            f"<trans_dfn>{language} definition {index}</trans_dfn>"
            "</translation>"
            "</sense>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<channel><total>1</total><start>1</start><num>100</num>"
        "<item><target_code>100</target_code><word>시험</word><pos>명사</pos>"
        "<link>https://krdict.korean.go.kr/kor/dicSearch/"
        "SearchView?ParaWordNo=100</link>"
        + "".join(senses)
        + "</item></channel>"
    ).encode("utf-8")


def _build_source(root: Path) -> Path:
    policy = _campaign_policy()
    plan = build_krdict_collection_plan(
        ["시험", "제외"], campaign_policy=policy
    )
    legacy = root / "legacy"
    legacy.mkdir()
    names = {"en": "영어", "zh": "중국어", "fr": "프랑스어"}
    records: list[dict] = []
    for request in plan["requests"]:
        language = request["language"]
        payload = _krdict_payload(names[language], language)
        raw_name = f"0000_{language}.xml"
        ref = publish_bytes_once(legacy / raw_name, payload)
        records.append(
            {
                "query_index": 0,
                "language": language,
                "endpoint": request["endpoint"],
                "params_without_key": dict(request["params_without_key"]),
                "status": "PAYLOAD_SCHEMA_CHECKED",
                "raw_file": raw_name,
                "sha256": ref["sha256"],
                "bytes": ref["bytes"],
                "summary": {},
            }
        )
    legacy_manifest = legacy / "source_manifest.json"
    publish_json_once(
        legacy_manifest,
        {
            "status": "COLLECTED_UNREVIEWED",
            "requests_made": len(records),
            "records": records,
        },
    )
    adapted = root / "adapted"
    import_existing_krdict_collection(
        legacy_manifest,
        plan,
        output_dir=adapted,
        campaign_policy=policy,
        production=False,
        scope_root=root,
    )
    return adapted / "source_manifest.json"


def _evidence(
    tag: str, *, subject_kind: str, subject_id: str, supports_label: str
) -> dict:
    payload = tag.encode("utf-8")
    core = {
        "evidence_id": tag,
        "subject_kind": subject_kind,
        "subject_id": subject_id,
        "supports_label": supports_label,
        "source_url": f"https://example.invalid/evidence/{tag}",
        "evidence_origin": "synthetic-human-reviewed-fixture",
        "source_name": "Synthetic Evidence Dictionary",
        "source_version": "fixture-v1",
        "source_license": "Synthetic fixture license",
        "source_license_url": "https://example.invalid/license",
        "sense_locator": tag,
        "payload_path": f"evidence/{tag}.txt",
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_bytes": len(payload),
        "retrieved_at": "2026-01-01T00:00:00Z",
        "conflicts_with": [],
    }
    return {**core, "record_sha256": _logical_hash(core)}


def _write_review_csv(
    path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _strict_evidence_record(
    evidence_id: str,
    *,
    subject_kind: str,
    subject_id: str,
    supports_label: str,
    payload: Path,
    scope_root: Path,
    evidence_origin: str = "synthetic-human-reviewed-fixture",
) -> dict:
    core = {
        "evidence_id": evidence_id,
        "subject_kind": subject_kind,
        "subject_id": subject_id,
        "supports_label": supports_label,
        "source_url": f"https://example.invalid/evidence/{evidence_id}",
        "evidence_origin": evidence_origin,
        "source_name": "Synthetic Evidence Dictionary",
        "source_version": "fixture-v1",
        "source_license": "Synthetic fixture license",
        "source_license_url": "https://example.invalid/license",
        "sense_locator": f"sense-{evidence_id}",
        "payload_path": str(payload.relative_to(scope_root)),
        "payload_sha256": _sha256_file(payload),
        "payload_bytes": payload.stat().st_size,
        "retrieved_at": "2026-01-01T00:00:00Z",
        "conflicts_with": [],
    }
    return {**core, "record_sha256": _logical_hash(core)}


def _build_csv_evidence_freeze(
    root: Path,
    *,
    source: Path | None = None,
    scope_root: Path | None = None,
    evidence_origin: str = "synthetic-human-reviewed-fixture",
) -> Path:
    source = _build_source(root) if source is None else source
    scope_root = root if scope_root is None else scope_root
    merge = merge_krdict_snapshot(source, production=False)
    blank = build_review_tables(merge)
    terms = [dict(row) for row in blank.terms]
    pairs = [dict(row) for row in blank.pairs]
    screen = [dict(row) for row in blank.screen]
    term_by_key = {
        (row["candidate_id"], row["lang"], row["term_index"]): row
        for row in terms
    }

    payload = root / "evidence" / "shared-attestation.txt"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"offline synthetic human-reviewed evidence payload\n")
    records: list[dict] = []
    review_date = "2026-01-01"
    reviewer = "synthetic-test-reviewer"

    pilot_indices: dict[str, int] = {}
    for pair in pairs:
        candidate_id = pair["candidate_id"]
        first_terms = {
            language: term_by_key[(candidate_id, language, "1")]
            for language in LANGUAGES
        }
        if (
            len(pilot_indices) < 60
            and len({term["term_canonical"] for term in first_terms.values()}) == 4
        ):
            pilot_indices[candidate_id] = len(pilot_indices)
    if len(pilot_indices) != 60:
        raise AssertionError("fixture source does not contain 60 identifiable candidates")

    for pair in pairs:
        candidate_id = pair["candidate_id"]
        if candidate_id not in pilot_indices:
            continue
        index = pilot_indices[candidate_id]
        selected = {
            language: term_by_key[(candidate_id, language, "1")]
            for language in LANGUAGES
        }
        for language in ("en", "fr"):
            term = selected[language]
            evidence_id = f"term-{index:02d}-{language}"
            term.update(
                {
                    "segmentation_decision": "APPROVED",
                    "translation_quality": "ATTESTED_SAME_SENSE",
                    "is_transliteration": "FALSE",
                    "quality_evidence_ids_json": json.dumps([evidence_id]),
                    "quality_note": "Synthetic source attestation reviewed offline.",
                    "term_review_status": "APPROVED_BY_RESEARCHER",
                    "term_reviewer": reviewer,
                    "term_review_date": review_date,
                }
            )
            records.append(
                _strict_evidence_record(
                    evidence_id,
                    subject_kind="TERM",
                    subject_id=term["term_id"],
                    supports_label="ATTESTED_SAME_SENSE",
                    payload=payload,
                    scope_root=scope_root,
                    evidence_origin=evidence_origin,
                )
            )

        if index < 12:
            subtype = "BORROWING_DIRECT"
            primary = "BORROWING_DOCUMENTED"
            direction = "EN_TO_FR"
            confidence = "MEDIUM"
            search_note = ""
        elif index < 24:
            subtype = "DISTINCT_ROUTES_REVIEWED"
            primary = "DISTINCT_ROUTES_REVIEWED"
            direction = "NONE"
            confidence = "MEDIUM"
            search_note = ""
        else:
            subtype = "INDETERMINATE"
            primary = "UNRESOLVED"
            direction = "UNKNOWN"
            confidence = "LOW"
            search_note = "Synthetic search found insufficient historical evidence."
        pair_evidence_id = f"pair-{index:02d}"
        pair.update(
            {
                "candidate_decision": "SELECT_PAIR",
                "selected_ko_term_id": selected["ko"]["term_id"],
                "selected_en_term_id": selected["en"]["term_id"],
                "selected_zh_term_id": selected["zh"]["term_id"],
                "selected_fr_term_id": selected["fr"]["term_id"],
                "selection_rationale": "Selected exact source-backed first terms.",
                "meaning_alignment_decision": "ALIGNED",
                "sense_alignment_note": "Synthetic aligned sense reviewed offline.",
                "source_alignment_checked": "TRUE",
                "answer_copy_checked": "TRUE",
                "synonym_cluster_id": f"synonym-{index:02d}",
                "pilot_selected": "TRUE",
                "etymology_subtype": subtype,
                "etymology_primary": primary,
                "relation_direction": direction,
                "shared_source": "",
                "historical_scope": "Synthetic historical scope.",
                "confidence": confidence,
                "etymology_evidence_ids_json": json.dumps([pair_evidence_id]),
                "evidence_search_note": search_note,
                "family_id": f"family-{index:02d}",
                "concept_review_status": "APPROVED_BY_RESEARCHER",
                "concept_reviewer": reviewer,
                "concept_review_date": review_date,
                "etymology_review_status": "APPROVED_BY_RESEARCHER",
                "etymology_reviewer": reviewer,
                "etymology_review_date": review_date,
            }
        )
        records.append(
            _strict_evidence_record(
                pair_evidence_id,
                subject_kind="EN_FR_PAIR",
                subject_id=en_fr_pair_evidence_subject_sha256(
                    candidate_id=pair["candidate_id"],
                    en_term_id=selected["en"]["term_id"],
                    en_term_sha256=selected["en"]["term_sha256"],
                    en_canonical_answer=selected["en"]["term_canonical"],
                    fr_term_id=selected["fr"]["term_id"],
                    fr_term_sha256=selected["fr"]["term_sha256"],
                    fr_canonical_answer=selected["fr"]["term_canonical"],
                ),
                supports_label=subtype,
                payload=payload,
                scope_root=scope_root,
                evidence_origin=evidence_origin,
            )
        )

    review_dir = root / "completed-review"
    terms_csv = review_dir / "terms_long.csv"
    pairs_csv = review_dir / "pair_selection_sheet.csv"
    screen_csv = review_dir / "untranslated_screen.csv"
    _write_review_csv(terms_csv, terms, TERM_FIELDS)
    _write_review_csv(pairs_csv, pairs, PAIR_FIELDS)
    _write_review_csv(screen_csv, screen, SCREEN_FIELDS)

    output = root / "csv-evidence-freeze"
    freeze_completed_review_csv_bundle(
        source,
        terms_csv,
        pairs_csv,
        screen_csv,
        evidence_records=records,
        output_dir=output,
        requirements=REQUIREMENTS,
        production=False,
        scope_root=scope_root,
    )
    return output


def _build_valid_freeze(root: Path) -> Path:
    source = _build_source(root)
    merge = merge_krdict_snapshot(source, production=False)
    tables = build_review_tables(merge)
    terms = {
        (row["candidate_id"], row["lang"], row["term_index"]): row
        for row in tables.terms
    }
    concept_reviews: list[dict] = []
    etymology_reviews: list[dict] = []
    for index, candidate in enumerate(merge["eligible_candidates"]):
        candidate_id = candidate["candidate_id"]
        selected = {
            language: terms[(candidate_id, language, "1")]
            for language in LANGUAGES
        }
        selections = {
            language: {
                "option_id": term["source_option_id"],
                "answer": term["term_canonical"],
                "source_span": [
                    int(term["source_span_start"]),
                    int(term["source_span_end"]),
                ],
                "selection_rationale": "Synthetic exact source span.",
            }
            for language, term in selected.items()
        }
        term_quality: dict[str, dict] = {}
        for language in ("en", "fr"):
            term = selected[language]
            evidence_id = f"quality-{index}-{language}"
            term_quality[language] = {
                "term_id": term["term_id"],
                "term_sha256": term["term_sha256"],
                "segmentation_decision": "APPROVED",
                "translation_quality": "ATTESTED_SAME_SENSE",
                "is_transliteration": False,
                "quality_note": "Synthetic term attestation.",
                "quality_evidence_ids": [evidence_id],
                "evidence": [
                    _evidence(
                        evidence_id,
                        subject_kind="TERM",
                        subject_id=term["term_id"],
                        supports_label="ATTESTED_SAME_SENSE",
                    )
                ],
                "qa": {
                    "status": "APPROVED_BY_RESEARCHER",
                    "reviewer": "synthetic-test-reviewer",
                    "review_date": "2026-01-01",
                },
            }
        concept_reviews.append(
            {
                "candidate_id": candidate_id,
                "candidate_sha256": candidate["candidate_sha256"],
                "synthetic_fixture": True,
                "selections": selections,
                "synonym_cluster_id": f"synonym-{index}",
                "term_quality_reviews": term_quality,
                "qa": {
                    "status": "APPROVED_BY_RESEARCHER",
                    "reviewer": "synthetic-test-reviewer",
                    "review_date": "2026-01-01",
                    "source_alignment_checked": True,
                    "answer_copy_checked": True,
                    "meaning_alignment_note": "Synthetic aligned sense.",
                },
            }
        )

        if index < 12:
            relation = "BORROWING_DOCUMENTED"
            subtype = "BORROWING_DIRECT"
            direction = "EN_TO_FR"
            confidence = "MEDIUM"
            evidence_ids = [f"etymology-{index}"]
            search_note = None
        elif index < 24:
            relation = "DISTINCT_ROUTES_REVIEWED"
            subtype = "DISTINCT_ROUTES_REVIEWED"
            direction = "NONE"
            confidence = "MEDIUM"
            evidence_ids = [f"etymology-{index}"]
            search_note = None
        else:
            relation = "UNRESOLVED"
            subtype = "INDETERMINATE"
            direction = "UNKNOWN"
            confidence = "LOW"
            evidence_ids = [f"etymology-{index}"]
            search_note = "Synthetic search found insufficient evidence."
        en_answer = selections["en"]["answer"]
        fr_answer = selections["fr"]["answer"]
        surface_pair_sha256 = _logical_hash(
            {"en": en_answer, "fr": fr_answer}
        )
        evidence_subject_sha256 = en_fr_pair_evidence_subject_sha256(
            candidate_id=candidate_id,
            en_term_id=selected["en"]["term_id"],
            en_term_sha256=selected["en"]["term_sha256"],
            en_canonical_answer=selected["en"]["term_canonical"],
            fr_term_id=selected["fr"]["term_id"],
            fr_term_sha256=selected["fr"]["term_sha256"],
            fr_canonical_answer=selected["fr"]["term_canonical"],
        )
        if evidence_ids:
            evidence = [
                _evidence(
                    evidence_ids[0],
                    subject_kind="EN_FR_PAIR",
                    subject_id=evidence_subject_sha256,
                    supports_label=subtype,
                )
            ]
        etymology_reviews.append(
            {
                "candidate_id": candidate_id,
                "candidate_sha256": candidate["candidate_sha256"],
                "synthetic_fixture": True,
                "answer_pair_sha256": surface_pair_sha256,
                "evidence_subject_sha256": evidence_subject_sha256,
                "pair": ["en", "fr"],
                "relation": relation,
                "relation_subtype": subtype,
                "relation_direction": direction,
                "shared_source": None,
                "confidence": confidence,
                "evidence_ids": evidence_ids,
                "family_id": f"family-{index}",
                "sense_alignment_note": "Synthetic aligned sense.",
                "historical_scope": "Synthetic historical scope.",
                "evidence": evidence,
                "evidence_search_note": search_note,
                "qa": {
                    "status": "APPROVED_BY_RESEARCHER",
                    "reviewer": "synthetic-test-reviewer",
                    "review_date": "2026-01-01",
                    "source_alignment_checked": True,
                    "answer_copy_checked": True,
                },
            }
        )
    output = root / "annotation-freeze"
    freeze_reviewed_dataset(
        source,
        concept_reviews,
        etymology_reviews,
        cohort_ids=[row["candidate_id"] for row in merge["eligible_candidates"]],
        output_dir=output,
        requirements=REQUIREMENTS,
        production=False,
        scope_root=root,
    )
    return output


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write(path: Path, data: bytes) -> None:
    path.chmod(0o600)
    path.write_bytes(data)


def _write_json(path: Path, value: object) -> None:
    _write(path, canonical_json_bytes(value))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    _write(path, b"".join(canonical_json_bytes(row) for row in rows))


def _history_pairs(concept_ids: list[str]) -> dict:
    ordered = sorted(concept_ids)
    pairs: list[dict] = []
    for offset in range(0, len(ordered), 2):
        core = {
            "pair_id": f"B36-pair-{offset // 2:02d}",
            "pair_index": offset // 2,
            "concept_ids": ordered[offset : offset + 2],
        }
        pairs.append({**core, "pair_sha256": _logical_hash(core)})
    core = {
        "schema_version": VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "project_id": load_pilot_config()["project_id"],
        "review_validation_mode": "SYNTHETIC_TEST_FIXTURE",
        "synthetic_fixture": True,
        "status": "PASS",
        "history_family": load_pilot_config()["preparation"]["history_family"],
        "pairing_rule": PAIRING_RULE,
        "concept_order": ordered,
        "pairs": pairs,
    }
    return {**core, "history_pair_set_sha256": _logical_hash(core)}


def _load_freeze(directory: Path) -> dict[str, object]:
    return {
        "manifest": json.loads(
            (directory / "annotation_freeze_manifest.json").read_text(
                encoding="utf-8"
            )
        ),
        "concepts": _load_jsonl(directory / "concepts.jsonl"),
        "etymology": _load_jsonl(directory / "etymology_en_fr.jsonl"),
        "cohort": json.loads((directory / "cohort.json").read_text(encoding="utf-8")),
        "audit": json.loads(
            (directory / "review_audit.json").read_text(encoding="utf-8")
        ),
        "history": json.loads(
            (directory / "history_pairs.json").read_text(encoding="utf-8")
        ),
    }


def _reseal(directory: Path, bundle: dict[str, object]) -> Path:
    manifest = bundle["manifest"]
    concepts = bundle["concepts"]
    etymology = bundle["etymology"]
    cohort = bundle["cohort"]
    audit = bundle["audit"]
    history = bundle["history"]
    assert isinstance(manifest, dict)
    assert isinstance(concepts, list)
    assert isinstance(etymology, list)
    assert isinstance(cohort, dict)
    assert isinstance(audit, dict)
    assert isinstance(history, dict)

    for concept in concepts:
        concept["concept_record_sha256"] = _logical_hash(
            _without(concept, "concept_record_sha256")
        )
    concept_hashes = {
        row["concept_id"]: row["concept_record_sha256"] for row in concepts
    }
    for row in etymology:
        row["concept_record_sha256"] = concept_hashes[row["concept_id"]]
        row["etymology_record_sha256"] = _logical_hash(
            _without(row, "etymology_record_sha256")
        )

    reconstructed = {
        **_without(audit, "review_bundle_sha256"),
        "concepts": concepts,
        "etymology": etymology,
    }
    audit["review_bundle_sha256"] = _logical_hash(reconstructed)
    manifest["review_bundle_sha256"] = audit["review_bundle_sha256"]

    paths = {
        "concepts": directory / "concepts.jsonl",
        "etymology": directory / "etymology_en_fr.jsonl",
        "cohort": directory / "cohort.json",
        "review_audit": directory / "review_audit.json",
        "history_pairs": directory / "history_pairs.json",
    }
    _write_jsonl(paths["concepts"], concepts)
    _write_jsonl(paths["etymology"], etymology)
    _write_json(paths["cohort"], cohort)
    _write_json(paths["review_audit"], audit)
    _write_json(paths["history_pairs"], history)
    for name, path in paths.items():
        manifest["artifacts"][name]["sha256"] = _sha256_file(path)
        manifest["artifacts"][name]["bytes"] = path.stat().st_size
    manifest["freeze_id"] = "annotation-" + _logical_hash(
        _without(manifest, "freeze_id")
    )[:20]
    manifest_path = directory / "annotation_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


class AnnotationFreezeAtomicPublicationTests(unittest.TestCase):
    @staticmethod
    def _staged_directory(root: Path, name: str) -> Path:
        source = root / name
        source.mkdir()
        (source / "annotation_freeze_manifest.json").write_text(
            name, encoding="utf-8"
        )
        return source

    def test_existing_empty_directory_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = self._staged_directory(root, "staged")
            destination = root / "freeze"
            destination.mkdir()

            with self.assertRaisesRegex(
                ContractViolation, "ANNOTATION_FREEZE_OUTPUT_EXISTS"
            ):
                prepare_data_module._rename_directory_noreplace(
                    source, destination
                )

            self.assertTrue(source.is_dir())
            self.assertEqual(list(destination.iterdir()), [])

    def test_concurrent_publishers_have_exactly_one_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sources = [
                self._staged_directory(root, "staged-a"),
                self._staged_directory(root, "staged-b"),
            ]
            destination = root / "freeze"
            barrier = threading.Barrier(2)

            def publish(source: Path) -> str:
                barrier.wait()
                try:
                    prepare_data_module._rename_directory_noreplace(
                        source, destination
                    )
                except ContractViolation as exc:
                    return exc.code
                return "PASS"

            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(publish, sources))

            self.assertCountEqual(
                outcomes, ["PASS", "ANNOTATION_FREEZE_OUTPUT_EXISTS"]
            )
            self.assertTrue(
                (destination / "annotation_freeze_manifest.json").is_file()
            )
            self.assertEqual(sum(source.exists() for source in sources), 1)

    def test_rename_failure_leaves_final_name_absent_and_source_intact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = self._staged_directory(root, "staged")
            destination = root / "freeze"
            failure = OSError(errno.EIO, "injected rename failure")

            with mock.patch.object(
                prepare_data_module, "_renameat2_noreplace", side_effect=failure
            ), mock.patch.object(prepare_data_module, "_fsync_directory") as fsync:
                with self.assertRaisesRegex(
                    ContractViolation, "ANNOTATION_FREEZE_ATOMIC_PUBLISH_FAILED"
                ):
                    prepare_data_module._rename_directory_noreplace(
                        source, destination
                    )

            self.assertTrue(source.is_dir())
            self.assertFalse(destination.exists())
            fsync.assert_not_called()

    def test_success_fsyncs_final_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = self._staged_directory(root, "staged")
            destination = root / "freeze"
            real_fsync = prepare_data_module._fsync_directory

            with mock.patch.object(
                prepare_data_module, "_fsync_directory", wraps=real_fsync
            ) as fsync:
                prepare_data_module._rename_directory_noreplace(
                    source, destination
                )

            fsync.assert_called_once_with(root)
            self.assertTrue(destination.is_dir())


class AnnotationVerifierHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name).resolve()
        cls.valid_directory = _build_valid_freeze(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def _copy(self, name: str) -> Path:
        destination = self.root / name
        shutil.copytree(self.valid_directory, destination)
        return destination

    def test_valid_source_bound_exact_60_freeze_verifies(self) -> None:
        verified = verify_annotation_freeze(
            self.valid_directory / "annotation_freeze_manifest.json",
            production=False,
        )
        self.assertEqual(
            verified["review_validation_mode"], "SYNTHETIC_TEST_FIXTURE"
        )
        self.assertIs(verified["synthetic_fixture"], True)
        self.assertEqual(len(verified["concepts"]), 60)
        self.assertEqual(len(verified["history_pairs"]["pairs"]), 30)
        self.assertEqual(verified["review_audit"]["coverage"]["n_total"], 60)
        self.assertEqual(
            verified["review_audit"]["coverage"]["n_related_identifiable"],
            12,
        )
        self.assertEqual(
            verified["review_audit"]["coverage"]["n_distinct_routes_identifiable"],
            12,
        )

    def test_csv_to_freeze_publishes_and_verifies_strict_evidence_offline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = _build_csv_evidence_freeze(root)
            manifest_path = frozen / "annotation_freeze_manifest.json"
            verified = verify_annotation_freeze(manifest_path, production=False)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIn("evidence_records", manifest["artifacts"])
            self.assertEqual(manifest["gates"]["evidence_payloads"], "PASS")
            self.assertEqual(len(manifest["evidence_payloads"]), 1)
            self.assertEqual(len(verified["evidence_records"]), 180)
            self.assertEqual(
                len({row["evidence_id"] for row in verified["evidence_records"]}),
                180,
            )
            self.assertEqual(
                Counter(
                    row["subject_kind"] for row in verified["evidence_records"]
                ),
                {"TERM": 120, "EN_FR_PAIR": 60},
            )
            self.assertEqual(len(verified["resolved_evidence_payloads"]), 1)
            self.assertEqual(verified["review_audit"]["coverage"]["n_total"], 60)
            self.assertEqual(
                verified["review_audit"]["coverage"][
                    "n_related_identifiable"
                ],
                12,
            )
            self.assertEqual(
                verified["review_audit"]["coverage"][
                    "n_distinct_routes_identifiable"
                ],
                12,
            )

            for kind in ("evidence-records", "evidence-payload"):
                with self.subTest(kind=kind):
                    tampered = root / f"tampered-{kind}"
                    shutil.copytree(frozen, tampered)
                    copied_manifest_path = (
                        tampered / "annotation_freeze_manifest.json"
                    )
                    copied_manifest = json.loads(
                        copied_manifest_path.read_text(encoding="utf-8")
                    )
                    if kind == "evidence-records":
                        ref = copied_manifest["artifacts"]["evidence_records"]
                    else:
                        ref = next(iter(copied_manifest["evidence_payloads"].values()))
                    artifact = tampered / ref["path"]
                    changed = bytearray(artifact.read_bytes())
                    changed[0] ^= 1
                    _write(artifact, bytes(changed))
                    with self.assertRaisesRegex(
                        ContractViolation, "ARTIFACT_HASH_MISMATCH"
                    ):
                        verify_annotation_freeze(
                            copied_manifest_path, production=False
                        )

    def test_real_source_synthetic_review_cannot_launder_into_production(self) -> None:
        real_source = (
            PROJECT_ROOT
            / "work/sources/krdict/hardened_v4r1_20260914T151550Z/source_manifest.json"
        )
        self.assertTrue(real_source.is_file())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = _build_csv_evidence_freeze(
                root,
                source=real_source,
                scope_root=Path("/"),
                evidence_origin="human-reviewed-official-reference",
            )
            manifest_path = frozen / "annotation_freeze_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["data_kind"], "REAL_KRDICT_API")
            self.assertEqual(
                manifest["review_validation_mode"], "SYNTHETIC_TEST_FIXTURE"
            )
            self.assertIs(manifest["synthetic_fixture"], True)
            # Root is widened only inside this offline regression so that a
            # missing mode gate would exercise the full old laundering path.
            with mock.patch.object(prepare_data_module, "WORK_ROOT", Path("/")):
                with self.assertRaisesRegex(
                    ContractViolation, "SYNTHETIC_FREEZE_NOT_PRODUCTION"
                ):
                    verify_annotation_freeze(manifest_path, production=True)

    def test_resealed_annotation_metadata_is_pinned(self) -> None:
        directory = self._copy("tampered-annotation-metadata")
        bundle = _load_freeze(directory)
        manifest = bundle["manifest"]
        assert isinstance(manifest, dict)
        manifest["implementation_revision"] = "forged-old-revision"
        manifest["project_id"] = "forged-project"
        manifest_path = _reseal(directory, bundle)
        with self.assertRaisesRegex(ContractViolation, "INVALID_ANNOTATION_FREEZE"):
            verify_annotation_freeze(manifest_path, production=False)

    def test_resealed_review_audit_metadata_is_pinned(self) -> None:
        directory = self._copy("tampered-review-audit-metadata")
        bundle = _load_freeze(directory)
        audit = bundle["audit"]
        assert isinstance(audit, dict)
        audit.update(
            {
                "schema_version": "forged-schema",
                "implementation_revision": "forged-revision",
                "project_id": "forged-project",
                "status": "BLOCKED_DATA_QA",
            }
        )
        manifest_path = _reseal(directory, bundle)
        with self.assertRaisesRegex(
            ContractViolation, "INVALID_REVIEW_AUDIT_METADATA"
        ):
            verify_annotation_freeze(manifest_path, production=False)

    def test_resealed_history_metadata_is_pinned(self) -> None:
        directory = self._copy("tampered-history-metadata")
        bundle = _load_freeze(directory)
        history = bundle["history"]
        assert isinstance(history, dict)
        history.update(
            {
                "schema_version": "forged-schema",
                "implementation_revision": "forged-revision",
                "project_id": "forged-project",
                "status": "BLOCKED",
            }
        )
        history["history_pair_set_sha256"] = _logical_hash(
            _without(history, "history_pair_set_sha256")
        )
        manifest_path = _reseal(directory, bundle)
        with self.assertRaisesRegex(
            ContractViolation, "INVALID_HISTORY_PAIR_METADATA"
        ):
            verify_annotation_freeze(manifest_path, production=False)

    def test_resealed_concept_and_etymology_metadata_is_pinned(self) -> None:
        cases = (
            ("concepts", "INVALID_CONCEPT_METADATA"),
            ("etymology", "INVALID_ETYMOLOGY_METADATA"),
        )
        for target, expected in cases:
            with self.subTest(target=target):
                directory = self._copy(f"tampered-{target}-metadata")
                bundle = _load_freeze(directory)
                rows = bundle[target]
                assert isinstance(rows, list)
                rows[0].update(
                    {
                        "schema_version": "forged-schema",
                        "implementation_revision": "forged-revision",
                        "project_id": "forged-project",
                        "status": "BLOCKED",
                    }
                )
                manifest_path = _reseal(directory, bundle)
                with self.assertRaisesRegex(ContractViolation, expected):
                    verify_annotation_freeze(manifest_path, production=False)

    def test_resealed_selected_term_quality_answer_mismatch_is_rejected(self) -> None:
        directory = self._copy("tampered-term-quality-answer")
        bundle = _load_freeze(directory)
        concepts = bundle["concepts"]
        assert isinstance(concepts, list)
        concepts[0]["selected_term_quality"]["en"]["answer"] = (
            "attacker-stale-answer"
        )
        manifest_path = _reseal(directory, bundle)
        with self.assertRaisesRegex(
            ContractViolation, "TERM_QUALITY_ANSWER_MISMATCH"
        ):
            verify_annotation_freeze(manifest_path, production=False)

    def test_resealed_frozen_record_extra_keys_are_rejected(self) -> None:
        cases = (
            ("concepts", "CONCEPT_RECORD_SCHEMA_MISMATCH"),
            ("etymology", "ETYMOLOGY_RECORD_SCHEMA_MISMATCH"),
        )
        for target, expected in cases:
            with self.subTest(target=target):
                directory = self._copy(f"tampered-{target}-extra-key")
                bundle = _load_freeze(directory)
                rows = bundle[target]
                assert isinstance(rows, list)
                rows[0]["unregistered_training_override"] = "FORGED"
                manifest_path = _reseal(directory, bundle)
                with self.assertRaisesRegex(ContractViolation, expected):
                    verify_annotation_freeze(manifest_path, production=False)

    def test_resealed_nested_term_quality_extra_key_is_rejected(self) -> None:
        directory = self._copy("tampered-term-quality-extra-key")
        bundle = _load_freeze(directory)
        concepts = bundle["concepts"]
        assert isinstance(concepts, list)
        concepts[0]["selected_term_quality"]["en"][
            "unregistered_override"
        ] = "FORGED"
        manifest_path = _reseal(directory, bundle)
        with self.assertRaisesRegex(
            ContractViolation, "INVALID_SELECTED_TERM_QUALITY_REVIEW_SCHEMA"
        ):
            verify_annotation_freeze(manifest_path, production=False)

    def test_evidence_payload_key_and_ref_hash_must_match_before_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = _build_csv_evidence_freeze(root)
            manifest_path = frozen / "annotation_freeze_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload_hash = next(iter(manifest["evidence_payloads"]))
            replacement = "f" * 64 if payload_hash != "f" * 64 else "e" * 64
            manifest["evidence_payloads"][payload_hash]["sha256"] = replacement
            manifest["freeze_id"] = "annotation-" + _logical_hash(
                _without(manifest, "freeze_id")
            )[:20]
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                ContractViolation, "EVIDENCE_PAYLOAD_SHA256_MISMATCH"
            ):
                verify_annotation_freeze(manifest_path, production=False)

    def test_evidence_payload_final_size_and_hash_share_one_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = _build_csv_evidence_freeze(root)
            manifest_path = frozen / "annotation_freeze_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload_ref = next(iter(manifest["evidence_payloads"].values()))
            payload_path = (frozen / payload_ref["path"]).resolve()
            real_read = prepare_data_module.read_regular_file_bytes_exact

            def changed_payload(path: Path, *, expected_bytes: int) -> bytes:
                data = real_read(path, expected_bytes=expected_bytes)
                if Path(path).resolve() == payload_path:
                    return data + b"changed-after-resolve"
                return data

            with mock.patch.object(
                prepare_data_module,
                "read_regular_file_bytes_exact",
                side_effect=changed_payload,
            ):
                with self.assertRaisesRegex(
                    ContractViolation, "EVIDENCE_PAYLOAD_SIZE_MISMATCH"
                ):
                    verify_annotation_freeze(manifest_path, production=False)

    def test_annotation_source_reopen_identities_are_cross_bound(self) -> None:
        directory = self._copy("tampered-source-reopen-identity")
        manifest_path = directory / "annotation_freeze_manifest.json"
        real_verify = prepare_data_module.verify_krdict_collection

        def changed_source_identity(*args: object, **kwargs: object) -> dict:
            verified = dict(real_verify(*args, **kwargs))
            verified["manifest_sha256"] = "f" * 64
            return verified

        with mock.patch.object(
            prepare_data_module,
            "verify_krdict_collection",
            side_effect=changed_source_identity,
        ):
            with self.assertRaisesRegex(
                ContractViolation, "ANNOTATION_SOURCE_BINDING_MISMATCH"
            ):
                verify_annotation_freeze(manifest_path, production=False)

    def test_rebuilt_merge_must_match_first_verified_source(self) -> None:
        directory = self._copy("tampered-rebuilt-source-identity")
        manifest_path = directory / "annotation_freeze_manifest.json"
        real_merge = prepare_data_module.merge_krdict_snapshot

        def changed_merge_identity(*args: object, **kwargs: object) -> dict:
            rebuilt = dict(real_merge(*args, **kwargs))
            rebuilt["source_set_sha256"] = "f" * 64
            rebuilt["merge_sha256"] = _logical_hash(
                _without(rebuilt, "merge_sha256")
            )
            return rebuilt

        with mock.patch.object(
            prepare_data_module,
            "merge_krdict_snapshot",
            side_effect=changed_merge_identity,
        ):
            with self.assertRaisesRegex(
                ContractViolation, "ANNOTATION_SOURCE_BINDING_MISMATCH"
            ):
                verify_annotation_freeze(manifest_path, production=False)

    def test_self_consistently_resealed_source_binding_tampering_is_rejected(self) -> None:
        cases = (
            ("candidate", "FROZEN_CONCEPT_SOURCE_MISMATCH"),
            ("option", "FROZEN_SELECTION_NOT_IN_SOURCE"),
            ("span", "FROZEN_SELECTION_SOURCE_MISMATCH"),
        )
        for name, expected in cases:
            with self.subTest(name=name):
                directory = self._copy(f"tampered-{name}")
                bundle = _load_freeze(directory)
                concepts = bundle["concepts"]
                assert isinstance(concepts, list)
                concept = concepts[0]
                if name == "candidate":
                    concept["source_candidate_sha256"] = "f" * 64
                    concept["qa"]["review_candidate_sha256"] = "f" * 64
                elif name == "option":
                    concept["selected_option_ids"]["en"] = concepts[1][
                        "selected_option_ids"
                    ]["en"]
                else:
                    start, end = concept["selected_source_spans"]["en"]
                    alternate = concept["answers"]["en"].replace(
                        "english-", "alternate-"
                    )
                    concept["selected_source_spans"]["en"] = [
                        end + 2,
                        end + 2 + len(alternate),
                    ]
                manifest_path = _reseal(directory, bundle)
                with self.assertRaisesRegex(ContractViolation, expected):
                    verify_annotation_freeze(manifest_path, production=False)

    def test_self_consistently_resealed_cohort_and_review_audit_tampering_is_rejected(self) -> None:
        directory = self._copy("tampered-coverage")
        bundle = _load_freeze(directory)
        concepts = bundle["concepts"]
        etymology = bundle["etymology"]
        cohort = bundle["cohort"]
        audit = bundle["audit"]
        assert isinstance(concepts, list)
        assert isinstance(etymology, list)
        assert isinstance(cohort, dict)
        assert isinstance(audit, dict)

        removed = {row["concept_id"] for row in concepts[-2:]}
        concepts[:] = [row for row in concepts if row["concept_id"] not in removed]
        etymology[:] = [row for row in etymology if row["concept_id"] not in removed]
        remaining = sorted(row["concept_id"] for row in concepts)
        identifiable = sorted(
            row["concept_id"] for row in concepts if row["identifiable"] is True
        )
        cohort.clear()
        cohort.update(
            {
                "concept_ids": remaining,
                "identifiable_concept_ids": identifiable,
            }
        )
        audit["cohort"] = dict(cohort)
        counts = Counter(row["relation"] for row in etymology)
        audit["coverage"].update(
            {
                "n_total": len(concepts),
                "n_identifiable": len(identifiable),
                "n_shared": len(concepts) - len(identifiable),
                "n_related_identifiable": counts["BORROWING_DOCUMENTED"]
                + counts["SHARED_SOURCE_DOCUMENTED"],
                "n_distinct_routes_identifiable": counts[
                    "DISTINCT_ROUTES_REVIEWED"
                ],
                "relation_counts_identifiable": dict(counts),
            }
        )
        audit["status"] = "PASS"
        audit["reasons"] = []
        bundle["history"] = _history_pairs(remaining)
        manifest_path = _reseal(directory, bundle)
        with self.assertRaisesRegex(
            ContractViolation, "ANNOTATION_REVIEW_AUDIT_MISMATCH"
        ):
            verify_annotation_freeze(manifest_path, production=False)


if __name__ == "__main__":
    unittest.main()
