"""Materialize immutable lexical examples without exposing future languages."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from freshstart.core import canonical

from .contracts import (
    ContractViolation,
    LANGUAGES,
    PROJECT_ROOT,
    canonical_json_bytes,
    load_json,
    require_sha256,
)


EVALUATION_PLAN_PATH = PROJECT_ROOT / "implementation" / "config" / "evaluation_plan.json"
STAGE_ACTIVE_LANGUAGES = {
    "T0": ("ko",),
    "T1": ("ko", "en"),
    "T2": ("ko", "en", "zh"),
    "T3": LANGUAGES,
    "H": LANGUAGES,
}
REVERSE_LANGUAGE_ORDER = ("ko", "fr", "zh", "en")


def validated_language_order(order=LANGUAGES):
    order = tuple(order)
    if order not in (LANGUAGES, REVERSE_LANGUAGE_ORDER):
        raise ContractViolation("UNSUPPORTED_LANGUAGE_ORDER")
    return order


def stage_active_languages(stage, language_order=LANGUAGES):
    order = validated_language_order(language_order)
    if stage not in STAGE_ACTIVE_LANGUAGES:
        raise ContractViolation("UNKNOWN_TRAINING_STAGE")
    return order[:len(STAGE_ACTIVE_LANGUAGES[stage])]


@dataclass(frozen=True)
class RecordSlot:
    record_id: str
    concept_id: str
    input_language: str
    target_language: str
    mode: str
    wrapper: str
    occurrence: int
    kind: str = "lexical"

    def __post_init__(self) -> None:
        if not self.record_id or not self.concept_id or self.occurrence < 0:
            raise ContractViolation("INVALID_RECORD_SLOT")
        if self.input_language not in LANGUAGES or self.target_language not in LANGUAGES:
            raise ContractViolation("INVALID_RECORD_LANGUAGE")
        if self.mode not in {"ANY", "REQUESTED"}:
            raise ContractViolation("INVALID_RECORD_MODE")


@dataclass(frozen=True)
class TrainingRecord:
    record_id: str
    concept_id: str
    input_language: str
    target_language: str
    mode: str
    wrapper: str
    occurrence: int
    kind: str
    prefix: str
    canonical_answer: str
    terminator: str
    source_record_hashes: tuple[str, ...]
    incidental_exposure_counts: tuple[tuple[str, int], ...]
    content_sha256: str

    @property
    def full_text(self) -> str:
        return self.prefix + self.canonical_answer + self.terminator

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_record_hashes"] = list(self.source_record_hashes)
        value["incidental_exposure_counts"] = [list(row) for row in self.incidental_exposure_counts]
        return value


@dataclass(frozen=True)
class EncodedRecord:
    record_id: str
    content_sha256: str
    input_ids: tuple[int, ...]
    response_mask: tuple[bool, ...]


class StageConceptView:
    """A structural projection that refuses access to inactive languages."""

    def __init__(self, concept: Mapping[str, Any], active_languages: Sequence[str], *, language_order=LANGUAGES):
        active = tuple(active_languages)
        order = validated_language_order(language_order)
        if not active or active != order[: len(active)]:
            raise ContractViolation("INVALID_ACTIVE_LANGUAGE_PREFIX")
        if not isinstance(concept.get("concept_id"), str) or not concept["concept_id"]:
            raise ContractViolation("INVALID_CONCEPT")
        answers = concept.get("answers")
        glosses = concept.get("glosses")
        if not isinstance(answers, Mapping) or set(answers) != set(LANGUAGES):
            raise ContractViolation("FOUR_ANSWERS_REQUIRED")
        if not isinstance(glosses, Mapping) or set(glosses) != set(LANGUAGES):
            raise ContractViolation("FOUR_GLOSSES_REQUIRED")
        self.concept_id = concept["concept_id"]
        self.active_languages = active
        self._answers = {language: canonical(str(answers[language])) for language in active}
        self._glosses = {language: str(glosses[language]) for language in active}
        if any(not value.strip() for value in self._glosses.values()):
            raise ContractViolation("EMPTY_ACTIVE_GLOSS")
        raw_hashes = concept.get("source_record_hashes", concept.get("source_refs", ()))
        if isinstance(raw_hashes, Mapping):
            raw_hashes = [
                item.get("sha256")
                for item in raw_hashes.values()
                if isinstance(item, Mapping) and item.get("sha256")
            ]
        if not isinstance(raw_hashes, Sequence) or isinstance(raw_hashes, (str, bytes)):
            raise ContractViolation("MISSING_RECORD_PROVENANCE")
        self.source_record_hashes = tuple(sorted(str(value) for value in raw_hashes))
        if not self.source_record_hashes:
            raise ContractViolation("MISSING_RECORD_PROVENANCE")
        for value in self.source_record_hashes:
            require_sha256(value, "INVALID_SOURCE_RECORD_HASH")

    def answer(self, language: str) -> str:
        try:
            return self._answers[language]
        except KeyError as exc:
            raise ContractViolation("FUTURE_TARGET_LANGUAGE_ACCESS") from exc

    def gloss(self, language: str) -> str:
        try:
            return self._glosses[language]
        except KeyError as exc:
            raise ContractViolation("FUTURE_INPUT_LANGUAGE_ACCESS") from exc

    def active_answers(self) -> Mapping[str, str]:
        return dict(self._answers)


def _prompt_policy(path: Path = EVALUATION_PLAN_PATH) -> dict[str, Any]:
    try:
        return load_json(path)["prompt_plan"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ContractViolation("INVALID_PROMPT_POLICY") from exc


def _record_content_without_hash(
    slot: RecordSlot,
    prefix: str,
    answer: str,
    source_hashes: tuple[str, ...],
    incidental_exposure_counts: tuple[tuple[str, int], ...],
) -> dict[str, Any]:
    return {
        "record_id": slot.record_id,
        "concept_id": slot.concept_id,
        "input_language": slot.input_language,
        "target_language": slot.target_language,
        "mode": slot.mode,
        "wrapper": slot.wrapper,
        "occurrence": slot.occurrence,
        "kind": slot.kind,
        "prefix": prefix,
        "canonical_answer": answer,
        "terminator": "\n",
        "source_record_hashes": list(source_hashes),
        "incidental_exposure_counts": [list(row) for row in incidental_exposure_counts],
    }


def materialize_record(
    concept: Mapping[str, Any],
    slot: RecordSlot,
    *,
    stage: str,
    policy_path: Path = EVALUATION_PLAN_PATH,
    language_order=LANGUAGES,
) -> TrainingRecord:
    if stage not in STAGE_ACTIVE_LANGUAGES:
        raise ContractViolation("UNKNOWN_TRAINING_STAGE")
    active = stage_active_languages(stage, language_order)
    if slot.input_language not in active or slot.target_language not in active:
        raise ContractViolation("FUTURE_LANGUAGE_IN_TRAINING_SLOT")
    if concept.get("concept_id") != slot.concept_id:
        raise ContractViolation("RECORD_CONCEPT_MISMATCH")
    view = StageConceptView(concept, active, language_order=language_order)
    prompt_policy = _prompt_policy(policy_path)
    if slot.wrapper not in tuple(prompt_policy["training_wrappers"]):
        raise ContractViolation("NONTRAINING_WRAPPER")
    try:
        template = prompt_policy["templates"][slot.wrapper][slot.input_language][slot.mode]
        language_name = prompt_policy["target_language_names"][slot.input_language][slot.target_language]
    except (KeyError, TypeError) as exc:
        raise ContractViolation("MISSING_PROMPT_TEMPLATE") from exc
    prefix = str(template).format(
        gloss=view.gloss(slot.input_language),
        target_language_name=language_name,
    )
    if not prefix:
        raise ContractViolation("EMPTY_PROMPT_PREFIX")
    answer = view.answer(slot.target_language)
    incidental = tuple(
        (language, prefix.count(expression))
        for language, expression in sorted(view.active_answers().items())
        if prefix.count(expression) > 0
    )
    content = _record_content_without_hash(
        slot,
        prefix,
        answer,
        view.source_record_hashes,
        incidental,
    )
    digest = hashlib.sha256(canonical_json_bytes(content)).hexdigest()
    return TrainingRecord(
        **{key: content[key] for key in (
            "record_id",
            "concept_id",
            "input_language",
            "target_language",
            "mode",
            "wrapper",
            "occurrence",
            "kind",
            "prefix",
            "canonical_answer",
            "terminator",
        )},
        source_record_hashes=tuple(content["source_record_hashes"]),
        incidental_exposure_counts=tuple(
            (str(row[0]), int(row[1])) for row in content["incidental_exposure_counts"]
        ),
        content_sha256=digest,
    )


def _encode_no_special(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes)):
        raise ContractViolation("TOKENIZER_RETURNED_INVALID_IDS")
    try:
        result = [int(value) for value in encoded]
    except (TypeError, ValueError) as exc:
        raise ContractViolation("TOKENIZER_RETURNED_INVALID_IDS") from exc
    if any(value < 0 for value in result):
        raise ContractViolation("TOKENIZER_RETURNED_INVALID_IDS")
    return result


def encode_record(record: TrainingRecord, tokenizer: Any, context_length: int = 256) -> EncodedRecord:
    prefix_ids = _encode_no_special(tokenizer, record.prefix)
    full_ids = _encode_no_special(tokenizer, record.full_text)
    if not prefix_ids or full_ids[: len(prefix_ids)] != prefix_ids:
        raise ContractViolation("UNSTABLE_TOKEN_BOUNDARY")
    if len(full_ids) <= len(prefix_ids):
        raise ContractViolation("EMPTY_RESPONSE_TOKENS")
    if len(full_ids) > context_length:
        raise ContractViolation("TRAINING_RECORD_TOO_LONG")
    response_mask = [False] * len(prefix_ids) + [True] * (len(full_ids) - len(prefix_ids))
    return EncodedRecord(
        record_id=record.record_id,
        content_sha256=record.content_sha256,
        input_ids=tuple(full_ids),
        response_mask=tuple(response_mask),
    )


def materialize_record_bank(
    concepts: Sequence[Mapping[str, Any]],
    slots: Sequence[RecordSlot],
    *,
    stage: str,
    policy_path: Path = EVALUATION_PLAN_PATH,
    language_order=LANGUAGES,
) -> dict[str, TrainingRecord]:
    by_id = {str(concept.get("concept_id")): concept for concept in concepts}
    if len(by_id) != len(concepts):
        raise ContractViolation("DUPLICATE_CONCEPT_ID")
    bank: dict[str, TrainingRecord] = {}
    for slot in slots:
        if slot.record_id in bank:
            raise ContractViolation("DUPLICATE_RECORD_ID")
        try:
            concept = by_id[slot.concept_id]
        except KeyError as exc:
            raise ContractViolation("MISSING_RECORD_CONCEPT") from exc
        bank[slot.record_id] = materialize_record(
            concept,
            slot,
            stage=stage,
            policy_path=policy_path,
            language_order=language_order,
        )
    return bank
