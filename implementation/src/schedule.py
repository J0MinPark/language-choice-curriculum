"""Deterministic record and optimizer-batch plans for S and B36 history."""

from __future__ import annotations

import hashlib
import itertools
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

from .contracts import ContractViolation, LANGUAGES, canonical_json_bytes
from .records import RecordSlot, STAGE_ACTIVE_LANGUAGES, TrainingRecord, stage_active_languages, validated_language_order


TARGET_WEIGHTS: dict[str, tuple[str, ...]] = {
    "T0": ("ko",),
    "T1": ("ko", "en"),
    "T2": ("ko", "en", "zh", "zh"),
    "T3": ("ko", "en", "zh", "fr", "fr", "fr"),
}
MODES = ("ANY", "REQUESTED")
TRAIN_WRAPPERS = ("train1", "train2")


@dataclass(frozen=True)
class PlannedBatch:
    optimizer_step: int
    record_ids: tuple[str, ...]


@dataclass(frozen=True)
class StagePlan:
    root_id: int
    stage: str
    active_languages: tuple[str, ...]
    optimizer_updates: int
    effective_batch_examples: int
    slots: tuple[RecordSlot, ...]
    batches: tuple[PlannedBatch, ...]
    counts: Mapping[str, Any]
    plan_sha256: str


@dataclass(frozen=True)
class HistoryUse:
    branch: str
    round_id: int
    optimizer_step: int
    pair_index: int
    record_id: str
    concept_id: str
    target_language: str
    kind: str


@dataclass(frozen=True)
class HistoryBranchPlan:
    branch: str
    uses: tuple[HistoryUse, ...]
    batches: tuple[PlannedBatch, ...]
    plan_sha256: str


@dataclass(frozen=True)
class HistoryPlan:
    root_id: int
    assignment: Mapping[str, int]
    pairs: tuple[tuple[str, str], ...]
    record_slots: tuple[RecordSlot, ...]
    branches: Mapping[str, HistoryBranchPlan]
    record_bank_schema_sha256: str
    frozen_pair_set_sha256: str | None = None


def _seed(*parts: Any) -> int:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _stage_targets(stage, language_order=LANGUAGES):
    mapping = dict(zip(LANGUAGES, validated_language_order(language_order)))
    return tuple(mapping[l] for l in TARGET_WEIGHTS[stage])


def _stage_factor_cycle(stage: str, language_order=LANGUAGES) -> list[tuple[str, str, str, str]]:
    if stage not in TARGET_WEIGHTS:
        raise ContractViolation("UNKNOWN_TRAINING_STAGE")
    active = stage_active_languages(stage, language_order)
    return list(itertools.product(_stage_targets(stage, language_order), active, MODES, TRAIN_WRAPPERS))


def _plan_digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_stage_plan(
    concept_ids: Sequence[str],
    *,
    root_id: int,
    stage: str,
    optimizer_updates: int = 3000,
    effective_batch_examples: int = 32,
    concept_factor_balance: bool = False,
    language_order=LANGUAGES,
) -> StagePlan:
    ids = tuple(str(value) for value in concept_ids)
    if not ids or len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ContractViolation("INVALID_STAGE_CONCEPTS")
    if root_id not in {4101, 4102, 4103, 4104}:
        raise ContractViolation("INVALID_ROOT_ID")
    if optimizer_updates <= 0 or effective_batch_examples <= 0:
        raise ContractViolation("INVALID_STAGE_SIZE")
    factor_cycle = _stage_factor_cycle(stage, language_order)
    total = optimizer_updates * effective_batch_examples
    if total % len(factor_cycle) or total % len(ids):
        raise ContractViolation("STAGE_PLAN_NOT_EXACTLY_BALANCED")

    factor_rng = random.Random(_seed("factors", root_id, stage))
    factors: list[tuple[str, str, str, str]] = []
    for _ in range(total // len(factor_cycle)):
        cycle = list(factor_cycle)
        factor_rng.shuffle(cycle)
        factors.extend(cycle)

    concept_rng = random.Random(_seed("concepts", root_id, stage))
    planned_concepts: list[str] = []
    for _ in range(total // len(ids)):
        cycle = list(ids)
        concept_rng.shuffle(cycle)
        planned_concepts.extend(cycle)

    if concept_factor_balance:
        # Whole Cartesian cycles remove per-concept factor sampling imbalance.
        # No evaluation wrappers, error lists, or observed scores enter this plan.
        joint_cycle = list(itertools.product(sorted(ids), factor_cycle))
        if total % len(joint_cycle):
            raise ContractViolation("CONCEPT_FACTOR_CYCLE_NOT_EXACT")
        joint_rng = random.Random(_seed("concept-factor-balanced-v1", root_id, stage))
        joint = []
        for _ in range(total // len(joint_cycle)):
            cycle = list(joint_cycle)
            joint_rng.shuffle(cycle)
            joint.extend(cycle)
        planned_concepts = [c for c, _ in joint]
        factors = [f for _, f in joint]

    occurrences: Counter[tuple[str, str]] = Counter()
    slots: list[RecordSlot] = []
    for concept_id, (target, input_language, mode, wrapper) in zip(planned_concepts, factors):
        occurrence = occurrences[(concept_id, target)]
        occurrences[(concept_id, target)] += 1
        record_id = f"S|{root_id}|{stage}|{concept_id}|{target}|{occurrence}"
        slots.append(
            RecordSlot(
                record_id=record_id,
                concept_id=concept_id,
                input_language=input_language,
                target_language=target,
                mode=mode,
                wrapper=wrapper,
                occurrence=occurrence,
                kind="lexical",
            )
        )
    batches = tuple(
        PlannedBatch(step, tuple(slot.record_id for slot in slots[start : start + effective_batch_examples]))
        for step, start in enumerate(range(0, total, effective_batch_examples), 1)
    )
    counts = audit_stage_plan(slots, root_id=root_id, stage=stage, language_order=language_order)
    if concept_factor_balance:
        counts["concept_factor_balance"] = audit_concept_factor_balance(slots, stage=stage, language_order=language_order)
    payload = {
        "root_id": root_id,
        "stage": stage,
        "optimizer_updates": optimizer_updates,
        "effective_batch_examples": effective_batch_examples,
        "slots": [asdict(slot) for slot in slots],
        "batches": [asdict(batch) for batch in batches],
    }
    return StagePlan(
        root_id=root_id,
        stage=stage,
        active_languages=stage_active_languages(stage, language_order),
        optimizer_updates=optimizer_updates,
        effective_batch_examples=effective_batch_examples,
        slots=tuple(slots),
        batches=batches,
        counts=counts,
        plan_sha256=_plan_digest(payload),
    )


def audit_concept_factor_balance(slots, *, stage, language_order=LANGUAGES):
    cells = Counter((s.concept_id, s.target_language, s.input_language, s.mode, s.wrapper) for s in slots)
    ids = {s.concept_id for s in slots}
    factors = Counter(_stage_factor_cycle(stage, language_order))
    cycles, remainder = divmod(len(slots), len(ids) * sum(factors.values()))
    if remainder or not cycles or len(cells) != len(ids) * len(factors):
        raise ContractViolation("CONCEPT_FACTOR_BALANCE_MISMATCH")
    for concept in ids:
        for factor, weight in factors.items():
            if cells[(concept, *factor)] != cycles * weight:
                raise ContractViolation("CONCEPT_FACTOR_BALANCE_MISMATCH")
    return {"status": "PASS", "complete_cartesian_cycles": cycles,
            "distinct_cells": len(cells), "min_cell_exposures": min(cells.values()),
            "max_cell_exposures": max(cells.values())}


def audit_stage_plan(
    slots: Sequence[RecordSlot],
    *,
    root_id: int,
    stage: str,
    language_order=LANGUAGES,
) -> dict[str, Any]:
    if not slots or stage not in TARGET_WEIGHTS:
        raise ContractViolation("INVALID_STAGE_PLAN")
    active = stage_active_languages(stage, language_order)
    if len({slot.record_id for slot in slots}) != len(slots):
        raise ContractViolation("DUPLICATE_RECORD_ID")
    for slot in slots:
        expected_prefix = f"S|{root_id}|{stage}|"
        if not slot.record_id.startswith(expected_prefix):
            raise ContractViolation("STAGE_RECORD_LINEAGE_MISMATCH")
        if slot.input_language not in active or slot.target_language not in active:
            raise ContractViolation("FUTURE_LANGUAGE_IN_STAGE_PLAN")
        if slot.mode not in MODES or slot.wrapper not in TRAIN_WRAPPERS:
            raise ContractViolation("INVALID_STAGE_FACTOR")
    target = Counter(slot.target_language for slot in slots)
    target_input = Counter((slot.target_language, slot.input_language) for slot in slots)
    target_mode = Counter((slot.target_language, slot.mode) for slot in slots)
    target_wrapper = Counter((slot.target_language, slot.wrapper) for slot in slots)
    concept_counts = Counter(slot.concept_id for slot in slots)
    if len(set(concept_counts.values())) != 1:
        raise ContractViolation("CONCEPT_EXPOSURE_NOT_BALANCED")
    weights = Counter(_stage_targets(stage, language_order))
    unit = len(slots) // sum(weights.values())
    if any(target[language] != weight * unit for language, weight in weights.items()):
        raise ContractViolation("TARGET_RATIO_MISMATCH")
    for language in weights:
        expected_input = target[language] // len(active)
        expected_binary = target[language] // 2
        if any(target_input[(language, value)] != expected_input for value in active):
            raise ContractViolation("INPUT_NOT_INDEPENDENTLY_BALANCED")
        if any(target_mode[(language, value)] != expected_binary for value in MODES):
            raise ContractViolation("MODE_NOT_BALANCED_WITHIN_TARGET")
        if any(target_wrapper[(language, value)] != expected_binary for value in TRAIN_WRAPPERS):
            raise ContractViolation("WRAPPER_NOT_BALANCED_WITHIN_TARGET")
    return {
        "total_records": len(slots),
        "target": dict(sorted(target.items())),
        "target_input": {"|".join(key): value for key, value in sorted(target_input.items())},
        "target_mode": {"|".join(key): value for key, value in sorted(target_mode.items())},
        "target_wrapper": {"|".join(key): value for key, value in sorted(target_wrapper.items())},
        "concept": dict(sorted(concept_counts.items())),
    }


def make_history_assignment(
    pairs: Sequence[tuple[str, str]],
    *,
    root_id: int,
) -> dict[str, int]:
    normalized = tuple((str(a), str(b)) for a, b in pairs)
    if root_id not in {4101, 4102, 4103, 4104}:
        raise ContractViolation("INVALID_ROOT_ID")
    flat = [concept for pair in normalized for concept in pair]
    if not normalized or len(flat) != len(set(flat)) or any(not value for value in flat):
        raise ContractViolation("INVALID_HISTORY_PAIRS")
    rng = random.Random(_seed("B36-assignment", root_id))
    assignment: dict[str, int] = {}
    for first, second in normalized:
        sign = rng.choice((-1, 1))
        assignment[first] = sign
        assignment[second] = -sign
    if sum(assignment.values()) != 0:
        raise ContractViolation("UNBALANCED_HISTORY_ASSIGNMENT")
    return assignment


def _history_record_slot(root_id: int, concept_id: str, language: str, occurrence: int) -> RecordSlot:
    offset = _seed("H-record-factors", root_id, concept_id, language) % 40
    position = (occurrence + offset) % 40
    input_language = LANGUAGES[position % 4]
    mode = MODES[(position // 4) % 2]
    wrapper = TRAIN_WRAPPERS[(position // 2) % 2]
    kind = "tail" if occurrence >= 36 else ("mutable" if language in {"en", "fr"} else "anchor")
    return RecordSlot(
        record_id=f"H|{root_id}|{concept_id}|{language}|{occurrence}",
        concept_id=concept_id,
        input_language=input_language,
        target_language=language,
        mode=mode,
        wrapper=wrapper,
        occurrence=occurrence,
        kind=kind,
    )


def _history_round_languages(
    assignment: Mapping[str, int],
    branch: str,
) -> list[tuple[str, Mapping[str, str]]]:
    rows: list[tuple[str, Mapping[str, str]]] = []
    for mutable_round in range(72):
        languages = {}
        for concept_id, sign in assignment.items():
            orientation = sign * (1 if branch == "A" else -1)
            languages[concept_id] = (
                ("en" if mutable_round < 36 else "fr")
                if orientation == 1
                else ("fr" if mutable_round < 36 else "en")
            )
        rows.append(("mutable", languages))
        if mutable_round % 2 == 1:
            rows.append(("anchor", {concept_id: "ko" for concept_id in assignment}))
            rows.append(("anchor", {concept_id: "zh" for concept_id in assignment}))
    for _ in range(4):
        for language in LANGUAGES:
            rows.append(("tail", {concept_id: language for concept_id in assignment}))
    if len(rows) != 160:
        raise ContractViolation("INVALID_HISTORY_ROUND_COUNT")
    return rows


def _build_history_branch(
    *,
    root_id: int,
    pairs: tuple[tuple[str, str], ...],
    assignment: Mapping[str, int],
    branch: str,
    effective_batch_examples: int,
) -> HistoryBranchPlan:
    if branch not in {"A", "B"} or effective_batch_examples % 2:
        raise ContractViolation("INVALID_HISTORY_BATCH_POLICY")
    occurrences: Counter[tuple[str, str]] = Counter()
    pair_units: list[tuple[int, int, str, tuple[str, str], tuple[str, str]]] = []
    round_rows = _history_round_languages(assignment, branch)
    for round_id, (kind, languages) in enumerate(round_rows):
        for pair_index, pair in enumerate(pairs):
            record_ids = []
            target_languages = []
            for concept_id in pair:
                language = languages[concept_id]
                occurrence = occurrences[(concept_id, language)]
                occurrences[(concept_id, language)] += 1
                record_ids.append(f"H|{root_id}|{concept_id}|{language}|{occurrence}")
                target_languages.append(language)
            pair_units.append(
                (round_id, pair_index, kind, tuple(record_ids), tuple(target_languages))
            )
    pairs_per_batch = effective_batch_examples // 2
    if len(pair_units) % pairs_per_batch:
        raise ContractViolation("HISTORY_BATCH_NOT_EXACT")
    uses: list[HistoryUse] = []
    batches: list[PlannedBatch] = []
    for step, start in enumerate(range(0, len(pair_units), pairs_per_batch), 1):
        units = pair_units[start : start + pairs_per_batch]
        batch_ids: list[str] = []
        for round_id, pair_index, kind, record_ids, target_languages in units:
            for record_id, concept_id, language in zip(
                record_ids, pairs[pair_index], target_languages
            ):
                uses.append(
                    HistoryUse(
                        branch=branch,
                        round_id=round_id,
                        optimizer_step=step,
                        pair_index=pair_index,
                        record_id=record_id,
                        concept_id=concept_id,
                        target_language=language,
                        kind=kind,
                    )
                )
                batch_ids.append(record_id)
        batches.append(PlannedBatch(step, tuple(batch_ids)))
    payload = {
        "root_id": root_id,
        "branch": branch,
        "uses": [asdict(use) for use in uses],
        "batches": [asdict(batch) for batch in batches],
    }
    return HistoryBranchPlan(branch, tuple(uses), tuple(batches), _plan_digest(payload))


def build_history_plan(
    pairs: Sequence[tuple[str, str]],
    *,
    root_id: int,
    effective_batch_examples: int = 32,
) -> HistoryPlan:
    normalized = tuple((str(a), str(b)) for a, b in pairs)
    assignment = make_history_assignment(normalized, root_id=root_id)
    slots = tuple(
        _history_record_slot(root_id, concept_id, language, occurrence)
        for concept_id in assignment
        for language in LANGUAGES
        for occurrence in range(40)
    )
    branches = {
        branch: _build_history_branch(
            root_id=root_id,
            pairs=normalized,
            assignment=assignment,
            branch=branch,
            effective_batch_examples=effective_batch_examples,
        )
        for branch in ("A", "B")
    }
    schema_hash = _plan_digest([asdict(slot) for slot in slots])
    plan = HistoryPlan(root_id, assignment, normalized, slots, branches, schema_hash)
    audit_history_plan(plan, effective_batch_examples=effective_batch_examples)
    return plan


def build_history_plan_from_verified_freeze(
    verified_freeze: Mapping[str, Any],
    *,
    root_id: int,
    effective_batch_examples: int = 32,
) -> HistoryPlan:
    """Build production B36 only from the pairing artifact in a verified freeze.

    ``build_history_plan`` remains available for synthetic schedule tests.  A
    production caller must use this boundary so an ad-hoc concept order or pair
    list cannot silently replace the preregistered frozen assignment units.
    """
    if (
        verified_freeze.get("verified") is not True
        or verified_freeze.get("status") != "PASS"
        or verified_freeze.get("freeze_gate_status") != "PASS"
        or verified_freeze.get("data_kind") != "REAL"
    ):
        raise ContractViolation("HISTORY_REQUIRES_VERIFIED_REAL_FREEZE")
    concepts = verified_freeze.get("concepts")
    history = verified_freeze.get("history_pairs")
    if not isinstance(concepts, Sequence) or isinstance(concepts, (str, bytes)):
        raise ContractViolation("INVALID_FROZEN_HISTORY_PAIRS")
    if not isinstance(history, Mapping):
        raise ContractViolation("INVALID_FROZEN_HISTORY_PAIRS")
    try:
        concept_ids = [str(row["concept_id"]) for row in concepts]
    except (KeyError, TypeError) as exc:
        raise ContractViolation("INVALID_FROZEN_HISTORY_PAIRS") from exc
    ordered_ids = sorted(concept_ids)
    if (
        len(ordered_ids) != 60
        or len(set(ordered_ids)) != 60
        or history.get("status") != "PASS"
        or history.get("history_family") != "B36"
        or history.get("concept_order") != ordered_ids
        or history.get("pairing_rule")
        != "lexicographically sort frozen concept_id and pair adjacent positions (0,1),(2,3),..."
    ):
        raise ContractViolation("INVALID_FROZEN_HISTORY_PAIRS")
    pair_set_hash = history.get("history_pair_set_sha256")
    if not isinstance(pair_set_hash, str) or pair_set_hash != _plan_digest(
        {key: value for key, value in history.items() if key != "history_pair_set_sha256"}
    ):
        raise ContractViolation("FROZEN_HISTORY_PAIR_HASH_MISMATCH")
    rows = history.get("pairs")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) != 30:
        raise ContractViolation("INVALID_FROZEN_HISTORY_PAIRS")
    normalized: list[tuple[str, str]] = []
    for pair_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ContractViolation("INVALID_FROZEN_HISTORY_PAIRS")
        pair_hash = row.get("pair_sha256")
        pair_core = {key: value for key, value in row.items() if key != "pair_sha256"}
        expected_ids = ordered_ids[pair_index * 2 : pair_index * 2 + 2]
        if (
            not isinstance(pair_hash, str)
            or pair_hash != _plan_digest(pair_core)
            or row.get("pair_id") != f"B36-pair-{pair_index:02d}"
            or row.get("pair_index") != pair_index
            or row.get("concept_ids") != expected_ids
        ):
            raise ContractViolation("FROZEN_HISTORY_PAIR_HASH_MISMATCH")
        normalized.append((expected_ids[0], expected_ids[1]))
    plan = build_history_plan(
        normalized,
        root_id=root_id,
        effective_batch_examples=effective_batch_examples,
    )
    return replace(plan, frozen_pair_set_sha256=pair_set_hash)


def audit_history_plan(plan: HistoryPlan, *, effective_batch_examples: int = 32) -> dict[str, Any]:
    a, b = plan.branches["A"], plan.branches["B"]
    expected_records = len(plan.pairs) * 2 * 160
    if not plan.pairs or len(a.uses) != expected_records or len(b.uses) != expected_records:
        raise ContractViolation("HISTORY_RECORD_COUNT_MISMATCH")
    if Counter(use.record_id for use in a.uses) != Counter(use.record_id for use in b.uses):
        raise ContractViolation("HISTORY_RECORD_MULTISET_MISMATCH")
    if len(set(use.record_id for use in a.uses)) != expected_records:
        raise ContractViolation("HISTORY_RECORD_NOT_USED_ONCE")
    for branch_plan in (a, b):
        if {use.round_id for use in branch_plan.uses} != set(range(160)):
            raise ContractViolation("INVALID_HISTORY_ROUNDS")
        if any(len(batch.record_ids) != effective_batch_examples for batch in branch_plan.batches):
            raise ContractViolation("INVALID_HISTORY_BATCH_SIZE")
        by_round_pair: dict[tuple[int, int], list[HistoryUse]] = defaultdict(list)
        for use in branch_plan.uses:
            by_round_pair[(use.round_id, use.pair_index)].append(use)
        if any(
            len(pair_uses) != 2 or len({use.optimizer_step for use in pair_uses}) != 1
            for pair_uses in by_round_pair.values()
        ):
            raise ContractViolation("HISTORY_PAIR_SPLIT_ACROSS_BATCH")
        counts = Counter((use.concept_id, use.target_language) for use in branch_plan.uses)
        if set(counts.values()) != {40}:
            raise ContractViolation("HISTORY_TARGET_COUNT_MISMATCH")
    for kind in ("anchor", "tail"):
        aa = [(use.round_id, use.optimizer_step, use.record_id) for use in a.uses if use.kind == kind]
        bb = [(use.round_id, use.optimizer_step, use.record_id) for use in b.uses if use.kind == kind]
        if aa != bb:
            raise ContractViolation(kind.upper() + "_HISTORY_DIFFERS")
    def last_step(branch_plan: HistoryBranchPlan) -> dict[tuple[str, str], int]:
        result: dict[tuple[str, str], int] = {}
        for use in branch_plan.uses:
            result[(use.concept_id, use.target_language)] = use.optimizer_step
        return result
    if last_step(a) != last_step(b):
        raise ContractViolation("LAST_TARGET_OPTIMIZER_STEP_DIFFERS")
    return {
        "status": "PASS",
        "concepts": len(plan.assignment),
        "rounds": 160,
        "target_records_per_branch": expected_records,
        "optimizer_updates_per_branch": len(a.batches),
        "per_concept_per_language": 40,
        "frozen_pair_set_sha256": plan.frozen_pair_set_sha256,
    }


def audit_history_record_content(
    plan: HistoryPlan,
    record_bank: Mapping[str, TrainingRecord],
) -> dict[str, Any]:
    expected_ids = {slot.record_id for slot in plan.record_slots}
    if set(record_bank) != expected_ids:
        raise ContractViolation("HISTORY_RECORD_BANK_MISMATCH")
    for record_id, record in record_bank.items():
        if record.record_id != record_id:
            raise ContractViolation("HISTORY_RECORD_CONTENT_ID_MISMATCH")
    slot_kind = {slot.record_id: slot.kind for slot in plan.record_slots}
    for branch_plan in plan.branches.values():
        if any(slot_kind.get(use.record_id) != use.kind for use in branch_plan.uses):
            raise ContractViolation("HISTORY_RECORD_KIND_MISMATCH")
    marginal_counts: dict[str, Any] = {}
    by_concept_target: dict[tuple[str, str], list[TrainingRecord]] = defaultdict(list)
    for record in record_bank.values():
        by_concept_target[(record.concept_id, record.target_language)].append(record)
    for key, records in by_concept_target.items():
        inputs = Counter(record.input_language for record in records)
        modes = Counter(record.mode for record in records)
        wrappers = Counter(record.wrapper for record in records)
        if set(inputs.values()) != {10} or set(modes.values()) != {20} or set(wrappers.values()) != {20}:
            raise ContractViolation("HISTORY_RECORD_FACTORS_NOT_BALANCED")
        marginal_counts["|".join(key)] = {
            "input": dict(sorted(inputs.items())),
            "mode": dict(sorted(modes.items())),
            "wrapper": dict(sorted(wrappers.items())),
        }
    counters = {}
    for branch, branch_plan in plan.branches.items():
        counters[branch] = Counter(
            (use.record_id, record_bank[use.record_id].content_sha256)
            for use in branch_plan.uses
        )
    if counters["A"] != counters["B"]:
        raise ContractViolation("HISTORY_CONTENT_MULTISET_MISMATCH")
    return {
        "status": "PASS",
        "record_bank_schema_sha256": plan.record_bank_schema_sha256,
        "distinct_record_content": len(record_bank),
        "factor_marginals": marginal_counts,
    }
