"""Exact registered-expression generation and continuation scoring.

The scorer accepts an already-loaded, already-device-bound causal LM.  It does
not select a device or mutate checkpoint/training state.  Probability events
are the tokenizer's single canonical continuation for ``answer + "\n"``;
they are not length-normalized and do not marginalize alternate segmentations.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from freshstart.core import ContractError as ReferenceContractError
from freshstart.core import (
    canonical as _reference_canonical,
    event_partition as _reference_event_partition,
    membership as _reference_membership,
    validate_token_events as _reference_validate_token_events,
)

from .artifacts import (
    canonical_json_bytes,
    read_verified_json,
    sha256_bytes,
    sha256_file,
)
from .contracts import ContractViolation as ContractError, LANGUAGES, VERSION


HASH_RE = re.compile(r"[0-9a-f]{64}")
REQUIRED_PROVENANCE = (
    "root_id",
    "phase",
    "stage",
    "branch",
    "checkpoint_sha256",
    "model_fingerprint",
    "freeze_sha256",
    "config_sha256",
    "tokenizer_sha256",
    "code_sha256",
    "evaluation_plan_sha256",
    "evaluation_record_hash_schema",
    "data_kind",
    "context_length",
    "maximum_new_tokens",
    "generation_strategy",
    "terminator",
    "candidate_universe",
)
REQUIRED_RECORD_FIELDS = (
    "record_id",
    "concept_id",
    "input_language",
    "format",
    "wrapper",
    "split",
    "mode",
    "prefix",
    "answers",
)

EVALUATION_RECORD_HASH_SCHEMA = "score-record-v1"
EVALUATION_RECORD_HASH_FIELDS = REQUIRED_RECORD_FIELDS
STAGE_ACTIVE_LANGUAGES = {
    "T0": LANGUAGES[:1],
    "T1": LANGUAGES[:2],
    "T2": LANGUAGES[:3],
    "T3": LANGUAGES,
    "H": LANGUAGES,
    "H_BASE": LANGUAGES,
    "H_A": LANGUAGES,
    "H_B": LANGUAGES,
}


def _translate_reference_contract(callable_: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return callable_(*args, **kwargs)
    except ReferenceContractError as exc:
        raise ContractError(str(exc)) from exc


def canonical(text: str) -> str:
    return _translate_reference_contract(_reference_canonical, text)


def membership(answers: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    return _translate_reference_contract(_reference_membership, answers)


def event_partition(
    answers: Mapping[str, str], logp_by_text: Mapping[str, float]
) -> dict[str, Any]:
    return _translate_reference_contract(
        _reference_event_partition, answers, logp_by_text
    )


def validate_token_events(events: Mapping[str, Sequence[int]]) -> None:
    _translate_reference_contract(_reference_validate_token_events, events)


def _require_sha256(value: Any, code: str) -> str:
    normalized = str(value).lower()
    if not HASH_RE.fullmatch(normalized):
        raise ContractError(code)
    return normalized


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)
    ids = encoded.ids if hasattr(encoded, "ids") else encoded
    try:
        result = [int(token_id) for token_id in ids]
    except (TypeError, ValueError) as exc:
        raise ContractError("TOKENIZER_DID_NOT_RETURN_TOKEN_IDS") from exc
    if any(token_id < 0 for token_id in result):
        raise ContractError("NEGATIVE_TOKEN_ID")
    return result


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        decoded = tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        try:
            decoded = tokenizer.decode(list(token_ids), skip_special_tokens=False)
        except TypeError:
            decoded = tokenizer.decode(list(token_ids))
    if not isinstance(decoded, str):
        raise ContractError("TOKENIZER_DID_NOT_RETURN_TEXT")
    return decoded


def semantic_model_fingerprint(model: Any) -> str:
    """Hash parameter/buffer names, dtypes, shapes and exact bytes."""
    import torch

    try:
        state = model.state_dict()
    except AttributeError as exc:
        raise ContractError("MODEL_STATE_DICT_REQUIRED") from exc
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(
            tensor.reshape(-1)
            .view(dtype=torch.uint8)
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


def rng_fingerprint(generators: Mapping[str, Any] | None = None) -> str:
    """Fingerprint all process RNGs whose preservation the checkpoint requires."""
    try:
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover - training environment contract
        raise ContractError("TORCH_NUMPY_REQUIRED_FOR_SCORING") from exc

    digest = hashlib.sha256()
    digest.update(repr(random.getstate()).encode("utf-8"))
    np_state = np.random.get_state()
    digest.update(str(np_state[0]).encode("ascii"))
    digest.update(np_state[1].tobytes())
    digest.update(repr(np_state[2:]).encode("ascii"))

    def add_tensor(label: str, state: Any) -> None:
        tensor = state.detach().cpu().contiguous()
        digest.update(label.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())

    add_tensor("torch_cpu", torch.get_rng_state())
    if torch.cuda.is_available():
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            add_tensor(f"torch_cuda_{index}", state)
    for name, generator in sorted((generators or {}).items()):
        add_tensor("generator_" + name, generator.get_state())
    return digest.hexdigest()


@contextmanager
def preserved_evaluation_state(
    model: Any,
    generators: Mapping[str, Any] | None = None,
):
    """Enter inference mode and prove that scoring consumed no training RNG."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - training environment contract
        raise ContractError("TORCH_REQUIRED_FOR_SCORING") from exc
    before = rng_fingerprint(generators)
    module_training_states = (
        [(module, bool(module.training)) for module in model.modules()]
        if hasattr(model, "modules")
        else []
    )
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.inference_mode():
            yield before
    finally:
        for module, training_state in module_training_states:
            module.training = training_state
        after = rng_fingerprint(generators)
        if after != before:
            raise ContractError("EVALUATION_CONSUMED_TRAINING_RNG")


def build_continuation_events(
    tokenizer: Any,
    prefix: str,
    answers: Mapping[str, str],
    *,
    context_length: int,
) -> list[dict[str, Any]]:
    """Build one context-specific token event per distinct canonical string."""
    if not isinstance(prefix, str):
        raise ContractError("PREFIX_MUST_BE_TEXT")
    if not isinstance(context_length, int) or context_length < 2:
        raise ContractError("INVALID_CONTEXT_LENGTH")
    grouped = membership(answers)
    prefix_ids = _encode(tokenizer, prefix)
    if not prefix_ids:
        raise ContractError("EMPTY_PREFIX_TOKENS")
    events: list[dict[str, Any]] = []
    token_events: dict[str, list[int]] = {}
    for text, languages in grouped.items():
        full_ids = _encode(tokenizer, prefix + text + "\n")
        if full_ids[: len(prefix_ids)] != prefix_ids or len(full_ids) <= len(prefix_ids):
            raise ContractError("UNSTABLE_TOKEN_BOUNDARY")
        if len(full_ids) > context_length:
            raise ContractError("SCORING_SEQUENCE_EXCEEDS_CONTEXT")
        continuation_ids = full_ids[len(prefix_ids) :]
        token_events[text] = continuation_ids
        events.append(
            {
                "canonical_text": text,
                "membership": list(languages),
                "prefix_token_ids": list(prefix_ids),
                "continuation_token_ids": continuation_ids,
                "continuation_token_count": len(continuation_ids),
                "full_token_count": len(full_ids),
            }
        )
    validate_token_events(token_events)
    return events


def _extract_logits(output: Any) -> Any:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, Mapping) and "logits" in output:
        return output["logits"]
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise ContractError("MODEL_OUTPUT_HAS_NO_LOGITS")


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        try:
            return next(model.buffers()).device
        except (AttributeError, StopIteration):
            return "cpu"


def _continuation_logprob(
    model: Any,
    prefix_ids: Sequence[int],
    continuation_ids: Sequence[int],
) -> dict[str, Any]:
    import torch

    prefix = [int(value) for value in prefix_ids]
    continuation = [int(value) for value in continuation_ids]
    if not prefix:
        raise ContractError("EMPTY_PREFIX_TOKENS")
    if not continuation:
        raise ContractError("EMPTY_TOKEN_EVENT")
    full = prefix + continuation
    inputs = torch.tensor([full[:-1]], dtype=torch.long, device=_model_device(model))
    try:
        output = model(input_ids=inputs, use_cache=False)
    except TypeError:
        output = model(input_ids=inputs)
    logits = _extract_logits(output)
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < len(full) - 1:
        raise ContractError("INVALID_MODEL_LOGIT_SHAPE")
    start = len(prefix) - 1
    stop = start + len(continuation)
    selected = logits[0, start:stop, :]
    targets = torch.tensor(continuation, dtype=torch.long, device=selected.device)
    if selected.shape[0] != targets.shape[0]:
        raise ContractError("LOGIT_TARGET_LENGTH_MISMATCH")
    if targets.numel() and int(targets.max()) >= selected.shape[-1]:
        raise ContractError("TOKEN_ID_OUTSIDE_MODEL_VOCAB")
    log_probs = torch.log_softmax(selected.float(), dim=-1)
    values = log_probs.gather(1, targets[:, None]).squeeze(1).double().cpu().tolist()
    token_log_probs = [float(value) for value in values]
    if any(math.isnan(value) or value > 1e-10 for value in token_log_probs):
        raise ContractError("INVALID_TOKEN_LOG_PROBABILITY")
    return {
        "token_log_probabilities": token_log_probs,
        "log_p": math.fsum(token_log_probs),
    }


def continuation_logprob(
    model: Any,
    prefix_ids: Sequence[int],
    continuation_ids: Sequence[int],
    *,
    generators: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    with preserved_evaluation_state(model, generators):
        return _continuation_logprob(model, prefix_ids, continuation_ids)


def partition_probability_events(
    answers: Mapping[str, str],
    logp_by_text: Mapping[str, float],
) -> dict[str, Any]:
    """Public pure formula layer, retaining finite-log underflow semantics."""
    result = event_partition(answers, logp_by_text)
    shared_mass = None
    if result["Q_membership"] is not None:
        shared_mass = math.fsum(
            mass
            for label, mass in result["Q_membership"].items()
            if "+" in label
        )
    return {
        **result,
        "shared_mass": shared_mass,
        "probability_event_definition": "P(canonical(answer + newline) | exact prefix, ANY)",
        "length_normalized": False,
        "alternate_segmentations_marginalized": False,
    }


def _score_probability_partition(
    model: Any,
    tokenizer: Any,
    prefix: str,
    answers: Mapping[str, str],
    *,
    context_length: int,
) -> dict[str, Any]:
    events = build_continuation_events(
        tokenizer, prefix, answers, context_length=context_length
    )
    logp_by_text: dict[str, float] = {}
    scored_events: list[dict[str, Any]] = []
    for event in events:
        score = _continuation_logprob(
            model,
            event["prefix_token_ids"],
            event["continuation_token_ids"],
        )
        logp = score["log_p"]
        logp_by_text[event["canonical_text"]] = logp
        scored_events.append(
            {
                **event,
                **score,
                "raw_p": math.exp(logp),
            }
        )
    partition = partition_probability_events(answers, logp_by_text)
    return {**partition, "events": scored_events}


def score_probability_partition(
    model: Any,
    tokenizer: Any,
    prefix: str,
    answers: Mapping[str, str],
    *,
    context_length: int,
    generators: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    with preserved_evaluation_state(model, generators):
        return _score_probability_partition(
            model,
            tokenizer,
            prefix,
            answers,
            context_length=context_length,
        )


def classify_first_line(
    decoded_continuation: str,
    answers: Mapping[str, str],
    requested_language: str | None,
) -> dict[str, Any]:
    if requested_language is not None and requested_language not in LANGUAGES:
        raise ContractError("UNKNOWN_REQUESTED_LANGUAGE")
    normalized_newlines = decoded_continuation.replace("\r\n", "\n").replace("\r", "\n")
    first_line = normalized_newlines.split("\n", 1)[0]
    if not first_line.strip():
        return {
            "class": "EMPTY",
            "first_line": first_line,
            "canonical_first_line": None,
            "registered_match": False,
            "membership": [],
            "request_compatible": False if requested_language is not None else None,
            "registered_other_language": False if requested_language is not None else None,
            "empty": True,
            "unregistered": False,
        }
    try:
        normalized = canonical(first_line)
    except ContractError:
        normalized = None
    languages = membership(answers).get(normalized, ()) if normalized is not None else ()
    if languages:
        compatible = requested_language in languages if requested_language is not None else None
        label = (
            "REGISTERED"
            if requested_language is None
            else (
                "REGISTERED_COMPATIBLE"
                if compatible
                else "REGISTERED_OTHER_LANGUAGE"
            )
        )
    else:
        compatible = False if requested_language is not None else None
        label = "UNREGISTERED"
    return {
        "class": label,
        "first_line": first_line,
        "canonical_first_line": normalized,
        "registered_match": bool(languages),
        "membership": list(languages),
        "request_compatible": compatible,
        "registered_other_language": (
            bool(languages) and not bool(compatible)
            if requested_language is not None
            else None
        ),
        "empty": False,
        "unregistered": not bool(languages),
    }


def _greedy_first_line(
    model: Any,
    tokenizer: Any,
    prefix: str,
    answers: Mapping[str, str],
    requested_language: str | None,
    *,
    context_length: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ContractError("INVALID_MAX_NEW_TOKENS")
    prefix_ids = _encode(tokenizer, prefix)
    if not prefix_ids:
        raise ContractError("EMPTY_PREFIX_TOKENS")
    if len(prefix_ids) >= context_length:
        raise ContractError("GENERATION_PREFIX_EXCEEDS_CONTEXT")
    generated: list[int] = []
    termination = "MAX_NEW_TOKENS"
    for _ in range(max_new_tokens):
        current = prefix_ids + generated
        if len(current) >= context_length:
            termination = "CONTEXT_LIMIT"
            break
        inputs = torch.tensor([current], dtype=torch.long, device=_model_device(model))
        try:
            output = model(input_ids=inputs, use_cache=False)
        except TypeError:
            output = model(input_ids=inputs)
        logits = _extract_logits(output)
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < len(current):
            raise ContractError("INVALID_MODEL_LOGIT_SHAPE")
        next_token = int(torch.argmax(logits[0, len(current) - 1, :]).item())
        generated.append(next_token)
        if "\n" in _decode(tokenizer, generated).replace("\r\n", "\n").replace("\r", "\n"):
            termination = "NEWLINE"
            break
    decoded = _decode(tokenizer, generated)
    classified = classify_first_line(decoded, answers, requested_language)
    if termination != "NEWLINE":
        classified = {
            **classified,
            "class": "UNREGISTERED_UNTERMINATED",
            "registered_match": False,
            "membership": [],
            "request_compatible": False if requested_language is not None else None,
            "registered_other_language": False if requested_language is not None else None,
            "empty": False,
            "unregistered": True,
        }
    return {
        **classified,
        "generated_token_ids": generated,
        "generated_token_count": len(generated),
        "decoded_continuation": decoded,
        "termination": termination,
    }


def greedy_first_line(
    model: Any,
    tokenizer: Any,
    prefix: str,
    answers: Mapping[str, str],
    requested_language: str | None,
    *,
    context_length: int,
    max_new_tokens: int,
    generators: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    with preserved_evaluation_state(model, generators):
        return _greedy_first_line(
            model,
            tokenizer,
            prefix,
            answers,
            requested_language,
            context_length=context_length,
            max_new_tokens=max_new_tokens,
        )


def aggregate_generation(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    mode_counts: dict[str, Counter[str]] = {}
    language_counts: dict[str, Counter[str]] = {
        language: Counter() for language in LANGUAGES
    }
    seen = 0
    for row in rows:
        mode = row.get("mode")
        generation = row.get("generation")
        if mode not in {"ANY", "REQUESTED"} or not isinstance(generation, Mapping):
            raise ContractError("INVALID_GENERATION_ROW")
        label = generation.get("class")
        if label not in {
            "EMPTY",
            "UNREGISTERED",
            "UNREGISTERED_UNTERMINATED",
            "REGISTERED",
            "REGISTERED_COMPATIBLE",
            "REGISTERED_OTHER_LANGUAGE",
        }:
            raise ContractError("UNKNOWN_GENERATION_CLASS")
        if mode == "ANY" and label not in {
            "REGISTERED",
            "UNREGISTERED",
            "UNREGISTERED_UNTERMINATED",
            "EMPTY",
        }:
            raise ContractError("REQUEST_LABEL_IN_ANY_MODE")
        if mode == "REQUESTED" and label not in {
            "REGISTERED_COMPATIBLE",
            "REGISTERED_OTHER_LANGUAGE",
            "UNREGISTERED",
            "UNREGISTERED_UNTERMINATED",
            "EMPTY",
        }:
            raise ContractError("ANY_LABEL_IN_REQUESTED_MODE")
        expected_registered = label in {
            "REGISTERED",
            "REGISTERED_COMPATIBLE",
            "REGISTERED_OTHER_LANGUAGE",
        }
        if bool(generation.get("registered_match")) != expected_registered:
            raise ContractError("INCONSISTENT_REGISTERED_MATCH")
        bucket = mode_counts.setdefault(mode, Counter())
        bucket["total"] += 1
        bucket[label] += 1
        if bool(generation.get("registered_match")):
            bucket["registered"] += 1
        if mode == "REQUESTED":
            requested = row.get("requested_language")
            if requested not in LANGUAGES:
                raise ContractError("REQUESTED_MODE_REQUIRES_LANGUAGE")
            language_counts[requested]["total"] += 1
            language_counts[requested][label] += 1
        elif row.get("requested_language") is not None:
            raise ContractError("ANY_MODE_MUST_NOT_REQUEST_LANGUAGE")
        seen += 1
    if not seen:
        raise ContractError("EMPTY_GENERATION_ROWS")

    s_by_mode: dict[str, Any] = {}
    for mode in sorted(mode_counts):
        counts = mode_counts[mode]
        s_by_mode[mode] = {
            "registered_numerator": counts["registered"],
            "denominator": counts["total"],
            "S": counts["registered"] / counts["total"],
            "classes": {
                label: counts[label]
                for label in (
                    "REGISTERED",
                    "REGISTERED_COMPATIBLE",
                    "REGISTERED_OTHER_LANGUAGE",
                    "UNREGISTERED",
                    "UNREGISTERED_UNTERMINATED",
                    "EMPTY",
                )
            },
        }
    by_language: dict[str, Any] = {}
    for language in LANGUAGES:
        counts = language_counts[language]
        if not counts["total"]:
            continue
        by_language[language] = {
            "denominator": counts["total"],
            "A_numerator": counts["REGISTERED_COMPATIBLE"],
            "C_numerator": counts["REGISTERED_OTHER_LANGUAGE"],
            "unregistered_numerator": counts["UNREGISTERED"],
            "unterminated_numerator": counts["UNREGISTERED_UNTERMINATED"],
            "empty_numerator": counts["EMPTY"],
            "A": counts["REGISTERED_COMPATIBLE"] / counts["total"],
            "C": counts["REGISTERED_OTHER_LANGUAGE"] / counts["total"],
        }
    return {"S_by_mode": s_by_mode, "requested_by_language": by_language}


def evaluation_record_content_sha256(record: Mapping[str, Any]) -> str:
    """Hash exactly the projection frozen as ``score-record-v1``."""
    missing = [key for key in EVALUATION_RECORD_HASH_FIELDS if key not in record]
    if missing:
        raise ContractError("MISSING_EVALUATION_FIELD:" + missing[0])
    content = {key: record[key] for key in EVALUATION_RECORD_HASH_FIELDS}
    if "requested_language" in record:
        content["requested_language"] = record.get("requested_language")
    return sha256_bytes(canonical_json_bytes(content))


def _validate_provenance(
    metadata: Mapping[str, Any], *, expected_data_kind: str
) -> dict[str, Any]:
    missing = [key for key in REQUIRED_PROVENANCE if key not in metadata]
    if missing:
        raise ContractError("MISSING_SCORER_PROVENANCE:" + missing[0])
    normalized = dict(metadata)
    for key in (
        "checkpoint_sha256",
        "model_fingerprint",
        "freeze_sha256",
        "config_sha256",
        "tokenizer_sha256",
        "code_sha256",
        "evaluation_plan_sha256",
    ):
        normalized[key] = _require_sha256(metadata[key], "INVALID_" + key.upper())
    if metadata["data_kind"] not in {"REAL", "SYNTHETIC_TEST_FIXTURE"}:
        raise ContractError("INVALID_DATA_KIND")
    if metadata["data_kind"] != expected_data_kind:
        raise ContractError(
            "REAL_SCORER_REQUIRES_REAL_DATA"
            if expected_data_kind == "REAL"
            else "FIXTURE_SCORER_REQUIRES_SYNTHETIC_DATA"
        )
    if metadata["evaluation_record_hash_schema"] != EVALUATION_RECORD_HASH_SCHEMA:
        raise ContractError("UNSUPPORTED_EVALUATION_RECORD_HASH_SCHEMA")
    try:
        normalized["context_length"] = int(metadata["context_length"])
        normalized["maximum_new_tokens"] = int(metadata["maximum_new_tokens"])
    except (TypeError, ValueError) as exc:
        raise ContractError("INVALID_SCORING_LENGTH_POLICY") from exc
    if normalized["context_length"] < 2 or normalized["maximum_new_tokens"] < 1:
        raise ContractError("INVALID_SCORING_LENGTH_POLICY")
    if metadata["generation_strategy"] != "MANUAL_GREEDY":
        raise ContractError("UNSUPPORTED_GENERATION_STRATEGY")
    if metadata["terminator"] != "NEWLINE":
        raise ContractError("UNSUPPORTED_SCORING_TERMINATOR")
    try:
        candidate_universe = tuple(metadata["candidate_universe"])
    except TypeError as exc:
        raise ContractError("CANDIDATE_UNIVERSE_MUST_REMAIN_FOUR_LANGUAGES") from exc
    if candidate_universe != LANGUAGES:
        raise ContractError("CANDIDATE_UNIVERSE_MUST_REMAIN_FOUR_LANGUAGES")
    normalized["candidate_universe"] = list(LANGUAGES)
    if metadata["data_kind"] == "REAL" and metadata.get("freeze_gate_status") != "PASS":
        raise ContractError("REAL_SCORING_REQUIRES_VERIFIED_FREEZE")
    if metadata["data_kind"] == "REAL" and (
        normalized["context_length"] != 256
        or normalized["maximum_new_tokens"] != 64
    ):
        raise ContractError("REAL_SCORING_POLICY_NOT_V4")
    phase = metadata["phase"]
    stage = metadata["stage"]
    branch = metadata["branch"]
    if phase == "S":
        if stage not in {"T0", "T1", "T2", "T3"} or branch is not None:
            raise ContractError("INVALID_PHASE_STAGE_BRANCH")
    elif phase == "H":
        if stage != "H" or branch not in {"A", "B"}:
            raise ContractError("INVALID_PHASE_STAGE_BRANCH")
    else:
        raise ContractError("INVALID_SCORING_PHASE")
    try:
        normalized["root_id"] = int(metadata["root_id"])
    except (TypeError, ValueError) as exc:
        raise ContractError("INVALID_ROOT_ID") from exc
    if normalized["root_id"] not in {4101, 4102, 4103, 4104}:
        raise ContractError("INVALID_ROOT_ID")
    return normalized


def _read_checkpoint_binding(
    checkpoint_directory: str | Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a committed checkpoint without deserializing its pickle payload."""
    from .contracts import WORK_ROOT, require_relative_to

    directory = require_relative_to(
        Path(checkpoint_directory), WORK_ROOT, "CHECKPOINT_OUTSIDE_WORK_ROOT"
    )
    if directory.is_symlink() or not directory.is_dir():
        raise ContractError("CHECKPOINT_NOT_COMMITTED")
    commit_path = directory / "COMMITTED"
    manifest_path = directory / "manifest.json"
    state_path = directory / "state.pt"
    if any(
        path.is_symlink() or not path.is_file()
        for path in (commit_path, manifest_path, state_path)
    ):
        raise ContractError("CHECKPOINT_NOT_COMMITTED")
    try:
        manifest_sha256 = commit_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise ContractError("CHECKPOINT_NOT_COMMITTED") from exc
    _require_sha256(manifest_sha256, "INVALID_CHECKPOINT_COMMIT")
    manifest = read_verified_json(
        manifest_path, expected_sha256=manifest_sha256
    )
    required_manifest = {
        "schema_version",
        "state_file",
        "state_sha256",
        "state_bytes",
        "model_fingerprint",
        "lineage",
    }
    if (
        manifest.get("schema_version") != "v4-full-state-1"
        or not required_manifest.issubset(manifest)
        or manifest.get("state_file") != "state.pt"
    ):
        raise ContractError("CHECKPOINT_SCHEMA_MISMATCH")
    state_sha256 = sha256_file(state_path)
    if (
        manifest.get("state_sha256") != state_sha256
        or provenance["checkpoint_sha256"] != state_sha256
    ):
        raise ContractError("CHECKPOINT_STATE_HASH_MISMATCH")
    declared_manifest_hash = provenance.get("checkpoint_manifest_sha256")
    if declared_manifest_hash is not None and (
        _require_sha256(
            declared_manifest_hash, "INVALID_CHECKPOINT_MANIFEST_SHA256"
        )
        != manifest_sha256
    ):
        raise ContractError("CHECKPOINT_MANIFEST_HASH_MISMATCH")
    if manifest.get("model_fingerprint") != provenance["model_fingerprint"]:
        raise ContractError("CHECKPOINT_MODEL_HASH_MISMATCH")
    if (
        not isinstance(manifest.get("state_bytes"), int)
        or isinstance(manifest.get("state_bytes"), bool)
        or manifest["state_bytes"] != state_path.stat().st_size
    ):
        raise ContractError("CHECKPOINT_STATE_SIZE_MISMATCH")
    _require_sha256(manifest.get("model_fingerprint"), "INVALID_MODEL_FINGERPRINT")
    lineage = manifest.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ContractError("INCOMPLETE_CHECKPOINT_LINEAGE")
    checkpoint_phase = (
        provenance["stage"]
        if provenance["phase"] == "S"
        else "H_" + str(provenance["branch"])
    )
    expected_lineage = {
        "root_id": provenance["root_id"],
        "phase": checkpoint_phase,
        "stage": provenance["stage"],
        "branch": provenance["branch"],
        "freeze_sha256": provenance["freeze_sha256"],
        "config_sha256": provenance["config_sha256"],
        "tokenizer_sha256": provenance["tokenizer_sha256"],
        "code_sha256": provenance["code_sha256"],
        "evaluation_plan_sha256": provenance["evaluation_plan_sha256"],
        "data_kind": "REAL",
    }
    for key, expected in expected_lineage.items():
        actual = lineage.get(key)
        if key == "root_id":
            try:
                actual = int(actual)
            except (TypeError, ValueError) as exc:
                raise ContractError("CHECKPOINT_SCORING_LINEAGE_MISMATCH") from exc
        if actual != expected:
            raise ContractError("CHECKPOINT_SCORING_LINEAGE_MISMATCH")
    if lineage.get("freeze_gate_status") != "PASS":
        raise ContractError("CHECKPOINT_FREEZE_GATE_NOT_PASS")
    try:
        from .contracts import PROJECT_ROOT, load_json

        expected_uuid = load_json(
            PROJECT_ROOT / "implementation/config/campaign_policy.json"
        )["gpu"]["expected_uuid"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ContractError("BLOCKED_GPU_POLICY") from exc
    binding = lineage.get("gpu_binding")
    runtime = lineage.get("runtime")
    if (
        not isinstance(binding, Mapping)
        or binding.get("physical_index") != 2
        or binding.get("logical_index") != 0
        or binding.get("uuid") != expected_uuid
        or not isinstance(lineage.get("budget_lease_id"), str)
        or not lineage["budget_lease_id"]
        or not isinstance(runtime, Mapping)
        or not all(
            runtime.get(key)
            for key in ("python", "torch", "transformers", "torch_cuda", "precision")
        )
    ):
        raise ContractError("REAL_CHECKPOINT_NOT_BOUND_TO_GPU2")
    return {
        "directory": str(directory),
        "manifest_sha256": manifest_sha256,
        "state_sha256": state_sha256,
        "manifest": manifest,
    }


def _serialized_tokenizer_object(tokenizer: Any) -> Mapping[str, Any]:
    candidates = (
        tokenizer,
        getattr(tokenizer, "backend_tokenizer", None),
        getattr(tokenizer, "_tokenizer", None),
    )
    for candidate in candidates:
        serializer = getattr(candidate, "to_str", None)
        if not callable(serializer):
            continue
        try:
            raw = serializer()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            value = json.loads(raw)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, Mapping):
            return value
    raise ContractError("REAL_TOKENIZER_OBJECT_UNVERIFIABLE")


def _validate_real_record(
    record: Mapping[str, Any],
    *,
    concepts: Mapping[str, Mapping[str, Any]],
    evaluation_plan: Mapping[str, Any],
    stage: str,
    language_order=LANGUAGES,
) -> None:
    allowed_fields = set(REQUIRED_RECORD_FIELDS) | {
        "record_content_sha256",
        "requested_language",
        "root_id",
        "phase",
        "stage",
        "branch",
        "checkpoint_sha256",
        "model_fingerprint",
        "freeze_sha256",
        "config_sha256",
        "tokenizer_sha256",
        "code_sha256",
        "evaluation_plan_sha256",
    }
    if set(record) - allowed_fields:
        raise ContractError("UNEXPECTED_EVALUATION_RECORD_FIELD")
    concept_id = str(record.get("concept_id", ""))
    try:
        concept = concepts[concept_id]
    except KeyError as exc:
        raise ContractError("EVALUATION_CONCEPT_NOT_IN_FREEZE") from exc
    answers = concept.get("answers")
    glosses = concept.get("glosses")
    if not isinstance(answers, Mapping) or not isinstance(glosses, Mapping):
        raise ContractError("INVALID_FROZEN_CONCEPT")
    record_answers = record.get("answers")
    if not isinstance(record_answers, Mapping) or dict(record_answers) != dict(answers):
        raise ContractError("EVALUATION_ANSWERS_NOT_FROZEN")
    input_language = record.get("input_language")
    if stage not in STAGE_ACTIVE_LANGUAGES:
        raise ContractError("INVALID_SCORING_STAGE")
    from .records import stage_active_languages
    active_languages = stage_active_languages(stage, language_order)
    if input_language not in active_languages:
        raise ContractError("FUTURE_INPUT_LANGUAGE_IN_EVALUATION")
    prompt_plan = evaluation_plan.get("prompt_plan")
    if not isinstance(prompt_plan, Mapping) or record.get("format") != prompt_plan.get(
        "format"
    ):
        raise ContractError("EVALUATION_FORMAT_NOT_FROZEN")
    wrapper = record.get("wrapper")
    split = record.get("split")
    development = tuple(prompt_plan.get("development_wrappers", ()))
    tests = tuple(prompt_plan.get("test_wrappers", ()))
    if (split == "dev" and wrapper not in development) or (
        split == "test" and wrapper not in tests
    ):
        raise ContractError("EVALUATION_WRAPPER_SPLIT_NOT_FROZEN")
    if split not in {"dev", "test"}:
        raise ContractError("EVALUATION_SPLIT_NOT_FROZEN")
    mode = record.get("mode")
    requested = record.get("requested_language")
    if mode == "ANY":
        if "requested_language" in record:
            raise ContractError("ANY_MODE_MUST_NOT_REQUEST_LANGUAGE")
        try:
            gloss = glosses[input_language]
        except KeyError as exc:
            raise ContractError("INVALID_FROZEN_CONCEPT") from exc
        format_values: dict[str, Any] = {"gloss": gloss}
    elif mode == "REQUESTED":
        if "requested_language" not in record or requested not in active_languages:
            raise ContractError("FUTURE_REQUESTED_LANGUAGE_IN_EVALUATION")
        try:
            language_name = prompt_plan["target_language_names"][input_language][
                requested
            ]
        except (KeyError, TypeError) as exc:
            raise ContractError("EVALUATION_PROMPT_NOT_FROZEN") from exc
        format_values = {
            "gloss": glosses.get(input_language),
            "target_language_name": language_name,
        }
        if not isinstance(format_values["gloss"], str) or not format_values["gloss"]:
            raise ContractError("INVALID_FROZEN_CONCEPT")
    else:
        raise ContractError("UNKNOWN_EVALUATION_MODE")
    try:
        template = prompt_plan["templates"][wrapper][input_language][mode]
        expected_prefix = str(template).format(**format_values)
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("EVALUATION_PROMPT_NOT_FROZEN") from exc
    if record.get("prefix") != expected_prefix:
        raise ContractError("EVALUATION_PREFIX_NOT_FROZEN")


def _verify_real_scoring_bindings(
    model: Any,
    tokenizer: Any,
    provenance: dict[str, Any],
    *,
    experiment_freeze_path: str | Path,
    checkpoint_directory: str | Path,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    from .build_tokenizer import verify_tokenizer_manifest
    from .contracts import load_json
    from .prepare_data import verify_experiment_freeze

    freeze = verify_experiment_freeze(Path(experiment_freeze_path), production=True)
    if (
        freeze.get("verified") is not True
        or freeze.get("status") != "PASS"
        or freeze.get("freeze_gate_status") != "PASS"
        or freeze.get("data_kind") != "REAL"
    ):
        raise ContractError("REAL_SCORING_REQUIRES_VERIFIED_FREEZE")
    if freeze.get("freeze_sha256") != provenance["freeze_sha256"]:
        raise ContractError("SCORING_FREEZE_HASH_MISMATCH")
    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ContractError("INVALID_EXPERIMENT_FREEZE")
    if artifacts.get("pilot_config", {}).get("sha256") != provenance[
        "config_sha256"
    ]:
        raise ContractError("SCORING_CONFIG_HASH_MISMATCH")
    if artifacts.get("evaluation_plan", {}).get("sha256") != provenance[
        "evaluation_plan_sha256"
    ]:
        raise ContractError("SCORING_EVALUATION_PLAN_HASH_MISMATCH")
    if freeze.get("tokenizer_file_sha256") != provenance["tokenizer_sha256"]:
        raise ContractError("SCORING_TOKENIZER_HASH_MISMATCH")
    if freeze.get("evaluation_record_hash_schema") != EVALUATION_RECORD_HASH_SCHEMA:
        raise ContractError("UNSUPPORTED_EVALUATION_RECORD_HASH_SCHEMA")

    resolved = freeze.get("resolved_artifacts")
    if not isinstance(resolved, Mapping) or "tokenizer" not in resolved:
        raise ContractError("INVALID_EXPERIMENT_FREEZE")
    tokenizer_info = verify_tokenizer_manifest(
        Path(str(resolved["tokenizer"])), production=True
    )
    if tokenizer_info.get("tokenizer_file_sha256") != provenance["tokenizer_sha256"]:
        raise ContractError("SCORING_TOKENIZER_HASH_MISMATCH")
    frozen_tokenizer = read_verified_json(Path(tokenizer_info["tokenizer_path"]))
    if canonical_json_bytes(dict(_serialized_tokenizer_object(tokenizer))) != canonical_json_bytes(
        frozen_tokenizer
    ):
        raise ContractError("TOKENIZER_OBJECT_ARTIFACT_MISMATCH")

    checkpoint = _read_checkpoint_binding(checkpoint_directory, provenance)
    from .checkpoint import load_checkpoint_payload

    checkpoint_payload, checkpoint_manifest = load_checkpoint_payload(
        Path(checkpoint["directory"]),
        expected_state_sha256=checkpoint["state_sha256"],
        expected_manifest_sha256=checkpoint["manifest_sha256"],
    )
    if checkpoint_manifest != checkpoint["manifest"]:
        raise ContractError("CHECKPOINT_CHANGED_DURING_SCORING_BINDING")
    actual_model_fingerprint = semantic_model_fingerprint(model)
    if actual_model_fingerprint != provenance["model_fingerprint"]:
        raise ContractError("MODEL_FINGERPRINT_MISMATCH")
    if actual_model_fingerprint != checkpoint["manifest"]["model_fingerprint"]:
        raise ContractError("CHECKPOINT_MODEL_HASH_MISMATCH")
    del checkpoint_payload
    provenance["checkpoint_manifest_sha256"] = checkpoint["manifest_sha256"]

    evaluation_plan_path = Path(str(resolved.get("evaluation_plan", "")))
    if evaluation_plan_path.is_symlink() or not evaluation_plan_path.is_file():
        raise ContractError("INVALID_EVALUATION_PLAN_ARTIFACT")
    evaluation_plan = load_json(evaluation_plan_path)
    try:
        model_policy = evaluation_plan["model"]
        generation_policy = evaluation_plan["generation"]
    except (KeyError, TypeError) as exc:
        raise ContractError("INVALID_EVALUATION_PLAN_ARTIFACT") from exc
    if (
        model_policy.get("context_length") != provenance["context_length"]
        or generation_policy.get("maximum_new_tokens")
        != provenance["maximum_new_tokens"]
        or generation_policy.get("strategy") != provenance["generation_strategy"]
        or generation_policy.get("terminator") != provenance["terminator"]
        or tuple(generation_policy.get("candidate_universe_at_all_stages", ()))
        != tuple(provenance["candidate_universe"])
    ):
        raise ContractError("SCORING_POLICY_NOT_FROZEN")
    return freeze, evaluation_plan


def _score_checkpoint_impl(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    provenance: dict[str, Any],
    *,
    context_length: int,
    max_new_tokens: int,
    generators: Mapping[str, Any] | None = None,
    require_record_content_hash: bool,
    frozen_concepts: Mapping[str, Mapping[str, Any]] | None = None,
    evaluation_plan: Mapping[str, Any] | None = None,
    progress_callback: Any | None = None,
    language_order=LANGUAGES,
) -> dict[str, Any]:
    if context_length != provenance["context_length"]:
        raise ContractError("CONTEXT_LENGTH_POLICY_MISMATCH")
    if max_new_tokens != provenance["maximum_new_tokens"]:
        raise ContractError("GENERATION_LENGTH_POLICY_MISMATCH")
    actual_fingerprint = semantic_model_fingerprint(model)
    if actual_fingerprint != provenance["model_fingerprint"]:
        raise ContractError("MODEL_FINGERPRINT_MISMATCH")
    if not records:
        raise ContractError("EMPTY_EVALUATION_PLAN")
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        missing = [key for key in REQUIRED_RECORD_FIELDS if key not in record]
        if missing:
            raise ContractError("MISSING_EVALUATION_FIELD:" + missing[0])
        record_id = str(record["record_id"])
        if not record_id or record_id in by_id:
            raise ContractError("DUPLICATE_OR_EMPTY_EVALUATION_RECORD_ID")
        if record.get("mode") == "ANY":
            if record.get("requested_language") is not None:
                raise ContractError("ANY_MODE_MUST_NOT_REQUEST_LANGUAGE")
        elif record.get("mode") == "REQUESTED":
            if record.get("requested_language") not in LANGUAGES:
                raise ContractError("REQUESTED_MODE_REQUIRES_LANGUAGE")
        else:
            raise ContractError("UNKNOWN_EVALUATION_MODE")
        if frozen_concepts is not None:
            if evaluation_plan is None:
                raise ContractError("INVALID_EVALUATION_PLAN_ARTIFACT")
            _validate_real_record(
                record,
                concepts=frozen_concepts,
                evaluation_plan=evaluation_plan,
                stage=provenance["stage"],
                language_order=language_order,
            )
        for key in ("root_id", "phase", "stage", "branch"):
            if key in record and str(record[key]) != str(provenance[key]):
                raise ContractError("MIXED_EVALUATION_" + key.upper())
        for key in (
            "checkpoint_sha256",
            "model_fingerprint",
            "freeze_sha256",
            "config_sha256",
            "tokenizer_sha256",
            "code_sha256",
            "evaluation_plan_sha256",
        ):
            if key in record and str(record[key]).lower() != provenance[key]:
                raise ContractError("MIXED_EVALUATION_" + key.upper())
        by_id[record_id] = record

    output_rows: list[dict[str, Any]] = []
    with preserved_evaluation_state(model, generators) as rng_before:
        for record_id in sorted(by_id):
            if progress_callback is not None:
                progress_callback()
            record = by_id[record_id]
            answers = record["answers"]
            if not isinstance(answers, Mapping):
                raise ContractError("FOUR_REGISTERED_ANSWERS_REQUIRED")
            membership(answers)
            if any(
                answers[language] != canonical(answers[language])
                for language in LANGUAGES
            ):
                raise ContractError("NONCANONICAL_REGISTERED_EXPRESSION")
            if record["input_language"] not in LANGUAGES:
                raise ContractError("UNKNOWN_INPUT_LANGUAGE")
            if not all(
                isinstance(record[field], str) and bool(record[field])
                for field in ("concept_id", "format", "wrapper", "split", "prefix")
            ):
                raise ContractError("INVALID_EVALUATION_RECORD_TEXT_FIELD")
            record_content_sha256 = evaluation_record_content_sha256(record)
            supplied_content_hash = record.get("record_content_sha256")
            if require_record_content_hash and supplied_content_hash is None:
                raise ContractError("EVALUATION_RECORD_CONTENT_HASH_REQUIRED")
            if supplied_content_hash is not None and (
                _require_sha256(
                    supplied_content_hash, "INVALID_EVALUATION_RECORD_CONTENT_HASH"
                )
                != record_content_sha256
            ):
                raise ContractError("EVALUATION_RECORD_CONTENT_HASH_MISMATCH")
            requested = record.get("requested_language")
            generation = _greedy_first_line(
                model,
                tokenizer,
                record["prefix"],
                answers,
                requested,
                context_length=context_length,
                max_new_tokens=max_new_tokens,
            )
            probability = None
            if record["mode"] == "ANY":
                probability = _score_probability_partition(
                    model,
                    tokenizer,
                    record["prefix"],
                    answers,
                    context_length=context_length,
                )
            cell = {
                "record_id": record_id,
                "record_content_sha256": record_content_sha256,
                "concept_id": record["concept_id"],
                "root_id": provenance["root_id"],
                "phase": provenance["phase"],
                "stage": provenance["stage"],
                "branch": provenance["branch"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "model_fingerprint": provenance["model_fingerprint"],
                "freeze_sha256": provenance["freeze_sha256"],
                "config_sha256": provenance["config_sha256"],
                "tokenizer_sha256": provenance["tokenizer_sha256"],
                "code_sha256": provenance["code_sha256"],
                "evaluation_plan_sha256": provenance["evaluation_plan_sha256"],
                "evaluation_record_hash_schema": provenance[
                    "evaluation_record_hash_schema"
                ],
                "data_kind": provenance["data_kind"],
                "input_language": record["input_language"],
                "format": record["format"],
                "wrapper": record["wrapper"],
                "split": record["split"],
                "mode": record["mode"],
                "prefix_sha256": hashlib.sha256(record["prefix"].encode("utf-8")).hexdigest(),
                "generation": generation,
                "probability": probability,
            }
            if "checkpoint_manifest_sha256" in provenance:
                cell["checkpoint_manifest_sha256"] = provenance[
                    "checkpoint_manifest_sha256"
                ]
            if "requested_language" in record:
                cell["requested_language"] = requested
            output_rows.append(cell)
    record_set = [
        {
            "record_id": row["record_id"],
            "record_content_sha256": row["record_content_sha256"],
        }
        for row in output_rows
    ]
    return {
        "schema_version": "score.v1",
        "format_version": VERSION,
        "provenance": provenance,
        "rng_fingerprint_before": rng_before,
        "rng_fingerprint_after": rng_before,
        "rng_unchanged": True,
        "n_rows": len(output_rows),
        "evaluation_record_set_sha256": sha256_bytes(
            canonical_json_bytes(record_set)
        ),
        "rows": output_rows,
        "generation_metrics": aggregate_generation(output_rows),
    }


def score_checkpoint(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    *,
    context_length: int,
    max_new_tokens: int,
    experiment_freeze_path: str | Path | None = None,
    checkpoint_directory: str | Path | None = None,
    generators: Mapping[str, Any] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Score REAL data only after verifying every frozen artifact binding.

    Synthetic fixtures deliberately use :func:`score_fixture_checkpoint`; this
    production entry point has no caller-controlled fingerprint bypass.
    """
    provenance = _validate_provenance(metadata, expected_data_kind="REAL")
    if experiment_freeze_path is None:
        raise ContractError("EXPERIMENT_FREEZE_PATH_REQUIRED")
    if checkpoint_directory is None:
        raise ContractError("CHECKPOINT_DIRECTORY_REQUIRED")
    freeze, evaluation_plan = _verify_real_scoring_bindings(
        model,
        tokenizer,
        provenance,
        experiment_freeze_path=experiment_freeze_path,
        checkpoint_directory=checkpoint_directory,
    )
    concepts_raw = freeze.get("concepts")
    if not isinstance(concepts_raw, list) or not concepts_raw:
        raise ContractError("INVALID_EXPERIMENT_FREEZE")
    frozen_concepts: dict[str, Mapping[str, Any]] = {}
    for concept in concepts_raw:
        if not isinstance(concept, Mapping):
            raise ContractError("INVALID_FROZEN_CONCEPT")
        concept_id = str(concept.get("concept_id", ""))
        if not concept_id or concept_id in frozen_concepts:
            raise ContractError("INVALID_FROZEN_CONCEPT")
        frozen_concepts[concept_id] = concept
    return _score_checkpoint_impl(
        model,
        tokenizer,
        records,
        provenance,
        context_length=context_length,
        max_new_tokens=max_new_tokens,
        generators=generators,
        require_record_content_hash=True,
        frozen_concepts=frozen_concepts,
        evaluation_plan=evaluation_plan,
        language_order=freeze.get("execution", {}).get("language_order", LANGUAGES),
        progress_callback=progress_callback,
    )


def score_fixture_checkpoint(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    *,
    context_length: int,
    max_new_tokens: int,
    generators: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Explicit, non-production scorer for deterministic synthetic tests."""
    provenance = _validate_provenance(
        metadata, expected_data_kind="SYNTHETIC_TEST_FIXTURE"
    )
    return _score_checkpoint_impl(
        model,
        tokenizer,
        records,
        provenance,
        context_length=context_length,
        max_new_tokens=max_new_tokens,
        generators=generators,
        require_record_content_hash=False,
    )
