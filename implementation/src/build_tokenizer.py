"""Deterministic Wiki40B/ko byte-BPE and corpus-token materialization.

The source snapshot is immutable and preserves exact UTF-8.  NFC is applied
inside the frozen tokenizer; whitespace and Wiki40B structural markers are not
collapsed or removed.  All production entry points reverify their complete
artifact chain and reject synthetic fixtures.
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
import tempfile
from array import array
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from freshstart.core import ContractError as ReferenceContractError
from freshstart.core import (
    canonical,
    spelling_similarity,
    token_jaccard,
    validate_token_events,
)

from .artifacts import (
    publish_bytes_once,
    publish_json_once,
    read_regular_file_bytes,
    read_regular_file_bytes_exact,
    read_verified_json_with_sha256,
    sha256_file,
)
from .contracts import (
    IMPLEMENTATION_REVISION,
    LANGUAGES,
    PROJECT_ROOT,
    VERSION,
    WORK_ROOT,
    ContractViolation,
    canonical_json_bytes,
    require_relative_to,
    require_sha256,
    sha256_bytes,
)
from .prepare_data import REAL_WIKI40B, SYNTHETIC, verify_wiki40b_snapshot


EVALUATION_PLAN_PATH = PROJECT_ROOT / "implementation/config/evaluation_plan.json"
EXPRESSION_LANGUAGE_PAIRS = (
    ("ko", "en"),
    ("ko", "zh"),
    ("ko", "fr"),
    ("en", "zh"),
    ("en", "fr"),
    ("zh", "fr"),
)
EXPRESSION_METRICS_SCHEMA = "expression-only-lexical-metrics-v1"
EXPRESSION_EXCLUDED_SPECIAL_TOKENS = ("<|endoftext|>", "<|pad|>")


def _logical_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _without(mapping: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in keys}


def _validate_embedded_hash(row: Mapping[str, Any], field: str, code: str) -> None:
    claimed = row.get(field)
    require_sha256(str(claimed or ""), code)
    if claimed != _logical_hash(_without(row, field)):
        raise ContractViolation(code)


def _verified_artifact_bytes(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> bytes:
    """Return one descriptor's bytes only when both frozen claims match."""
    require_sha256(expected_sha256, "INVALID_ARTIFACT_SHA256")
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
    ):
        raise ContractViolation("INVALID_ARTIFACT_SIZE")
    raw = read_regular_file_bytes_exact(path, expected_bytes=expected_bytes)
    if len(raw) != expected_bytes:
        raise ContractViolation("ARTIFACT_SIZE_MISMATCH")
    if sha256_bytes(raw) != expected_sha256:
        raise ContractViolation("ARTIFACT_HASH_MISMATCH")
    return raw


def _artifact_ref(ref: Mapping[str, Any], base: Path) -> dict[str, Any]:
    raw = Path(str(ref["path"])).resolve()
    try:
        stored_path = str(raw.relative_to(base.resolve()))
    except ValueError:
        stored_path = str(raw)
    return {
        "path": stored_path,
        "sha256": require_sha256(str(ref["sha256"])),
        "bytes": int(ref["bytes"]),
    }


def _resolve_artifact(
    ref: Mapping[str, Any],
    *,
    base: Path,
    production: bool,
    scope_root: Path = WORK_ROOT,
) -> Path:
    required = {"path", "sha256", "bytes"}
    if not isinstance(ref, Mapping) or not required.issubset(ref):
        raise ContractViolation("INVALID_ARTIFACT_REF")
    expected_hash = require_sha256(str(ref["sha256"]), "INVALID_ARTIFACT_SHA256")
    expected_bytes = ref["bytes"]
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
    ):
        raise ContractViolation("INVALID_ARTIFACT_SIZE")
    raw = Path(str(ref["path"]))
    unresolved = raw if raw.is_absolute() else base / raw
    if unresolved.is_symlink():
        raise ContractViolation("ARTIFACT_NOT_REGULAR_FILE")
    path = unresolved.resolve(strict=False)
    if production:
        require_relative_to(path, scope_root)
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise ContractViolation("ARTIFACT_SIZE_MISMATCH")
    if sha256_file(path) != expected_hash:
        raise ContractViolation("ARTIFACT_HASH_MISMATCH")
    return path


def _tokenizer_bindings() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
    except ImportError as exc:
        raise ContractViolation("TOKENIZERS_DEPENDENCY_MISSING") from exc
    return Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


def load_evaluation_plan(
    path: Path = EVALUATION_PLAN_PATH,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ContractViolation("EVALUATION_PLAN_NOT_REGULAR_FILE")
    if (expected_sha256 is None) != (expected_bytes is None):
        raise ContractViolation("INVALID_ARTIFACT_REF")
    try:
        if expected_sha256 is None:
            raw = read_regular_file_bytes(source)
        else:
            checked_sha256 = require_sha256(
                str(expected_sha256), "INVALID_ARTIFACT_SHA256"
            )
            if (
                not isinstance(expected_bytes, int)
                or isinstance(expected_bytes, bool)
                or expected_bytes < 0
            ):
                raise ContractViolation("INVALID_ARTIFACT_SIZE")
            raw = read_regular_file_bytes_exact(
                source, expected_bytes=expected_bytes
            )
            if sha256_bytes(raw) != checked_sha256:
                raise ContractViolation("ARTIFACT_HASH_MISMATCH")
        plan = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractViolation("INVALID_EVALUATION_PLAN") from exc
    if not isinstance(plan, dict) or plan.get("version") != IMPLEMENTATION_REVISION:
        raise ContractViolation("BLOCKED_IMPLEMENTATION_REVISION")
    policy = plan.get("tokenizer")
    required = {
        "type": "BYTE_LEVEL_BPE",
        "add_prefix_space": False,
        "use_regex": True,
        "special_tokens_in_id_order": ["<|endoftext|>", "<|pad|>"],
        "document_separator": "<|endoftext|>",
        "partial_final_document": "truncate only the materialized corpus token stream at the cap and record discarded tokens",
    }
    if not isinstance(policy, dict) or any(policy.get(key) != value for key, value in required.items()):
        raise ContractViolation("INVALID_TOKENIZER_POLICY")
    for key in (
        "vocab_size",
        "min_frequency",
        "training_document_cap",
        "training_utf8_byte_cap",
        "corpus_model_token_cap",
    ):
        value = policy.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ContractViolation("INVALID_TOKENIZER_POLICY")
    if policy["vocab_size"] >= 65_536:
        raise ContractViolation("TOKENIZER_VOCAB_EXCEEDS_UINT16")
    if plan.get("normalization") != "NFC_AND_WHITESPACE_ONLY":
        raise ContractViolation("INVALID_NORMALIZATION_POLICY")
    return plan


def _iter_verified_documents(source: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    index = source["index"]
    path = Path(source["documents_path"])
    with path.open("rb") as handle:
        for row in index:
            if handle.tell() != row["offset"]:
                raise ContractViolation("WIKI40B_INDEX_MISMATCH")
            size_raw = handle.read(8)
            if len(size_raw) != 8:
                raise ContractViolation("WIKI40B_DOCUMENT_TRUNCATED")
            size = int.from_bytes(size_raw, "big")
            payload = handle.read(size)
            if len(payload) != size or sha256_bytes(payload) != row["text_sha256"]:
                raise ContractViolation("WIKI40B_DOCUMENT_HASH_MISMATCH")
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ContractViolation("WIKI40B_INVALID_UTF8") from exc
            yield {**row, "text": text}
        if handle.read(1):
            raise ContractViolation("WIKI40B_DOCUMENT_TRAILING_BYTES")


def iter_staged_documents(
    source_manifest: Path,
    *,
    production: bool = True,
) -> Iterator[dict[str, Any]]:
    """Reverify and yield exact source documents in frozen order."""
    source = verify_wiki40b_snapshot(source_manifest, production=production)
    yield from _iter_verified_documents(source)


def _tokenizer_json_structure(
    serialized: Mapping[str, Any], policy: Mapping[str, Any]
) -> None:
    model = serialized.get("model")
    pre = serialized.get("pre_tokenizer")
    decoder = serialized.get("decoder")
    normalizer = serialized.get("normalizer")
    if not isinstance(model, Mapping) or model.get("type") != "BPE":
        raise ContractViolation("TOKENIZER_MODEL_MISMATCH")
    if model.get("unk_token") is not None or model.get("byte_fallback") is not False:
        raise ContractViolation("TOKENIZER_MODEL_MISMATCH")
    if (
        not isinstance(pre, Mapping)
        or pre.get("type") != "ByteLevel"
        or pre.get("add_prefix_space") is not False
        or pre.get("use_regex") is not True
    ):
        raise ContractViolation("TOKENIZER_PRETOKENIZER_MISMATCH")
    if not isinstance(decoder, Mapping) or decoder.get("type") != "ByteLevel":
        raise ContractViolation("TOKENIZER_DECODER_MISMATCH")
    if not isinstance(normalizer, Mapping) or normalizer.get("type") != "NFC":
        raise ContractViolation("TOKENIZER_NORMALIZER_MISMATCH")
    added = serialized.get("added_tokens")
    specials = policy["special_tokens_in_id_order"]
    if not isinstance(added, list):
        raise ContractViolation("TOKENIZER_SPECIAL_TOKEN_MISMATCH")
    found = {
        row.get("content"): row
        for row in added
        if isinstance(row, Mapping) and row.get("special") is True
    }
    for expected_id, token in enumerate(specials):
        if found.get(token, {}).get("id") != expected_id:
            raise ContractViolation("TOKENIZER_SPECIAL_TOKEN_MISMATCH")


def _verify_tokenizer_object(tokenizer: Any, policy: Mapping[str, Any]) -> None:
    _, _, _, _, pre_tokenizers, _ = _tokenizer_bindings()
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    if len(vocab) != policy["vocab_size"] or len(set(vocab.values())) != len(vocab):
        raise ContractViolation("TOKENIZER_VOCAB_SIZE_MISMATCH")
    for expected_id, token in enumerate(policy["special_tokens_in_id_order"]):
        if tokenizer.token_to_id(token) != expected_id:
            raise ContractViolation("TOKENIZER_SPECIAL_TOKEN_MISMATCH")
    alphabet = set(pre_tokenizers.ByteLevel.alphabet())
    if len(alphabet) != 256 or not alphabet.issubset(vocab):
        raise ContractViolation("TOKENIZER_BYTE_ALPHABET_INCOMPLETE")
    try:
        serialized = json.loads(tokenizer.to_str())
    except (TypeError, json.JSONDecodeError) as exc:
        raise ContractViolation("TOKENIZER_SERIALIZATION_INVALID") from exc
    _tokenizer_json_structure(serialized, policy)


def train_byte_bpe(
    source_manifest: Path,
    *,
    output_dir: Path,
    production: bool = True,
    evaluation_plan_path: Path = EVALUATION_PLAN_PATH,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Train and immutably publish the frozen byte-level BPE tokenizer."""
    source = verify_wiki40b_snapshot(source_manifest, production=production)
    plan = load_evaluation_plan(evaluation_plan_path)
    policy = plan["tokenizer"]
    policy_path = Path(evaluation_plan_path).resolve()
    if production and policy_path != EVALUATION_PLAN_PATH.resolve():
        raise ContractViolation("PRODUCTION_EVALUATION_PLAN_PATH_MISMATCH")
    if source["selection"]["document_cap"] != policy["training_document_cap"]:
        raise ContractViolation("TOKENIZER_SOURCE_SELECTION_MISMATCH")
    if source["selection"]["utf8_byte_cap"] != policy["training_utf8_byte_cap"]:
        raise ContractViolation("TOKENIZER_SOURCE_SELECTION_MISMATCH")

    Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers = _tokenizer_bindings()
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=policy["vocab_size"],
        min_frequency=policy["min_frequency"],
        show_progress=False,
        special_tokens=policy["special_tokens_in_id_order"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(
        (row["text"] for row in _iter_verified_documents(source)),
        trainer=trainer,
        length=source["staged_num_examples"],
    )
    _verify_tokenizer_object(tokenizer, policy)
    serialized = tokenizer.to_str(pretty=False).encode("utf-8")
    try:
        roundtrip = Tokenizer.from_str(serialized.decode("utf-8"))
    except Exception as exc:
        raise ContractViolation("TOKENIZER_ROUNDTRIP_FAILED") from exc
    _verify_tokenizer_object(roundtrip, policy)
    if roundtrip.to_str(pretty=False).encode("utf-8") != serialized:
        raise ContractViolation("TOKENIZER_SERIALIZATION_NOT_STABLE")

    output_dir = require_relative_to(
        Path(output_dir), scope_root, "TOKENIZER_OUTPUT_OUTSIDE_SCOPE"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer_ref = publish_bytes_once(output_dir / "tokenizer.json", serialized)
    source_path = Path(source_manifest).resolve()
    source_ref = {
        "path": str(source_path),
        "sha256": sha256_file(source_path),
        "bytes": source_path.stat().st_size,
    }
    plan_ref = {
        "path": str(policy_path),
        "sha256": sha256_file(policy_path),
        "bytes": policy_path.stat().st_size,
    }
    core = {
        "schema_version": VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "status": "PASS",
        "data_kind": source["data_kind"],
        "tokenizer_kind": "BYTE_LEVEL_BPE",
        "source_snapshot": source_ref,
        "source_snapshot_sha256": source["source_snapshot_sha256"],
        "evaluation_plan": plan_ref,
        "tokenizer_policy_sha256": _logical_hash(policy),
        "training_document_count": source["staged_num_examples"],
        "training_source_utf8_bytes": source["staged_utf8_bytes"],
        "training_document_order_sha256": source["document_order_sha256"],
        "input_normalization": "NFC_ONLY_WHITESPACE_AND_WIKI40B_MARKERS_PRESERVED",
        "vocab_size": policy["vocab_size"],
        "min_frequency": policy["min_frequency"],
        "special_tokens_in_id_order": policy["special_tokens_in_id_order"],
        "byte_alphabet_size": 256,
        "tokenizers_version": importlib.metadata.version("tokenizers"),
        "artifact": _artifact_ref(tokenizer_ref, output_dir),
        "tokenizer_file_sha256": tokenizer_ref["sha256"],
    }
    manifest = {**core, "tokenizer_manifest_sha256": _logical_hash(core)}
    manifest_ref = publish_json_once(output_dir / "tokenizer_manifest.json", manifest)
    return {**manifest, "manifest_artifact": _artifact_ref(manifest_ref, output_dir)}


def verify_tokenizer_manifest(
    manifest_source: Path,
    *,
    production: bool = True,
) -> dict[str, Any]:
    path = Path(manifest_source)
    manifest, manifest_sha256 = read_verified_json_with_sha256(path)
    if (
        manifest.get("schema_version") != VERSION
        or manifest.get("implementation_revision") != IMPLEMENTATION_REVISION
        or manifest.get("status") != "PASS"
        or manifest.get("tokenizer_kind") != "BYTE_LEVEL_BPE"
    ):
        raise ContractViolation("INVALID_TOKENIZER_MANIFEST")
    if production and manifest.get("data_kind") != REAL_WIKI40B:
        raise ContractViolation("SYNTHETIC_TOKENIZER_NOT_PRODUCTION")
    _validate_embedded_hash(
        manifest, "tokenizer_manifest_sha256", "TOKENIZER_MANIFEST_HASH_MISMATCH"
    )
    source_path = _resolve_artifact(
        manifest.get("source_snapshot", {}),
        base=path.parent,
        production=production,
    )
    plan_ref = manifest.get("evaluation_plan")
    if not isinstance(plan_ref, Mapping):
        raise ContractViolation("INVALID_ARTIFACT_REF")
    plan_path = _resolve_artifact(
        plan_ref,
        base=path.parent,
        production=production,
        scope_root=PROJECT_ROOT,
    )
    if production and plan_path != EVALUATION_PLAN_PATH.resolve():
        raise ContractViolation("PRODUCTION_EVALUATION_PLAN_PATH_MISMATCH")
    plan_sha256 = require_sha256(
        str(plan_ref.get("sha256", "")), "INVALID_ARTIFACT_SHA256"
    )
    plan = load_evaluation_plan(
        plan_path,
        expected_sha256=plan_sha256,
        expected_bytes=plan_ref.get("bytes"),
    )
    policy = plan["tokenizer"]
    if manifest.get("tokenizer_policy_sha256") != _logical_hash(policy):
        raise ContractViolation("TOKENIZER_POLICY_HASH_MISMATCH")
    tokenizer_path = _resolve_artifact(
        manifest.get("artifact", {}),
        base=path.parent,
        production=production,
    )
    tokenizer_ref = manifest.get("artifact")
    if not isinstance(tokenizer_ref, Mapping):
        raise ContractViolation("INVALID_ARTIFACT_REF")
    tokenizer_file_sha256 = require_sha256(
        str(manifest.get("tokenizer_file_sha256", "")),
        "INVALID_ARTIFACT_SHA256",
    )
    if tokenizer_ref.get("sha256") != tokenizer_file_sha256:
        raise ContractViolation("TOKENIZER_FILE_HASH_MISMATCH")
    tokenizer_bytes = _verified_artifact_bytes(
        tokenizer_path,
        expected_sha256=tokenizer_file_sha256,
        expected_bytes=tokenizer_ref.get("bytes"),
    )
    source = verify_wiki40b_snapshot(source_path, production=production)
    if (
        source.get("source_snapshot_sha256") != manifest.get("source_snapshot_sha256")
        or source.get("staged_num_examples") != manifest.get("training_document_count")
        or source.get("staged_utf8_bytes") != manifest.get("training_source_utf8_bytes")
        or source.get("document_order_sha256")
        != manifest.get("training_document_order_sha256")
    ):
        raise ContractViolation("TOKENIZER_SOURCE_BINDING_MISMATCH")
    Tokenizer, _, _, _, _, _ = _tokenizer_bindings()
    try:
        tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
    except Exception as exc:
        raise ContractViolation("TOKENIZER_ARTIFACT_INVALID") from exc
    _verify_tokenizer_object(tokenizer, policy)
    if manifest.get("vocab_size") != policy["vocab_size"]:
        raise ContractViolation("TOKENIZER_VOCAB_SIZE_MISMATCH")
    return {
        **manifest,
        "manifest_path": str(path.resolve()),
        "manifest_sha256": manifest_sha256,
        "tokenizer_path": str(tokenizer_path),
        "source_manifest_path": str(source_path),
        "evaluation_plan_path": str(plan_path),
        "evaluation_plan_sha256": plan_sha256,
        "policy": policy,
    }


def load_verified_tokenizer(
    manifest_source: Path,
    *,
    production: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Return a tokenizer only after the entire source/policy chain verifies."""
    verified = verify_tokenizer_manifest(manifest_source, production=production)
    Tokenizer, _, _, _, _, _ = _tokenizer_bindings()
    artifact = verified.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ContractViolation("INVALID_ARTIFACT_REF")
    tokenizer_bytes = _verified_artifact_bytes(
        Path(verified["tokenizer_path"]),
        expected_sha256=str(verified["tokenizer_file_sha256"]),
        expected_bytes=artifact.get("bytes"),
    )
    try:
        tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
    except Exception as exc:
        raise ContractViolation("TOKENIZER_ARTIFACT_INVALID") from exc
    _verify_tokenizer_object(tokenizer, verified["policy"])
    return tokenizer, verified


def _encode(
    tokenizer: Any,
    text: str,
    *,
    require_explicit_no_special_tokens: bool = False,
) -> list[int]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError as exc:
        if require_explicit_no_special_tokens:
            raise ContractViolation("TOKENIZER_CANNOT_DISABLE_SPECIAL_TOKENS") from exc
        encoded = tokenizer.encode(text)
    values = encoded.ids if hasattr(encoded, "ids") else encoded
    if isinstance(values, Mapping):
        values = values.get("input_ids")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ContractViolation("TOKENIZER_RETURNED_INVALID_IDS")
    try:
        result = [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ContractViolation("TOKENIZER_RETURNED_INVALID_IDS") from exc
    if any(value < 0 for value in result):
        raise ContractViolation("TOKENIZER_RETURNED_INVALID_IDS")
    return result


def _expression_excluded_special_token_ids(tokenizer: Any) -> dict[str, int]:
    token_to_id = getattr(tokenizer, "token_to_id", None)
    if not callable(token_to_id):
        raise ContractViolation("TOKENIZER_SPECIAL_TOKEN_LOOKUP_REQUIRED")
    result: dict[str, int] = {}
    for token in EXPRESSION_EXCLUDED_SPECIAL_TOKENS:
        try:
            token_id = token_to_id(token)
        except Exception as exc:
            raise ContractViolation("TOKENIZER_SPECIAL_TOKEN_LOOKUP_FAILED") from exc
        if (
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
        ):
            raise ContractViolation("TOKENIZER_SPECIAL_TOKEN_ID_MISSING")
        result[token] = token_id
    if len(set(result.values())) != len(result):
        raise ContractViolation("TOKENIZER_SPECIAL_TOKEN_IDS_NOT_DISTINCT")
    return result


def expression_only_similarity_metrics(
    tokenizer: Any, answers: Mapping[str, Any]
) -> dict[str, Any]:
    """Return deterministic four-language form metrics for one concept.

    Character lengths and edit similarities use the scorer's canonical NFC
    expressions.  Token metrics encode those expressions alone: no prompt,
    padding, answer terminator, or automatically added special token is part of
    the event passed to the tokenizer.
    """

    if not isinstance(answers, Mapping) or set(answers) != set(LANGUAGES):
        raise ContractViolation("FOUR_LANGUAGE_ANSWERS_REQUIRED")

    excluded_special_token_ids = _expression_excluded_special_token_ids(tokenizer)
    excluded_ids = set(excluded_special_token_ids.values())
    canonical_answers: dict[str, str] = {}
    token_ids: dict[str, list[int]] = {}
    language_metrics: dict[str, dict[str, int]] = {}
    for language in LANGUAGES:
        value = answers[language]
        if not isinstance(value, str):
            raise ContractViolation("INVALID_REGISTERED_EXPRESSION")
        try:
            normalized = canonical(value)
        except ReferenceContractError as exc:
            code = str(exc)
            if not code or any(character.isspace() for character in code):
                code = "INVALID_REGISTERED_EXPRESSION"
            raise ContractViolation(code) from exc
        ids = _encode(
            tokenizer,
            normalized,
            require_explicit_no_special_tokens=True,
        )
        if not ids:
            raise ContractViolation("EMPTY_EXPRESSION_TOKENS")
        if excluded_ids.intersection(ids):
            raise ContractViolation("EXPRESSION_CONTAINS_RESERVED_SPECIAL_TOKEN")
        canonical_answers[language] = normalized
        token_ids[language] = ids
        language_metrics[language] = {
            "nfc_char_length": len(normalized),
            "token_length": len(ids),
        }

    pair_metrics: dict[str, dict[str, float]] = {}
    for left, right in EXPRESSION_LANGUAGE_PAIRS:
        key = f"{left}-{right}"
        try:
            surface = spelling_similarity(
                canonical_answers[left], canonical_answers[right]
            )
            token_overlap = token_jaccard(token_ids[left], token_ids[right])
        except ReferenceContractError as exc:
            code = str(exc)
            if not code or any(character.isspace() for character in code):
                code = "INVALID_EXPRESSION_METRIC"
            raise ContractViolation(code) from exc
        pair_metrics[key] = {
            "surface_levenshtein_similarity": surface,
            "token_set_jaccard": token_overlap,
        }

    return {
        "schema_version": EXPRESSION_METRICS_SCHEMA,
        "normalization": "NFC_AND_WHITESPACE_ONLY",
        "tokenization_scope": (
            "CANONICAL_EXPRESSION_ONLY_NO_PROMPT_PADDING_RESERVED_SPECIAL_IDS_OR_TERMINATOR"
        ),
        "excluded_special_token_ids": excluded_special_token_ids,
        "language_order": list(LANGUAGES),
        "pair_order": [f"{left}-{right}" for left, right in EXPRESSION_LANGUAGE_PAIRS],
        "languages": language_metrics,
        "pairs": pair_metrics,
    }


def _uint16_le_bytes(values: array[int]) -> bytes:
    copy = array("H", values)
    if sys.byteorder != "little":
        copy.byteswap()
    return copy.tobytes()


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def materialize_corpus_tokens(
    source_manifest: Path,
    tokenizer_manifest: Path,
    *,
    output_dir: Path,
    production: bool = True,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Encode the frozen source and truncate only the token stream at 2M."""
    source = verify_wiki40b_snapshot(source_manifest, production=production)
    tokenizer, tokenizer_info = load_verified_tokenizer(
        tokenizer_manifest, production=production
    )
    if sha256_file(Path(source_manifest)) != tokenizer_info["source_snapshot"]["sha256"]:
        raise ContractViolation("CORPUS_TOKENIZER_SOURCE_MISMATCH")
    policy = tokenizer_info["policy"]
    token_cap = int(policy["corpus_model_token_cap"])
    eos_token = policy["document_separator"]
    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is None:
        raise ContractViolation("DOCUMENT_SEPARATOR_TOKEN_MISSING")
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if vocab_size >= 65_536:
        raise ContractViolation("TOKENIZER_VOCAB_EXCEEDS_UINT16")

    materialized = array("H")
    index_rows: list[dict[str, Any]] = []
    source_token_count = 0
    partial_ordinal: int | None = None
    for row in _iter_verified_documents(source):
        document_ids = _encode(tokenizer, row["text"]) + [int(eos_id)]
        if any(token >= vocab_size for token in document_ids):
            raise ContractViolation("TOKEN_ID_OUTSIDE_VOCAB")
        source_start = source_token_count
        source_token_count += len(document_ids)
        remaining = max(0, token_cap - len(materialized))
        kept = min(remaining, len(document_ids))
        materialized_start = len(materialized)
        if kept:
            materialized.extend(document_ids[:kept])
        partial = 0 < kept < len(document_ids)
        if partial:
            if partial_ordinal is not None:
                raise ContractViolation("MULTIPLE_PARTIAL_DOCUMENTS")
            partial_ordinal = row["ordinal"]
        index_rows.append(
            {
                "ordinal": row["ordinal"],
                "source_text_sha256": row["text_sha256"],
                "source_token_start": source_start,
                "source_token_count": len(document_ids),
                "materialized_token_start": materialized_start,
                "materialized_token_count": kept,
                "discarded_token_count": len(document_ids) - kept,
                "partial": partial,
            }
        )
    materialized_count = len(materialized)
    if production and materialized_count != token_cap:
        raise ContractViolation("CORPUS_TOKEN_CAP_NOT_REACHED")
    if materialized_count > token_cap:
        raise ContractViolation("CORPUS_TOKEN_CAP_EXCEEDED")

    output_dir = require_relative_to(
        Path(output_dir), scope_root, "CORPUS_TOKEN_OUTPUT_OUTSIDE_SCOPE"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    tokens_ref = publish_bytes_once(
        output_dir / "corpus_tokens.uint16le", _uint16_le_bytes(materialized)
    )
    index_ref = publish_bytes_once(
        output_dir / "corpus_tokens.index.jsonl", _jsonl_bytes(index_rows)
    )
    source_path = Path(source_manifest).resolve()
    tokenizer_manifest_path = Path(tokenizer_manifest).resolve()
    core = {
        "schema_version": VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "status": "PASS",
        "data_kind": source["data_kind"],
        "corpus_kind": "WIKI40B_KO_CAUSAL_TOKEN_STREAM",
        "source_snapshot": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "bytes": source_path.stat().st_size,
        },
        "source_snapshot_sha256": source["source_snapshot_sha256"],
        "tokenizer_manifest": {
            "path": str(tokenizer_manifest_path),
            "sha256": sha256_file(tokenizer_manifest_path),
            "bytes": tokenizer_manifest_path.stat().st_size,
        },
        "tokenizer_file_sha256": tokenizer_info["tokenizer_file_sha256"],
        "tokenizer_policy_sha256": tokenizer_info["tokenizer_policy_sha256"],
        "dtype": "uint16-le",
        "token_width_bytes": 2,
        "document_separator": {"text": eos_token, "token_id": eos_id},
        "normalization": "TOKENIZER_NFC_ONLY_WHITESPACE_AND_WIKI40B_MARKERS_PRESERVED",
        "source_document_count": len(index_rows),
        "source_document_order_sha256": source["document_order_sha256"],
        "source_token_count_including_separators": source_token_count,
        "materialized_token_cap": token_cap,
        "materialized_token_count": materialized_count,
        "discarded_token_count": source_token_count - materialized_count,
        "partial_final_document": {
            "policy": policy["partial_final_document"],
            "ordinal": partial_ordinal,
        },
        "artifacts": {
            "tokens": _artifact_ref(tokens_ref, output_dir),
            "index": _artifact_ref(index_ref, output_dir),
        },
        "token_stream_sha256": tokens_ref["sha256"],
    }
    manifest = {**core, "corpus_manifest_sha256": _logical_hash(core)}
    manifest_ref = publish_json_once(output_dir / "corpus_manifest.json", manifest)
    return {**manifest, "manifest_artifact": _artifact_ref(manifest_ref, output_dir)}


def _read_jsonl(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        raw = _verified_artifact_bytes(
            path,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
        )
        for line in raw.splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ContractViolation("JSONL_ROW_NOT_OBJECT")
                rows.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractViolation("INVALID_JSONL") from exc
    return rows


def verify_corpus_manifest(
    manifest_source: Path,
    *,
    production: bool = True,
) -> dict[str, Any]:
    """Re-encode the source and compare every stored token/count/index row."""
    path = Path(manifest_source)
    manifest, manifest_sha256 = read_verified_json_with_sha256(path)
    if (
        manifest.get("schema_version") != VERSION
        or manifest.get("implementation_revision") != IMPLEMENTATION_REVISION
        or manifest.get("status") != "PASS"
        or manifest.get("corpus_kind") != "WIKI40B_KO_CAUSAL_TOKEN_STREAM"
    ):
        raise ContractViolation("INVALID_CORPUS_MANIFEST")
    if production and manifest.get("data_kind") != REAL_WIKI40B:
        raise ContractViolation("SYNTHETIC_CORPUS_NOT_PRODUCTION")
    _validate_embedded_hash(
        manifest, "corpus_manifest_sha256", "CORPUS_MANIFEST_HASH_MISMATCH"
    )
    source_path = _resolve_artifact(
        manifest.get("source_snapshot", {}),
        base=path.parent,
        production=production,
    )
    tokenizer_manifest_path = _resolve_artifact(
        manifest.get("tokenizer_manifest", {}),
        base=path.parent,
        production=production,
    )
    tokenizer, tokenizer_info = load_verified_tokenizer(
        tokenizer_manifest_path, production=production
    )
    source = verify_wiki40b_snapshot(source_path, production=production)
    if sha256_file(source_path) != tokenizer_info["source_snapshot"]["sha256"]:
        raise ContractViolation("CORPUS_TOKENIZER_SOURCE_MISMATCH")
    if (
        manifest.get("source_snapshot_sha256") != source["source_snapshot_sha256"]
        or manifest.get("source_document_order_sha256") != source["document_order_sha256"]
        or manifest.get("tokenizer_file_sha256") != tokenizer_info["tokenizer_file_sha256"]
        or manifest.get("tokenizer_policy_sha256") != tokenizer_info["tokenizer_policy_sha256"]
    ):
        raise ContractViolation("CORPUS_SOURCE_OR_TOKENIZER_BINDING_MISMATCH")
    if manifest.get("dtype") != "uint16-le" or manifest.get("token_width_bytes") != 2:
        raise ContractViolation("CORPUS_DTYPE_MISMATCH")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"tokens", "index"}:
        raise ContractViolation("CORPUS_ARTIFACT_SET_INVALID")
    tokens_path = _resolve_artifact(
        artifacts["tokens"], base=path.parent, production=production
    )
    index_path = _resolve_artifact(
        artifacts["index"], base=path.parent, production=production
    )
    materialized_count = manifest.get("materialized_token_count")
    if (
        not isinstance(materialized_count, int)
        or isinstance(materialized_count, bool)
        or materialized_count < 1
    ):
        raise ContractViolation("CORPUS_TOKEN_FILE_SIZE_MISMATCH")
    token_ref = artifacts["tokens"]
    index_ref = artifacts["index"]
    if not isinstance(token_ref, Mapping) or not isinstance(index_ref, Mapping):
        raise ContractViolation("INVALID_ARTIFACT_REF")
    token_stream_sha256 = require_sha256(
        str(manifest.get("token_stream_sha256", "")),
        "INVALID_ARTIFACT_SHA256",
    )
    if token_ref.get("sha256") != token_stream_sha256:
        raise ContractViolation("CORPUS_TOKEN_HASH_MISMATCH")
    token_bytes = _verified_artifact_bytes(
        tokens_path,
        expected_sha256=token_stream_sha256,
        expected_bytes=token_ref.get("bytes"),
    )
    if len(token_bytes) != materialized_count * 2:
        raise ContractViolation("CORPUS_TOKEN_FILE_SIZE_MISMATCH")
    policy = tokenizer_info["policy"]
    if manifest.get("materialized_token_cap") != policy["corpus_model_token_cap"]:
        raise ContractViolation("CORPUS_TOKEN_CAP_MISMATCH")
    if materialized_count > policy["corpus_model_token_cap"]:
        raise ContractViolation("CORPUS_TOKEN_CAP_EXCEEDED")
    if production and materialized_count != policy["corpus_model_token_cap"]:
        raise ContractViolation("CORPUS_TOKEN_CAP_NOT_REACHED")

    try:
        import numpy as np
    except ImportError as exc:
        raise ContractViolation("NUMPY_DEPENDENCY_MISSING") from exc
    token_values = np.frombuffer(
        token_bytes,
        dtype=np.dtype("<u2"),
        count=materialized_count,
    )
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if materialized_count and int(token_values.max()) >= vocab_size:
        raise ContractViolation("TOKEN_ID_OUTSIDE_VOCAB")
    index_rows = _read_jsonl(
        index_path,
        expected_sha256=str(index_ref.get("sha256", "")),
        expected_bytes=index_ref.get("bytes"),
    )
    if len(index_rows) != source["staged_num_examples"]:
        raise ContractViolation("CORPUS_INDEX_COUNT_MISMATCH")
    eos = manifest.get("document_separator")
    if (
        not isinstance(eos, Mapping)
        or eos.get("text") != policy["document_separator"]
        or eos.get("token_id") != tokenizer.token_to_id(policy["document_separator"])
    ):
        raise ContractViolation("DOCUMENT_SEPARATOR_MISMATCH")

    source_cursor = 0
    materialized_cursor = 0
    source_token_total = 0
    discarded_total = 0
    partial_ordinals: list[int] = []
    expected_row_fields = {
        "ordinal",
        "source_text_sha256",
        "source_token_start",
        "source_token_count",
        "materialized_token_start",
        "materialized_token_count",
        "discarded_token_count",
        "partial",
    }
    for source_row, index_row in zip(
        _iter_verified_documents(source), index_rows, strict=True
    ):
        if set(index_row) != expected_row_fields:
            raise ContractViolation("CORPUS_INDEX_SCHEMA_MISMATCH")
        document_ids = _encode(tokenizer, source_row["text"]) + [int(eos["token_id"])]
        kept = min(
            max(0, policy["corpus_model_token_cap"] - materialized_cursor),
            len(document_ids),
        )
        expected_partial = 0 < kept < len(document_ids)
        if (
            index_row["ordinal"] != source_row["ordinal"]
            or index_row["source_text_sha256"] != source_row["text_sha256"]
            or index_row["source_token_start"] != source_cursor
            or index_row["source_token_count"] != len(document_ids)
            or index_row["materialized_token_start"] != materialized_cursor
            or index_row["materialized_token_count"] != kept
            or index_row["discarded_token_count"] != len(document_ids) - kept
            or index_row["partial"] is not expected_partial
        ):
            raise ContractViolation("CORPUS_INDEX_MISMATCH")
        if kept and not np.array_equal(
            token_values[materialized_cursor : materialized_cursor + kept],
            np.asarray(document_ids[:kept], dtype=np.dtype("<u2")),
        ):
            raise ContractViolation("CORPUS_TOKEN_CONTENT_MISMATCH")
        if expected_partial:
            partial_ordinals.append(source_row["ordinal"])
        source_cursor += len(document_ids)
        source_token_total += len(document_ids)
        materialized_cursor += kept
        discarded_total += len(document_ids) - kept
    expected_partial_ordinal = partial_ordinals[0] if partial_ordinals else None
    if len(partial_ordinals) > 1:
        raise ContractViolation("MULTIPLE_PARTIAL_DOCUMENTS")
    if (
        source_token_total != manifest.get("source_token_count_including_separators")
        or materialized_cursor != materialized_count
        or discarded_total != manifest.get("discarded_token_count")
        or manifest.get("source_document_count") != len(index_rows)
        or manifest.get("partial_final_document", {}).get("ordinal")
        != expected_partial_ordinal
        or manifest.get("partial_final_document", {}).get("policy")
        != policy["partial_final_document"]
    ):
        raise ContractViolation("CORPUS_ACCOUNTING_MISMATCH")
    return {
        **manifest,
        "manifest_path": str(path.resolve()),
        "manifest_sha256": manifest_sha256,
        "tokens_path": str(tokens_path),
        "index_path": str(index_path),
        "verified_by_reencoding": True,
    }


def load_corpus_token_memmap(
    manifest_source: Path,
    *,
    production: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Return a read-only memmap of a private copy of verified token bytes."""
    verified = verify_corpus_manifest(manifest_source, production=production)
    try:
        import numpy as np
    except ImportError as exc:
        raise ContractViolation("NUMPY_DEPENDENCY_MISSING") from exc
    artifacts = verified.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(
        artifacts.get("tokens"), Mapping
    ):
        raise ContractViolation("INVALID_ARTIFACT_REF")
    token_ref = artifacts["tokens"]
    token_bytes = _verified_artifact_bytes(
        Path(verified["tokens_path"]),
        expected_sha256=str(verified["token_stream_sha256"]),
        expected_bytes=token_ref.get("bytes"),
    )
    expected_bytes = int(verified["materialized_token_count"]) * 2
    if len(token_bytes) != expected_bytes:
        raise ContractViolation("CORPUS_TOKEN_FILE_SIZE_MISMATCH")

    # Mapping the verified source path again would reopen the TOCTOU window.
    # An anonymous private file keeps memmap semantics while the returned
    # mapping remains attached only to bytes already checked above.
    with tempfile.TemporaryFile(mode="w+b") as private_copy:
        private_copy.write(token_bytes)
        private_copy.flush()
        private_copy.seek(0)
        values = np.memmap(
            private_copy,
            dtype=np.dtype("<u2"),
            mode="r",
            shape=(verified["materialized_token_count"],),
        )
    return values, verified


def _failure_code(exc: Exception) -> str:
    if isinstance(exc, ContractViolation):
        return exc.code
    if isinstance(exc, ReferenceContractError):
        text = str(exc)
        return text if text and not any(ch.isspace() for ch in text) else type(exc).__name__
    return type(exc).__name__


def audit_lexical_token_boundaries(
    tokenizer_manifest: Path,
    concepts: Sequence[Mapping[str, Any]],
    *,
    output_path: Path | None = None,
    production: bool = True,
    evaluation_plan_path: Path = EVALUATION_PLAN_PATH,
    scope_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    """Audit every frozen wrapper/mode/target context before experiment freeze."""
    tokenizer, tokenizer_info = load_verified_tokenizer(
        tokenizer_manifest, production=production
    )
    plan_path = Path(evaluation_plan_path).resolve()
    plan_ref = tokenizer_info.get("evaluation_plan")
    if not isinstance(plan_ref, Mapping):
        raise ContractViolation("INVALID_ARTIFACT_REF")
    if plan_path != Path(tokenizer_info["evaluation_plan_path"]).resolve():
        from .prompt_boundary_revision import resolve_boundary_plan

        plan_ref = resolve_boundary_plan(evaluation_plan_path, plan_ref)
    plan_sha256 = require_sha256(
        str(plan_ref.get("sha256", "")), "INVALID_ARTIFACT_SHA256"
    )
    plan = load_evaluation_plan(
        plan_path,
        expected_sha256=plan_sha256,
        expected_bytes=plan_ref.get("bytes"),
    )
    prompt_plan = plan["prompt_plan"]
    templates = prompt_plan["templates"]
    context_length = int(plan["model"]["context_length"])
    failures: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    contexts_checked = 0
    contexts_failed = 0
    lexical_metrics: list[dict[str, Any]] = []
    concept_hashes: list[str] = []

    concept_ids = [
        concept.get("concept_id") if isinstance(concept, Mapping) else None
        for concept in concepts
    ]
    if (
        any(not isinstance(value, str) or not value for value in concept_ids)
        or len(concept_ids) != len(set(concept_ids))
    ):
        raise ContractViolation("DUPLICATE_OR_EMPTY_CONCEPT_ID")
    ordered_concepts = sorted(concepts, key=lambda row: str(row["concept_id"]))

    for concept in ordered_concepts:
        concept_id = concept.get("concept_id")
        answers = concept.get("answers")
        glosses = concept.get("glosses")
        record_hash = concept.get("concept_record_sha256")
        require_sha256(str(record_hash or ""), "CONCEPT_RECORD_HASH_MISSING")
        if record_hash != _logical_hash(_without(concept, "concept_record_sha256")):
            raise ContractViolation("CONCEPT_RECORD_HASH_MISMATCH")
        if set(answers or {}) != set(LANGUAGES) or set(glosses or {}) != set(LANGUAGES):
            raise ContractViolation("FOUR_LANGUAGE_CONCEPT_REQUIRED")
        canonical_answers = {language: canonical(str(answers[language])) for language in LANGUAGES}
        concept_hashes.append(record_hash)
        expression_metrics = expression_only_similarity_metrics(tokenizer, answers)
        language_metrics = expression_metrics["languages"]
        pair_metrics = expression_metrics["pairs"]
        lexical_metrics.append(
            {
                "concept_id": concept_id,
                "concept_record_sha256": record_hash,
                # Retain the original v4 flat fields for existing consumers.
                "en_token_length": language_metrics["en"]["token_length"],
                "fr_token_length": language_metrics["fr"]["token_length"],
                "en_fr_token_jaccard": pair_metrics["en-fr"]["token_set_jaccard"],
                "four_language_metrics": expression_metrics,
            }
        )
        distinct_answers = sorted(set(canonical_answers.values()))
        for wrapper, by_input in sorted(templates.items()):
            for input_language in LANGUAGES:
                for mode in ("ANY", "REQUESTED"):
                    requested_values: Sequence[str | None] = (
                        (None,) if mode == "ANY" else LANGUAGES
                    )
                    for requested_language in requested_values:
                        contexts_checked += 1
                        context = {
                            "concept_id": concept_id,
                            "wrapper": wrapper,
                            "input_language": input_language,
                            "mode": mode,
                            "requested_language": requested_language,
                        }
                        try:
                            template = by_input[input_language][mode]
                            language_name = ""
                            if requested_language is not None:
                                language_name = prompt_plan["target_language_names"][input_language][requested_language]
                            prefix = template.format(
                                gloss=glosses[input_language],
                                target_language_name=language_name,
                            )
                            prefix_ids = _encode(tokenizer, prefix)
                            if not prefix_ids:
                                raise ContractViolation("EMPTY_PREFIX_TOKENS")
                            events: dict[str, list[int]] = {}
                            for answer in distinct_answers:
                                full_ids = _encode(tokenizer, prefix + answer + "\n")
                                if full_ids[: len(prefix_ids)] != prefix_ids:
                                    raise ContractViolation("UNSTABLE_TOKEN_BOUNDARY")
                                continuation = full_ids[len(prefix_ids) :]
                                if not continuation:
                                    raise ContractViolation("EMPTY_TOKEN_EVENT")
                                if len(full_ids) > context_length:
                                    raise ContractViolation("LEXICAL_SEQUENCE_EXCEEDS_CONTEXT")
                                events[answer] = continuation
                            validate_token_events(events)
                        except Exception as exc:
                            code = _failure_code(exc)
                            failures[code] += 1
                            contexts_failed += 1
                            if len(examples) < 200:
                                examples.append({**context, "code": code})
    core = {
        "schema_version": VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "status": "PASS" if not failures else "BLOCKED_TOKEN_BOUNDARY",
        "data_kind": "REAL" if production else SYNTHETIC,
        "tokenizer_manifest_sha256": tokenizer_info["manifest_sha256"],
        "tokenizer_file_sha256": tokenizer_info["tokenizer_file_sha256"],
        "evaluation_plan_sha256": plan_sha256,
        "concept_record_hashes": sorted(concept_hashes),
        "contexts_checked": contexts_checked,
        "contexts_passed": contexts_checked - contexts_failed,
        "contexts_failed": contexts_failed,
        "failure_counts": dict(sorted(failures.items())),
        "failure_examples_first_200": examples,
        "expression_only_metrics": lexical_metrics,
        "context_length": context_length,
        "event_definition": "exact tokenizer continuation for canonical_answer + newline after exact prefix",
    }
    audit = {**core, "audit_sha256": _logical_hash(core)}
    if output_path is not None:
        target = require_relative_to(
            Path(output_path), scope_root, "TOKEN_AUDIT_OUTSIDE_SCOPE"
        )
        publish_json_once(target, audit)
    return audit


__all__ = [
    "EVALUATION_PLAN_PATH",
    "EXPRESSION_LANGUAGE_PAIRS",
    "EXPRESSION_METRICS_SCHEMA",
    "EXPRESSION_EXCLUDED_SPECIAL_TOKENS",
    "audit_lexical_token_boundaries",
    "expression_only_similarity_metrics",
    "iter_staged_documents",
    "load_corpus_token_memmap",
    "load_evaluation_plan",
    "load_verified_tokenizer",
    "materialize_corpus_tokens",
    "train_byte_bpe",
    "verify_corpus_manifest",
    "verify_tokenizer_manifest",
]
