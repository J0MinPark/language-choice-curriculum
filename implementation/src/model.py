"""Exact v4 GPT-2 construction, optimizer policy, and registered losses."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from .contracts import ContractViolation, PROJECT_ROOT, load_json


EVALUATION_PLAN_PATH = PROJECT_ROOT / "implementation" / "config" / "evaluation_plan.json"
EXPECTED_PARAMETER_COUNT = 40_044_544


@dataclass(frozen=True)
class ModelPolicy:
    vocab_size: int
    layers: int
    hidden_size: int
    attention_heads: int
    ffn_size: int
    context_length: int
    dropout: float
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int


def load_model_policy(path: Path = EVALUATION_PLAN_PATH) -> ModelPolicy:
    try:
        plan = load_json(path)
        tokenizer = plan["tokenizer"]
        model = plan["model"]
        special = tokenizer["special_tokens_in_id_order"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ContractViolation("INVALID_MODEL_POLICY") from exc
    if special != ["<|endoftext|>", "<|pad|>"]:
        raise ContractViolation("INVALID_SPECIAL_TOKEN_POLICY")
    policy = ModelPolicy(
        vocab_size=int(tokenizer["vocab_size"]),
        layers=int(model["layers"]),
        hidden_size=int(model["hidden_size"]),
        attention_heads=int(model["attention_heads"]),
        ffn_size=int(model["ffn_size"]),
        context_length=int(model["context_length"]),
        dropout=float(model["dropout"]),
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=1,
    )
    if policy != ModelPolicy(16384, 10, 512, 8, 2048, 256, 0.0, 0, 0, 1):
        raise ContractViolation("MODEL_POLICY_NOT_V4")
    return policy


def make_gpt2_config(policy: ModelPolicy | None = None) -> GPT2Config:
    policy = policy or load_model_policy()
    config = GPT2Config(
        vocab_size=policy.vocab_size,
        n_positions=policy.context_length,
        n_ctx=policy.context_length,
        n_embd=policy.hidden_size,
        n_layer=policy.layers,
        n_head=policy.attention_heads,
        n_inner=policy.ffn_size,
        resid_pdrop=policy.dropout,
        embd_pdrop=policy.dropout,
        attn_pdrop=policy.dropout,
        bos_token_id=policy.bos_token_id,
        eos_token_id=policy.eos_token_id,
        pad_token_id=policy.pad_token_id,
        tie_word_embeddings=True,
        use_cache=False,
    )
    # Transformers otherwise auto-selects SDPA for this config/version.  The
    # frozen implementation policy requires eager attention for replay.
    config._attn_implementation = "eager"
    return config


def seed_all(seed: int, *, include_cuda: bool) -> None:
    if not isinstance(seed, int) or seed < 0:
        raise ContractViolation("INVALID_ROOT_SEED")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if include_cuda:
        from .gpu_guard import require_literal_gpu2_mask

        require_literal_gpu2_mask()
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ContractViolation("BLOCKED_GPU2_UNAVAILABLE")
        torch.cuda.manual_seed_all(seed)


def configure_determinism() -> dict[str, Any]:
    """Set, and return, deterministic backend choices frozen for replay."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "cublas_workspace_config": __import__("os").environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def build_random_model(
    *,
    seed: int,
    policy: ModelPolicy | None = None,
    require_v4_parameter_count: bool = True,
) -> GPT2LMHeadModel:
    """Construct from config only; pretrained identifiers are not accepted."""
    seed_all(seed, include_cuda=False)
    model = GPT2LMHeadModel(make_gpt2_config(policy))
    model.tie_weights()
    count = sum(parameter.numel() for parameter in model.parameters())
    if require_v4_parameter_count and count != EXPECTED_PARAMETER_COUNT:
        raise ContractViolation("MODEL_PARAMETER_COUNT_MISMATCH")
    return model


def parameter_group_signature(model: torch.nn.Module) -> dict[str, tuple[str, ...]]:
    """Apply the frozen matrix/bias/norm/embedding decay rule by name and shape."""
    embedding_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, torch.nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    normalization_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, torch.nn.LayerNorm)
        for parameter in module.parameters(recurse=False)
    }
    decay: list[str] = []
    no_decay: list[str] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        identity = id(parameter)
        if identity in seen:
            raise ContractViolation("DUPLICATE_TRAINABLE_PARAMETER")
        seen.add(identity)
        if (
            parameter.ndim >= 2
            and identity not in embedding_parameter_ids
            and identity not in normalization_parameter_ids
        ):
            decay.append(name)
        else:
            no_decay.append(name)
    all_trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if (
        not decay
        or not no_decay
        or len(decay) + len(no_decay) != len(seen)
        or seen != all_trainable
    ):
        raise ContractViolation("INVALID_PARAMETER_GROUPS")
    return {"decay": tuple(decay), "no_decay": tuple(no_decay)}


def build_adamw(
    model: torch.nn.Module,
    training_policy: Mapping[str, Any],
    *,
    learning_rate: float | None = None,
) -> tuple[torch.optim.AdamW, dict[str, tuple[str, ...]]]:
    if training_policy.get("optimizer") != "torch.optim.AdamW":
        raise ContractViolation("OPTIMIZER_POLICY_MISMATCH")
    if training_policy.get("parameter_decay_rule") != (
        "matrix weights decay; bias, normalization weights, and embeddings do not decay"
    ):
        raise ContractViolation("PARAMETER_DECAY_POLICY_MISMATCH")
    signature = parameter_group_signature(model)
    by_name = dict(model.named_parameters())
    lr = float(training_policy["learning_rate"] if learning_rate is None else learning_rate)
    betas = tuple(float(value) for value in training_policy["betas"])
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [by_name[name] for name in signature["decay"]],
                "weight_decay": float(training_policy["weight_decay"]),
                "group_name": "decay",
            },
            {
                "params": [by_name[name] for name in signature["no_decay"]],
                "weight_decay": 0.0,
                "group_name": "no_decay",
            },
        ],
        lr=lr,
        betas=betas,
        eps=float(training_policy["eps"]),
        foreach=False,
        fused=False,
    )
    return optimizer, signature


class ConstantSchedule:
    """Explicit state for a registered constant learning-rate schedule."""

    def __init__(self, optimizer: torch.optim.Optimizer, learning_rate: float, phase: str):
        if not math.isfinite(learning_rate) or learning_rate <= 0 or not phase:
            raise ContractViolation("INVALID_SCHEDULER_STATE")
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)
        self.phase = phase
        self.steps = 0
        self._apply()

    def _apply(self) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def step(self) -> None:
        self.steps += 1
        self._apply()

    def transition(self, *, phase: str, learning_rate: float) -> None:
        if not phase or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ContractViolation("INVALID_SCHEDULER_TRANSITION")
        self.phase = phase
        self.learning_rate = float(learning_rate)
        self._apply()

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "CONSTANT_EXPLICIT_STATE",
            "learning_rate": self.learning_rate,
            "phase": self.phase,
            "steps": self.steps,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("kind") != "CONSTANT_EXPLICIT_STATE":
            raise ContractViolation("SCHEDULER_KIND_MISMATCH")
        learning_rate = float(state["learning_rate"])
        steps = int(state["steps"])
        phase = str(state["phase"])
        if not math.isfinite(learning_rate) or learning_rate <= 0 or steps < 0 or not phase:
            raise ContractViolation("INVALID_SCHEDULER_STATE")
        self.learning_rate = learning_rate
        self.steps = steps
        self.phase = phase
        self._apply()


def lexical_response_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Response CE: token mean per example, followed by the batch mean."""
    if logits.ndim != 3 or input_ids.ndim != 2 or response_mask.shape != input_ids.shape:
        raise ContractViolation("INVALID_LEXICAL_LOSS_SHAPE")
    if logits.shape[:2] != input_ids.shape or logits.shape[2] < 2:
        raise ContractViolation("INVALID_LEXICAL_LOSS_SHAPE")
    mask = response_mask[:, 1:].to(dtype=torch.bool)
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ContractViolation("INVALID_ATTENTION_MASK_SHAPE")
        mask &= attention_mask[:, 1:].to(dtype=torch.bool)
    counts = mask.sum(dim=1)
    if bool((counts == 0).any().item()):
        raise ContractViolation("EMPTY_RESPONSE_TOKENS")
    shifted_logits = logits[:, :-1, :].float()
    shifted_targets = input_ids[:, 1:]
    token_losses = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_targets.reshape(-1),
        reduction="none",
    ).reshape_as(shifted_targets)
    example_losses = (token_losses * mask).sum(dim=1) / counts
    return example_losses.mean()


def corpus_causal_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ordinary causal next-token CE over every non-padding continuation."""
    if logits.ndim != 3 or input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape:
        raise ContractViolation("INVALID_CORPUS_LOSS_SHAPE")
    targets = input_ids[:, 1:]
    losses = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    if attention_mask is None:
        return losses.mean()
    if attention_mask.shape != input_ids.shape:
        raise ContractViolation("INVALID_ATTENTION_MASK_SHAPE")
    mask = attention_mask[:, 1:].to(dtype=torch.bool)
    if not bool(mask.any().item()):
        raise ContractViolation("EMPTY_CORPUS_TOKENS")
    return losses.masked_select(mask).mean()
