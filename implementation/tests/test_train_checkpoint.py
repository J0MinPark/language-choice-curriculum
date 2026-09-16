from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from implementation.src.budget import (
    BudgetCaps,
    BudgetStop,
    GpuBudgetLedger,
    require_disk_reservation,
)
from implementation.src.checkpoint import (
    ResumeContract,
    capture_training_state,
    checkpoint_state_fingerprint,
    load_checkpoint_payload,
    restore_training_state,
    save_checkpoint,
)
from implementation.src.contracts import ContractViolation, canonical_json_bytes
from implementation.src.gpu_guard import _cuda_visible_memory_mib, require_literal_gpu2_mask, require_physical_gpu2
from implementation.src.model import (
    ConstantSchedule,
    lexical_response_loss,
    load_model_policy,
    make_gpt2_config,
    parameter_group_signature,
)
from implementation.src.records import (
    RecordSlot,
    StageConceptView,
    encode_record,
    materialize_record,
    materialize_record_bank,
)
from implementation.src.schedule import (
    audit_history_plan,
    audit_history_record_content,
    build_history_plan,
    build_history_plan_from_verified_freeze,
    build_stage_plan,
)
from implementation.src.train import (
    StatefulBatchStream,
    TrainBatch,
    evaluate_without_rng_consumption,
    audit_initialization_fingerprints,
    require_campaign_training_gates,
    run_cpu_fixture_steps,
    validate_phase_transition,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
GPU2_UUID = "GPU-de18b86c-419a-795a-0667-c7ae03bf800f"


def fixture_concept(concept_id: str) -> dict:
    return {
        "concept_id": concept_id,
        "answers": {
            "ko": "한국어" + concept_id,
            "en": "english" + concept_id,
            "zh": "中文" + concept_id,
            "fr": "français" + concept_id,
        },
        "glosses": {
            "ko": "한국어 정의 " + concept_id,
            "en": "English definition " + concept_id,
            "zh": "中文定义 " + concept_id,
            "fr": "définition française " + concept_id,
        },
        "source_record_hashes": [HASH_A],
    }


class ByteTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(text.encode("utf-8"))


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


class GpuGuardTests(unittest.TestCase):
    def test_literal_mask_has_no_alternative_or_fallback(self):
        for value in (None, "", "0", "1", "2,0", GPU2_UUID, " 2"):
            environment = {} if value is None else {"CUDA_VISIBLE_DEVICES": value}
            with self.assertRaisesRegex(ContractViolation, "BLOCKED_GPU2_MASK"):
                require_literal_gpu2_mask(environment)
        with self.assertRaisesRegex(ContractViolation, "BLOCKED_GPU_DEVICE_ORDER"):
            require_literal_gpu2_mask({"CUDA_VISIBLE_DEVICES": "2"})
        require_literal_gpu2_mask(
            {"CUDA_VISIBLE_DEVICES": "2", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        )

    def test_physical_uuid_and_process_are_proved(self):
        process_queries = 0

        def output(argv):
            nonlocal process_queries
            joined = " ".join(argv)
            if "memory.reserved" in joined:
                return f"{GPU2_UUID}, 97887, 638\n"
            if "query-gpu" in joined:
                return (
                    "0, GPU-zero, 00000000:01:00.0, other, 100\n"
                    f"2, {GPU2_UUID}, 00000000:60:00.0, expected, 97887\n"
                )
            process_queries += 1
            return "" if process_queries == 1 else f"{GPU2_UUID}, {os.getpid()}\n"

        class Properties:
            name = "expected"
            total_memory = 97249 * 1024 * 1024

        class Cuda:
            @staticmethod
            def is_available(): return True
            @staticmethod
            def device_count(): return 1
            @staticmethod
            def set_device(index): self.assertEqual(index, 0)
            @staticmethod
            def synchronize(index): self.assertEqual(index, 0)
            @staticmethod
            def get_device_properties(index): return Properties()
            @staticmethod
            def is_bf16_supported(): return True

        class Version:
            cuda = "12.8"

        class FakeTorch:
            __version__ = "test"
            version = Version()
            cuda = Cuda()
            @staticmethod
            def empty(size, device):
                self.assertEqual((size, device), (1, "cuda:0"))
                return object()

        binding = require_physical_gpu2(
            environ={"CUDA_VISIBLE_DEVICES": "2", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
            command_output=output,
            torch_module=FakeTorch(),
        )
        self.assertEqual(binding.physical_index, 2)
        self.assertEqual(binding.logical_index, 0)
        self.assertEqual(binding.uuid, GPU2_UUID)

        Properties.total_memory = 97887 * 1024 * 1024
        with self.assertRaisesRegex(ContractViolation, "BLOCKED_GPU2_PROPERTY_MISMATCH"):
            require_physical_gpu2(
                environ={"CUDA_VISIBLE_DEVICES": "2", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
                command_output=output, torch_module=FakeTorch(),
            )

    def test_memory_accounting_fails_closed(self):
        physical = {"uuid": GPU2_UUID, "total_memory_mib": 97887}
        for raw in (
            "", f"{GPU2_UUID}, 97887, N/A", f"{GPU2_UUID}, 97887, -1",
            f"{GPU2_UUID}, 97887, 97887", f"{GPU2_UUID}, 97888, 638",
            "GPU-wrong, 97887, 638", f"{GPU2_UUID}, 97887, 638\n" * 2,
        ):
            with self.subTest(raw=raw), self.assertRaisesRegex(
                ContractViolation, "BLOCKED_GPU2_MEMORY_ACCOUNTING"
            ):
                _cuda_visible_memory_mib(physical, lambda argv: raw)
        self.assertEqual(_cuda_visible_memory_mib(
            physical, lambda argv: f"{GPU2_UUID}, 97887, 0"
        ), 97887)

    def test_cuda_unavailable_never_falls_back(self):
        calls = 0
        def output(argv):
            nonlocal calls
            if "query-gpu" in " ".join(argv):
                return f"2, {GPU2_UUID}, 00000000:60:00.0, expected, 97887\n"
            calls += 1
            return ""
        class Version: cuda = "12.8"
        class Cuda:
            @staticmethod
            def is_available(): return False
            @staticmethod
            def device_count(): return 0
        class FakeTorch:
            version = Version()
            cuda = Cuda()
        with self.assertRaisesRegex(ContractViolation, "BLOCKED_GPU2_UNAVAILABLE"):
            require_physical_gpu2(
                environ={"CUDA_VISIBLE_DEVICES": "2", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
                command_output=output,
                torch_module=FakeTorch(),
            )


class BudgetTests(unittest.TestCase):
    def test_prior_usage_campaign_and_root_caps_are_reserved(self):
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            ledger = GpuBudgetLedger(
                Path(directory) / "gpu.jsonl",
                BudgetCaps(24.0, 0.953, 6.0),
                monotonic=clock,
            )
            lease = ledger.reserve(run_id="run1", root_id=4101, phase="T0", reserved_seconds=100)
            snapshot = ledger.snapshot()
            self.assertAlmostEqual(
                snapshot["campaign_gpu_hours_charged_or_reserved"],
                0.953 + 100 / 3600,
            )
            # A dangling/crashed lease remains charged at its full reservation.
            self.assertEqual(snapshot["per_root_gpu_seconds_charged_or_reserved"]["4101"], 100)
            clock.value += 10
            ledger.finish(lease, "PASS")
            self.assertEqual(
                ledger.snapshot()["per_root_gpu_seconds_charged_or_reserved"]["4101"],
                10,
            )

    def test_budget_deadline_and_duplicate_finish(self):
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            ledger = GpuBudgetLedger(Path(directory) / "gpu.jsonl", BudgetCaps(24, 0.953, 6), monotonic=clock)
            lease = ledger.reserve(run_id="run", root_id=4101, phase="T0", reserved_seconds=20)
            clock.value += 16
            with self.assertRaises(BudgetStop):
                ledger.require_time_remaining(lease, checkpoint_grace_seconds=5)
            ledger.finish(lease, "BLOCKED_GPU_BUDGET_DEADLINE")
            with self.assertRaisesRegex(ContractViolation, "GPU_LEASE_NOT_ACTIVE"):
                ledger.finish(lease, "DUPLICATE")

    def test_root_and_campaign_overcommit_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = GpuBudgetLedger(Path(directory) / "gpu.jsonl", BudgetCaps(1.0, 0.5, 0.25))
            with self.assertRaisesRegex(ContractViolation, "BLOCKED_ROOT_GPU_BUDGET"):
                ledger.reserve(run_id="root", root_id=4101, phase="T0", reserved_seconds=901)
            with self.assertRaisesRegex(ContractViolation, "BLOCKED_CAMPAIGN_GPU_BUDGET"):
                ledger.reserve(run_id="campaign", root_id=None, phase="REPLAY", reserved_seconds=1801)

    def test_disk_headroom_is_not_spendable(self):
        result = require_disk_reservation(Path("."), planned_bytes=20, emergency_free_bytes=10, available_bytes=31)
        self.assertEqual(result["projected_free_bytes"], 11)
        with self.assertRaisesRegex(ContractViolation, "BLOCKED_DISK_BUDGET"):
            require_disk_reservation(Path("."), planned_bytes=22, emergency_free_bytes=10, available_bytes=31)


class ModelRecordScheduleTests(unittest.TestCase):
    def test_exact_model_config(self):
        policy = load_model_policy()
        config = make_gpt2_config(policy)
        self.assertEqual((config.n_layer, config.n_embd, config.n_head, config.n_inner), (10, 512, 8, 2048))
        self.assertEqual((config.n_positions, config.vocab_size), (256, 16384))
        self.assertEqual((config.resid_pdrop, config.embd_pdrop, config.attn_pdrop), (0.0, 0.0, 0.0))
        self.assertEqual(config._attn_implementation, "eager")

    def test_parameter_decay_rule(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(5, 3)
                self.norm = torch.nn.LayerNorm(3)
                self.linear = torch.nn.Linear(3, 4)
        signature = parameter_group_signature(Toy())
        self.assertEqual(signature["decay"], ("linear.weight",))
        self.assertIn("embedding.weight", signature["no_decay"])
        self.assertIn("norm.weight", signature["no_decay"])
        self.assertIn("linear.bias", signature["no_decay"])

    def test_lexical_loss_is_example_then_batch_mean(self):
        logits = torch.tensor(
            [
                [[0.0, 0.0], [4.0, -4.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
        ids = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])
        response = torch.tensor([[0, 0, 1, 0], [0, 1, 1, 1]], dtype=torch.bool)
        attention = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
        result = lexical_response_loss(logits, ids, response, attention)
        one = torch.nn.functional.cross_entropy(logits[0, 1].view(1, 2), ids[0, 2].view(1))
        three = torch.nn.functional.cross_entropy(logits[1, :3], ids[1, 1:])
        self.assertTrue(torch.allclose(result, (one + three) / 2))

    def test_future_language_is_structurally_inaccessible(self):
        concept = fixture_concept("c")
        view = StageConceptView(concept, ("ko",))
        with self.assertRaisesRegex(ContractViolation, "FUTURE_TARGET_LANGUAGE_ACCESS"):
            view.answer("en")
        slot = RecordSlot("id", "c", "ko", "en", "ANY", "train1", 0)
        with self.assertRaisesRegex(ContractViolation, "FUTURE_LANGUAGE_IN_TRAINING_SLOT"):
            materialize_record(concept, slot, stage="T0")

    def test_materialized_record_and_token_boundary(self):
        slot = RecordSlot("id", "c", "ko", "ko", "REQUESTED", "train1", 0)
        record = materialize_record(fixture_concept("c"), slot, stage="T0")
        encoded = encode_record(record, ByteTokenizer(), context_length=1024)
        self.assertEqual(len(encoded.input_ids), len(encoded.response_mask))
        self.assertGreater(sum(encoded.response_mask), 1)
        self.assertNotIn("English definition", record.full_text)

    def test_t3_supercycle_exact(self):
        plan = build_stage_plan(["a", "b", "c", "d"], root_id=4101, stage="T3", optimizer_updates=3)
        self.assertEqual(plan.counts["target"], {"en": 16, "fr": 48, "ko": 16, "zh": 16})
        self.assertEqual(len(plan.batches), 3)
        self.assertTrue(all(len(batch.record_ids) == 32 for batch in plan.batches))

    def test_b36_rounds_batches_pairs_and_content(self):
        pairs = [(f"c{i:02d}a", f"c{i:02d}b") for i in range(30)]
        plan = build_history_plan(pairs, root_id=4101)
        audit = audit_history_plan(plan)
        self.assertEqual(audit["target_records_per_branch"], 9600)
        self.assertEqual(audit["optimizer_updates_per_branch"], 300)
        concepts = [fixture_concept(value) for pair in pairs for value in pair]
        bank = materialize_record_bank(concepts, plan.record_slots, stage="H")
        self.assertEqual(audit_history_record_content(plan, bank)["distinct_record_content"], 9600)

    def test_b36_production_plan_consumes_hash_bound_frozen_pairs(self):
        concepts = [fixture_concept(f"c{index:02d}") for index in range(60)]
        rows = []
        for pair_index in range(30):
            core = {
                "pair_id": f"B36-pair-{pair_index:02d}",
                "pair_index": pair_index,
                "concept_ids": [f"c{pair_index * 2:02d}", f"c{pair_index * 2 + 1:02d}"],
            }
            rows.append(
                {
                    **core,
                    "pair_sha256": hashlib.sha256(canonical_json_bytes(core)).hexdigest(),
                }
            )
        history_core = {
            "schema_version": "4.0.0",
            "implementation_revision": "4.0.0-r1",
            "status": "PASS",
            "history_family": "B36",
            "pairing_rule": "lexicographically sort frozen concept_id and pair adjacent positions (0,1),(2,3),...",
            "concept_order": [f"c{index:02d}" for index in range(60)],
            "pairs": rows,
        }
        pair_set_hash = hashlib.sha256(canonical_json_bytes(history_core)).hexdigest()
        freeze = {
            "verified": True,
            "status": "PASS",
            "freeze_gate_status": "PASS",
            "data_kind": "REAL",
            "concepts": concepts,
            "history_pairs": {
                **history_core,
                "history_pair_set_sha256": pair_set_hash,
            },
        }
        plan = build_history_plan_from_verified_freeze(freeze, root_id=4101)
        self.assertEqual(plan.frozen_pair_set_sha256, pair_set_hash)
        self.assertEqual(plan.pairs[0], ("c00", "c01"))
        tampered = copy.deepcopy(freeze)
        tampered["history_pairs"]["pairs"][0]["concept_ids"].reverse()
        with self.assertRaisesRegex(ContractViolation, "FROZEN_HISTORY_PAIR_HASH_MISMATCH"):
            build_history_plan_from_verified_freeze(tampered, root_id=4101)


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 5)
        self.projection = torch.nn.Linear(5, 7)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        return type("Output", (), {"logits": self.projection(self.embedding(input_ids))})()


def make_batches(count=14):
    batches = []
    for index in range(count):
        ids = torch.tensor([[1, 2, 3, (index + 4) % 7], [2, 3, 4, (index + 5) % 7]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        response = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.bool)
        batches.append(
            TrainBatch(
                (f"r{index}a", f"r{index}b"),
                (f"{index:064x}", f"{index + 100:064x}"),
                ids,
                mask,
                response,
            )
        )
    return batches


def make_progress(stream):
    return {
        "global_step": 0,
        "phase_step": 0,
        "loader_state": stream.state_dict(),
        "accumulation_step": 0,
        "examples_seen": 0,
        "model_tokens_seen": 0,
        "plan_sha256": HASH_A,
    }


def make_lineage():
    return {
        "root_id": 4101,
        "phase": "T3",
        "stage": "T3",
        "branch": None,
        "phase_parent_sha256": HASH_B,
        "resume_parent_sha256": None,
        "initial_model_fingerprint": HASH_A,
        "freeze_sha256": HASH_A,
        "config_sha256": HASH_A,
        "tokenizer_sha256": HASH_A,
        "code_sha256": HASH_A,
        "evaluation_plan_sha256": HASH_A,
        "data_kind": "SYNTHETIC_TEST_FIXTURE",
    }


def make_training_objects(seed=19):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = TinyLM()
    signature = {"all": tuple(name for name, _ in model.named_parameters())}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False, fused=False)
    scheduler = ConstantSchedule(optimizer, 0.001, "T3")
    generator = torch.Generator().manual_seed(77)
    stream = StatefulBatchStream(make_batches(), plan_sha256=HASH_A, generator=generator)
    return model, optimizer, scheduler, generator, stream, signature


class CheckpointReplayTests(unittest.TestCase):
    def _run(self, model, optimizer, scheduler, stream, progress, steps):
        return run_cpu_fixture_steps(
            model,
            optimizer,
            scheduler,
            stream,
            progress,
            data_kind="SYNTHETIC_TEST_FIXTURE",
            steps=steps,
            loss_kind="lexical",
            grad_clip_norm=1.0,
        )[0]

    def test_continuous_equals_save_resume(self):
        model, optimizer, scheduler, generator, stream, signature = make_training_objects()
        progress = make_progress(stream)
        self._run(model, optimizer, scheduler, stream, progress, 2)
        payload = capture_training_state(
            model,
            optimizer,
            scheduler=scheduler,
            scaler=None,
            generators={"loader": generator},
            progress=progress,
            lineage=make_lineage(),
            parameter_group_signature=signature,
            deterministic_settings={"fixture": True},
            require_cuda_rng=False,
        )
        expected_traces = self._run(model, optimizer, scheduler, stream, progress, 10)
        expected_state = checkpoint_state_fingerprint(
            model,
            optimizer,
            scheduler,
            progress,
            generators={"loader": generator},
        )

        resumed_model, resumed_optimizer, resumed_scheduler, resumed_generator, resumed_stream, resumed_signature = make_training_objects(999)
        contract = ResumeContract(
            4101, "T3", "T3", None,
            HASH_A, HASH_A, HASH_A, HASH_A, HASH_A, HASH_A,
            "SYNTHETIC_TEST_FIXTURE", False,
            phase_parent_sha256=HASH_B,
        )
        resumed_progress = restore_training_state(
            payload,
            resumed_model,
            resumed_optimizer,
            scheduler=resumed_scheduler,
            scaler=None,
            generators={"loader": resumed_generator},
            expected=contract,
            parameter_group_signature=resumed_signature,
            expected_deterministic_settings={"fixture": True},
        )
        resumed_stream.load_state_dict(resumed_progress["loader_state"])
        actual_traces = self._run(
            resumed_model, resumed_optimizer, resumed_scheduler, resumed_stream, resumed_progress, 10
        )
        self.assertEqual(expected_traces, actual_traces)
        self.assertEqual(
            expected_state,
            checkpoint_state_fingerprint(
                resumed_model,
                resumed_optimizer,
                resumed_scheduler,
                resumed_progress,
                generators={"loader": resumed_generator},
            ),
        )

    def test_disk_roundtrip_and_corruption_rejected(self):
        model, optimizer, scheduler, generator, stream, signature = make_training_objects()
        progress = make_progress(stream)
        self._run(model, optimizer, scheduler, stream, progress, 1)
        payload = capture_training_state(
            model, optimizer, scheduler=scheduler, scaler=None,
            generators={"loader": generator}, progress=progress, lineage=make_lineage(),
            parameter_group_signature=signature, deterministic_settings={"fixture": True},
            require_cuda_rng=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            ref = save_checkpoint(path, payload)
            loaded, manifest = load_checkpoint_payload(
                path,
                expected_state_sha256=ref.state_sha256,
                expected_manifest_sha256=ref.manifest_sha256,
            )
            self.assertEqual(loaded["progress"], progress)
            self.assertEqual(manifest["model_fingerprint"], ref.model_fingerprint)
            state_path = path / "state.pt"
            os.chmod(state_path, 0o644)
            with state_path.open("ab") as handle:
                handle.write(b"damage")
            with self.assertRaises(Exception):
                load_checkpoint_payload(path, expected_state_sha256=ref.state_sha256)

    def test_mid_accumulation_and_loader_mismatch_rejected(self):
        model, optimizer, scheduler, generator, stream, signature = make_training_objects()
        progress = make_progress(stream)
        progress["accumulation_step"] = 1
        with self.assertRaisesRegex(ContractViolation, "SAVE_ONLY_AT_ACCUMULATION_BOUNDARY"):
            capture_training_state(
                model, optimizer, scheduler=scheduler, scaler=None,
                generators={"loader": generator}, progress=progress, lineage=make_lineage(),
                parameter_group_signature=signature, deterministic_settings={}, require_cuda_rng=False,
            )
        state = stream.state_dict()
        state["next_record_ids"] = ["wrong"]
        with self.assertRaisesRegex(ContractViolation, "LOADER_NEXT_RECORD_MISMATCH"):
            stream.load_state_dict(state)

    def test_different_optimizer_class_is_rejected(self):
        model, optimizer, scheduler, generator, stream, signature = make_training_objects()
        progress = make_progress(stream)
        payload = capture_training_state(
            model,
            optimizer,
            scheduler=scheduler,
            scaler=None,
            generators={"loader": generator},
            progress=progress,
            lineage=make_lineage(),
            parameter_group_signature=signature,
            deterministic_settings={"fixture": True},
            require_cuda_rng=False,
        )
        other_model = TinyLM()
        other_optimizer = torch.optim.SGD(other_model.parameters(), lr=0.001)
        other_scheduler = ConstantSchedule(other_optimizer, 0.001, "T3")
        other_generator = torch.Generator().manual_seed(77)
        with self.assertRaisesRegex(ContractViolation, "OPTIMIZER_CLASS_MISMATCH"):
            restore_training_state(
                payload,
                other_model,
                other_optimizer,
                scheduler=other_scheduler,
                scaler=None,
                generators={"loader": other_generator},
                expected=ResumeContract(
                    4101,
                    "T3",
                    "T3",
                    None,
                    HASH_A,
                    HASH_A,
                    HASH_A,
                    HASH_A,
                    HASH_A,
                    HASH_A,
                    "SYNTHETIC_TEST_FIXTURE",
                    False,
                    phase_parent_sha256=HASH_B,
                ),
                parameter_group_signature=signature,
                expected_deterministic_settings={"fixture": True},
            )

    def test_evaluation_rng_use_is_detected_and_rolled_back(self):
        generator = torch.Generator().manual_seed(1)
        before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(ContractViolation, "EVALUATION_CONSUMED_TRAINING_RNG"):
            evaluate_without_rng_consumption(
                lambda: (random.random(), np.random.random(), torch.rand(1)),
                generators={"loader": generator},
            )
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_cpu_path_rejects_real_data(self):
        model, optimizer, scheduler, generator, stream, _ = make_training_objects()
        with self.assertRaisesRegex(ContractViolation, "CPU_PATH_REQUIRES_SYNTHETIC_FIXTURE"):
            run_cpu_fixture_steps(
                model, optimizer, scheduler, stream, make_progress(stream),
                data_kind="REAL", steps=1, loss_kind="lexical", grad_clip_norm=1,
            )

    def test_budget_stop_checkpoints_at_optimizer_boundary(self):
        model, optimizer, scheduler, _, stream, _ = make_training_objects()
        progress = make_progress(stream)
        checks = 0
        saved = []

        def budget_check():
            nonlocal checks
            checks += 1
            if checks == 2:
                raise BudgetStop("BLOCKED_GPU_BUDGET_DEADLINE")

        with self.assertRaisesRegex(BudgetStop, "BLOCKED_GPU_BUDGET_DEADLINE"):
            run_cpu_fixture_steps(
                model,
                optimizer,
                scheduler,
                stream,
                progress,
                data_kind="SYNTHETIC_TEST_FIXTURE",
                steps=2,
                loss_kind="lexical",
                grad_clip_norm=1,
                budget_check=budget_check,
                boundary_checkpoint=lambda reason: saved.append(
                    (reason, progress["phase_step"], progress["accumulation_step"])
                ),
            )
        self.assertEqual(saved, [("BLOCKED_GPU_BUDGET_DEADLINE", 1, 0)])

    def test_lineage_rejects_cross_root_and_skips(self):
        parent = {"lineage": {"root_id": 4101, "phase": "T1", "branch": None}}
        validate_phase_transition(child_phase="T2", child_root_id=4101, parent_manifest=parent, resume=False)
        with self.assertRaisesRegex(ContractViolation, "CROSS_ROOT_LINEAGE"):
            validate_phase_transition(child_phase="T2", child_root_id=4102, parent_manifest=parent, resume=False)
        with self.assertRaisesRegex(ContractViolation, "INVALID_PHASE_TRANSITION"):
            validate_phase_transition(child_phase="T3", child_root_id=4101, parent_manifest=parent, resume=False)
        bad_fork = {"lineage": {"root_id": 4101, "phase": "H_BASE", "branch": "A"}}
        for branch in ("H_A", "H_B"):
            with self.assertRaisesRegex(ContractViolation, "INVALID_HISTORY_FORK_PARENT"):
                validate_phase_transition(
                    child_phase=branch,
                    child_root_id=4101,
                    parent_manifest=bad_fork,
                    resume=False,
                )
        validate_phase_transition(
            child_phase="H_B",
            child_root_id=4101,
            parent_manifest={"lineage": {"root_id": 4101, "phase": "H_B", "branch": "B"}},
            resume=True,
        )
        with self.assertRaisesRegex(ContractViolation, "INVALID_RESUME_BRANCH"):
            validate_phase_transition(
                child_phase="H_B",
                child_root_id=4101,
                parent_manifest={"lineage": {"root_id": 4101, "phase": "H_B", "branch": "A"}},
                resume=True,
            )

    def test_campaign_expansion_and_unique_initializations(self):
        base = {name: "PASS" for name in ("DATA_QA", "EXPERIMENT_FREEZE", "CPU_INTEGRATION", "GPU_REPLAY")}
        require_campaign_training_gates(root_id=4101, phase="T0", gates=base, main_enabled=False)
        with self.assertRaisesRegex(ContractViolation, "ROOT_4101_T3_MEASUREMENT"):
            require_campaign_training_gates(root_id=4102, phase="T0", gates=base, main_enabled=False)
        with self.assertRaisesRegex(ContractViolation, "ROOT_4104_T3_READINESS"):
            require_campaign_training_gates(
                root_id=4101,
                phase="H_BASE",
                gates={**base, **{
                    f"ROOT_{root}_{kind}": "PASS"
                    for root in (4101, 4102, 4103)
                    for kind in ("T3_READINESS", "T3_MEASUREMENT")
                }},
                main_enabled=False,
            )
        audit_initialization_fingerprints({4101: HASH_A, 4102: HASH_B}, require_all_roots=False)
        with self.assertRaisesRegex(ContractViolation, "ROOT_INITIALIZATION_NOT_DISTINCT"):
            audit_initialization_fingerprints({4101: HASH_A, 4102: HASH_A}, require_all_roots=False)


if __name__ == "__main__":
    unittest.main()
