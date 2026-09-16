"""End-to-end fixed-endpoint pilot on physical GPU2, with verified resume.

Every optimizer step runs through gpu2_training_session/run_gpu2_stage_steps.
The journal, not filenames or console logs, determines completed work.
"""
from dataclasses import asdict
from pathlib import Path
import gc
import platform
import uuid

import torch

from .artifacts import publish_json_once, read_verified_json, sha256_file
from .budget import BudgetCaps, BudgetStop, GpuBudgetLedger, ExclusiveGpu2Lock, require_disk_reservation
from .build_tokenizer import load_verified_tokenizer, load_corpus_token_memmap
from .checkpoint import ResumeContract, capture_training_state, save_checkpoint, load_checkpoint_payload, restore_training_state
from .contracts import ContractViolation, WORK_ROOT, PROJECT_ROOT, LANGUAGES
from .integrity import verify_implementation_manifest, verify_cpu_check_evidence
from .model import build_random_model, build_adamw, ConstantSchedule, configure_determinism, seed_all
from .pilot_execution_freeze import PHASES, TRAJECTORY_REVISION, REVERSE_REVISION, verify_execution_freeze
from .records import stage_active_languages
from .pilot_journal import PilotJournal
from .pilot_streams import corpus_stream, lexical_stream, evaluation_records
from .report import _verify_gpu_replay_evidence, compute_readiness, compute_measurement, LINEAGE_FIELDS, _probability_values
from .score import score_checkpoint, semantic_model_fingerprint
from .train import (gpu2_training_session, run_gpu2_stage_steps, BoundaryStopController,
    validate_phase_transition, require_campaign_training_gates, PHASE_PARENT,
    audit_initialization_fingerprints, evaluate_without_rng_consumption)
from .semantic_experiment_freeze import _hash
from .schedule import build_history_plan_from_verified_freeze


def bound_json(ref):
    value = read_verified_json(Path(ref["path"]), expected_sha256=ref["sha256"])
    if Path(ref["path"]).stat().st_size != ref["bytes"]:
        raise ContractViolation("RUNTIME_ARTIFACT_SIZE_MISMATCH")
    return value


def gate_results(scored, records, concepts, stage, *, language_order=LANGUAGES):
    """Recompute exact gates from score rows, never accept a supplied PASS."""
    provenance = scored["provenance"]
    primary = [r for r in records if r["input_language"] == "ko" and r["split"] == "dev"]
    results = {}
    for name, mode in (("readiness", "REQUESTED"), ("measurement", "ANY")):
        if name == "measurement" and stage != "T3": continue
        subset = [r for r in primary if r["mode"] == mode]
        ids = {r["record_id"] for r in subset}
        rows = [r for r in scored["rows"] if r["record_id"] in ids]
        plan = {k:provenance[k] for k in LINEAGE_FIELDS}
        plan.update({"expected_record_ids": sorted(ids),
            "expected_record_content_sha256_by_id":{r["record_id"]:r["record_content_sha256"] for r in subset},
            "input_language":"ko","format":"RD","split":"dev","mode":mode,
            "wrappers":["dev1","dev2"], "active_languages":list(stage_active_languages(stage, language_order)),
            "language_order":list(language_order),
            "gate_each_wrapper_and_requested_language":True,"minimum_unrounded":0.9,
            "identifiable_concept_ids":[r["concept_id"] for r in concepts if len(set(r["answers"].values())) == 4],
            "rho_min":0.6,"wrapper_abs_max":0.15,"z_threshold":0.05,"low_z_fraction_max":0.1})
        results[name] = compute_readiness(rows, plan) if name == "readiness" else compute_measurement(rows, plan)
    return results


def phase_order(execution=None):
    execution = execution or {}
    roots = execution.get("roots", (4101,4102,4103,4104))
    start = PHASES.index(execution.get("start_phase", "INIT"))
    return [(r,p) for r in roots for p in PHASES[start:6]] + ([
        (r,p) for r in roots for p in PHASES[6:]] if execution.get("history_enabled", True) else [])


def phase_learning_rate(execution, phase, completed_steps):
    reduction = execution.get("learning_rate_reduction", {})
    if phase in reduction.get("stages", []) and completed_steps >= reduction["after_updates"]:
        return reduction["learning_rate"]
    return 0.0001 if phase.startswith("H_") else 0.0003


def history_contrast(a, b, assignment):
    """Matched concept/prompt contrasts, not a main-study prediction claim."""
    def index(scores):
        out = {}
        for row in scores["rows"]:
            if row["mode"] != "ANY": continue
            key=(row["concept_id"],row["input_language"],row["wrapper"],row["split"])
            if key in out: raise ContractViolation("DUPLICATE_HISTORY_OBSERVATION")
            out[key]=row
        return out
    left,right=index(a),index(b)
    if set(left)!=set(right) or not left: raise ContractViolation("HISTORY_EVALUATION_SET_MISMATCH")
    pa,pb=a["provenance"],b["provenance"]
    if pa.get("branch")!="A" or pb.get("branch")!="B" or any(pa.get(k)!=pb.get(k) for k in (
        "root_id","phase","stage","freeze_sha256","config_sha256","tokenizer_sha256","code_sha256","evaluation_plan_sha256")):
        raise ContractViolation("HISTORY_EVALUATION_LINEAGE_MISMATCH")
    rows=[]
    for key in sorted(left):
        qa,za=_probability_values(left[key]); qb,zb=_probability_values(right[key])
        sign=assignment[key[0]]
        h={l:sign*(qa[l]-qb[l]) for l in LANGUAGES}
        rows.append({"concept_id":key[0],"input_language":key[1],"wrapper":key[2],"split":key[3],
            "g":sign,"h":h,"r":(h["fr"]-h["en"])/2,"Z_A":za,"Z_B":zb})
    return {"schema_version":"pilot-history-contrast-v1","root_id":pa["root_id"],"rows":rows,
        "main_prediction_status":"NOT_RUN_NOT_AUTHORIZED","etymology_status":"NOT_RUN",
        "interpretation":"paired additional-exposure schedule contrast; not isolated concept causality"}


class PilotRuntime:
    def __init__(self, *, freeze_path, implementation_manifest, cpu_checks, replay,
                 run_directory, resume=False, pause_after=None):
        self.directory = Path(run_directory).absolute()
        if self.directory != self.directory.resolve() or not self.directory.is_relative_to(WORK_ROOT):
            raise ContractViolation("PILOT_DIRECTORY_OUTSIDE_WORK")
        self.freeze_path = Path(freeze_path)
        self.manifest_path = Path(implementation_manifest)
        self.cpu_path, self.replay_path = Path(cpu_checks), Path(replay)
        self.freeze = verify_execution_freeze(self.freeze_path)
        if self.freeze.get("execution", {}).get("runner") == "accuracy_probe":
            raise ContractViolation("USE_DEDICATED_ACCURACY_PROBE_RUNNER")
        implementation = verify_implementation_manifest(self.manifest_path)
        cpu = verify_cpu_check_evidence(self.cpu_path)
        self.code = implementation["implementation_manifest_sha256"]
        self.cpu_hash = cpu["cpu_check_sha256"]
        if cpu["implementation"]["implementation_manifest_sha256"] != self.code:
            raise ContractViolation("PILOT_CPU_CODE_MISMATCH")
        _verify_gpu_replay_evidence(self.replay_path, freeze_sha256=self.freeze["freeze_sha256"],
            code_sha256=self.code, cpu_check_sha256=self.cpu_hash)
        self.binding = {"freeze_sha256":self.freeze["freeze_sha256"],"code_sha256":self.code,
            "cpu_sha256":self.cpu_hash,"replay_sha256":sha256_file(self.replay_path)}
        self.resume = resume
        self.pause_after = pause_after
        policy = read_verified_json(PROJECT_ROOT / "implementation/config/campaign_policy.json")["budget"]
        self.ledger = GpuBudgetLedger(WORK_ROOT / "ledger/gpu_budget.json", BudgetCaps(
            policy["campaign_gpu_hours_cap"],policy["prior_gpu_hours_user_reported"],policy["per_root_gpu_hours_cap"]))
        self.emergency = policy["minimum_emergency_free_bytes"]
        self.plan = read_verified_json(Path(self.freeze["resolved_artifacts"]["evaluation_plan"]))
        self.tokenizer, _ = load_verified_tokenizer(Path(self.freeze["resolved_artifacts"]["tokenizer"]))
        self.tokens = None
        self.journal = None

    def _import_full_state(self):
        execution = self.freeze.get("execution", {})
        source = execution.get("import_full_state")
        root, phase = execution.get("import_root_id",4101), execution.get("import_phase","T1")
        if not source or self.journal.latest("PHASE_COMPLETE",root,phase):
            return
        # The exact source is pinned in the verified exploratory freeze. This is
        # an explicit cross-revision continuation, not an independent new root.
        payload, _ = load_checkpoint_payload(Path(source["path"]),
            expected_state_sha256=source["state_sha256"],
            expected_manifest_sha256=source["manifest_sha256"])
        lineage = payload["lineage"]
        expected = {"root_id":root,"phase":phase,"stage":phase,"branch":None,"data_kind":"REAL",
            "freeze_sha256":source["freeze_sha256"],"code_sha256":source["code_sha256"],
            "config_sha256":self.freeze["artifacts"]["pilot_config"]["sha256"],
            "tokenizer_sha256":self.freeze["tokenizer_file_sha256"],
            "evaluation_plan_sha256":self.freeze["artifacts"]["evaluation_plan"]["sha256"]}
        if any(lineage.get(k) != v for k,v in expected.items()) or payload["progress"]["phase_step"] != 3000:
            raise ContractViolation("EXPLORATORY_SOURCE_LINEAGE_MISMATCH")
        previous_lineage = dict(lineage)
        lineage.update({"freeze_sha256":self.freeze["freeze_sha256"],"code_sha256":self.code,
            "exploratory_source":source,"source_lineage":previous_lineage})
        require_disk_reservation(self.directory,planned_bytes=600*1024**2,emergency_free_bytes=self.emergency)
        ref = save_checkpoint(self.directory/"checkpoints"/f"{root}_{phase}_import_{uuid.uuid4().hex[:12]}",payload).as_dict()
        checked, _ = self._load_checkpoint(ref,root,phase)
        if checked["semantic_fingerprints"] != payload["semantic_fingerprints"]:
            raise ContractViolation("EXPLORATORY_IMPORT_STATE_CHANGED")
        self.journal.append("FULL_STATE_IMPORT",{"root_id":root,"phase":phase,"source":source,
            "checkpoint":ref,"independent_initialization":False,
            "preserved":["model","optimizer","scheduler","rng","progress"]})
        self.journal.append("PHASE_COMPLETE",{"root_id":root,"phase":phase,"phase_step":3000,
            "checkpoint":ref,"origin":"IMPORTED_NOT_RETRAINED"})

    def _load_checkpoint(self, ref, root, phase):
        path = Path(ref["path"])
        if path != path.resolve() or not path.is_relative_to(self.directory / "checkpoints"):
            raise ContractViolation("PILOT_CHECKPOINT_OUTSIDE_RUN")
        payload, manifest = load_checkpoint_payload(path, expected_state_sha256=ref["state_sha256"],
            expected_manifest_sha256=ref["manifest_sha256"])
        lineage = payload["lineage"]
        expected = {"root_id":root,"phase":phase,"freeze_sha256":self.freeze["freeze_sha256"],
            "code_sha256":self.code,"config_sha256":self.freeze["artifacts"]["pilot_config"]["sha256"],
            "tokenizer_sha256":self.freeze["tokenizer_file_sha256"],
            "evaluation_plan_sha256":self.freeze["artifacts"]["evaluation_plan"]["sha256"],"data_kind":"REAL"}
        if any(lineage.get(k) != v for k,v in expected.items()):
            raise ContractViolation("PILOT_CHECKPOINT_LINEAGE_MISMATCH")
        return payload, manifest

    def _root_gates(self, root):
        completed = self.journal.latest("PHASE_COMPLETE",root,"T3")
        if not completed:
            raise ContractViolation("MISSING_T3_ENDPOINT")
        _, manifest = self._load_checkpoint(completed["checkpoint"], root, "T3")
        evidence = self.journal.latest("EVALUATION",root,"T3")
        if not evidence or evidence["checkpoint"]["state_sha256"] != completed["checkpoint"]["state_sha256"]:
            raise ContractViolation("T3_EVALUATION_NOT_ENDPOINT")
        scores, records = bound_json(evidence["scores"]), bound_json(evidence["records"])["records"]
        if scores["provenance"]["checkpoint_sha256"] != manifest["state_sha256"]:
            raise ContractViolation("T3_GATE_LINEAGE_MISMATCH")
        return gate_results(scores, records, self.freeze["concepts"], "T3", language_order=self.freeze.get("execution",{}).get("language_order",LANGUAGES))

    def _training_gates(self, root, phase):
        if self._trajectory_replication():
            if root not in self.freeze["execution"]["roots"] or phase not in PHASES[:6]:
                raise ContractViolation("TRAJECTORY_REPLICATION_SCOPE_VIOLATION")
            # __init__ verifies real-data freeze, exact CPU evidence and GPU replay.
            # The user's new replication request makes outcome gates diagnostic;
            # it does not convert failed gates into PASS or authorize H.
            return
        gates = {k:"PASS" for k in ("DATA_QA","EXPERIMENT_FREEZE","CPU_INTEGRATION","GPU_REPLAY")}
        required = (4101,4102,4103,4104) if phase.startswith("H_") else ((4101,) if root != 4101 else ())
        for r in required:
            for name,value in self._root_gates(r).items():
                gates[f"ROOT_{r}_T3_{name.upper()}"] = value["status"]
        require_campaign_training_gates(root_id=root,phase=phase,gates=gates,main_enabled=False)

    def _trajectory_replication(self):
        return getattr(self,"freeze",{}).get("exploratory_revision") in (TRAJECTORY_REVISION, REVERSE_REVISION)

    def _restore(self, payload, model, optimizer, scheduler, generator, signature):
        lineage = payload["lineage"]
        expected = ResumeContract(**{k:lineage[k] for k in (
            "root_id","phase","stage","branch","freeze_sha256","config_sha256","tokenizer_sha256",
            "code_sha256","evaluation_plan_sha256","data_kind","initial_model_fingerprint",
            "phase_parent_sha256","resume_parent_sha256")},
            plan_sha256=payload["progress"]["plan_sha256"],require_cuda_rng=True)
        return restore_training_state(payload,model,optimizer,scheduler=scheduler,scaler=None,
            generators={"loader":generator},expected=expected,parameter_group_signature=signature,
            expected_deterministic_settings=configure_determinism())

    def _checkpoint(self, objects, lineage, progress, root, phase, reason):
        model,optimizer,scheduler,generator,signature = objects
        require_disk_reservation(self.directory,planned_bytes=600*1024**2,emergency_free_bytes=self.emergency)
        payload = capture_training_state(model,optimizer,scheduler=scheduler,scaler=None,
            generators={"loader":generator},progress=progress,lineage=lineage,
            parameter_group_signature=signature,deterministic_settings=configure_determinism(),require_cuda_rng=True)
        dest = self.directory / "checkpoints" / f"{root}_{phase}_{progress['phase_step']:06d}_{uuid.uuid4().hex[:12]}"
        ref = save_checkpoint(dest,payload).as_dict()
        del payload
        # Read back the complete model/optimizer/RNG state before announcing it.
        checked,_ = self._load_checkpoint(ref,root,phase)
        del checked
        self.journal.append("CHECKPOINT", {"root_id":root,"phase":phase,"phase_step":progress["phase_step"],
            "checkpoint":ref,"reason":reason})
        lineage["resume_parent_sha256"] = ref["state_sha256"] if phase != "INIT" else None
        return ref

    def _prune_intermediates(self, root, phase):
        """Only this run's superseded periodic states; never endpoint evidence.

        Called after the successor checkpoint and its evaluation are committed.
        Keep manifests/commit markers/trace/score hashes and record every removal.
        """
        candidates = [e["data"] for e in self.journal.events if e["kind"] == "CHECKPOINT"
            and e["data"]["reason"] == "PERIODIC"]
        pruned = {e["data"]["state_sha256"] for e in self.journal.events if e["kind"] == "STATE_PRUNED"}
        for item in candidates[:-2]:
            ref = item["checkpoint"]
            if ref["state_sha256"] in pruned: continue
            path = Path(ref["path"])/"state.pt"
            if path != path.resolve() or not path.is_relative_to(self.directory/"checkpoints"):
                raise ContractViolation("CHECKPOINT_RETENTION_SCOPE_MISMATCH")
            if path.exists():
                if sha256_file(path) != ref["state_sha256"]: raise ContractViolation("CHECKPOINT_RETENTION_HASH_MISMATCH")
                path.unlink()
            self.journal.append("STATE_PRUNED",{"root_id":item["root_id"],"phase":item["phase"],"state_sha256":ref["state_sha256"],
                "path":str(path),"reason":"SUPERSEDED_PERIODIC_STATE","manifest_and_scores_retained":True})

    def _evaluate(self, objects, ref, root, phase, step, endpoint, session, stop):
        stage = "H" if phase.startswith("H_") else phase
        language_order = self.freeze.get("execution",{}).get("language_order",LANGUAGES)
        records = evaluation_records(self.freeze["concepts"],self.plan,stage,endpoint=endpoint,language_order=language_order)
        evaluation_id = f"{root}_{phase}_{step:06d}_{uuid.uuid4().hex[:12]}"
        record_ref = publish_json_once(self.directory / "evaluations" / (evaluation_id+"_records.json"),
            {"schema_version":"pilot-evaluation-records-v1","records":records})
        provenance = {"root_id":root,"phase":"H" if stage == "H" else "S", "stage":stage,
            "branch":phase[-1] if stage == "H" else None, "checkpoint_sha256":ref["state_sha256"],
            "checkpoint_manifest_sha256":ref["manifest_sha256"],"model_fingerprint":ref["model_fingerprint"],
            "freeze_sha256":self.freeze["freeze_sha256"],"config_sha256":self.freeze["artifacts"]["pilot_config"]["sha256"],
            "tokenizer_sha256":self.freeze["tokenizer_file_sha256"],"code_sha256":self.code,
            "evaluation_plan_sha256":self.freeze["artifacts"]["evaluation_plan"]["sha256"],
            "evaluation_record_hash_schema":"score-record-v1","data_kind":"REAL","freeze_gate_status":"PASS",
            "context_length":256,"maximum_new_tokens":64,"generation_strategy":"MANUAL_GREEDY",
            "terminator":"NEWLINE","candidate_universe":list(LANGUAGES)}
        def heartbeat():
            session.ledger.require_time_remaining(session.lease,checkpoint_grace_seconds=120)
            if stop.requested: raise ContractViolation("INTERRUPTED_AT_BOUNDARY")
        scores = score_checkpoint(objects[0],self.tokenizer,records,provenance,
            context_length=256,max_new_tokens=64,experiment_freeze_path=self.freeze_path,
            checkpoint_directory=ref["path"],generators={"loader":objects[3]},progress_callback=heartbeat)
        score_ref = publish_json_once(self.directory / "evaluations" / (evaluation_id+"_scores.json"), scores)
        gates = gate_results(scores,records,self.freeze["concepts"],stage,language_order=language_order)
        event = {"root_id":root,"phase":phase,"phase_step":step,"endpoint":endpoint,
            "checkpoint":ref,"records":record_ref,"scores":score_ref,"gates":gates}
        self.journal.append("EVALUATION",event)
        print(f"EVALUATION root={root} phase={phase} step={step} readiness={gates['readiness']['status']}",flush=True)
        return event

    def _phase(self, root, phase):
        self._training_gates(root,phase)
        verify_implementation_manifest(self.manifest_path)
        generator = torch.Generator(device="cpu").manual_seed(root+1)
        if phase in ("INIT","H_BASE"):
            # Zero-update state still has a reconstructible loader contract.
            stream,audit = corpus_stream(self.tokens[:256],self.token_sha,generator)
            endpoint = 0
        elif phase == "CORPUS":
            stream,audit = corpus_stream(self.tokens,self.token_sha,generator)
            endpoint = len(stream)
        else:
            stream,audit = lexical_stream(self.freeze,root,phase,self.tokenizer,generator)
            endpoint = len(stream)
        plan_ref = publish_json_once(self.directory / "plans" / f"{root}_{phase}_{uuid.uuid4().hex[:12]}.json",audit)
        self.journal.append("PLAN",{"root_id":root,"phase":phase,"artifact":plan_ref})
        snapshot = self.ledger.snapshot()
        remaining = min(snapshot["campaign_gpu_hours_remaining"]*3600,
            6*3600-snapshot["per_root_gpu_seconds_charged_or_reserved"].get(str(root),0))
        if remaining <= 240: raise BudgetStop("BLOCKED_PILOT_BUDGET")
        latest = self.journal.latest("CHECKPOINT",root,phase)
        parent = None if phase == "INIT" else self.journal.latest("PHASE_COMPLETE",root,PHASE_PARENT[phase])
        if phase != "INIT" and not parent: raise ContractViolation("MISSING_PHASE_ENDPOINT_PARENT")
        with gpu2_training_session(ledger=self.ledger,lock_path=WORK_ROOT/"locks/gpu2.lock",
                run_id=f"{self.directory.name}-{root}-{phase}-{uuid.uuid4().hex[:12]}",root_id=root,phase=phase,
                reserved_seconds=min(1800,remaining-1),checkpoint_grace_seconds=120,experiment_freeze_path=self.freeze_path) as session, BoundaryStopController() as stop:
            seed_all(root,include_cuda=True)
            model = build_random_model(seed=root).to("cuda:0")
            optimizer,signature = build_adamw(model,self.plan["training"])
            scheduler = ConstantSchedule(optimizer,self.plan["training"]["learning_rate"],"INIT")
            initial = semantic_model_fingerprint(model)
            objects = (model,optimizer,scheduler,generator,signature)
            progress = {"global_step":0,"phase_step":0,"accumulation_step":0,"examples_seen":0,
                "model_tokens_seen":0,"plan_sha256":stream.plan_sha256,"loader_state":stream.state_dict()}
            if latest or parent:
                ref = (latest or parent)["checkpoint"]
                expected_phase = phase if latest else PHASE_PARENT[phase]
                payload,manifest = self._load_checkpoint(ref,root,expected_phase)
                if phase != "INIT": validate_phase_transition(child_phase=phase,child_root_id=root,parent_manifest=manifest,resume=bool(latest))
                if payload["lineage"]["initial_model_fingerprint"] != initial:
                    raise ContractViolation("INITIALIZATION_FINGERPRINT_MISMATCH")
                restored = self._restore(payload,*objects)
                if latest:
                    progress = restored
                    stream.load_state_dict(progress["loader_state"])
                else:
                    progress.update({k:restored[k] for k in ("global_step","examples_seen","model_tokens_seen")})
                del payload,manifest
            if phase == "INIT":
                fingerprints = {e["data"]["root_id"]:e["data"]["initial_model_fingerprint"] for e in self.journal.events if e["kind"] == "INITIALIZATION"}
                if self._trajectory_replication():
                    source=self.freeze["execution"]["reference_run"]
                    reference=read_verified_json(Path(source["path"]),expected_sha256=source["sha256"])
                    fingerprints.update({e["data"]["root_id"]:e["data"]["initial_model_fingerprint"]
                        for e in reference["events"] if e["kind"]=="INITIALIZATION"})
                fingerprints[root] = initial
                audit_initialization_fingerprints(fingerprints,require_all_roots=False)
                if not self.journal.latest("INITIALIZATION",root):
                    self.journal.append("INITIALIZATION",{"root_id":root,"initial_model_fingerprint":initial})
            lr = phase_learning_rate(self.freeze["execution"], phase, progress["phase_step"])
            scheduler.transition(phase=phase,learning_rate=lr)
            lineage = {"root_id":root,"phase":phase,"stage":"H" if phase.startswith("H_") else phase,
                "branch":phase[-1] if phase in ("H_A","H_B") else None,
                "phase_parent_sha256":parent["checkpoint"]["state_sha256"] if parent else None,
                "resume_parent_sha256":latest["checkpoint"]["state_sha256"] if latest and phase != "INIT" else None,
                "initial_model_fingerprint":initial,"freeze_sha256":self.freeze["freeze_sha256"],
                "config_sha256":self.freeze["artifacts"]["pilot_config"]["sha256"],"tokenizer_sha256":self.freeze["tokenizer_file_sha256"],
                "code_sha256":self.code,"evaluation_plan_sha256":self.freeze["artifacts"]["evaluation_plan"]["sha256"],
                "data_kind":"REAL","freeze_gate_status":"PASS","gpu_binding":session.binding.as_dict(),
                "budget_lease_id":session.lease.lease_id,"runtime":{"python":platform.python_version(),"torch":torch.__version__,
                    "transformers":__import__("transformers").__version__,"torch_cuda":torch.version.cuda,
                    "precision":"BF16_AUTOCAST_WITH_FP32_PARAMETERS_AND_ADAMW"}}
            ref = latest["checkpoint"] if latest else None
            def save(reason): return self._checkpoint(objects,lineage,progress,root,phase,reason)
            def evaluate_if_due():
                if phase in ("INIT","CORPUS","H_BASE") or not progress["phase_step"]: return
                previous = self.journal.latest("EVALUATION",root,phase)
                if not previous or previous["checkpoint"]["state_sha256"] != ref["state_sha256"]:
                    evaluate_without_rng_consumption(
                        lambda:self._evaluate(objects,ref,root,phase,progress["phase_step"],progress["phase_step"]==endpoint,session,stop),
                        generators={"loader":generator},include_cuda=True)
            if ref and (progress["phase_step"] % 200 == 0 or progress["phase_step"] == endpoint): evaluate_if_due()
            while progress["phase_step"] < endpoint:
                scheduler.transition(phase=phase,learning_rate=phase_learning_rate(
                    self.freeze["execution"],phase,progress["phase_step"]))
                steps = min(200-progress["phase_step"]%200,endpoint-progress["phase_step"])
                # Reserve enough space BEFORE updates so a boundary save remains possible.
                require_disk_reservation(self.directory,planned_bytes=1200*1024**2,emergency_free_bytes=self.emergency)
                traces,status = run_gpu2_stage_steps(session,model,optimizer,scheduler,stream,progress,
                    checkpoint_grace_seconds=120,steps=steps,loss_kind="corpus" if phase=="CORPUS" else "lexical",
                    grad_clip_norm=1.0,stop_controller=stop,boundary_checkpoint=save)
                if status == "FIXED_ENDPOINT_REACHED":
                    ref = save("FIXED_ENDPOINT" if progress["phase_step"]==endpoint else "PERIODIC")
                trace_ref = publish_json_once(self.directory/"traces"/f"{root}_{phase}_{progress['phase_step']}_{uuid.uuid4().hex[:12]}.json",
                    {"schema_version":"pilot-training-trace-v1","traces":[asdict(r) for r in traces]})
                self.journal.append("TRACE",{"root_id":root,"phase":phase,"artifact":trace_ref})
                if status != "FIXED_ENDPOINT_REACHED": raise ContractViolation(status)
                evaluate_if_due()
                self._prune_intermediates(root,phase)
                print(f"TRAIN root={root} phase={phase} step={progress['phase_step']}/{endpoint}",flush=True)
                if self.pause_after == (root,phase,progress["phase_step"]):
                    raise ContractViolation("PAUSED_AFTER_REQUESTED_BOUNDARY")
            if ref is None: ref = save("ZERO_UPDATE_ENDPOINT")
            self.journal.append("PHASE_COMPLETE",{"root_id":root,"phase":phase,"checkpoint":ref,
                "phase_step":endpoint,"plan":plan_ref})
        del objects,model,optimizer,scheduler,stream
        gc.collect()
        torch.cuda.empty_cache()
        if phase == "T3" and not self._trajectory_replication():
            gates = self._root_gates(root)
            for name in ("readiness","measurement"):
                if gates[name]["status"] != "PASS": raise ContractViolation(gates[name]["status"])

    def _history_result(self, root):
        scores=[]
        refs=[]
        for branch in ("A","B"):
            event=self.journal.latest("EVALUATION",root,"H_"+branch)
            endpoint=self.journal.latest("PHASE_COMPLETE",root,"H_"+branch)
            if not event or not endpoint or event["checkpoint"] != endpoint["checkpoint"] or not event["endpoint"]:
                raise ContractViolation("HISTORY_RESULT_REQUIRES_FIXED_ENDPOINTS")
            self._load_checkpoint(endpoint["checkpoint"],root,"H_"+branch)
            scores.append(bound_json(event["scores"])); refs.append(event["scores"])
        assignment=build_history_plan_from_verified_freeze(self.freeze,root_id=root).assignment
        result=history_contrast(*scores,assignment)
        result["score_artifacts"]=refs
        ref=publish_json_once(self.directory/"history"/f"{root}_{uuid.uuid4().hex[:12]}.json",result)
        self.journal.append("HISTORY_RESULT",{"root_id":root,"artifact":ref})

    def run(self):
        self.directory.mkdir(parents=True,exist_ok=True)
        # Nonblocking lock includes CPU preparation and closes duplicate-launch race.
        with ExclusiveGpu2Lock(self.directory/"run.lock"):
            self.journal = PilotJournal(self.directory/"journal",self.binding)
            if self.journal.events and not self.resume: raise ContractViolation("PILOT_RESUME_FLAG_REQUIRED")
            self.journal.append("START",{"resume":self.resume,"pid":__import__("os").getpid()})
            status = "INCOMPLETE"
            try:
                self.tokens,corpus = load_corpus_token_memmap(Path(self.freeze["resolved_artifacts"]["corpus"]),production=True)
                self.token_sha = corpus["token_stream_sha256"]
                self._import_full_state()
                execution = self.freeze.get("execution", {})
                for root,phase in phase_order(execution):
                    if self.journal.latest("PHASE_COMPLETE",root,phase): continue
                    if self._trajectory_replication() and phase == "INIT":
                        require_disk_reservation(self.directory,planned_bytes=4*1024**3,emergency_free_bytes=self.emergency)
                        if root != execution["benchmark_first_root"]:
                            measured = self.ledger.snapshot()["per_root_gpu_seconds_charged_or_reserved"].get("4102",0)/3600
                            needed = max(execution["minimum_remaining_hours_before_later_root"],1.5*measured)
                            if self.ledger.snapshot()["campaign_gpu_hours_remaining"] < needed:
                                raise ContractViolation("BLOCKED_REPLICATION_PROJECTED_BUDGET")
                        self.journal.append("REPLICATION_ROOT_START",{"root_id":root,"independent_initialization":True,
                            "budget_before":self.ledger.snapshot()})
                    while True:
                        try:
                            self._phase(root,phase)
                            break
                        except BudgetStop as exc:
                            if str(exc) != "BLOCKED_GPU_BUDGET_DEADLINE": raise
                            self.journal.append("LEASE_BOUNDARY",{"root_id":root,"phase":phase,"status":str(exc)})
                            gc.collect()
                            torch.cuda.empty_cache()
                # Completion cannot skip a readiness failure on the final S stage.
                if execution.get("history_enabled", True):
                    for root in execution.get("roots", (4101,4102,4103,4104)):
                        self._training_gates(root,"H_A")
                        if not self.journal.latest("HISTORY_RESULT",root): self._history_result(root)
                    status = "PILOT_COMPLETE"
                elif self._trajectory_replication():
                    for root in execution["roots"]:
                        self.journal.append("REPLICATION_OUTCOME_GATES",{"root_id":root,"gates":self._root_gates(root),
                            "gate_failure_is_not_success":True})
                    status = "REVERSE_ORDER_COMPLETE" if self.freeze.get("exploratory_revision")==REVERSE_REVISION else "TRAJECTORY_REPLICATION_COMPLETE"
                else:
                    # Resume may skip a completed phase: recheck its gates here.
                    for root in execution["roots"]:
                        for gate in self._root_gates(root).values():
                            if gate["status"] != "PASS": raise ContractViolation(gate["status"])
                    status = "EXPLORATORY_PREPARATION_COMPLETE"
            except Exception as exc:
                status = str(exc)
                self.journal.append("STOP",{"status":status,"error_type":type(exc).__name__})
            finally:
                summary = {"schema_version":"pilot-runtime-run-v1","status":status,"binding":self.binding,
                    "budget":self.ledger.snapshot(),"main_enabled":False,"events":self.journal.events,
                    "journal_tip_sha256":self.journal.events[-1]["event_sha256"],
                    "exploratory_revision":self.freeze.get("exploratory_revision"),
                    "scientific_training_started":any(e["kind"]=="TRACE" or
                        (e["kind"]=="CHECKPOINT" and e["data"].get("phase_step",0)>0) for e in self.journal.events),
                    "uncommitted_updates_possible":status not in ("PILOT_COMPLETE","EXPLORATORY_PREPARATION_COMPLETE","TRAJECTORY_REPLICATION_COMPLETE","REVERSE_ORDER_COMPLETE","PAUSED_AFTER_REQUESTED_BOUNDARY")}
                ref = publish_json_once(self.directory/f"run_summary_{len(self.journal.events):06d}.json",summary)
                print({"status":status,"summary":ref},flush=True)
            return summary
