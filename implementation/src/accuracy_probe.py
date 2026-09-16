"""Bounded paired accuracy experiment; same source state, records, and budget.

No candidate is promoted to production automatically. The train-prompt mixture
is independent of error IDs and never includes dev/test prefixes.
"""
from dataclasses import replace, asdict
from pathlib import Path
import gc
import hashlib

import torch

from .artifacts import read_verified_json, publish_json_once, sha256_file
from .contracts import ContractViolation, WORK_ROOT, PROJECT_ROOT, LANGUAGES
from .pilot_execution_freeze import verify_execution_freeze, PROBE_REVISION, PROBE_NEXT_REVISION, PROBE_CONTINUE_REVISION
from .pilot_runtime import PilotRuntime, gate_results
from .pilot_journal import PilotJournal
from .pilot_streams import LazyBatchStream, LazyLexicalBatches, evaluation_records
from .schedule import build_stage_plan
from .records import materialize_record_bank, encode_record
from .semantic_experiment_freeze import _hash
from .model import build_random_model, build_adamw, ConstantSchedule, configure_determinism
from .checkpoint import load_checkpoint_payload, capture_training_state, save_checkpoint
from .build_tokenizer import load_verified_tokenizer
from .integrity import verify_implementation_manifest, verify_cpu_check_evidence
from .report import _verify_gpu_replay_evidence
from .score import score_checkpoint, preserved_evaluation_state, semantic_model_fingerprint
from .budget import BudgetCaps, GpuBudgetLedger, ExclusiveGpu2Lock, require_disk_reservation
from .train import gpu2_training_session, run_gpu2_stage_steps, BoundaryStopController, evaluate_without_rng_consumption


def mix_prompt(record, concept, prompt, *, all_ko_prompts=False):
    # Half of KO training records, selected without outcomes. Foreign input
    # records and all answers remain exactly the same as in both control arms.
    digest=hashlib.sha256(record.record_id.encode()).digest()
    if record.input_language!="ko" or (not all_ko_prompts and digest[0]%2):
        return record
    header=("뜻풀이: " if record.wrapper=="train1" else "정의: ")+concept["glosses"]["ko"]
    language=prompt["target_language_names"]["ko"][record.target_language]
    instruction=(f"{language}로 답하세요" if record.mode=="REQUESTED" else "한 단어로 답하세요")
    prefix=(header+"\n"+instruction+":" if digest[1]%2 else instruction+".\n"+header+"\n답:")
    for wrapper in (*prompt["development_wrappers"],*prompt["test_wrappers"]):
        held=prompt["templates"][wrapper]["ko"][record.mode].format(
            gloss=concept["glosses"]["ko"],target_language_name=language)
        if prefix==held: raise ContractViolation("PROMPT_MIX_HELDOUT_COLLISION")
    incidental=tuple((l,prefix.count(y)) for l,y in sorted(concept["answers"].items()) if prefix.count(y))
    changed=replace(record,prefix=prefix,kind="lexical_prompt_mix",incidental_exposure_counts=incidental)
    core=changed.as_dict(); core.pop("content_sha256")
    return replace(changed,content_sha256=_hash(core))


def probe_stream(freeze, tokenizer, generator, *, prompt_mix, concept_factor_balance=False, all_ko_prompts=False):
    plan=build_stage_plan([c["concept_id"] for c in freeze["concepts"]],root_id=4101,stage="T3",
        optimizer_updates=freeze["execution"]["additional_updates"],concept_factor_balance=concept_factor_balance)
    policy=Path(freeze["resolved_artifacts"]["evaluation_plan"])
    prompt=read_verified_json(policy)["prompt_plan"]
    bank=materialize_record_bank(freeze["concepts"],plan.slots,stage="T3",policy_path=policy)
    concepts={c["concept_id"]:c for c in freeze["concepts"]}
    if prompt_mix:
        bank={rid:mix_prompt(r,concepts[r.concept_id],prompt,all_ko_prompts=all_ko_prompts) for rid,r in bank.items()}
    cache={}; encoded={}
    for rid,r in bank.items():
        key=(r.prefix,r.canonical_answer)
        if key not in cache: cache[key]=encode_record(r,tokenizer)
        encoded[rid]=replace(cache[key],record_id=rid,content_sha256=r.content_sha256)
    record_hash=_hash([(rid,r.content_sha256) for rid,r in sorted(bank.items())])
    plan_hash=_hash({"schedule":plan.plan_sha256,"records":record_hash})
    audit={"schedule_sha256":plan.plan_sha256,"record_bank_sha256":record_hash,"plan_sha256":plan_hash,
        "counts":plan.counts,"augmented_records":sum(r.kind=="lexical_prompt_mix" for r in bank.values()),
        "max_sequence_tokens":max(len(e.input_ids) for e in encoded.values()),
        "target_slots_sha256":_hash([(r.record_id,r.concept_id,r.target_language,r.canonical_answer) for r in bank.values()])}
    return LazyBatchStream(LazyLexicalBatches(plan.batches,encoded),plan_sha256=plan_hash,generator=generator),audit


def compare_arms(baseline, arms, *, selection_version=1):
    if selection_version not in (1,2): raise ContractViolation("UNKNOWN_PROBE_SELECTION_VERSION")
    def counts(g): return {k:v["compatible_numerator"] for k,v in g["readiness"]["cells"].items()}
    base=counts(baseline); control=counts(arms["constant"])
    result={}
    for name,gates in arms.items():
        cells=counts(gates)
        gains={l:sum(cells[f"{w}:{l}"]-base[f"{w}:{l}"] for w in ("dev1","dev2")) for l in LANGUAGES}
        control_gains={l:sum(cells[f"{w}:{l}"]-control[f"{w}:{l}"] for w in ("dev1","dev2")) for l in LANGUAGES}
        candidate=(name!="constant" and all(gains[l]>=2 and control_gains[l]>=2 for l in ("zh","fr"))
            and all(cells[f"{w}:{l}"]>=max(base[f"{w}:{l}"],control[f"{w}:{l}"])-1
                    for w in ("dev1","dev2") for l in ("ko","en")))
        if selection_version==2:
            candidate=(name!="constant" and gates["readiness"]["status"]=="PASS"
                and gains["zh"]>=2 and control_gains["zh"]>=1 and gains["fr"]>=0 and control_gains["fr"]>=-1
                and all(cells[f"{w}:{l}"]>=max(base[f"{w}:{l}"],control[f"{w}:{l}"])-1
                        for w in ("dev1","dev2") for l in ("ko","en")))
        result[name]={"gains_over_baseline_out_of_120":gains,"gains_over_constant_out_of_120":control_gains,
            "promising_exploratory_candidate":candidate,
            "all_original_gates_pass":all(g["status"]=="PASS" for g in gates.values())}
    return result


def continuation_result(baseline, arms):
    if set(arms)!={"lower_lr"}:
        raise ContractViolation("CONTINUATION_REQUIRES_ONLY_SELECTED_ARM")
    gates=arms["lower_lr"]
    return {"arm":"lower_lr", "concurrent_control":False,
        "compatible_count_change_by_cell":{k:v["compatible_numerator"]-baseline["readiness"]["cells"][k]["compatible_numerator"]
            for k,v in gates["readiness"]["cells"].items()},
        "all_original_gates_pass":all(g["status"]=="PASS" for g in gates.values()),
        "interpretation":"descriptive dev-informed continuation; not causal improvement or held-out confirmation"}


def continuation_lineage(source, **updates):
    # A saved fingerprint describes the *parent* weights. capture_training_state
    # computes the new fingerprint; retaining the parent at the top level would
    # correctly fail after the first optimizer update. Keep it in provenance.
    lineage={k:v for k,v in source.items() if k!="model_fingerprint"}
    return {**lineage,"source_lineage":dict(source),**updates}


def probe_source_event(source, execution):
    arm=execution.get("source_arm")
    if arm is None:
        return [e["data"] for e in source["events"] if e["kind"]=="EVALUATION" and e["data"]["phase"]=="T3" and e["data"]["endpoint"]][-1]
    evaluation=[e["data"] for e in source["events"] if e["kind"]=="EVALUATION" and e["data"].get("arm")==arm][-1]
    endpoint=[e["data"] for e in source["events"] if e["kind"]=="CHECKPOINT" and e["data"].get("arm")==arm and e["data"]["reason"]=="FIXED_ENDPOINT"][-1]
    return {**evaluation,"checkpoint":endpoint["checkpoint"]}


def prune_probe_midpoint(directory, ref, endpoint, journal, arm):
    # Only this new run's verified superseded midpoint. All final models and
    # all records, scores, trace and manifest evidence remain intact.
    path=Path(ref["path"])/"state.pt"
    if path!=path.resolve() or not path.is_relative_to(directory/"checkpoints") or ref==endpoint:
        raise ContractViolation("PROBE_RETENTION_SCOPE_MISMATCH")
    checked,_=load_checkpoint_payload(Path(endpoint["path"]),expected_state_sha256=endpoint["state_sha256"],expected_manifest_sha256=endpoint["manifest_sha256"])
    del checked
    if sha256_file(path)!=ref["state_sha256"]: raise ContractViolation("PROBE_RETENTION_HASH_MISMATCH")
    path.unlink()
    journal.append("STATE_PRUNED",{"arm":arm,"path":str(path),"sha256":ref["state_sha256"],
        "reason":"VERIFIED_MIDPOINT_SUPERSEDED_BY_SCORED_ENDPOINT","endpoint":endpoint,"metadata_and_scores_retained":True})


def run_probe(*, freeze_path, manifest_path, cpu_path, replay_path, directory):
    directory=Path(directory).absolute()
    if directory!=directory.resolve() or not directory.is_relative_to(WORK_ROOT) or directory.exists():
        raise ContractViolation("PROBE_REQUIRES_NEW_WORK_DIRECTORY")
    freeze=verify_execution_freeze(freeze_path)
    if freeze.get("exploratory_revision") not in (PROBE_REVISION,PROBE_NEXT_REVISION,PROBE_CONTINUE_REVISION): raise ContractViolation("WRONG_PROBE_FREEZE")
    code=verify_implementation_manifest(manifest_path)["implementation_manifest_sha256"]
    cpu=verify_cpu_check_evidence(cpu_path)
    if cpu["implementation"]["implementation_manifest_sha256"]!=code: raise ContractViolation("CPU_CODE_MISMATCH")
    _verify_gpu_replay_evidence(replay_path,freeze_sha256=freeze["freeze_sha256"],code_sha256=code,cpu_check_sha256=cpu["cpu_check_sha256"])
    execution=freeze["execution"]
    source=read_verified_json(Path(execution["source_summary_path"]),expected_sha256=execution["source_summary_sha256"])
    event=probe_source_event(source,execution)
    source_ref=event["checkpoint"]
    source_scores=read_verified_json(Path(event["scores"]["path"]),expected_sha256=event["scores"]["sha256"])
    source_records=read_verified_json(Path(event["records"]["path"]),expected_sha256=event["records"]["sha256"])["records"]
    if source_scores["provenance"]["checkpoint_sha256"]!=source_ref["state_sha256"]:
        raise ContractViolation("PROBE_SOURCE_SCORE_CHECKPOINT_MISMATCH")
    baseline=gate_results(source_scores,source_records,freeze["concepts"],"T3")
    plan=read_verified_json(Path(freeze["resolved_artifacts"]["evaluation_plan"]))
    policy=read_verified_json(PROJECT_ROOT/"implementation/config/campaign_policy.json")["budget"]
    ledger=GpuBudgetLedger(WORK_ROOT/"ledger/gpu_budget.json",BudgetCaps(policy["campaign_gpu_hours_cap"],policy["prior_gpu_hours_user_reported"],policy["per_root_gpu_hours_cap"]))
    tokenizer,_=load_verified_tokenizer(Path(freeze["resolved_artifacts"]["tokenizer"]))
    planned_bytes=((len(execution["arms"])+1)*500*1024**2 if execution.get("prune_verified_midpoints") else 3300*1024**2)
    require_disk_reservation(directory.parent,planned_bytes=planned_bytes,emergency_free_bytes=policy["minimum_emergency_free_bytes"])
    directory.mkdir(parents=True)
    journal=PilotJournal(directory/"journal",{"freeze":freeze["freeze_sha256"],"code":code,"cpu":cpu["cpu_check_sha256"],"replay":sha256_file(replay_path)})
    results={}; status="INCOMPLETE"
    with ExclusiveGpu2Lock(directory/"run.lock"):
        try:
            for arm,settings in execution["arms"].items():
                verify_implementation_manifest(manifest_path)
                generator=torch.Generator(device="cpu").manual_seed(4102)
                stream,audit=probe_stream(freeze,tokenizer,generator,prompt_mix=settings["prompt_mix"],
                    concept_factor_balance=settings.get("concept_factor_balance",False),all_ko_prompts=settings.get("all_ko_prompts",False))
                journal.append("PLAN",{"arm":arm,"audit":audit,"settings":settings,"source":source_ref})
                with gpu2_training_session(ledger=ledger,lock_path=WORK_ROOT/"locks/gpu2.lock",run_id=directory.name+"-"+arm,
                        root_id=4101,phase="T3",reserved_seconds=1500,checkpoint_grace_seconds=120,
                        experiment_freeze_path=freeze_path) as session, BoundaryStopController() as stop:
                    configure_determinism()
                    model=build_random_model(seed=4101).to("cuda:0")
                    optimizer,signature=build_adamw(model,plan["training"])
                    scheduler=ConstantSchedule(optimizer,0.0003,"T3")
                    payload,_=load_checkpoint_payload(Path(source_ref["path"]),expected_state_sha256=source_ref["state_sha256"],
                        expected_manifest_sha256=source_ref["manifest_sha256"])
                    source_freeze=source["binding"].get("freeze_sha256",source["binding"].get("freeze"))
                    if payload["lineage"]["freeze_sha256"]!=source_freeze or payload["lineage"]["root_id"]!=4101:
                        raise ContractViolation("PROBE_SOURCE_LINEAGE_MISMATCH")
                    progress=PilotRuntime._restore(None,payload,model,optimizer,scheduler,generator,signature)
                    scheduler.transition(phase="T3",learning_rate=settings["learning_rate"])
                    progress.update({"plan_sha256":stream.plan_sha256,"loader_state":stream.state_dict()})
                    lineage=continuation_lineage(payload["lineage"],exploratory_arm=arm,
                        freeze_sha256=freeze["freeze_sha256"],code_sha256=code,resume_parent_sha256=source_ref["state_sha256"],
                        budget_lease_id=session.lease.lease_id,gpu_binding=session.binding.as_dict())
                    del payload
                    def save(reason):
                        require_disk_reservation(directory,planned_bytes=600*1024**2,emergency_free_bytes=policy["minimum_emergency_free_bytes"])
                        state=capture_training_state(model,optimizer,scheduler=scheduler,scaler=None,generators={"loader":generator},
                            progress=progress,lineage=lineage,parameter_group_signature=signature,
                            deterministic_settings=configure_determinism(),require_cuda_rng=True)
                        ref=save_checkpoint(directory/"checkpoints"/f"{arm}_{progress['phase_step']}",state).as_dict()
                        checked,_=load_checkpoint_payload(Path(ref["path"]),expected_state_sha256=ref["state_sha256"],expected_manifest_sha256=ref["manifest_sha256"])
                        if checked["semantic_fingerprints"]!=state["semantic_fingerprints"]: raise ContractViolation("PROBE_CHECKPOINT_READBACK_MISMATCH")
                        journal.append("CHECKPOINT",{"arm":arm,"reason":reason,"checkpoint":ref,"phase_step":progress["phase_step"]})
                        return ref
                    for half in range(2):
                        traces,training_status=run_gpu2_stage_steps(session,model,optimizer,scheduler,stream,progress,
                            checkpoint_grace_seconds=120,steps=180,loss_kind="lexical",grad_clip_norm=1.0,
                            stop_controller=stop,boundary_checkpoint=save)
                        trace=publish_json_once(directory/f"{arm}_trace_{half}.json",{"traces":[asdict(t) for t in traces]})
                        journal.append("TRACE",{"arm":arm,"artifact":trace})
                        if training_status!="FIXED_ENDPOINT_REACHED": raise ContractViolation(training_status)
                        ref=save("PAIRED_MIDPOINT" if half==0 else "FIXED_ENDPOINT")
                        if half==0:
                            midpoint=ref
                            # Real disk save/restore in every candidate arm.
                            state,_=load_checkpoint_payload(Path(ref["path"]),expected_state_sha256=ref["state_sha256"],expected_manifest_sha256=ref["manifest_sha256"])
                            progress=PilotRuntime._restore(None,state,model,optimizer,scheduler,generator,signature)
                            stream.load_state_dict(progress["loader_state"])
                            lineage["resume_parent_sha256"]=ref["state_sha256"]
                            del state
                        print(f"PROBE {arm} {180*(half+1)}/360",flush=True)
                    records=evaluation_records(freeze["concepts"],plan,"T3",endpoint=False)
                    provenance={**source_scores["provenance"],"checkpoint_sha256":ref["state_sha256"],
                        "checkpoint_manifest_sha256":ref["manifest_sha256"],"model_fingerprint":ref["model_fingerprint"],
                        "freeze_sha256":freeze["freeze_sha256"],"code_sha256":code}
                    def heartbeat():
                        ledger.require_time_remaining(session.lease,checkpoint_grace_seconds=120)
                        if stop.requested: raise ContractViolation("INTERRUPTED_AT_BOUNDARY")
                    scores=evaluate_without_rng_consumption(lambda:score_checkpoint(model,tokenizer,records,provenance,
                        context_length=256,max_new_tokens=64,experiment_freeze_path=freeze_path,checkpoint_directory=ref["path"],
                        generators={"loader":generator},progress_callback=heartbeat),generators={"loader":generator},include_cuda=True)
                    score_ref=publish_json_once(directory/f"{arm}_scores.json",scores)
                    record_ref=publish_json_once(directory/f"{arm}_records.json",{"records":records})
                    results[arm]=gate_results(scores,records,freeze["concepts"],"T3")
                    journal.append("EVALUATION",{"arm":arm,"gates":results[arm],"scores":score_ref,"records":record_ref})
                    if execution.get("prune_verified_midpoints"):
                        prune_probe_midpoint(directory,midpoint,ref,journal,arm)
                    print(f"PROBE {arm} EVALUATED",flush=True)
                del model,optimizer,scheduler,stream
                gc.collect(); torch.cuda.empty_cache()
            status="EXPLORATORY_COMPARISON_COMPLETE"
        except Exception as exc:
            status=str(exc); journal.append("STOP",{"status":status,"error_type":type(exc).__name__})
        complete=len(results)==len(execution["arms"])
        single=execution.get("single_candidate_continuation",False)
        result={"status":status,"baseline":baseline,"arms":results,
            "comparison":compare_arms(baseline,results,selection_version=execution.get("selection_version",1)) if complete and not single else None,
            "continuation":continuation_result(baseline,results) if complete and single else None,
            "events":journal.events,"binding":journal.binding,"budget":ledger.snapshot(),
            "automatic_retraining_started":False,"main_enabled":False,
            "interpretation":freeze["interpretation"]+"; dev-informed, not a held-out confirmation"}
        print(publish_json_once(directory/"run_summary.json",result),flush=True)
    return result
