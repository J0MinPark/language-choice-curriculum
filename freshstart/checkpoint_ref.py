"""In-memory PyTorch full-state reference. Not the actual data-loader/GPU integration.

Caller owns loader state and must restore its iterator/position before continuing.
Only import this module in a Torch + NumPy environment.
"""
from __future__ import annotations
import copy
import hashlib
import random
import numpy as np
import torch
from .core import ContractError

REQUIRED_EXTRA={'root_id','stage','global_step','loader_state','accumulation_step','config_sha256','tokenizer_sha256'}

def fingerprint(state) -> str:
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            t=x.detach().cpu().contiguous(); h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=lambda k:str(k)):
                h.update(repr(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode())
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(state);return h.hexdigest()

def capture(model, optimizer, extra: dict, scheduler=None, scaler=None, generators=None):
    if optimizer is None:raise ContractError('OPTIMIZER_STATE_REQUIRED')
    if not REQUIRED_EXTRA.issubset(extra):raise ContractError('INCOMPLETE_CHECKPOINT_METADATA')
    if extra['accumulation_step']!=0:raise ContractError('SAVE_ONLY_AT_ACCUMULATION_BOUNDARY')
    return {'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
            'extra':copy.deepcopy(extra),'python_rng':random.getstate(),'numpy_rng':copy.deepcopy(np.random.get_state()),
            'torch_rng':torch.get_rng_state().clone(),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'generators':{k:g.get_state().clone() for k,g in (generators or {}).items()},
            'scheduler':copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
            'scaler':copy.deepcopy(scaler.state_dict()) if scaler is not None else None}

def restore(state, model, optimizer, scheduler=None, scaler=None, generators=None):
    if optimizer is None or 'optimizer' not in state:raise ContractError('OPTIMIZER_STATE_REQUIRED')
    if bool(scheduler is not None)!=bool(state['scheduler'] is not None):raise ContractError('SCHEDULER_MISMATCH')
    if bool(scaler is not None)!=bool(state['scaler'] is not None):raise ContractError('SCALER_MISMATCH')
    if set(generators or {})!=set(state['generators']):raise ContractError('GENERATOR_SET_MISMATCH')
    model.load_state_dict(copy.deepcopy(state['model']));optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    if scheduler is not None:scheduler.load_state_dict(copy.deepcopy(state['scheduler']))
    if scaler is not None:scaler.load_state_dict(copy.deepcopy(state['scaler']))
    random.setstate(state['python_rng']);np.random.set_state(copy.deepcopy(state['numpy_rng']))
    torch.set_rng_state(state['torch_rng'].clone())
    if state['cuda_rng'] is not None:
        if not torch.cuda.is_available():raise ContractError('CUDA_STATE_WITHOUT_CUDA')
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    for k,g in (generators or {}).items():g.set_state(state['generators'][k].clone())
    return copy.deepcopy(state['extra'])
