"""Probability events, similarity and paired effects; standard library only."""
from __future__ import annotations
import math
import unicodedata
from collections import defaultdict
from typing import Mapping, Sequence
from . import LANGUAGES

class ContractError(ValueError):
    """A research/data contract was violated; never silently repair it."""

def canonical(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ContractError("EMPTY_EXPRESSION")
    if any(ch in text for ch in ("\n", "\r", "\x00")):
        raise ContractError("MULTILINE_OR_NUL_EXPRESSION")
    return " ".join(unicodedata.normalize("NFC", text).split())

def membership(answers: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    if set(answers) != set(LANGUAGES):
        raise ContractError("EXPECTED_KO_EN_ZH_FR")
    groups: dict[str, list[str]] = defaultdict(list)
    for lang in LANGUAGES:
        groups[canonical(answers[lang])].append(lang)
    return {word: tuple(langs) for word, langs in groups.items()}

def logsumexp(xs: Sequence[float]) -> float:
    if not xs or any(math.isnan(v) or v == math.inf for v in xs):
        raise ContractError("INVALID_LOG_VALUES")
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(math.fsum(math.exp(v-m) for v in xs))

def event_partition(answers: Mapping[str, str], logp_by_text: Mapping[str, float]) -> dict:
    """Inputs are log probabilities of COMPLETE canonical answer+terminator events.

    Tokenization/boundaries must have been audited by the actual model scorer.
    No length normalization. Shared strings form ONE event, not one per language.
    """
    members = membership(answers)
    if set(members) != set(logp_by_text):
        raise ContractError("PROBABILITY_EVENT_SET_MISMATCH")
    vals = [float(logp_by_text[y]) for y in members]
    if any(math.isnan(v) or v > 0 for v in vals):
        raise ContractError("INVALID_LOG_PROBABILITY")
    lz = logsumexp(vals)
    if lz > 1e-10:
        raise ContractError("REGISTERED_MASS_EXCEEDS_ONE")
    identifiable = all(len(ls) == 1 for ls in members.values())
    rawp = {y: math.exp(logp_by_text[y]) for y in members}
    if lz == -math.inf:
        return {"logZ": None, "logZ_status": "NEGATIVE_INFINITY", "Z": 0.0,
                "raw_p": rawp, "Q_membership": None, "Q_language": None,
                "identifiable": identifiable, "status": "UNDEFINED_ZERO_MASS"}
    q = {"+".join(members[y]): math.exp(logp_by_text[y]-lz) for y in members}
    lower = {l: q.get(l, 0.0) for l in LANGUAGES}
    upper = {l: math.fsum(v for k,v in q.items() if l in k.split("+")) for l in LANGUAGES}
    return {"logZ": lz, "logZ_status": "FINITE", "Z": math.exp(lz),
            "raw_p": rawp, "Q_membership": q,
            "Q_language": lower if identifiable else None,
            "assignment_lower": lower, "assignment_upper": upper,
            "identifiable": identifiable, "status": "DEFINED"}

def validate_token_events(events: Mapping[str, Sequence[int]]) -> None:
    """Reject duplicate or prefix-overlapping canonical token events."""
    seqs = [tuple(v) for v in events.values()]
    if not seqs or any(not x for x in seqs):
        raise ContractError("EMPTY_TOKEN_EVENT")
    if len(set(seqs)) != len(seqs):
        raise ContractError("TOKEN_EVENT_COLLISION")
    for i, a in enumerate(seqs):
        for j, b in enumerate(seqs):
            if i != j and len(a) <= len(b) and b[:len(a)] == a:
                raise ContractError("NON_PREFIX_FREE_TOKEN_EVENTS")

def continuation_tokens(encode, prefix: str, answer: str) -> list[int]:
    canonical(answer)
    p = list(encode(prefix))
    both = list(encode(prefix + answer + "\n"))
    if not p or both[:len(p)] != p or len(both) <= len(p):
        raise ContractError("UNSTABLE_TOKEN_BOUNDARY")
    return both[len(p):]

def generated_label(first_line: str, answers: Mapping[str,str], requested: str|None) -> dict:
    if requested is not None and requested not in LANGUAGES:
        raise ContractError("UNKNOWN_REQUESTED_LANGUAGE")
    try:
        m = membership(answers).get(canonical(first_line), ())
    except ContractError:
        if first_line.strip():
            raise
        m = ()
    return {"registered_match": bool(m), "membership": list(m),
            "request_compatible": (requested in m) if requested is not None else None,
            "registered_other_language": (bool(m) and requested not in m) if requested is not None else None,
            "unregistered": not bool(m)}

def signed_change(qa: Mapping[str,float], qb: Mapping[str,float], g: int) -> dict[str,float]:
    if g not in (-1,1) or set(qa) != set(LANGUAGES) or set(qb) != set(LANGUAGES):
        raise ContractError("INVALID_EFFECT_INPUT")
    for q in (qa,qb):
        if any(not math.isfinite(v) or v < 0 or v > 1 for v in q.values()):
            raise ContractError("INVALID_Q")
        if not math.isclose(math.fsum(q.values()),1.0,abs_tol=1e-8):
            raise ContractError("Q_DOES_NOT_SUM_TO_ONE")
    return {l: g*(qa[l]-qb[l]) for l in LANGUAGES}

def vector_mae(truth: Sequence[Sequence[float]], pred: Sequence[Sequence[float]]) -> float:
    if not truth or len(truth)!=len(pred) or any(len(t)!=4 or len(p)!=4 for t,p in zip(truth,pred)):
        raise ContractError("INVALID_MAE_SHAPE")
    values=[abs(a-b) for t,p in zip(truth,pred) for a,b in zip(t,p)]
    if any(not math.isfinite(v) for v in values):
        raise ContractError("NONFINITE_MAE")
    return math.fsum(values)/len(values)

def spelling_similarity(a: str,b: str) -> float:
    a,b=canonical(a),canonical(b)
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1):
            cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return 1.0-prev[-1]/max(len(a),len(b))

def token_jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    # Caller passes expression-only tokens; no prompt, separator or padding tokens.
    aa,bb=set(a),set(b)
    if not aa or not bb:
        raise ContractError("EMPTY_EXPRESSION_TOKENS")
    return len(aa&bb)/len(aa|bb)
