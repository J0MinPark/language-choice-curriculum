"""CPU-only literal-expression screen of the actually consumed frozen corpus."""
import json
import re
import struct
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer
from implementation.src.artifacts import read_verified_json, sha256_file, publish_json_once
from implementation.src.semantic_experiment_freeze import _hash

ROOT = Path(__file__).resolve().parents[1]


def counts(texts, term):
    if not term:
        raise ValueError('empty expression')
    literal = re.compile('(?=' + re.escape(term) + ')')
    bounded = re.compile(r'(?<!\w)' + re.escape(term) + r'(?!\w)')
    raw = [len(literal.findall(t)) for t in texts]
    return {'substring_occurrences': sum(raw), 'documents_with_substring': sum(n > 0 for n in raw),
            'unicode_word_bounded_occurrences': sum(len(bounded.findall(t)) for t in texts)}


def main():
    parent = read_verified_json(ROOT/'work/freeze/accuracy_selected_360_v3/experiment_freeze.json',
        expected_sha256='6f6cc0d52b4162f1018b560c6f95419cb9258deda7e053e0a12bdc7950e39dfd')
    source = ROOT/'work/runs/pilot_jm02_4e30ec2/run_summary_000263.json'
    run = read_verified_json(source, expected_sha256='a8d55cd39abbe55b3124b74ff72921c15d3700aa305393848948598ff13d3ac4')
    ref = parent['artifacts']['corpus']
    corpus = read_verified_json(Path(ref['path']), expected_sha256=ref['sha256'])
    base = Path(ref['path']).parent
    for a in corpus['artifacts'].values():
        if sha256_file(base/a['path']) != a['sha256']:
            raise ValueError('corpus artifact hash mismatch')
    tokpath = ROOT/'work/tokenizer/wiki40b_ko_bpe_v4r1/tokenizer.json'
    if sha256_file(tokpath) != corpus['tokenizer_file_sha256']:
        raise ValueError('tokenizer hash mismatch')
    tokens = list(struct.unpack('<2000000H', (base/corpus['artifacts']['tokens']['path']).read_bytes()))
    spans = []
    for event in run['events']:
        if event['kind'] != 'TRACE' or event['data']['phase'] != 'CORPUS':
            continue
        a = event['data']['artifact']
        trace = read_verified_json(Path(a['path']), expected_sha256=a['sha256'])
        for step in trace['traces']:
            for rid, digest in zip(step['record_ids'], step['content_sha256s'], strict=True):
                prefix, start, end = rid.split(':'); start, end = int(start), int(end)
                if prefix != 'corpus' or digest != _hash({'token_stream':corpus['token_stream_sha256'], 'start':start, 'end':end}):
                    raise ValueError('actual trace content mismatch')
                spans.append((start, end))
    expected = [(s, min(s+256, len(tokens))) for s in range(0, len(tokens), 256)]
    if Counter(spans) != Counter(expected):
        raise ValueError('actual corpus coverage mismatch')
    tokenizer = Tokenizer.from_file(str(tokpath))
    docs = []
    covered = 0
    for line in (base/corpus['artifacts']['index']['path']).read_text().splitlines():
        row = json.loads(line); n = row['materialized_token_count']
        if not n:
            continue
        start = row['materialized_token_start']
        if start != covered:
            raise ValueError('document index gap')
        covered += n
        docs.append(tokenizer.decode(tokens[start:start+n], skip_special_tokens=False))
    if covered != len(tokens):
        raise ValueError('document coverage mismatch')
    rows = []
    for concept in parent['concepts']:
        for lang, term in concept['answers'].items():
            rows.append({'concept_id':concept['concept_id'], 'language':lang, 'expression':term,
                         **counts(docs, term)})
    totals = {}
    for lang in ('ko','en','zh','fr'):
        subset = [r for r in rows if r['language'] == lang]
        totals[lang] = {'expressions':len(subset), 'expressions_found':sum(r['substring_occurrences'] > 0 for r in subset),
                       'substring_occurrences':sum(r['substring_occurrences'] for r in subset),
                       'unicode_word_bounded_occurrences':sum(r['unicode_word_bounded_occurrences'] for r in subset)}
    report = {'status':'COMPLETE_LITERAL_SCREEN_NOT_SEMANTIC_EXPOSURE', 'source_run':str(source),
        'source_run_sha256':sha256_file(source), 'corpus_manifest':ref,
        'script_sha256':sha256_file(Path(__file__)), 'actual_trace_coverage_verified':True,
        'tokens':len(tokens), 'documents_including_partial':len(docs), 'rows':rows, 'totals':totals,
        'method':'Exact case-sensitive decoded frozen-token text; overlapping substring and Unicode word-boundary sensitivity counts, per document; no cross-document matches.',
        'limitations':['Literal homographs/substrings are not verified concept mentions.',
            'Unicode word boundaries miss Korean inflections and Chinese segmentation; neither count is a semantic frequency estimate.',
            'Counts describe unique corpus stream, not complete expressions within each 256-token training block.',
            'Shared strings/concepts may be counted in multiple expression rows; language totals are not disjoint.',
            'Only 2M training tokens screened; tokenizer training used a larger corpus, not screened here.',
            'Lexical-stage prompt incidental exposure and Qwen pretraining data are outside this screen.']}
    out = ROOT/'work/reports/corpus_expression_audit_v1.json'
    publish_json_once(out, report)
    print(json.dumps({'report':str(out), 'totals':totals, 'documents':len(docs)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
