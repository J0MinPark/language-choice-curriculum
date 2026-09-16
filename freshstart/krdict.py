"""Bounded exact-query KRDICT collection. Requires a user-owned API key.

No retry loop. No URL/key logged. Collector yields unreviewed source candidates.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from .core import ContractError
from .reports import write_json

LANG_CODE={'en':'1','fr':'3','zh':'11'}
LANG_NAME={'en':'영어','fr':'프랑스어','zh':'중국어'}
BASE='https://krdict.korean.go.kr/api/search'

class SameHostRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        p=urllib.parse.urlparse(newurl)
        if p.scheme!='https' or p.hostname!='krdict.korean.go.kr':
            raise ContractError('UNEXPECTED_REDIRECT_HOST')
        return super().redirect_request(req,fp,code,msg,headers,newurl)

def parse_response(payload: bytes, expected_lang: str) -> dict:
    if expected_lang not in LANG_CODE: raise ContractError('UNKNOWN_API_LANGUAGE')
    if len(payload)>8*1024*1024: raise ContractError('RESPONSE_SIZE_LIMIT')
    upper=payload.upper()
    if b'<!DOCTYPE' in upper or b'<!ENTITY' in upper: raise ContractError('UNSAFE_XML')
    try: root=ET.fromstring(payload)
    except ET.ParseError: raise ContractError('MALFORMED_XML') from None
    if root.tag=='error':
        code=root.findtext('error_code','UNKNOWN')
        raise ContractError('KRDICT_API_ERROR_'+(code if re.fullmatch(r'\d{3}',code) else 'UNKNOWN'))
    if root.tag!='channel': raise ContractError('UNEXPECTED_XML_ROOT')
    try: total=int(root.findtext('total',''))
    except ValueError: raise ContractError('MISSING_TOTAL') from None
    if total<0: raise ContractError('NEGATIVE_TOTAL')
    items=root.findall('item'); out=[]
    for item in items:
        if item.findtext('pos')!='명사': continue
        target=item.findtext('target_code'); word=item.findtext('word')
        if not target or not word: raise ContractError('MISSING_ENTRY_ID')
        for sense in item.findall('sense'):
            so=sense.findtext('sense_order'); definition=sense.findtext('definition')
            if not so or not definition: raise ContractError('MISSING_SENSE')
            translations=[]
            for tr in sense.findall('translation'):
                lang=tr.findtext('trans_lang')
                if lang and lang not in (LANG_NAME[expected_lang],LANG_CODE[expected_lang],expected_lang):
                    continue
                translations.append({'raw_trans_word':tr.findtext('trans_word'),
                                     'raw_trans_definition':tr.findtext('trans_dfn')})
            out.append({'target_code':target,'sense_order':so,'ko_word':word,'ko_definition':definition,
                        'origin_raw':item.findtext('origin'),'pos':'명사','target_language':expected_lang,
                        'translations':translations, 'entry_url':item.findtext('link'),
                        'qa_status':'PENDING'})
    return {'total_entries':total,'returned_entries':len(items),'all_exact_entries_returned':len(items)>=total,
            'candidate_senses':out}

def collect(query_file: Path, output_dir: Path, max_requests: int=600, key: str|None=None) -> dict:
    key=key if key is not None else os.environ.get('KRDICT_API_KEY','')
    if not re.fullmatch(r'[0-9a-fA-F]{32}',key or ''):
        return {'status':'BLOCKED_CREDENTIALS','requests_made':0,
                'reason':'Set a valid-format key in the invoking terminal, or provide a legal raw snapshot. Never send the key in chat.'}
    queries=[q.strip() for q in query_file.read_text(encoding='utf-8').splitlines() if q.strip() and not q.startswith('#')]
    if len(set(queries))!=len(queries): raise ContractError('DUPLICATE_QUERY')
    if len(queries)*3>max_requests: raise ContractError('REQUEST_BUDGET_TOO_SMALL')
    output_dir.mkdir(parents=True,exist_ok=True)
    if any(output_dir.iterdir()): raise ContractError('RAW_OUTPUT_NOT_EMPTY_USE_NEW_RUN')
    records=[]; n=0; incomplete=False
    for index,q in enumerate(queries):
        for lang in ('en','zh','fr'):
            params={'key':key,'q':q,'translated':'y','trans_lang':LANG_CODE[lang],
                    'advanced':'y','method':'exact','pos':'1','part':'word','num':'100','start':'1'}
            safe={k:v for k,v in params.items() if k!='key'}
            item={'query_index':index,'language':lang,'endpoint':BASE,'params_without_key':safe}
            try:
                req=urllib.request.Request(BASE+'?'+urllib.parse.urlencode(params),
                    headers={'User-Agent':'LexicalChoiceResearch/4.0 (bounded exact-query collector)'})
                n+=1
                with urllib.request.build_opener(SameHostRedirect()).open(req,timeout=20) as response:
                    if response.status!=200: raise ContractError('HTTP_STATUS_'+str(response.status))
                    final=urllib.parse.urlparse(response.geturl())
                    if final.hostname!='krdict.korean.go.kr': raise ContractError('UNEXPECTED_REDIRECT_HOST')
                    payload=response.read(8*1024*1024+1)
                if key.encode() in payload: raise ContractError('CREDENTIAL_ECHO_IN_RESPONSE')
                parsed=parse_response(payload,lang)
                name=f'{index:04d}_{lang}.xml'; (output_dir/name).write_bytes(payload)
                item.update({'status':'PAYLOAD_SCHEMA_CHECKED','raw_file':name,
                             'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload),
                             'summary':parsed})
                incomplete |= not parsed['all_exact_entries_returned']
            except Exception as e:
                # urllib exceptions can contain the full URL with its key: never stringify them.
                item.update({'status':'BLOCKED_SOURCE','error_type':type(e).__name__})
                if isinstance(e,ContractError): item['safe_error_code']=str(e)
                records.append(item)
                result={'status':'BLOCKED_SOURCE','requests_made':n,'records':records}
                write_json(output_dir/'source_manifest.json',result)
                return result
            records.append(item)
            write_json(output_dir/'source_manifest.json',{'status':'IN_PROGRESS','requests_made':n,'records':records})
            time.sleep(.3)
    result={'status':'BLOCKED_TRUNCATED_EXACT_QUERY' if incomplete else 'COLLECTED_UNREVIEWED',
            'requests_made':n,'source_population_complete':False,'records':records,
            'scope':'Predeclared exact-word query list, NOT an exhaustive dictionary sample. No annotation or training completed.'}
    write_json(output_dir/'source_manifest.json',result)
    return result
