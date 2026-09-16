"""Verify archived JSON bytes and recompute final results without GPU or weights."""
import gzip,hashlib,json,math,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from implementation.choice_trajectory import primary_cells,aggregate,WRAPPERS
s=json.loads((ROOT/'paper/summary.json').read_text());index=json.loads((ROOT/'evidence/index.json').read_text());original=Path(index['original_root'])
def read(ref):
 path=Path(ref['path']);rel=str(path.relative_to(original)) if path.is_absolute() else str(path)
 item=index['objects'][rel];raw=gzip.decompress((ROOT/item['object']).read_bytes());sha=hashlib.sha256(raw).hexdigest()
 assert sha==item['sha256']==ref['sha256'],rel
 return json.loads(raw)
for rel,ref in index['objects'].items():read({'path':rel,'sha256':ref['sha256']})
results=dict(s['forward_roots']);results['4101_reverse']=s['reverse_phases']['T3']
for seed,r in results.items():
 score=read(r['scores']);records=read(r['records'])['records'];ids=sorted(r['concept_Q2']);assert len(ids)==60
 cells=primary_cells(score['rows'],records,ids,score['provenance']['checkpoint_sha256']);a=aggregate(cells)
 assert a['wrappers']==r['per_wrapper'],seed
 for l,expected in r['Q2_mean'].items():
  actual=statistics.mean(statistics.mean(cells[i][w]['Q'][l] for w in WRAPPERS) for i in ids)
  assert math.isclose(actual,expected,abs_tol=1e-14), (seed,l)
 print(seed,'PASS: recomputed Q2, requested accuracy, Z and generation counts')
print('PASS:',len(index['objects']),'archived JSON hashes; all five final comparisons reproduced. No GPU used.')
