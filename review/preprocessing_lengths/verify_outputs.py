import csv,gzip,json
from collections import Counter
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
counts=Counter();bases=Counter();bad=0
for split in ('train','validation','inference'):
 with gzip.open(root/f'data/preprocessed/main/{split}.csv.gz','rt') as f:
  for r in csv.DictReader(f):
   n=int(r['valid_length']);counts[r['source'],split,n]+=1;bases[split]+=n
   bad+=len(r['sequence'])!=n or int(r['chunk_end_exclusive'])-int(r['chunk_start_0based'])!=n
 print(split,'read complete',flush=True)
expected=Counter()
with (out/'length_distribution.csv').open() as f:
 for r in csv.DictReader(f):expected[r['prefix'].replace('-','_'),r['split'],int(r['length_bp'])]+=int(r['chunks'])
result={'main_actual_chunks':sum(counts.values()),'bases_by_split':dict(bases),'invalid_lengths':bad,'matches_manifest_length_distribution':counts==expected,'different_keys':len(set(counts)|set(expected)) if counts!=expected else 0}
(out/'verification.json').write_text(json.dumps(result,indent=2));print(result)
