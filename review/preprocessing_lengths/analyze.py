import csv,gzip,json
from pathlib import Path
from collections import Counter,defaultdict
root=Path(__file__).resolve().parents[2]; out=Path(__file__).resolve().parent
stats=defaultdict(Counter); hist=defaultdict(Counter); reasons=defaultdict(Counter); examples=[]
with gzip.open(root/'data/manifests/genomes.tsv.gz','rt') as f:
 for r in csv.DictReader(f,delimiter='\t'):
  p,s=r['prefix'],r['split']; n=int(r['sequence_length'])
  if s=='quarantine': reasons[p][r['exclusion_reason']]+=1; continue
  c=stats[p,s]; c['contigs']+=1; c['bases']+=n
  full,rem=divmod(n,5000)
  if n<=5000: chunks=[(n,'full_5kb' if n==5000 else 'short_contig')]
  elif 0<rem<500:
   a=(5000+rem)//2; chunks=[(5000,'full_5kb')]*(full-1)+[(a,'rebalanced'),(5000+rem-a,'rebalanced')]
  else: chunks=[(5000,'full_5kb')]*full+([(rem,'tail')] if rem else [])
  for length,kind in chunks:
   c['total']+=1;c[kind]+=1;hist[p,s][length]+=1
   if length<500: c['lt500']+=1
  if n!=5000 and any(k!='full_5kb' for _,k in chunks) and len(examples)<30:
   examples.append({k:r[k] for k in ('prefix','split','genome_id','contig_id','sequence_length')}|{'last_chunks':chunks[-2:]})
fields=['prefix','split','contigs','bases','total','full_5kb','short_contig','tail','rebalanced','lt500']
with (out/'prefix_split_counts.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for (p,s),c in sorted(stats.items()):w.writerow({'prefix':p,'split':s,**{k:c[k] for k in fields[2:]}})
with (out/'length_distribution.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['prefix','split','length_bp','chunks'])
 for (p,s),h in sorted(hist.items()):
  for n,c in sorted(h.items()):w.writerow([p,s,n,c])
agg=defaultdict(Counter)
for (p,s),c in stats.items():agg[p].update(c)
result={'prefix_totals':dict(agg),'split_totals':{s:dict(sum((c for (p,t),c in stats.items() if t==s),Counter())) for s in ('train','validation','inference')},'quarantine_reasons':dict(reasons),'examples':examples}
(out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
print(json.dumps(result,indent=2,ensure_ascii=False))
