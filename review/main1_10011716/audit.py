"""Read-only audit of existing inputs and predictions; writes only review artifacts."""
from pathlib import Path
import hashlib
import json
import itertools
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/main/main1_10011716'
RESULT = ROOT / 'results/main1_10011716/main'
report = {}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def save():
    (OUT/'audit_results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')

def metrics(y,p):
    pred=p.argmax(axis=1)
    return dict(n=len(y), confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist(),
                macro_f1=float(f1_score(y,pred,average='macro')), accuracy=float(accuracy_score(y,pred)),
                nll=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()))

frame = pd.read_parquet(RESULT/'chunk_probabilities.parquet')
sys.path.insert(0,str(ROOT/'scripts/03_inference'))
from evaluation import aggregate
report['recomputed']={}
for level in ['chunk','contig','genome']:
    table=aggregate(frame,['p0','p1'],level)
    p=table[['p0','p1']].to_numpy()
    y=table.labels.to_numpy(int)
    m=metrics(y,p)
    original=json.loads((RESULT/level/'metrics.json').read_text())
    m['matches_saved_confusion_matrix']=m['confusion_matrix']==original['confusion_matrix']
    m['saved_macro_f1_difference']=m['macro_f1']-original['macro_f1']
    # Undo scalar temperature for a check of classification invariance.
    raw=p**0.75; raw/=raw.sum(1,keepdims=True)
    m['uncalibrated_nll']=metrics(y,raw)['nll']
    m['temperature_prediction_changes']=int((p.argmax(1)!=raw.argmax(1)).sum())
    report['recomputed'][level]=m
    if level=='genome':
        table['predicted_label']=p.argmax(1)
        table['correct']=table.labels==table.predicted_label
        table.loc[~table.correct].to_csv(OUT/'genome_errors.csv',index=False)
        report['genome_source_accuracy']=table.groupby('source').correct.agg(['count','mean']).to_dict('index')
        report['source_macro_accuracy']=float(table.groupby('source').correct.mean().mean())
        report['equal_split_group_accuracy']=float(table.groupby('split_group').correct.mean().mean())

history=pd.read_csv(RUN/'history.csv')
eps=.05
floor=-(1-eps/2)*np.log(1-eps/2)-(eps/2)*np.log(eps/2)
report['loss']={'binary_label_smoothing_floor':float(floor),
                'first_validation_loss':float(history.val_loss.iloc[0]),
                'minimum_validation_loss':float(history.val_loss.min()),
                'last_validation_loss':float(history.val_loss.iloc[-1])}
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].plot(history.epoch,history.train_loss,label='Train')
axes[0].plot(history.epoch,history.val_loss,label='Validation')
axes[0].axhline(floor,color='gray',ls='--',label='Smoothed CE lower bound')
axes[0].set(xlabel='Epoch',ylabel='Weighted chunk cross-entropy')
axes[0].legend()
axes[1].plot(history.epoch,history.genome_macro_f1,label='Validation genome macro-F1')
axes[1].axhline(report['recomputed']['genome']['macro_f1'],color='gray',ls='--',label='Holdout genome macro-F1')
axes[1].axvline(2.4,color='green',ls=':',label='Selected checkpoint')
axes[1].set(xlabel='Epoch',ylabel='Macro-F1',ylim=(.94,1))
axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT/'loss_vs_f1.png',dpi=170); plt.close(fig)
save()
print('Prediction recomputation done',flush=True)

manifest_cols=['record_id','source','genome_id','contig_id','base_group_id','split_group','split','main_label','taxonomy_genus','taxonomy_order','host_assembly_id','votu','canonical_sha256']
manifest=pd.read_csv(ROOT/'data/manifests/genomes.tsv.gz',sep='\t',usecols=manifest_cols,dtype=str,keep_default_na=False)
active=manifest[manifest.split.isin(['train','validation','inference'])]
report['manifest']={}
for key in ['genome_id','base_group_id','split_group','canonical_sha256']:
    report['manifest'][key+'_cross_split']=int((active.groupby(key).split.nunique()>1).sum())
group_splits=active.groupby('base_group_id').split.first().to_dict()
edges=pd.read_csv(ROOT/'data/manifests/similarity_edges.tsv',sep='\t')
edge_bad=[]; eligible=0
for row in edges.itertuples():
    a,b=group_splits.get(row.left_group),group_splits.get(row.right_group)
    if a and b:
        eligible+=1
        if a!=b: edge_bad.append([row.left_group,row.right_group,a,b])
report['manifest']['similarity_edges']={'total':len(edges),'both_groups_active':eligible,'cross_split':len(edge_bad),'examples':edge_bad[:10]}
for source,key in [('NCLDV','taxonomy_genus'),('EVE_NCLDV','host_assembly_id'),('EVE_MIRUS','host_assembly_id'),('PHAGE','votu')]:
    sub=active[(active.source==source)&(active[key]!='')]
    if source=='NCLDV':
        sub=sub[sub.base_group_id.str.startswith('NCLDV_GENUS::')]
    report['manifest'][source+'_'+key]={'unique':int(sub[key].nunique()),'cross_split':int((sub.groupby(key).split.nunique()>1).sum())}
ncldv=active[active.source=='NCLDV'].drop_duplicates('genome_id')
report['manifest']['NCLDV_genus_grouped_genomes_by_split']=ncldv.assign(grouped=ncldv.base_group_id.str.startswith('NCLDV_GENUS::')).groupby('split').grouped.agg(['count','sum']).to_dict('index')
source_split_genomes=active.drop_duplicates(['genome_id','split']).groupby(['split','source']).size().unstack(fill_value=0)
source_split_genomes.to_csv(OUT/'source_split_genomes.csv')
save(); print('Manifest and similarity-edge audit done',flush=True)

expected=json.loads((ROOT/'data/preprocessed/dataset_checksums.json').read_text())
run_meta=json.loads((RUN/'run_metadata.json').read_text())
inf_meta=json.loads((RESULT/'inference_metadata.json').read_text())
report['fingerprints']={}
for rel,reference in [
    ('data/manifests/genomes.tsv.gz',run_meta['data']['manifest_sha256']),
    ('runs/main/main1_10011716/best.pt',inf_meta['model_sha256']),
    *[(f'data/preprocessed/main/{s}.csv.gz',expected['files'][f'data/preprocessed/main/{s}.csv.gz']) for s in ['train','validation','inference']]]:
    actual=sha(ROOT/rel)
    report['fingerprints'][rel]={'sha256':actual,'matches':actual==reference}
report['fingerprints']['train_matches_run']=run_meta['data']['files']['train']==expected['files']['data/preprocessed/main/train.csv.gz']
report['fingerprints']['validation_matches_run']=run_meta['data']['files']['validation']==expected['files']['data/preprocessed/main/validation.csv.gz']
report['fingerprints']['inference_matches_result']=inf_meta['data']['files']['inference']==expected['files']['data/preprocessed/main/inference.csv.gz']
save(); print('File fingerprint audit done',flush=True)

# Independently recompute canonical sequence hashes, keeping holdout hashes in RAM.
rc=str.maketrans('ACGTN','TGCAN')
def canonical(sequence):
    return hashlib.sha256(min(sequence,sequence.translate(rc)[::-1]).encode()).digest()

hash_sets={}; id_sets={}; counts={}; overlaps={}; wrong_manifest={}; inf_alignment=[]
lookup=active.drop_duplicates('record_id').set_index('record_id')
for split in ['inference','validation','train']:
    ids={key:set() for key in ['genome_id','split_group','record_id']}
    seqset=set() if split!='train' else None
    count=bp=bad=0; hit_counts={other:0 for other in hash_sets}
    for chunk in pd.read_csv(ROOT/f'data/preprocessed/main/{split}.csv.gz',chunksize=25000,
                             usecols=['sequence','labels','source','genome_id','split_group','record_id','valid_length','chunk_id']):
        count+=len(chunk); bp+=int(chunk.valid_length.sum())
        for key in ids: ids[key].update(chunk[key])
        refs=lookup.reindex(chunk.record_id)
        bad+=int(((refs.split.to_numpy()!=split)|(refs.genome_id.to_numpy()!=chunk.genome_id.to_numpy())|
                   (refs.main_label.to_numpy()!=chunk.labels.astype(str).to_numpy())).sum())
        if split=='inference':
            saved=frame.iloc[count-len(chunk):count]
            inf_alignment.append(bool(np.array_equal(saved.chunk_id.to_numpy(),chunk.chunk_id.to_numpy()) and np.array_equal(saved.labels.to_numpy(),chunk.labels.to_numpy())))
        for seq in chunk.sequence:
            digest=canonical(seq)
            if seqset is not None: seqset.add(digest)
            for other,hashes in hash_sets.items():
                hit_counts[other]+=int(digest in hashes)
        if count%250000==0: print(f'{split}: {count:,} chunks checked',flush=True)
    counts[split]={'chunks':count,'bases':bp}
    wrong_manifest[split]=bad
    id_sets[split]=ids
    if seqset is not None: hash_sets[split]=seqset
    for other,hits in hit_counts.items(): overlaps[other+'__'+split]=hits
    report['independent_chunk_audit']={'counts':counts,'cross_split_canonical_hash_matching_rows':overlaps,
        'manifest_mismatching_rows':wrong_manifest,'inference_prediction_order_and_labels_match':all(inf_alignment),
        'id_overlaps':{a+'__'+b:{k:len(id_sets[a][k]&id_sets[b][k]) for k in ids} for a,b in itertools.combinations(id_sets,2)}}
    save(); print(f'{split} complete: {count:,} chunks, sequence overlaps {hit_counts}',flush=True)
print('Audit complete',flush=True)
