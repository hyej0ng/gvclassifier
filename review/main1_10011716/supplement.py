from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
cols=['source','genome_id','base_group_id','split','taxonomy_genus']
d=pd.read_csv(ROOT/'data/manifests/genomes.tsv.gz',sep='\t',usecols=cols,dtype=str,keep_default_na=False)
d=d[(d.source=='NCLDV')&d.split.isin(['train','validation','inference'])].drop_duplicates('genome_id')
known=d[d.base_group_id.str.startswith('NCLDV_GENUS::')]
out={'usable_NCLDV_genus_cross_split':int((known.groupby('taxonomy_genus').split.nunique()>1).sum()),
     'usable_NCLDV_genus_unique':int(known.taxonomy_genus.nunique()),
     'unknown_NCLDV_genus_names':sorted(d[~d.base_group_id.str.startswith('NCLDV_GENUS::')].taxonomy_genus.unique().tolist())}
p=pd.read_csv(ROOT/'results/main1_10011716/main/genome/predictions.csv.gz')
n=p[p.source=='NCLDV'].merge(d[['genome_id','base_group_id']],on='genome_id',validate='one_to_one')
n['genus_grouped']=n.base_group_id.str.startswith('NCLDV_GENUS::')
n['correct']=n.true_label==n.predicted_label
out['NCLDV_holdout_genus_grouped_accuracy']=n.groupby('genus_grouped').correct.agg(['count','sum','mean']).to_dict('index')
(OUT/'supplement_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

h=pd.read_csv(OUT/'validation_detailed.csv')
floor=-(.975*np.log(.975)+.025*np.log(.025))
fig,axes=plt.subplots(1,2,figsize=(11,4))
train=pd.read_csv(ROOT/'runs/main/main1_10011716/history.csv')
axes[0].plot(train.epoch,train.train_loss,label='Train')
axes[0].plot(train.epoch,train.val_loss,label='Validation')
axes[0].axhline(floor,color='gray',ls='--',label='Smoothed CE lower bound')
axes[0].set(xlabel='Epoch',ylabel='Weighted chunk cross-entropy')
axes[0].legend(fontsize=8)
axes[1].plot(h.epoch,h.eval_genome_macro_f1,label='Validation genome macro-F1')
axes[1].plot(h.epoch,h.eval_chunk_macro_f1,label='Validation chunk macro-F1')
axes[1].axhline(.9829994991625246,color='gray',ls='--',label='Holdout genome macro-F1')
axes[1].axvline(2.4,color='green',ls=':',label='Selected checkpoint')
axes[1].set(xlabel='Epoch',ylabel='Macro-F1',ylim=(.88,1))
axes[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'loss_vs_f1.png',dpi=170);plt.close(fig)
