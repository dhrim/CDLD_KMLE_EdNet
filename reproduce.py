"""Rebuild paper tables and figures from the final run-level results."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'

def reproduce(out=None):
    out=Path(out) if out else ROOT/'outputs'
    out.mkdir(parents=True,exist_ok=True)
    read=lambda name:pd.read_csv(DATA/(name+'.csv'))
    raw=read('irt_all_metrics')
    means=raw.groupby(['dataset','unit','target','features'],as_index=False).r2.mean()
    table1=means.pivot(index=['dataset','unit','target'],columns='features',values='r2')[['B1','B1+S','B1+L']].reset_index()
    ci=read('irt_paired_bootstrap_summary')
    for base,label in [('B1','summary'),('B1+S','svd64')]:
        c=ci[(ci.baseline==base)&(ci.extended=='B1+L')][['dataset','unit','target','estimate','ci_low','ci_high']]
        table1=table1.merge(c.rename(columns={k:f'{label}_{k}' for k in ['estimate','ci_low','ci_high']}),on=['dataset','unit','target'],validate='one_to_one')
        np.testing.assert_allclose(table1['B1+L']-table1[base],table1[f'{label}_estimate'],atol=1e-10)
    table1=table1.rename(columns={'B1':'Summary','B1+S':'Summary+SVD64','B1+L':'Summary+CDLD64'})
    part=read('part_seed_metrics').groupby('features',as_index=False)[['log_loss','balanced_accuracy','macro_f1']].mean()
    pred=read('rq1_seed_metrics').groupby(['dataset','protocol','model','k'],as_index=False)[['auc','score_correlation','log_loss','brier','accuracy']].mean()
    table3=pred[(pred.model.isin(['CDLD','2PL']))&((pred.protocol=='warm')|((pred.protocol=='cold')&(pred.k==40)))].copy()
    ni=read('rq1_cold_ni');table3=table3.merge(ni[ni.comparator=='2PL'][['dataset','difference','repeat_lower_one_sided95','repeat_noninferior']],on='dataset',how='left')
    table3['repeat_noninferior']=table3['repeat_noninferior'].astype(object)
    table3.loc[table3.protocol=='warm',['difference','repeat_lower_one_sided95','repeat_noninferior']]=np.nan
    tables={'table1_irt':table1,'table2_part':part,'table2_part_difference':read('rq2b_log_loss_corrected'),'table3_prediction':table3,'supplement_prediction':pred,'supplement_noninferiority':ni,'supplement_irt_all':raw.groupby(['dataset','unit','target','features']).r2.agg(mean='mean',minimum='min',maximum='max').reset_index(), 'supplement_rt_sample':read('rt_sample_counts'),'supplement_irt_initial':read('rq2a_summary'),'supplement_rt_means':read('rt_seed_metrics').groupby('condition',as_index=False)[['mse','rmse','r2']].mean(),'supplement_rt_differences':read('rt_contrasts'),'supplement_metadata_means':read('augmentation_seed_metrics').groupby('protocol',as_index=False).mean(numeric_only=True).drop(columns='seed'),'supplement_metadata_differences':read('augmentation_contrasts'),'supplement_csp':read('csp_gates'),'supplement_tag':read('tag_summary'),'supplement_irt_stability':read('irt_stability_unaligned')}
    for name,frame in tables.items():frame.to_csv(out/(name+'.csv'),index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    targets=['Rasch_theta','2PL_theta','Rasch_b','2PL_b','2PL_log_a']
    labels=['Rasch ability','2PL ability','Rasch difficulty','2PL difficulty','2PL log discrimination']
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharex=True,sharey=True)
    for ax,ds in zip(axes,['KMLE','EdNet']):
        for base,offset,color,marker,label in [('B1',-.12,'#007997','o','vs Summary'),('B1+S',.12,'#c16b20','s','vs Summary+SVD64')]:
            g=ci[(ci.dataset==ds)&(ci.baseline==base)&(ci.extended=='B1+L')].set_index('target').loc[targets]
            ax.errorbar(g.estimate,np.arange(5)+offset,xerr=[g.estimate-g.ci_low,g.ci_high-g.estimate],fmt=marker,color=color,capsize=3,label=label)
        ax.axvline(0,color='gray',ls='--');ax.set(title=ds,yticks=range(5),yticklabels=labels,xlabel='R² difference: Summary+CDLD64 − comparator',xlim=(-.25,.35))
    axes[0].invert_yaxis();axes[1].legend(frameon=False);fig.tight_layout();fig.savefig(out/'figure1.png',dpi=240);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,ds in zip(axes,['KMLE','EdNet']):
        for model in ['CDLD','2PL','Rasch']:
            g=pred[(pred.dataset==ds)&(pred.protocol=='cold')&(pred.model==model)].sort_values('k')
            ax.plot(g.k,g.score_correlation,'o-',label=model)
        ax.set(title=ds,xlabel='Adaptation responses per student',xticks=[0,4,8,12,16,20,40])
    axes[0].set_ylabel('Student score correlation (Pearson r)');axes[1].legend(frameon=False);fig.tight_layout();fig.savefig(out/'figure2.png',dpi=240);plt.close(fig)
    return tables

if __name__=='__main__':
    tables=reproduce()
    print(f'Wrote {len(tables)} tables and 2 figures to {ROOT / "outputs"}')
