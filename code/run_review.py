"""Exploratory review-response comparisons; no CDLD fitting or selection."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os
WORKSPACE = Path(os.environ.get("CDLD_WORKSPACE", Path(__file__).resolve().parents[1] / "work"))
import json,time,traceback,importlib.util,gc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold,GridSearchCV
ROOT=(WORKSPACE / '20260922_cdld_kmle_review_52025292')
SRC=WORKSPACE;OUT=ROOT/'review_analysis'
spec=importlib.util.spec_from_file_location('original',Path(__file__).with_name('run_static_probes.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
START=time.time();completed=0

def save(p,x):
 t=p.with_suffix('.pending');t.write_text(json.dumps(x,indent=2));t.replace(p)
def status(phase,**kw):save(OUT/'analysis_status.json',dict(phase=phase,completed=completed,total=600,updated=time.time(),elapsed=time.time()-START,**kw))
def ridge(ids,x,y):
 order=np.argsort(ids);x=x[order];y=y[order];ids=ids[order];pred=np.full(len(y),np.nan);rows=[]
 for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=20260920).split(ids)):
  model=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler()),('ridge',Ridge())])
  cv=KFold(5,shuffle=True,random_state=20260920+fold+1)
  search=GridSearchCV(model,{'ridge__alpha':old.RIDGE_ALPHAS},scoring='neg_mean_squared_error',cv=cv,n_jobs=1)
  search.fit(x[tr],y[tr]);pred[te]=search.predict(x[te]);rows.append(dict(fold=fold,alpha=float(search.best_params_['ridge__alpha'])))
 return pred[np.argsort(order)],rows

def summaries(data,train,nu,ni):
 pc=np.zeros(nu);ps=np.zeros(nu);ic=np.zeros(ni);ys=np.zeros(ni)
 for start in range(0,len(train),1000000):
  a=data[train[start:start+1000000]];u=a['person'];i=a['item'];y=a['correct'].astype(float)
  pc+=np.bincount(u,minlength=nu);ps+=np.bincount(u,weights=y,minlength=nu);ic+=np.bincount(i,minlength=ni);ys+=np.bincount(i,weights=y,minlength=ni)
 pr=np.divide(ps,pc,out=np.full(nu,ps.sum()/pc.sum()),where=pc>0);ir=np.divide(ys,ic,out=np.full(ni,ys.sum()/ic.sum()),where=ic>0)
 n=np.zeros(ni);sx=np.zeros(ni);sy=np.zeros(ni);sxx=np.zeros(ni);syy=np.zeros(ni);sxy=np.zeros(ni)
 for start in range(0,len(train),1000000):
  a=data[train[start:start+1000000]];keep=pc[a['person']]>1;a=a[keep];u=a['person'];i=a['item'];y=a['correct'].astype(float);x=(ps[u]-y)/(pc[u]-1)
  n+=np.bincount(i,minlength=ni)
  for acc,val in [(sx,x),(sy,y),(sxx,x*x),(syy,y*y),(sxy,x*y)]:acc+=np.bincount(i,weights=val,minlength=ni)
 den=np.sqrt(np.maximum(0,n*sxx-sx*sx)*np.maximum(0,n*syy-sy*sy));corr=np.divide(n*sxy-sx*sy,den,out=np.full(ni,np.nan),where=(den>0)&(n>1))
 smooth=(ys+.5)/(ic+1)
 return pr,pc,ir,ic,np.column_stack([ir,np.log1p(ic),corr,np.log(smooth/(1-smooth))])

def main():
 global completed
 OUT.mkdir(parents=True,exist_ok=True)
 previous=pd.read_csv(SRC/'20260920_representation_study/results/static_probes/metrics.csv')
 for ds in ['KMLE','EdNet']:
  data=np.load(SRC/'20260914_cdld_reviewer/cache'/ds/'full/responses.npy',mmap_mode='r')
  for seed in range(42,47):
   result=OUT/f'{ds}_seed{seed}';result.mkdir(exist_ok=True)
   d=SRC/'20260917_cdld_finder_5x10/results'/ds/'full'/f'warm_seed{seed}'
   split=np.load(d/'split.npz');train=split['train_rows'];ids_u=split['evaluation_people'];u=np.load(d/'CDLD_user_latents.npy',mmap_mode='r');v=np.load(d/'CDLD_item_latents.npy',mmap_mode='r')
   status('response_summaries',dataset=ds,seed=seed)
   pr,pc,ir,ic,b1i=summaries(data,train,len(u),len(v));b0u=pr[ids_u,None];b0i=np.column_stack([ir,np.log1p(ic)]);b1u=np.column_stack([pr[ids_u],np.log1p(pc[ids_u])])
   pars=pd.read_csv(d/'2PL_item_parameters.csv');rasch=pd.read_csv(d/'Rasch_item_parameters.csv');targets_i={'Rasch_b':rasch.b.to_numpy(float),'2PL_b':pars.b.to_numpy(float),'2PL_log_a':np.log(pars.a.to_numpy(float))}
   good=rasch.estimable.to_numpy(bool)&pars.estimable.to_numpy(bool)&np.isfinite(np.column_stack(list(targets_i.values()))).all(axis=1);ids_i=np.flatnonzero(good)
   pred=np.load(d/'predictions.npz');jobs=[('student',name+'_theta',ids_u,pred['theta|'+name+'|-1'],b0u,np.asarray(u[ids_u]),b1u) for name in ['Rasch','2PL']]
   jobs += [('item',name,ids_i,y[good],b0i[good],np.asarray(v[ids_i]),b1i[good]) for name,y in targets_i.items()]
   # Gate all ten old B0/L cells before evaluating any new feature set for this seed.
   gates=[];saved={}
   for unit,target,ids,y,b0,latent,b1 in jobs:
    for name,x in [('B0',b0),('L',latent)]:
     status('reproduction_gate',dataset=ds,seed=seed,target=target,features=name)
     est,folds=old.nested_ridge(ids,x,y);met=old.regression_metrics(y,est)
     original_name='latent64' if name=='L' else ('response_rate' if unit=='student' else 'response_summaries')
     match=previous[(previous.dataset==ds)&(previous.seed==seed)&(previous.target==target)&(previous.features==original_name)]
     assert len(match)==1
     delta=abs(met['r2']-float(match.iloc[0].r2));gates.append(dict(target=target,features=name,r2_difference=delta))
     assert delta<1e-5,('reproduction mismatch',ds,seed,target,name,delta)
     saved[(target,name)]=(est,folds,met)
   save(result/'reproduction_gate.json',gates)
   status('linear64_svd',dataset=ds,seed=seed)
   # Float64 sparse residuals. Only saved train rows contribute.
   a=data[train];matrix=csr_matrix((a['correct'].astype(float)-ir[a['item']],(a['person'],a['item'])),shape=(len(u),len(v)))
   del a
   rng=np.random.default_rng(20260922);v0=rng.standard_normal(min(matrix.shape))
   uu,s,vh=svds(matrix,k=64,solver='arpack',tol=1e-6,maxiter=10000,v0=v0)
   order=np.argsort(s)[::-1];s=s[order];uu=uu[:,order];vh=vh[order]
   # Check numerical residual, independent of downstream predictive performance.
   residual=np.linalg.norm(matrix@vh.T-uu*s,axis=0)/np.maximum(s,1e-15)
   residual2=np.linalg.norm(matrix.T@uu-vh.T*s,axis=0)/np.maximum(s,1e-15)
   save(result/'svd_diagnostics.json',dict(singular_values=s.tolist(),left_relative_residual=residual.tolist(),right_relative_residual=residual2.tolist()))
   assert max(residual.max(),residual2.max())<1e-4,'SVD residual failure'
   su=uu[ids_u]*s;si=vh.T[ids_i]*s
   np.savez(result/'linear64.npz',student_ids=ids_u,item_ids=ids_i,student=su,item=si,singular_values=s)
   del matrix,uu,vh;gc.collect()
   metrics=[];frames=[];fold_records=[]
   random_u={k:np.random.default_rng(k).standard_normal((len(u),64))[ids_u] for k in range(2026092201,2026092204)}
   random_i={k:np.random.default_rng(k).standard_normal((len(v),64))[ids_i] for k in range(2026092201,2026092204)}
   for unit,target,ids,y,b0,latent,b1 in jobs:
    linear=su if unit=='student' else si;random=random_u if unit=='student' else random_i
    inputs={'B0':b0,'L':latent,'B1':b1,'B1+L':np.column_stack([b1,latent]),'S':linear,'B1+S':np.column_stack([b1,linear])}
    for k,r in random.items():inputs[f'R{k}']=r;inputs[f'B1+R{k}']=np.column_stack([b1,r])
    for name,x in inputs.items():
     status('nested_regression',dataset=ds,seed=seed,target=target,features=name)
     if name in ['B0','L']:est,folds,met=saved[(target,name)]
     else:est,folds=ridge(ids,x,y);met=old.regression_metrics(y,est)
     record=dict(dataset=ds,seed=seed,unit=unit,target=target,features=name)
     metrics.append({**record,**met});frames.append(pd.DataFrame({**record,'entity_id':ids,'actual':y,'predicted':est}));fold_records.extend({**record,**f} for f in folds)
     completed+=1
     pd.DataFrame(metrics).to_csv(result/'metrics.partial.csv',index=False)
    # Persist each complete target to survive an interruption.
    pd.concat(frames,ignore_index=True).to_csv(result/'oof_predictions.partial.csv.gz',index=False,compression='gzip')
   pd.DataFrame(metrics).to_csv(result/'metrics.csv',index=False);pd.DataFrame(fold_records).to_csv(result/'fold_parameters.csv',index=False)
   (result/'oof_predictions.partial.csv.gz').rename(result/'oof_predictions.csv.gz')
   save(result/'complete.json',dict(cells=len(metrics),time=time.time()))
   del train,split,jobs,random_u,random_i,saved;gc.collect()
 status('comparisons_complete_bootstrap_pending')
if __name__=='__main__':
 try:main()
 except Exception:status('failed',error=traceback.format_exc());raise
