"""Apply the original global-ID conditional paired bootstrap to new OOF predictions."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os
WORKSPACE = Path(os.environ.get("CDLD_WORKSPACE", Path(__file__).resolve().parents[1] / "work"))
import json,time,traceback,importlib.util
import numpy as np
import pandas as pd
ROOT=(WORKSPACE / '20260922_cdld_kmle_review_52025292');OUT=ROOT/'review_analysis'
spec=importlib.util.spec_from_file_location('inference',Path(__file__).with_name('analyze_static_probes.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def status(phase,**kw):
 p=OUT/'bootstrap_status.json';t=p.with_suffix('.pending');t.write_text(json.dumps(dict(phase=phase,updated=time.time(),**kw),indent=2));t.replace(p)
try:
 if not all((OUT/f'{ds}_seed{seed}/complete.json').exists() for ds in ['KMLE','EdNet'] for seed in range(42,47)):
  raise RuntimeError('Run run_review.py to completion first.')
 rows=[];draws={};metrics=[]
 for ds in ['KMLE','EdNet']:
  frames=[]
  for seed in range(42,47):
   p=OUT/f'{ds}_seed{seed}';assert json.loads((p/'complete.json').read_text())['cells']==60
   metrics.append(pd.read_csv(p/'metrics.csv'));frames.append(pd.read_csv(p/'oof_predictions.csv.gz'))
  frame=pd.concat(frames,ignore_index=True)
  for unit,target in [('student','Rasch_theta'),('student','2PL_theta'),('item','Rasch_b'),('item','2PL_b'),('item','2PL_log_a')]:
   for baseline,extended in [('B1','B1+L'),('B1+S','B1+L'),('B0','L')]:
    status('bootstrap',completed=len(rows),total=30,dataset=ds,target=target,baseline=baseline,extended=extended)
    summary,values=old.bootstrap_contrast(frame,unit,ds,target,baseline,extended)
    # The bootstrap tail fraction is descriptive, not a new null-hypothesis p-value.
    summary['bootstrap_fraction_nonpositive_corrected']=summary.pop('p_improvement_le_0')
    rows.append(summary);draws[f'{ds}_{target}_{baseline}_{extended}']=values
    pd.DataFrame(rows).to_csv(OUT/'paired_bootstrap_summary.partial.csv',index=False)
 pd.DataFrame(rows).to_csv(OUT/'paired_bootstrap_summary.csv',index=False)
 np.savez_compressed(OUT/'bootstrap_draws.npz',**draws)
 allmetrics=pd.concat(metrics,ignore_index=True);assert len(allmetrics)==600
 allmetrics.to_csv(OUT/'all_metrics.csv',index=False)
 allmetrics.groupby(['dataset','unit','target','features']).r2.agg(['mean','std','min','max']).reset_index().to_csv(OUT/'mean_r2.csv',index=False)
 status('complete',completed=30,total=30)
except Exception:
 status('failed',error=traceback.format_exc());raise
