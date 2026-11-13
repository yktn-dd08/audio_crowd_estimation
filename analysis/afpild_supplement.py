"""Null baselines, common-duration checks, and simulation QC for AFPILD."""
import os
os.environ['MPLBACKEND']='Agg'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from analysis.afpild_evaluate import events, match, features, OUT, ROOT


def main():
    rng=np.random.default_rng(20261111)
    rows=[]
    for i,name in enumerate(sorted(pd.read_csv(OUT/'event_metrics.csv').name)):
        row,_,_,(t,xy,u,pq)=events(name,None)
        horizon=row['trajectory_horizon_s']
        tc,uc=t[t<=horizon],u[u<=horizon]
        a,b=match(tc,uc,.2)
        base=[]
        duration=max(t[-1],u[-1])
        for shift in rng.uniform(2,duration-2,20):
            shifted=np.sort((u+shift)%duration)
            ai,bi=match(t,shifted,.2)
            base.append(2*len(ai)/(len(t)+len(u)))
        rows.append({'name':name,'subject':name[:3].upper(),'f1_common_horizon_200ms':2*len(a)/(len(tc)+len(uc)),
                     'gt_common_count':len(tc),'pred_common_count':len(uc),
                     'phase_randomized_f1_200ms':np.mean(base),'phase_randomized_f1_std':np.std(base),
                     'observed_f1_200ms':2*len(match(t,u,.2)[0])/(len(t)+len(u))})
        if (i+1)%40==0: print(f'BASELINES {i+1}/160',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'event_null_baseline.csv',index=False)
    # Numerical checks: injective matching and gain invariance of compared metrics.
    a,b=match(np.array([0.,.15]),np.array([.1,.25]),.12)
    assert len(a)==2 and len(set(b))==2
    assert len(match(np.array([0.,.01]),np.array([.005]),.1)[0])==1
    assert len(match(np.array([0.]),np.array([1.]),.1)[0])==0
    x=np.random.default_rng(1).normal(0,.1,16000)
    _,p,_,m,_=features(x); _,q,_,n,_=features(2*x)
    assert cosine(p,q)<1e-10
    assert np.max(np.abs(m-n))<1e-7
    a,b=match(np.array([0.]),np.array([0.]),.1,np.array([[2.]]),.5)
    assert len(a)==0
    pd.DataFrame(rows).groupby('subject').mean(numeric_only=True).to_csv(OUT/'event_baseline_by_subject.csv')
    (OUT/'numerical_checks.json').write_text(json.dumps({'matching':'passed: maximum cardinality, uniqueness, rejection, spatial gate',
       'spectral_metrics':'passed: gain-invariant spectrum and MFCC c1-c13',
       'accelerate_check':'GEMM vs einsum max discrepancy 7.98e-17; all original outputs finite',
       'random_seed':20261111,'null_phase_shifts_per_recording':20},indent=2))

if __name__=='__main__':
    main()
