"""Band-restricted raw-spectrum comparisons and subject-cluster bootstrap."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial.distance import cosine,jensenshannon
p=Path('workspace/afpild/evaluation'); freq=np.fft.rfftfreq(1024,1/16000); rows=[]
for f in sorted((p/'cache').glob('*.npz')):
 d=np.load(f); s=d['sim_spectrum']; r=d['real_spectrum']
 for low,hi in [(0,250),(250,2000),(2000,8001),(250,8001)]:
  mask=(freq>=low)&(freq<hi); a=s[mask]/s[mask].sum(); b=r[mask]/r[mask].sum()
  rows.append({'name':f.stem,'subject':f.stem[:3].upper(),'low_hz':low,'high_hz':hi,'cosine':1-cosine(a,b),'js_distance':jensenshannon(a,b)})
pd.DataFrame(rows).to_csv(p/'audio_band_metrics.csv',index=False)
print(pd.DataFrame(rows).groupby(['low_hz','high_hz'])[['cosine','js_distance']].mean())
rng=np.random.default_rng(20261111); results={}
for file,cols in [('event_null_baseline.csv',['observed_f1_200ms','phase_randomized_f1_200ms']),('audio_mismatched_baseline.csv',['matched_spectral_cosine_4ch_mean_psd','other_subject_mean_cosine'])]:
 d=pd.read_csv(p/file); d['subject']=d.name.str[:3]; d['difference']=d[cols[0]]-d[cols[1]]; v=d.groupby('subject').difference.mean().to_numpy(); boot=v[rng.integers(0,len(v),(5000,len(v)))].mean(axis=1); results[file]={'difference':v.mean(),'subject_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist()}
(p/'baseline_uncertainty.json').write_text(json.dumps(results,indent=2)); print(results)
