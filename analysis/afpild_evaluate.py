"""Event reconstruction and raw-audio domain comparison (no denoising)."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ.setdefault('MPLCONFIGDIR', str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'workspace/afpild/evaluation/mplconfig'))
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal, stats, spatial
from scipy.optimize import linear_sum_assignment
from scipy.fft import dct
from shapely import wkt
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'workspace/afpild/evaluation'
FS, HOP, NFFT = 16000, 256, 1024
EPS = 1e-15


def match(t, u, limit, distances=None, radius=None):
    """Maximum-cardinality then minimum-time-error injective assignment."""
    dt = np.abs(t[:, None] - u[None, :])
    valid = dt <= limit
    if radius is not None:
        valid &= distances <= radius
    cost = np.full((len(t), len(u) + len(t)), len(t) + 1., dtype=float)
    cost[:, :len(u)] = np.where(valid, dt / max(limit, EPS), 1e9)
    i, j = linear_sum_assignment(cost)
    keep = j < len(u)
    return i[keep], j[keep]


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > EPS and np.std(y) > EPS else 0.


def events(name, manifest):
    subject, session = name.split('_')
    gt = pd.read_csv(ROOT / f'data/gis/afpild/crowd/afpild_footstep_events/events/{subject.upper()}/{name}_footsteps.csv')
    t, xy = gt.time_s.to_numpy(), gt[['x_m', 'y_m']].to_numpy()
    js = json.loads((ROOT / f'workspace/afpild/sim/{name}/footstep_info.json').read_text())
    assert len(js) == 1
    pred = pd.DataFrame(next(iter(js.values()))['time_series']).sort_values('t')
    u, pq = pred.t.to_numpy(), pred[['x', 'y']].to_numpy()
    # The simulator only has floor(last annotation time) seconds of trajectory.
    tagged = pd.read_csv(ROOT / f'workspace/data/gis/afpild/crowd/asplaht/test/{name}_crowd.csv')
    coords = np.asarray(wkt.loads(tagged.geom.iloc[0]).coords)
    horizon = len(coords) - 1
    assert np.allclose(coords[0], xy[0], atol=1e-8)
    distances = spatial.distance.cdist(xy, pq)
    rows = []
    for threshold in (.05, .1, .2, .3, .5):
        a, b = match(t, u, threshold)
        rows.append({'name': name, 'subject': subject.upper(), 'session': int(session),
                     'time_tolerance_s': threshold, 'space_tolerance_m': np.nan,
                     'gt': len(t), 'pred': len(u), 'tp': len(a), 'precision': len(a)/len(u),
                     'recall': len(a)/len(t), 'f1': 2*len(a)/(len(t)+len(u))})
    for radius in (.2, .5, 1.):
        a, b = match(t, u, .2, distances, radius)
        rows.append({'name': name, 'subject': subject.upper(), 'session': int(session),
                     'time_tolerance_s': .2, 'space_tolerance_m': radius,
                     'gt': len(t), 'pred': len(u), 'tp': len(a), 'precision': len(a)/len(u),
                     'recall': len(a)/len(t), 'f1': 2*len(a)/(len(t)+len(u))})
    a, b = match(t, u, .2)
    dt, dist = u[b]-t[a], distances[a, b]
    matched = pd.DataFrame({'name': name, 'gt_index': a, 'pred_index': b,
                            'gt_t': t[a], 'pred_t': u[b], 'dt_s': dt, 'xy_error_m': dist})
    angle = np.abs(np.angle(np.exp(1j*(np.arctan2(pq[b, 1], pq[b, 0])-np.arctan2(xy[a, 1], xy[a, 0]))))) * 180/np.pi
    interpolated = np.column_stack([np.interp(u, t, xy[:, k]) for k in range(2)])
    reconstruction = np.linalg.norm(pq-interpolated, axis=1)
    # Equal ordinal pairing is a drift diagnostic, not event correspondence.
    rank = np.linspace(0, 1, 101)
    rank_drift = np.quantile(u, rank) - np.quantile(t, rank)
    bins = np.arange(0, max(t[-1], u[-1])+10, 10.)
    cg, cp = np.histogram(t, bins)[0], np.histogram(u, bins)[0]
    row = {'name': name, 'subject': subject.upper(), 'session': int(session),
           'gt_count': len(t), 'pred_count': len(u), 'count_bias_pct': 100*(len(u)/len(t)-1),
           'gt_last_s': t[-1], 'pred_last_s': u[-1], 'trajectory_horizon_s': horizon,
           'gt_outside_horizon': int(np.sum(t > horizon)),
           'step_length_m': tagged.foot_step.iloc[0],
           'timing_mae_s_at_200ms': np.abs(dt).mean(), 'timing_bias_s_at_200ms': dt.mean(),
           'xy_mae_m_at_200ms': dist.mean(), 'xy_median_m_at_200ms': np.median(dist),
           'xy_p95_m_at_200ms': np.quantile(dist, .95), 'azimuth_mae_deg_at_200ms': angle.mean(),
           'radius_mae_m_at_200ms': np.abs(np.linalg.norm(pq[b],axis=1)-np.linalg.norm(xy[a],axis=1)).mean(),
           'reconstruction_mae_m': reconstruction.mean(),
           'reconstruction_p95_m': np.quantile(reconstruction, .95),
           'gt_interval_mean_s': np.diff(t).mean(), 'pred_interval_mean_s': np.diff(u).mean(),
           'gt_interval_cv': np.diff(t).std()/np.diff(t).mean(),
           'pred_interval_cv': np.diff(u).std()/np.diff(u).mean(),
           'interval_wasserstein_s': stats.wasserstein_distance(np.diff(t), np.diff(u)),
           'count_10s_mae': np.abs(cg-cp).mean(), 'count_10s_correlation': corr(cg,cp),
           'rank_time_drift_p95_s': np.quantile(np.abs(rank_drift), .95)}
    return row, rows, matched, (t, xy, u, pq)


def features(y):
    _, _, z = signal.stft(y, FS, nperseg=NFFT, noverlap=NFFT-HOP, boundary=None, padded=False)
    power = np.abs(z)**2
    freq = np.fft.rfftfreq(NFFT, 1/FS)
    psd = power.mean(axis=1) + EPS
    norm = psd/psd.sum()
    envelope = np.sqrt(np.mean(np.lib.stride_tricks.sliding_window_view(y, NFFT)[::HOP]**2, axis=1)+EPS)
    # Fixed triangular mel filterbank; log-mel mean without gain term (DCT c0).
    mel = lambda x: 2595*np.log10(1+x/700)
    hz = lambda x: 700*(10**(x/2595)-1)
    edges = hz(np.linspace(mel(0), mel(FS/2), 42))
    filters = np.array([np.maximum(0,np.minimum((freq-edges[i])/(edges[i+1]-edges[i]),
                        (edges[i+2]-freq)/(edges[i+2]-edges[i+1]))) for i in range(40)])
    # einsum avoids spurious floating-point warnings in macOS Accelerate GEMM.
    melp = np.einsum('ij,jk->ik', filters, power, optimize=False) + EPS
    mfcc = dct(np.log(melp), type=2, axis=0, norm='ortho')[1:14].mean(axis=1)
    magnitude = np.sqrt(power)
    flux = np.maximum(0, np.diff(magnitude/(np.linalg.norm(magnitude,axis=0)+EPS),axis=1)).sum(axis=0)
    centered = envelope-envelope.mean()
    ac = signal.fftconvolve(centered, centered[::-1], mode='full')[len(centered)-1:]
    ac /= ac[0]+EPS
    lo, hi = int(.3*FS/HOP), int(1.*FS/HOP)
    cadence_period = (lo+np.argmax(ac[lo:hi]))*HOP/FS
    summary = {'rms': np.sqrt(np.mean(y*y)), 'crest_db': 20*np.log10(np.max(np.abs(y))/(np.sqrt(np.mean(y*y))+EPS)+EPS),
               'centroid_hz': np.dot(freq,norm), 'rolloff85_hz': freq[np.searchsorted(np.cumsum(norm),.85)],
               'flatness': np.exp(np.mean(np.log(psd)))/psd.mean(),
               'low_0_250_fraction': norm[freq<250].sum(),
               'mid_250_2000_fraction': norm[(freq>=250)&(freq<2000)].sum(),
               'high_2000_8000_fraction': norm[freq>=2000].sum(),
               'envelope_cv': envelope.std()/envelope.mean(),
               'onset_flux_mean': flux.mean(), 'acf_period_s': cadence_period}
    return summary, norm, envelope, mfcc, power


def load_audio(name, offset):
    simulated, fs = sf.read(ROOT / f'workspace/afpild/sim/{name}/sim_mic0000.wav')
    assert fs == FS and simulated.ndim == 1 and np.isfinite(simulated).all()
    assert np.max(np.abs(simulated)) > 0
    path = ROOT / f'workspace/afpild/AFPILD_v1/{name[:3].upper()}/{name}_footstep_audio.wav'
    info = sf.info(path)
    assert info.samplerate == FS and info.channels == 4
    start = int(round(offset*FS))
    length = min(len(simulated), info.frames-start)
    real, _ = sf.read(path, start=start, frames=length)
    return simulated[:length], real


def audio(name, offset):
    simulated, real = load_audio(name, offset)
    ss, sp, se, sm, _ = features(simulated)
    rows, vectors = [], []
    for channel in range(4):
        rs, rp, re, rm, _ = features(real[:, channel])
        # Correlation is diagnostic only: different source recordings are not phase aligned.
        re, se0 = re[:len(se)], se[:len(re)]
        maxlag = round(.25*FS/HOP)
        lag_results = [(lag, corr(np.roll(se0,lag)[maxlag:-maxlag],re[maxlag:-maxlag])) for lag in range(-maxlag,maxlag+1)]
        lag, best = max(lag_results,key=lambda x:x[1])
        logdiff = 10*np.log10(sp+EPS)-10*np.log10(rp+EPS)
        rows.append({'name': name, 'subject': name[:3].upper(), 'session': int(name[-1]), 'channel': channel,
                     'duration_s': len(simulated)/FS, 'audio_origin_s': offset,
                     'spectral_cosine': np.dot(sp,rp)/(np.linalg.norm(sp)*np.linalg.norm(rp)),
                     'spectral_js_distance': spatial.distance.jensenshannon(sp,rp),
                     'gain_invariant_log_spectral_rmse_db': np.sqrt(np.mean((logdiff-logdiff.mean())**2)),
                     'mfcc_1_13_distance': np.linalg.norm(sm-rm),
                     'envelope_correlation': corr(se0,re), 'best_envelope_correlation_250ms': best,
                     'best_envelope_lag_s': lag*HOP/FS,
                     'waveform_correlation': corr(simulated,real[:,channel]),
                     **{f'sim_{k}': v for k,v in ss.items()}, **{f'real_{k}': v for k,v in rs.items()}})
        vectors.append(rp)
    return rows, sp, np.mean(vectors,axis=0), sm


def final_tables():
    event = pd.read_csv(OUT/'event_metrics.csv')
    scores = pd.read_csv(OUT/'event_detection.csv')
    aud = pd.read_csv(OUT/'audio_metrics.csv')
    subject_means = event.groupby('subject').mean(numeric_only=True)
    rng = np.random.default_rng(20261111)
    ci = {}
    for key in ['count_bias_pct','xy_mae_m_at_200ms','reconstruction_mae_m','interval_wasserstein_s']:
        values = subject_means[key].to_numpy()
        boot = values[rng.integers(0,len(values),(5000,len(values)))].mean(axis=1)
        ci[key] = {'subject_mean': values.mean(), 'ci95': np.quantile(boot,[.025,.975]).tolist()}
    subject_means.to_csv(OUT/'event_by_subject.csv')
    event.groupby('session').mean(numeric_only=True).to_csv(OUT/'event_by_session.csv')
    aud.groupby('session').mean(numeric_only=True).to_csv(OUT/'audio_by_session.csv')
    grouped = scores.fillna({'space_tolerance_m':-1}).groupby(['time_tolerance_s','space_tolerance_m'])
    aggregate = grouped[['gt','pred','tp']].sum()
    aggregate['precision'] = aggregate.tp/aggregate.pred
    aggregate['recall'] = aggregate.tp/aggregate['gt']
    aggregate['f1'] = 2*aggregate.tp/(aggregate['gt']+aggregate.pred)
    aggregate.to_csv(OUT/'event_detection_summary.csv')
    npz = [np.load(p) for p in sorted((OUT/'cache').glob('*.npz'))]
    sp = np.array([x['sim_spectrum'] for x in npz]); rp = np.array([x['real_spectrum'] for x in npz])
    cosine = 1-spatial.distance.cdist(sp,rp,metric='cosine')
    # Same recording vs other subjects (exclude all same-subject sessions).
    names = sorted(event.name.tolist())
    mask = np.array([[a[:3]!=b[:3] for b in names] for a in names])
    matched = np.diag(cosine)
    null = np.array([cosine[i,mask[i]].mean() for i in range(len(names))])
    baseline = pd.DataFrame({'name':names,'matched_spectral_cosine_4ch_mean_psd':matched,
                             'other_subject_mean_cosine':null,'paired_advantage':matched-null})
    baseline.to_csv(OUT/'audio_mismatched_baseline.csv',index=False)
    summary = {'sessions':len(event),'gt_events':int(event.gt_count.sum()),'pred_events':int(event.pred_count.sum()),
               'event_means':event.mean(numeric_only=True).to_dict(), 'audio_means':aud.mean(numeric_only=True).to_dict(),
               'subject_bootstrap':ci,'spectral_matched_mean':matched.mean(),
               'spectral_other_subject_mean':null.mean(), 'matched_advantage':(matched-null).mean(),
               'bootstrap_unit':'subject; 40 independent clusters, 4 sessions each',
               'denoising':False,'audio_channel_policy':'sim mono versus each raw recording channel separately'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    return event, scores, aud, summary


def figures(event, scores, aud):
    fig, ax = plt.subplots(2,3,figsize=(14,8))
    ax[0,0].scatter(event.gt_count,event.pred_count,s=12,alpha=.6)
    ax[0,0].plot([200,700],[200,700],'k--',lw=1)
    ax[0,0].set(xlabel='Annotated event count',ylabel='Simulated event count')
    plain = scores[scores.space_tolerance_m.isna()]
    for ses, group in plain.groupby('session'):
        m=group.groupby('time_tolerance_s').f1.mean()
        ax[0,1].plot(m.index*1000,m.values,marker='o',label=f'Session {ses}')
    ax[0,1].set(xlabel='Time tolerance (ms)',ylabel='Mean F1'); ax[0,1].legend(fontsize=8)
    ax[0,2].hist(event.reconstruction_mae_m,bins=25)
    ax[0,2].set(xlabel='Interpolated trajectory error (m)',ylabel='Recordings')
    ax[1,0].scatter(event.gt_interval_cv,event.pred_interval_cv,s=12,alpha=.6)
    ax[1,0].set(xlabel='Annotated interval CV',ylabel='Simulated interval CV')
    ax[1,1].hist(aud.groupby('name').spectral_cosine.mean(),bins=25)
    ax[1,1].set(xlabel='Spectral cosine (mean over 4 channels)',ylabel='Recordings')
    ax[1,2].hist(aud.groupby('name').envelope_correlation.mean(),bins=25)
    ax[1,2].set(xlabel='Synchronized RMS envelope correlation',ylabel='Recordings')
    fig.tight_layout(); fig.savefig(OUT/'overview.png',dpi=160); plt.close(fig)
    ranking=aud.groupby('name').spectral_cosine.mean().sort_values()
    examples=[ranking.index[0],ranking.index[len(ranking)//2],ranking.index[-1]]
    for name in examples:
        offset=aud[aud.name==name].audio_origin_s.iloc[0]
        sim,real=load_audio(name,offset)
        _,sp,se,_,ps=features(sim); _,rp,re,_,pr=features(real[:,0])
        er,_,_,(t,xy,u,pq)=events(name,None)
        fig,ax=plt.subplots(3,2,figsize=(14,10))
        ax[0,0].plot(xy[:,0],xy[:,1],'.-',ms=2,lw=.4,label='Annotation')
        ax[0,0].plot(pq[:,0],pq[:,1],'.',ms=1,label='Simulation'); ax[0,0].axis('equal'); ax[0,0].legend()
        ax[0,0].set(xlabel='X (m)',ylabel='Y (m)')
        ax[0,1].eventplot([t[t<10],u[u<10]],colors=['C0','C1']); ax[0,1].set(xlabel='Time from first annotation (s)',yticks=[0,1],yticklabels=['Annotation','Simulation'])
        times=np.arange(min(len(se),len(re)))*HOP/FS
        ax[1,0].plot(times,se[:len(times)]/np.mean(se),lw=.6,label='Simulation')
        ax[1,0].plot(times,re[:len(times)]/np.mean(re),lw=.6,label='Real ch0'); ax[1,0].set(xlim=(0,10),xlabel='Time (s)',ylabel='RMS / mean RMS'); ax[1,0].legend()
        freq=np.fft.rfftfreq(NFFT,1/FS)
        ax[1,1].plot(freq,10*np.log10(sp+EPS),label='Simulation'); ax[1,1].plot(freq,10*np.log10(rp+EPS),label='Real ch0'); ax[1,1].set(xlabel='Frequency (Hz)',ylabel='Normalized PSD (dB)'); ax[1,1].legend()
        for a,p,title in [(ax[2,0],ps,'Simulation'),(ax[2,1],pr,'Real ch0')]:
            db=10*np.log10(p/np.max(p)+EPS)
            a.imshow(db,origin='lower',aspect='auto',extent=[0,p.shape[1]*HOP/FS,0,8000],vmin=-70,vmax=0,cmap='magma')
            a.set(xlim=(0,10),title=title,xlabel='Time (s)',ylabel='Frequency (Hz)')
        fig.suptitle(f'{name}: raw audio and event reconstruction'); fig.tight_layout(); fig.savefig(OUT/f'comparison_{name}.png',dpi=150); plt.close(fig)
        # Equal RMS preview clips: silence between real then synthetic, limit peaks jointly.
        duration=10*FS
        clip1,clip2=real[:duration,0].copy(),sim[:duration].copy()
        clip1/=np.sqrt(np.mean(clip1**2))+EPS; clip2/=np.sqrt(np.mean(clip2**2))+EPS
        joined=np.concatenate([clip1,np.zeros(FS),clip2]); joined*=.95/np.max(np.abs(joined))
        sf.write(OUT/f'listen_{name}_real_then_sim.wav',joined,FS,subtype='PCM_16')
    (OUT/'examples.json').write_text(json.dumps(examples))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--finalize',action='store_true'); args=parser.parse_args()
    os.chdir(ROOT); OUT.mkdir(exist_ok=True); (OUT/'cache').mkdir(exist_ok=True)
    manifest=pd.read_csv('data/gis/afpild/crowd/afpild_footstep_events/session_manifest.csv')
    for r in manifest.itertuples():
        name=f'{r.subject_id.lower()}_{r.session}'
        folder=ROOT/f'workspace/afpild/sim/{name}'
        cache=OUT/f'cache/{name}.npz'
        if not (folder/'run_manifest.json').exists() or cache.exists():
            continue
        row, detection, pairs, _=events(name,manifest)
        audio_rows, sp, rp, mfcc=audio(name,r.zero_origin_audio_time_s)
        (OUT/f'cache/{name}.json').write_text(json.dumps({'event':row,'detection':detection,'audio':audio_rows}))
        pairs.to_csv(OUT/f'cache/{name}_matches.csv',index=False)
        # Write the completion marker last so an interrupted run is resumable.
        np.savez_compressed(cache,sim_spectrum=sp,real_spectrum=rp,mfcc=mfcc)
        print(f'EVALUATED {name}',flush=True)
    cached=[json.loads(p.read_text()) for p in sorted((OUT/'cache').glob('s*.json'))]
    if not cached:
        return
    pd.DataFrame([c['event'] for c in cached]).to_csv(OUT/'event_metrics.csv',index=False)
    pd.DataFrame([r for c in cached for r in c['detection']]).to_csv(OUT/'event_detection.csv',index=False)
    pd.DataFrame([r for c in cached for r in c['audio']]).to_csv(OUT/'audio_metrics.csv',index=False)
    if args.finalize:
        assert len(cached)==160, f'Only {len(cached)} of 160 complete'
        event,scores,aud,summary=final_tables()
        figures(event,scores,aud)
        print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
