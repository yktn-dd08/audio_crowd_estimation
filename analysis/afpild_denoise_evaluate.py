"""Denoise all AFPILD recordings and compare against fixed simulations."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND']='Agg'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import spatial
from analysis.afpild_evaluate import ROOT, FS, HOP, EPS, features, corr
from tools.afpild_multiband import noise_frames, estimate_noise, denoise, transform, invert

OUT=ROOT/'workspace/afpild/evaluation/denoise'
AUDIO=ROOT/'workspace/afpild/AFPILD_v1_mbss'
MODE='quiet'


def comparisons(sp,se,sm,y):
    summary, p, e, m, _=features(y)
    maxlag=round(.25*FS/HOP)
    lag,best=max([(lag,corr(np.roll(se,lag)[maxlag:-maxlag],e[maxlag:-maxlag]))
                  for lag in range(-maxlag,maxlag+1)],key=lambda x:x[1])
    logdiff=10*np.log10(sp+EPS)-10*np.log10(p+EPS)
    row={'spectral_cosine':float(1-spatial.distance.cosine(sp,p)),
         'spectral_js_distance':float(spatial.distance.jensenshannon(sp,p)),
         'gain_invariant_log_spectral_rmse_db':float(np.sqrt(np.mean((logdiff-logdiff.mean())**2))),
         'mfcc_1_13_distance':float(np.linalg.norm(sm-m)),
         'envelope_correlation':corr(se,e), 'best_envelope_correlation_250ms':best,
         'best_envelope_lag_s':lag*HOP/FS,
         **{f'real_{k}':float(v) for k,v in summary.items()}}
    freq=np.fft.rfftfreq(1024,1/FS)
    for low,hi,label in [(0,250,'0_250'),(250,2000,'250_2000'),(2000,8001,'2000_8000'),(250,8001,'250_8000')]:
        mask=(freq>=low)&(freq<hi)
        row[f'band_cosine_{label}']=float(1-spatial.distance.cosine(sp[mask],p[mask]))
    return row,p,e


def process(r, raw_table):
    name=f'{r.subject_id.lower()}_{r.session}'
    cache=OUT/'cache'/f'{name}.json'
    if cache.exists(): return
    raw_path=ROOT/f'workspace/afpild/AFPILD_v1/{r.subject_id}/{name}_footstep_audio.wav'
    y,fs=sf.read(raw_path); assert fs==FS and y.shape[1]==4
    sim,fs=sf.read(ROOT/f'workspace/afpild/sim/{name}/sim_mic0000.wav'); assert fs==FS
    start=int(round(r.zero_origin_audio_time_s*FS)); length=min(len(sim),len(y)-start)
    ss,sp,se,sm,_=features(sim[:length])
    prefix=y[:int(3.1*FS)]
    indices, selection=noise_frames(prefix,r.zero_origin_audio_time_s,MODE)
    npsd=estimate_noise(prefix,indices)
    enhanced=np.zeros_like(y); rows=[]; spectra=[]; diagnostics=[]
    for ch in range(4):
        enhanced[:,ch],diag=denoise(y[:,ch],npsd[ch])
        clean=enhanced[start:start+length,ch]
        row,p,env=comparisons(sp,se,sm,clean)
        old=raw_table.loc[(raw_table.name==name)&(raw_table.channel==ch)].iloc[0]
        rows.append({'name':name,'subject':r.subject_id,'session':int(r.session),'channel':ch,
                     'duration_s':length/FS, 'audio_origin_s':r.zero_origin_audio_time_s,
                     **row, 'removed_energy_db':float(10*np.log10((np.mean(y[start:start+length,ch]**2)+EPS)/(np.mean(clean**2)+EPS))),
                     'raw_clean_waveform_correlation':corr(y[start:start+length,ch],clean),
                     'sim_centroid_hz':ss['centroid_hz'],
                     'envelope_correlation_delta':row['envelope_correlation']-old.envelope_correlation})
        diagnostics.append(diag); spectra.append(p)
    audio_path=AUDIO/r.subject_id/f'{name}_footstep_audio_mbss.wav'
    audio_path.parent.mkdir(parents=True,exist_ok=True)
    sf.write(audio_path,enhanced,FS,subtype='FLOAT')
    assert sf.info(audio_path).frames==len(y)
    # Fixed sensitivity set chosen without looking at similarity scores.
    sensitivity=[]
    if MODE == 'quiet' and r.subject_id in ('S01','S10','S20','S40'):
        idx,info=noise_frames(prefix,r.zero_origin_audio_time_s,'first3s')
        n2=estimate_noise(prefix,idx)
        # A second analysis isolates the noise estimate; nothing is tuned to simulations.
        for ch in range(4):
            alt,_=denoise(y[:,ch],n2[ch])
            row,_,_=comparisons(sp,se,sm,alt[start:start+length])
            sensitivity.append({'name':name,'subject':r.subject_id,'session':int(r.session),'channel':ch,**row})
    # Verify that unchanged raw metrics are numerically identical on the first session.
    raw_check={}
    if name=='s01_1':
        check,_,_=comparisons(sp,se,sm,y[start:start+length,0])
        original=raw_table.loc[(raw_table.name==name)&(raw_table.channel==0)].iloc[0]
        for key in ('spectral_cosine','mfcc_1_13_distance','envelope_correlation','gain_invariant_log_spectral_rmse_db'):
            raw_check[key]=abs(float(original[key])-check[key]); assert raw_check[key]<1e-9
    np.savez_compressed(OUT/'cache'/f'{name}.npz',sim_spectrum=sp,clean_spectrum=np.mean(spectra,axis=0),noise_power=npsd)
    cache.write_text(json.dumps({'rows':rows,'selection':selection,'diagnostics':diagnostics,
                                'sensitivity':sensitivity,'raw_recomputation_abs_difference':raw_check,
                                'raw_file_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                                'denoised_file':str(audio_path.relative_to(ROOT)),
                                'denoised_file_sha256':hashlib.sha256(audio_path.read_bytes()).hexdigest()},indent=2))
    print(f'DENOISED {name}',flush=True)


def main():
    global MODE, OUT, AUDIO
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int,default=160)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--noise-mode',choices=['quiet','first3s'],default='quiet')
    args=parser.parse_args()
    MODE=args.noise_mode
    if MODE == 'first3s':
        OUT=ROOT/'workspace/afpild/evaluation/denoise_first3s'
        AUDIO=ROOT/'workspace/afpild/AFPILD_v1_mbss_first3s'
    os.chdir(ROOT); (OUT/'cache').mkdir(parents=True,exist_ok=True)
    config={'method':'AFPILD-compatible multiband spectral subtraction; explicit Kamath & Loizou 2002 based settings',
            'exact_afpild_reproduction':False, 'sample_rate':16000,'window':'Hamming, 20ms', 'hop_ms':10,
            'nfft':512,'bands_hz':[0,2000,4000,6000,8000],'delta_by_band':[2.5,2.5,2.5,1.5],
            'alpha':'5 if SNR < -5; 4 - 0.15*SNR if -5 <= SNR <= 20; 1 if SNR > 20',
            'snr_definition':'10log10(sum(noisy smoothed power) / sum(estimated noise power)) per band and frame',
            'beta':.002,'floor':'negative/nonpositive estimates replaced with beta * smoothed noisy power',
            'temporal_magnitude_smoothing':[.09,.25,.32,.25,.09],
            'other_spectral_smoothing':'Not implemented; the original paper cites a separate prior smoother',
            'noise_selection':'lowest joint 4-channel energy 30% frames in [0.01, min(3, first accepted annotation - .2) - .01] sec; minimum 20 frames',
            'noise_model':'fixed per-channel mean power of selected frames; no speech VAD/no online adaptation',
            'sensitivity':'all 4 channels of all 4 sessions for S01,S10,S20,S40, noise mean over first 3 sec without quiet selection',
            'phase':'original per-channel phase', 'output':'4-channel float32 WAV; same start, frames, fs; no peak/gain normalization',
            'evaluation':'same crop and features as original evaluation; simulated audio unchanged',
            'sources':['https://www.researchgate.net/publication/376251270_AFPILD_Acoustic_footstep_dataset_collected_using_one_microphone_array_and_LiDAR_sensor_for_person_identification_and_localization',
                       'https://doczz.net/doc/7989100/a-multi-band-spectral-subtraction'],
            'implementation_sha256':hashlib.sha256((ROOT/'tools/afpild_multiband.py').read_bytes()).hexdigest()}
    if MODE == 'first3s':
        config['noise_selection']='all complete frames in the first 3 seconds; mean power per channel'
        config['sensitivity']='Full 160-recording quiet-frame comparison stored separately in evaluation/denoise'
    conf=OUT/'method.json'
    if conf.exists(): assert json.loads(conf.read_text())==config,'Settings changed; use a new output folder'
    else: conf.write_text(json.dumps(config,indent=2))
    manifest=pd.read_csv(ROOT/'data/gis/afpild/crowd/afpild_footstep_events/session_manifest.csv')
    raw=pd.read_csv(ROOT/'workspace/afpild/evaluation/audio_metrics.csv')
    for r in manifest.iloc[args.start:args.limit].itertuples(): process(r,raw)
    cached=[json.loads(p.read_text()) for p in sorted((OUT/'cache').glob('s*.json'))]
    pd.DataFrame([r for c in cached for r in c['rows']]).to_csv(OUT/'audio_metrics_after.csv',index=False)
    pd.DataFrame([r for c in cached for r in c['sensitivity']]).to_csv(OUT/'sensitivity_first3s.csv',index=False)
    print(f'COMPLETE {len(cached)}/160',flush=True)

if __name__=='__main__': main()
