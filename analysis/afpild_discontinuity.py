"""Read-only diagnosis of apparent gaps in generated AFPILD footsteps."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
 os.environ[k]='1'
os.environ['MPLBACKEND']='Agg'
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
from simulation.util import CrowdSim
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'workspace/afpild/evaluation/discontinuity'


def spans(mask,rate):
 edges=np.diff(np.r_[False,mask,False].astype(int))
 return [(float(a/rate),float(b/rate)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]


def main():
 os.chdir(ROOT);OUT.mkdir(exist_ok=True)
 name='s06_3';folder=ROOT/f'workspace/afpild/sim/{name}'
 info=json.loads((folder/'signal_info.json').read_text())
 js=json.loads((folder/'footstep_info.json').read_text());ev=js['0']['time_series'];tag=js['0']['foot_tag']
 paths=json.loads((ROOT/'workspace/afpild/evaluation/environment.json').read_text())['sound_index_paths'][tag]
 sound=[]
 for p in paths:
  y,fs=sf.read(p);y=y.mean(axis=1) if y.ndim>1 else y
  if fs!=16000:
   import librosa
   y=librosa.resample(y,orig_sr=fs,target_sr=16000);fs=16000
  active=np.flatnonzero(np.abs(y)>np.max(np.abs(y))*1e-3)
  sound.append({'file':p,'duration_s':len(y)/fs,'active_span_minus60db_s':(active[-1]-active[0]+1)/fs,
                'start_abs_over_peak':float(abs(y[0])/np.max(np.abs(y))),
                'end_abs_over_peak':float(abs(y[-1])/np.max(np.abs(y)))})
 sim=CrowdSim.from_shp('data/gis/afpild/roi/afpild_roi01.shp',height=3.,max_order=3)
 sim.set_microphone(np.array(info['microphone']))
 point=[ev[0]['x'],ev[0]['y'],ev[0]['z']]
 rir_info=[]
 for order in [0,3,10]:
  sim.room_info['max_order']=order;room=sim.create_room();room.add_source(point,signal=np.array([1.]))
  room.compute_rir();h=room.rir[0][0]
  nz=np.flatnonzero(abs(h)>np.max(abs(h))*1e-6)
  energy=np.cumsum(h*h);energy/=energy[-1]
  rir_info.append({'order':order,'buffer_length_s':len(h)/16000,'first_nonzero_s':int(nz[0])/16000,
                   'last_nonzero_s':int(nz[-1])/16000,'support_duration_s':int(nz[-1]-nz[0]+1)/16000,
                   'energy_99_time_s':float(np.searchsorted(energy,.99)/16000),
                   'absorption_by_wall':[np.asarray(w.absorption).tolist() for w in room.walls],
                   'ray_tracing_enabled':room.simulator_state['rt_needed']})
  np.save(OUT/f'rir_order{order}.npy',h)
 y,fs=sf.read(folder/'sim_mic0000.wav')
 window=80;trim=y[:len(y)//window*window];env=np.sqrt((trim.reshape(-1,window)**2).mean(axis=1))
 silent=env<env.max()*1e-3;gaps=spans(silent,fs/window)
 long=[(a,b) for a,b in gaps if b-a>=.02]
 duration=np.array([b-a for a,b in long])
 # Re-render 12 seconds in a single room, crossing the old 10-second batch boundary.
 sim.room_info['max_order']=3;room=sim.create_room()
 for foot in ev:
  if foot['t']>=12: continue
  room.add_source([foot['x'],foot['y'],foot['z']],signal=sim.foot_sound.get_wav(tag,foot['sound_index']),delay=foot['t'])
 room.simulate();fresh=room.mic_array.signals[0]
 count=min(int(11.5*fs),len(fresh),len(y));expected=y[:count]*info['signal_max']
 err=np.max(np.abs(fresh[:count]-expected))
 result={'name':name,'foot_tag':tag,'source_fragments':sound,
         'footstep_interval_median_s':float(np.median(np.diff([e['t'] for e in ev]))),
         'rir':rir_info,'near_silence_definition':'5ms RMS more than 60dB below maximum 5ms RMS',
         'near_silence_fraction':float(silent.mean()),'gaps_at_least_20ms':len(long),
         'gap_duration_median_s':float(np.median(duration)),'gap_duration_p95_s':float(np.quantile(duration,.95)),
         'first_10s_gaps':[(a,b) for a,b in long if a<10],
         'single_room_vs_saved_first11_5s_max_abs_error':float(err),
         'single_room_vs_saved_relative_max_error':float(err/(np.max(np.abs(expected))+1e-20))}
 (OUT/'diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
