"""Validate saved MBSS WAVs, unchanged inputs, OLA, and spectral floor."""
from pathlib import Path
import hashlib
import json
import numpy as np
import soundfile as sf
from tools.afpild_multiband import transform,invert,subtract,NFFT,BETA
from analysis.afpild_evaluate import ROOT


def main():
    rng=np.random.default_rng(1);x=rng.normal(0,.1,16037);_,_,z=transform(x)
    error=float(np.max(np.abs(x-invert(z,len(x)))));assert error<1e-12
    z=np.full((NFFT//2+1,11),.01,dtype=complex)
    enhanced,_=subtract(z,np.full(NFFT//2+1,.01));assert np.allclose(np.abs(enhanced)**2,BETA*np.abs(z)**2)
    validation=[]
    for mode,folder in [('quiet','denoise'),('first3s','denoise_first3s')]:
        out=ROOT/f'workspace/afpild/evaluation/{folder}'
        cached=sorted((out/'cache').glob('s*.json'));assert len(cached)==160,(mode,len(cached))
        examples=[]
        for f in cached:
            name=f.stem;metadata=json.loads(f.read_text());audio=ROOT/metadata['denoised_file']
            raw=ROOT/f'workspace/afpild/AFPILD_v1/{name[:3].upper()}/{name}_footstep_audio.wav'
            assert hashlib.sha256(raw.read_bytes()).hexdigest()==metadata['raw_file_sha256']
            assert hashlib.sha256(audio.read_bytes()).hexdigest()==metadata['denoised_file_sha256']
            ri,di=sf.info(raw),sf.info(audio)
            assert ri.samplerate==di.samplerate==16000 and ri.channels==di.channels==4 and ri.frames==di.frames
            assert di.subtype=='FLOAT'
            peak=0.
            with sf.SoundFile(audio) as stream:
                for block in stream.blocks(blocksize=262144):
                    assert np.isfinite(block).all()
                    peak=max(peak,float(np.max(np.abs(block))))
            assert peak>0
            examples.append({'name':name,'peak':peak,'frames':di.frames})
        result={'recordings':160,'channels':4,'sample_rate':16000,'same_frame_counts':True,
                'all_saved_samples_finite':True,'all_saved_outputs_nonzero':True,'float32_no_normalization':True,
                'original_sha256_unchanged':True,'output_sha256_matches':True,
                'stft_istft_reconstruction_max_abs_error':error,'nonpositive_power_floor':'passed',
                'max_output_peak':max(r['peak'] for r in examples),'raw_recomputation_abs_difference':json.loads(cached[0].read_text())['raw_recomputation_abs_difference']}
        (out/'numerical_checks.json').write_text(json.dumps(result,indent=2));validation.append({mode:result})
    print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
