"""Reproducible AFPILD tagging and simulation; run from repository root."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse
import hashlib
import glob
import importlib.metadata
import json
import random
import sys
from pathlib import Path
import numpy as np
from preprocess.footstep_sound import add_foot_tag
from simulation.crowd_audio import audio_crowd_simulation

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=160)
    args = parser.parse_args()
    os.chdir(ROOT)
    evaluation = Path('workspace/afpild/evaluation')
    evaluation.mkdir(parents=True, exist_ok=True)
    environment = evaluation / 'environment.json'
    if not environment.exists():
        versions = {key: importlib.metadata.version(key) for key in
                    ('numpy', 'scipy', 'pandas', 'librosa', 'soundfile',
                     'pyroomacoustics', 'geopandas', 'joblib', 'matplotlib', 'torch')}
        versions['python'] = sys.version
        sound_paths = {Path(folder).name: glob.glob(folder + '/*.wav')
                       for folder in glob.glob('data/ambient_sound/footstep/*')
                       if Path(folder).is_dir()}
        environment.write_text(json.dumps({'versions': versions,
                                          'sound_index_paths': sound_paths}, indent=2))
    files = sorted(Path('data/gis/afpild/crowd/org').glob('*.csv'))
    assert len(files) == 160
    assert {p.name[:3] for p in files} == {f's{i:02}' for i in range(1, 41)}
    room = Path('data/gis/afpild/roi/afpild_roi01.shp')
    records = []
    for i, source in enumerate(files[:args.limit]):
        name = source.stem.replace('_crowd', '')
        subject = name.split('_')[0].upper()
        template = Path(f'workspace/afpild/foottag/foottag_template_test_{subject}.json')
        tagged = Path('workspace/data/gis/afpild/crowd/asplaht/test') / source.name
        output = Path('workspace/afpild/sim') / name
        seed = 20261111 + i
        signature = {'source_sha256': sha(source), 'template_sha256': sha(template),
                     'config_sha256': sha('footstep_config.json'), 'room_sha256': sha(room),
                     'util_sha256': sha('simulation/util.py'),
                     'crowd_audio_sha256': sha('simulation/crowd_audio.py'),
                     'max_order': 3, 'seed': seed}
        marker = output / 'run_manifest.json'
        complete = marker.exists() and json.loads(marker.read_text()) == signature
        if not complete:
            add_foot_tag(str(source), str(tagged), 'footstep_config.json', str(template))
            random.seed(seed)
            np.random.seed(seed)
            audio_crowd_simulation(str(tagged), str(room), str(output), max_order=3)
            marker.write_text(json.dumps(signature, indent=2))
        records.append({'session': name, 'source': str(source), 'template': str(template),
                        'tagged': str(tagged), 'output': str(output), **signature})
        Path('workspace/afpild/evaluation/run_manifest.json').write_text(json.dumps(records, indent=2))
        print(f'COMPLETE {i+1}/{len(files[:args.limit])}: {name}', flush=True)

if __name__ == '__main__':
    main()
