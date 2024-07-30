# audio_crowd_estimation
## installation
- for mac
```commandline
python -m venv venv
source venv/bin/activate
pip install -r requirement-cpu.txt
```

- for windows with gpu
```commandline
python -m venv venv
venv¥Script¥activate
pip install -r requirement.txt
```

## data preparation
### speech audio
You can download speech audio dataset as belows:
```pycon
from simulation.util import Data
Data.get_cmu_arctic()
```

### ambient sound
You can download ambient sound dataset as following URL.
- https://github.com/karolpiczak/ESC-50

### DISCO dataset
