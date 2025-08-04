# audio_crowd_estimation
## Installation
- for mac
```commandline
python -m venv venv
source venv/bin/activate
pip install -r requirement-cpu.txt
```

- for windows with gpu
```commandline
python -m venv venv
venv\Script\activate
pip install -r requirement.txt
```

## Data preparation
### speech audio
You can download speech audio dataset as belows:
```pycon
from simulation.util import Data
Data.get_cmu_arctic()
```
### DISCO dataset

### ESC-50: ambient sound
You can download ambient sound dataset as following URL.
- https://github.com/karolpiczak/ESC-50

### folder structure for dataset
![folder_structure](./img/folder_structure.png)

Before audio crowd simulation, you need to parse some ambient sound to each segment.

```commandline
python -m preprocess.footstep_sound -opt segment -i .\data\ambient_sound\audio -o .\data\ambient_sound\footstep
```

## How to execute
### Create people flow trajectories from PostgreSQL + PostGIS
Before executing following command line, please check your PostgreSQL environments.

```commandline
python -m database.crowd_data -opt export -oc data/gis/marunouchi/crowd/roi1_crowd0806_09.csv -dn marunouchi -lo marunouchi -st 2020-08-06 09:00:00 -et 2020-08-06 10:00:00 -rs data/gis/marunouchi/roi/marunouchi_roi1.shp
```

### Add foottag information to trajectory data
```commandline
python -m preprocess.footstep_sound -opt foot_tag -i data/gis/marunouchi/crowd/org/roi1_crowd0806_09.csv -o data/gis/marunouchi/crowd/asphalt/train/roi1_crowd0806_09.csv -f template/foottag/foottag_template_train.json
```
### Audio simulation (footstep) from people flow trajectories

```commandline
python -m simulation.util -c ./workspace/gis/test_f1.csv -s ./workspace/gis/room/marunouchi.shp -o ./workspace/sim/f1
```

### Audio crowd estimation

```commandline
python -m analysis.model_training -opt cv -i ./workspace/sim/f1 -o ./workspace/sim/f1/model2 -d sim -e 10
```