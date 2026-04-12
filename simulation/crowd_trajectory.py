import json
import argparse
from simulation.util import *

logger = get_logger('simulation.crowd_trajectory')


def execute_json(config_path):
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    option = cfg['option']
    common_param = cfg['param']
    if option == 'crowd_trajectory':
        for task, param in cfg['task_list'].items():
            logger.info(f"Executing task: {task}")
            output_path = param['output_csv']
            crowd_trj = CrowdTrajectory()
            crowd_trj.set_crowd_trajectory(
                person_num=param.get('person_num', common_param.get('person_num', 100)),
                start_time=param.get('start_time', common_param.get('start_time', 0)),
                end_time=param.get('end_time', common_param.get('end_time', 100)),
                v=param.get('v', common_param.get('v', 1.5)),
                v_sigma=param.get('v_sigma', common_param.get('v_sigma', 0.3)),
                exist_time=param.get('exist_time', common_param.get('exist_time', 60)),
                exist_time_sigma=param.get('exist_time_sigma', common_param.get('exist_time_sigma', 0.0)),
                dir_division=param.get('dir_division', common_param.get('dir_division', 8)),
                datetime_str=param.get('datetime_str', common_param.get('datetime_str', '2024-01-01 00:00:00'))
            )
            crowd_trj.to_csv(output_path)
            logger.info(f"Saved crowd trajectory to {output_path}")


def test():
    crowd_trj = CrowdTrajectory()
    crowd_trj.set_crowd_trajectory(
        person_num=100,
        start_time=0,
        end_time=100,
        v=1.5,
        v_sigma=0.3,
        exist_time=60,
        exist_time_sigma=0.0,
        dir_division=8,
        datetime_str='2024-01-01 00:00:00'
    )
    crowd_trj.to_csv('crowd_trajectory.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    args = parser.parse_args()
    execute_json(args.config)
