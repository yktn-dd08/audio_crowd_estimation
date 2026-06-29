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
            roi_path = param.get('roi_shp', common_param.get('roi_shp', None))
            roi_shp = gpd.read_file(roi_path).geometry[0] if roi_path else None
            crowd_trj = CrowdTrajectory(room_polygon=roi_shp)
            crowd_trj.set_crowd_trajectory(
                person_num=param.get('person_num', common_param.get('person_num', 100)),
                start_time=param.get('start_time', common_param.get('start_time', 0)),
                end_time=param.get('end_time', common_param.get('end_time', 100)),
                v=param.get('v', common_param.get('v', 1.5)),
                v_sigma=param.get('v_sigma', common_param.get('v_sigma', 0.3)),
                dir_division=param.get('dir_division', common_param.get('dir_division', 8)),
                datetime_str=param.get('datetime_str', common_param.get('datetime_str', '2024-01-01 00:00:00'))
            )
            crowd_trj.to_csv(output_path)
            logger.info(f"Saved crowd trajectory to {output_path}")
    elif option == 'social_force':
        for task, param in cfg['task_list'].items():
            logger.info(f"Executing task: {task}")
            output_path = param['output_csv']
            roi_path = param.get('roi_shp', common_param.get('roi_shp', None))
            roi_shp = gpd.read_file(roi_path).geometry[0] if roi_path else None
            wall_path = param.get('wall_shp', common_param.get('wall_shp', None))
            wall_shp = MultiPolygon(gpd.read_file(wall_path).geometry.tolist()) if wall_path else None
            crowd_trj = SocialForceSimulation(
                roi_polygon=roi_shp,
                wall=wall_shp,
                dt=param.get('dt', common_param.get('dt', 1.0)),
                desired_speed=param.get('desired_speed', common_param.get('desired_speed', 1.5)),
                velocity_noise_std=param.get('velocity_noise_std', common_param.get('velocity_noise_std', 0.3)),
                c_obs=param.get('c_obs', common_param.get('c_obs', 2000.0)),
                r_obs=param.get('r_obs', common_param.get('r_obs', 0.08)),
                c_wall=param.get('c_wall', common_param.get('c_wall', 2000.0)),
                r_wall=param.get('r_wall', common_param.get('r_wall', 0.08)),
            )
            crowd_trj.run(
                person_num=param.get('person_num', common_param.get('person_num', 100)),
                start_time=param.get('start_time', common_param.get('start_time', '2024-01-01 00:00:00')),
                simulation_time=param.get('simulation_time', common_param.get('simulation_time', 100)),
                goal_flag=param.get('goal_flag', common_param.get('goal_flag', False))
            )
            crowd_trj.to_csv(output_path)
            if 'output_mp4' in param:
                crowd_trj.create_video(param['output_mp4'], fps=10)
            logger.info(f'Saved crowd trajectory to {output_path}')



def test():
    crowd_trj = CrowdTrajectory()
    crowd_trj.set_crowd_trajectory_old(
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
