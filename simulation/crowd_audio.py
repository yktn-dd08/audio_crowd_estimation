import argparse
from simulation.util import *
logger = get_logger('simulation.crowd_audio')


def audio_crowd_simulation(crowd_csv, room_shp, output_folder, mic_shp=None, snr=None, time_unit=10.0,
                           distance_list=None, grid_shp=None, x_idx_range=None, y_idx_range=None,
                           height=3.0, max_order=0):
    signal_info = {}
    if distance_list is None:
        distance_list = [1, 10, 20, 30, 40, 50]
    signal_info['distance_list'] = distance_list

    crowd_list = Crowd.csv_to_crowd_list(crowd_csv)
    st_time, ed_time = min([c.start_time for c in crowd_list]), max([c.start_time for c in crowd_list])
    signal_info['simulation_time'] = [st_time, ed_time]
    logger.info(f'simulation time: {st_time} - {ed_time}')

    # print('Reading Simulation room shapefile.')
    crowd_sim = CrowdSim.from_shp(room_shp, height, max_order)
    signal_info['shp_file'] = room_shp
    signal_info['corner'] = []
    for i, corner in enumerate(crowd_sim.room_info['corners'].T):
        logger.debug(f'corner {i} - {corner.tolist()}')
        signal_info['corner'].append(corner.tolist())
    logger.info('sampling rate - ' + str(crowd_sim.room_info['fs']))
    logger.info('max_order - ' + str(crowd_sim.room_info['max_order']))
    signal_info['sampling_rate'] = crowd_sim.room_info['fs']
    signal_info['max_order'] = crowd_sim.room_info['max_order']
    crowd_sim.set_crowd(crowd_list)
    room_center = crowd_sim.room_info['corners'].mean(axis=1)

    if mic_shp is None:
        id_list = [0]
        mic_location = np.array([[room_center[0], room_center[1], 0.05]])

    else:
        df = gpd.read_file(mic_shp)
        id_list = [dr['id'] for _, dr in df.iterrows() if crowd_sim.room_polygon.contains(dr['geometry'])]
        mic_location = np.array([[dr['geometry'].x, dr['geometry'].y, dr['height']]
                                 for _, dr in df.iterrows() if crowd_sim.room_polygon.contains(dr['geometry'])])
    crowd_sim.set_microphone(mic_location)
    signal_info['microphone'] = mic_location.tolist()
    for i, mic in enumerate(mic_location.tolist()):
        logger.debug(f'microphone {i} - {mic}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    signals = crowd_sim.simulation(multi_process=True, time_unit=time_unit)

    # add noise
    max_signal = np.abs(signals).max()

    # 信号値はmaxで割って保存するが、これを残しておかないと複数ケースのシミュレーションを比較できない(スケールがバラバラになる)
    signal_info |= {'signal_max': float(max_signal), 'info': {}}

    if snr is not None:
        sigma_n = crowd_sim.generate_noise_std(snr=snr)
        signals += np.random.normal(0.0, sigma_n, size=signals.shape)
        signal_info['noise_sigma'] = float(sigma_n)

    signals = signals / max_signal

    for s in range(len(signals)):
        wav_path = f'{output_folder}/sim_mic{id_list[s]:04}.wav'
        logger.info(f'save wav file - {wav_path}')
        wavfile.write(wav_path, SR, signals[s])
        signal_info['info'][f'ch{s}'] = {'wav_path': wav_path}

    # save crowd with each range
    if grid_shp is None:
        crowd_path = f'{output_folder}/crowd.csv'
        logger.info(f'save crowd csv - {crowd_path}')
        crowd_sim.crowd_density_to_csv(crowd_path)
        signal_info['crowd_path'] = crowd_path
        for s in range(len(signals)):
            each_crowd = f'{output_folder}/crowd_mic{id_list[s]:04}.csv'
            logger.info(f'save crowd from mic{id_list[s]} - {each_crowd}')
            crowd_sim.crowd_density_from_each_mic(mic_index=id_list[s], csv_name=each_crowd,
                                                  distance_list=distance_list)
            signal_info['info'][f'ch{s}'] |= {'each_crowd': each_crowd}
    else:
        crowd_path = f'{output_folder}/crowd_grid.csv'
        crowd_sim.crowd_dendity_from_grid_shp(grid_shp=grid_shp, csv_name=crowd_path,
                                              x_idx_range=x_idx_range, y_idx_range=y_idx_range)
        signal_info['crowd_path'] = crowd_path

    with open(f'{output_folder}/signal_info.json', 'w') as f:
        json.dump(signal_info, f, indent=4)
    return


def footstep_simulation_task(config_path):
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    task_list = cfg['task_list']
    # Get common parameters
    common_roi_shp = cfg.get('roi_shp', None)
    common_mic_shp = cfg.get('mic_shp', None)
    common_grid_shp = cfg.get('grid_shp', None)
    common_snr = cfg.get('snr', None)
    common_distance_list = cfg.get('distance_list', None)
    common_x_idx_range = cfg.get('x_idx_range', None)
    common_y_idx_range = cfg.get('y_idx_range', None)
    common_height = cfg.get('height', 3.0)
    common_max_order = cfg.get('max_order', 0)

    for task_name, task_cfg in task_list.items():
        logger.info(f'[task name]: {task_name}')
        audio_crowd_simulation(
            crowd_csv=task_cfg['crowd_csv'],
            output_folder=task_cfg['output_folder'],
            mic_shp=task_cfg.get('params', {}).get('mic_shp', common_mic_shp),
            room_shp=task_cfg.get('params', {}).get('roi_shp', common_roi_shp),
            grid_shp=task_cfg.get('params', {}).get('grid_shp', common_grid_shp),
            snr=task_cfg.get('params', {}).get('snr', common_snr),
            distance_list=task_cfg.get('params', {}).get('distance_list', common_distance_list),
            time_unit=task_cfg.get('params', {}).get('time_unit', 10.0),
            x_idx_range=task_cfg.get('params', {}).get('x_idx_range', common_x_idx_range),
            y_idx_range=task_cfg.get('params', {}).get('y_idx_range', common_y_idx_range),
            height=task_cfg.get('params', {}).get('height', common_height),
            max_order=task_cfg.get('params', {}).get('max_order', common_max_order)
        )
    logger.info(f'{len(task_list)} tasks are completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--option', type=str, required=True,
                        choices=['footstep'], help='simulation option')
    parser.add_argument('-i', '--input-config', type=str, required=True,
                        help='input simulation config file (JSON)')
    args = parser.parse_args()
    if args.option == 'footstep':
        footstep_simulation_task(args.input_config)
