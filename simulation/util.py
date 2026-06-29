import glob
import json
import os
import math
import random
import argparse
from datetime import datetime, timedelta
from matplotlib.animation import FuncAnimation, FFMpegWriter

import librosa
import pandas as pd
import torch
import shapely
import wave as wave
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.io import wavfile
from shapely import wkt
from shapely.geometry import LineString, Point, Polygon, MultiPoint, MultiPolygon
# from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import split, nearest_points, orient
from joblib import Parallel, delayed
from tqdm import tqdm
from common.logger import get_logger
from database.common import *
from tools.visualize import change_speed_ffmpeg


INPUT_DIR = './data/speech'
OUTPUT_DIR = './output'

CH = 'channels'
FS = 'sampling_rate'
SR = 16000
ROOM_SIZE = np.array([10.0, 10.0, 3.0])
MIC_INTERVAL = 0.02
MAX_SPEED = 2.0
MAX_ACC = 1.0

# logging.basicConfig(
#     level=logging.DEBUG,
#     format='Line %(lineno)d: %(name)s: %(message)s',
#     # datefmt='[%Y-%m-%d ]',
#     handlers=[RichHandler(markup=True, rich_tracebacks=True)]
# )
# logger = logging.getLogger('simulation.util')
# for mod in ['numba', 'matplotlib']:
#     logging.getLogger(mod).setLevel(logging.CRITICAL)
logger = get_logger('simulation.util')


class Data:
    CMU = 'cmu_arctic'

    @staticmethod
    def get_cmu_arctic(console_out=False):
        path = f'{INPUT_DIR}/{Data.CMU}'
        if not os.path.exists(path):
            os.makedirs(path)
            logger.debug(msg=f'{path} created.')
            if console_out:
                print(f'Created folder: {path}')
        pra.datasets.CMUArcticCorpus(basedir=path, download=True)


class SimOld:
    def __init__(self, src_num, mic_num=2, sampling_rate=16000, indoor=True, bins=1024):
        self.mic_num = mic_num
        self.src_num = src_num
        self.microphone_array = None
        self.source_location = [None] * src_num
        self.signal = [None] * src_num
        self.fs = sampling_rate
        self.indoor = indoor
        self.room = pra.ShoeBox(ROOM_SIZE, fs=sampling_rate, max_order=3, absorption=0.3)
        self.each_room = [pra.ShoeBox(ROOM_SIZE, fs=sampling_rate, max_order=3, absorption=0.3) for _ in range(src_num)]
        self.room_size = ROOM_SIZE
        if indoor:
            self.room = pra.AnechoicRoom(fs=sampling_rate)
            self.each_room = [pra.AnechoicRoom(fs=sampling_rate) for _ in range(src_num)]
            self.room_size = None

        # parameter for moving sound sources
        self.move = [False] * src_num
        self.bins = 1024
        self.move_overlap = True
        self.block_num = [None] * src_num

    def set_room(self, size: np.array, max_order=3, absorption=0.3):
        self.indoor = False
        self.room = pra.ShoeBox(size, fs=self.fs, max_order=max_order, absorption=absorption)
        self.each_room = [pra.ShoeBox(size, fs=self.fs, max_order=max_order, absorption=absorption)
                          for _ in range(self.src_num)]

    def set_microphone_array(self, location=None):
        """
        Set the location of microphone array
        :param location: numpy.array (mic_num x 3-dimension)
        :return:
        """
        if location is None:
            location = self.create_microphone_array()

        assert location.shape[1] == 3, 'Shape of location should be (mic_num, 3))'
        self.mic_num = len(location)
        self.microphone_array = location
        return

    def create_microphone_array(self, distance=MIC_INTERVAL, center_pos=None, angle=0.0, array_type='line'):
        """
        Create locations of microphone array automatically (array type: line or circle)
        :param distance: microphone max interval (line case), radius of microphone array (circle case)
        :param center_pos: center position of microphone array (default: center position of room)
        :param angle: rotation angle of microphone array (only XY)
        :param array_type: line or circle
        :return: microphone array location: np.array(mic_num * 3 dimension)
        """
        assert array_type in ['line', 'circle'], 'array_type should be "line" or "circle"'
        if center_pos is None:
            if self.room_size is None:
                center_pos = np.array([0.0, 0.0, 0.0])
            else:
                center_pos = self.room_size / 2

        absolute_pos = None
        if array_type == 'line':
            relative_pos = [r * np.array([np.cos(angle), np.sin(angle), 0.0])
                            for r in np.linspace(-distance/2, distance/2, self.src_num)]
            absolute_pos = center_pos + relative_pos
        elif array_type == 'circle':
            Exception('Not yet implement.')

        return absolute_pos

    def create_source_location(self, radius, angle, center_pos=None):
        if center_pos is None:
            if self.room_size is None:
                center_pos = np.array([0.0, 0.0, 0.0])
            else:
                center_pos = self.room_size / 2
        relative_pos = radius * np.array([np.cos(angle), np.sin(angle), 0.0])
        return center_pos + relative_pos

    def set_source_location(self, src_idx, location):
        self.source_location[src_idx] = location
        self.move[src_idx] = False
        return

    def set_source_path(self, src_idx, trajectory):
        """
        set sound path
        :param src_idx:
        :param trajectory: numpy.array(trajectory_length x 3 dimension)
        :return:
        """
        if self.signal[src_idx] is None:
            Exception(f'Not set up signal {src_idx}.')
        signal_length = len(self.signal[src_idx])
        bin_num = int(signal_length / self.bins) + 1
        if self.move_overlap:
            bin_num = int((signal_length - self.bins / 2) / (self.bins / 2)) + 1

        t_new = np.linspace(0.0, 1.0, bin_num)
        trj_len, dim = trajectory.shape
        t_old = np.linspace(0.0, 1.0, trj_len)
        assert dim == 2 or dim == 3, 'trajectory should be time-series data of coordinates with 2 or 3 dimension.'

        traj_interp = np.vstack([np.interp(t_new, t_old, trajectory[:, d]) for d in range(dim)]).T
        self.source_location[src_idx] = traj_interp
        self.move[src_idx] = True
        self.block_num = bin_num
        return

    def set_source_sound(self, src_idx, signal, offset=0):
        self.signal[src_idx] = np.concatenate([np.zeros(int(offset * self.fs)), signal])
        return

    def get_sound_block(self, blk_idx, src_idx):
        tmp = np.zeros(len(self.signal[src_idx]))
        st_idx = blk_idx * self.bins / 2 if self.move_overlap else blk_idx * self.bins
        ed_idx = blk_idx * self.bins / 2 + self.bins if self.move_overlap else (blk_idx + 1) * self.bins
        tmp[st_idx:ed_idx] = self.signal[st_idx:ed_idx] * pra.hann(self.bins)
        # in case of half overlap, multiply each sound block of moving source by hanning window for normalization.
        window = pra.hann(self.bins) if self.move_overlap else 1
        tmp[st_idx:ed_idx] = self.signal[st_idx:ed_idx] * window
        return tmp

    def simulation(self):
        for s in range(self.src_num):
            if self.source_location[s] is None or self.signal[s] is None:
                Exception(f'Cannot simulate because the location or signal of source {s} is not installed yet.')
        if self.microphone_array is None:
            Exception('Cannot simulate because the microphone array is not set up yet.')

        # set up room object
        self.room.add_microphone_array(pra.MicrophoneArray(self.microphone_array.T, fs=self.fs))
        for s in range(self.src_num):
            if self.move[s]:
                for b in self.block_num:
                    sig_block = self.get_sound_block(blk_idx=b, src_idx=s)
                    self.room.add_source(position=self.source_location[s][b], signal=sig_block)
            else:
                self.room.add_source(position=self.source_location[s], signal=self.signal[s])

        for s in range(self.src_num):
            self.each_room[s].add_microphone_array(pra.MicrophoneArray(self.microphone_array.T, fs=self.fs))
            if self.move[s]:
                for b in self.block_num:
                    sig_block = self.get_sound_block(blk_idx=b, src_idx=s)
                    self.each_room[s].add_source(position=self.source_location[s][b], signal=sig_block)
            else:
                self.each_room[s].add_source(position=self.source_location[s], signal=self.signal[s])

        # simulation
        self.room.simulate()
        for s in range(self.src_num):
            self.each_room[s].simulate()

        return

    def output(self, folder, combine=True):
        if not os.path.isdir(folder):
            os.makedirs(folder)
        #
        # observation = self.room.mic_array.signals
        # if combine and self.mic_num == 2:
        #     util.Data.write_wav(path=f'{folder}/observation_stereo.wav', data=observation,
        #                         info={util.FS: FS, util.CH: 2})
        # else:
        #     for i in range(len(observation)):
        #         util.Data.write_wav(path=f'{folder}/observation_ch{i}.wav', data=observation[i],
        #                             info={util.FS: self.fs, util.CH: 1})
        # for s in range(self.src_num):
        #     each_sig = self.each_room[s].mic_array.signals
        #     if combine and self.mic_num == 2:
        #         util.Data.write_wav(path=f'{folder}/signal{s}_stereo.wav', data=each_sig,
        #                             info={util.FS: FS, util.CH: 2})
        #     else:
        #         for i in range(len(each_sig)):
        #             util.Data.write_wav(path=f'{folder}/signal{s}_ch{i}.wav', data=each_sig[i],
        #                                 info={util.FS: self.fs, util.CH: 1})
        return


class Crowd:
    @classmethod
    def csv_to_crowd_list(cls, filename):
        logger.info(f'read csv - {filename}')
        df = pd.read_csv(filename).dropna()
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['geom'] = df['geom'].apply(lambda x: wkt.loads(x))
        start_time = df['start_time'].min()

        crowd_list = [cls(path=dr['geom'],
                          start_time=(dr['start_time']-start_time).total_seconds(),
                          height=1.7 if 'height' not in df.columns else dr['height'],
                          foot_step=1.7 * 0.35 if 'foot_step' not in df.columns else dr['foot_step'],
                          name=str(dr['id']),
                          foot_tag=None if 'foot_tag' not in df.columns else dr['foot_tag'])
                      for _, dr in df.iterrows()]

        return crowd_list

    def __init__(self, path: LineString,
                 start_time: int | float,
                 height: float,
                 foot_step: float,
                 time_step: float = 1.0,
                 name=None,
                 foot_tag=None):
        self.time_step = time_step
        self.path = path
        self.start_time = start_time
        self.height = height
        self.foot_step = foot_step
        self.duration = len(path.coords) * time_step
        self.name = name
        self.foot_tag = foot_tag

    def time_interpolate(self, t: float | int):
        if self.start_time > t or t >= self.start_time + self.duration:
            return None
        t_step_float = (t - self.start_time) / self.time_step
        t_step_int = int(t_step_float)
        p1 = self.path.coords[t_step_int]
        p2 = self.path.coords[t_step_int + 1]
        line = LineString([p1, p2])
        return line.interpolate(distance=t_step_float - t_step_int, normalized=True)

    @staticmethod
    def __get_point_index_float(line: LineString, point: Point):
        """
        Nポイントで構成されるLineStringについて、内分点が何ポイント目に相当するか抽出
        Parameters
        ----------
        line: LineString
        point

        Returns
        -------
        result: float

        """
        c = line.coords
        dist_parts = [0] + [Point(c[i]).distance(Point(c[i + 1])) for i in range(len(c) - 1)]

        for i, l in enumerate(line.coords):
            if l == point.coords[0]:
                return i
        d = line.distance(point)
        gc = split(line, point.buffer(d + 1.0e-10))
        if len(gc.geoms) != 2:
            Exception(f'Multiple split points: {point}')

        tmp_length = LineString([gc.geoms[0].coords[-2], gc.geoms[1].coords[1]]).length
        tmp_div = LineString([gc.geoms[0].coords[-2], point.coords[0]]).length
        result = len(gc.geoms[0].coords) - 2 + tmp_div / tmp_length
        return result

    def get_foot_points(self):
        """

        :return: [{'t': time(float), 'point': point(Point)}]
        """
        walking_distance = self.path.length
        dist = 0.0

        res = []
        pc = self.path.coords
        dist_parts = [0] + [Point(pc[i]).distance(Point(pc[i + 1])) for i in range(len(pc) - 1)]
        while dist < walking_distance:
            point = self.path.interpolate(distance=dist, normalized=False)

            # extract time index for each point
            for line_index in range(len(dist_parts) - 1):
                if sum(dist_parts[:line_index + 1]) <= dist < sum(dist_parts[:line_index + 2]):
                    dist_part = Point(pc[line_index]).distance(point) / dist_parts[line_index + 1]
                    res.append({'t': line_index + dist_part + self.start_time, 'point': point, 'dist': dist})

                # if sum(dist_parts[:line_index + 1]) == dist:
                #     res.append({'t': line_index, 'point': point, 'dist': dist})
                # elif sum(dist_parts[:line_index + 1]) < dist:
                #     if dist < sum(dist_parts[:line_index + 2]):
                #         dist_part = Point(pc[line_index]).distance(point) / dist_parts[line_index + 1]
                #         res.append({'t': line_index + dist_part, 'point': point, 'dist': dist})
                #     else:
                #         pass
                # pass
            # if sum(dist_parts[:line_index]) == dist:
            #     res.append({'t': line_index, 'point': point})
            # elif sum(dist_parts[:line_index]) > dist:
            #     Exception('')
            # else:
            #     while sum(dist_parts[:line_index]) < dist < sum(dist_parts[:line_index + 1]):
            #
            #         pass
            #
            # res.append({'t': Crowd.__get_point_index_float(self.path, point), 'point': point})
            # TODO randomize foot_step
            dist += self.foot_step

        return res


class FootstepSound:
    FOLDER = './data/ambient_sound/footstep'

    def __init__(self, sampling_rate=16000):
        self.fs = sampling_rate
        self.folder_list = glob.glob(f'{FootstepSound.FOLDER}/*')
        self.file_dict = {os.path.basename(fl): glob.glob(f'{fl}/*.wav') for fl in self.folder_list}
        self.wav_tag = list(self.file_dict.keys())
        self.wav_dict = {tag: [FootstepSound.__read_wav_file(f, self.fs) for f in files]
                         for tag, files in self.file_dict.items()}

    @staticmethod
    def __read_wav_file(filename, fs, normalize=True):
        _fs, signal = wavfile.read(filename)
        signal = signal.astype(float)

        # Convert stereo to mono if necessary
        if signal.ndim > 1:
            signal = signal.mean(axis=1)

        if _fs != fs:
            signal = librosa.resample(y=signal, orig_sr=_fs, target_sr=fs)

        # Validate signal before normalization
        if len(signal) == 0:
            logger.warning(f'Empty signal from {filename}')
            return np.array([0.0])  # Return minimal signal

        if normalize:
            std = signal.std()
            if std > 0:
                signal = signal / std
            else:
                logger.warning(f'Signal from {filename} has zero standard deviation. Using original signal.')
        return signal

    def get_rnd_sound_index(self, tag):
        wav_list = self.wav_dict[tag]
        index = random.randint(0, len(wav_list) - 1)
        return index

    def get_wav(self, tag, index=-1):
        wav_list = self.wav_dict[tag]
        if index < 0 or index > len(wav_list) - 1:
            index = self.get_rnd_sound_index(tag)
        return wav_list[index]

    def get_tags(self):
        return self.wav_tag

    def get_foot_std(self, tag):
        return np.array([w.std() for w in self.wav_dict[tag]]).mean()


class CrowdSim:
    @classmethod
    def from_shp(cls, shp_path, height=3.0, max_order=0):
        logger.info(msg=f'read shp - {shp_path}')
        df = gpd.read_file(shp_path)
        room_polygon = df['geometry'].loc[0]
        return cls(room_polygon=room_polygon, height=height, max_order=max_order)

    def __init__(self, room_polygon: Polygon, height=3.0, max_order=0):
        self.crowd_list = None
        self.footstep = []
        self.room_polygon = room_polygon
        room_coordinates = np.array(room_polygon.exterior.coords[:-1])
        self.room_info = {'corners': room_coordinates.T, 'fs': SR, 'max_order': max_order}
        self.room_height = height
        self.mic_info = None
        self.foot_sound = FootstepSound(sampling_rate=SR)

        # self.foot_tags = []  # setting the foot tag for each person
        self.person_sound_info = [{}]  # setting the sound information(trajectory, foot tag and footstep index)

        # self.room = pra.Room.from_corners(**self.room_info)
        # self.room.extrude(height=height)
        return

    def create_room(self):
        room = pra.Room.from_corners(**self.room_info)
        room.extrude(height=self.room_height)
        room.add_microphone_array(self.mic_info.T)
        return room

    def set_crowd(self, crowd_list: list[Crowd]):
        self.crowd_list = crowd_list
        # TODO ここが時間かかるので将来的にマルチプロセスにする必要あり -> なぜかマルチプロセスの方が遅いため保留
        #
        # def get_foot_points_wrapper(index):
        #     return self.crowd_list[index].get_foot_points(), index
        #
        # res = Parallel(n_jobs=-1)(delayed(get_foot_points_wrapper)(i) for i in range(len(crowd_list)))
        # res.sort(key=lambda x: x[1])
        # self.footstep = [r[0] for r in res]
        self.footstep = [crowd.get_foot_points() for crowd in tqdm(self.crowd_list, desc='[CrowdSet]')]
        # foot_tags = [self.foot_sound.get_tags()[0] for _ in self.crowd_list]
        foot_tags = [self.foot_sound.get_tags()[0] if crowd.foot_tag is None else crowd.foot_tag
                     for crowd in self.crowd_list]
        self.person_sound_info = [
            {
                'id': i,
                'foot_tag': foot_tags[i],
                'time_series': [
                    {
                        't': foot['t'],
                        'x': foot['point'].x,
                        'y': foot['point'].y,
                        'z': 0.0,
                        'sound_index': self.foot_sound.get_rnd_sound_index(foot_tags[i])
                    }
                    for foot in self.footstep[i]
                ]
            }
            for i in range(len(self.crowd_list))
        ]
        return

    def set_microphone(self, mic_loc: np.array):
        """
        Set the locations of multiple microphones.
        Parameters
        ----------
        mic_loc
        numpy.array(mic_num x 3)

        Returns
        -------

        """
        mic_loc_tmp = mic_loc
        if mic_loc.shape[1] == 2:
            mic_loc_tmp = np.zeros((len(mic_loc), 3))
            mic_loc_tmp[:, 0:2] = mic_loc
        self.mic_info = mic_loc_tmp

        return

    def person_sim(self, index):
        """
        Get audio signal for footstep of person {index}
        Parameters
        ----------
        index: int
            index of person

        Returns
        -------
        result: np.array
            audio signal
        """
        # TODO 一人の移動時間が長い場合にメモリエラーが出るため、pyroomacousticsのシミュレーションを分割する必要あり

        """
        {
                'id': i,
                'foot_tag': foot_tags[i],
                'time_series': [
                    {
                        't': foot['t'],
                        'x': foot['x'],
                        'y': foot['y'],
                        'z': 0.0,
                        'sound_index': self.foot_sound.get_rnd_sound_index(foot_tags[i])
                    }
                    for foot in self.footstep[i]
                ]
            }
        """
        if self.crowd_list is None:
            Exception('Not set crowd data.')
        room = self.create_room()
        person_footstep = self.footstep[index]

        # Handle empty footstep list
        if not person_footstep:
            logger.warning(f'No footsteps for person {index}')
            return np.zeros([room.n_mics, 1])

        foot_tag = self.foot_sound.get_tags()[0]
        offset_time = min([foot['t'] for foot in person_footstep])
        for foot in person_footstep:
            p = [foot['point'].x, foot['point'].y, 0.0]
            if room.is_inside(p):
                try:
                    sig = self.foot_sound.get_wav(foot_tag, -1)
                    # Validate signal before adding source
                    if sig is None or len(sig) == 0 or np.any(np.isnan(sig)) or np.any(np.isinf(sig)):
                        logger.warning(f'Invalid signal for person {index}, foot_tag {foot_tag}')
                        continue
                    room.add_source(position=p,
                                    signal=sig,
                                    delay=foot['t']-offset_time)
                except Exception as e:
                    logger.warning(f'Error adding source for person {index}: {str(e)}')
                    continue

        if room.n_sources == 0:
            logger.debug(f'No valid sources added for person {index}')
            return np.zeros([room.n_mics, 1])

        try:
            logger.debug(f'Simulating person {index} with {room.n_sources} source(s)')
            room.simulate()
            simulated_sound = room.mic_array.signals
        except Exception as e:
            logger.warning(f'Simulation failed for person {index} with {room.n_sources} sources: {type(e).__name__}: {str(e)}')
            return np.zeros([room.n_mics, 1])

        result = np.concatenate([np.zeros([room.n_mics, int(offset_time*room.fs)]), simulated_sound],
                                axis=1)
        return result

    def multi_process_simulation(self, duration=10.0):
        """
        self.person_sound_info = [
            {
                'id': i,
                'foot_tag': foot_tags[i],
                'time_series': [
                    {
                        't': foot['t'],
                        'x': foot['point'].x,
                        'y': foot['point'].y,
                        'z': 0.0,
                        'sound_index': self.foot_sound.get_rnd_sound_index(foot_tags[i])
                    }
                    for foot in self.footstep[i]
                ]
            }
            for i in range(len(self.crowd_list))
        ]
        """
        tmp_df = pd.DataFrame(data=[{'id': psi['id'], 'foot_tag': psi['foot_tag'],
                                     't': ts['t'], 'x': ts['x'], 'y': ts['y'], 'z': ts['z'],
                                     'sound_index': ts['sound_index']}
                                    for psi in self.person_sound_info for ts in psi['time_series']])
        tmp_df['group'] = (tmp_df['t'] / duration).astype(int)
        sim_param_group = [{'group': g, 'foot_tag': gr_df['foot_tag'].tolist(), 't': gr_df['t'].tolist(),
                            'x': gr_df['x'].tolist(), 'y': gr_df['y'].tolist(), 'z': gr_df['z'].tolist(),
                            'sound_index': gr_df['sound_index'].tolist(), 'offset': gr_df['t'].min(),
                            'size': len(gr_df)}
                           for g, gr_df in tmp_df.groupby('group')]

        def _sim_group(index):
            sim_param = sim_param_group[index]
            room = self.create_room()
            for i in range(sim_param['size']):
                p = [sim_param['x'][i], sim_param['y'][i], sim_param['z'][i]]
                sig = self.foot_sound.get_wav(sim_param['foot_tag'][i], sim_param['sound_index'][i])
                if room.is_inside(p):
                    delay = sim_param['t'][i] - sim_param['offset']
                    try:
                        # Validate signal before adding source
                        if sig is None or len(sig) == 0 or np.any(np.isnan(sig)) or np.any(np.isinf(sig)):
                            logger.warning(f'Invalid signal for group {sim_param["group"]}, sound_index {sim_param["sound_index"][i]}')
                            continue
                        room.add_source(position=p, signal=sig, delay=delay)
                    except Exception as e:
                        logger.warning(f'Error adding source (position: {p}, delay:{delay} ) for group {sim_param["group"]}: {str(e)}')
                        continue
            simulated_sound = np.zeros([room.n_mics, 1])
            if room.n_sources > 0:
                try:
                    # logger.debug(f'Simulating group {sim_param["group"]} with {room.n_sources} source(s)')
                    room.simulate()
                    simulated_sound = room.mic_array.signals
                except Exception as e:
                    logger.warning(f'Simulation failed for group {sim_param["group"]} with {room.n_sources} sources: {type(e).__name__}: {str(e)}')
                    simulated_sound = np.zeros([room.n_mics, 1])
            result = {'delay': int(room.fs * sim_param['offset']), 'signal': simulated_sound,
                      'group': sim_param['group']}
            return result

        res_list = Parallel(n_jobs=-1)(delayed(_sim_group)(i) for i in tqdm(range(len(sim_param_group)),
                                                                            desc='[Audio crowd simulation]'))
        # res_list = [self.__sim_group(sp) for sp in tqdm(sim_param_group)]
        signal_size = max([r['delay'] + r['signal'].shape[1] for r in res_list])
        channel = res_list[0]['signal'].shape[0]
        sim_result = np.zeros([channel, signal_size])
        for r in res_list:
            each_size = r['signal'].shape[1]
            sim_result[:, r['delay']:r['delay']+each_size] += r['signal']
        return sim_result

    def __sim_group(self, sim_param):
        room = self.create_room()
        for i in range(sim_param['size']):
            p = [sim_param['x'][i], sim_param['y'][i], sim_param['z'][i]]
            sig = self.foot_sound.get_wav(sim_param['foot_tag'][i], sim_param['sound_index'][i])
            if room.is_inside(p):
                room.add_source(position=p, signal=sig, delay=sim_param['t'][i]-sim_param['offset'])
        simulated_sound = np.zeros([room.n_mics, 1])
        if room.n_sources > 0:
            room.simulate()
            simulated_sound = room.mic_array.signals
        result = {'delay': int(room.fs * sim_param['offset']), 'signal': simulated_sound, 'group': sim_param['group']}
        return result

    def simulation(self, multi_process=True, time_unit=10.0):
        logger.info(f'audio simulation start - {len(self.crowd_list)} people')
        logger.info(f'multi_process - {multi_process}')
        if multi_process:
            return self.multi_process_simulation(duration=time_unit)
            # people_sound = Parallel(n_jobs=-1)(delayed(self.person_sim)(i) for i in tqdm(range(len(self.crowd_list))))
            # ch = people_sound[0].shape[0]
            # audio_size = max([ps.shape[1] for ps in people_sound])
            # sim_result = np.zeros((ch, audio_size))
            # for ps in people_sound:
            #     sim_result[:, :ps.shape[1]] += ps
            # return sim_result
        else:
            people_sound = [self.person_sim(i) for i in tqdm(range(len(self.crowd_list)))]

            # Handle empty results
            if not people_sound:
                logger.warning('No audio signals generated from any person')
                # Return empty signal with proper shape
                return np.zeros((len(self.mic_info), 1))

            ch = people_sound[0].shape[0]
            audio_size = max([ps.shape[1] for ps in people_sound])
            sim_result = np.zeros((ch, audio_size))
            for ps in people_sound:
                sim_result[:, :ps.shape[1]] += ps
            return sim_result

    def generate_noise_std(self, snr):
        # TODO ここはwav_tagの割合でうまく調整する予定
        sigma_s = self.foot_sound.get_foot_std(self.foot_sound.wav_tag[0])
        sigma_n = sigma_s * (10 ** (-snr / 20))
        return sigma_n

    def save_footstep_shp(self, filename):
        # TODO
        return

    def plot(self, filename):
        """
        Simulation Room 2D-plot. It will be deprecated.
        :param filename: png filename
        :return:
        """
        room = pra.Room.from_corners(**self.room_info)
        room.add_microphone_array(self.mic_info[:, 0:2].T)
        for person_footstep in self.footstep:
            for foot in person_footstep:
                p = [foot['point'].x, foot['point'].y]
                if room.is_inside(p):
                    room.add_source(position=p)
        room.plot()
        plt.savefig(filename)
        plt.close()
        plt.cla()
        return

    def get_person_moving_features(self, index):
        """
        Get GeoDataFrame of index-th person as Moving Features Format (id, t, geometry)

        Parameters
        ----------
        index : int
            index of person.

        Returns
        -------
        df : GeoDataFrame
        """
        crowd = self.crowd_list[index]
        line = crowd.path.coords
        record_num = len(line)
        df = gpd.GeoDataFrame({'t': [crowd.start_time + crowd.time_step * r for r in range(record_num - 1)]},
                              geometry=[LineString([line[r], line[r + 1]]) for r in range(record_num - 1)])
        df['id'] = index
        return df

    def get_all_moving_features(self, verbose=True):
        """
        Get GeoDataFrame of all people as Moving Features Format (id, t, geometry)
        Returns
        -------
        GeoDataFrame
        """
        crowd_df = pd.concat([self.get_person_moving_features(index)
                              for index in tqdm(range(len(self.crowd_list)), disable=not verbose,
                                                desc='[Generate Moving Features]')])
        return crowd_df.reset_index().drop('index', axis=1)

    def decomposed_moving_feature(self, verbose=True):
        """
        Get GeoDataFrame of all people as Moving Features Format (id, t, geometry)
        Returns
        -------
        GeoDataFrame
        """
        crowd_df = pd.concat([self.get_person_moving_features(index)
                              for index in tqdm(range(len(self.crowd_list)), disable=not verbose)])
        return crowd_df

    def crowd_density_to_csv(self, csv_name):
        duration = int(max([c.start_time + c.duration for c in self.crowd_list]))
        crowd_density = np.zeros(duration)
        for c in self.crowd_list:
            st_idx = int(c.start_time)
            ed_idx = int(c.start_time+c.duration)
            crowd_density[st_idx:ed_idx] += 1
        df = pd.DataFrame(data={'t': [t for t in range(duration)], 'crowd': crowd_density.tolist()})
        df.to_csv(csv_name, index=False)
        return

    def crowd_density_from_each_mic(self, mic_index: int, csv_name: str, distance_list: list, time_step=1.0,
                                    verbose=True):
        mic_geom = Point(self.mic_info[mic_index][0:2])
        max_time_index = int(max([c.start_time + c.duration for c in self.crowd_list]) / time_step) + 1

        out_df = pd.DataFrame(np.zeros((max_time_index, len(distance_list) + 1)),
                              columns=['count'] + [f'distance_{d}' for d in distance_list],
                              index=[t for t in range(max_time_index)])
        out_df.index.name = 'time_index'

        for i in tqdm(range(len(self.crowd_list)), desc='[Crowd Counting]', disable=not verbose):
            trj_df = self.get_person_moving_features(i)
            trj_df['time_index'] = (trj_df['t'] / time_step).astype(int)
            trj_df['distance'] = trj_df['geometry'].apply(lambda x: Point(x.coords[0]).distance(mic_geom))
            trj_df['count'] = 1.0
            for d in distance_list:
                trj_df[f'distance_{d}'] = trj_df['distance'].apply(lambda x: 1 if x < d else 0)
            trj_df = trj_df[['time_index', 'count'] + [f'distance_{d}' for d in distance_list]].set_index('time_index')
            out_df.loc[trj_df.index] += trj_df.loc[trj_df.index]

        out_df = out_df.astype(int).reset_index()
        out_df['t'] = out_df['time_index'] * time_step
        folder = os.path.dirname(csv_name)
        os.makedirs(folder, exist_ok=True)
        out_df.to_csv(csv_name, index=False)
        return

    def crowd_dendity_from_grid_shp(self, grid_shp: str, csv_name: str, time_step=1.0,
                                    x_idx_range=None, y_idx_range=None, geom_flag=True, verbose=True):
        # グリッドデータの読み込み
        logger.info('Crowd density calculation from grid shapefile.')
        grid_df = gpd.read_file(grid_shp)
        if grid_df.crs is None:
            grid_df.set_crs(epsg=BASE_SRID, inplace=True)
        # グリッドの絞り込み
        if x_idx_range is not None:
            grid_df = grid_df[(grid_df['x_idx'] >= x_idx_range[0]) & (grid_df['x_idx'] <= x_idx_range[1])]
        if y_idx_range is not None:
            grid_df = grid_df[(grid_df['y_idx'] >= y_idx_range[0]) & (grid_df['y_idx'] <= y_idx_range[1])]
        logger.info(f'grid num: {len(grid_df)}')

        # 人流データの取得
        logger.info('Get all moving features of crowd.')
        crowd_df = self.get_all_moving_features(verbose=verbose)
        crowd_df.set_crs(grid_df.crs, inplace=True)

        # 人流データとグリッドデータの空間結合
        logger.info('Calculating number of people flow at each grid.')
        join_df = gpd.sjoin(crowd_df, grid_df[['grid_id', 'geometry']], how='left', predicate='intersects')
        cnt_df = join_df.groupby(['t', 'grid_id'])['id'].nunique().reset_index(name='count')

        # 全時刻 x 全グリッドの組み合わせを作成
        t_df = pd.DataFrame({'t': np.arange(min([c.start_time for c in self.crowd_list]),
                                            max([c.start_time + c.duration for c in self.crowd_list]) + time_step,
                                            time_step)})
        g_df = pd.DataFrame({'grid_id': grid_df['grid_id'].unique()})
        tg_df = t_df.merge(g_df, how='cross')

        cnt_df = tg_df.merge(cnt_df, on=['t', 'grid_id'], how='left').fillna({'count': 0})
        cnt_df['count'] = cnt_df['count'].astype(int)
        cnt_df['time_index'] = (cnt_df['t'] / time_step).astype(int)
        GEOM_COL = 'geometry'
        join_columns = [c for c in grid_df.columns if c != GEOM_COL]
        if geom_flag:
            join_columns.append(GEOM_COL)
        cnt_df = pd.merge(cnt_df, grid_df[join_columns], on='grid_id', how='left')

        logger.info('Saving crowd density csv file.')
        folder = os.path.dirname(csv_name)
        os.makedirs(folder, exist_ok=True)
        cnt_df.to_csv(csv_name, index=False)
        return
"""
from simulation.util import *
crowd_list = Crowd.csv_to_crowd_list('./data/gis/marunouchi/crowd/roi1_crowd0807_15.csv')
crowd_sim = CrowdSim.from_shp('./data/gis/marunouchi/roi/marunouchi_roi1.shp')
mic_shp = './data/gis/marunouchi/mic/marunouchi_mic30.shp'
crowd_sim.set_crowd(crowd_list)

df = gpd.read_file(mic_shp)
id_list = [dr['id'] for _, dr in df.iterrows() if crowd_sim.room_polygon.contains(dr['geometry'])]
mic_location = np.array([[dr['geometry'].x, dr['geometry'].y, dr['height']]
                         for _, dr in df.iterrows() if crowd_sim.room_polygon.contains(dr['geometry'])])
crowd_sim.set_microphone(mic_location)

mic_index = 0
time_step = 1.0
distance_list = [20, 40, 60, 80, 100, 1000]
mic_geom = Point(crowd_sim.mic_info[mic_index][0:2])
max_time_index = int(max([c.start_time + c.duration for c in crowd_sim.crowd_list]) / time_step) + 1

out_df = pd.DataFrame(np.zeros((max_time_index, len(distance_list))),
                      columns=[f'distance_{d}' for d in distance_list],
                      index=[t for t in range(max_time_index)])
out_df.index.name = 'time_index'

for i in tqdm(range(len(crowd_sim.crowd_list))):
    trj_df = crowd_sim.get_person_moving_features(i)
    trj_df['time_index'] = (trj_df['t'] / time_step).astype(int)
    trj_df['distance'] = trj_df['geometry'].apply(lambda x: Point(x.coords[0]).distance(mic_geom))
    for d in distance_list:
        trj_df[f'distance_{d}'] = trj_df['distance'].apply(lambda x: 1 if x < d else 0)
    trj_df = trj_df[['time_index'] + [f'distance_{d}' for d in distance_list]].set_index('time_index')
    out_df.loc[trj_df.index] += trj_df.loc[trj_df.index]
"""

class PersonTrajectory:
    def __init__(
            self,
            pid: int,
            room_polygon: Polygon,
            start_time: int | float = 0.0,
            duration: int | float = 3600.0,
            start_point: Point = None,
            v: float = 1.0,
            v_sigma: float = 0.0,
            dir_sigma: float = None
    ):
        """
        一人分の移動軌跡を生成するクラス
        壁反射なしの単純なランダムウォークで生成する
        Parameters
        ----------
        pid
        room_polygon
        start_time
        start_point
        v
        v_sigma
        dir_sigma
        """
        self.pid = pid
        self.start_time = start_time
        self.duration = duration
        self.room_polygon = room_polygon
        self.start_point = start_point if start_point is not None else self.__get_random_point_from_room_edge()
        self.v = v
        self.v_sigma = v_sigma
        self.dir_sigma = math.pi / 8 if dir_sigma is None else dir_sigma
        self.trajectory = None
        return

    def __get_random_point_from_room_edge(self, buffer=0.01):
        """
        room_polygonの外周からランダムに点を選ぶ
        Parameters
        ----------
        buffer: float
            外周からの距離。正の値なら外周から内側に、負の値なら外周から外側に点が選ばれる
        Returns
        -------
        result: Point
        """
        line_string = self.room_polygon.buffer(distance=-buffer).exterior
        total_length = line_string.length
        random_length = random.uniform(0, total_length)
        return line_string.interpolate(random_length)

    def __get_direction_from_point(self, point):
        """
        入力された点から部屋の中心に向かう方向を計算する
        Parameters
        ----------
        point: Point
            入力点
        Returns
        -------
        result: float
            部屋の中心に向かう方向（ラジアン）
        """
        room_center = self.room_polygon.centroid
        return math.atan2(room_center.y - point.y, room_center.x - point.x)

    def generate_trajectory(self, time_step=1.0, simulation_total_time=3600.0):
        current_point = self.start_point
        # current_dir = random.uniform(0, 2 * math.pi)
        current_dir = random.gauss(self.__get_direction_from_point(current_point), math.pi / 16.0)
        while not self.room_polygon.contains(
                Point(
                    current_point.x + self.v * 5 * math.cos(current_dir),
                    current_point.y + self.v * 5 * math.sin(current_dir)
                )
        ):
            current_dir = random.gauss(self.__get_direction_from_point(current_point), math.pi / 16.0)
        history = [(self.start_time, current_point, current_dir)]
        duration = simulation_total_time - self.start_time if self.duration is None else self.duration
        idx = 0
        for t in np.arange(self.start_time + time_step, self.start_time + duration, time_step):
            # 速度と方向にランダムノイズを加える
            v_t = max(0.0, random.gauss(self.v, self.v_sigma))
            # 最初だけは方向にノイズを加えない（点が部屋の外に行く可能性があるため）
            dir_t = random.gauss(current_dir, self.dir_sigma) if idx > 0 else current_dir
            # 次の点を計算する
            new_x = current_point.x + v_t * time_step * math.cos(dir_t)
            new_y = current_point.y + v_t * time_step * math.sin(dir_t)
            new_point = Point(new_x, new_y)

            current_point = new_point
            current_dir = dir_t
            history.append((t, current_point, current_dir))
            idx += 1
            if not self.room_polygon.contains(new_point):
                # 部屋の外に出たら終了
                break

        self.trajectory = LineString([h[1] for h in history])
        return


class PersonTrajectoryOld:
    def __init__(
            self,
            pid: int,
            start_time: int | float,
            end_time: int | float,
            start_point: Point,
            v: float = 1.0,
            v_sigma: float = 0.0,
            dir_sigma: float = None,
            room_polygon: Polygon = None
    ):
        """
        一人分の移動軌跡を生成するクラス
        壁に当たった場合は反射するように生成する
        Parameters
        ----------
        pid: int
            person id
        start_time: int or float
            person start time (second)
        end_time: int or float
            person end time (second)
        start_point: Point
            person start point
        v: float
            person speed (m/s)
        v_sigma: float
            standard deviation of speed
        dir_sigma: float
            standard deviation of direction (radian)
        room_polygon: Polygon
            simulation room polygon. If the generated point is outside of the room, it will be reflected
        """
        self.pid = pid
        self.start_time = start_time
        self.end_time = end_time
        self.start_point = start_point
        self.v = v
        self.dir_sigma = math.pi / 8 if dir_sigma is None else dir_sigma
        self.v_sigma = v_sigma
        self.room_polygon = room_polygon
        if self.room_polygon is not None and not self.room_polygon.contains(self.start_point):
            # 開始点が部屋の外にある場合はpolygon内のランダムな点を開始点とする
            minx, miny, maxx, maxy = self.room_polygon.bounds
            while True:
                random_point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                if self.room_polygon.contains(random_point):
                    self.start_point = random_point
                    break
        self.trajectory = None
        return

    def generate_trajectory(self, time_step=1.0):
        current_point = self.start_point
        current_dir = random.uniform(0, 2 * math.pi)
        history = [(self.start_time, current_point, current_dir)]
        for t in np.arange(self.start_time + time_step, self.end_time, time_step):
            # 速度と方向にランダムノイズを加える
            v_t = max(0.0, random.gauss(self.v, self.v_sigma))
            dir_t = random.gauss(current_dir, self.dir_sigma)
            # 次の点を計算する
            new_x = current_point.x + v_t * time_step * math.cos(dir_t)
            new_y = current_point.y + v_t * time_step * math.sin(dir_t)
            new_point = Point(new_x, new_y)


            if self.room_polygon is not None and not self.room_polygon.contains(new_point):
                # 部屋の外に出ないように反射させる
                line = LineString([current_point, new_point])
                intersection = line.intersection(self.room_polygon.boundary)
                if intersection.is_empty:
                    # 交点がない場合は元の位置に留まる
                    new_point = current_point
                else:
                    # 交点が複数ある場合は最も近い交点を選ぶ
                    if isinstance(intersection, Point):
                        nearest_intersection = intersection
                    else:
                        nearest_intersection = min(intersection.geoms, key=lambda p: current_point.distance(p))
                    # 反射後の点を計算する
                    reflect_dir = math.atan2(nearest_intersection.y - current_point.y,
                                             nearest_intersection.x - current_point.x) + math.pi
                    # v_t = v_t - nearest_intersection.distance(current_point) / time_step
                    new_x = nearest_intersection.x + v_t * time_step * math.cos(reflect_dir)
                    new_y = nearest_intersection.y + v_t * time_step * math.sin(reflect_dir)
                    new_point = Point(new_x, new_y)
                    # 反射後の点が部屋の外に出ないようにする
                    if self.room_polygon is not None and not self.room_polygon.contains(new_point):
                        new_point = nearest_intersection
                    current_dir = math.atan2(new_point.y - current_point.y, new_point.x - current_point.x)
            current_point = new_point
            history.append((t, current_point, current_dir))
        self.trajectory = LineString([h[1] for h in history])
        return


class CrowdTrajectory:
    @staticmethod
    def from_csv(csv_path):
        # TODO 実装中
        crowd_df = pd.read_csv(csv_path)
        crowd_df['geom'] = crowd_df['geom'].apply(lambda x: wkt.loads(x))
        crowd_traj = CrowdTrajectory()
        crowd_traj.crowd_trj_df = crowd_df
        logger.info(f'crowd trajectory loaded from {csv_path}')
        return crowd_traj

    def __init__(
            self,
            room_polygon: Polygon = None,
            room_size = 100
    ):
        self.crowd_trj_df = None
        self.person_trajectories = []
        default_coords = ((-room_size/2, -room_size/2),
                          (-room_size/2, room_size/2),
                          (room_size/2, room_size/2),
                          (room_size/2, -room_size/2),
                          (-room_size/2, -room_size/2))
        self.room_polygon = Polygon(default_coords) if room_polygon is None else room_polygon
        logger.debug(f'CrowdTrajectory initialized with room polygon: {self.room_polygon}')
        return

    def set_crowd_trajectory_old(
            self,
            person_num,
            start_time,
            end_time,
            v=1.5,
            v_sigma=0.3,
            exist_time=60.0,
            exist_time_sigma=0.0,
            dir_division=8,
            datetime_str='2024-01-01 00:00:00'
    ):
        for pid in range(person_num):
            delta_time = max(int(exist_time + random.gauss(0, exist_time_sigma)), 1)
            each_time = int(random.random() * (end_time - start_time) + start_time)
            start_point = Point(random.uniform(self.room_polygon.bounds[0], self.room_polygon.bounds[2]),
                                random.uniform(self.room_polygon.bounds[1], self.room_polygon.bounds[3]))
            person_trajectory = PersonTrajectoryOld(
                pid=pid,
                start_time=each_time,
                end_time=each_time + delta_time,
                start_point=start_point,
                v=v+random.gauss(0, 0.1),
                v_sigma=v_sigma,
                dir_sigma=math.pi / dir_division,
                room_polygon=self.room_polygon
            )
            self.person_trajectories.append(person_trajectory)
        logger.info(f'set crowd setting - person num: {person_num}, time range: ({start_time}, {end_time}), v: {v}, exist_time: {exist_time}')

        dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        trj_list = []
        for pt in self.person_trajectories:
            pt.generate_trajectory()
            start_time = dt + timedelta(seconds=pt.start_time)
            line_string = pt.trajectory
            trj_list.append({'id': pt.pid, 'start_time': start_time, 'geom': line_string})
        self.crowd_trj_df = pd.DataFrame(trj_list)
        logger.info('crowd trajectory generated.')
        return

    def set_crowd_trajectory(
            self,
            person_num,
            start_time,
            end_time,
            v=1.5,
            v_sigma=0.3,
            dir_division=8,
            datetime_str='2024-01-01 00:00:00'
    ):
        for pid in range(person_num):
            each_start_time = int(random.random() * (end_time - start_time) * 0.95 + start_time)
            if pid == 0:
                each_start_time = 0
            person_trajectory = PersonTrajectory(
                pid=pid,
                start_time=each_start_time,
                room_polygon=self.room_polygon,
                v=v+random.gauss(0, 0.1),
                v_sigma=v_sigma,
                dir_sigma=math.pi / dir_division
            )
            self.person_trajectories.append(person_trajectory)

        logger.info(f'set crowd setting - person num: {person_num}, time range: ({start_time}, {end_time}), v: {v}')
        sim_total_time = end_time - start_time
        dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        trj_list = []
        pbar = tqdm(total=person_num)
        for pt in self.person_trajectories:
            pt.generate_trajectory(simulation_total_time=sim_total_time)
            start_time = dt + timedelta(seconds=pt.start_time)
            line_string = pt.trajectory
            trj_list.append({'id': pt.pid, 'start_time': start_time, 'geom': line_string})
            pbar.update(1)
        self.crowd_trj_df = pd.DataFrame(trj_list)
        logger.info('crowd trajectory generated.')
        return

    def to_csv(self, filename):
        assert self.crowd_trj_df is not None, 'No trajectory data to save. Please run set_crowd_trajectory() first.'
        dir_name = os.path.dirname(filename)
        os.makedirs(dir_name, exist_ok=True)
        self.crowd_trj_df.to_csv(filename, index=False)
        logger.info(f'crowd trajectory saved to {filename}')
        # if shp_flag:
        #     shp_name = f'{dir_name}/roi1_random.shp'
        #     roi_df = pd.DataFrame({'id':0, 'geom': self.room_polygon})
        #     gpd.GeoDataFrame(roi_df, geometry='geom').to_file(shp_name)
        #     logger.info(f'room polygon saved to {shp_name}')
        return

    def create_video(self, filename, fps=1, width=800, height=800, dt=3.0, tail_alpha=0.7, mov_speed=10.0):
        assert self.crowd_trj_df is not None, 'No trajectory data to visualize.'

        df = self.crowd_trj_df.copy()
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['geom'] = df['geom'].apply(lambda g: wkt.loads(g) if isinstance(g, str) else g)

        if len(df) == 0:
            raise ValueError('crowd_trj_df is empty.')

        if fps <= 0:
            raise ValueError('fps must be greater than 0.')

        if dt < 0:
            raise ValueError('dt must be non-negative.')

        base_time = df['start_time'].min()

        person_data = []
        for _, row in df.iterrows():
            geom = row['geom']
            coords = np.asarray(geom.coords, dtype=float)
            if len(coords) == 0:
                continue

            start_sec = (row['start_time'] - base_time).total_seconds()
            duration = max(len(coords) - 1, 0)

            person_data.append({
                'pid': row['id'],
                'start_sec': start_sec,
                'end_sec': start_sec + duration,
                'coords': coords
            })
        logger.info(f'creating video - {filename} (person num: {len(person_data)}, fps: {fps}, dt: {dt})')

        if len(person_data) == 0:
            raise ValueError('No valid trajectory data found.')

        total_duration = max(p['end_sec'] for p in person_data)

        def point_at_time(coords, rel_t):
            """
            coords: shape (N, 2)
            rel_t : trajectory-relative time [sec]
            """
            if len(coords) == 1:
                return coords[0]

            rel_t = max(0.0, min(rel_t, len(coords) - 1))
            i0 = int(np.floor(rel_t))
            i1 = min(i0 + 1, len(coords) - 1)

            if i0 == i1:
                return coords[i0]

            w = rel_t - i0
            return (1.0 - w) * coords[i0] + w * coords[i1]

        def tail_coords(coords, t0, t1):
            """
            軌跡の [t0, t1] 秒区間を切り出した折れ線座標を返す
            """
            if t1 < t0:
                return None

            if len(coords) == 1:
                return coords[[0]]

            start_pt = point_at_time(coords, t0)
            end_pt = point_at_time(coords, t1)

            points = [start_pt]

            mid_start = int(np.floor(t0)) + 1
            mid_end = int(np.ceil(t1)) - 1
            if mid_start <= mid_end:
                for idx in range(mid_start, mid_end + 1):
                    points.append(coords[idx])

            points.append(end_pt)

            out = np.asarray(points, dtype=float)

            # 連続同一点の重複を除去
            if len(out) >= 2:
                keep = [0]
                for i in range(1, len(out)):
                    if not np.allclose(out[i], out[keep[-1]]):
                        keep.append(i)
                out = out[keep]

            return out

        dpi = 100
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

        # room polygon
        room_x, room_y = self.room_polygon.exterior.xy
        ax.plot(room_x, room_y, color='black', linewidth=1.5)

        minx, miny, maxx, maxy = self.room_polygon.bounds
        pad_x = max((maxx - minx) * 0.05, 1.0e-6)
        pad_y = max((maxy - miny) * 0.05, 1.0e-6)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)

        title_text = ax.set_title('')

        # person artists
        line_artists = []
        head_artists = []
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(person_data), 20)))

        for i, p in enumerate(person_data):
            color = colors[i % len(colors)]
            line, = ax.plot([], [], '-', color=color, alpha=tail_alpha, linewidth=2)
            head, = ax.plot([], [], 'o', color=color, markersize=4)
            line_artists.append(line)
            head_artists.append(head)
        frame_num = int(np.ceil(total_duration * fps)) + 1

        pbar = tqdm(total=frame_num)
        def update(frame_idx):
            current_t = frame_idx / fps

            for i, p in enumerate(person_data):
                if current_t < p['start_sec'] or current_t > p['end_sec']:
                    line_artists[i].set_data([], [])
                    head_artists[i].set_data([], [])
                    continue

                rel_t1 = current_t - p['start_sec']
                rel_t0 = max(0.0, rel_t1 - dt)

                trj = tail_coords(p['coords'], rel_t0, rel_t1)
                if trj is None or len(trj) == 0:
                    line_artists[i].set_data([], [])
                    head_artists[i].set_data([], [])
                    continue

                line_artists[i].set_data(trj[:, 0], trj[:, 1])
                head_artists[i].set_data([trj[-1, 0]], [trj[-1, 1]])

            current_dt = base_time + timedelta(seconds=current_t)
            title_text.set_text(current_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            pbar.update(1)
            return [title_text] + line_artists + head_artists

        logger.info(f'starting animation (total_duration: {total_duration:.2f} sec, frame_num: {frame_num})')
        anim = FuncAnimation(
            fig,
            update,
            frames=frame_num,
            interval=1000 / fps,
            blit=True
        )

        dir_name = os.path.dirname(filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        writer = FFMpegWriter(fps=fps)
        anim.save(filename, writer=writer)
        plt.close(fig)
        logger.info(f'saved video - {filename}')
        if mov_speed != 1.0:
            logger.info(f'moving video with speed {mov_speed}x')
            os.rename(filename, f'{filename}.tmp')
            change_speed_ffmpeg(f'{filename}.tmp', filename, mov_speed)
            os.remove(f'{filename}.tmp')
        return

def test_crowd_sim(person_num=1000, room_size=100):
    crowd_traj = CrowdTrajectory(room_size=room_size)
    crowd_traj.set_crowd_trajectory(
        person_num=person_num,
        start_time=0,
        end_time=600,
        v=1.5,
        v_sigma=0.3,
        dir_division=8,
        datetime_str='2024-01-01 00:00:00'
    )
    crowd_traj.to_csv(f'./workspace/random_traj{person_num}.csv')
    crowd_traj.create_video(f'./workspace/random_traj{person_num}.mp4')

    return

def test():
    crowd_list = Crowd.csv_to_crowd_list('./workspace/gis/test_light.csv')
    # df = pd.read_csv('./workspace/gis/test_light.csv')
    # df['geom'] = df['geom'].apply(lambda x: wkt.loads(x))
    # df['start_time'] = pd.to_datetime(df['start_time'])
    # st = df['start_time'].min()
    # df['t'] = df['start_time'].apply(lambda x: (x - st).total_seconds())
    #
    # path = wkt.loads(df['geom'].loc[0])
    # start_time = 0
    # height = 1.8
    # foot_step = height * 0.35
    # print('id: ', df['id'].loc[0])
    # print('foot_step: ', foot_step)
    # c = Crowd(path, start_time, height, foot_step)
    crowd_sim = CrowdSim.from_shp('./workspace/gis/room/marunouchi.shp')
    crowd_sim.set_crowd(crowd_list)
    room_center = crowd_sim.room_info['corners'].mean(axis=1)
    crowd_sim.set_microphone(np.array([[room_center[0]-0.01, room_center[1], 0.8],
                                       [room_center[0]+0.01, room_center[1], 0.8]]))
    signals = crowd_sim.simulation()
    signals = signals / signals.max()
    wavfile.write('./workspace/gis/marunouchi_footstep.wav', SR, signals.T)
    return


class SocialForcePerson:
    def __init__(
        self,
        person_id: int,
        position: tuple | list | np.ndarray,
        initial_velocity: tuple | list | np.ndarray,
        start_time: int | float,
        goal: tuple | list | np.ndarray = None,
        mass: float = 80.0,
        desired_speed: float = 1.3,
        relaxation_time:float = 0.5,
        radius: float = 0.3,
    ):
        """
        Social Force Modelの一人分の移動軌跡を生成するクラス
        Parameters
        ----------
        person_id: int
            人のID
        position: tuple or list or np.ndarray
            初期位置 (x, y)
        initial_velocity: tuple or list or np.ndarray
            初期速度 (vx, vy)
        start_time: int or float
            シミュレーション開始時間 (s)
        goal: tuple or list or np.ndarray or None
            目的地の位置 (x, y)。Noneの場合は目的地なしとする
        mass: float
            人の質量 (kg)
        desired_speed: float
            目的地に向かうときの希望速度 (m/s)
        relaxation_time: float
            目的地に向かうときの加速の速さを表すパラメータ (s)
        radius: float
            人の半径 (m)
        """
        self.person_id = person_id
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(initial_velocity, dtype=float)
        self.goal = None if goal is None else np.array(goal, dtype=float)

        self.start_time = start_time
        self.is_active = False
        self.is_finished = False

        self.mass = mass
        self.desired_speed = desired_speed
        self.relaxation_time = relaxation_time
        self.radius = radius

        self.trajectory = []

        # stuck判定用: 前回の位置を保存する
        self.prev_position = self.position.copy()
        self.stuck_duration = 0.0
        self.finish_reason = None
        self.motion_history = []

    def desired_direction(self):
        """
        目的地がある場合は目的地に向かう単位ベクトルを返す。ない場合は現在の速度の単位ベクトルを返す。
        Returns
        -------
        result: np.ndarray
            目的地がある場合は目的地に向かう単位ベクトル、ない場合は現在の速度の単位ベクトル
        """
        if self.goal is not None:
            direction = self.goal - self.position
        else:
            direction = self.velocity

        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return np.zeros(2)

        return direction / norm

    def driving_force(self):
        """
        Social Force Modelの駆動力を計算する
        Returns
        -------
        result: np.ndarray
            駆動力ベクトル
        """
        desired_velocity = self.desired_speed * self.desired_direction()
        return self.mass * (desired_velocity - self.velocity) / self.relaxation_time

    def interaction_force(
            self,
            others: list['SocialForcePerson'] = None,
            c_obs=2000.0,
            r_obs=0.08
    ):
        """
        他の人との相互作用力を計算する
        Parameters
        ----------
        others: list of SocialForcePerson
            他の人のリスト
        c_obs: float
            他の人との距離が0のときの反発力の大きさを表すパラメータ
        r_obs: float
            他の人との距離が大きくなると反発力が減少する速さを表すパラメータ

        Returns
        -------

        """
        force = np.zeros(2)
        if others is None:
            return force

        for other in others:
            if other is self:
                continue
            if not other.is_active or other.is_finished:
                continue

            diff = self.position - other.position
            distance = np.linalg.norm(diff)

            if distance < 1e-8:
                continue

            direction = diff / distance
            combined_radius = self.radius + other.radius

            force += c_obs * np.exp((combined_radius - distance) / r_obs) * direction

        return force

    def wall_force(
            self,
            walls: Polygon | MultiPolygon = None,
            c_wall: float = 500.0,
            r_wall: float = 0.7,
            cutoff_distance: float = 3.0,
    ):
        if walls is None:
            return np.zeros(2)

        point = Point(float(self.position[0]), float(self.position[1]))
        total_force = np.zeros(2)

        if isinstance(walls, Polygon):
            wall_polygons = [walls]
        elif isinstance(walls, MultiPolygon):
            wall_polygons = list(walls.geoms)
        else:
            return total_force

        for wall_poly in wall_polygons:
            boundary_lines = [wall_poly.exterior] + list(wall_poly.interiors)

            for boundary in boundary_lines:
                _, nearest_wall_point = nearest_points(point, boundary)

                nearest = np.array(
                    [nearest_wall_point.x, nearest_wall_point.y],
                    dtype=float,
                )

                diff = self.position - nearest
                distance = np.linalg.norm(diff)

                if distance < 1e-8:
                    continue

                if distance > cutoff_distance:
                    continue

                direction = diff / distance

                force = c_wall * np.exp(
                    (self.radius - distance) / r_wall
                ) * direction

                total_force += force

        return total_force

    def update(
            self,
            dt: float = 0.01,
            others: list['SocialForcePerson'] = None,
            walls: Polygon | MultiPolygon = None,
            c_obs: float = 2000.0,
            r_obs: float = 0.08,
            c_wall: float = 2000.0,
            r_wall: float = 0.08
    ):
        """
        次の時間ステップにおける位置と速度を更新する
        Parameters
        ----------
        dt: float
            時間ステップ (s)
        others: list of SocialForcePerson
            他の人のリスト
        walls: Polygon or MultiPolygon or None
            壁のポリゴン。エージェントはこの壁に当たらないように移動する。Noneの場合は壁なしとする
        c_obs: float
            他の人との距離が0のときの反発力の大きさを表すパラメータ
        r_obs: float
            他の人との距離が大きくなると反発力が減少する速さを表すパラメータ
        c_wall: float
            壁に近いほど強い反発力の大きさを表すパラメータ
        r_wall: float
            壁に近いほど強い反発力の減少する速さを表すパラメータ

        Returns
        -------

        """
        if not self.is_active or self.is_finished:
            return

        total_force = self.driving_force()
        total_force += self.interaction_force(others, c_obs=c_obs, r_obs=r_obs)
        total_force += self.wall_force(walls, c_wall=c_wall, r_wall=r_wall)

        acceleration = total_force / self.mass
        acc_norm = np.linalg.norm(acceleration)
        if acc_norm > MAX_ACC:
            acceleration = acceleration / acc_norm * MAX_ACC
        self.velocity += acceleration * dt
        speed = np.linalg.norm(self.velocity)
        if speed > MAX_SPEED:
            self.velocity = self.velocity / speed * MAX_SPEED
        self.position += self.velocity * dt

        self.project_out_of_walls(walls)

    def record_trajectory(self, current_time, absolute_time):
        res = {
            "pid": self.person_id,
            "time": current_time,
            "datetime": absolute_time,
            "x": self.position[0],
            "y": self.position[1],
            "vx": self.velocity[0],
            "vy": self.velocity[1]
        }
        if self.goal is not None:
            res |= {
                "goal_x": self.goal[0],
                "goal_y": self.goal[1]
            }
        self.trajectory.append(res)

    def project_out_of_walls(self, walls, margin=0.05):
        if walls is None:
            return

        point = Point(float(self.position[0]), float(self.position[1]))

        if not walls.contains(point):
            return

        nearest_self, nearest_wall = nearest_points(point, walls.boundary)

        nearest = np.array(
            [nearest_wall.x, nearest_wall.y],
            dtype=float,
        )

        diff = self.position - nearest
        norm = np.linalg.norm(diff)

        if norm < 1e-8:
            # 境界上で法線が取れない場合は、現在速度の逆向きへ戻す
            v_norm = np.linalg.norm(self.velocity)
            if v_norm < 1e-8:
                normal = np.array([1.0, 0.0])
            else:
                normal = -self.velocity / v_norm
        else:
            normal = diff / norm

        # 壁の外へ少し余裕を持って押し出す
        self.position = nearest + normal * (self.radius + margin)

        # 壁へ向かう速度成分を消す
        vn = np.dot(self.velocity, normal)
        if vn < 0:
            self.velocity -= vn * normal


class SocialForceSimulation:
    def __init__(
        self,
        roi_polygon: Polygon,
        wall: Polygon | MultiPolygon = None,
        dt: float | int = 1,
        desired_speed: float = 1.3,
        velocity_noise_std=0.2,
        c_obs: float = 2000.0,
        r_obs: float = 0.08,
        c_wall: float = 2000.0,
        r_wall: float = 0.08,
        stuck_speed_threshold: float = 0.3,
        stuck_time_threshold: float = 5.0,
        wall_oscillation_distance: float = 1.0,
        wall_oscillation_time_window: float = 8.0,
        wall_oscillation_min_path_length: float = 3.0,
        wall_oscillation_efficiency_threshold: float = 0.25,
        wall_oscillation_min_reverse_count: int = 2,
        wall_oscillation_min_projection_count: int = 2,
        seed: int = 0,
    ):
        """
        Social Force Modelのシミュレーションを実行するクラス
        Parameters
        ----------
        roi_polygon: Polygon
            シミュレーションの関心領域を表すPolygon。エージェントはこの領域内にいる間だけ移動する。
            領域外に出たエージェントはシミュレーションから除外される
        wall: Polygon or MultiPolygon or None
            壁・障害物を表すPolygonまたはMultiPolygon。Noneの場合は壁なしとする
        dt: float or int
            シミュレーションの時間刻み [s]
        desired_speed: float
            エージェントの希望速度 [m/s]
        velocity_noise_std: float
            エージェントの初期速度に加えるノイズの標準偏差
        c_obs: float
            他の人との距離が0のときの反発力の大きさを表すパラメータ
        r_obs: float
            他の人との距離が大きくなると反発力が減少する速さを表すパラメータ
        c_wall: float
            壁に近いほど強い反発力の大きさを表すパラメータ
        r_wall: float
            壁に近いほど強い反発力の減少する速さを表すパラメータ
        stuck_speed_threshold: float
            stuck判定のための速度閾値。速度がこの値以下で一定時間以上続くとstuckと判定される
        stuck_time_threshold: float
            stuck判定のための時間閾値。速度がstuck_speed_threshold以下でこの時間以上続くとstuckと判定される
        wall_oscillation_distance: float
            壁振動判定で「壁の近く」とみなす壁からの距離 [m]
        wall_oscillation_time_window: float
            壁振動判定に使う直近履歴の時間幅 [s]
        wall_oscillation_min_path_length: float
            壁振動判定に必要な直近履歴内の最小移動距離 [m]
        wall_oscillation_efficiency_threshold: float
            直近履歴の正味移動距離 / 経路長。この値以下なら往復運動とみなす
        wall_oscillation_min_reverse_count: int
            直近履歴内で必要な進行方向反転回数
        wall_oscillation_min_projection_count: int
            直近履歴内で必要なfree_areaへの投影回数
        seed: int or None
            乱数シード。Noneの場合はランダムなシードを使用
        """
        self.roi_area = orient(roi_polygon, sign=1.0)
        self.walls = wall

        self.dt = dt
        self.desired_speed = desired_speed
        self.velocity_noise_std = velocity_noise_std

        self.rng = np.random.default_rng(seed)
        self.sfm_params = {
            'c_obs': c_obs,
            'r_obs': r_obs,
            'c_wall': c_wall,
            'r_wall': r_wall,
        }
        self.persons = []
        self.crowd_trj_df = None

        self.stuck_speed_threshold = stuck_speed_threshold
        self.stuck_time_threshold = stuck_time_threshold
        self.wall_oscillation_distance = wall_oscillation_distance
        self.wall_oscillation_time_window = wall_oscillation_time_window
        self.wall_oscillation_min_path_length = wall_oscillation_min_path_length
        self.wall_oscillation_efficiency_threshold = wall_oscillation_efficiency_threshold
        self.wall_oscillation_min_reverse_count = wall_oscillation_min_reverse_count
        self.wall_oscillation_min_projection_count = wall_oscillation_min_projection_count

        clearance = 0.35  # person radius + margin 相当
        self.free_area = self.roi_area

        if self.walls is not None:
            self.free_area = self.roi_area.difference(
                self.walls.buffer(clearance)
            )

    def _check_stuck(self, person):
        """
        連続して stuck_time_threshold 秒以上、移動速度が小さい場合に除外する。
        """

        displacement = np.linalg.norm(person.position - person.prev_position)
        speed = displacement / max(self.dt, 1.0e-8)

        if speed < self.stuck_speed_threshold:
            person.stuck_duration += self.dt
        else:
            person.stuck_duration = 0.0

        person.prev_position = person.position.copy()

        if person.stuck_duration >= self.stuck_time_threshold:
            person.is_active = False
            person.is_finished = True
            person.finish_reason = "stuck"
            return True

        return False

    def _motion_history_limit(self):
        return max(
            3,
            int(math.ceil(self.wall_oscillation_time_window / max(self.dt, 1.0e-8))) + 1,
        )

    def _append_motion_history(self, person, projected_to_free_area=False):
        if not hasattr(person, 'motion_history'):
            person.motion_history = []

        wall_distance = math.inf
        if self.walls is not None:
            point = Point(float(person.position[0]), float(person.position[1]))
            wall_distance = float(self.walls.distance(point))

        person.motion_history.append(
            {
                'position': person.position.copy(),
                'wall_distance': wall_distance,
                'projected_to_free_area': bool(projected_to_free_area),
            }
        )

        max_samples = self._motion_history_limit()
        if len(person.motion_history) > max_samples:
            person.motion_history = person.motion_history[-max_samples:]

    def _check_wall_oscillation(self, person):
        """
        壁近傍で往復し続けるエージェントを除外する。

        stuck判定は低速停止だけを見るため、速度はあるが同じ壁際を
        行き来するケースをここで別途検出する。
        """

        if self.walls is None:
            return False

        history = getattr(person, 'motion_history', [])
        min_samples = max(4, min(6, self._motion_history_limit()))
        if len(history) < min_samples:
            return False

        positions = np.asarray([sample['position'] for sample in history], dtype=float)
        wall_distances = np.asarray([sample['wall_distance'] for sample in history], dtype=float)
        projection_count = sum(sample['projected_to_free_area'] for sample in history)

        near_wall_count = int(np.sum(wall_distances <= self.wall_oscillation_distance))
        near_wall_required = max(2, len(history) // 2)
        if (
            near_wall_count < near_wall_required
            and projection_count < self.wall_oscillation_min_projection_count
        ):
            return False

        diffs = np.diff(positions, axis=0)
        step_lengths = np.linalg.norm(diffs, axis=1)
        path_length = float(np.sum(step_lengths))
        if path_length < self.wall_oscillation_min_path_length:
            return False

        net_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
        efficiency = net_displacement / max(path_length, 1.0e-8)
        if efficiency > self.wall_oscillation_efficiency_threshold:
            return False

        moving = step_lengths > 1.0e-8
        if int(np.sum(moving)) < 2:
            return False

        directions = diffs[moving] / step_lengths[moving, None]
        reverse_count = int(np.sum(np.sum(directions[:-1] * directions[1:], axis=1) < -0.3))
        if reverse_count < self.wall_oscillation_min_reverse_count:
            return False

        person.is_active = False
        person.is_finished = True
        person.finish_reason = 'wall_oscillation'
        return True

    def _sample_boundary_point_and_inward_normal(self):
        """
        関心領域の境界上のランダムな点と、その点における内向き法線ベクトルをサンプリングする
        Returns
        -------
        position: np.ndarray
            境界上の点の座標 (x, y)
        inward_normal: np.ndarray
            境界上の点における内向き法線ベクトル (nx, ny)
        """
        exterior = self.roi_area.exterior
        boundary_length = exterior.length

        distance_on_boundary = self.rng.uniform(0, boundary_length)
        point = exterior.interpolate(distance_on_boundary)

        coords = list(exterior.coords)

        accumulated = 0.0
        selected_segment = None

        for p1, p2 in zip(coords[:-1], coords[1:]):
            p1 = np.array(p1, dtype=float)
            p2 = np.array(p2, dtype=float)

            segment = p2 - p1
            segment_length = np.linalg.norm(segment)

            if accumulated + segment_length >= distance_on_boundary:
                selected_segment = segment
                break

            accumulated += segment_length

        if selected_segment is None:
            selected_segment = np.array(coords[1]) - np.array(coords[0])

        tangent = selected_segment / np.linalg.norm(selected_segment)

        # Polygonを反時計回りにしているので、進行方向左側が内側
        inward_normal = np.array([-tangent[1], tangent[0]])

        position = np.array([point.x, point.y], dtype=float)

        return position, inward_normal

    def _sample_goal_point_from_start(
            self,
            start_point: np.ndarray,
            initial_velocity: np.ndarray,
            min_d_ratio=0.25
    ):
        """
        目的地の位置を、関心領域の境界上のスタート位置から一定距離以上離れた位置からランダムにサンプリングする
        Parameters
        ----------
        start_point: np.ndarray
            スタート位置の座標 (x, y)
        initial_velocity: np.ndarray
            スタート位置における初期速度ベクトル (vx, vy
        min_d_ratio: float
            目的地がスタート位置から少なくともこの割合以上離れるようにするためのパラメータ。0.0以上0.5未満の値を指定する。
            例えば0.25を指定した場合、目的地はスタート位置から関心領域の外周に沿った距離がスタート位置から外周全体の距離の25%以上、
            75%以下の位置からランダムにサンプリングされる。

        Returns
        -------
        goal_point: Point
            目的地の座標 (x, y)
        """
        start_point = Point(float(start_point[0]), float(start_point[1]))
        exterior = self.roi_area.exterior
        if self.walls is None:
            # 壁がない場合は、スタート位置から初期速度方向へ伸ばした直線と外周の交点を目的地とする
            ext_len = exterior.length
            vector_line = LineString(
                [
                    (start_point.x, start_point.y),
                    (start_point.x + initial_velocity[0] * ext_len, start_point.y + initial_velocity[1] * ext_len)
                ]
            )
            vector_line = vector_line.difference(start_point.buffer(0.01))
            goal_point = exterior.intersection(vector_line)
            if goal_point.is_empty:
                raise Exception('Failed to sample goal point: no intersection between vector line and exterior')
            if isinstance(goal_point, Point):
                return goal_point.coords[0]
            elif isinstance(goal_point, MultiPoint):
                # 複数の交点がある場合は、スタート位置から最も遠い交点を選ぶ
                max_dist = -1.0
                selected_point = None
                for geom in goal_point.geoms:
                    if isinstance(geom, Point):
                        dist = start_point.distance(geom)
                        if dist > max_dist:
                            max_dist = dist
                            selected_point = geom
                if selected_point is not None:
                    return selected_point.coords[0]
                else:
                    raise Exception('Failed to sample goal point: no valid intersection point found')
            else:
                raise Exception('Failed to sample goal point: unexpected geometry type for intersection result')
        else:
            # 壁がない領域の外周を取得する
            exterior = exterior.difference(self.walls)
            ext_len = exterior.length
            start_ratio = exterior.project(start_point) / ext_len
            # 外周領域から、スタート位置から一定距離以上離れた位置をランダムにサンプリングする
            goal_ratio = (start_ratio + min_d_ratio + self.rng.uniform(0, 1 - min_d_ratio * 2)) % 1.0
            goal_point = exterior.interpolate(goal_ratio * ext_len)
            return goal_point.coords[0]

    def _generate_initial_velocity(self, inward_normal):
        """
        初期速度を生成する。基本的には内向き法線方向に希望速度を持たせるが、ノイズも加える
        Parameters
        ----------
        inward_normal: np.ndarray
            境界上の点における内向き法線ベクトル (nx, ny)

        Returns
        -------
        initial_velocity: np.ndarray
            初期速度ベクトル (vx, vy)
        """
        noise = self.rng.normal(
            loc=0.0,
            scale=self.velocity_noise_std,
            size=2,
        )

        direction = inward_normal + noise
        norm = np.linalg.norm(direction)

        if norm < 1e-8:
            direction = inward_normal
        else:
            direction = direction / norm

        return self.desired_speed * direction

    def _create_persons(
            self,
            person_num: int,
            simulation_time: float,
            walls: Polygon | MultiPolygon = None,
            goal_flag: bool = False
    ):
        """
        初期位置を関心領域の境界上からランダムにサンプリングし、初期速度を内向き法線方向に希望速度を持たせて生成する
        Parameters
        ----------
        person_num: int
            発生させるエージェント数
        simulation_time: float
            シミュレーション秒数 [s]
        walls: Polygon or MultiPolygon or None
            壁のポリゴン。エージェントはこの壁に当たらないように移動する。Noneの場合は壁なしとする
        goal_flag: bool
            目的地を設定するかどうかのフラグ。Trueの場合は、関心領域内のランダムな点を目的地として設定する。Falseの場合は目的地なしとする

        Returns
        -------

        """
        if walls is None:
            walls = self.walls
        self.persons = []

        start_time = self.rng.uniform(
            low=0.0,
            high=simulation_time,
            size=person_num,
        )

        start_time.sort()
        start_time = start_time.tolist()

        for person_id in range(person_num):
            position, inward_normal = self._sample_boundary_point_and_inward_normal()

            if walls is not None:
                point = Point(float(position[0]), float(position[1]))
                if walls.contains(point):
                    # 壁の内側にサンプリングされてしまった場合は、壁の外側にサンプリングされるまで再サンプリングする
                    while True:
                        position, inward_normal = self._sample_boundary_point_and_inward_normal()
                        point = Point(float(position[0]), float(position[1]))
                        if not walls.contains(point):
                            break

            ini_vel = self._generate_initial_velocity(inward_normal)

            person = SocialForcePerson(
                person_id=person_id,
                position=position,
                initial_velocity=ini_vel,
                start_time=start_time[person_id],
                goal=None if not goal_flag else self._sample_goal_point_from_start(position, ini_vel),
                desired_speed=self.desired_speed,
            )

            self.persons.append(person)

    def _project_to_free_area(self, person):
        point = Point(float(person.position[0]), float(person.position[1]))

        if self.free_area.covers(point):
            return False

        _, nearest_free = nearest_points(point, self.free_area)

        new_pos = np.array(
            [nearest_free.x, nearest_free.y],
            dtype=float,
        )

        move_back = new_pos - person.position
        norm = np.linalg.norm(move_back)

        person.position = new_pos

        # 壁内へ突っ込む速度を弱める
        if norm > 1e-8:
            normal = move_back / norm
            vn = np.dot(person.velocity, normal)

            if vn < 0:
                person.velocity -= vn * normal

        person.velocity *= 0.3
        return True

    def step(self, current_time, absolute_time):
        """
        1ステップ分シミュレーションを進める
        Parameters
        ----------
        current_time: float
            シミュレーション開始からの経過時間 (s)
        absolute_time: datetime
            シミュレーション開始日時 + current_time

        Returns
        -------
        None
        """

        for person in self.persons:
            if person.is_finished:
                continue

            if (not person.is_active) and current_time >= person.start_time:
                person.is_active = True

            if not person.is_active:
                continue

            person.update(
                dt=self.dt,
                others=self.persons,
                walls=self.walls,
                c_obs=self.sfm_params['c_obs'],
                r_obs=self.sfm_params['r_obs'],
                c_wall=self.sfm_params['c_wall'],
                r_wall=self.sfm_params['r_wall'],
            )
            projected_to_free_area = self._project_to_free_area(person)
            self._append_motion_history(person, projected_to_free_area)

            if self._check_wall_oscillation(person):
                continue

            if self._check_stuck(person):
                continue

            point = Point(person.position[0], person.position[1])

            if not self.roi_area.contains(point):
                person.is_active = False
                person.is_finished = True
                person.finish_reason = "out_of_roi"
                continue

            person.record_trajectory(
                current_time=current_time,
                absolute_time=absolute_time,
            )

    def run(self, person_num, simulation_time, start_time, goal_flag=False):
        """

        Parameters
        ----------
        person_num: int
            発生させるエージェント数
        simulation_time: float
            シミュレーション秒数 [s]
        start_time: datetime or str
            シミュレーション開始日時 (datetime型または文字列)
        goal_flag: bool
            目的地を設定するかどうかのフラグ。Trueの場合は、関心領域内のランダムな点を目的地として設定する。Falseの場合は目的地なしとする

        Returns
        -------

        """

        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)

        logger.info(f'starting simulation - simulation_time: {simulation_time:.2f} sec, start_time: {start_time}')
        self._create_persons(
            person_num=person_num,
            simulation_time=simulation_time,
            walls=self.walls,
            goal_flag=goal_flag,
        )
        logger.info(f'created {len(self.persons)} persons')

        steps = int(simulation_time / self.dt)

        # tqdmを使用してシミュレーションの進行状況を表示
        pbar = tqdm(total=steps, desc='Simulating', unit='step')
        for step_idx in range(steps):
            current_time = step_idx * self.dt
            absolute_time = start_time + timedelta(seconds=current_time)

            self.step(
                current_time=current_time,
                absolute_time=absolute_time,
            )
            pbar.update(1)
        pbar.close()
        logger.info('simulation finished')

        records = []
        for person in self.persons:
            records.extend(person.trajectory)
        trajectory_df = pd.DataFrame(records)
        trj_list = [
            {
                'id': pid,
                'start_time': min(grp_df['datetime']),
                'geom': LineString(grp_df.sort_values(by='datetime')[['x', 'y']].values),
                'goal': Point(grp_df[['goal_x', 'goal_y']].iloc[0]) if 'goal_x' in grp_df.columns and 'goal_y' in grp_df.columns else None
            }
            for pid, grp_df in trajectory_df.groupby('pid') if len(grp_df) > 1
        ]

        self.crowd_trj_df = pd.DataFrame(trj_list)
        return self.crowd_trj_df

    def to_csv(self, filename):
        assert self.crowd_trj_df is not None, 'No trajectory data to save. Please run SocialForceSimulation.run() first.'
        dir_name = os.path.dirname(filename)
        os.makedirs(dir_name, exist_ok=True)
        self.crowd_trj_df.to_csv(filename, index=False)
        logger.info(f'crowd trajectory saved to {filename}')
        return

    def create_video(self, filename, fps=10, width=800, height=800, dt=3.0, tail_alpha=0.7, mov_speed=10.0):
        assert self.crowd_trj_df is not None, 'No trajectory data to visualize.'

        df = self.crowd_trj_df.copy()
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['geom'] = df['geom'].apply(lambda g: wkt.loads(g) if isinstance(g, str) else g)

        if len(df) == 0:
            raise ValueError('crowd_trj_df is empty.')

        if fps <= 0:
            raise ValueError('fps must be greater than 0.')

        if dt < 0:
            raise ValueError('dt must be non-negative.')

        base_time = df['start_time'].min()

        person_data = []
        for _, row in df.iterrows():
            geom = row['geom']
            coords = np.asarray(geom.coords, dtype=float)
            if len(coords) == 0:
                continue

            start_sec = (row['start_time'] - base_time).total_seconds()
            duration = max(len(coords) - 1, 0)

            person_data.append({
                'pid': row['id'],
                'start_sec': start_sec,
                'end_sec': start_sec + duration,
                'coords': coords
            })
        logger.info(f'creating video - {filename} (person num: {len(person_data)}, fps: {fps}, dt: {dt})')

        if len(person_data) == 0:
            raise ValueError('No valid trajectory data found.')

        total_duration = max(p['end_sec'] for p in person_data)

        def point_at_time(coords, rel_t):
            """
            coords: shape (N, 2)
            rel_t : trajectory-relative time [sec]
            """
            if len(coords) == 1:
                return coords[0]

            rel_t = max(0.0, min(rel_t, len(coords) - 1))
            i0 = int(np.floor(rel_t))
            i1 = min(i0 + 1, len(coords) - 1)

            if i0 == i1:
                return coords[i0]

            w = rel_t - i0
            return (1.0 - w) * coords[i0] + w * coords[i1]

        def tail_coords(coords, t0, t1):
            """
            軌跡の [t0, t1] 秒区間を切り出した折れ線座標を返す
            """
            if t1 < t0:
                return None

            if len(coords) == 1:
                return coords[[0]]

            start_pt = point_at_time(coords, t0)
            end_pt = point_at_time(coords, t1)

            points = [start_pt]

            mid_start = int(np.floor(t0)) + 1
            mid_end = int(np.ceil(t1)) - 1
            if mid_start <= mid_end:
                for idx in range(mid_start, mid_end + 1):
                    points.append(coords[idx])

            points.append(end_pt)

            out = np.asarray(points, dtype=float)

            # 連続同一点の重複を除去
            if len(out) >= 2:
                keep = [0]
                for i in range(1, len(out)):
                    if not np.allclose(out[i], out[keep[-1]]):
                        keep.append(i)
                out = out[keep]

            return out

        dpi = 100
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

        # ROI polygon
        room_x, room_y = self.roi_area.exterior.xy
        ax.plot(room_x, room_y, color='black', linewidth=1.5, label='ROI')

        # Wall polygons
        if self.walls is not None:
            if isinstance(self.walls, Polygon):
                wall_geoms = [self.walls]
            elif isinstance(self.walls, MultiPolygon):
                wall_geoms = list(self.walls.geoms)
            else:
                wall_geoms = []

            for wall_poly in wall_geoms:
                wx, wy = wall_poly.exterior.xy
                ax.fill(
                    wx,
                    wy,
                    facecolor='gray',
                    edgecolor='black',
                    alpha=0.5,
                    linewidth=1.0,
                    zorder=1,
                )

                # 穴があるPolygonの場合
                for interior in wall_poly.interiors:
                    ix, iy = interior.xy
                    ax.fill(
                        ix,
                        iy,
                        facecolor='white',
                        edgecolor='black',
                        alpha=1.0,
                        linewidth=1.0,
                        zorder=2,
                    )

        minx, miny, maxx, maxy = self.roi_area.bounds
        pad_x = max((maxx - minx) * 0.05, 1.0e-6)
        pad_y = max((maxy - miny) * 0.05, 1.0e-6)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)

        title_text = ax.set_title('')

        # person artists
        line_artists = []
        head_artists = []
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(person_data), 20)))

        for i, p in enumerate(person_data):
            color = colors[i % len(colors)]
            line, = ax.plot([], [], '-', color=color, alpha=tail_alpha, linewidth=2)
            head, = ax.plot([], [], 'o', color=color, markersize=4)
            line_artists.append(line)
            head_artists.append(head)
        frame_num = int(np.ceil(total_duration * fps)) + 1

        pbar = tqdm(total=frame_num)
        def update(frame_idx):
            current_t = frame_idx / fps

            for i, p in enumerate(person_data):
                if current_t < p['start_sec'] or current_t > p['end_sec']:
                    line_artists[i].set_data([], [])
                    head_artists[i].set_data([], [])
                    continue

                rel_t1 = current_t - p['start_sec']
                rel_t0 = max(0.0, rel_t1 - dt)

                trj = tail_coords(p['coords'], rel_t0, rel_t1)
                if trj is None or len(trj) == 0:
                    line_artists[i].set_data([], [])
                    head_artists[i].set_data([], [])
                    continue

                line_artists[i].set_data(trj[:, 0], trj[:, 1])
                head_artists[i].set_data([trj[-1, 0]], [trj[-1, 1]])

            current_dt = base_time + timedelta(seconds=current_t)
            title_text.set_text(current_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            pbar.update(1)
            return [title_text] + line_artists + head_artists

        logger.info(f'starting animation (total_duration: {total_duration:.2f} sec, frame_num: {frame_num})')
        anim = FuncAnimation(
            fig,
            update,
            frames=frame_num,
            interval=1000 / fps,
            blit=True
        )

        dir_name = os.path.dirname(filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        writer = FFMpegWriter(fps=fps)
        anim.save(filename, writer=writer)
        plt.close(fig)
        logger.info(f'saved video - {filename}')
        if mov_speed != 1.0:
            logger.info(f'moving video with speed {mov_speed}x')
            os.rename(filename, f'{filename}.tmp')
            change_speed_ffmpeg(f'{filename}.tmp', filename, mov_speed)
            os.remove(f'{filename}.tmp')
        return



def audio_crowd_simulation_(crowd_csv, room_shp, output_folder, mic_shp=None, snr=None, time_unit=10.0,
                            distance_list=None, grid_shp=None):
    signal_info = {}
    if distance_list is None:
        distance_list = [1, 10, 20, 30, 40, 50]
    signal_info['distance_list'] = distance_list

    crowd_list = Crowd.csv_to_crowd_list(crowd_csv)
    st_time, ed_time = min([c.start_time for c in crowd_list]), max([c.start_time for c in crowd_list])
    signal_info['simulation_time'] = [st_time, ed_time]
    logger.info(f'simulation time: {st_time} - {ed_time}')

    # print('Reading Simulation room shapefile.')
    crowd_sim = CrowdSim.from_shp(room_shp)
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
    crowd_path = f'{output_folder}/crowd.csv'
    logger.info(f'save crowd csv - {crowd_path}')
    crowd_sim.crowd_density_to_csv(crowd_path)
    signal_info['crowd_path'] = crowd_path
    for s in range(len(signals)):
        each_crowd = f'{output_folder}/crowd_mic{id_list[s]:04}.csv'
        logger.info(f'save crowd from mic{id_list[s]} - {each_crowd}')
        crowd_sim.crowd_density_from_each_mic(mic_index=id_list[s], csv_name=each_crowd, distance_list=distance_list)
        signal_info['info'][f'ch{s}'] |= {'each_crowd': each_crowd}

    with open(f'{output_folder}/signal_info.json', 'w') as f:
        json.dump(signal_info, f, indent=4)

    # with open(f'{output_folder}/log.json', 'w', encoding='utf-8') as f:
    #     json.dump(log_info, f, indent=4)
    return


"""
from simulation.util import *
room_size = 40
default_coords = ((-room_size / 2, -room_size / 2),
                  (-room_size / 2, room_size / 2),
                  (room_size / 2, room_size / 2),
                  (room_size / 2, -room_size / 2),
                  (-room_size / 2, -room_size / 2))
room_polygon = Polygon(default_coords)
wall_polygon1 = Polygon(((0, -10), (0, 10), (5, 10), (5, -10), (0, -10)))
wall_polygon2 = Polygon(((-10, -30), (-10, -10), (10, -15), (10, -30), (-10, -30)))
walls = MultiPolygon([wall_polygon1, wall_polygon2])
sfm = SocialForceSimulation(roi_polygon=room_polygon, wall=walls)
hoge_df = sfm.run(person_num=1000, simulation_time=100, start_time='2025-01-01 00:00:00')
sfm.create_video('./workspace/sfm_p10000_test.mp4', fps=10)

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--crowd-csv')
    parser.add_argument('-s', '--room-shp')
    parser.add_argument('-o', '--output-folder')
    parser.add_argument('-m', '--mic-shp')
    parser.add_argument('-g', '--grid_shp')
    parser.add_argument('-n', '--snr', default=None, type=float)
    parser.add_argument('-t', '--time-unit', default=10.0, type=float)
    parser.add_argument('-opt', '--option', choices=['footstep', 'env'],
                        default='footstep')
    parser.add_argument('-dl', '--distance-list', nargs='+', type=int, default=[1, 10, 20, 30, 40, 50])
    args = parser.parse_args()
    if args.option == 'footstep':
        audio_crowd_simulation_(args.crowd_csv, args.room_shp, args.output_folder, args.mic_shp, args.snr,
                                args.time_unit, args.distance_list, args.grid_shp)
