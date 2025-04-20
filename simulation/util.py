import glob
import json
import os
import random
import argparse
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
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import split
from joblib import Parallel, delayed
from tqdm import tqdm
from common.logger import get_logger


INPUT_DIR = './data/speech'
OUTPUT_DIR = './output'

CH = 'channels'
FS = 'sampling_rate'
SR = 16000
ROOM_SIZE = np.array([10.0, 10.0, 3.0])
MIC_INTERVAL = 0.02

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
                          name=str(dr['id']))
                      for _, dr in df.iterrows()]

        return crowd_list

    def __init__(self, path: LineString,
                 start_time: int | float,
                 height: float,
                 foot_step: float,
                 time_step: float = 1.0,
                 name=None):
        self.time_step = time_step
        self.path = path
        self.start_time = start_time
        self.height = height
        self.foot_step = foot_step
        self.duration = len(path.coords) * time_step
        self.name = name

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
        :param line:
        :param point:
        :return:
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
        return len(gc.geoms[0].coords) - 2 + tmp_div / tmp_length

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
    def __read_wav_file(filename, fs):
        _fs, signal = wavfile.read(filename)
        signal = signal.astype(float)
        if _fs != fs:
            signal = librosa.resample(y=signal, orig_sr=_fs, target_sr=fs)
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
    def from_shp(cls, shp_path):
        logger.info(msg=f'read shp - {shp_path}')
        df = gpd.read_file(shp_path)
        room_polygon = df['geometry'].loc[0]
        return cls(room_polygon=room_polygon, height=3.0)

    def __init__(self, room_polygon: Polygon, height=3.0):
        self.crowd_list = None
        self.footstep = []
        self.room_polygon = room_polygon
        room_coordinates = np.array(room_polygon.exterior.coords[:-1])
        self.room_info = {'corners': room_coordinates.T, 'fs': SR, 'max_order': 0}
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
        self.footstep = [crowd.get_foot_points() for crowd in tqdm(self.crowd_list, desc='[CrowdSet')]
        foot_tags = [self.foot_sound.get_tags()[0] for _ in self.crowd_list]
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
        index: index of person

        Returns audio signal
        -------
        # TODO 一人の移動時間が長い場合にメモリエラーが出るため、pyroomacousticsのシミュレーションを分割する必要あり

        """
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
        foot_tag = self.foot_sound.get_tags()[0]
        offset_time = min([foot['t'] for foot in person_footstep])
        for foot in person_footstep:
            p = [foot['point'].x, foot['point'].y, 0.0]
            if room.is_inside(p):
                room.add_source(position=p,
                                signal=self.foot_sound.get_wav(foot_tag, -1),
                                delay=foot['t']-offset_time)
        if room.n_sources == 0:
            return np.zeros([room.n_mics, 1])
        room.simulate()
        simulated_sound = room.mic_array.signals
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
                    room.add_source(position=p, signal=sig, delay=sim_param['t'][i] - sim_param['offset'])
            simulated_sound = np.zeros([room.n_mics, 1])
            if room.n_sources > 0:
                room.simulate()
                simulated_sound = room.mic_array.signals
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
        index: index of person.

        Returns
        -------
        GeoDataFrame
        """
        crowd = self.crowd_list[index]
        line = crowd.path.coords
        record_num = len(line)
        df = gpd.GeoDataFrame({'t': [crowd.start_time + crowd.time_step * r for r in range(record_num - 1)]},
                              geometry=[LineString([line[r], line[r + 1]]) for r in range(record_num - 1)])
        df['id'] = index
        return df

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


def audio_crowd_simulation(crowd_csv, room_shp, output_folder, mic_shp=None, snr=None, time_unit=10.0,
                           distance_list=None):
    signal_info = {}
    if distance_list is None:
        distance_list = [1, 10, 20, 30, 40, 50]
    signal_info['distance_list'] = distance_list

    crowd_list = Crowd.csv_to_crowd_list(crowd_csv)
    st_time, ed_time = min([c.start_time for c in crowd_list]), max([c.start_time for c in crowd_list])
    signal_info['simulation_time'] = [str(st_time), str(ed_time)]
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--crowd-csv')
    parser.add_argument('-s', '--room-shp')
    parser.add_argument('-o', '--output-folder')
    parser.add_argument('-m', '--mic-shp')
    parser.add_argument('-n', '--snr', default=None, type=float)
    parser.add_argument('-t', '--time-unit', default=10.0, type=float)
    parser.add_argument('-opt', '--option', choices=['footstep', 'env'],
                        default='footstep')
    parser.add_argument('-dl', '--distance-list', nargs='+', type=int, default=[1, 10, 20, 30, 40, 50])
    args = parser.parse_args()
    if args.option == 'footstep':
        audio_crowd_simulation(args.crowd_csv, args.room_shp, args.output_folder, args.mic_shp, args.snr,
                               args.time_unit, args.distance_list)
