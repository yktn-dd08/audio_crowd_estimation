import glob
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


INPUT_DIR = './data/speech'
OUTPUT_DIR = './output'

CH = 'channels'
FS = 'sampling_rate'
SR = 16000
ROOM_SIZE = np.array([10.0, 10.0, 3.0])
MIC_INTERVAL = 0.02


class Data:
    CMU = 'cmu_arctic'

    @staticmethod
    def get_cmu_arctic(console_out=False):
        path = f'{INPUT_DIR}/{Data.CMU}'
        if not os.path.exists(path):
            os.makedirs(path)
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
        df = pd.read_csv(filename)
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

    def get_wav(self, tag, index=-1):
        wav_list = self.wav_dict[tag]
        if index < 0 or index > len(wav_list) - 1:
            index = random.randint(0, len(wav_list) - 1)
        return wav_list[index]

    def get_tags(self):
        return self.wav_tag


class CrowdSim:
    @classmethod
    def from_shp(cls, shp_path):
        df = gpd.read_file(shp_path)
        room_polygon = df['geometry'].loc[0]
        return cls(room_polygon=room_polygon, height=3.0)

    def __init__(self, room_polygon: Polygon, height=3.0):
        self.crowd_list = None
        self.footstep = []
        room_coordinates = np.array(room_polygon.exterior.coords[:-1])
        self.room_info = {'corners': room_coordinates.T, 'fs': SR, 'max_order': 0}
        self.room_height = height
        self.mic_info = None
        self.foot_sound = FootstepSound(sampling_rate=SR)

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
        self.footstep = [crowd.get_foot_points() for crowd in self.crowd_list]
        return

    def set_microphone(self, mic_loc: np.array):
        """
        set the locations of multiple microphones
        :param mic_loc: numpy.array(mic_num x 3)
        :return:
        """
        mic_loc_tmp = mic_loc
        if mic_loc.shape[1] == 2:
            mic_loc_tmp = np.zeros((len(mic_loc), 3))
            mic_loc_tmp[:, 0:2] = mic_loc
        self.mic_info = mic_loc_tmp

        return

    def person_sim(self, index):
        if self.crowd_list is None:
            Exception('Not set crowd data.')
        person_footstep = self.footstep[index]
        foot_tag = self.foot_sound.get_tags()[0]
        room = self.create_room()
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

    def simulation(self):
        print(f'# of people: {len(self.crowd_list)}')
        people_sound = Parallel(n_jobs=-1)(delayed(self.person_sim)(i) for i in tqdm(range(len(self.crowd_list))))
        ch = people_sound[0].shape[0]
        audio_size = max([ps.shape[1] for ps in people_sound])
        sim_result = np.zeros((ch, audio_size))
        for ps in people_sound:
            sim_result[:, :ps.shape[1]] += ps
        return sim_result

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


def audio_crowd_simulation(crowd_csv, room_shp, output_folder, mic_json=None):
    print('Reading Crowd CSV.')
    crowd_list = Crowd.csv_to_crowd_list(crowd_csv)
    print(f'[# of people] {len(crowd_list)}')
    print(f'[Simulation time] {min([c.start_time for c in crowd_list])} - {max([c.start_time for c in crowd_list])}')
    print(f'[Crowd footstep] {min([c.foot_step for c in crowd_list])} - {max([c.foot_step for c in crowd_list])}')

    print('Reading Simulation room shapefile.')
    crowd_sim = CrowdSim.from_shp(room_shp)
    print(f'[Room info]: {crowd_sim.room_info}')
    crowd_sim.set_crowd(crowd_list)
    room_center = crowd_sim.room_info['corners'].mean(axis=1)
    if mic_json is None:
        crowd_sim.set_microphone(np.array([[room_center[0] - 0.01, room_center[1], 0.8],
                                           [room_center[0] + 0.01, room_center[1], 0.8]]))
    else:
        pass

    if not os.path.exists(output_folder):
        print(f'Create folder: {output_folder}')
        os.makedirs(output_folder)
    print('Simulation start.')
    signals = crowd_sim.simulation()
    signals = signals / signals.max()
    for s in range(len(signals)):
        wavfile.write(f'{output_folder}/sim{s}.wav', SR, signals[s])

    print('Writing crowd density CSV.')
    crowd_sim.crowd_density_to_csv(f'{output_folder}/crowd.csv')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--crowd-csv')
    parser.add_argument('-s', '--room-shp')
    parser.add_argument('-o', '--output-folder')
    parser.add_argument('-m', '--mic-json')
    parser.add_argument('-opt', '--option', choices=['footstep', 'env'],
                        default='footstep')
    args = parser.parse_args()
    if args.option == 'footstep':
        audio_crowd_simulation(args.crowd_csv, args.room_shp, args.output_folder)
