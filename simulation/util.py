import os

import pandas as pd
import torch
import shapely
import wave as wave
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt


INPUT_DIR = './data/speech'
OUTPUT_DIR = './output'

CH = 'channels'
FS = 'sampling_rate'
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
    def read_csv(cls, filename):
        df = pd.read_csv(filename)
        return

    def __init__(self, path, start_time: int, time_step: float = 1.0):
        self.time_step = 1.0
        pass

    def interpolate(self, t: int):
        return


class CrowdSim:
    def __init__(self, room):
        return

    def set_crowd(self, crowd: Crowd):
        return
