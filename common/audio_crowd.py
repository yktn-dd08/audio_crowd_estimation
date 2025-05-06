import os
import json
import glob
import librosa
import argparse
import numpy as np
import pandas as pd

from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from shapely import wkt
from shapely.geometry import LineString, Point
from shapely.ops import split
from common.logger import get_logger
from pydantic import BaseModel, field_serializer

FS = 16000
logger = get_logger(__name__)


# class Crowd(BaseModel):
#     time_step: float
#     # path: LineString
#     path: str
#     start_time: float
#     height: float
#     foot_step: float
#     # self.duration = len(path.coords) * time_step
#     name: str
#
#     # @field_serializer('path')
#     # def path_serialize(self, path) -> LineString:
#     #     return wkt.loads(path)
#     #
#     @classmethod
#     def csv_to_crowd_list(cls, filename):
#         logger.info(f'read csv - {filename}')
#         df = pd.read_csv(filename).dropna()
#         df['start_time'] = pd.to_datetime(df['start_time'])
#         df['geom'] = df['geom'].apply(lambda x: wkt.loads(x))
#         start_time = df['start_time'].min()
#
#         crowd_list = [cls(path=dr['geom'],
#                           start_time=(dr['start_time']-start_time).total_seconds(),
#                           height=1.7 if 'height' not in df.columns else dr['height'],
#                           foot_step=1.7 * 0.35 if 'foot_step' not in df.columns else dr['foot_step'],
#                           name=str(dr['id']))
#                       for _, dr in df.iterrows()]
#
#         return crowd_list
#
#     # def __init__(self, path: LineString,
#     #              start_time: int | float,
#     #              height: float,
#     #              foot_step: float,
#     #              time_step: float = 1.0,
#     #              name=None):
#     #     self.time_step = time_step
#     #     self.path = path
#     #     self.start_time = start_time
#     #     self.height = height
#     #     self.foot_step = foot_step
#     #     self.duration = len(path.coords) * time_step
#     #     self.name = name
#
#     def time_interpolate(self, t: float | int):
#         duration = len(self.path.coords) * self.time_step
#         if self.start_time > t or t >= self.start_time + duration:
#             return None
#         t_step_float = (t - self.start_time) / self.time_step
#         t_step_int = int(t_step_float)
#         p1 = self.path.coords[t_step_int]
#         p2 = self.path.coords[t_step_int + 1]
#         line = LineString([p1, p2])
#         return line.interpolate(distance=t_step_float - t_step_int, normalized=True)
#
#     @staticmethod
#     def __get_point_index_float(line: LineString, point: Point):
#         """
#         Nポイントで構成されるLineStringについて、内分点が何ポイント目に相当するか抽出
#         :param line:
#         :param point:
#         :return:
#         """
#         c = line.coords
#         dist_parts = [0] + [Point(c[i]).distance(Point(c[i + 1])) for i in range(len(c) - 1)]
#
#         for i, l in enumerate(line.coords):
#             if l == point.coords[0]:
#                 return i
#         d = line.distance(point)
#         gc = split(line, point.buffer(d + 1.0e-10))
#         if len(gc.geoms) != 2:
#             Exception(f'Multiple split points: {point}')
#
#         tmp_length = LineString([gc.geoms[0].coords[-2], gc.geoms[1].coords[1]]).length
#         tmp_div = LineString([gc.geoms[0].coords[-2], point.coords[0]]).length
#         return len(gc.geoms[0].coords) - 2 + tmp_div / tmp_length
#
#     def get_foot_points(self):
#         """
#
#         :return:
#         [
#             {
#                 't': time(float),
#                 'point': point(Point)
#             }
#         ]
#         """
#         walking_distance = self.path.length
#         dist = 0.0
#
#         res = []
#         pc = self.path.coords
#         dist_parts = [0] + [Point(pc[i]).distance(Point(pc[i + 1])) for i in range(len(pc) - 1)]
#         while dist < walking_distance:
#             point = self.path.interpolate(distance=dist, normalized=False)
#
#             # extract time index for each point
#             for line_index in range(len(dist_parts) - 1):
#                 if sum(dist_parts[:line_index + 1]) <= dist < sum(dist_parts[:line_index + 2]):
#                     dist_part = Point(pc[line_index]).distance(point) / dist_parts[line_index + 1]
#                     res.append({'t': line_index + dist_part + self.start_time, 'point': point, 'dist': dist})
#
#             # TODO randomize foot_step
#             dist += self.foot_step
#
#         return res


def get_first_singular_index(data, min_max):
    assert min_max in ['min', 'max']
    dif = np.diff(data, n=1)
    if min_max == 'min':
        for j in range(len(dif) - 1):
            if dif[j] < 0.0 <= dif[j + 1]:
                return j
    else:
        for j in range(len(dif) - 1):
            if dif[j] > 0.0 >= dif[j + 1]:
                return j
    return -1


class EnvSoundInfo(BaseModel):
    file: str
    segment: list[float]
    type: str | None = None
    flag: bool = True

    def read_org_wav(self, org_folder, resample=False):
        fs, signal = wavfile.read(f'{org_folder}/{self.file}')
        if resample and fs != FS:
            signal = librosa.resample(y=signal.astype(float), orig_sr=fs, target_sr=FS)
            fs = FS
        return fs, signal

    def read_wav_segments(self, folder, resample=True, normalize=False):
        file_pattern = self.file.replace('.wav', '-*.wav')
        file_list = glob.glob(f'{folder}/{file_pattern}')
        file_list.sort()
        res = []
        for f in file_list:
            fs, signal = wavfile.read(f)
            if resample and fs != FS:
                signal = librosa.resample(y=signal.astype(float), orig_sr=fs, target_sr=FS)
                fs = FS
            if normalize:
                signal = signal / signal.std()
            res.append((fs, signal))
        return res

    def save_segment(self, org_folder, res_folder, alignment=0.1, sig_max=None):
        fs, signal = self.read_org_wav(org_folder, False)
        for i in range(len(self.segment) - 1):
            # extract signal segment
            start_time, end_time = int(self.segment[i] * fs), int(self.segment[i + 1] * fs)
            signal_seg = signal[start_time:end_time]

            # signal alignment
            singular_index = get_first_singular_index(data=gaussian_filter1d(input=signal_seg ** 2,
                                                                             sigma=int(fs * 0.01),
                                                                             mode='constant'),
                                                      min_max='max')
            align_index = int(alignment * fs)
            if singular_index >= align_index:
                signal_seg = signal_seg[singular_index-align_index:]
            else:
                signal_seg = np.concatenate([np.zeros(align_index-singular_index), signal_seg])

            # signal normalization
            signal_max = np.max(np.abs(signal_seg))
            # 最大で除算するとパワーで規格化できない -> 音のエネルギーに差が出る -> どうするか検討中
            # signal_seg = signal_seg / (signal_seg.std() * 3)
            signal_seg = signal_seg / signal_max
            # TODO この後にnp.int16で規格化する
            # if normalize:
            #     signal_seg = signal_seg / signal_seg.std()

            # save signal segment as wav file
            os.makedirs(res_folder, exist_ok=True)
            wav_name = self.file.replace('.wav', f'-{i}.wav')
            wavfile.write(filename=f'{res_folder}/{wav_name}', data=signal_seg, rate=fs)
        return


class EnvSoundConfig(BaseModel):
    config: dict[str, EnvSoundInfo]

    @classmethod
    def read_json(cls, file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            s = json.load(f)
        return cls(**s)


class SoundCondition(BaseModel):
    type: list[str] | None = None
    tag: list[str] | None = None

    def include(self, sound_info: EnvSoundInfo, tag: str = None):
        if not sound_info.flag:
            return False
        if tag is None:
            tag = sound_info.file.replace('.wav', '')
        if self.type is None:
            if self.tag is None:
                return True
            else:
                return tag in self.tag
        else:
            if self.tag is None:
                return sound_info.type in self.type
            else:
                return (tag in self.tag) and (sound_info.type in self.type)


class FootTagSetting(BaseModel):
    include: SoundCondition | None = None
    exclude: SoundCondition | None = None
    random_state: int

    @classmethod
    def read_json(cls, file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            s = json.load(f)
        return cls(**s)

    def set_seed(self):
        np.random.seed(self.random_state)
        return

    def satisfied_tag_list(self, env_sound_config: EnvSoundConfig):
        tag_list = []
        for tag, sound_info in env_sound_config.config.items():
            if sound_info.flag:
                # includeに何も入っていない場合は全てのfoottagを入れる
                include_flag = self.include.include(sound_info, tag) if self.include else True
                # if self.include == {}:
                #     include_flag = True
                # excludeに何も入っていない場合はfoottagの除外対象なしとする
                exclude_flag = self.exclude.include(sound_info, tag) if self.exclude else False
                # if self.exclude == {}:
                #     exclude_flag = False
                if include_flag and not exclude_flag:
                    tag_list.append(tag)
        return tag_list

    def get_tag_list(self, env_sound_config: EnvSoundConfig, tag_num: int):
        self.set_seed()
        tag_list = self.satisfied_tag_list(env_sound_config)
        return np.random.choice(tag_list, tag_num).tolist()


def footstep_sound_segmentation(input_folder, output_folder, config_json):
    conf = EnvSoundConfig.read_json(config_json)
    for tag, sound_info in conf.config.items():
        if sound_info.flag:
            res_folder = f'{output_folder}/{tag}'
            sound_info.save_segment(org_folder=input_folder, res_folder=res_folder)
            logger.info(f'{tag}: # of segments={len(sound_info.segment) - 1}, saved footstep segments in {res_folder}.')
        else:
            logger.info(f'{tag}: flag={sound_info.flag}, segmentation skipped.')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--option', type=str, choices=['segment'], default='segment')
    parser.add_argument('-i', '--input-folder', type=str)
    parser.add_argument('-o', '--output-folder', type=str)
    parser.add_argument('-c', '--config-json', type=str)
    args = parser.parse_args()
    if args.option == 'segment':
        footstep_sound_segmentation(args.input_folder, args.output_folder, args.config_json)
    pass
