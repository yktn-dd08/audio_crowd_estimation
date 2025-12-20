import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.io import wavfile
from common.logger import get_logger
logger = get_logger('__tools.create_calibrated_dataset__')


def calculate_calibration(wav_file, step_num=22):
    """
    信号の標準偏差が1になるようにするためのキャリブレーションパラメータを計算する
    Parameters
    ----------
    wav_file: str
        キャリブレーション用のWAVファイルのパス
    step_num: int
        キャリブレーション用のWAVファイルに含まれる歩数

    Returns
    -------
    calib_param: float
        キャリブレーションパラメータ
    """
    logger.info(f'Reading WAV file for calibration: {wav_file}')
    fs, signal = wavfile.read(wav_file)
    if len(signal.shape) > 1:
        signal = signal.mean(axis=1)
    signal = signal.astype(float)
    logger.info(f'Step number of footstep: {step_num}')

    # 一歩分の信号の二乗平均を計算
    signal_power = np.sum(signal**2) / step_num

    # 一歩分の信号のパワーが1になるようにするためのキャリブレーションパラメータ
    # calib_param = 1.0 / np.sqrt(signal_power)
    calib_param = 1.0 / signal.std()
    logger.info(f'Calibration parameter: {calib_param}')
    return calib_param


def interpolate_kpi_from_annotation(csv_file, output_csv, start_time_col, end_time_col, target_col,
                                    start_time=0.0, end_time=None, delta_time=1.0):
    """
    開始時間と終了時間、その間のKPIから構成されるcsvファイルを読み込み、delta_time秒ごとのKPIを線形補間して保存する関数

    Parameters
    ----------
    csv_file: str
        入力のアノテーションCSVファイル
    output_csv: str
        出力の補間後のCSVファイル
    start_time_col: str
        開始時間の列名
    end_time_col: str
        終了時間の列名
    start_time: float
        補間の開始時間（秒）
    end_time: float
        補間の終了時間（秒）
    target_col: str
        KPIの列名
    delta_time: float
        補間の時間間隔（秒）

    Returns
    -------

    """
    # CSVファイルを読み込む
    logger.info(f'Reading annotation CSV file: {csv_file}')
    df = pd.read_csv(csv_file)

    # 開始時間と終了時間の列を取得
    start_times = df[start_time_col].values
    end_times = df[end_time_col].values

    # KPIの列を取得
    kpi_values = df[target_col].values
    start_time_min = start_time
    end_time_max = end_times.max() - delta_time if end_time is None else end_time - delta_time
    assert start_time_min >= 0.0
    time_length = int((end_time_max - start_time_min) / delta_time)

    def time2index(t_):
        for idx in range(len(df)):
            if start_times[idx] <= t_ <= end_times[idx]:
                return idx
        return -1

    def calc_kpi(start_idx_, end_idx_, time_s, time_e):
        assert end_idx_ >= start_idx_ >= 0, f'start_idx: {start_idx_}, end_idx: {end_idx_}, time_s: {time_s}, time_e: {time_e}'
        if start_idx_ == -1 or end_idx_ == -1:
            return 0.0
        if start_idx_ == end_idx_:
            return kpi_values[start_idx_] * (time_e - time_s)
        total_time = 0.0
        total_kpi = 0.0
        for idx in range(start_idx_, end_idx_ + 1):
            if idx == start_idx_:
                seg_time = end_times[idx] - time_s
                total_time += seg_time
                total_kpi += kpi_values[idx] * seg_time
            elif idx == end_idx_:
                seg_time = time_e - start_times[idx]
                total_time += seg_time
                total_kpi += kpi_values[idx] * seg_time
            else:
                seg_time = end_times[idx] - start_times[idx]
                total_time += seg_time
                total_kpi += kpi_values[idx] * seg_time
        return total_kpi / total_time

    out_dict = {'time_idx': [], 't': [], 'distance_3': [], 'start_idx': [], 'end_idx': []}
    for i in range(time_length):
        t = i * delta_time + start_time_min
        start_idx = time2index(t)
        end_idx = time2index(t + delta_time)
        out_dict['time_idx'].append(i)
        out_dict['t'].append(t)
        out_dict['distance_3'].append(calc_kpi(start_idx, end_idx, t, t + delta_time))
        out_dict['start_idx'].append(start_idx)
        out_dict['end_idx'].append(end_idx)

    # # 補間後の時間とKPIを格納するリスト
    # interpolated_times = []
    # interpolated_kpis = []
    #
    # # 各区間ごとに補間を行う
    # for i in range(len(start_times)):
    #     start_time = start_times[i]
    #     end_time = end_times[i]
    #     kpi_value = kpi_values[i]
    #
    #     # delta_time秒ごとの時間点を生成
    #     times = np.arange(start_time, end_time, delta_time)
    #     if len(times) == 0 or times[-1] < end_time:
    #         times = np.append(times, end_time)
    #
    #     # 各時間点に対してKPI値を割り当てる（ここでは一定値として扱う）
    #     kpis = np.full_like(times, kpi_value, dtype=float)
    #
    #     # リストに追加
    #     interpolated_times.extend(times)
    #     interpolated_kpis.extend(kpis)

    # 補間結果をデータフレームに変換して保存
    logger.info(f'Writing interpolated CSV file to: {output_csv}')
    folder = os.path.dirname(output_csv)
    os.makedirs(folder, exist_ok=True)
    interpolated_df = pd.DataFrame(out_dict)
    interpolated_df['count'] = 1
    interpolated_df.to_csv(output_csv, index=False)


def extract_wav_segment(input_wav, start_time, end_time, output_wav):
    """
    wavファイルと開始時間、終了時間を入力として、指定された時間範囲の音声を切り出して保存する関数
    Parameters
    ----------
    input_wav: str
        入力WAVファイルのパス
    start_time: float
        切り出し開始時間（秒）
    end_time: float
        切り出し終了時間（秒）
    output_wav: str
        出力WAVファイルのパス

    Returns
    -------

    """
    # 入力wavファイルのサンプリング周波数とデータを読み込む
    logger.info(f'[WAV segmentation] Reading from {input_wav}')
    fs, data = wavfile.read(input_wav)

    # 開始時間と終了時間をサンプル数に変換
    start_sample = int(start_time * fs)
    end_sample = len(data) if end_time is None else int(end_time * fs)

    # 音声データの範囲を切り出す
    segment = data[start_sample:end_sample]

    # 切り出した音声データを新しいwavファイルとして保存
    folder = os.path.dirname(output_wav)
    os.makedirs(folder, exist_ok=True)
    logger.info(f'Writing to {output_wav}')
    wavfile.write(output_wav, fs, segment)


def create_calibrated_folder2(input_wav, calib_wav, annotation_csv, output_folder):
    """
    キャリブレーション用のWAVファイルを用いて、入力WAVファイルをキャリブレーションし、指定されたフォルダに保存する
    officeにて録音した時のアノテーションデータに対応(2025/09/23)
    new version
    Parameters
    ----------
    input_wav: str
        入力WAVファイルのパス
    calib_wav: str
        キャリブレーション用のWAVファイルのパス
    annotation_csv: str
        アノテーションCSVファイルのパス
    output_folder: str
        キャリブレーション後のWAVファイルを保存するフォルダ

    Returns
    -------

    """
    calib_param = calculate_calibration(calib_wav)

    logger.info(f'Reading input WAV file: {input_wav}')
    fs, signal = wavfile.read(input_wav)
    if len(signal.shape) > 1:
        signal = signal.mean(axis=1)
    signal = signal.astype(float)

    # キャリブレーションを適用
    calibrated_signal = signal * calib_param
    signal_max = np.max(np.abs(calibrated_signal))
    calibrated_signal /= signal_max  # -1.0 ~ 1.0に正規化

    # 出力フォルダが存在しない場合は作成
    os.makedirs(output_folder, exist_ok=True)
    annotation_df = pd.read_csv(annotation_csv)
    distance_list = ['distance_3']

    # キャリブレーション後のWAVファイルを保存
    output_wav = os.path.join(output_folder, os.path.basename(input_wav))
    logger.info(f'Writing calibrated WAV file to: {output_wav}')
    wavfile.write(output_wav, fs, calibrated_signal)

    # annotaion_csvをoutput_folderにコピー
    output_csv = os.path.join(output_folder, os.path.basename(annotation_csv))
    annotation_df.to_csv(output_csv, index=False)

    # signal_infoを作成
    signal_info = {
        'distance_list': distance_list,
        'sampling_rate': fs,
        'microphone': [[0, 0, 0.01]],
        'signal_max': signal_max,
        'info': {
            'ch0': {
                'wav_path': output_wav,
                'each_crowd': output_csv
            }
        }
    }
    # jsonファイルとして保存
    output_json = os.path.join(output_folder, 'signal_info.json')
    with open(output_json, 'w') as f:
        json.dump(signal_info, f, indent=4)
    logger.info(f'Writing signal info JSON file to: {output_json}')
    return


def create_calibrated_folder(input_wav, calib_wav, step_num, annotation_csv, output_folder):
    """
    キャリブレーション用のWAVファイルを用いて、入力WAVファイルをキャリブレーションし、指定されたフォルダに保存する
    homeで録音した時のアノテーションデータに対応(2025/09/14の時のデータ)
    old version
    Parameters
    ----------
    input_wav: str
        入力WAVファイルのパス
    calib_wav: str
        キャリブレーション用のWAVファイルのパス
    step_num: int
        キャリブレーション用のWAVファイルに含まれる歩数
    annotation_csv: str
        アノテーションCSVファイルのパス
    output_folder: str
        キャリブレーション後のWAVファイルを保存するフォルダ

    Returns
    -------

    """
    calib_param = calculate_calibration(calib_wav, step_num)

    logger.info(f'Reading input WAV file: {input_wav}')
    fs, signal = wavfile.read(input_wav)
    if len(signal.shape) > 1:
        signal = signal.mean(axis=1)
    signal = signal.astype(float)

    # キャリブレーションを適用
    calibrated_signal = signal * calib_param
    signal_max = np.max(np.abs(calibrated_signal))
    calibrated_signal /= signal_max  # -1.0 ~ 1.0に正規化

    annotation_df = pd.read_csv(annotation_csv)
    distance_list = [int(d.replace('distance_', '')) for d in annotation_df.columns if d.startswith('distance_')]
    # 出力フォルダが存在しない場合は作成
    os.makedirs(output_folder, exist_ok=True)

    # キャリブレーション後のWAVファイルを保存
    output_wav = os.path.join(output_folder, os.path.basename(input_wav))
    logger.info(f'Writing calibrated WAV file to: {output_wav}')
    wavfile.write(output_wav, fs, calibrated_signal)

    # annotaion_csvをoutput_folderにコピー
    output_csv = os.path.join(output_folder, os.path.basename(annotation_csv))
    annotation_df.to_csv(output_csv, index=False)

    # signal_infoを作成
    signal_info = {
        'distance_list': distance_list,
        'sampling_rate': fs,
        'microphone': [[0, 0, 0.01]],
        'signal_max': signal_max,
        'info': {
            'ch0': {
                'wav_path': output_wav,
                'each_crowd': output_csv
            }
        }
    }
    # jsonファイルとして保存
    output_json = os.path.join(output_folder, 'signal_info.json')
    with open(output_json, 'w') as f:
        json.dump(signal_info, f, indent=4)
    logger.info(f'Writing signal info JSON file to: {output_json}')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create calibrated dataset from real recordings.')
    parser.add_argument('-opt', '--option', type=str, required=True,
                        choices=['calib', 'annotation', 'segment'],)
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='Path to the input WAV file.')
    parser.add_argument('-c', '--calibration-file', type=str,
                        help='Path to the calibration WAV file.')
    parser.add_argument('-s', '--step-num', type=int, default=22,
                        help='Number of steps for calibration calculation.')
    parser.add_argument('-a', '--annotation-csv', type=str,
                        help='Path to the annotation CSV file.')
    parser.add_argument('-oc', '--output-csv', type=str,
                        help='Path to save the interpolated CSV file (used with annotation option).')
    parser.add_argument('-tc', '--target-col', type=str, default='count',
                        help='Target column name for interpolation (used with annotation option).')
    parser.add_argument('-st', '--start-time', type=float, default=0.0,
                        help='Start time for interpolation (used with annotation option).')
    parser.add_argument('-et', '--end-time', type=float, default=None,
                        help='End time for interpolation (used with annotation option).')
    parser.add_argument('-o', '--output-folder', type=str,
                        help='Folder to save the calibrated WAV files.')
    args = parser.parse_args()
    if args.option == 'annotation':
        interpolate_kpi_from_annotation(
            args.annotation_csv, args.output_csv,
            start_time_col='start_time', end_time_col='end_time', target_col=args.target_col,
            start_time=args.start_time, end_time=args.end_time, delta_time=1.0
        )
        extract_wav_segment(
            args.input_file, args.start_time, args.end_time,
            args.output_csv.replace('.csv', '.wav')
        )

    elif args.option == 'segment':
        extract_wav_segment(
            args.input_file, args.start_time, args.end_time,
            args.calibration_file
        )

    elif args.option == 'calib':
        # create_calibrated_folder(args.input_file, args.calibration_file, args.step_num,
        #                          args.annotation_csv, args.output_folder)
        create_calibrated_folder2(
            input_wav=args.input_file,
            calib_wav=args.calibration_file,
            annotation_csv=args.annotation_csv,
            output_folder=args.output_folder
        )
