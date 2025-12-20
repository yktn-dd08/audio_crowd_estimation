import os
import argparse
import pandas as pd
import numpy as np
from scipy.io import wavfile

# wavファイルと開始時間、終了時間を入力として、指定された時間範囲の音声を切り出して保存する関数
def extract_wav_segment(input_wav, start_time, end_time, output_wav):
    # 入力wavファイルのサンプリング周波数とデータを読み込む
    print('read from', input_wav)
    fs, data = wavfile.read(input_wav)

    # 開始時間と終了時間をサンプル数に変換
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)

    # 音声データの範囲を切り出す
    segment = data[start_sample:end_sample]

    # 切り出した音声データを新しいwavファイルとして保存
    print('write to', output_wav)
    wavfile.write(output_wav, fs, segment)


# 開始時間と終了時間、その間のKPIから構成されるcsvファイルを読み込み、delta_time秒ごとのKPIを線形補間して保存する関数
def interpolate_kpi(csv_file, delta_time, output_csv):
    # CSVファイルを読み込む
    df = pd.read_csv(csv_file)

    # 開始時間と終了時間の列を取得
    start_times = df['start_time'].values
    end_times = df['end_time'].values

    # KPIの列を取得（ここでは'score'列を例として使用）
    kpi_values = df['score'].values

    # 補間後の時間とKPIを格納するリスト
    interpolated_times = []
    interpolated_kpis = []

    # 各区間ごとに補間を行う
    for i in range(len(start_times)):
        start_time = start_times[i]
        end_time = end_times[i]
        kpi_value = kpi_values[i]

        # delta_time秒ごとの時間点を生成
        times = np.arange(start_time, end_time, delta_time)
        if len(times) == 0 or times[-1] < end_time:
            times = np.append(times, end_time)

        # 各時間点に対してKPI値を割り当てる（ここでは一定値として扱う）
        kpis = np.full_like(times, kpi_value, dtype=float)

        # リストに追加
        interpolated_times.extend(times)
        interpolated_kpis.extend(kpis)

    # 補間結果をデータフレームに変換して保存
    interpolated_df = pd.DataFrame({
        'time': interpolated_times,
        'kpi': interpolated_kpis
    })
    interpolated_df.to_csv(output_csv, index=False)

# argparseを使ってコマンドライン引数を処理するメイン関数
def main():
    # optionにてwav, csvのどちらを処理するかを指定
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--option', type=str, required=True, choices=['wav', 'csv'])
    parser.add_argument('-i', '--input-file', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, required=True)
    parser.add_argument('-s', '--start-time', type=float, default=0.0)
    parser.add_argument('-e', '--end-time', type=float, default=10.0)
    parser.add_argument('-d', '--delta-time', type=float, default=0.1)
    args = parser.parse_args()
    if args.option == 'wav':
        print(args)
        extract_wav_segment(args.input_file, args.start_time, args.end_time, args.output_file)
    elif args.option == 'csv':
        interpolate_kpi(args.input_file, args.delta_time, args.output_file)
    else:
        raise ValueError("Invalid option. Use 'wav' or 'csv'.")


if __name__ == '__main__':
    main()
