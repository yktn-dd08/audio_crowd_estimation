import os
import glob
import argparse
import pandas as pd
from common.logger import get_logger
from database.datasets.eth_datasets import store_people_flow


FPS = 23.98
logger = get_logger('database.datasets.dut_datasets')


def read_dut_people_flow_csv(
        csv_path: str,
        frame_number_column='frame',
        pid_column='id',
        x_column='x_est',
        y_column='y_est'
):
    df = pd.read_csv(csv_path)
    df['t'] = df[frame_number_column] / FPS
    df = df[['t', pid_column, x_column, y_column]]
    df.columns = ['t', 'pid', 'x', 'y']
    return df


def merge_multiple_csv(input_csv_list, output_csv):
    df_list = []
    for i, input_csv in enumerate(input_csv_list):
        logger.info(f'Read {input_csv}')
        each_df = read_dut_people_flow_csv(input_csv)
        each_df['t'] += i * 60  # 各csvの時間をずらす（例: 60秒ずつ）
        df_list.append(each_df)
    merged_df = pd.concat(df_list, ignore_index=True)
    logger.info(f'Write {output_csv}')
    folder = os.path.dirname(output_csv)
    os.makedirs(folder, exist_ok=True)
    merged_df.to_csv(output_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--option', type=str, choices=['preprocess', 'store'])
    parser.add_argument('-dh', '--db-host', type=str, default='localhost')
    parser.add_argument('-dp', '--db-port', type=int, default=5432)
    parser.add_argument('-du', '--db-user', type=str, default='postgres')
    parser.add_argument('-dw', '--db-pw', type=str, default='postgres')
    parser.add_argument('-dn', '--db-name', type=str)
    parser.add_argument('-lo', '--layout-name', type=str)
    parser.add_argument('-i', '--input-csv', type=str, nargs='*')
    parser.add_argument('-o', '--output-csv', type=str)
    args = parser.parse_args()
    _pg_url = f'postgresql://{args.db_user}:{args.db_pw}@{args.db_host}:{args.db_port}/{args.db_name}'

    if args.option == 'preprocess':
        logger.info(f'Preprocess people flow data: {args.input_csv} -> {args.output_csv}')
        if len(args.input_csv) == 1 and '*' not in args.input_csv[0]:  # 単一のcsvファイルが指定された場合
            merge_multiple_csv([args.input_csv[0]], args.output_csv)
        elif len(args.input_csv) == 1 and '*' in args.input_csv[0]:  # ワイルドカードが指定された場合
            csv_list = glob.glob(args.input_csv[0])
            merge_multiple_csv(sorted(csv_list), args.output_csv)
        else:  # 複数のcsvファイルが指定された場合
            merge_multiple_csv(args.input_csv, args.output_csv)
        # df = read_dut_people_flow_csv(args.input_csv)
        # folder = os.path.dirname(args.output_csv)
        # os.makedirs(folder, exist_ok=True)
        # df.to_csv(args.output_csv, index=False)
    elif args.option == 'store':
        logger.info(f'Store people flow data: {args.input_csv[0]} -> {args.layout_name}')
        store_people_flow(_pg_url, args.layout_name, args.input_csv[0])
