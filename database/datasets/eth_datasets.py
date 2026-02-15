import argparse
import os.path

import numpy as np
import pandas as pd
from scipy import interpolate
from sqlalchemy import create_engine, text
from common.logger import get_logger

FPS = 2.5
logger = get_logger('database.datasets.eth_datasets')


def get_fpl_from_frame_number(df: pd.DataFrame, frame_number_column='frame_number', pid_column='pedestrian_ID'):
    """
    pd.DataFrameに格納sれているデータから、人流データが何フレームに一回記録されているかを取得する。

    Parameters
    ----------
    df: pd.DataFrame
        フレーム番号とPIDが格納されている人流データのDataFrame
    frame_number_column: str
        フレーム番号が格納されているカラム名
    pid_column: str
        PIDが格納されているカラム名

    Returns
    -------
    fpl: float
        frame per log
    """
    # frame per logのリストを作成
    delta_frame_list = []
    for _, g_df in df.groupby(pid_column):
        frame_numbers =sorted(g_df[frame_number_column].unique())
        delta_frame_list.extend([j - i for i, j in zip(frame_numbers[:-1], frame_numbers[1:])])

    # frame per logの出現回数をカウント
    cnt_dict = {k: delta_frame_list.count(k) for k in set(delta_frame_list)}

    # もっとも多いframe per logを取得
    fpl = list(cnt_dict.keys())[0]
    for k, v in cnt_dict.items():
        if v > cnt_dict[fpl]:
            fpl = k
    return fpl


def load_people_flow_df(
        csv_path: str,
        frame_number_column='frame_number',
        pid_column='pedestrian_ID',
        x_column='pos_x',
        y_column='pos_y'
):
    """
    人流データのCSVファイルを読み込み、frame per logを取得する。

    Parameters
    ----------
    csv_path: str
        人流データのCSVファイルパス
    frame_number_column: str
        フレーム番号が格納されているカラム名
    pid_column: str
        PIDが格納されているカラム名
    x_column: str
        x座標が格納されているカラム名
    y_column: str
        y座標が格納されているカラム名
    Returns
    -------
    df: pd.DataFrame
        読み込んだ人流データのDataFrame
    """
    logger.info(f'Load people flow data from {csv_path}')
    df = pd.read_csv(csv_path)
    fpl = get_fpl_from_frame_number(df, frame_number_column, pid_column)
    logger.info(f'frame per log: {fpl}')
    df['t'] = (df[frame_number_column] - df[frame_number_column].min()) / fpl / FPS
    df = df[['t', pid_column, x_column, y_column]]
    df.columns = ['t', 'pid', 'x', 'y']
    return df


def interpolate_people_flow_df(df: pd.DataFrame, dt=0.1):
    """
    人流データのDataFrameを指定したfpsで補間する。
    Parameters
    ----------
    df: pd.DataFrame
        人流データのDataFrame
    dt: float
        補間後のdelta_t[s]

    Returns
    -------

    """
    logger.info(f'Interpolate people flow data with dt={dt} [s]')
    res_df = pd.DataFrame()
    for pid, gr_df in df.groupby('pid'):
        t_org = gr_df['t'].values
        if len(t_org) < 2:
            res_df = pd.concat([res_df, gr_df], axis=0)
            continue
        t_interp = np.arange(t_org.min(), t_org.max(), dt)
        t_interp = np.array([t for t in t_interp if t < t_org.max()])
        x_org = gr_df['x'].values
        y_org = gr_df['y'].values
        # logger.info(f'[pid: {pid}] Interpolate with {t_org} original points -> {t_interp} interpolated points')
        # f_x = interpolate.interp1d(t_org, x_org, kind='quadratic')
        # f_y = interpolate.interp1d(t_org, y_org, kind='quadratic')
        f_x = interpolate.interp1d(t_org, x_org)
        f_y = interpolate.interp1d(t_org, y_org)
        x_interp = f_x(t_interp)
        y_interp = f_y(t_interp)
        n_df = pd.DataFrame({'t': t_interp, 'pid': int(pid), 'x': x_interp, 'y': y_interp})
        res_df = pd.concat([res_df, n_df], axis=0)
    return res_df


def store_people_flow(pg_url: str, layout_name: str, input_csv: str, offset_t: float = 1577836800.0):
    """
    人流データのDataFrameをPostgreSQLに保存する。

    Parameters
    ----------
    pg_url: str
        PostgreSQLの接続URL
    layout_name: str
        保存先のレイアウト名
    input_csv: str
        保存する人流データのCSVファイルパス(proprocess済みのファイル)
    offset_t: float
        tに加算するオフセット値[s]

    Returns
    -------

    """
    df = pd.read_csv(input_csv)
    df['t'] = df['t'] + offset_t
    logger.info(f'pg_url: {pg_url}, Copy to {layout_name}_org')
    engine = create_engine(pg_url)
    df.to_sql(f'{layout_name}_org', engine, if_exists='replace', index=False)
    sql1 = f'''DROP TABLE IF EXISTS {layout_name}_point;
    SELECT to_timestamp(t::int)::timestamp without time zone AS t, pid AS id, ST_MakePoint(AVG(x), AVG(y)) AS geom
    INTO {layout_name}_point FROM {layout_name}_org GROUP BY t::int, pid;'''
    sql2 = f'''DROP TABLE IF EXISTS {layout_name}_model;
    WITH w AS (SELECT id, t AS start_time, LEAD(t) OVER (PARTITION BY id ORDER BY t) AS end_time,
    geom AS p1, LEAD(geom) OVER (PARTITION BY id ORDER BY t) AS p2 FROM {layout_name}_point)
    SELECT id, start_time, end_time, ST_MakeLine(p1, p2) AS line INTO {layout_name}_model
    FROM w WHERE end_time IS NOT NULL AND p2 IS NOT NULL AND end_time - start_time = INTERVAL '1 second'
    ORDER BY id, start_time;
    CREATE INDEX index_id_{layout_name}_model ON {layout_name}_model USING btree(id);
    CREATE INDEX index_start_time_{layout_name}_model ON {layout_name}_model USING btree(start_time);
    CREATE INDEX index_end_time_{layout_name}_model ON {layout_name}_model USING btree(end_time);
    CREATE INDEX index_line_{layout_name}_model ON {layout_name}_model USING gist(line);'''
    with engine.connect() as con:
        logger.info(f'Execute SQL - {sql1.replace("    ", "")}')
        con.execute(text(sql1))
        con.commit()
        logger.info(f'Execute SQL - {sql2.replace("    ", "")}')
        con.execute(text(sql2))
        con.commit()
    return


def preprocess_people_flow(input_csv, output_csv):
    df = load_people_flow_df(input_csv)
    df = interpolate_people_flow_df(df, 0.1)

    folder = os.path.dirname(output_csv)
    os.makedirs(folder, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return

# def preprocess_people_flow(
#         input_csv: str,
#         pg_url: str,
#         table_name: str,
#         dt: float = 0.1,
#         frame_number_column='frame_number',
#         pid_column='pedestrian_ID',
#         x_column='pos_x',
#         y_column='pos_y'
# ):
#     """
#     人流データのCSVファイルを読み込み、補間してPostgreSQLに保存する。
#
#     Parameters
#     ----------
#     input_csv: str
#         人流データのCSVファイルパス
#     pg_url: str
#         PostgreSQLの接続URL
#     table_name: str
#         保存先のテーブル名
#     dt: float
#         補間後のdelta_t[s]
#     frame_number_column: str
#         フレーム番号が格納されているカラム名
#     pid_column: str
#         PIDが格納されているカラム名
#     x_column: str
#         x座標が格納されているカラム名
#     y_column: str
#         y座標が格納されているカラム名
#
#     Returns
#     -------
#
#     """
#     df = load_people_flow_df(
#         input_csv,
#         frame_number_column,
#         pid_column,
#         x_column,
#         y_column
#     )
#     interp_df = interpolate_people_flow_df(df, dt)
#     store_people_flow(pg_url, table_name, interp_df)
#     return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--option', type=str, choices=['preprocess', 'store'])
    # parser.add_argument('-i', '--input-file', type=str, help='input csv file path')
    parser.add_argument('-dh', '--db-host', type=str, default='localhost', help='database host')
    parser.add_argument('-dp', '--db-port', type=str, default='5432', help='database port')
    parser.add_argument('-du', '--db-user', type=str, default='postgres', help='database user')
    parser.add_argument('-dw', '--db-pw', type=str, default='postgres', help='database password')
    parser.add_argument('-dn', '--db-name', type=str, help='database name')
    parser.add_argument('-lo', '--layout-name', type=str, help='layout name')

    parser.add_argument('-i', '--input-file', type=str, help='input csv file path')
    parser.add_argument('-o', '--output-file', type=str, help='output csv file path')

    args = parser.parse_args()
    _pg_url = f'postgresql://{args.db_user}:{args.db_pw}@{args.db_host}:{args.db_port}/{args.db_name}'
    if args.option == 'preprocess':
        logger.info(f'Preprocess people flow data: {args.input_file} -> {args.output_file}')
        preprocess_people_flow(args.input_file, args.output_file)
    elif args.option == 'store':
        logger.info(f'Store people flow data to PostgreSQL: {args.input_file} -> {args.layout_name}')
        store_people_flow(_pg_url, args.layout_name, args.input_file)
