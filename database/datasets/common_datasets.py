import argparse
import os.path

import numpy as np
import pandas as pd
from scipy import interpolate
from sqlalchemy import create_engine, text
from common.logger import get_logger

FPS = 2.5
logger = get_logger('database.datasets.common_datasets')


def interpolate_people_flow_df(df: pd.DataFrame, dt=0.1):
    """
    人流データのDataFrameを指定したfpsで補間する。
    Parameters
    ----------
    df: pd.DataFrame
        人流データのDataFrame
        columns: [t, pid, x, y]
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


def store_people_flow(pg_url: str, layout_name: str, df: pd.DataFrame, offset_t: float = 1577836800.0):
    """
    人流データのDataFrameをPostgreSQLに保存する。

    Parameters
    ----------
    pg_url: str
        PostgreSQLの接続URL
    layout_name: str
        保存先のレイアウト名
    df: pd.DataFrame
        保存する人流データのDataFrame。カラムは't', 'pid', 'x', 'y'を想定。
    offset_t: float
        tに加算するオフセット値[s]

    Returns
    -------

    """
    # df = pd.read_csv(input_csv)
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