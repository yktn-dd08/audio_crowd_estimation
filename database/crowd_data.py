import argparse
import os.path
import time

import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from sqlalchemy import create_engine
from shapely import line_merge
from shapely.geometry import MultiLineString, LineString
from joblib import Parallel, delayed
from common.logger import get_logger
logger = get_logger('database.crowd_data')


def get_person_trajectory(table_name, _id, conn, where):
    sub_where = where
    if sub_where != '':
        sub_where += f' AND id = {_id} AND line IS NOT NULL'
    else:
        sub_where = f'WHERE id = {_id} AND line IS NOT NULL'
    sql = f'SELECT id, start_time, end_time, line FROM {table_name} {sub_where} ORDER BY start_time;'
    g_df = gpd.read_postgis(sql, conn, geom_col='line')

    data_list = []
    tmp = {}
    for i, dr in g_df.iterrows():
        if i == 0:
            # tmp = {'start_time': dr['start_time'], 'line': dr['line']}
            tmp = {'start_time': dr['start_time'], 'coordinates': list(dr['line'].coords)}
        else:
            if tmp['coordinates'][-1] == dr['line'].coords[0]:
                # in case of merging LineString
                tmp['coordinates'].extend(dr['line'].coords[1:])
            else:
                tmp['geom'] = LineString(tmp['coordinates'])
                data_list.append(tmp)
                tmp = {'start_time': dr['start_time'], 'coordinates': list(dr['line'].coords)}
            #     pass
            # line_merge_try = line_merge(MultiLineString([tmp['line'], dr['line']]))
            # if line_merge_try.geom_type == 'MultiLineString':
            #     data_list.append(tmp)
            #     tmp = {'start_time': dr['start_time'], 'line': dr['line']}
            # else:
            #     tmp['line'] = line_merge_try
            if i == len(g_df) - 1:
                tmp['geom'] = LineString(tmp['coordinates'])
                data_list.append(tmp)
    o_df = gpd.GeoDataFrame(data=data_list, geometry='geom')
    o_df.drop('coordinates', axis=1, inplace=True)
    # o_df = gpd.read_postgis(sql, conn, geom_col='line')
    o_df['id'] = _id
    return o_df


def get_where(start_time, end_time, roi_wkt):
    if isinstance(start_time, list):
        start_time = f'{start_time[0]} {start_time[1]}'
    if isinstance(end_time, list):
        end_time = f'{end_time[0]} {end_time[1]}'

    t_where = 'start_time '
    if start_time is None:
        if end_time is None:
            t_where = ''
        else:
            t_where += f'<= \'{end_time}\''
    else:
        if end_time is None:
            t_where += f'>= \'{start_time}\''
        else:
            t_where += f'BETWEEN \'{start_time}\' AND \'{end_time}\''
    r_where = '' if roi_wkt is None else f'ST_Contains(\'{roi_wkt}\'::geometry, line)'
    where = 'WHERE '
    if t_where == '':
        if r_where == '':
            where = ''
        else:
            where += r_where
    else:
        if r_where == '':
            where += t_where
        else:
            where += r_where + ' AND ' + t_where
    return where


def load_trj_df(pg_url, table_name, start_time, end_time, roi_wkt, distance=5):
    """
    Load trajectory data as GeoPandas DataFrame from PostgreSQL+PostGIS
    Parameters
    ----------
    pg_url: URL for database access (postgresql://user:password@host:port/database)
    table_name: table name of people flow data
    start_time: start time for data extraction
    end_time: end time for data extraction
    roi_wkt: wkt of ROI for data extraction
    distance: walking distance threshold for data extraction

    Returns GeoPandas DataFrame (column: id, t, geom: Linestring)
    -------
    """
    # t_where = 'start_time '
    # if start_time is None:
    #     if end_time is None:
    #         t_where = ''
    #     else:
    #         t_where += f'<= \'{end_time}\''
    # else:
    #     if end_time is None:
    #         t_where += f'>= \'{start_time}\''
    #     else:
    #         t_where += f'BETWEEN \'{start_time}\' AND \'{end_time}\''
    # r_where = '' if roi_wkt is None else f'ST_Contains(\'{roi_wkt}\'::geometry, line)'
    # where = 'WHERE '
    # if t_where == '':
    #     if r_where == '':
    #         where = ''
    #     else:
    #         where += r_where
    # else:
    #     if r_where == '':
    #         where += t_where
    #     else:
    #         where += r_where + ' AND ' + t_where

    logger.info(f'time range: {start_time} - {end_time}')
    logger.info(f'roi range: {roi_wkt}')

    sql = f'''SELECT id, MIN(start_time) AS start_time, MAX(end_time) AS end_time,
    ST_LineMerge(ST_Collect(line ORDER BY start_time)) AS geom,
    ST_AsText(ST_LineMerge(ST_Collect(line ORDER BY start_time))) AS wkt,
    ST_Length(ST_LineMerge(ST_Collect(line ORDER BY start_time))) AS len
    FROM {table_name} {get_where(start_time, end_time, roi_wkt)} GROUP BY id
    HAVING ST_Length(ST_LineMerge(ST_Collect(line ORDER BY start_time))) > {distance};'''

    with create_engine(pg_url).connect() as conn:
        logger.info(f'SQL - {sql.replace("\n", "").replace("    ", " ")}')
        s = time.time()
        df = gpd.read_postgis(sql=sql, con=conn, geom_col='geom')
        logger.info(f'SQL Result - Acquired {len(df)} records - Execution time: {time.time() - s} sec.')
        out_df = df.loc[df['wkt'].apply(lambda x: not x.startswith('MULTI'))]
        out_df = out_df[['id', 'start_time', 'geom']]
        logger.info(f'Acquired LINESTRING of {len(out_df)} people.')
        id_list = df['id'].loc[df['wkt'].apply(lambda x: x.startswith('MULTI'))].tolist()

        if len(id_list) > 0:
            # prc_list = Parallel(n_jobs=-1)(delayed(get_person_trajectory)(table_name, i, conn) for i in tqdm(id_list))
            logger.info(f'Need postprocess to merge MULTILINESTRING into LINESTRING for {len(id_list)} people.')
            prc_list = [get_person_trajectory(table_name, i, conn, get_where(start_time, end_time, roi_wkt))
                        for i in tqdm(id_list, desc='[Postprocess]')]
            prc_df = pd.concat(prc_list, axis=0)
            out_df = pd.concat([out_df, prc_df], axis=0).reset_index().drop('index', axis=1)

    return out_df


def set_crowd_data(trj_df):
    id_list = pd.unique(trj_df['id']).tolist()
    start_time = trj_df['start_time'].min()
    # 身長セット、歩幅セット、歩いている時間セット
    return


def get_roi_wkt(roi_shp):
    df = gpd.read_file(roi_shp)
    geom = df['geometry'].loc[0]
    return geom.wkt


def export_trj_csv(pg_url, table_name, start_time, end_time, roi_shp, output_csv):
    # if isinstance(start_time, list):
    #     start_time = f'{start_time[0]} {start_time[1]}'
    # if isinstance(end_time, list):
    #     end_time = f'{end_time[0]} {end_time[1]}'
    trj_df = load_trj_df(pg_url, table_name, start_time, end_time,
                         None if roi_shp is None else get_roi_wkt(roi_shp))
    folder = os.path.dirname(output_csv)
    os.makedirs(folder, exist_ok=True)
    trj_df.to_csv(output_csv, index=False)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--option', type=str, choices=['export'], default='export')
    parser.add_argument('-oc', '--output-csv', type=str)

    parser.add_argument('-dh', '--db-host', type=str, default='localhost')
    parser.add_argument('-dp', '--db-port', type=int, default=5432)
    parser.add_argument('-du', '--db-user', type=str, default='postgres')
    parser.add_argument('-dw', '--db-pw', type=str, default='postgres')
    parser.add_argument('-dn', '--db-name', type=str, default='marunouchi1')
    parser.add_argument('-lo', '--layout', type=str, default='marunouchi')
    parser.add_argument('-st', '--start-time', nargs=2, type=str)
    parser.add_argument('-et', '--end-time', nargs=2, type=str)
    parser.add_argument('-rs', '--roi-shp', type=str, default=None)
    args = parser.parse_args()

    _pg_url = f'postgresql://{args.db_user}:{args.db_pw}@{args.db_host}:{args.db_port}/{args.db_name}'
    if args.option == 'export':
        export_trj_csv(_pg_url, args.layout + '_model', args.start_time, args.end_time, args.roi_shp, args.output_csv)
