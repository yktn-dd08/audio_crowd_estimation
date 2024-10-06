import argparse
import os.path
import time

import pandas as pd
import geopandas as gpd
from database.crowd_data import get_where, get_roi_wkt
from sqlalchemy import create_engine
from shapely.geometry import Polygon, Point

from common.logger import get_logger
logger = get_logger('database.layout_data')


def create_rectangle_roi_shp(pg_url, table_name, start_time, end_time, roi_wkt, output_shp, percent=0.03):
    sql_cnt = f'SELECT COUNT(*) AS cnt FROM {table_name};'
    # sql = f'''SELECT LEAST(MIN(ST_X(ST_PointN(line, 1))), MIN(ST_X(ST_PointN(line, 2)))) as x_min,
    # GREATEST(MAX(ST_X(ST_PointN(line, 1))), MAX(ST_X(ST_PointN(line, 2)))) as x_max,
    # LEAST(MIN(ST_Y(ST_PointN(line, 1))), MIN(ST_Y(ST_PointN(line, 2)))) as y_min,
    # GREATEST(MAX(ST_Y(ST_PointN(line, 1))), MAX(ST_Y(ST_PointN(line, 2)))) as y_max
    # FROM {table_name} {get_where(start_time, end_time, roi_wkt)};'''
    with create_engine(pg_url).connect() as con:
        s = time.time()
        cnt_df = pd.read_sql(sql_cnt, con)
        cnt = cnt_df['cnt'].loc[0]
        logger.info(f'SQL - {sql_cnt}')

        def get_boundary_coords(x_y, min_max):
            col = 'ST_X(ST_Centroid(line))' if x_y == 'x' else 'ST_Y(ST_Centroid(line))'
            desc = '' if min_max == 'min' else 'DESC'
            sql = f'''SELECT {col} as {x_y} FROM {table_name} {get_where(start_time, end_time, roi_wkt)}
            ORDER BY {col} {desc} LIMIT {int(cnt * percent)};'''
            logger.info(f'SQL - {sql.replace("    ", "").replace("\n", " ")}')
            df = pd.read_sql(sql, con)
            param = df[x_y].max() if min_max == 'min' else df[x_y].min()
            return param

        # sql_x1 = f'''SELECT ST_X(ST_Centroid(line)) as x FROM {table_name} {get_where(start_time, end_time, roi_wkt)}
        # ORDER BY ST_X(ST_Centroid(line)) LIMIT {int(cnt * percent)};'''
        # logger.info(f'SQL - {sql_x1.replace("    ", "").replace("\n", " ")}')
        # df = pd.read_sql(sql_x1, con)
        x_min = get_boundary_coords('x', 'min')
        x_max = get_boundary_coords('x', 'max')
        y_min = get_boundary_coords('y', 'min')
        y_max = get_boundary_coords('y', 'max')
        pol = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        logger.info(f'SQL Result - Polygon: {pol} - Total exec time: {time.time() - s} sec.')
    # x_min, x_max, y_min, y_max = df['x_min'].loc[0], df['x_max'].loc[0], df['y_min'].loc[0], df['y_max'].loc[0]
    roi = Polygon(pol)
    gdf = gpd.GeoDataFrame(data=[{'id': 0}], geometry=[roi])
    folder = os.path.dirname(output_shp)
    os.makedirs(folder, exist_ok=True)
    gdf.to_file(filename=output_shp, driver='ESRI Shapefile')
    logger.info(f'create shp: {output_shp}')
    return


def create_mic_shp(pg_url, mic_shp, x_num=1, y_num=1):

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