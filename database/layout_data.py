import argparse
import itertools

import numpy as np
import os.path
import time

import pandas as pd
import geopandas as gpd
from database.crowd_data import get_where, get_roi_wkt
from sqlalchemy import create_engine
from shapely.geometry import Polygon, Point

from common.logger import get_logger
logger = get_logger('database.layout_data')


def get_rectangle_roi(pg_url, table_name, start_time, end_time, roi_wkt, percent=0.03):
    with create_engine(pg_url).connect() as con:
        s = time.time()
        sql_cnt = f'SELECT COUNT(*) AS cnt FROM {table_name} {get_where(start_time, end_time, roi_wkt)};'
        cnt_df = pd.read_sql(sql_cnt, con)
        cnt = cnt_df['cnt'].loc[0]
        logger.info(f'SQL - {sql_cnt}')

        def xy(x_y, min_max):
            col = 'ST_X(ST_Centroid(line))' if x_y == 'x' else 'ST_Y(ST_Centroid(line))'
            desc = '' if min_max == 'min' else 'DESC'
            sql = f'''SELECT {col} as {x_y} FROM {table_name} {get_where(start_time, end_time, roi_wkt)}
            ORDER BY {col} {desc} LIMIT {int(cnt * percent)};'''
            logger.info(f'SQL - {sql.replace("    ", "").replace("\n", " ")}')
            df = pd.read_sql(sql, con)
            param = df[x_y].max() if min_max == 'min' else df[x_y].min()
            return param

        x_min = xy('x', 'min')
        x_max = xy('x', 'max')
        y_min = xy('y', 'min')
        y_max = xy('y', 'max')
        pol = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        logger.info(f'SQL Result - Polygon: {pol} - Total exec time: {time.time() - s} sec.')
    roi = Polygon(pol)
    return roi


def create_rectangle_roi_shp(pg_url, table_name, start_time, end_time, roi_shp, output_shp, percent=0.03):
    # sql_cnt = f'SELECT COUNT(*) AS cnt FROM {table_name};'
    # # sql = f'''SELECT LEAST(MIN(ST_X(ST_PointN(line, 1))), MIN(ST_X(ST_PointN(line, 2)))) as x_min,
    # # GREATEST(MAX(ST_X(ST_PointN(line, 1))), MAX(ST_X(ST_PointN(line, 2)))) as x_max,
    # # LEAST(MIN(ST_Y(ST_PointN(line, 1))), MIN(ST_Y(ST_PointN(line, 2)))) as y_min,
    # # GREATEST(MAX(ST_Y(ST_PointN(line, 1))), MAX(ST_Y(ST_PointN(line, 2)))) as y_max
    # # FROM {table_name} {get_where(start_time, end_time, roi_wkt)};'''
    # with create_engine(pg_url).connect() as con:
    #     s = time.time()
    #     cnt_df = pd.read_sql(sql_cnt, con)
    #     cnt = cnt_df['cnt'].loc[0]
    #     logger.info(f'SQL - {sql_cnt}')
    #
    #     def get_boundary_coords(x_y, min_max):
    #         col = 'ST_X(ST_Centroid(line))' if x_y == 'x' else 'ST_Y(ST_Centroid(line))'
    #         desc = '' if min_max == 'min' else 'DESC'
    #         sql = f'''SELECT {col} as {x_y} FROM {table_name} {get_where(start_time, end_time, roi_wkt)}
    #         ORDER BY {col} {desc} LIMIT {int(cnt * percent)};'''
    #         logger.info(f'SQL - {sql.replace("    ", "").replace("\n", " ")}')
    #         df = pd.read_sql(sql, con)
    #         param = df[x_y].max() if min_max == 'min' else df[x_y].min()
    #         return param
    #
    #     # sql_x1 = f'''SELECT ST_X(ST_Centroid(line)) as x FROM {table_name} {get_where(start_time, end_time, roi_wkt)}
    #     # ORDER BY ST_X(ST_Centroid(line)) LIMIT {int(cnt * percent)};'''
    #     # logger.info(f'SQL - {sql_x1.replace("    ", "").replace("\n", " ")}')
    #     # df = pd.read_sql(sql_x1, con)
    #     x_min = get_boundary_coords('x', 'min')
    #     x_max = get_boundary_coords('x', 'max')
    #     y_min = get_boundary_coords('y', 'min')
    #     y_max = get_boundary_coords('y', 'max')
    #     pol = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    #     logger.info(f'SQL Result - Polygon: {pol} - Total exec time: {time.time() - s} sec.')
    # # x_min, x_max, y_min, y_max = df['x_min'].loc[0], df['x_max'].loc[0], df['y_min'].loc[0], df['y_max'].loc[0]
    # roi = Polygon(pol)
    roi_wkt = None if roi_shp is None else get_roi_wkt(roi_shp)
    roi = get_rectangle_roi(pg_url, table_name, start_time, end_time, roi_wkt, percent)
    gdf = gpd.GeoDataFrame(data=[{'id': 0}], geometry=[roi], crs=2450)
    folder = os.path.dirname(output_shp)
    os.makedirs(folder, exist_ok=True)
    gdf.to_file(filename=output_shp, driver='ESRI Shapefile')
    logger.info(f'create shp: {output_shp}')
    return


def create_mic_shp(pg_url, table_name, start_time, end_time, roi_shp, mic_shp, height, x_num=1, y_num=1):
    assert x_num > 0 and y_num > 0, 'Input positive values as x_num and y_num.'
    roi_wkt = None if roi_shp is None else get_roi_wkt(roi_shp)
    roi = get_rectangle_roi(pg_url, table_name, start_time, end_time, roi_wkt)
    x_min = min([c[0] for c in roi.exterior.coords])
    x_max = max([c[0] for c in roi.exterior.coords])
    y_min = min([c[1] for c in roi.exterior.coords])
    y_max = max([c[1] for c in roi.exterior.coords])
    x_buf = (x_max - x_min) / x_num / 2
    x_list = np.linspace(x_min + x_buf, x_max - x_buf, x_num)
    # x_list = x_list[1:len(x_list) - 1]
    y_buf = (y_max - y_min) / y_num / 2
    y_list = np.linspace(y_min + y_buf, y_max - y_buf, y_num)
    # y_list = y_list[1:len(y_list) - 1]
    geom_list = [Point(x, y) for x, y in itertools.product(x_list, y_list)]
    df = gpd.GeoDataFrame(data=[{'id': i, 'height': height, 'x': geom_list[i].x, 'y': geom_list[i].y}
                                for i in range(len(geom_list))],
                          geometry=geom_list, crs=2450)
    folder = os.path.dirname(mic_shp)
    os.makedirs(folder, exist_ok=True)
    df.to_file(filename=mic_shp, driver='ESRI Shapefile')
    logger.info(f'create shp: {mic_shp}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--option', type=str, choices=['roi', 'mic'])
    parser.add_argument('-os', '--output-shp', type=str)

    parser.add_argument('-dh', '--db-host', type=str, default='localhost')
    parser.add_argument('-dp', '--db-port', type=int, default=5432)
    parser.add_argument('-du', '--db-user', type=str, default='postgres')
    parser.add_argument('-dw', '--db-pw', type=str, default='postgres')
    parser.add_argument('-dn', '--db-name', type=str, default='marunouchi1')
    parser.add_argument('-lo', '--layout', type=str, default='marunouchi')
    parser.add_argument('-st', '--start-time', nargs=2, type=str)
    parser.add_argument('-et', '--end-time', nargs=2, type=str)
    parser.add_argument('-rs', '--roi-shp', type=str, default=None)
    parser.add_argument('-xn', '--x-num', type=int, default=1)
    parser.add_argument('-yn', '--y-num', type=int, default=1)
    parser.add_argument('-ht', '--height', type=int, default=0.01)
    args = parser.parse_args()

    _pg_url = f'postgresql://{args.db_user}:{args.db_pw}@{args.db_host}:{args.db_port}/{args.db_name}'
    if args.option == 'roi':
        create_rectangle_roi_shp(_pg_url, args.layout+'_model', args.start_time, args.end_time, args.roi_shp,
                                 args.output_shp)
    elif args.option == 'mic':
        create_mic_shp(_pg_url, args.layout+'_model', args.start_time, args.end_time, args.roi_shp, args.output_shp,
                       args.height, args.x_num, args.y_num)
