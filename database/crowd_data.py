import argparse
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from sqlalchemy import create_engine
from shapely import line_merge
from shapely.geometry import MultiLineString
from joblib import Parallel, delayed
import logging
from rich.logging import RichHandler


logging.basicConfig(
    level=logging.DEBUG,
    format='Line %(lineno)d: %(name)s: %(message)s',
    # datefmt='[%Y-%m-%d ]',
    handlers=[RichHandler(markup=True, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


def get_person_trajectory(table_name, id, conn):
    sql = f'select id, start_time, end_time, line from {table_name} where id = {id} and line IS NOT NULL order by start_time;'
    g_df = gpd.read_postgis(sql, conn, geom_col='line')

    data_list = []
    tmp = {}
    for i, dr in g_df.iterrows():
        if i == 0:
            tmp = {'start_time': dr['start_time'], 'line': dr['line']}
        else:
            line_merge_try = line_merge(MultiLineString([tmp['line'], dr['line']]))
            if line_merge_try.geom_type == 'MultiLineString':
                data_list.append(tmp)
                tmp = {'start_time': dr['start_time'], 'line': dr['line']}
            else:
                tmp['line'] = line_merge_try
            if i == len(g_df) - 1:
                data_list.append(tmp)

    o_df = gpd.read_postgis(sql, conn, geom_col='line')
    o_df['id'] = id
    return o_df


def load_trj_df(pg_url, table_name, start_time, end_time, distance=5):
    """
    Load trajectory data as GeoPandas DataFrame from PostgreSQL+PostGIS
    Parameters
    ----------
    pg_url: URL for database access (postgresql://user:password@host:port/database)
    table_name: table name of people flow data
    start_time: start time for data extraction
    end_time: end time for data extraction
    distance: walking distance threshold for data extraction

    Returns GeoPandas DataFrame (column: id, t, geom: Linestring)
    -------
    """

    t_where = f'start_time between \'{start_time}\' and \'{end_time}\''

    sql = f'''select id, min(start_time) as start_time, max(end_time) as end_time,
    st_linemerge(st_collect(line order by start_time)) as geom,
    st_astext(st_linemerge(st_collect(line order by start_time))) as wkt,
    st_length(st_linemerge(st_collect(line order by start_time))) as len
    from {table_name} where {t_where} group by id
    having st_length(st_linemerge(st_collect(line order by start_time))) > {distance};'''

    with create_engine(pg_url).connect() as conn:
        logger.info(f'SQL - {sql.replace("¥n", "").replace("    ", " ")}')
        df = gpd.read_postgis(sql=sql, con=conn, geom_col='geom')
        out_df = df.loc[df['wkt'].apply(lambda x: not x.startswith('MULTI'))]
        out_df = out_df[['id', 'start_time', 'geom']]
        logger.info(f'Got LINESTRING of {len(out_df)} people.')
        id_list = df['id'].loc[df['wkt'].apply(lambda x: x.startswith('MULTI'))].tolist()

        # prc_list = Parallel(n_jobs=-1)(delayed(get_person_trajectory)(table_name, i, conn) for i in tqdm(id_list))
        logger.info(f'Postprocessing - merging MULTILINESTRING into LINESTRING for {len(id_list)} people.')
        prc_list = [get_person_trajectory(table_name, i, conn) for i in tqdm(id_list)]
        prc_df = pd.concat(prc_list, axis=0)
        out_df = pd.concat([out_df, prc_df], axis=0).reset_index().drop('index', axis=1)

    return out_df


def set_crowd_data(trj_df):
    id_list = pd.unique(trj_df['id']).tolist()
    start_time = trj_df['start_time'].min()
    # 身長セット、歩幅セット、歩いている時間セット
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-opt', '--option', type=str, choices=['vel', 'stay'])
    parser.add_argument('-of', '--output-folder', type=str)

    parser.add_argument('-dh', '--db-host', type=str, default='localhost')
    parser.add_argument('-dp', '--db-port', type=int, default=5432)
    parser.add_argument('-du', '--db-user', type=str, default='postgres')
    parser.add_argument('-dw', '--db-pw', type=str, default='postgres')
    parser.add_argument('-dn', '--db-name', type=str, default='marunouchi1')
    parser.add_argument('-lo', '--layout', type=str, default='marunouchi')
    parser.add_argument('-st', '--start-time', nargs=2, type=str)
    parser.add_argument('-et', '--end-time', nargs=2, type=str)
    args = parser.parse_args()

    _pg_url = f'postgresql://{args.db_user}:{args.db_pw}@{args.db_host}:{args.db_port}/{args.db_name}'
    trj_df = load_trj_df(_pg_url, args.layout+'_model', '2020-08-08 00:00:00','2020-08-09 00:00:00')
    trj_df.to_csv('./test.csv')
