import datetime
import argparse
import os.path

import numpy as np
import pandas as pd
from common.logger import get_logger
from database.datasets.common_datasets import interpolate_people_flow_df, store_people_flow

logger = get_logger('database.datasets.gc_datasets')


def store_gc_datasets(
        pg_url: str,
        layout_name: str,
        input_csv: str,
        offset_datetime: list[str] = None,
        # offset_t: float = 1577836800.0
):
    if offset_datetime is None:
        offset_datetime = ['2011-06-20', '09:00:00']
    offset_t = datetime.datetime.fromisoformat(' '.join(offset_datetime)).timestamp()
    df = pd.read_csv(input_csv)
    df = df[['timestamp', 'agent_id', 'pos_x', 'pos_y']]
    df.columns = ['t', 'pid', 'x', 'y']

    df = interpolate_people_flow_df(df, 0.1)
    store_people_flow(pg_url, layout_name, df, offset_t)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--option', type=str, choices=['store'])
    parser.add_argument('-dh', '--db-host', type=str, default='localhost', help='database host')
    parser.add_argument('-dp', '--db-port', type=str, default='5432', help='database port')
    parser.add_argument('-du', '--db-user', type=str, default='postgres', help='database user')
    parser.add_argument('-dw', '--db-pw', type=str, default='postgres', help='database password')
    parser.add_argument('-dn', '--db-name', type=str, help='database name')
    parser.add_argument('-lo', '--layout-name', type=str, help='layout name')

    parser.add_argument('-i', '--input-csv', type=str, required=True, help='Input CSV file path')
    args = parser.parse_args()
    _pg_url = f'postgresql://{args.db_user}:{args.db_pw}@{args.db_host}:{args.db_port}/{args.db_name}'

    if args.option == 'store':
        store_gc_datasets(
            pg_url=_pg_url,
            layout_name=args.layout_name,
            input_csv=args.input_csv
        )
