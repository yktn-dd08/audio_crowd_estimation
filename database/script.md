# Database Script Usage Guide
## how to execute crowd_data.py

## how to execute layout_data.py
### roi

### mic

### csv2shp

- XYカラムを持つCSVファイルをシェープファイルに変換。マイクロフォンの配置情報などをシェープファイルで管理したい場合に使用。
```commandline
python -m database.layout_data -opt csv2shp -ic <input_csv_path> -os <output_shp_path>
```
- 下記のようなCSVファイルを想定
- input_csv_path: data/gis/marunouchi/mic/marunouchi_mic09_3m.csv

| id | height | x      | y      | x_idx | y_idx |
|----|--------|--------|--------|-------|-------|
| 0  | 0.01   | 54.603 | 19.696 | -1    | -1    |
| 1  | 0.01   | 54.603 | 16.696 | -1    | 0     |
| 2  | 0.01   | 54.603 | 13.696 | -1    | 1     |
| 3  | 0.01   | 51.603 | 19.696 | 0     | -1    |
| 4  | 0.01   | 51.603 | 16.696 | 0     | 0     |
| 5  | 0.01   | 51.603 | 13.696 | 0     | 1     |
| 6  | 0.01   | 48.603 | 19.696 | 1     | -1    |
| 7  | 0.01   | 48.603 | 16.696 | 1     | 0     |
| 8  | 0.01   | 48.603 | 13.696 | 1     | 1     |

### csv2grid

- XYカラムを持つCSVファイルをグリッド形式に変換。人流の数を計算するためのグリッド管理を想定。
```commandline
python -m database.layout_data -opt csv2grid -ic <input_csv_path> -os <output_grid_path> -bs <buffer_size>
```
- buffer_size: グリッドの1セルの半分のサイズ（メートル単位）
- 下記のようなCSVファイルを想定
- input_csv_path: data/gis/marunouchi/mic/pf_grid.csv
- x,yカラムを中心に、buffer_size=0.5mでグリッド化する

| grid_id | x       | y      | x_idx | y_idx |
|---------|---------|--------|-------|-------|
| 0       | 101.603 | 26.696 | -50   | -10   |
| 1       | 100.603 | 26.696 | -49   | -10   |
| 2       | 99.603  | 26.696 | -48   | -10   |
| 3       | 98.603  | 26.696 | -47   | -10   |
| 4       | 97.603  | 26.696 | -46   | -10   |
| 5       | 96.603  | 26.696 | -45   | -10   |
| 6       | 95.603  | 26.696 | -44   | -10   |
| 7       | 94.603  | 26.696 | -43   | -10   |
| 8       | 93.603  | 26.696 | -42   | -10   |

## SQL script
- 以下のSQLスクリプトは、時間ごとの人流密度データを計算し、`marunouchi_density`に格納する。各グリッドごとに各時間に存在する人の数が含まれる。
```sql
drop table if exists marunouchi_density;

with times as (select distinct start_time as t from marunouchi_model),
agg as (select f.start_time as t, g.id as gid, count(distinct f.id) as count from marunouchi_model as f join marunouchi_grid as g on st_intersects(f.line, g.geom) group by f.start_time, g.id, g.geom)
select times.t, g.id as gid, coalesce(agg.count, 0) as count, g.geom as geom into marunouchi_density from times cross join marunouchi_grid as g left join agg on agg.t=times.t and agg.gid = g.id order by times.t, g.id;

create index if not exists sidx_marunouchi_density_geom on marunouchi_density using gist(geom);
create index if not exists marunouchi_density_t_idx on marunouchi_density using btree(t);
```