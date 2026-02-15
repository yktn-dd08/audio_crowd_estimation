# Database Script Usage Guide
## Pedestrian Dataset
### ATC dataset
- 以下のURLより人流データをダウンロード可能
- https://dil.atr.jp/crest2010_HRI/ATC_dataset/

- 下記SQLにてTableを作成しておいてCSVファイルをインポートする
```sql
CREATE TABLE data (
	time DOUBLE PRECISION NOT NULL,
	id INTEGER NOT NULL DEFAULT '-1',
	x INTEGER NOT NULL DEFAULT '0',
	y INTEGER NOT NULL DEFAULT '0',
	z DOUBLE PRECISION NOT NULL DEFAULT '0.000',
	vel DOUBLE PRECISION NOT NULL DEFAULT '0.000',
	mTheta DOUBLE PRECISION NOT NULL DEFAULT '0.000',
	oTheta DOUBLE PRECISION NOT NULL DEFAULT '0.000'
)
```
- インポート後は下記SQLにてフォーマット変換を行う
```sql
DROP TABLE IF EXISTS atc_point;
SELECT id, to_timestamp(time::integer)::timestamp without time zone AS t, ST_Point(AVG(x)/1000, AVG(y)/1000) AS geom into atc_point
FROM data GROUP BY time::integer, id HAVING id > -1;

DROP TABLE IF EXISTS atc_model;
WITH w AS (
  SELECT
    id,
    t AS start_time,
    LEAD(t)    OVER (PARTITION BY id ORDER BY t) AS end_time,
    geom       AS p1,
    LEAD(geom) OVER (PARTITION BY id ORDER BY t) AS p2
  FROM atc_point
)
SELECT id, start_time, end_time, ST_MakeLine(p1, p2) AS line
INTO atc_model FROM w WHERE end_time IS NOT NULL AND p2 IS NOT NULL AND end_time - start_time = INTERVAL '1 second'
ORDER BY id, start_time;

CREATE INDEX index_id_atc_model ON atc_model USING btree(id);
CREATE INDEX index_start_time_atc_model ON atc_model USING btree(start_time);
CREATE INDEX index_end_time_atc_model ON atc_model USING btree(end_time);
CREATE INDEX index_line_atc_model ON atc_model USING gist(line);
```

### ETH dataset
- 事前にCSVファイルをダウンロードしておき、カラム情報を入れておく必要がある。
- CSVファイルのカラムは以下のような形式を想定。
- CSV example: data/gis/ewap/org/seq_eth.csv

| frame_number | pedestrian_ID | pos_x | pos_z | pos_y | v_x | v_z | v_y |
|--------------|---------------|-------|-------|-------|-----|-----|-----|
| 780          | 1             | 8.456 | 0.0   | 3.588 | 1.6 | 0.0 | 1.7 |
| 786          | 1             | 9.125 | 0.0   | 3.658 | 1.6 | 0.0 | 3.2 |

- 下記コマンドを打つことでPostgreSQLにインポート可能
```commandline
python -m database.datasets.eth_datasets -opt preprocess -i ./data/gis/ewap/org/seq_hotel.csv -o ./data/gis/ewap/org/ewap_hotel.csv
python -m database.datasets.eth_datasets -opt store -i ./data/gis/ewap/org/ewap_eth.csv -dn ewap -lo eth
```
### DUT dataset
- 事前にCSVファイルをダウンロードしておく
- URL: https://github.com/dongfang-steven-yang/vci-dataset-dut/tree/master
- 下記コマンドにより前処理
```commandline
python -m database.datasets.dut_datasets -opt preprocess -i ./data/gis/dut/org/trajectories_filtered/intersection_*_traj_ped_filtered.csv -o ./data/gis/dut/org/ped/intersection_ped.csv
python -m database.datasets.dut_datasets -opt preprocess -i ./data/gis/dut/org/trajectories_filtered/roundabout_*_traj_ped_filtered.csv -o ./data/gis/dut/org/ped/roundabout_ped.csv
```
- 下記コマンドによりPostgreSQLにインポート
```commandline
python -m database.datasets.dut_datasets -opt store -i ./data/gis/dut/org/ped/intersection_ped.csv -dn dut -lo intersection
python -m database.datasets.dut_datasets -opt store -i ./data/gis/dut/org/ped/roundabout_ped.csv -dn dut -lo roundabout
```

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
