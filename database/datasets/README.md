# database/datasets

公開人流データを PostgreSQL に取り込み、次のテーブルを作成するスクリプトです。

| テーブル | 内容 |
| --- | --- |
| `<layout>_org` | 元データ（`t, pid, x, y`） |
| `<layout>_point` | 1 秒ごとの位置 |
| `<layout>_model` | 1 秒ごとの移動線分（`id, start_time, end_time, line`） |

作成した `<layout>_model` テーブルは [database/README.md](../README.md) の `crowd_data.py` / `layout_data.py` で参照します。
ATC データセットはスクリプトを使わず SQL で直接取り込みます（[database/script.md](../script.md) 参照）。

コマンドはリポジトリのルートから実行します。

## スクリプト一覧

| モジュール | `-opt` | 内容 |
| --- | --- | --- |
| `database.datasets.eth_datasets` | `preprocess` | ETH/UCY 形式（`frame_number, pedestrian_ID, pos_x, pos_z, pos_y, ...`）を読み、0.1秒間隔に補間して `t, pid, x, y` のCSVを出力 |
| | `store` | 前処理済みCSVをDBに格納（`<layout>_org` は置き換え） |
| `database.datasets.dut_datasets` | `preprocess` | DUT 形式（`frame, id, x_est, y_est`）の複数CSVを結合して出力。`-i` はファイル列挙またはワイルドカード（引用符で囲む） |
| | `store` | 前処理済みCSVをDBに格納 |
| `database.datasets.edinburgh_datasets` | `store` | `timestamp, agent_id, pos_x, pos_y` のCSVを補間してDBに格納。ファイル名は `edinburgh01Sep.csv` のように日付を含める必要あり（2010年の当該日 09:00 を起点とする） |
| `database.datasets.gc_datasets` | `store` | Grand Central（`timestamp, agent_id, pos_x, pos_y`）を補間してDBに格納（2011-06-20 09:00 を起点） |

`common_datasets.py` は補間・DB 格納の共通関数で、コマンドとしては使いません。

## 共通オプション

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `-opt` | | 実行モード（上表） |
| `-dh, -dp, -du, -dw` | `localhost`, `5432`, `postgres`, `postgres` | DB ホスト / ポート / ユーザー / パスワード |
| `-dn` | | DB 名 |
| `-lo` | | レイアウト名（テーブル名の接頭辞） |
| `-i` | | 入力CSV（eth は `--input-file`、それ以外は `--input-csv`） |
| `-o` | | 出力CSV（`preprocess` のみ） |

## 実行例

ETH:
```bash
python -m database.datasets.eth_datasets -opt preprocess -i ./data/gis/ewap/org/seq_eth.csv -o ./data/gis/ewap/org/ewap_eth.csv
```
```bash
python -m database.datasets.eth_datasets -opt store -i ./data/gis/ewap/org/ewap_eth.csv -dn ewap -lo eth
```

DUT（https://github.com/dongfang-steven-yang/vci-dataset-dut ）:
```bash
python -m database.datasets.dut_datasets -opt preprocess -i "./data/gis/dut/org/trajectories_filtered/intersection_*_traj_ped_filtered.csv" -o ./data/gis/dut/org/ped/intersection_ped.csv
```
```bash
python -m database.datasets.dut_datasets -opt store -i ./data/gis/dut/org/ped/intersection_ped.csv -dn dut -lo intersection
```

Edinburgh / Grand Central:
```bash
python -m database.datasets.edinburgh_datasets -opt store -i ./data/gis/edinburgh/edinburgh01Sep.csv -dn edinburgh -lo edinburgh
```
```bash
python -m database.datasets.gc_datasets -opt store -i ./data/gis/gc/gc.csv -dn gc -lo gc
```
