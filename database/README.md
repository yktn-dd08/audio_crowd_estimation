# database

PostgreSQL + PostGIS に格納した人流データから、人流軌跡CSVや ROI・マイク・グリッドのシェープファイルを作成するモジュールです。

| ファイル | 内容 |
| --- | --- |
| `crowd_data.py` | 指定時間帯・ROI 内の人流軌跡を CSV にエクスポート |
| `layout_data.py` | ROI・マイク・グリッドのシェープファイルを作成 |
| `datasets/` | 公開人流データを DB に取り込むスクリプト（[datasets/README.md](datasets/README.md)） |
| [script.md](script.md) | ATC データの取り込みなど、DB 側で実行する SQL の手順 |

コマンドはリポジトリのルートから実行します。DB 接続オプション（`-dh/-dp/-du/-dw/-dn`）の既定値は `localhost:5432`、ユーザー・パスワードとも `postgres` です。

両モジュールとも、DB の `<layout>_model` テーブル（1 秒ごとの移動線分 `id, start_time, end_time, line`）を参照します。このテーブルは [datasets/README.md](datasets/README.md) または [script.md](script.md) の手順で作成します。

## crowd_data.py：人流軌跡のエクスポート

`<layout>_model` テーブルから、指定時間帯・ROI 内の人ごとの軌跡を LINESTRING として抽出し、CSV に出力します。
移動距離が 5 m 以下の人は除外されます。出力CSVの列は `id, start_time, geom`（WKT）です。

```bash
python -m database.crowd_data -opt export -dn marunouchi -lo marunouchi -st 2020-08-06 09:00:00 -et 2020-08-06 10:00:00 -rs data/gis/marunouchi/roi/marunouchi_roi1.shp -oc data/gis/marunouchi/crowd/roi1_crowd0806_09.csv
```

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `-opt` | `export` | `export`：コマンドライン引数で1件抽出 / `json`：JSON設定で一括抽出 |
| `-ic` | | `json` 時の設定ファイル |
| `-oc` | | 出力CSV |
| `-dh, -dp, -du, -dw` | `localhost`, `5432`, `postgres`, `postgres` | DB 接続情報 |
| `-dn` | `marunouchi1` | DB 名 |
| `-lo` | `marunouchi` | レイアウト名（`<layout>_model` を参照） |
| `-st`, `-et` | なし | 開始・終了日時。`日付 時刻` の **2引数** で指定。省略時は時間で絞り込まない |
| `-rs` | なし | ROI のシェープファイル。軌跡がROIに含まれる区間だけを抽出 |

### JSON による一括抽出

テンプレート: [template/database/01_export_csv.json](../template/database/01_export_csv.json)

```bash
python -m database.crowd_data -opt json -ic template/database/01_export_csv.json
```
```json
{
    "option": "export",
    "db_host": "localhost", "db_port": 5432, "db_user": "postgres", "db_pw": "postgres",
    "db_name": "atc",
    "layout_name": "atc",
    "roi_shp": "./data/gis/atc/roi/atc_roi01.shp",
    "task_list": {
        "roi1_crowd1024_09": {
            "output_csv": "./data/gis/atc/crowd/roi1_crowd1024_09.csv",
            "start_time": ["2012-10-24", "09:00:00"],
            "end_time": ["2012-10-24", "10:00:00"]
        }
    }
}
```
トップレベルのキー（`db_*`, `layout_name`, `roi_shp`, `start_time`, `end_time`, `output_csv`）はいずれもタスク側で上書きできます。

## layout_data.py：ROI・マイク・グリッドの作成

```bash
python -m database.layout_data -opt <roi|mic|csv2mic|csv2grid> ...
```

| `-opt` | 内容 | 主に使うオプション |
| --- | --- | --- |
| `roi` | DB の人流分布から、x・y それぞれ上下 3% を除いた矩形 ROI を作成 | `-dn -lo -st -et -rs -os` |
| `mic` | 上記矩形 ROI を `x_num × y_num` に等分した各セル中心にマイクを配置 | `-dn -lo -st -et -rs -xn -yn -ht -os` |
| `csv2mic` | `x, y` 列（他の列は属性として保持）を持つCSVからマイク位置の点シェープファイルを作成 | `-ic -os` |
| `csv2grid` | `x, y` 列を中心とした一辺 `2 × buffer_size` の正方形グリッドを作成 | `-ic -os -bf` |

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `-os` | | 出力シェープファイル |
| `-ic` | | 入力CSV（`csv2mic`, `csv2grid`） |
| `-rs` | なし | 絞り込み用 ROI シェープファイル（`roi`, `mic`） |
| `-xn`, `-yn` | `1`, `1` | マイクの x / y 方向の個数（`mic`） |
| `-ht` | `0.05` | マイク高さ [m]（`mic`） |
| `-bf` | `0.5` | グリッド一辺の半分 [m]（`csv2grid`） |
| `-dh -dp -du -dw -dn -lo -st -et` | | DB 接続・抽出条件（`crowd_data.py` と同じ） |

音響シミュレーションで使うマイクのシェープファイルには `id` と `height` の属性が必要です。`csv2mic` の場合は CSV に `id, height, x, y` 列を用意してください。

```bash
python -m database.layout_data -opt csv2mic -ic data/gis/marunouchi/mic/marunouchi_mic09_3m.csv -os data/gis/marunouchi/mic/marunouchi_mic09_3m.shp
```
```bash
python -m database.layout_data -opt csv2grid -ic data/gis/marunouchi/mic/pf_grid.csv -os data/gis/marunouchi/grid/pf_grid.shp -bf 0.5
```

入力CSVの例は [script.md](script.md) を参照してください。

## 注意点

- [script.md](script.md) には `-opt csv2shp`、`-bs` という記載がありますが、実際のオプションは `csv2mic`、`-bf` です。
