# tools

パイプライン実行、設定 JSON の一括生成、結果の集計、可視化、実録音データの処理などの補助ツールです。

| ファイル | 内容 |
| --- | --- |
| `pipeline.py` | 複数のモジュール関数を JSON で順番に実行 |
| `create_sim_json.py` | 速度 × 人数ごとのシミュレーション設定 JSON を生成 |
| `create_ml_json.py` | 学習設定 JSON を生成する関数（CLI なし） |
| `compare_model_training.py` | 複数モデルの精度 JSON を 1 つの CSV に集計 |
| `output_acc_from_result.py` | 正解値の範囲を限定した精度を出力 |
| `visualize.py` | log-mel / スペクトログラム / 散布図の画像出力 |
| `crowd_histogram.py` | マイクからの距離別人数分布の可視化 |
| `create_heatmap_gif.py` | グリッド人数のヒートマップ動画 |
| `create_calibrated_dataset.py` | 実録音を学習用フォルダ形式に変換 |
| `denoise.py` | スペクトルサブトラクション / ウィーナーフィルタによる雑音除去 |
| `afpild_multiband.py` | AFPILD 評価用の多帯域スペクトルサブトラクション（[analysis/README.md](../analysis/README.md) 参照） |
| `correct_simulation_path.py` | シミュレーション出力内のパス修正用 |

コマンドはリポジトリのルートから実行します。

## pipeline.py：パイプライン実行

任意のモジュール関数を 1 つの JSON で順番に実行します。

```bash
python -m tools.pipeline <pipeline設定JSON>
```

```json
{
    "env_param": {"$SIM": "./workspace/sim/atc/monaural"},
    "pipelines": [
        {
            "module": "preprocess.footstep_sound",
            "function": "add_foot_tag",
            "params": {"config_json": "./footstep_config.json", "foot_tag_json": "./template/foottag/foottag_template_train.json"},
            "task_list": {
                "t1": {"input_csv": "./data/gis/atc/crowd/org/roi2_crowd1024_09.csv", "output_csv": "./data/gis/atc/crowd/asphalt/train/roi2_crowd1024_09.csv"}
            }
        },
        {
            "module": "simulation.crowd_audio",
            "function": "audio_crowd_simulation",
            "params": {"room_shp": "./data/gis/atc/roi/atc_roi02.shp", "mic_shp": "./data/gis/atc/mic/atc_mic01.shp"},
            "task_list": {
                "t1": {"crowd_csv": "./data/gis/atc/crowd/asphalt/train/roi2_crowd1024_09.csv", "output_folder": "$SIM/asphalt/train/roi2_crowd1024_09"}
            }
        }
    ]
}
```

| キー | 説明 |
| --- | --- |
| `pipelines[].module`, `function` | 呼び出す関数（必須） |
| `pipelines[].params` | 関数へのキーワード引数 |
| `pipelines[].task_list` | 指定すると、各タスクの値で `params` を上書きして繰り返し実行 |
| `env_param` | `$` を含むキーをパラメータ内の文字列（リスト要素も含む）で置換 |

`execute_json` 系の関数（例: `simulation.crowd_audio.footstep_simulation_task` に `config_path` を渡す）を並べることもできます。

## create_sim_json.py：設定 JSON の一括生成

速度 × 人数の組み合わせごとに、シミュレーション設定 JSON（[simulation/README.md](../simulation/README.md) 参照）を生成します。

```bash
python -m tools.create_sim_json -opt rnd_trj -s setting.json -v 0.8 1.3 1.8 -p 100 500 1000 -o template/simulation/trajectory/rec_rnd_crowd_trajectory.json
```

| オプション | 説明 |
| --- | --- |
| `-opt` | `rnd_trj`（ランダム歩行軌跡）/ `sfm_trj`（SFM 軌跡）/ `audio`（音響シミュレーション）。必須 |
| `-o` | 出力 JSON。必須 |
| `-v`, `-p` | 速度 [m/s] と人数のリスト。必須（`audio` でもダミー値が必要） |
| `-s` | 基本設定 JSON（`rnd_trj`, `audio`） |
| `-c` | codex による SFM パラメータ探索結果 JSON（`sfm_trj`） |

- `rnd_trj`：基本設定 `{"roi_shp", "wall_shp"(任意), "output_folder", "tag"}`。v, p の組み合わせごとに `dir_division ∈ {8,4,2}` × `v_sigma ∈ {0.3,0.2,0.1,0.0}` の 12 シナリオを生成
- `audio`：基本設定 `{"glob_path", "roi_shp", "mic_shp", "output_folder", "snr", "height", "max_order"}`。`glob_path` に一致する全軌跡CSVのタスクを生成
- `sfm_trj`：現状 CLI からは動作しません（下記「注意点」）

## create_ml_json.py

学習設定 JSON（[analysis/README.md](../analysis/README.md) 参照）を生成する `create_training_json()` を提供します。CLI はないため Python から呼び出します。

```python
from tools.create_ml_json import create_training_json
create_training_json(
    output_json='./workspace/json/train.json',
    input_glob_list=['./workspace/sim/atc/monaural/asphalt/train/*'],
    valid_glob_list=['./workspace/sim/atc/monaural/asphalt/test/*'],
    model_folder='./workspace/model/train/xxx',
    model_name='SimpleCNN2',
    model_param={'option': 'train', 'target': ['distance_3'], 'time_sec': 5, 'frame_num': 501, 'freq_num': 64},
)
```
`model_param` には `option`, `target`, `time_sec`, `frame_num`, `freq_num`（必須）と、`device`, `batch_size`（64）, `epoch`（200）, `log_scale`（true）, `time_agg`（true）, `model_param`（モデル固有の値で既定値を上書き）を指定できます。対応モデルは `SimpleCNN2` などファイル内の `DEFAULT_PARAM` にあるものです。

## compare_model_training.py / output_acc_from_result.py：結果の集計

```bash
python -m tools.compare_model_training -g ./workspace/model/train/asphalt -o ./workspace/model/compare.csv
```
```bash
python -m tools.output_acc_from_result -g ./workspace/model/train/asphalt -r 3 8
```

| コマンド | オプション | 説明 |
| --- | --- | --- |
| `compare_model_training` | `-g` | モデルフォルダを含む親フォルダ。各フォルダの `train_acc_*.json` / `test_acc_*.json` を縦持ち（`indicator, value, condition, target, ...`）で結合。フォルダ名 `<place>_<algorithm>_v<速度>_p<人数>_...` から条件列も付与 |
| | `-o` | 出力CSV |
| `output_acc_from_result` | `-g` | モデルフォルダを含む親フォルダ。各フォルダの `test_result.csv` を読む |
| | `-r` | 正解値の範囲（既定 `3.0 8.0`）。結果は各フォルダに `test_acc_<target>_range3.0-8.0.json` などの名前で保存 |

## visualize.py

```bash
python -m tools.visualize -opt logmel -i workspace/sim/xxx/sim_mic0000.wav -o workspace/visualize/logmel.png
```
```bash
python -m tools.visualize -opt scatter -i workspace/model/xxx/test_result.csv -o workspace/visualize/scatter.png -xc target_distance_3 -yc predict_distance_3 -gt SimpleCNN2
```

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `-opt` | `logmel` | `logmel`（log-mel スペクトログラム）/ `stft`（スペクトログラム）/ `scatter`（散布図） |
| `-i`, `-o` | | 入力（WAV または CSV）・出力画像 |
| `-xc`, `-yc` | | 散布図の x / y 列名 |
| `-xl`, `-yl` | `Actual [people/s]`, `Estimated [people/s]` | 軸ラベル（スペース区切りで複数語可） |
| `-gt` | | グラフタイトル（`scatter` では必須） |

## crowd_histogram.py

マイク（中心点）からの距離ごとの人数分布を可視化します。

```bash
python -m tools.crowd_histogram -opt anime -i data/gis/atc/crowd/org/roi1_crowd1024_09.csv -s data/gis/atc/mic/atc_mic01.shp -o workspace/visualize/hist.mp4
```

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `-opt` | `anime` | `anime`：距離ヒストグラムの動画（+ 箱ひげ図 PNG）/ `simple`：時間 × 距離の密度図 / `json`：一括処理 |
| `-i` | | 軌跡CSV |
| `-s` | | 中心点のシェープファイル |
| `-o` | | 出力 MP4（`anime`） |
| `-of` | | 出力フォルダ（`simple`） |
| `-l` | `True` | 人数に `log1p` をかけるか（`True` / `False`） |
| `-dr` | `0 30` | 距離範囲 [m] |
| `-db` | 0.5 | 距離ビン幅 [m] |
| `-tb` | 5 | 時間ビン幅 [s]（`simple`） |
| `-th` | 3.0 | しきい値距離 [m]（`simple`） |
| `-ic` | | 一括処理 JSON（`json`）。`option: "simple"`、`param`（`shp_path`, `output_folder` など）、`task_list`（`csv_path` または `glob_path`）を指定 |

## create_heatmap_gif.py

`crowd_grid.csv` のように `x{i}_y{j}` 形式の列を持つ CSV から、グリッド人数のヒートマップ動画を作成します。

```bash
python -m tools.create_heatmap_gif -opt gif -i workspace/sim/xxx/crowd_grid.csv -o workspace/visualize/heatmap.gif --fps 2 --vmin 0 --vmax 5
```

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `-opt` | `gif` | `gif` のみ実装（`png` は未実装） |
| `-i`, `-o` | | 入力CSV・出力ファイル |
| `--fps` | 1.0 | フレームレート |
| `--cmap` | `jet` | カラーマップ |
| `--vmin`, `--vmax` | 自動 | カラースケールの範囲 |
| `--title` | `Heatmap Animation` | タイトル |

## create_calibrated_dataset.py

実録音を、シミュレーションデータと同じフォルダ形式（WAV + 人数CSV + `signal_info.json`）に変換します。

```bash
python -m tools.create_calibrated_dataset -opt calib -i rec.wav -c calib.wav -a annotation.csv -o workspace/record/office/xxx
```

| `-opt` | 内容 | 使うオプション |
| --- | --- | --- |
| `calib` | キャリブレーション用 WAV から求めた係数で入力 WAV を補正し、アノテーションと合わせて学習用フォルダを作成 | `-i -c -a -o` |
| `annotation` | `start_time, end_time, <target>` 形式のアノテーションCSVを 1 秒刻みに補間して `-oc` に保存し、同じ区間の WAV を `-oc` の拡張子を `.wav` にしたファイルに切り出す | `-i -a -oc -tc -st -et` |
| `segment` | 入力 WAV の `-st`〜`-et` 秒を `-c` のパスへ切り出す | `-i -c -st -et` |

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `-i` | 必須 | 入力 WAV |
| `-c` | | キャリブレーション用 WAV（`segment` では出力 WAV） |
| `-a` | | アノテーションCSV |
| `-oc` | | 補間後CSVの出力先（`annotation`） |
| `-tc` | `count` | 補間する列名 |
| `-st`, `-et` | 0.0, 末尾 | 開始・終了時刻 [s] |
| `-s` | 22 | キャリブレーションのステップ数（現行の `calib` では未使用） |
| `-o` | | 出力フォルダ（`calib`） |

## denoise.py

```bash
python -m tools.denoise -opt ss -i input.wav -o output.wav -s 0.0 -e 3.0
```

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `-opt` | `ss` | `ss`（スペクトルサブトラクション）/ `wf`（ウィーナーフィルタ） |
| `-i`, `-o` | | 入力・出力 WAV |
| `-s`, `-e` | 0.0, なし | 雑音推定に使う区間の開始・終了 [s] |

## 注意点

- `create_sim_json.py -opt sfm_trj` は `sfm_trj_json()` の呼び出し引数が不足しており、CLI からは動作しません。
