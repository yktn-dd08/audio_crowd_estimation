# simulation

人流軌跡のシミュレーションと、足音の音響シミュレーションを行うモジュールです。

| ファイル | 内容 |
| --- | --- |
| `crowd_trajectory.py` | ランダム歩行 / Social Force Model による人流軌跡の生成 |
| `crowd_audio.py` | 足音タグ付き軌跡から各マイクの受音信号を生成（pyroomacoustics） |
| `util.py` | シミュレーション本体（`CrowdTrajectory`, `SocialForceSimulation`, `CrowdSim` など） |

コマンドはリポジトリのルートから実行します。

## crowd_trajectory.py：人流軌跡シミュレーション

JSON 設定に従って人流軌跡を生成し、[database/README.md](../database/README.md) の `crowd_data.py` と同じ形式（`id, start_time, geom`）のCSVを出力します。

```bash
python -m simulation.crowd_trajectory -c template/simulation/trajectory/sfm_crowd_trajectory.json
```

| オプション | 説明 |
| --- | --- |
| `-c, --config` | 設定 JSON |

JSON の構成：
```json
{
    "option": "social_force",
    "overwrite": false,
    "param": { "<共通パラメータ>": "..." },
    "task_list": {
        "<タスク名>": { "output_csv": "<出力CSV>", "output_mp4": "<任意: 動画出力>", "<個別パラメータ>": "..." }
    }
}
```
タスク側のパラメータが `param` より優先されます。`output_csv` が既に存在するタスクは `overwrite: true` でない限りスキップします。

### `option: "random"`（ランダム歩行）

| キー | 既定値 | 説明 |
| --- | --- | --- |
| `roi_shp` | なし | 歩行領域（ROI）のシェープファイル |
| `wall_shp` | なし | 壁（障害物）のシェープファイル |
| `person_num` | 100 | 人数 |
| `start_time`, `end_time` | 0, 100 | 出現時間の範囲 [s] |
| `v`, `v_sigma` | 1.5, 0.3 | 歩行速度の平均・標準偏差 [m/s] |
| `dir_division` | 8 | 進行方向の分割数 |
| `datetime_str` | `2024-01-01 00:00:00` | 出力CSVの基準日時 |
| `fps` | 10 | `output_mp4` を出力する場合のフレームレート |

### `option: "social_force"`（Social Force Model）

| キー | 既定値 | 説明 |
| --- | --- | --- |
| `roi_shp`, `wall_shp` | なし | ROI・壁のシェープファイル |
| `person_num` | 100 | 人数 |
| `start_time` | `2024-01-01 00:00:00` | 基準日時 |
| `simulation_time` | 100 | シミュレーション時間 [s] |
| `dt` | 1.0 | 時間刻み [s] |
| `desired_speed` | 1.5 | 希望速度 [m/s] |
| `velocity_noise_std` | 0.3 | 速度ノイズの標準偏差 |
| `c_obs`, `r_obs` | 2000.0, 0.08 | 他者からの反発力の強さ・到達距離 |
| `c_wall`, `r_wall` | 2000.0, 0.08 | 壁からの反発力の強さ・到達距離 |
| `goal_flag` | false | ゴールを設定するか |

テンプレート: [template/simulation/trajectory/](../template/simulation/trajectory/)。
速度・人数を変えた設定の一括生成は [tools/README.md](../tools/README.md) の `create_sim_json.py` を参照してください。

## crowd_audio.py：音響シミュレーション

足音タグ付き軌跡（[preprocess/README.md](../preprocess/README.md) の `foot_tag`）から、各マイク位置での受音信号をシミュレーションします。

```bash
python -m simulation.crowd_audio -opt footstep -i template/simulation/audio/atc/atc_footstep_simulation_template.json
```

| オプション | 説明 |
| --- | --- |
| `-opt` | `footstep`（現状これのみ） |
| `-i, --input-config` | 設定 JSON（必須） |

設定 JSON（テンプレート: [template/simulation/audio/](../template/simulation/audio/)）:
```json
{
    "roi_shp": "./data/gis/atc/roi/atc_roi02.shp",
    "mic_shp": "./data/gis/atc/mic/atc_mic01.shp",
    "snr": null,
    "height": 3.0,
    "max_order": 0,
    "overwrite": false,
    "task_list": {
        "asphalt/train/roi2_crowd1024_09": {
            "crowd_csv": "./data/gis/atc/crowd/asphalt/train/roi2_crowd1024_09.csv",
            "output_folder": "./workspace/sim/atc/monaural/asphalt/train/roi2_crowd1024_09",
            "params": {}
        }
    }
}
```

| キー | 既定値 | 説明 |
| --- | --- | --- |
| `roi_shp` | 必須 | 部屋（シミュレーション領域）のシェープファイル |
| `mic_shp` | なし | マイク位置のシェープファイル（属性 `id`, `height` が必要）。ROI 内のマイクのみ使用。省略時は ROI 中心・高さ 0.05 m に1本 |
| `height` | 3.0 | 部屋の高さ [m] |
| `max_order` | 0 | 壁面反射の最大次数（0 は直接音のみ） |
| `snr` | なし | 加える白色雑音の SNR [dB]。`null` で雑音なし |
| `distance_list` | `[1,2,...,10,20,30,40,50]` | マイクからの距離別人数を集計する半径 [m] |
| `grid_shp` | なし | 指定するとマイク別ではなくグリッド別の人数を集計 |
| `x_idx_range`, `y_idx_range` | なし | グリッド集計の対象インデックス範囲 |
| `time_unit` | 10.0 | 1回のシミュレーションで処理する時間長 [s]（タスクの `params` でのみ指定可能） |
| `overwrite` | false | `output_folder/signal_info.json` が存在するタスクを再実行するか |

上記はタスクの `params` で個別に上書きできます（`roi_shp` も `params.roi_shp` で上書き可）。

入力CSVには `id, start_time, geom` が必要です。`foot_tag`, `height`, `foot_step`, `time_step` 列があれば使用し、なければ既定値（身長 1.7 m、歩幅 1.7 × 0.35 m、1 歩 1.0 s）になります。

### 出力フォルダの内容

| ファイル | 内容 |
| --- | --- |
| `sim_micXXXX.wav` | マイクごとのシミュレーション音声（全マイク共通の最大値で正規化） |
| `crowd.csv` | ROI 全体の人数（列 `t, crowd`） |
| `crowd_micXXXX.csv` | 各マイクについて 1 秒ごとの人数（列 `count, distance_<半径>..., t`）。`distance_3` はマイクから 3 m 以内の人数 |
| `crowd_grid.csv` | グリッド指定時のグリッド別人数 |
| `footstep_info.json` | 各足音の発生情報 |
| `signal_info.json` | 上記ファイルのパス、正規化係数 `signal_max`、マイク位置などのメタ情報（学習時に参照） |

この出力フォルダを [analysis/README.md](../analysis/README.md) の `model_training.py` の入力に指定します。

## util.py

`python -m simulation.util` でも 1 件の音響シミュレーションを実行できますが、旧形式の CLI です（`-c` 軌跡CSV、`-s` ROI、`-o` 出力フォルダ、`-m` マイク、`-g` グリッド、`-n` SNR、`-t` time_unit、`-dl` 距離リスト）。通常は `crowd_audio.py` を使ってください。

## 注意点

- [template/simulation/trajectory/random_crowd_trajectory.json](../template/simulation/trajectory/random_crowd_trajectory.json) は `option` が `crowd_trajectory` になっており、何も実行されません。`random` に変更してください。
- `template/simulation/audio/` の多く（11ファイル）で `max_orders` と書かれていますが、読まれるキーは `max_order` です（反射次数は 0 になります）。
