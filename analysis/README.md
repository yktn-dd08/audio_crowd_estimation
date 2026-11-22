# analysis

モデルの学習・推論・ハイパーパラメータ探索と、AFPILD データを用いたシミュレーション評価を行うモジュールです。

| ファイル | 内容 |
| --- | --- |
| `model_training.py` | 学習・推論・Optuna 探索・ファインチューニング |
| `model_common.py` | 学習ループ、評価指標、結果の出力などの共通関数 |
| `export_least_square_csv.py` | 最小二乗法用に平均パワーと人数の組を CSV 出力 |
| `test_least_square_training.py` | 最小二乗モデルのテスト |
| `afpild_*.py` | AFPILD データでの足音シミュレーション評価 |
| `model_training_old.py` | 旧実装（参考用） |

コマンドはリポジトリのルートから実行します。

## model_training.py：学習・推論・チューニング

```bash
python -m analysis.model_training -c template/model_train/SimpleCNN2/sfm2atc_cnn_f5_d3.json
```

| オプション | 説明 |
| --- | --- |
| `-c, --input-config-json` | 設定 JSON（必須） |
| `-opt`, `-d` | **使われません**。実行モードは JSON の `option`、デバイスは JSON の `device` で指定します |

JSON のトップレベル：

| キー | 既定値 | 説明 |
| --- | --- | --- |
| `option` | `train` | 実行モード（下表） |
| `device` | 自動 | `cuda` / `mps` / `cpu`。省略時は cuda → mps → cpu の順に自動選択 |
| `overwrite` | false | `model_folder` に同名 JSON がある場合も再実行するか |
| `train` / `predict` / `optuna` / `finetune` | | モードごとの設定ブロック |

| `option` | 内容 | 設定ブロック |
| --- | --- | --- |
| `train` | 学習 | `train` |
| `predict` | 学習済みモデルで推論・評価 | `predict` |
| `tuning` | Optuna によるハイパーパラメータ探索 | `optuna` |
| `finetune` | 学習済みモデルの一部レイヤを再学習 | `finetune` |
| `train_multitask` / `tuning_multitask` | 距離帯別人数を補助タスクとするマルチタスク学習 | `train` / `optuna` |
| `train_multichannel` / `tuning_multichannel` | 複数マイクを入力するモデルの学習（`MultiChannelSimpleCNN`） | `train` / `optuna` |

> `predict_multitask` と `finetune_multitask` は未実装、`predict_multichannel` と `finetune_multichannel` は何も実行しません。

### 設定ブロックのキー

| キー | 対象モード | 説明 |
| --- | --- | --- |
| `input_folder_list` | 全て | 学習（推論）データ。音響シミュレーションの出力フォルダ（`signal_info.json` を含む。[simulation/README.md](../simulation/README.md) 参照）のリスト |
| `valid_folder_list` | train / optuna / finetune | 検証データのフォルダリスト |
| `model_folder` | 全て | モデルの保存先（predict / finetune では読み込み元） |
| `pretrain_model_folder` | finetune | 事前学習済みモデルのフォルダ |
| `output_folder` | predict | 推論結果の出力先 |
| `model_name` | 全て | モデル名（下表） |
| `target` | 全て（multichannel を除く） | 推定対象。`count`（範囲内の全人数）または `distance_<半径>`（マイクから半径 m 以内の人数）。既定は `["count"]` |
| `model_param` | 全て | モデルのパラメータ（下記） |
| `epoch` | train / optuna / finetune | エポック数 |
| `batch_size` | 全て | バッチサイズ |
| `log_scale` | 全て | 目的変数を `log(1 + y)` で学習するか |
| `time_agg` | 全て | 目的変数を `time_sec` 区間の時間平均にするか（false の場合は区間内の時系列） |
| `n_trials` | optuna | 試行回数（既定 1000） |
| `sampler` | optuna | `tpe`（既定）または `random` |
| `tasks`, `weight` | multitask | 補助タスク名（`range_a-b` = a〜b m の人数）とその損失の重み |

設定例（抜粋）:
```json
{
    "train": {
        "input_folder_list": ["./workspace/sim/sfm/monaural/asphalt/train/roi1_sfm_crowd0101_00"],
        "valid_folder_list": ["./workspace/sim/atc/monaural/asphalt/test/roi1_crowd1031_14"],
        "model_folder": "./workspace/model/train/asphalt/sfm2atc_SimpleCNN2",
        "model_name": "SimpleCNN2",
        "target": ["distance_3"],
        "batch_size": 64,
        "model_param": {
            "time_sec": 5, "frame_num": 501, "freq_num": 64,
            "layer_num": 3, "kernel_size": [9, 23, 23], "dilation_size": [3, 1, 1],
            "inter_ch": [16, 13, 12], "pool_size": [2, 1, 1],
            "lr": 0.00001, "weight_decay": 0.0, "criterion": "MAE"
        },
        "epoch": 50,
        "log_scale": true,
        "time_agg": true
    }
}
```

### `model_param` の主なキー

| キー | 説明 |
| --- | --- |
| `time_sec` | 1 サンプルの時間長 [s]（例: 5） |
| `frame_num`, `freq_num` | log-mel 特徴量のフレーム数・メル帯域数（例: 501, 64） |
| `layer_num`, `kernel_size`, `dilation_size`, `inter_ch`, `pool_size` | CNN の層数と各層の設定（リストの長さは `layer_num`） |
| `lr`, `weight_decay` | Adam の学習率（既定 1e-5）・重み減衰（既定 0） |
| `criterion` | 損失関数 `MAE` または `MSE`（既定 MSE） |
| `out_features` | 出力次元。省略時は `target` の数 |
| `channel_num` | 入力マイク数（multichannel） |
| `finetune` | 再学習するレイヤ（finetune）。`0` を含めば最初の畳み込み層、そうでなく `-1` を含めば回帰層のみを学習 |

各モデルのコンストラクタに存在するキーだけが渡されます（モデル定義は [model/](../model/)）。

| `model_name` | 説明 |
| --- | --- |
| `SimpleCNN`, `SimpleCNN2`, `SimpleCNN3` | log-mel を入力とする 1 次元 CNN（`SimpleCNN2` が主に使われている） |
| `MultiChannelSimpleCNN` | 複数マイク入力の CNN |
| `AreaSpecificCNN`, `AreaSpecificCNN2` | 領域別推定用 CNN（非推奨） |
| `Conv1dTransformer` | 1 次元畳み込み + Transformer |
| `ASTRegressor` | Audio Spectrogram Transformer（Hugging Face の事前学習済みモデル名を `model_param.model_name` で指定） |
| `VGGishLinear`, `VGGishTransformer` | VGGish 特徴量ベース |
| `LeastSquareModel`, `LeastSquareModel2` | 最小二乗法による線形モデル（`epoch`, `lr` などは無視） |

### Optuna 探索範囲の書き方

`optuna` ブロックの `model_param` では、値を辞書にすると探索対象になります。
```json
"layer_num": 3,
"kernel_size":   {"type": "list[int]", "size": "layer_num", "low": 3, "high": 31},
"lr":            {"type": "float", "low": 1e-6, "high": 1e-3},
"criterion":     {"type": "category", "category": ["MAE", "MSE"]}
```
- `type`: `int` / `float` / `category`、またはそのリスト版 `list[int]` など（`list[list[int]]` で 2 次元リスト）
- `size`: リスト長。数値、または別のキー名（例: `"layer_num"`）を指定

### 出力

| モード | 出力ファイル（`model_folder` または `output_folder`） |
| --- | --- |
| train / finetune | `<model_name>_model.pt`、`loss.png`、`train_result.csv` / `test_result.csv`（`target_*`, `predict_*` 列）、`*_acc_*.json`（精度指標）、散布図、設定 JSON のコピー |
| tuning | `trial<番号>/`（各試行の `loss.png`, `model_params.json` など）、`best_params.json`、`optuna_trials.csv` |
| predict | `result.csv`、`acc.json`、`scatter.png`（`log_scale` 時は `scatter_log.png` も） |

テンプレート: 学習 [template/model_train/](../template/model_train/)、探索 [template/model_tuning/](../template/model_tuning/)。
多くのテンプレートは `option` を省略しているため `train` として実行されます。

結果の集計・可視化は [tools/README.md](../tools/README.md) の `compare_model_training.py`、`output_acc_from_result.py`、`visualize.py` を参照してください。

## export_least_square_csv.py

学習設定 JSON（`train` ブロック）のデータについて、平均パワーと人数の組を `<model_folder>/train_xy.csv`, `valid_xy.csv` に出力します。

```bash
python -m analysis.export_least_square_csv -c template/model_train/FixedLinear/ls_train_f5_d3.json
```

| オプション | 説明 |
| --- | --- |
| `-c, --input-config-json` | 学習設定 JSON |

## AFPILD 評価（afpild_*.py）

AFPILD データセット（マイクアレイ + LiDAR による足音データ）の実測軌跡を用いて、足音シミュレーションと実録音を比較します。
`data/gis/afpild/` 以下のデータと `workspace/afpild/foottag/` の foottag 設定が必要です。
これらのスクリプトは `python -m` ではなく、リポジトリのルートで `PYTHONPATH=.` を付けて直接実行します。

実行順：
```bash
PYTHONPATH=. python analysis/afpild_pipeline.py --limit 160
```
```bash
PYTHONPATH=. python analysis/afpild_evaluate.py --finalize
```
```bash
PYTHONPATH=. python analysis/afpild_supplement.py
```
```bash
python analysis/afpild_audio_bands.py
```
```bash
python analysis/afpild_report.py
```

| スクリプト | オプション | 内容 |
| --- | --- | --- |
| `afpild_pipeline.py` | `--limit`（既定 160） | 160 試行の軌跡に足音タグを付与し、反射次数 3 で音響シミュレーション。入力のハッシュが変わらなければ完了済みはスキップ |
| `afpild_evaluate.py` | `--finalize` | 足音イベントの再現性と、実録音との音響特徴の比較。`--finalize` で全 160 件揃った後に集計表・図を作成 |
| `afpild_supplement.py` | | ランダム化ベースライン、共通時間長での比較、シミュレーション品質確認 |
| `afpild_audio_bands.py` | | 周波数帯域別のスペクトル比較と被験者単位のブートストラップ |
| `afpild_report.py` | | `workspace/afpild/evaluation/report.html` を作成 |
| `afpild_discontinuity.py` | | 生成された足音の途切れの診断（読み取りのみ） |

### 雑音除去を適用した評価

多帯域スペクトルサブトラクション（[tools/afpild_multiband.py](../tools/afpild_multiband.py)）を実録音に適用して比較します。

```bash
PYTHONPATH=. python analysis/afpild_denoise_evaluate.py --noise-mode quiet
```
```bash
PYTHONPATH=. python analysis/afpild_denoise_evaluate.py --noise-mode first3s
```
```bash
PYTHONPATH=. python analysis/afpild_denoise_report.py
```

| スクリプト | オプション | 内容 |
| --- | --- | --- |
| `afpild_denoise_evaluate.py` | `--noise-mode`（`quiet`：静かなフレームから雑音推定 / `first3s`：冒頭 3 秒から推定）、`--start`, `--limit`（処理する試行の範囲） | 雑音除去と評価指標の計算 |
| `afpild_denoise_report.py` | | 雑音除去前後の比較レポートを作成 |
| `afpild_denoise_validate.py` | | 保存された雑音除去済み WAV の検証 |

## テスト

```bash
python -m analysis.test_least_square_training
```

## 注意点

- `model_training.py` の `-opt` / `-d` は受け付けますが使われません（JSON の `option` / `device` を使用）。
- `template/model_tuning/` のうち `option` がなく `optuna` ブロックだけのものは `train` として実行され、`train` キーが無いためエラーになります。`"option": "tuning"` を追加してください。
