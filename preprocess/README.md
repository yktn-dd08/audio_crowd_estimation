# preprocess

足音音源の準備（セグメンテーション・足音タグ付与）と、シミュレーション結果・DISCO データの特徴量計算を行うモジュールです。

| ファイル | 内容 |
| --- | --- |
| `footstep_sound.py` | 足音のセグメンテーション、人流軌跡への足音タグ付与 |
| `sim_data.py` | シミュレーション出力からの log-mel 特徴量計算 |
| `disco_data.py` | DISCO データセットの前処理・特徴量計算 |

コマンドはリポジトリのルートから実行します。

## footstep_sound.py

```bash
python -m preprocess.footstep_sound -opt <segment|foot_tag|json> ...
```

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `-opt` | `segment` | 実行モード |
| `-i` | | 入力（`segment`：元音源フォルダ / `foot_tag`：軌跡CSV） |
| `-o` | | 出力（`segment`：出力フォルダ / `foot_tag`：出力CSV） |
| `-c` | `./footstep_config.json` | 足音音源の設定ファイル |
| `-f` | | foottag 設定ファイル（`foot_tag`） |
| `-ic` | | 一括処理用の JSON（`json`） |

### `-opt segment`：足音セグメンテーション

[footstep_config.json](../footstep_config.json) の各音源（`flag: false` のものは除く）を `segment` に書かれた時刻 [s] の区間ごとに切り出し、相互相関で位置合わせしてから `<出力フォルダ>/<タグ名>/` に保存します。
元音源には ESC-50（https://github.com/karolpiczak/ESC-50 ）の WAV を使います。

```bash
python -m preprocess.footstep_sound -opt segment -i ./data/ambient_sound/audio -o ./data/ambient_sound/footstep
```

`footstep_config.json` の形式：
```json
{
    "config": {
        "1-155858-A-25": {
            "file": "1-155858-A-25.wav",
            "category": "footsteps",
            "segment": [0.0, 0.9, 1.6, 2.1, 2.7, 3.3, 3.9, 4.5, 5.0],
            "type": "asphalt"
        }
    }
}
```
`type` は路面種別（`asphalt`, `wood`, `pebble`, `water` など）で、足音タグ付与時の絞り込みに使います。

### `-opt foot_tag`：足音タグ付与

軌跡CSVの人物（`id`）ごとに、足音音源（`foot_tag`）・身長（`height` [m]）・歩幅（`foot_step` = 身長 × 0.35）をランダムに割り当てた列を追加します。

```bash
python -m preprocess.footstep_sound -opt foot_tag -i data/gis/marunouchi/crowd/org/roi1_crowd0806_09.csv -o data/gis/marunouchi/crowd/asphalt/train/roi1_crowd0806_09.csv -f template/foottag/foottag_template_train.json
```

foottag 設定（[template/foottag/](../template/foottag/)）:
```json
{
    "include": {"type": ["asphalt"]},
    "exclude": {"tag": ["1-155858-A-25"]},
    "random_state": 0,
    "height": [160.0, 180.0]
}
```

| キー | 説明 |
| --- | --- |
| `include` | 使用する音源の条件。`type`（路面種別）・`tag`（音源名）のリスト。省略時は全音源 |
| `exclude` | 除外する音源の条件。形式は `include` と同じ |
| `random_state` | 乱数シード（必須） |
| `height` | 身長の範囲 [cm]。一様分布で割り当て。省略時は 170 cm 固定 |

学習用とテスト用で別の音源を使うために、`foottag_template_train.json` / `foottag_template_test.json` のように `exclude` を使い分けています。

### `-opt json`：一括処理

```bash
python -m preprocess.footstep_sound -opt json -ic template/database/02_add_foot_tag.json
```

`option: "foot_tag"`（または `"segment"`）の場合は `task_list` の各タスクを処理します（テンプレート: [template/database/02_add_foot_tag.json](../template/database/02_add_foot_tag.json)）:
```json
{
    "option": "foot_tag",
    "foot_tag_json": "./template/foottag/foottag_template_test.json",
    "config_json": "./footstep_config.json",
    "task_list": {
        "asphalt/test/roi1_crowd1031_09": {
            "input": "./data/gis/atc/crowd/org/roi1_crowd1031_09.csv",
            "output": "./data/gis/atc/crowd/asphalt/test/roi1_crowd1031_09.csv"
        }
    }
}
```
タスクごとに `input`, `output`, `config_json`, `foot_tag_json` を指定できます（省略時はトップレベルの値）。

`option: "foot_tag_all"` の場合は、`param.path`（glob）に一致する全CSVを、パス中の `/<folder_before>/` を `/<folder_after>/` に置き換えた場所へ出力します（既存ファイルはスキップ。下記「注意点」参照）。

## sim_data.py

```bash
python -m preprocess.sim_data -opt feature -i workspace/sim/xxx -o workspace/sim/xxx/feature/d005 -s 5.0
```

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `-opt` | `plot` | `feature`：log-mel 特徴量と人数を計算し `<出力>/feature.pickle` に保存 / `plot`：スペクトログラムと人数の図を出力 |
| `-i` | | 音響シミュレーションの出力フォルダ |
| `-o` | | 出力フォルダ |
| `-s` | 1.0 | 窓長 = シフト幅 [s] |

学習（[analysis/README.md](../analysis/README.md)）は WAV から直接特徴量を計算するため、`-opt feature` の実行は必須ではありません。

## disco_data.py

DISCO データセット（`audio/`, `image/`, `density_map/` を含むフォルダ）を処理します。

```bash
python -m preprocess.disco_data -opt preprocess -i <DISCOフォルダ> -o <出力フォルダ>
```
```bash
python -m preprocess.disco_data -opt feature -i <DISCOフォルダ> -o <出力フォルダ>
```

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `-opt` | `preprocess` | `preprocess`：密度マップ・画像・音声の予備解析 / `feature`：音声の特徴量解析 |
| `-i` | | DISCO データセットのフォルダ |
| `-o` | | 出力フォルダ（ファイルごとにサブフォルダを作成） |

## 注意点

- `foot_tag_all` は、内部で `foot_tag_all()` に `config_json` と `foot_tag_json` が逆順で渡されているため、そのままでは動作しません。
- `sim_data.py -opt plot` は旧形式の出力（`sim0.wav`）を前提としているため、現在の出力には使えません。
- `python -m common.audio_crowd -opt segment` も同様のセグメンテーションを行いますが、旧実装（相互相関による位置合わせなし）です。
