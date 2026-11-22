# video

動画・録音のアノテーションから人数データを作成するためのスクリプトです。

| ファイル | 内容 |
| --- | --- |
| `create_crowd_from_annotation.py` | WAV の切り出し、区間アノテーションの時系列への展開 |
| `test_annotate_feet.py` | MediaPipe による足首位置検出のテスト |

コマンドはリポジトリのルートから実行します。実録音を学習用フォルダ形式にする場合は [tools/README.md](../tools/README.md) の `create_calibrated_dataset.py` も参照してください。

## create_crowd_from_annotation.py

```bash
python -m video.create_crowd_from_annotation -opt wav -i rec.wav -o rec_cut.wav -s 10 -e 70
```
```bash
python -m video.create_crowd_from_annotation -opt csv -i annotation.csv -o kpi.csv -d 0.1
```

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `-opt` | 必須 | `wav`：WAV の切り出し / `csv`：`start_time, end_time, score` のCSVを `-d` 秒間隔の `time, kpi` に展開 |
| `-i`, `-o` | 必須 | 入力・出力ファイル |
| `-s`, `-e` | 0.0, 10.0 | 切り出し区間 [s]（`wav`） |
| `-d` | 0.1 | 時間間隔 [s]（`csv`） |

## test_annotate_feet.py

MediaPipe で動画中の足首位置を検出し、動画に重ねて描画します。`opencv-python` と `mediapipe` が必要です。

```bash
python -m video.test_annotate_feet --input in.mp4 --output out.mp4 --csv ankles.csv
```

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `--input` | 必須 | 入力 MP4 |
| `--output` | 必須 | 出力 MP4 |
| `--csv` | なし | フレームごとの足首座標を保存する CSV |
| `--alpha` | 0.7 | 重ね描きの不透明度（0〜1） |
| `--smooth` | 0.3 | EMA 平滑化係数（0 で無効） |
| `--min_vis` | 0.5 | 描画する最小信頼度 |
| `--codec` | `mp4v` | 一時動画の FourCC（`mp4v`, `avc1` など） |
