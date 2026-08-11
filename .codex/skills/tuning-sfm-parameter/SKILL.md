---
name: tuning-sfm-parameter
description: Social Force Model (SFM) の c_obs、r_obs、c_wall、r_wall を複数の目標平均速度と person_num 条件に対して適応的にバッチ探索し、各条件で指定されたK件を必ず出力する。添付JSONを基準に十分な候補数を探索する依頼、K件保証付きSFM調整、軌跡CSVの定量評価、上位候補だけの動画確認、再現可能なJSON結果出力が必要なときに使用する。
---

# SFM parameter tuning

添付または指定されたSFM設定JSONを基準に、`target_mean_speed x person_num` の全条件をまとめて探索する。LLMは探索オーケストレーションと最終要約だけに使い、候補生成、シミュレーション実行、軌跡メトリクス計算、ランキング、JSON生成は `scripts/` に任せる。

## 最重要ルール

- 1候補ごとにLLMで次候補を考えない。
- CSV全体をLLMで解析しない。Pythonが生成した小さいsummary JSONだけを見る。
- 全候補の動画をLLMで見ない。定量評価で絞った上位候補だけ確認する。
- `parameter_sets` をK件未満のまま成功として出力しない。各条件で厳密にK件を出力する。
- 標準探索は `coarse 48件 -> 上位3中心のrefine各12件 -> K不足時の拡張バッチ各48件` とする。
- strict合格がK件に満たない場合は最大1024件まで適応的に探索する。それでも不足する場合だけ、評価可能なhard constraint違反候補を明示的なfallbackとして補う。
- シミュレーション失敗、空CSV、NaN/inf、評価不能候補はfallbackにも使わない。評価可能候補自体がK件未満なら結果JSONを確定せず、明示的に失敗する。
- 1000回の逐次試行は行わない。さらに増やす場合は `--max-candidates` と `--workers` を明示する。
- `v` を指定された場合、この実装ではまず `desired_speed=v` としてシミュレーションへ渡す。同時に、出力軌跡から実現平均速度を計算し、`v` との差をランキングに使う。
- `crowd_trajectory.py` は task側の値を `param` より優先する。探索対象の `person_num`、`desired_speed`、4つのSFMパラメータは task側と `param` 側の両方へ同じ値を書き、古いoverrideを残さない。
- `dt=1` では強すぎる力や広すぎる減衰距離を避ける。特に `r_wall > 1.0` は標準探索では使わない。

## 書き込み制約

実際のプロジェクトでこのSkillを実行するときは、プロジェクト内の `./.codex/workspace/tuning-sfm-parameter/` 以下だけに生成物を書く。ユーザーが別の作業ディレクトリを明示した場合のみ変更する。入力ファイルそのものは上書きせず、作業用コピーを作る。

## 入力

最低限、次を受け取る。

- ベースJSON
- `target_mean_speed`: 例 `[0.8, 1.0, 1.3, 1.6]`
- `person_num`: 例 `[100, 300, 500, 800]`
- `K`: 各条件の最終候補数

不足している場合は、ベースJSONにある `desired_speed` と `person_num` を単一条件として使える。探索範囲が未指定なら、ベースJSONの4パラメータを中心に標準レンジを使う。

## 標準探索レンジ

`dt=1` では振動を抑えることを優先し、ベース値 `(c_obs0, r_obs0, c_wall0, r_wall0)` から次の保守的な coarse 範囲を使う。

- `c_obs`: `0.70x` から `1.25x`
- `r_obs`: `0.75x` から `1.15x`
- `c_wall`: `0.70x` から `1.25x`
- `r_wall`: `0.80x` から `1.10x`、ただし上限 `0.95`

refine は coarse上位3件それぞれの周囲を探索し、`c_*` は ±10%、`r_*` は ±7.5% とする。refine候補は元のcoarse範囲内にクリップする。

strict合格がK件未満なら、`scripts/generate_candidates.py` の決定論的な低偏差系列で探索範囲を段階的に拡張する。最大拡張範囲は、`c_*: 0.35x–1.75x`、`r_obs: 0.40x–1.30x`、`r_wall: 0.40x–1.15x`（上限0.95）とする。拡張はK不足時だけ行う。

## バッチ探索ワークフロー

1. ベースJSONを読み、SFMパラメータの位置を確認する。
2. `target_mean_speed x person_num` の直積を作る。
3. `references/implementation_contract.md` の設定優先順位とCSV仕様に従い、`scripts/search_sfm.py` を1回実行して全条件を処理する。
4. coarse探索では各条件あたり標準48候補、かつ最低でも`2K`候補を評価する。
5. hard constraint に違反する候補を除外する。
6. coarse上位3件の近傍を各12件以上refineする。
7. strict合格がK件未満なら、48件単位で拡張探索を繰り返す。
8. 最大探索数に達してもstrictが不足する場合、評価可能なconstraint違反候補をfallbackとして補い、`constraint_status` と `flags` で明示する。
9. coarse/refine/拡張探索中は `output_mp4` を削除し、動画生成を行わない。
10. 各条件で厳密にK件の `parameter_sets` を `sfm_search_result.json` に出力する。K未満なら処理を失敗させる。
11. 必要なら `--render-top N` で上位候補だけ再実行して動画を生成し、目視確認する。
12. `scripts/validate_result.py` で全条件が厳密にK件か検証する。

## 実行例

```bash
python ./.codex/skills/tuning-sfm-parameter/scripts/search_sfm.py \
  --config path/to/base.json \
  --target-speeds 0.8 1.0 1.3 1.6 \
  --person-nums 100 300 500 800 \
  --top-k 5 \
  --candidates 48 \
  --rounds 4 \
  --refine-centers 3 \
  --refine-candidates 12 \
  --expansion-batch 48 \
  --max-candidates 1024 \
  --workers 4 \
  --render-top 3 \
  --workspace ./.codex/workspace/tuning-sfm-parameter
```

デフォルトでは、プロジェクトルートから次を呼ぶ。

```bash
python -m simulation.crowd_trajectory -c <generated_config.json>
```

異なる起動方法が必要なら `--command-template` を使う。

## 軌跡評価

この実装のSFM CSVは1人1行の `id,start_time,geom,goal` 形式である。`geom` はWKT `LINESTRING` であり、各座標間隔は設定した `dt` とみなす。汎用の `x/y/time` 列検出は使わない。詳細は `references/implementation_contract.md` を参照する。

`scripts/evaluate_sfm_csv.py` と `scripts/search_sfm.py` は次を計算する。

- `mean_speed`, `p95_speed`, `max_speed`
- `mean_acc`, `p95_acc`, `max_acc`: 速度ベクトル差から計算し、同じ速さで逆向きになる振動も検出する
- `mean_speed_change_acc`: 速度の大きさだけの変化量（補助指標）
- `jitter_mean_deg`, `jitter_large_turn_rate`
- `reversal_rate`, `severe_reversal_rate`: 120度/150度以上の方向反転率
- `two_step_backtrack_rate`: A-B-A型の短周期往復率
- `oscillating_agent_ratio`: 2回以上の短周期往復を持つ人物の割合
- `path_tortuosity_mean`
- `target_speed_error`
- `collision_count`, `collision_pair_rate`, `collision_frame_rate`
- `output_person_num`, `output_person_ratio`

人物半径は現実装のデフォルト `0.3 m` に合わせ、中心間距離 `< 0.6 m` を近接衝突とする。`start_time` と `dt` から各エージェントの時刻を同期して判定する。

CSVには `finish_reason`、壁投影回数、stuck判定、wall oscillation判定は保存されない。これらをLLMに推測させない。厳密な件数が必要ならシミュレータ側へイベント出力を追加する。

## Hard constraints

次の候補はstrict推薦として不合格とする。

- シミュレーション失敗
- CSVが存在しない、空、NaN/infで評価不能
- 明らかな数値発散
- `severe_reversal_rate > 0.10`
- `two_step_backtrack_rate > 0.08`
- `oscillating_agent_ratio > 0.15`

閾値未満でも `reversal_rate > 0.03`、`two_step_backtrack_rate > 0.02`、`oscillating_agent_ratio > 0.05` は flags を付け、スコアで強く減点する。

strict候補がK件に満たず最大探索数へ達した場合、CSVが正常に評価できたhard constraint違反候補だけをfallback推薦へ使える。fallbackには `constraint_status: "fallback"` と具体的な `hard_*` flagを必ず付け、strict候補より後に並べる。不正なCSVやシミュレーション失敗は決して推薦しない。

壁侵入が継続する場合、パラメータ探索だけで解決せず、有効領域への投影または壁横断判定の修正を優先する。

## スコアリング

スコアはPython側で固定し、LLMに毎回判断させない。

標準では、利用可能なメトリクスだけを正規化して加重平均する。

- target speed agreement: 0.25
- vector acceleration: 0.20
- oscillation (reversal/backtrack/oscillating agents): 0.25
- path tortuosity: 0.05
- collision penalty: 0.20
- output person ratio: 0.05

振動抑制を優先するため、平均方向変化だけではなく短周期の反転・往復を独立に評価する。速度の大きさが一定でも方向が反転すれば vector acceleration と oscillation の両方で減点する。

CSVから直接取得できない壁投影・wall oscillationはスコアへ入れず、上位候補の動画で確認する。衝突率は定量スコアへ入れる。

研究目的に応じて重みを変える場合は、変更値を結果JSONの `scoring` に残す。

## 動画確認

動画の目視評価は上位K件、または各条件の上位3件までに限定する。確認項目は以下。

- 高速な振動やジッタがない
- 人同士の不自然な重なりが少ない
- 壁侵入や壁際振動がない
- 不自然な散乱がない
- 流れが連続している
- ROI内部で突然消えない

## 代表的な失敗モード

- 壁付近で振動: `r_wall` または `c_wall` を下げる。
- 壁に侵入: パラメータより実装上の投影・壁横断判定を確認する。
- 過剰散乱: `c_obs` または `r_obs` を下げる。
- 密集・重なり: `c_obs` または `r_obs` を少し上げる。
- 壁を無視: `c_wall` または `r_wall` を少し上げる。ただし `dt=1` では `r_wall <= 1.0` を標準とする。

## 出力形式

最終結果は必ず `references/output_schema.md` のJSON構造に従い、`sfm_search_result.json` として保存する。

各 `target_mean_speed x person_num` 条件について、`parameter_sets` は厳密にK件とする。strictを先、fallbackを後に置き、各tier内ではスコア降順にする。候補ごとの長文reasonは出力しない。`constraint_status`、短い `flags`、数値メトリクスだけにする。出力後は必ず `scripts/validate_result.py` を実行する。

## ユーザーへの返答

- 最終JSONへのパス
- 条件ごとの上位K件
- 探索した範囲と候補数
- hard reject数
- strict/fallback推薦数
- 必要なら次の狭い探索範囲

を簡潔に示す。探索途中の全ログや全候補は本文に貼らない。
