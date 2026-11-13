"""Build the Japanese, local HTML report from validated AFPILD metrics."""
from pathlib import Path
import json
import html
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'workspace/afpild/evaluation'
s=json.loads((P/'summary.json').read_text())
e=s['event_means']; a=s['audio_means']
d=pd.read_csv(P/'event_detection_summary.csv')
b=pd.read_csv(P/'event_null_baseline.csv')
c=json.loads((P/'baseline_uncertainty.json').read_text())
v=pd.read_csv(P/'simulation_validation.csv')
examples=json.loads((P/'examples.json').read_text())

def table(headers,rows):
 return '<table><thead><tr>'+''.join(f'<th>{html.escape(str(x))}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'

body='''<h1>AFPILD 足音シミュレーション・評価</h1>
<p class="meta">実施日：2026年11月11日（日本時間）／40名 × 4試行＝160件</p>
<p class="lead">歩数と平均的な歩行間隔は概ね再現できました。一方、一歩ごとの発音時刻には位相ランダム化に対する優位性が見られず、音声も特定の被験者・試行の録音を再現したとは言えません。中高域の周波数分布には共通点がありますが、全帯域の分布と時間変動には大きな差があります。</p>
<h2>1. 実行した内容</h2>
<ul><li>全160入力CSVに <code>preprocess.footstep_sound.add_foot_tag</code> を実行。ファイル名の S01〜S40 と同名の foottag_template_test_Sxx.json を対応づけました。</li>
<li>保存先は指定どおり <code>workspace/data/gis/afpild/crowd/asplaht/test</code>。asplaht の綴りも指定どおりです。</li>
<li><code>simulation.crowd_audio.audio_crowd_simulation</code> で全160件を生成。room_shp は唯一の <code>data/gis/afpild/roi/afpild_roi01.shp</code>、<b>max_order=3</b>、出力は <code>workspace/afpild/sim/sxx_n/</code>。</li>
<li>未指定設定は既存関数の既定値を使用：室高3m、中央の単一マイク(0,0,0.05)m、16kHz、10秒単位の計算、追加雑音なし。ROIは6.6m四方です。</li>
<li>全40テンプレートの音源タグは同じ <code>1-155858-A-25</code>。被験者別の実録音から音源を学習・抽出したものではありません。テンプレートの height は変更せず、既存関数により foot_step = height[m] × 0.35 を使用しました。</li>
<li>乱数seed、入力・テンプレート・主要コードのSHA256、ライブラリの版、音源のインデックス順を保存しました。既存の未コミット変更は編集していません。</li></ul>
<h2>2. 評価時の同期とデータの意味</h2>
<p>正解CSVの0秒は「最初の採用イベント」です。実録音を session_manifest.csv の zero_origin_audio_time_s だけ進めてから比較しました。音声間の比較範囲は共通する長さに制限し、末尾をゼロ埋めして相関を稼ぐことはしていません。イベント時刻に対する音響伝搬・音源波形の立ち上がりの追加補正はしていません。</p>
<p><b>正解の sample_loc は足音区間の切り出し境界であり、接地瞬間を直接測った値ではありません。</b>足の座標抽出に失敗した際の身体位置による代用も含まれ、フラグがないため分離できません。したがって、以下は注釈由来イベントの再構成評価です。</p>
<p>入力軌跡も同じ注釈座標からPCHIPで1秒間隔に補間されたものです。元注釈と独立した位置推定精度の検証にはなりません。指定テンプレートが今回の正解件数などで調整済みかは検証しておらず、歩数一致を未知条件への汎化と解釈できません。</p>
<h2>3. 接地イベントの評価</h2>'''
body+=table(['指標','結果','解釈'],[
 ['正解／生成イベント数',f"{s['gt_events']:,} / {s['pred_events']:,}",'総数では408歩少ない（−0.656%）'],
 ['試行別の歩数バイアス平均',f"{e['count_bias_pct']:.3f}%",'被験者単位bootstrap 95% CI：−1.021〜−0.353%'],
 ['試行別の歩数絶対誤差平均','9.54歩','総数の誤差は試行間の過大・過小を相殺している'],
 ['平均歩行間隔：正解／生成',f"{e['gt_interval_mean_s']:.3f} / {e['pred_interval_mean_s']:.3f}秒",'平均的な歩調は近い'],
 ['歩行間隔CV：正解／生成',f"{e['gt_interval_cv']:.3f} / {e['pred_interval_cv']:.3f}",'生成側の時間間隔のばらつきが大きい'],
 ['歩行間隔分布のWasserstein距離',f"{e['interval_wasserstein_s']*1000:.1f}ms",'小さいほど分布が近い。95% CI：42.7〜51.6ms'],
 ['10秒窓ごとの歩数誤差',f"{e['count_10s_mae']:.2f}歩",f"窓内歩数の相関：{e['count_10s_correlation']:.3f}（試行平均）"],
 ['時刻差≤200msで照合した座標誤差',f"{e['xy_mae_m_at_200ms']:.3f}m",'95% CI：0.160〜0.189m。照合できたイベントのみ'],
 ['同上の方位角／半径誤差',f"{e['azimuth_mae_deg_at_200ms']:.2f}度 / {e['radius_mae_m_at_200ms']:.3f}m",'座標誤差を方向と距離に分解'],
 ['同一時刻の補間軌跡との座標誤差',f"{e['reconstruction_mae_m']:.3f}m",'全生成点。95% CI：0.083〜0.109m。独立した正解ではない']])
body+='''<p>以下は、各正解イベントと生成イベントの<b>重複しない一対一対応</b>を求めた結果です。許容差内の対応数を最大にし、同数なら時刻誤差を最小化します。Precision＝対応数/生成数、Recall＝対応数/正解数。集計は全イベントを合算したmicro値です。</p>'''
rows=[]
for r in d.itertuples():
 rows.append([f'{r.time_tolerance_s*1000:.0f}ms','制限なし' if r.space_tolerance_m<0 else f'{r.space_tolerance_m:g}m',f'{r.tp:,}',f'{r.precision:.3f}',f'{r.recall:.3f}',f'{r.f1:.3f}'])
body+=table(['許容時間差','許容座標差','対応数','Precision','Recall','F1'],rows)
body+=f'''<p><b>200msでのF1約0.697だけでは、正確な足取りの再現とは言えません。</b>試行平均（macro）では観測F1={b.observed_f1_200ms.mean():.4f}、生成時刻を録音内で循環シフトした基準F1={b.phase_randomized_f1_200ms.mean():.4f}でした。各試行20回のランダム位相を使い、差は{c['event_null_baseline.csv']['difference']:.5f}、被験者単位bootstrap 95% CIは−0.00465〜0.00465です。個々の時刻が一致するという証拠は見られません。</p>
<p>時刻差200msで対応したイベントの時刻絶対誤差は平均{e['timing_mae_s_at_200ms']*1000:.1f}ms。ただし200ms以下だけを選んだ条件付きの値で、取りこぼした約30%のイベントを含む全体誤差ではありません。位置誤差にも同じ選択の影響があります。</p>
<p>1秒刻みの入力軌跡は末尾の端数秒を含まないため、計228個の正解イベントが軌跡の終端より後にあります。共通する時刻範囲だけに限定したmacro F1は{b.f1_common_horizon_200ms.mean():.4f}で、結論は変わりません。両列の同じ分位点を比べた時刻差の95パーセンタイルは試行平均{e['rank_time_drift_p95_s']:.2f}秒でした。これは位相ずれ・歩数差の診断であり、一歩ごとの対応誤差ではありません。</p>
<p>入力軌跡には3m/sを超える1秒区間が52か所（10試行、最大5.95m/s）あります。固定の歩幅で軌跡上を刻む方式のため、座標の飛びが短い発音間隔を生む可能性があります。これは観測に基づく原因候補であり、外れ値を修正した比較実験は今回行っていません。</p>
<h2>4. 生成音声と実録音</h2>
<p>実録音は4チャンネル、今回の生成音は1チャンネルです。実録音を平均して位相干渉させず、各チャンネルを個別に比較しました。表は160試行×4チャンネル＝640比較の平均です。STFTはHann窓1024サンプル（64ms）、hop256（16ms）。PSDは総和1にして音量の違いを除いています。</p>'''
body+=table(['指標','生成音','実録音'],[
 ['スペクトル重心',f"{a['sim_centroid_hz']:.0f}Hz",f"{a['real_centroid_hz']:.0f}Hz"],
 ['0〜250Hzのパワー割合',f"{100*a['sim_low_0_250_fraction']:.1f}%",f"{100*a['real_low_0_250_fraction']:.1f}%"],
 ['250〜2000Hzのパワー割合',f"{100*a['sim_mid_250_2000_fraction']:.1f}%",f"{100*a['real_mid_250_2000_fraction']:.1f}%"],
 ['2000Hz以上のパワー割合',f"{100*a['sim_high_2000_8000_fraction']:.1f}%",f"{100*a['real_high_2000_8000_fraction']:.1f}%"],
 ['RMS包絡の変動係数',f"{a['sim_envelope_cv']:.3f}",f"{a['real_envelope_cv']:.3f}"],
 ['クレストファクタ',f"{a['sim_crest_db']:.1f}dB",f"{a['real_crest_db']:.1f}dB"]])
body+=table(['比較指標','値','意味'],[
 ['全帯域PSD cosine',f"{a['spectral_cosine']:.3f}",'1に近いほど分布が似る。今回は低い'],
 ['PSD Jensen–Shannon距離',f"{a['spectral_js_distance']:.3f}",'自然対数、0が一致（上限√ln2）'],
 ['利得差を除いたlog-spectrum RMSE',f"{a['gain_invariant_log_spectral_rmse_db']:.2f}dB",'帯域全体の形状差'],
 ['MFCC c1〜c13の平均ベクトル間距離',f"{a['mfcc_1_13_distance']:.2f}",'大域音量c0を除外。絶対的な良否の閾値はない'],
 ['同期したRMS包絡相関',f"{a['envelope_correlation']:.4f}",'同じ時刻での音量変動はほぼ対応していない'],
 ['±250msで最適化した包絡相関',f"{a['best_envelope_correlation_250ms']:.4f}",'時刻補正で改善できる範囲も小さい。診断用'],
 ['250〜2000Hz限定のPSD cosine','0.703','4チャンネル平均PSDの帯域内正規化による比較'],
 ['2000Hz以上限定のPSD cosine','0.663','中高域の大まかな特徴には共通点がある']])
body+=f'''<p>対応する実録音とのcosineは{ s['spectral_matched_mean']:.4f}、別の被験者の全試行との平均は{s['spectral_other_subject_mean']:.4f}でした（ここは4チャンネル平均PSDを使用）。差の95% CIは−0.01099〜0.01092です。<b>特定の被験者や試行に固有の音色が再現されたという証拠はありません。</b>全テンプレートが同一の足音タグを使うという設定とも整合します。</p>
<p>定性的には、図から生成音は「背景が弱く、離散的なパルスが目立つ」「実録音より中域寄り」という特徴が読み取れます。実録音は「低域の持続成分が強い」「背景に対する足音の振幅変動が小さい」傾向があります。中高域の分布と、断続的な足音としての性質は共通しますが、音の質感や発音位置は一致していません。聴感としては生成音の方が明瞭・硬質に感じられる可能性がありますが、これは特徴量と図からの推測です。人による聴取実験やMOS評価は行っていません。</p>
<p><b>雑音除去は行っていません。</b>背景雑音と足音を混同しないよう、全帯域と帯域別の結果を併記しています。帯域別比較はPSDの該当帯域を抽出して正規化したもので、音声にフィルタや雑音除去を施したものではありません。低周波成分の全てを雑音と断定できず、足音・床の応答・マイク特性なども含みます。そのため今回の結果から「雑音を除いても似ていない」とは結論できません。</p>
<p>波形の点ごとの相関は約0ですが、異なる音源断片を使った音声では位相が一致しないので、それ自体は品質指標にしていません。実録音の自己相関の周期は低周波背景や倍周期の影響を受けるため、歩調の正解にはイベント間隔を使いました。生成音は試行ごとに最大振幅を1にして保存されるため、生のRMS差を実環境の音圧差と解釈できません。</p>
<p>既存関数の signal_info.json にある simulation_time は軌跡の開始時刻の最小・最大を記録しており、今回は全件 [0,0] です。継続時間を表すメタデータとしては使えないため、評価ではWAVのサンプル数とイベント時刻から比較範囲を計算しました。</p>
<h2>5. 全体図と代表例</h2>
<img src="overview.png" alt="全160試行の評価概要">
<p>代表例は音声PSD cosineの最小・中央値・最大から機械的に選びました。いずれも実録音はch0を図示。試聴ファイルは「実録音10秒 → 無音1秒 → 生成音10秒」の順です。各区間を等RMSにしてから全体のピークを0.95に抑えており、元の音圧比較には使えません。評価値自体はこの試聴用処理の前に計算しました。</p>'''
for name,label in zip(examples,['PSD類似度が最低','PSD類似度が中央値','PSD類似度が最高']):
 body+=f'<h3>{name}：{label}</h3><img src="comparison_{name}.png" alt="{name}の詳細比較"><audio controls src="listen_{name}_real_then_sim.wav"></audio>'
body+='''<h2>6. 解釈と次に改善する順序</h2>
<ol><li><b>接地位相をモデル化する。</b>歩数の近さだけでは実録音の発音時刻が合わないため、最初の接地位相と可変の歩幅・歩調が必要です。正解イベントを直接駆動にした比較実験を追加すると、タイミングモデルと音源・音響モデルの誤差を分離できます。</li>
<li><b>入力軌跡の品質を確認する。</b>大きな座標ジャンプと、1秒補間による局所的な曲率・速度の変化が短い発音間隔に関係する可能性があります。元注釈を保存したうえで補間法と外れ値処理を比較する価値があります。</li>
<li><b>被験者・靴条件に対応する音源を使う。</b>現在は全員同じ音源タグなので、被験者固有の音色や靴変更を再現する設計になっていません。音源抽出・調整用と評価用の録音を分離する必要があります。</li>
<li><b>録音環境を校正する。</b>現状はROIを壁として扱い、既定の材質・室高・単一マイクを使用しています。max_order=3は反射次数の指定で、実室の残響時間の再現を保証しません。実室の壁・吸音、マイクアレイ配置、周波数応答、雑音モデルを校正する余地があります。</li>
<li><b>雑音除去を行う場合は原音との両方で比較する。</b>AFPILD論文3.2節には、冒頭の約3秒の環境音を利用した多帯域スペクトル減算が記載されています。今回の解析では適用していません。実装時は手法・パラメータ・雑音推定区間を明記し、足音の低域が失われる可能性も含めて検証する必要があります。</li></ol>
<h2>7. 保存物と再実行</h2>
<ul><li>foottag付き軌跡：<code>workspace/data/gis/afpild/crowd/asplaht/test/*.csv</code>（160件）</li>
<li>生成音・接地イベント：<code>workspace/afpild/sim/sxx_n/</code> の sim_mic0000.wav、footstep_info.json、signal_info.json、crowd.csv、crowd_mic0000.csv、run_manifest.json</li>
<li>試行別結果：<a href="event_metrics.csv">event_metrics.csv</a>、<a href="event_detection.csv">event_detection.csv</a>、<a href="audio_metrics.csv">audio_metrics.csv</a>、<a href="audio_band_metrics.csv">audio_band_metrics.csv</a></li>
<li>基準との比較：<a href="event_null_baseline.csv">event_null_baseline.csv</a>、<a href="audio_mismatched_baseline.csv">audio_mismatched_baseline.csv</a>、<a href="baseline_uncertainty.json">baseline_uncertainty.json</a></li>
<li>要約・検証：<a href="summary.json">summary.json</a>、<a href="simulation_validation.csv">simulation_validation.csv</a>、<a href="numerical_checks.json">numerical_checks.json</a>、<a href="environment.json">environment.json</a>、<a href="run_manifest.json">run_manifest.json</a></li>
<li>被験者別・試行条件別結果：event_by_subject.csv、event_by_session.csv、audio_by_session.csv。照合した個別イベント：cache/sxx_n_matches.csv</li></ul>
<p>リポジトリのルートから実行。追加したスクリプトは analysis/afpild_*.py です。シミュレーションは入力等のSHA256が一致する完了済み試行を再利用します。評価を再計算する場合は evaluation/cache を別名で退避してから実行してください。</p>
<pre>PYTHONPATH=. .venv/bin/python analysis/afpild_pipeline.py
PYTHONPATH=. .venv/bin/python analysis/afpild_evaluate.py --finalize
PYTHONPATH=. .venv/bin/python analysis/afpild_supplement.py
.venv/bin/python analysis/afpild_audio_bands.py
.venv/bin/python analysis/afpild_report.py</pre>
<p>検証：全160件の必須ファイルを確認、反射次数3、非ゼロ・有限の音声、接地件数と軌跡長/歩幅の理論件数の一致、全接地点のROI内包含を確認しました。一対一対応の重複防止・対応数最大化・範囲外拒否・空間条件、およびスペクトル・MFCCの利得不変性を数値検証しました。評価中にmacOSの行列演算で出た浮動小数点警告は、別計算との最大差7.98×10⁻¹⁷、出力有限値を確認し、再実行用コードではeinsumに切り替えました。</p>
<p>95%信頼区間は、同一被験者内の4試行の依存を考慮し、被験者平均を単位とした5000回bootstrapです。母集団への一般化には40名を独立な標本と扱う仮定が必要です。単純な160試行独立としての過小な区間は使っていません。</p>
<h2>出典</h2>
<ul><li><a href="https://github.com/Cratial/AFPILD-CRNN">AFPILD公式公開リポジトリ</a>：収録条件、4試行の衣服・靴条件、注釈の定義。</li>
<li><a href="https://www.researchgate.net/publication/376251270_AFPILD_Acoustic_footstep_dataset_collected_using_one_microphone_array_and_LiDAR_sensor_for_person_identification_and_localization">Wu et al., AFPILD, Information Fusion 104, 102181, 2024（著者公開原稿）</a>：3.2節の前処理、3.3節の位置注釈処理。DOI: 10.1016/j.inffus.2023.102181。</li>
<li>ローカルの session_manifest.csv、afpild_footstep_events/README_ja.txt、AFPILD_v1_aligned/README_PROCESSING.md、AFPILD_v1_aligned_1s/README.md および既存実装を照合して時刻・座標を確認しました。</li></ul>
'''
if (P/'denoise_first3s/report.html').exists():
 body = '<p class="lead"><b>追加評価：</b><a href="denoise_first3s/report.html">多帯域スペクトル減算による雑音除去の前後比較（全160試行）</a>を保存しました。このページの音声評価は雑音除去前の結果です。</p>' + body

page='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AFPILD 足音シミュレーション評価</title><style>
body{max-width:1080px;margin:36px auto;padding:0 22px;color:#183040;background:#f8fafc;font:16px/1.85 -apple-system,BlinkMacSystemFont,"Hiragino Sans",sans-serif}h1{font-size:30px}h2{margin-top:44px;border-bottom:2px solid #c7d7df;padding-bottom:8px}h3{margin-top:32px}.meta{color:#607786}.lead{background:#e5f2f4;border-left:5px solid #167b82;padding:20px}table{border-collapse:collapse;width:100%;background:white;font-size:14px;margin:24px 0}th,td{border:1px solid #d1dce2;padding:9px 13px;text-align:left}th{background:#e4edf2}tr:nth-child(even){background:#f4f7fa}img{width:100%;height:auto;border:1px solid #d1dce2;background:white}code,pre{font-family:ui-monospace,monospace;font-size:13px;overflow-wrap:anywhere}pre{padding:18px;background:#e4edf2;overflow:auto}a{color:#126a8f}audio{width:100%;margin:14px 0}li{margin:9px 0}@media print{body{background:white}h2{break-after:avoid}img,table{break-inside:avoid}audio{display:none}}</style><body>'''+body+'</body></html>'
(P/'report.html').write_text(page)
print(P/'report.html')
