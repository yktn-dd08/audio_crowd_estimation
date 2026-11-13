"""Paired before/after evaluation and report for the AFPILD MBSS experiment."""
import os
os.environ['MPLBACKEND']='Agg'
import json
import html
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import spatial, signal
import matplotlib.pyplot as plt
from analysis.afpild_evaluate import ROOT,FS,HOP,EPS,features,load_audio

P=ROOT/'workspace/afpild/evaluation'
OUT=P/'denoise_first3s'
METRICS={'spectral_cosine':('PSD cosine ↑',True),
 'spectral_js_distance':('PSD JS distance ↓',False),
 'gain_invariant_log_spectral_rmse_db':('Log-spectrum RMSE (dB) ↓',False),
 'mfcc_1_13_distance':('MFCC c1–c13 distance ↓',False),
 'envelope_correlation':('RMS envelope correlation ↑',True),
 'best_envelope_correlation_250ms':('Best RMS correlation ±250ms ↑',True)}


def paired_summary(before,after,keys):
    merged=before.merge(after,on=['name','subject','session','channel'],suffixes=('_before','_after'),validate='one_to_one')
    rng=np.random.default_rng(20261111); rows=[]
    for key in keys:
        delta=merged[f'{key}_after']-merged[f'{key}_before']
        means=delta.groupby(merged.subject).mean().to_numpy()
        boot=means[rng.integers(0,len(means),(5000,len(means)))].mean(axis=1)
        recording_delta=delta.groupby(merged.name).mean()
        high=METRICS[key][1] if key in METRICS else None
        rows.append({'metric':key,'before_mean':merged[f'{key}_before'].mean(),
                     'after_mean':merged[f'{key}_after'].mean(),'delta':delta.mean(),
                     'delta_ci95_low':np.quantile(boot,.025),'delta_ci95_high':np.quantile(boot,.975),
                     'improved_recordings':int(((recording_delta>0) if high else (recording_delta<0)).sum()) if high is not None else None})
    return merged,pd.DataFrame(rows)


def baseline_and_bands(names):
    sp=[]; raw=[]; clean=[]
    for name in names:
        old=np.load(P/f'cache/{name}.npz'); new=np.load(OUT/f'cache/{name}.npz')
        assert np.allclose(old['sim_spectrum'],new['sim_spectrum'],rtol=1e-9,atol=1e-12)
        sp.append(old['sim_spectrum']);raw.append(old['real_spectrum']);clean.append(new['clean_spectrum'])
    sp,raw,clean=np.array(sp),np.array(raw),np.array(clean)
    other=np.array([[a[:3]!=b[:3] for b in names] for a in names])
    rows=[]
    for label,real in [('before',raw),('after',clean)]:
        sim=1-spatial.distance.cdist(sp,real,metric='cosine')
        for i,name in enumerate(names):
            rows.append({'name':name,'subject':name[:3].upper(),'condition':label,
                         'matched_cosine':sim[i,i],'other_subject_cosine':sim[i,other[i]].mean(),
                         'matched_advantage':sim[i,i]-sim[i,other[i]].mean()})
    null=pd.DataFrame(rows);null.to_csv(OUT/'mismatched_baseline.csv',index=False)
    freq=np.fft.rfftfreq(1024,1/FS); rows=[]
    for low,hi,label in [(0,250,'0–250Hz'),(250,2000,'250–2000Hz'),(2000,8001,'2000–8000Hz'),(250,8001,'250–8000Hz')]:
        mask=(freq>=low)&(freq<hi)
        for i,name in enumerate(names):
            rows.append({'name':name,'subject':name[:3].upper(),'band':label,
                         'before_cosine':1-spatial.distance.cosine(sp[i,mask],raw[i,mask]),
                         'after_cosine':1-spatial.distance.cosine(sp[i,mask],clean[i,mask])})
    band=pd.DataFrame(rows);band.to_csv(OUT/'band_comparison.csv',index=False)
    return null,band


def figures(before,after,summary):
    fig,ax=plt.subplots(2,3,figsize=(14,8))
    for axis,key in zip(ax.flat,METRICS):
        x=before.groupby('name')[key].mean();y=after.groupby('name')[key].mean()
        axis.scatter(x,y,s=13,alpha=.65)
        lo=min(x.min(),y.min());hi=max(x.max(),y.max());axis.plot([lo,hi],[lo,hi],'k--',lw=.8)
        axis.set(xlabel='Before denoising',ylabel='After MBSS',title=METRICS[key][0])
    fig.suptitle('AFPILD: fixed simulation versus 160 real recordings (mean over 4 channels)')
    fig.tight_layout();fig.savefig(OUT/'before_after.png',dpi=160);plt.close(fig)
    examples=json.loads((P/'examples.json').read_text())
    for name in examples:
        offset=after[after.name==name].audio_origin_s.iloc[0]
        sim,raw=load_audio(name,offset)
        clean,_=sf.read(ROOT/f'workspace/afpild/AFPILD_v1_mbss_first3s/{name[:3].upper()}/{name}_footstep_audio_mbss.wav',
                        start=round(offset*FS),frames=len(sim))
        arrays=[raw[:,0],clean[:,0],sim]
        results=[features(x) for x in arrays]
        fig,axes=plt.subplots(3,2,figsize=(14,10))
        freq=np.fft.rfftfreq(1024,1/FS)
        for label,result in zip(['Raw ch0','MBSS ch0','Simulation'],results):
            axes[0,0].plot(freq,10*np.log10(result[1]+EPS),label=label,lw=1)
        axes[0,0].set(xlabel='Frequency (Hz)',ylabel='Normalized PSD (dB)');axes[0,0].legend(fontsize=8)
        for label,result in zip(['Raw ch0','MBSS ch0','Simulation'],results):
            env=result[2];axes[0,1].plot(np.arange(len(env))*HOP/FS,env/env.mean(),label=label,lw=.7)
        axes[0,1].set(xlim=(0,10),xlabel='Time from first annotation (s)',ylabel='RMS / mean RMS');axes[0,1].legend(fontsize=8)
        for axis,result,label in zip([axes[1,0],axes[1,1],axes[2,0]],results,['Raw ch0','MBSS ch0','Simulation']):
            power=result[-1]
            axis.imshow(10*np.log10(power/power.max()+EPS),origin='lower',aspect='auto',extent=[0,power.shape[1]*HOP/FS,0,8000],vmin=-70,vmax=0,cmap='magma')
            axis.set(xlim=(0,10),title=label,xlabel='Time (s)',ylabel='Frequency (Hz)')
        axes[2,1].axis('off')
        axes[2,1].text(0, .9,'Same time crop; original simulation unchanged.\nSpectrograms: peak-relative dB per signal.\nRMS envelopes: normalized by each mean.\nLevels in these plots are not absolute SPL.',va='top',fontsize=11)
        fig.suptitle(name+': before / after MBSS / simulated');fig.tight_layout();fig.savefig(OUT/f'comparison_{name}.png',dpi=150);plt.close(fig)
        # Timbre preview: every excerpt has equal RMS, followed by one shared gain.
        clip=[x[:10*FS].copy() for x in arrays]
        equal=[x/(np.sqrt(np.mean(x*x))+EPS) for x in clip]
        joined=np.concatenate([equal[0],np.zeros(FS//2),equal[1],np.zeros(FS//2),equal[2]])
        joined*=.95/np.max(np.abs(joined))
        sf.write(OUT/f'listen_{name}_equal_rms_raw_mbss_sim.wav',joined,FS,subtype='PCM_16')
        # Shared-gain raw vs clean exposes the actual attenuation; simulation is omitted.
        joined=np.concatenate([clip[0],np.zeros(FS//2),clip[1]])
        joined*=.95/np.max(np.abs(joined))
        sf.write(OUT/f'listen_{name}_shared_gain_raw_mbss.wav',joined,FS,subtype='PCM_16')
    return examples


def table(headers,rows):
    return '<table><tr>'+''.join(f'<th>{html.escape(str(x))}</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in row)+'</tr>' for row in rows)+'</table>'


def report(before,after,summary,null,band,sensitivity,examples):
    lookup=summary.set_index('metric');cfg=json.loads((OUT/'method.json').read_text())
    body='''<h1>AFPILD：多帯域スペクトル減算の前後比較</h1><p>2026年11月11日／40名×4試行×4チャンネル＝640比較</p>
<p class="lead">実録音に多帯域スペクトル減算（MBSS）を適用し、同じ生成音・同じ時間区間・同じ指標で比較しました。各指標の改善幅と、発音タイミングの一致を分けて解釈します。</p>
<h2>方法と再現範囲</h2>
<p>AFPILD論文3.2節の「冒頭約3秒の環境音を使う多帯域スペクトル減算」に沿った実装です。ただしAFPILD論文では帯域・係数等の具体値が示されていないため、同論文の処理の完全再現ではありません。今回の具体的な式・係数はKamath &amp; Loizou (2002)の式4〜9に基づき、16kHzに適用する設定を固定しました。類似度を見てパラメータを調整していません。</p>
<ul><li>Hamming窓20ms、hop10ms、FFT512、4等分帯域0–2–4–6–8kHz。帯域係数δ＝[2.5, 2.5, 2.5, 1.5]。</li>
<li>帯域ごとのSNRからαを算出。SNR&lt;−5dBでは5、−5〜20dBでは4−0.15×SNR、20dB超では1。SNRは観測パワー/推定雑音パワー。</li>
<li>振幅スペクトルを5フレームの重み[0.09,0.25,0.32,0.25,0.09]で平滑化後、パワーを減算。非正の値は観測パワー×0.002で置換。原位相を使って逆STFT。原著が別文献を参照する追加スペクトル平滑化は実装していません。</li>
<li>主解析の雑音推定：録音冒頭3秒の完全なフレームを全て使い、チャンネル別の平均雑音パワーを算出。</li>
<li>最初の採用注釈は2.10〜3.78秒にあるため、冒頭3秒には足音等が混入する可能性があります。論文の方針をそのまま具体化した主解析に加え、以下の推定法を比較します。</li>
<li>感度分析：全160試行で、冒頭から min(3秒, 最初の採用注釈−0.2秒) までの4ch平均エネルギーが低い30%（最低20フレーム）を共通に選び、ch別に雑音を推定。これは今回補った設定です。静かなフレームも純粋な雑音である保証はありません。</li></ul>
<p>処理後WAVは4ch・16kHz・録音全長を保持するfloat32です。ピーク正規化、切り詰め、時刻補正は行っていません。生成音は変更せず、前回と同じオフセット・共通長で各chを個別に比較しました。評価用STFTは前回どおりHann窓64ms、hop16msです。</p>
<h2>全160試行の結果</h2>'''
    rows=[]
    jp={'spectral_cosine':'PSD cosine（↑）','spectral_js_distance':'PSD JS距離（↓）',
        'gain_invariant_log_spectral_rmse_db':'利得差を除いたlog-spectrum RMSE [dB]（↓）',
        'mfcc_1_13_distance':'MFCC c1〜c13距離（↓）','envelope_correlation':'RMS包絡相関（↑）',
        'best_envelope_correlation_250ms':'±250ms最適化後のRMS包絡相関（↑）'}
    for key in METRICS:
        r=lookup.loc[key]
        rows.append([jp[key],f'{r.before_mean:.4f}',f'{r.after_mean:.4f}',f'{r.delta:+.4f}',
                     f'[{r.delta_ci95_low:+.4f}, {r.delta_ci95_high:+.4f}]',f'{int(r.improved_recordings)}/160'])
    body+=table(['指標','前','後','後−前','差の95% CI','改善した試行'],rows)
    cosine=lookup.loc['spectral_cosine']; js=lookup.loc['spectral_js_distance']
    mfcc=lookup.loc['mfcc_1_13_distance']; envelope=lookup.loc['envelope_correlation']
    outcome=f'''<p class="lead"><b>再評価の結果：</b>冒頭3秒の環境音を使う主解析では、PSD cosineは{cosine.before_mean:.3f}→{cosine.after_mean:.3f}、JS距離は{js.before_mean:.3f}→{js.after_mean:.3f}、MFCC距離は{mfcc.before_mean:.2f}→{mfcc.after_mean:.2f}でした。RMS包絡相関は{envelope.before_mean:.4f}→{envelope.after_mean:.4f}で、発音の時間変動が一致するという結果にはなっていません。周波数分布の一部が近づくことと、個々の足音の時刻・音色を再現することは区別が必要です。</p>'''
    body=body.replace('<h2>方法と再現範囲</h2>',outcome+'<h2>方法と再現範囲</h2>')
    body+='''<p>上表は640比較の平均です。95% CIは被験者内の4試行・4chを平均した後、40名を単位に5000回のpaired bootstrapで算出しました。改善試行数は4ch平均の前後差で判定しています。↑／↓は類似度が良くなる方向を示します。複数指標の検定補正を行ったp値ではなく、差の推定区間です。</p>
<img src="before_after.png" alt="前後の全試行比較">
<h2>周波数分布と時間変動</h2>'''
    rows=[]
    for key,label,scale,unit in [('real_low_0_250_fraction','0〜250Hzのパワー割合',100,'%'),
                                ('real_mid_250_2000_fraction','250〜2000Hzのパワー割合',100,'%'),
                                ('real_high_2000_8000_fraction','2000Hz以上のパワー割合',100,'%'),
                                ('real_centroid_hz','スペクトル重心',1,'Hz'),
                                ('real_envelope_cv','RMS包絡のCV',1,''),
                                ('real_crest_db','クレストファクタ',1,'dB')]:
        rows.append([label,f'{before[key].mean()*scale:.3f}{unit}',f'{after[key].mean()*scale:.3f}{unit}'])
    body+=table(['実録音の特徴','前','後'],rows)
    body+=table(['比較帯域','cosine 前','cosine 後'],[[label,f'{g.before_cosine.mean():.4f}',f'{g.after_cosine.mean():.4f}'] for label,g in band.groupby('band',sort=False)])
    body+='''<p>帯域別および次の他被験者比較は、4ch平均PSDを使っています。各chのcosineを平均する全帯域表とは集計順序が異なります。帯域別の値は該当帯域内を正規化した形状の一致であり、その帯域の音圧・エネルギーの一致ではありません。</p>'''
    body+='''<p>cosineは強いスペクトル成分の重なりに影響されやすく、JS距離は正規化した分布全体、log-spectrum RMSEやMFCCは別の重みで音色の差を捉えます。そのため一つの指標だけが改善しても矛盾ではなく、残った帯域や失われた帯域の違いを確認する必要があります。</p>'''
    body+=table(['条件','対応録音とのcosine','他被験者とのcosine','対応することの優位'],[[label,f'{g.matched_cosine.mean():.4f}',f'{g.other_subject_cosine.mean():.4f}',f'{g.matched_advantage.mean():+.4f}'] for label,g in null.groupby('condition',sort=False)])
    body+=f'''<p>実録音の比較区間の総エネルギーは、処理後に平均{after.removed_energy_db.mean():.2f}dB減少しました。これは除去された雑音だけを測った値ではなく、足音成分の減衰も含み得ます。クリーンな参照録音がないため、真のSNR改善量・足音の歪み・知覚品質の改善量は確定できません。</p>'''
    body+='<h2>雑音推定区間の感度分析</h2>'
    selected=sensitivity[['name','subject','session','channel']]
    pre=before.merge(selected,on=['name','subject','session','channel'],validate='one_to_one')
    main=after.merge(selected,on=['name','subject','session','channel'],validate='one_to_one')
    rows=[]
    for key in METRICS:
        rows.append([jp[key],f'{pre[key].mean():.4f}',f'{main[key].mean():.4f}',f'{sensitivity[key].mean():.4f}'])
    body+=table(['指標（160試行640ch）','前','冒頭3秒平均推定（主解析）','静かなフレーム推定'],rows)
    body+='''<p>両推定法とも全40名・全160試行・全4チャンネルを比較しています。冒頭3秒を雑音として過大推定すると足音を消す可能性があるため、推定法に依存する変化も評価対象です。</p>
<p>静かなフレームを使う場合はcosineが0.091→0.173に上がる一方、JS距離は0.633→0.641、log-spectrum RMSEは7.164→7.727dB、MFCC距離は9.981→10.279となりました。この差から、雑音の推定方法を明記せずに「雑音除去で類似度が改善する」と一括りに結論することはできません。</p>
<h2>図と試聴</h2>
<p>代表例は前回と同じ3件を使い、今回の改善幅で選び直していません。最初の音声は「原音10秒→MBSS10秒→生成音10秒」（間に0.5秒無音）、各区間を等RMSにした音色比較用です。次の音声は原音とMBSSに共通の利得をかけ、実際の減衰の大きさを残しています。評価用WAVにはいずれの試聴処理も行っていません。</p>'''
    for name in examples:
        body+=f'<h3>{name}</h3><img src="comparison_{name}.png" alt="{name}の前後比較"><p>等RMS：原音 → MBSS → 生成音</p><audio controls src="listen_{name}_equal_rms_raw_mbss_sim.wav"></audio><p>共通利得：原音 → MBSS</p><audio controls src="listen_{name}_shared_gain_raw_mbss.wav"></audio>'
    body+='''<h2>解釈上の注意</h2>
<ul><li>周波数分布が生成音に近づくことと、雑音除去の品質が高いことは同じではありません。生成音をクリーンな正解として扱っていません。</li>
<li>スペクトル減算とその平滑化は、持続成分・弱い足音・高域成分の除去やmusical noiseを生む可能性があります。低域の全てを雑音とは断定できません。</li>
<li>前後で発音タイミング・生成イベントは変えていないため、接地イベントのF1や座標誤差は前回のままです。音量包絡の相関は、雑音除去で時間変動が見えやすくなるかを評価する別の指標です。</li>
<li>5フレームの両側平滑化を含むオフライン処理です。録音の開始時刻やサンプル数は保持しますが、過渡音の形状は変わり得ます。包絡相関の変化を物理的な接地時刻の修復と解釈できません。</li>
<li>人による聴取実験やMOS評価は行っていません。図からの特徴の解釈と、確認用の試聴素材を提供しています。</li></ul>
<p>比較図からは、持続的な背景成分が減って個々の過渡音が見えやすくなる一方、生成音にある広帯域のパルス構造と実録音の残留成分は一致していないことが読み取れます。これは図・特徴量に基づく定性的な解釈です。波形の位相一致や特定被験者の音色の再現を証明するものではありません。</p>
<h2>保存物・再実行</h2>
<ul><li>雑音除去済み全160録音：<code>workspace/afpild/AFPILD_v1_mbss_first3s/Sxx/sxx_n_footstep_audio_mbss.wav</code></li>
<li><a href="audio_metrics_after.csv">処理後640比較</a>、<a href="paired_metrics.csv">前後の対応表</a>、<a href="paired_summary.csv">指標別の差と信頼区間</a></li>
<li><a href="band_comparison.csv">帯域別比較</a>、<a href="mismatched_baseline.csv">他被験者比較</a>、<a href="../denoise/audio_metrics_after.csv">静かなフレーム推定の全件評価</a></li>
<li><a href="method.json">手法と設定</a>、<a href="summary.json">集計JSON</a>、<a href="numerical_checks.json">数値検証</a>、cache内に選択雑音フレーム・推定PSD・入力/出力SHA256</li></ul>
<pre>PYTHONPATH=. .venv/bin/python analysis/afpild_denoise_evaluate.py --noise-mode quiet
PYTHONPATH=. .venv/bin/python analysis/afpild_denoise_evaluate.py --noise-mode first3s
PYTHONPATH=. .venv/bin/python analysis/afpild_denoise_report.py</pre>
<p>完了済み処理はcacheを使って再開できます。手法設定または実装のSHA256を変えた場合は再利用せず、新しい出力先を使う必要があります。</p>
<h2>出典</h2>
<ul><li><a href="https://www.researchgate.net/publication/376251270_AFPILD_Acoustic_footstep_dataset_collected_using_one_microphone_array_and_LiDAR_sensor_for_person_identification_and_localization">Wu et al., AFPILD, 3.2節（著者公開原稿）</a>：多帯域スペクトル減算と冒頭約3秒の環境音の利用。</li>
<li><a href="https://doczz.net/doc/7989100/a-multi-band-spectral-subtraction">Kamath &amp; Loizou (2002), A multi-band spectral subtraction method for enhancing speech corrupted by colored noise（原著の全文転載）</a>：式4〜9、時間平滑化、減算係数・床値。DOI:10.1109/ICASSP.2002.5745591。</li></ul>
<p><a href="../report.html">前回のイベント・原音評価レポートへ</a></p>'''
    css='''body{max-width:1100px;margin:35px auto;padding:0 22px;background:#f8fafc;color:#19313f;font:16px/1.85 -apple-system,BlinkMacSystemFont,"Hiragino Sans",sans-serif}h1{font-size:30px}h2{margin-top:42px;border-bottom:2px solid #cedce4}table{border-collapse:collapse;width:100%;font-size:14px;margin:22px 0;background:white}th,td{padding:9px;border:1px solid #cedce4;text-align:left}th{background:#e1edf1}tr:nth-child(even){background:#f0f5f7}img{width:100%;height:auto;background:white}audio{width:100%}.lead{background:#e2f0f2;border-left:5px solid #157e87;padding:18px}a{color:#126b92}pre{padding:18px;background:#e1edf1;overflow:auto}code{font-size:13px;overflow-wrap:anywhere}li{margin:8px 0}'''
    (OUT/'report.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AFPILD 雑音除去前後の比較</title><style>'+css+'</style><body>'+body+'</body></html>')


def main():
    before=pd.read_csv(P/'audio_metrics.csv');after=pd.read_csv(OUT/'audio_metrics_after.csv')
    assert len(before)==len(after)==640
    keys=list(METRICS)+['real_low_0_250_fraction','real_centroid_hz','real_envelope_cv']
    paired,summary=paired_summary(before,after,keys)
    assert np.isfinite(after.select_dtypes('number').to_numpy()).all()
    paired.to_csv(OUT/'paired_metrics.csv',index=False);summary.to_csv(OUT/'paired_summary.csv',index=False)
    after.groupby('subject').mean(numeric_only=True).to_csv(OUT/'by_subject_after.csv')
    after.groupby('session').mean(numeric_only=True).to_csv(OUT/'by_session_after.csv')
    names=sorted(after.name.unique());null,band=baseline_and_bands(names)
    sensitivity=pd.read_csv(P/'denoise/audio_metrics_after.csv');assert len(sensitivity)==640
    _,quiet_summary=paired_summary(before,sensitivity,keys)
    quiet_summary.to_csv(OUT/'quiet_noise_paired_summary.csv',index=False)
    _,noise_summary=paired_summary(sensitivity,after,keys)
    noise_summary.to_csv(OUT/'noise_estimate_comparison.csv',index=False)
    examples=figures(before,after,summary)
    records=summary.astype(object).where(pd.notna(summary),None).to_dict('records')
    (OUT/'summary.json').write_text(json.dumps({'sessions':160,'comparisons':640,'paired_metrics':records,
                    'energy_reduction_db_mean':after.removed_energy_db.mean(),
                    'baseline_means':null.groupby('condition').mean(numeric_only=True).to_dict(),
                    'band_means':band.groupby('band').mean(numeric_only=True).to_dict(),
                    'sensitivity_comparisons':640,'examples':examples},indent=2,allow_nan=False))
    report(before,after,summary,null,band,sensitivity,examples)
    print(summary.to_string(index=False));print(OUT/'report.html')

if __name__=='__main__':main()
