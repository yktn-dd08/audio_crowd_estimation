/* No CDN dependencies: Canvas plots and Web Audio share the audio clock. */
'use strict';
const $ = id => document.getElementById(id);
const colors = ['#6ed8ff','#ffbd69','#cb9dff','#6ae2af','#ff829a','#eee181','#81aaff'];
let context, gain, source, buffer, samples, playing=false, position=0, started=0, speed=1;
let csvRows=[], headers=[], tracks=[], groups=[], specImage=null, audioName='', csvName='';
let waveLimit=1;
let timeView=[0,1], xyView=[-1,1,-1,1], duration=0, offset=0, maxGap=.5;
let audioGeneration=0, spectrumGeneration=0;
let spectrum=null;
let linearSpectrum=null,melSpectrum=null;
let dbMin=-100,dbMax=0;
const canvases=['wave','spec','trajectory'].map($);
const status=(message,error=false)=>{ $('status').textContent=message; $('status').classList.toggle('error',error); };
const report=e=>status(e.message || String(e),true);
function audioContext(){
  if(!context){context=new (window.AudioContext || window.webkitAudioContext)(); gain=context.createGain();gain.gain.value=Number($('volume').value);gain.connect(context.destination);}
  return context;
}
function now(){return playing ? Math.min(duration,position+(context.currentTime-started)*speed) : position;}
function halt(){if(source){source.onended=null;source.stop();source.disconnect();source=null;} playing=false;}
function pause(){position=now();halt();$('play').textContent='▶ 再生';draw();}
async function play(){
  if(!buffer) return;
  await audioContext().resume();
  describeAudio();
  if(playing) return;
  if(position>=duration) position=0;
  source=context.createBufferSource();
  // A selected channel is auditioned in mono; mix averages all input channels.
  const mono=context.createBuffer(1,samples.length,buffer.sampleRate);
  mono.copyToChannel(samples,0);source.buffer=mono;source.playbackRate.value=speed;source.connect(gain);
  started=context.currentTime;playing=true;
  source.onended=()=>{position=duration;halt();$('play').textContent='▶ 再生';draw();};
  source.start(0,position);$('play').textContent='❚❚ 一時停止';
}
function seek(t){const resume=playing;pause();position=Math.max(0,Math.min(duration,t));draw();if(resume) play().catch(report);}
$('play').onclick=()=>{if(playing) pause();else play().catch(report);};
$('stop').onclick=()=>{pause();position=0;draw();};
$('seek').oninput=e=>seek(Number(e.target.value));
$('rate').onchange=()=>{const resume=playing;pause();speed=Number($('rate').value);if(resume)play().catch(report);};
$('volume').oninput=()=>{if(gain)gain.gain.value=Number($('volume').value);};
function names(){ $('names').textContent=`音声: ${audioName || '未選択'}　／　CSV: ${csvName || '未選択'}`; }
// Read uncompressed RIFF WAV directly; never pass these samples to a resampler.
function decodeWavPCM(bytes){
  const v=new DataView(bytes),tag=p=>String.fromCharCode(...new Uint8Array(bytes,p,4));
  if(bytes.byteLength<12||tag(0)!=='RIFF'||tag(8)!=='WAVE')return null;
  let format=null,data=null;
  for(let pos=12;pos+8<=v.byteLength;){
    const name=tag(pos),size=v.getUint32(pos+4,true),start=pos+8;
    if(start+size>v.byteLength)throw new Error('WAVのチャンクが途中で切れています。');
    if(name==='fmt '&&size>=16){
      let code=v.getUint16(start,true);
      if(code===0xfffe&&size>=40){
        const guid=new Uint8Array(bytes,start+24,16);
        if(guid.slice(2).some((b,i)=>b!==[0,0,0,0,16,0,128,0,0,170,0,56,155,113][i]))return null;
        code=v.getUint16(start+24,true);
      }
      format={code,channels:v.getUint16(start+2,true),sr:v.getUint32(start+4,true),stride:v.getUint16(start+12,true),bits:v.getUint16(start+14,true)};
    }
    if(name==='data')data={start,size};
    pos=start+size+(size%2);
  }
  if(!format||!data)throw new Error('WAVにfmtまたはdataチャンクがありません。');
  const {code,channels,sr,stride,bits}=format;
  if(!((code===1&&[8,16,24,32].includes(bits))||(code===3&&[32,64].includes(bits))))return null;
  if(!channels||!sr||stride!==channels*bits/8||data.size%stride)throw new Error('WAVのチャンネル数・サンプル構造が不正です。');
  const length=data.size/stride,arrays=Array.from({length:channels},()=>new Float32Array(length));
  for(let i=0;i<length;i++)for(let ch=0;ch<channels;ch++){
    const p=data.start+i*stride+ch*bits/8;
    let value;
    if(code===3)value=bits===32?v.getFloat32(p,true):v.getFloat64(p,true);
    else if(bits===8)value=(v.getUint8(p)-128)/128;
    else if(bits===16)value=v.getInt16(p,true)/32768;
    else if(bits===24){let n=v.getUint8(p)|(v.getUint8(p+1)<<8)|(v.getUint8(p+2)<<16);if(n&0x800000)n-=0x1000000;value=n/8388608;}
    else value=v.getInt32(p,true)/2147483648;
    if(!Number.isFinite(value))throw new Error('WAVに非有限の振幅値があります。');
    arrays[ch][i]=value;
  }
  return {sampleRate:sr,numberOfChannels:channels,length,duration:length/sr,getChannelData:ch=>arrays[ch],nativePCM:true};
}
function flacSampleRate(bytes){
  const v=new Uint8Array(bytes);
  // STREAMINFO is the first metadata block in a native FLAC stream.
  if(v.length>=42&&String.fromCharCode(...v.slice(0,4))==='fLaC'&&(v[4]&127)===0)
    return (v[18]<<12)|(v[19]<<4)|(v[20]>>4);
  return null;
}
async function decodeInput(bytes){
  const pcm=decodeWavPCM(bytes);if(pcm)return pcm;
  const sr=flacSampleRate(bytes);
  if(sr){
    const offline=new OfflineAudioContext(1,1,sr);
    const decoded=await offline.decodeAudioData(bytes);
    decoded.nativeRate=true;return decoded;
  }
  // Unknown/compressed formats: do not silently claim original-rate samples.
  return audioContext().decodeAudioData(bytes);
}
function describeAudio(){
  if(!buffer)return;
  const preserved=buffer.nativePCM||buffer.nativeRate;
  $('audioDetails').textContent=`表示・解析: ${buffer.sampleRate.toLocaleString()} Hz / ${buffer.length.toLocaleString()} samples / ${buffer.numberOfChannels} ch — ${buffer.nativePCM?'WAV元サンプル（リサンプリングなし）':buffer.nativeRate?'元周波数でデコード（リサンプリングなし）':'ブラウザデコード（元周波数は未確認・変換の可能性あり）'}。再生出力: ${context?context.sampleRate.toLocaleString()+' Hz':'再生開始時に決定'}${preserved?'（出力機器に合わせた変換は再生時のみ）':''}。`;
}
async function loadAudio(bytes,name){
  const generation=++audioGeneration;
  ++spectrumGeneration;
  pause();$('play').disabled=true;$('stop').disabled=true;$('seek').disabled=true;
  status('音声を読み込み中…');
  let decoded;
  try { decoded=await decodeInput(bytes); }
  catch(e){ if(generation!==audioGeneration)return;buffer=null;samples=null;specImage=null;spectrum=null;linearSpectrum=null;melSpectrum=null;duration=0;invalidate();throw new Error('音声をデコードできません。このブラウザが対応する WAV / MP3 / FLAC 等を使用してください。 '+e.message); }
  if(generation!==audioGeneration)return;
  buffer=decoded;duration=buffer.duration;position=0;timeView=[0,duration];audioName=name;names();
  describeAudio();
  $('seek').max=duration; $('seek').disabled=false;$('play').disabled=false;$('stop').disabled=false;
  const ch=$('channel');ch.replaceChildren(new Option('全Ch平均','mix'));
  for(let i=0;i<buffer.numberOfChannels;i++)ch.add(new Option(`Ch ${i+1}`,String(i)));
  ch.value='0'; ch.disabled=false;
  await selectChannel();
}
async function selectChannel(){
  if(!buffer)return;
  pause();audioLayers=null;const b=buffer, v=$('channel').value;
  if(v==='mix'){
    samples=new Float32Array(b.length);
    for(let ch=0;ch<b.numberOfChannels;ch++){const a=b.getChannelData(ch);for(let i=0;i<a.length;i++)samples[i]+=a[i]/b.numberOfChannels;}
  }else samples=b.getChannelData(Number(v));
  let peak=0;for(let i=0;i<samples.length;i++)peak=Math.max(peak,Math.abs(samples[i]));
  waveLimit=Math.max(1e-5,peak*1.1);
  await makeSpectrum(samples,b.sampleRate);
}
$('channel').onchange=()=>selectChannel().catch(report);
$('audioFile').onchange=async e=>{const file=e.target.files[0];if(file)try{await loadAudio(await file.arrayBuffer(),file.name);}catch(error){report(error);}};
$('csvFile').onchange=async e=>{const file=e.target.files[0];if(file)try{loadCSV(await file.text(),file.name);}catch(error){report(error);}};
// RFC 4180-style parsing, including escaped quotes, CRLF and quoted newlines.
function parseCSV(text){
  text=text.replace(/^\uFEFF/,'');let rows=[],row=[],field='',quoted=false;
  for(let i=0;i<text.length;i++){
    const c=text[i];
    if(c==='"'){if(quoted && text[i+1]==='"'){field+='"';i++;}else if(quoted || field==='')quoted=!quoted;else throw new Error('CSVの引用符が不正です。');}
    else if(c===',' && !quoted){row.push(field);field='';}
    else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&text[i+1]==='\n')i++;row.push(field);if(row.some(v=>v.trim()!==''))rows.push(row);row=[];field='';}
    else field+=c;
  }
  if(quoted)throw new Error('CSVの引用符が閉じられていません。');
  row.push(field);if(row.some(v=>v.trim()!==''))rows.push(row);
  if(rows.length<2)throw new Error('CSVには列名と1行以上のデータが必要です。');
  const width=rows[0].length;
  if(rows.some(r=>r.length!==width))throw new Error('CSVの列数が一致しません。カンマ区切りのCSVを指定してください。');
  return rows;
}
const aliases={time:['clip_time_s','time_s','timestamp','time','t','datetime','original_time_s'],id:['source_id','person_id','agent_id','track_id','id'],segment:['segment_id','segment'],x:['x_m','x','pos_x','position_x'],y:['y_m','y','pos_y','position_y']};
function loadCSV(text,name){
  const parsed=parseCSV(text);const nextHeaders=parsed[0].map(s=>s.trim());
  if(new Set(nextHeaders).size!==nextHeaders.length)throw new Error('CSVに重複した列名があります。');
  headers=nextHeaders;csvRows=parsed.slice(1);csvName=name;names();$('mapping').replaceChildren();
  for(const [key,label] of Object.entries({time:'時刻',id:'人物ID',segment:'区間ID',x:'X座標',y:'Y座標'})){
    const select=document.createElement('select');select.id=`map_${key}`;select.add(new Option('選択なし',''));
    headers.forEach((h,i)=>select.add(new Option(h,String(i))));
    const found=aliases[key].map(a=>headers.findIndex(h=>h.toLowerCase()===a)).find(i=>i>=0);
    select.value=found===undefined?'':String(found);
    const wrap=document.createElement('label');wrap.append(label,select);$('mapping').append(wrap);
  }
  // Estimate a safe interpolation limit from within-segment sampling intervals.
  const ti=$('map_time').value, si=$('map_segment').value, ii=$('map_id').value;
  const byKey=new Map();
  if(ti!=='')for(const r of csvRows){const key=JSON.stringify([ii===''?'':r[ii],si===''?'':r[si]]);const t=parseTime(r[ti]);if(Number.isFinite(t)){if(!byKey.has(key))byKey.set(key,[]);byKey.get(key).push(t);}}
  const steps=[];for(const times of byKey.values()){times.sort((a,b)=>a-b);for(let i=1;i<times.length;i++)if(times[i]>times[i-1])steps.push(times[i]-times[i-1]);}
  steps.sort((a,b)=>a-b);$('gap').value=steps.length?Math.max(.001,steps[Math.floor(steps.length/2)]*3).toPrecision(5):.5;
  applyMapping();
}
function parseTime(value){if(value===undefined || value.trim()==='')return NaN;const n=Number(value);return Number.isFinite(n)?n:Date.parse(value)/1000;}
function applyMapping(){
  try{
    if(!csvRows.length)return;
    const mapping=Object.fromEntries(Object.keys(aliases).map(k=>[k,$(`map_${k}`).value]));
    for(const key of ['time','x','y'])if(mapping[key]==='')throw new Error(`${key} の列を選択して「設定を適用」を押してください。`);
    const nextOffset=Number($('offset').value), nextGap=Number($('gap').value);
    if(!Number.isFinite(nextOffset)||!Number.isFinite(nextGap)||nextGap<=0)throw new Error('オフセットは有限の数、最大間隔は正の数を指定してください。');
    const grouped=new Map();let invalid=0;
    for(const row of csvRows){
      const t=parseTime(row[mapping.time]), x=row[mapping.x].trim()===''?NaN:Number(row[mapping.x]), y=row[mapping.y].trim()===''?NaN:Number(row[mapping.y]);
      if(![t,x,y].every(Number.isFinite)){invalid++;continue;}
      const id=mapping.id===''?'track':row[mapping.id], segment=mapping.segment===''?'':row[mapping.segment];
      const key=JSON.stringify([id,segment]);if(!grouped.has(key))grouped.set(key,{id,segment,points:[]});grouped.get(key).points.push({t,x,y});
    }
    if(!grouped.size)throw new Error('有効な時刻・座標がありません。列設定を確認してください。');
    offset=nextOffset;maxGap=nextGap;groups=[...new Set([...grouped.values()].map(g=>g.id))];
    tracks=[...grouped.values()].map(g=>{g.points.sort((a,b)=>a.t-b.t);g.points=g.points.filter((p,i,a)=>i===a.length-1||p.t!==a[i+1].t);g.color=colors[groups.indexOf(g.id)%colors.length];return g;});
    resetXY();$('legend').replaceChildren();for(const [i,id] of groups.entries()){const span=document.createElement('span');span.style.color=colors[i%colors.length];span.textContent=`● ${id}`;$('legend').append(span);}
    const times=tracks.flatMap(g=>[g.points[0].t,g.points.at(-1).t]);const lo=Math.min(...times)+offset,hi=Math.max(...times)+offset;
    status(`${csvRows.length-invalid} 行 / ${groups.length} ID / 軌跡 ${lo.toFixed(3)}–${hi.toFixed(3)} 秒${invalid?`（無効な ${invalid} 行を除外）`:''}${buffer&&(lo>duration||hi<0)?' ⚠ 音声時間と重なっていません。オフセットを調整してください。':''}`);draw();
  }catch(e){report(e);}
}
$('apply').onclick=applyMapping;
function resetXY(){
  let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
  for(const g of tracks)for(const p of g.points){xmin=Math.min(xmin,p.x);xmax=Math.max(xmax,p.x);ymin=Math.min(ymin,p.y);ymax=Math.max(ymax,p.y);}
  if(!Number.isFinite(xmin)){xyView=[-1,1,-1,1];return;}
  const dx=Math.max(xmax-xmin,.1)*.1,dy=Math.max(ymax-ymin,.1)*.1;xyView=[xmin-dx,xmax+dx,ymin-dy,ymax+dy];
}
function currentPoint(points,t){
  if(!points.length||t<points[0].t-1e-8||t>points.at(-1).t+1e-8)return null;
  if(t<=points[0].t)return points[0];if(t>=points.at(-1).t)return points.at(-1);
  let lo=0,hi=points.length-1;while(hi-lo>1){const m=(lo+hi)>>1;if(points[m].t<=t)lo=m;else hi=m;}
  const a=points[lo],b=points[hi];if(Math.abs(t-a.t)<1e-8)return a;if(Math.abs(t-b.t)<1e-8)return b;if(b.t-a.t>maxGap)return null;
  const u=(t-a.t)/(b.t-a.t);return {x:a.x+(b.x-a.x)*u,y:a.y+(b.y-a.y)*u};
}
// Radix-2 FFT; bound STFT columns to keep long recordings interactive.
function fft(re,im){
  const n=re.length;for(let i=1,j=0;i<n;i++){let bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){[re[i],re[j]]=[re[j],re[i]];[im[i],im[j]]=[im[j],im[i]];}}
  for(let len=2;len<=n;len<<=1){const angle=-2*Math.PI/len,wr=Math.cos(angle),wi=Math.sin(angle);for(let i=0;i<n;i+=len){let ur=1,ui=0;for(let j=0;j<len/2;j++){const a=i+j,b=a+len/2,vr=re[b]*ur-im[b]*ui,vi=re[b]*ui+im[b]*ur;re[b]=re[a]-vr;im[b]=im[a]-vi;re[a]+=vr;im[a]+=vi;const nr=ur*wr-ui*wi;ui=ur*wi+ui*wr;ur=nr;}}}
}
function colorAt(value,name){
  const v=Math.max(0,Math.min(1,value));
  if(name==='jet')return [3,2,1].map(center=>255*Math.max(0,Math.min(1,1.5-Math.abs(4*v-center))));
  if(name==='gray')return [255*v,255*v,255*v];
  const stops=name==='viridis'
    ? [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]]
    : [[9,17,37],[63,44,118],[172,55,126],[244,118,83],[255,239,156]];
  const q=v*(stops.length-1),j=Math.min(stops.length-2,Math.floor(q)),f=q-j;
  return stops[j].map((c,i)=>c*(1-f)+stops[j+1][i]*f);
}
const hzToMel=hz=>2595*Math.log10(1+hz/700);
const melToHz=mel=>700*(10**(mel/2595)-1);
function melFilters(sr,n,bands=128){
  const max=hzToMel(sr/2),edges=Array.from({length:bands+2},(_,i)=>melToHz(max*i/(bands+1)));
  return Array.from({length:bands},(_,j)=>{
    const weights=[];
    for(let k=Math.max(0,Math.ceil(edges[j]*n/sr));k<=Math.min(n/2,Math.floor(edges[j+2]*n/sr));k++){
      const hz=k*sr/n,w=Math.min((hz-edges[j])/(edges[j+1]-edges[j]),(edges[j+2]-hz)/(edges[j+2]-edges[j+1]));
      if(w>0)weights.push([k,w]);
    }
    return weights;
  });
}
function makeMelSpectrum(linear,sr){
  const {cols,bins,levels}=linear,bands=128,filters=melFilters(sr,2048,bands),out=new Float32Array(cols*bands);
  for(let col=0;col<cols;col++)for(let j=0;j<bands;j++){
    let power=0;
    for(const [k,w] of filters[j])power+=w*10**(levels[(bins-1-k)*cols+col]/10);
    out[(bands-1-j)*cols+col]=10*Math.log10(Math.max(1e-16,power));
  }
  return {cols,bins:bands,levels:out};
}
function updateSpectrumView(){
  if($('spectrumType').value==='mel'){
    if(linearSpectrum&&!melSpectrum)melSpectrum=makeMelSpectrum(linearSpectrum,buffer.sampleRate);
    spectrum=melSpectrum;
  }else spectrum=linearSpectrum;
  $('spectrumTitle').textContent=$('spectrumType').value==='mel'?'メルスペクトログラム (128帯域)':'スペクトログラム (dB)';
  recolorSpectrum();
}
$('spectrumType').onchange=updateSpectrumView;
function spectralUnit(){return $('spectrumType').value==='mel'?'dB ref=1':'dBFS';}
function recolorSpectrum(){
  if(!spectrum){specImage=null;invalidate();return;}
  const {cols,bins,levels}=spectrum,image=new ImageData(cols,bins);
  const lut=Array.from({length:256},(_,i)=>colorAt(i/255,$('colormap').value));
  for(let i=0;i<levels.length;i++){
    const level=Math.round(Math.max(0,Math.min(1,(levels[i]-dbMin)/(dbMax-dbMin)))*255);
    const rgb=lut[level],idx=i*4;
    image.data[idx]=rgb[0];image.data[idx+1]=rgb[1];image.data[idx+2]=rgb[2];image.data[idx+3]=255;
  }
  specImage=document.createElement('canvas');specImage.width=cols;specImage.height=bins;
  specImage.getContext('2d').putImageData(image,0,0);invalidate();
}
$('colormap').onchange=recolorSpectrum;
function applyDbRange(){
  const lower=$('dbMin').value,upper=$('dbMax').value;
  const lo=Number(lower),hi=Number(upper);
  if(lower.trim()===''||upper.trim()===''||!Number.isFinite(lo)||!Number.isFinite(hi)||lo>=hi){
    $('dbError').textContent='下限・上限には有限の数値を入力し、下限を上限より小さくしてください。';
    return false;
  }
  dbMin=lo;dbMax=hi;$('dbError').textContent='';recolorSpectrum();return true;
}
$('spectrumSettings').onsubmit=e=>{e.preventDefault();applyDbRange();};
function drawColorbar(){
  const canvas=$('specColorbar'),{c,l,r,w}=frame(canvas),top=6,height=14;
  const gradient=c.createLinearGradient(l,0,r,0);
  for(let i=0;i<=255;i++){
    const rgb=colorAt(i/255,$('colormap').value).map(Math.round);
    gradient.addColorStop(i/255,`rgb(${rgb.join(',')})`);
  }
  c.fillStyle=gradient;c.fillRect(l,top,r-l,height);
  c.strokeStyle='#aabbd3';c.strokeRect(l,top,r-l,height);
  c.fillStyle='#aabbd3';c.textBaseline='alphabetic';
  for(let i=0;i<=4;i++){
    const x=l+(r-l)*i/4,value=dbMin+(dbMax-dbMin)*i/4;
    c.beginPath();c.moveTo(x,top+height);c.lineTo(x,top+height+4);c.stroke();
    c.textAlign=i===0?'left':i===4?'right':'center';
    c.fillText(Number(value.toPrecision(5)).toString(),x,top+height+18);
  }
  c.textAlign='right';c.fillText(spectralUnit(),w-4,52);
  canvas.setAttribute('aria-label',`カラーバー: ${dbMin} 〜 ${dbMax} ${spectralUnit()} / ${$('colormap').value}`);
}
async function makeSpectrum(data,sr){
  const token=++spectrumGeneration;specImage=null;spectrum=null;linearSpectrum=null;melSpectrum=null;audioLayers=null;status('スペクトログラムを計算中…');draw();
  const n=2048,bins=n/2+1,hop=Math.max(n/4,Math.ceil(data.length/1800)),cols=Math.max(1,Math.ceil(data.length/hop));
  // Keep unclipped dB values so display limits can change without another FFT.
  const levels=new Float32Array(cols*bins),re=new Float64Array(n),im=new Float64Array(n),win=Float64Array.from({length:n},(_,i)=>.5-.5*Math.cos(2*Math.PI*i/(n-1)));
  for(let col=0;col<cols;col++){
    const center=(col+.5)*data.length/cols;im.fill(0);
    for(let i=0;i<n;i++)re[i]=(data[Math.floor(center)-n/2+i]||0)*win[i];fft(re,im);
    for(let k=0;k<bins;k++){
      const db=20*Math.log10(Math.max(1e-8,Math.hypot(re[k],im[k])*2/(n/2)));
      levels[(bins-1-k)*cols+col]=db;
    }
    if(col%40===0){await new Promise(resolve=>setTimeout(resolve,0));if(token!==spectrumGeneration)return;}
  }
  if(token!==spectrumGeneration)return;
  linearSpectrum={cols,bins,levels};updateSpectrumView();
  status(`表示・解析 ${sr.toLocaleString()} Hz / ${buffer.numberOfChannels} ch / ${duration.toFixed(3)} 秒。2048点 Hann窓。通常: 片側振幅dBFS / メル: 三角フィルタのパワー和dB (ref=1)`);draw();
}
function frame(canvas){
  const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=rect.width,h=rect.height;
  if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
  const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.fillStyle='#111a29';c.fillRect(0,0,w,h);c.font='11px system-ui';
  return {c,w,h,l:58,r:w-16,t:12,b:h-30};
}
function formatTick(v,max){return Math.abs(v)<1e-12?'0':max>100?v.toFixed(0):Math.abs(max)<.1?v.toPrecision(2):v.toFixed(1);}
function grid(f,xmin,xmax,ymin,ymax,xlabel,ylabel,yTick=null){
  const {c,l,r,t,b,w,h}=f;c.strokeStyle='#2e3a50';c.fillStyle='#aabbd3';c.lineWidth=1;
  for(let i=0;i<=4;i++){const x=l+(r-l)*i/4,y=b-(b-t)*i/4;c.beginPath();c.moveTo(x,t);c.lineTo(x,b);c.moveTo(l,y);c.lineTo(r,y);c.stroke();c.textAlign='center';c.fillText((xmin+(xmax-xmin)*i/4).toFixed(2),x,b+16);c.textAlign='right';c.fillText(yTick?yTick(i/4):formatTick(ymin+(ymax-ymin)*i/4,ymax),l-6,y+4);}
  c.textAlign='right';c.fillText(xlabel,w-5,h-2);c.textAlign='left';c.fillText(ylabel,3,10);
}
function drawAudio(kind,timestamp){
  const f=frame($(kind)),{c,l,r,t,b}=f,[lo,hi]=timeView,span=hi-lo;
  if(kind==='spec'&&specImage&&duration)c.drawImage(specImage,lo/duration*specImage.width,0,span/duration*specImage.width,specImage.height,l,t,r-l,b-t);
  const mel=kind==='spec'&&$('spectrumType').value==='mel',nyquist=buffer?buffer.sampleRate/2:1;
  grid(f,lo,hi,kind==='wave'?-waveLimit:0,kind==='wave'?waveLimit:nyquist,'秒',kind==='wave'?'振幅 (auto)':mel?'Hz (mel)':'Hz',mel?u=>melToHz(hzToMel(nyquist)*(0.5+128*u)/129).toFixed(0):null);
  c.save();c.beginPath();c.rect(l,t,r-l,b-t);c.clip();
  if(kind==='wave'&&samples){
    c.strokeStyle='#76d8ff';c.lineWidth=1;c.beginPath();
    const sr=buffer.sampleRate;
    for(let px=0;px<r-l;px++){
      const start=Math.max(0,Math.floor((lo+px/(r-l)*span)*sr)),end=Math.min(samples.length,Math.max(start+1,Math.ceil((lo+(px+1)/(r-l)*span)*sr)));
      let min=Infinity,max=-Infinity;for(let i=start;i<end;i++){min=Math.min(min,samples[i]);max=Math.max(max,samples[i]);}
      if(end>start){c.moveTo(l+px,(t+b)/2-max/waveLimit*(b-t)/2);c.lineTo(l+px,(t+b)/2-min/waveLimit*(b-t)/2);}
    }c.stroke();
  }
  const x=l+(timestamp-lo)/span*(r-l);c.strokeStyle='#ff6c8b';c.lineWidth=2;c.beginPath();c.moveTo(x,t);c.lineTo(x,b);c.stroke();c.restore();
}
function xyGeometry(f){const [xmin,xmax,ymin,ymax]=xyView;const scale=Math.min((f.r-f.l)/(xmax-xmin),(f.b-f.t)/(ymax-ymin));return {x:v=>(f.l+f.r)/2+(v-(xmin+xmax)/2)*scale,y:v=>(f.t+f.b)/2-(v-(ymin+ymax)/2)*scale,scale};}
function positionLabel(track,p,xLabel,yLabel){
  const number=v=>(Math.abs(v)<.0005?0:v).toFixed(3);
  return [`ID: ${track.id}`,`${xLabel}: ${number(p.x)}`,`${yLabel}: ${number(p.y)}`];
}
function drawPositionLabel(c,f,px,py,lines,color){
  if(px<f.l||px>f.r||py<f.t||py>f.b)return;
  const pad=5,lineHeight=15;
  const width=Math.min(f.r-f.l-8,Math.max(...lines.map(line=>c.measureText(line).width))+pad*2);
  const height=lines.length*lineHeight+pad*2;
  const left=Math.max(f.l+4,Math.min(f.r-width-4,px+12+width>f.r?px-width-12:px+12));
  const top=Math.max(f.t+4,Math.min(f.b-height-4,py-height-8));
  c.fillStyle='rgba(10,18,31,0.88)';c.fillRect(left,top,width,height);
  c.strokeStyle=color;c.lineWidth=1;c.strokeRect(left,top,width,height);
  c.fillStyle='#fff';c.textAlign='left';c.textBaseline='top';
  lines.forEach((line,i)=>c.fillText(line,left+pad,top+pad+i*lineHeight,width-pad*2));
  c.textBaseline='alphabetic';
}
function drawTrajectory(timestamp){
  const f=frame($('trajectory')),{c,l,r,t,b}=f,g=xyGeometry(f),[xmin,xmax,ymin,ymax]=xyView;
  const cx=(xmin+xmax)/2,cy=(ymin+ymax)/2,dx=(r-l)/g.scale/2,dy=(b-t)/g.scale/2;
  const xLabel=$('map_x')?.selectedOptions[0]?.textContent||'X',yLabel=$('map_y')?.selectedOptions[0]?.textContent||'Y';
  grid(f,cx-dx,cx+dx,cy-dy,cy+dy,xLabel,yLabel);
  c.save();c.beginPath();c.rect(l,t,r-l,b-t);c.clip();
  const labels=[];
  for(const track of tracks){
    c.strokeStyle=track.color;c.fillStyle=track.color;c.globalAlpha=.4;c.lineWidth=1.5;c.beginPath();
    track.points.forEach((p,i)=>{if(!i||p.t-track.points[i-1].t>maxGap)c.moveTo(g.x(p.x),g.y(p.y));else c.lineTo(g.x(p.x),g.y(p.y));});c.stroke();
    // Also show isolated observations in the overview.
    for(let i=0;i<track.points.length;i++){const p=track.points[i];if((!i||p.t-track.points[i-1].t>maxGap)&&(i===track.points.length-1||track.points[i+1].t-p.t>maxGap)){c.beginPath();c.arc(g.x(p.x),g.y(p.y),2,0,Math.PI*2);c.fill();}}
    const p=currentPoint(track.points,timestamp-offset);if(p){c.globalAlpha=1;c.beginPath();c.arc(g.x(p.x),g.y(p.y),7,0,2*Math.PI);c.fill();c.strokeStyle='#fff';c.lineWidth=2;c.stroke();labels.push({track,p});}
  }
  c.globalAlpha=1;
  for(const {track,p} of labels)drawPositionLabel(c,f,g.x(p.x),g.y(p.y),positionLabel(track,p,xLabel,yLabel),track.color);
  c.restore();
}
// Cache static audio plots: animation only copies layers and paints the cursor.
let audioLayers=null,layerKey='';
function draw(){
  const ts=now();$('seek').value=ts;$('time').textContent=`${ts.toFixed(3)} / ${duration.toFixed(3)} s`;
  const key=JSON.stringify([timeView,canvases.slice(0,2).map(c=>[c.clientWidth,c.clientHeight]),window.devicePixelRatio]);
  if(!audioLayers || key!==layerKey){
    drawColorbar();
    audioLayers={};for(const id of ['wave','spec']){drawAudio(id,-1e100);const copy=document.createElement('canvas');copy.width=$(id).width;copy.height=$(id).height;copy.getContext('2d').drawImage($(id),0,0);audioLayers[id]=copy;}layerKey=key;
  }
  for(const id of ['wave','spec']){const f=frame($(id)),{c,l,r,t,b}=f;c.drawImage(audioLayers[id],0,0,f.w,f.h);const x=l+(ts-timeView[0])/(timeView[1]-timeView[0])*(r-l);if(x>=l&&x<=r){c.strokeStyle='#ff6c8b';c.lineWidth=2;c.beginPath();c.moveTo(x,t);c.lineTo(x,b);c.stroke();}}
  drawTrajectory(ts);
}
function invalidate(){audioLayers=null;draw();}
function resetTime(){timeView=[0,duration||1];invalidate();}
function clampTime(lo,hi){const total=duration||1,span=Math.min(total,Math.max(.0001,hi-lo));lo=Math.max(0,Math.min(total-span,lo));timeView=[lo,lo+span];}
for(const canvas of canvases){
  const xy=canvas.id==='trajectory';let drag=null;
  canvas.onwheel=e=>{e.preventDefault();const f=frame(canvas),rect=canvas.getBoundingClientRect(),mx=e.clientX-rect.left,my=e.clientY-rect.top,factor=Math.exp(Math.max(-1,Math.min(1,e.deltaY*.002)));
    if(xy){const g=xyGeometry(f),cx=(xyView[0]+xyView[1])/2,cy=(xyView[2]+xyView[3])/2,x=cx+(mx-(f.l+f.r)/2)/g.scale,y=cy-(my-(f.t+f.b)/2)/g.scale;xyView=xyView.map((v,i)=>(i<2?x:y)+(v-(i<2?x:y))*factor);}
    else{const u=Math.max(0,Math.min(1,(mx-f.l)/(f.r-f.l))),anchor=timeView[0]+u*(timeView[1]-timeView[0]),span=(timeView[1]-timeView[0])*factor;clampTime(anchor-u*span,anchor+(1-u)*span);}draw();};
  canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,time:[...timeView],xy:[...xyView],moved:false};canvas.setPointerCapture(e.pointerId);};
  canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>4)drag.moved=true;if(!drag.moved)return;const f=frame(canvas);
    if(xy){const scale=Math.min((f.r-f.l)/(drag.xy[1]-drag.xy[0]),(f.b-f.t)/(drag.xy[3]-drag.xy[2]));xyView=drag.xy.map((v,i)=>v+(i<2?-dx:dy)/scale);}
    else{const dt=-dx/(f.r-f.l)*(drag.time[1]-drag.time[0]);clampTime(drag.time[0]+dt,drag.time[1]+dt);}draw();};
  canvas.onpointerup=e=>{if(drag&&!drag.moved&&!xy&&buffer){const rect=canvas.getBoundingClientRect(),f=frame(canvas),u=Math.max(0,Math.min(1,(e.clientX-rect.left-f.l)/(f.r-f.l)));seek(timeView[0]+u*(timeView[1]-timeView[0]));}drag=null;};
  canvas.onpointercancel=()=>{drag=null;};canvas.ondblclick=()=>{if(xy){resetXY();draw();}else resetTime();};
}
document.querySelectorAll('[data-reset]').forEach(button=>button.onclick=()=>{if(button.dataset.reset==='xy'){resetXY();draw();}else resetTime();});
document.querySelectorAll('[data-full]').forEach(button=>button.onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await button.closest('.panel').requestFullscreen();}catch(e){report(new Error('このNotebookでは全画面表示が制限されています。保存したHTMLをブラウザで開いてください。'));}});
const resizeObserver=new ResizeObserver(()=>draw());
for(const canvas of canvases)resizeObserver.observe(canvas);
resizeObserver.observe($('specColorbar'));
let lastFrame=0;function tick(ms){if(playing&&ms-lastFrame>=30){draw();lastFrame=ms;}requestAnimationFrame(tick);}requestAnimationFrame(tick);
async function init(){
  try{
    const payload=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob($('payload').textContent.trim()),c=>c.charCodeAt(0))));
    if(payload.csv)loadCSV(payload.csv,payload.csvName);
    if(payload.audio)await loadAudio(Uint8Array.from(atob(payload.audio),c=>c.charCodeAt(0)).buffer,payload.audioName);
    draw();
  }catch(e){report(e);}
}
init();
