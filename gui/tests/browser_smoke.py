"""Optional browser test. Requires websocket-client and a Chrome executable."""
import base64,json,pathlib,subprocess,tempfile,time,urllib.request
import argparse
import array,hashlib,wave
import websocket
from gui import SyncPlayer

parser=argparse.ArgumentParser(description="Browser smoke test (audio must be longer than 6 seconds).")
parser.add_argument("audio")
parser.add_argument("trajectory")
parser.add_argument("--chrome", default="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
parser.add_argument("--screenshot")
args=parser.parse_args()
profile=tempfile.TemporaryDirectory(prefix='sync-player-verify-')
html_path=SyncPlayer(args.audio,args.trajectory).save(pathlib.Path(profile.name)/'viewer.html')
chrome=subprocess.Popen([args.chrome,'--headless','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-port=0','--remote-allow-origins=*','--autoplay-policy=no-user-gesture-required','--user-data-dir='+profile.name,'about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
try:
 for _ in range(100):
  portfile=pathlib.Path(profile.name)/'DevToolsActivePort'
  if portfile.exists():break
  time.sleep(.1)
 port=portfile.read_text().splitlines()[0]
 tabs=json.load(urllib.request.urlopen('http://127.0.0.1:'+port+'/json'))
 ws=websocket.create_connection(tabs[0]['webSocketDebuggerUrl'],timeout=30,suppress_origin=True)
 seq=0
 def call(method,params={}):
  global seq
  seq+=1;ws.send(json.dumps({'id':seq,'method':method,'params':params}))
  while True:
   response=json.loads(ws.recv())
   if response.get('id')==seq:
    if 'error' in response:raise RuntimeError(response['error'])
    return response.get('result',{})
 def evaluate(expression):
  result=call('Runtime.evaluate',{'expression':expression,'returnByValue':True,'awaitPromise':True})
  if 'exceptionDetails' in result:raise RuntimeError(result['exceptionDetails'])
  return result.get('result',{}).get('value')
 call('Page.enable');call('Emulation.setDeviceMetricsOverride',{'width':1250,'height':960,'deviceScaleFactor':1,'mobile':False})
 call('Page.navigate',{'url':html_path.as_uri()})
 for _ in range(200):
  state=evaluate('({ready: typeof specImage!=="undefined" && !!specImage, status:document.getElementById("status")?.textContent})')
  if state['ready']:break
  time.sleep(.1)
 print('loaded',state,flush=True)
 assert state['ready'],state
 print('data',evaluate('({duration, channels:buffer.numberOfChannels, tracks:tracks.length, ids:groups, view:timeView, gap:maxGap, points:tracks.reduce((n,t)=>n+t.points.length,0)})'),flush=True)
 assert evaluate('buffer.numberOfChannels>=1 && duration>6')
 # Compare every decoded PCM16 sample against Python's independent WAV reader.
 try:
  with wave.open(args.audio,'rb') as wav:
   if wav.getsampwidth()==2:
    count,rate,channels=wav.getnframes(),wav.getframerate(),wav.getnchannels()
    pcm=array.array('h',wav.readframes(count))
    expected=[hashlib.sha256(array.array('f',(v/32768 for v in pcm[ch::channels])).tobytes()).hexdigest() for ch in range(channels)]
    actual=evaluate('(async()=>{const hashes=[];for(let ch=0;ch<buffer.numberOfChannels;ch++){const digest=await crypto.subtle.digest("SHA-256",buffer.getChannelData(ch));hashes.push([...new Uint8Array(digest)].map(v=>v.toString(16).padStart(2,"0")).join(""));}return {rate:buffer.sampleRate,count:buffer.length,hashes,native:buffer.nativePCM};})()')
    assert actual==dict(rate=rate,count=count,hashes=expected,native=True),actual
    print('original PCM samples',dict(rate=rate,count=count,channels=channels,all_samples_identical=True),flush=True)
 except wave.Error:
  pass
 print('play',evaluate('(async()=>{await play();await new Promise(r=>setTimeout(r,350));return {playing,now:now(),state:context.state};})()'),flush=True)
 assert evaluate('playing && now()>.1')
 assert evaluate('$("colormap").value==="jet"')
 print('dB range / colorbar',evaluate('''(()=>{
   const levels=spectrum.levels,gen=spectrumGeneration,active=source,view=JSON.stringify(timeView);
   const c=$("specColorbar").getContext("2d"),original=c.fillText,labels=[];
   c.fillText=function(text,...args){labels.push(String(text));return original.call(this,text,...args);};
   try{$("dbMin").value="-140";$("dbMax").value="-20";$("spectrumSettings").requestSubmit();}
   finally{c.fillText=original;}
   if(dbMin!==-140||dbMax!==-20||!labels.includes("-140")||!labels.includes("-20")||!labels.includes("dBFS"))throw new Error("Range / colorbar labels incorrect");
   if(!playing||active!==source||gen!==spectrumGeneration||levels!==spectrum.levels||view!==JSON.stringify(timeView))throw new Error("Range edit disturbed playback / FFT / zoom");
   const image=specImage.getContext("2d").getImageData(0,0,spectrum.cols,spectrum.bins).data;
   let unclipped=false;
   for(let i=0;i<levels.length;i+=101){
     const q=Math.round(Math.max(0,Math.min(1,(levels[i]+140)/120))*255);
     const rgb=colorAt(q/255,$("colormap").value).map(Math.round);
     if(rgb.some((v,j)=>Math.abs(v-image[i*4+j])>1))throw new Error("Image colors do not match dB limits");
     if(levels[i]<-100)unclipped=true;
   }
   if(!unclipped)throw new Error("Raw values below the former display floor were lost");
   const rendered=specImage;
   for(const [lo,hi] of [["0","0"],["10","-10"],["","0"]]){
     $("dbMin").value=lo;$("dbMax").value=hi;
     if(applyDbRange()||dbMin!==-140||dbMax!==-20||specImage!==rendered||!$("dbError").textContent)throw new Error("Invalid dB limits were accepted");
   }
   $("dbMin").value="-100";$("dbMax").value="0";$("spectrumSettings").requestSubmit();
   if($("dbError").textContent||!$("specColorbar").getAttribute("aria-label").includes("-100 〜 0"))throw new Error("Colorbar / validation did not reset");
   return {ok:true,labels};
 })()'''),flush=True)
 print('colormaps',evaluate('(()=>{const levels=spectrum.levels,gen=spectrumGeneration,start=now(),view=JSON.stringify(timeView),before=specImage.getContext("2d").getImageData(0,0,1,1).data.slice();for(const map of ["viridis","warm","gray"]){$("colormap").value=map;$("colormap").dispatchEvent(new Event("change"));}const gray=specImage.getContext("2d").getImageData(0,0,1,1).data;const ok=playing&&levels===spectrum.levels&&gen===spectrumGeneration&&view===JSON.stringify(timeView)&&now()>=start&&gray[0]===gray[1]&&gray[1]===gray[2]&&before.some((v,i)=>v!==gray[i]);$("colormap").value="jet";$("colormap").dispatchEvent(new Event("change"));if(!ok)throw new Error("Palette switch changed playback/FFT or failed to recolor");return {ok};})()'),flush=True)
 print('seek',evaluate('(async()=>{seek(5);await new Promise(r=>setTimeout(r,100));return {time:now(),playing,point:currentPoint(tracks[0].points,5)};})()'),flush=True)
 assert evaluate('playing && now()>=5 && now()<5.5')
 evaluate('pause();');time.sleep(.15)
 assert evaluate('!playing && Math.abs(now()-position)<1e-8')
 print('channel',evaluate('(async()=>{$("channel").value=String(Math.min(2,buffer.numberOfChannels-1));await selectChannel();return {selected:$("channel").value,ready:!!specImage};})()'),flush=True)
 print('algorithms',evaluate('(()=>{const a=parseCSV("time,x,y,id\\r\\n0,1,2,\\\"A,B\\\"\\r\\n1,2,3,\\\"C\\\"\\\"D\\\"");const p=currentPoint([{t:0,x:0,y:0},{t:.1,x:1,y:2}],.05);return {csv:a,interpolation:p,gap:currentPoint([{t:0,x:0,y:0},{t:4,x:1,y:2}],2),outside:currentPoint([{t:0,x:0,y:0},{t:1,x:1,y:2}],-1)};})()'),flush=True)
 assert evaluate('currentPoint([{t:0,x:0,y:0},{t:4,x:1,y:2}],2)===null')
 print('mel filterbank',evaluate('''(()=>{
   const raw=samples,gen=spectrumGeneration,linear=linearSpectrum,view=JSON.stringify(timeView);
   $("spectrumType").value="mel";$("spectrumType").dispatchEvent(new Event("change"));
   if(spectrum.bins!==128||samples!==raw||spectrumGeneration!==gen||JSON.stringify(timeView)!==view)throw new Error("Mel switch disturbed source data / FFT / zoom");
   const j=30,col=Math.min(5,spectrum.cols-1),filters=melFilters(buffer.sampleRate,2048),weights=filters[j];
   let power=0;for(const [k,w] of weights)power+=w*10**(linear.levels[(linear.bins-1-k)*linear.cols+col]/10);
   const expected=10*Math.log10(Math.max(1e-16,power)),actual=spectrum.levels[(127-j)*spectrum.cols+col];
   if(Math.abs(actual-expected)>.0001)throw new Error("Mel power sum incorrect");
   if(Math.abs(melToHz(hzToMel(1000))-1000)>1e-8)throw new Error("Mel axis incorrect");
   const mel=melSpectrum;
   $("spectrumType").value="linear";$("spectrumType").dispatchEvent(new Event("change"));
   if(spectrum!==linear||melSpectrum!==mel)throw new Error("Spectrum cache incorrect");
   return {bands:mel.bins,powerSumVerified:true,originalSamplesUnchanged:true};
 })()'''),flush=True)
 print('zoom',evaluate('(()=>{$("wave").dispatchEvent(new WheelEvent("wheel",{deltaY:-300,clientX:300,clientY:300,cancelable:true}));return timeView;})()'),flush=True)
 assert evaluate('timeView[1]-timeView[0]<duration')
 print('coordinate overlay',evaluate('(()=>{const c=$("trajectory").getContext("2d"),original=c.fillText,rendered=[];c.fillText=function(text,...args){rendered.push(String(text));return original.call(this,text,...args);};try{drawTrajectory(5.05);}finally{c.fillText=original;}const p=currentPoint(tracks[0].points,5.05-offset);if(!p)throw new Error("Sample needs a valid trajectory at 5.05 seconds");const expected=positionLabel(tracks[0],p,$("map_x").selectedOptions[0].textContent,$("map_y").selectedOptions[0].textContent);if(!expected.every(line=>rendered.includes(line)))throw new Error("Coordinate overlay missing");return expected;})()'),flush=True)
 evaluate('$("spectrumType").value="mel";updateSpectrumView();resetTime();position=5;draw();')
 shot=call('Page.captureScreenshot',{'format':'png','captureBeyondViewport':False})
 if args.screenshot:pathlib.Path(args.screenshot).write_bytes(base64.b64decode(shot['data']))
 print('end',evaluate('(async()=>{seek(duration-.15);await play();await new Promise(r=>setTimeout(r,400));return {playing,position};})()'),flush=True)
 assert evaluate('!playing && position===duration')
 print('PASS',flush=True)
finally:
 chrome.terminate();chrome.wait(timeout=10)
 profile.cleanup()
