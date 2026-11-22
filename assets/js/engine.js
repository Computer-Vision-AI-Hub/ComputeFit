/* ============================== HELPERS ============================== */
const GB = 1e9;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmtGB = b => (b/GB >= 100 ? (b/GB).toFixed(0) : b/GB >= 10 ? (b/GB).toFixed(1) : (b/GB).toFixed(2)) + ' GB';
const fmtInt = x => Math.round(x).toLocaleString('en-US');
const fmtMs = ms => ms>=1000 ? (ms/1000).toFixed(ms>=10000?0:2)+' s' : ms>=10 ? Math.round(ms)+' ms' : ms.toFixed(1)+' ms';
const fmtS = s => !isFinite(s) ? '∞' : s>=120 ? (s/60).toFixed(1)+' min' : s>=10 ? s.toFixed(0)+' s' : s.toFixed(1)+' s';
const fmtUSD = x => '$'+Math.round(x).toLocaleString('en-US');
const fmtN = x => x>=100 ? fmtInt(x) : x>=10 ? x.toFixed(0) : x.toFixed(1);
const num = (id, d) => { const v = parseFloat($(id).value); return isFinite(v) && v>0 ? v : d; };
const usableFrac = hw => hw.usable || ({gpu:0.92, jetson:0.65, apple:0.7, cpu:0.6, npu:0.6})[hw.kind];
const status = (ok, margin) => ok ? (margin ? 'warn' : 'pass') : 'fail';
const hwLabel = (hw,n) => `${n>1?n+'× ':''}${hw.name}`;

/* ============================== VISION MODE ============================== */
function runtimeFor(choice, hw){
  if(choice==='pytorch') return {name:'PyTorch', speed: hw.kind==='cpu'?0.6:0.5, add: hw.kind==='gpu'?3:5, pytorch:true};
  if(choice==='onnx')    return {name:'ONNX Runtime', speed:0.7, add:1};
  if(hw.kind==='npu') return {name:hw.rt, speed:1, add:0};
  return {name:hw.rt, speed:1, add:0};
}
const VISION = {
  label:'vision',
  supports: (hw,m) => !(hw.kind==='npu' && m && !m.npu),
  evaluate(m,w,hw,n){
    const notes=[];
    if(hw.kind==='npu' && !m.npu) return {unsupported:`${hw.name} can't run ${m.name} (transformer layers aren't supported on the Hailo compiler path used here).`};
    let prec = w.prec;
    const rt = runtimeFor(w.rt, hw);
    if(hw.int8only && prec!=='int8'){ prec='int8'; notes.push('Hailo accelerators run INT8 only — the model is quantized when compiled.'); }
    if(rt.pytorch && prec==='int8'){ prec='fp16'; notes.push('PyTorch has no fast INT8 path — export to TensorRT/OpenVINO for INT8.'); }
    if(hw.kind==='npu' && w.rt!=='auto'){ notes.push('Hailo needs a compiled .hef model; ONNX/PyTorch only run on the Pi CPU.'); }
    const scale = (w.imgsz/m.base)**2;
    const gf = m.gf*scale;
    let eff = hw.cv * m.util * rt.speed;
    if(prec==='fp32') eff *= (hw.kind==='gpu'||hw.kind==='jetson') ? 0.45 : hw.kind==='apple' ? 0.6 : 1;
    if(prec==='int8') eff *= hw.int8only ? 1 : hw.int8x;
    if(hw.kind==='npu' && w.rt!=='auto'){ eff = 0.118*m.util*rt.speed; }
    const perDev = Math.ceil(w.streams/n);
    const maxB = hw.kind==='gpu' ? 8 : hw.kind==='jetson' ? 4 : 1;
    const b = Math.max(1, Math.min(perDev, maxB));
    const comp = (gf*b*(b>1?0.9:1) + hw.ck)/eff;          // ms
    const inferMs = hw.ovh + rt.add + comp;
    const ppMs = hw.pp * (m.nmsFree?0.6:1) * (m.task==='seg'?1.4:1) * Math.max(1, w.imgsz/640);
    const latency = inferMs + ppMs;
    const cpuOnly = hw.kind==='cpu' || (hw.kind==='npu' && w.rt!=='auto');
    const accFps = cpuOnly ? 1000/(inferMs+ppMs) : b*1000/inferMs;
    const ppFps = cpuOnly ? Infinity : 1000/ppMs * ({gpu:6, jetson:3, apple:6, npu:2})[hw.kind];
    const capacity = n*Math.min(accFps, ppFps);
    const need = w.streams*w.fps;
    const bytes = prec==='fp32'?4:prec==='fp16'?2:1;
    const memNeed = m.params*1e6*bytes*1.2 + gf*b*1.5e6*(bytes/2) + ({gpu:0.8, jetson:0.5, apple:0.5, cpu:0.2, npu:0.15})[hw.kind]*GB;
    const memUsable = hw.mem*GB*usableFrac(hw);
    const fits = memNeed <= memUsable;
    if(prec==='int8' && !m.int8ok) notes.push(`${m.name} loses more accuracy in INT8 than CNNs do — validate mAP after quantization.`);
    const cams = Math.floor(capacity / w.fps);
    const r = {hw,n,prec,rt,gf,b,inferMs,ppMs,latency,capacity,need,accFps,ppFps,memNeed,memUsable,fits,cams,notes,cpuOnly,
      checks:[
        {k:'Delay per frame', v:fmtMs(latency), raw:latency, ratio: latency/w.lat, target:`need ≤ ${fmtMs(w.lat)}`, st: status(latency<=w.lat, latency>w.lat*0.85)},
        {k:'Frames per second', v:fmtN(capacity), unit:'FPS', ratio: need/capacity, target:`need ${fmtN(need)} (${w.streams} × ${fmtN(w.fps)})`, st: status(capacity>=need, capacity<need*1.15)},
        {k:'Memory', v:fmtGB(memNeed), ratio: memNeed/memUsable, target:`of ${fmtGB(memUsable)} available${n>1?' per device':''}`, st: status(fits, memNeed>memUsable*0.9)},
      ]};
    r.pass = fits && latency<=w.lat && capacity>=need;
    r.headroom = Math.min(capacity/need, w.lat/latency, memUsable*0.9/memNeed);
    r.score = r.headroom;
    r.rows = [['Delay per frame', fmtMs(latency)], ['Max FPS', fmtN(capacity)], [`Cameras @ ${fmtN(w.fps)} FPS`, String(cams)], ['Power', hw.tdp*n+' W'], ['Headroom', r.headroom.toFixed(1)+'×']];
    return r;
  },
  answer(r,m,w){
    const hw = hwLabel(r.hw,r.n);
    const job = `${m.displayName} on ${w.streams} camera${w.streams>1?'s':''} at ${fmtN(w.fps)} FPS`;
    if(r.pass) return `<em>Yes.</em> Your ${esc(hw)} can run ${esc(job)} — it has capacity for about <b>${r.cams}</b> camera${r.cams===1?'':'s'}.`;
    if(!r.fits) return `<em>No.</em> ${esc(m.name)} doesn't fit in the memory of your ${esc(hw)}.`;
    const bits=[];
    if(r.capacity<r.need) bits.push(`it reaches ${fmtN(r.capacity)} FPS but you need ${fmtN(r.need)}`);
    if(r.latency>w.lat) bits.push(`each frame takes ${fmtMs(r.latency)} (limit ${fmtMs(w.lat)})`);
    return `<em>No.</em> Your ${esc(hw)} can't keep up with ${esc(job)}: ${bits.join(', and ')}.`;
  },
  sub:(r,m,w)=>`${esc(m.name)} · ${TASKNAME[m.task]} · ${w.imgsz}px · ${r.prec.toUpperCase()} · ${esc(r.rt.name)}${r.b>1?` · batch ${r.b}`:''}`,
  mem(r){ const w = r.memNeed - (r.memNeed*0); return {need:r.memNeed, usable:r.memUsable, parts:[['Model + activations + runtime', r.memNeed, '#042C53']]}; },
  why(r,m,w){
    if(!r.fits) return ['Memory','The model and its working memory are bigger than what this device has free.'];
    const latFail = r.latency>w.lat, capFail = r.capacity<r.need;
    if(!latFail && !capFail){
      const lim = r.cpuOnly ? 'CPU compute' : r.accFps <= r.ppFps ? 'accelerator compute' : 'CPU pre/post-processing';
      return ['Nothing — all targets met', `Spare capacity: ${r.headroom.toFixed(1)}×. When you add cameras, the first limit you'll hit is ${lim}.`];
    }
    if(capFail && !r.cpuOnly && r.ppFps < r.accFps) return ['CPU pre/post-processing','Resizing frames and decoding results on the CPU is slower than the model itself. Use GPU decoding (DeepStream / NVDEC) or a faster CPU.'];
    if(r.cpuOnly) return ['CPU compute','The whole model runs on the CPU. An accelerator (Hailo, Jetson, or a GPU) is typically 10–50× faster.'];
    if(latFail && r.ppMs > r.inferMs) return ['Pre/post-processing','Most of the delay is image resizing and result decoding, not the model.'];
    return ['Accelerator compute', `The model needs ${fmtN(r.gf)} GFLOPs per frame — too much for this device at ${fmtN(r.need)} FPS. A smaller model, lower input size, INT8, or a faster accelerator fixes it.`];
  },
  variants(m,w){
    const out=[];
    const precs = ['fp32','fp16','int8'];
    if(w.rt!=='auto') out.push([{...w, rt:'auto'}, m, 'export to the device\'s optimized runtime (TensorRT / OpenVINO / NCNN)', 'no accuracy change']);
    for(let i=precs.indexOf(w.prec)+1;i<precs.length;i++) out.push([{...w, rt:'auto', prec:precs[i]}, m, `use <b>${precs[i].toUpperCase()}</b> with the optimized runtime`, precs[i]==='int8'?'usually <1–2 mAP loss, needs calibration images':'no meaningful accuracy change']);
    for(const s of [512,416,320]) if(s < w.imgsz && m.base>=512) out.push([{...w, rt:'auto', imgsz:s}, m, `lower the input size to <b>${s}px</b>`, 'small objects get harder to detect']);
    const sizes = ['x','l','m','s','n'];
    const parts = m.id.match(/^(.*)-([nslmx])$/);
    if(parts && ['yolo26','yolo11','yolov8'].includes(m.fam)){
      for(let i=sizes.indexOf(parts[2])+1;i<sizes.length;i++){
        const sm = CV.find(x=>x.id===`${parts[1]}-${sizes[i]}`); if(sm) out.push([{...w, rt:'auto'}, {...sm, displayName:sm.name, catalog:m.catalog}, `switch to the smaller <b>${sm.name}</b>`, 'lower mAP — retrain and compare on your data']);
      }
    }
    return out;
  },
  deploy(m,w,r){
    const hw=r.hw, b=r.b, imgsz=w.imgsz, int8 = r.prec==='int8';
    const wt = m.catalog ? 'best' : (m.fam==='sam'||m.fam==='rtdetr') ? m.id : m.name.toLowerCase();
    if(!m.ultra) return {name:'ONNX → '+hw.rt, why:'Export the PyTorch model to ONNX, then build an optimized engine for the device.',
      cmd: hw.kind==='gpu'||hw.kind==='jetson' ? `python -c "import torch; ...; torch.onnx.export(model, x, 'model.onnx')"\ntrtexec --onnx=model.onnx --saveEngine=model.engine ${int8?'--int8':'--fp16'}${b>1?` --minShapes=input:1x3x${imgsz}x${imgsz} --optShapes=input:${b}x3x${imgsz}x${imgsz} --maxShapes=input:${b}x3x${imgsz}x${imgsz}`:''}` : `# export to ONNX, then\novc model.onnx  # OpenVINO / or onnxruntime directly`};
    let cmd, name, why;
    const file = m.catalog ? 'best' : wt;
    if(hw.kind==='npu' && w.rt==='auto'){ name='Hailo Dataflow Compiler → HailoRT'; why='Hailo runs compiled .hef files. Export ONNX, then quantize + compile with calibration images.';
      cmd=`yolo export model=${file}.pt format=onnx imgsz=${imgsz} opset=11\nhailomz compile --ckpt ${file}.onnx --hw-arch ${hw.id==='pi5h8l'?'hailo8l':'hailo8'} --calib-path ./calib_images/ --yaml <model>.yaml`; }
    else if(hw.kind==='gpu' && !hw.amd&&!hw.intel || hw.kind==='jetson'){ name = w.streams>4 ? 'TensorRT (+ DeepStream for many cameras)' : 'TensorRT'; why = w.streams>4 ? 'TensorRT engine for speed; DeepStream decodes RTSP streams on the GPU so the CPU isn\'t the bottleneck.' : 'Fastest runtime on NVIDIA hardware. Build the engine on the target device itself.';
      cmd=`yolo export model=${file}.pt format=engine imgsz=${imgsz} ${int8?'int8=True data=your_data.yaml':r.prec==='fp16'?'half=True':''}${b>1?` batch=${b} dynamic=True`:''}\nyolo predict model=${file}.engine source="rtsp://camera-ip/stream" imgsz=${imgsz}`; }
    else if(hw.amd){ name='ONNX Runtime (MIGraphX / ROCm)'; why='Best-supported path for AMD GPUs.'; cmd=`yolo export model=${file}.pt format=onnx imgsz=${imgsz} ${r.prec==='fp16'?'half=True':''}${b>1?' dynamic=True':''}`; }
    else if(hw.id==='intel'){ name='OpenVINO'; why='Intel\'s runtime — 2–3× faster than plain ONNX on Intel CPUs/iGPUs.'; cmd=`yolo export model=${file}.pt format=openvino imgsz=${imgsz} ${int8?'int8=True data=your_data.yaml':'half=True'}\nyolo predict model=${file}_openvino_model/ source=0`; }
    else if(hw.kind==='cpu'){ name='NCNN'; why='Fastest Ultralytics export for ARM CPUs like the Raspberry Pi.'; cmd=`yolo export model=${file}.pt format=ncnn imgsz=${imgsz}\nyolo predict model=${file}_ncnn_model source=0`; }
    else if(hw.kind==='apple'){ name='Core ML'; why='Uses the Apple Neural Engine and GPU.'; cmd=`yolo export model=${file}.pt format=coreml imgsz=${imgsz} ${int8?'int8=True':'half=True'} nms=True`; }
    if(m.fam==='sam'){ why += ' SAM models are loaded via Ultralytics\' SAM interface; the image encoder dominates the cost.'; cmd = `from ultralytics import SAM\nmodel = SAM("${m.id.replace('sam2.1-','sam2.1_')}.pt")\nmodel("image.jpg", bboxes=[100, 100, 400, 400])`; }
    if(m.catalog) cmd += `\n\n# or run it with the Computer Vision AI Hub container:\ndocker run ${hw.kind==='gpu'||hw.kind==='jetson'?'--gpus all ':''}-p 7860:7860 --entrypoint python docker.io/lukasiktar/computervisionaihub:latest app.py`;
    return {name, why, cmd};
  },
  tips(m,w,r){
    const t=[];
    if(r.prec==='fp32') t.push(['16','Use FP16','On GPUs and Jetson, FP16 is ~2× faster than FP32 with no practical accuracy loss.']);
    if(w.rt==='pytorch') t.push(['RT','Export instead of running .pt',`PyTorch adds Python overhead (~${r.rt.add} ms/frame) and runs ~2× slower than an exported TensorRT/OpenVINO model.`]);
    if(r.prec!=='int8' && !r.hw.int8only && r.hw.int8x>1.2 && m.int8ok) t.push(['I8','INT8 quantization',`INT8 is ~${r.hw.int8x}× faster on the model part here. Calibrate with 300–1000 real images from your cameras.`]);
    if(w.streams>4) t.push(['DS','Many cameras → decode on the GPU','At this camera count, video decoding and resizing on the CPU often becomes the real bottleneck. NVIDIA DeepStream or GStreamer with NVDEC keeps it on the GPU.']);
    if(r.b>1) t.push(['B','Batch frames',`Batching ${r.b} frames together raises throughput but adds a little delay. Your estimate already assumes batch ${r.b}.`]);
    if(w.fps>=30 && w.lat<=40) t.push(['L','Tight latency',`For ≤${w.lat} ms end-to-end, keep the camera on a wired link, use a hardware decoder, and avoid queueing more than 1–2 frames.`]);
    if(!m.nmsFree) t.push(['26','Consider YOLO26','YOLO26 is NMS-free, which removes a post-processing step and makes edge exports (TensorRT, OpenVINO, Hailo) simpler and faster.']);
    if(m.fam==='sam') t.push(['SA','Encode once, prompt many times','SAM\'s image encoder is the expensive part. For several prompts on the same image, run the encoder once and reuse the embedding.']);
    if(m.task==='reid') t.push(['TRK','Tracking = detector + tracker (+ ReID)','ByteTrack needs no neural network and adds about 1–3 ms of CPU time. Appearance ReID (BoT-SORT, DeepSORT) runs this model once per person crop, so multiply its cost by the people per frame. Use it only when people cross paths or leave and re-enter.']);
    if(m.fam==='sam' && /sam2/.test(m.id)) t.push(['TRK','Video tracking with SAM 2','SAM 2 follows a prompted object through a video using a memory of earlier frames. It still runs its image encoder on every frame, so size it at your video frame rate.']);
    if(m.task==='ocr') t.push(['OCR','Documents? Compare with a VLM','Classic OCR (detect, then recognize) is much cheaper than a VLM for printed text. Use a VLM (Florence-2, Qwen2.5-VL) when layout, handwriting or understanding the content matters.']);
    if(m.task==='recon3d') t.push(['3D','Cost grows with the number of views','The numbers are per view (VGGT) or per image pair (DUSt3R). Reconstructing 20 views costs about 20× as much; batch the views in one call and keep FP16/BF16.']);
    if(m.task==='depth') t.push(['D','Relative vs metric depth','Depth Anything V2 predicts relative depth. For distances in meters use a metric fine-tune or scale the map with a known reference.']);
    if(m.approx) t.push(['~','Estimated FLOPs','Parameter counts are published; FLOPs for this model are estimated from its architecture, so expect more error than for YOLO models — benchmark it.']);
    r.notes.forEach(n=>t.unshift(['!','Note',n]));
    return t.slice(0,6);
  },
  assumptions:[
    'Speed is calibrated to Ultralytics\' published YOLO26 latencies (T4, TensorRT FP16, batch 1) and scaled to each device by its effective compute.',
    'Delay = runtime overhead + model compute + pre/post-processing (resize, decode, and NMS for non-YOLO26 models).',
    'Compute scales with input size squared. Several cameras on a GPU are batched (up to 8 frames; 4 on Jetson).',
    'Video decoding of camera streams is not included — at many 1080p/4K streams it needs hardware decoding.',
  ],
};

/* ============================== IMAGE GEN MODE ============================== */
const GENMODE = {
  label:'gen',
  supports: hw => hw.gen && hw.kind!=='npu',
  evaluate(m,w,hw,n){
    if(!hw.gen) return {unsupported:`${hw.name} is too slow for image generation.`};
    const notes=[];
    const pr = (w.res*w.res)/(m.base*m.base);
    const wb = {fp16:2, fp8:1, nf4:0.56}[w.prec], teb = w.prec==='fp16'?2:1;
    const den = m.params*GB*wb, te = m.te*GB*teb, vae = 0.17*GB, act = m.act*GB*pr*w.batch, ctx = 0.7*GB;
    const memNeed = den + (w.offload?0:te) + vae + act + ctx;
    const memUsable = hw.mem*GB*usableFrac(hw);
    const fits = memNeed <= memUsable;
    const fp8c = w.prec==='fp8' && hw.fp8>0;
    if(w.prec==='fp8' && !hw.fp8) notes.push('This GPU has no FP8 tensor cores: FP8 saves memory but won\'t be faster.');
    const util = hw.kind==='apple' ? 0.3 : 0.15;
    const eff = hw.fp16*1e12*util*(fp8c?1.4:1)*(w.prec==='nf4'?0.85:1)*(hw.amd?0.75:1);
    const stepS = m.gf*1e9*Math.pow(pr,1.2)*w.batch/eff + 0.003;
    const vaeS = m.vae*1e9*pr*w.batch/eff;
    const teS = w.offload ? (m.te>1?2.5:0.4) : 0.05;
    const reqS = w.steps*stepS + vaeS + teS + 0.1;           // one request (batch images)
    const capacity = n*3600/reqS*w.batch;                      // images per hour
    const rho = w.perHour/capacity;
    const waitS = rho<1 ? reqS*(1 + rho/(2*(1-rho))) : Infinity;
    const r = {hw,n,memNeed,memUsable,fits,den,te:w.offload?0:te,vae,act,ctx,stepS,reqS,capacity,rho,waitS,notes,fp8c,
      checks:[
        {k:'Time per image', v: fits?fmtS(waitS):'—', ratio: fits? waitS/w.maxS : null, target: fits?`need ≤ ${fmtS(w.maxS)} · ${fmtS(reqS)} when idle`:'does not fit', st: fits?status(waitS<=w.maxS, waitS>w.maxS*0.85):'na'},
        {k:'Images per hour', v: fits?fmtInt(capacity):'—', ratio: fits? w.perHour/capacity : null, target:`need ${fmtInt(w.perHour)}`, st: fits?status(capacity>=w.perHour, capacity<w.perHour*1.15):'na'},
        {k:'GPU memory', v:fmtGB(memNeed), ratio: memNeed/memUsable, target:`of ${fmtGB(memUsable)} available`, st: status(fits, memNeed>memUsable*0.9)},
      ]};
    r.pass = fits && waitS<=w.maxS && capacity>=w.perHour;
    r.headroom = Math.min(capacity/w.perHour, w.maxS/waitS, memUsable*0.9/memNeed);
    r.score = r.headroom;
    r.rows = [['Time per image', fmtS(reqS)], ['Speed', (1/stepS).toFixed(1)+' steps/s'], ['Images per hour', fmtInt(capacity)], ['Memory used', Math.round(memNeed/memUsable*100)+'%'], ['Headroom', r.headroom.toFixed(1)+'×']];
    return r;
  },
  answer(r,m,w){
    const hw = hwLabel(r.hw,r.n);
    if(r.pass) return `<em>Yes.</em> Your ${esc(hw)} makes a ${w.res}px ${esc(m.name)} image in about <b>${fmtS(r.reqS)}</b> and handles ${fmtInt(r.capacity)} images/hour.`;
    if(!r.fits) return `<em>No.</em> ${esc(m.name)} needs ${fmtGB(r.memNeed)} but your ${esc(hw)} has ${fmtGB(r.memUsable)} free.`;
    const bits=[];
    if(r.capacity<w.perHour) bits.push(`it makes ${fmtInt(r.capacity)} images/hour, you need ${fmtInt(w.perHour)}`);
    if(r.waitS>w.maxS) bits.push(`each image takes ${fmtS(r.waitS)} at peak (limit ${fmtS(w.maxS)})`);
    return `<em>No.</em> Your ${esc(hw)} is too slow: ${bits.join(', and ')}.`;
  },
  sub:(r,m,w)=>`${esc(m.name)} · ${w.res}×${w.res} · ${w.steps} steps · ${w.prec.toUpperCase()}${w.batch>1?` · ${w.batch} per request`:''}${w.offload?' · text encoder offloaded':''}`,
  mem: r => ({need:r.memNeed, usable:r.memUsable, parts:[['Image model',r.den,'#042C53'],['Text encoder',r.te,'#0C8567'],['Activations + VAE',r.act+r.vae,'#85B7EB'],['Runtime',r.ctx,'#B9C6D6']]}),
  why(r,m,w){
    if(!r.fits) return ['GPU memory', r.te>r.den*0.3 ? `The text encoder alone takes ${fmtGB(r.te)}. Offloading it or using FP8/NF4 is the cheapest fix.` : 'The image model is larger than the free memory. Use FP8 or NF4 weights, or a bigger GPU.'];
    if(r.pass) return ['Nothing — all targets met', `Spare capacity: ${r.headroom.toFixed(1)}×. Image generation is compute-bound: more TFLOPS = faster images.`];
    return ['GPU compute', `Each image needs ${w.steps} denoising steps at ${(1/r.stepS).toFixed(1)} steps/s. Fewer steps (distilled models), FP8 on newer GPUs, or more/faster GPUs fix it.`];
  },
  variants(m,w){
    const out=[];
    if(w.prec==='fp16') out.push([{...w, prec:'fp8'}, m, 'use <b>FP8</b> weights', 'barely visible quality change']);
    if(!w.offload) out.push([{...w, offload:true}, m, '<b>offload the text encoder</b> after encoding', 'adds ~1–3 s per prompt']);
    if(w.prec!=='nf4') out.push([{...w, prec:'nf4', offload:true}, m, 'use <b>NF4 / GGUF Q4</b> weights and offload the text encoder', 'slight loss of fine detail']);
    if(m.id==='fluxd') { const s=GEN.find(x=>x.id==='fluxs'); out.push([{...w, steps:s.steps}, {...s, displayName:s.name}, 'switch to <b>FLUX.1 [schnell]</b> (4 steps)', 'faster, a bit less detail; Apache-2.0 license']); }
    if(m.id==='sdxl') { const s=GEN.find(x=>x.id==='sdxlt'); out.push([{...w, steps:s.steps}, {...s, displayName:s.name}, 'switch to <b>SDXL Lightning</b> (4 steps)', 'faster, slightly less variety']); }
    if(w.steps>20 && m.steps>8) out.push([{...w, steps:20}, m, 'use <b>20 steps</b>', 'usually close in quality']);
    if(w.res>m.base) out.push([{...w, res:m.base}, m, `generate at <b>${m.base}px</b> and upscale`, 'upscaler adds a second or two']);
    return out;
  },
  deploy(m,w,r){
    const ckpt = {sd15:'stable-diffusion-v1-5/stable-diffusion-v1-5', sdxl:'stabilityai/stable-diffusion-xl-base-1.0', sdxlt:'ByteDance/SDXL-Lightning', sd35m:'stabilityai/stable-diffusion-3.5-medium', sd35l:'stabilityai/stable-diffusion-3.5-large', fluxs:'black-forest-labs/FLUX.1-schnell', fluxd:'black-forest-labs/FLUX.1-dev'}[m.id];
    const api = w.perHour>100 || r.n>1;
    if(api) return {name:'Diffusers behind an API (one worker per GPU)', why:`For ${fmtInt(w.perHour)} images/hour, run one model copy per GPU and a queue in front.${r.hw.kind==='gpu'&&!r.hw.amd&&!r.hw.intel?' TensorRT or torch.compile adds roughly 20–40% speed.':''}`,
      cmd:`import torch\nfrom diffusers import DiffusionPipeline\npipe = DiffusionPipeline.from_pretrained("${ckpt}", torch_dtype=torch.bfloat16).to("cuda")\n${w.offload?'pipe.enable_model_cpu_offload()\n':''}${r.hw.kind==='gpu'?'pipe.transformer = torch.compile(pipe.transformer) if hasattr(pipe, "transformer") else pipe.transformer\n':''}img = pipe("a photo of ...", num_inference_steps=${w.steps}, height=${w.res}, width=${w.res}).images[0]`};
    return {name:'ComfyUI', why:'Free node-based UI with built-in FP8/GGUF support and smart memory management.',
      cmd:`git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI\npip install -r requirements.txt\npython main.py --listen${r.memNeed>r.memUsable*0.85?' --lowvram':''}\n# put the ${w.prec==='fp16'?'':w.prec.toUpperCase()+' '}${m.name} checkpoint in models/${m.id.startsWith('flux')||m.id.startsWith('sd35')?'diffusion_models':'checkpoints'}/`};
  },
  tips(m,w,r){
    const t=[];
    r.notes.forEach(n=>t.push(['!','Note',n]));
    if(m.te>1 && !w.offload) t.push(['TE','The text encoder is big',`${m.name} uses T5-XXL (~${(m.te*2).toFixed(0)} GB in FP16). It only runs once per prompt, so offloading it frees VRAM for little cost.`]);
    if(w.prec==='fp16' && r.hw.fp8>0) t.push(['F8','FP8 on this GPU',`This GPU has FP8 tensor cores: FP8 roughly halves memory and is ~1.4× faster.`]);
    if(m.cfg && w.steps>12) t.push(['DS','Distilled / turbo variants','Lightning, Turbo, Hyper-SD or LCM LoRAs bring 25–30 steps down to 4–8 with a modest quality trade-off.']);
    if(w.perHour>100) t.push(['Q','Queue and batch','At high volume, a request queue plus batching 2–4 images per call raises throughput by keeping the GPU busy.']);
    t.push(['C','Compile the model','torch.compile, TensorRT (NVIDIA) or stable-fast typically give 20–40% faster steps after a one-time warm-up.']);
    if(r.n>1) t.push(['N','Multiple GPUs','Image models don\'t split well across GPUs — run one full copy per GPU; each extra GPU adds throughput, not single-image speed.']);
    return t.slice(0,6);
  },
  assumptions:[
    'Speed = denoising steps × compute per step ÷ (GPU TFLOPS × ~15% utilization), calibrated to typical SDXL/FLUX speeds on RTX 4090 and H100.',
    'Memory = image model + text encoder (unless offloaded) + VAE + activations (grow with resolution and batch) + runtime.',
    'Peak wait uses a simple queue model: as the GPU gets busier, requests wait longer behind each other.',
    'Each GPU runs its own model copy; multi-GPU speeds up throughput, not a single image.',
  ],
};

/* ============================== LLM MODE ============================== */
const LLMMODE = {
  label:'llm',
  supports: hw => hw.llm && hw.kind!=='npu',
  evaluate(m,w,hw,n){
    if(!hw.llm) return {unsupported:`${hw.name} can't run LLMs.`};
    const prompt = Math.min(w.prompt, w.ctx);
    const wBytes = m.params*GB*PREC[w.prec].bits/8;
    const kvPerTok = (m.kvl ? m.kvl : 2*m.kv*m.hd) * m.layers * (w.kvprec==='fp8'?1:2);
    const kvReserve = kvPerTok*w.ctx*w.users;
    const usable = hw.mem*GB*usableFrac(hw)*n;
    const hidden = m.heads*m.hd;
    const overhead = (hw.kind==='gpu'?0.6:0.3)*GB*n + wBytes*0.04 + Math.min(prompt*Math.ceil(w.users*0.25),8192)*hidden*24;
    const need = wBytes + kvReserve + overhead + (w.xBytes||0);
    const fits = need<=usable;
    const maxCtx = Math.max(0, Math.floor((usable-wBytes-overhead)/(kvPerTok*w.users)));
    const bwEff = {gpu:0.75, jetson:0.5, apple:0.65, cpu:0.6}[hw.kind];
    const tpEff = n===1 ? 1 : (hw.link==='nvlink' ? {2:.92,4:.87,8:.82}[n] : {2:.82,4:.68,8:.55}[n]);
    const BW = hw.bw*GB*bwEff*n*tpEff;
    const peak = ((w.prec==='fp8' && hw.fp8>0) ? hw.fp8 : hw.fp16)*1e12*n*tpEff;
    const mfu = hw.kind==='cpu'?0.3:0.45;
    const ratio = Math.min(1, m.active/m.params);
    const touched = ratio>=1 ? 1 : 1-Math.pow(1-ratio, w.users);
    const memStep = (wBytes*touched*(ratio<1?1.4:1) + kvPerTok*(prompt+w.output/2)*w.users)/BW;
    const compStep = 2*m.active*GB*w.users/(peak*mfu);
    const commLat = n>1 ? m.layers*2*(hw.link==='nvlink'?8e-6:25e-6) : 0;
    const step = Math.max(memStep, compStep) + commLat + 1e-3;
    const tps = 1/step;
    const pf = P => 2*m.active*GB*P + 2*m.layers*P*P*hidden;
    const burst = Math.max(1, Math.ceil(w.users*0.25));
    const ttft = (w.users>1?burst:1)*(pf(prompt)+(w.xFlops||0))/(peak*mfu) + wBytes*(ratio>=1?1:Math.min(1,ratio*8))/BW + (w.users>1?step:0) + commLat*4 + 0.005;
    const r = {hw,n,wBytes,kvReserve,overhead,need,usable,fits,maxCtx,tps,ttft,agg:w.users/step,commLat,bound:memStep>=compStep?'bandwidth':'compute',notes:[],
      checks:[
        {k:'Speed per user', v: fits?fmtN(tps):'—', ratio: fits? w.t_tps/tps : null, unit: fits?'tokens/s':'', target:`need ≥ ${w.t_tps}`, st: fits?status(tps>=w.t_tps, tps<w.t_tps*1.15):'na'},
        {k:'Time to first word', v: fits?fmtMs(ttft*1000):'—', ratio: fits? ttft*1000/w.t_ttft : null, target:`need ≤ ${fmtMs(w.t_ttft)}`, st: fits?status(ttft*1000<=w.t_ttft, ttft*1000>w.t_ttft*0.87):'na'},
        {k:'Memory', v:fmtGB(need), ratio: need/usable, target:`of ${fmtGB(usable)} available`, st: status(fits, need>usable*0.9)},
      ]};
    r.pass = fits && tps>=w.t_tps && ttft*1000<=w.t_ttft;
    r.headroom = Math.min(tps/w.t_tps, w.t_ttft/(ttft*1000), usable*0.9/need);
    r.score = r.headroom;
    r.rows = [['Speed per user', fmtN(tps)+' tok/s'], ['First word', fmtMs(ttft*1000)], ['Total throughput', fmtInt(r.agg)+' tok/s'], ['Memory used', Math.round(need/usable*100)+'%'], ['Headroom', r.headroom.toFixed(1)+'×']];
    return r;
  },
  answer(r,m,w){
    const hw = hwLabel(r.hw,r.n);
    if(r.pass) return `<em>Yes.</em> Your ${esc(hw)} runs ${esc(m.name)} at about <b>${fmtN(r.tps)} tokens/s</b> per user for ${w.users} user${w.users>1?'s':''}.`;
    if(!r.fits) return `<em>No.</em> ${esc(m.name)} needs ${fmtGB(r.need)} but your ${esc(hw)} has ${fmtGB(r.usable)} free.`;
    const bits=[]; if(r.tps<w.t_tps) bits.push(`it generates ${fmtN(r.tps)} tokens/s (need ${w.t_tps})`); if(r.ttft*1000>w.t_ttft) bits.push(`the first word takes ${fmtMs(r.ttft*1000)}`);
    return `<em>No.</em> Your ${esc(hw)} is too slow: ${bits.join(', and ')}.`;
  },
  sub:(r,m,w)=>`${esc(m.name)} · ${PREC[w.prec].label} · ${w.kvprec.toUpperCase()} KV · ${w.users} user${w.users>1?'s':''} · ${fmtInt(w.ctx)} context`,
  mem: r => ({need:r.need, usable:r.usable, parts:[['Weights',r.wBytes,'#042C53'],['KV cache (context)',r.kvReserve,'#0C8567'],['Runtime',r.overhead,'#B9C6D6']]}),
  why(r,m,w){
    if(!r.fits) return ['Memory', r.wBytes+r.overhead>r.usable ? 'The model weights alone are bigger than free memory.' : `The weights fit, but the context memory for ${w.users} user(s) × ${fmtInt(w.ctx)} tokens doesn't. At this user count, max context is ~${fmtInt(r.maxCtx)} tokens.`];
    if(r.pass) return ['Nothing — all targets met', `Spare capacity: ${r.headroom.toFixed(1)}×. Next limit: memory ${r.bound}.`];
    if(r.tps<w.t_tps) return ['Memory bandwidth','Every generated token reads the whole model from memory. Fewer bytes (lower precision) or faster memory fixes it.'];
    return ['Compute (prompt processing)','Reading the prompt is heavy math. FP8 on newer GPUs, shorter prompts or prefix caching help.'];
  },
  variants(m,w){
    const order=['fp16','fp8','int8','q6','q5','q4','int4'], out=[];
    if(w.kvprec==='fp16') out.push([{...w, kvprec:'fp8'}, m, 'use an <b>FP8 KV cache</b>', 'negligible quality change']);
    for(let i=order.indexOf(w.prec)+1;i<order.length;i++) out.push([{...w, prec:order[i], kvprec:'fp8'}, m, `use <b>${PREC[order[i]].label}</b> weights + FP8 KV cache`, PREC[order[i]].quality]);
    return out;
  },
  deploy(m,w,r){
    const id = m.hf || 'your-org/your-model', gguf = ['q4','q5','q6','int8'].includes(w.prec);
    if(r.hw.kind!=='gpu' || (gguf && w.users<=4)){
      const q = {q4:'Q4_K_M',q5:'Q5_K_M',q6:'Q6_K',int8:'Q8_0',fp16:'F16',fp8:'Q8_0',int4:'IQ4_XS'}[w.prec];
      return {name:'llama.cpp / Ollama', why:'Lightweight and fast for 1–4 users; works on Jetson, Apple and CPU too.', cmd:`llama-server -m ${id.split('/').pop()}-${q}.gguf \\\n  -c ${w.ctx*w.users} -np ${w.users} -ngl ${r.hw.kind==='cpu'?0:999} -fa on`};
    }
    return {name:'vLLM', why:'Continuous batching serves many users efficiently.', cmd:`vllm serve ${id} \\\n  --tensor-parallel-size ${r.n} --max-model-len ${w.ctx} --max-num-seqs ${w.users} \\\n  --gpu-memory-utilization 0.92 --enable-prefix-caching${w.kvprec==='fp8'?' --kv-cache-dtype fp8':''}${w.prec==='fp8'?' --quantization fp8':''}`};
  },
  tips(m,w,r){
    const t=[];
    if(r.n>1 && r.hw.link==='pcie') t.push(['PC','PCIe multi-GPU adds delay',`Splitting across ${r.n} GPUs over PCIe adds ≈${(r.commLat*1000).toFixed(1)} ms per token.`]);
    if(w.users<=4) t.push(['SD','Speculative decoding','A small draft model typically gives 1.5–2.5× faster generation for few users with identical output.']);
    else t.push(['CB','Continuous batching','Use vLLM/SGLang so long prompts don\'t stall other users.']);
    if(m.active<m.params*0.5) t.push(['MoE','Mixture-of-experts',`Needs memory for all ${m.params}B parameters but runs at the speed of a ~${Math.round(m.active)}B model.`]);
    return t;
  },
  assumptions:['Memory = weights + KV cache (grows with context × users) + runtime.','Speed: each token reads all active weights from memory (bandwidth-bound) or is compute-bound at high user counts.','Typical prompt 1,000 tokens, answer 400 tokens; 25% of users sending at once for "busy" first-word time.'],
};


/* ============================== VLM MODE ==============================
   params = language-model parameters (B), vparams = vision encoder (B).
   Image tokens: fixedTok per image, or (resolution ÷ pxTok)² clamped to min/max. */
const VLMS = [
  {id:'smolvlm-256m',  name:'SmolVLM 256M',        hf:'HuggingFaceTB/SmolVLM-256M-Instruct', params:0.135, vparams:0.093, layers:30, heads:9,  kv:3,  hd:64,  pxTok:64, minTok:64, maxTok:1100, defRes:512},
  {id:'smolvlm2-2b',   name:'SmolVLM2 2.2B',       hf:'HuggingFaceTB/SmolVLM2-2.2B-Instruct', params:1.71, vparams:0.43, layers:24, heads:32, kv:32, hd:64,  pxTok:42.7, minTok:81, maxTok:1400, defRes:768},
  {id:'moondream2',    name:'Moondream 2',         hf:'vikhyatk/moondream2',                 params:1.42, vparams:0.43, layers:24, heads:32, kv:32, hd:64,  fixedTok:729, vpatches:729, defRes:378},
  {id:'florence2-l',   name:'Florence-2 Large',    hf:'microsoft/Florence-2-large',          params:0.41, vparams:0.36, layers:12, heads:16, kv:16, hd:64,  fixedTok:577, vpatches:2300, defRes:768, note:'Task-prompted (detection, OCR, captioning) rather than free chat.'},
  {id:'paligemma2-3b', name:'PaliGemma 2 3B',      hf:'google/paligemma2-3b-pt-448',         params:2.61, vparams:0.41, layers:26, heads:8,  kv:4,  hd:256, pxTok:14, minTok:256, maxTok:4096, defRes:448, resList:[224,448,896]},
  {id:'phi35-vision',  name:'Phi-3.5 Vision 4.2B', hf:'microsoft/Phi-3.5-vision-instruct',   params:3.82, vparams:0.30, layers:32, heads:32, kv:32, hd:96,  pxTok:28, minTok:144, maxTok:2400, defRes:768},
  {id:'qwen25vl-3b',   name:'Qwen2.5-VL 3B',       hf:'Qwen/Qwen2.5-VL-3B-Instruct',         params:3.09, vparams:0.67, layers:36, heads:16, kv:2,  hd:128, pxTok:28, minTok:64, maxTok:16384, defRes:768},
  {id:'gemma3-4b',     name:'Gemma 3 4B',          hf:'google/gemma-3-4b-it',                params:3.88, vparams:0.42, layers:34, heads:8,  kv:4,  hd:256, fixedTok:256, vpatches:4096, defRes:896},
  {id:'qwen25vl-7b',   name:'Qwen2.5-VL 7B',       hf:'Qwen/Qwen2.5-VL-7B-Instruct',         params:7.62, vparams:0.68, layers:28, heads:28, kv:4,  hd:128, pxTok:28, minTok:64, maxTok:16384, defRes:768},
  {id:'minicpm-v26',   name:'MiniCPM-V 2.6 8B',    hf:'openbmb/MiniCPM-V-2_6',               params:7.62, vparams:0.42, layers:28, heads:28, kv:4,  hd:128, pxTok:56, minTok:64, maxTok:640, defRes:768},
  {id:'internvl3-8b',  name:'InternVL3 8B',        hf:'OpenGVLab/InternVL3-8B',              params:7.62, vparams:0.30, layers:28, heads:28, kv:4,  hd:128, pxTok:28, minTok:256, maxTok:3328, defRes:768},
  {id:'gemma3-12b',    name:'Gemma 3 12B',         hf:'google/gemma-3-12b-it',               params:11.77, vparams:0.42, layers:48, heads:16, kv:8, hd:240, fixedTok:256, vpatches:4096, defRes:896},
  {id:'pixtral-12b',   name:'Pixtral 12B',         hf:'mistralai/Pixtral-12B-2409',          params:12.2, vparams:0.40, layers:40, heads:32, kv:8,  hd:128, pxTok:16, minTok:64, maxTok:4096, defRes:768},
  {id:'gemma3-27b',    name:'Gemma 3 27B',         hf:'google/gemma-3-27b-it',               params:27.0, vparams:0.42, layers:62, heads:32, kv:16, hd:128, fixedTok:256, vpatches:4096, defRes:896},
  {id:'qwen25vl-32b',  name:'Qwen2.5-VL 32B',      hf:'Qwen/Qwen2.5-VL-32B-Instruct',        params:32.8, vparams:0.68, layers:64, heads:40, kv:8,  hd:128, pxTok:28, minTok:64, maxTok:16384, defRes:768},
  {id:'qwen25vl-72b',  name:'Qwen2.5-VL 72B',      hf:'Qwen/Qwen2.5-VL-72B-Instruct',        params:72.7, vparams:0.68, layers:80, heads:64, kv:8,  hd:128, pxTok:28, minTok:64, maxTok:16384, defRes:768},
];
VLMS.forEach(m => { m.active = m.params; });
function vlmTok(m, res){ return m.fixedTok || Math.max(m.minTok||64, Math.min(m.maxTok||16384, Math.round((res/m.pxTok)**2))); }
const VLMMODE = {
  label:'vlm',
  supports: hw => hw.llm && hw.kind!=='npu',
  evaluate(m,w,hw,n){
    if(!hw.llm) return {unsupported:`${hw.name} can't run VLMs.`};
    const tok = vlmTok(m,w.res), prompt = w.imgs*tok + 60, ctx = prompt + w.out + 64;
    const patches = m.vpatches || Math.min(16000, (w.res/14)**2);
    const vis = m.vparams*GB*2;
    const base = {prec:w.prec, kvprec:w.kvprec, ctx, prompt, output:w.out, t_tps:1, t_ttft:1e12, xBytes:vis, xFlops:w.imgs*2*m.vparams*GB*patches};
    const us = hw.kind==='gpu' ? [1,2,4,8,16,32] : [1,2,4];
    let pick=null, best=null;
    for(const u of us){
      const L = LLMMODE.evaluate(m,{...base, users:u},hw,n);
      if(!L.fits){ if(!best) best = {L,u,S:Infinity,cap:0}; break; }
      const S = L.ttft + w.out/L.tps, cap = u*60/S, c = {L,u,S,cap};
      if(!best || cap>best.cap) best = c;
      if(cap>=w.rpm){ pick = c; break; }
    }
    const {L,u,S,cap} = pick||best, fits = L.fits;
    const r = {hw,n,fits,need:L.need,usable:L.usable,wBytes:L.wBytes,kvReserve:L.kvReserve,overhead:L.overhead,vis,S,cap,u,tok,prompt,tps:L.tps,ttft:L.ttft,commLat:L.commLat,notes:[],
      checks:[
        {k:'Time per answer', v: fits?fmtS(S):'—', ratio: fits? S/w.maxS : null, target: fits?`need ≤ ${fmtS(w.maxS)}`:'does not fit', st: fits?status(S<=w.maxS, S>w.maxS*0.85):'na'},
        {k:'Answers per minute', v: fits?fmtN(cap):'—', ratio: fits? w.rpm/cap : null, target:`need ${fmtN(w.rpm)}`, st: fits?status(cap>=w.rpm, cap<w.rpm*1.15):'na'},
        {k:'Memory', v:fmtGB(L.need), ratio:L.need/L.usable, target:`of ${fmtGB(L.usable)} available`, st: status(fits, L.need>L.usable*0.9)},
      ]};
    r.pass = fits && S<=w.maxS && cap>=w.rpm;
    r.headroom = fits ? Math.min(cap/w.rpm, w.maxS/S, L.usable*0.9/L.need) : L.usable*0.9/L.need;
    r.score = r.headroom;
    r.rows = [['Time per answer', fits?fmtS(S):'—'], ['Answers / min', fits?fmtN(cap):'—'], ['Read image', fits?fmtMs(L.ttft*1000):'—'], ['Memory used', Math.round(L.need/L.usable*100)+'%'], ['Headroom', r.headroom.toFixed(1)+'×']];
    return r;
  },
  headline: r => [fmtS(r.S), '/ answer'],
  answer(r,m,w){
    const hw = hwLabel(r.hw,r.n);
    if(r.pass) return `<em>Yes.</em> Your ${esc(hw)} answers a question about ${w.imgs>1?w.imgs+' images':'an image'} with ${esc(m.name)} in about <b>${fmtS(r.S)}</b>, up to ${fmtN(r.cap)} answers per minute.`;
    if(!r.fits) return `<em>No.</em> ${esc(m.name)} needs ${fmtGB(r.need)} but your ${esc(hw)} has ${fmtGB(r.usable)} free.`;
    const bits=[]; if(r.S>w.maxS) bits.push(`each answer takes ${fmtS(r.S)} (limit ${fmtS(w.maxS)})`); if(r.cap<w.rpm) bits.push(`it handles ${fmtN(r.cap)} answers per minute, you need ${fmtN(w.rpm)}`);
    return `<em>No.</em> Your ${esc(hw)} is too slow for ${esc(m.name)}: ${bits.join(', and ')}.`;
  },
  sub:(r,m,w)=>`${esc(m.name)} · ${w.imgs} image${w.imgs>1?'s':''} × ${fmtInt(r.tok)} tokens (${w.res}px) · ${w.out}-token answer · ${PREC[w.prec].label}${r.u>1?` · ${r.u} requests in parallel`:''}`,
  mem: r => ({need:r.need, usable:r.usable, parts:[['Language model',r.wBytes,'#042C53'],['Vision encoder',r.vis,'#378ADD'],['KV cache (image + text tokens)',r.kvReserve,'#0C8567'],['Runtime',r.overhead,'#B9C6D6']]}),
  why(r,m,w){
    if(!r.fits) return ['Memory', r.kvReserve>r.wBytes*0.5 ? 'The image tokens fill the context memory. Lower the image resolution or the number of images per request.' : 'The model weights are bigger than free memory. Use a lower precision or a smaller VLM.'];
    if(r.pass) return ['Nothing — all targets met', `Spare capacity: ${r.headroom.toFixed(1)}×. ${r.ttft > r.S*0.5 ? 'Most of the time goes into reading the image (compute).' : 'Most of the time goes into writing the answer (memory bandwidth).'}`];
    if(r.ttft > r.S*0.5) return ['Compute (reading the image)', `Each image becomes ${fmtInt(r.tok)} tokens the model must process before it can answer — that takes ${fmtMs(r.ttft*1000)}. A lower resolution, fewer images or more GPU compute fixes it.`];
    return ['Memory bandwidth (writing the answer)', `The answer is generated at ${fmtN(r.tps)} tokens/s. Shorter answers, a lower precision or a smaller model fix it.`];
  },
  variants(m,w){
    const order=['fp16','fp8','int8','q6','q5','q4','int4'], out=[];
    if(!m.fixedTok){ const rl = (m.resList||[336,448,768,1024,1536]).filter(x=>x<w.res).reverse(); for(const x of rl.slice(0,2)) out.push([{...w,res:x}, m, `lower the image resolution to <b>${x}px</b> (${fmtInt(vlmTok(m,x))} tokens per image)`, 'small text and fine detail get harder to read']); }
    if(w.out>60) out.push([{...w,out:50}, m, 'ask for a <b>short answer</b> (≈50 tokens, e.g. JSON or one sentence)', 'less explanation in the output']);
    for(let i=order.indexOf(w.prec)+1;i<order.length;i++) out.push([{...w,prec:order[i],kvprec:'fp8'}, m, `use <b>${PREC[order[i]].label}</b> weights`, PREC[order[i]].quality]);
    return out;
  },
  deploy(m,w,r){
    const id = m.hf || 'your-org/your-vlm';
    if(m.id==='florence2-l') return {name:'Transformers', why:'Florence-2 runs through the Transformers library with task prompts such as <OD>, <OCR> or <CAPTION>.', cmd:`from transformers import AutoProcessor, AutoModelForCausalLM\nmodel = AutoModelForCausalLM.from_pretrained("${id}", trust_remote_code=True, torch_dtype="auto").to("cuda")\nprocessor = AutoProcessor.from_pretrained("${id}", trust_remote_code=True)\ninputs = processor(text="<OD>", images=image, return_tensors="pt").to("cuda")\nout = model.generate(**inputs, max_new_tokens=${w.out})`};
    if(r.hw.kind==='gpu' && !r.hw.amd&&!r.hw.intel && (r.u>1 || ['fp16','fp8','int4'].includes(w.prec))) return {name:'vLLM', why:'Batches several image requests together and exposes an OpenAI-compatible API.', cmd:`vllm serve ${id} \\\n  --tensor-parallel-size ${r.n} --max-model-len ${Math.max(4096, Math.ceil((r.prompt+w.out+64)/1024)*1024)} \\\n  --limit-mm-per-prompt image=${w.imgs} --max-num-seqs ${Math.max(r.u,1)} --gpu-memory-utilization 0.92${w.prec==='fp8'?' --quantization fp8':''}`};
    const q = {q4:'Q4_K_M',q5:'Q5_K_M',q6:'Q6_K',int8:'Q8_0',fp16:'F16',fp8:'Q8_0',int4:'Q4_K_M'}[w.prec];
    return {name:'llama.cpp / Ollama', why:'Lightweight runtime for single requests; works on Jetson, Apple and CPU. The vision encoder ships as a separate "mmproj" file.', cmd:`llama-server -m ${id.split('/').pop()}-${q}.gguf \\\n  --mmproj mmproj-${id.split('/').pop()}-f16.gguf \\\n  -c ${Math.max(4096, Math.ceil((r.prompt+w.out+64)/1024)*1024)} -ngl ${r.hw.kind==='cpu'?0:999} -fa on`};
  },
  tips(m,w,r){
    const t=[];
    if(m.note) t.push(['!','Note',m.note]);
    if(m.approx) t.push(['!','Imported model','Vision-encoder size and image-token count are assumed (0.4B, 28 px per token) — treat results as rough.']);
    t.push(['DET','Fixed classes? Use a detector',`For a known list of objects, a YOLO model is typically 50–500× cheaper per frame than a VLM. Keep the VLM for open-ended questions, reading text, or describing scenes.`]);
    t.push(['CAS','Cascade: detector first, VLM on events','Run a small detector on every frame and send only the interesting frames or crops to the VLM. That turns a 30 FPS problem into a few requests per minute.']);
    if(!m.fixedTok && r.tok>1000) t.push(['PX','Image tokens dominate',`At ${w.res}px each image is ${fmtInt(r.tok)} tokens. Cropping to the region of interest or halving the resolution cuts cost about 4×.`]);
    if(w.out>100) t.push(['OUT','Ask for structured, short output','A JSON answer of 30–60 tokens is several times faster than a paragraph.']);
    if(w.imgs>1) t.push(['VID','Video = many images',`${w.imgs} frames per request multiplies the image tokens. Sample fewer frames or use a model with video token compression.`]);
    return t.slice(0,6);
  },
  assumptions:[
    'Each image is converted to tokens (fixed per model, or resolution ÷ pixels-per-token squared). Those tokens are processed like a long prompt before the answer starts.',
    'Time per answer = vision encoder + reading image and text tokens (compute) + answer tokens ÷ generation speed (memory bandwidth).',
    'Answers per minute assumes requests are batched in parallel on GPUs (up to 32) while memory allows; up to 4 on other devices.',
    'Memory = language model + vision encoder (FP16) + KV cache for image and text tokens + runtime.',
    'VLM numbers are less well calibrated than vision and LLM ones: image-token counts vary with aspect ratio and each model\'s tiling scheme.',
  ],
};

/* ============================== VLA MODE ==============================
   Vision-language-action (robot control) models.
   params = language backbone (B), vparams = vision encoders (B), act = action head / expert (B),
   cparams = backbone parameters actually computed (e.g. only the first half of the layers),
   ar = autoregressive action tokens per inference (0 for chunked / flow / diffusion heads),
   steps = flow / diffusion denoising steps, chunk = actions per inference, tokCam = image tokens per camera. */
const VLAS = [
  {id:'openvla',     name:'OpenVLA 7B',              hf:'openvla/openvla-7b', params:6.74, vparams:0.76, act:0, cams:1, tokCam:256, textTok:40, patches:256, ar:7, steps:0, chunk:1, ovh:3, fam:'openvla', note:'Autoregressive: one token per action dimension and one action per inference, so it is memory-bandwidth-bound and slow.'},
  {id:'openvla-oft', name:'OpenVLA-OFT 7B',          link:'https://openvla-oft.github.io', params:6.74, vparams:0.76, act:0, cams:2, tokCam:256, textTok:40, patches:256, ar:0, oft:true, steps:0, chunk:8, ovh:3, fam:'openvla', note:'Same 7B backbone, but predicts a whole chunk in one parallel pass.'},
  {id:'pi0',         name:'π0 (PaliGemma 3B + action expert)', link:'https://github.com/Physical-Intelligence/openpi', params:2.51, vparams:0.40, act:0.30, cams:2, tokCam:256, textTok:48, patches:256, ar:0, steps:10, chunk:50, ovh:0, fam:'openpi'},
  {id:'pi0-fast',    name:'π0-FAST',                 link:'https://github.com/Physical-Intelligence/openpi', params:2.51, vparams:0.40, act:0, cams:2, tokCam:256, textTok:48, patches:256, ar:30, steps:0, chunk:10, ovh:0, fam:'openpi', note:'Compresses a chunk into about 30 discrete tokens, then decodes them one by one.'},
  {id:'pi05',        name:'π0.5',                    link:'https://github.com/Physical-Intelligence/openpi', params:2.51, vparams:0.40, act:0.30, cams:2, tokCam:256, textTok:48, patches:256, ar:0, steps:10, chunk:50, ovh:8, fam:'openpi'},
  {id:'smolvla',     name:'SmolVLA 450M',            hf:'lerobot/smolvla_base', params:0.26, cparams:0.18, vparams:0.09, act:0.10, cams:2, tokCam:64, textTok:24, patches:1024, ar:0, steps:10, chunk:50, ovh:2, fam:'lerobot', note:'Runs only the first half of the language layers; built for consumer GPUs and laptops.'},
  {id:'gr00t-n1',    name:'GR00T N1 2B',             hf:'nvidia/GR00T-N1-2B', params:0.94, vparams:0.40, act:0.86, cams:1, tokCam:64, textTok:24, patches:256, ar:0, steps:4, chunk:16, ovh:30, fam:'gr00t'},
  {id:'gr00t-n15',   name:'GR00T N1.5 3B',          hf:'nvidia/GR00T-N1.5-3B', params:2.0, vparams:0.40, act:0.60, cams:1, tokCam:64, textTok:24, patches:256, ar:0, steps:4, chunk:16, ovh:20, fam:'gr00t'},
  {id:'gr00t-n16',   name:'GR00T N1.6 3B',          hf:'nvidia/GR00T-N1.6-3B', params:1.9, vparams:0.40, act:0.70, cams:1, tokCam:256, textTok:24, patches:256, ar:0, steps:4, chunk:16, ovh:10, fam:'gr00t'},
  {id:'gr00t-n17',   name:'GR00T N1.7 3B',          link:'https://github.com/NVIDIA/Isaac-GR00T', params:1.0, vparams:0.40, act:1.40, cams:1, tokCam:256, textTok:24, patches:256, ar:0, steps:4, chunk:16, ovh:8, fam:'gr00t', note:'Parameter split between backbone and action head is approximate.'},
  {id:'rdt-1b',      name:'RDT-1B',                  hf:'robotics-diffusion-transformer/rdt-1b', params:0, vparams:0.40, act:1.20, cams:3, tokCam:729, textTok:0, patches:729, ar:0, steps:5, chunk:64, ovh:10, fam:'diffusion', note:'The T5 language embedding is computed once per instruction and is not counted.'},
  {id:'octo-base',   name:'Octo-Base 93M',           hf:'rail-berkeley/octo-base-1.5', params:0.09, vparams:0, act:0.003, cams:2, tokCam:256, textTok:16, patches:256, ar:0, steps:20, stepMs:0.5, chunk:4, ovh:3, fam:'octo'},
  {id:'act',         name:'ACT 84M',                 link:'https://tonyzhaozh.github.io/aloha/', params:0, vparams:0.011, act:0.073, cams:4, tokCam:300, textTok:0, patches:1200, ar:0, steps:1, stepMs:1.5, chunk:50, ovh:2, fam:'small', note:'Small policy trained per task; no language input.'},
  {id:'diff-policy', name:'Diffusion Policy 157M',   link:'https://diffusion-policy.cs.columbia.edu', params:0, vparams:0.022, act:0.135, cams:2, tokCam:100, textTok:0, patches:1200, ar:0, steps:10, stepMs:1.0, chunk:16, ovh:2, fam:'small', note:'Small policy trained per task; no language input.'},
];
function vlaDefaults(m){
  const d = {params:0, vparams:0.4, act:0, cams:1, tokCam:256, textTok:32, patches:256, ar:0, steps:0, chunk:10, ovh:0};
  for(const k in d) if(m[k]==null) m[k] = d[k];
}
VLAS.forEach(vlaDefaults);
const VLAMODE = {
  label:'vla',
  supports: hw => hw.llm && hw.kind!=='npu',
  evaluate(m,w,hw,n){
    if(!hw.llm || hw.kind==='npu') return {unsupported:`${hw.name} can't run VLA models.`};
    const bits = PREC[w.prec].bits, opt = w.rt==='opt', kind = hw.kind;
    const lmB = m.params*GB*bits/8, actB = m.act*GB*bits/8, vis = m.vparams*GB*2, wBytes = lmB+actB;
    const usable = hw.mem*GB*usableFrac(hw);
    const overhead = (kind==='gpu'?0.8:0.4)*GB + wBytes*0.05 + w.cams*0.08*GB;
    const need = wBytes+vis+overhead, fits = need<=usable;
    const uF = {gpu:0.35, jetson:0.3, apple:0.3, cpu:0.25}[kind]*(opt?1.3:1);
    const BW = hw.bw*GB*{gpu:0.75, jetson:0.5, apple:0.65, cpu:0.6}[kind];
    const peak = ((w.prec==='fp8' && hw.fp8>0) ? hw.fp8 : hw.fp16)*1e12*uF;
    const peakV = hw.fp16*1e12*uF*0.8;
    const chunk = Math.max(1, w.chunk), steps = m.steps>0 ? Math.max(1, w.steps||m.steps) : 0;
    const tok = w.cams*m.tokCam + m.textTok + (m.oft ? chunk*7 : 0);
    const cp = (m.cparams!=null ? m.cparams : m.params)*GB;
    const tVis = w.cams*2*m.vparams*GB*m.patches/peakV + vis/BW;
    const tPre = cp>0 ? Math.max(2*cp*tok/peak, cp*bits/8/BW) : 0;
    const tTok = m.ar>0 ? m.ar*(lmB/BW + 0.0005) : 0;
    const stepMs = (m.stepMs!=null ? m.stepMs : {gpu:1.8, jetson:3.5, apple:3, cpu:6}[kind])*(opt?0.6:1);
    const tAct = steps>0 ? steps*(Math.max(actB/BW, 2*m.act*GB*chunk/peak) + stepMs/1000) : 0;
    const ovh = ((m.ovh||0)*(opt?0.4:1) + {gpu:0, jetson:4, apple:3, cpu:5}[kind])/1000;
    const L = tVis+tPre+tTok+tAct+ovh, ms = L*1000, aps = chunk/L;
    const parts = [['Reading the cameras', tVis], ['Reading cameras + instruction', tPre], ['Writing action tokens', tTok], [steps>0?`Denoising (${steps} steps)`:'Action head', tAct], ['Software overhead', ovh]];
    const big = parts.reduce((a,b)=>b[1]>a[1]?b:a);
    const r = {hw,n:1,fits,need,usable,wBytes,vis,overhead,L,ms,aps,chunk,steps,tok,tVis,tPre,tTok,tAct,big,notes:[],
      checks:[
        {k:'Reaction time', v: fits?fmtMs(ms):'—', ratio: fits? ms/w.lat : null, target: fits?`need ≤ ${fmtMs(w.lat)}`:'does not fit', st: fits?status(ms<=w.lat, ms>w.lat*0.85):'na'},
        {k:'Actions per second', v: fits?fmtN(aps):'—', ratio: fits? w.hz/aps : null, unit: fits?'Hz':'', target:`need ≥ ${fmtN(w.hz)}`, st: fits?status(aps>=w.hz, aps<w.hz*1.15):'na'},
        {k:'Memory', v:fmtGB(need), ratio:need/usable, target:`of ${fmtGB(usable)} available`, st: status(fits, need>usable*0.9)},
      ]};
    r.pass = fits && ms<=w.lat && aps>=w.hz;
    r.headroom = fits ? Math.min(w.lat/ms, aps/w.hz, usable*0.9/need) : usable*0.9/need;
    r.score = r.headroom;
    r.rows = [['Reaction time', fits?fmtMs(ms):'—'], ['Actions / s', fits?fmtN(aps)+' Hz':'—'], ['Inferences / s', fits?fmtN(1/L):'—'], ['Memory used', Math.round(need/usable*100)+'%'], ['Headroom', r.headroom.toFixed(1)+'×']];
    return r;
  },
  headline: r => [fmtN(r.aps), 'actions/s'],
  answer(r,m,w){
    const hw = hwLabel(r.hw,r.n);
    if(r.pass) return `<em>Yes.</em> Your ${esc(hw)} runs ${esc(m.name)} with a reaction time of about <b>${fmtMs(r.ms)}</b> and up to <b>${fmtN(r.aps)} actions per second</b>.`;
    if(!r.fits) return `<em>No.</em> ${esc(m.name)} needs ${fmtGB(r.need)} but your ${esc(hw)} has ${fmtGB(r.usable)} free.`;
    const bits=[]; if(r.ms>w.lat) bits.push(`the reaction time is ${fmtMs(r.ms)} (limit ${fmtMs(w.lat)})`); if(r.aps<w.hz) bits.push(`it supplies ${fmtN(r.aps)} actions per second, you need ${fmtN(w.hz)}`);
    return `<em>No.</em> Your ${esc(hw)} is too slow for ${esc(m.name)}: ${bits.join(', and ')}.`;
  },
  sub:(r,m,w)=>`${esc(m.name)} · ${w.cams} camera${w.cams>1?'s':''} · ${r.chunk}-action chunk · ${m.ar>0?'autoregressive actions':r.steps>0?r.steps+' denoising steps':'single-pass actions'} · ${PREC[w.prec].label}${w.rt==='opt'?' · optimized runtime':''}`,
  mem: r => ({need:r.need, usable:r.usable, parts:[['Weights',r.wBytes,'#042C53'],['Vision encoders',r.vis,'#378ADD'],['Runtime',r.overhead,'#B9C6D6']]}),
  why(r,m,w){
    if(!r.fits) return ['Memory', r.wBytes+r.vis>r.usable*0.9 ? 'The model weights alone are bigger than free memory. Use a lower precision or a smaller model.' : 'The weights fit, but the runtime buffers do not. Use fewer cameras or a lower precision.'];
    if(r.pass) return ['Nothing — all targets met', `Spare capacity: ${r.headroom.toFixed(1)}×. Largest part of the time: ${r.big[0].toLowerCase()} (${fmtMs(r.big[1]*1000)}).`];
    const k = r.big[0];
    if(k==='Writing action tokens') return ['Memory bandwidth (writing action tokens)', `Every action token reads the whole ${m.params}B backbone from memory — ${fmtMs(r.tTok*1000)} per inference. A chunked model (OpenVLA-OFT, π0, GR00T), FP8/INT4 weights or faster memory fixes it.`];
    if(k.startsWith('Denoising')) return ['Denoising steps', `The action head runs ${r.steps} times, ${fmtMs(r.tAct*1000)} in total. Fewer steps or an optimized runtime (CUDA graphs, TensorRT) cut it.`];
    if(k==='Reading cameras + instruction') return ['Compute (reading cameras and instruction)', `${fmtInt(r.tok)} tokens must be processed every inference — ${fmtMs(r.tPre*1000)}. Fewer cameras, FP8 on newer GPUs or more GPU compute fixes it.`];
    if(k==='Reading the cameras') return ['Compute (vision encoders)', `The image encoders take ${fmtMs(r.tVis*1000)}. Fewer or smaller camera images help.`];
    return ['Software overhead', `A fixed ${fmtMs(r.big[1]*1000)} goes to framework overhead. An optimized runtime removes most of it.`];
  },
  variants(m,w){
    const out=[], st = w.steps || m.steps;
    if(m.steps>0 && st>1){ const s = Math.max(1, Math.round(st/2)); out.push([{...w,steps:s}, m, `cut the denoising steps to <b>${s}</b>`, 'usually small quality change for flow-matching policies']); }
    if(w.cams>1) out.push([{...w,cams:w.cams-1}, m, `use <b>${w.cams-1} camera${w.cams-1>1?'s':''}</b>`, 'less visual information, e.g. drop one side view']);
    if(w.rt!=='opt') out.push([{...w,rt:'opt'}, m, 'use an <b>optimized runtime</b> (TensorRT, torch.compile, CUDA graphs)', 'engineering work, same model quality']);
    for(const p of ['fp8','int8','int4']) if(PREC[p].bits<PREC[w.prec].bits) out.push([{...w,prec:p}, m, `use <b>${PREC[p].label}</b> weights`, PREC[p].quality+' — validate task success']);
    return out;
  },
  deploy(m,w,r){
    const gpu = r.hw.kind==='gpu'||r.hw.kind==='jetson', dev = gpu ? 'cuda' : 'cpu';
    if(m.fam==='openvla') return {name:'Transformers (+ FlashAttention)', why: m.oft ? 'OpenVLA-OFT checkpoints and the parallel-decoding head come from the OFT repository; the backbone loads like OpenVLA.' : 'Official OpenVLA loading code; one action per call.', cmd: m.oft ? `git clone https://github.com/moojink/openvla-oft\ncd openvla-oft && pip install -e .\n# follow the repo's LIBERO / ALOHA instructions to load a finetuned checkpoint` :
`import torch\nfrom transformers import AutoModelForVision2Seq, AutoProcessor\nprocessor = AutoProcessor.from_pretrained("${m.hf}", trust_remote_code=True)\nvla = AutoModelForVision2Seq.from_pretrained("${m.hf}", torch_dtype=torch.bfloat16, trust_remote_code=True).to("${dev}:0" if "${dev}"=="cuda" else "cpu")\naction = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)`};
    if(m.fam==='openpi') return {name:'openpi policy server', why:'Physical Intelligence\'s reference server: the policy runs on the GPU box and the robot client streams observations to it.', cmd:`git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi\ncd openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync\nuv run scripts/serve_policy.py policy:checkpoint \\\n  --policy.config=${m.id==='pi0-fast'?'pi0_fast_droid':m.id==='pi05'?'pi05_droid':'pi0_fast_droid'} \\\n  --policy.dir=<checkpoint dir>   # see the README for the checkpoint list`};
    if(m.fam==='lerobot') return {name:'LeRobot', why:'Hugging Face LeRobot loads SmolVLA and supports asynchronous inference through a policy server.', cmd:`pip install "lerobot[smolvla]"\npython -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; p = SmolVLAPolicy.from_pretrained('${m.hf}')"\n# import path can differ between LeRobot versions; see the SmolVLA model card`};
    if(m.fam==='gr00t') return {name:'Isaac GR00T', why:'NVIDIA\'s repo includes a policy server and TensorRT export scripts; Jetson Thor, DGX Spark and RTX PRO systems are supported.', cmd:`git clone https://github.com/NVIDIA/Isaac-GR00T\ncd Isaac-GR00T && uv sync && uv pip install -e .\n# checkpoint: ${m.hf||'see the repo README'}\n# then follow the README to start the policy server and (optionally) export the action head to TensorRT`};
    return {name:'Reference repository', why:'Small policies run from their own research repositories (PyTorch or JAX).', cmd:`# ${m.name}\n# ${m.hf?'weights: https://huggingface.co/'+m.hf:'see the project page for code and weights'}\n# export to ONNX / TensorRT if you need lower latency than eager PyTorch`};
  },
  tips(m,w,r){
    const t=[];
    if(m.note) t.push(['!','Note',m.note]);
    t.push(['ASY','Run inference asynchronously','Compute the next chunk while the robot executes the current one (real-time chunking, SmolVLA async inference). The robot then never waits, and a delay up to about half a chunk is hidden.']);
    t.push(['NET','Keep the GPU close to the robot','A GPU box on the same wired network adds only 1–10 ms; Wi-Fi or the cloud adds 10–100+ ms to the reaction time and is not included here.']);
    if(m.ar>0) t.push(['PAR','Switch to a chunked action head','One-token-at-a-time action decoding is the bottleneck. Parallel decoding (OpenVLA-OFT) or flow heads (π0, GR00T) are 5–20× faster per action.']);
    if(m.steps>4) t.push(['STP','Fewer denoising steps',`Going from ${w.steps||m.steps} to 4–5 steps is often enough for smooth motion and saves ${fmtMs(r.tAct*1000*0.5)} or more.`]);
    t.push(['LOW','Add a fast low-level controller','Let the VLA output targets at 5–30 Hz and run a PD / impedance controller at 200+ Hz underneath. This is how humanoid and dexterous systems reach high control rates with 100 ms policies.']);
    return t.slice(0,6);
  },
  assumptions:[
    'Reaction time = vision encoders + reading camera and instruction tokens (compute-bound) + action generation + a fixed software overhead. Network and robot-controller delay are not included.',
    'Autoregressive models (OpenVLA, π0-FAST) read the whole backbone once per action token (memory-bandwidth-bound). Flow / diffusion heads repeat a small network once per denoising step with a fixed per-step cost.',
    'Actions per second = actions per chunk ÷ time per inference, an open-loop upper bound; a robot that must react to every new frame is limited by the reaction time instead.',
    'Memory = backbone + action head (at the chosen precision) + vision encoders (FP16) + runtime buffers. One device only.',
    'Calibrated against published figures: π0 ≈ 73 ms (RTX 4090), OpenVLA ≈ 6 Hz (RTX 4090), OpenVLA-OFT ≈ 73 ms for an 8-action chunk (A100), GR00T N1 ≈ 64 ms (L40). Parameter splits and image-token counts for newer models are approximate.',
  ],
};
VISION.headline = r => [fmtN(r.capacity), 'FPS'];
GENMODE.headline = r => [fmtS(r.reqS), '/ image'];
LLMMODE.headline = r => [fmtN(r.tps), 'tokens/s'];

/* ============================== OPTIONS ============================== */
function costOf(hw,n,c){ return c.pay==='cloud' ? (hw.cloud==null?null:hw.cloud*n) : hw.price*n; }
function pickOptions(mode,m,w,cur,c){
  const all=[];
  for(const hw of HW){
    if(!mode.supports(hw,m)) continue;
    if(c.edge && hw.g!=='Edge devices') continue;
    if((hw.lap||hw.legacy||hw.phone) && !c.legacy && !hw.mine) continue;
    if(hw.inst && c.pay!=='cloud') continue;
    for(const n of [1,2,4,8]){
      if(n>hw.max) continue;
      if(hw.fixedN && n!==hw.fixedN) continue;
      if(mode===VLAMODE && (n>1 || hw.amd || hw.intel)) continue;   // robot stacks (CUDA / Jetson) are NVIDIA-first
      if((mode===LLMMODE || mode===VLMMODE) && n>1 && hw.kind!=='gpu') continue;
      const cost = costOf(hw,n,c); if(cost==null) continue;
      if(c.maxW && hw.tdp*n > c.maxW) continue;
      const r = mode.evaluate(m,w,hw,n); if(r.unsupported) continue;
      r.cost = cost; all.push(r);
    }
  }
  return tierPicks(preferInst(all,c),cur,c);
}
function preferInst(all,c){ const ins = all.filter(r=>r.hw.inst); return c.pay==='cloud' && ins.some(r=>r.pass) ? ins : all; }
function tierPicks(all,cur,c){
  let pool = all.filter(r=>r.pass && !(cur && r.hw.id===cur.hw.id && r.n===cur.n));
  if(cur && cur.pass) pool = pool.filter(r=>r.score > cur.score*1.25);
  // someone already on an edge device usually needs to stay on the edge
  if(cur && cur.hw.g==='Edge devices' && pool.some(r=>r.hw.g==='Edge devices')) pool = pool.filter(r=>r.hw.g==='Edge devices');
  pool.sort((a,b)=>a.cost-b.cost || b.score-a.score);
  if(!pool.length) return {picks:[]};
  const inBudget = pool.filter(r=>!c.budget || c.pay==='cloud' || r.cost<=c.budget);
  const base = inBudget.length ? inBudget : pool;
  const cheapest = base[0], picks=[];
  let rec=null;
  if(cheapest.headroom>=1.5) picks.push({tier:'Recommended · best value', r:cheapest, rec:true, why:'Cheapest option that meets every target with ≥1.5× spare capacity.'});
  else {
    picks.push({tier:'Budget', r:cheapest, why:'Cheapest option that meets every target — little room to grow.'});
    rec = base.find(r=>r.headroom>=1.5 && r.score>=cheapest.score) || base.find(r=>r!==cheapest && r.score>cheapest.score);
    if(rec) picks.push({tier:'Recommended', r:rec, rec:true, why:'Meets targets with ≥1.5× spare capacity for growth and peaks.'});
  }
  const top = rec || cheapest, cap = Math.max(cheapest.cost*4, top.cost*2);
  const perf = pool.filter(r=>r!==cheapest && r!==rec && r.cost<=cap && r.score>top.score*1.2 && (!c.budget || c.pay==='cloud' || r.cost<=c.budget*1.5)).sort((a,b)=>b.score-a.score)[0];
  if(perf) picks.push({tier:'Performance', r:perf, why:'Most speed within a sensible price step up.'});
  return {picks, overBudget: c.budget && c.pay==='buy' && !inBudget.length};
}

