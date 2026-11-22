/* ==========================================================================
   DATA — edit these tables to update specs and prices.
   mem GB · bw GB/s · fp16/fp8 = dense tensor TFLOPS (LLM, VLM & image gen)
   cv = effective TFLOPS for CNN inference at FP16, ck = fixed per-inference work,
   ovh = runtime overhead ms, pp = pre/post-processing ms, int8x = INT8 speed-up.
   price ≈ USD street price · cloud ≈ USD per hour · tdp = watts
   ========================================================================== */
const HW = [
  // Edge
  {id:'pi5',     g:'Edge devices', name:'Raspberry Pi 5 8GB (CPU)', kind:'cpu', mem:8, bw:17, fp16:0.1, fp8:0, cv:0.118, ck:4.6, int8x:1.3, ovh:0, pp:10, price:90, cloud:null, tdp:12, max:1, llm:true, gen:false, rt:'NCNN'},
  {id:'pi5h8l',  g:'Edge devices', name:'Raspberry Pi 5 + Hailo-8L (13 TOPS)', kind:'npu', mem:8, bw:17, fp16:0, fp8:0, cv:3, ck:3, int8x:1, int8only:true, ovh:1.5, pp:8, price:160, cloud:null, tdp:15, max:1, llm:false, gen:false, rt:'HailoRT'},
  {id:'pi5h8',   g:'Edge devices', name:'Raspberry Pi 5 + Hailo-8 (26 TOPS)', kind:'npu', mem:8, bw:17, fp16:0, fp8:0, cv:6, ck:4, int8x:1, int8only:true, ovh:1.5, pp:8, price:290, cloud:null, tdp:18, max:1, llm:false, gen:false, rt:'HailoRT'},
  {id:'jnano',   g:'Edge devices', name:'Jetson Nano 4GB (legacy)', kind:'jetson', mem:4, bw:25.6, fp16:0.47, fp8:0, cv:0.3, ck:1, int8x:1, ovh:2, pp:8, price:150, cloud:null, tdp:10, max:1, llm:true, gen:false, rt:'TensorRT'},
  {id:'jorinnano',g:'Edge devices',name:'Jetson Orin Nano Super 8GB', kind:'jetson', mem:8, bw:102, fp16:17, fp8:0, cv:6, ck:6, int8x:1.7, ovh:1, pp:5, price:249, cloud:null, tdp:25, max:1, llm:true, gen:true, rt:'TensorRT'},
  {id:'jorinnx', g:'Edge devices', name:'Jetson Orin NX 16GB', kind:'jetson', mem:16, bw:102, fp16:25, fp8:0, cv:8, ck:7, int8x:1.7, ovh:1, pp:4, price:900, cloud:null, tdp:25, max:1, llm:true, gen:true, rt:'TensorRT'},
  {id:'jagx',    g:'Edge devices', name:'Jetson AGX Orin 64GB', kind:'jetson', mem:64, bw:205, fp16:85, fp8:0, cv:18, ck:10, int8x:1.7, ovh:0.8, pp:3, price:2000, cloud:null, tdp:60, max:1, llm:true, gen:true, rt:'TensorRT'},
  {id:'intel',   g:'Edge devices', name:'Mini PC · Intel Core Ultra 7 (CPU only)', kind:'cpu', mem:32, bw:90, fp16:1.5, fp8:0, cv:1.2, ck:10, int8x:1.8, ovh:0, pp:3, price:700, cloud:null, tdp:65, max:1, llm:true, gen:false, rt:'OpenVINO'},
  // Desktop
  {id:'3060',    g:'Desktop GPUs', name:'RTX 3060 12GB', kind:'gpu', mem:12, bw:360, fp16:51, fp8:0, price:280, cloud:null, tdp:170, max:4, link:'pcie'},
  {id:'4060ti',  g:'Desktop GPUs', name:'RTX 4060 Ti 16GB', kind:'gpu', mem:16, bw:288, fp16:88, fp8:176, price:450, cloud:null, tdp:165, max:4, link:'pcie'},
  {id:'3090',    g:'Desktop GPUs', name:'RTX 3090 24GB (used)', kind:'gpu', mem:24, bw:936, fp16:142, fp8:0, price:800, cloud:0.22, tdp:350, max:4, link:'pcie'},
  {id:'7900xtx', g:'Desktop GPUs', name:'Radeon RX 7900 XTX 24GB', kind:'gpu', mem:24, bw:960, fp16:123, fp8:0, price:900, cloud:null, tdp:355, max:4, link:'pcie', amd:true, rt:'ONNX Runtime (MIGraphX)'},
  {id:'4090',    g:'Desktop GPUs', name:'RTX 4090 24GB', kind:'gpu', mem:24, bw:1008, fp16:330, fp8:660, price:2000, cloud:0.34, tdp:450, max:4, link:'pcie'},
  {id:'5090',    g:'Desktop GPUs', name:'RTX 5090 32GB', kind:'gpu', mem:32, bw:1792, fp16:419, fp8:838, price:2600, cloud:0.69, tdp:575, max:4, link:'pcie'},
  {id:'m4max',   g:'Desktop GPUs', name:'Apple M4 Max 128GB', kind:'apple', mem:128, bw:546, fp16:30, fp8:0, cv:6, ck:8, int8x:1.2, ovh:1, pp:2, price:4700, cloud:null, tdp:120, max:1, rt:'Core ML'},
  // Workstation
  {id:'a6000',   g:'Workstation GPUs', name:'RTX A6000 48GB', kind:'gpu', mem:48, bw:768, fp16:155, fp8:0, price:4200, cloud:0.49, tdp:300, max:4, link:'pcie'},
  {id:'6000ada', g:'Workstation GPUs', name:'RTX 6000 Ada 48GB', kind:'gpu', mem:48, bw:960, fp16:364, fp8:728, price:6800, cloud:0.77, tdp:300, max:4, link:'pcie'},
  {id:'pro6000', g:'Workstation GPUs', name:'RTX PRO 6000 Blackwell 96GB', kind:'gpu', mem:96, bw:1792, fp16:500, fp8:1000, price:8500, cloud:1.79, tdp:600, max:4, link:'pcie'},
  {id:'m3ultra', g:'Workstation GPUs', name:'Apple M3 Ultra 512GB', kind:'apple', mem:512, bw:819, fp16:45, fp8:0, cv:9, ck:10, int8x:1.2, ovh:1, pp:2, price:9500, cloud:null, tdp:200, max:1, rt:'Core ML'},
  // Data center
  {id:'t4',      g:'Data center GPUs', name:'NVIDIA T4 16GB', kind:'gpu', mem:16, bw:320, fp16:65, fp8:0, price:900, cloud:0.35, tdp:70, max:8, link:'pcie'},
  {id:'l4',      g:'Data center GPUs', name:'NVIDIA L4 24GB', kind:'gpu', mem:24, bw:300, fp16:121, fp8:242, price:2500, cloud:0.44, tdp:72, max:8, link:'pcie'},
  {id:'l40s',    g:'Data center GPUs', name:'NVIDIA L40S 48GB', kind:'gpu', mem:48, bw:864, fp16:362, fp8:733, price:8000, cloud:0.86, tdp:350, max:8, link:'pcie'},
  {id:'a100',    g:'Data center GPUs', name:'A100 80GB SXM', kind:'gpu', mem:80, bw:2039, fp16:312, fp8:0, price:15000, cloud:1.19, tdp:400, max:8, link:'nvlink'},
  {id:'h100',    g:'Data center GPUs', name:'H100 80GB SXM', kind:'gpu', mem:80, bw:3350, fp16:989, fp8:1979, price:27000, cloud:1.99, tdp:700, max:8, link:'nvlink'},
  {id:'h200',    g:'Data center GPUs', name:'H200 141GB SXM', kind:'gpu', mem:141, bw:4800, fp16:989, fp8:1979, price:33000, cloud:3.00, tdp:700, max:8, link:'nvlink'},
  {id:'b200',    g:'Data center GPUs', name:'B200 180GB', kind:'gpu', mem:180, bw:8000, fp16:2250, fp8:4500, price:45000, cloud:5.00, tdp:1000, max:8, link:'nvlink'},
  {id:'mi300x',  g:'Data center GPUs', name:'AMD MI300X 192GB', kind:'gpu', mem:192, bw:5300, fp16:1307, fp8:2615, price:20000, cloud:2.00, tdp:750, max:8, link:'nvlink', amd:true, rt:'ONNX Runtime (MIGraphX)'},
];

/* ---- more GPUs. Spec: [id, name, arch, memGB, GB/s, FP32 TFLOPS, watts, price].
   FP16 tensor (dense) is derived from FP32 by architecture; laptop parts use sustained clocks. ---- */
const ARCH = {
  blackwell:{l:'NVIDIA RTX 50 (Blackwell)', m:4, f8:2},
  ada:      {l:'NVIDIA RTX 40 / Ada',        m:4, f8:2},
  ampere:   {l:'NVIDIA RTX 30 / Ampere',     m:4, f8:0},
  turing:   {l:'NVIDIA RTX 20 (Turing)',     m:8, f8:0, legacy:1},
  gtx16:    {l:'NVIDIA GTX 16 (no tensor cores)', m:2, f8:0, legacy:1},
  pascal:   {l:'NVIDIA GTX 10 (Pascal)',     m:1, f8:0, legacy:1},
  rdna4:    {l:'AMD RX 9000 (RDNA 4)',       m:4, f8:2, amd:1},
  rdna3:    {l:'AMD RX 7000 (RDNA 3)',       m:2, f8:0, amd:1},
  rdna2:    {l:'AMD RX 6000 (RDNA 2)',       m:2, f8:0, amd:1, legacy:1},
  arc2:     {l:'Intel Arc B-series (Battlemage)', m:8, f8:0, intel:1},
  arc1:     {l:'Intel Arc A-series (Alchemist)',  m:7, f8:0, intel:1, legacy:1},
  manual:   {l:"Other — I'll enter FP16 TFLOPS", m:0, f8:0},
};
function mkDev(g, id, name, arch, mem, bw, fp32, tdp, price, x){
  const a = ARCH[arch], fp16 = Math.round(fp32*a.m*10)/10;
  return Object.assign({id, g, name, kind:'gpu', mem, bw, fp16, fp8: a.f8 ? Math.round(fp16*a.f8) : 0, price, cloud:null, tdp, max:2, link:'pcie',
    amd:!!a.amd, intel:!!a.intel, legacy:!!a.legacy, rt: a.amd ? 'ONNX Runtime (MIGraphX)' : a.intel ? 'OpenVINO' : 'TensorRT'}, x||{});
}
[
  ['1060','GTX 1060 6GB','pascal',6,192,4.4,120,120], ['1070','GTX 1070 8GB','pascal',8,256,6.5,150,130], ['1080ti','GTX 1080 Ti 11GB','pascal',11,484,11.3,250,200],
  ['1650','GTX 1650 4GB','gtx16',4,128,3.0,75,130], ['1660s','GTX 1660 Super 6GB','gtx16',6,336,5.0,125,150],
  ['2060','RTX 2060 6GB','turing',6,336,6.5,160,140], ['2070s','RTX 2070 Super 8GB','turing',8,448,9.1,215,200], ['2080','RTX 2080 8GB','turing',8,448,10.1,215,220], ['2080ti','RTX 2080 Ti 11GB','turing',11,616,13.4,250,300],
  ['3050','RTX 3050 8GB','ampere',8,224,9.1,130,210], ['3060ti','RTX 3060 Ti 8GB','ampere',8,448,16.2,200,300], ['3070','RTX 3070 8GB','ampere',8,448,20.3,220,330], ['3080','RTX 3080 10GB','ampere',10,760,29.8,320,450], ['3080ti','RTX 3080 Ti 12GB','ampere',12,912,34.1,350,550],
  ['4060','RTX 4060 8GB','ada',8,272,15.1,115,300], ['4070','RTX 4070 12GB','ada',12,504,29.1,200,550], ['4070s','RTX 4070 Super 12GB','ada',12,504,35.5,220,600], ['4070tis','RTX 4070 Ti Super 16GB','ada',16,672,44.1,285,800], ['4080s','RTX 4080 Super 16GB','ada',16,736,52.2,320,1000],
  ['5060ti','RTX 5060 Ti 16GB','blackwell',16,448,23.7,180,430], ['5070','RTX 5070 12GB','blackwell',12,672,30.9,250,550], ['5070ti','RTX 5070 Ti 16GB','blackwell',16,896,43.9,300,750], ['5080','RTX 5080 16GB','blackwell',16,960,56.3,360,1000],
  ['rx6600xt','Radeon RX 6600 XT 8GB','rdna2',8,256,10.6,160,200], ['rx6800xt','Radeon RX 6800 XT 16GB','rdna2',16,512,20.7,300,350],
  ['rx7600','Radeon RX 7600 8GB','rdna3',8,288,21.5,165,260], ['rx7700xt','Radeon RX 7700 XT 12GB','rdna3',12,432,35.2,245,380], ['rx7800xt','Radeon RX 7800 XT 16GB','rdna3',16,624,37.3,263,450], ['rx7900xt','Radeon RX 7900 XT 20GB','rdna3',20,800,51.5,315,700],
  ['rx9060xt','Radeon RX 9060 XT 16GB','rdna4',16,322,25.6,160,350], ['rx9070xt','Radeon RX 9070 XT 16GB','rdna4',16,640,48.7,304,600],
  ['a380','Intel Arc A380 6GB','arc1',6,186,4.2,75,120], ['a750','Intel Arc A750 8GB','arc1',8,512,17.2,225,200], ['a770','Intel Arc A770 16GB','arc1',16,560,19.7,225,300],
  ['b580','Intel Arc B580 12GB','arc2',12,456,14.6,190,250], ['arcb60','Intel Arc Pro B60 24GB','arc2',24,456,12.0,200,700],
].forEach(d => HW.push(mkDev('Desktop GPUs', ...d)));
/* laptop GPUs — price = typical laptop price with this GPU; sustained (not boost) clocks */
[
  ['l1650','GTX 1650 Laptop 4GB','gtx16',4,128,2.6,50,900], ['l1660ti','GTX 1660 Ti Laptop 6GB','gtx16',6,288,4.9,80,1000],
  ['l2060','RTX 2060 Laptop 6GB','turing',6,336,5.5,80,1100], ['l2070','RTX 2070 Laptop 8GB','turing',8,448,7.0,115,1300], ['l2080','RTX 2080 Laptop (Super / Max-Q) 8GB','turing',8,448,8.5,150,1800],
  ['l3050','RTX 3050 Laptop 4GB','ampere',4,192,5.3,60,800], ['l3060','RTX 3060 Laptop 6GB','ampere',6,336,12.0,115,1100], ['l3070','RTX 3070 Laptop 8GB','ampere',8,448,15.5,125,1400], ['l3080','RTX 3080 Laptop 16GB','ampere',16,448,18.0,150,1900],
  ['l4050','RTX 4050 Laptop 6GB','ada',6,192,10.0,115,900], ['l4060','RTX 4060 Laptop 8GB','ada',8,256,12.4,115,1100], ['l4070','RTX 4070 Laptop 8GB','ada',8,256,17.0,115,1500], ['l4080','RTX 4080 Laptop 12GB','ada',12,432,28.7,150,2200], ['l4090','RTX 4090 Laptop 16GB','ada',16,576,36.5,150,3000],
  ['l5050','RTX 5050 Laptop 8GB','blackwell',8,384,11.7,100,1000], ['l5060','RTX 5060 Laptop 8GB','blackwell',8,384,15.2,100,1300], ['l5070','RTX 5070 Laptop 8GB','blackwell',8,384,21.2,100,1600], ['l5070ti','RTX 5070 Ti Laptop 12GB','blackwell',12,672,26.4,115,2000], ['l5080','RTX 5080 Laptop 16GB','blackwell',16,896,35.5,150,2600], ['l5090','RTX 5090 Laptop 24GB','blackwell',24,896,48.5,150,3400],
].forEach(d => HW.push(mkDev('Laptop GPUs', ...d, {lap:true, max:1})));
/* workstation / data-centre additions */
[
  ['rtxa4000','RTX A4000 16GB','ampere',16,448,19.2,140,700], ['rtxa5000','RTX A5000 24GB','ampere',24,768,27.8,230,1500], ['rtx4000ada','RTX 4000 Ada 20GB','ada',20,360,26.7,130,1250], ['rtx5000ada','RTX 5000 Ada 32GB','ada',32,576,65.3,250,4000],
  ['pro4000','RTX PRO 4000 Blackwell 24GB','blackwell',24,672,36.8,140,1600], ['pro5000','RTX PRO 5000 Blackwell 48GB','blackwell',48,1344,64.5,300,4600], ['w7900','Radeon PRO W7900 48GB','rdna3',48,864,61.3,295,3500],
].forEach(d => HW.push(mkDev('Workstation GPUs', ...d, {max:4, legacy:false})));
HW.push(
  {id:'v100',g:'Data center GPUs',name:'NVIDIA V100 32GB (legacy)',kind:'gpu',mem:32,bw:900,fp16:125,fp8:0,price:1500,cloud:0.45,tdp:300,max:8,link:'nvlink',legacy:true},
  {id:'a10g',g:'Data center GPUs',name:'NVIDIA A10G 24GB',kind:'gpu',mem:24,bw:600,fp16:125,fp8:0,price:3000,cloud:0.55,tdp:150,max:8,link:'pcie'},
  {id:'a40',g:'Data center GPUs',name:'NVIDIA A40 48GB',kind:'gpu',mem:48,bw:696,fp16:150,fp8:0,price:4500,cloud:0.40,tdp:300,max:8,link:'pcie'},
  {id:'l40',g:'Data center GPUs',name:'NVIDIA L40 48GB',kind:'gpu',mem:48,bw:864,fp16:181,fp8:362,price:7000,cloud:0.69,tdp:300,max:8,link:'pcie'},
  {id:'a10040',g:'Data center GPUs',name:'A100 40GB SXM',kind:'gpu',mem:40,bw:1555,fp16:312,fp8:0,price:9000,cloud:0.80,tdp:400,max:8,link:'nvlink'},
  {id:'h100pcie',g:'Data center GPUs',name:'H100 80GB PCIe',kind:'gpu',mem:80,bw:2000,fp16:756,fp8:1513,price:25000,cloud:1.60,tdp:350,max:8,link:'pcie'},
  {id:'b300',g:'Data center GPUs',name:'B300 288GB',kind:'gpu',mem:288,bw:8000,fp16:2250,fp8:4500,price:55000,cloud:6.00,tdp:1400,max:8,link:'nvlink'},
  {id:'mi325x',g:'Data center GPUs',name:'AMD MI325X 256GB',kind:'gpu',mem:256,bw:6000,fp16:1307,fp8:2615,price:25000,cloud:2.40,tdp:1000,max:8,link:'nvlink',amd:true,rt:'ONNX Runtime (MIGraphX)'},
  {id:'mi355x',g:'Data center GPUs',name:'AMD MI355X 288GB',kind:'gpu',mem:288,bw:8000,fp16:2500,fp8:5000,price:35000,cloud:3.50,tdp:1400,max:8,link:'nvlink',amd:true,rt:'ONNX Runtime (MIGraphX)'},
  {id:'gaudi3',g:'Data center GPUs',name:'Intel Gaudi 3 128GB',kind:'gpu',mem:128,bw:3700,fp16:1835,fp8:1835,price:15000,cloud:null,tdp:900,max:8,link:'pcie',intel:true,legacy:true,rt:'OpenVINO'},
  // mini PCs, dev kits and unified-memory machines
  {id:'dgxspark',g:'Mini PCs & dev kits',name:'NVIDIA DGX Spark (GB10) 128GB',kind:'gpu',mem:128,bw:273,fp16:125,fp8:250,price:4000,cloud:null,tdp:240,max:1,usable:0.85},
  {id:'strixhalo',g:'Mini PCs & dev kits',name:'AMD Ryzen AI Max+ 395 · 128GB mini PC',kind:'gpu',mem:96,bw:256,fp16:59,fp8:0,price:2000,cloud:null,tdp:120,max:1,usable:0.8,amd:true,rt:'ONNX Runtime (MIGraphX)'},
  {id:'jthor',g:'Edge devices',name:'Jetson AGX Thor 128GB',kind:'jetson',mem:128,bw:273,fp16:260,fp8:520,cv:50,ck:14,int8x:1.7,ovh:0.8,pp:3,price:3500,cloud:null,tdp:130,max:1,llm:true,gen:true,rt:'TensorRT'},
  {id:'iph14p',g:'Phones & tablets',name:'iPhone 14 Pro · A16 · 6GB',kind:'apple',mem:6,bw:51.2,fp16:3.4,fp8:0,cv:0.7,ck:2.7,int8x:1.2,ovh:1.5,pp:4,price:600,cloud:null,tdp:8,max:1,usable:0.5,phone:true,rt:'Core ML'},
  {id:'iph15p',g:'Phones & tablets',name:'iPhone 15 Pro · A17 Pro · 8GB',kind:'apple',mem:8,bw:51.2,fp16:4.3,fp8:0,cv:1.5,ck:3.9,int8x:1.2,ovh:1.5,pp:4,price:700,cloud:null,tdp:8,max:1,usable:0.5,phone:true,rt:'Core ML'},
  {id:'iph16p',g:'Phones & tablets',name:'iPhone 16 Pro · A18 Pro · 8GB',kind:'apple',mem:8,bw:60,fp16:5,fp8:0,cv:1.7,ck:4.2,int8x:1.2,ovh:1.5,pp:4,price:800,cloud:null,tdp:8,max:1,usable:0.5,phone:true,rt:'Core ML'},
  {id:'iph17p',g:'Phones & tablets',name:'iPhone 17 Pro · A19 Pro · 12GB',kind:'apple',mem:12,bw:76.8,fp16:20,fp8:0,cv:2.6,ck:5.2,int8x:1.2,ovh:1.5,pp:4,price:1100,cloud:null,tdp:8,max:1,usable:0.5,phone:true,rt:'Core ML'},
  {id:'iph18p',g:'Phones & tablets',name:'iPhone 18 Pro · A20 Pro · 12GB (estimated specs)',kind:'apple',mem:12,bw:96,fp16:22,fp8:0,cv:3.0,ck:5.5,int8x:1.2,ovh:1.5,pp:4,price:1100,cloud:null,tdp:8,max:1,usable:0.5,phone:true,rt:'Core ML'},
  {id:'m4',g:'Desktop GPUs',name:'Apple M4 16GB (Mac mini / MacBook Air)',kind:'apple',mem:16,bw:120,fp16:8,fp8:0,cv:1.6,ck:4,int8x:1.2,ovh:1,pp:3,price:600,cloud:null,tdp:30,max:1,rt:'Core ML'},
  {id:'m4pro',g:'Desktop GPUs',name:'Apple M4 Pro 48GB',kind:'apple',mem:48,bw:273,fp16:18,fp8:0,cv:3.4,ck:6,int8x:1.2,ovh:1,pp:2.5,price:2400,cloud:null,tdp:70,max:1,rt:'Core ML'}
);


/* ---- cloud instances: [id, provider, instance, GPU in this table, GPUs, vCPU, RAM GB, USD/hour on-demand Linux, US region]
   Prices checked Nov 2026; they change often — edit here or add "instances" rows in data.json. ---- */
const CLOUD = [
  ['aws-g4dn-xl','AWS','g4dn.xlarge','t4',1,4,16,0.526], ['aws-g5-xl','AWS','g5.xlarge','a10g',1,4,16,1.006], ['aws-g5-12xl','AWS','g5.12xlarge','a10g',4,48,192,5.672], ['aws-g5-48xl','AWS','g5.48xlarge','a10g',8,192,768,16.288],
  ['aws-g6-xl','AWS','g6.xlarge','l4',1,4,16,0.805], ['aws-g6e-xl','AWS','g6e.xlarge','l40s',1,4,32,1.861], ['aws-g6e-12xl','AWS','g6e.12xlarge','l40s',4,48,384,10.493],
  ['aws-p4d','AWS','p4d.24xlarge','a10040',8,96,1152,21.96], ['aws-p5','AWS','p5.48xlarge','h100',8,192,2048,55.04],
  ['gcp-g2-4','Google Cloud','g2-standard-4','l4',1,4,16,0.707], ['gcp-g2-48','Google Cloud','g2-standard-48','l4',4,48,192,4.002],
  ['gcp-a2-1g','Google Cloud','a2-highgpu-1g','a10040',1,12,85,3.673], ['gcp-a2-4g','Google Cloud','a2-highgpu-4g','a10040',4,48,340,14.694], ['gcp-a3-8g','Google Cloud','a3-highgpu-8g','h100',8,208,1872,88.49],
  ['az-nc24-a100','Azure','NC24ads A100 v4','a100',1,24,220,3.67], ['az-nc96-a100','Azure','NC96ads A100 v4','a100',4,96,880,14.69], ['az-nd96-h100','Azure','ND96isr H100 v5','h100',8,96,1900,98.32],
];
function mkInst([id,prov,inst,base,n,vcpu,ram,usd]){
  const b = HW.find(h=>h.id===base && !h.inst); if(!b) return null;
  return {...b, id, g:'Cloud instances', name:`${prov} ${inst} · ${n>1?n+'× ':''}${b.name}`, price:0, cloud:+(usd/n).toFixed(4), max:n, fixedN:n, legacy:false, lap:false,
    inst:{prov, name:inst, n, vcpu, ram, usd}};
}
CLOUD.forEach(row => { const h = mkInst(row); if(h) HW.push(h); });

// derive vision constants for discrete GPUs
function deriveHW(h){
  if(h.kind==='gpu' && h.cv==null){ const nn = h.amd||h.intel; h.cv = h.fp16*0.3*(nn?0.7:1); h.ck = 20*Math.sqrt(h.cv/20); h.int8x = nn?1.2:1.7; h.ovh = h.g==='Data center GPUs'?0.3:0.4; h.pp = 2; }
  if(h.llm==null) h.llm = true; if(h.gen==null) h.gen = true; if(!h.rt) h.rt='TensorRT'; if(!h.link) h.link='none';
  if(!h.g) h.g = 'Other devices'; if(!h.max) h.max = 1; if(h.cloud===undefined) h.cloud = null; if(!h.price) h.price = 0; if(!h.tdp) h.tdp = 0;
}
HW.forEach(deriveHW);
const GROUP_ORDER = ['My devices','Edge devices','Phones & tablets','Mini PCs & dev kits','Laptop GPUs','Desktop GPUs','Workstation GPUs','Data center GPUs','Cloud instances'];
const gIdx = g => { const i = GROUP_ORDER.indexOf(g); return i<0 ? 99 : i; };
function sortHW(){ HW.sort((a,b) => gIdx(a.g)-gIdx(b.g) || (a.price||0)-(b.price||0) || (a.cloud||0)-(b.cloud||0)); }
sortHW();
/* ---- devices the user adds in the page (kept in this browser) ---- */
let DEVS = [];
function loadDevs(){ try{ DEVS = JSON.parse(localStorage.getItem('computefit.devices')||'[]') || []; }catch(e){ DEVS = []; } }
function saveDevs(){ try{ localStorage.setItem('computefit.devices', JSON.stringify(DEVS)); }catch(e){} }
function applyDevs(){
  for(let i=HW.length-1;i>=0;i--) if(HW[i].mine) HW.splice(i,1);
  DEVS.forEach(d => { const h = {...d, mine:true, g:'My devices'}; deriveHW(h); HW.push(h); });
  sortHW();
}

/* ---- Computer-vision models. params M, gf = GFLOPs at base input size ---- */
const CV = [];
const addFam = (fam, label, task, base, rows, extra={}) => rows.forEach(([s,p,gf]) =>
  CV.push({id:`${fam}${task==='det'?'':'-'+task}-${s}`, fam, name:`${label}${s}${task==='det'?'':'-'+task}`, group:extra.group||label.trim(), task, base, params:p, gf, nmsFree:!!extra.nmsFree, util:extra.util||1, npu:extra.npu!==false, int8ok:extra.int8ok!==false, ultra:extra.ultra!==false}));
addFam('yolo26','YOLO26','det',640,[['n',2.4,5.4],['s',9.5,20.7],['m',20.4,68.2],['l',24.8,86.4],['x',55.7,193.9]],{nmsFree:true,group:'YOLO26 (Ultralytics, 2026)'});
addFam('yolo26','YOLO26','seg',640,[['n',2.7,9.1],['s',10.4,34.2],['m',23.6,121.5],['l',28.0,139.8],['x',62.8,313.5]],{nmsFree:true,group:'YOLO26 (Ultralytics, 2026)'});
addFam('yolo26','YOLO26','obb',1024,[['n',2.5,14.0],['s',9.8,55.1],['m',21.2,183.3],['l',25.6,230.0],['x',57.6,516.5]],{nmsFree:true,group:'YOLO26 (Ultralytics, 2026)'});
addFam('yolo26','YOLO26','pose',640,[['n',2.9,7.5],['s',10.4,23.9],['m',21.5,73.1],['l',25.9,91.3],['x',57.6,201.7]],{nmsFree:true,group:'YOLO26 (Ultralytics, 2026)'});
addFam('yolo26','YOLO26','cls',224,[['n',2.8,0.5],['s',6.7,1.6],['m',11.6,4.9],['l',14.1,6.2],['x',29.6,13.6]],{nmsFree:true,group:'YOLO26 (Ultralytics, 2026)'});
addFam('yolo11','YOLO11','det',640,[['n',2.6,6.5],['s',9.4,21.5],['m',20.1,68.0],['l',25.3,86.9],['x',56.9,194.9]],{group:'YOLO11 / YOLOv8'});
addFam('yolo11','YOLO11','seg',640,[['n',2.9,10.4],['s',10.1,35.5],['m',22.4,123.3],['l',27.6,142.2],['x',62.1,319.0]],{group:'YOLO11 / YOLOv8'});
addFam('yolov8','YOLOv8','det',640,[['n',3.2,8.7],['s',11.2,28.6],['m',25.9,78.9],['l',43.7,165.2],['x',68.2,257.8]],{group:'YOLO11 / YOLOv8'});
CV.push({id:'rtdetr-l',fam:'rtdetr',name:'RT-DETR-l',group:'Transformers (RT-DETR, SAM)',task:'det',base:640,params:32,gf:110,nmsFree:true,util:0.73,npu:false,int8ok:false,ultra:true});
CV.push({id:'rtdetr-x',fam:'rtdetr',name:'RT-DETR-x',group:'Transformers (RT-DETR, SAM)',task:'det',base:640,params:67,gf:232,nmsFree:true,util:0.73,npu:false,int8ok:false,ultra:true});
[['mobilesam','MobileSAM',10,38,0.5],['sam2.1-t','SAM 2.1 tiny',39,300,0.3],['sam2.1-s','SAM 2.1 small',46,350,0.3],['sam2.1-b','SAM 2.1 base+',81,520,0.3],['sam2.1-l','SAM 2.1 large',224,1200,0.3],['sam-b','SAM ViT-B',91,487,0.3],['sam-h','SAM ViT-H',636,2976,0.3],['sam3','SAM 3',848,2600,0.3]]
  .forEach(([id,name,p,gf,u])=>CV.push({id,fam:'sam',name,group:'Transformers (RT-DETR, SAM)',task:'segany',base:1024,params:p,gf,nmsFree:true,util:u,npu:false,int8ok:false,ultra:true}));
[['resnet50','ResNet-50',25.6,4.1,224,1],['effb0','EfficientNet-B0',5.3,0.39,224,0.6],['vitb16','ViT-B/16',86,17.6,224,0.5],['clipl14','CLIP ViT-L/14 (image encoder)',304,81,224,0.5],['dinov2l','DINOv2 ViT-L/14',304,78,224,0.5],['dav2s','Depth Anything V2 Small',24.8,50,518,0.5],['dav2l','Depth Anything V2 Large',335,680,518,0.5]]
  .forEach(([id,name,p,gf,base,u])=>CV.push({id,fam:'backbone',name,group:'Classification, embeddings & depth',task:id.startsWith('dav2')?'depth':'cls',base,params:p,gf,nmsFree:true,util:u,npu:u>=0.6,int8ok:u>=0.6,ultra:false}));

[['osnet-x1','OSNet-x1.0 ReID (per person crop)',2.2,2.0,256,0.6,'reid',false,true,true,'Appearance model for tracking: run once per detected person. Cost = crops per frame × this.'],
 ['osnet-x025','OSNet-x0.25 ReID (per person crop)',0.2,0.16,256,0.5,'reid',false,true,true,'Tiny appearance model for trackers on weak hardware.'],
 ['dav2b','Depth Anything V2 Base',97.5,200,518,0.5,'depth',true,false,false,'Relative depth from a single image.'],
 ['ppocr-m','PaddleOCR mobile (detect + recognize)',15,12,640,0.5,'ocr',true,true,true,'Printed-text OCR for edge devices; cost grows with text lines per page.'],
 ['ppocr-s','PaddleOCR server (detect + recognize)',110,130,640,0.5,'ocr',true,false,true,'More accurate OCR for documents and scene text.'],
 ['doctr','docTR (DBNet-R50 + CRNN)',41,190,1024,0.55,'ocr',true,false,true,'Document OCR with a detection and a recognition network.'],
 ['dust3r','DUSt3R ViT-L (image pair → 3D)',571,1600,512,0.5,'recon3d',true,false,false,'Cost is per image pair.'],
 ['vggt','VGGT (per view → depth, camera pose, point map)',1200,3300,518,0.5,'recon3d',true,false,false,'Feed-forward 3D reconstruction; cost grows with the number of views.']]
  .forEach(([id,name,p,gf,base,u,task,approx,npu,int8ok,note])=>CV.push({id,fam:'extra',name,group:'Tracking, depth, OCR & 3D',task,base,params:p,gf,nmsFree:true,util:u,npu,int8ok,approx,note,ultra:false}));
// The user's own catalog (Computer Vision AI Hub) → mapped to base architectures
// Bundled snapshot of the Computer Vision AI Hub catalog (replaced by the live models.json when reachable)
const HUB_SNAPSHOT = [{"id":"xray-contraband-yolo26","name":"X-ray Contraband Detector - YOLO26","base_model":"yolo26s","task":"detection","image_size":640,"size_mb":20.3,"summary":"Detects prohibited items (guns, knives, batons, etc.) in X-ray baggage scans.","metrics":{"mAP50_95":0.816}},{"id":"jetson-nano-detector","name":"Jetson Nano Detector","base_model":"yolo26s","task":"detection","image_size":640,"size_mb":20.3,"summary":"Performs the detection of structure elements on Jetson Nano.","metrics":{"mAP50_95":0.856}},{"id":"jetson-nano-detector2","name":"Jetson Nano Detector 2","base_model":"yolo26m","task":"detection","image_size":640,"size_mb":44.1,"summary":"Performs the detection of structure elements on Jetson Nano.","metrics":{"mAP50_95":0.903}},{"id":"raspberry-pi-elements-detector","name":"Raspberry Pi elements detector","base_model":"yolo26n","task":"detection","image_size":640,"size_mb":5.4,"summary":"Performs the detection of structure elements on Raspberry Pi.","metrics":{"mAP50_95":0.945}},{"id":"raspberry-pi-elements-detector-rt-detr","name":"Raspberry Pi elements detector RT-DETR","base_model":"rtdetr-l","task":"detection","image_size":640,"size_mb":66.3,"summary":"Performs the detection of structure elements on Raspberry Pi.","metrics":{"mAP50_95":0.94}},{"id":"arduino-uno-elements-detector","name":"Arduino Uno elements detector","base_model":"yolo26n","task":"detection","image_size":640,"size_mb":5.4,"summary":"Performs the detection of structure elements on Arduino Uno.","metrics":{"mAP50_95":0.911}},{"id":"traffic-vehicle-detection","name":"Traffic Vehicle Detector","base_model":"yolo26n","task":"detection","image_size":640,"size_mb":5.4,"summary":"Detects cars, busses and trucks in the traffic recorded using CCTV facing the traffic.","metrics":{"mAP50_95":0.553}},{"id":"traffic-vehicle-detection2","name":"Traffic Vehicle Detector 2","base_model":"yolo26s","task":"detection","image_size":640,"size_mb":20.3,"summary":"Detects cars, busses and trucks in the traffic recorded using CCTV facing the traffic.","metrics":{"mAP50_95":0.57}},{"id":"football-player-detector","name":"Football player detector","base_model":"yolo26s","task":"detection","image_size":640,"size_mb":20.3,"summary":"Detects football players and the ball on macro view.","metrics":{"mAP50_95":0.499}},{"id":"football-player-detector2","name":"Football player detector2","base_model":"yolo26n","task":"detection","image_size":640,"size_mb":5.4,"summary":"Detects football players and the ball on macro view.","metrics":{"mAP50_95":0.49}},{"id":"barcode-detector","name":"Barcode detector","base_model":"yolo26n","task":"detection","image_size":640,"size_mb":5.4,"summary":"Performs the barcode detection.","metrics":{"mAP50_95":0.835}},{"id":"barcode-obb-detector","name":"Barcode detector OBB","base_model":"yolo26n-obb","task":"obb","image_size":640,"size_mb":21.6,"summary":"Performs the barcode detection using oriented bounding boxes.","metrics":{"mAP50_95":0.852}},{"id":"barcode-obb-detector2","name":"Barcode detector OBB 2","base_model":"yolo26m-obb","task":"obb","image_size":640,"size_mb":47.6,"summary":"Performs the barcode detection using oriented bounding boxes.","metrics":{"mAP50_95":0.84}},{"id":"dental-radiography-yolo26","name":"Dental Radiography - YOLO","base_model":"yolo26m","task":"detection","image_size":640,"size_mb":44.1,"summary":"The model detects various conditions from dental radiography images .","metrics":{"mAP50_95":0.393}},{"id":"dental-radiography-rt-detr","name":"Dental Radiography - RT-DETR","base_model":"rtdetr-l","task":"detection","image_size":640,"size_mb":66.3,"summary":"The model detects various conditions from dental radiography images .","metrics":{"mAP50_95":0.466}},{"id":"aerial-sheep-detector-yolo26","name":"Aerial Sheep Detector","base_model":"yolo26s","task":"detection","image_size":640,"size_mb":66.4,"summary":"Detects sheep in aerial images.","metrics":{"mAP50_95":0.552}},{"id":"parking-spot-detector","name":"Parking spot detector","base_model":"yolo26s-obb","task":"obb","image_size":640,"size_mb":21.8,"summary":"Detects empty/occupied parking spots.","metrics":{"mAP50_95":0.963}},{"id":"parking-spot-detector2","name":"Parking spot detector 2","base_model":"yolo26n-obb","task":"obb","image_size":640,"size_mb":5.85,"summary":"Detects empty/occupied parking spots.","metrics":{"mAP50_95":0.933}},{"id":"parking-spot-detector3","name":"Parking spot detector 3","base_model":"yolo26s-obb","task":"obb","image_size":640,"size_mb":5.85,"summary":"Detects empty/occupied parking spots.","metrics":{"mAP50_95":0.94}},{"id":"aerial-traffic-detector","name":"Aerial traffic detector","base_model":"yolo26s","task":"detection","image_size":640,"size_mb":20.3,"summary":"Detects vehicles in aerial images and tracks their movement.","metrics":{"mAP50_95":0.94}},{"id":"thremal-fire-detector","name":"Thermal fire detector","base_model":"yolo26s","task":"detection","image_size":640,"size_mb":20.3,"summary":"Detects fire from thermal images.","metrics":{"mAP50_95":0.388}},{"id":"license-plate-detector-rt-detr","name":"License Plate Detector - RT-DETR","base_model":"rtdetr-l","task":"detection","image_size":640,"size_mb":66.2,"summary":"Detects license plates on cars.","metrics":{"mAP50_95":0.777}},{"id":"license-plate-detector-yolo26","name":"License Plate Detector","base_model":"yolo26s","task":"detection","image_size":640,"size_mb":20.3,"summary":"Detects license plates on cars.","metrics":{"mAP50_95":0.841}},{"id":"license-plate-ocr-detector-rt-detr","name":"License Plate OCR Detector - RT-DETR","base_model":"rtdetr-l","task":"detection","image_size":640,"size_mb":66.4,"summary":"Detects numbers and letters on license plates and performs OCR on images.","metrics":{"mAP50_95":0.806}},{"id":"license-plate-ocr-detector","name":"License plate OCR detector","base_model":"yolo26s","task":"detection","image_size":640,"size_mb":20.4,"summary":"Detects numbers and letters on license plates and performs OCR on images.","metrics":{"mAP50_95":0.8}},{"id":"license-place-ocr-detector2","name":"License plate OCR detector 2","base_model":"yolo26n","task":"detection","image_size":640,"size_mb":5.4,"summary":"Detects numbers and letters on license plates and performs OCR on images.","metrics":{"mAP50_95":0.797}},{"id":"license-place-ocr-detector3","name":"License plate OCR detector 3","base_model":"yolo26m","task":"detection","image_size":640,"size_mb":44.1,"summary":"Detects numbers and letters on license plates and performs OCR on images.","metrics":{"mAP50_95":0.802}},{"id":"military-vehicle-detector-yolo26","name":"Military Vehicle Detector","base_model":"yolo26s","task":"detection","image_size":640,"size_mb":20.3,"summary":"Detects military vehicles.","metrics":{"mAP50_95":0.492}},{"id":"military-vehicle-detector2-yolo26","name":"Military Vehicle Detector 2","base_model":"yolo26m","task":"detection","image_size":640,"size_mb":20.3,"summary":"Detects military vehicles.","metrics":{"mAP50_95":0.505}},{"id":"military-vehicle-detector-rt-detr","name":"Military Vehicle Detector - RT-DETR","base_model":"rtdetr-l","task":"detection","image_size":640,"size_mb":66.2,"summary":"Detects military vehicles.","metrics":{"mAP50_95":0.532}},{"id":"road-line-segmentator","name":"Road line segmentator","base_model":"yolo26s-seg","task":"segmentation","image_size":640,"size_mb":23.3,"summary":"Segments the road lines from the environment.","metrics":{"mAP50_95":0.939}},{"id":"road-line-segmentator2","name":"Road line segmentator 2","base_model":"yolo26s-seg","task":"segmentation","image_size":640,"size_mb":23.3,"summary":"Segments the road lines from the environment.","metrics":{"mAP50_95":0.865}},{"id":"road-line-segmentator3","name":"Road line segmentator 3","base_model":"yolo26s-seg","task":"segmentation","image_size":640,"size_mb":23.3,"summary":"Segments the lines on the road from the environment. Enables to detect dashed and solid lines on the left and right side of the road.","metrics":{"mAP50_95":0.716}},{"id":"road-line-segmentator4","name":"Road line segmentator 4","base_model":"yolo26s-seg","task":"segmentation","image_size":640,"size_mb":23.3,"summary":"Segments the lines on the road from the environment. Enables to detect dashed and solid lines on both sides of the road.","metrics":{"mAP50_95":0.723}}];
const TASKNAME = {det:'detection', seg:'segmentation', obb:'oriented boxes', pose:'pose', cls:'classification', segany:'promptable segmentation', depth:'depth estimation', reid:'tracking (re-identification)', ocr:'text recognition (OCR)', recon3d:'3D reconstruction'};

/* ---- Image generation models. params/te in B, gf = GFLOPs per denoising step at base res (incl. CFG) ---- */
const GEN = [
  {id:'sd15',  name:'Stable Diffusion 1.5', params:0.86, te:0.12, base:512,  gf:1600,  vae:600,  act:0.8, steps:25, cfg:true},
  {id:'sdxl',  name:'SDXL 1.0',             params:2.57, te:0.82, base:1024, gf:6000,  vae:2500, act:2.5, steps:30, cfg:true},
  {id:'sdxlt', name:'SDXL Lightning / Turbo (4-step)', params:2.57, te:0.82, base:1024, gf:3000, vae:2500, act:2.5, steps:4, cfg:false},
  {id:'sd35m', name:'Stable Diffusion 3.5 Medium', params:2.5, te:5.6, base:1024, gf:9000,  vae:2500, act:2.5, steps:28, cfg:true},
  {id:'sd35l', name:'Stable Diffusion 3.5 Large',  params:8.1, te:5.6, base:1024, gf:28000, vae:2500, act:4.0, steps:28, cfg:true},
  {id:'fluxs', name:'FLUX.1 [schnell] (4-step)',   params:12, te:4.9, base:1024, gf:40000, vae:2500, act:4.0, steps:4,  cfg:false},
  {id:'fluxd', name:'FLUX.1 [dev] / Kontext',      params:12, te:4.9, base:1024, gf:40000, vae:2500, act:4.0, steps:28, cfg:false},
];

/* ---- LLMs ---- */
const PREC = {
  fp16:{bits:16,label:'FP16 / BF16',quality:'reference quality'}, fp8:{bits:8,label:'FP8',quality:'near-lossless'},
  int8:{bits:8.5,label:'INT8 / Q8_0',quality:'near-lossless'}, q6:{bits:6.56,label:'Q6_K (GGUF)',quality:'near-lossless'},
  q5:{bits:5.69,label:'Q5_K_M (GGUF)',quality:'very small loss'}, q4:{bits:4.85,label:'Q4_K_M (GGUF)',quality:'small loss'},
  int4:{bits:4.25,label:'INT4 AWQ/GPTQ · MXFP4',quality:'small loss'},
};
const LLMS = [
  {id:'llama32-3b', name:'Llama 3.2 3B', hf:'meta-llama/Llama-3.2-3B-Instruct', params:3.21, active:3.21, layers:28, heads:24, kv:8, hd:128},
  {id:'qwen25-7b',  name:'Qwen2.5 7B', hf:'Qwen/Qwen2.5-7B-Instruct', params:7.62, active:7.62, layers:28, heads:28, kv:4, hd:128},
  {id:'llama31-8b', name:'Llama 3.1 8B', hf:'meta-llama/Llama-3.1-8B-Instruct', params:8.03, active:8.03, layers:32, heads:32, kv:8, hd:128},
  {id:'qwen3-8b',   name:'Qwen3 8B', hf:'Qwen/Qwen3-8B', params:8.19, active:8.19, layers:36, heads:32, kv:8, hd:128},
  {id:'gemma2-9b',  name:'Gemma 2 9B', hf:'google/gemma-2-9b-it', params:9.24, active:9.24, layers:42, heads:16, kv:8, hd:256},
  {id:'qwen25-14b', name:'Qwen2.5 14B', hf:'Qwen/Qwen2.5-14B-Instruct', params:14.7, active:14.7, layers:48, heads:40, kv:8, hd:128},
  {id:'gptoss-20b', name:'gpt-oss-20b (MoE)', hf:'openai/gpt-oss-20b', params:20.9, active:3.6, layers:24, heads:64, kv:8, hd:64},
  {id:'mistral-24b',name:'Mistral Small 24B', hf:'mistralai/Mistral-Small-24B-Instruct-2501', params:23.6, active:23.6, layers:40, heads:32, kv:8, hd:128},
  {id:'qwen3-30b',  name:'Qwen3 30B-A3B (MoE)', hf:'Qwen/Qwen3-30B-A3B', params:30.5, active:3.3, layers:48, heads:32, kv:4, hd:128},
  {id:'qwen25-32b', name:'Qwen2.5 32B', hf:'Qwen/Qwen2.5-32B-Instruct', params:32.8, active:32.8, layers:64, heads:40, kv:8, hd:128},
  {id:'llama33-70b',name:'Llama 3.3 70B', hf:'meta-llama/Llama-3.3-70B-Instruct', params:70.6, active:70.6, layers:80, heads:64, kv:8, hd:128},
  {id:'gptoss-120b',name:'gpt-oss-120b (MoE)', hf:'openai/gpt-oss-120b', params:117, active:5.1, layers:36, heads:64, kv:8, hd:64},
  {id:'deepseek-v3',name:'DeepSeek V3 / R1 671B (MoE)', hf:'deepseek-ai/DeepSeek-V3', params:671, active:37, layers:61, heads:128, kv:128, hd:128, kvl:576},
];

