/* ============================== APP STATE ============================== */
const MODES = {vision:VISION, gen:GENMODE, llm:LLMMODE, vlm:VLMMODE, vla:VLAMODE};
const TYPES = [['vision','Computer vision'],['gen','Image generation'],['llm','LLM'],['vlm','VLM'],['vla','VLA (robots)']];
const TYPE_LABEL = Object.fromEntries(TYPES);
const HUB_URL = 'https://computer-vision-ai-hub.github.io/ComputerVisionAIHubLibrary.github.io/models.json';
const HUB_SITE = 'https://computer-vision-ai-hub.github.io/ComputerVisionAIHubLibrary.github.io/';
const DEFAULT_WL = {
  vision:{prec:'fp16', rt:'auto', streams:1, fps:30, lat:50},
  gen:{prec:'fp16', batch:1, offload:false, perHour:30, maxS:20},
  llm:{prec:'q4', kvprec:'fp16', users:1, ctx:8192, prompt:1000, output:400, t_ttft:1000, t_tps:20},
  vlm:{prec:'q4', kvprec:'fp16', imgs:1, out:100, rpm:6, maxS:5},
  vla:{prec:'fp16', rt:'pt', hz:10, lat:200},
};
const WL = JSON.parse(JSON.stringify(DEFAULT_WL));
const C = {pay:'buy', budget:0, maxW:0, edge:false, hours:8, kwh:0.25, legacy:false};
const SEL = {hw:'jorinnano', n:1};
const OV = {};                       // per-model overrides (input size, steps, resolution)
const F = {type:'all', q:'', fits:false, sort:'default'};
let ITEMS = [], HUB = HUB_SNAPSHOT, hubLive = false, CUSTOM = [], FITS = new Map(), CUR = null;
/* ---- LIVE DATA: keeps the page current without new commits to this file ----
   1. HUB_URL      your catalog's models.json, read on every visit
   2. HF trending  newest popular LLMs / VLMS from the Hugging Face API (cached 24 h per browser)
   3. DATA_URL     optional overrides file: {"hardware":[…],"vision":[…],"gen":[…],"llm":[…],"vlm":[…],"vla":[…]}
                   entries are merged by "id" */
const DATA_URL = 'data.json';
const HF_TRENDING = {llm:'text-generation', vlm:'image-text-to-text'}, HF_LIMIT = 6, HF_TTL = 24*3600*1000;
let LIVE = [], hfLive = false, dataLive = false;
const els = {};
const slug = s => String(s).toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'');
const cap1 = s => s.charAt(0).toUpperCase() + s.slice(1);
function baseToId(b){
  b = String(b||'').toLowerCase();
  const y = b.match(/^(yolov?\d+)([nsmlx])(?:-(seg|obb|pose|cls))?$/);
  if(y) return `${y[1]}${y[3]?'-'+y[3]:''}-${y[2]}`;
  return CV.some(x=>x.id===b) ? b : null;
}
function loadCustom(){ try{ CUSTOM = JSON.parse(localStorage.getItem('computefit.custom')||'[]') || []; }catch(e){ CUSTOM = []; } }
function saveCustom(){ try{ localStorage.setItem('computefit.custom', JSON.stringify(CUSTOM)); }catch(e){} }

function buildItems(){
  ITEMS = [];
  for(const h of HUB){
    const base = CV.find(x=>x.id===baseToId(h.base_model)); if(!base) continue;
    const sz = +h.image_size || 640, map = h.metrics && h.metrics.mAP50_95;
    ITEMS.push({id:'hub-'+slug(h.id), type:'vision', hub:true, name:h.name, family:h.base_model, task:TASKNAME[base.task], group:'Hub catalog',
      summary:h.summary || `${cap1(TASKNAME[base.task])} model from the Computer Vision AI Hub catalog.`,
      m:{...base, displayName:`${h.name} (${base.name})`, catalog:h.name}, per:{imgsz:sz},
      meta:[`${base.params}M params`, `${sz}px`, map!=null?`mAP50-95 ${(+map).toFixed(2)}`:null], link:HUB_SITE+'#m='+encodeURIComponent(h.id), linkText:'Hub catalog page'});
  }
  for(const m of CV) ITEMS.push({id:slug(m.id), type:'vision', name:m.name, family:m.fam, task:TASKNAME[m.task], group:m.group,
    summary:`${cap1(TASKNAME[m.task])} · ${m.group}. Pretrained base model.${m.note?' '+m.note:''}${m.approx?' FLOPs are estimated from the architecture.':''}`, m:{...m, displayName:m.name}, per:{imgsz:m.base},
    meta:[`${m.params}M params`, `${m.gf} GFLOPs`, `${m.base}px`], link:m.ultra?'https://docs.ultralytics.com/models/':null, linkText:'Ultralytics docs'});
  for(const m of GEN) ITEMS.push({id:slug(m.id), type:'gen', name:m.name, family:m.id, task:'text-to-image', group:'Diffusion',
    summary:`Text-to-image diffusion model, ${m.steps} steps at ${m.base}px by default.`, m:{...m, displayName:m.name}, per:{res:m.base, steps:m.steps},
    meta:[`${m.params}B params`, `${m.base}px`, `${m.steps} steps`]});
  for(const m of LLMS) ITEMS.push(llmItem(m));
  for(const m of VLMS) ITEMS.push(vlmItem(m));
  for(const m of VLAS) ITEMS.push(vlaItem(m));
  const known = new Set(ITEMS.map(i=>i.id));
  for(const c of LIVE){ const it = c.type==='vlm' ? vlmItem(c.m) : llmItem(c.m); if(!known.has(it.id)){ known.add(it.id); ITEMS.push({...it, live:true, family:'trending', group:'Trending on Hugging Face', summary:'Trending on Hugging Face right now. '+it.summary+(c.type==='vlm'?' Vision encoder size is assumed.':'')}); } }
  for(const c of CUSTOM) ITEMS.push({...(c.type==='vlm' ? vlmItem(c.m) : llmItem(c.m)), custom:true, group:'Imported'});
}
function llmItem(m){ const moe = m.active < m.params*0.9;
  return {id:slug(m.id||m.name), type:'llm', name:m.name, family:moe?'MoE':'dense', task:'text', group:'Language models',
    summary:`${m.params}B-parameter language model${moe?`, mixture-of-experts (${m.active}B active per token)`:''}.`, m:{...m, displayName:m.name}, per:{},
    meta:[`${m.params}B params`, `${m.layers} layers`, moe?`${m.active}B active`:null], link:m.hf?'https://huggingface.co/'+m.hf:null, linkText:m.hf}; }
function vlaItem(m){
  const tot = m.params+m.vparams+m.act;
  return {id:slug(m.id||m.name), type:'vla', name:m.name, family:'VLA', task:'robot actions', group:'Vision-language-action',
    summary:`Robot-control model: reads camera images and an instruction, outputs ${m.chunk>1?'a chunk of '+m.chunk+' actions':'one action'} per inference${m.ar>0?' (autoregressive)':m.steps>0?` with ${m.steps} flow / diffusion steps`:''}.${m.note?' '+m.note:''}`,
    m:{...m, displayName:m.name}, per:{cams:m.cams, chunk:m.chunk, steps:m.steps},
    meta:[`${tot.toFixed(tot<1?2:1)}B params`, `${m.chunk}-action chunk`, m.ar>0?'autoregressive':m.steps>0?`${m.steps} flow steps`:'single pass'], link:m.link||(m.hf?'https://huggingface.co/'+m.hf:null), linkText:m.hf||'Project page'}; }
function vlmItem(m){ const res = m.defRes||768;
  return {id:slug(m.id||m.name), type:'vlm', name:m.name, family:'VLM', task:'image + text', group:'Vision-language models',
    summary:`Vision-language model: answers questions about images, reads text, describes scenes.${m.note?' '+m.note:''}`, m:{...m, displayName:m.name}, per:{res},
    meta:[`${(m.params+m.vparams).toFixed(m.params<1?2:1)}B params`, `${fmtInt(vlmTok(m,res))} tokens / image`], link:m.hf?'https://huggingface.co/'+m.hf:null, linkText:m.hf}; }

const wOf = item => ({...WL[item.type], ...item.per, ...(OV[item.id]||{})});
const curHW = () => HW.find(h=>h.id===SEL.hw) || null;
const nFor = (hw,type) => !hw ? 1 : (type==='vla' || ((type==='llm'||type==='vlm') && hw.kind!=='gpu')) ? 1 : hw.fixedN ? hw.fixedN : Math.min(SEL.n, hw.max);
function fitOf(item){
  const mode = MODES[item.type], w = wOf(item), hw = curHW();
  let r = null, st = 'nohw';
  if(hw){
    if(!mode.supports(hw,item.m)) st = 'unsupported';
    else { r = mode.evaluate(item.m,w,hw,nFor(hw,item.type)); if(r.unsupported){ st='unsupported'; r=null; } else st = r.pass ? (r.checks.some(c=>c.st==='warn')?'warn':'pass') : 'fail'; }
  }
  let from = null;
  if(st!=='pass' && st!=='warn'){ const up = pickOptions(mode,item.m,w,null,C); from = up.picks[0] ? up.picks[0].r : null; }
  return {st, r, w, from, hw};
}
const priceOf = r => C.pay==='cloud' ? '$'+r.cost.toFixed(2)+'/h' : fmtUSD(r.cost);
const loadOf = r => r.fits ? 1/Math.max(r.headroom,0.01) : 9.99;
const pctTxt = l => l>=9.99 ? '>999%' : Math.round(l*100)+'%';

/* ============================== GRID ============================== */
function ringSVG(pct, track){
  const Cc = 2*Math.PI*30;
  return `<svg viewBox="0 0 72 72" aria-hidden="true"><circle cx="36" cy="36" r="30" fill="none" stroke="${track||'rgba(4,44,83,.1)'}" stroke-width="7"/><circle cx="36" cy="36" r="30" fill="none" stroke="var(--sc)" stroke-width="7" stroke-linecap="round" stroke-dasharray="${Cc}" stroke-dashoffset="${Cc*(1-Math.max(0,Math.min(1,pct)))}"/></svg>`;
}
function tabText(f){
  if(f.st==='pass') return `fits · ${f.r.headroom.toFixed(1)}×`;
  if(f.st==='warn') return 'fits · tight';
  if(f.st==='fail') return f.r.fits ? 'too slow' : 'no memory';
  if(f.st==='unsupported') return 'not supported';
  return f.from ? `from ${priceOf(f.from)}` : 'no device fits';
}
function cardHTML(item){
  const f = FITS.get(item.id), mode = MODES[item.type], r = f.r;
  let visual, stats;
  if(r){
    const l = loadOf(r), [v,u] = r.fits ? mode.headline(r) : ["Doesn't fit", ''];
    const need = f.st==='fail' && f.from ? `<br><b>needs ${esc(hwLabel(f.from.hw,f.from.n))}</b>` : '';
    visual = `<div class="fit-ring">${ringSVG(l)}<b>${r.fits?pctTxt(l):'RAM'}</b></div>
      <div class="fit-txt"><p class="fit-big">${v} <small>${u}</small></p><p class="fit-small">${r.fits?'on':'needs '+fmtGB(mode.mem(r).need)+' ·'} ${esc(f.hw.name)}${need}</p></div>`;
    const cs = r.fits ? [r.checks[0], r.checks[1]] : [r.checks[2]];
    stats = `<div class="card-stats">${cs.map(c=>`<div><div class="stat-label">${c.k} <b>${c.v}${c.unit?' '+c.unit:''}</b></div><div class="meter ${c.st}" aria-hidden="true"><span style="width:${Math.round(Math.min(1,Math.max(.03,c.ratio||0))*100)}%"></span></div></div>`).join('')}</div>`;
  } else {
    const big = f.st==='unsupported' ? 'Not supported' : f.from ? esc(hwLabel(f.from.hw,f.from.n)) : 'No single device';
    const small = f.st==='unsupported' ? `on ${esc(f.hw.name)}` : f.from ? `cheapest that meets the targets · ${priceOf(f.from)}` : 'meets the default targets';
    visual = `<div class="fit-ring">${ringSVG(0)}<b>${f.st==='unsupported'?'✕':'?'}</b></div><div class="fit-txt"><p class="fit-big" style="font-size:1.05rem">${big}</p><p class="fit-small">${small}</p></div>`;
    stats = `<p class="card-from">${f.from ? `Runs from <b>${esc(hwLabel(f.from.hw,f.from.n))}</b> (${priceOf(f.from)}).` : 'No device in the database meets the default targets — open it to relax them.'}</p>`;
  }
  const meta = item.meta.filter(Boolean).concat(r ? [fmtGB(mode.mem(r).need)+' memory'] : []);
  return `<article class="model-card st-${f.st}">
    <span class="card-tab">${esc(item.family)} <b>${tabText(f)}</b></span>
    <div class="card-image">${visual}<span class="task-badge">${esc(item.task)}</span></div>
    <div class="card-body">
      <h2 class="card-name"><a href="#fit-${item.id}" data-fit="${item.id}">${esc(item.name)}</a></h2>
      <p class="card-summary">${esc(item.summary)}</p>
      ${stats}
      <div class="card-meta">${meta.map(x=>`<span>${esc(x)}</span>`).join('')}</div>
      <div class="card-actions"><button type="button" class="btn try-btn" data-fit="${item.id}">Check fit</button><button type="button" class="btn secondary" data-info="${item.id}">Details</button></div>
    </div></article>`;
}
function visibleItems(){
  const q = F.q.trim().toLowerCase();
  let list = ITEMS.filter(it => (F.type==='all' || (F.type==='hub' ? it.hub : F.type==='live' ? it.live : it.type===F.type))
    && (!q || `${it.name} ${it.family} ${it.task} ${it.group} ${TYPE_LABEL[it.type]}`.toLowerCase().includes(q))
    && (!F.fits || ['pass','warn'].includes(FITS.get(it.id).st)));
  const rank = {pass:0, warn:1, fail:2, nohw:2, unsupported:3};
  const memOf = it => { const f = FITS.get(it.id); return f.r ? MODES[it.type].mem(f.r).need : (it.m.params||0)*(it.type==='vision'?1e6:1e9)*2; };
  if(F.sort==='fit') list = list.slice().sort((a,b)=>{ const x=FITS.get(a.id), y=FITS.get(b.id); return rank[x.st]-rank[y.st] || (y.r?y.r.headroom:0)-(x.r?x.r.headroom:0); });
  if(F.sort==='light') list = list.slice().sort((a,b)=>memOf(a)-memOf(b));
  if(F.sort==='name') list = list.slice().sort((a,b)=>a.name.localeCompare(b.name));
  return list;
}
function renderPills(){
  const n = t => ITEMS.filter(it => t==='all' || (t==='hub' ? it.hub : t==='live' ? it.live : it.type===t)).length;
  const pills = [['all','All'], ...TYPES, ['hub','My Hub models']].concat(LIVE.length ? [['live','New · trending']] : []);
  els.filters.innerHTML = pills.map(([t,l])=>`<button type="button" class="pill" data-type="${t}" aria-pressed="${F.type===t}">${l} <span class="n">${n(t)}</span></button>`).join('')
    + `<button type="button" class="pill toggle" data-fits aria-pressed="${F.fits}">Fits my hardware</button>`;
}
function renderGrid(){
  FITS = new Map(ITEMS.map(it => [it.id, fitOf(it)]));
  const list = visibleItems(), hw = curHW();
  const fitsN = ITEMS.filter(it => ['pass','warn'].includes(FITS.get(it.id).st)).length;
  els.grid.innerHTML = list.map(cardHTML).join('');
  els.state.hidden = list.length>0;
  els.state.innerHTML = list.length ? '' : `<p>No models match these filters.</p><button type="button" class="btn secondary" id="reset-filters">Clear filters</button>`;
  els.count.innerHTML = `<b>${list.length}</b> of ${ITEMS.length} models${hw ? ` · <b>${fitsN}</b> fit ${esc(hw.name)}` : ' · showing the cheapest device for each'} · <span title="Hub catalog: ${hubLive?'read live from the catalog':'bundled snapshot'} · Hugging Face trending: ${hfLive?'live':'not loaded'}${dataLive?' · data.json overrides applied':''}">data: ${hubLive||hfLive||dataLive?'live':'bundled snapshot'}</span>`;
  document.querySelector('[data-stat=models]').textContent = ITEMS.length;
  document.querySelector('[data-stat=devices]').textContent = HW.length;
  document.querySelector('[data-stat=fits]').textContent = hw ? fitsN : '–';
  const v=WL.vision, g=WL.gen, l=WL.llm, x=WL.vlm, a=WL.vla;
  $('targets-note').innerHTML = `Default targets used for the cards — <b>Vision:</b> ${v.streams} camera${v.streams>1?'s':''} × ${fmtN(v.fps)} FPS, delay ≤ ${v.lat} ms, ${v.prec.toUpperCase()}. <b>Image generation:</b> ${fmtInt(g.perHour)} images/hour, ≤ ${g.maxS} s each. <b>LLM:</b> ${l.users} user${l.users>1?'s':''}, ≥ ${l.t_tps} tokens/s, first word ≤ ${fmtInt(l.t_ttft)} ms. <b>VLM:</b> ${x.imgs} image${x.imgs>1?'s':''} per question, ${fmtN(x.rpm)} answers/minute, ≤ ${x.maxS} s each. <b>VLA:</b> ≥ ${fmtN(a.hz)} actions/s, reaction ≤ ${fmtInt(a.lat)} ms. Targets you change inside a model apply to every card of that type. <button type="button" class="chip" id="reset-targets">Reset to defaults</button>`;
  renderHero();
}
function renderHero(){
  const it = ITEMS.find(i=>i.id==='hub-traffic-vehicle-detection2') || ITEMS[0]; if(!it) return;
  const f = FITS.get(it.id), col = {pass:'var(--mint-bright)', warn:'#F2C46D', fail:'#F0A08C', unsupported:'#F0A08C', nohw:'var(--sky)'}[f.st];
  let rows = '', sub, label, ring;
  if(f.r){ rows = f.r.checks.map(c=>`<div><dt>${c.k}</dt><dd>${c.v}${c.unit?' '+c.unit:''}</dd></div>`).join(''); sub = `on ${f.hw.name}`;
    label = f.st==='fail' ? 'does not meet targets' : `meets targets · ${f.r.headroom.toFixed(1)}× headroom`; ring = `${ringSVG(loadOf(f.r),'rgba(255,255,255,.14)')}<b>${f.r.fits?pctTxt(loadOf(f.r)):'RAM'}</b>`; }
  else { sub = f.from ? `runs from ${hwLabel(f.from.hw,f.from.n)}` : (f.hw?`not supported on ${f.hw.name}`:'pick your hardware'); label = f.from ? `from ${priceOf(f.from)}` : 'no device'; ring = `${ringSVG(0,'rgba(255,255,255,.14)')}<b>?</b>`;
    if(f.from) rows = f.from.rows.slice(0,3).map(([k,v])=>`<div><dt>${k}</dt><dd>${v}</dd></div>`).join(''); }
  $('hero-shot').innerHTML = `<div class="shot-top" style="--sc:${col}"><div class="fit-ring" style="width:86px;height:86px">${ring}</div><div><p class="shot-title">${esc(it.name)}</p><p class="shot-sub">${esc(it.family)} · ${esc(sub)}</p></div></div><dl class="shot-rows">${rows}</dl>`;
  const lab = $('hero-label'); lab.textContent = label; lab.className = 'det-label' + (f.st==='warn'?' warn':['fail','unsupported'].includes(f.st)?' fail':'');
}
function hwSelectHTML(mode, m, selId){
  if(selId===undefined) selId = SEL.hw;
  const list = HW.filter(h => !mode || mode.supports(h,m)), groups = {};
  list.forEach(h => (groups[h.g] = groups[h.g]||[]).push(h));
  const cur = HW.find(h=>h.id===selId) || null, unsup = mode && cur && !mode.supports(cur,m) ? `<option value="${cur.id}" selected>${esc(cur.name)} — not supported</option>` : '';
  return `<option value="none">I don't have hardware yet</option>` + unsup + Object.entries(groups).map(([g,hs])=>`<optgroup label="${esc(g)}">${hs.map(h=>`<option value="${h.id}"${h.id===selId?' selected':''}>${esc(h.name)}</option>`).join('')}</optgroup>`).join('');
}
function renderAll(){ renderPills(); els.hw.innerHTML = hwSelectHTML(); els.hw.value = curHW() ? SEL.hw : 'none'; renderGrid(); if(typeof fillStack==='function') fillStack(); }

/* ============================== DETAILS DRAWER ============================== */
function openDrawer(item){
  const f = FITS.get(item.id) || fitOf(item), mode = MODES[item.type], m = item.m, w = f.w;
  const tile = (k,v) => `<div class="tile"><dt>${k}</dt><dd>${v}</dd></div>`;
  const tiles = item.type==='vision' ? [['Params', m.params+'M'], ['GFLOPs', m.gf], ['Native size', m.base+'px'], ['Your size', w.imgsz+'px']]
    : item.type==='gen' ? [['Image model', m.params+'B'], ['Text encoder', m.te+'B'], ['Native size', m.base+'px'], ['Steps', w.steps]]
    : item.type==='llm' ? [['Params', m.params+'B'], ['Active', m.active+'B'], ['Layers', m.layers], ['KV heads', m.kv]]
    : item.type==='vla' ? [['Backbone', m.params?m.params+'B':'—'], ['Action head', m.act?m.act+'B':'tokens'], ['Cameras', w.cams], ['Chunk', w.chunk+' actions']]
    : [['Language', m.params+'B'], ['Vision', m.vparams+'B'], ['Tokens / image', fmtInt(vlmTok(m,w.res))], ['Layers', m.layers]];
  const devs = HW.filter(h => mode.supports(h,m)).map(h => { const r = mode.evaluate(m,w,h,1); return {h, ok: !r.unsupported && r.pass}; });
  const okN = devs.filter(d=>d.ok).length;
  const vb = {pass:['pass','fits your hardware'], warn:['warn','fits · tight'], fail:['fail','does not meet targets'], unsupported:['fail','not supported'], nohw:['','no hardware selected']}[f.st];
  els.drawer.innerHTML = `
    <div class="drawer-head"><div><p class="drawer-id">${esc(item.id)}</p><h2 id="drawer-title">${esc(item.name)}</h2></div>
      <button type="button" class="icon-btn" data-drawer-close aria-label="Close details"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg></button></div>
    <div class="drawer-body">
      <p class="drawer-summary">${esc(item.summary)}</p>
      <div class="badges"><span class="badge base">${esc(item.family)}</span><span class="badge">${esc(TYPE_LABEL[item.type])}</span><span class="badge">${esc(item.task)}</span><span class="badge ${vb[0]}">${vb[1]}</span></div>
      <dl class="tiles">${tiles.map(([k,v])=>tile(k,v)).join('')}</dl>
      <div><p class="block-label">Runs on · ${okN} of ${devs.length} single devices at the default targets</p>
        <div class="chips">${devs.map(d=>d.ok ? `<button type="button" class="chip ok" data-pick-hw="${d.h.id}" title="Check fit on this device">${esc(d.h.name)}</button>` : `<span class="chip no">${esc(d.h.name)}</span>`).join('')}</div></div>
      <div><p class="block-label">Facts</p><dl class="facts">
        <dt>Type</dt><dd>${esc(TYPE_LABEL[item.type])}</dd><dt>Group</dt><dd>${esc(item.group)}</dd>
        ${item.type==='vision'?`<dt>Base architecture</dt><dd>${esc(m.name)}</dd><dt>INT8 friendly</dt><dd>${m.int8ok?'yes':'validate accuracy'}</dd>`:''}
        ${item.link?`<dt>Source</dt><dd><a href="${esc(item.link)}" target="_blank" rel="noopener">${esc(item.linkText||'link')}</a></dd>`:''}
        ${item.custom?`<dt>Imported</dt><dd><button type="button" class="chip" data-remove="${item.id}">remove from this browser</button></dd>`:''}
      </dl></div>
    </div>
    <div class="drawer-foot"><button type="button" class="btn try-btn" data-fit="${item.id}">Check fit on my hardware</button></div>`;
  els.drawerOverlay.hidden = false; document.body.classList.add('modal-open');
  els.drawer.scrollTop = 0; els.drawer.querySelector('[data-drawer-close]').focus();
}
function closeDrawer(){ els.drawerOverlay.hidden = true; if(els.fitModal.hidden && els.addModal.hidden) document.body.classList.remove('modal-open'); }

/* ============================== CHECK-FIT MODAL ============================== */
const fld = (k,label,o={}) => ({k,label,scope:'wl',...o});
function fieldSpec(item,w){
  const m = item.m, t = item.type;
  const precL = Object.entries(PREC).map(([k,p])=>[k,p.label]);
  if(t==='vision'){
    const sizes = m.base===224 ? [160,224,288,384] : m.base===518 ? [364,518,700] : (m.task==='reid'||m.task==='ocr'||m.task==='recon3d') ? [0.5,0.75,1,1.5].map(f=>Math.round(m.base*f/32)*32) : m.fam==='sam' ? [512,768,1024] : [320,416,512,640,768,960,1024,1280];
    return {model:[fld('imgsz','Input size (px)',{opts:sizes.map(s=>[s,s+(s===m.base?' (native)':'')]),scope:'ov',num:1}), fld('prec','Precision',{opts:[['fp32','FP32'],['fp16','FP16'],['int8','INT8']]}), fld('rt','Runtime',{opts:[['auto','Best for the device'],['onnx','ONNX Runtime'],['pytorch','PyTorch (.pt)']],wide:1})],
      targets:[fld('streams','Cameras',{num:1,int:1}), fld('fps','FPS each',{num:1}), fld('lat','Max delay (ms)',{num:1,wide:1})],
      presets:[['1 camera, real-time',{streams:1,fps:30,lat:50}],['Factory inspection',{streams:1,fps:60,lat:20}],['CCTV, 8 cameras',{streams:8,fps:15,lat:200}],['Parking, 4 cams',{streams:4,fps:10,lat:300}],['Drone / robot',{streams:1,fps:30,lat:40}],['Photo batch',{streams:1,fps:5,lat:2000}]]};
  }
  if(t==='gen') return {model:[fld('res','Image size',{opts:(m.base===512?[512,768]:[768,1024,1280,1536]).map(r=>[r,`${r}×${r}`]),scope:'ov',num:1}), fld('steps','Steps',{num:1,int:1,scope:'ov'}), fld('prec','Precision',{opts:[['fp16','FP16 / BF16'],['fp8','FP8'],['nf4','NF4 / GGUF Q4']]}), fld('batch','Images per request',{num:1,int:1}), fld('offload','Offload text encoder after encoding',{chk:1,wide:1})],
    targets:[fld('perHour','Images per hour',{num:1}), fld('maxS','Max wait (s)',{num:1})],
    presets:[['Personal use',{perHour:20,maxS:30}],['Team tool',{perHour:150,maxS:20}],['Production API',{perHour:1000,maxS:15}],['Real-time preview',{perHour:600,maxS:2}]]};
  if(t==='llm') return {model:[fld('prec','Precision',{opts:precL}), fld('kvprec','KV cache',{opts:[['fp16','FP16'],['fp8','FP8 (½ memory)']]}), fld('ctx','Max context (tokens)',{num:1,int:1,wide:1})],
    targets:[fld('users','Users at once',{num:1,int:1}), fld('t_tps','Min tokens/s',{num:1}), fld('t_ttft','Max time to first word (ms)',{num:1,wide:1})],
    presets:[['Personal chat',{users:1,t_tps:20,t_ttft:1000}],['Team assistant',{users:10,t_tps:20,t_ttft:1500}],['Coding agent',{users:2,t_tps:50,t_ttft:800}]]};
  if(t==='vla') return {model:[fld('prec','Precision',{opts:[['fp16','FP16 / BF16'],['fp8','FP8'],['int8','INT8'],['int4','INT4']]}), fld('rt','Runtime',{opts:[['pt','PyTorch / JAX (default)'],['opt','Optimized (TensorRT, CUDA graphs)']]}), fld('cams','Cameras',{num:1,int:1,scope:'ov'}), fld('chunk','Actions per inference',{num:1,int:1,scope:'ov'})].concat(m.steps>0 ? [fld('steps','Denoising steps',{num:1,int:1,scope:'ov',wide:1})] : []),
    targets:[fld('hz','Control rate (actions/s)',{num:1}), fld('lat','Max reaction time (ms)',{num:1})],
    presets:[['Slow tabletop arm',{hz:5,lat:400}],['Teleop-style 10 Hz',{hz:10,lat:200}],['Responsive arm 30 Hz',{hz:30,lat:80}],['Humanoid 50 Hz',{hz:50,lat:40}]]};
  const rl = m.fixedTok ? [m.defRes||768] : (m.resList||[336,448,768,1024,1536]);
  return {model:[fld('res','Image resolution (px)',{opts:rl.map(r=>[r,`${r}px · ${fmtInt(vlmTok(m,r))} tokens`]),scope:'ov',num:1,wide:1}), fld('prec','Precision',{opts:precL,wide:1})],
    targets:[fld('imgs','Images per question',{num:1,int:1}), fld('out','Answer length (tokens)',{num:1,int:1}), fld('rpm','Answers per minute',{num:1}), fld('maxS','Max time per answer (s)',{num:1})],
    presets:[['Ask about a photo',{imgs:1,out:100,rpm:6,maxS:5}],['Camera event check',{imgs:1,out:40,rpm:30,maxS:2}],['Document / OCR',{imgs:1,out:400,rpm:4,maxS:20}],['Video clip, 8 frames',{imgs:8,out:150,rpm:2,maxS:15}]]};
}
function fieldHTML(f, val){
  const a = `data-k="${f.k}" data-scope="${f.scope}"${f.num?' data-num="1"':''}${f.int?' data-int="1"':''}${f.opt?' data-opt="1"':''}`;
  if(f.chk) return `<label class="fld chk${f.wide?' wide':''}"><input type="checkbox" ${a}${val?' checked':''}>${f.label}</label>`;
  if(f.opts) return `<label class="fld${f.wide?' wide':''}">${f.label}<select ${a}>${f.opts.map(([v,l])=>`<option value="${v}"${String(v)===String(val)?' selected':''}>${esc(l)}</option>`).join('')}</select></label>`;
  return `<label class="fld${f.wide?' wide':''}">${f.label}<input type="number" step="any" min="0" ${a} value="${val||''}"${f.ph?` placeholder="${f.ph}"`:''}></label>`;
}
function renderSide(){
  const item = CUR, w = wOf(item), spec = fieldSpec(item,w), mode = MODES[item.type], hw = curHW();
  const fixedN = hw && hw.fixedN && item.type!=='vla' ? hw.fixedN : 0;
  const max = hw ? (item.type==='vla' || ((item.type==='llm'||item.type==='vlm') && hw.kind!=='gpu') ? 1 : hw.max) : 1;
  const cF = [fld('pay','Buy or rent',{opts:[['buy','Buy hardware'],['cloud','Rent in cloud']],scope:'c',wide:1}), fld('budget','Max budget ($)',{num:1,opt:1,scope:'c',ph:'no limit'}), fld('maxW','Max power (W)',{num:1,opt:1,scope:'c',ph:'no limit'}), fld('hours','Use per day (h)',{num:1,scope:'c'}), fld('legacy','Include laptops, phones, niche & older hardware in suggestions',{chk:1,scope:'c',wide:1}), fld('kwh','Electricity ($/kWh)',{num:1,opt:1,scope:'c'}), fld('edge','Edge devices only',{chk:1,scope:'c',wide:1})];
  els.fitSide.innerHTML = `
    <div><p class="block-label">Model settings</p><div class="fld-grid">${spec.model.map(f=>fieldHTML(f,w[f.k])).join('')}</div></div>
    <div><p class="block-label">Your targets</p><div class="side-chips">${spec.presets.map(([l],i)=>`<button type="button" data-preset="${i}">${l}</button>`).join('')}</div><div class="fld-grid">${spec.targets.map(f=>fieldHTML(f,w[f.k])).join('')}</div></div>
    <div><p class="block-label">Your hardware</p><div class="fld-grid"><label class="fld wide">Device<select data-k="hw" data-scope="sel">${hwSelectHTML(mode,item.m)}</select></label>
      <p class="modal-note wide" style="grid-column:1/-1;margin:-4px 0 0">Not listed? <button type="button" class="chip" data-add-dev>Add your device</button></p>
      <label class="fld wide">How many<select data-k="n" data-scope="sel" data-num="1"${max===1||fixedN?' disabled':''}>${(fixedN?[fixedN]:[1,2,4,8].filter(n=>n<=max)).map(n=>`<option value="${n}"${fixedN||n===Math.min(SEL.n,max)?' selected':''}>${n}×</option>`).join('')}</select></label></div></div>
    <details><summary class="block-label" style="cursor:pointer;margin:0">Constraints &amp; costs</summary><div class="fld-grid" style="margin-top:8px">${cF.map(f=>fieldHTML(f,C[f.k])).join('')}</div></details>
    <button type="button" class="btn secondary" id="copy-summary">Copy summary</button>
    <button type="button" class="btn secondary" id="make-report">Create shareable report</button>`;
}
const METRIC_ANCHOR = {'Delay per frame':'m-delay','Frames per second':'m-fps','Memory':'m-memory','GPU memory':'m-memory','Time per image':'m-imgtime','Images per hour':'m-imghour','Speed per user':'m-tps','Time to first word':'m-ttft','Time per answer':'m-vlmtime','Answers per minute':'m-vlmrate','Reaction time':'m-vlareact','Actions per second':'m-vlarate'};
const infoBtn = id => `<button type="button" class="info" data-metric="${id}" title="What does this mean?">?</button>`;
function tileHTML(ch){
  const word = {pass:'✓ ', warn:'≈ ', fail:'✗ ', na:''}[ch.st], a = METRIC_ANCHOR[ch.k];
  const meter = ch.ratio!=null && isFinite(ch.ratio) ? `<div class="meter ${ch.st}"><span style="width:${Math.round(Math.min(1,Math.max(.03,ch.ratio))*100)}%"></span></div>` : '';
  return `<div class="tile ${ch.st}"><dt>${ch.k}${a?infoBtn(a):''}</dt><dd>${ch.v}${ch.unit?` <small>${ch.unit}</small>`:''}</dd><p class="t-target">${word}${ch.target}</p>${meter}</div>`;
}
function memHTML(mm, fits){
  const tot = Math.max(mm.usable, mm.need), pct = x => (x/tot*100).toFixed(2)+'%';
  return `<div><div class="bar">${mm.parts.map(([,v,c])=>`<span style="width:${pct(v)};background:${c}"></span>`).join('')}${fits?'':`<span style="position:absolute;top:0;bottom:0;left:${(mm.usable/tot*100).toFixed(2)}%;width:2px;background:var(--bad)"></span>`}</div>
    <div class="legend">${mm.parts.filter(p=>p[1]>0).map(([k,v,c])=>`<span><i style="background:${c}"></i>${k} ${fmtGB(v)}</span>`).join('')}<span>· ${fmtGB(mm.usable)} usable</span></div></div>`;
}
function optHTML(p){
  const r = p.r, price = C.pay==='cloud' ? '$'+r.cost.toFixed(2)+' <small>/ hour</small>' : fmtUSD(r.cost);
  const rows = r.rows.concat(r.hw.inst ? [['Instance', `${r.hw.inst.vcpu} vCPU · ${r.hw.inst.ram} GB RAM`]] : []).concat(C.pay==='cloud' ? [[`Per month (${C.hours} h/day)`, fmtUSD(r.cost*C.hours*30)]] : []);
  return `<div class="opt ${p.rec?'rec':''}"><span class="card-tab">${p.tier.toLowerCase()}</span><h4>${esc(hwLabel(r.hw,r.n))}</h4><div class="price">${price}</div>
    <dl class="facts">${rows.map(([k,v])=>`<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl><p class="w">${p.why}</p>
    ${r.stack ? `<button type="button" class="btn secondary" data-stk-use="${r.hw.id}" data-stk-n="${r.n}" style="margin-top:auto">Use in the planner</button>` : `<button type="button" class="btn secondary" data-use-hw="${r.hw.id}" data-use-n="${r.n}" style="margin-top:auto">Check this device</button>`}</div>`;
}
function tcoHTML(r){
  const hw = r.hw, n = r.n; if(hw.cloud==null || !hw.price) return '';
  const hMo = C.hours*30, cloudMo = hw.cloud*n*hMo, powerMo = hw.tdp*n*0.7*hMo/1000*C.kwh, buy = hw.price*n*1.25, save = cloudMo-powerMo, be = save>0 ? buy/save : Infinity;
  return `<div class="sec"><p class="block-label">Rent or buy · ${esc(hwLabel(hw,n))} ${infoBtn('m-tco')}</p>
    <p>At ${C.hours} h/day, renting costs about <b>${fmtUSD(cloudMo)}/month</b>. Buying costs about <b>${fmtUSD(buy)}</b> up front (incl. ~25% for the host machine) plus <b>${fmtUSD(powerMo)}/month</b> electricity. ${isFinite(be) ? `Buying pays off after about <b>${be<1?'1':be.toFixed(0)} month${be>=1.5?'s':''}</b>.` : 'At this usage, renting stays cheaper.'}</p>
    <table class="tco"><thead><tr><th>After</th><th class="num">Rent</th><th class="num">Buy + power</th></tr></thead><tbody>${[12,24,36].map(mo=>`<tr><td>${mo} months</td><td class="num">${fmtUSD(cloudMo*mo)}</td><td class="num">${fmtUSD(buy+powerMo*mo)}</td></tr>`).join('')}</tbody></table></div>`;
}

/* ============================== ACCURACY TRADE-OFFS ============================== */
const LVL = ['none','negligible','small','noticeable'];
function accInfo(item, p){
  const t = item.type, m = item.m;
  if(t==='vision') return {fp32:[0,'reference'], fp16:[1,'no measurable change in published tests'], int8: m.int8ok===false ? [3,'transformer models lose more in INT8 — keep FP16 or use quantization-aware training'] : [2,'about 1–7 mAP points with simple post-training quantization, depending on model and calibration; calibrate on your own images (QAT recovers most)']}[p];
  if(t==='gen') return {fp16:[0,'reference'], fp8:[1,'visually near-identical in most comparisons'], nf4:[2,'slight loss of fine detail and prompt adherence']}[p];
  if(t==='vla') return {fp16:[0,'reference'], fp8:[1,'about the same task success in published tests'], int8:[1,'about the same when calibrated'], int4:[2,'matched bf16 success in the OpenVLA paper but dropped in other studies — validate on your task']}[p];
  return {fp16:[0,'reference'], fp8:[1,'close to lossless'], int8:[1,'close to lossless (Q8_0: perplexity +0.01%)'], q6:[1,'close to lossless (about +0.07% perplexity)'], q5:[1,'tiny (about +0.24% perplexity)'], q4:[2,'small (about +0.9% perplexity on a 7B model); larger on models under 3B'], int4:[2,'small to moderate (AWQ / GPTQ about +1–3% perplexity); check code and math tasks']}[p];
}
function accRows(item,w,hw,n){
  const mode = MODES[item.type], precs = {vision:['fp32','fp16','int8'], gen:['fp16','fp8','nf4'], llm:Object.keys(PREC), vlm:Object.keys(PREC), vla:['fp16','fp8','int8','int4']}[item.type], out = [];
  for(const p of precs){
    const r = mode.evaluate(item.m,{...w,prec:p},hw,n); if(r.unsupported) continue;
    const [lvl,note] = accInfo(item,p) || [0,''], hl = mode.headline(r);
    const need = r.memNeed!=null ? r.memNeed : r.need;
    out.push({p, label: (item.type==='vision'?{fp32:'FP32',fp16:'FP16',int8:'INT8'}[p]:item.type==='gen'?{fp16:'FP16 / BF16',fp8:'FP8',nf4:'NF4 / GGUF Q4'}[p]:PREC[p].label), speed: r.fits===false?'—':`${hl[0]} ${hl[1]}`, mem:fmtGB(need), lvl, note, pass:!!r.pass, fits:r.fits!==false, cur:p===w.prec});
  }
  return out;
}
function accHTML(item,w,hw,cur){
  if(!cur) return '';
  const rows = accRows(item,w,hw,cur.n); if(rows.length<2) return '';
  return `<div class="sec"><p class="block-label">Accuracy trade-off · speed vs quality ${infoBtn('m-acc')}</p>
    <table class="tco acc"><thead><tr><th>Precision</th><th class="num">Speed</th><th class="num">Memory</th><th>Quality change</th><th>Your targets</th></tr></thead><tbody>${rows.map(r=>`<tr${r.cur?' class="cur"':''}><td>${esc(r.label)}${r.cur?' <small>(selected)</small>':''}</td><td class="num">${esc(r.speed)}</td><td class="num">${r.mem}</td><td><span class="lv l${r.lvl}" title="${LVL[r.lvl]}"><i style="width:${(r.lvl+1)*25}%"></i></span> <small>${esc(r.note)}</small></td><td>${r.pass?'✓ meets':r.fits?'too slow':'no memory'}</td></tr>`).join('')}</tbody></table>
    <p class="modal-note" style="margin-top:6px">Quality changes are typical published figures, not measurements of your data. Always re-check accuracy on your own test set after lowering precision.</p></div>`;
}
let lastText = '';
function renderFit(){
  const item = CUR, mode = MODES[item.type], m = item.m, w = wOf(item), hw = curHW(), out = [];
  let cur = null;
  if(hw && mode.supports(hw,m)){ cur = mode.evaluate(m,w,hw,nFor(hw,item.type)); if(cur.unsupported) cur = null; }
  if(cur){
    const st = cur.pass ? (cur.checks.some(x=>x.st==='warn')?'warn':'pass') : 'fail';
    const badge = {pass:'meets your targets', warn:'meets targets · no margin', fail:'does not meet targets'}[st];
    const l = loadOf(cur), Cc = 2*Math.PI*44, [wt,wd] = mode.why(cur,m,w);
    out.push(`<div class="verdict ${st}"><span class="card-tab">verdict · ${badge}</span>
      <div><p class="answer">${mode.answer(cur,m,w)}</p><p class="sub">${mode.sub(cur,m,w)}</p></div>
      <div class="gauge" title="Load = 1 ÷ headroom"><svg viewBox="0 0 100 100" aria-hidden="true"><circle cx="50" cy="50" r="44" fill="none" stroke="rgba(4,44,83,.09)" stroke-width="9"/><circle cx="50" cy="50" r="44" fill="none" stroke="var(--sc)" stroke-width="9" stroke-linecap="round" stroke-dasharray="${Cc}" stroke-dashoffset="${Cc*(1-Math.min(1,l))}"/></svg>
        <div><b>${cur.fits?pctTxt(l):'no fit'}</b><span>${cur.fits?'load':'memory'}</span>${cur.fits?`<span>${cur.headroom.toFixed(1)}× headroom</span>`:''}</div></div></div>`);
    out.push(`<dl class="tiles three">${cur.checks.map(tileHTML).join('')}</dl>`);
    out.push(memHTML(mode.mem(cur), cur.fits));
    out.push(accHTML(item,w,hw,cur));
    out.push(`<div class="callout"><p class="block-label">What's holding it back: ${wt} ${infoBtn('m-bottleneck')}</p><p>${wd}</p></div>`);
    if(!cur.pass) for(const [w2,m2,desc,cost] of mode.variants(m,w)){
      const r2 = mode.evaluate(m2,w2,hw,cur.n);
      if(!r2.unsupported && r2.pass){ out.push(`<div class="callout fix"><p class="block-label">Free fix — try this before buying anything ${infoBtn('m-fix')}</p><p>Your current hardware meets every target if you ${desc}. Result: ${r2.rows.slice(0,3).map(([k,v])=>`${k.toLowerCase()} <b>${v}</b>`).join(', ')}.</p><p>Trade-off: ${cost}.</p></div>`); break; }
    }
  } else if(hw){
    out.push(`<div class="verdict fail"><span class="card-tab">verdict · not supported</span><div><p class="answer"><em>Not supported.</em> ${esc(hw.name)} can't run ${esc(m.name)}.</p><p class="sub">Pick another device on the right, or use one of the options below.</p></div></div>`);
  } else {
    out.push(`<div class="verdict"><span class="card-tab">no hardware yet</span><div><p class="answer">Here's what to get for ${esc(item.name)}.</p><p class="sub">Pick your current device on the right to check it instead.</p></div></div>`);
  }
  const up = pickOptions(mode,m,w,cur,C);
  const title = cur && cur.pass ? 'Need more headroom? Step-up options' : 'Hardware that meets your targets';
  if(up.picks.length) out.push(`<div class="sec"><p class="block-label">${title} · ${C.pay==='cloud'?'cloud rental':'to buy'}${C.edge?' · edge only':''} ${infoBtn('m-tiers')}</p><div class="opts">${up.picks.map(optHTML).join('')}</div>
    <p class="modal-note" style="margin-top:10px">${C.pay==='cloud'?'Cloud instances: on-demand Linux prices in a US region (Nov 2026); GPU-marketplace entries are indicative. ':''}Approximate prices — check current listings before buying.${up.overBudget?' Nothing meets the targets within your budget; these are the cheapest that do.':''}</p></div>`);
  else if(!(cur && cur.pass)) out.push(`<div class="callout"><p class="block-label">${title}</p><p>Nothing in the database meets all targets with these settings${C.edge?' on an edge device':''}${C.maxW?` under ${C.maxW} W`:''}. Try a smaller model, lower precision or input size, or relaxed targets.</p></div>`);
  const dt = cur && cur.pass ? cur : (up.picks.find(p=>p.rec)||up.picks[0]||{}).r;
  if(dt){ const d = mode.deploy(m,w,dt);
    out.push(`<div class="sec"><p class="block-label">How to deploy it · ${esc(hwLabel(dt.hw,dt.n))}</p><p><b>${esc(d.name)}.</b> ${esc(d.why)}</p><div class="run multiline"><button type="button" class="copy-btn" data-copy>copy</button><code>${esc(d.cmd)}</code></div></div>`); }
  const tr = cur || dt;
  if(tr){ const tips = mode.tips(m,w,tr); if(tips.length) out.push(`<div class="sec"><p class="block-label">Advice</p><ul class="tips">${tips.map(([i,t,d])=>`<li><span class="ic">${i}</span><div><b>${t}.</b> ${d}</div></li>`).join('')}</ul></div>`); }
  const tc = (up.picks.find(p=>p.rec)||up.picks[0]||{}).r; if(tc) out.push(tcoHTML(tc));
  out.push(`<details class="guide"><summary>How this result is estimated</summary><div class="guide-body"><ul style="margin:0;padding-left:18px">${mode.assumptions.map(a=>`<li>${a}</li>`).join('')}<li>Real results vary roughly ±20–30%. Benchmark before buying.</li></ul><p style="margin-top:10px">Every metric and formula: <button type="button" class="chip" data-metric="metrics">Metrics explained</button></p></div></details>`);
  els.fitOut.innerHTML = out.join('');
  $('fit-note').textContent = `${TYPE_LABEL[item.type]} · ${item.task} · change anything on the right and the result updates.`;
  const L = [`ComputeFit — ${m.displayName}`];
  if(cur){ L.push(`${hwLabel(cur.hw,cur.n)} — ${cur.pass?'meets targets':'does NOT meet targets'}`); cur.checks.forEach(x=>L.push(`  ${x.k}: ${x.v}${x.unit?' '+x.unit:''} (${x.target})`)); }
  if(up.picks.length){ L.push('Options:'); up.picks.forEach(p=>L.push(`  ${p.tier}: ${hwLabel(p.r.hw,p.r.n)} — ${priceOf(p.r)} — ${p.r.rows.slice(0,2).map(x=>x.join(' ')).join(', ')}`)); }
  L.push('Estimates (±20–30%) — validate with a benchmark.'); lastText = L.join('\n');
}
function setHash(h){ try{ history.replaceState(null,'', h ? '#'+h : location.pathname+location.search); }catch(e){} }
function openFit(id){
  const item = ITEMS.find(i=>i.id===id); if(!item) return;
  CUR = item; closeDrawer();
  $('fit-title').textContent = item.name;
  renderSide(); renderFit();
  els.fitModal.hidden = false; document.body.classList.add('modal-open');
  els.fitModal.querySelector('.modal').scrollTop = 0; setHash('fit-'+id); $('fit-close').focus();
}
function closeFit(){ if(els.fitModal.hidden) return; els.fitModal.hidden = true; document.body.classList.remove('modal-open'); CUR = null; setHash(''); els.hw.value = curHW()?SEL.hw:'none'; renderGrid(); }
function gotoMetric(id){
  closeFit(); closeDrawer();
  const el = document.getElementById(id); if(!el) return;
  const d = el.closest('details'); if(d) d.open = true;
  el.scrollIntoView({block:'start'}); el.classList.add('flash'); setTimeout(()=>el.classList.remove('flash'), 2400);
}
async function copyText(btn, text, done){ const old = btn.textContent; try{ await navigator.clipboard.writeText(text); btn.textContent = done; btn.classList.add('copied'); }catch(e){ btn.textContent = 'select & copy'; } setTimeout(()=>{ btn.textContent = old; btn.classList.remove('copied'); }, 1600); }

/* ============================== ADD A MODEL (Hugging Face) ============================== */
function parseConfig(raw, totalB){
  const c = raw.text_config ? {...raw, ...raw.text_config} : raw.llm_config ? {...raw, ...raw.llm_config} : raw;
  const hidden = c.hidden_size||c.d_model||c.n_embd, layers = c.num_hidden_layers||c.n_layer||c.num_layers, heads = c.num_attention_heads||c.n_head;
  if(!hidden||!layers||!heads) throw new Error('config is missing hidden_size / num_hidden_layers / num_attention_heads');
  const kv = c.num_key_value_heads||heads, hd = c.head_dim||Math.round(hidden/heads), inter = c.intermediate_size||4*hidden, vocab = c.vocab_size||32000;
  const E = c.num_local_experts||c.n_routed_experts||c.num_experts||0, k = c.num_experts_per_tok||c.moe_topk||0;
  const moeI = c.moe_intermediate_size||inter, sh = c.n_shared_experts||(c.shared_expert_intermediate_size?1:0), shI = c.shared_expert_intermediate_size||moeI, fd = c.first_k_dense_replace||0;
  const attn = hidden*heads*hd + 2*hidden*kv*hd + heads*hd*hidden, dense = 3*hidden*inter, emb = vocab*hidden*(c.tie_word_embeddings?1:2);
  let tot, act;
  if(E&&k){ const ml = layers-fd; tot = ml*(attn+E*3*hidden*moeI+sh*3*hidden*shI)+fd*(attn+dense)+emb; act = ml*(attn+k*3*hidden*moeI+sh*3*hidden*shI)+fd*(attn+dense)+emb; }
  else tot = act = layers*(attn+dense)+emb;
  const total = totalB ? totalB*GB : tot;
  return {params:+(total/GB).toFixed(2), active:+(total*act/tot/GB).toFixed(2), layers, heads, kv, hd, kvl: c.kv_lora_rank ? c.kv_lora_rank+(c.qk_rope_head_dim||64) : null};
}
function addStatus(t, cls){ const el = $('add-status'); el.className = 'try-status '+(cls||''); el.innerHTML = t; }
function toVLM(m){ const v = 0.4; m = {...m, params:+Math.max(0.1, m.params - (m.params>1?v:0)).toFixed(2), vparams:v, pxTok:28, minTok:64, maxTok:4096, defRes:768, approx:true}; m.active = Math.min(m.active, m.params); return m; }
function addCustom(parsed, name, hf, type){
  let m = {...parsed, id:'custom-'+slug(name), name, hf};
  if(type==='vlm') m = toVLM(m);
  CUSTOM = CUSTOM.filter(c=>c.m.id!==m.id).concat([{type, m}]); saveCustom(); buildItems(); renderAll();
  els.addModal.hidden = true; openFit(slug(m.id));
}
async function importHF(){
  const id = $('add-id').value.trim().replace(/^https?:\/\/huggingface\.co\//,'').replace(/\/$/,''), type = $('add-type').value;
  if(!/^[\w.-]+\/[\w.-]+$/.test(id)){ addStatus('Enter a model id like <code>Qwen/Qwen2.5-7B-Instruct</code>.','error'); return; }
  addStatus('Fetching from Hugging Face…');
  try{
    const [cr, ar] = await Promise.all([fetch(`https://huggingface.co/${id}/resolve/main/config.json`), fetch(`https://huggingface.co/api/models/${id}`).catch(()=>null)]);
    if(!cr.ok) throw new Error(cr.status===401||cr.status===403 ? 'gated' : 'notfound');
    const cfg = await cr.json(); let total = parseFloat($('add-params').value)||null;
    try{ if(!total && ar && ar.ok){ const a = await ar.json(); if(a.safetensors && a.safetensors.total) total = a.safetensors.total/GB; } }catch(e){}
    addCustom(parseConfig(cfg,total), id, id, type);
  }catch(err){
    addStatus(err.message==='gated' ? 'This model is gated (needs a login). Copy its <code>config.json</code> from the model page and paste it below.' : err.message.startsWith('config') ? esc(err.message) : 'Couldn\'t reach Hugging Face from this page (previews block outside requests; your hosted copy can). Paste the <code>config.json</code> below instead.','error');
    $('add-paste').open = true;
  }
}
function usePaste(){
  try{ const raw = JSON.parse($('add-cfg').value), name = $('add-id').value.trim() || raw._name_or_path || 'Custom model';
    addCustom(parseConfig(raw, parseFloat($('add-params').value)||null), name, /^[\w.-]+\/[\w.-]+$/.test(name)?name:null, $('add-type').value);
  }catch(e){ addStatus('Could not read that JSON: '+esc(e.message),'error'); }
}
async function loadHub(){ // 1. your catalog, live
  try{ const r = await fetch(HUB_URL, {cache:'no-store'}); if(!r.ok) return; const d = await r.json(), ms = Array.isArray(d) ? d : d.models;
    if(Array.isArray(ms) && ms.length){ HUB = ms; hubLive = true; refresh(); } }catch(e){}
}

const refresh = () => { buildItems(); if(els.fitModal.hidden) renderAll(); };
async function loadData(){
  try{ const r = await fetch(DATA_URL, {cache:'no-store'}); if(!r.ok) return; const d = await r.json();
    const merge = (arr, extra, after) => (extra||[]).forEach(x => { if(!x || !x.id) return; const cur = arr.find(a=>a.id===x.id); if(cur) Object.assign(cur, x); else arr.push(x); if(after) after(cur||x); });
    merge(HW, d.hardware, deriveHW); merge(GEN, d.gen); merge(LLMS, d.llm, m=>{ if(m.active==null) m.active = m.params; }); merge(VLMS, d.vlm, m=>{ m.active = m.params; }); merge(VLAS, d.vla, vlaDefaults); (d.instances||[]).forEach(row=>{ const h = mkInst(row); if(h){ const i = HW.findIndex(x=>x.id===h.id); if(i>=0) HW[i]=h; else HW.push(h); deriveHW(h); } }); sortHW(); applyDevs();
    merge(CV, d.vision, m=>{ m.fam = m.fam||'custom'; m.group = m.group||'Added models'; m.task = m.task||'det'; m.base = m.base||640; m.util = m.util||1; if(m.npu==null) m.npu = true; if(m.int8ok==null) m.int8ok = true; if(m.nmsFree==null) m.nmsFree = false; });
    dataLive = true; refresh(); }catch(e){}
}
async function loadTrending(){
  try{ const c = JSON.parse(localStorage.getItem('computefit.trending')||'null'); if(c && Date.now()-c.t < HF_TTL && Array.isArray(c.items)){ LIVE = c.items; hfLive = true; refresh(); return; } }catch(e){}
  const skip = /gguf|awq|gptq|mlx|[-_](4|8)bit|onnx|lora|adapter|exl2|bnb/i, found = [];
  const knownHF = new Set([...LLMS, ...VLMS].map(m=>(m.hf||'').toLowerCase()));
  for(const [type, tag] of Object.entries(HF_TRENDING)){
    try{
      const r = await fetch(`https://huggingface.co/api/models?pipeline_tag=${tag}&sort=trendingScore&limit=30`); if(!r.ok) continue;
      const ids = (await r.json()).map(x=>x.id||x.modelId).filter(id => id && !skip.test(id) && !knownHF.has(id.toLowerCase())).slice(0, HF_LIMIT*2);
      let n = 0;
      for(const id of ids){
        if(n>=HF_LIMIT) break;
        try{
          const [cr, ar] = await Promise.all([fetch(`https://huggingface.co/${id}/resolve/main/config.json`), fetch(`https://huggingface.co/api/models/${id}`).catch(()=>null)]);
          if(!cr.ok) continue;                                   // gated or no config: skip quietly
          let total = null; try{ if(ar && ar.ok){ const a = await ar.json(); if(a.safetensors && a.safetensors.total) total = a.safetensors.total/GB; } }catch(e){}
          let m = {...parseConfig(await cr.json(), total), id:'hf-'+slug(id), name:id.split('/').pop().replace(/[-_]/g,' '), hf:id};
          if(!(m.params>=0.05)) continue;
          if(type==='vlm') m = toVLM(m);
          found.push({type, m}); n++;
        }catch(e){}
      }
    }catch(e){}
  }
  if(found.length){ LIVE = found; hfLive = true; try{ localStorage.setItem('computefit.trending', JSON.stringify({t:Date.now(), items:found})); }catch(e){} refresh(); }
}


/* ============================== ADD A DEVICE ============================== */
function closeDev(){ els.devModal.hidden = true; if(els.fitModal.hidden && els.addModal.hidden && els.drawerOverlay.hidden) document.body.classList.remove('modal-open'); }
function openDev(){ els.devModal.hidden = false; document.body.classList.add('modal-open'); renderDevList(); devPreview(); $('dev-name').focus(); }
function renderDevList(){
  $('dev-list').innerHTML = DEVS.length ? `<p class="block-label">Your devices</p><div class="chips">${DEVS.map(d=>`<button type="button" class="chip" data-rm-dev="${esc(d.id)}" title="Remove from this browser">${esc(d.name)} ✕</button>`).join('')}</div>` : '';
}
function devFromForm(forExport){
  const gv = id => parseFloat($(id).value), name = $('dev-name').value.trim(), a = ARCH[$('dev-arch').value];
  const mem = gv('dev-mem'), bw = gv('dev-bw'), fp32 = gv('dev-fp32'), o16 = gv('dev-fp16');
  const fp16 = o16>0 ? o16 : (a.m && fp32>0 ? Math.round(fp32*a.m*10)/10 : 0);
  const err = !name ? 'Give the device a name.' : !(mem>0) ? 'Enter the memory size in GB.' : !(bw>0) ? 'Enter the memory bandwidth in GB/s.' : !(fp16>0) ? (a.m ? 'Enter the FP32 TFLOPS from the spec sheet.' : 'Enter the FP16 tensor TFLOPS.') : '';
  if(err){ $('dev-preview').textContent = err; return null; }
  const lap = $('dev-lap').checked, price = Math.max(0, gv('dev-price')||0), tdp = Math.max(0, gv('dev-tdp')||0);
  const d = {id:(forExport?'':'my-')+slug(name), name, kind:'gpu', mem, bw, fp16, fp8: !o16 && a.f8 ? Math.round(fp16*a.f8) : 0, price, cloud:null, tdp, max: lap?1:2, link:'pcie'};
  if(a.amd){ d.amd = true; d.rt = 'ONNX Runtime (MIGraphX)'; } if(a.intel){ d.intel = true; d.rt = 'OpenVINO'; }
  if(lap){ d.lap = true; } if(forExport) d.g = lap ? 'Laptop GPUs' : 'Desktop GPUs';
  return d;
}
function devPreview(){
  const a = ARCH[$('dev-arch').value], fp32 = parseFloat($('dev-fp32').value), o16 = parseFloat($('dev-fp16').value);
  $('dev-fp32').disabled = !a.m; if(!a.m) $('dev-fp32').value = '';
  const fp16 = o16>0 ? o16 : (a.m && fp32>0 ? fp32*a.m : 0);
  $('dev-preview').textContent = fp16>0 ? `Estimated FP16 tensor speed ≈ ${fmtN(fp16)} TFLOPS${!o16 && a.f8 ? ` · FP8 ≈ ${fmtN(fp16*a.f8)}` : ''} (FP32 × ${a.m}${a.m===8?' on RTX 20':''}). Overwrite it if the spec sheet lists a different number.` : 'FP16 speed is derived from FP32 TFLOPS and the architecture, or enter it yourself.';
}
function saveDevice(){
  const d = devFromForm(false); if(!d) return;
  DEVS = DEVS.filter(x=>x.id!==d.id).concat(d); saveDevs(); applyDevs();
  SEL.hw = d.id; SEL.n = 1; renderDevList();
  $('dev-preview').textContent = `Saved “${d.name}” — it is selected as your hardware.`;
  renderAll(); if(!els.fitModal.hidden){ renderSide(); renderFit(); }
  setTimeout(closeDev, 700);
}


const RP_CSS = `
.rp{max-width:840px;margin:0 auto;padding:28px;color:#042C53;font-family:Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-size:14px;line-height:1.5}
.rp h1{font-size:24px;font-weight:600;margin:0}.rp h2{font-size:15px;font-weight:600;margin:22px 0 8px;padding-bottom:4px;border-bottom:2px solid #185FA5;color:#185FA5}
.rp p{margin:6px 0}.rp small,.rp .muted{color:#55677E}
.rp-head{display:flex;align-items:center;gap:14px;padding-bottom:14px;border-bottom:3px solid #042C53}.rp-head svg{width:52px;height:52px;flex:none}.rp-head .meta{margin-left:auto;text-align:right;font-size:12px;color:#55677E}
.rp-verdict{border:2px solid #042C53;border-radius:10px;padding:12px 16px;margin-top:16px}.rp-verdict.pass{border-color:#0C8567;background:#EEF8F4}.rp-verdict.warn{border-color:#B7791F;background:#FBF5E8}.rp-verdict.fail{border-color:#B3402B;background:#FBEFEC}
.rp-verdict .a{font-size:16px;margin:0}.rp-verdict em{font-style:normal;font-weight:700}.rp-verdict .s{font-size:12px;color:#55677E;margin:4px 0 0}
.rp table{width:100%;border-collapse:collapse;font-size:13px}.rp th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#55677E;padding:5px 8px;border-bottom:1px solid #DCE4EE}.rp td{padding:6px 8px;border-bottom:1px solid #E8EDF4;vertical-align:top}.rp .num{text-align:right;font-variant-numeric:tabular-nums}
.rp pre{background:#F2F5F9;border:1px solid #DCE4EE;border-radius:8px;padding:10px 12px;font-size:11.5px;white-space:pre-wrap;word-break:break-word;font-family:"JetBrains Mono",ui-monospace,monospace}
.rp ul{margin:6px 0;padding-left:18px}.rp li{margin:3px 0}
.rp .ok{color:#0C8567;font-weight:600}.rp .no{color:#B3402B;font-weight:600}.rp .mid{color:#B7791F;font-weight:600}
.rp .lv{display:inline-block;width:44px;height:7px;border-radius:4px;background:#E1E8F0;vertical-align:middle;overflow:hidden;margin-right:6px}.rp .lv i{display:block;height:100%;background:#0C8567}.rp .lv.l2 i{background:#C98A12}.rp .lv.l3 i{background:#B3402B}
.rp .ver{font-size:12px;margin:10px 0 0;padding:6px 10px;border-radius:6px}.rp .ver.good{background:#EEF8F4;color:#0C8567}.rp .ver.bad{background:#FBEFEC;color:#B3402B}
.rp-sign{display:flex;gap:14px;align-items:center;margin-top:26px;padding:14px 16px;border:2px solid #042C53;border-radius:10px;page-break-inside:avoid}.rp-sign svg{width:56px;height:56px;flex:none}.rp-sign .t{flex:1}.rp-sign .id{text-align:right;font-size:12px;color:#55677E}.rp-sign code{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:11px;color:#042C53}
.rp-foot{font-size:11px;color:#55677E;margin-top:10px}
h2,table,pre{page-break-inside:avoid}
@media print{.rp{padding:0}}
`;

/* ============================== STACK PLANNER ============================== */
const STACK = {rows:[], hw:'__init', n:1};
const STK_PRESETS = [
  ['Smart retail camera', [['yolo26-m',{streams:4,fps:15}],['osnet-x1',{streams:4,fps:5}],['qwen25vl-3b',{rpm:6}]]],
  ['Warehouse robot', [['yolo26-s',{}],['dav2s',{}],['smolvla',{}]]],
  ['Document pipeline', [['ppocr-s',{streams:1,fps:2,lat:500}],['qwen25vl-3b',{rpm:4,maxS:20}]]],
  ['Edge assistant', [['llama32-3b',{}],['yolo26-n',{}],['smolvlm2-2b',{}]]],
];
const STK_COLORS = ['#042C53','#378ADD','#0C8567','#5DCAA5','#C98A12','#B3402B','#7A5AA6','#B9C6D6'];
const stkItem = row => ITEMS.find(i=>i.id===row.id);
const stkList = () => STACK.rows.map(r=>{ const item = stkItem(r); return item ? {item, w:{...wOf(item), ...r.ov}} : null; }).filter(Boolean);
const stkHW = () => HW.find(h=>h.id===STACK.hw) || null;
const nUse = x => x.memNeed!=null ? x.memNeed : x.need;
const uUse = x => x.memUsable!=null ? x.memUsable : x.usable;
function stackEval(hw, N, list){
  N = Math.max(1, N|0);
  const sh = (hw.kind==='gpu'?0.6:0.3)*GB, usable1 = hw.mem*GB*usableFrac(hw), big = {...hw, mem:hw.mem*1000};
  const items = list.map(({item,w})=>{
    const mode = MODES[item.type];
    if(!mode.supports(hw,item.m)) return {item,w,bad:`${hw.name} can't run ${item.name}.`};
    let r = mode.evaluate(item.m,w,hw,1); if(r.unsupported) return {item,w,bad:r.unsupported};
    let k = 1;
    if(nUse(r)>uUse(r) && N>1 && hw.kind==='gpu' && item.type!=='vla'){
      for(const kk of [2,4,8]){ if(kk>N || kk>hw.max) break; const r2 = mode.evaluate(item.m,w,hw,kk); if(!r2.unsupported && nUse(r2)<=uUse(r2)){ r = r2; k = kk; break; } }
    }
    const fits = nUse(r)<=uUse(r), rc = mode.evaluate(item.m,w,big,k);
    return {item,w,r,k,fits,need:nUse(r),solo:!!r.pass && fits,hl:mode.headline(r),load: fits ? 1/Math.max(rc.headroom,0.01) : 9.99};
  });
  const gpus = Array.from({length:N},()=>({mem:0,load:0,n:0}));
  let ok = items.every(i=>!i.bad && i.fits);
  if(ok){
    for(const it of items.slice().sort((a,b)=>b.need-a.need)){
      const per = it.need/it.k, cand = gpus.map((g,i)=>i).sort((a,b)=>gpus[a].load-gpus[b].load || gpus[a].mem-gpus[b].mem), chosen = [];
      for(const gi of cand){ const g = gpus[gi], add = per - (g.n>0 ? Math.min(sh, per*0.5) : 0); if(g.mem+add<=usable1){ chosen.push([gi,add]); if(chosen.length===it.k) break; } }
      if(chosen.length<it.k){ it.nofit = true; ok = false; continue; }
      chosen.forEach(([gi,add])=>{ const g = gpus[gi]; g.mem += add; g.load += it.load; g.n++; });
      it.gpus = chosen.map(c=>c[0]+1);
    }
  }
  const memUse = gpus.reduce((a,g)=>a+g.mem,0), memCap = usable1*N, maxLoad = Math.max(...gpus.map(g=>g.load)), allSolo = items.every(i=>i.solo);
  const headroom = ok ? 1/Math.max(maxLoad,0.01) : 0, pass = ok && maxLoad<=1 && allSolo;
  return {hw,n:N,stack:true,items,gpus,ok,pass,allSolo,memUse,memCap,maxLoad,headroom,score:headroom,fits:ok,
    rows:[['Models', String(items.length)],['Memory used', Math.round(memUse/memCap*100)+'%'],['Combined load', ok?Math.round(maxLoad*100)+'%':'—'],['Headroom', ok?headroom.toFixed(1)+'×':'—']]};
}
function stackOptions(list,cur,c){
  const all = [];
  for(const hw of HW){
    if(c.edge && hw.g!=='Edge devices') continue;
    if((hw.lap||hw.legacy||hw.phone) && !c.legacy && !hw.mine) continue;
    if(hw.inst && c.pay!=='cloud') continue;
    const ns = hw.fixedN ? [hw.fixedN] : hw.kind==='gpu' ? [1,2,4].filter(n=>n<=hw.max) : [1];
    for(const n of ns){
      const cost = costOf(hw,n,c); if(cost==null || (c.pay==='buy' && !cost)) continue;
      if(c.maxW && hw.tdp*n>c.maxW) continue;
      const r = stackEval(hw,n,list); r.cost = cost; all.push(r);
    }
  }
  return tierPicks(preferInst(all,c),cur,c);
}
function stkNSelect(){
  const hw = stkHW(), max = hw ? (hw.fixedN ? 1 : hw.kind==='gpu' ? Math.min(hw.max,4) : 1) : 1;
  const opts = hw && hw.fixedN ? [hw.fixedN] : [1,2,4].filter(n=>n<=max);
  if(!opts.includes(STACK.n)) STACK.n = opts[0];
  $('stk-n').innerHTML = opts.map(n=>`<option value="${n}"${n===STACK.n?' selected':''}>${n}×</option>`).join(''); $('stk-n').disabled = opts.length<2;
}
function fillStack(){
  if(!$('stk-hw')) return;
  if(STACK.hw==='__init') STACK.hw = SEL.hw;
  const groups = {}; ITEMS.forEach(i=>(groups[i.type] = groups[i.type]||[]).push(i));
  const sel = $('stk-add-sel').value;
  $('stk-add-sel').innerHTML = TYPES.map(([t,l])=>groups[t] ? `<optgroup label="${esc(l)}">${groups[t].map(i=>`<option value="${esc(i.id)}">${esc(i.name)}</option>`).join('')}</optgroup>` : '').join('');
  if(sel) $('stk-add-sel').value = sel;
  $('stk-hw').innerHTML = hwSelectHTML(null,null,STACK.hw); $('stk-hw').value = stkHW() ? STACK.hw : 'none';
  $('stk-presets').innerHTML = STK_PRESETS.map(([l],i)=>`<button type="button" data-stk-preset="${i}">${esc(l)}</button>`).join('');
  stkNSelect(); renderStackRows(); renderStackOut();
}
function renderStackRows(){
  $('stk-rows').innerHTML = STACK.rows.length ? STACK.rows.map((row,i)=>{
    const item = stkItem(row); if(!item) return '';
    const w = {...wOf(item), ...row.ov}, tf = fieldSpec(item,wOf(item)).targets;
    return `<div class="stk-row"><div class="stk-row-h"><b>${esc(item.name)}</b><span class="chip">${esc(TYPE_LABEL[item.type])}</span><button type="button" class="icon-btn" data-stk-rm="${i}" aria-label="Remove ${esc(item.name)}"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg></button></div>
      <div class="fld-grid">${tf.map(f=>`<label class="fld">${f.label}<input type="number" step="any" min="0" data-stk-i="${i}" data-stk-k="${f.k}" value="${w[f.k]}"></label>`).join('')}</div></div>`; }).join('')
    : '<p class="modal-note">Pick an example above, or add models below.</p>';
}
function stackVerdict(cur,hw){
  const k = cur.items.length, bad = cur.items.filter(i=>i.bad), nf = cur.items.filter(i=>i.nofit||(!i.fits&&!i.bad)), slow = cur.items.filter(i=>!i.solo && i.fits && !i.bad);
  const st = cur.pass ? (cur.maxLoad>0.85?'warn':'pass') : 'fail', hwn = hwLabel(hw,cur.n);
  let a, sub = `${k} model${k>1?'s':''} · ${cur.n>1?cur.n+' devices · ':''}memory ${fmtGB(cur.memUse)} of ${fmtGB(cur.memCap)}${cur.ok?` · combined load ${Math.round(cur.maxLoad*100)}%`:''}`;
  if(cur.pass) a = `<em>Yes.</em> Your ${esc(hwn)} carries all ${k} models at about <b>${Math.round(cur.maxLoad*100)}%</b> combined load${st==='warn'?' — tight, with little room for peaks':''}.`;
  else if(bad.length) a = `<em>No.</em> ${esc(bad[0].bad)}`;
  else if(nf.length) a = `<em>No.</em> ${esc(nf.map(i=>i.item.name).join(', '))} ${nf.length>1?'do':'does'} not fit in the memory of your ${esc(hwn)} together with the rest.`;
  else if(slow.length) a = `<em>No.</em> ${esc(slow.map(i=>i.item.name).join(', '))} can't meet ${slow.length>1?'their':'its'} own targets on your ${esc(hwn)} even alone.`;
  else a = `<em>No.</em> The models together need ${Math.round(cur.maxLoad*100)}% of your ${esc(hwn)} — over its capacity.`;
  return {st, a, sub};
}
function stackTable(cur){
  return `<table class="stk-table"><thead><tr><th>Model</th><th>Alone</th><th class="num">Load</th><th class="num">Memory</th>${cur.n>1?'<th>Device #</th>':''}</tr></thead><tbody>${cur.items.map(it=>it.bad ? `<tr><td>${esc(it.item.name)}</td><td colspan="${cur.n>1?4:3}">not supported</td></tr>` : `<tr><td>${esc(it.item.name)}</td><td>${it.fits?`${esc(it.hl[0])} ${esc(it.hl[1])} ${it.solo?'✓':'✗'}`:'no memory'}</td><td class="num">${it.fits?Math.round(it.load*100)+'%':'—'}</td><td class="num">${fmtGB(it.need)}</td>${cur.n>1?`<td>${it.gpus?it.gpus.join(', '):'—'}</td>`:''}</tr>`).join('')}</tbody></table>`;
}
function renderStackOut(){
  const list = stkList(), hw = stkHW(), out = [];
  if(!list.length){ $('stk-out').innerHTML = '<div class="callout"><p class="block-label">Nothing to size yet</p><p>Add models on the left. The planner combines their memory and load on one device and suggests hardware if it does not fit.</p></div>'; return; }
  let cur = null;
  if(hw){ cur = stackEval(hw, hw.fixedN || STACK.n, list); }
  if(cur){
    const v = stackVerdict(cur,hw);
    out.push(`<div class="verdict ${v.st}"><span class="card-tab">stack verdict · ${v.st==='pass'?'fits':v.st==='warn'?'fits · tight':'does not fit'}</span><div><p class="answer">${v.a}</p><p class="sub">${v.sub}</p></div></div>`);
    const soloN = cur.items.filter(i=>i.solo).length;
    out.push(`<dl class="tiles three">${tileHTML({k:'Memory', v:fmtGB(cur.memUse), ratio:cur.memUse/cur.memCap, target:`of ${fmtGB(cur.memCap)} available`, st:cur.ok?status(true,cur.memUse>cur.memCap*0.9):'fail'})}${tileHTML({k:'Combined load', v:cur.ok?Math.round(cur.maxLoad*100)+'%':'—', ratio:cur.ok?cur.maxLoad:null, target:'need ≤ 100%', st:cur.ok?status(cur.maxLoad<=1,cur.maxLoad>0.85):'na'})}${tileHTML({k:'Models meeting targets alone', v:`${soloN} / ${cur.items.length}`, ratio:soloN/cur.items.length, target:'need all', st:soloN===cur.items.length?'pass':'fail'})}</dl>`);
    out.push(stackTable(cur));
    out.push(memHTML({need:cur.memUse, usable:cur.memCap, parts:cur.items.filter(i=>!i.bad).map((i,x)=>[i.item.name,i.need,STK_COLORS[x%STK_COLORS.length]])}, cur.ok));
    out.push('<p class="modal-note">Sharing a device adds waiting time between models. A model with a tight latency target may need its own GPU, or a higher priority stream.</p>');
  } else out.push('<div class="callout"><p class="block-label">No device selected</p><p>Pick a device on the left to check it, or see the hardware options below.</p></div>');
  const up = stackOptions(list,cur,C);
  out.push(up.picks.length ? `<div class="sec"><p class="block-label">${cur&&cur.pass?'Need more headroom? Step-up options':'Hardware that carries the whole stack'} · ${C.pay==='cloud'?'cloud rental':'to buy'}${C.edge?' · edge only':''}</p><div class="opts">${up.picks.map(optHTML).join('')}</div><p class="modal-note" style="margin-top:10px">Approximate prices. Rental options use on-demand cloud instances. Change buy / rent, budget and power limits under Constraints in any model's check.</p></div>`
    : '<div class="callout"><p class="block-label">Hardware options</p><p>Nothing in the database carries this whole stack under the current constraints. Remove a model, relax a target or allow rental.</p></div>');
  out.push('<div style="display:flex;gap:8px;flex-wrap:wrap"><button type="button" class="btn primary" id="stk-report">Create shareable report</button></div>');
  $('stk-out').innerHTML = out.join('');
}

/* ============================== REPORT ============================== */
const RP = {kind:'single', item:null, link:'', id:'', hash:'', date:'', verified:null};
const b64e = o => btoa(unescape(encodeURIComponent(JSON.stringify(o)))).replace(/\+/g,'-').replace(/\//g,'_').replace(/=+$/,'');
const b64d = t => { t = t.replace(/-/g,'+').replace(/_/g,'/'); while(t.length%4) t += '='; return JSON.parse(decodeURIComponent(escape(atob(t)))); };
async function sha(str){
  try{ const b = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(str)); return [...new Uint8Array(b)].map(x=>x.toString(16).padStart(2,'0')).join(''); }
  catch(e){ let h1=0xdeadbeef, h2=0x41c6ce57; for(let i=0;i<str.length;i++){ const c = str.charCodeAt(i); h1 = Math.imul(h1^c,2654435761); h2 = Math.imul(h2^c,1597334677); }
    h1 = Math.imul(h1^(h1>>>16),2246822507)^Math.imul(h2^(h2>>>13),3266489909); h2 = Math.imul(h2^(h2>>>16),2246822507)^Math.imul(h1^(h1>>>13),3266489909);
    const x = (h2>>>0).toString(16).padStart(8,'0')+(h1>>>0).toString(16).padStart(8,'0'); return (x+x+x+x).slice(0,64); }
}
function reportState(){
  const st = RP.kind==='stack';
  const base = {v:1, d:RP.date, wl:WL, c:C, hw: st?STACK.hw:SEL.hw, n: st?STACK.n:SEL.n, cl:$('rp-client').value.trim(), pr:$('rp-project').value.trim()};
  return st ? {...base, k:'t', r:STACK.rows} : {...base, k:'s', i:RP.item, o:OV[RP.item]||{}};
}
function applyState(st){
  Object.keys(st.wl||{}).forEach(k => Object.assign(WL[k] = WL[k]||{}, st.wl[k]));
  Object.assign(C, st.c||{});
  if(st.k==='t'){ STACK.rows = st.r||[]; STACK.hw = st.hw; STACK.n = st.n||1; }
  else { SEL.hw = st.hw; SEL.n = st.n||1; if(st.o && st.i) OV[st.i] = st.o; }
  RP.date = st.d || ''; $('rp-client').value = st.cl||''; $('rp-project').value = st.pr||'';
}
const rpStat = {pass:'<span class="ok">✓ meets</span>', warn:'<span class="mid">≈ tight</span>', fail:'<span class="no">✗ misses</span>', na:'—'};
const rpTable = (head, rows) => `<table><thead><tr>${head.map(h=>`<th${h[0]==='#'?' class="num"':''}>${esc(h.replace(/^#/,''))}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map((c,i)=>`<td${head[i][0]==='#'?' class="num"':''}>${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
function rpOptions(up){
  if(!up.picks.length) return '<p>No device in the database meets all targets with these settings.</p>';
  return rpTable(['Option','Device',"#Price",'Key results'], up.picks.map(p=>[esc(p.tier), esc(hwLabel(p.r.hw,p.r.n)), C.pay==='cloud'?'$'+p.r.cost.toFixed(2)+'/h':fmtUSD(p.r.cost), p.r.rows.slice(0,3).map(x=>esc(x[0])+': '+esc(x[1])).join(' · ')]))
    + `<p class="muted">${C.pay==='cloud'?'Cloud prices are on-demand, US region. ':''}Prices are approximate; check current listings.</p>`;
}
function rpSingle(){
  const item = ITEMS.find(i=>i.id===RP.item);
  if(!item) return '<p>This model is stored only in the browser that created the report (custom or trending model), so it cannot be rebuilt here.</p>';
  const mode = MODES[item.type], m = item.m, w = wOf(item), hw = curHW(), N = nFor(hw,item.type), spec = fieldSpec(item,w);
  let cur = null; if(hw && mode.supports(hw,m)){ cur = mode.evaluate(m,w,hw,N); if(cur.unsupported) cur = null; }
  const up = pickOptions(mode,m,w,cur,C), fv = f => { const v = w[f.k]; if(f.opts){ const o = f.opts.find(x=>String(x[0])===String(v)); return o?o[1]:v; } return f.chk ? (v?'yes':'no') : v; };
  const o = [];
  if(cur){
    const st = cur.pass ? (cur.checks.some(x=>x.st==='warn')?'warn':'pass') : 'fail';
    o.push(`<div class="rp-verdict ${st}"><p class="a">${mode.answer(cur,m,w)}</p><p class="s">${mode.sub(cur,m,w)}</p></div>`);
  } else o.push(`<div class="rp-verdict"><p class="a">${hw?`<em>Not supported.</em> ${esc(hw.name)} can't run ${esc(m.name)}.`:'No hardware selected — see the options below.'}</p></div>`);
  o.push('<h2>Requirements</h2>'+rpTable(['Setting','Value'], spec.model.concat(spec.targets).map(f=>[esc(f.label), esc(String(fv(f)))])));
  if(cur){
    o.push(`<h2>Result on ${esc(hwLabel(cur.hw,cur.n))}</h2>`+rpTable(['Metric','#Result','Requirement','Status'], cur.checks.map(c=>[esc(c.k), esc(c.v+(c.unit?' '+c.unit:'')), esc(c.target), rpStat[c.st]||'—'])));
    const mm = mode.mem(cur); o.push('<h2>Memory</h2>'+rpTable(['Part','#Size'], mm.parts.filter(x=>x[1]>0).map(x=>[esc(x[0]), fmtGB(x[1])]).concat([['<b>Total / available</b>', `<b>${fmtGB(mm.need)} / ${fmtGB(mm.usable)}</b>`]])));
    const [wt,wd] = mode.why(cur,m,w); o.push(`<h2>What limits it</h2><p><b>${esc(wt)}.</b> ${wd}</p>`);
    const ar = accRows(item,w,hw,cur.n); if(ar.length>1) o.push('<h2>Accuracy trade-off</h2>'+rpTable(['Precision','#Speed','#Memory','Quality change','Targets'], ar.map(r=>[esc(r.label)+(r.cur?' (selected)':''), esc(r.speed), r.mem, `<span class="lv l${r.lvl}"><i style="width:${(r.lvl+1)*25}%"></i></span>${esc(r.note)}`, r.pass?'<span class="ok">✓</span>':r.fits?'<span class="no">too slow</span>':'<span class="no">no memory</span>']))+'<p class="muted">Typical published quality changes, not measurements on your data.</p>');
  }
  o.push(`<h2>${cur&&cur.pass?'Step-up options':'Hardware that meets the targets'} (${C.pay==='cloud'?'cloud rental':'to buy'})</h2>`+rpOptions(up));
  const dt = cur && cur.pass ? cur : (up.picks.find(p=>p.rec)||up.picks[0]||{}).r;
  if(dt){ const d = mode.deploy(m,w,dt); o.push(`<h2>Deployment · ${esc(hwLabel(dt.hw,dt.n))}</h2><p><b>${esc(d.name)}.</b> ${esc(d.why)}</p><pre>${esc(d.cmd)}</pre>`); }
  const tr = cur || dt; if(tr){ const tips = mode.tips(m,w,tr).slice(0,4); if(tips.length) o.push('<h2>Advice</h2><ul>'+tips.map(t=>`<li><b>${t[1]}.</b> ${t[2]}</li>`).join('')+'</ul>'); }
  o.push('<h2>Assumptions</h2><ul>'+mode.assumptions.map(a=>`<li>${a}</li>`).join('')+'</ul>');
  return {title:`Hardware fit report — ${m.displayName}`, html:o.join('')};
}
function rpStack(){
  const list = stkList(), hw = stkHW(), o = [];
  if(!list.length) return {title:'Hardware fit report — stack', html:'<p>The stack is empty.</p>'};
  let cur = hw ? stackEval(hw, hw.fixedN||STACK.n, list) : null;
  if(cur){ const v = stackVerdict(cur,hw); o.push(`<div class="rp-verdict ${v.st}"><p class="a">${v.a}</p><p class="s">${v.sub}</p></div>`); }
  else o.push('<div class="rp-verdict"><p class="a">No device selected — see the hardware options below.</p></div>');
  o.push('<h2>Models in the stack</h2>'+rpTable(['Model','Type','Targets'], list.map(({item,w})=>[esc(item.name), esc(TYPE_LABEL[item.type]), fieldSpec(item,w).targets.map(f=>esc(f.label)+': '+esc(String(w[f.k]))).join(' · ')])));
  if(cur){
    const multi = cur.n>1, rowsT = cur.items.map(it=>{
      const dev = multi ? [it.gpus ? it.gpus.join(', ') : '—'] : [];
      if(it.bad) return [esc(it.item.name),'not supported','—','—'].concat(multi?['—']:[]);
      const alone = it.fits ? esc(it.hl[0]+' '+it.hl[1])+(it.solo?' <span class="ok">✓</span>':' <span class="no">✗</span>') : 'no memory';
      return [esc(it.item.name), alone, it.fits?Math.round(it.load*100)+'%':'—', fmtGB(it.need)].concat(dev);
    });
    rowsT.push(['<b>Total</b>','', cur.ok?'<b>'+Math.round(cur.maxLoad*100)+'%</b>':'—', '<b>'+fmtGB(cur.memUse)+' / '+fmtGB(cur.memCap)+'</b>'].concat(multi?['']:[]));
    o.push('<h2>Result on '+esc(hwLabel(cur.hw,cur.n))+'</h2>'+rpTable(['Model','Alone','#Load','#Memory'].concat(multi?['Device #']:[]), rowsT));
  }
  const up = stackOptions(list,cur,C);
  o.push(`<h2>${cur&&cur.pass?'Step-up options':'Hardware that carries the whole stack'} (${C.pay==='cloud'?'cloud rental':'to buy'})</h2>`+rpOptions(up));
  o.push('<h2>Assumptions</h2><ul><li>Each model is evaluated alone on the device; combined load is the sum of the shares of capacity they need (1 ÷ headroom). Memory adds each model\'s weights and buffers, counting the shared runtime once per device.</li><li>With several devices, models are spread across them; a model that does not fit one device is split over 2–8 devices.</li><li>Sharing adds waiting time between models: latency-critical models may need their own device.</li></ul>');
  return {title:'Hardware fit report — model stack', html:o.join('')};
}
async function renderReport(){
  const st = reportState(), b = b64e(st), full = await sha(b);
  RP.hash = full; RP.id = 'CF-'+full.slice(0,8).toUpperCase(); RP.link = location.href.split('#')[0]+'#report='+b+'.'+full.slice(0,12);
  const rep = RP.kind==='stack' ? rpStack() : rpSingle(), body = typeof rep==='string' ? {title:'Hardware fit report', html:rep} : rep;
  RP.title = body.title;
  const mark = (document.querySelector('.brand-mark')||{outerHTML:''}).outerHTML.replace('class="brand-mark"','');
  const ver = RP.verified===true ? '<p class="ver good">✓ Integrity check passed: the inputs in this report match the hash in the link that opened it.</p>' : RP.verified===false ? '<p class="ver bad">⚠ Integrity check failed: this link was changed after it was created. Treat the numbers with caution.</p>' : '';
  $('rp-body').innerHTML = `<div class="rp-head">${mark}<div><h1>${esc(body.title.replace('Hardware fit report — ','').replace('Hardware fit report','Hardware fit report'))}</h1><div class="muted">Hardware fit report</div></div>
    <div class="meta">${RP.date?esc(RP.date)+'<br>':''}${st.cl?'Prepared for '+esc(st.cl)+'<br>':''}${st.pr?'Project: '+esc(st.pr)+'<br>':''}Report ${RP.id}</div></div>${ver}${body.html}
    <div class="rp-sign">${mark}<div class="t"><b>Computer Vision AI Hub</b><br><span class="muted">Hardware fit analysis prepared with ComputeFit</span><br><span class="muted">${esc(HUB_SITE.replace(/^https?:[/][/]/,'').replace(/[/]$/,''))}</span></div>
      <div class="id">Report ID <b>${RP.id}</b><br>Integrity hash<br><code>${full.slice(0,24)}…</code></div></div>
    <p class="rp-foot">Estimates carry roughly ±20–30% uncertainty; validate with a benchmark on your own data before buying hardware. Prices are approximate. The hash is computed from the report inputs and lets anyone opening the share link detect edits; it is a checksum, not a certificate-based digital signature.</p>`;
}
function openReport(kind, itemId){
  RP.kind = kind; RP.item = itemId || (CUR && CUR.id) || null; if(!RP.date) RP.date = new Date().toISOString().slice(0,10);
  els.rpModal.hidden = false; document.body.classList.add('modal-open'); renderReport();
}
function closeReport(){ els.rpModal.hidden = true; if(els.fitModal.hidden && els.addModal.hidden && els.devModal.hidden && els.drawerOverlay.hidden) document.body.classList.remove('modal-open'); }
function rpPrint(){ document.body.classList.add('printing-report'); window.print(); setTimeout(()=>document.body.classList.remove('printing-report'),800); }
function rpDownload(){
  const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${esc(RP.title||'Report')} · ${RP.id}</title><style>${RP_CSS}</style></head><body><div class="rp">${$('rp-body').innerHTML}</div></body></html>`;
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([html],{type:'text/html'})); a.download = `${RP.id}-hardware-report.html`; document.body.appendChild(a); a.click(); setTimeout(()=>{ URL.revokeObjectURL(a.href); a.remove(); },500);
}
window.addEventListener('afterprint',()=>document.body.classList.remove('printing-report'));

/* ============================== WIRING ============================== */
function init(){
  Object.assign(els, {rpModal:$('report-modal'), devModal:$('dev-modal'), grid:$('grid'), state:$('state'), count:$('count'), filters:$('type-filters'), hw:$('hw'), drawer:$('drawer'), drawerOverlay:$('drawer-overlay'), fitModal:$('fit-modal'), fitOut:$('fit-out'), fitSide:$('fit-side'), addModal:$('add-modal')});
  { const st = document.createElement('style'); st.textContent = RP_CSS; document.head.appendChild(st); }
  loadCustom(); loadDevs(); applyDevs(); buildItems(); renderAll();
  $('dev-arch').innerHTML = Object.entries(ARCH).map(([k,a])=>`<option value="${k}">${esc(a.l)}</option>`).join('');
  $('dev-arch').value = 'ada'; devPreview();
  $('search').addEventListener('input', e=>{ F.q = e.target.value; renderGrid(); });
  $('sort').addEventListener('change', e=>{ F.sort = e.target.value; renderGrid(); });
  els.hw.addEventListener('change', e=>{ SEL.hw = e.target.value; SEL.n = 1; renderGrid(); });
  els.filters.addEventListener('click', e=>{ const b = e.target.closest('button'); if(!b) return; if('fits' in b.dataset) F.fits = !F.fits; else F.type = b.dataset.type; renderPills(); renderGrid(); });
  document.addEventListener('click', e=>{
    const t = e.target, q = s => t.closest(s);
    let b;
    if((b = q('[data-fit]'))){ e.preventDefault(); openFit(b.dataset.fit); }
    else if((b = q('[data-info]'))){ const it = ITEMS.find(i=>i.id===b.dataset.info); if(it) openDrawer(it); }
    else if((b = q('[data-pick-hw]'))){ SEL.hw = b.dataset.pickHw; SEL.n = 1; const id = els.drawer.querySelector('[data-fit]').dataset.fit; renderGrid(); openFit(id); }
    else if((b = q('[data-remove]'))){ CUSTOM = CUSTOM.filter(c=>slug(c.m.id)!==b.dataset.remove); saveCustom(); closeDrawer(); buildItems(); renderAll(); }
    else if(q('[data-drawer-close]') || t===els.drawerOverlay) closeDrawer();
    else if(q('#fit-close') || t===els.fitModal) closeFit();
    else if(q('#add-close') || t===els.addModal){ els.addModal.hidden = true; document.body.classList.remove('modal-open'); }
    else if(q('#make-report')){ openReport('single', CUR && CUR.id); }
    else if(q('#stk-report')){ openReport('stack'); }
    else if(q('#rp-close') || t===els.rpModal){ closeReport(); }
    else if(q('#rp-print')){ rpPrint(); }
    else if((b = q('#rp-link'))){ copyText(b, RP.link, 'copied ✓'); }
    else if(q('#rp-dl')){ rpDownload(); }
    else if(q('#stk-add')){ const id = $('stk-add-sel').value; if(id){ STACK.rows.push({id, ov:{}}); renderStackRows(); renderStackOut(); } }
    else if((b = q('[data-stk-rm]'))){ STACK.rows.splice(+b.dataset.stkRm,1); renderStackRows(); renderStackOut(); }
    else if((b = q('[data-stk-preset]'))){ const pr = STK_PRESETS[+b.dataset.stkPreset]; STACK.rows = pr[1].filter(([id])=>ITEMS.some(i=>i.id===id)).map(([id,ov])=>({id,ov:{...ov}})); renderStackRows(); renderStackOut(); }
    else if((b = q('[data-stk-use]'))){ STACK.hw = b.dataset.stkUse; STACK.n = +b.dataset.stkN; $('stk-hw').innerHTML = hwSelectHTML(null,null,STACK.hw); stkNSelect(); renderStackOut(); $('stack').scrollIntoView({block:'start'}); }
    else if(q('#nav-add-dev') || q('[data-add-dev]')){ openDev(); }
    else if(q('#dev-close') || t===els.devModal){ closeDev(); }
    else if((b = q('[data-rm-dev]'))){ DEVS = DEVS.filter(d=>d.id!==b.dataset.rmDev); saveDevs(); if(SEL.hw===b.dataset.rmDev) SEL.hw = 'none'; applyDevs(); renderDevList(); renderAll(); if(!els.fitModal.hidden){ renderSide(); renderFit(); } }
    else if(q('#dev-copy')){ const d = devFromForm(true); if(d) copyText(q('#dev-copy'), JSON.stringify({hardware:[d]}, null, 2), 'copied ✓'); }
    else if(q('#nav-add')){ els.addModal.hidden = false; document.body.classList.add('modal-open'); addStatus(''); $('add-id').focus(); }
    else if((b = q('[data-metric]'))) gotoMetric(b.dataset.metric);
    else if((b = q('[data-copy]'))) copyText(b, b.parentElement.querySelector('code').textContent, 'copied ✓');
    else if((b = q('#copy-summary'))) copyText(b, lastText, 'Copied ✓');
    else if((b = q('[data-use-hw]'))){ SEL.hw = b.dataset.useHw; SEL.n = +b.dataset.useN; renderSide(); renderFit(); els.fitModal.querySelector('.modal').scrollTop = 0; }
    else if((b = q('[data-preset]')) && CUR){ Object.assign(WL[CUR.type], fieldSpec(CUR,wOf(CUR)).presets[+b.dataset.preset][1]); renderSide(); renderFit(); }
    else if(q('#reset-filters')){ F.type='all'; F.q=''; F.fits=false; $('search').value=''; renderPills(); renderGrid(); }
    else if(q('#reset-targets')){ Object.keys(DEFAULT_WL).forEach(k=>WL[k] = {...DEFAULT_WL[k]}); Object.keys(OV).forEach(k=>delete OV[k]); renderGrid(); }
    else if(q('#add-paste-btn')) usePaste();
  });
  const onSide = e=>{
    const el = e.target, k = el.dataset.k; if(!k || !CUR) return;
    let v = el.type==='checkbox' ? el.checked : el.value;
    if(el.dataset.num){ v = parseFloat(v); if(!isFinite(v) || v<=0){ if(el.dataset.opt) v = 0; else return; } if(el.dataset.int) v = Math.max(1, Math.round(v)); }
    const sc = el.dataset.scope;
    if(sc==='wl') WL[CUR.type][k] = v; else if(sc==='ov') (OV[CUR.id] = OV[CUR.id]||{})[k] = v; else if(sc==='c') C[k] = v;
    else { SEL[k] = v; if(k==='hw'){ SEL.n = 1; renderSide(); } }
    renderFit();
  };
  els.fitSide.addEventListener('input', onSide);
  $('add-form').addEventListener('submit', e=>{ e.preventDefault(); importHF(); });
  $('dev-form').addEventListener('submit', e=>{ e.preventDefault(); saveDevice(); });
  $('dev-form').addEventListener('input', devPreview);
  $('stk-rows').addEventListener('input', e=>{ const el = e.target, row = STACK.rows[+el.dataset.stkI]; if(!el.dataset.stkK || !row) return; const v = parseFloat(el.value); if(isFinite(v) && v>0) row.ov[el.dataset.stkK] = v; renderStackOut(); });
  $('stk-hw').addEventListener('change', e=>{ STACK.hw = e.target.value; STACK.n = 1; stkNSelect(); renderStackOut(); });
  $('stk-n').addEventListener('change', e=>{ STACK.n = +e.target.value; renderStackOut(); });
  $('report-modal').addEventListener('input', e=>{ if(e.target.id==='rp-client' || e.target.id==='rp-project'){ RP.verified = null; renderReport(); } });
  document.addEventListener('keydown', e=>{
    if(e.key==='Escape'){ if(!els.rpModal.hidden) closeReport(); else if(!els.devModal.hidden) closeDev(); else if(!els.addModal.hidden){ els.addModal.hidden = true; document.body.classList.remove('modal-open'); } else if(!els.fitModal.hidden) closeFit(); else closeDrawer(); }
    else if(e.key==='/' && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)){ e.preventDefault(); $('search').focus(); }
  });
  const h = (location.hash||'').slice(1);
  if(h.startsWith('fit-')) openFit(h.slice(4)); else if(h && document.getElementById(h) && h!=='top' && h!=='models') gotoMetric(h);
  const rm = (location.hash||'').match(/^#report=([A-Za-z0-9_-]+)\.([0-9a-f]+)$/);
  if(rm){ try{ const st = b64d(rm[1]); applyState(st); renderAll(); sha(rm[1]).then(hh=>{ RP.verified = hh.slice(0,12)===rm[2]; openReport(st.k==='t'?'stack':'single', st.i); }); }catch(e){} }
  loadHub(); loadData(); loadTrending();
}
init();
