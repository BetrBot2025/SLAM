"""
Interactive web-based nuScenes BEV visualizer.

Run:
    python visualize_nuscenes_web.py
    python visualize_nuscenes_web.py --root lidar_detection/data/nuScenes_mini --port 5500

Controls (in browser):
    Scroll          zoom in / out
    Drag            pan
    R               reset view
    ← / →           previous / next frame
    Space           play / pause
"""
import argparse, base64, json, os, sys, threading, webbrowser
from collections import Counter
import numpy as np
from flask import Flask, jsonify

sys.path.insert(0, ".")
from lidar_detection.data.nuscenes_loader import NuScenesDataset
from lidar_detection.config import PipelineConfig

DS        = None
DATAROOT  = None
VERSION   = None
PC_RANGE  = None

app = Flask(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  API
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML

@app.route("/api/info")
def api_info():
    meta = os.path.join(DATAROOT, VERSION)
    with open(os.path.join(meta, "scene.json"))  as f: scenes_raw  = json.load(f)
    with open(os.path.join(meta, "sample.json")) as f: samples_raw = json.load(f)

    scene_name = {s["token"]: s["name"]                for s in scenes_raw}
    scene_desc = {s["token"]: s.get("description", "") for s in scenes_raw}
    tok_scene  = {s["token"]: s["scene_token"]         for s in samples_raw}

    scenes, prev, start = [], None, 0
    for idx, tok in enumerate(DS._sample_tokens):
        name = scene_name.get(tok_scene.get(tok, ""), "unknown")
        if name != prev:
            if prev is not None:
                scenes[-1]["end"] = idx - 1
            scenes.append({"name": name,
                           "desc": scene_desc.get(tok_scene.get(tok, ""), ""),
                           "start": idx, "end": idx})
            prev = name
    if scenes:
        scenes[-1]["end"] = len(DS) - 1

    return jsonify({"total": len(DS), "scenes": scenes, "pc_range": PC_RANGE})


@app.route("/api/sample/<int:idx>")
def api_sample(idx):
    if not (0 <= idx < len(DS)):
        return jsonify({"error": "out of range"}), 400
    points, gt_boxes = DS[idx]
    return jsonify({
        "idx":       idx,
        "total":     len(DS),
        "n_pts":     int(len(points)),
        "pts_b64":   base64.b64encode(points.astype(np.float32).tobytes()).decode(),
        "gt_boxes":  gt_boxes,
        "cls_counts": dict(Counter(b["class_name"] for b in gt_boxes)),
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Front-end  (single-page, no external dependencies)
# ─────────────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>nuScenes Visualizer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f0f23;color:#e0e0e0;font-family:'Segoe UI',monospace;height:100vh;overflow:hidden;user-select:none}
#app{display:flex;flex-direction:column;height:100vh}
header{background:#16163a;padding:7px 16px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #2a2a5a;flex-shrink:0}
header span{font-size:13px;color:#90caf9}
#frame-info{color:#aaa;font-size:12px}
#main{display:flex;flex:1;overflow:hidden}

/* ── sidebar ── */
#sidebar{width:210px;background:#111128;border-right:1px solid #222;overflow-y:auto;padding:10px 8px;flex-shrink:0}
.sec{margin-bottom:14px}
.sec-label{font-size:10px;color:#555;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:5px;padding-left:2px}
.scene-item{padding:5px 7px;cursor:pointer;border-radius:4px;font-size:11px;color:#888;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.scene-item:hover{background:#1e2a4e;color:#ccc}
.scene-item.active{background:#1a3560;color:#90caf9;font-weight:600}
.cls-row{display:flex;align-items:center;gap:7px;padding:4px 3px;cursor:pointer;border-radius:3px}
.cls-row:hover{background:#1a1a3a}
.cls-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.cls-lbl{font-size:12px}
input[type=range]{width:100%;accent-color:#2196F3;margin:4px 0}
#sample-lbl{font-size:11px;color:#888;margin-bottom:3px}
button{background:#1a2a4a;color:#90caf9;border:1px solid #2a4a7a;border-radius:3px;padding:5px 10px;font-size:11px;cursor:pointer;transition:background .15s}
button:hover{background:#2196F3;color:#fff;border-color:#2196F3}
button.on{background:#2196F3;color:#fff;border-color:#2196F3}
#btn-reset{width:100%;margin-top:4px}

/* ── canvas area ── */
#content{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}
#canvas-wrap{flex:1;position:relative;overflow:hidden;background:#1a1a2e}
#bev{display:block;cursor:crosshair;width:100%;height:100%}
#tooltip{position:absolute;pointer-events:none;background:rgba(0,0,0,.75);color:#e0e0e0;font-size:11px;padding:4px 8px;border-radius:3px;border:1px solid #444;display:none;white-space:nowrap}

/* ── bottom bars ── */
#stats{background:#13132e;padding:5px 14px;font-size:12px;color:#999;display:flex;gap:18px;flex-shrink:0;border-top:1px solid #1e1e3e}
#stats span{display:flex;align-items:center;gap:4px}
#ctrl{background:#0e0e26;padding:7px 14px;display:flex;align-items:center;gap:10px;flex-shrink:0;border-top:1px solid #1a1a3a}
#hint{background:#09091a;padding:2px 14px;font-size:10px;color:#444;flex-shrink:0}
.fps-wrap{display:flex;align-items:center;gap:6px;font-size:11px;color:#777;margin-left:auto}
.fps-wrap input{width:80px}
</style>
</head>
<body>
<div id="app">
  <header>
    <span>nuScenes BEV Visualizer</span>
    <span id="frame-info">loading…</span>
  </header>
  <div id="main">
    <aside id="sidebar">
      <div class="sec">
        <div class="sec-label">Scenes</div>
        <div id="scene-list"></div>
      </div>
      <div class="sec">
        <div class="sec-label">Sample</div>
        <div id="sample-lbl">0 / 0</div>
        <input type="range" id="s-slider" min="0" max="0" value="0">
      </div>
      <div class="sec">
        <div class="sec-label">Classes</div>
        <div id="cls-filters"></div>
      </div>
      <div class="sec">
        <div class="sec-label">View</div>
        <button id="btn-reset">⟳ Reset view</button>
      </div>
    </aside>
    <div id="content">
      <div id="canvas-wrap">
        <canvas id="bev"></canvas>
        <div id="tooltip"></div>
      </div>
      <div id="stats">
        <span id="st-pts">— pts</span>
        <span id="st-gt">— GT</span>
        <span id="st-cls"></span>
      </div>
      <div id="ctrl">
        <button id="btn-prev">◀ Prev</button>
        <button id="btn-play">▶ Play</button>
        <button id="btn-next">Next ▶</button>
        <div class="fps-wrap">
          FPS <input type="range" id="fps" min="1" max="10" value="3">
          <span id="fps-lbl">3</span>
        </div>
      </div>
      <div id="hint">Scroll: zoom &nbsp;|&nbsp; Drag: pan &nbsp;|&nbsp; R: reset view &nbsp;|&nbsp; ← →: frames &nbsp;|&nbsp; Space: play/pause</div>
    </div>
  </div>
</div>
<script>
'use strict';

// ── config ──────────────────────────────────────────────────────────────────
const CLS_COLORS = {Car:'#2196F3', Pedestrian:'#4CAF50', Cyclist:'#FF9800'};
const PLASMA = [
  [13,  8,135],[84,  2,163],[139, 10,165],[185, 50,137],
  [219, 92, 104],[244,136, 73],[254,188, 43],[240,249, 33]
];

// ── state ────────────────────────────────────────────────────────────────────
const S = {
  idx:0, total:0, scenes:[], pcRange:[0,-39.68,-3,69.12,39.68,1],
  data:null, playing:false, timer:null, fps:3,
  cls:{Car:true, Pedestrian:true, Cyclist:true},
  zoom:8, panX:0, panY:0,
  drag:false, dragStart:null, panStart:null,
  defaultZoom:8,
};

// ── plasma colormap ───────────────────────────────────────────────────────────
function plasma(t) {
  t = Math.max(0, Math.min(1, t));
  const pos = t * (PLASMA.length-1), i = Math.floor(pos), f = pos-i;
  if (i >= PLASMA.length-1) return PLASMA[PLASMA.length-1];
  const [r1,g1,b1]=PLASMA[i],[r2,g2,b2]=PLASMA[i+1];
  return [r1+f*(r2-r1)|0, g1+f*(g2-g1)|0, b1+f*(b2-b1)|0];
}

// ── decode base64 float32 ────────────────────────────────────────────────────
function decodeF32(b64) {
  const bin = atob(b64), n = bin.length;
  const buf = new ArrayBuffer(n), u8 = new Uint8Array(buf);
  for (let i=0;i<n;i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

// ── world → screen ────────────────────────────────────────────────────────────
// nuScenes LiDAR: x=forward y=left z=up
// BEV canvas: forward=up  left=left  ego at (cx, cy)
function w2s(wx, wy, W, H) {
  return [W/2 + S.panX - wy*S.zoom,
          H/2 + S.panY - wx*S.zoom];
}

// ── render ────────────────────────────────────────────────────────────────────
function render() {
  const canvas = document.getElementById('bev');
  const W = canvas.width, H = canvas.height;
  if (!W || !H || !S.data) return;
  const ctx = canvas.getContext('2d');
  const pts = decodeF32(S.data.pts_b64);
  const N = pts.length >> 2;   // /4
  const [,,zMin,,,zMax] = S.pcRange;

  // ── points via ImageData (fast pixel write) ──────────────────────────────
  const img  = ctx.createImageData(W, H);
  const d    = img.data;
  // fill background
  const [BR,BG,BB] = [26,26,46];
  for (let p=0; p<d.length; p+=4) { d[p]=BR; d[p+1]=BG; d[p+2]=BB; d[p+3]=255; }

  for (let i=0;i<N;i++) {
    const px=pts[i*4], py=pts[i*4+1], pz=pts[i*4+2];
    const [sx,sy] = w2s(px, py, W, H);
    const ix=sx|0, iy=sy|0;
    if (ix<0||ix>=W||iy<0||iy>=H) continue;
    const [r,g,b] = plasma((pz-zMin)/(zMax-zMin));
    const o=(iy*W+ix)*4;
    d[o]=r; d[o+1]=g; d[o+2]=b; d[o+3]=255;
  }
  ctx.putImageData(img, 0, 0);

  // ── detection range overlay ────────────────────────────────────────────────
  const [xMin,yMin,,xMax,yMax] = S.pcRange;
  const [tlx,tly] = w2s(xMax,  yMax, W, H);
  const [brx,bry] = w2s(xMin, yMin, W, H);
  ctx.strokeStyle='rgba(255,255,255,0.12)';
  ctx.lineWidth=1;
  ctx.setLineDash([6,4]);
  ctx.strokeRect(tlx, tly, brx-tlx, bry-tly);
  ctx.setLineDash([]);

  // ── GT boxes ─────────────────────────────────────────────────────────────
  for (const box of S.data.gt_boxes) {
    if (!S.cls[box.class_name]) continue;
    drawBox(ctx, box, W, H);
  }

  // ── ego marker ────────────────────────────────────────────────────────────
  const [ex,ey] = w2s(0,0,W,H);
  const [fx,fy] = w2s(4,0,W,H);  // 4m forward arrow
  ctx.fillStyle='#fff';
  ctx.beginPath(); ctx.arc(ex,ey,4,0,Math.PI*2); ctx.fill();
  ctx.strokeStyle='rgba(255,255,255,0.6)';
  ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(ex,ey); ctx.lineTo(fx,fy); ctx.stroke();

  // ── scale bar ──────────────────────────────────────────────────────────────
  const barM = S.zoom>=4 ? 10 : 20;
  const barPx = barM * S.zoom;
  const bx=W-barPx-16, by=H-14;
  ctx.strokeStyle='#667'; ctx.lineWidth=1.5;
  ctx.beginPath();
  ctx.moveTo(bx,by); ctx.lineTo(bx+barPx,by);
  ctx.moveTo(bx,by-3); ctx.lineTo(bx,by+3);
  ctx.moveTo(bx+barPx,by-3); ctx.lineTo(bx+barPx,by+3);
  ctx.stroke();
  ctx.fillStyle='#667'; ctx.font='10px monospace';
  ctx.fillText(barM+'m', bx+barPx/2-10, by-5);
}

function drawBox(ctx, box, W, H) {
  const [cx,cy]  = box.center;
  const [l,w]    = box.size;
  const yaw      = box.yaw;
  const col      = CLS_COLORS[box.class_name]||'#fff';
  const cos=Math.cos(yaw), sin=Math.sin(yaw);
  const hl=l/2, hw=w/2;
  const local=[[hl,hw],[hl,-hw],[-hl,-hw],[-hl,hw]];
  const sc = local.map(([dx,dy])=>w2s(cx+cos*dx-sin*dy, cy+sin*dx+cos*dy, W, H));

  ctx.strokeStyle=col; ctx.lineWidth=1.5; ctx.setLineDash([5,3]);
  ctx.beginPath(); ctx.moveTo(...sc[0]);
  for(let i=1;i<4;i++) ctx.lineTo(...sc[i]);
  ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);

  // direction arrow
  const fm=[(sc[0][0]+sc[1][0])/2,(sc[0][1]+sc[1][1])/2];
  const [bcx,bcy]=w2s(cx,cy,W,H);
  ctx.strokeStyle=col; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(bcx,bcy); ctx.lineTo(...fm); ctx.stroke();

  // label
  if (S.zoom > 5) {
    ctx.fillStyle=col;
    ctx.font=`${Math.max(9,S.zoom*0.9)|0}px monospace`;
    ctx.fillText(box.class_name[0], bcx+3, bcy-2);
  }
}

// ── resize ────────────────────────────────────────────────────────────────────
function resize() {
  const wrap = document.getElementById('canvas-wrap');
  const canvas = document.getElementById('bev');
  canvas.width  = wrap.clientWidth;
  canvas.height = wrap.clientHeight;
  S.defaultZoom = Math.min(canvas.width/100, canvas.height/100);
  render();
}

// ── data fetch ────────────────────────────────────────────────────────────────
async function load(idx) {
  idx = Math.max(0, Math.min(S.total-1, idx));
  const r = await fetch(`/api/sample/${idx}`);
  S.data = await r.json();
  S.idx  = idx;
  document.getElementById('s-slider').value = idx;
  document.getElementById('sample-lbl').textContent = `${idx+1} / ${S.total}`;
  document.getElementById('frame-info').textContent  = `Frame ${idx} / ${S.total-1}`;

  // stats bar
  document.getElementById('st-pts').textContent = `⬡ ${S.data.n_pts.toLocaleString()} pts`;
  document.getElementById('st-gt').textContent  = `GT: ${S.data.gt_boxes.length}`;
  document.getElementById('st-cls').innerHTML   =
    Object.entries(S.data.cls_counts||{})
      .map(([c,n])=>`<span style="color:${CLS_COLORS[c]||'#aaa'}">${c}: ${n}</span>`)
      .join(' ');

  // highlight active scene
  document.querySelectorAll('.scene-item').forEach((el,i)=>{
    const sc=S.scenes[i];
    el.classList.toggle('active', sc && S.idx>=sc.start && S.idx<=sc.end);
  });

  render();
}

async function init() {
  const r = await fetch('/api/info');
  const info = await r.json();
  S.total   = info.total;
  S.scenes  = info.scenes;
  if (info.pc_range) S.pcRange = info.pc_range;

  document.getElementById('s-slider').max = S.total-1;

  // scene list
  const sl = document.getElementById('scene-list');
  info.scenes.forEach(sc=>{
    const el=document.createElement('div');
    el.className='scene-item';
    el.textContent=`${sc.name} [${sc.end-sc.start+1}]`;
    el.title=sc.desc||'';
    el.onclick=()=>load(sc.start);
    sl.appendChild(el);
  });

  // class filters
  const cf = document.getElementById('cls-filters');
  Object.entries(CLS_COLORS).forEach(([cls,col])=>{
    const row=document.createElement('div');
    row.className='cls-row';
    row.innerHTML=`<div class="cls-dot" id="dot-${cls}" style="background:${col}"></div>
                   <span class="cls-lbl" id="lbl-${cls}" style="color:${col}">${cls}</span>`;
    row.onclick=()=>{
      S.cls[cls]=!S.cls[cls];
      document.getElementById(`dot-${cls}`).style.opacity=S.cls[cls]?1:0.25;
      document.getElementById(`lbl-${cls}`).style.color=S.cls[cls]?col:'#555';
      render();
    };
    cf.appendChild(row);
  });

  resize();
  await load(0);
}

// ── navigation ────────────────────────────────────────────────────────────────
function prev()  { load(S.idx-1); }
function next()  { load(S.idx+1); }
function togglePlay() {
  S.playing=!S.playing;
  const btn=document.getElementById('btn-play');
  btn.textContent=S.playing?'⏸ Pause':'▶ Play';
  btn.classList.toggle('on',S.playing);
  if (S.playing) S.timer=setInterval(()=>load((S.idx+1)%S.total), 1000/S.fps);
  else           clearInterval(S.timer);
}

// ── canvas events ─────────────────────────────────────────────────────────────
(function(){
  const canvas=document.getElementById('bev');
  canvas.addEventListener('wheel',e=>{
    e.preventDefault();
    S.zoom=Math.max(1.5,Math.min(120,S.zoom*(e.deltaY<0?1.15:0.87)));
    render();
  },{passive:false});
  canvas.addEventListener('mousedown',e=>{
    S.drag=true; S.dragStart=[e.clientX,e.clientY];
    S.panStart=[S.panX,S.panY]; canvas.style.cursor='grabbing';
  });
  canvas.addEventListener('mousemove',e=>{
    if(!S.drag)return;
    S.panX=S.panStart[0]+(e.clientX-S.dragStart[0]);
    S.panY=S.panStart[1]+(e.clientY-S.dragStart[1]);
    render();
  });
  ['mouseup','mouseleave'].forEach(ev=>canvas.addEventListener(ev,()=>{
    S.drag=false; canvas.style.cursor='crosshair';
  }));
})();

// ── keyboard ──────────────────────────────────────────────────────────────────
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT') return;
  if(e.key==='ArrowLeft'||e.key==='a')  prev();
  if(e.key==='ArrowRight'||e.key==='d') next();
  if(e.key===' ')  { e.preventDefault(); togglePlay(); }
  if(e.key==='r'||e.key==='R') {
    S.zoom=S.defaultZoom; S.panX=0; S.panY=0; render();
  }
});

// ── wire up static controls ───────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded',()=>{
  document.getElementById('btn-prev').onclick  = prev;
  document.getElementById('btn-next').onclick  = next;
  document.getElementById('btn-play').onclick  = togglePlay;
  document.getElementById('btn-reset').onclick = ()=>{
    S.zoom=S.defaultZoom; S.panX=0; S.panY=0; render();
  };
  document.getElementById('s-slider').oninput=e=>load(+e.target.value);
  document.getElementById('fps').oninput=e=>{
    S.fps=+e.target.value;
    document.getElementById('fps-lbl').textContent=S.fps;
    if(S.playing){ clearInterval(S.timer); S.timer=setInterval(()=>load((S.idx+1)%S.total),1000/S.fps); }
  };
  new ResizeObserver(resize).observe(document.getElementById('canvas-wrap'));
  init();
});
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",    default="lidar_detection/data/nuScenes_mini")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--port",    type=int, default=5500)
    args = parser.parse_args()

    DATAROOT = args.root
    VERSION  = args.version
    PC_RANGE = PipelineConfig().voxel.point_cloud_range

    print(f"Loading {VERSION} from {DATAROOT} …")
    DS = NuScenesDataset(DATAROOT, VERSION)

    url = f"http://localhost:{args.port}"
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"Visualizer → {url}")
    print("Press Ctrl+C to stop.")
    app.run(port=args.port, debug=False, use_reloader=False)
