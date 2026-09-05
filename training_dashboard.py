"""
Live training dashboard.

Start training first:
    python train_nuscenes.py

Then open the dashboard in a separate terminal:
    python training_dashboard.py

Or point at a different log file:
    python training_dashboard.py --log my_run.jsonl --port 5501
"""
import argparse
import json
import os
import sys
import threading
import webbrowser
from flask import Flask, jsonify

app = Flask(__name__)
LOG_PATH = "training_log.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
#  API
# ─────────────────────────────────────────────────────────────────────────────

def _read_log():
    records = []
    if not os.path.exists(LOG_PATH):
        return records
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


@app.route("/")
def index():
    return HTML


@app.route("/api/metrics")
def api_metrics():
    return jsonify(_read_log())


# ─────────────────────────────────────────────────────────────────────────────
#  Front-end
# ─────────────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#f5f6fa;color:#1a1a2e;font-family:'Segoe UI',monospace;min-height:100vh}
header{background:#fff;padding:10px 20px;display:flex;justify-content:space-between;
       align-items:center;border-bottom:1px solid #dde1f0;box-shadow:0 1px 4px rgba(0,0,0,.06)}
h1{font-size:15px;color:#1565C0;font-weight:600}
#status-pill{font-size:11px;padding:3px 10px;border-radius:12px;font-weight:600}
.pill-live{background:#e8f5e9;color:#2e7d32;border:1px solid #a5d6a7}
.pill-done{background:#fff3e0;color:#e65100;border:1px solid #ffcc80}
.pill-wait{background:#f5f5f5;color:#999;border:1px solid #ddd}

#layout{display:grid;grid-template-columns:220px 1fr;gap:0;height:calc(100vh - 47px)}

/* ── sidebar ── */
#sidebar{background:#fff;border-right:1px solid #dde1f0;padding:14px 12px;
         overflow-y:auto;display:flex;flex-direction:column;gap:14px}
.card{background:#f8f9fe;border:1px solid #dde1f0;border-radius:6px;padding:12px}
.card-title{font-size:10px;color:#90a0c0;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}
.metric{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px}
.metric-label{font-size:11px;color:#6b7a99}
.metric-value{font-size:13px;color:#1a1a2e;font-weight:600}
.metric-value.hi{color:#1565C0}
.metric-value.good{color:#2e7d32}
.progress-wrap{margin-top:8px}
.progress-bar-bg{background:#e8eaf6;border-radius:4px;height:6px;overflow:hidden;margin-top:4px}
.progress-bar-fg{background:linear-gradient(90deg,#1976D2,#00BCD4);height:6px;
                 border-radius:4px;transition:width .4s ease}
.progress-label{display:flex;justify-content:space-between;font-size:10px;color:#90a0c0;margin-top:3px}

/* ── main ── */
#main{display:flex;flex-direction:column;overflow:hidden}
.charts-grid{display:grid;grid-template-columns:1fr 1fr;gap:1px;
             background:#dde1f0;flex:1;overflow:hidden}
.chart-panel{background:#fff;padding:14px 16px;display:flex;flex-direction:column}
.chart-title{font-size:11px;color:#90a0c0;text-transform:uppercase;letter-spacing:.8px;
             margin-bottom:10px;flex-shrink:0}
.chart-wrap{flex:1;position:relative;min-height:0}
canvas{position:absolute;inset:0}

/* ── epoch table ── */
#table-panel{background:#fff;border-top:1px solid #dde1f0;max-height:180px;
             overflow-y:auto;flex-shrink:0}
table{width:100%;border-collapse:collapse;font-size:11px}
thead th{position:sticky;top:0;background:#f8f9fe;color:#90a0c0;padding:6px 12px;
         text-align:right;font-weight:600;font-size:10px;text-transform:uppercase;
         letter-spacing:.6px;border-bottom:1px solid #dde1f0}
thead th:first-child{text-align:left}
tbody tr:hover{background:#f0f4ff}
tbody td{padding:5px 12px;text-align:right;color:#444;border-bottom:1px solid #eef0f8}
tbody td:first-child{text-align:left;color:#1a1a2e}
td.loss-good{color:#2e7d32}
td.loss-mid {color:#e65100}
td.loss-bad {color:#c62828}

#refresh-bar{background:#f8f9fe;padding:4px 14px;font-size:10px;color:#aaa;
             display:flex;justify-content:space-between;border-top:1px solid #dde1f0}
</style>
</head>
<body>
<header>
  <h1>PointPillars — Training Dashboard</h1>
  <span id="status-pill" class="pill-wait">● Waiting for log</span>
</header>

<div id="layout">
  <aside id="sidebar">
    <div class="card" id="card-progress">
      <div class="card-title">Progress</div>
      <div class="metric"><span class="metric-label">Epoch</span>
                          <span class="metric-value hi" id="m-epoch">—</span></div>
      <div class="metric"><span class="metric-label">Step</span>
                          <span class="metric-value" id="m-step">—</span></div>
      <div class="progress-wrap">
        <div class="progress-bar-bg"><div class="progress-bar-fg" id="pb-epoch" style="width:0%"></div></div>
        <div class="progress-label"><span>Epochs</span><span id="pb-epoch-lbl">0%</span></div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Latest Loss</div>
      <div class="metric"><span class="metric-label">Total</span>
                          <span class="metric-value hi" id="m-loss">—</span></div>
      <div class="metric"><span class="metric-label">Cls (focal)</span>
                          <span class="metric-value" id="m-cls">—</span></div>
      <div class="metric"><span class="metric-label">Reg (smooth-L1)</span>
                          <span class="metric-value" id="m-reg">—</span></div>
    </div>

    <div class="card">
      <div class="card-title">Timing</div>
      <div class="metric"><span class="metric-label">Last epoch</span>
                          <span class="metric-value" id="m-dur">—</span></div>
      <div class="metric"><span class="metric-label">ETA</span>
                          <span class="metric-value good" id="m-eta">—</span></div>
      <div class="metric"><span class="metric-label">Elapsed</span>
                          <span class="metric-value" id="m-elapsed">—</span></div>
    </div>

    <div class="card">
      <div class="card-title">Config</div>
      <div class="metric"><span class="metric-label">Dataset</span>
                          <span class="metric-value" id="cfg-ds">—</span></div>
      <div class="metric"><span class="metric-label">Steps/epoch</span>
                          <span class="metric-value" id="cfg-steps">—</span></div>
      <div class="metric"><span class="metric-label">Total epochs</span>
                          <span class="metric-value" id="cfg-epochs">—</span></div>
    </div>
  </aside>

  <div id="main">
    <div class="charts-grid">
      <div class="chart-panel">
        <div class="chart-title">Step Loss (recent 400 steps)</div>
        <div class="chart-wrap"><canvas id="chart-step"></canvas></div>
      </div>
      <div class="chart-panel">
        <div class="chart-title">Epoch Average Loss</div>
        <div class="chart-wrap"><canvas id="chart-epoch"></canvas></div>
      </div>
    </div>

    <div id="table-panel">
      <table>
        <thead>
          <tr><th>Epoch</th><th>Avg Loss</th><th>Cls Loss</th>
              <th>Reg Loss</th><th>Duration</th></tr>
        </thead>
        <tbody id="epoch-tbody"></tbody>
      </table>
    </div>

    <div id="refresh-bar">
      <span id="last-update">Not yet updated</span>
      <span>Auto-refresh every 3s</span>
    </div>
  </div>
</div>

<script>
'use strict';

// ── chart setup ──────────────────────────────────────────────────────────────
const CHART_DEFAULTS = {
  responsive: true, maintainAspectRatio: false,
  animation: { duration: 300 },
  plugins: { legend: { labels: { color: '#444', font: { size: 11 } } } },
  scales: {
    x: { ticks: { color: '#90a0c0', maxTicksLimit: 8, font: { size: 10 } },
         grid:  { color: 'rgba(0,0,0,0.05)' } },
    y: { ticks: { color: '#90a0c0', font: { size: 10 } },
         grid:  { color: 'rgba(0,0,0,0.07)' } },
  },
};

function makeChart(id, label, color, fill=false) {
  return new Chart(document.getElementById(id), {
    type: 'line',
    data: { labels: [], datasets: [{
      label, data: [], borderColor: color,
      backgroundColor: fill ? color+'22' : 'transparent',
      borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill,
    }]},
    options: { ...CHART_DEFAULTS },
  });
}

function makeMultiChart(id, datasets) {
  return new Chart(document.getElementById(id), {
    type: 'line',
    data: { labels: [], datasets: datasets.map(([label,color])=>({
      label, data: [], borderColor: color,
      backgroundColor: 'transparent',
      borderWidth: 2, pointRadius: 2, pointHoverRadius: 4, tension: 0.3,
    }))},
    options: { ...CHART_DEFAULTS },
  });
}

const stepChart  = makeChart('chart-step',  'Step loss',  '#2196F3', true);
const epochChart = makeMultiChart('chart-epoch', [
  ['Total avg', '#90caf9'], ['Cls', '#4CAF50'], ['Reg', '#FF9800']
]);

// ── state ────────────────────────────────────────────────────────────────────
let cfg = null, startTs = null;

// ── helpers ──────────────────────────────────────────────────────────────────
function fmt(v) { return typeof v === 'number' ? v.toFixed(4) : '—'; }

function fmtDur(s) {
  if (!s || s < 0) return '—';
  if (s < 60)  return s.toFixed(0) + 's';
  if (s < 3600) return (s/60).toFixed(1) + 'm';
  return (s/3600).toFixed(1) + 'h';
}

function lossClass(v) {
  if (v < 0.3) return 'loss-good';
  if (v < 1.0) return 'loss-mid';
  return 'loss-bad';
}

function setText(id, v) {
  const el = document.getElementById(id);
  if (el) el.textContent = v;
}

// ── update ───────────────────────────────────────────────────────────────────
async function refresh() {
  let records;
  try {
    const r = await fetch('/api/metrics');
    records = await r.json();
  } catch { return; }

  if (!records.length) return;

  // Partition records
  const cfgRec    = records.find(r => r.type === 'config');
  const stepRecs  = records.filter(r => r.type === 'step');
  const epochRecs = records.filter(r => r.type === 'epoch');
  const done      = records.some(r => r.type === 'done');

  // Status pill
  const pill = document.getElementById('status-pill');
  if (done) {
    pill.textContent = '● Training complete';
    pill.className = 'pill-done';
  } else if (epochRecs.length || stepRecs.length) {
    pill.textContent = '● Live';
    pill.className = 'pill-live';
  } else {
    pill.textContent = '● Waiting for data';
    pill.className = 'pill-wait';
  }

  // Config
  if (cfgRec && !cfg) {
    cfg = cfgRec;
    if (cfgRec.timestamp && !startTs) startTs = cfgRec.timestamp;
    setText('cfg-ds',     cfgRec.dataset_size?.toLocaleString() ?? '—');
    setText('cfg-steps',  cfgRec.steps_per_epoch ?? '—');
    setText('cfg-epochs', cfgRec.epochs ?? '—');
  }

  // Step chart (last 400 steps)
  const recent = stepRecs.slice(-400);
  stepChart.data.labels   = recent.map((_,i) => i+1);
  stepChart.data.datasets[0].data = recent.map(r => r.loss);
  stepChart.update('none');

  // Epoch chart
  epochChart.data.labels = epochRecs.map(r => `Ep ${r.epoch}`);
  epochChart.data.datasets[0].data = epochRecs.map(r => r.avg_loss);
  epochChart.data.datasets[1].data = epochRecs.map(r => r.avg_cls);
  epochChart.data.datasets[2].data = epochRecs.map(r => r.avg_reg);
  epochChart.update('none');

  // Sidebar metrics
  const lastStep  = stepRecs[stepRecs.length - 1];
  const lastEpoch = epochRecs[epochRecs.length - 1];
  const totalEpochs = cfgRec?.epochs ?? lastEpoch?.total_epochs ?? '?';
  const stepsPerEp  = cfgRec?.steps_per_epoch ?? '?';

  if (lastStep) {
    setText('m-epoch', `${lastStep.epoch} / ${totalEpochs}`);
    setText('m-step',  `${lastStep.step} / ${stepsPerEp}`);
    setText('m-loss',  fmt(lastStep.loss));
    setText('m-cls',   fmt(lastStep.cls_loss));
    setText('m-reg',   fmt(lastStep.reg_loss));
    const epPct = (lastStep.epoch - 1 + lastStep.step / stepsPerEp) / totalEpochs * 100;
    document.getElementById('pb-epoch').style.width = Math.min(100, epPct).toFixed(1) + '%';
    document.getElementById('pb-epoch-lbl').textContent = epPct.toFixed(0) + '%';
  }

  if (lastEpoch) {
    setText('m-dur', fmtDur(lastEpoch.duration_s));
    const doneEpochs = lastEpoch.epoch;
    const remEpochs  = (totalEpochs === '?' ? 0 : totalEpochs) - doneEpochs;
    if (remEpochs > 0 && lastEpoch.duration_s) {
      const avgDur = epochRecs.reduce((s,r)=>s+r.duration_s,0) / epochRecs.length;
      setText('m-eta', fmtDur(avgDur * remEpochs));
    } else if (done) {
      setText('m-eta', 'Done');
    }
  }

  if (startTs) {
    setText('m-elapsed', fmtDur(Date.now()/1000 - startTs));
  }

  // Epoch table (newest first)
  const tbody = document.getElementById('epoch-tbody');
  tbody.innerHTML = '';
  for (const r of [...epochRecs].reverse()) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r.epoch} / ${r.total_epochs ?? totalEpochs}</td>
      <td class="${lossClass(r.avg_loss)}">${fmt(r.avg_loss)}</td>
      <td>${fmt(r.avg_cls)}</td>
      <td>${fmt(r.avg_reg)}</td>
      <td>${fmtDur(r.duration_s)}</td>`;
    tbody.appendChild(tr);
  }

  setText('last-update', 'Updated ' + new Date().toLocaleTimeString());
}

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",  default="training_log.jsonl")
    parser.add_argument("--port", type=int, default=5501)
    args = parser.parse_args()

    LOG_PATH = args.log
    url = f"http://localhost:{args.port}"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Dashboard → {url}")
    print(f"Reading metrics from: {LOG_PATH}")
    app.run(port=args.port, debug=False, use_reloader=False)
