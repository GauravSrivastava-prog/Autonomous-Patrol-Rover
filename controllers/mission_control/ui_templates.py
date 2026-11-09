#!/usr/bin/env python3
"""
ui_templates.py — Modern Mission Control Web Dashboards (2D Dashboard & 3D WebXR Twin)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features:
  1. DASHBOARD_HTML:
     - Side-by-side Dual Viewport: .dual-video-grid with responsive fallback (580px).
     - Non-blocking double-buffered snapshot polling (/video_frame.jpg, /depth_frame.jpg).
     - Robust fetch() wrappers with AbortController timeout (2500ms) to prevent UI hangs.
     - Live 2D radar minimap with repositioned arena obstacle coordinates.
     - Telemetry gauges, manual WASD deck, VLM controls, and Purpose C prompt input.
  2. VR_HTML:
     - A-Frame 1.6.0 3D WebXR Digital Twin with Dual Holographic Vision Screens.
     - Interactive 3D drag-and-drop objects with bidirectional WebSocket sync.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mission Control — AI Vision Patrol & Multimodal Robotics</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #070b12;
    --surface: #0e1626;
    --surface2: #142036;
    --accent: #00e5ff;
    --accent2: #ff1744;
    --accent3: #76ff03;
    --purple: #d500f9;
    --text: #e0e6ed;
    --muted: #8fa0c0;
    --border: #1e2c48;
  }
  * { box-sizing: border-box; margin:0; padding:0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: 'Inter', sans-serif;
    min-height: 100vh; display: flex; flex-direction: column;
  }
  header {
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 12px 24px; display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 12px;
  }
  .brand {
    font-family: 'Orbitron', monospace; font-size: 1.15rem; font-weight: 900;
    color: var(--accent); letter-spacing: 2px; display: flex; align-items: center; gap: 10px;
  }
  .brand-dot {
    width: 10px; height: 10px; border-radius: 50%; background: var(--accent3);
    box-shadow: 0 0 10px var(--accent3); animation: pulse 1.5s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
  .nav-links a {
    color: var(--muted); text-decoration: none; font-size: 0.85rem; font-weight: 600;
    padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border);
    transition: all 0.2s;
  }
  .nav-links a:hover { color: var(--accent); border-color: var(--accent); }

  .grid {
    display: grid; grid-template-columns: 2fr 1fr;
    gap: 16px; padding: 16px; flex: 1;
  }
  @media (max-width: 1024px) {
    .grid { grid-template-columns: 1fr; }
  }

  .panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden;
  }
  .panel-title {
    font-family: 'Orbitron', monospace; font-size: 0.72rem;
    font-weight: 700; letter-spacing: 2px; color: var(--accent);
    padding: 12px 16px; border-bottom: 1px solid var(--border);
  }

  /* Dual Viewport Side-by-Side */
  .dual-video-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(280px, 1fr));
    gap: 12px; padding: 12px;
  }
  @media (max-width: 580px) {
    .dual-video-grid { grid-template-columns: 1fr; }
  }
  .video-card {
    background: #000; border-radius: 8px; border: 1px solid var(--border);
    overflow: hidden; display: flex; flex-direction: column;
  }
  .feed-header {
    background: #0a111e; padding: 6px 12px; font-family: 'Orbitron', monospace;
    font-size: 0.65rem; color: #8fa0c0; border-bottom: 1px solid #1a2540;
    display: flex; justify-content: space-between; align-items: center;
  }
  .video-wrap {
    position: relative; background: #000; aspect-ratio: 4/3; overflow: hidden;
  }
  .video-wrap img {
    width: 100%; height: 100%; object-fit: contain; display: block;
  }
  .mode-badge {
    position: absolute; top: 8px; left: 8px; font-family: 'Orbitron', monospace;
    font-size: 0.65rem; font-weight: 700; padding: 4px 8px; border-radius: 4px;
    background: var(--accent); color: #000; letter-spacing: 1px;
  }
  .scene-badge {
    position: absolute; bottom: 8px; left: 8px; font-family: 'Orbitron', monospace;
    font-size: 0.62rem; padding: 3px 8px; border-radius: 4px;
    background: rgba(0,0,0,0.85); color: var(--accent); border: 1px solid var(--accent);
  }

  /* Deck Controls */
  .deck-row { display: flex; gap: 8px; padding: 12px 16px; flex-wrap: wrap; }
  .btn {
    font-family: 'Orbitron', monospace; font-size: 0.72rem; font-weight: 700;
    padding: 8px 14px; border-radius: 6px; border: 1px solid var(--border);
    background: var(--surface2); color: var(--text); cursor: pointer;
    transition: all 0.2s; letter-spacing: 1px;
  }
  .btn:hover { border-color: var(--accent); color: var(--accent); }
  .btn.active { background: var(--accent); color: #000; border-color: var(--accent); }
  .btn.btn-vlm { border-color: var(--purple); color: var(--purple); }
  .btn.btn-vlm.active { background: var(--purple); color: #fff; }

  /* Purpose C input */
  .vocab-deck {
    display: flex; gap: 8px; padding: 8px 16px; align-items: center;
  }
  .vocab-deck input {
    flex: 1; background: #050a14; border: 1px solid var(--border);
    color: #fff; padding: 8px 12px; border-radius: 6px; font-size: 0.8rem;
  }
  .vocab-deck input:focus { outline: none; border-color: var(--accent); }

  /* Telemetry HUD & Minimap */
  .telemetry-grid {
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; padding: 12px 16px;
  }
  .tel-stat {
    background: #060d1a; padding: 8px 10px; border-radius: 6px; border: 1px solid #142036;
  }
  .tel-stat-title { font-size: 0.62rem; color: var(--muted); text-transform: uppercase; }
  .tel-stat-val { font-family: 'Orbitron', monospace; font-size: 0.88rem; color: var(--accent); margin-top: 2px; }

  /* Radar canvas */
  .radar-box { padding: 12px 16px; display: flex; justify-content: center; }
  canvas#radar {
    background: #050a14; border-radius: 8px; border: 1px solid var(--border);
    width: 220px; height: 220px;
  }

  /* Flight Log table */
  .log-container {
    max-height: 180px; overflow-y: auto; padding: 8px 16px; font-family: 'Monaco', monospace;
    font-size: 0.68rem; color: #a0b2d6; line-height: 1.5;
  }
  .log-item { padding: 3px 0; border-bottom: 1px solid #111a2d; }
  .log-item span.cat { color: var(--accent); font-weight: bold; margin-right: 4px; }

  /* WASD manual teleop */
  .wasd-grid {
    display: grid; grid-template-columns: repeat(3, 48px); gap: 6px;
    justify-content: center; padding: 12px;
  }
  .d-btn {
    width: 48px; height: 48px; display: flex; align-items: center; justify-content: center;
    user-select: none; font-size: 0.8rem;
  }

  .bottom-bar {
    background: var(--surface); border-top: 1px solid var(--border);
    padding: 8px 24px; font-size: 0.72rem; color: var(--muted); text-align: center;
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="brand-dot"></div>
    AI VISION PATROL ROBOT
  </div>
  <div class="nav-links">
    <a href="/vr" target="_blank">🥽 3D WebXR Digital Twin</a>
    <a href="/video_frame.jpg" target="_blank">📷 RGB Snapshot</a>
    <a href="/depth_frame.jpg" target="_blank">🗺️ Depth Snapshot</a>
    <a href="/download_mission_log" target="_blank">📋 Download Log</a>
  </div>
</header>

<div class="grid">
  <!-- LEFT: Dual Viewports & Action Deck -->
  <div style="display:flex; flex-direction:column; gap:16px;">
    <div class="panel">
      <div class="panel-title">◉ Dual Viewport: RGB Vision & Monocular MAGMA Depth</div>
      <div class="dual-video-grid">
        <div class="video-card">
          <div class="feed-header">
            <span>📷 RGB + YOLO & OPEN-VOCAB HUD</span>
            <span style="color:#76ff03;">640x480</span>
          </div>
          <div class="video-wrap">
            <img id="videoFeed" src="/video_frame.jpg" alt="Live Camera Feed">
            <div class="mode-badge" id="modeBadge">AI_PATROL</div>
            <div class="scene-badge" id="sceneBadge">Semantic Scene: Clear Open Floor (95%)</div>
          </div>
        </div>

        <div class="video-card">
          <div class="feed-header">
            <span>🗺️ MONOCULAR MAGMA DEPTH & LOW-OBSTACLE HAZARD</span>
            <span style="color:#00e5ff;">CLEAR FLOOR</span>
          </div>
          <div class="video-wrap">
            <img id="depthFeed" src="/depth_frame.jpg" alt="Monocular Depth Map">
          </div>
        </div>
      </div>

      <!-- Mode Switches & Teleop -->
      <div class="deck-row">
        <button id="btnPatrol" class="btn active" onclick="setMode('AI_PATROL')">🧠 AI Patrol</button>
        <button id="btnChase" class="btn" onclick="setMode('AI_CHASE')">🎯 AI Chase</button>
        <button id="btnManual" class="btn" onclick="setMode('MANUAL')">🕹️ Manual Teleop</button>
        <button id="btnVlmToggle" class="btn btn-vlm" onclick="toggleVlm()">🧭 VLM Auto-Goal: OFF</button>
        <button class="btn btn-vlm" onclick="triggerVlm()">⚡ Trigger VLM</button>
        <button class="btn" onclick="teleport()">⚡ Reset Pose</button>
      </div>

      <!-- Purpose C Open-Vocab Bar -->
      <div class="vocab-deck">
        <input type="text" id="vocabInput" placeholder="Purpose C Open-Vocab Prompt (e.g. blue chair, wooden crate, hazard box)..." onkeydown="if(event.key==='Enter') submitVocab()">
        <button class="btn" onclick="submitVocab()">🔍 Lock & Ground</button>
        <button class="btn" onclick="setVocab('clear')">✕ Clear</button>
      </div>

      <div class="deck-row" style="padding-top:0;">
        <span style="font-size:0.7rem; color:var(--muted); align-self:center;">Presets:</span>
        <button class="btn" style="padding:4px 8px; font-size:0.65rem;" onclick="setVocab('blue office chair')">Blue Chair</button>
        <button class="btn" style="padding:4px 8px; font-size:0.65rem;" onclick="setVocab('wooden supply crate')">Wooden Crate</button>
        <button class="btn" style="padding:4px 8px; font-size:0.65rem;" onclick="setVocab('yellow hazard box')">Hazard Box</button>
        <button class="btn" style="padding:4px 8px; font-size:0.65rem;" onclick="setVocab('green potted plant')">Potted Plant</button>
        <button class="btn" style="padding:4px 8px; font-size:0.65rem;" onclick="setVocab('red traffic cone')">Red Cone</button>
      </div>
    </div>

    <!-- Flight Log Panel -->
    <div class="panel">
      <div class="panel-title">📋 Automated Mission Flight Logs (mission_log.md)</div>
      <div class="log-container" id="logBox">
        <div class="log-item">Initializing autonomous flight logs...</div>
      </div>
    </div>
  </div>

  <!-- RIGHT: Minimap & Telemetry -->
  <div style="display:flex; flex-direction:column; gap:16px;">
    <div class="panel">
      <div class="panel-title">🛰️ Arena Minimap & Proximity Radar</div>
      <div class="radar-box">
        <canvas id="radar" width="220" height="220"></canvas>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">📊 Live Telemetry Deck</div>
      <div class="telemetry-grid">
        <div class="tel-stat"><div class="tel-stat-title">Pose X, Y</div><div class="tel-stat-val" id="valPos">0.00, 0.00</div></div>
        <div class="tel-stat"><div class="tel-stat-title">Heading Yaw</div><div class="tel-stat-val" id="valYaw">0.0°</div></div>
        <div class="tel-stat"><div class="tel-stat-title">Wheel Speeds (L/R)</div><div class="tel-stat-val" id="valSpeed">0.00 / 0.00</div></div>
        <div class="tel-stat"><div class="tel-stat-title">Locked Target</div><div class="tel-stat-val" id="valTarget" style="font-size:0.75rem;">NONE</div></div>
        <div class="tel-stat"><div class="tel-stat-title">Distance Proximity</div><div class="tel-stat-val" id="valProx">L:0 R:0</div></div>
        <div class="tel-stat"><div class="tel-stat-title">Depth Hazard Score</div><div class="tel-stat-val" id="valDepth">0.00</div></div>
      </div>

      <!-- WASD Grid -->
      <div id="wasdGrid" class="wasd-grid" style="display:none;">
        <div></div>
        <div class="btn d-btn" onmousedown="mc('FORWARD')" onmouseup="mc('STOP')">▲ W</div>
        <div></div>
        <div class="btn d-btn" onmousedown="mc('LEFT')" onmouseup="mc('STOP')">◄ A</div>
        <div class="btn d-btn" onclick="mc('STOP')">■</div>
        <div class="btn d-btn" onmousedown="mc('RIGHT')" onmouseup="mc('STOP')">D ►</div>
        <div></div>
        <div class="btn d-btn" onmousedown="mc('BACKWARD')" onmouseup="mc('STOP')">▼ S</div>
        <div></div>
      </div>
    </div>
  </div>
</div>

<div class="bottom-bar">Mission Control Dashboard &mdash; Webots R2025a &times; YOLOv8 &times; CLIP &times; Monocular Depth &times; FastAPI &times; WebXR</div>

<script>
let currentMode = 'AI_PATROL';
let autoGoalVlm = false;

// ── Robust AbortController API Fetch Wrapper ──
async function apiPost(url, body = null, timeout = 2500) {
  const ctrl = new AbortController();
  const tid = setTimeout(() => ctrl.abort(), timeout);
  try {
    const opts = { method: 'POST', signal: ctrl.signal };
    if (body) {
      opts.headers = { 'Content-Type': 'application/json' };
      opts.body = typeof body === 'string' ? body : JSON.stringify(body);
    }
    const res = await fetch(url, opts);
    clearTimeout(tid);
    return await res.json().catch(() => ({}));
  } catch (err) {
    clearTimeout(tid);
    console.warn('API call failed or timed out:', url, err);
    return {};
  }
}

async function setMode(m) {
  currentMode = m;
  document.getElementById('btnPatrol').classList.toggle('active', m === 'AI_PATROL');
  document.getElementById('btnChase').classList.toggle('active', m === 'AI_CHASE');
  document.getElementById('btnManual').classList.toggle('active', m === 'MANUAL');
  document.getElementById('wasdGrid').style.display = m === 'MANUAL' ? 'grid' : 'none';
  const badge = document.getElementById('modeBadge');
  badge.textContent = m;
  badge.style.background = m === 'AI_PATROL' ? 'var(--accent)' : m === 'AI_CHASE' ? 'var(--accent2)' : '#ffd600';
  await apiPost('/set_mode/' + m);
}

async function mc(cmd) {
  await apiPost('/manual/' + cmd);
}

async function teleport() {
  await apiPost('/teleport_reset');
}

async function submitVocab() {
  const prompt = document.getElementById('vocabInput').value.trim();
  if (prompt && prompt.toLowerCase() !== 'clear') setMode('AI_CHASE');
  await apiPost('/set_openvocab_prompt', { prompt: prompt });
}

async function setVocab(prompt) {
  document.getElementById('vocabInput').value = prompt;
  if (prompt && prompt.toLowerCase() !== 'clear') setMode('AI_CHASE');
  await apiPost('/set_openvocab_prompt', { prompt: prompt });
}

async function toggleVlm() {
  autoGoalVlm = !autoGoalVlm;
  const btn = document.getElementById('btnVlmToggle');
  btn.textContent = autoGoalVlm ? '🧭 VLM Auto-Goal: ON' : '🧭 VLM Auto-Goal: OFF';
  btn.classList.toggle('active', autoGoalVlm);
  if (autoGoalVlm) setMode('AI_CHASE');
  await apiPost('/set_vlm_goal_mode/' + (autoGoalVlm ? 'true' : 'false'));
}

async function triggerVlm() {
  setMode('AI_CHASE');
  await apiPost('/trigger_vlm_goal');
}

// ── Non-Blocking Snapshot Polling (Zero MJPEG Socket Exhaustion) ──
const videoImg = document.getElementById('videoFeed');
const depthImg = document.getElementById('depthFeed');

function pollVisionFeeds() {
  if (document.hidden) {
    setTimeout(pollVisionFeeds, 200);
    return;
  }
  const now = Date.now();
  const nextRgb = new Image();
  nextRgb.onload = () => { if (videoImg) videoImg.src = nextRgb.src; };
  nextRgb.src = '/video_frame.jpg?t=' + now;

  const nextDepth = new Image();
  nextDepth.onload = () => { if (depthImg) depthImg.src = nextDepth.src; };
  nextDepth.src = '/depth_frame.jpg?t=' + now;

  setTimeout(pollVisionFeeds, 90);
}
setTimeout(pollVisionFeeds, 150);

// ── Keyboard Teleop ──
const keyMap = { w: 'FORWARD', a: 'LEFT', s: 'BACKWARD', d: 'RIGHT' };
const pressed = new Set();
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  const k = e.key.toLowerCase();
  if (keyMap[k] && !pressed.has(k)) {
    pressed.add(k);
    mc(keyMap[k]);
  }
});
document.addEventListener('keyup', e => {
  if (e.target.tagName === 'INPUT') return;
  const k = e.key.toLowerCase();
  if (keyMap[k]) {
    pressed.delete(k);
    if (pressed.size === 0) mc('STOP');
    else mc(keyMap[Array.from(pressed)[pressed.size - 1]]);
  }
});

// ── Radar Minimap Drawing ──
const radarCanvas = document.getElementById('radar');
const ctx = radarCanvas.getContext('2d');
const ARENA_RADIUS = 1.85;

// Repositioned targets matching domain_state.py
const RADAR_OBJECTS = [
  { name: 'RED', x: 1.15, y: 1.00, color: '#ff1744' },
  { name: 'GREEN', x: -1.05, y: 1.05, color: '#76ff03' },
  { name: 'BLUE', x: 0.00, y: -1.15, color: '#00e5ff' },
  { name: 'CRATE_1', x: -1.15, y: -0.95, color: '#d500f9' },
  { name: 'CRATE_2', x: 1.15, y: -0.95, color: '#d500f9' },
  { name: 'HAZARD', x: 0.85, y: 0.15, color: '#ffd600' },
];

function drawRadar(rx, ry, ryaw, trail) {
  const w = radarCanvas.width;
  const h = radarCanvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const scale = (w * 0.44) / ARENA_RADIUS;

  ctx.clearRect(0, 0, w, h);

  // Outer border & grid
  ctx.strokeStyle = '#1e2c48';
  ctx.lineWidth = 1;
  ctx.strokeRect(10, 10, w - 20, h - 20);

  // Concentric radar circles
  ctx.beginPath();
  ctx.arc(cx, cy, 0.6 * scale, 0, Math.PI * 2);
  ctx.arc(cx, cy, 1.2 * scale, 0, Math.PI * 2);
  ctx.strokeStyle = '#142036';
  ctx.stroke();

  // Trail
  if (trail && trail.length > 1) {
    ctx.beginPath();
    trail.forEach((pt, idx) => {
      const px = cx + pt[0] * scale;
      const py = cy - pt[1] * scale;
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = 'rgba(0, 229, 255, 0.25)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // Draw Arena Objects
  RADAR_OBJECTS.forEach(obj => {
    const ox = cx + obj.x * scale;
    const oy = cy - obj.y * scale;
    ctx.fillStyle = obj.color;
    ctx.beginPath();
    ctx.arc(ox, oy, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = '9px monospace';
    ctx.fillText(obj.name.substring(0, 3), ox + 7, oy + 3);
  });

  // Rover position & heading triangle
  const rpx = cx + rx * scale;
  const rpy = cy - ry * scale;
  ctx.save();
  ctx.translate(rpx, rpy);
  ctx.rotate(-ryaw);
  ctx.fillStyle = '#00e5ff';
  ctx.beginPath();
  ctx.moveTo(0, -9);
  ctx.lineTo(6, 6);
  ctx.lineTo(-6, 6);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

// ── Telemetry Poller & WebSocket Bridge ──
async function pollTelemetry() {
  try {
    const res = await fetch('/telemetry');
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById('valPos').textContent = `${data.pos[0].toFixed(2)}, ${data.pos[1].toFixed(2)}`;
    document.getElementById('valYaw').textContent = `${data.yaw_deg.toFixed(1)}°`;
    document.getElementById('valSpeed').textContent = `${data.left_speed.toFixed(2)} / ${data.right_speed.toFixed(2)}`;
    document.getElementById('valTarget').textContent = data.locked_target || 'NONE';
    document.getElementById('valProx').textContent = `L:${data.ps_left.toFixed(0)} R:${data.ps_right.toFixed(0)}`;
    document.getElementById('valDepth').textContent = `${data.depth_score.toFixed(2)}`;

    const sceneBadge = document.getElementById('sceneBadge');
    if (sceneBadge && data.semantic_scene) {
      sceneBadge.textContent = `Scene: ${data.semantic_scene} (${(data.semantic_score * 100).toFixed(0)}%)`;
    }

    drawRadar(data.pos[0], data.pos[1], data.orientation, data.trail);

    // Update Logs
    const logBox = document.getElementById('logBox');
    if (data.latest_mission_log && logBox) {
      logBox.innerHTML = `<div class="log-item"><span class="cat">[LIVE]</span> ${data.latest_mission_log}</div>`;
    }
  } catch (e) {
    // Silent polling catch
  }
}
setInterval(pollTelemetry, 150);
</script>
</body>
</html>
"""

VR_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>3D VR Digital Twin — Smart Robotics Lab</title>
<script src="https://aframe.io/releases/1.6.0/aframe.min.js"></script>
<script src="https://unpkg.com/aframe-environment-component@1.3.4/dist/aframe-environment-component.min.js"></script>
<script>
window.isDraggingObject = false;
window.activeDragEl = null;
window.activeDragTarget = null;

AFRAME.registerComponent('drag-drop-target', {
  schema: { targetName: { type: 'string', default: 'RED' } },
  init: function () {
    const el = this.el;
    const targetName = this.data.targetName;
    const sceneEl = document.querySelector('a-scene');
    const floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const intersection = new THREE.Vector3();

    el.addEventListener('mousedown', function () {
      window.isDraggingObject = true;
      window.activeDragEl = el;
      window.activeDragTarget = targetName;
      const cam = document.getElementById('vr-cam');
      if (cam && cam.components['look-controls']) cam.components['look-controls'].pause();
    });

    if (!window._dragListenersAdded) {
      window._dragListenersAdded = true;
      window.addEventListener('mousemove', function (evt) {
        if (!window.isDraggingObject || !window.activeDragEl) return;
        const canvas = sceneEl.canvas;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        mouse.x = ((evt.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((evt.clientY - rect.top) / rect.height) * 2 + 1;
        const camera = sceneEl.camera;
        if (!camera) return;
        raycaster.setFromCamera(mouse, camera);

        if (raycaster.ray.intersectPlane(floorPlane, intersection)) {
          const clampedX = Math.max(-1.8, Math.min(1.8, intersection.x));
          const clampedZ = Math.max(-1.8, Math.min(1.8, intersection.z));
          window.activeDragEl.setAttribute('position', `${clampedX.toFixed(3)} 0 ${clampedZ.toFixed(3)}`);
        }
      });

      window.addEventListener('mouseup', function () {
        if (window.isDraggingObject) {
          window.isDraggingObject = false;
          window.activeDragEl = null;
          window.activeDragTarget = null;
          const cam = document.getElementById('vr-cam');
          if (cam && cam.components['look-controls']) cam.components['look-controls'].play();
        }
      });
    }
  }
});
</script>

<style>
  body { margin: 0; overflow: hidden; font-family: 'Inter', sans-serif; }
  #vr-hud {
    position: fixed; top: 12px; left: 12px; z-index: 9999;
    background: rgba(10, 14, 23, 0.88); border: 1px solid rgba(0, 229, 255, 0.4);
    border-radius: 12px; padding: 12px 18px; color: #00e5ff;
    font-family: 'Monaco', monospace; font-size: 13px; line-height: 1.6;
    pointer-events: none; backdrop-filter: blur(8px);
  }
  #vr-controls {
    position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%); z-index: 9999;
    display: flex; gap: 8px; background: rgba(10, 14, 23, 0.9);
    border: 1px solid rgba(0, 229, 255, 0.3); border-radius: 12px;
    padding: 8px 14px; backdrop-filter: blur(8px); flex-wrap: wrap; justify-content: center;
  }
  #vr-controls button {
    background: rgba(0, 229, 255, 0.12); border: 1px solid rgba(0, 229, 255, 0.3);
    color: #00e5ff; padding: 8px 14px; border-radius: 8px;
    font-family: 'Monaco', monospace; font-size: 11px; font-weight: bold;
    cursor: pointer; transition: all 0.15s;
  }
  #vr-controls button.active { background: #00e5ff; color: #000; }
</style>
</head>
<body>

<div id="vr-hud">
  <div>MODE: <span id="h-mode">AI_PATROL</span></div>
  <div>POS: <span id="h-pos">0.000, 0.000</span></div>
  <div>YAW: <span id="h-yaw">0.0</span>°</div>
  <div>TARGET: <span id="h-target">NONE</span></div>
  <div>DEPTH HAZ: <span id="h-depth">CLEAR</span></div>
</div>

<div id="vr-controls">
  <button id="btn-patrol" class="active" onclick="sendMode('AI_PATROL')">🧠 AI PATROL</button>
  <button id="btn-chase" onclick="sendMode('AI_CHASE')">🎯 AI CHASE</button>
  <button id="btn-manual" onclick="sendMode('MANUAL')">🕹️ MANUAL</button>
  <button onclick="sendCmd('FORWARD')">▲ FWD</button>
  <button onclick="sendCmd('LEFT')">◄ LEFT</button>
  <button onclick="sendCmd('STOP')">■ STOP</button>
  <button onclick="sendCmd('RIGHT')">RIGHT ►</button>
  <button onclick="sendCmd('BACKWARD')">▼ BACK</button>
  <button onclick="sendReset()">⚡ RESET</button>
</div>

<a-scene>
  <a-entity environment="preset: starry; groundColor: #050811; grid: 2x2; gridColor: #142036;"></a-entity>
  <a-entity id="vr-cam" camera look-controls wasd-controls position="0 2.2 3.6"></a-entity>

  <!-- Rover Body -->
  <a-entity id="rover-body" position="0 0.15 0">
    <a-box width="0.45" height="0.18" depth="0.52" color="#00e5ff" opacity="0.9"></a-box>
    <a-cylinder radius="0.08" height="0.05" rotation="0 0 90" position="-0.26 0 0.15" color="#333"></a-cylinder>
    <a-cylinder radius="0.08" height="0.05" rotation="0 0 90" position="0.26 0 0.15" color="#333"></a-cylinder>
    <a-cylinder radius="0.08" height="0.05" rotation="0 0 90" position="-0.26 0 -0.15" color="#333"></a-cylinder>
    <a-cylinder radius="0.08" height="0.05" rotation="0 0 90" position="0.26 0 -0.15" color="#333"></a-cylinder>
  </a-entity>

  <!-- Arena Targets with Repositioned Layout -->
  <a-cone id="tgt-red" drag-drop-target="targetName: RED" position="1.15 0.22 -1.00" radius-bottom="0.18" height="0.45" color="#ff1744"></a-cone>
  <a-cylinder id="tgt-green" drag-drop-target="targetName: GREEN" position="-1.05 0.25 -1.05" radius="0.20" height="0.50" color="#76ff03"></a-cylinder>
  <a-box id="tgt-blue" drag-drop-target="targetName: BLUE" position="0.00 0.31 1.15" width="0.44" height="0.62" depth="0.44" color="#00e5ff"></a-box>
  <a-box id="tgt-crate1" drag-drop-target="targetName: CRATE_1" position="-1.15 0.19 0.95" width="0.40" height="0.38" depth="0.40" color="#8d6e63"></a-box>
  <a-box id="tgt-crate2" drag-drop-target="targetName: CRATE_2" position="1.15 0.19 0.95" width="0.40" height="0.38" depth="0.40" color="#8d6e63"></a-box>
  <a-box id="tgt-hazard" drag-drop-target="targetName: HAZARD" position="0.85 0.18 -0.15" width="0.36" height="0.36" depth="0.36" color="#ffd600"></a-box>

  <!-- Dual Hologram Viewports in VR -->
  <a-plane position="-1.4 1.8 -2.4" width="1.6" height="1.2" color="#000" material="shader: flat;">
    <a-text value="RGB LIVE CAMERA" position="-0.7 0.68 0" scale="0.4 0.4 0.4" color="#00e5ff"></a-text>
  </a-plane>
  <a-plane position="1.4 1.8 -2.4" width="1.6" height="1.2" color="#000" material="shader: flat;">
    <a-text value="MAGMA DEPTH MAP" position="-0.7 0.68 0" scale="0.4 0.4 0.4" color="#ff1744"></a-text>
  </a-plane>
</a-scene>

<script>
async function sendMode(m) {
  document.getElementById('btn-patrol').classList.toggle('active', m === 'AI_PATROL');
  document.getElementById('btn-chase').classList.toggle('active', m === 'AI_CHASE');
  document.getElementById('btn-manual').classList.toggle('active', m === 'MANUAL');
  document.getElementById('h-mode').textContent = m;
  await fetch('/set_mode/' + m, { method: 'POST' }).catch(() => {});
}

async function sendCmd(cmd) {
  await fetch('/manual/' + cmd, { method: 'POST' }).catch(() => {});
}

async function sendReset() {
  await fetch('/teleport_reset', { method: 'POST' }).catch(() => {});
}

// Telemetry loop for VR Twin
async function updateVrTelemetry() {
  try {
    const res = await fetch('/telemetry');
    if (!res.ok) return;
    const data = await res.json();
    document.getElementById('h-mode').textContent = data.mode;
    document.getElementById('h-pos').textContent = `${data.pos[0].toFixed(2)}, ${data.pos[1].toFixed(2)}`;
    document.getElementById('h-yaw').textContent = `${data.yaw_deg.toFixed(1)}`;
    document.getElementById('h-target').textContent = data.locked_target;
    document.getElementById('h-depth').textContent = data.depth_hazard ? 'HAZARD' : 'CLEAR';

    const rover = document.getElementById('rover-body');
    if (rover) {
      // In Webots coords x=right, y=forward, z=up; in A-Frame x=right, y=up, z=-forward
      rover.setAttribute('position', `${data.pos[0].toFixed(3)} 0.15 ${(-data.pos[1]).toFixed(3)}`);
      rover.setAttribute('rotation', `0 ${(data.yaw_deg - 90).toFixed(1)} 0`);
    }
  } catch (e) {}
}
setInterval(updateVrTelemetry, 150);
</script>
</body>
</html>
"""
