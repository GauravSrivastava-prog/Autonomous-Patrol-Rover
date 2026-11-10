"""
app.py — Pure FastAPI Local Engine & Full-Bleed GTA 3D Mission Control HUD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Architecture:
  1. Pure FastAPI + Uvicorn Local-First Architecture (Zero Gradio Bloat)
  2. 1-Command Evaluator Launch via `uvx` / `pipx` (`patrol-rover`)
  3. Full-Bleed 100vw x 100vh Widescreen 3D GTA Chase Viewport
  4. Glassmorphism HUD Overlay (Circular Minimap, Telemetry, Mode Dock & D-Pad)
  5. Real-Time Hardware-Accelerated Local AI Telemetry (Apple MPS / CUDA / CPU)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import time
import argparse
import threading
import webbrowser
from typing import Optional, Dict, Any

from fastapi import FastAPI, Response, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from core_engine import RobotController, ControlMode, ARENA_OBJECTS

PORT = int(os.environ.get("PORT", 7860))

# ──────────────────────────────────────────────────────────────────────────────
# 1. Initialize Robotics Engine (30Hz Background Physics & 3D HD Rendering)
# ──────────────────────────────────────────────────────────────────────────────
controller = RobotController()
controller.start()

# ──────────────────────────────────────────────────────────────────────────────
# 2. FastAPI Application Setup & Streaming Routes
# ──────────────────────────────────────────────────────────────────────────────
api = FastAPI(
    title="Autonomous Patrol Rover — GTA Mission Deck",
    description="Local-First 3D Robotics Platform with Hardware-Accelerated Multimodal Vision",
    version="2.0.0"
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 30 FPS MJPEG Stream Generators ──
def stream_mjpeg(feed_fn):
    """Generates continuous multipart/x-mixed-replace MJPEG stream at ~30 FPS."""
    while True:
        data = feed_fn()
        if data:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        time.sleep(0.033)

@api.get("/video_feed")
def video_feed():
    return StreamingResponse(
        stream_mjpeg(lambda: controller.latest_rgb),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@api.get("/depth_feed")
def depth_feed():
    return StreamingResponse(
        stream_mjpeg(lambda: controller.latest_depth),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ── Snapshot Endpoints ──
@api.get("/video_frame.jpg")
def get_video_frame():
    return Response(content=controller.latest_rgb, media_type="image/jpeg")

@api.get("/depth_frame.jpg")
def get_depth_frame():
    return Response(content=controller.latest_depth, media_type="image/jpeg")

# ── REST Control & Telemetry APIs ──
@api.get("/telemetry")
def get_telemetry():
    return controller.get_telemetry()

@api.post("/set_mode/{mode}")
def set_mode(mode: str):
    try:
        m = mode.upper()
        controller.set_mode(ControlMode(m))
        return {"status": "ok", "mode": m}
    except ValueError:
        return JSONResponse({"status": "error", "message": f"Invalid mode: {mode}"}, status_code=400)

@api.post("/set_view/{view}")
def set_view(view: str):
    v = view.upper()
    controller.set_view_mode(v)
    return {"status": "ok", "view": v}

@api.post("/manual/{cmd}")
def manual_move(cmd: str):
    c = cmd.upper()
    controller.set_manual(c)
    return {"status": "ok", "cmd": c}

@api.post("/set_target")
async def set_target(request: Request, prompt: Optional[str] = None):
    p = prompt
    if not p:
        try:
            body = await request.json()
            p = body.get("prompt", "")
        except Exception:
            pass
    if p:
        controller.set_chase_prompt(p)
    return {"status": "ok", "prompt": p}

@api.post("/reset")
def reset_rover():
    controller.reset_teleport()
    return {"status": "ok"}

@api.get("/mission_logs")
def get_mission_logs():
    return {
        "status": "ok",
        "logs": f"[{time.strftime('%H:%M:%S')}] {controller.state.status_text}",
        "total": 1,
    }

# ──────────────────────────────────────────────────────────────────────────────
# 3. Full-Bleed GTA-Style Mission HUD (Glassmorphism / HTML5 / CSS3 / Vanilla JS)
# ──────────────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Autonomous Patrol Rover — GTA Mission Control</title>
  <style>
    :root {
      --bg: #070a10;
      --card-bg: rgba(11, 18, 32, 0.78);
      --card-border: rgba(0, 229, 255, 0.28);
      --accent: #00e5ff;
      --accent-glow: rgba(0, 229, 255, 0.45);
      --accent-warn: #ffd600;
      --accent-red: #ff1744;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; user-select: none; -webkit-user-select: none; }
    html, body {
      margin: 0; padding: 0; width: 100vw; height: 100vh;
      overflow: hidden; background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    }

    /* 1. Full-Bleed 100vw x 100vh 3D Chase Viewport */
    #viewport-container {
      position: absolute; top: 0; left: 0; width: 100vw; height: 100vh;
      overflow: hidden; background: #000; z-index: 1;
    }
    #rgbFeed {
      width: 100%; height: 100%; object-fit: cover;
      position: absolute; top: 0; left: 0; z-index: 1; display: block;
    }

    /* 2. Glassmorphism HUD Overlay Layer (z-index: 10) */
    .hud-layer {
      position: absolute; top: 0; left: 0; width: 100vw; height: 100vh;
      pointer-events: none; z-index: 10;
      display: grid; grid-template-rows: auto 1fr auto;
      padding: 16px;
    }
    .interactive { pointer-events: auto; }

    .frosted {
      background: var(--card-bg);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.65), inset 0 0 12px rgba(0, 229, 255, 0.06);
    }

    /* Top HUD Elements */
    .top-bar {
      display: flex; justify-content: space-between; align-items: flex-start;
      gap: 16px;
    }

    /* Top-Left: Depth Map PiP + Neural Accelerator Telemetry */
    .top-left-box {
      display: flex; flex-direction: column; gap: 8px; width: 280px;
    }
    .pip-container {
      position: relative; width: 220px; aspect-ratio: 16/9;
      border-radius: 8px; overflow: hidden; border: 2px solid rgba(255, 145, 0, 0.75);
      box-shadow: 0 4px 16px rgba(255, 145, 0, 0.35); background: #000;
    }
    .pip-container img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .pip-tag {
      position: absolute; top: 4px; left: 6px; font-size: 9px;
      font-weight: 800; color: #ff9100; letter-spacing: 0.5px;
      text-shadow: 0 1px 3px #000;
    }
    .view-toggle-btn {
      position: absolute; bottom: 4px; right: 4px;
      background: rgba(0, 0, 0, 0.75); border: 1px solid var(--accent);
      color: var(--accent); font-size: 10px; font-weight: 700;
      padding: 3px 6px; border-radius: 4px; cursor: pointer;
    }
    .view-toggle-btn:hover { background: var(--accent); color: #000; }

    .telemetry-card {
      padding: 10px; width: 220px;
    }
    .badge-pill {
      font-size: 9px; font-weight: 800; padding: 3px 7px; border-radius: 12px;
      background: rgba(0, 229, 255, 0.15); border: 1px solid var(--accent);
      color: var(--accent); letter-spacing: 0.5px; display: inline-block; margin-bottom: 6px;
    }
    .vlm-row {
      display: flex; align-items: center; justify-content: space-between;
      font-size: 9px; font-family: monospace; margin-bottom: 3px;
    }
    .vlm-bar-bg {
      flex: 1; height: 4px; background: rgba(255, 255, 255, 0.1);
      border-radius: 2px; overflow: hidden; margin: 0 6px;
    }
    .vlm-bar-fill {
      height: 100%; width: 5%; background: var(--accent);
      transition: width 0.3s ease;
    }

    /* Top-Right: Mode Dock & Search */
    .top-right-box {
      display: flex; flex-direction: column; align-items: flex-end; gap: 8px;
    }
    .mode-dock {
      display: flex; gap: 6px; padding: 6px;
    }
    .pill-btn {
      background: rgba(255, 255, 255, 0.06); color: var(--text);
      border: 1px solid rgba(255, 255, 255, 0.18); padding: 8px 14px;
      border-radius: 20px; cursor: pointer; font-size: 11px; font-weight: 700;
      transition: all 0.15s;
    }
    .pill-btn:hover { background: rgba(0, 229, 255, 0.22); border-color: var(--accent); }
    .pill-btn.active {
      background: var(--accent) !important; color: #000 !important;
      border-color: var(--accent) !important; box-shadow: 0 0 12px var(--accent-glow);
    }

    .search-box {
      display: flex; gap: 6px; padding: 6px; width: 340px;
    }
    .search-box input {
      flex: 1; background: rgba(0, 0, 0, 0.45); border: 1px solid var(--card-border);
      color: #fff; padding: 8px 12px; border-radius: 6px; font-size: 12px; outline: none;
    }
    .search-box input:focus { border-color: var(--accent); }
    .search-box button {
      background: var(--accent); color: #000; border: none; font-weight: 800;
      padding: 0 16px; border-radius: 6px; cursor: pointer; font-size: 11px;
    }

    /* Bottom HUD Elements */
    .bottom-bar {
      display: flex; justify-content: space-between; align-items: flex-end;
      gap: 16px;
    }

    /* Bottom-Left: Circular 240px GTA Radar Minimap */
    .radar-box {
      display: flex; flex-direction: column; align-items: center; gap: 6px;
    }
    #radarCanvas {
      width: 220px; height: 220px; border-radius: 50%;
      background: radial-gradient(circle, rgba(10, 18, 30, 0.90) 0%, rgba(5, 8, 14, 0.96) 100%);
      border: 2px solid var(--accent);
      box-shadow: 0 0 20px rgba(0, 229, 255, 0.35), inset 0 0 24px rgba(0, 0, 0, 0.85);
      display: block;
    }
    .status-pill {
      padding: 5px 12px; font-family: monospace; font-size: 11px;
      color: var(--accent); font-weight: 600; letter-spacing: 0.5px;
      text-align: center; white-space: nowrap;
    }

    /* Bottom-Right: Tactile D-Pad & WASD Indicators */
    .controls-box {
      padding: 10px; width: 175px;
    }
    .d-pad {
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px;
    }
    .d-btn {
      aspect-ratio: 1; background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 8px;
      color: #fff; font-size: 12px; font-weight: 800; cursor: pointer;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      transition: all 0.1s;
    }
    .d-btn span { font-size: 8px; opacity: 0.7; }
    .d-btn:hover { background: rgba(0, 229, 255, 0.25); border-color: var(--accent); }
    .d-btn.active {
      background: var(--accent) !important; color: #000 !important;
      box-shadow: 0 0 12px var(--accent-glow);
    }
  </style>
</head>
<body>

  <!-- Full-Bleed 100vw x 100vh 3D GTA Chase Viewport -->
  <div id="viewport-container">
    <img id="rgbFeed" src="/video_feed" alt="3D Live POV Viewport" />
  </div>

  <!-- Glassmorphism Floating HUD Layer -->
  <div class="hud-layer">
    
    <!-- Top Bar -->
    <div class="top-bar">
      <!-- Top Left: Depth Map PiP + Hardware Accelerator Telemetry -->
      <div class="top-left-box interactive">
        <div class="pip-container">
          <span class="pip-tag">MONOCULAR DEPTH (MAGMA)</span>
          <img id="depthFeed" src="/depth_feed" alt="Monocular Depth Feed" />
          <button id="btnViewToggle" class="view-toggle-btn" onclick="(window.toggleView||toggleView)()">🎥 POV: 3RD PERSON</button>
        </div>

        <div class="frosted telemetry-card">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span id="hwBadge" class="badge-pill">🍎 APPLE MPS ACCELERATED</span>
            <span id="fpsCounter" style="font-size:10px; font-family:monospace; color:var(--text-muted);">30.0 FPS</span>
          </div>
          <div style="font-size:10px; font-weight:700; color:var(--accent); margin-bottom:2px; display:flex; justify-content:space-between;">
            <span>MULTIMODAL VLM MATCH</span>
            <span id="vlmActiveTarget" style="color:#fff;">--</span>
          </div>
          <div id="yoloCount" style="font-size:9px; font-family:monospace; color:#38bdf8; margin-bottom:5px;">🎯 YOLO DETECTIONS: 0</div>
          <div id="vlmScoresList">
            <!-- Dynamic similarity bars -->
          </div>
        </div>
      </div>

      <!-- Top Right: Mode Selection Dock & Natural Language Search -->
      <div class="top-right-box interactive">
        <div class="frosted mode-dock">
          <button id="btnPatrol" class="pill-btn active" onclick="(window.setMode||setMode)('AI_PATROL')">🧠 AI Patrol</button>
          <button id="btnChase" class="pill-btn" onclick="(window.setMode||setMode)('AI_CHASE')">🎯 AI Chase</button>
          <button id="btnManual" class="pill-btn" onclick="(window.setMode||setMode)('MANUAL')">🕹️ Manual</button>
          <button id="btnReset" class="pill-btn" onclick="(window.postApi||postApi)('/reset')">🔄 Reset (0,0)</button>
        </div>

        <div class="frosted search-box">
          <input id="promptInput" type="text" placeholder="Seek target (e.g. blue chair, wooden crate)..." />
          <button id="btnSeek" onclick="(window.sendTarget||sendTarget)()">Seek</button>
        </div>
      </div>
    </div>

    <!-- Center Viewport Spacer -->
    <div></div>

    <!-- Bottom Bar -->
    <div class="bottom-bar">
      <!-- Bottom-Left: Circular 240px GTA Radar Minimap -->
      <div class="radar-box interactive">
        <canvas id="radarCanvas" width="220" height="220"></canvas>
        <div id="telemetryStatus" class="frosted status-pill">
          POS: (0.00, 0.00) | YAW: 0.0° | SPEED: 0.0/0.0 | INITIALIZING...
        </div>
      </div>

      <!-- Bottom-Right: Tactile Manual D-Pad -->
      <div class="controls-box frosted interactive">
        <div class="d-pad">
          <div></div>
          <button id="dpadFwd" class="d-btn">▲<span>W</span></button>
          <div></div>
          <button id="dpadLeft" class="d-btn">◄<span>A</span></button>
          <button id="dpadStop" class="d-btn" style="background:rgba(239,68,68,0.3); border-color:#ef4444;">■<span>SPC</span></button>
          <button id="dpadRight" class="d-btn">►<span>D</span></button>
          <div></div>
          <button id="dpadBack" class="d-btn">▼<span>S</span></button>
          <div></div>
        </div>
      </div>
    </div>

  </div>

  <script>
    // ── Global Scope Core Actions ──
    window.postApi = function(url) {
      return fetch(url, { method: 'POST' }).catch(err => console.warn('POST failed:', url, err));
    };

    window.setMode = function(m) {
      window.postApi('/set_mode/' + m);
      document.querySelectorAll('.mode-dock button').forEach(b => b.classList.remove('active'));
      if (m === 'AI_PATROL') document.getElementById('btnPatrol')?.classList.add('active');
      if (m === 'AI_CHASE') document.getElementById('btnChase')?.classList.add('active');
      if (m === 'MANUAL') document.getElementById('btnManual')?.classList.add('active');
    };

    window.setView = function(v) {
      window.postApi('/set_view/' + v);
      const btn = document.getElementById('btnViewToggle');
      if (btn) btn.textContent = v === 'TPP' ? '🎥 POV: 3RD PERSON' : '🎥 POV: 1ST PERSON';
    };

    window.toggleView = function() {
      curView = curView === 'TPP' ? 'FPP' : 'TPP';
      window.setView(curView);
    };

    window.manual = function(cmd) {
      window.postApi('/manual/' + cmd);
    };

    window.sendTarget = function() {
      const input = document.getElementById('promptInput');
      if (!input) return;
      const q = input.value.trim();
      if (q) {
        fetch('/set_target?prompt=' + encodeURIComponent(q), { method: 'POST' });
        window.setMode('AI_CHASE');
      }
    };

    // Local scope declarations
    function postApi(url) { return window.postApi(url); }
    function setMode(m) { return window.setMode(m); }
    function setView(v) { return window.setView(v); }
    function toggleView() { return window.toggleView(); }
    function manual(cmd) { return window.manual(cmd); }
    function sendTarget() { return window.sendTarget(); }

    // ── Radar Minimap Drawing Routine ──
    const ARENA_RADAR_LANDMARKS = [
      { name: 'RED', x: 1.15, y: 1.00, color: '#ff1744' },
      { name: 'GRN', x: -1.05, y: 1.05, color: '#00e676' },
      { name: 'BLU', x: 0.00, y: -1.15, color: '#00e5ff' },
      { name: 'CR1', x: -1.15, y: -0.95, color: '#e040fb' },
      { name: 'CR2', x: 1.15, y: -0.95, color: '#e040fb' },
      { name: 'HAZ', x: 0.85, y: 0.15, color: '#ffd600' },
    ];

    function drawRadar(rx = 0, ry = 0, ryaw = 0.35, trail = []) {
      const canvas = document.getElementById('radarCanvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, 220, 220);

      const cx = 110, cy = 110, scale = 48.0;

      // Concentric Range Rings
      ctx.strokeStyle = 'rgba(0, 229, 255, 0.22)';
      ctx.lineWidth = 1;
      [0.6, 1.2, 1.85].forEach(r => {
        ctx.beginPath();
        ctx.arc(cx, cy, r * scale, 0, Math.PI * 2);
        ctx.stroke();
      });

      // Crosshairs
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.beginPath();
      ctx.moveTo(cx - 1.85 * scale, cy); ctx.lineTo(cx + 1.85 * scale, cy);
      ctx.moveTo(cx, cy - 1.85 * scale); ctx.lineTo(cx, cy + 1.85 * scale);
      ctx.stroke();

      // Dashed 4m Arena Boundary
      ctx.strokeStyle = 'rgba(0, 229, 255, 0.35)';
      ctx.setLineDash([4, 4]);
      const bSize = 1.85 * scale;
      ctx.strokeRect(cx - bSize, cy - bSize, bSize * 2, bSize * 2);
      ctx.setLineDash([]);

      // Forward 60-degree Camera FOV Sector Cone
      ctx.save();
      ctx.translate(cx + rx * scale, cy - ry * scale);
      ctx.rotate(-ryaw);
      ctx.fillStyle = 'rgba(0, 229, 255, 0.14)';
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.arc(0, 0, 52, -Math.PI / 6 - Math.PI / 2, Math.PI / 6 - Math.PI / 2);
      ctx.closePath();
      ctx.fill();
      ctx.restore();

      // Landmarks
      ARENA_RADAR_LANDMARKS.forEach(lm => {
        const px = cx + lm.x * scale;
        const py = cy - lm.y * scale;
        ctx.fillStyle = lm.color;
        ctx.beginPath(); ctx.arc(px, py, 4.5, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '8px monospace';
        ctx.fillText(lm.name, px + 6, py + 3);
      });

      // Historical Path Trail
      if (trail && trail.length > 1) {
        ctx.strokeStyle = 'rgba(0, 229, 255, 0.45)';
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        trail.forEach((pt, i) => {
          const tx = cx + pt[0] * scale, ty = cy - pt[1] * scale;
          if (i === 0) ctx.moveTo(tx, ty); else ctx.lineTo(tx, ty);
        });
        ctx.stroke();
      }

      // Rover Heading Arrow & Dot
      ctx.save();
      ctx.translate(cx + rx * scale, cy - ry * scale);
      ctx.rotate(-ryaw);
      ctx.fillStyle = '#00e5ff';
      ctx.beginPath();
      ctx.moveTo(0, -10); ctx.lineTo(6, 6); ctx.lineTo(-6, 6);
      ctx.closePath();
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.beginPath(); ctx.arc(0, 0, 2, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    }

    // Initial render of minimap
    drawRadar(0, 0, 0.35, []);

    // ── Live Telemetry Poller (Calm 600ms Cycle) ──
    let curView = 'TPP';
    async function pollTelemetry() {
      try {
        const res = await fetch('/telemetry');
        if (res.ok) {
          const d = await res.json();
          
          // Status Pill
          const statusEl = document.getElementById('telemetryStatus');
          if (statusEl) {
            statusEl.textContent = `POS: (${d.x.toFixed(2)}, ${d.y.toFixed(2)}) | YAW: ${d.yaw_deg}° | SPD: ${d.speed_l}/${d.speed_r} | ${d.status.toUpperCase()}`;
          }

          // Hardware & FPS Badge
          const hwEl = document.getElementById('hwBadge');
          if (hwEl && d.neural_badge) hwEl.textContent = d.neural_badge;
          const fpsEl = document.getElementById('fpsCounter');
          if (fpsEl && d.fps) fpsEl.textContent = `${d.fps.toFixed(1)} FPS`;
          const yoloCountEl = document.getElementById('yoloCount');
          if (yoloCountEl && d.yolo_detections !== undefined) {
            yoloCountEl.textContent = `🎯 YOLO DETECTIONS: ${d.yolo_detections.length} IN VIEW`;
          }

          // Mode Button Sync
          if (d.mode) {
            document.querySelectorAll('.mode-dock button').forEach(b => b.classList.remove('active'));
            if (d.mode === 'AI_PATROL') document.getElementById('btnPatrol')?.classList.add('active');
            else if (d.mode === 'AI_CHASE') document.getElementById('btnChase')?.classList.add('active');
            else if (d.mode === 'MANUAL') document.getElementById('btnManual')?.classList.add('active');
          }

          // View Mode Sync
          if (d.view_mode && d.view_mode !== curView) {
            curView = d.view_mode;
            const btn = document.getElementById('btnViewToggle');
            if (btn) btn.textContent = curView === 'TPP' ? '🎥 POV: 3RD PERSON' : '🎥 POV: 1ST PERSON';
          }

          // Draw Minimap
          drawRadar(d.x, d.y, (d.yaw_deg || 0) * Math.PI / 180, d.trail);

          // Update VLM Confidence Bars
          if (d.vlm_scores) {
            const container = document.getElementById('vlmScoresList');
            if (container) {
              container.innerHTML = Object.entries(d.vlm_scores).map(([k, s]) => {
                const pct = Math.round(s * 100);
                const color = pct > 65 ? '#00e5ff' : pct > 35 ? '#ffd600' : '#64748b';
                return `
                  <div class="vlm-row">
                    <span style="color:#cbd5e1; width:50px;">${k}</span>
                    <div class="vlm-bar-bg">
                      <div class="vlm-bar-fill" style="width:${pct}%; background:${color};"></div>
                    </div>
                    <span style="color:${color}; width:28px; text-align:right;">${pct}%</span>
                  </div>
                `;
              }).join('');
            }
            if (d.target && d.target !== 'NONE') {
              const activeLabel = document.getElementById('vlmActiveTarget');
              if (activeLabel) activeLabel.textContent = d.target.split(' ')[0].toUpperCase();
            }
          }
        }
      } catch(e) {
        // Fallback gracefully
      }
      setTimeout(pollTelemetry, 600);
    }
    setTimeout(pollTelemetry, 300);

    // ── Event Handlers & Tactile D-Pad ──
    function initEvents() {
      document.getElementById('btnPatrol')?.addEventListener('click', () => window.setMode('AI_PATROL'));
      document.getElementById('btnChase')?.addEventListener('click', () => window.setMode('AI_CHASE'));
      document.getElementById('btnManual')?.addEventListener('click', () => window.setMode('MANUAL'));
      document.getElementById('btnReset')?.addEventListener('click', () => window.postApi('/reset'));
      document.getElementById('btnSeek')?.addEventListener('click', window.sendTarget);
      document.getElementById('btnViewToggle')?.addEventListener('click', window.toggleView);

      document.getElementById('promptInput')?.addEventListener('keydown', e => {
        if (e.key === 'Enter') window.sendTarget();
      });

      const bindBtn = (id, cmd) => {
        const el = document.getElementById(id);
        if (!el) return;
        const start = (e) => { e.preventDefault(); window.manual(cmd); el.classList.add('active'); };
        const stop = (e) => { e.preventDefault(); window.manual('STOP'); el.classList.remove('active'); };
        el.addEventListener('mousedown', start);
        el.addEventListener('mouseup', stop);
        el.addEventListener('mouseleave', stop);
        el.addEventListener('touchstart', start, { passive: false });
        el.addEventListener('touchend', stop, { passive: false });
        el.addEventListener('touchcancel', stop, { passive: false });
      };

      bindBtn('dpadFwd', 'FORWARD');
      bindBtn('dpadBack', 'BACKWARD');
      bindBtn('dpadLeft', 'LEFT');
      bindBtn('dpadRight', 'RIGHT');
      document.getElementById('dpadStop')?.addEventListener('click', (e) => {
        e.preventDefault(); window.manual('STOP');
      });
    }
    initEvents();

    // ── Keyboard WASD / Arrow Teleoperation & 'V' POV Toggle ──
    const keyMap = {
      w: 'FORWARD', a: 'LEFT', s: 'BACKWARD', d: 'RIGHT', ' ': 'STOP',
      arrowup: 'FORWARD', arrowleft: 'LEFT', arrowdown: 'BACKWARD', arrowright: 'RIGHT'
    };
    const keyBtnMap = {
      FORWARD: 'dpadFwd', LEFT: 'dpadLeft', BACKWARD: 'dpadBack', RIGHT: 'dpadRight', STOP: 'dpadStop'
    };
    const activeKeys = new Set();

    document.addEventListener('keydown', e => {
      if (e.target.tagName === 'INPUT') return;
      const k = e.key.toLowerCase();
      if (k === 'v') {
        window.toggleView();
        return;
      }
      if (keyMap[k] && !activeKeys.has(k)) {
        activeKeys.add(k);
        window.setMode('MANUAL');
        const cmd = keyMap[k];
        window.manual(cmd);
        document.getElementById(keyBtnMap[cmd])?.classList.add('active');
      }
    });

    document.addEventListener('keyup', e => {
      if (e.target.tagName === 'INPUT') return;
      const k = e.key.toLowerCase();
      if (keyMap[k]) {
        activeKeys.delete(k);
        const cmd = keyMap[k];
        document.getElementById(keyBtnMap[cmd])?.classList.remove('active');
        if (activeKeys.size === 0) window.manual('STOP');
        else {
          const lastKey = Array.from(activeKeys)[activeKeys.size - 1];
          window.manual(keyMap[lastKey]);
          document.getElementById(keyBtnMap[keyMap[lastKey]])?.classList.add('active');
        }
      }
    });
  </script>
</body>
</html>
"""

@api.get("/", response_class=HTMLResponse)
def index_root():
    """Serves the full-bleed GTA HUD directly at the root."""
    return HTMLResponse(content=DASHBOARD_HTML)

@api.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    """Compatible alias route for /dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)

# ──────────────────────────────────────────────────────────────────────────────
# 4. 1-Command CLI & Evaluator Launcher (uvx / pipx / local)
# ──────────────────────────────────────────────────────────────────────────────
def main():
    """1-Command Entrypoint for local execution via `uvx patrol-rover` or direct python."""
    parser = argparse.ArgumentParser(
        description="Autonomous AI Vision Patrol Rover Mission Control & GTA HUD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind server")
    parser.add_argument("--port", type=int, default=PORT, help="Port to run web server on")
    parser.add_argument("--no-open", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    # Automatically launch mission control in evaluator's browser
    if not args.no_open:
        def _open():
            time.sleep(1.2)
            url = f"http://{args.host}:{args.port}"
            print(f"🚀 Opening mission control HUD at {url}...")
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    print(f"🚀 Patrol Rover starting on http://{args.host}:{args.port} (Pure FastAPI Local Engine)")
    try:
        uvicorn.run("app:api", host=args.host, port=args.port, reload=False)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down rover controller cleanly...")
        controller.stop()

if __name__ == "__main__":
    main()
