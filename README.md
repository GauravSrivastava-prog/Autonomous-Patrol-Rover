---
title: AI Vision Patrol Robot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🤖 AI Vision Patrol Robot — Webots R2025a + Multimodal AI & Robotics Platform

> An autonomous 3D robot featuring HD computer vision, zero-shot multimodal semantic analysis, monocular depth obstacle avoidance, open-vocabulary grounding, dynamic VLM goal selection, flight logging, and an A-Frame WebXR 3D Digital Twin with bidirectional drag-and-drop object synchronization.

---

## 🌟 Key Architecture & Upgrades

```
┌────────────────────────────────────────────────────────┐
│  Webots R2025a (3D Simulation Physics & Supervisor)     │
│  - 4m×4m Glass Arena with warm amber gold top rails    │
│  - 3 Movable Targets: Red Beacon, Green Plant, Blue    │
│  - 2 Low-Profile Obstacles: Cable Box, Tool Box        │
│  - HD 640×480 Camera Sensor Node                       │
│  - Supervisor API (Translation, Physics Reset)         │
└──────────────────────────┬─────────────────────────────┘
                           │ IPC (<extern>)
┌──────────────────────────▼─────────────────────────────┐
│  Mission Control Server (FastAPI + Asyncio)            │
│  - Webots IPC Loop (~30 Hz)                            │
│  - HD JPEG Quality 95 Streamer                         │
│  - Background Multi-Model AI Engine (ai_worker_loop):  │
│      * Purpose A: Dynamic VLM Goal Planner             │
│      * Purpose B: 640x480 MAGMA Monocular Depth Map    │
│      * Purpose C: Open-Vocabulary Target Grounder      │
│      * Purpose D: Automated Flight Mission Logger      │
│      * YOLOv8n + OpenAI CLIP Semantic Throttling       │
│  - WebSocket Telemetry & Drag-and-Drop Handler         │
└────────────────────┬─────────────────────┬─────────────┘
                     │                     │
         HTTP & SSE  ▼                     ▼  WebSockets (~30 Hz)
┌─────────────────────────────┐   ┌──────────────────────────────┐
│  2D Mission Control         │   │  A-Frame WebXR Digital Twin  │
│  http://localhost:8000/     │   │  http://localhost:8000/vr   │
│  - Dual Viewport (RGB+Depth)│   │  - 3D Digital Twin Arena     │
│  - Open-Vocab Prompt Input  │   │  - Dual Hologram Screens     │
│  - VLM Auto-Goal Deck       │   │  - 3D Drag-and-Drop Targets  │
│  - Live Mission Log Terminal│   │  - Gaze & Keyboard Controls  │
└─────────────────────────────┘   └──────────────────────────────┘
```

---

## 🚀 The 4 Core Multimodal AI Purposes

### 🧭 Purpose A: Dynamic Goal Selection & Semantic Navigation (VLM + CLIP Planner)
- **Autonomous Tactical Planning**: Evaluates laboratory scene geometry and visited target history to decide the next waypoint (`RED`, `GREEN`, `BLUE`).
- **Natural Language Instructions**: Generates real-time reasoning sentences (e.g. *"VLM Planner: Tactical waypoint switch -> GREEN (Potted Plant) to inspect environmental flora perimeter"*).
- **Interactive Control**: Toggleable VLM Auto-Goal mode on both 2D dashboard and 3D VR deck with manual trigger endpoint `POST /trigger_vlm_goal`.

### ◈ Purpose B: Monocular Depth Map & Low-Obstacle Avoidance
- **High-Definition Depth Stream**: Real-time 640×480 depth map rendered in high-contrast `MAGMA` colormap, served live at `GET /depth_feed` and `GET /depth_frame.jpg`.
- **Floor ROI Analysis**: Continuously inspects the bottom-center near-field floor zone (`y: 320..470, x: 180..460`) split into left and right halves.
- **Low-Profile Obstacle Avoidance**: Detects obstacles sitting below the ray height of the rover's infrared distance sensors (`LOW_CABLE_BOX` and `SMALL_TOOLBOX`), triggering an immediate `[DEPTH HAZARD: LOW OBSTACLE - REROUTING]` warning and automated steering avoidance around the clearer side.

### 🔍 Purpose C: Open-Vocabulary Target Tracking
- **Free-Text Zero-Shot Prompting**: Input any arbitrary natural language prompt (e.g., `"wooden crate"`, `"yellow hazard box"`, `"potted plant"`, `"blue chair"`) via `POST /set_openvocab_prompt` or WebSocket action `SET_OPENVOCAB`.
- **Region Grounding**: Extracts candidate object bounding boxes and scores them against the prompt embedding using OpenAI CLIP.
- **Neon-Magenta HUD & Visual Servoing**: Highlights the grounded target with a magenta box `(255, 0, 255)`, crosshair, and confidence score, feeding the target center coordinate directly into `AI_CHASE` for smooth visual servoing.

### 📋 Purpose D: Automated Mission Flight Logging & Captioning
- **Multimodal Flight Log**: Every 5 seconds, synthesizes telemetry pose, active mode, semantic classification, depth hazard status, and locked target into a human-readable flight log entry.
- **Persistent Markdown Log**: Automatically writes structured log entries to `mission_log.md` in the project root.
- **Live Terminal & Export**: View logs streaming in real-time on the 2D dashboard terminal, query `GET /mission_logs`, or click **"Download mission_log.md"** (`GET /download_mission_log`).

---

## 🕹️ Control Modes

| Mode | Key | Behavior |
|------|-----|----------|
| 🧠 **AI_PATROL** | `P` | Autonomous exploration with wall bounce, physical stall watchdog, and depth-based low obstacle avoidance. Safely throttles speed when CLIP detects fragile objects. |
| 🎯 **AI_CHASE** | `C` | Visual servoing tracking target beacons or open-vocabulary objects. Maintains a clean 15cm standoff distance (smoothly backs away if too close). |
| 🕹️ **MANUAL** | `M` | Full teleoperation via `W / A / S / D` keys or dashboard buttons. |

---

## 📡 API Endpoints & WebSocket Protocol

### HTTP Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | 2D Dual-Viewport Mission Control Dashboard |
| `GET` | `/vr` | A-Frame WebXR 3D Digital Twin with Dual Hologram Screens |
| `GET` | `/video_feed` | HD RGB + YOLO + Open-Vocab MJPEG video stream (640×480) |
| `GET` | `/video_frame.jpg` | Single HD RGB JPEG snapshot for WebGL/VR textures |
| `GET` | `/depth_feed` | Live Monocular Depth Map MJPEG stream (640×480, MAGMA) |
| `GET` | `/depth_frame.jpg` | Single Depth JPEG snapshot for WebGL/VR textures |
| `GET` | `/telemetry` | Full telemetry JSON packet including depth, VLM, and open-vocab states |
| `POST` | `/set_mode/{mode}` | Switch mode (`AI_PATROL`, `AI_CHASE`, `MANUAL`) |
| `POST` | `/manual/{cmd}` | Send teleop command (`FORWARD`, `BACKWARD`, `LEFT`, `RIGHT`, `STOP`) |
| `POST` | `/teleport_reset` | Reset robot pose to origin $(0, 0, 0)$ |
| `POST` | `/trigger_vlm_goal` | Manually trigger Purpose A VLM goal selection |
| `POST` | `/set_vlm_goal_mode/{enabled}` | Toggle Purpose A VLM Auto-Goal mode (`true`/`false`) |
| `POST` | `/set_openvocab_prompt` | Set Purpose C prompt (`{"prompt": "wooden crate"}`) |
| `GET` | `/mission_logs` | Retrieve recent mission flight logs in JSON |
| `GET` | `/download_mission_log` | Download persistent `mission_log.md` flight log file |

### WebSocket Protocol (`/ws`)

#### Downstream Broadcast (~30 Hz):
```json
{
  "x": 0.452,
  "y": -0.812,
  "z": 0.060,
  "yaw": 42.5,
  "vx": 3.75,
  "vy": 3.75,
  "mode": "AI_PATROL",
  "detected": "RED",
  "ps_left": 45.0,
  "ps_right": 32.0,
  "semantic_scene": "Clear Open Floor",
  "semantic_score": 0.94,
  "depth_hazard": false,
  "depth_score": 0.18,
  "auto_goal_vlm": false,
  "vlm_goal_instruction": "VLM Planner: Tactical waypoint switch -> GREEN (Potted Plant)",
  "preferred_target": "GREEN",
  "openvocab_prompt": "wooden crate",
  "openvocab_detected": true,
  "latest_mission_log": "[04:05:12] POS (0.45, -0.81) | YAW 42.5° | MODE: AI_PATROL..."
}
```

#### Upstream Client Actions:
- **Drag Target**: `{"action": "DRAG_OBJECT", "target": "RED", "x": 0.5, "y": 0.2}`
- **Change Mode**: `{"action": "SET_MODE", "mode": "AI_CHASE"}`
- **Teleop**: `{"action": "MANUAL_CMD", "cmd": "FORWARD"}`
- **Reset**: `{"action": "RESET_POSE"}`
- **Set Open-Vocab**: `{"action": "SET_OPENVOCAB", "prompt": "wooden crate"}`
- **VLM Plan Goal**: `{"action": "VLM_SELECT_GOAL"}`
- **Set VLM Mode**: `{"action": "SET_VLM_GOAL_MODE", "enabled": true}`

---

## 🛠️ Quick Start

### 1. Prerequisites
- **macOS** (Apple Silicon or Intel)
- **Webots R2025a** installed in `/Applications/Webots.app`
- **Python 3.11+** virtual environment

### 2. Launch Demo
```bash
chmod +x run_demo.sh
./run_demo.sh
```

- **2D Mission Control Dashboard**: [http://localhost:8000](http://localhost:8000)
- **3D WebXR Digital Twin**: [http://localhost:8000/vr](http://localhost:8000/vr)
- **Mission Log**: [http://localhost:8000/download_mission_log](http://localhost:8000/download_mission_log)

---

## 🤗 Deploy to Hugging Face Spaces (24/7 Zero-Friction Cloud)

This repository is pre-configured with `app.py`, `Dockerfile`, `packages.txt`, and `requirements.txt` for direct deployment on Hugging Face Spaces.

### Option A: Create a Docker Space (Recommended)
1. Go to [Hugging Face Spaces](https://huggingface.co/new-space).
2. Set Space name (e.g., `ai-vision-patrol-robot`), license, and select **Docker** as Space SDK.
3. Push this repository to your Space remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<YOUR-USERNAME>/<SPACE-NAME>
   git push space main
   ```
4. Hugging Face will automatically build the container and serve the interactive Gradio + FastAPI + WebXR dashboard on port `7860`.

### Option B: Run locally on dynamic port
```bash
python3 controllers/mission_control/mission_control.py --port 7860
# or via master launcher:
python3 app.py
```

---

## 👥 Authors & Academic Context

- **Course**: Algorithm and Robotics Lab, Semester 7
- **Platform**: Webots R2025a Supervisor, FastAPI, OpenCV, YOLOv8, OpenAI CLIP, Monocular Depth, Three.js / A-Frame WebXR
