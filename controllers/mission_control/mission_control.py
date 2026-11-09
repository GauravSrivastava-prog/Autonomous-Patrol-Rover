#!/usr/bin/env python3
"""
mission_control.py — Master Facade & FastAPI Router for AI Vision Patrol Robot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Architecture:
  Clean OOP Facade delegating responsibilities to 4 specialized submodules:
    1. domain_state.py      -> ArenaObject, ARENA_OBJECTS, RoverTelemetryState, MissionLogger
    2. perception_engine.py -> Instant <1ms Open-Vocab Resolver, Depth & VLM Planner
    3. physics_simulator.py -> 30Hz Simulation, Solid Collisions & 3D Camera Renderer
    4. ui_templates.py      -> Modern 2D Dashboard & 3D WebXR Digital Twin HTML

Zero Breaking Changes:
  Exposes all module-level exports expected by app.py:
  (app, state, STATE, ControlMode, plan_vlm_goal, start_background_threads,
   ensure_ws_broadcast, MISSION_LOG_PATH, ARENA_OBJECTS, ARENA_OBSTACLES)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys

# ──────────────────────────────────────────────────────────────────────────────
# Bulletproof sys.path Resolution
# ──────────────────────────────────────────────────────────────────────────────
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

_ROOT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
if not os.path.exists(os.path.join(_ROOT_DIR, "worlds")):
    _ROOT_DIR = os.getcwd()

MISSION_LOG_PATH = os.path.join(_ROOT_DIR, "mission_log.md")

import math
import time
import json
import asyncio
import argparse
import threading
from contextlib import asynccontextmanager
from typing import Set, Optional, Dict, Tuple, Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Import OOP Modules
from domain_state import (
    ArenaObject,
    ARENA_OBJECTS,
    ARENA_OBSTACLES,
    ControlMode,
    RoverTelemetryState,
    MissionLogger,
    CAM_W,
    CAM_H,
    ROVER_RADIUS,
    ARENA_BOUND,
    MAX_SPEED,
    TIMESTEP,
)
from perception_engine import PerceptionEngine, VLM_TARGET_CATALOG
from physics_simulator import ArenaPhysicsSimulator, SyntheticCameraRenderer
from ui_templates import DASHBOARD_HTML, VR_HTML

# ──────────────────────────────────────────────────────────────────────────────
# Singleton System Components
# ──────────────────────────────────────────────────────────────────────────────
state = RoverTelemetryState()
STATE = state  # Backward compatibility alias for app.py

logger = MissionLogger(MISSION_LOG_PATH)
perception = PerceptionEngine()
simulator = ArenaPhysicsSimulator(state, perception, logger)


# ──────────────────────────────────────────────────────────────────────────────
# Facade Functions (Backwards Compatibility for app.py & tests)
# ──────────────────────────────────────────────────────────────────────────────
def plan_vlm_goal() -> Tuple[str, str]:
    """Exposed facade function for VLM Auto-Goal selection."""
    return perception.plan_vlm_goal(state, logger)


def resolve_openvocab_coords(prompt: str, cur_rx: float = 0.0, cur_ry: float = 0.0) -> Tuple[float, float, str]:
    """Exposed facade function for coordinate resolution."""
    k, obj = perception.resolve_prompt(prompt, cur_rx, cur_ry)
    return (obj.x, obj.y, obj.name)


def resolve_openvocab_object(prompt: str, cur_rx: float = 0.0, cur_ry: float = 0.0) -> Tuple[str, Dict[str, Any]]:
    """Exposed facade function for object dictionary resolution."""
    k, obj = perception.resolve_prompt(prompt, cur_rx, cur_ry)
    return (k, {
        "id": obj.key,
        "name": obj.name,
        "label": obj.label,
        "coords": obj.coords,
        "radius": obj.radius,
        "height": obj.height,
        "color": obj.color_bgr,
        "shape": obj.shape,
        "aliases": obj.aliases,
    })


def resolve_rover_collisions(rx: float, ry: float) -> Tuple[float, float, bool]:
    """Exposed facade function for solid-body collision pushback."""
    return simulator.resolve_collisions(rx, ry)


def render_synthetic_arena(rx: float, ry: float, ryaw: float, targets: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """Exposed facade function for 3D synthetic camera rendering."""
    with state.lock:
        mode = state.mode
        locked_label = state.locked_target
        preferred = state.preferred_target
        prompt = state.openvocab_prompt
        scene = state.semantic_scene
        score = state.semantic_score
        lv = state.left_speed
        rv = state.right_speed

    target_key = "NONE"
    if prompt:
        target_key, _ = perception.resolve_prompt(prompt, rx, ry)
    elif preferred in ARENA_OBJECTS:
        target_key = preferred
    elif mode == ControlMode.AI_CHASE:
        target_key = min(
            ARENA_OBJECTS.keys(),
            key=lambda k: math.hypot(rx - targets.get(k, ARENA_OBJECTS[k].coords)[0], ry - targets.get(k, ARENA_OBJECTS[k].coords)[1])
        )

    return simulator.renderer.render(
        rx, ry, ryaw, targets, mode, target_key, locked_label,
        simulator.vlm_inspect_timer, scene, score, lv, rv
    )


def project_arena_object_to_2d(rx: float, ry: float, ryaw: float, obj_dict: Any, coords: Optional[Tuple[float, float]] = None) -> Optional[dict]:
    """Exposed facade projection function for 3D-to-2D bounding box math."""
    if isinstance(obj_dict, dict):
        key = obj_dict.get("id", obj_dict.get("key", "OBJ"))
        c = coords or obj_dict.get("coords", (obj_dict.get("x", 0.0), obj_dict.get("y", 0.0)))
        obj = ArenaObject(
            key=key,
            name=obj_dict.get("name", "Object"),
            label=obj_dict.get("label", "Object"),
            x=c[0],
            y=c[1],
            radius=obj_dict.get("radius", 0.25),
            height=obj_dict.get("height", 0.40),
            color_bgr=obj_dict.get("color", (255, 255, 255)),
            shape=obj_dict.get("shape", "box"),
            aliases=obj_dict.get("aliases", []),
        )
    else:
        obj = obj_dict
        c = coords or (obj.x, obj.y)
    return simulator.renderer.project_object(rx, ry, ryaw, obj, c)


def draw_target_hud_bracket(frame: np.ndarray, proj: dict, label: str, color: Tuple[int, int, int], dist: float, confidence: float = 0.95):
    """Exposed facade function for rendering corner brackets and pill banner."""
    x1, y1, x2, y2 = proj["box"]
    sx, sy = proj["sx"], proj["sy"]
    pw, ph = proj["pw"], proj["ph"]

    c_len = max(6, min(18, pw // 4, ph // 4))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    cv2.line(frame, (x1, y1), (x1 + c_len, y1), color, 2)
    cv2.line(frame, (x1, y1), (x1, y1 + c_len), color, 2)
    cv2.line(frame, (x2, y1), (x2 - c_len, y1), color, 2)
    cv2.line(frame, (x2, y1), (x2, y1 + c_len), color, 2)
    cv2.line(frame, (x1, y2), (x1 + c_len, y2), color, 2)
    cv2.line(frame, (x1, y2), (x1, y2 - c_len), color, 2)
    cv2.line(frame, (x2, y2), (x2 - c_len, y2), color, 2)
    cv2.line(frame, (x2, y2), (x2, y2 - c_len), color, 2)
    cv2.drawMarker(frame, (sx, sy), color, cv2.MARKER_CROSS, 14, 1)

    banner_text = f"LOCKED: {label.upper()} [{dist:.2f}m | {int(confidence*100)}%]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(banner_text, font, 0.38, 1)
    pill_h = th + 10
    pill_w = tw + 16

    if y1 - pill_h - 4 >= 10:
        py1, py2 = y1 - pill_h - 4, y1 - 4
    else:
        py1, py2 = y2 + 4, y2 + pill_h + 4

    px1 = max(4, min(CAM_W - pill_w - 4, sx - pill_w // 2))
    px2 = px1 + pill_w
    cv2.rectangle(frame, (px1, py1), (px2, py2), (18, 22, 32), -1)
    cv2.rectangle(frame, (px1, py1), (px2, py2), color, 1)
    cv2.putText(frame, banner_text, (px1 + 8, py1 + th + 5), font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket Bridge & Telemetry Broadcaster
# ──────────────────────────────────────────────────────────────────────────────
class ConnectionManager:
    """Manages active WebSocket connections."""
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: str):
        dead = []
        for conn in self.active_connections:
            try:
                await conn.send_text(message)
            except Exception:
                dead.append(conn)
        for d in dead:
            self.active_connections.discard(d)


ws_manager = ConnectionManager()
_ws_broadcast_task: Optional[asyncio.Task] = None


async def ws_broadcast_loop():
    """30Hz WebSocket telemetry broadcaster."""
    try:
        while True:
            if ws_manager.active_connections:
                packet = json.dumps(state.to_dict())
                await ws_manager.broadcast(packet)
            await asyncio.sleep(1.0 / 30.0)
    except asyncio.CancelledError:
        pass


def ensure_ws_broadcast():
    """Idempotently starts ws_broadcast_loop in the active running asyncio loop."""
    global _ws_broadcast_task
    if _ws_broadcast_task is None or _ws_broadcast_task.done():
        try:
            loop = asyncio.get_running_loop()
            _ws_broadcast_task = loop.create_task(ws_broadcast_loop())
        except RuntimeError:
            pass


_THREADS_LOCK = threading.Lock()
_THREADS_STARTED = False


def start_background_threads():
    """Starts the 30Hz robotics physics simulator and AI worker threads."""
    global _THREADS_STARTED
    with _THREADS_LOCK:
        if _THREADS_STARTED:
            return
        _THREADS_STARTED = True
        simulator.start()
        ensure_ws_broadcast()
        print("✅ Background Robotics Engine & 30Hz Simulator Active.")


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI Application & Lifespan
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    start_background_threads()
    ensure_ws_broadcast()
    yield
    simulator.stop()
    global _ws_broadcast_task
    if _ws_broadcast_task and not _ws_broadcast_task.done():
        _ws_broadcast_task.cancel()
        try:
            await _ws_broadcast_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Mission Control Dashboard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI Routes & Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    return DASHBOARD_HTML


@app.get("/vr", response_class=HTMLResponse)
def get_vr_twin():
    return VR_HTML


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    ensure_ws_broadcast()
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                action = msg.get("action", "")
                if action == "SET_MODE":
                    mode_str = msg.get("mode", "AI_PATROL")
                    try:
                        m = ControlMode(mode_str)
                        with state.lock:
                            state.mode = m
                            state.manual_cmd = "STOP"
                            state.depth_obstacle_detected = False
                    except ValueError:
                        pass
                elif action == "MANUAL_CMD":
                    with state.lock:
                        state.manual_cmd = msg.get("cmd", "STOP").upper()
                elif action == "RESET_POSE":
                    with state.lock:
                        state.teleport_flag = True
                elif action == "SET_OPENVOCAB":
                    p = msg.get("prompt", "").strip()
                    with state.lock:
                        state.openvocab_prompt = p
                        if p and p.lower() not in ["clear", "none", ""]:
                            state.mode = ControlMode.AI_CHASE
                            state.depth_obstacle_detected = False
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


@app.get("/telemetry")
def get_telemetry():
    return JSONResponse(state.to_dict())


# ── Mode & Control Endpoints ──
@app.post("/set_mode/{mode}")
@app.post("/mode/{mode}")
@app.post("/control/mode/{mode}")
def set_mode_endpoint(mode: str):
    try:
        m = ControlMode(mode)
    except ValueError:
        return JSONResponse({"error": f"Invalid mode: {mode}"}, status_code=400)

    with state.lock:
        state.mode = m
        state.depth_obstacle_detected = False
        if m == ControlMode.MANUAL:
            state.manual_cmd = "STOP"
            state.left_speed = 0.0
            state.right_speed = 0.0
        elif m == ControlMode.AI_CHASE:
            state.manual_cmd = "NONE"
            if state.locked_target in ["NONE", "", "PATROL"] and not state.openvocab_prompt:
                nearest_k = min(
                    ARENA_OBJECTS.keys(),
                    key=lambda k: math.hypot(state.rx - ARENA_OBJECTS[k].x, state.ry - ARENA_OBJECTS[k].y)
                )
                state.preferred_target = nearest_k
                state.locked_target = ARENA_OBJECTS[nearest_k].name
        elif m == ControlMode.AI_PATROL:
            state.manual_cmd = "NONE"
            state.locked_target = "PATROL"

    logger.log_event(f"Control mode changed to {m.value}", category="MODE SWITCH", state=state)
    return {"mode": m.value, "status": "active"}


@app.post("/control")
async def control_unified(request: Request):
    """Unified endpoint accepting {'mode': ...} or {'cmd': ...}."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    if "mode" in body:
        return set_mode_endpoint(body["mode"])
    if "cmd" in body:
        cmd = str(body["cmd"]).upper()
        with state.lock:
            state.manual_cmd = cmd
        return {"cmd": cmd}
    return {"status": "ok"}


@app.post("/manual/{cmd}")
def manual_cmd_endpoint(cmd: str):
    cmd_u = cmd.upper()
    with state.lock:
        state.manual_cmd = cmd_u
    return {"cmd": cmd_u}


@app.post("/teleport_reset")
def teleport_reset_endpoint():
    with state.lock:
        state.teleport_flag = True
    return {"status": "teleporting"}


# ── Purpose C: Instant Open-Vocab Target Grounding ──
@app.post("/set_openvocab_prompt")
@app.post("/open_vocab")
async def set_openvocab_prompt_endpoint(request: Request):
    prompt = ""
    try:
        data = await request.json()
        if isinstance(data, dict):
            prompt = data.get("prompt", "")
        elif isinstance(data, str):
            prompt = data
    except Exception:
        pass
    prompt = prompt.strip()

    with state.lock:
        if prompt.lower() in ["clear", "none", ""]:
            state.openvocab_prompt = ""
            state.openvocab_target = None
            logger.log_event("Open-vocabulary target cleared", category="CHASE LOCK", state=state)
        else:
            state.openvocab_prompt = prompt
            state.mode = ControlMode.AI_CHASE
            state.depth_obstacle_detected = False
            t_k, t_obj = perception.resolve_prompt(prompt, state.rx, state.ry)
            state.preferred_target = t_k
            state.locked_target = t_obj.name
            logger.log_event(f"Open-vocab target locked: '{prompt}' -> {t_obj.name} (switched to AI_CHASE)", category="CHASE LOCK", state=state)

    return {"openvocab_prompt": state.openvocab_prompt, "target": state.locked_target, "mode": "AI_CHASE"}


# ── Purpose A: Dynamic VLM Tactical Planner ──
@app.post("/trigger_vlm_goal")
@app.post("/vlm_plan")
def trigger_vlm_goal_endpoint():
    inst, target = perception.plan_vlm_goal(state, logger)
    with state.lock:
        state.mode = ControlMode.AI_CHASE
        state.depth_obstacle_detected = False
    return {"instruction": inst, "preferred_target": target}


@app.post("/set_vlm_goal_mode/{enabled}")
@app.post("/toggle_vlm_auto")
def set_vlm_goal_mode_endpoint(enabled: str):
    val = enabled.lower() in ["true", "1", "yes", "on"]
    with state.lock:
        state.auto_goal_vlm = val
        state.depth_obstacle_detected = False
        if val:
            state.mode = ControlMode.AI_CHASE
    if val:
        perception.plan_vlm_goal(state, logger)
    logger.log_event(f"VLM Auto-Goal set to {'ON' if val else 'OFF'}", category="VLM GOAL", state=state)
    return {"auto_goal_vlm": val}


# ── Purpose B: Snapshot Endpoints (Non-Blocking, High Speed) ──
@app.get("/video_frame.jpg")
def video_frame_snapshot():
    with state.lock:
        content = state.frame_jpeg
    return Response(
        content=content,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
    )


@app.get("/depth_frame.jpg")
def depth_frame_snapshot():
    with state.lock:
        content = state.depth_jpeg
    return Response(
        content=content,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
    )


# ── MJPEG Video Streaming (Backward Compatibility) ──
async def _gen_stream(is_depth: bool = False):
    try:
        while True:
            with state.lock:
                frame = state.depth_jpeg if is_depth else state.frame_jpeg
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            await asyncio.sleep(0.06)
    except asyncio.CancelledError:
        pass


@app.get("/video_feed")
def video_feed_stream():
    return StreamingResponse(_gen_stream(is_depth=False), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/depth_feed")
def depth_feed_stream():
    return StreamingResponse(_gen_stream(is_depth=True), media_type="multipart/x-mixed-replace; boundary=frame")


# ── Purpose D: Flight Logs ──
@app.get("/mission_logs")
def get_mission_logs(request: Request, format: Optional[str] = None):
    with state.lock:
        logs_list = list(state.mission_logs)
    logs_str = "\n".join(logs_list) if logs_list else "Standby. Simulation active."

    # Return clean plain text if requested via format parameter or text/plain Accept header
    accept = request.headers.get("accept", "")
    if format in ("text", "plain") or ("text/plain" in accept and "application/json" not in accept):
        return Response(content=logs_str, media_type="text/plain; charset=utf-8")

    return {
        "status": "ok",
        "logs": logs_str,
        "logs_list": logs_list,
        "total": len(logs_list),
    }


@app.get("/download_mission_log")
def download_mission_log():
    if not os.path.exists(MISSION_LOG_PATH):
        logger.log_event("Mission log file initialized", category="INIT", state=state)
    return FileResponse(MISSION_LOG_PATH, media_type="text/markdown", filename="mission_log.md")


# ──────────────────────────────────────────────────────────────────────────────
# Standalone Runner
# ──────────────────────────────────────────────────────────────────────────────
def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Starts background threads and runs Uvicorn ASGI server."""
    start_background_threads()
    print("=" * 65)
    print("  AI VISION PATROL ROBOT — MISSION CONTROL FACADE ACTIVE")
    print(f"  📡 Dashboard:  http://{host}:{port}/dashboard")
    print(f"  🥽 VR Twin:    http://{host}:{port}/vr")
    print(f"  ⚡ Telemetry:  http://{host}:{port}/telemetry")
    print("=" * 65)
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Vision Patrol Robot — Mission Control Server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--host", type=str, default=os.environ.get("HOST", "0.0.0.0"))
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
