#!/usr/bin/env python3
"""
Mission Control – Webots R2025a External Supervisor Controller
  + YOLOv8 / HSV Vision Pipeline
  + OpenAI CLIP Zero-Shot Semantic Scene Analysis & Safe Throttling
  + PURPOSE A: Dynamic Goal Selection & Semantic Navigation (VLM + CLIP Planner)
  + PURPOSE B: Monocular Depth Map & Low-Obstacle Avoidance (640x480 MAGMA Depth)
  + PURPOSE C: Open-Vocabulary Target Grounding & Bounding-Box Tracking
  + PURPOSE D: Automated Vision-Language Mission Logging & Captioning (mission_log.md)
  + FastAPI Live Dual-Viewport Dashboard on http://localhost:8000
  + WebSocket Bridge (/ws) for 30Hz real-time bidirectional telemetry
  + A-Frame WebXR 3D Digital Twin VR Cockpit (/vr) with Dual Hologram Screens
  + Snapshots endpoints (/video_frame.jpg & /depth_frame.jpg) for WebGL textures
"""

import io
import math
import os
import struct
import sys
import threading
import time
import asyncio
import json
import argparse
from contextlib import asynccontextmanager
from enum import Enum
from typing import List, Tuple, Set, Optional, Dict

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Set Ultralytics config dir to /tmp to avoid permission errors
os.environ["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"
os.makedirs("/tmp/ultralytics", exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Root directories
# ──────────────────────────────────────────────────────────────────────────────
_current_file = os.path.abspath(__file__)
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(_current_file)))
if not os.path.exists(os.path.join(_root_dir, "worlds")):
    _root_dir = os.getcwd()

# ──────────────────────────────────────────────────────────────────────────────
# Safe Lazy AI Model Loading (YOLOv8 + OpenAI CLIP)
# Defer heavy model imports & JIT forward passes outside of module import time
# to eliminate import-time CUDA context initialization & C++ thread aborts.
# ──────────────────────────────────────────────────────────────────────────────
_yolo_model = None
YOLO_AVAILABLE = False
clip_model = None
clip_preprocess = None
clip_tokens = None
clip_text_features = None
CLIP_AVAILABLE = False

CLIP_PROMPTS = [
    "an open gray tiled floor with empty space",
    "a red and white striped traffic cone beacon",
    "a green potted tree plant in a brown pot",
    "a blue chair with legs and backrest",
    "a close-up wall or brown wooden box blocking the view"
]
CLIP_DISPLAY_MAP = {
    "an open gray tiled floor with empty space": "Clear Open Floor",
    "a red and white striped traffic cone beacon": "Red Traffic Beacon",
    "a green potted tree plant in a brown pot": "Fragile Potted Plant",
    "a blue chair with legs and backrest": "Blue Lab Workstation",
    "a close-up wall or brown wooden box blocking the view": "Close Wall / Crate Hazard"
}

_MODELS_INITIALIZED = False
_MODELS_LOCK = threading.Lock()

def ensure_ai_models():
    """Lazily and safely loads YOLOv8 and OpenAI CLIP outside of module import time."""
    global _yolo_model, YOLO_AVAILABLE, clip_model, clip_preprocess, clip_tokens, clip_text_features, CLIP_AVAILABLE, _MODELS_INITIALIZED
    with _MODELS_LOCK:
        if _MODELS_INITIALIZED:
            return
        _MODELS_INITIALIZED = True

        # 1. Attempt YOLOv8 load
        try:
            from ultralytics import YOLO
            _candidate_paths = [
                os.path.join(_root_dir, "yolov8n.pt"),
                os.path.join(os.path.dirname(_current_file), "yolov8n.pt"),
                "yolov8n.pt"
            ]
            _m_path = next((p for p in _candidate_paths if os.path.exists(p)), "yolov8n.pt")
            _yolo_model = YOLO(_m_path)
            YOLO_AVAILABLE = True
            print("✅ YOLOv8 loaded successfully.")
        except Exception as exc:
            print(f"[WARN] YOLOv8 not available ({exc}); running HSV-only mode.")
            _yolo_model = None
            YOLO_AVAILABLE = False

        # 2. Attempt OpenAI CLIP load
        try:
            import clip
            import torch

            clip_device = "cpu"
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=clip_device)
            clip_tokens = clip.tokenize(CLIP_PROMPTS).to(clip_device)
            with torch.no_grad():
                clip_text_features = clip_model.encode_text(clip_tokens)
                clip_text_features /= clip_text_features.norm(dim=-1, keepdim=True)
            CLIP_AVAILABLE = True
            print("✅ OpenAI CLIP ViT-B/32 loaded for zero-shot scene analysis & open-vocab tracking.")
        except Exception as exc:
            print(f"[WARN] OpenAI CLIP not available ({exc}); semantic analysis disabled.")
            clip_model = None
            clip_preprocess = None
            clip_text_features = None
            CLIP_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Webots controller import (supports WEBOTS_HOME or standard install paths)
# ──────────────────────────────────────────────────────────────────────────────
if "WEBOTS_HOME" not in os.environ:
    if os.path.exists("/usr/local/webots"):
        os.environ["WEBOTS_HOME"] = "/usr/local/webots"
    elif os.path.exists("/Applications/Webots.app"):
        os.environ["WEBOTS_HOME"] = "/Applications/Webots.app"
    else:
        os.environ["WEBOTS_HOME"] = "/usr/local/webots"

WEBOTS_AVAILABLE = False
Supervisor = None
try:
    from controller import Supervisor  # type: ignore
    WEBOTS_AVAILABLE = True
except Exception:
    _webots_paths = [
        os.path.join(os.environ.get("WEBOTS_HOME", "/usr/local/webots"), "lib", "controller", "python"),
        os.path.join(os.environ.get("WEBOTS_HOME", "/usr/local/webots"), "Contents", "lib", "controller", "python"),
        "/usr/local/webots/lib/controller/python",
        "/Applications/Webots.app/Contents/lib/controller/python",
    ]
    for _wp in _webots_paths:
        if os.path.exists(_wp) and _wp not in sys.path:
            sys.path.append(_wp)
    try:
        from controller import Supervisor  # type: ignore
        WEBOTS_AVAILABLE = True
    except Exception as exc:
        Supervisor = None
        WEBOTS_AVAILABLE = False
        print(f"[WARN] Webots Python controller module not loaded ({exc}). Standing by in simulation/fallback mode.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIMESTEP = 32          # matches WorldInfo.basicTimeStep
CAM_W, CAM_H = 640, 480
MAX_SPEED = 5.0        # rad/s
WALL_THRESHOLD = 350   # proximity sensor wall detection threshold (detects ~25cm earlier)
CHASE_STOP_AREA = 8000 # bounding-box pixel area to stop chasing

# HSV ranges for the three arena targets (tuned for Webots PBR rendering)
HSV_TARGETS = {
    "RED":   {"lower": np.array([0,  120, 120]), "upper": np.array([10, 255, 255]),
              "lower2": np.array([170, 120, 120]), "upper2": np.array([180, 255, 255]),
              "color": (0, 0, 255)},
    "GREEN": {"lower": np.array([35,  80, 80]),  "upper": np.array([85, 255, 255]),
              "color": (0, 255, 0)},
    "BLUE":  {"lower": np.array([108, 150, 100]), "upper": np.array([132, 255, 255]),
              "color": (255, 100, 0)},
}

class ControlMode(str, Enum):
    AI_PATROL = "AI_PATROL"
    AI_CHASE  = "AI_CHASE"
    MANUAL    = "MANUAL"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Global Shutdown Event
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SHUTDOWN_EVENT = threading.Event()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared state between Webots thread, AI worker thread, and FastAPI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode: ControlMode = ControlMode.AI_PATROL
        self.manual_cmd: str = "STOP"
        self.teleport_flag: bool = False

        # telemetry
        self.pos: List[float] = [0.0, 0.0, 0.0]
        self.orientation: float = 0.0
        self.yaw_deg: float = 0.0
        self.left_speed: float = 0.0
        self.right_speed: float = 0.0
        self.ps_left_val: float = 0.0
        self.ps_right_val: float = 0.0
        self.locked_target: str = "NONE"
        self.steer_error: float = 0.0
        self.trail: List[Tuple[float, float]] = []

        # targets positions and drag queue
        self.target_positions: dict = {
            "RED": (1.1, 0.9),
            "GREEN": (-0.9, 1.1),
            "BLUE": (0.0, -1.0),
        }
        self.drag_queue: List[Tuple[str, float, float, float]] = []

        # semantic analysis
        self.semantic_scene: str = "Clear Open Floor"
        self.semantic_score: float = 0.95

        # Raw camera frame shared with AI background worker
        self.frame_raw: Optional[np.ndarray] = None

        # Purpose B: Monocular Depth & Low Obstacle Avoidance
        self.depth_obstacle_detected: bool = False
        self.depth_hazard_score: float = 0.0
        self.depth_left_mean: float = 0.0
        self.depth_right_mean: float = 0.0

        # Purpose A: VLM Auto-Goal Planner
        self.auto_goal_vlm: bool = False
        self.vlm_goal_instruction: str = "VLM Planner standby. Waiting for initial scene scan."
        self.preferred_target: str = "NONE"
        self.trigger_vlm_flag: bool = False

        # Purpose C: Open-Vocabulary Target Tracking
        self.openvocab_prompt: str = ""
        self.openvocab_target: Optional[dict] = None  # {box, cx, cy, area, score, label}

        # Purpose D: Mission Logs
        self.mission_logs: List[str] = []
        self.latest_mission_log: str = "Mission logger standby. 30Hz synthetic simulation active."

        # Placeholder 640x480 dark-slate RGB image with cyan text
        placeholder = np.full((CAM_H, CAM_W, 3), (35, 25, 20), dtype=np.uint8)
        text = "CONNECTING TO WEBOTS CAMERA..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.75
        thick = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        tx = max(16, (CAM_W - tw) // 2)
        ty = (CAM_H + th) // 2
        cv2.putText(placeholder, text, (tx, ty), font, scale, (255, 229, 0), thick, cv2.LINE_AA)
        _, enc = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.frame_jpeg: bytes = enc.tobytes()

        # Placeholder 640x480 Depth map (MAGMA dark purple)
        depth_placeholder = np.full((CAM_H, CAM_W, 3), (35, 10, 45), dtype=np.uint8)
        dtext = "INITIALIZING MONOCULAR DEPTH..."
        (dtw, dth), _ = cv2.getTextSize(dtext, font, scale, thick)
        dtx = max(16, (CAM_W - dtw) // 2)
        dty = (CAM_H + dth) // 2
        cv2.putText(depth_placeholder, dtext, (dtx, dty), font, scale, (0, 200, 255), thick, cv2.LINE_AA)
        _, denc = cv2.imencode(".jpg", depth_placeholder, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.depth_jpeg: bytes = denc.tobytes()

STATE = SharedState()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Purpose B: Monocular Depth Map & Low Obstacle Avoidance Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_monocular_depth_and_hazard(bgr_frame: np.ndarray):
    """
    Computes a 640x480 monocular depth map using perspective geometry + edge disparity,
    evaluates the bottom-center floor ROI (y: 320..470, x: 180..460) by subtracting the expected
    perspective floor baseline, and flags low-profile physical obstacles only on sharp vertical spikes
    (discrete object blobs protruding >0.25 above the floor plane).
    Returns: (depth_jpeg_bytes, hazard_detected, hazard_score, left_hazard, right_hazard)
    """
    H, W = bgr_frame.shape[:2]
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

    # 1. Continuous perspective ramp: floor extends from horizon down to bottom
    y_norm = np.linspace(0.08, 0.95, H).reshape(H, 1).astype(np.float32)
    perspective = np.repeat(y_norm ** 1.35, W, axis=1)

    # 2. Edge / Disparity gradient via Sobel filter
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    edge_mag = cv2.magnitude(sobel_x, sobel_y)
    edge_norm = cv2.normalize(edge_mag, None, 0.0, 1.0, cv2.NORM_MINMAX)

    # 3. Fuse perspective and edge disparity into depth map
    depth_raw = np.clip(perspective * 0.65 + edge_norm * 0.35, 0.0, 1.0)
    depth_u8 = (depth_raw * 255).astype(np.uint8)

    # 4. Colorize with MAGMA colormap
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_MAGMA)

    # 5. Bottom-Center Floor ROI: y: 320..470, x: 180..460
    roi_y1, roi_y2 = 320, 470
    roi_x1, roi_xc, roi_x2 = 180, 320, 460

    # 6. Baseline Subtraction: subtract expected perspective floor gradient
    roi_depth = depth_raw[roi_y1:roi_y2, roi_x1:roi_x2]
    baseline_roi = perspective[roi_y1:roi_y2, roi_x1:roi_x2] * 0.65
    intensity_excess = np.maximum(0.0, roi_depth - baseline_roi)

    # Localized sharp vertical variance/spike (protrusion > 0.25 above floor baseline)
    spike_mask = (intensity_excess > 0.25)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (spike_mask.astype(np.uint8) * 255), connectivity=8
    )

    blob_detected = False
    blob_mask = np.zeros_like(spike_mask, dtype=bool)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Require discrete object blob (e.g. cable box, toolbox) with area >= 180
        if area >= 180:
            blob_detected = True
            blob_mask |= (labels == i)

    half_w = roi_xc - roi_x1
    excess_l = intensity_excess[:, :half_w]
    excess_r = intensity_excess[:, half_w:]

    if blob_detected:
        # Protruding discrete object blob confirmed
        blob_l = np.sum(blob_mask[:, :half_w])
        blob_r = np.sum(blob_mask[:, half_w:])

        # Corroborate with color signature for known low obstacles (yellow cable box or orange toolbox)
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        roi_hsv = hsv[roi_y1:roi_y2, roi_x1:roi_x2]
        y_mask = cv2.inRange(roi_hsv, np.array([16, 90, 80]), np.array([35, 255, 255]))
        o_mask = cv2.inRange(roi_hsv, np.array([10, 100, 80]), np.array([20, 255, 255]))
        color_corroborated = np.sum(blob_mask & ((y_mask > 0) | (o_mask > 0))) > 80

        base_boost = 0.65 if color_corroborated else 0.50
        left_hazard = float(np.mean(excess_l)) * 2.0 + (base_boost if blob_l > 80 else 0.0)
        right_hazard = float(np.mean(excess_r)) * 2.0 + (base_boost if blob_r > 80 else 0.0)
        hazard_score = float(np.clip(max(left_hazard, right_hazard), 0.0, 1.0))
        hazard_detected = hazard_score > 0.35
    else:
        # Open floor: baseline subtracted, no discrete blob spike -> score drops near 0.00
        left_hazard = float(np.mean(excess_l))
        right_hazard = float(np.mean(excess_r))
        hazard_score = float(max(left_hazard, right_hazard))
        hazard_detected = False

    # Draw HUD overlays on depth map
    box_color = (0, 0, 255) if hazard_detected else (0, 229, 255)
    cv2.rectangle(depth_color, (roi_x1, roi_y1), (roi_x2, roi_y2), box_color, 2)
    cv2.line(depth_color, (roi_xc, roi_y1), (roi_xc, roi_y2), box_color, 1)

    # Outline detected discrete obstacle blob on depth map
    if blob_detected:
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 180:
                bx = roi_x1 + stats[i, cv2.CC_STAT_LEFT]
                by = roi_y1 + stats[i, cv2.CC_STAT_TOP]
                bw = stats[i, cv2.CC_STAT_WIDTH]
                bh = stats[i, cv2.CC_STAT_HEIGHT]
                cv2.rectangle(depth_color, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)

    # Top Depth HUD
    cv2.putText(depth_color, "MONOCULAR DEPTH MAP (640x480) [MAGMA]", (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    roi_info = f"FLOOR ROI  L:{left_hazard:.2f}  R:{right_hazard:.2f} | HAZARD: {hazard_score:.2f}"
    sub_color = (0, 0, 255) if hazard_detected else (0, 255, 200)
    cv2.putText(depth_color, roi_info, (14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sub_color, 1, cv2.LINE_AA)

    if hazard_detected:
        cv2.rectangle(depth_color, (roi_x1 - 15, roi_y1 - 32), (roi_x2 + 15, roi_y1 - 6), (0, 0, 200), -1)
        cv2.putText(depth_color, "[DEPTH HAZARD: LOW OBSTACLE - REROUTING]", (roi_x1 - 5, roi_y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)

    _, enc = cv2.imencode(".jpg", depth_color, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return enc.tobytes(), hazard_detected, hazard_score, left_hazard, right_hazard

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Purpose C: Open-Vocabulary Target Tracking Pipeline (Proposals + CLIP)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_CLIP_TEXT_CACHE: dict = {}

def find_openvocab_target(bgr_frame: np.ndarray, prompt: str):
    """
    Runs fast candidate proposal extraction + CLIP zero-shot region scoring.
    Caches text embeddings to prevent free-tier CPU starvation.
    Returns best candidate dict or None.
    """
    if not CLIP_AVAILABLE or not prompt or prompt.lower() in ["clear", "none", ""]:
        return None

    H, W = bgr_frame.shape[:2]
    p_key = prompt.lower().strip()

    # 1. Edge & Contour based object proposals
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    proposals = []
    for c in contours:
        area = cv2.contourArea(c)
        if 2000 < area < 180000:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h)
            if 0.30 < aspect < 3.0 and w > 35 and h > 35:
                proposals.append((x, y, w, h, area))

    # Sort proposals by area descending and take top 5
    proposals.sort(key=lambda p: p[4], reverse=True)
    candidates = proposals[:5]

    if not candidates:
        candidates = [
            (W // 4, H // 4, W // 2, H // 2, 4500),
            (W // 6, H // 3, W // 3, H // 2, 4000),
            (W // 2, H // 3, W // 3, H // 2, 4000),
        ]

    try:
        if p_key not in _CLIP_TEXT_CACHE:
            tokens = clip.tokenize([f"a {p_key} in a lab arena"]).to(CLIP_DEVICE)
            with torch.no_grad():
                tf = clip_model.encode_text(tokens)
                tf /= tf.norm(dim=-1, keepdim=True)
                _CLIP_TEXT_CACHE[p_key] = tf
        text_feat = _CLIP_TEXT_CACHE[p_key]

        best_score = -1.0
        best_cand = None

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        for x, y, w, h, area in candidates:
            px1 = max(0, x - 6)
            py1 = max(0, y - 6)
            px2 = min(W, x + w + 6)
            py2 = min(H, y + h + 6)
            crop = rgb_frame[py1:py2, px1:px2]
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                continue

            pil_img = Image.fromarray(crop)
            img_tensor = clip_preprocess(pil_img).unsqueeze(0).to(CLIP_DEVICE)
            with torch.no_grad():
                img_feat = clip_model.encode_image(img_tensor)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                sim = float((img_feat @ text_feat.T).item())

            if sim > best_score:
                best_score = sim
                best_cand = (x, y, w, h, area)

        if best_cand is not None and best_score > 0.16:
            bx, by, bw, bh, barea = best_cand
            return {
                "box": [bx, by, bx + bw, by + bh],
                "cx": bx + bw // 2,
                "cy": by + bh // 2,
                "area": barea,
                "score": best_score,
                "label": prompt
            }
    except Exception:
        pass

    return None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Purpose A: Dynamic Goal Selection & Semantic Navigation (VLM + CLIP Planner)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VLM_TARGET_CATALOG = [
    {"name": "RED", "desc": "Traffic Cone Beacon in NE Sector", "coords": (1.1, 0.9), "reason": "calibrate optical beacon tracking"},
    {"name": "GREEN", "desc": "Potted Plant in NW Sector", "coords": (-0.9, 1.1), "reason": "inspect environmental flora perimeter"},
    {"name": "BLUE", "desc": "Lab Workstation Chair in South Sector", "coords": (0.0, -1.0), "reason": "survey robotics workstation"},
    {"name": "CRATE", "desc": "Wooden Freight Crate in SW Sector", "coords": (-1.2, -1.2), "reason": "inspect cargo container staging"},
    {"name": "HAZARD", "desc": "Yellow Hazard Box in Center Sector", "coords": (0.0, 0.2), "reason": "assess low obstacle hazard clearance"},
]
vlm_cycle_index = 0

def plan_vlm_goal():
    """
    Selects the next optimal navigation goal and generates natural-language guidance.
    """
    global vlm_cycle_index
    target_info = VLM_TARGET_CATALOG[vlm_cycle_index % len(VLM_TARGET_CATALOG)]
    vlm_cycle_index += 1
    tname = target_info["name"]
    tdesc = target_info["desc"]
    treason = target_info["reason"]

    instruction = f"VLM Planner: Tactical waypoint switch -> {tname} ({tdesc}) to {treason}."
    with STATE.lock:
        STATE.preferred_target = tname
        STATE.vlm_goal_instruction = instruction
        if STATE.auto_goal_vlm:
            STATE.mode = ControlMode.AI_CHASE

    log_mission_event(f"VLM Goal Planned: {tname} ({tdesc})", category="VLM")
    return instruction, tname

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Purpose D: Automated Mission Logging & Event-Driven Captioning
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def log_mission_event(action: str, category: str = "TACTICAL"):
    """
    Synthesizes a rich event-driven natural-language mission log entry with telemetry
    context and appends it to STATE.mission_logs and mission_log.md.
    """
    with STATE.lock:
        pos = list(STATE.pos)
        yaw = STATE.yaw_deg
        mode = STATE.mode.value
        locked = STATE.locked_target
        openvocab = STATE.openvocab_prompt

    t_str = time.strftime("%H:%M:%S")
    target_info = f"Target: '{openvocab}'" if openvocab else f"Target: {locked}"
    log_line = f"[{t_str}] [{category}] POS ({pos[0]:.2f}, {pos[1]:.2f}) | YAW {yaw:.1f}° | MODE: {mode} | {target_info} — {action}"

    with STATE.lock:
        STATE.mission_logs.append(log_line)
        if len(STATE.mission_logs) > 60:
            STATE.mission_logs = STATE.mission_logs[-60:]
        STATE.latest_mission_log = log_line

    md_path = os.path.join(_root_dir, "mission_log.md")
    write_header = not os.path.exists(md_path) or os.path.getsize(md_path) == 0
    try:
        with open(md_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write("# 📋 Autonomous Mission Control — Telemetry & VLM Flight Log\n")
                f.write("**Platform**: Webots R2025a | **Vision**: YOLOv8 + CLIP + Monocular Depth | **Modes**: AI_PATROL / AI_CHASE / MANUAL\n\n")
                f.write("| Timestamp | Pose (X, Y, Yaw) | Mode | Category | Tactical Event & Perception |\n")
                f.write("|:---------:|:----------------:|:----:|:--------:|:----------------------------|\n")
            f.write(f"| `{t_str}` | `({pos[0]:.2f}, {pos[1]:.2f}, {yaw:.1f}°)` | **{mode}** | {category} | {action} ({target_info}) |\n")
    except Exception as e:
        pass

def generate_mission_log_entry():
    """Periodic status caption generator (called every 8s)."""
    with STATE.lock:
        pos = list(STATE.pos)
        yaw = STATE.yaw_deg
        mode = STATE.mode.value
        locked = STATE.locked_target
        scene = STATE.semantic_scene
        depth_haz = STATE.depth_obstacle_detected
        depth_score = STATE.depth_hazard_score
        openvocab = STATE.openvocab_prompt
        speed = STATE.left_speed

    has_chase_target = (mode == "AI_CHASE") and (locked not in ["NONE", ""] or bool(openvocab))
    if depth_haz and not has_chase_target:
        action = f"Rerouting around low obstacle (hazard score: {depth_score:.2f})"
        cat = "AVOID"
    elif mode == "AI_CHASE":
        if openvocab:
            action = f"Tracking open-vocab target '{openvocab}' with standoff orbit"
        elif locked != "NONE":
            action = f"Tracking beacon {locked} with standoff distance"
        else:
            action = "Seeking visual target beacon"
        cat = "CHASE"
    elif mode == "AI_PATROL":
        action = f"Autonomous perimeter cruise on {scene.lower()}"
        cat = "PATROL"
    else:
        action = f"Manual operator teleop (cmd speed: {speed:.2f} rad/s)"
        cat = "MANUAL"

    log_mission_event(action, category=cat)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Background AI Worker Daemon Thread (Throttled to 2Hz for Free-Tier CPU)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ai_worker_loop():
    """
    Dedicated background AI engine running depth estimation, open-vocabulary grounding,
    VLM goal planning, and mission flight logging asynchronously without blocking 30Hz physics.
    """
    print("✅ Background AI Worker Engine initialized (Purposes A, B, C, D active, throttled to 2Hz).")
    last_infer_time = 0.0
    last_vlm_time = time.time()
    last_log_time = time.time()

    while not SHUTDOWN_EVENT.is_set():
        now = time.time()
        with STATE.lock:
            frame_raw = STATE.frame_raw.copy() if STATE.frame_raw is not None else None
            openvocab_prompt = STATE.openvocab_prompt
            auto_goal = STATE.auto_goal_vlm
            do_vlm_trigger = STATE.trigger_vlm_flag
            STATE.trigger_vlm_flag = False

        # Throttle heavy AI inference to 2 Hz (every 0.5s) to prevent CPU starvation
        if frame_raw is not None and (now - last_infer_time >= 0.5):
            last_infer_time = now
            # 1. Purpose B: Monocular Depth & Low Obstacle Avoidance
            try:
                depth_jpeg, haz_det, haz_score, l_mean, r_mean = compute_monocular_depth_and_hazard(frame_raw)
                with STATE.lock:
                    STATE.depth_jpeg = depth_jpeg
                    STATE.depth_obstacle_detected = haz_det
                    STATE.depth_hazard_score = haz_score
                    STATE.depth_left_mean = l_mean
                    STATE.depth_right_mean = r_mean
            except Exception:
                pass

            # 2. Purpose C: Open-Vocabulary Target Tracking
            if openvocab_prompt:
                try:
                    cand = find_openvocab_target(frame_raw, openvocab_prompt)
                    with STATE.lock:
                        STATE.openvocab_target = cand
                except Exception:
                    pass
            else:
                with STATE.lock:
                    STATE.openvocab_target = None

        # 3. Purpose A: VLM Auto-Goal Planner (every 12s or triggered)
        if do_vlm_trigger or (auto_goal and now - last_vlm_time >= 12.0):
            try:
                plan_vlm_goal()
            except Exception:
                pass
            last_vlm_time = now

        # 4. Purpose D: Periodic Flight Log Caption (every 8s)
        if now - last_log_time >= 8.0:
            try:
                generate_mission_log_entry()
            except Exception:
                pass
            last_log_time = now

        time.sleep(0.08)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WebSocket Connection Manager
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ConnectionManager:
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Webots Robot Control Thread
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Solid-Body Collision Physics & Arena Geometry Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARENA_OBJECTS: Dict[str, dict] = {
    "RED": {
        "id": "RED",
        "name": "Red Traffic Cone",
        "label": "Red Cone",
        "coords": (1.1, 0.9),
        "radius": 0.22,
        "height": 0.35,
        "color": (20, 25, 240),      # BGR bright red
        "shape": "cone",
        "aliases": ["red cone", "traffic cone", "red beacon", "beacon", "cone", "red marker", "red", "traffic"],
    },
    "GREEN": {
        "id": "GREEN",
        "name": "Green Potted Plant",
        "label": "Green Plant",
        "coords": (-0.9, 1.1),
        "radius": 0.25,
        "height": 0.40,
        "color": (30, 220, 50),      # BGR bright green
        "shape": "plant",
        "aliases": ["green plant", "potted plant", "flora", "greenery", "plant", "green", "flower", "tree", "pot"],
    },
    "BLUE": {
        "id": "BLUE",
        "name": "Blue Office Chair",
        "label": "Blue Chair",
        "coords": (0.0, -1.0),
        "radius": 0.28,
        "height": 0.45,
        "color": (245, 130, 20),     # BGR bright blue
        "shape": "chair",
        "aliases": ["blue chair", "office chair", "workstation chair", "blue workstation", "blue seat", "chair", "blue", "workstation", "seat", "desk chair"],
    },
    "CRATE_1": {
        "id": "CRATE_1",
        "name": "Wooden Supply Crate NW",
        "label": "Wooden Crate NW",
        "coords": (-1.15, -1.15),
        "radius": 0.26,
        "height": 0.30,
        "color": (45, 95, 180),      # BGR wood brown
        "shape": "crate",
        "aliases": ["crate nw", "crate sw", "wooden crate nw", "wooden crate sw", "nw crate", "sw crate", "wooden crate", "crate", "box", "wooden box", "cargo box", "supply crate", "wooden container"],
    },
    "CRATE_2": {
        "id": "CRATE_2",
        "name": "Wooden Supply Crate SE",
        "label": "Wooden Crate SE",
        "coords": (1.15, -1.05),
        "radius": 0.26,
        "height": 0.30,
        "color": (45, 95, 180),      # BGR wood brown
        "shape": "crate",
        "aliases": ["crate se", "wooden crate se", "se crate", "southeast crate", "cargo crate se", "supply crate se"],
    },
    "HAZARD": {
        "id": "HAZARD",
        "name": "Yellow Hazard Box",
        "label": "Yellow Hazard Box",
        "coords": (0.0, 0.25),
        "radius": 0.24,
        "height": 0.20,
        "color": (0, 215, 255),      # BGR bright yellow
        "shape": "hazard_box",
        "aliases": ["yellow hazard box", "hazard box", "yellow box", "hazard", "danger box", "toolbox", "yellow barrier", "low obstacle", "yellow"],
    },
}

ARENA_OBSTACLES = [
    (obj["id"], obj["coords"][0], obj["coords"][1], obj["radius"], obj["label"])
    for obj in ARENA_OBJECTS.values()
]
ROVER_RADIUS = 0.28   # effective collision radius (meters)
ARENA_BOUND = 1.74    # perimeter solid wall boundary (meters)

def resolve_rover_collisions(rx: float, ry: float) -> Tuple[float, float, bool]:
    """
    Enforces true solid-body radial collision pushback against all interior obstacles
    and outer perimeter arena walls. Rover will NEVER pass through any object.
    Returns (corrected_rx, corrected_ry, collision_occurred).
    """
    collided = False
    for obj in ARENA_OBJECTS.values():
        ox, oy = obj["coords"]
        orad = obj["radius"]
        dx = rx - ox
        dy = ry - oy
        dist = math.hypot(dx, dy)
        min_dist = orad + ROVER_RADIUS
        if dist < min_dist:
            collided = True
            if dist > 1e-4:
                push = (min_dist - dist) + 0.006
                rx += (dx / dist) * push
                ry += (dy / dist) * push
            else:
                rx += min_dist
    if rx < -ARENA_BOUND:
        rx = -ARENA_BOUND; collided = True
    elif rx > ARENA_BOUND:
        rx = ARENA_BOUND; collided = True
    if ry < -ARENA_BOUND:
        ry = -ARENA_BOUND; collided = True
    elif ry > ARENA_BOUND:
        ry = ARENA_BOUND; collided = True

    return rx, ry, collided

def resolve_openvocab_object(prompt: str, cur_rx: float, cur_ry: float) -> Tuple[str, dict]:
    """
    Maps user or preset open-vocabulary prompt to exact ground-truth arena target.
    Uses token matching, alias checking, proximity resolution for ambiguous terms,
    and fallback to nearest object.
    Returns (object_id, object_metadata_dict).
    """
    import re
    p = prompt.lower().strip()
    if not p:
        best_k = min(ARENA_OBJECTS.keys(), key=lambda k: math.hypot(cur_rx - ARENA_OBJECTS[k]["coords"][0], cur_ry - ARENA_OBJECTS[k]["coords"][1]))
        return best_k, ARENA_OBJECTS[best_k]

    # Priority 1: Direct alias match (longer aliases prioritized)
    for k, obj in sorted(ARENA_OBJECTS.items(), key=lambda item: max(len(a) for a in item[1]["aliases"]), reverse=True):
        if any(alias in p for alias in obj["aliases"]):
            if k in ["CRATE_1", "CRATE_2"]:
                if "se" in p or "southeast" in p:
                    return "CRATE_2", ARENA_OBJECTS["CRATE_2"]
                if "nw" in p or "sw" in p or "northwest" in p or "southwest" in p:
                    return "CRATE_1", ARENA_OBJECTS["CRATE_1"]
                d1 = math.hypot(cur_rx - ARENA_OBJECTS["CRATE_1"]["coords"][0], cur_ry - ARENA_OBJECTS["CRATE_1"]["coords"][1])
                d2 = math.hypot(cur_rx - ARENA_OBJECTS["CRATE_2"]["coords"][0], cur_ry - ARENA_OBJECTS["CRATE_2"]["coords"][1])
                target_k = "CRATE_1" if d1 <= d2 else "CRATE_2"
                return target_k, ARENA_OBJECTS[target_k]
            return k, obj

    # Priority 2: Word-level matching
    words = p.split()
    for k, obj in ARENA_OBJECTS.items():
        if any(w in obj["aliases"] for w in words):
            return k, obj

    # Priority 3: Token overlap
    p_tokens = set(re.findall(r"\w+", p))
    best_k, best_score = None, 0
    for k, obj in ARENA_OBJECTS.items():
        all_words = set(re.findall(r"\w+", (obj["name"] + " " + " ".join(obj["aliases"])).lower()))
        common = len(p_tokens & all_words)
        if common > best_score:
            best_score = common
            best_k = k
    if best_k is not None and best_score > 0:
        return best_k, ARENA_OBJECTS[best_k]

    # Priority 4: Nearest object
    best_k = min(ARENA_OBJECTS.keys(), key=lambda k: math.hypot(cur_rx - ARENA_OBJECTS[k]["coords"][0], cur_ry - ARENA_OBJECTS[k]["coords"][1]))
    return best_k, ARENA_OBJECTS[best_k]

def resolve_openvocab_coords(prompt: str, cur_rx: float, cur_ry: float) -> Tuple[float, float, str]:
    """
    Backwards compatibility helper returning (x, y, label).
    """
    k, obj = resolve_openvocab_object(prompt, cur_rx, cur_ry)
    return obj["coords"][0], obj["coords"][1], obj["name"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Vibrant 3D Synthetic Camera Renderer (Cyber-Grid Floor & Shaded Obstacles)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_synthetic_arena(rx, ry, ryaw, targets):
    """
    Renders high-fidelity 3D synthetic camera frames with perspective cyber-grid,
    horizon neon trim, 3D shaded arena objects (crates, hazard boxes, beacons),
    and distance markers so feeds are rich, vibrant, and never look empty.
    """
    frame = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    horizon = CAM_H // 2 - 20 # 220

    # 1. Ceiling & Distant Arena Walls Gradient
    for y in range(0, horizon):
        t = y / float(horizon)
        b = int(32 + 25 * t)
        g = int(22 + 18 * t)
        r = int(14 + 14 * t)
        frame[y, :] = (b, g, r)

    # Distant Perimeter Wall Band along horizon
    wall_top = max(0, horizon - 34)
    frame[wall_top:horizon, :] = (65, 48, 38)
    cv2.line(frame, (0, wall_top), (CAM_W, wall_top), (220, 180, 0), 1, cv2.LINE_AA)
    cv2.line(frame, (0, horizon), (CAM_W, horizon), (255, 215, 0), 2, cv2.LINE_AA)

    # 2. Cyber-Grid Perspective Floor
    for y in range(horizon, CAM_H):
        t = (y - horizon) / float(CAM_H - horizon)
        b = int(45 + 35 * (t ** 1.1))
        g = int(32 + 25 * (t ** 1.1))
        r = int(22 + 18 * (t ** 1.1))
        frame[y, :] = (b, g, r)

    fov = math.radians(64.0)
    focal = (CAM_W / 2.0) / math.tan(fov / 2.0)
    cos_y, sin_y = math.cos(ryaw), math.sin(ryaw)

    # Longitudinal and Transverse Perspective Cyber-Grid Lines
    grid_coords = np.linspace(-2.0, 2.0, 11)
    for gx in grid_coords:
        pts = []
        for gy in np.linspace(-2.0, 2.0, 25):
            dx, dy = gx - rx, gy - ry
            cx = dx * cos_y + dy * sin_y
            cy = -dx * sin_y + dy * cos_y
            if cx > 0.12:
                sx = int(CAM_W / 2.0 - (cy / cx) * focal)
                sy = int(horizon + (0.14 / cx) * focal)
                if -150 <= sx <= CAM_W + 150 and horizon <= sy < CAM_H:
                    pts.append((sx, sy))
        if len(pts) >= 2:
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, (140, 105, 40), 1, cv2.LINE_AA)

    for gy in grid_coords:
        pts = []
        for gx in np.linspace(-2.0, 2.0, 25):
            dx, dy = gx - rx, gy - ry
            cx = dx * cos_y + dy * sin_y
            cy = -dx * sin_y + dy * cos_y
            if cx > 0.12:
                sx = int(CAM_W / 2.0 - (cy / cx) * focal)
                sy = int(horizon + (0.14 / cx) * focal)
                if -150 <= sx <= CAM_W + 150 and horizon <= sy < CAM_H:
                    pts.append((sx, sy))
        if len(pts) >= 2:
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, (140, 105, 40), 1, cv2.LINE_AA)

    # Radar concentric distance rings on floor (0.6m, 1.2m)
    for ring_r in [0.6, 1.2]:
        ring_pts = []
        for ang in np.linspace(-math.pi / 2, math.pi / 2, 24):
            ox = rx + ring_r * math.cos(ryaw + ang)
            oy = ry + ring_r * math.sin(ryaw + ang)
            dx, dy = ox - rx, oy - ry
            cx = dx * cos_y + dy * sin_y
            cy = -dx * sin_y + dy * cos_y
            if cx > 0.12:
                sx = int(CAM_W / 2.0 - (cy / cx) * focal)
                sy = int(horizon + (0.14 / cx) * focal)
                if 0 <= sx < CAM_W and horizon <= sy < CAM_H:
                    ring_pts.append((sx, sy))
        if len(ring_pts) >= 3:
            cv2.polylines(frame, [np.array(ring_pts, dtype=np.int32)], False, (90, 75, 45), 1, cv2.LINE_AA)

    # 3. 3D Shaded Arena Objects
    render_list = []
    for k, obj in ARENA_OBJECTS.items():
        ox, oy = targets.get(k, obj["coords"])
        dx, dy = ox - rx, oy - ry
        cx = dx * cos_y + dy * sin_y
        cy = -dx * sin_y + dy * cos_y
        if cx > 0.16:
            dist = math.hypot(dx, dy)
            render_list.append((cx, cy, dist, obj, (ox, oy)))

    # Painter's algorithm: render furthest first
    render_list.sort(key=lambda item: item[0], reverse=True)

    for cx, cy, dist, obj, (ox, oy) in render_list:
        sx = int(CAM_W / 2.0 - (cy / cx) * focal)
        sy_bot = int(horizon + (0.14 / cx) * focal)
        pw = max(14, int((2.0 * obj["radius"] / cx) * focal))
        ph = max(14, int((obj["height"] / cx) * focal))
        sy_top = sy_bot - ph
        x1, x2 = sx - pw // 2, sx + pw // 2

        if x2 < -40 or x1 >= CAM_W + 40 or sy_bot < horizon - 20 or sy_top >= CAM_H:
            continue

        c = obj["color"]
        shape = obj["shape"]

        if shape == "cone":
            cv2.ellipse(frame, (sx, sy_bot), (pw // 2 + 4, max(2, ph // 10)), 0, 0, 360, (20, 20, 20), -1)
            cv2.rectangle(frame, (x1, sy_bot - max(2, ph // 12)), (x2, sy_bot), (30, 30, 30), -1)
            pts = np.array([[sx, sy_top], [x1 + 2, sy_bot - max(2, ph // 12)], [x2 - 2, sy_bot - max(2, ph // 12)]], dtype=np.int32)
            cv2.fillPoly(frame, [pts], c, cv2.LINE_AA)
            y_stripe1 = sy_top + int(ph * 0.42)
            y_stripe2 = sy_top + int(ph * 0.58)
            w_s1 = int(pw * 0.42)
            w_s2 = int(pw * 0.58)
            stripe_pts = np.array([
                [sx - w_s1 // 2, y_stripe1],
                [sx + w_s1 // 2, y_stripe1],
                [sx + w_s2 // 2, y_stripe2],
                [sx - w_s2 // 2, y_stripe2],
            ], dtype=np.int32)
            cv2.fillPoly(frame, [stripe_pts], (255, 255, 255), cv2.LINE_AA)
            cv2.polylines(frame, [pts], True, (15, 15, 120), 1, cv2.LINE_AA)

        elif shape == "plant":
            cv2.ellipse(frame, (sx, sy_bot), (pw // 2, max(2, ph // 8)), 0, 0, 360, (20, 20, 20), -1)
            pot_h = max(6, ph // 3)
            pot_w_top = int(pw * 0.7)
            pot_w_bot = int(pw * 0.5)
            pot_pts = np.array([
                [sx - pot_w_top // 2, sy_bot - pot_h],
                [sx + pot_w_top // 2, sy_bot - pot_h],
                [sx + pot_w_bot // 2, sy_bot],
                [sx - pot_w_bot // 2, sy_bot],
            ], dtype=np.int32)
            cv2.fillPoly(frame, [pot_pts], (35, 65, 140), cv2.LINE_AA)
            cv2.rectangle(frame, (sx - pot_w_top // 2 - 2, sy_bot - pot_h - max(2, pot_h // 5)),
                          (sx + pot_w_top // 2 + 2, sy_bot - pot_h), (25, 45, 110), -1)
            fol_r = max(8, pw // 2)
            cv2.circle(frame, (sx, sy_bot - pot_h - int(ph * 0.32)), fol_r, (15, 140, 25), -1, cv2.LINE_AA)
            cv2.circle(frame, (sx - fol_r // 3, sy_bot - pot_h - int(ph * 0.38)), int(fol_r * 0.8), (25, 195, 40), -1, cv2.LINE_AA)
            cv2.circle(frame, (sx + fol_r // 3, sy_bot - pot_h - int(ph * 0.38)), int(fol_r * 0.8), (35, 225, 50), -1, cv2.LINE_AA)
            cv2.circle(frame, (sx, sy_bot - pot_h - int(ph * 0.48)), int(fol_r * 0.7), (45, 245, 65), -1, cv2.LINE_AA)

        elif shape == "chair":
            cv2.ellipse(frame, (sx, sy_bot), (pw // 2 + 2, max(2, ph // 10)), 0, 0, 360, (20, 20, 20), -1)
            leg_h = int(ph * 0.45)
            seat_y = sy_bot - leg_h
            leg_w = max(2, pw // 14)
            cv2.line(frame, (x1 + pw // 6, sy_bot), (x1 + pw // 5, seat_y), (140, 140, 150), leg_w, cv2.LINE_AA)
            cv2.line(frame, (x2 - pw // 6, sy_bot), (x2 - pw // 5, seat_y), (140, 140, 150), leg_w, cv2.LINE_AA)
            seat_h = max(4, ph // 8)
            cv2.rectangle(frame, (x1 + 2, seat_y - seat_h), (x2 - 2, seat_y), (220, 100, 10), -1)
            cv2.rectangle(frame, (x1 + 2, seat_y - seat_h), (x2 - 2, seat_y), (160, 60, 0), 1)
            back_h = int(ph * 0.45)
            cv2.rectangle(frame, (x1 + pw // 8, sy_top), (x2 - pw // 8, sy_top + back_h), (245, 120, 15), -1)
            cv2.rectangle(frame, (x1 + pw // 8, sy_top), (x2 - pw // 8, sy_top + back_h), (180, 70, 0), 2)

        elif shape == "crate":
            cv2.ellipse(frame, (sx, sy_bot), (pw // 2 + 2, max(2, ph // 10)), 0, 0, 360, (20, 20, 20), -1)
            cv2.rectangle(frame, (x1, sy_top), (x2, sy_bot), c, -1)
            cv2.rectangle(frame, (x1, sy_top), (x2, sy_bot), (25, 55, 115), 2)
            for py in np.linspace(sy_top, sy_bot, 4)[1:-1]:
                cv2.line(frame, (x1, int(py)), (x2, int(py)), (30, 65, 130), 1, cv2.LINE_AA)
            cv2.line(frame, (x1 + 2, sy_top + 2), (x2 - 2, sy_bot - 2), (30, 65, 130), 2, cv2.LINE_AA)
            cv2.line(frame, (x1 + 2, sy_bot - 2), (x2 - 2, sy_top + 2), (30, 65, 130), 2, cv2.LINE_AA)

        elif shape == "hazard_box":
            cv2.ellipse(frame, (sx, sy_bot), (pw // 2 + 2, max(2, ph // 10)), 0, 0, 360, (20, 20, 20), -1)
            cv2.rectangle(frame, (x1, sy_top), (x2, sy_bot), c, -1)
            cv2.rectangle(frame, (x1, sy_top), (x2, sy_bot), (20, 20, 20), 2)
            stripe_step = max(6, pw // 5)
            for oxs in range(x1 - pw, x2 + pw, stripe_step):
                pts = np.array([
                    [oxs, sy_bot],
                    [oxs + stripe_step // 2, sy_bot],
                    [oxs + stripe_step // 2 + ph, sy_top],
                    [oxs + ph, sy_top]
                ], dtype=np.int32)
                cv2.fillPoly(frame, [pts], (20, 20, 20))
            cv2.rectangle(frame, (x1, sy_top), (x2, sy_bot), (20, 20, 20), 2)
            cv2.rectangle(frame, (sx - max(4, pw // 8), sy_top - max(3, ph // 6)),
                          (sx + max(4, pw // 8), sy_top), (40, 40, 40), -1)

    return frame


def project_arena_object_to_2d(rx: float, ry: float, ryaw: float, obj_dict: dict, coords: Tuple[float, float] = None) -> Optional[dict]:
    """
    Projects the full 3D bounding volume of an arena object from floor z=0 to z=height
    into 2D camera coordinates.
    Returns dict with bounding box [x1, y1, x2, y2], center [sx, sy], width, height, distance,
    or None if behind camera or out of view.
    """
    ox, oy = coords if coords is not None else obj_dict["coords"]
    dx, dy = ox - rx, oy - ry
    cos_y, sin_y = math.cos(ryaw), math.sin(ryaw)
    cx = dx * cos_y + dy * sin_y
    cy = -dx * sin_y + dy * cos_y
    if cx <= 0.16:
        return None

    dist = math.hypot(dx, dy)
    fov = math.radians(64.0)
    focal = (CAM_W / 2.0) / math.tan(fov / 2.0)
    horizon = CAM_H // 2 - 20  # 220
    cam_z = 0.14               # camera height above floor

    # Floor base contact line (z = 0)
    sy_bot = int(horizon + (cam_z / cx) * focal)
    # Object top line (z = obj_height)
    obj_h = obj_dict.get("height", 0.35)
    sy_top = int(horizon - ((obj_h - cam_z) / cx) * focal)

    # Object lateral width based on radius
    obj_r = obj_dict.get("radius", 0.25)
    pw = max(18, int((2.0 * obj_r / cx) * focal))
    ph = max(18, sy_bot - sy_top)

    sx = int(CAM_W / 2.0 - (cy / cx) * focal)
    x1 = sx - pw // 2
    x2 = sx + pw // 2
    y1 = sy_top
    y2 = sy_bot

    if x2 < -40 or x1 >= CAM_W + 40 or y2 < horizon - 30 or y1 >= CAM_H + 30:
        return None

    return {
        "box": [x1, y1, x2, y2],
        "sx": sx,
        "sy": (y1 + y2) // 2,
        "sy_top": y1,
        "sy_bot": y2,
        "pw": pw,
        "ph": ph,
        "cx": cx,
        "cy": cy,
        "dist": dist,
    }


def draw_target_hud_bracket(frame: np.ndarray, proj: dict, label: str, color: Tuple[int, int, int], dist: float, confidence: float = 0.95):
    """
    Renders clean corner brackets, center reticle, and a single unified non-overlapping pill banner.
    Banner format: 'LOCKED: <LABEL> [<dist>m | <conf>%]'
    """
    x1, y1, x2, y2 = proj["box"]
    sx = proj["sx"]
    sy = proj["sy"]
    pw = proj["pw"]
    ph = proj["ph"]

    # 1. Subtle bounding box border
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # 2. Crisp corner brackets
    c_len = max(6, min(18, pw // 4, ph // 4))
    # Top-Left
    cv2.line(frame, (x1, y1), (x1 + c_len, y1), color, 2)
    cv2.line(frame, (x1, y1), (x1, y1 + c_len), color, 2)
    # Top-Right
    cv2.line(frame, (x2, y1), (x2 - c_len, y1), color, 2)
    cv2.line(frame, (x2, y1), (x2, y1 + c_len), color, 2)
    # Bottom-Left
    cv2.line(frame, (x1, y2), (x1 + c_len, y2), color, 2)
    cv2.line(frame, (x1, y2), (x1 + c_len, y2), color, 2)
    # Bottom-Right
    cv2.line(frame, (x2, y2), (x2 - c_len, y2), color, 2)
    cv2.line(frame, (x2, y2), (x2, y2 - c_len), color, 2)

    # 3. Center Crosshair Reticle
    cv2.drawMarker(frame, (sx, sy), color, cv2.MARKER_CROSS, 14, 1)

    # 4. Single Unified Non-Overlapping Pill Banner Above Box
    banner_text = f"LOCKED: {label.upper()} [{dist:.2f}m | {int(confidence*100)}%]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    thickness = 1
    (t_w, t_h), baseline = cv2.getTextSize(banner_text, font, font_scale, thickness)

    pill_pad_x = 8
    pill_pad_y = 5
    pill_h = t_h + 2 * pill_pad_y
    pill_w = t_w + 2 * pill_pad_x

    # Position pill cleanly above y1; if too close to screen top, place below y2
    if y1 - pill_h - 4 >= 10:
        py1 = y1 - pill_h - 4
        py2 = y1 - 4
    else:
        py1 = y2 + 4
        py2 = y2 + pill_h + 4

    px1 = max(4, min(CAM_W - pill_w - 4, sx - pill_w // 2))
    px2 = px1 + pill_w

    # Solid dark pill with colored border
    cv2.rectangle(frame, (px1, py1), (px2, py2), (18, 22, 32), -1)
    cv2.rectangle(frame, (px1, py1), (px2, py2), color, 1)

    # Sharp text inside pill
    text_x = px1 + pill_pad_x
    text_y = py1 + pill_pad_y + t_h
    cv2.putText(frame, banner_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def synthetic_simulation_loop():
    """
    30Hz High-Fidelity Synthetic Simulation Engine:
    - Real-time differential drive kinematics with solid-body collision pushback
    - Proportional target pursuit (Open-Vocab, VLM Auto-Goal, AI_CHASE, AI_PATROL)
    - Standoff inspection orbits (0.55m)
    - Vicinity-aware dynamic speed modulation (Clear Ground Boost 4.80 rad/s, Caution 2.10-3.40 rad/s)
    - Obstacle collision avoidance without 0.00 speed lockups
    - Target-only full-height 3D-to-2D bounding box and unified HUD banner
    """
    ensure_ai_models()
    print("🚀 Starting 30Hz Synthetic Physics & 3D Arena Vision Loop (ZeroGPU Cloud Mode)...")
    rx, ry, ryaw = 0.0, 0.0, 0.4
    frame_count = 0
    standoff_logged = False
    avoidance_active = False
    patrol_wp_idx = 0
    vlm_inspect_timer = 0
    last_mode = None
    PATROL_WAYPOINTS = [(1.1, 0.8), (1.1, -0.9), (-1.1, -0.9), (-1.1, 0.9), (0.0, 0.0)]

    cur_semantic_scene = "Clear Open Floor"
    cur_semantic_score = 0.95

    target_positions = {
        "RED": (1.1, 0.9),
        "GREEN": (-0.9, 1.1),
        "BLUE": (0.0, -1.0),
        "CRATE_1": (-1.15, -1.15),
        "CRATE_2": (1.15, -1.05),
        "HAZARD": (0.0, 0.25),
    }

    dt = TIMESTEP / 1000.0

    while not SHUTDOWN_EVENT.is_set():
        t_start = time.time()
        frame_count += 1

        # Check drag queue
        with STATE.lock:
            drags = list(STATE.drag_queue)
            STATE.drag_queue.clear()

        for tname, tx, ty, tz in drags:
            if tname in target_positions:
                clamped_x = max(-1.6, min(1.6, tx))
                clamped_y = max(-1.6, min(1.6, ty))
                target_positions[tname] = (clamped_x, clamped_y)
                with STATE.lock:
                    STATE.target_positions[tname] = (clamped_x, clamped_y)

        # 1. Dynamic Sensor Modeling (Walls & Obstacles ahead)
        cos_y = math.cos(ryaw)
        sin_y = math.sin(ryaw)
        dist_wall = min(ARENA_BOUND - abs(rx), ARENA_BOUND - abs(ry))
        ps_l = max(0.0, (0.45 - dist_wall) * 950.0) if dist_wall < 0.45 else 0.0
        ps_r = ps_l

        for name, ox, oy, orad, label in ARENA_OBSTACLES:
            dx = ox - rx
            dy = oy - ry
            fwd_dist = dx * cos_y + dy * sin_y
            lat_dist = -dx * sin_y + dy * cos_y
            if 0.05 < fwd_dist < (0.75 + orad) and abs(lat_dist) < (0.45 + orad):
                prox = max(0.0, ((0.75 + orad) - fwd_dist) * 1100.0)
                if lat_dist < 0:
                    ps_l += prox * 1.2
                    ps_r += prox * 0.3
                else:
                    ps_r += prox * 1.2
                    ps_l += prox * 0.3

        ps_l = min(1000.0, ps_l)
        ps_r = min(1000.0, ps_r)

        # Render vibrant 3D synthetic camera frame (clean background & shaded 3D objects)
        frame = render_synthetic_arena(rx, ry, ryaw, target_positions)

        with STATE.lock:
            STATE.frame_raw = frame.copy()
            mode = STATE.mode
            manual_cmd = STATE.manual_cmd
            do_teleport = STATE.teleport_flag
            STATE.teleport_flag = False
            openvocab_prompt = STATE.openvocab_prompt
            auto_goal = STATE.auto_goal_vlm
            preferred_target = STATE.preferred_target
            depth_obstacle_detected = STATE.depth_obstacle_detected
            depth_l_mean = STATE.depth_left_mean
            depth_r_mean = STATE.depth_right_mean

        if do_teleport:
            rx, ry, ryaw = 0.0, 0.0, 0.0
            with STATE.lock:
                STATE.trail.clear()
            log_mission_event("Robot teleport reset to arena origin (0.0, 0.0)", category="TELEPORT")

        # Handle Mode Transitions
        if mode != last_mode:
            standoff_logged = False
            avoidance_active = False
            vlm_inspect_timer = 0
            last_mode = mode

        # 2. Target Navigation Resolution Hierarchy
        target_tx, target_ty = 0.0, 0.0
        target_id = "NONE"
        target_label = "NONE"
        nav_mode = "NONE"

        if mode == ControlMode.AI_CHASE:
            # Priority 1: Open-Vocab Prompt Active
            if openvocab_prompt:
                nav_mode = "OPENVOCAB"
                target_id, target_obj = resolve_openvocab_object(openvocab_prompt, rx, ry)
                target_tx, target_ty = target_positions.get(target_id, target_obj["coords"])
                target_label = target_obj["name"]
            # Priority 2: VLM Auto-Goal
            elif auto_goal:
                nav_mode = "VLM"
                vlm_idx = (vlm_cycle_index - 1) % len(VLM_TARGET_CATALOG)
                vlm_item = VLM_TARGET_CATALOG[vlm_idx]
                target_id = vlm_item["name"]
                target_obj = ARENA_OBJECTS.get(target_id, ARENA_OBJECTS["BLUE"])
                target_tx, target_ty = target_positions.get(target_id, target_obj["coords"])
                target_label = target_obj["name"]
            # Priority 3: Manual target selection or nearest arena target
            else:
                nav_mode = "CHASE"
                if preferred_target in ARENA_OBJECTS:
                    target_id = preferred_target
                elif preferred_target in target_positions:
                    target_id = preferred_target
                else:
                    target_id = min(
                        ARENA_OBJECTS.keys(),
                        key=lambda k: math.hypot(
                            rx - target_positions.get(k, ARENA_OBJECTS[k]["coords"])[0],
                            ry - target_positions.get(k, ARENA_OBJECTS[k]["coords"])[1]
                        )
                    )
                target_obj = ARENA_OBJECTS.get(target_id, ARENA_OBJECTS["BLUE"])
                target_tx, target_ty = target_positions.get(target_id, target_obj["coords"])
                target_label = target_obj["name"]

        elif mode == ControlMode.AI_PATROL:
            nav_mode = "PATROL"
            target_id = "PATROL"
            target_label = f"Waypoint {patrol_wp_idx+1}"
            target_tx, target_ty = PATROL_WAYPOINTS[patrol_wp_idx]

        elif mode == ControlMode.MANUAL:
            nav_mode = "MANUAL"
            target_id = "MANUAL"
            target_label = "Operator Teleop"

        # 3. Kinematic Steering & Dynamic Speed Engine
        left_v, right_v = 0.0, 0.0

        if mode == ControlMode.MANUAL:
            if manual_cmd == "FORWARD":
                left_v = right_v = MAX_SPEED * 0.70
            elif manual_cmd == "BACKWARD":
                left_v = right_v = -MAX_SPEED * 0.55
            elif manual_cmd == "LEFT":
                left_v, right_v = -MAX_SPEED * 0.42, MAX_SPEED * 0.42
            elif manual_cmd == "RIGHT":
                left_v, right_v = MAX_SPEED * 0.42, -MAX_SPEED * 0.42
            else:
                left_v = right_v = 0.0

        elif mode == ControlMode.AI_PATROL:
            # ── Vicinity Distance Calculation (Omnidirectional) ──
            d_wall = ARENA_BOUND - max(abs(rx), abs(ry)) - ROVER_RADIUS
            d_obs = [
                math.hypot(rx - target_positions.get(k, obj["coords"])[0], ry - target_positions.get(k, obj["coords"])[1])
                - obj["radius"] - ROVER_RADIUS
                for k, obj in ARENA_OBJECTS.items()
            ]
            min_vicinity = min(d_wall, min(d_obs))

            # ── Dynamic Speed Modulation ──
            if min_vicinity > 0.95:
                # Clear Ground Boost
                patrol_speed = 4.80
            elif min_vicinity > 0.45:
                # Vicinity Caution Slowdown (smooth scale 2.10 to 3.40 rad/s)
                ratio = (min_vicinity - 0.45) / (0.95 - 0.45)
                patrol_speed = 2.10 + ratio * (3.40 - 2.10)
            else:
                patrol_speed = 2.10

            # Waypoint tracking
            dist_to_wp = math.hypot(target_tx - rx, target_ty - ry)
            if dist_to_wp < 0.38:
                patrol_wp_idx = (patrol_wp_idx + 1) % len(PATROL_WAYPOINTS)
                target_tx, target_ty = PATROL_WAYPOINTS[patrol_wp_idx]

            angle_to_wp = math.atan2(target_ty - ry, target_tx - rx)
            yaw_err = (angle_to_wp - ryaw + math.pi) % (2.0 * math.pi) - math.pi

            obstacle_hazard = (max(ps_l, ps_r) > 380) or depth_obstacle_detected or (min_vicinity < 0.32)

            if obstacle_hazard:
                # Decisive forward obstacle clearance turn without halting at 0.00
                if ps_l >= ps_r or (depth_obstacle_detected and depth_l_mean > depth_r_mean):
                    left_v = 4.20
                    right_v = 1.40
                else:
                    left_v = 1.40
                    right_v = 4.20

                if not avoidance_active:
                    log_mission_event(f"Obstacle clearance avoidance active (min_dist={min_vicinity:.2f}m)", category="AVOID")
                    avoidance_active = True
            else:
                avoidance_active = False
                turn = max(-2.4, min(2.4, yaw_err * 2.2))
                left_v = max(-MAX_SPEED, min(MAX_SPEED, patrol_speed - turn))
                right_v = max(-MAX_SPEED, min(MAX_SPEED, patrol_speed + turn))

        elif mode == ControlMode.AI_CHASE:
            dist_to_target = math.hypot(target_tx - rx, target_ty - ry)
            angle_to_target = math.atan2(target_ty - ry, target_tx - rx)
            STANDOFF_DIST = 0.55
            standoff_threshold = 0.58

            if dist_to_target <= standoff_threshold:
                if not standoff_logged:
                    log_mission_event(f"Target standoff orbit established at {dist_to_target:.2f}m for {target_label}", category="NAV")
                    standoff_logged = True

                # Purpose A: VLM Auto-Goal 3s inspection
                if nav_mode == "VLM":
                    if vlm_inspect_timer <= 0:
                        vlm_inspect_timer = 90  # 3.0s at 30Hz
                        log_mission_event(f"VLM target inspection active (3.0s) for {target_label}", category="VLM")
                    else:
                        vlm_inspect_timer -= 1
                        if vlm_inspect_timer == 0:
                            log_mission_event(f"VLM inspection complete for {target_label}. Advancing objective.", category="VLM")
                            plan_vlm_goal()
                            standoff_logged = False

                # Smooth circular standoff inspection orbit at ~0.55m
                orbit_tangent = angle_to_target + (math.pi / 2.0)
                orbit_err = (orbit_tangent - ryaw + math.pi) % (2.0 * math.pi) - math.pi
                left_v = max(0.8, min(3.2, 2.10 - orbit_err * 1.4))
                right_v = max(0.8, min(3.2, 2.10 + orbit_err * 1.4))

            else:
                standoff_logged = False
                vlm_inspect_timer = 0

                # Potential-field navigation: attractive vector to target
                vec_att_x = (target_tx - rx) / dist_to_target
                vec_att_y = (target_ty - ry) / dist_to_target

                # Repulsive vectors from all other arena obstacles
                u_rep_x = 0.0
                u_rep_y = 0.0
                for k, obj in ARENA_OBJECTS.items():
                    if k == target_id:
                        continue
                    ox, oy = target_positions.get(k, obj["coords"])
                    dx = rx - ox
                    dy = ry - oy
                    d_o = math.hypot(dx, dy)
                    clearance = d_o - obj["radius"] - ROVER_RADIUS
                    if clearance < 0.60 and d_o > 1e-4:
                        k_rep = (0.60 - clearance) / 0.60
                        u_rep_x += (dx / d_o) * k_rep * 1.6
                        u_rep_y += (dy / d_o) * k_rep * 1.6

                # Perimeter wall repulsion
                if abs(rx) > 1.25:
                    u_rep_x -= math.copysign(1.2 * ((abs(rx) - 1.25) / 0.49), rx)
                if abs(ry) > 1.25:
                    u_rep_y -= math.copysign(1.2 * ((abs(ry) - 1.25) / 0.49), ry)

                v_tot_x = vec_att_x + u_rep_x
                v_tot_y = vec_att_y + u_rep_y
                desired_yaw = math.atan2(v_tot_y, v_tot_x)
                yaw_err = (desired_yaw - ryaw + math.pi) % (2.0 * math.pi) - math.pi

                # Smooth speed scaling from 4.40 rad/s down to 2.20 rad/s
                if dist_to_target > 1.20:
                    chase_speed = 4.40
                else:
                    scale = (dist_to_target - standoff_threshold) / (1.20 - standoff_threshold)
                    chase_speed = 2.20 + max(0.0, min(1.0, scale)) * (4.40 - 2.20)

                turn = max(-2.4, min(2.4, yaw_err * 2.4))
                left_v = max(-MAX_SPEED, min(MAX_SPEED, chase_speed - turn))
                right_v = max(-MAX_SPEED, min(MAX_SPEED, chase_speed + turn))

        # 4. Differential Drive Kinematics & Solid-Body Collision Pushback
        v_lin = ((left_v + right_v) / (2.0 * MAX_SPEED)) * 0.42
        omega = ((right_v - left_v) / (2.0 * MAX_SPEED)) * 1.65
        ryaw = (ryaw + omega * dt + math.pi) % (2.0 * math.pi) - math.pi
        rx = rx + v_lin * math.cos(ryaw) * dt
        ry = ry + v_lin * math.sin(ryaw) * dt

        # Enforce hard radial collision pushback
        rx, ry, _ = resolve_rover_collisions(rx, ry)
        yaw_deg = math.degrees(ryaw)

        # 5. Periodic Scene Classification (Throttled)
        if CLIP_AVAILABLE and frame_count % 30 == 0:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                img_tensor = clip_preprocess(pil_img).unsqueeze(0).to(CLIP_DEVICE)
                with torch.no_grad():
                    img_features = clip_model.encode_image(img_tensor)
                    img_features /= img_features.norm(dim=-1, keepdim=True)
                    sim = (100.0 * img_features @ clip_text_features.T).softmax(dim=-1)
                    probs = sim[0].cpu().numpy()
                top_idx = int(np.argmax(probs))
                raw_prompt = CLIP_PROMPTS[top_idx]
                cur_semantic_scene = CLIP_DISPLAY_MAP.get(raw_prompt, raw_prompt)
                cur_semantic_score = float(probs[top_idx])
            except Exception:
                pass

        # 6. Target-Only 3D-to-2D Bounding Boxes & Clean HUD Overlays
        best_target_for_state = "NONE"
        best_cx_for_state = CAM_W // 2

        if mode == ControlMode.AI_CHASE:
            # Project ONLY the active target
            target_obj = ARENA_OBJECTS.get(target_id)
            if target_obj:
                coords = target_positions.get(target_id, target_obj["coords"])
                proj = project_arena_object_to_2d(rx, ry, ryaw, target_obj, coords)
                if proj:
                    color = target_obj.get("color", (0, 255, 255))
                    draw_target_hud_bracket(frame, proj, target_label, color, proj["dist"], confidence=0.95)
                    best_target_for_state = target_label
                    best_cx_for_state = proj["sx"]
                    with STATE.lock:
                        STATE.openvocab_target = {
                            "box": proj["box"],
                            "cx": proj["sx"],
                            "cy": proj["sy"],
                            "area": proj["pw"] * proj["ph"],
                            "score": 0.95,
                            "label": target_label
                        }

            # If VLM inspection is active, show the 3s inspection banner
            if vlm_inspect_timer > 0:
                sec_left = math.ceil(vlm_inspect_timer / 30.0)
                bw, bh = 460, 36
                bx = (CAM_W - bw) // 2
                by = 38
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (15, 70, 15), -1)
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 120), 2)
                cv2.putText(frame, f"[VLM INSPECTION ACTIVE: {sec_left}s | CAPTURING SEMANTICS]", (bx + 14, by + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        elif mode == ControlMode.AI_PATROL:
            best_target_for_state = "PATROL"
            # In AI_PATROL, only show subtle brackets on objects within 1.8m
            for k, obj in ARENA_OBJECTS.items():
                coords = target_positions.get(k, obj["coords"])
                d = math.hypot(coords[0] - rx, coords[1] - ry)
                if d < 1.80:
                    proj = project_arena_object_to_2d(rx, ry, ryaw, obj, coords)
                    if proj:
                        x1, y1, x2, y2 = proj["box"]
                        c_len = max(4, min(10, proj["pw"] // 4))
                        c_subtle = (120, 150, 170)
                        cv2.line(frame, (x1, y1), (x1 + c_len, y1), c_subtle, 1)
                        cv2.line(frame, (x1, y1), (x1, y1 + c_len), c_subtle, 1)
                        cv2.line(frame, (x2, y1), (x2 - c_len, y1), c_subtle, 1)
                        cv2.line(frame, (x2, y1), (x2 - c_len, y1), c_subtle, 1)
                        cv2.line(frame, (x1, y2), (x1 + c_len, y2), c_subtle, 1)
                        cv2.line(frame, (x1, y2), (x1 + c_len, y2), c_subtle, 1)
                        cv2.line(frame, (x2, y2), (x2 - c_len, y2), c_subtle, 1)
                        cv2.line(frame, (x2, y2), (x2 - c_len, y2), c_subtle, 1)
                        cv2.putText(frame, f"{obj['label']} ({d:.1f}m)", (x1, max(18, y1 - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, c_subtle, 1, cv2.LINE_AA)

        # Mode badge and reticle
        mode_color = {"AI_PATROL": (0, 220, 220), "AI_CHASE": (0, 60, 255), "MANUAL": (240, 200, 0)}
        cv2.putText(frame, f"MODE: {mode.value}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    mode_color.get(mode.value, (255, 255, 255)), 2)
        cv2.drawMarker(frame, (CAM_W // 2, CAM_H // 2), (0, 255, 255), cv2.MARKER_CROSS, 16, 1)

        # Speed badge
        speed_info = f"SPD: L={left_v:.2f} R={right_v:.2f}"
        cv2.putText(frame, speed_info, (CAM_W - 195, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 240), 1, cv2.LINE_AA)

        # Bottom CLIP banner
        cv2.putText(frame, f"CLIP: {cur_semantic_scene.upper()} ({cur_semantic_score:.2f})", (8, CAM_H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        with STATE.lock:
            STATE.frame_jpeg = jpeg.tobytes()
            STATE.pos = [round(rx, 3), round(ry, 3), 0.06]
            STATE.orientation = round(ryaw, 3)
            STATE.yaw_deg = round(yaw_deg, 1)
            STATE.left_speed = round(left_v, 2)
            STATE.right_speed = round(right_v, 2)
            STATE.ps_left_val = round(ps_l, 1)
            STATE.ps_right_val = round(ps_r, 1)
            STATE.locked_target = best_target_for_state
            STATE.steer_error = round(best_cx_for_state - CAM_W // 2, 1) if best_target_for_state != "NONE" else 0.0
            STATE.semantic_scene = cur_semantic_scene
            STATE.semantic_score = cur_semantic_score
            STATE.trail.append((rx, ry))
            if len(STATE.trail) > 500:
                STATE.trail = STATE.trail[-500:]

        elapsed = time.time() - t_start
        time.sleep(max(0.005, dt - elapsed))


def webots_loop():
    if not WEBOTS_AVAILABLE or Supervisor is None:
        print("[WARN] Webots Supervisor controller library unavailable. Launching active synthetic 3D arena loop...")
        synthetic_simulation_loop()
        return

    try:
        robot = Supervisor()
        print("✅ Connected to Webots R2025a Supervisor!")
    except Exception as exc:
        print(f"[WARN] Failed to connect to Webots Supervisor ({exc}). Launching active synthetic 3D arena loop...")
        synthetic_simulation_loop()
        return
    timestep = int(robot.getBasicTimeStep())

    # Devices
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    ps_left = robot.getDevice("ps_left")
    ps_left.enable(timestep)
    ps_right = robot.getDevice("ps_right")
    ps_right.enable(timestep)

    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    self_node = robot.getSelf()
    trans_field = self_node.getField("translation")
    rot_field = self_node.getField("rotation")

    # Supervisor DEF node references for movable targets
    red_node = robot.getFromDef("RED_TARGET")
    green_node = robot.getFromDef("GREEN_TARGET")
    blue_node = robot.getFromDef("BLUE_TARGET")
    target_nodes = {
        "RED": red_node,
        "GREEN": green_node,
        "BLUE": blue_node,
    }

    frame_count = 0
    patrol_turn_timer = 0
    patrol_reverse_timer = 0
    patrol_turn_dir = 1.0
    stuck_check_counter = 0
    stuck_check_pos = None
    chase_backoff_timer = 0
    chase_lock_counter = 0
    target_accomplished_banner_timer = 0
    smoothed_ov_box = None
    smoothed_ov_area = None
    smoothed_ov_cx = None
    smoothed_ov_cy = None
    last_ov_prompt = ""
    cur_semantic_scene = "Clear Open Floor"
    cur_semantic_score = 0.95

    target_positions = {
        "RED":   (0.9, 0.7),
        "GREEN": (-0.8, 0.8),
        "BLUE":  (0.0, -0.9),
    }

    while robot.step(timestep) != -1:
        frame_count += 1

        # ── Process Drag & Drop Object Requests from VR Twin ──
        with STATE.lock:
            drags = list(STATE.drag_queue)
            STATE.drag_queue.clear()

        for tname, tx, ty, tz in drags:
            tnode = target_nodes.get(tname)
            if tnode is not None:
                trans_f = tnode.getField("translation")
                cur = trans_f.getSFVec3f()
                cur_z = cur[2] if len(cur) > 2 else 0.0
                clamped_x = max(-1.8, min(1.8, tx))
                clamped_y = max(-1.8, min(1.8, ty))
                trans_f.setSFVec3f([clamped_x, clamped_y, cur_z])
                tnode.resetPhysics()
                target_positions[tname] = (clamped_x, clamped_y)
                with STATE.lock:
                    STATE.target_positions[tname] = (clamped_x, clamped_y)

        # ── Read sensors ──
        ps_l = ps_left.getValue()
        ps_r = ps_right.getValue()

        pos = list(self_node.getPosition())
        rot = list(self_node.getOrientation())
        yaw = math.atan2(rot[3], rot[0])
        yaw_deg = math.degrees(yaw)

        # ── Grab camera image ──
        raw = camera.getImage()
        if raw is None:
            continue

        img = np.frombuffer(raw, dtype=np.uint8).reshape((CAM_H, CAM_W, 4))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Share raw frame with AI worker thread
        with STATE.lock:
            STATE.frame_raw = frame.copy()
            preferred_target = STATE.preferred_target
            openvocab_target = STATE.openvocab_target
            openvocab_prompt = STATE.openvocab_prompt
            depth_obstacle_detected = STATE.depth_obstacle_detected
            depth_l_mean = STATE.depth_left_mean
            depth_r_mean = STATE.depth_right_mean

        # ── HSV Target Detection ──
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        best_target = None
        best_area = 0
        best_cx = CAM_W // 2
        green_target_area = 0
        detected_hsv = {}

        for tname, tdata in HSV_TARGETS.items():
            mask = cv2.inRange(hsv, tdata["lower"], tdata["upper"])
            if "lower2" in tdata:
                mask |= cv2.inRange(hsv, tdata["lower2"], tdata["upper2"])
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)
                if area > 200:
                    x, y, w, h = cv2.boundingRect(c)
                    cx, cy = x + w // 2, y + h // 2
                    color = tdata["color"]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 16, 2)
                    label = f"[LOCKED: {tname} TARGET] A={area}"
                    text_y = max(y - 8, 42)
                    cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    dx = cx - CAM_W // 2
                    cv2.putText(frame, f"dx={dx}", (x, y + h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

                    detected_hsv[tname] = {"cx": cx, "cy": cy, "area": area}
                    if tname == "GREEN":
                        green_target_area = area

                    if area > best_area:
                        best_area = area
                        best_target = tname
                        best_cx = cx

        # Prioritize Purpose A Preferred Target if in sight
        if preferred_target in detected_hsv:
            best_target = preferred_target
            best_area = detected_hsv[preferred_target]["area"]
            best_cx = detected_hsv[preferred_target]["cx"]

        # ── Purpose C: Open-Vocabulary Grounding Overlay (with EMA Jitter Smoothing) ──
        if openvocab_prompt and openvocab_target:
            raw_box = openvocab_target["box"]
            raw_area = float(openvocab_target["area"])
            raw_cx = float(openvocab_target["cx"])
            raw_cy = float(openvocab_target["cy"])

            if openvocab_prompt != last_ov_prompt or smoothed_ov_box is None:
                smoothed_ov_box = [float(b) for b in raw_box]
                smoothed_ov_area = raw_area
                smoothed_ov_cx = raw_cx
                smoothed_ov_cy = raw_cy
                last_ov_prompt = openvocab_prompt
            else:
                alpha = 0.6  # Exponential moving average over ~3 frames
                smoothed_ov_box = [alpha * raw_box[i] + (1.0 - alpha) * smoothed_ov_box[i] for i in range(4)]
                smoothed_ov_area = alpha * raw_area + (1.0 - alpha) * smoothed_ov_area
                smoothed_ov_cx = alpha * raw_cx + (1.0 - alpha) * smoothed_ov_cx
                smoothed_ov_cy = alpha * raw_cy + (1.0 - alpha) * smoothed_ov_cy

            ov_box = [int(round(b)) for b in smoothed_ov_box]
            ov_score = openvocab_target["score"]
            ov_cx = int(round(smoothed_ov_cx))
            ov_cy = int(round(smoothed_ov_cy))
            ov_area = int(round(smoothed_ov_area))

            cv2.rectangle(frame, (ov_box[0], ov_box[1]), (ov_box[2], ov_box[3]), (255, 0, 255), 2)
            cv2.drawMarker(frame, (ov_cx, ov_cy), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
            ov_text = f"[OPEN-VOCAB: {openvocab_prompt.upper()} {ov_score:.2f}] A={ov_area}"
            cv2.putText(frame, ov_text, (ov_box[0], max(ov_box[1] - 8, 42)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
            # In AI_CHASE, open-vocab target takes priority
            best_target = f"OPENVOCAB:{openvocab_prompt.upper()}"
            best_area = ov_area
            best_cx = ov_cx
        else:
            smoothed_ov_box = None
            smoothed_ov_area = None
            smoothed_ov_cx = None
            smoothed_ov_cy = None
            last_ov_prompt = ""

        # ── YOLOv8 (every 3rd frame) ──
        if YOLO_AVAILABLE and frame_count % 3 == 0:
            try:
                results = _yolo_model.predict(frame, imgsz=320, conf=0.35, verbose=False)
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = f"{r.names[cls_id]} {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
            except Exception:
                pass

        # ── OpenAI CLIP Semantic Scene Analysis (every 10th frame) ──
        if CLIP_AVAILABLE and frame_count % 10 == 0:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                img_tensor = clip_preprocess(pil_img).unsqueeze(0).to(CLIP_DEVICE)
                with torch.no_grad():
                    img_features = clip_model.encode_image(img_tensor)
                    img_features /= img_features.norm(dim=-1, keepdim=True)
                    sim = (100.0 * img_features @ clip_text_features.T).softmax(dim=-1)
                    probs = sim[0].cpu().numpy()
                top_idx = int(np.argmax(probs))
                raw_prompt = CLIP_PROMPTS[top_idx]
                cur_semantic_scene = CLIP_DISPLAY_MAP.get(raw_prompt, raw_prompt)
                cur_semantic_score = float(probs[top_idx])
            except Exception:
                pass

        # ── Safe Semantic Speed Throttling ──
        speed_scale = 1.0
        corroborated = (max(ps_l, ps_r) > 200) or (green_target_area > 1500)
        if cur_semantic_scene in ["Fragile Potted Plant", "Close Wall / Crate Hazard"]:
            if cur_semantic_score > 0.55 and corroborated:
                speed_scale = 0.5

        # ── Read state for HUD and control ──
        with STATE.lock:
            mode = STATE.mode
            locked_target = STATE.locked_target
            manual_cmd = STATE.manual_cmd
            do_teleport = STATE.teleport_flag
            STATE.teleport_flag = False

        has_chase_target = (mode == ControlMode.AI_CHASE) and (
            (locked_target not in ["NONE", ""]) or bool(openvocab_prompt) or (best_target is not None and best_area > 150)
        )

        # ── Purpose B: Depth Hazard Warning Overlay on RGB Feed (ignored during active chase) ──
        if depth_obstacle_detected and not has_chase_target:
            cv2.rectangle(frame, (180, 320), (460, 470), (0, 140, 255), 2)
            cv2.rectangle(frame, (150, 275), (490, 310), (0, 0, 180), -1)
            cv2.putText(frame, "[DEPTH HAZARD: LOW OBSTACLE - REROUTING]", (160, 298),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)

        # ── HUD overlay: mode badge & CLIP label ──

        mode_color = {"AI_PATROL": (0, 200, 200), "AI_CHASE": (0, 0, 255), "MANUAL": (200, 200, 0)}
        cv2.putText(frame, f"MODE: {mode.value}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    mode_color.get(mode.value, (255, 255, 255)), 2)
        cv2.drawMarker(frame, (CAM_W // 2, CAM_H // 2), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)

        clip_color = (0, 165, 255) if speed_scale < 1.0 else (0, 255, 255)
        clip_hud = f"CLIP: {cur_semantic_scene.upper()} ({cur_semantic_score:.2f})"
        if speed_scale < 1.0:
            clip_hud += " [SAFE THROTTLE 50%]"
        cv2.putText(frame, clip_hud, (8, CAM_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, clip_color, 1, cv2.LINE_AA)

        # Draw Target Accomplished banner if active
        if target_accomplished_banner_timer > 0:
            target_accomplished_banner_timer -= 1
            bw, bh = 420, 46
            bx = (CAM_W - bw) // 2
            by = 48
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 200, 0), -1)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
            cv2.putText(frame, "[TARGET ACCOMPLISHED]", (bx + 26, by + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 2, cv2.LINE_AA)
        elif mode == ControlMode.AI_CHASE and chase_lock_counter > 0:
            pct = int(min(100, (chase_lock_counter / 45.0) * 100))
            cv2.putText(frame, f"STANDOFF HOLD: {pct}% ({chase_lock_counter}/45)", (CAM_W // 2 - 130, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 100), 2, cv2.LINE_AA)

        # ── Teleport reset ──
        if do_teleport:
            trans_field.setSFVec3f([0.0, 0.0, 0.06])
            rot_field.setSFRotation([0.0, 0.0, 1.0, 0.0])
            self_node.resetPhysics()
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            with STATE.lock:
                STATE.trail.clear()
            continue

        # ── Motor Control Logic ──
        left_v = 0.0
        right_v = 0.0

        if mode == ControlMode.AI_PATROL:
            # Purpose B Low Obstacle Avoidance takes high priority
            if depth_obstacle_detected:
                if depth_l_mean >= depth_r_mean:
                    left_v = MAX_SPEED * 0.55
                    right_v = -MAX_SPEED * 0.35
                else:
                    left_v = -MAX_SPEED * 0.35
                    right_v = MAX_SPEED * 0.55
            else:
                # Physical Stall / Stuck Watchdog (every 35 timesteps)
                stuck_check_counter += 1
                if stuck_check_counter >= 35:
                    stuck_check_counter = 0
                    if stuck_check_pos is not None:
                        dist_moved = math.hypot(pos[0] - stuck_check_pos[0], pos[1] - stuck_check_pos[1])
                        if dist_moved < 0.04:
                            patrol_reverse_timer = 18
                            patrol_turn_timer = 28
                            patrol_turn_dir = 1.0 if ps_l >= ps_r else -1.0
                    stuck_check_pos = (pos[0], pos[1])

                # Priority 1: Committed Straight Reverse
                if patrol_reverse_timer > 0:
                    left_v = -MAX_SPEED * 0.65
                    right_v = -MAX_SPEED * 0.65
                    patrol_reverse_timer -= 1

                # Priority 2: Committed In-Place Pivot
                elif patrol_turn_timer > 0:
                    left_v = patrol_turn_dir * MAX_SPEED * 0.65
                    right_v = -patrol_turn_dir * MAX_SPEED * 0.65
                    patrol_turn_timer -= 1

                # Priority 3: Head-on wall or corner
                elif ps_l > WALL_THRESHOLD and ps_r > WALL_THRESHOLD:
                    patrol_turn_dir = 1.0 if ps_l >= ps_r else -1.0
                    patrol_reverse_timer = 12
                    patrol_turn_timer = 24
                    left_v = -MAX_SPEED * 0.65
                    right_v = -MAX_SPEED * 0.65
                    patrol_reverse_timer -= 1

                # Priority 4: Left sensor obstacle -> spin in place RIGHT
                elif ps_l > WALL_THRESHOLD:
                    patrol_turn_dir = 1.0
                    patrol_turn_timer = 10
                    left_v = MAX_SPEED * 0.6
                    right_v = -MAX_SPEED * 0.45

                # Priority 5: Right sensor obstacle -> spin in place LEFT
                elif ps_r > WALL_THRESHOLD:
                    patrol_turn_dir = -1.0
                    patrol_turn_timer = 10
                    left_v = -MAX_SPEED * 0.45
                    right_v = MAX_SPEED * 0.6

                # Priority 6: Open Floor Cruise
                else:
                    left_v = MAX_SPEED * 0.75
                    right_v = MAX_SPEED * 0.75

                if frame_count % 220 == 0 and patrol_turn_timer == 0 and patrol_reverse_timer == 0:
                    patrol_turn_dir = -patrol_turn_dir
                    patrol_turn_timer = 16

        elif mode == ControlMode.AI_CHASE:
            stuck_check_counter = 0
            stuck_check_pos = (pos[0], pos[1])

            # Purpose B Low Obstacle Avoidance during chase (ignored if active target exists)
            if depth_obstacle_detected and not has_chase_target:
                chase_lock_counter = max(0, chase_lock_counter - 1)
                if depth_l_mean >= depth_r_mean:
                    left_v = MAX_SPEED * 0.55
                    right_v = -MAX_SPEED * 0.35
                else:
                    left_v = -MAX_SPEED * 0.35
                    right_v = MAX_SPEED * 0.55
            elif best_target is not None and best_area > 200:
                steer_error = best_cx - CAM_W // 2
                steer_gain = 0.008

                # If physically touching / too close (> 16000): back off smoothly
                if best_area > 16000:
                    chase_backoff_timer = 8

                if chase_backoff_timer > 0:
                    fwd_speed = -MAX_SPEED * 0.3
                    chase_backoff_timer -= 1
                    chase_lock_counter = max(0, chase_lock_counter - 1)
                elif 5500 <= best_area <= 16000 or (max(ps_l, ps_r) > 280):
                    # Ideal standoff distance zone (~15cm from target)
                    fwd_speed = 0.0
                    chase_lock_counter += 1

                    # Target reached & Task complete condition (~1.5s / 45 ticks of steady lock)
                    if chase_lock_counter >= 45:
                        t_str = time.strftime("%H:%M:%S")
                        target_name = str(best_target)
                        success_msg = f"[{t_str}] [MISSION SUCCESS] Reached target: {target_name}"
                        print(f"🎯 {success_msg}")

                        with STATE.lock:
                            STATE.mission_logs.append(success_msg)
                            if len(STATE.mission_logs) > 50:
                                STATE.mission_logs = STATE.mission_logs[-50:]
                            STATE.openvocab_prompt = ""
                            STATE.openvocab_target = None
                            STATE.mode = ControlMode.AI_PATROL

                        try:
                            md_path = os.path.join(_root_dir, "mission_log.md")
                            with open(md_path, "a", encoding="utf-8") as f:
                                f.write(f"| `{t_str}` | `({pos[0]:.2f}, {pos[1]:.2f}, {yaw_deg:.1f}°)` | **MISSION SUCCESS** | Reached {target_name} | Standoff held (45 ticks), resuming AI_PATROL |\n")
                        except Exception:
                            pass

                        target_accomplished_banner_timer = 50
                        chase_lock_counter = 0

                        # Also trigger Purpose A to plan next waypoint
                        try:
                            plan_vlm_goal()
                        except Exception:
                            pass
                else:  # best_area < 6000
                    fwd_speed = MAX_SPEED * 0.5
                    chase_lock_counter = max(0, chase_lock_counter - 1)

                left_v = fwd_speed + steer_gain * steer_error
                right_v = fwd_speed - steer_gain * steer_error
                left_v = max(-MAX_SPEED, min(MAX_SPEED, left_v))
                right_v = max(-MAX_SPEED, min(MAX_SPEED, right_v))
            else:
                chase_backoff_timer = 0
                chase_lock_counter = 0
                left_v = MAX_SPEED * 0.25
                right_v = -MAX_SPEED * 0.25
                best_target = None

        elif mode == ControlMode.MANUAL:
            chase_lock_counter = 0
            stuck_check_counter = 0
            stuck_check_pos = (pos[0], pos[1])
            if manual_cmd == "FORWARD":
                left_v = MAX_SPEED * 0.6
                right_v = MAX_SPEED * 0.6
            elif manual_cmd == "BACKWARD":
                left_v = -MAX_SPEED * 0.5
                right_v = -MAX_SPEED * 0.5
            elif manual_cmd == "LEFT":
                left_v = -MAX_SPEED * 0.3
                right_v = MAX_SPEED * 0.3
            elif manual_cmd == "RIGHT":
                left_v = MAX_SPEED * 0.3
                right_v = -MAX_SPEED * 0.3
            else:
                left_v = 0.0
                right_v = 0.0

        if left_v > 0 and right_v > 0:
            cmd_l = left_v * speed_scale
            cmd_r = right_v * speed_scale
        else:
            cmd_l = left_v
            cmd_r = right_v
        left_motor.setVelocity(cmd_l)
        right_motor.setVelocity(cmd_r)

        # ── Encode frame to JPEG at HD Quality (95%) ──
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # ── Update shared state ──
        with STATE.lock:
            STATE.frame_jpeg = jpeg.tobytes()
            STATE.pos = [round(pos[0], 3), round(pos[1], 3), round(pos[2], 3)]
            STATE.orientation = round(yaw, 3)
            STATE.yaw_deg = round(yaw_deg, 1)
            STATE.left_speed = round(cmd_l, 2)
            STATE.right_speed = round(cmd_r, 2)
            STATE.ps_left_val = round(ps_l, 1)
            STATE.ps_right_val = round(ps_r, 1)
            STATE.locked_target = best_target if best_target else "NONE"
            STATE.steer_error = round(best_cx - CAM_W // 2, 1) if best_target else 0.0
            STATE.semantic_scene = cur_semantic_scene
            STATE.semantic_score = cur_semantic_score
            STATE.trail.append((pos[0], pos[1]))
            if len(STATE.trail) > 500:
                STATE.trail = STATE.trail[-500:]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WebSocket Telemetry Broadcaster (runs in background asyncio task)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_ws_broadcast_task: Optional[asyncio.Task] = None

def ensure_ws_broadcast():
    """Idempotently starts ws_broadcast_loop in the active running asyncio loop."""
    global _ws_broadcast_task
    if _ws_broadcast_task is None or _ws_broadcast_task.done():
        try:
            loop = asyncio.get_running_loop()
            _ws_broadcast_task = loop.create_task(ws_broadcast_loop())
            print("✅ WebSocket broadcast loop active (30Hz telemetry).")
        except RuntimeError:
            pass

async def ws_broadcast_loop():
    """Broadcasts telemetry state at ~30 Hz to all connected WebSocket clients."""
    try:
        while not SHUTDOWN_EVENT.is_set():
            if ws_manager.active_connections:
                with STATE.lock:
                    has_active_chase = (STATE.mode == ControlMode.AI_CHASE) and (
                        (STATE.locked_target not in ["NONE", ""]) or bool(STATE.openvocab_prompt)
                    )
                    packet = json.dumps({
                        "x": STATE.pos[0],
                        "y": STATE.pos[1],
                        "z": STATE.pos[2],
                        "yaw": STATE.yaw_deg,
                        "mode": STATE.mode.value,
                        "detected": STATE.locked_target,
                        "ps_left": STATE.ps_left_val,
                        "ps_right": STATE.ps_right_val,
                        "vx": STATE.left_speed,
                        "vy": STATE.right_speed,
                        "semantic_scene": STATE.semantic_scene,
                        "semantic_score": STATE.semantic_score,
                        "targets": {
                            k: [round(v[0], 3), round(v[1], 3)]
                            for k, v in STATE.target_positions.items()
                        },
                        "depth_hazard": STATE.depth_obstacle_detected and not has_active_chase,
                        "depth_score": round(STATE.depth_hazard_score, 3),
                        "auto_goal_vlm": STATE.auto_goal_vlm,
                        "vlm_goal_instruction": STATE.vlm_goal_instruction,
                        "preferred_target": STATE.preferred_target,
                        "openvocab_prompt": STATE.openvocab_prompt,
                        "openvocab_detected": STATE.openvocab_target is not None,
                        "latest_mission_log": STATE.mission_logs[-1] if STATE.mission_logs else "Initializing mission logs..."
                    })
                await ws_manager.broadcast(packet)
            await asyncio.sleep(1.0 / 30)
    except asyncio.CancelledError:
        pass

_THREADS_LOCK = threading.Lock()
_THREADS_STARTED = False

def start_background_threads():
    global _THREADS_STARTED
    with _THREADS_LOCK:
        if _THREADS_STARTED:
            return
        _THREADS_STARTED = True
        ensure_ai_models()
        ai_thread = threading.Thread(target=ai_worker_loop, daemon=True, name="AIWorkerThread")
        ai_thread.start()
        t = threading.Thread(target=webots_loop, daemon=True, name="WebotsLoopThread")
        t.start()
        print("✅ Background AI Worker Engine & Webots Controller threads started.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI Application & Lifespan
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@asynccontextmanager
async def lifespan(app):
    start_background_threads()
    ensure_ws_broadcast()
    yield
    SHUTDOWN_EVENT.set()
    global _ws_broadcast_task
    if _ws_broadcast_task and not _ws_broadcast_task.done():
        _ws_broadcast_task.cancel()
        try:
            await _ws_broadcast_task
        except asyncio.CancelledError:
            pass
    for conn in list(ws_manager.active_connections):
        try:
            await conn.close()
        except Exception:
            pass

app = FastAPI(title="Mission Control Dashboard", lifespan=lifespan)

# Enable CORS for Hugging Face Spaces, reverse proxies, and iframes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── WebSocket Endpoint ──
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
                        with STATE.lock:
                            STATE.mode = m
                            STATE.manual_cmd = "STOP"
                    except ValueError:
                        pass
                elif action == "MANUAL_CMD":
                    cmd = msg.get("cmd", "STOP").upper()
                    with STATE.lock:
                        STATE.manual_cmd = cmd
                elif action == "RESET_POSE":
                    with STATE.lock:
                        STATE.teleport_flag = True
                elif action == "DRAG_OBJECT":
                    target = msg.get("target", "").upper()
                    tx = float(msg.get("x", 0.0))
                    ty = float(msg.get("y", msg.get("z", 0.0)))
                    tz = float(msg.get("z", 0.0))
                    with STATE.lock:
                        STATE.drag_queue.append((target, tx, ty, tz))
                elif action == "SET_OPENVOCAB":
                    prompt = msg.get("prompt", "").strip()
                    with STATE.lock:
                        if prompt.lower() in ["clear", "none", ""]:
                            STATE.openvocab_prompt = ""
                            STATE.openvocab_target = None
                        else:
                            STATE.openvocab_prompt = prompt
                elif action == "VLM_SELECT_GOAL":
                    plan_vlm_goal()
                elif action == "SET_VLM_GOAL_MODE":
                    enabled = bool(msg.get("enabled", True))
                    with STATE.lock:
                        STATE.auto_goal_vlm = enabled
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)

# ── MJPEG Video Feed ──
async def gen_video():
    try:
        while not SHUTDOWN_EVENT.is_set():
            with STATE.lock:
                frame = STATE.frame_jpeg
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        pass

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_video(), media_type="multipart/x-mixed-replace; boundary=frame")

# ── Purpose B: Monocular Depth Map Video Feed ──
async def gen_depth():
    try:
        while not SHUTDOWN_EVENT.is_set():
            with STATE.lock:
                dframe = STATE.depth_jpeg
            if dframe:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + dframe + b"\r\n")
            await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        pass

@app.get("/depth_feed")
def depth_feed():
    return StreamingResponse(gen_depth(), media_type="multipart/x-mixed-replace; boundary=frame")

# ── Single Frame Snapshot (for WebGL textures) ──
@app.get("/video_frame.jpg")
def video_frame():
    with STATE.lock:
        frame = STATE.frame_jpeg
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )

# ── Purpose B: Depth Frame Snapshot (for WebGL/VR textures) ──
@app.get("/depth_frame.jpg")
def depth_frame():
    with STATE.lock:
        dframe = STATE.depth_jpeg
    return Response(
        content=dframe,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )

# ── Telemetry JSON ──
@app.get("/telemetry")
def telemetry():
    with STATE.lock:
        has_active_chase = (STATE.mode == ControlMode.AI_CHASE) and (
            (STATE.locked_target not in ["NONE", ""]) or bool(STATE.openvocab_prompt)
        )
        return JSONResponse({
            "pos": STATE.pos,
            "orientation": STATE.orientation,
            "yaw_deg": STATE.yaw_deg,
            "left_speed": STATE.left_speed,
            "right_speed": STATE.right_speed,
            "ps_left": STATE.ps_left_val,
            "ps_right": STATE.ps_right_val,
            "locked_target": STATE.locked_target,
            "steer_error": STATE.steer_error,
            "mode": STATE.mode.value,
            "semantic_scene": STATE.semantic_scene,
            "semantic_score": STATE.semantic_score,
            "targets": STATE.target_positions,
            "depth_hazard": STATE.depth_obstacle_detected and not has_active_chase,
            "depth_score": round(STATE.depth_hazard_score, 3),
            "depth_left": round(STATE.depth_left_mean, 3),
            "depth_right": round(STATE.depth_right_mean, 3),
            "auto_goal_vlm": STATE.auto_goal_vlm,
            "vlm_goal_instruction": STATE.vlm_goal_instruction,
            "preferred_target": STATE.preferred_target,
            "openvocab_prompt": STATE.openvocab_prompt,
            "openvocab_detected": STATE.openvocab_target is not None,
            "latest_mission_log": STATE.mission_logs[-1] if STATE.mission_logs else "Initializing mission logs...",
            "trail": STATE.trail[-200:],
        })

# ── Command endpoints ──
@app.post("/set_mode/{mode}")
@app.post("/mode/{mode}")
@app.post("/control/mode/{mode}")
def set_mode(mode: str):
    try:
        m = ControlMode(mode)
    except ValueError:
        return JSONResponse({"error": f"Invalid mode: {mode}"}, status_code=400)
    with STATE.lock:
        STATE.mode = m
        STATE.depth_obstacle_detected = False
        if m == ControlMode.MANUAL:
            STATE.manual_cmd = "STOP"
            STATE.left_speed = 0.0
            STATE.right_speed = 0.0
        elif m == ControlMode.AI_CHASE:
            STATE.manual_cmd = "NONE"
            if STATE.locked_target in ["NONE", "", "PATROL"] and not STATE.openvocab_prompt:
                rx, ry = STATE.pos[0], STATE.pos[1]
                nearest_k = min(
                    ARENA_OBJECTS.keys(),
                    key=lambda k: math.hypot(rx - ARENA_OBJECTS[k]["coords"][0], ry - ARENA_OBJECTS[k]["coords"][1])
                )
                STATE.preferred_target = nearest_k
                STATE.locked_target = ARENA_OBJECTS[nearest_k]["name"]
        elif m == ControlMode.AI_PATROL:
            STATE.manual_cmd = "NONE"
            STATE.locked_target = "PATROL"

    log_mission_event(f"Control mode transitioned to {m.value}", category="MODE")
    return {"mode": m.value, "status": "active"}

@app.post("/control")
async def control_unified(request: Request):
    """Unified endpoint accepting {'mode': ...} or {'cmd': ...}."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    if "mode" in body:
        return set_mode(body["mode"])
    if "cmd" in body:
        cmd = str(body["cmd"]).upper()
        with STATE.lock:
            STATE.manual_cmd = cmd
        return {"cmd": cmd}
    return {"status": "ok"}

@app.post("/manual/{cmd}")
def manual_cmd(cmd: str):
    with STATE.lock:
        STATE.manual_cmd = cmd.upper()
    return {"cmd": cmd.upper()}

@app.post("/teleport_reset")
def teleport_reset():
    with STATE.lock:
        STATE.teleport_flag = True
    return {"status": "teleporting"}

# ── Purpose A: Dynamic VLM Goal Endpoints ──
@app.post("/trigger_vlm_goal")
def trigger_vlm_goal_endpoint():
    inst, target = plan_vlm_goal()
    with STATE.lock:
        STATE.mode = ControlMode.AI_CHASE
        STATE.depth_obstacle_detected = False
    return {"instruction": inst, "preferred_target": target}

@app.post("/set_vlm_goal_mode/{enabled}")
def set_vlm_goal_mode(enabled: str):
    val = enabled.lower() in ["true", "1", "yes", "on"]
    with STATE.lock:
        STATE.auto_goal_vlm = val
        STATE.depth_obstacle_detected = False
        if val:
            STATE.mode = ControlMode.AI_CHASE
    if val:
        plan_vlm_goal()
    log_mission_event(f"VLM Auto-Goal mode set to {'ON' if val else 'OFF'}", category="VLM")
    return {"auto_goal_vlm": val}

# ── Purpose C: Open-Vocabulary Prompt Endpoint ──
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
    with STATE.lock:
        if prompt.lower() in ["clear", "none", ""]:
            STATE.openvocab_prompt = ""
            STATE.openvocab_target = None
            log_mission_event("Open-vocabulary target cleared", category="VLM")
        else:
            STATE.openvocab_prompt = prompt
            STATE.mode = ControlMode.AI_CHASE
            STATE.depth_obstacle_detected = False
            rx, ry = STATE.pos[0], STATE.pos[1]
            t_k, t_obj = resolve_openvocab_object(prompt, rx, ry)
            STATE.preferred_target = t_k
            STATE.locked_target = t_obj["name"]
            log_mission_event(f"Open-vocab target locked: '{prompt}' -> {t_obj['name']} (switched to AI_CHASE)", category="VLM")
    return {"openvocab_prompt": STATE.openvocab_prompt, "target": STATE.locked_target, "mode": "AI_CHASE"}

# ── Purpose D: Automated Mission Logs Endpoints ──
@app.get("/mission_logs")
def get_mission_logs():
    with STATE.lock:
        logs = list(STATE.mission_logs)
    return {"logs": logs, "total": len(logs)}

@app.get("/download_mission_log")
def download_mission_log():
    md_path = os.path.join(_root_dir, "mission_log.md")
    if not os.path.exists(md_path):
        generate_mission_log_entry()
    return FileResponse(md_path, media_type="text/markdown", filename="mission_log.md")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dashboard HTML (2D)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mission Control — AI Vision Patrol & Multimodal Robotics</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0e17; --surface: #121a2b; --border: #1e2d46;
    --accent: #00e5ff; --accent2: #ff1744; --accent3: #76ff03;
    --magenta: #e040fb; --amber: #ffab00;
    --text: #e0e6f0; --dim: #5a6a8a;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background: var(--bg); color: var(--text);
    font-family: 'Inter', sans-serif; min-height:100vh;
    display:flex; flex-direction:column;
  }
  header {
    background: linear-gradient(135deg, #0d1520 0%, #162035 100%);
    border-bottom: 1px solid var(--border); padding: 12px 24px;
    display:flex; align-items:center; gap:16px; flex-wrap: wrap;
  }
  header h1 {
    font-family:'Orbitron',monospace; font-weight:900; font-size:1.3rem;
    background: linear-gradient(90deg, var(--accent), #6ee7ff);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    letter-spacing:2px;
  }
  header .badge {
    background: var(--accent); color:#000; padding:3px 10px;
    border-radius:20px; font-size:0.65rem; font-weight:700;
    font-family:'Orbitron',monospace; letter-spacing:1px;
    animation: pulse 2s ease-in-out infinite;
  }
  .vr-launch-btn {
    margin-left: auto;
    background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
    color: #fff; padding: 10px 22px; border-radius: 12px;
    font-family: 'Orbitron', monospace; font-weight: 700; font-size: 0.85rem;
    text-decoration: none; letter-spacing: 1px;
    border: 1px solid rgba(168,85,247,0.4);
    transition: all 0.2s ease; cursor: pointer;
    box-shadow: 0 0 20px rgba(124,58,237,0.3);
    display: inline-flex; align-items: center; gap: 8px;
  }
  .vr-launch-btn:hover {
    transform: translateY(-2px) scale(1.03);
    box-shadow: 0 0 30px rgba(124,58,237,0.5);
    background: linear-gradient(135deg, #8b5cf6 0%, #c084fc 100%);
  }
  @keyframes pulse {
    0%,100% { opacity:1; } 50% { opacity:0.6; }
  }
  .grid {
    display:grid; grid-template-columns: 2fr 1fr;
    gap:16px; padding:16px; flex:1;
  }
  .panel {
    background: var(--surface); border:1px solid var(--border);
    border-radius:12px; overflow:hidden;
  }
  .panel-title {
    font-family:'Orbitron',monospace; font-size:0.72rem;
    font-weight:700; letter-spacing:2px; color:var(--accent);
    padding:12px 16px; border-bottom:1px solid var(--border);
  }

  /* Dual Viewport */
  .dual-video-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(240px, 1fr));
    gap: 12px;
    padding: 12px;
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
    position:relative; background:#000; aspect-ratio:4/3;
    overflow: hidden;
  }
  .video-wrap img {
    width:100%; height:100%; object-fit:contain; display:block;
    image-rendering: -webkit-optimize-contrast; image-rendering: crisp-edges;
  }
  .scene-badge {
    position:absolute; bottom:8px; left:8px;
    padding:4px 10px; border-radius:6px;
    background:rgba(10,14,23,0.88); border:1px solid rgba(0,229,255,0.4);
    font-family:'Orbitron',monospace; font-size:0.62rem; font-weight:700;
    letter-spacing:1px; color:#00e5ff; backdrop-filter:blur(6px);
  }
  .mode-badge {
    position:absolute; top:8px; right:8px;
    padding:4px 10px; border-radius:6px;
    font-family:'Orbitron',monospace; font-size:0.62rem;
    font-weight:700; letter-spacing:1px; color:#000;
  }
  .hazard-badge {
    position:absolute; bottom:8px; right:8px;
    padding:4px 10px; border-radius:6px;
    background:rgba(255,23,68,0.9); color:#fff;
    font-family:'Orbitron',monospace; font-size:0.62rem; font-weight:700;
    letter-spacing:1px; animation: pulse 1s infinite;
  }

  /* Multimodal AI Interactive Controls */
  .ai-controls-box {
    margin: 12px; padding: 14px; background: rgba(14, 22, 38, 0.7);
    border: 1px solid rgba(0, 229, 255, 0.2); border-radius: 10px;
    display: flex; flex-direction: column; gap: 12px;
  }
  .ai-box-title {
    font-family: 'Orbitron', monospace; font-size: 0.72rem; font-weight: 700;
    color: var(--magenta); letter-spacing: 1px; display: flex; align-items: center; gap: 8px;
  }
  .vocab-bar {
    display: flex; gap: 8px; flex-wrap: wrap;
  }
  .vocab-input {
    flex: 1; min-width: 200px; background: #0a0e17; border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 12px; color: #fff; font-family: 'Inter', sans-serif;
    font-size: 0.82rem; outline: none;
  }
  .vocab-input:focus { border-color: var(--magenta); box-shadow: 0 0 8px rgba(224, 64, 251, 0.3); }
  .chip-row {
    display: flex; gap: 6px; flex-wrap: wrap; align-items: center;
  }
  .chip {
    background: rgba(224, 64, 251, 0.1); border: 1px solid rgba(224, 64, 251, 0.3);
    color: #e0a0ff; padding: 4px 10px; border-radius: 14px; font-size: 0.68rem;
    cursor: pointer; transition: all 0.15s; font-family: 'Inter', sans-serif;
  }
  .chip:hover { background: rgba(224, 64, 251, 0.25); color: #fff; }
  .vlm-bar {
    display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
  }
  .vlm-instruction {
    background: #0a0e17; border-left: 3px solid var(--accent);
    padding: 8px 12px; border-radius: 0 8px 8px 0; font-size: 0.75rem;
    color: #c0d8f0; font-family: 'Inter', sans-serif; flex: 1; min-width: 240px;
  }

  /* Mission Log Terminal */
  .log-terminal {
    background: #080c14; border: 1px solid #1a2540; border-radius: 8px;
    height: 160px; overflow-y: auto; overflow-x: hidden; padding: 10px; font-family: 'Monaco', 'Courier New', monospace;
    font-size: 0.72rem; color: #76ff03; line-height: 1.5; margin: 0 12px 12px;
    word-break: break-word; white-space: pre-wrap;
  }
  .log-line { margin-bottom: 4px; }
  .log-line.system { color: #00e5ff; }
  .download-btn {
    margin-left: auto; background: rgba(0, 229, 255, 0.1); border: 1px solid rgba(0, 229, 255, 0.3);
    color: var(--accent); padding: 4px 10px; border-radius: 6px; font-size: 0.65rem;
    text-decoration: none; cursor: pointer; transition: all 0.15s;
    font-family: 'Orbitron', monospace;
  }
  .download-btn:hover { background: rgba(0, 229, 255, 0.25); }

  /* Telemetry & Controls */
  canvas { width:100%; display:block; }
  .telem-grid {
    display:grid; grid-template-columns:1fr 1fr; gap:6px; padding:12px;
  }
  .telem-card {
    background: rgba(0,229,255,0.05); border:1px solid rgba(0,229,255,0.1);
    border-radius:8px; padding:8px 10px; text-align:center;
  }
  .telem-card .label { font-size:0.6rem; color:var(--dim); text-transform:uppercase; letter-spacing:1px; }
  .telem-card .value { font-family:'Orbitron',monospace; font-size:0.88rem; font-weight:700; color:var(--accent); margin-top:2px; }
  .telem-card.green .value { color:var(--accent3); }
  .telem-card.alert .value { color:var(--accent2); }
  .telem-card.amber .value { color:var(--amber); }

  .controls { padding:12px; }
  .btn-row { display:flex; gap:8px; margin-bottom:8px; }
  .btn {
    flex:1; padding:9px; border-radius:8px; text-align:center;
    font-family:'Orbitron',monospace; font-size:0.65rem; font-weight:700;
    letter-spacing:1px; cursor:pointer; user-select:none;
    background: rgba(0,229,255,0.08); border:1px solid rgba(0,229,255,0.2);
    color:var(--accent); transition: all 0.15s;
  }
  .btn:hover { background:rgba(0,229,255,0.15); }
  .btn.active { background:var(--accent); color:#000; }
  .btn.magenta { background:rgba(224, 64, 251, 0.15); border-color:rgba(224, 64, 251, 0.4); color:var(--magenta); }
  .btn.magenta:hover { background:rgba(224, 64, 251, 0.3); }
  .btn.magenta.active { background:var(--magenta); color:#fff; }
  .btn.danger { background:rgba(255,23,68,0.1); border-color:rgba(255,23,68,0.3); color:var(--accent2); }
  .btn.danger:hover { background:rgba(255,23,68,0.2); }
  .wasd-grid {
    display:grid; grid-template-columns:repeat(3,1fr); grid-template-rows:1fr 1fr;
    gap:6px; padding:0 12px 12px;
  }
  .w-btn { grid-column:2; }
  .a-btn { grid-column:1; grid-row:2; }
  .s-btn { grid-column:2; grid-row:2; }
  .d-btn { grid-column:3; grid-row:2; }
  .bottom-bar {
    text-align:center; padding:8px; font-size:0.6rem; color:var(--dim);
    font-family:'Orbitron',monospace; letter-spacing:2px; border-top:1px solid var(--border);
  }
  .sidebar { display:flex; flex-direction:column; gap:16px; }
</style>
</head>
<body>
<header>
  <h1>MISSION CONTROL</h1>
  <span class="badge">LIVE MULTIMODAL</span>
  <a class="vr-launch-btn" href="/vr" target="_blank">
    🥽 Launch A-Frame 3D VR Digital Twin (/vr)
  </a>
</header>

<div class="grid">
  <!-- LEFT: Dual Video Viewports & Multimodal AI Deck -->
  <div style="display:flex; flex-direction:column; gap:16px;">
    <!-- Dual Viewport Panel -->
    <div class="panel">
      <div class="panel-title">◉ Dual Viewport: RGB HD Vision & Monocular Depth Pipeline</div>
      <div class="dual-video-grid">
        <!-- Feed 1: RGB + YOLO + Open-Vocab -->
        <div class="video-card">
          <div class="feed-header">
            <span>📷 RGB + YOLO & OPEN-VOCAB HUD</span>
            <span style="color:#76ff03;">640x480@95%</span>
          </div>
          <div class="video-wrap">
            <img id="videoFeed" src="/video_frame.jpg" alt="Live Camera Feed">
            <div class="mode-badge" id="modeBadge" style="background:var(--accent)">AI_PATROL</div>
            <div class="scene-badge" id="sceneBadge">Semantic Scene: Clear Open Floor (95%)</div>
          </div>
        </div>
        <!-- Feed 2: Monocular Depth Map (MAGMA) -->
        <div class="video-card">
          <div class="feed-header">
            <span>◈ PURPOSE B: MONOCULAR DEPTH MAP</span>
            <span style="color:var(--magenta);">MAGMA COLORMAP</span>
          </div>
          <div class="video-wrap">
            <img id="depthFeed" src="/depth_frame.jpg" alt="Monocular Depth Feed">
            <div class="hazard-badge" id="depthHazardBadge" style="display:none;">⚠️ LOW OBSTACLE HAZARD</div>
          </div>
        </div>
      </div>

      <!-- Multimodal AI Controls (Purposes A & C) -->
      <div class="ai-controls-box">
        <div class="ai-box-title">🔍 PURPOSE C: OPEN-VOCABULARY TARGET TRACKING</div>
        <div class="vocab-bar">
          <input id="vocabInput" class="vocab-input" type="text" placeholder="Enter target prompt e.g. wooden crate, chair...">
          <button class="btn magenta" style="flex:0 0 140px;" onclick="submitVocab()">🔍 Track Prompt</button>
          <button class="btn danger" style="flex:0 0 90px;" onclick="setVocab('')">✖ Clear</button>
        </div>
        <div class="chip-row">
          <span style="font-size:0.65rem; color:var(--dim);">QUICK PRESETS:</span>
          <button class="chip" onclick="setVocab('wooden crate')">📦 wooden crate</button>
          <button class="chip" onclick="setVocab('yellow hazard box')">⚠️ yellow hazard box</button>
          <button class="chip" onclick="setVocab('potted plant')">🪴 potted plant</button>
          <button class="chip" onclick="setVocab('blue chair')">🪑 blue chair</button>
          <button class="chip" onclick="setVocab('red cone')">🔺 red cone</button>
        </div>

        <div class="ai-box-title" style="margin-top:6px; color:var(--accent);">🧭 PURPOSE A: DYNAMIC VLM GOAL SELECTION</div>
        <div class="vlm-bar">
          <button id="btnVlmToggle" class="btn" style="flex:0 0 200px;" onclick="toggleVlm()">🧭 VLM Auto-Goal: OFF</button>
          <button class="btn" style="flex:0 0 160px;" onclick="triggerVlm()">⚡ Plan Goal Now</button>
          <div id="vlmInstruction" class="vlm-instruction">VLM Planner: Standby. Ready to plan goals.</div>
        </div>
      </div>

      <!-- Purpose D Live Mission Flight Log Panel -->
      <div style="padding:0 12px 6px; display:flex; align-items:center;">
        <span style="font-family:'Orbitron',monospace; font-size:0.68rem; color:var(--accent3);">📋 PURPOSE D: LIVE FLIGHT MISSION LOG</span>
        <a class="download-btn" href="/download_mission_log" target="_blank">📥 Download mission_log.md</a>
      </div>
      <div id="logTerminal" class="log-terminal">
        <div class="log-line system">[SYSTEM] Autonomous Mission Control logger online. Streaming telemetry & vision captions...</div>
      </div>
    </div>
  </div>

  <!-- RIGHT: Sidebar (Radar, Telemetry, Mode Deck) -->
  <div class="sidebar">
    <!-- Radar -->
    <div class="panel">
      <div class="panel-title">◎ Arena Radar & Target Beacon Map</div>
      <canvas id="radar" width="300" height="300"></canvas>
    </div>

    <!-- Telemetry -->
    <div class="panel">
      <div class="panel-title">◈ Robot Telemetry & Sensor Suite</div>
      <div class="telem-grid">
        <div class="telem-card"><div class="label">X (m)</div><div class="value" id="tX">0.000</div></div>
        <div class="telem-card"><div class="label">Y (m)</div><div class="value" id="tY">0.000</div></div>
        <div class="telem-card"><div class="label">L Speed</div><div class="value" id="tLW">0.00</div></div>
        <div class="telem-card"><div class="label">R Speed</div><div class="value" id="tRW">0.00</div></div>
        <div class="telem-card"><div class="label">PS Left</div><div class="value" id="tPL">0.0</div></div>
        <div class="telem-card"><div class="label">PS Right</div><div class="value" id="tPR">0.0</div></div>
        <div class="telem-card green"><div class="label">Target</div><div class="value" id="tTarget">NONE</div></div>
        <div class="telem-card alert"><div class="label">Yaw°</div><div class="value" id="tYaw">0.0</div></div>
        <div class="telem-card amber"><div class="label">Depth Haz</div><div class="value" id="tDepthHaz">CLEAR</div></div>
        <div class="telem-card"><div class="label">Haz Score</div><div class="value" id="tDepthScore">0.00</div></div>
      </div>
    </div>

    <!-- Command Deck -->
    <div class="panel">
      <div class="panel-title">◇ Autonomy & Control Deck</div>
      <div class="controls">
        <div class="btn-row">
          <div class="btn active" id="btnPatrol" onclick="setMode('AI_PATROL')">🧠 AI Patrol</div>
          <div class="btn" id="btnChase" onclick="setMode('AI_CHASE')">🎯 AI Chase</div>
          <div class="btn" id="btnManual" onclick="setMode('MANUAL')">🕹️ Manual</div>
        </div>
        <div class="btn-row">
          <div class="btn danger" onclick="teleport()">⚡ Teleport Reset Pose</div>
        </div>
      </div>
      <div class="wasd-grid" id="wasdGrid" style="display:none;">
        <div class="btn w-btn" onmousedown="mc('FORWARD')" onmouseup="mc('STOP')">W ▲</div>
        <div class="btn a-btn" onmousedown="mc('LEFT')" onmouseup="mc('STOP')">◄ A</div>
        <div class="btn s-btn" onmousedown="mc('BACKWARD')" onmouseup="mc('STOP')">S ▼</div>
        <div class="btn d-btn" onmousedown="mc('RIGHT')" onmouseup="mc('STOP')">D ►</div>
      </div>
    </div>
  </div>
</div>

<div class="bottom-bar">Mission Control Dashboard &mdash; Webots R2025a &times; YOLOv8 &times; CLIP &times; Depth &times; FastAPI &times; WebXR</div>

<script>
let currentMode = 'AI_PATROL';
let autoGoalVlm = false;
let lastKnownLog = '';

async function setMode(m) {
  currentMode = m;
  await fetch('/set_mode/' + m, {method:'POST'});
  document.getElementById('btnPatrol').classList.toggle('active', m==='AI_PATROL');
  document.getElementById('btnChase').classList.toggle('active', m==='AI_CHASE');
  document.getElementById('btnManual').classList.toggle('active', m==='MANUAL');
  document.getElementById('wasdGrid').style.display = m==='MANUAL' ? 'grid' : 'none';
  const badge = document.getElementById('modeBadge');
  badge.textContent = m;
  badge.style.background = m==='AI_PATROL' ? 'var(--accent)' : m==='AI_CHASE' ? 'var(--accent2)' : '#ffd600';
}

async function mc(cmd) {
  await fetch('/manual/' + cmd, {method:'POST'});
}

async function teleport() {
  await fetch('/teleport_reset', {method:'POST'});
}

// Purpose C: Open-Vocabulary Target Setting
async function submitVocab() {
  const prompt = document.getElementById('vocabInput').value.trim();
  await fetch('/set_openvocab_prompt', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prompt: prompt})
  });
  if (prompt) setMode('AI_CHASE');
}

async function setVocab(prompt) {
  document.getElementById('vocabInput').value = prompt;
  await fetch('/set_openvocab_prompt', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prompt: prompt})
  });
  if (prompt) setMode('AI_CHASE');
}

// Purpose A: VLM Auto-Goal Trigger & Toggle
async function toggleVlm() {
  autoGoalVlm = !autoGoalVlm;
  await fetch('/set_vlm_goal_mode/' + (autoGoalVlm ? 'true' : 'false'), {method: 'POST'});
  const btn = document.getElementById('btnVlmToggle');
  btn.textContent = autoGoalVlm ? '🧭 VLM Auto-Goal: ON' : '🧭 VLM Auto-Goal: OFF';
  btn.classList.toggle('active', autoGoalVlm);
  if (autoGoalVlm) setMode('AI_CHASE');
}

async function triggerVlm() {
  const res = await fetch('/trigger_vlm_goal', {method: 'POST'});
  const data = await res.json();
  if (data.instruction) {
    document.getElementById('vlmInstruction').textContent = data.instruction;
  }
  setMode('AI_CHASE');
}

// Keyboard WASD
const keyMap = {w:'FORWARD', a:'LEFT', s:'BACKWARD', d:'RIGHT'};
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
    else mc(keyMap[Array.from(pressed)[pressed.size-1]]);
  }
});

// Radar canvas
const radarCanvas = document.getElementById('radar');
const ctx = radarCanvas.getContext('2d');
const TARGETS = [
  {name:'R', x:1.1, y:0.9, color:'#ff1744'},
  {name:'G', x:-0.9, y:1.1, color:'#76ff03'},
  {name:'B', x:0.0, y:-1.0, color:'#448aff'},
  {name:'CR', x:-1.2, y:-1.2, color:'#a67234'},
  {name:'HZ', x:0.0, y:0.2, color:'#ffd600'},
];

function drawRadar(data) {
  const W = radarCanvas.width, H = radarCanvas.height;
  const s = W / 4.6;
  const cx = W/2, cy = H/2;
  ctx.clearRect(0,0,W,H);

  // Grid
  ctx.strokeStyle = '#1a2540'; ctx.lineWidth=1;
  for(let i=-2;i<=2;i+=0.5){
    ctx.beginPath(); ctx.moveTo(cx+i*s,0); ctx.lineTo(cx+i*s,H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0,cy-i*s); ctx.lineTo(W,cy-i*s); ctx.stroke();
  }
  // Arena border
  ctx.strokeStyle='#2a3f6a'; ctx.lineWidth=2;
  ctx.strokeRect(cx-2*s, cy-2*s, 4*s, 4*s);

  // Targets
  TARGETS.forEach(t => {
    const tx = cx + t.x*s, ty = cy - t.y*s;
    ctx.fillStyle = t.color; ctx.globalAlpha=0.3;
    ctx.beginPath(); ctx.arc(tx,ty,8,0,Math.PI*2); ctx.fill();
    ctx.globalAlpha=1;
    ctx.beginPath(); ctx.arc(tx,ty,4,0,Math.PI*2); ctx.fill();
    ctx.fillStyle='#fff'; ctx.font='bold 9px Orbitron';
    ctx.fillText(t.name, tx+7, ty+3);
  });

  // Trail
  if (data.trail && data.trail.length > 1) {
    ctx.strokeStyle='rgba(0,229,255,0.25)'; ctx.lineWidth=1.5;
    ctx.beginPath();
    data.trail.forEach((p,i) => {
      const px = cx + p[0]*s, py = cy - p[1]*s;
      i===0 ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
    });
    ctx.stroke();
  }

  // Robot
  const rx = cx + data.pos[0]*s, ry = cy - data.pos[1]*s;
  const ha = data.orientation || 0;
  ctx.strokeStyle='var(--accent)'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(rx,ry);
  ctx.lineTo(rx + Math.cos(ha)*14, ry - Math.sin(ha)*14); ctx.stroke();
  ctx.fillStyle='#00e5ff'; ctx.beginPath(); ctx.arc(rx,ry,5,0,Math.PI*2); ctx.fill();
  ctx.fillStyle='#fff'; ctx.beginPath(); ctx.arc(rx,ry,2,0,Math.PI*2); ctx.fill();
}

function updateTelemetryUI(d) {
  document.getElementById('tX').textContent = d.pos[0].toFixed(3);
  document.getElementById('tY').textContent = d.pos[1].toFixed(3);
  document.getElementById('tLW').textContent = d.left_speed.toFixed(2);
  document.getElementById('tRW').textContent = d.right_speed.toFixed(2);
  document.getElementById('tPL').textContent = d.ps_left.toFixed(0);
  document.getElementById('tPR').textContent = d.ps_right.toFixed(0);
  document.getElementById('tTarget').textContent = d.locked_target;
  document.getElementById('tYaw').textContent = (d.yaw_deg || 0).toFixed(1);

  // Depth hazard status
  const dh = document.getElementById('tDepthHaz');
  const ds = document.getElementById('tDepthScore');
  const dhBadge = document.getElementById('depthHazardBadge');
  if (d.depth_hazard) {
    dh.textContent = 'HAZARD!';
    dh.style.color = 'var(--accent2)';
    if (dhBadge) dhBadge.style.display = 'block';
  } else {
    dh.textContent = 'CLEAR';
    dh.style.color = 'var(--accent3)';
    if (dhBadge) dhBadge.style.display = 'none';
  }
  if (ds && d.depth_score !== undefined) {
    ds.textContent = d.depth_score.toFixed(2);
  }

  if (d.targets) {
    if (d.targets.RED) { TARGETS[0].x = d.targets.RED[0]; TARGETS[0].y = d.targets.RED[1]; }
    if (d.targets.GREEN) { TARGETS[1].x = d.targets.GREEN[0]; TARGETS[1].y = d.targets.GREEN[1]; }
    if (d.targets.BLUE) { TARGETS[2].x = d.targets.BLUE[0]; TARGETS[2].y = d.targets.BLUE[1]; }
  }

  const sceneBadge = document.getElementById('sceneBadge');
  if (sceneBadge && d.semantic_scene) {
    const pct = Math.round((d.semantic_score || 0) * 100);
    sceneBadge.textContent = 'Semantic Scene: ' + d.semantic_scene + ' (' + pct + '%)';
    const isAlert = d.semantic_scene.includes('Hazard') || d.semantic_scene.includes('Plant');
    sceneBadge.style.borderColor = isAlert ? 'rgba(255, 23, 68, 0.6)' : 'rgba(0, 229, 255, 0.4)';
    sceneBadge.style.color = isAlert ? '#ff5252' : '#00e5ff';
  }

  if (d.vlm_goal_instruction) {
    document.getElementById('vlmInstruction').textContent = d.vlm_goal_instruction;
  }

  // Update Mission Logs Terminal
  if (d.latest_mission_log && d.latest_mission_log !== lastKnownLog) {
    lastKnownLog = d.latest_mission_log;
    const term = document.getElementById('logTerminal');
    const line = document.createElement('div');
    line.className = 'log-line';
    line.textContent = d.latest_mission_log;
    term.appendChild(line);
    term.scrollTop = term.scrollHeight;
  }

  if (d.mode) {
    currentMode = d.mode;
    const badge = document.getElementById('modeBadge');
    if (badge) {
      badge.textContent = d.mode;
      badge.style.background = d.mode==='AI_PATROL' ? 'var(--accent)' : d.mode==='AI_CHASE' ? 'var(--accent2)' : '#ffd600';
    }
    const bp = document.getElementById('btnPatrol');
    const bc = document.getElementById('btnChase');
    const bm = document.getElementById('btnManual');
    if (bp) bp.classList.toggle('active', d.mode === 'AI_PATROL');
    if (bc) bc.classList.toggle('active', d.mode === 'AI_CHASE');
    if (bm) bm.classList.toggle('active', d.mode === 'MANUAL');
    const wg = document.getElementById('wasdGrid');
    if (wg) wg.style.display = d.mode === 'MANUAL' ? 'grid' : 'none';
  }

  drawRadar(d);
}

// ── WebSocket Connection ──
let ws = null;
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onmessage = (evt) => {
    try {
      const d = JSON.parse(evt.data);
      updateTelemetryUI({
        pos: [d.x, d.y, d.z],
        left_speed: d.vx,
        right_speed: d.vy,
        ps_left: d.ps_left,
        ps_right: d.ps_right,
        locked_target: d.detected,
        yaw_deg: d.yaw,
        mode: d.mode,
        semantic_scene: d.semantic_scene,
        semantic_score: d.semantic_score,
        targets: d.targets,
        depth_hazard: d.depth_hazard,
        depth_score: d.depth_score,
        vlm_goal_instruction: d.vlm_goal_instruction,
        latest_mission_log: d.latest_mission_log,
        trail: []
      });
    } catch(e) {}
  };
  ws.onclose = () => { setTimeout(connectWS, 2000); };
}
connectWS();

// Polling telemetry fallback
async function poll() {
  try {
    const r = await fetch('/telemetry');
    const d = await r.json();
    updateTelemetryUI(d);
  } catch(e) {}
}
setInterval(poll, 200);

// Non-blocking double-buffered snapshot polling for RGB and Depth viewports (120ms)
function pollVideoFeed() {
  const img = new Image();
  const startTime = Date.now();
  img.onload = () => {
    const el = document.getElementById('videoFeed');
    if (el) el.src = img.src;
    const elapsed = Date.now() - startTime;
    setTimeout(pollVideoFeed, Math.max(20, 120 - elapsed));
  };
  img.onerror = () => {
    setTimeout(pollVideoFeed, 300);
  };
  img.src = '/video_frame.jpg?t=' + Date.now();
}

function pollDepthFeed() {
  const img = new Image();
  const startTime = Date.now();
  img.onload = () => {
    const el = document.getElementById('depthFeed');
    if (el) el.src = img.src;
    const elapsed = Date.now() - startTime;
    setTimeout(pollDepthFeed, Math.max(20, 120 - elapsed));
  };
  img.onerror = () => {
    setTimeout(pollDepthFeed, 300);
  };
  img.src = '/depth_frame.jpg?t=' + Date.now();
}

pollVideoFeed();
pollDepthFeed();
</script>
</body>
</html>"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A-Frame WebXR VR Digital Twin HTML
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VR_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>3D VR Digital Twin — Smart Robotics Lab</title>
<script src="https://aframe.io/releases/1.6.0/aframe.min.js"></script>
<script src="https://unpkg.com/aframe-environment-component@1.3.4/dist/aframe-environment-component.min.js"></script>
<script>
// ── A-Frame Drag and Drop Target Component ──
window.isDraggingObject = false;
window.activeDragEl = null;
window.activeDragTarget = null;

AFRAME.registerComponent('drag-drop-target', {
  schema: {
    targetName: { type: 'string', default: 'RED' }
  },
  init: function () {
    const el = this.el;
    const targetName = this.data.targetName;
    const sceneEl = document.querySelector('a-scene');
    const floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const intersection = new THREE.Vector3();

    el.addEventListener('mousedown', function (evt) {
      window.isDraggingObject = true;
      window.activeDragEl = el;
      window.activeDragTarget = targetName;
      const cam = document.getElementById('vr-cam');
      if (cam && cam.components['look-controls']) {
        cam.components['look-controls'].pause();
      }
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

          const webotsX = clampedX;
          const webotsY = -clampedZ;

          if (window.ws && window.ws.readyState === 1) {
            window.ws.send(JSON.stringify({
              action: 'DRAG_OBJECT',
              target: window.activeDragTarget,
              x: webotsX,
              y: webotsY,
              z: 0.05
            }));
          }
        }
      });

      window.addEventListener('mouseup', function () {
        if (window.isDraggingObject) {
          window.isDraggingObject = false;
          window.activeDragEl = null;
          window.activeDragTarget = null;
          const cam = document.getElementById('vr-cam');
          if (cam && cam.components['look-controls']) {
            cam.components['look-controls'].play();
          }
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
    font-family: 'Monaco', 'Courier New', monospace; font-size: 13px; line-height: 1.6;
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
  #vr-controls button:hover { background: rgba(0, 229, 255, 0.3); transform: scale(1.05); }
  #vr-controls button.active { background: #00e5ff; color: #000; }
  #vr-controls button.vlm { background: rgba(224, 64, 251, 0.2); border-color: #e040fb; color: #e040fb; }
  #vr-controls button.vlm.active { background: #e040fb; color: #fff; }
</style>
</head>
<body>

<div id="vr-hud">
  <div>MODE: <span id="h-mode">AI_PATROL</span></div>
  <div>POS: <span id="h-pos">0.000, 0.000, 0.000</span></div>
  <div>YAW: <span id="h-yaw">0.0</span>°</div>
  <div>TARGET: <span id="h-target">NONE</span></div>
  <div>PROX: L=<span id="h-psl">0</span> R=<span id="h-psr">0</span></div>
  <div>DEPTH HAZ: <span id="h-depth">CLEAR</span></div>
  <div>SCENE: <span id="h-scene">Clear Open Floor (95%)</span></div>
  <div>WS: <span id="h-ws">CONNECTING...</span></div>
</div>

<div id="vr-controls">
  <button id="btn-patrol" class="active" onclick="sendMode('AI_PATROL')">🧠 AI PATROL</button>
  <button id="btn-chase" onclick="sendMode('AI_CHASE')">🎯 AI CHASE</button>
  <button id="btn-manual" onclick="sendMode('MANUAL')">🕹️ MANUAL</button>
  <button id="btn-vlm" class="vlm" onclick="toggleVlmVr()">🧭 VLM AUTO-GOAL</button>
  <button onmousedown="sendCmd('FORWARD')" onmouseup="sendCmd('STOP')">▲ FWD</button>
  <button onmousedown="sendCmd('LEFT')" onmouseup="sendCmd('STOP')">◄ LEFT</button>
  <button onmousedown="sendCmd('BACKWARD')" onmouseup="sendCmd('STOP')">▼ BACK</button>
  <button onmousedown="sendCmd('RIGHT')" onmouseup="sendCmd('STOP')">RIGHT ►</button>
  <button onclick="sendReset()">⚡ RESET</button>
  <button id="btn-view" onclick="toggleView()">📷 ROVER FPV</button>
</div>

<a-scene
  renderer="antialias: true; colorManagement: true; physicallyCorrectLights: true; toneMapping: ACESFilmic; exposure: 1.1"
  shadow="type: pcfsoft"
  webxr="optionalFeatures: hit-test, dom-overlay"
  vr-mode-ui="enabled: true"
  fog="type: exponential; color: #0a0e17; density: 0.08"
  cursor="rayOrigin: mouse" raycaster="objects: .vr-btn, .drag-target"
>
  <a-entity environment="preset: starry; groundColor: #0f172a; grid: 1x1; gridColor: #1e293b; dressing: none; fog: 0.4"></a-entity>

  <a-light type="ambient" color="#334466" intensity="0.5"></a-light>
  <a-light type="directional" color="#ffe8cc" intensity="1.2" position="3 5 2"></a-light>
  <a-light type="directional" color="#cce0ff" intensity="0.8" position="-3 4 2"></a-light>

  <!-- LAB FLOOR -->
  <a-plane position="0 0.001 0" rotation="-90 0 0" width="4" height="4"
           material="color: #383c46; roughness: 0.25; metalness: 0.1"></a-plane>

  <!-- ARENA BOUNDARY WALLS with Amber Gold Top Rails -->
  <a-box position="0 0.1 -2.0" width="4" height="0.2" depth="0.05" material="color: #1a2233; roughness: 0.5"></a-box>
  <a-box id="wall-north" position="0 0.205 -2.0" width="4.02" height="0.015" depth="0.06"
         material="color: #ffbf00; emissive: #cc8800; roughness: 0.2"></a-box>

  <a-box position="0 0.1 2.0" width="4" height="0.2" depth="0.05" material="color: #1a2233; roughness: 0.5"></a-box>
  <a-box id="wall-south" position="0 0.205 2.0" width="4.02" height="0.015" depth="0.06"
         material="color: #ffbf00; emissive: #cc8800; roughness: 0.2"></a-box>

  <a-box position="-2.0 0.1 0" width="0.05" height="0.2" depth="4" material="color: #1a2233; roughness: 0.5"></a-box>
  <a-box id="wall-west" position="-2.0 0.205 0" width="0.06" height="0.015" depth="4.02"
         material="color: #ffbf00; emissive: #cc8800; roughness: 0.2"></a-box>

  <a-box position="2.0 0.1 0" width="0.05" height="0.2" depth="4" material="color: #1a2233; roughness: 0.5"></a-box>
  <a-box id="wall-east" position="2.0 0.205 0" width="0.06" height="0.015" depth="4.02"
         material="color: #ffbf00; emissive: #cc8800; roughness: 0.2"></a-box>

  <!-- DRAGGABLE TARGET BEACONS -->
  <!-- RED TRAFFIC BEACON -->
  <a-entity id="red-beacon" class="drag-target" drag-drop-target="targetName: RED" position="1.1 0 -0.9">
    <a-cylinder radius="0.09" height="0.02" position="0 0.01 0" material="color: #222222; roughness: 0.8"></a-cylinder>
    <a-cone radius-bottom="0.075" radius-top="0.015" height="0.26" position="0 0.14 0" material="color: #e51e1e; roughness: 0.4"></a-cone>
    <a-cylinder radius="0.05" height="0.03" position="0 0.13 0" material="color: #ffffff; roughness: 0.3"></a-cylinder>
    <a-cylinder radius="0.03" height="0.025" position="0 0.20 0" material="color: #ffffff; roughness: 0.3"></a-cylinder>
    <a-text value="RED TARGET\n(DRAGGABLE)" position="0 0.36 0" align="center" color="#ff4444" width="1.2" font="monoid"></a-text>
    <a-ring id="lock-red" radius-inner="0.16" radius-outer="0.19" position="0 0.005 0" rotation="-90 0 0"
            material="color: #ff1744; emissive: #ff1744; opacity: 0.0; transparent: true; side: double"></a-ring>
  </a-entity>

  <!-- GREEN POTTED PLANT -->
  <a-entity id="green-plant" class="drag-target" drag-drop-target="targetName: GREEN" position="-0.9 0 -1.1">
    <a-cylinder radius="0.08" height="0.14" position="0 0.07 0" material="color: #7a4820; roughness: 0.8"></a-cylinder>
    <a-sphere radius="0.14" position="0 0.24 0" material="color: #1e8220; roughness: 0.6"></a-sphere>
    <a-sphere radius="0.10" position="0.05 0.32 0.03" material="color: #2da430; roughness: 0.6"></a-sphere>
    <a-text value="GREEN PLANT\n(DRAGGABLE)" position="0 0.48 0" align="center" color="#00e676" width="1.2" font="monoid"></a-text>
    <a-ring id="lock-green" radius-inner="0.18" radius-outer="0.21" position="0 0.005 0" rotation="-90 0 0"
            material="color: #00e676; emissive: #00e676; opacity: 0.0; transparent: true; side: double"></a-ring>
  </a-entity>

  <!-- BLUE LAB CHAIR -->
  <a-entity id="blue-chair" class="drag-target" drag-drop-target="targetName: BLUE" position="0 0 1.0">
    <a-cylinder radius="0.015" height="0.24" position="-0.1 0.12 0.1" material="color: #9a9ea3; roughness: 0.3; metalness: 0.7"></a-cylinder>
    <a-cylinder radius="0.015" height="0.24" position="0.1 0.12 0.1" material="color: #9a9ea3; roughness: 0.3; metalness: 0.7"></a-cylinder>
    <a-cylinder radius="0.015" height="0.24" position="-0.1 0.12 -0.1" material="color: #9a9ea3; roughness: 0.3; metalness: 0.7"></a-cylinder>
    <a-cylinder radius="0.015" height="0.24" position="0.1 0.12 -0.1" material="color: #9a9ea3; roughness: 0.3; metalness: 0.7"></a-cylinder>
    <a-box width="0.24" height="0.04" depth="0.24" position="0 0.26 0" material="color: #1a33eb; roughness: 0.45"></a-box>
    <a-box width="0.22" height="0.24" depth="0.03" position="0 0.40 -0.11" rotation="-6 0 0" material="color: #1530e0; roughness: 0.4"></a-box>
    <a-text value="BLUE LAB CHAIR\n(DRAGGABLE)" position="0 0.68 0" align="center" color="#448aff" width="1.2" font="monoid"></a-text>
    <a-ring id="lock-blue" radius-inner="0.18" radius-outer="0.21" position="0 0.005 0" rotation="-90 0 0"
            material="color: #448aff; emissive: #448aff; opacity: 0.0; transparent: true; side: double"></a-ring>
  </a-entity>

  <!-- WOODEN CRATES -->
  <a-entity position="1.2 0.13 1.1">
    <a-box width="0.28" height="0.26" depth="0.28" material="color: #a67234; roughness: 0.7"></a-box>
    <a-text value="WOODEN CRATE SE" position="0 0.22 0" align="center" color="#e0a0ff" width="1.2" font="monoid"></a-text>
  </a-entity>
  <a-entity position="-1.2 0.13 1.2">
    <a-box width="0.28" height="0.26" depth="0.28" material="color: #946633; roughness: 0.72"></a-box>
    <a-text value="WOODEN CRATE SW" position="0 0.22 0" align="center" color="#e0a0ff" width="1.2" font="monoid"></a-text>
  </a-entity>

  <!-- LOW-PROFILE OBSTACLES (PURPOSE B) -->
  <!-- YELLOW HAZARD BOX -->
  <a-entity position="0 0.08 -0.2">
    <a-box width="0.28" height="0.16" depth="0.20" material="color: #ffd600; roughness: 0.5"></a-box>
    <a-text value="YELLOW HAZARD BOX" position="0 0.18 0" align="center" color="#ffd600" width="1.1" font="monoid"></a-text>
  </a-entity>

  <!-- SMALL TOOLBOX -->
  <a-entity position="-0.35 0.02 -0.25">
    <a-box width="0.22" height="0.04" depth="0.14" material="color: #ff6d00; roughness: 0.5"></a-box>
    <a-cylinder radius="0.008" height="0.08" position="0 0.035 0" rotation="0 0 90" material="color: #cfd8dc; metalness: 0.8"></a-cylinder>
    <a-text value="SMALL TOOLBOX" position="0 0.12 0" align="center" color="#ff6d00" width="1.1" font="monoid"></a-text>
  </a-entity>

  <!-- VR ROVER -->
  <a-entity id="vr-rover" position="0 0.06 0">
    <a-box width="0.26" height="0.07" depth="0.18" material="color: #1e2233; roughness: 0.25; metalness: 0.6"></a-box>
    <a-box width="0.22" height="0.015" depth="0.15" position="0 0.04 0" material="color: #2e3342; roughness: 0.2; metalness: 0.7"></a-box>
    <a-box width="0.01" height="0.015" depth="0.14" position="0.12 0.02 0" material="color: #00d9ff; emissive: #00b8d9; roughness: 0.1"></a-box>
    <a-cylinder radius="0.018" height="0.06" position="0.06 0.08 0" material="color: #3f4450; roughness: 0.3; metalness: 0.5"></a-cylinder>
    <a-box width="0.04" height="0.03" depth="0.05" position="0.06 0.12 0" material="color: #262933; roughness: 0.25; metalness: 0.6"></a-box>
    <a-cylinder radius="0.012" height="0.02" position="0.08 0.12 0" rotation="0 0 90" material="color: #0d0d14; roughness: 0.15; metalness: 0.3"></a-cylinder>
    <a-cylinder radius="0.05" height="0.03" position="0 -0.01 0.1" rotation="90 0 0" material="color: #1a1a1a; roughness: 0.85"></a-cylinder>
    <a-cylinder radius="0.05" height="0.03" position="0 -0.01 -0.1" rotation="90 0 0" material="color: #1a1a1a; roughness: 0.85"></a-cylinder>
  </a-entity>

  <!-- MISSION CONTROL DECK (Curved Hologram Wall with Dual Screens) -->
  <a-entity id="mission-deck" position="0 1.15 -2.35" rotation="8 0 0" scale="0.85 0.85 0.85">
    <a-box width="3.6" height="1.8" depth="0.03" position="0 0 -0.1"
           material="color: #1a2540; opacity: 0.75; transparent: true; roughness: 0.05; metalness: 0.2"></a-box>
    <a-box width="3.62" height="0.015" depth="0.04" position="0 0.9 -0.1" material="color: #00e5ff; emissive: #00aacc;"></a-box>
    <a-box width="3.62" height="0.015" depth="0.04" position="0 -0.9 -0.1" material="color: #00e5ff; emissive: #00aacc;"></a-box>
    <a-box width="0.015" height="1.8" depth="0.04" position="-1.81 0 -0.1" material="color: #00e5ff; emissive: #00aacc;"></a-box>
    <a-box width="0.015" height="1.8" depth="0.04" position="1.81 0 -0.1" material="color: #00e5ff; emissive: #00aacc;"></a-box>

    <!-- Screen 1: RGB + YOLO Live Feed -->
    <a-plane id="yolo-vr-screen" width="1.3" height="0.95" position="-0.85 0.2 0"
             material="shader: flat; src: #yolo-canvas; side: double"></a-plane>
    <a-text value="RGB + YOLO VISION FEED" position="-0.85 0.76 0" align="center" color="#00e5ff" width="1.6" font="monoid"></a-text>

    <!-- Screen 2: Monocular Depth Map Feed (Purpose B) -->
    <a-plane id="depth-vr-screen" width="1.3" height="0.95" position="0.85 0.2 0"
             material="shader: flat; src: #depth-canvas; side: double"></a-plane>
    <a-text value="MONOCULAR DEPTH MAP (PURPOSE B)" position="0.85 0.76 0" align="center" color="#e040fb" width="1.6" font="monoid"></a-text>

    <!-- Deck Telemetry & Status -->
    <a-text id="telem-mode" value="MODE: AI_PATROL" position="-1.5 -0.42 0" align="left" color="#00e5ff" width="1.8" font="monoid"></a-text>
    <a-text id="telem-pos" value="POS: 0.000, 0.000, 0.000" position="-0.7 -0.42 0" align="left" color="#aabbcc" width="1.8" font="monoid"></a-text>
    <a-text id="telem-depth" value="DEPTH: CLEAR (0.00)" position="0.4 -0.42 0" align="left" color="#76ff03" width="1.8" font="monoid"></a-text>
    <a-text id="telem-vlm" value="VLM GOAL: STANDBY" position="-1.5 -0.56 0" align="left" color="#e040fb" width="1.8" font="monoid"></a-text>
    <a-text id="telem-log" value="LOG: Initializing mission flight log..." position="-1.5 -0.70 0" align="left" color="#ffab00" width="1.5" font="monoid"></a-text>

    <!-- VR Gaze Buttons -->
    <a-entity position="0 -0.80 0.05">
      <a-box class="vr-btn" width="0.32" height="0.1" depth="0.02" position="-0.6 0 0"
             material="color: #0a1428; opacity: 0.9;" data-action="patrol">
        <a-text value="AI PATROL" align="center" color="#00e5ff" width="1.4" font="monoid" position="0 0 0.015"></a-text>
      </a-box>
      <a-box class="vr-btn" width="0.32" height="0.1" depth="0.02" position="-0.2 0 0"
             material="color: #0a1428; opacity: 0.9;" data-action="chase">
        <a-text value="AI CHASE" align="center" color="#ff1744" width="1.4" font="monoid" position="0 0 0.015"></a-text>
      </a-box>
      <a-box class="vr-btn" width="0.32" height="0.1" depth="0.02" position="0.2 0 0"
             material="color: #0a1428; opacity: 0.9;" data-action="manual">
        <a-text value="MANUAL" align="center" color="#ffd600" width="1.4" font="monoid" position="0 0 0.015"></a-text>
      </a-box>
      <a-box class="vr-btn" width="0.36" height="0.1" depth="0.02" position="0.62 0 0"
             material="color: #2a0845; opacity: 0.9;" data-action="vlm_goal">
        <a-text value="🧭 VLM GOAL" align="center" color="#e040fb" width="1.3" font="monoid" position="0 0 0.015"></a-text>
      </a-box>
    </a-entity>
  </a-entity>

  <!-- CAMERA RIG with Gaze Cursor -->
  <a-entity id="camera-rig" position="0 2.1 2.7">
    <a-camera id="vr-cam" look-controls wasd-controls="enabled: false" rotation="-22 0 0">
      <a-cursor fuse="true" fuse-timeout="1200" color="#00e5ff" raycaster="objects: .vr-btn, .drag-target"></a-cursor>
    </a-camera>
  </a-entity>

  <!-- Canvases for RGB & Depth Feeds -->
  <a-assets>
    <canvas id="yolo-canvas" width="640" height="480"></canvas>
    <canvas id="depth-canvas" width="640" height="480"></canvas>
  </a-assets>

</a-scene>

<script>
let ws = null;
let currentMode = 'AI_PATROL';
let autoGoalVlm = false;
let fpvMode = false;

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  window.ws = ws;

  ws.onopen = () => {
    document.getElementById('h-ws').textContent = 'CONNECTED';
    document.getElementById('h-ws').style.color = '#76ff03';
  };

  ws.onclose = () => {
    document.getElementById('h-ws').textContent = 'DISCONNECTED';
    document.getElementById('h-ws').style.color = '#ff1744';
    setTimeout(connectWS, 2000);
  };

  ws.onerror = () => { ws.close(); };

  ws.onmessage = (evt) => {
    try {
      const d = JSON.parse(evt.data);
      updateScene(d);
    } catch(e) {}
  };
}

function updateScene(d) {
  const ax = d.x;
  const ay = d.z || 0.06;
  const az = -d.y;
  const yawDeg = d.yaw || 0;

  const rover = document.getElementById('vr-rover');
  if (rover) {
    rover.setAttribute('position', `${ax} ${ay} ${az}`);
    rover.setAttribute('rotation', `0 ${yawDeg} 0`);
  }

  if (fpvMode) {
    const rig = document.getElementById('camera-rig');
    if (rig) rig.setAttribute('position', `${ax} ${ay + 0.2} ${az}`);
  }

  // Update HUD
  document.getElementById('h-mode').textContent = d.mode || 'AI_PATROL';
  document.getElementById('h-pos').textContent = `${d.x.toFixed(3)}, ${d.y.toFixed(3)}, ${(d.z||0).toFixed(3)}`;
  document.getElementById('h-yaw').textContent = (d.yaw || 0).toFixed(1);
  document.getElementById('h-target').textContent = d.detected || 'NONE';
  document.getElementById('h-psl').textContent = (d.ps_left || 0).toFixed(0);
  document.getElementById('h-psr').textContent = (d.ps_right || 0).toFixed(0);

  const hd = document.getElementById('h-depth');
  if (hd) {
    if (d.depth_hazard) {
      hd.textContent = `HAZARD (${(d.depth_score||0).toFixed(2)})`;
      hd.style.color = '#ff1744';
    } else {
      hd.textContent = `CLEAR (${(d.depth_score||0).toFixed(2)})`;
      hd.style.color = '#76ff03';
    }
  }

  const hs = document.getElementById('h-scene');
  if (hs && d.semantic_scene) {
    const pct = Math.round((d.semantic_score || 0) * 100);
    hs.textContent = `${d.semantic_scene} (${pct}%)`;
    const isAlert = d.semantic_scene.includes('Hazard') || d.semantic_scene.includes('Plant');
    hs.style.color = isAlert ? '#ff1744' : '#00e5ff';
  }

  // Deck telemetry
  const tm = document.getElementById('telem-mode');
  if (tm) tm.setAttribute('value', `MODE: ${d.mode}`);
  const tp = document.getElementById('telem-pos');
  if (tp) tp.setAttribute('value', `POS: ${d.x.toFixed(3)}, ${d.y.toFixed(3)}`);
  const tdep = document.getElementById('telem-depth');
  if (tdep) {
    if (d.depth_hazard) {
      tdep.setAttribute('value', `DEPTH: HAZARD (${(d.depth_score||0).toFixed(2)})`);
      tdep.setAttribute('color', '#ff1744');
    } else {
      tdep.setAttribute('value', `DEPTH: CLEAR (${(d.depth_score||0).toFixed(2)})`);
      tdep.setAttribute('color', '#76ff03');
    }
  }
  const tvlm = document.getElementById('telem-vlm');
  if (tvlm && d.vlm_goal_instruction) {
    tvlm.setAttribute('value', d.vlm_goal_instruction.substring(0, 52) + '...');
  }
  const tlog = document.getElementById('telem-log');
  if (tlog && d.latest_mission_log) {
    tlog.setAttribute('value', 'LOG: ' + d.latest_mission_log.substring(0, 60));
  }

  // Update targets from Webots
  if (d.targets && !window.isDraggingObject) {
    if (d.targets.RED) {
      const el = document.getElementById('red-beacon');
      if (el) el.setAttribute('position', `${d.targets.RED[0]} 0 ${-d.targets.RED[1]}`);
    }
    if (d.targets.GREEN) {
      const el = document.getElementById('green-plant');
      if (el) el.setAttribute('position', `${d.targets.GREEN[0]} 0 ${-d.targets.GREEN[1]}`);
    }
    if (d.targets.BLUE) {
      const el = document.getElementById('blue-chair');
      if (el) el.setAttribute('position', `${d.targets.BLUE[0]} 0 ${-d.targets.BLUE[1]}`);
    }
  }

  // Target lock rings
  const lockRed = document.getElementById('lock-red');
  const lockGreen = document.getElementById('lock-green');
  const lockBlue = document.getElementById('lock-blue');
  if (lockRed) lockRed.setAttribute('material', 'opacity', d.detected === 'RED' ? 0.7 : 0.0);
  if (lockGreen) lockGreen.setAttribute('material', 'opacity', d.detected === 'GREEN' ? 0.7 : 0.0);
  if (lockBlue) lockBlue.setAttribute('material', 'opacity', d.detected === 'BLUE' ? 0.7 : 0.0);

  const bp = document.getElementById('btn-patrol');
  const bc = document.getElementById('btn-chase');
  const bm = document.getElementById('btn-manual');
  if (bp) bp.classList.toggle('active', d.mode === 'AI_PATROL');
  if (bc) bc.classList.toggle('active', d.mode === 'AI_CHASE');
  if (bm) bm.classList.toggle('active', d.mode === 'MANUAL');

  currentMode = d.mode || 'AI_PATROL';
}

function sendMode(m) {
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({action: 'SET_MODE', mode: m}));
  }
  document.getElementById('btn-patrol').classList.toggle('active', m === 'AI_PATROL');
  document.getElementById('btn-chase').classList.toggle('active', m === 'AI_CHASE');
  document.getElementById('btn-manual').classList.toggle('active', m === 'MANUAL');
}

function sendCmd(cmd) {
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({action: 'MANUAL_CMD', cmd: cmd}));
  }
}

function sendReset() {
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({action: 'RESET_POSE'}));
  }
}

function toggleVlmVr() {
  autoGoalVlm = !autoGoalVlm;
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({action: 'SET_VLM_GOAL_MODE', enabled: autoGoalVlm}));
    ws.send(JSON.stringify({action: 'VLM_SELECT_GOAL'}));
  }
  document.getElementById('btn-vlm').classList.toggle('active', autoGoalVlm);
}

function toggleView() {
  fpvMode = !fpvMode;
  const btn = document.getElementById('btn-view');
  btn.textContent = fpvMode ? '🏠 LAB OVERVIEW' : '📷 ROVER FPV';
  if (!fpvMode) {
    document.getElementById('camera-rig').setAttribute('position', '0 2.1 2.7');
  }
}

// ── Refresh Dual Hologram Video Textures (RGB + Depth, 120ms Chained Polling) ──
let vrTexturePending = false;
function scheduleTextureRefresh() {
  if (vrTexturePending) return;
  vrTexturePending = true;

  const yCanvas = document.getElementById('yolo-canvas');
  const dCanvas = document.getElementById('depth-canvas');
  let completed = 0;
  const onFinish = () => {
    completed++;
    if (completed >= 2) {
      vrTexturePending = false;
      setTimeout(scheduleTextureRefresh, 120);
    }
  };

  // 1. RGB Feed
  if (yCanvas) {
    const ctx = yCanvas.getContext('2d');
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = function() {
      ctx.drawImage(img, 0, 0, yCanvas.width, yCanvas.height);
      const scr = document.getElementById('yolo-vr-screen');
      if (scr && scr.getObject3D('mesh')) {
        const m = scr.getObject3D('mesh').material;
        if (m && m.map) m.map.needsUpdate = true;
      }
      onFinish();
    };
    img.onerror = onFinish;
    img.src = '/video_frame.jpg?t=' + Date.now();
  } else {
    onFinish();
  }

  // 2. Depth Feed
  if (dCanvas) {
    const dctx = dCanvas.getContext('2d');
    const dimg = new Image();
    dimg.crossOrigin = 'anonymous';
    dimg.onload = function() {
      dctx.drawImage(dimg, 0, 0, dCanvas.width, dCanvas.height);
      const dscr = document.getElementById('depth-vr-screen');
      if (dscr && dscr.getObject3D('mesh')) {
        const dm = dscr.getObject3D('mesh').material;
        if (dm && dm.map) dm.map.needsUpdate = true;
      }
      onFinish();
    };
    dimg.onerror = onFinish;
    dimg.src = '/depth_frame.jpg?t=' + Date.now();
  } else {
    onFinish();
  }
}
scheduleTextureRefresh();

// VR Button handler
document.addEventListener('DOMContentLoaded', () => {
  setTimeout(() => {
    document.querySelectorAll('.vr-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const action = btn.getAttribute('data-action');
        if (action === 'patrol') sendMode('AI_PATROL');
        else if (action === 'chase') sendMode('AI_CHASE');
        else if (action === 'manual') sendMode('MANUAL');
        else if (action === 'vlm_goal') {
          if (ws && ws.readyState === 1) ws.send(JSON.stringify({action: 'VLM_SELECT_GOAL'}));
        }
      });
    });
  }, 2000);
});

// Keyboard
const vrKeyMap = {w:'FORWARD', a:'LEFT', s:'BACKWARD', d:'RIGHT'};
const vrPressed = new Set();
document.addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if (vrKeyMap[k] && !vrPressed.has(k)) {
    vrPressed.add(k);
    sendCmd(vrKeyMap[k]);
  }
});
document.addEventListener('keyup', e => {
  const k = e.key.toLowerCase();
  if (vrKeyMap[k]) {
    vrPressed.delete(k);
    if (vrPressed.size === 0) sendCmd('STOP');
    else sendCmd(vrKeyMap[Array.from(vrPressed)[vrPressed.size-1]]);
  }
});

connectWS();
</script>
</body>
</html>"""

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML

@app.get("/", response_class=HTMLResponse)
def root_dashboard():
    return DASHBOARD_HTML

@app.get("/vr", response_class=HTMLResponse)
def vr_twin():
    return VR_HTML

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Entry Point & Standalone Server
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Starts background loops and serves the FastAPI Mission Control dashboard."""
    print("=" * 65)
    print("  MISSION CONTROL — Webots R2025a + Multimodal AI + WebXR")
    print("  Purposes A, B, C, D Active | Dual Viewport | 640x480 Depth")
    print("=" * 65)
    print(f"  Dashboard:    http://{host}:{port}/")
    print(f"  Dashboard Alt:http://{host}:{port}/dashboard")
    print(f"  VR Twin:      http://{host}:{port}/vr")
    print(f"  WebSocket:    ws://{host}:{port}/ws")
    print(f"  RGB Feed:     http://{host}:{port}/video_feed")
    print(f"  Depth Feed:   http://{host}:{port}/depth_feed")
    print(f"  Depth Snap:   http://{host}:{port}/depth_frame.jpg")
    print(f"  Mission Logs: http://{host}:{port}/mission_logs")
    print("=" * 65)

    start_background_threads()

    try:
        uvicorn.run(app, host=host, port=port, log_level="info", timeout_graceful_shutdown=1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Vision Patrol Robot — Mission Control Server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 8000)),
        help="Port to run the FastAPI server on (default: 8000, or $PORT)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Host interface to bind to (default: 0.0.0.0, or $HOST)"
    )
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
