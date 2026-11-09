#!/usr/bin/env python3
"""
domain_state.py — Domain Models, Arena Geometry, and Rover Telemetry State
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OOP Data Architecture:
  1. ArenaObject: Strongly-typed arena obstacle with spatial & visual properties.
  2. ARENA_OBJECTS: Repositioned arena catalog (center corridor open, trap fixed).
  3. ControlMode: Enum for AI_PATROL, AI_CHASE, and MANUAL modes.
  4. RoverTelemetryState: Encapsulated rover telemetry, vision buffers, and locks.
  5. MissionLogger: Structured, deduplicated event logging to mission_log.md.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import cv2

# Global Display Constants
CAM_W: int = 640
CAM_H: int = 480
ROVER_RADIUS: float = 0.28   # effective collision radius (meters)
ARENA_BOUND: float = 1.85    # perimeter wall boundary (meters)
MAX_SPEED: float = 5.0       # max wheel rotational velocity (rad/s)
TIMESTEP: int = 32           # simulation step in milliseconds


class ControlMode(str, Enum):
    AI_PATROL = "AI_PATROL"
    AI_CHASE = "AI_CHASE"
    MANUAL = "MANUAL"


@dataclass
class ArenaObject:
    """Strongly-typed arena obstacle / target model."""
    key: str
    name: str
    label: str
    x: float
    y: float
    radius: float
    height: float
    color_bgr: Tuple[int, int, int]
    shape: str
    aliases: List[str] = field(default_factory=list)

    @property
    def coords(self) -> Tuple[float, float]:
        return (self.x, self.y)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Repositioned Arena Layout (Center Obstacle Trap Permanently Resolved)
# HAZARD moved from (0.0, 0.25) to (0.85, 0.15) so the central patrol axis is clear!
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARENA_OBJECTS: Dict[str, ArenaObject] = {
    "RED": ArenaObject(
        key="RED",
        name="Red Traffic Cone",
        label="Red Cone",
        x=1.15,
        y=1.00,
        radius=0.18,
        height=0.45,
        color_bgr=(20, 25, 240),
        shape="cone",
        aliases=["red cone", "traffic cone", "red beacon", "beacon", "cone", "red marker", "red", "traffic"],
    ),
    "GREEN": ArenaObject(
        key="GREEN",
        name="Green Potted Plant",
        label="Green Plant",
        x=-1.05,
        y=1.05,
        radius=0.20,
        height=0.50,
        color_bgr=(30, 220, 50),
        shape="plant",
        aliases=["green plant", "potted plant", "flora", "greenery", "plant", "green", "flower", "tree", "pot"],
    ),
    "BLUE": ArenaObject(
        key="BLUE",
        name="Blue Office Chair",
        label="Blue Chair",
        x=0.00,
        y=-1.15,
        radius=0.22,
        height=0.62,
        color_bgr=(245, 130, 20),
        shape="chair",
        aliases=["blue chair", "office chair", "workstation chair", "blue workstation", "blue seat", "chair", "blue", "workstation", "seat", "desk chair"],
    ),
    "CRATE_1": ArenaObject(
        key="CRATE_1",
        name="Wooden Supply Crate",
        label="Wooden Crate NW",
        x=-1.15,
        y=-0.95,
        radius=0.20,
        height=0.38,
        color_bgr=(45, 95, 180),
        shape="crate",
        aliases=["crate nw", "crate sw", "wooden crate nw", "wooden crate sw", "nw crate", "sw crate", "wooden crate", "crate", "box", "wooden box", "cargo box", "supply crate", "wooden container", "wood"],
    ),
    "CRATE_2": ArenaObject(
        key="CRATE_2",
        name="Wooden Cargo Crate",
        label="Wooden Crate SE",
        x=1.15,
        y=-0.95,
        radius=0.20,
        height=0.38,
        color_bgr=(45, 95, 180),
        shape="crate",
        aliases=["crate se", "wooden crate se", "se crate", "southeast crate", "cargo crate se", "supply crate se", "cargo crate", "cargo box se"],
    ),
    "HAZARD": ArenaObject(
        key="HAZARD",
        name="Yellow Hazard Box",
        label="Yellow Hazard Box",
        x=0.85,
        y=0.15,
        radius=0.18,
        height=0.36,
        color_bgr=(0, 215, 255),
        shape="hazard_box",
        aliases=["yellow hazard box", "hazard box", "yellow box", "hazard", "danger box", "toolbox", "yellow barrier", "low obstacle", "yellow", "danger"],
    ),
}

# Legacy tuple list for backward compatibility
ARENA_OBSTACLES: List[Tuple[str, float, float, float, str]] = [
    (obj.key, obj.x, obj.y, obj.radius, obj.label)
    for obj in ARENA_OBJECTS.values()
]


class RoverTelemetryState:
    """Encapsulates all rover state fields, vision buffers, and concurrency locks."""

    def __init__(self):
        self.lock = threading.Lock()
        self.mode: ControlMode = ControlMode.AI_PATROL
        self.manual_cmd: str = "STOP"
        self.manual_vx: float = 0.0
        self.manual_vy: float = 0.0
        self.teleport_flag: bool = False

        # Odometry and Telemetry
        self.pos: List[float] = [0.0, 0.0, 0.06]
        self.orientation: float = 0.0
        self.yaw_deg: float = 0.0
        self.left_speed: float = 0.0
        self.right_speed: float = 0.0
        self.ps_left_val: float = 0.0
        self.ps_right_val: float = 0.0
        self.locked_target: str = "NONE"
        self.steer_error: float = 0.0
        self.trail: List[Tuple[float, float]] = []

        # Target positions dictionary
        self.target_positions: Dict[str, Tuple[float, float]] = {
            k: obj.coords for k, obj in ARENA_OBJECTS.items()
        }
        self.drag_queue: List[Tuple[str, float, float, float]] = []

        # Purpose A: VLM Auto-Goal
        self.auto_goal_vlm: bool = False
        self.vlm_goal_instruction: str = "VLM Planner standby. Waiting for initial scene scan."
        self.preferred_target: str = "NONE"
        self.trigger_vlm_flag: bool = False

        # Purpose B: Monocular Depth & Clearance
        self.depth_obstacle_detected: bool = False
        self.depth_hazard_score: float = 0.0
        self.depth_left_mean: float = 0.0
        self.depth_right_mean: float = 0.0

        # Purpose C: Open-Vocabulary Tracking
        self.openvocab_prompt: str = ""
        self.openvocab_target: Optional[dict] = None

        # Purpose D: Mission Logs
        self.mission_logs: List[str] = []
        self.latest_mission_log: str = "Mission logger standby. 30Hz synthetic simulation active."

        # Vision & Semantic Classification
        self.semantic_scene: str = "Clear Open Floor"
        self.semantic_score: float = 0.95
        self.detections: list = []
        self.frame_raw: Optional[np.ndarray] = None

        # Pre-allocate dark placeholder JPEG frames
        self.frame_jpeg: bytes = self._generate_placeholder("CONNECTING TO 3D ARENA CAMERA...", (255, 229, 0))
        self.depth_jpeg: bytes = self._generate_placeholder("INITIALIZING MAGMA DEPTH...", (0, 200, 255), bg=(35, 10, 45))

    @property
    def rx(self) -> float:
        return self.pos[0]

    @property
    def ry(self) -> float:
        return self.pos[1]

    @property
    def rz(self) -> float:
        return self.pos[2]

    def _generate_placeholder(self, text: str, color: Tuple[int, int, int], bg: Tuple[int, int, int] = (30, 25, 20)) -> bytes:
        img = np.full((CAM_H, CAM_W, 3), bg, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.70, 2)
        tx = max(16, (CAM_W - tw) // 2)
        ty = (CAM_H + th) // 2
        cv2.putText(img, text, (tx, ty), font, 0.70, color, 2, cv2.LINE_AA)
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return enc.tobytes()

    def to_dict(self) -> Dict[str, Any]:
        """Returns clean telemetry dict for JSON endpoints and WebSockets."""
        with self.lock:
            has_active_chase = (self.mode == ControlMode.AI_CHASE) and (
                (self.locked_target not in ["NONE", ""]) or bool(self.openvocab_prompt)
            )
            return {
                "pos": [round(c, 3) for c in self.pos],
                "orientation": round(self.orientation, 3),
                "yaw_deg": round(self.yaw_deg, 1),
                "left_speed": round(self.left_speed, 2),
                "right_speed": round(self.right_speed, 2),
                "ps_left": round(self.ps_left_val, 1),
                "ps_right": round(self.ps_right_val, 1),
                "locked_target": self.locked_target,
                "steer_error": round(self.steer_error, 2),
                "mode": self.mode.value,
                "semantic_scene": self.semantic_scene,
                "semantic_score": round(self.semantic_score, 2),
                "targets": {k: [round(c, 2) for c in v] for k, v in self.target_positions.items()},
                "depth_hazard": self.depth_obstacle_detected and not has_active_chase,
                "depth_score": round(self.depth_hazard_score, 3),
                "depth_left": round(self.depth_left_mean, 3),
                "depth_right": round(self.depth_right_mean, 3),
                "auto_goal_vlm": self.auto_goal_vlm,
                "vlm_goal_instruction": self.vlm_goal_instruction,
                "preferred_target": self.preferred_target,
                "openvocab_prompt": self.openvocab_prompt,
                "openvocab_detected": self.openvocab_target is not None,
                "latest_mission_log": self.latest_mission_log,
                "trail": self.trail[-200:],
            }

    # Dict compatibility helper
    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def get(self, key: str, default: Any = None):
        return getattr(self, key, default)


class MissionLogger:
    """Purpose D: Structured, deduplicated flight logger writing to mission_log.md."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._lock = threading.Lock()
        self._last_logged_action: str = ""
        self._last_logged_time: float = 0.0

    def log_event(self, action: str, category: str = "TACTICAL", state: Optional[RoverTelemetryState] = None):
        """Appends formatted event to logs if not duplicate within 1.0s window."""
        now = time.time()
        with self._lock:
            if action == self._last_logged_action and (now - self._last_logged_time) < 1.0:
                return
            self._last_logged_action = action
            self._last_logged_time = now

        t_str = time.strftime("%H:%M:%S")
        if state is not None:
            with state.lock:
                pos = list(state.pos)
                yaw = state.yaw_deg
                mode = state.mode.value
                locked = state.locked_target
                ov = state.openvocab_prompt
            tgt_info = f"Target: '{ov}'" if ov else f"Target: {locked}"
            log_line = f"[{t_str}] [{category}] POS ({pos[0]:.2f}, {pos[1]:.2f}) | YAW {yaw:.1f}° | MODE: {mode} | {tgt_info} — {action}"
        else:
            log_line = f"[{t_str}] [{category}] — {action}"

        if state is not None:
            with state.lock:
                state.mission_logs.append(log_line)
                if len(state.mission_logs) > 60:
                    state.mission_logs = state.mission_logs[-60:]
                state.latest_mission_log = log_line

        # Persist to markdown
        write_header = not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                if write_header:
                    f.write("# 📋 Autonomous Mission Control — Telemetry & VLM Flight Log\n")
                    f.write("**Platform**: Webots R2025a | **Modes**: AI_PATROL / AI_CHASE / MANUAL\n\n")
                    f.write("| Timestamp | Category | Tactical Event & Perception |\n")
                    f.write("|:---------:|:--------:|:----------------------------|\n")
                f.write(f"| `{t_str}` | **{category}** | {action} |\n")
        except Exception:
            pass
