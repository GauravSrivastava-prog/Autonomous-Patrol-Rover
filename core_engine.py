"""
core_engine.py — High-Performance 3D Vision Patrol & Multimodal Robotics Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Architecture:
  1. True GTA 3rd-Person Chase ("TPP") Camera Rig (1280x720 HD @ 30+ FPS)
     - Positioned 1.4m behind and 0.75m above rover with -20° downward pitch.
     - 1st-Person Bumper ("FPP") toggle mode via 'V' key or HUD button.
  2. 3D Depth-Sorted Perspective Graphics Engine (Painter's Algorithm)
     - High-contrast synthwave perspective floor grid & perimeter amber rails.
     - 3D rover model with glowing cyan headlights, red taillights & rotating wheels.
     - Extruded vertical 3D targets with ground contact shadows & floating HUD tags.
  3. Hardware-Aware Local AI Pipeline (Apple Silicon MPS / NVIDIA CUDA / CPU)
     - Ultralytics YOLOv8 object detector loading local tuned weights if present.
     - OpenAI CLIP open-vocabulary multimodal semantic resolver & cache lookup.
  4. Reactive Unicycle Kinematics & Autonomous Movement Modes
     - AI Patrol: Waypoint cruising with continuous 360° obstacle pushaway.
     - AI Chase: Navigates to CLIP-resolved target & locks into 0.55m standoff orbit.
     - Manual: Zero-latency WASD / tactile D-Pad teleoperation.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import logging
import warnings

# Suppress runtime warnings and noisy library loggers
warnings.filterwarnings("ignore")
try:
    from transformers.utils import logging as tf_logging
    tf_logging.set_verbosity_error()
except Exception:
    pass
for _log in ("transformers", "ultralytics", "torch"):
    logging.getLogger(_log).setLevel(logging.ERROR)

logger = logging.getLogger("patrol_rover.core_engine")
logger.setLevel(logging.WARNING)

# Direct Ultralytics config to /tmp to prevent sandbox permission warnings
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

import time
import math
import json
import pickle
import threading
import collections
import re
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────────
# 1. Hardware Detection & Device Pinning (Apple MPS / CUDA / CPU)
# ──────────────────────────────────────────────────────────────────────────────
def detect_hardware_device() -> Tuple[str, str]:
    """Detects available acceleration hardware and returns (torch_device, display_badge)."""
    if torch.backends.mps.is_available():
        return "mps", "🍎 APPLE MPS ACCELERATED"
    elif torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return "cuda", f"🚀 CUDA: {name}"
    return "cpu", "⚡ MULTI-CORE CPU"

NEURAL_DEVICE, NEURAL_BADGE = detect_hardware_device()

# ──────────────────────────────────────────────────────────────────────────────
# 2. Rendering Geometry & Camera Constants (1280x720 HD Widescreen)
# ──────────────────────────────────────────────────────────────────────────────
CAMERA_W = 1280
CAMERA_H = 720
# 68-degree horizontal FOV focal length calculation
FOCAL_LENGTH = int(CAMERA_W / (2.0 * math.tan(math.radians(34.0))))  # ~948 px

ARENA_BOUND = 2.00        # 4m x 4m glass arena (+/- 2.0m)
ROVER_RADIUS = 0.22       # Collision clearance envelope
MAX_WHEEL_SPEED = 8.00    # Max rad/s for wheel kinematics (DWA open-space cruising)

# ──────────────────────────────────────────────────────────────────────────────
# 3. 3D Arena Taxonomy & Physical Landmark Objects
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ArenaObject:
    name: str
    x: float
    y: float
    radius: float
    height: float
    bgr_color: Tuple[int, int, int]
    aliases: List[str]
    keywords: List[str]

ARENA_OBJECTS: Dict[str, ArenaObject] = {
    "RED": ArenaObject(
        name="Red Traffic Cone",
        x=1.15, y=1.00, radius=0.20, height=0.55,
        bgr_color=(35, 45, 235),  # Crimson Red
        aliases=["red cone", "traffic cone", "red", "cone", "beacon", "pylon"],
        keywords=["red", "traffic", "cone", "beacon", "pylon", "safety", "orange", "hazard"]
    ),
    "GREEN": ArenaObject(
        name="Green Potted Plant",
        x=-1.05, y=1.05, radius=0.22, height=0.62,
        bgr_color=(50, 205, 50),  # Emerald Green
        aliases=["green plant", "potted plant", "plant", "green", "flower", "bush", "tree"],
        keywords=["green", "plant", "potted", "flower", "bush", "tree", "vegetation", "nature"]
    ),
    "BLUE": ArenaObject(
        name="Blue Office Chair",
        x=0.00, y=-1.15, radius=0.24, height=0.68,
        bgr_color=(235, 130, 25),  # Deep Blue
        aliases=["blue chair", "office chair", "chair", "blue", "seat", "desk chair"],
        keywords=["blue", "office", "chair", "seat", "desk", "furniture", "swivel", "armchair"]
    ),
    "CRATE_1": ArenaObject(
        name="Wooden Supply Crate",
        x=-1.15, y=-0.95, radius=0.20, height=0.40,
        bgr_color=(75, 115, 155),  # Warm Timber Ochre
        aliases=["crate", "wooden crate", "box", "wooden", "supply", "cargo"],
        keywords=["wooden", "crate", "box", "supply", "cargo", "wood", "timber", "pallet"]
    ),
    "CRATE_2": ArenaObject(
        name="Cargo Storage Container",
        x=1.15, y=-0.95, radius=0.20, height=0.40,
        bgr_color=(140, 100, 160),  # Industrial Steel Magenta
        aliases=["secondary crate", "cargo box", "storage container", "metal container"],
        keywords=["cargo", "container", "freight", "metal", "storage", "shipping", "trunk"]
    ),
    "HAZARD": ArenaObject(
        name="Yellow Hazard Box",
        x=0.85, y=0.15, radius=0.18, height=0.38,
        bgr_color=(35, 215, 245),  # High-Voltage Yellow
        aliases=["hazard box", "yellow box", "danger", "hazard", "toolbox", "yellow"],
        keywords=["yellow", "hazard", "danger", "box", "warning", "caution", "toolbox"]
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# 3.1 Canonical Semantic Taxonomy for Multi-Token Attribute Weighting
# ──────────────────────────────────────────────────────────────────────────────
OBJECT_SEMANTICS: Dict[str, Dict[str, Any]] = {
    "RED": {
        "class_key": "red_cone",
        "colors": {"red", "crimson", "orange"},
        "nouns": {"cone", "traffic", "beacon", "pylon"},
        "phrases": ["red cone", "traffic cone", "red traffic cone", "beacon", "pylon"],
    },
    "GREEN": {
        "class_key": "green_plant",
        "colors": {"green", "emerald"},
        "nouns": {"plant", "potted", "flower", "bush", "tree", "vegetation"},
        "phrases": ["green plant", "potted plant", "green potted plant", "plant", "potted"],
    },
    "BLUE": {
        "class_key": "blue_chair",
        "colors": {"blue", "cobalt", "cyan"},
        "nouns": {"chair", "office", "seat", "armchair", "desk"},
        "phrases": ["blue chair", "office chair", "blue office chair", "chair"],
    },
    "CRATE_1": {
        "class_key": "wooden_crate",
        "colors": {"wooden", "wood", "timber", "brown", "ochre"},
        "nouns": {"crate", "supply", "pallet"},
        "phrases": ["wooden crate", "wood crate", "supply crate", "crate"],
    },
    "CRATE_2": {
        "class_key": "wooden_crate",
        "colors": {"cargo", "metal", "steel", "magenta", "purple"},
        "nouns": {"container", "storage", "freight", "trunk"},
        "phrases": ["cargo container", "storage container", "cargo box", "metal container"],
    },
    "HAZARD": {
        "class_key": "hazard_box",
        "colors": {"yellow", "amber", "gold"},
        "nouns": {"hazard", "danger", "toolbox", "warning"},
        "phrases": ["hazard box", "yellow hazard box", "yellow box", "hazard"],
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# 4. State Model & Kinematics Specification
# ──────────────────────────────────────────────────────────────────────────────
class ControlMode(str, Enum):
    MANUAL = "MANUAL"
    AI_PATROL = "AI_PATROL"
    AI_CHASE = "AI_CHASE"

@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.35
    left_speed: float = 0.0
    right_speed: float = 0.0
    wheel_angle_l: float = 0.0
    wheel_angle_r: float = 0.0
    mode: ControlMode = ControlMode.AI_PATROL
    target_key: str = "NONE"
    target_name: str = "NONE"
    target_x: float = 0.0
    target_y: float = 0.0
    target_conf: float = 0.0
    status_text: str = "AI Patrol: Waypoint Cruising"
    trail: List[Tuple[float, float]] = field(default_factory=list)
    view_mode: str = "TPP"  # "TPP" (GTA 3rd-Person Chase) or "FPP" (1st-Person Bumper)
    vlm_scores: Dict[str, float] = field(default_factory=lambda: {k: 0.15 for k in ARENA_OBJECTS.keys()})
    yolo_detections: List[Dict[str, Any]] = field(default_factory=list)
    neural_device: str = NEURAL_DEVICE
    neural_badge: str = NEURAL_BADGE
    fps: float = 30.0
    lock: threading.Lock = field(default_factory=threading.Lock)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Multimodal AI Resolver (YOLOv8 + OpenAI CLIP + Pre-Computed Semantic Cache)
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ML_MODELS_DIR = BASE_DIR / "ml_pipeline" / "models"

def _is_valid_file(path: Path, min_bytes: int = 512) -> bool:
    """Verifies file existence, readability, and non-empty status."""
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except Exception:
        return False

class MultimodalPerceptionEngine:
    """
    Hardware-accelerated neural perception loader with Cloud-to-Edge tensor remapping:
    - Deserializes CUDA checkpoints to Apple MPS / CPU using safe map_location.
    - Seamlessly falls back to ONNX (rover_yolo_tuned.onnx) or baseline yolov8n.pt.
    - Handles cross-platform pickle deserialization for clip_semantic_cache.pkl.
    - Runs sub-millisecond CLIP open-vocabulary cosine similarity queries.
    """
    def __init__(self, device: str = NEURAL_DEVICE):
        self.device = device
        self.yolo_model = None
        self.yolo_source = "STANDBY"
        self.clip_model = None
        self.clip_processor = None
        self.semantic_cache = None
        self.cache_source = "STANDBY"
        self.is_loaded = False
        self._init_models()

    def _init_models(self):
        """Asynchronously attempts to load YOLOv8 and CLIP with safe cross-platform remapping."""
        def _loader():
            # ──────────────────────────────────────────────────────────────
            # 1. Load CLIP Semantic Cache (.pkl or .json) with CUDA-safe remap
            # ──────────────────────────────────────────────────────────────
            cache_candidates = [
                MODELS_DIR / "clip_semantic_cache.pkl",
                MODELS_DIR / "clip_semantic_cache.json",
                ML_MODELS_DIR / "clip_semantic_cache.pkl",
                ML_MODELS_DIR / "clip_semantic_cache.json",
            ]

            for cp in cache_candidates:
                if not _is_valid_file(cp, min_bytes=64):
                    continue
                try:
                    raw_cache = None
                    if cp.suffix.lower() == ".pkl":
                        # CRUCIAL: torch.load with map_location='cpu' prevents crash if Colab pickled CUDA tensors!
                        try:
                            raw_cache = torch.load(cp, map_location="cpu", weights_only=False)
                        except Exception:
                            with open(cp, "rb") as f:
                                raw_cache = pickle.load(f)
                    elif cp.suffix.lower() == ".json":
                        with open(cp, "r", encoding="utf-8") as f:
                            raw_cache = json.load(f)

                    if raw_cache:
                        # Extract and normalize embeddings dictionary
                        data_dict = raw_cache.get("embeddings", raw_cache) if isinstance(raw_cache, dict) else raw_cache
                        embeddings = {}
                        if isinstance(data_dict, dict):
                            for k, v in data_dict.items():
                                if isinstance(v, (list, tuple)):
                                    vec = np.array(v, dtype=np.float32)
                                elif isinstance(v, torch.Tensor):
                                    vec = v.detach().cpu().numpy().astype(np.float32).flatten()
                                elif isinstance(v, np.ndarray):
                                    vec = v.astype(np.float32).flatten()
                                else:
                                    continue
                                norm = np.linalg.norm(vec)
                                if norm > 1e-6:
                                    vec = vec / norm
                                embeddings[k.lower()] = vec

                        if embeddings:
                            self.semantic_cache = {"embeddings": embeddings}
                            self.cache_source = f"{cp.name} ({len(embeddings)} classes)"
                            logger.debug(f"Loaded {len(embeddings)} class embeddings from {cp}")
                            break
                except Exception as cache_err:
                    logger.debug(f"Failed to load {cp}: {cache_err}")

            # ──────────────────────────────────────────────────────────────
            # 2. Load YOLOv8 Detector (PyTorch map_location MPS/CPU -> ONNX -> Base)
            # ──────────────────────────────────────────────────────────────
            pt_candidates = [
                MODELS_DIR / "rover_yolo_tuned.pt",
                BASE_DIR / "ml_pipeline" / "runs" / "train" / "rover_yolo_run" / "weights" / "best.pt",
            ]
            onnx_candidates = [
                MODELS_DIR / "rover_yolo_tuned.onnx",
                BASE_DIR / "ml_pipeline" / "runs" / "train" / "rover_yolo_run" / "weights" / "best.onnx",
            ]

            # Priority A: Tuned PyTorch Weights (.pt) with explicit CUDA->MPS/CPU deserialization
            for pt_path in pt_candidates:
                if not _is_valid_file(pt_path, min_bytes=1024 * 100):
                    continue
                try:
                    size_mb = pt_path.stat().st_size / (1024 * 1024)
                    logger.debug(f"Found tuned YOLO checkpoint: {pt_path} ({size_mb:.2f} MB)")
                    from ultralytics import YOLO

                    # Safe pre-check: verify deserialization without CUDA hardware crash
                    try:
                        _ = torch.load(pt_path, map_location="cpu", weights_only=False)
                    except Exception as torch_load_err:
                        logger.debug(f"Pre-flight torch.load warning: {torch_load_err}")

                    yolo = YOLO(str(pt_path))
                    # Map to Apple Silicon Metal (MPS) or CPU
                    model_dev = "cpu"
                    if self.device == "mps" and torch.backends.mps.is_available():
                        try:
                            yolo.to("mps")
                            model_dev = "mps"
                        except Exception as mps_err:
                            logger.debug(f"MPS pinning deferred ({mps_err}); executing on CPU")
                            yolo.to("cpu")
                    elif self.device == "cuda" and torch.cuda.is_available():
                        yolo.to("cuda")
                        model_dev = "cuda"
                    else:
                        yolo.to("cpu")

                    self.yolo_model = yolo
                    self.yolo_source = f"TUNED_PT ({size_mb:.1f}MB, {model_dev.upper()})"
                    logger.debug(f"Loaded YOLOv8 Tuned Model: {pt_path} on {model_dev.upper()}")
                    break
                except Exception as pt_err:
                    logger.debug(f"PyTorch .pt load failed ({pt_path}): {pt_err}. Testing ONNX fallback...")

            # Priority B: ONNX Runtime Fallback if .pt failed or absent
            if self.yolo_model is None:
                for onnx_path in onnx_candidates:
                    if not _is_valid_file(onnx_path, min_bytes=1024 * 100):
                        continue
                    try:
                        size_mb = onnx_path.stat().st_size / (1024 * 1024)
                        logger.debug(f"Engaging ONNX Runtime fallback: {onnx_path} ({size_mb:.2f} MB)")
                        from ultralytics import YOLO
                        yolo_onnx = YOLO(str(onnx_path), task="detect")
                        self.yolo_model = yolo_onnx
                        self.yolo_source = f"TUNED_ONNX ({size_mb:.1f}MB)"
                        logger.debug(f"Loaded YOLOv8 ONNX Runtime Model: {onnx_path}")
                        break
                    except Exception as onnx_err:
                        logger.debug(f"ONNX Runtime load failed ({onnx_path}): {onnx_err}")

            # Priority C: Baseline yolov8n.pt Fallback
            if self.yolo_model is None:
                try:
                    from ultralytics import YOLO
                    yolo_base = YOLO("yolov8n.pt")
                    if self.device == "mps" and torch.backends.mps.is_available():
                        try:
                            yolo_base.to("mps")
                        except Exception:
                            yolo_base.to("cpu")
                    elif self.device == "cuda" and torch.cuda.is_available():
                        yolo_base.to("cuda")
                    else:
                        yolo_base.to("cpu")
                    self.yolo_model = yolo_base
                    self.yolo_source = f"BASE_YOLOV8N ({self.device.upper()})"
                    logger.debug(f"Loaded baseline yolov8n.pt on {self.device.upper()}")
                except Exception as base_err:
                    logger.debug(f"Baseline YOLO deferred: {base_err}")
                    self.yolo_source = "GEOMETRIC_FALLBACK"

            # ──────────────────────────────────────────────────────────────
            # 3. Load OpenAI CLIP Transformer Model (Asynchronous background)
            # ──────────────────────────────────────────────────────────────
            try:
                from transformers import CLIPModel, CLIPProcessor
                try:
                    self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(self.device)
                    self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
                except Exception:
                    self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                    self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model.eval()
                logger.debug(f"Loaded OpenAI CLIP transformer on {self.device.upper()}")
            except Exception as e:
                logger.debug(f"OpenAI CLIP transformer deferred: {e}")

            self.is_loaded = True

        threading.Thread(target=_loader, daemon=True).start()

    def compute_similarity(self, prompt: str) -> Dict[str, float]:
        """
        Computes cosine similarity between target prompt and all arena objects.
        Incorporates explicit multi-token parsing (color attributes + object nouns)
        grounded in OpenAI CLIP zero-shot semantic embeddings.
        """
        if not prompt or not prompt.strip():
            return {k: 0.15 for k in ARENA_OBJECTS.keys()}

        clean_p = prompt.lower().strip()
        tokens = set(re.findall(r"\b\w+\b", clean_p))

        # Identify all colors present in the query
        all_colors = {c: k for k, v in OBJECT_SEMANTICS.items() for c in v["colors"]}
        query_colors = {c: all_colors[c] for c in tokens if c in all_colors}

        prompt_vec = None
        # Path 1: Live CLIP Text Embedding
        if (self.clip_model is not None and self.clip_processor is not None):
            try:
                inputs = self.clip_processor(text=[clean_p], return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    out = self.clip_model.get_text_features(**inputs)
                    feat = getattr(out, "pooler_output", out)
                    if hasattr(feat, "text_embeds") and feat.text_embeds is not None:
                        feat = feat.text_embeds
                    if hasattr(feat, "norm"):
                        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
                    prompt_vec = feat.detach().cpu().numpy().squeeze(0)
            except Exception as clip_err:
                logger.debug(f"Live CLIP inference fallback: {clip_err}")

        cache_embs = self.semantic_cache.get("embeddings", {}) if self.semantic_cache else {}

        scores = {}
        for key, spec in OBJECT_SEMANTICS.items():
            base = 0.15
            cls_key = spec["class_key"]
            if prompt_vec is not None and cls_key in cache_embs:
                dot = float(np.dot(prompt_vec, cache_embs[cls_key]))
                base = max(0.12, (dot + 1.0) / 2.0)
            elif cls_key in cache_embs:
                base = 0.20

            score = base
            # Full exact phrase bonus (highest match)
            if any(p in clean_p for p in spec["phrases"]):
                score += 0.25
            # Noun match bonus (nouns carry primary semantic identity, e.g. "plant" in "plant blue")
            if any(n in tokens for n in spec["nouns"]):
                score += 0.35
            # Color attribute match bonus (e.g. "blue" in "blue chair")
            if any(c in tokens for c in spec["colors"]):
                score += 0.25
            # Cross-color penalty: if query explicitly specifies another color, penalize non-matching colors
            if query_colors and not any(c in tokens for c in spec["colors"]):
                score -= 0.15

            scores[key] = max(0.08, min(0.98, score))

        return self._normalize_scores(scores)

    def _normalize_scores(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        max_s = max(raw_scores.values()) if raw_scores else 1.0
        if max_s > 0.40:
            return {k: round(v, 3) for k, v in raw_scores.items()}
        return {k: round(v * (0.85 / max(1e-4, max_s)), 3) for k, v in raw_scores.items()}

# ──────────────────────────────────────────────────────────────────────────────
# 6. Primary Robotics Controller & 3D Simulation Loop
# ──────────────────────────────────────────────────────────────────────────────
class RobotController:
    """
    Full-featured autonomous rover controller:
    - 30 Hz kinematic integration loop with solid-body obstacle avoidance.
    - 1280x720 HD 3D perspective projection graphics engine.
    - True GTA 3rd-person chase camera rig (-20° pitch, 1.4m behind, 0.75m elevation).
    """
    WAYPOINTS: List[Tuple[float, float]] = [
        (0.00, 1.25),   # North open corridor (between Green and Red)
        (-0.60, 0.10),  # West open corridor (clear of west wall and crates)
        (-0.35, -0.35), # South open corridor (well north of Blue and east of Crate 1)
        (0.15, 0.60),   # East open corridor (between Hazard, Red and center)
    ]

    def __init__(self, init_perception: bool = True):
        self.state = RobotState()
        self.perception = MultimodalPerceptionEngine(NEURAL_DEVICE) if init_perception else None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.perception_thread: Optional[threading.Thread] = None

        self.patrol_idx = 0
        self.standoff_dist = 0.55
        self.manual_cmd = "STOP"
        self.escape_timer = 0.0
        self.escape_cooldown = 0.0
        self.escape_turn_sign = 1.0
        self.pos_history = collections.deque(maxlen=35)
        self.prev_ai_vl = 0.0
        self.prev_ai_vr = 0.0
        self.active_tracks: Dict[str, Dict[str, Any]] = {}
        self.last_cam_params: Optional[Tuple[float, float, float, float, float]] = None

        # Frame buffers
        self.latest_rgb: bytes = b""
        self.latest_depth: bytes = b""
        self.latest_clean_frame: Optional[np.ndarray] = None
        self.frame_counter = 0
        self.last_fps_time = time.time()

        # Pre-render initial frame
        self._render_frames()

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            if self.perception:
                self.perception_thread = threading.Thread(target=self._perception_loop, daemon=True)
                self.perception_thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.perception_thread and self.perception_thread.is_alive():
            self.perception_thread.join(timeout=1.0)

    @staticmethod
    def is_in_frontal_sector(rx: float, ry: float, ryaw: float,
                             obj_x: float, obj_y: float,
                             max_dist: float = 2.50,
                             max_angle_deg: float = 45.0) -> Tuple[bool, float, float]:
        """
        Computes 3D relative bearing and distance from rover forward vector to object.
        Strictly gates detections to frontal sector [-45°, +45°] and distance <= 2.5m.
        """
        dx = obj_x - rx
        dy = obj_y - ry
        dist = math.hypot(dx, dy)
        if dist > max_dist or dist < 1e-4:
            return False, dist, 0.0
        target_yaw = math.atan2(dy, dx)
        bearing = (target_yaw - ryaw + math.pi) % (2.0 * math.pi) - math.pi
        in_sector = abs(bearing) <= math.radians(max_angle_deg)
        return in_sector, dist, math.degrees(bearing)

    @staticmethod
    def compute_iou(box1: List[int], box2: List[int]) -> float:
        """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h
        area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
        union_area = area1 + area2 - inter_area
        return inter_area / float(union_area)

    @classmethod
    def apply_strict_nms(cls, detections: List[Dict[str, Any]], iou_thresh: float = 0.45) -> List[Dict[str, Any]]:
        """
        Enforces strict Non-Maximum Suppression (IoU <= 0.45) and 1-box-per-entity deduplication
        to eliminate colliding, overlapping multi-box predictions.
        """
        if not detections:
            return []
        sorted_dets = sorted(detections, key=lambda d: d.get("conf", 0.0), reverse=True)
        selected = []
        selected_keys = set()

        for det in sorted_dets:
            box = det.get("box", [])
            if len(box) != 4:
                continue
            key = det.get("key", None)
            if key and key in selected_keys:
                continue

            overlap = False
            for s in selected:
                iou = cls.compute_iou(box, s["box"])
                if iou > iou_thresh:
                    overlap = True
                    break
                cx1, cy1 = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
                cx2, cy2 = (s["box"][0] + s["box"][2]) / 2.0, (s["box"][1] + s["box"][3]) / 2.0
                if math.hypot(cx1 - cx2, cy1 - cy2) < 45.0:
                    overlap = True
                    break

            if not overlap:
                selected.append(det)
                if key:
                    selected_keys.add(key)

        return selected

    def _compute_frustum_bboxes(self, cam_x: float, cam_y: float, cam_z: float,
                                cam_yaw: float, cam_pitch: float,
                                rx: float, ry: float, ryaw: float,
                                is_close_range: bool) -> List[Dict[str, Any]]:
        """
        Computes 2D viewport bounding boxes for arena objects strictly inside
        the rover's 3D frontal sector cone [-45°, +45°] and distance <= 2.5m.
        """
        label_map = {
            "RED": "traffic cone",
            "GREEN": "potted plant",
            "BLUE": "chair",
            "CRATE_1": "wooden crate",
            "CRATE_2": "wooden crate",
            "HAZARD": "hazard box",
        }
        boxes = []
        for key, obj in ARENA_OBJECTS.items():
            # PHASE 1: Frontal Sector Gating (strictly [-45°, +45°] and distance <= 2.5m)
            in_sector, dist, bearing_deg = self.is_in_frontal_sector(rx, ry, ryaw, obj.x, obj.y, max_dist=2.50, max_angle_deg=45.0)
            if not in_sector:
                continue

            pts = []
            for i in range(8):
                ang = i * (2.0 * math.pi / 8.0)
                px = obj.x + obj.radius * math.cos(ang)
                py = obj.y + obj.radius * math.sin(ang)
                for pz in (0.015, obj.height):
                    proj = self._project_point(px, py, pz, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
                    if proj:
                        pts.append((proj[0], proj[1]))

            if len(pts) >= 4:
                x1 = min(p[0] for p in pts)
                x2 = max(p[0] for p in pts)
                y1 = min(p[1] for p in pts)
                y2 = max(p[1] for p in pts)

                # Check if box intersects screen viewport
                if x2 > 0 and x1 < CAMERA_W and y2 > 0 and y1 < CAMERA_H:
                    bw, bh = x2 - x1, y2 - y1
                    if bw >= 12 and bh >= 12:
                        # Dynamic Proximity Fallback: Expand ROI padding when close
                        if dist < 0.65 or is_close_range:
                            pad_x = int(bw * 0.08)
                            pad_y = int(bh * 0.08)
                            x1 = max(0, x1 - pad_x)
                            y1 = max(0, y1 - pad_y)
                            x2 = min(CAMERA_W - 1, x2 + pad_x)
                            y2 = min(CAMERA_H - 1, y2 + pad_y)
                        else:
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(CAMERA_W - 1, x2)
                            y2 = min(CAMERA_H - 1, y2)

                        conf = round(min(0.96, max(0.85, 0.98 - 0.05 * dist)), 2)
                        boxes.append({
                            "label": label_map.get(key, "object"),
                            "conf": conf,
                            "box": [x1, y1, x2, y2],
                            "key": key,
                            "obj_x": obj.x,
                            "obj_y": obj.y,
                            "dist": round(dist, 2),
                            "bearing": round(bearing_deg, 1)
                        })
        return boxes

    def _perception_loop(self):
        """
        Asynchronous perception worker: runs YOLOv8 with strict NMS (conf=0.55, iou=0.45),
        Frontal Sector Gating ([-45°, +45°], d <= 2.5m), and 3-Frame Temporal Confidence Hysteresis.
        """
        while self.running:
            time.sleep(0.10)
            with self.state.lock:
                clean_img = self.latest_clean_frame.copy() if self.latest_clean_frame is not None else None
                rx, ry, ryaw = self.state.x, self.state.y, self.state.yaw

            if clean_img is None:
                continue

            try:
                h, w = clean_img.shape[:2]
                small_img = cv2.resize(clean_img, (640, 360))

                # Proximity analysis
                d_to_objs = [math.hypot(rx - obj.x, ry - obj.y) for obj in ARENA_OBJECTS.values()]
                min_obj_dist = min(d_to_objs) if d_to_objs else 1.0
                is_close_range = (min_obj_dist < 0.50)

                # Compute ground-truth frustum boxes for frontal sector
                cam_params = getattr(self, "last_cam_params", None)
                frustum_boxes = []
                if cam_params:
                    cx, cy, cz, cyaw, cpitch = cam_params
                    frustum_boxes = self._compute_frustum_bboxes(cx, cy, cz, cyaw, cpitch, rx, ry, ryaw, is_close_range)

                raw_detections = []
                # Phase 2: YOLOv8 inference with strict thresholds (conf=0.55, iou=0.45)
                if self.perception and self.perception.is_loaded and self.perception.yolo_model is not None:
                    dev = "mps" if self.perception.device == "mps" and torch.backends.mps.is_available() else "cpu"
                    try:
                        results = self.perception.yolo_model.predict(
                            small_img,
                            conf=0.55,
                            iou=0.45,
                            verbose=False,
                            device=dev
                        )
                    except Exception:
                        results = self.perception.yolo_model.predict(
                            small_img,
                            conf=0.55,
                            iou=0.45,
                            verbose=False,
                            device="cpu"
                        )

                    if results and len(results) > 0:
                        boxes = results[0].boxes
                        names = getattr(results[0], "names", {})
                        if boxes is not None and len(boxes) > 0:
                            for b in boxes:
                                cls_id = int(b.cls[0].item())
                                conf = float(b.conf[0].item())
                                if conf < 0.55:
                                    continue
                                label = names.get(cls_id, f"object_{cls_id}")
                                xyxy = b.xyxy[0].cpu().numpy()
                                x1 = int(xyxy[0] * (w / 640.0))
                                y1 = int(xyxy[1] * (h / 360.0))
                                x2 = int(xyxy[2] * (w / 640.0))
                                y2 = int(xyxy[3] * (h / 360.0))
                                det_cx = (x1 + x2) / 2.0
                                det_cy = (y1 + y2) / 2.0

                                # Map detection center to 3D arena coordinates
                                matched_key = None
                                matched_obj = None
                                min_match_dist = 180.0
                                for k, obj in ARENA_OBJECTS.items():
                                    if cam_params:
                                        p_center = self._project_point(obj.x, obj.y, obj.height / 2.0, cx, cy, cz, cyaw, cpitch)
                                        if p_center:
                                            d_screen = math.hypot(det_cx - p_center[0], det_cy - p_center[1])
                                            if d_screen < min_match_dist:
                                                min_match_dist = d_screen
                                                matched_key = k
                                                matched_obj = obj

                                if matched_obj:
                                    obj_x, obj_y = matched_obj.x, matched_obj.y
                                else:
                                    # Fallback estimate from camera forward ray
                                    obj_x = rx + 1.2 * math.cos(ryaw)
                                    obj_y = ry + 1.2 * math.sin(ryaw)

                                # Frontal Sector Gating: [-45°, +45°] and dist <= 2.5m
                                in_sector, dist_r, bearing_deg = self.is_in_frontal_sector(rx, ry, ryaw, obj_x, obj_y, max_dist=2.50, max_angle_deg=45.0)
                                if not in_sector:
                                    continue

                                if is_close_range:
                                    bw, bh = x2 - x1, y2 - y1
                                    pad_x = int(bw * 0.08)
                                    pad_y = int(bh * 0.08)
                                    x1 = max(0, x1 - pad_x)
                                    y1 = max(0, y1 - pad_y)
                                    x2 = min(w - 1, x2 + pad_x)
                                    y2 = min(h - 1, y2 + pad_y)

                                raw_detections.append({
                                    "label": label,
                                    "conf": round(conf, 2),
                                    "box": [x1, y1, x2, y2],
                                    "key": matched_key,
                                    "obj_x": obj_x,
                                    "obj_y": obj_y,
                                    "dist": round(dist_r, 2),
                                    "bearing": round(bearing_deg, 1)
                                })

                # Dynamic Proximity Fallback: Ensure frontal sector visible objects are covered
                if frustum_boxes:
                    for fb in frustum_boxes:
                        covered = False
                        for det in raw_detections:
                            if det.get("key") == fb.get("key") or self.compute_iou(fb["box"], det["box"]) > 0.35:
                                covered = True
                                break
                        if not covered and (is_close_range or len(raw_detections) == 0):
                            raw_detections.append(fb)

                # Apply Strict NMS (iou_thresh=0.45)
                filtered_candidates = self.apply_strict_nms(raw_detections, iou_thresh=0.45)

                # Phase 2: Temporal Confidence Hysteresis Buffer (Confirmation >= 3 consecutive frames)
                updated_tracks = {}
                matched_track_keys = set()

                for det in filtered_candidates:
                    d_box = det["box"]
                    d_label = det["label"]
                    d_conf = det["conf"]
                    d_key = det.get("key")
                    d_cx = (d_box[0] + d_box[2]) / 2.0
                    d_cy = (d_box[1] + d_box[3]) / 2.0

                    best_match_key = None
                    best_match_dist = 140.0
                    for t_key, track in self.active_tracks.items():
                        if t_key in matched_track_keys:
                            continue
                        same_key = (d_key and track.get("key") == d_key)
                        same_label = (track["label"] == d_label)
                        t_cx = (track["box"][0] + track["box"][2]) / 2.0
                        t_cy = (track["box"][1] + track["box"][3]) / 2.0
                        dist_px = math.hypot(d_cx - t_cx, d_cy - t_cy)
                        if same_key or (same_label and dist_px < best_match_dist):
                            best_match_dist = dist_px
                            best_match_key = t_key

                    if best_match_key is not None:
                        matched_track_keys.add(best_match_key)
                        prev = self.active_tracks[best_match_key]
                        hits = prev.get("consecutive_hits", 1) + 1
                        confirmed = (hits >= 3)
                        # Exponential smoothing (EMA: 0.70 / 0.30)
                        old_b = prev["box"]
                        smooth_box = [
                            int(0.70 * d_box[0] + 0.30 * old_b[0]),
                            int(0.70 * d_box[1] + 0.30 * old_b[1]),
                            int(0.70 * d_box[2] + 0.30 * old_b[2]),
                            int(0.70 * d_box[3] + 0.30 * old_b[3]),
                        ]
                        smooth_conf = round(0.70 * d_conf + 0.30 * prev["conf"], 2)
                        updated_tracks[best_match_key] = {
                            "key": d_key or prev.get("key", best_match_key),
                            "label": d_label,
                            "conf": smooth_conf,
                            "box": smooth_box,
                            "consecutive_hits": hits,
                            "confirmed": confirmed,
                            "miss_count": 0,
                            "obj_x": det.get("obj_x", prev.get("obj_x")),
                            "obj_y": det.get("obj_y", prev.get("obj_y")),
                        }
                    else:
                        new_id = d_key or f"{d_label}_{int(d_cx)}_{int(d_cy)}"
                        updated_tracks[new_id] = {
                            "key": d_key or new_id,
                            "label": d_label,
                            "conf": d_conf,
                            "box": d_box,
                            "consecutive_hits": 1,
                            "confirmed": False,  # Probationary: requires >= 3 frames
                            "miss_count": 0,
                            "obj_x": det.get("obj_x"),
                            "obj_y": det.get("obj_y"),
                        }

                # Manage tracks not detected in this frame
                for t_key, track in self.active_tracks.items():
                    if t_key not in updated_tracks and t_key not in matched_track_keys:
                        ox, oy = track.get("obj_x"), track.get("obj_y")
                        in_sec = True
                        if ox is not None and oy is not None:
                            in_sec, _, _ = self.is_in_frontal_sector(rx, ry, ryaw, ox, oy, max_dist=2.50, max_angle_deg=45.0)
                        if not in_sec:
                            continue
                        track["miss_count"] = track.get("miss_count", 0) + 1
                        if track["miss_count"] <= 2 and track.get("confirmed", False):
                            updated_tracks[t_key] = track

                self.active_tracks = updated_tracks

                # Only render confirmed bounding boxes (consecutive_hits >= 3 and conf >= 0.55)
                confirmed_dets = [
                    {
                        "label": t["label"],
                        "conf": t["conf"],
                        "box": t["box"],
                        "key": t.get("key", "")
                    }
                    for t in self.active_tracks.values()
                    if t.get("confirmed", False)
                ]

                # Final NMS deduplication pass
                final_detections = self.apply_strict_nms(confirmed_dets, iou_thresh=0.45)

                with self.state.lock:
                    self.state.yolo_detections = final_detections
            except Exception:
                pass

    # ── Command Handlers ──
    def set_mode(self, mode: ControlMode):
        with self.state.lock:
            self.state.mode = mode
            if mode == ControlMode.AI_PATROL:
                self.state.status_text = "AI Patrol: Waypoint Cruising"
            elif mode == ControlMode.MANUAL:
                self.state.status_text = "Manual Override: Standby"

    def set_view_mode(self, v: str):
        v = v.upper()
        if v in ("TPP", "FPP", "TOP", "TOPDOWN"):
            with self.state.lock:
                self.state.view_mode = v

    def set_manual(self, cmd: str):
        with self.state.lock:
            self.state.mode = ControlMode.MANUAL
            self.manual_cmd = cmd.upper()
            self.state.status_text = f"Manual Teleop: {self.manual_cmd}"

    def set_chase_prompt(self, prompt: str):
        scores = self.perception.compute_similarity(prompt)
        best_key = max(scores.keys(), key=lambda k: scores[k])
        target_obj = ARENA_OBJECTS[best_key]
        best_conf = scores[best_key]

        with self.state.lock:
            self.state.mode = ControlMode.AI_CHASE
            self.state.target_key = best_key
            self.state.target_name = target_obj.name
            self.state.target_x = target_obj.x
            self.state.target_y = target_obj.y
            self.state.target_conf = best_conf
            self.state.vlm_scores = scores
            self.state.status_text = f"AI CHASE: LOCKED {target_obj.name.upper()} ({int(best_conf * 100)}%) AT ({target_obj.x:.2f}, {target_obj.y:.2f})"

    def reset_teleport(self):
        with self.state.lock:
            self.state.x, self.state.y, self.state.yaw = 0.0, 0.0, 0.35
            self.state.left_speed, self.state.right_speed = 0.0, 0.0
            self.state.trail.clear()
            self.state.target_key = "NONE"
            self.state.target_name = "NONE"
            self.state.target_x = 0.0
            self.state.target_y = 0.0
            self.state.target_conf = 0.0
            self.state.status_text = "AI PATROL: WAYPOINT CRUISING" if self.state.mode == ControlMode.AI_PATROL else "Rover reset to origin (0, 0)"
        self.escape_timer = 0.0
        self.escape_cooldown = 0.0
        self.pos_history.clear()
        self.prev_ai_vl = 0.0
        self.prev_ai_vr = 0.0
        self.active_tracks.clear()

    # ── Simulation & Rendering Loop (30+ FPS) ──
    def _loop(self):
        dt = 0.033  # ~30Hz
        while self.running:
            t0 = time.time()
            self._step(dt)
            self._render_frames()

            self.frame_counter += 1
            if time.time() - self.last_fps_time >= 1.0:
                with self.state.lock:
                    self.state.fps = round(self.frame_counter / (time.time() - self.last_fps_time), 1)
                self.frame_counter = 0
                self.last_fps_time = time.time()

            elapsed = time.time() - t0
            time.sleep(max(0.002, dt - elapsed))

    def _step(self, dt: float):
        with self.state.lock:
            mode = self.state.mode
            rx, ry, ryaw = self.state.x, self.state.y, self.state.yaw

        vl, vr = 0.0, 0.0

        # Mode 1: MANUAL TELEOPERATION
        if mode == ControlMode.MANUAL:
            cmd = self.manual_cmd
            if cmd == "FORWARD":    vl, vr = 4.80, 4.80
            elif cmd == "BACKWARD": vl, vr = -3.40, -3.40
            elif cmd == "LEFT":     vl, vr = -2.90, 2.90
            elif cmd == "RIGHT":    vl, vr = 2.90, -2.90
            else:                   vl, vr = 0.00, 0.00

        # Mode 2: AI PATROL (Pure Pursuit Lookahead + Dynamic Window Approach DWA)
        elif mode == ControlMode.AI_PATROL:
            now = time.time()
            self.pos_history.append((now, rx, ry))
            if self.escape_cooldown > 0.0:
                self.escape_cooldown -= dt

            d_wall_x = (ARENA_BOUND - ROVER_RADIUS) - abs(rx)
            d_wall_y = (ARENA_BOUND - ROVER_RADIUS) - abs(ry)
            min_wall_d = min(d_wall_x, d_wall_y)
            d_obs = [
                math.hypot(rx - obj.x, ry - obj.y) - obj.radius - ROVER_RADIUS
                for obj in ARENA_OBJECTS.values()
            ]
            min_obs = min(d_obs)
            min_dist = min(min_wall_d, min_obs)

            # Check displacement over last ~0.8s (24 ticks)
            disp = 1.0
            if len(self.pos_history) >= 24:
                disp = math.hypot(rx - self.pos_history[0][1], ry - self.pos_history[0][2])

            # Emergency escape fallback if physically wedged against obstacle/wall
            if self.escape_timer <= 0.0 and self.escape_cooldown <= 0.0:
                is_stuck = (len(self.pos_history) >= 30 and disp < 0.05 and min_dist < 0.20)
                is_corner_wedged = (min_wall_d < 0.18 and min_obs < 0.18)
                if is_stuck or is_corner_wedged:
                    self.escape_timer = 1.0
                    self.escape_cooldown = 2.5
                    best_wp_idx = min(range(len(self.WAYPOINTS)), key=lambda i: math.hypot(self.WAYPOINTS[i][0] - rx, self.WAYPOINTS[i][1] - ry))
                    self.patrol_idx = best_wp_idx
                    wx, wy = self.WAYPOINTS[self.patrol_idx]
                    escape_target_yaw = math.atan2(wy - ry, wx - rx)
                    diff = (escape_target_yaw - ryaw + math.pi) % (2.0 * math.pi) - math.pi
                    self.escape_turn_sign = 1.0 if diff >= 0 else -1.0

            if self.escape_timer > 0.0:
                self.escape_timer -= dt
                with self.state.lock:
                    self.state.status_text = "AI PATROL: CORNER ESCAPE"

                wx, wy = self.WAYPOINTS[self.patrol_idx]
                escape_target_yaw = math.atan2(wy - ry, wx - rx)
                diff = (escape_target_yaw - ryaw + math.pi) % (2.0 * math.pi) - math.pi

                if abs(diff) > 0.35:
                    vl_raw = 2.00 - 4.50 * self.escape_turn_sign
                    vr_raw = 2.00 + 4.50 * self.escape_turn_sign
                else:
                    vl_raw = 5.20
                    vr_raw = 5.20

                if min_dist > 0.35 and abs(diff) < 0.25:
                    self.escape_timer = 0.0
                    self.pos_history.clear()
            else:
                # ── PHASE 1: Lookahead Waypoint Tracking (L = 0.60m) ──
                curr_wp = self.WAYPOINTS[self.patrol_idx]
                next_wp = self.WAYPOINTS[(self.patrol_idx + 1) % len(self.WAYPOINTS)]
                dist_to_curr = math.hypot(curr_wp[0] - rx, curr_wp[1] - ry)

                # Capture circle: advance waypoint when within 0.35m
                L_LOOKAHEAD = 0.60
                if dist_to_curr < 0.35:
                    self.patrol_idx = (self.patrol_idx + 1) % len(self.WAYPOINTS)
                    curr_wp = self.WAYPOINTS[self.patrol_idx]
                    next_wp = self.WAYPOINTS[(self.patrol_idx + 1) % len(self.WAYPOINTS)]
                    dist_to_curr = math.hypot(curr_wp[0] - rx, curr_wp[1] - ry)

                # Smoothly blend target coordinate toward next waypoint inside lookahead radius
                if dist_to_curr < L_LOOKAHEAD:
                    blend = max(0.0, min(1.0, 1.0 - (dist_to_curr / L_LOOKAHEAD)))
                    tx = (1.0 - blend) * curr_wp[0] + blend * next_wp[0]
                    ty = (1.0 - blend) * curr_wp[1] + blend * next_wp[1]
                else:
                    tx, ty = curr_wp[0], curr_wp[1]

                # ── PHASE 2: Dynamic Window Approach (DWA) Vector Blending ──
                # 1. Attraction Vector toward lookahead target
                dx, dy = tx - rx, ty - ry
                d_tgt = max(1e-4, math.hypot(dx, dy))
                ux_att, uy_att = dx / d_tgt, dy / d_tgt

                # 2. Distance-weighted exponential repulsion from obstacles
                fx_rep, fy_rep = 0.0, 0.0
                fwd_obs_d = float('inf')
                for obj in ARENA_OBJECTS.values():
                    ox, oy = rx - obj.x, ry - obj.y
                    dist_center = math.hypot(ox, oy)
                    clearance = dist_center - obj.radius - ROVER_RADIUS
                    if clearance < 0.90:
                        mag = math.exp(-max(0.0, clearance) / 0.25) * 1.50
                        fx_rep += (ox / max(1e-4, dist_center)) * mag
                        fy_rep += (oy / max(1e-4, dist_center)) * mag

                    # Forward obstacle clearance
                    dx_obj, dy_obj = obj.x - rx, obj.y - ry
                    fwd_proj = dx_obj * math.cos(ryaw) + dy_obj * math.sin(ryaw)
                    lat_proj = abs(-dx_obj * math.sin(ryaw) + dy_obj * math.cos(ryaw))
                    if fwd_proj > 0 and lat_proj < (obj.radius + ROVER_RADIUS + 0.15):
                        fwd_clear = fwd_proj - obj.radius - ROVER_RADIUS
                        if fwd_clear < fwd_obs_d:
                            fwd_obs_d = fwd_clear

                # 3. Distance-weighted exponential repulsion from walls
                if d_wall_x < 0.65:
                    sign = -1.0 if rx > 0 else 1.0
                    fx_rep += sign * math.exp(-max(0.0, d_wall_x) / 0.20) * 1.80
                if d_wall_y < 0.65:
                    sign = -1.0 if ry > 0 else 1.0
                    fy_rep += sign * math.exp(-max(0.0, d_wall_y) / 0.20) * 1.80

                # Forward wall clearance calculation
                fwd_wall_d = float('inf')
                cos_y, sin_y = math.cos(ryaw), math.sin(ryaw)
                if cos_y > 1e-3:    fwd_wall_d = min(fwd_wall_d, (ARENA_BOUND - ROVER_RADIUS - rx) / cos_y)
                elif cos_y < -1e-3:  fwd_wall_d = min(fwd_wall_d, (-ARENA_BOUND + ROVER_RADIUS - rx) / cos_y)
                if sin_y > 1e-3:    fwd_wall_d = min(fwd_wall_d, (ARENA_BOUND - ROVER_RADIUS - ry) / sin_y)
                elif sin_y < -1e-3:  fwd_wall_d = min(fwd_wall_d, (-ARENA_BOUND + ROVER_RADIUS - ry) / sin_y)

                effective_fwd_clear = min(fwd_obs_d, fwd_wall_d, min_wall_d + 0.30)
                clearance_total = min_dist

                # 4. Blended desired heading vector
                v_des_x = ux_att + fx_rep
                v_des_y = uy_att + fy_rep
                desired_yaw = math.atan2(v_des_y, v_des_x)
                yaw_err = (desired_yaw - ryaw + math.pi) % (2.0 * math.pi) - math.pi

                # 5. Open-space acceleration & velocity profiling
                # When no obstacles within 1.2m, ramp linear speed up to MAX_WHEEL_SPEED (8.0 rad/s)
                # Apply smooth braking when entering 0.6m safety bubble
                if effective_fwd_clear >= 1.20 and clearance_total >= 0.50:
                    target_lin = MAX_WHEEL_SPEED
                    status_msg = "AI PATROL: OPEN-SPACE CRUISING (8.0 RAD/S)"
                elif effective_fwd_clear > 0.60:
                    ratio = (effective_fwd_clear - 0.60) / (1.20 - 0.60)
                    target_lin = 5.20 + ratio * (MAX_WHEEL_SPEED - 5.20)
                    status_msg = "AI PATROL: WAYPOINT CRUISING"
                else:
                    # Immediate 0.6m safety bubble: proportional braking
                    target_lin = max(3.00, 5.20 * (max(0.10, effective_fwd_clear) / 0.60))
                    status_msg = "AI PATROL: DWA AVOIDANCE BUBBLE"

                with self.state.lock:
                    self.state.status_text = status_msg

                # 6. Smooth forward drive and curvature steering (never zero out forward speed)
                fwd_drive = target_lin * max(0.45, math.cos(yaw_err))
                steer = max(-4.0, min(4.0, yaw_err * 2.8))

                vl_raw = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, fwd_drive - steer))
                vr_raw = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, fwd_drive + steer))

            # 7. Dynamic Window Approach: acceleration slew-rate limiting (a_max = 8.0 rad/s^2)
            a_max = 8.0 * dt
            vl = self.prev_ai_vl + max(-a_max, min(a_max, vl_raw - self.prev_ai_vl))
            vr = self.prev_ai_vr + max(-a_max, min(a_max, vr_raw - self.prev_ai_vr))
            self.prev_ai_vl, self.prev_ai_vr = vl, vr

        # Mode 3: AI CHASE & INSPECTION ORBIT
        elif mode == ControlMode.AI_CHASE:
            tgt = ARENA_OBJECTS.get(self.state.target_key, ARENA_OBJECTS["BLUE"])
            tgt_conf = getattr(self.state, "target_conf", 0.94)
            dist = math.hypot(tgt.x - rx, tgt.y - ry)
            desired_yaw = math.atan2(tgt.y - ry, tgt.x - rx)
            yaw_err = (desired_yaw - ryaw + math.pi) % (2.0 * math.pi) - math.pi

            if dist <= self.standoff_dist:
                # Lock into orbital inspection around target
                orbit_angle = desired_yaw + (math.pi / 2.0)
                orbit_err = (orbit_angle - ryaw + math.pi) % (2.0 * math.pi) - math.pi
                vl = max(0.9, min(3.10, 2.20 - orbit_err * 1.5))
                vr = max(0.9, min(3.10, 2.20 + orbit_err * 1.5))
                self.state.status_text = f"STANDOFF HELD: {tgt.name.upper()} ({int(tgt_conf * 100)}%) AT ({tgt.x:.2f}, {tgt.y:.2f}) [0.55M ORBIT]"
            else:
                speed = 4.40 if abs(yaw_err) < 0.45 else 2.60
                turn = max(-2.5, min(2.5, yaw_err * 2.8))
                vl = speed - turn
                vr = speed + turn
                self.state.status_text = f"AI CHASE: INTERCEPTING {tgt.name.upper()} ({int(tgt_conf * 100)}%) AT ({tgt.x:.2f}, {tgt.y:.2f})"

        # Unicycle kinematics integration
        wheel_base = 0.28
        wheel_radius = 0.045
        v_linear = (vl + vr) * 0.5 * wheel_radius
        omega = (vr - vl) * wheel_radius / wheel_base

        rx += v_linear * math.cos(ryaw) * dt
        ry += v_linear * math.sin(ryaw) * dt
        ryaw = (ryaw + omega * dt + math.pi) % (2.0 * math.pi) - math.pi

        # Solid-body collision repulsion with objects
        for obj in ARENA_OBJECTS.values():
            dx, dy = rx - obj.x, ry - obj.y
            dist = math.hypot(dx, dy)
            min_dist = obj.radius + ROVER_RADIUS
            if dist < min_dist:
                push = (min_dist - dist) + 0.010
                if dist > 1e-4:
                    rx += (dx / dist) * push
                    ry += (dy / dist) * push
                else:
                    rx += min_dist

        # Wall clamp
        bound = ARENA_BOUND - ROVER_RADIUS
        rx = max(-bound, min(bound, rx))
        ry = max(-bound, min(bound, ry))

        with self.state.lock:
            self.state.x, self.state.y, self.state.yaw = rx, ry, ryaw
            self.state.left_speed, self.state.right_speed = vl, vr
            self.state.wheel_angle_l = (self.state.wheel_angle_l + vl * dt * 4.0) % (2.0 * math.pi)
            self.state.wheel_angle_r = (self.state.wheel_angle_r + vr * dt * 4.0) % (2.0 * math.pi)
            if len(self.state.trail) == 0 or math.hypot(rx - self.state.trail[-1][0], ry - self.state.trail[-1][1]) > 0.06:
                self.state.trail.append((round(rx, 2), round(ry, 2)))
                if len(self.state.trail) > 160:
                    self.state.trail.pop(0)

    # ── 3D Perspective Projection Mathematics & Wall Clipping ──
    def _to_cam(self, px: float, py: float, pz: float,
                cam_x: float, cam_y: float, cam_z: float,
                cam_yaw: float, pitch: float) -> Tuple[float, float, float]:
        dx = px - cam_x
        dy = py - cam_y
        dz = pz - cam_z
        cos_y, sin_y = math.cos(cam_yaw), math.sin(cam_yaw)
        fwd = dx * cos_y + dy * sin_y
        lat = -dx * sin_y + dy * cos_y
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        z_cam = fwd * cos_p + dz * sin_p
        y_cam = -fwd * sin_p + dz * cos_p
        x_cam = lat
        return x_cam, y_cam, z_cam

    def _cam_to_screen(self, x_cam: float, y_cam: float, z_cam: float) -> Tuple[int, int]:
        sx = int(CAMERA_W / 2.0 - (x_cam / max(0.01, z_cam)) * FOCAL_LENGTH)
        sy = int(CAMERA_H / 2.0 - (y_cam / max(0.01, z_cam)) * FOCAL_LENGTH)
        return sx, sy

    def _project_point(self, px: float, py: float, pz: float,
                       cam_x: float, cam_y: float, cam_z: float,
                       cam_yaw: float, pitch: float) -> Optional[Tuple[int, int, float]]:
        x_cam, y_cam, z_cam = self._to_cam(px, py, pz, cam_x, cam_y, cam_z, cam_yaw, pitch)
        if z_cam <= 0.12:
            return None
        sx, sy = self._cam_to_screen(x_cam, y_cam, z_cam)
        return sx, sy, z_cam

    def _clip_polygon_near_plane(self, poly_cam: List[Tuple[float, float, float]],
                                 z_near: float = 0.08) -> Optional[List[Tuple[float, float, float]]]:
        if not poly_cam:
            return None
        out_pts: List[Tuple[float, float, float]] = []
        n = len(poly_cam)
        for i in range(n):
            curr = poly_cam[i]
            prev = poly_cam[(i - 1 + n) % n]
            curr_in = curr[2] >= z_near
            prev_in = prev[2] >= z_near

            if curr_in:
                if not prev_in:
                    dz = curr[2] - prev[2]
                    if abs(dz) > 1e-9:
                        t = (z_near - prev[2]) / dz
                        ix = prev[0] + t * (curr[0] - prev[0])
                        iy = prev[1] + t * (curr[1] - prev[1])
                        out_pts.append((ix, iy, z_near))
                out_pts.append(curr)
            elif prev_in:
                dz = curr[2] - prev[2]
                if abs(dz) > 1e-9:
                    t = (z_near - prev[2]) / dz
                    ix = prev[0] + t * (curr[0] - prev[0])
                    iy = prev[1] + t * (curr[1] - prev[1])
                    out_pts.append((ix, iy, z_near))

        if len(out_pts) < 3:
            return None
        return out_pts

    def _project_polygon(self, world_pts: List[Tuple[float, float, float]],
                         cam_x: float, cam_y: float, cam_z: float,
                         cam_yaw: float, pitch: float,
                         z_near: float = 0.08) -> Optional[np.ndarray]:
        cam_pts = [self._to_cam(px, py, pz, cam_x, cam_y, cam_z, cam_yaw, pitch) for px, py, pz in world_pts]
        clipped = self._clip_polygon_near_plane(cam_pts, z_near=z_near)
        if clipped is None:
            return None
        screen_pts = [self._cam_to_screen(x, y, z) for x, y, z in clipped]
        return np.array(screen_pts, dtype=np.int32)

    def _clip_segment_near_plane(self, p1_cam: Tuple[float, float, float],
                                 p2_cam: Tuple[float, float, float],
                                 z_near: float = 0.08) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        x1, y1, z1 = p1_cam
        x2, y2, z2 = p2_cam
        in1 = z1 >= z_near
        in2 = z2 >= z_near
        if not in1 and not in2:
            return None
        if in1 and in2:
            return self._cam_to_screen(x1, y1, z1), self._cam_to_screen(x2, y2, z2)
        dz = z2 - z1
        if abs(dz) < 1e-9:
            return None
        t = (z_near - z1) / dz
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        if in1 and not in2:
            return self._cam_to_screen(x1, y1, z1), self._cam_to_screen(ix, iy, z_near)
        else:
            return self._cam_to_screen(ix, iy, z_near), self._cam_to_screen(x2, y2, z2)

    # ── 1280x720 HD GTA Chase Graphics Engine (Solid Room & Military Rover) ──
    def _render_frames(self):
        with self.state.lock:
            rx, ry, ryaw = self.state.x, self.state.y, self.state.yaw
            vl, vr = self.state.left_speed, self.state.right_speed
            mode = self.state.mode
            target_key = self.state.target_key
            target_name = self.state.target_name
            view_mode = self.state.view_mode
            wheel_l = self.state.wheel_angle_l
            wheel_r = self.state.wheel_angle_r

        # Canvas setup: Solid indoor laboratory ceiling / upper wall fill (zero black void)
        frame = np.full((CAMERA_H, CAMERA_W, 3), (180, 185, 192), dtype=np.uint8)

        c_bound = 1.85  # Strict interior bounding box (0.15m safety margin inside 2.0m walls)

        # ── Step 1: Camera Rig Tuning & Spring-Arm Containment ──
        if view_mode == "TPP":
            # True GTA 3rd-Person Chase with Ray-AABB Spring-Arm Clamping
            d_behind_max = 1.40
            min_arm = 0.35

            # Direction vector from rover to camera
            vx = -math.cos(ryaw)
            vy = -math.sin(ryaw)

            test_rx = max(-c_bound + 0.01, min(c_bound - 0.01, rx))
            test_ry = max(-c_bound + 0.01, min(c_bound - 0.01, ry))

            t_hit = 1.0
            if vx > 1e-6:
                t_x = (c_bound - test_rx) / (vx * d_behind_max)
                t_hit = min(t_hit, max(0.0, t_x))
            elif vx < -1e-6:
                t_x = (-c_bound - test_rx) / (vx * d_behind_max)
                t_hit = min(t_hit, max(0.0, t_x))

            if vy > 1e-6:
                t_y = (c_bound - test_ry) / (vy * d_behind_max)
                t_hit = min(t_hit, max(0.0, t_y))
            elif vy < -1e-6:
                t_y = (-c_bound - test_ry) / (vy * d_behind_max)
                t_hit = min(t_hit, max(0.0, t_y))

            clamped_dist = max(min_arm, d_behind_max * t_hit)
            raw_cam_x = test_rx + clamped_dist * vx
            raw_cam_y = test_ry + clamped_dist * vy

            pull_fraction = max(0.0, min(1.0, 1.0 - (clamped_dist - min_arm) / max(0.01, (d_behind_max - min_arm))))

            # Smooth lateral shoulder offset along the wall toward the arena center
            u_lat_x = -math.sin(ryaw)
            u_lat_y = math.cos(ryaw)
            dot_center = (-rx) * u_lat_x + (-ry) * u_lat_y
            sign_lat = 1.0 if dot_center >= 0 else -1.0
            lat_mag = 0.36 * pull_fraction

            cam_x = raw_cam_x + sign_lat * lat_mag * u_lat_x
            cam_y = raw_cam_y + sign_lat * lat_mag * u_lat_y
            cam_x = max(-c_bound, min(c_bound, cam_x))
            cam_y = max(-c_bound, min(c_bound, cam_y))

            # Dynamic elevation: subtle raise (0.75m -> 0.79m)
            cam_z = 0.75 + 0.04 * pull_fraction

            # Look-at target on rover chassis
            tgt_x = rx
            tgt_y = ry
            tgt_z = 0.05

            dx = tgt_x - cam_x
            dy = tgt_y - cam_y
            dz = tgt_z - cam_z
            d_xy = math.hypot(dx, dy)
            look_pitch = math.atan2(dz, max(0.15, d_xy))

            # Pitch aims at target + offset so rover sits in lower-middle viewport
            target_pitch = look_pitch + math.radians(6.5)
            target_pitch = max(math.radians(-48.0), min(math.radians(-20.0), target_pitch))

            cam_pitch = (1.0 - pull_fraction) * math.radians(-20.0) + pull_fraction * target_pitch

            # Smooth camera yaw blend towards rover target
            aim_yaw = math.atan2(dy, dx)
            yaw_diff = (aim_yaw - ryaw + math.pi) % (2.0 * math.pi) - math.pi
            cam_yaw = ryaw + pull_fraction * yaw_diff

            horizon_y = int(CAMERA_H / 2.0 + math.tan(cam_pitch) * FOCAL_LENGTH * 0.40)
        elif view_mode in ("TOP", "TOPDOWN"):
            # High-angle tactical overhead surveillance view
            cam_x = max(-c_bound, min(c_bound, rx - 0.40 * math.cos(ryaw)))
            cam_y = max(-c_bound, min(c_bound, ry - 0.40 * math.sin(ryaw)))
            cam_z = 2.40
            cam_yaw = ryaw
            cam_pitch = math.radians(-62.0)
            horizon_y = -40
        else:
            # 1st-Person Bumper View ("FPP")
            cam_x = max(-c_bound, min(c_bound, rx + 0.16 * math.cos(ryaw)))
            cam_y = max(-c_bound, min(c_bound, ry + 0.16 * math.sin(ryaw)))
            cam_z = 0.16
            cam_yaw = ryaw
            cam_pitch = 0.02
            horizon_y = int(CAMERA_H / 2.0)

        # Store synchronized camera rig parameters for perception worker
        self.last_cam_params = (cam_x, cam_y, cam_z, cam_yaw, cam_pitch)

        # ── Pass 1: Ceiling & Reflective Dark Polished Concrete Floor ──
        # Visible Overhead LED Light Troffers on Ceiling
        if horizon_y > 40:
            cv2.rectangle(frame, (int(CAMERA_W * 0.22), 20), (int(CAMERA_W * 0.42), 55), (255, 255, 250), -1)
            cv2.rectangle(frame, (int(CAMERA_W * 0.22), 20), (int(CAMERA_W * 0.42), 55), (140, 145, 150), 2)
            cv2.rectangle(frame, (int(CAMERA_W * 0.58), 20), (int(CAMERA_W * 0.78), 55), (255, 255, 250), -1)
            cv2.rectangle(frame, (int(CAMERA_W * 0.58), 20), (int(CAMERA_W * 0.78), 55), (140, 145, 150), 2)

        # Fill Ground with Solid Reflective Dark Polished Concrete (From Horizon down)
        if 0 <= horizon_y < CAMERA_H:
            cv2.rectangle(frame, (0, horizon_y), (CAMERA_W, CAMERA_H), (45, 45, 48), -1)

        # 3D floor quad clipped against camera near-plane
        floor_world = [
            (-ARENA_BOUND, -ARENA_BOUND, 0.0),
            (ARENA_BOUND, -ARENA_BOUND, 0.0),
            (ARENA_BOUND, ARENA_BOUND, 0.0),
            (-ARENA_BOUND, ARENA_BOUND, 0.0),
        ]
        floor_poly = self._project_polygon(floor_world, cam_x, cam_y, cam_z, cam_yaw, cam_pitch, z_near=0.08)
        if floor_poly is not None:
            cv2.fillPoly(frame, [floor_poly], (45, 45, 48))

        # High-Contrast Perspective Floor Grid & Markings (Clipped to Near-Plane)
        grid_step = 0.40
        num_grid = int(ARENA_BOUND / grid_step)
        for i in range(-num_grid, num_grid + 1):
            gx = i * grid_step
            p1_cam = self._to_cam(gx, -ARENA_BOUND, 0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            p2_cam = self._to_cam(gx, ARENA_BOUND, 0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            seg = self._clip_segment_near_plane(p1_cam, p2_cam, z_near=0.08)
            if seg:
                color = (65, 68, 74) if i != 0 else (90, 95, 105)
                thickness = 2 if i == 0 else 1
                cv2.line(frame, seg[0], seg[1], color, thickness)

        for j in range(-num_grid, num_grid + 1):
            gy = j * grid_step
            p1_cam = self._to_cam(-ARENA_BOUND, gy, 0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            p2_cam = self._to_cam(ARENA_BOUND, gy, 0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            seg = self._clip_segment_near_plane(p1_cam, p2_cam, z_near=0.08)
            if seg:
                color = (65, 68, 74) if j != 0 else (90, 95, 105)
                thickness = 2 if j == 0 else 1
                cv2.line(frame, seg[0], seg[1], color, thickness)

        # Center Floor Ring
        c_proj = self._project_point(0, 0, 0.001, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
        if c_proj:
            scale_c = FOCAL_LENGTH / max(0.1, c_proj[2])
            cv2.circle(frame, (c_proj[0], c_proj[1]), max(8, int(0.45 * scale_c)), (85, 90, 100), 2)

        # ── Pass 2: Ground Contact Shadows (Drawn Directly on Floor Before Walls) ──
        for obj in ARENA_OBJECTS.values():
            b_proj = self._project_point(obj.x, obj.y, 0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            if b_proj:
                bx, by, bz = b_proj
                scale = FOCAL_LENGTH / max(0.1, bz)
                r_px = max(6, int(obj.radius * scale))
                shadow_overlay = frame.copy()
                cv2.ellipse(shadow_overlay, (bx, by), (r_px + 8, max(4, int(r_px * 0.45))),
                            0, 0, 360, (0, 0, 0), -1)
                cv2.addWeighted(shadow_overlay, 0.48, frame, 0.52, 0, frame)

        if view_mode in ("TPP", "TOP", "TOPDOWN"):
            r_proj = self._project_point(rx, ry, 0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            if r_proj:
                rcx, rcy, rz = r_proj
                scale = FOCAL_LENGTH / max(0.1, rz)
                sw_w = max(18, int(0.36 * scale))
                sw_h = max(8, int(0.22 * scale))
                shadow_overlay = frame.copy()
                cv2.ellipse(shadow_overlay, (rcx, rcy), (sw_w, sw_h),
                            int(-math.degrees(ryaw)), 0, 360, (0, 0, 0), -1)
                cv2.addWeighted(shadow_overlay, 0.45, frame, 0.55, 0, frame)

                # Forward Headlight Ground Illumination Beam (Crisp Pure White)
                beam_len = 0.95
                beam_spread = 0.35
                b_left = self._project_point(rx + beam_len * math.cos(ryaw) - beam_spread * math.sin(ryaw),
                                             ry + beam_len * math.sin(ryaw) + beam_spread * math.cos(ryaw),
                                             0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
                b_right = self._project_point(rx + beam_len * math.cos(ryaw) + beam_spread * math.sin(ryaw),
                                              ry + beam_len * math.sin(ryaw) - beam_spread * math.cos(ryaw),
                                              0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
                if b_left and b_right:
                    beam_poly = np.array([[rcx, rcy], [b_left[0], b_left[1]], [b_right[0], b_right[1]]], dtype=np.int32)
                    beam_layer = frame.copy()
                    cv2.fillPoly(beam_layer, [beam_poly], (245, 248, 255))
                    cv2.addWeighted(beam_layer, 0.14, frame, 0.86, 0, frame)

        # ── Pass 3: Solid Enclosing Facility Walls (Near-Plane Clipped, Culled & Depth Sorted) ──
        wall_h = 2.40
        baseboard_h = 0.14
        near_z = 0.08
        B = ARENA_BOUND

        wall_specs = [
            # South wall (y = -B, facing +Y into room)
            {
                "center": (0.0, -B), "inward": (0.0, 1.0),
                "quad": [(-B, -B, 0.0), (B, -B, 0.0), (B, -B, wall_h), (-B, -B, wall_h)],
                "bb_quad": [(-B, -B, 0.0), (B, -B, 0.0), (B, -B, baseboard_h), (-B, -B, baseboard_h)],
                "bb_line": [(-B, -B, baseboard_h), (B, -B, baseboard_h)],
                "top_line": [(-B, -B, wall_h), (B, -B, wall_h)],
            },
            # East wall (x = +B, facing -X into room)
            {
                "center": (B, 0.0), "inward": (-1.0, 0.0),
                "quad": [(B, -B, 0.0), (B, B, 0.0), (B, B, wall_h), (B, -B, wall_h)],
                "bb_quad": [(B, -B, 0.0), (B, B, 0.0), (B, B, baseboard_h), (B, -B, baseboard_h)],
                "bb_line": [(B, -B, baseboard_h), (B, B, baseboard_h)],
                "top_line": [(B, -B, wall_h), (B, B, wall_h)],
            },
            # North wall (y = +B, facing -Y into room)
            {
                "center": (0.0, B), "inward": (0.0, -1.0),
                "quad": [(B, B, 0.0), (-B, B, 0.0), (-B, B, wall_h), (B, B, wall_h)],
                "bb_quad": [(B, B, 0.0), (-B, B, 0.0), (-B, B, baseboard_h), (B, B, baseboard_h)],
                "bb_line": [(B, B, baseboard_h), (-B, B, baseboard_h)],
                "top_line": [(B, B, wall_h), (-B, B, wall_h)],
            },
            # West wall (x = -B, facing +X into room)
            {
                "center": (-B, 0.0), "inward": (1.0, 0.0),
                "quad": [(-B, B, 0.0), (-B, -B, 0.0), (-B, -B, wall_h), (-B, B, wall_h)],
                "bb_quad": [(-B, B, 0.0), (-B, -B, 0.0), (-B, -B, baseboard_h), (-B, B, baseboard_h)],
                "bb_line": [(-B, B, baseboard_h), (-B, -B, baseboard_h)],
                "top_line": [(-B, B, wall_h), (-B, -B, wall_h)],
            },
        ]

        # Depth-sort walls from furthest to closest to camera
        for w in wall_specs:
            w["dist"] = math.hypot(cam_x - w["center"][0], cam_y - w["center"][1])
        wall_specs.sort(key=lambda w: w["dist"], reverse=True)

        for w in wall_specs:
            # Backface culling: verify camera is on interior side of the wall
            vx = cam_x - w["center"][0]
            vy = cam_y - w["center"][1]
            dot = vx * w["inward"][0] + vy * w["inward"][1]
            if dot <= 0.01:
                continue

            # Render drywall quad
            wall_poly = self._project_polygon(w["quad"], cam_x, cam_y, cam_z, cam_yaw, cam_pitch, z_near=near_z)
            if wall_poly is not None:
                cv2.fillPoly(frame, [wall_poly], (235, 238, 240))
                cv2.polylines(frame, [wall_poly], True, (190, 195, 200), 1)

            # Render dark graphite baseboard quad
            bb_poly = self._project_polygon(w["bb_quad"], cam_x, cam_y, cam_z, cam_yaw, cam_pitch, z_near=near_z)
            if bb_poly is not None:
                cv2.fillPoly(frame, [bb_poly], (30, 32, 35))

            # Baseboard top trim line
            p1_cam = self._to_cam(w["bb_line"][0][0], w["bb_line"][0][1], w["bb_line"][0][2],
                                  cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            p2_cam = self._to_cam(w["bb_line"][1][0], w["bb_line"][1][1], w["bb_line"][1][2],
                                  cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            seg_bb = self._clip_segment_near_plane(p1_cam, p2_cam, z_near=near_z)
            if seg_bb is not None:
                cv2.line(frame, seg_bb[0], seg_bb[1], (50, 52, 56), 1)

            # High-vis yellow/amber hazard safety coping
            p1_cam = self._to_cam(w["top_line"][0][0], w["top_line"][0][1], w["top_line"][0][2],
                                  cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            p2_cam = self._to_cam(w["top_line"][1][0], w["top_line"][1][1], w["top_line"][1][2],
                                  cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            seg_top = self._clip_segment_near_plane(p1_cam, p2_cam, z_near=near_z)
            if seg_top is not None:
                cv2.line(frame, seg_top[0], seg_top[1], (25, 195, 245), 3)

        # ── Step 4: Depth-Sorted Rendering of 3D Objects & Rover (Painter's Algorithm) ──
        render_queue = []

        # Enqueue 6 Target Objects
        for key, obj in ARENA_OBJECTS.items():
            proj = self._project_point(obj.x, obj.y, 0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            if proj:
                render_queue.append({
                    "type": "object",
                    "key": key,
                    "obj": obj,
                    "z_cam": proj[2],
                    "base_proj": proj
                })

        # Enqueue Rover Chassis (Only in TPP and TOP views)
        if view_mode in ("TPP", "TOP", "TOPDOWN"):
            r_proj = self._project_point(rx, ry, 0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            if r_proj:
                render_queue.append({
                    "type": "rover",
                    "z_cam": r_proj[2],
                    "base_proj": r_proj
                })

        # Sort back-to-front (descending z_cam)
        render_queue.sort(key=lambda item: item["z_cam"], reverse=True)

        for item in render_queue:
            if item["type"] == "object":
                self._render_3d_object(frame, item["obj"], item["key"], target_key,
                                       cam_x, cam_y, cam_z, cam_yaw, cam_pitch, render_shadow=False)
            elif item["type"] == "rover":
                self._render_3d_rover(frame, rx, ry, ryaw, vl, vr, wheel_l, wheel_r,
                                      cam_x, cam_y, cam_z, cam_yaw, cam_pitch, render_shadow=False)

        # Store clean un-annotated frame for perception worker & depth map
        with self.state.lock:
            self.latest_clean_frame = frame.copy()

        # ── Step 4.5: Render Real-Time YOLO Bounding Boxes & Tactical Badges ──
        disp_frame = frame.copy()
        with self.state.lock:
            cur_dets = list(self.state.yolo_detections)
            cur_target = self.state.target_name.lower()

        for det in cur_dets:
            box = det.get("box", [])
            if len(box) == 4:
                x1, y1, x2, y2 = [int(v) for v in box]
                label = str(det.get("label", "object"))
                conf = float(det.get("conf", 0.0))

                # Highlight if matches current chase target
                is_target = cur_target != "none" and (label.lower() in cur_target or cur_target in label.lower())
                color = (120, 255, 0) if is_target else (255, 229, 0)  # Neon Green / Neon Cyan (BGR)

                # Bounding box border
                cv2.rectangle(disp_frame, (x1, y1), (x2, y2), color, 1)

                # Tactical Corner Brackets (GTA / Cyberpunk style)
                c_len = min(16, max(6, (x2 - x1) // 4, (y2 - y1) // 4))
                cv2.line(disp_frame, (x1, y1), (x1 + c_len, y1), color, 2)
                cv2.line(disp_frame, (x1, y1), (x1, y1 + c_len), color, 2)
                cv2.line(disp_frame, (x2, y1), (x2 - c_len, y1), color, 2)
                cv2.line(disp_frame, (x2, y1), (x2, y1 + c_len), color, 2)
                cv2.line(disp_frame, (x1, y2), (x1 + c_len, y2), color, 2)
                cv2.line(disp_frame, (x1, y2), (x1, y2 - c_len), color, 2)
                cv2.line(disp_frame, (x2, y2), (x2 - c_len, y2), color, 2)
                cv2.line(disp_frame, (x2, y2), (x2, y2 - c_len), color, 2)

                # Tactical Label Pill
                tag = f"🎯 {label.upper()} {int(conf * 100)}%"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                tag_y = max(y1 - 6, th + 8)
                cv2.rectangle(disp_frame, (x1, tag_y - th - 5), (x1 + tw + 10, tag_y + 3), (11, 18, 32), -1)
                cv2.rectangle(disp_frame, (x1, tag_y - th - 5), (x1 + tw + 10, tag_y + 3), color, 1)
                cv2.putText(disp_frame, tag, (x1 + 5, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Step 5: Encode Final 1280x720 Widescreen JPEG ──
        _, rgb_enc = cv2.imencode(".jpg", disp_frame, [cv2.IMWRITE_JPEG_QUALITY, 84])
        self.latest_rgb = rgb_enc.tobytes()

        # ── Step 6: Generate Real-Time Monocular MAGMA Depth Map ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Vertical gradient ramp modeling geometric scene depth falloff
        ramp = np.linspace(0.12, 0.98, CAMERA_H, dtype=np.float32).reshape(CAMERA_H, 1)
        depth_norm = np.clip(np.repeat(ramp, CAMERA_W, axis=1) * 0.70 + (gray.astype(np.float32) / 255.0) * 0.30, 0, 1)
        depth_u8 = (depth_norm * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_MAGMA)
        _, depth_enc = cv2.imencode(".jpg", depth_color, [cv2.IMWRITE_JPEG_QUALITY, 80])
        self.latest_depth = depth_enc.tobytes()

    # ── Realistic 3D Asset Rendering Engine (Traffic Cone, Potted Plant, Chair, Crates, Hazard Box) ──
    def _render_3d_object(self, frame, obj: ArenaObject, key: str, target_key: str,
                          cam_x, cam_y, cam_z, cam_yaw, cam_pitch, render_shadow: bool = True):
        b_proj = self._project_point(obj.x, obj.y, 0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
        t_proj = self._project_point(obj.x, obj.y, obj.height, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
        if not b_proj or not t_proj:
            return

        bx, by, bz = b_proj
        tx, ty, tz = t_proj

        scale = FOCAL_LENGTH / max(0.1, bz)
        r_px = max(6, int(obj.radius * scale))

        # 1. Soft Realistic Ground Contact Shadow
        if render_shadow:
            shadow_overlay = frame.copy()
            cv2.ellipse(shadow_overlay, (bx, by), (r_px + 8, max(4, int(r_px * 0.45))),
                        0, 0, 360, (0, 0, 0), -1)
            cv2.addWeighted(shadow_overlay, 0.48, frame, 0.52, 0, frame)

        # 2. Asset-Specific High-Fidelity 3D Geometry
        if key == "RED":
            # ── Traffic Cone: Heavy Square Rubber Base + Orange Cone + Dual White Collars ──
            base_half = obj.radius * 1.05
            base_corners = [
                (obj.x - base_half, obj.y - base_half),
                (obj.x + base_half, obj.y - base_half),
                (obj.x + base_half, obj.y + base_half),
                (obj.x - base_half, obj.y + base_half),
            ]
            base_pts = [self._project_point(cx, cy, 0.015, cam_x, cam_y, cam_z, cam_yaw, cam_pitch) for cx, cy in base_corners]
            if all(base_pts):
                poly = np.array([[p[0], p[1]] for p in base_pts], dtype=np.int32)
                cv2.fillPoly(frame, [poly], (25, 25, 30))
                cv2.polylines(frame, [poly], True, (50, 50, 55), 1)

            # Orange Conical Body Facets
            num_facets = 12
            pts_bot, pts_top, pts_band1_b, pts_band1_t, pts_band2_b, pts_band2_t = [], [], [], [], [], []
            for i in range(num_facets):
                ang = (i / num_facets) * 2.0 * math.pi
                cos_a, sin_a = math.cos(ang), math.sin(ang)
                # Outer conical contour
                pb = self._project_point(obj.x + obj.radius * 0.85 * cos_a, obj.y + obj.radius * 0.85 * sin_a, 0.02, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
                pt = self._project_point(obj.x + obj.radius * 0.20 * cos_a, obj.y + obj.radius * 0.20 * sin_a, obj.height, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
                p1b = self._project_point(obj.x + obj.radius * 0.65 * cos_a, obj.y + obj.radius * 0.65 * sin_a, obj.height * 0.35, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
                p1t = self._project_point(obj.x + obj.radius * 0.52 * cos_a, obj.y + obj.radius * 0.52 * sin_a, obj.height * 0.50, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
                p2b = self._project_point(obj.x + obj.radius * 0.40 * cos_a, obj.y + obj.radius * 0.40 * sin_a, obj.height * 0.65, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
                p2t = self._project_point(obj.x + obj.radius * 0.30 * cos_a, obj.y + obj.radius * 0.30 * sin_a, obj.height * 0.78, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
                if pb and pt: pts_bot.append(pb); pts_top.append(pt)
                if p1b and p1t: pts_band1_b.append(p1b); pts_band1_t.append(p1t)
                if p2b and p2t: pts_band2_b.append(p2b); pts_band2_t.append(p2t)

            if len(pts_bot) >= 4:
                # Main Orange Body
                for i in range(len(pts_bot)):
                    j = (i + 1) % len(pts_bot)
                    poly = np.array([[pts_bot[i][0], pts_bot[i][1]], [pts_bot[j][0], pts_bot[j][1]],
                                     [pts_top[j][0], pts_top[j][1]], [pts_top[i][0], pts_top[i][1]]], dtype=np.int32)
                    shade = 0.70 + 0.30 * math.sin(i / len(pts_bot) * math.pi)
                    cv2.fillPoly(frame, [poly], (int(10 * shade), int(85 * shade), int(245 * shade)))

                # Reflective White Collars
                if len(pts_band1_b) == len(pts_bot):
                    for i in range(len(pts_band1_b)):
                        j = (i + 1) % len(pts_band1_b)
                        poly1 = np.array([[pts_band1_b[i][0], pts_band1_b[i][1]], [pts_band1_b[j][0], pts_band1_b[j][1]],
                                          [pts_band1_t[j][0], pts_band1_t[j][1]], [pts_band1_t[i][0], pts_band1_t[i][1]]], dtype=np.int32)
                        cv2.fillPoly(frame, [poly1], (235, 235, 245))
                if len(pts_band2_b) == len(pts_bot):
                    for i in range(len(pts_band2_b)):
                        j = (i + 1) % len(pts_band2_b)
                        poly2 = np.array([[pts_band2_b[i][0], pts_band2_b[i][1]], [pts_band2_b[j][0], pts_band2_b[j][1]],
                                          [pts_band2_t[j][0], pts_band2_t[j][1]], [pts_band2_t[i][0], pts_band2_t[i][1]]], dtype=np.int32)
                        cv2.fillPoly(frame, [poly2], (235, 235, 245))

        elif key == "GREEN":
            # ── Potted Plant: Terracotta Saucer/Pot + Woody Stems + Layered Foliage Canopy ──
            pot_h = obj.height * 0.38
            pts_pot_b, pts_pot_t = [], []
            num_facets = 10
            for i in range(num_facets):
                ang = (i / num_facets) * 2.0 * math.pi
                pb = self._project_point(obj.x + obj.radius * 0.75 * math.cos(ang), obj.y + obj.radius * 0.75 * math.sin(ang), 0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
                pt = self._project_point(obj.x + obj.radius * 0.85 * math.cos(ang), obj.y + obj.radius * 0.85 * math.sin(ang), pot_h, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
                if pb and pt: pts_pot_b.append(pb); pts_pot_t.append(pt)

            if len(pts_pot_b) >= 4:
                for i in range(len(pts_pot_b)):
                    j = (i + 1) % len(pts_pot_b)
                    poly = np.array([[pts_pot_b[i][0], pts_pot_b[i][1]], [pts_pot_b[j][0], pts_pot_b[j][1]],
                                     [pts_pot_t[j][0], pts_pot_t[j][1]], [pts_pot_t[i][0], pts_pot_t[i][1]]], dtype=np.int32)
                    shade = 0.65 + 0.35 * math.sin(i / len(pts_pot_b) * math.pi)
                    cv2.fillPoly(frame, [poly], (int(42 * shade), int(90 * shade), int(195 * shade)))  # Terracotta BGR

                # Dark Potting Soil
                top_poly = np.array([[p[0], p[1]] for p in pts_pot_t], dtype=np.int32)
                cv2.fillPoly(frame, [top_poly], (25, 28, 35))

            # Woody Branch
            trunk_p = self._project_point(obj.x, obj.y, pot_h + 0.12, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            if trunk_p:
                cv2.line(frame, (bx, by - int(pot_h * scale * 0.4)), (trunk_p[0], trunk_p[1]), (30, 60, 95), max(3, int(4 * scale / 900.0)))

            # Lush Botanical Foliage Canopy (Clustered Leaf Spheres)
            leaf_clusters = [
                (0.00, 0.00, obj.height * 0.78, obj.radius * 0.95, (38, 175, 45)),
                (-0.06, 0.04, obj.height * 0.68, obj.radius * 0.70, (28, 145, 35)),
                (0.07, -0.05, obj.height * 0.70, obj.radius * 0.75, (52, 205, 60)),
                (0.00, 0.03, obj.height * 0.92, obj.radius * 0.65, (45, 185, 52)),
            ]
            for lx, ly, lz, lr, col in leaf_clusters:
                cp = self._project_point(obj.x + lx, obj.y + ly, lz, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
                if cp:
                    c_rad = max(6, int(lr * scale))
                    cv2.circle(frame, (cp[0], cp[1]), c_rad, col, -1)
                    cv2.circle(frame, (cp[0], cp[1]), c_rad, (int(col[0] * 0.7), int(col[1] * 0.7), int(col[2] * 0.7)), 1)

        elif key == "BLUE":
            # ── Office Chair: 5-Star Chrome Base + Lift Cylinder + Cobalt Blue Cushion + Backrest ──
            # 5-Star Chrome Base
            for i in range(5):
                ang = (i / 5.0) * 2.0 * math.pi
                leg_p = self._project_point(obj.x + obj.radius * 0.95 * math.cos(ang), obj.y + obj.radius * 0.95 * math.sin(ang), 0.02, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
                if leg_p:
                    cv2.line(frame, (bx, by), (leg_p[0], leg_p[1]), (210, 215, 225), max(2, int(3 * scale / 900.0)))
                    cv2.circle(frame, (leg_p[0], leg_p[1]), max(2, int(0.025 * scale)), (30, 30, 35), -1)

            # Chrome Pneumatic Cylinder
            seat_z = obj.height * 0.48
            seat_p = self._project_point(obj.x, obj.y, seat_z, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            if seat_p:
                cv2.line(frame, (bx, by), (seat_p[0], seat_p[1]), (220, 225, 235), max(3, int(6 * scale / 900.0)))

            # Cobalt Blue Ergonomic Seat Cushion
            sc_half = obj.radius * 0.75
            seat_corners = [
                (obj.x - sc_half, obj.y - sc_half), (obj.x + sc_half, obj.y - sc_half),
                (obj.x + sc_half, obj.y + sc_half), (obj.x - sc_half, obj.y + sc_half),
            ]
            sc_pts = [self._project_point(cx, cy, seat_z, cam_x, cam_y, cam_z, cam_yaw, cam_pitch) for cx, cy in seat_corners]
            if all(sc_pts):
                poly = np.array([[p[0], p[1]] for p in sc_pts], dtype=np.int32)
                cv2.fillPoly(frame, [poly], (215, 90, 18))  # Deep Cobalt Blue BGR
                cv2.polylines(frame, [poly], True, (245, 125, 45), 1)

            # Curved Ergonomic Mesh Backrest
            back_b = self._project_point(obj.x - sc_half * 0.8, obj.y, seat_z + 0.02, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            back_t = self._project_point(obj.x - sc_half * 0.8, obj.y, obj.height, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            if back_b and back_t:
                bw = max(8, int(sc_half * 1.6 * scale))
                bh = max(10, back_b[1] - back_t[1])
                cv2.rectangle(frame, (back_t[0] - bw // 2, back_t[1]), (back_t[0] + bw // 2, back_b[1]), (235, 105, 25), -1)
                cv2.rectangle(frame, (back_t[0] - bw // 2, back_t[1]), (back_t[0] + bw // 2, back_b[1]), (255, 150, 60), 1)

        elif key.startswith("CRATE"):
            # ── Wooden Supply Crate: Timber Planks + Cross Bracing + Corner Plates ──
            cw = obj.radius * 0.95
            ch = obj.height
            c_bot = [
                self._project_point(obj.x - cw, obj.y - cw, 0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch),
                self._project_point(obj.x + cw, obj.y - cw, 0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch),
                self._project_point(obj.x + cw, obj.y + cw, 0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch),
                self._project_point(obj.x - cw, obj.y + cw, 0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch),
            ]
            c_top = [
                self._project_point(obj.x - cw, obj.y - cw, ch, cam_x, cam_y, cam_z, cam_yaw, cam_pitch),
                self._project_point(obj.x + cw, obj.y - cw, ch, cam_x, cam_y, cam_z, cam_yaw, cam_pitch),
                self._project_point(obj.x + cw, obj.y + cw, ch, cam_x, cam_y, cam_z, cam_yaw, cam_pitch),
                self._project_point(obj.x - cw, obj.y + cw, ch, cam_x, cam_y, cam_z, cam_yaw, cam_pitch),
            ]
            if all(c_bot) and all(c_top):
                # 4 Sides
                for i in range(4):
                    j = (i + 1) % 4
                    side_poly = np.array([[c_bot[i][0], c_bot[i][1]], [c_bot[j][0], c_bot[j][1]],
                                          [c_top[j][0], c_top[j][1]], [c_top[i][0], c_top[i][1]]], dtype=np.int32)
                    shade = 0.70 + 0.25 * (i % 2)
                    cv2.fillPoly(frame, [side_poly], (int(55 * shade), int(110 * shade), int(160 * shade)))  # Timber Pine BGR
                    # X-Bracing Planks
                    cv2.line(frame, (c_bot[i][0], c_bot[i][1]), (c_top[j][0], c_top[j][1]), (35, 75, 115), 2)
                    cv2.line(frame, (c_bot[j][0], c_bot[j][1]), (c_top[i][0], c_top[i][1]), (35, 75, 115), 2)
                    cv2.polylines(frame, [side_poly], True, (30, 65, 95), 1)

                # Top Face
                top_poly = np.array([[p[0], p[1]] for p in c_top], dtype=np.int32)
                cv2.fillPoly(frame, [top_poly], (65, 125, 180))
                cv2.polylines(frame, [top_poly], True, (45, 85, 125), 1)

        elif key == "HAZARD":
            # ── Hazard Box: High-Vis Safety Yellow Enclosure + Black Chevrons + Chrome Latches ──
            hw_x = obj.radius * 0.90
            hw_y = obj.radius * 0.75
            hh = obj.height
            h_bot = [
                self._project_point(obj.x - hw_x, obj.y - hw_y, 0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch),
                self._project_point(obj.x + hw_x, obj.y - hw_y, 0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch),
                self._project_point(obj.x + hw_x, obj.y + hw_y, 0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch),
                self._project_point(obj.x - hw_x, obj.y + hw_y, 0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch),
            ]
            h_top = [
                self._project_point(obj.x - hw_x, obj.y - hw_y, hh, cam_x, cam_y, cam_z, cam_yaw, cam_pitch),
                self._project_point(obj.x + hw_x, obj.y - hw_y, hh, cam_x, cam_y, cam_z, cam_yaw, cam_pitch),
                self._project_point(obj.x + hw_x, obj.y + hw_y, hh, cam_x, cam_y, cam_z, cam_yaw, cam_pitch),
                self._project_point(obj.x - hw_x, obj.y + hw_y, hh, cam_x, cam_y, cam_z, cam_yaw, cam_pitch),
            ]
            if all(h_bot) and all(h_top):
                for i in range(4):
                    j = (i + 1) % 4
                    side_poly = np.array([[h_bot[i][0], h_bot[i][1]], [h_bot[j][0], h_bot[j][1]],
                                          [h_top[j][0], h_top[j][1]], [h_top[i][0], h_top[i][1]]], dtype=np.int32)
                    shade = 0.80 + 0.20 * (i % 2)
                    cv2.fillPoly(frame, [side_poly], (int(15 * shade), int(210 * shade), int(245 * shade)))  # Hazard Yellow BGR
                    # Black Diagonal Caution Band
                    mid1 = ((h_bot[i][0] + h_top[i][0]) // 2, (h_bot[i][1] + h_top[i][1]) // 2)
                    mid2 = ((h_bot[j][0] + h_top[j][0]) // 2, (h_bot[j][1] + h_top[j][1]) // 2)
                    cv2.line(frame, mid1, mid2, (20, 20, 25), max(3, int(6 * scale / 900.0)))
                    cv2.polylines(frame, [side_poly], True, (10, 160, 200), 1)

                top_poly = np.array([[p[0], p[1]] for p in h_top], dtype=np.int32)
                cv2.fillPoly(frame, [top_poly], (25, 225, 255))
                cv2.polylines(frame, [top_poly], True, (10, 170, 210), 1)

        else:
            # Fallback cylindrical extrusion
            num_facets = 10
            facet_pts_b = []
            facet_pts_t = []
            for i in range(num_facets):
                ang = (i / num_facets) * 2.0 * math.pi
                pb = self._project_point(obj.x + obj.radius * math.cos(ang), obj.y + obj.radius * math.sin(ang), 0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
                pt = self._project_point(obj.x + obj.radius * math.cos(ang), obj.y + obj.radius * math.sin(ang), obj.height, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
                if pb and pt: facet_pts_b.append(pb); facet_pts_t.append(pt)

            if len(facet_pts_b) >= 3 and len(facet_pts_b) == len(facet_pts_t):
                for i in range(len(facet_pts_b)):
                    j = (i + 1) % len(facet_pts_b)
                    poly = np.array([[facet_pts_b[i][0], facet_pts_b[i][1]], [facet_pts_b[j][0], facet_pts_b[j][1]],
                                     [facet_pts_t[j][0], facet_pts_t[j][1]], [facet_pts_t[i][0], facet_pts_t[i][1]]], dtype=np.int32)
                    cv2.fillPoly(frame, [poly], obj.bgr_color)

        # 3. Target HUD Reticle (If active chase target)
        if key == target_key:
            reticle_y = ty - 18
            cv2.circle(frame, (tx, reticle_y), 12, (0, 229, 255), 2)
            cv2.circle(frame, (tx, reticle_y), 4, (0, 229, 255), -1)
            cv2.line(frame, (tx - 18, reticle_y), (tx + 18, reticle_y), (0, 229, 255), 1)
            cv2.line(frame, (tx, reticle_y - 18), (tx, reticle_y + 18), (0, 229, 255), 1)

            tag = f"LOCKED: {obj.name.upper()}"
            cv2.putText(frame, tag, (tx - len(tag) * 4, reticle_y - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 229, 255), 1, cv2.LINE_AA)

    # ── 3D Rover Model Rendering (Military Prototype: Matte Gunmetal & Subtle White LEDs) ──
    def _render_3d_rover(self, frame, rx, ry, ryaw, vl, vr, wheel_l, wheel_r,
                         cam_x, cam_y, cam_z, cam_yaw, cam_pitch, render_shadow: bool = True):
        r_proj = self._project_point(rx, ry, 0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
        if not r_proj:
            return

        rcx, rcy, rz = r_proj
        scale = FOCAL_LENGTH / max(0.1, rz)

        # 1. Realistic Ambient Occlusion Ground Contact Shadow
        if render_shadow:
            sw_w = max(18, int(0.36 * scale))
            sw_h = max(8, int(0.22 * scale))
            shadow_overlay = frame.copy()
            cv2.ellipse(shadow_overlay, (rcx, rcy), (sw_w, sw_h),
                        int(-math.degrees(ryaw)), 0, 360, (0, 0, 0), -1)
            cv2.addWeighted(shadow_overlay, 0.45, frame, 0.55, 0, frame)

            # Subtle Forward Headlight Ground Illumination Beam (Crisp Pure White)
            beam_len = 0.95
            beam_spread = 0.35
            b_left = self._project_point(rx + beam_len * math.cos(ryaw) - beam_spread * math.sin(ryaw),
                                         ry + beam_len * math.sin(ryaw) + beam_spread * math.cos(ryaw),
                                         0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            b_right = self._project_point(rx + beam_len * math.cos(ryaw) + beam_spread * math.sin(ryaw),
                                          ry + beam_len * math.sin(ryaw) - beam_spread * math.cos(ryaw),
                                          0.0, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            if b_left and b_right:
                beam_poly = np.array([[rcx, rcy], [b_left[0], b_left[1]], [b_right[0], b_right[1]]], dtype=np.int32)
                beam_layer = frame.copy()
                cv2.fillPoly(beam_layer, [beam_poly], (245, 248, 255))  # Crisp white headlight beam
                cv2.addWeighted(beam_layer, 0.14, frame, 0.86, 0, frame)

        # 3. Sleek Elongated Matte Gunmetal AGV Chassis Hull
        bw_2 = 0.18  # elongated length
        bh_2 = 0.085 # sleek narrow width
        chassis_corners_local = [
            (bw_2, -bh_2 * 0.65, 0.032), (bw_2, bh_2 * 0.65, 0.032),     # Sloped front bumper
            (bw_2 * 0.70, bh_2, 0.035), (-bw_2 * 0.80, bh_2, 0.035),    # Right side
            (-bw_2, bh_2 * 0.70, 0.032), (-bw_2, -bh_2 * 0.70, 0.032),  # Rear deflector
            (-bw_2 * 0.80, -bh_2, 0.035), (bw_2 * 0.70, -bh_2, 0.035)   # Left side
        ]
        chassis_poly_pts = []
        for lx, ly, lz in chassis_corners_local:
            wx = rx + lx * math.cos(ryaw) - ly * math.sin(ryaw)
            wy = ry + lx * math.sin(ryaw) + ly * math.cos(ryaw)
            pt = self._project_point(wx, wy, lz, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            if pt:
                chassis_poly_pts.append((pt[0], pt[1]))

        if len(chassis_poly_pts) >= 6:
            c_poly = np.array(chassis_poly_pts, dtype=np.int32)
            cv2.fillPoly(frame, [c_poly], (36, 38, 44))  # Tactical Matte Titanium Hull
            cv2.polylines(frame, [c_poly], True, (65, 72, 82), 1)  # Milled Bevel Edge

        # 3.2 Sloped Front Armor Nose Wedge
        nose_local = [
            (bw_2 * 0.95, -bh_2 * 0.60, 0.020), (bw_2 * 0.95, bh_2 * 0.60, 0.020),
            (bw_2 * 0.40, bh_2 * 0.70, 0.045), (bw_2 * 0.40, -bh_2 * 0.70, 0.045)
        ]
        nose_poly_pts = []
        for lx, ly, lz in nose_local:
            wx = rx + lx * math.cos(ryaw) - ly * math.sin(ryaw)
            wy = ry + lx * math.sin(ryaw) + ly * math.cos(ryaw)
            pt = self._project_point(wx, wy, lz, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            if pt:
                nose_poly_pts.append((pt[0], pt[1]))
        if len(nose_poly_pts) == 4:
            n_poly = np.array(nose_poly_pts, dtype=np.int32)
            cv2.fillPoly(frame, [n_poly], (45, 48, 54))  # 30° Sloped Armor Glacis Plate
            cv2.polylines(frame, [n_poly], True, (75, 82, 92), 1)

        # 3.5 Tactical Military Run-Flat Wheels (Matte Black Lug Treads & Tungsten Hubs)
        for side in (-bh_2 * 1.15, bh_2 * 1.15):
            for fx in (-bw_2 * 0.55, bw_2 * 0.45):
                wx = rx + fx * math.cos(ryaw) - side * math.sin(ryaw)
                wy = ry + fx * math.sin(ryaw) + side * math.cos(ryaw)
                wpt = self._project_point(wx, wy, 0.028, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
                if wpt:
                    tire_w = max(4, int(0.025 * scale))
                    tire_h = max(6, int(0.040 * scale))
                    cv2.rectangle(frame, (wpt[0] - tire_w, wpt[1] - tire_h),
                                  (wpt[0] + tire_w, wpt[1] + tire_h), (18, 18, 20), -1)
                    cv2.rectangle(frame, (wpt[0] - tire_w, wpt[1] - tire_h),
                                  (wpt[0] + tire_w, wpt[1] + tire_h), (40, 42, 46), 1)
                    cv2.rectangle(frame, (wpt[0] - max(1, tire_w // 2), wpt[1] - max(1, tire_h // 3)),
                                  (wpt[0] + max(1, tire_w // 2), wpt[1] + max(1, tire_h // 3)), (75, 80, 88), -1)

        # 4. Subtle White LED Headlights (Twin Flush Micro-Projectors — Pure White, Zero Cartoon Circles)
        for side in (-0.055, 0.055):
            hl_wx = rx + bw_2 * math.cos(ryaw) - side * math.sin(ryaw)
            hl_wy = ry + bw_2 * math.sin(ryaw) + side * math.cos(ryaw)
            hl_pt = self._project_point(hl_wx, hl_wy, 0.025, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
            if hl_pt:
                pw = max(2, int(0.014 * scale))
                ph = max(1, int(0.008 * scale))
                cv2.rectangle(frame, (hl_pt[0] - pw, hl_pt[1] - ph),
                              (hl_pt[0] + pw, hl_pt[1] + ph), (255, 255, 255), -1)

        # 5. Rear Subtle Crimson Marker Light (Thin Recessed Strip)
        tl_wx = rx - bw_2 * math.cos(ryaw)
        tl_wy = ry - bw_2 * math.sin(ryaw)
        tl_pt = self._project_point(tl_wx, tl_wy, 0.035, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
        if tl_pt:
            tw = max(4, int(0.032 * scale))
            th = max(1, int(0.006 * scale))
            cv2.rectangle(frame, (tl_pt[0] - tw, tl_pt[1] - th),
                          (tl_pt[0] + tw, tl_pt[1] + th), (25, 25, 185), -1)

        # 6. Centered, Distinct Black LiDAR Puck (Centered on Top Deck — Zero Offset, Zero Cartoon Colors)
        lidar_pt = self._project_point(rx, ry, 0.065, cam_x, cam_y, cam_z, cam_yaw, cam_pitch)
        if lidar_pt:
            lp_w = max(4, int(0.028 * scale))
            lp_h = max(2, int(0.018 * scale))
            # Distinct cylindrical black puck
            cv2.rectangle(frame, (lidar_pt[0] - lp_w, lidar_pt[1] - lp_h),
                          (lidar_pt[0] + lp_w, lidar_pt[1] + lp_h), (16, 16, 18), -1)
            cv2.rectangle(frame, (lidar_pt[0] - lp_w, lidar_pt[1] - lp_h),
                          (lidar_pt[0] + lp_w, lidar_pt[1] + lp_h), (35, 38, 42), 1)
            # Optical horizontal mirror band
            cv2.line(frame, (lidar_pt[0] - lp_w + 1, lidar_pt[1]),
                     (lidar_pt[0] + lp_w - 1, lidar_pt[1]), (48, 52, 58), 1)

    # ── Telemetry API Payload ──
    def get_telemetry(self) -> Dict[str, Any]:
        with self.state.lock:
            # Enriched hardware & model status for HUD
            y_src = self.perception.yolo_source if self.perception else "OFFLINE"
            c_src = self.perception.cache_source if self.perception else "OFFLINE"
            enriched_badge = f"{self.state.neural_badge} | YOLO: {y_src} | CLIP: {c_src}"
            return {
                "x": round(self.state.x, 2),
                "y": round(self.state.y, 2),
                "yaw_deg": round(math.degrees(self.state.yaw), 1),
                "speed_l": round(self.state.left_speed, 2),
                "speed_r": round(self.state.right_speed, 2),
                "mode": self.state.mode.value,
                "target": self.state.target_name,
                "target_key": self.state.target_key,
                "target_x": round(self.state.target_x, 2),
                "target_y": round(self.state.target_y, 2),
                "target_conf": round(self.state.target_conf, 2),
                "status": self.state.status_text,
                "view_mode": self.state.view_mode,
                "vlm_scores": self.state.vlm_scores,
                "yolo_detections": self.state.yolo_detections,
                "neural_device": self.state.neural_device,
                "neural_badge": enriched_badge,
                "fps": self.state.fps,
                "trail": self.state.trail[-120:],
            }
