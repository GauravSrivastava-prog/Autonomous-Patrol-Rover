#!/usr/bin/env python3
"""
physics_simulator.py — Real-Time 30Hz Physics Simulation & 3D Camera Renderer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OOP Responsibilities:
  1. SyntheticCameraRenderer:
     - Full 3D-to-2D perspective rendering (cyber-grid floor, radar distance rings).
     - Full-volume 3D bounding boxes spanning z=0 floor to z=height.
     - Target-only neon corner brackets & pill banner in AI_CHASE and Open-Vocab.
     - Subtle distance brackets for objects within 1.8m in AI_PATROL.
  2. ArenaPhysicsSimulator:
     - Solid-body radial collision physics against all 6 arena objects and walls (±1.85m).
     - Seamless 3-Mode State Machine (AI_PATROL, AI_CHASE, MANUAL) with zero stuck states.
     - AI_PATROL 360° vicinity-aware speed modulation:
       * Clear Ground Speed-Up: min_vicinity > 0.90m -> 4.80 rad/s
       * Vicinity Caution Slowdown: min_vicinity <= 0.85m -> 2.10-3.20 rad/s
     - AI_CHASE potential-field pursuit with tangential obstacle repulsion & 0.58m standoff orbit.
     - 3.0s VLM Auto-Goal semantic inspection countdown at <0.65m standoff.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import time
import threading
from typing import Dict, Tuple, Optional, List, Any
import numpy as np
import cv2

from domain_state import (
    ArenaObject,
    ARENA_OBJECTS,
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
from perception_engine import PerceptionEngine


class SyntheticCameraRenderer:
    """3D Camera Renderer generating cyber-grid floor, shaded 3D shapes, and HUD overlays."""

    def __init__(self):
        self.fov = math.radians(64.0)
        self.focal = (CAM_W / 2.0) / math.tan(self.fov / 2.0)
        self.horizon = CAM_H // 2 - 20   # 220 px
        self.cam_z = 0.14                # camera height above floor

    def project_object(self, rx: float, ry: float, ryaw: float, obj: ArenaObject, coords: Tuple[float, float]) -> Optional[dict]:
        """Projects full 3D object volume (z=0 floor to z=height) into 2D camera coordinates."""
        ox, oy = coords
        dx, dy = ox - rx, oy - ry
        cos_y, sin_y = math.cos(ryaw), math.sin(ryaw)
        cx = dx * cos_y + dy * sin_y
        cy = -dx * sin_y + dy * cos_y
        if cx <= 0.16:
            return None

        dist = math.hypot(dx, dy)
        sy_bot = int(self.horizon + (self.cam_z / cx) * self.focal)
        sy_top = int(self.horizon - ((obj.height - self.cam_z) / cx) * self.focal)
        pw = max(18, int((2.0 * obj.radius / cx) * self.focal))
        ph = max(18, sy_bot - sy_top)
        sx = int(CAM_W / 2.0 - (cy / cx) * self.focal)

        x1, x2 = sx - pw // 2, sx + pw // 2
        y1, y2 = sy_top, sy_bot

        if x2 < -40 or x1 >= CAM_W + 40 or y2 < self.horizon - 30 or y1 >= CAM_H + 30:
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

    def render(
        self,
        rx: float,
        ry: float,
        ryaw: float,
        target_positions: Dict[str, Tuple[float, float]],
        mode: ControlMode,
        locked_target_key: str,
        locked_target_label: str,
        vlm_inspect_timer: int,
        cur_semantic_scene: str,
        cur_semantic_score: float,
        left_v: float,
        right_v: float,
    ) -> np.ndarray:
        """Renders rich 3D camera viewport with cyber-grid floor, shaded objects, and single HUD banner."""
        frame = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        horizon = self.horizon

        # 1. Distant Walls / Ceiling Gradient
        for y in range(0, horizon):
            t = y / float(horizon)
            frame[y, :] = (int(32 + 25 * t), int(22 + 18 * t), int(14 + 14 * t))

        wall_top = max(0, horizon - 34)
        frame[wall_top:horizon, :] = (65, 48, 38)
        cv2.line(frame, (0, wall_top), (CAM_W, wall_top), (220, 180, 0), 1, cv2.LINE_AA)
        cv2.line(frame, (0, horizon), (CAM_W, horizon), (255, 215, 0), 2, cv2.LINE_AA)

        # 2. Cyber-Grid Floor
        for y in range(horizon, CAM_H):
            t = (y - horizon) / float(CAM_H - horizon)
            frame[y, :] = (int(45 + 35 * (t ** 1.1)), int(32 + 25 * (t ** 1.1)), int(22 + 18 * (t ** 1.1)))

        cos_y, sin_y = math.cos(ryaw), math.sin(ryaw)
        focal = self.focal

        # Grid lines
        for gx in np.linspace(-2.0, 2.0, 11):
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

        for gy in np.linspace(-2.0, 2.0, 11):
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

        # Concentric floor radar rings (0.6m, 1.2m)
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

        # 3. 3D Shaded Objects (Painter's sorting: furthest first)
        render_list = []
        for k, obj in ARENA_OBJECTS.items():
            ox, oy = target_positions.get(k, obj.coords)
            dx, dy = ox - rx, oy - ry
            cx = dx * cos_y + dy * sin_y
            cy = -dx * sin_y + dy * cos_y
            if cx > 0.16:
                render_list.append((cx, cy, math.hypot(dx, dy), obj, (ox, oy)))

        render_list.sort(key=lambda item: item[0], reverse=True)

        for cx, cy, dist, obj, coords in render_list:
            sx = int(CAM_W / 2.0 - (cy / cx) * focal)
            sy_bot = int(horizon + (0.14 / cx) * focal)
            pw = max(14, int((2.0 * obj.radius / cx) * focal))
            ph = max(14, int((obj.height / cx) * focal))
            sy_top = sy_bot - ph
            x1, x2 = sx - pw // 2, sx + pw // 2

            if x2 < -40 or x1 >= CAM_W + 40 or sy_bot < horizon - 20 or sy_top >= CAM_H:
                continue

            c = obj.color_bgr
            shape = obj.shape

            if shape == "cone":
                cv2.ellipse(frame, (sx, sy_bot), (pw // 2 + 4, max(2, ph // 10)), 0, 0, 360, (20, 20, 20), -1)
                cv2.rectangle(frame, (x1, sy_bot - max(2, ph // 12)), (x2, sy_bot), (30, 30, 30), -1)
                pts = np.array([[sx, sy_top], [x1 + 2, sy_bot - max(2, ph // 12)], [x2 - 2, sy_bot - max(2, ph // 12)]], dtype=np.int32)
                cv2.fillPoly(frame, [pts], c, cv2.LINE_AA)
                y_stripe1 = sy_top + int(ph * 0.42)
                y_stripe2 = sy_top + int(ph * 0.58)
                w_s1, w_s2 = int(pw * 0.42), int(pw * 0.58)
                stripe_pts = np.array([
                    [sx - w_s1 // 2, y_stripe1], [sx + w_s1 // 2, y_stripe1],
                    [sx + w_s2 // 2, y_stripe2], [sx - w_s2 // 2, y_stripe2]
                ], dtype=np.int32)
                cv2.fillPoly(frame, [stripe_pts], (255, 255, 255), cv2.LINE_AA)
                cv2.polylines(frame, [pts], True, (15, 15, 120), 1, cv2.LINE_AA)

            elif shape == "plant":
                cv2.ellipse(frame, (sx, sy_bot), (pw // 2, max(2, ph // 8)), 0, 0, 360, (20, 20, 20), -1)
                pot_h = max(6, ph // 3)
                pot_w_top, pot_w_bot = int(pw * 0.7), int(pw * 0.5)
                pot_pts = np.array([
                    [sx - pot_w_top // 2, sy_bot - pot_h], [sx + pot_w_top // 2, sy_bot - pot_h],
                    [sx + pot_w_bot // 2, sy_bot], [sx - pot_w_bot // 2, sy_bot]
                ], dtype=np.int32)
                cv2.fillPoly(frame, [pot_pts], (35, 65, 140), cv2.LINE_AA)
                fol_r = max(8, pw // 2)
                cv2.circle(frame, (sx, sy_bot - pot_h - int(ph * 0.32)), fol_r, (15, 140, 25), -1, cv2.LINE_AA)
                cv2.circle(frame, (sx - fol_r // 3, sy_bot - pot_h - int(ph * 0.38)), int(fol_r * 0.8), (25, 195, 40), -1, cv2.LINE_AA)
                cv2.circle(frame, (sx + fol_r // 3, sy_bot - pot_h - int(ph * 0.38)), int(fol_r * 0.8), (35, 225, 50), -1, cv2.LINE_AA)

            elif shape == "chair":
                cv2.ellipse(frame, (sx, sy_bot), (pw // 2 + 2, max(2, ph // 10)), 0, 0, 360, (20, 20, 20), -1)
                leg_h = int(ph * 0.45)
                seat_y = sy_bot - leg_h
                leg_w = max(2, pw // 14)
                cv2.line(frame, (x1 + pw // 6, sy_bot), (x1 + pw // 5, seat_y), (140, 140, 150), leg_w, cv2.LINE_AA)
                cv2.line(frame, (x2 - pw // 6, sy_bot), (x2 - pw // 5, seat_y), (140, 140, 150), leg_w, cv2.LINE_AA)
                seat_h = max(4, ph // 8)
                cv2.rectangle(frame, (x1 + 2, seat_y - seat_h), (x2 - 2, seat_y), (220, 100, 10), -1)
                back_h = int(ph * 0.45)
                cv2.rectangle(frame, (x1 + pw // 8, sy_top), (x2 - pw // 8, sy_top + back_h), (245, 120, 15), -1)

            elif shape == "crate":
                cv2.ellipse(frame, (sx, sy_bot), (pw // 2 + 2, max(2, ph // 10)), 0, 0, 360, (20, 20, 20), -1)
                cv2.rectangle(frame, (x1, sy_top), (x2, sy_bot), c, -1)
                cv2.rectangle(frame, (x1, sy_top), (x2, sy_bot), (25, 55, 115), 2)
                cv2.line(frame, (x1 + 2, sy_top + 2), (x2 - 2, sy_bot - 2), (30, 65, 130), 2, cv2.LINE_AA)
                cv2.line(frame, (x1 + 2, sy_bot - 2), (x2 - 2, sy_top + 2), (30, 65, 130), 2, cv2.LINE_AA)

            elif shape == "hazard_box":
                cv2.ellipse(frame, (sx, sy_bot), (pw // 2 + 2, max(2, ph // 10)), 0, 0, 360, (20, 20, 20), -1)
                cv2.rectangle(frame, (x1, sy_top), (x2, sy_bot), c, -1)
                cv2.rectangle(frame, (x1, sy_top), (x2, sy_bot), (20, 20, 20), 2)
                stripe_step = max(6, pw // 5)
                for oxs in range(x1 - pw, x2 + pw, stripe_step):
                    pts = np.array([
                        [oxs, sy_bot], [oxs + stripe_step // 2, sy_bot],
                        [oxs + stripe_step // 2 + ph, sy_top], [oxs + ph, sy_top]
                    ], dtype=np.int32)
                    cv2.fillPoly(frame, [pts], (20, 20, 20))

        # 4. Target-Only HUD Overlays (Zero Duplicate Clutter)
        if mode == ControlMode.AI_CHASE and locked_target_key in ARENA_OBJECTS:
            target_obj = ARENA_OBJECTS[locked_target_key]
            coords = target_positions.get(locked_target_key, target_obj.coords)
            proj = self.project_object(rx, ry, ryaw, target_obj, coords)
            if proj is not None:
                bx1, by1, bx2, by2 = proj["box"]
                bsx, bsy = proj["sx"], proj["sy"]
                bpw, bph = proj["pw"], proj["ph"]
                bcolor = target_obj.color_bgr

                # Corner brackets
                c_len = max(6, min(18, bpw // 4, bph // 4))
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), bcolor, 1)
                cv2.line(frame, (bx1, by1), (bx1 + c_len, by1), bcolor, 2)
                cv2.line(frame, (bx1, by1), (bx1, by1 + c_len), bcolor, 2)
                cv2.line(frame, (bx2, by1), (bx2 - c_len, by1), bcolor, 2)
                cv2.line(frame, (bx2, by1), (bx2, by1 + c_len), bcolor, 2)
                cv2.line(frame, (bx1, by2), (bx1 + c_len, by2), bcolor, 2)
                cv2.line(frame, (bx1, by2), (bx1, by2 - c_len), bcolor, 2)
                cv2.line(frame, (bx2, by2), (bx2 - c_len, by2), bcolor, 2)
                cv2.line(frame, (bx2, by2), (bx2, by2 - c_len), bcolor, 2)
                cv2.drawMarker(frame, (bsx, bsy), bcolor, cv2.MARKER_CROSS, 14, 1)

                # Single Unified Pill Banner
                banner_text = f"LOCKED: {target_obj.name.upper()} [{proj['dist']:.2f}m | 96%]"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(banner_text, font, 0.38, 1)
                pill_h = th + 10
                pill_w = tw + 16

                if by1 - pill_h - 4 >= 10:
                    py1, py2 = by1 - pill_h - 4, by1 - 4
                else:
                    py1, py2 = by2 + 4, by2 + pill_h + 4

                px1 = max(4, min(CAM_W - pill_w - 4, bsx - pill_w // 2))
                px2 = px1 + pill_w
                cv2.rectangle(frame, (px1, py1), (px2, py2), (18, 22, 32), -1)
                cv2.rectangle(frame, (px1, py1), (px2, py2), bcolor, 1)
                cv2.putText(frame, banner_text, (px1 + 8, py1 + th + 5), font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

            # VLM Inspection countdown banner
            if vlm_inspect_timer > 0:
                sec_left = math.ceil(vlm_inspect_timer / 30.0)
                bw, bh = 460, 36
                bx, by = (CAM_W - bw) // 2, 38
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (15, 70, 15), -1)
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 120), 2)
                cv2.putText(frame, f"[VLM INSPECTION ACTIVE: {sec_left}s | CAPTURING SEMANTICS]",
                            (bx + 14, by + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        elif mode == ControlMode.AI_PATROL:
            # Subtle distance brackets on objects within 1.8m
            for k, obj in ARENA_OBJECTS.items():
                coords = target_positions.get(k, obj.coords)
                d = math.hypot(coords[0] - rx, coords[1] - ry)
                if d < 1.80:
                    proj = self.project_object(rx, ry, ryaw, obj, coords)
                    if proj is not None:
                        bx1, by1, bx2, by2 = proj["box"]
                        c_subtle = (120, 150, 170)
                        c_len = max(4, min(10, proj["pw"] // 4))
                        cv2.line(frame, (bx1, by1), (bx1 + c_len, by1), c_subtle, 1)
                        cv2.line(frame, (bx1, by1), (bx1, by1 + c_len), c_subtle, 1)
                        cv2.line(frame, (bx2, by1), (bx2 - c_len, by1), c_subtle, 1)
                        cv2.line(frame, (bx2, by1), (bx2, by1 + c_len), c_subtle, 1)
                        cv2.putText(frame, f"{obj.label} ({d:.1f}m)", (bx1, max(18, by1 - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, c_subtle, 1, cv2.LINE_AA)

        # Top Mode Badge and Reticle
        mode_colors = {
            "AI_PATROL": (0, 220, 220),
            "AI_CHASE": (0, 60, 255),
            "MANUAL": (240, 200, 0),
        }
        cv2.putText(frame, f"MODE: {mode.value}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    mode_colors.get(mode.value, (255, 255, 255)), 2)
        cv2.drawMarker(frame, (CAM_W // 2, CAM_H // 2), (0, 255, 255), cv2.MARKER_CROSS, 16, 1)

        speed_text = f"SPD: L={left_v:.2f} R={right_v:.2f}"
        cv2.putText(frame, speed_text, (CAM_W - 195, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 240), 1, cv2.LINE_AA)

        cv2.putText(frame, f"CLIP: {cur_semantic_scene.upper()} ({cur_semantic_score:.2f})", (8, CAM_H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        return frame


class ArenaPhysicsSimulator:
    """Encapsulates differential drive kinematics, solid-body collisions, and 3-mode state machine."""

    def __init__(self, state: RoverTelemetryState, perception: PerceptionEngine, logger: MissionLogger):
        self.state = state
        self.perception = perception
        self.logger = logger
        self.renderer = SyntheticCameraRenderer()

        # Pose & Kinematics
        self.rx: float = 0.0
        self.ry: float = 0.0
        self.ryaw: float = 0.4

        self.patrol_wp_idx: int = 0
        self.PATROL_WAYPOINTS: List[Tuple[float, float]] = [
            (1.10, 0.75), (1.10, -0.75), (-1.10, -0.75), (-1.10, 0.75)
        ]

        # Timers and Flags
        self.standoff_logged: bool = False
        self.avoidance_active: bool = False
        self.vlm_inspect_timer: int = 0
        self.last_mode: Optional[ControlMode] = None
        self._is_running: bool = False
        self._thread: Optional[threading.Thread] = None

    def resolve_collisions(self, rx: float, ry: float) -> Tuple[float, float, bool]:
        """Solid-body radial pushback ensuring rover never phases through objects or walls."""
        collided = False
        with self.state.lock:
            target_positions = dict(self.state.target_positions)

        for k, obj in ARENA_OBJECTS.items():
            ox, oy = target_positions.get(k, obj.coords)
            dx = rx - ox
            dy = ry - oy
            dist = math.hypot(dx, dy)
            min_dist = obj.radius + ROVER_RADIUS
            if dist < min_dist:
                collided = True
                if dist > 1e-4:
                    push = (min_dist - dist) + 0.006
                    rx += (dx / dist) * push
                    ry += (dy / dist) * push
                else:
                    rx += min_dist

        bound = ARENA_BOUND - ROVER_RADIUS
        if rx < -bound: rx = -bound; collided = True
        elif rx > bound: rx = bound; collided = True
        if ry < -bound: ry = -bound; collided = True
        elif ry > bound: ry = bound; collided = True

        return rx, ry, collided

    def step_simulation(self, dt: float):
        """Single 30Hz simulation step."""
        with self.state.lock:
            mode = self.state.mode
            manual_cmd = self.state.manual_cmd
            teleport = self.state.teleport_flag
            self.state.teleport_flag = False
            openvocab_prompt = self.state.openvocab_prompt
            auto_goal = self.state.auto_goal_vlm
            preferred_target = self.state.preferred_target
            depth_obstacle = self.state.depth_obstacle_detected
            depth_l = self.state.depth_left_mean
            depth_r = self.state.depth_right_mean
            target_positions = dict(self.state.target_positions)

        if teleport:
            self.rx, self.ry, self.ryaw = 0.0, 0.0, 0.0
            with self.state.lock:
                self.state.trail.clear()
            self.logger.log_event("Rover teleport reset to arena origin (0, 0)", category="TELEPORT", state=self.state)

        # Mode Transition Reset (Zero Stuck States)
        if mode != self.last_mode:
            self.standoff_logged = False
            self.avoidance_active = False
            self.vlm_inspect_timer = 0
            self.last_mode = mode

        # Dynamic Sensor Modeling
        cos_y, sin_y = math.cos(self.ryaw), math.sin(self.ryaw)
        dist_wall = min(ARENA_BOUND - abs(self.rx), ARENA_BOUND - abs(self.ry))
        ps_l = max(0.0, (0.45 - dist_wall) * 950.0) if dist_wall < 0.45 else 0.0
        ps_r = ps_l

        for k, obj in ARENA_OBJECTS.items():
            ox, oy = target_positions.get(k, obj.coords)
            dx = ox - self.rx
            dy = oy - self.ry
            fwd_dist = dx * cos_y + dy * sin_y
            lat_dist = -dx * sin_y + dy * cos_y
            if 0.05 < fwd_dist < (0.75 + obj.radius) and abs(lat_dist) < (0.45 + obj.radius):
                prox = max(0.0, ((0.75 + obj.radius) - fwd_dist) * 1100.0)
                if lat_dist < 0:
                    ps_l += prox * 1.2
                    ps_r += prox * 0.3
                else:
                    ps_r += prox * 1.2
                    ps_l += prox * 0.3

        ps_l = min(1000.0, ps_l)
        ps_r = min(1000.0, ps_r)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Mode Controller & Velocity Calculation
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        left_v, right_v = 0.0, 0.0
        locked_key: str = "NONE"
        locked_label: str = "NONE"

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
            locked_key = "PATROL"
            locked_label = f"Waypoint {self.patrol_wp_idx + 1}"

            # 360° Omnidirectional Vicinity Proximity
            d_wall = ARENA_BOUND - max(abs(self.rx), abs(self.ry)) - ROVER_RADIUS
            d_obs = [
                math.hypot(self.rx - target_positions.get(k, obj.coords)[0], self.ry - target_positions.get(k, obj.coords)[1])
                - obj.radius - ROVER_RADIUS
                for k, obj in ARENA_OBJECTS.items()
            ]
            min_vicinity = min(d_wall, min(d_obs))

            # Dynamic Speed Modulation: Speed up in open space, slow down in vicinity
            if min_vicinity > 0.90:
                patrol_speed = 4.80   # Clear Ground Boost
            elif min_vicinity <= 0.85:
                # Proportional slowdown between 2.10 and 3.20 rad/s
                ratio = max(0.0, min(1.0, (min_vicinity - 0.40) / 0.45))
                patrol_speed = 2.10 + ratio * (3.20 - 2.10)
            else:
                patrol_speed = 3.80

            wp_x, wp_y = self.PATROL_WAYPOINTS[self.patrol_wp_idx]
            dist_to_wp = math.hypot(wp_x - self.rx, wp_y - self.ry)
            if dist_to_wp < 0.38:
                self.patrol_wp_idx = (self.patrol_wp_idx + 1) % len(self.PATROL_WAYPOINTS)
                wp_x, wp_y = self.PATROL_WAYPOINTS[self.patrol_wp_idx]

            angle_to_wp = math.atan2(wp_y - self.ry, wp_x - self.rx)
            yaw_err = (angle_to_wp - self.ryaw + math.pi) % (2.0 * math.pi) - math.pi

            obstacle_hazard = (max(ps_l, ps_r) > 380) or depth_obstacle or (min_vicinity < 0.32)
            if obstacle_hazard:
                # Decisive forward clearance turn without stopping at 0.00
                if ps_l >= ps_r or (depth_obstacle and depth_l > depth_r):
                    left_v = 4.20
                    right_v = 1.40
                else:
                    left_v = 1.40
                    right_v = 4.20

                if not self.avoidance_active:
                    self.logger.log_event(f"Obstacle clearance avoidance active (min_dist={min_vicinity:.2f}m)", category="AVOID", state=self.state)
                    self.avoidance_active = True
            else:
                self.avoidance_active = False
                turn = max(-2.4, min(2.4, yaw_err * 2.2))
                left_v = max(-MAX_SPEED, min(MAX_SPEED, patrol_speed - turn))
                right_v = max(-MAX_SPEED, min(MAX_SPEED, patrol_speed + turn))

        elif mode == ControlMode.AI_CHASE:
            # Resolve Target
            if openvocab_prompt:
                locked_key, target_obj = self.perception.resolve_prompt(openvocab_prompt, self.rx, self.ry)
                locked_label = target_obj.name
                target_tx, target_ty = target_positions.get(locked_key, target_obj.coords)
            elif auto_goal and preferred_target in ARENA_OBJECTS:
                locked_key = preferred_target
                target_obj = ARENA_OBJECTS[locked_key]
                locked_label = target_obj.name
                target_tx, target_ty = target_positions.get(locked_key, target_obj.coords)
            else:
                # Nearest arena target
                locked_key = min(
                    ARENA_OBJECTS.keys(),
                    key=lambda k: math.hypot(
                        self.rx - target_positions.get(k, ARENA_OBJECTS[k].coords)[0],
                        self.ry - target_positions.get(k, ARENA_OBJECTS[k].coords)[1]
                    )
                )
                target_obj = ARENA_OBJECTS[locked_key]
                locked_label = target_obj.name
                target_tx, target_ty = target_positions.get(locked_key, target_obj.coords)

            dist_to_target = math.hypot(target_tx - self.rx, target_ty - self.ry)
            angle_to_target = math.atan2(target_ty - self.ry, target_tx - self.rx)
            standoff_threshold = 0.58

            if dist_to_target <= standoff_threshold:
                if not self.standoff_logged:
                    self.logger.log_event(f"Target standoff established at {dist_to_target:.2f}m for {locked_label}", category="STANDOFF", state=self.state)
                    self.standoff_logged = True

                # Purpose A: VLM Auto-Goal 3s inspection
                if auto_goal:
                    if self.vlm_inspect_timer <= 0:
                        self.vlm_inspect_timer = 90  # 3.0s at 30Hz
                        self.logger.log_event(f"VLM target arrived: {locked_label}. Commencing 3.0s inspection.", category="VLM GOAL", state=self.state)
                    else:
                        self.vlm_inspect_timer -= 1
                        if self.vlm_inspect_timer == 0:
                            self.logger.log_event(f"VLM inspection complete for {locked_label}. Advancing objective.", category="VLM GOAL", state=self.state)
                            self.perception.plan_vlm_goal(self.state, self.logger)
                            self.standoff_logged = False

                # Smooth standoff orbit around target
                orbit_tangent = angle_to_target + (math.pi / 2.0)
                orbit_err = (orbit_tangent - self.ryaw + math.pi) % (2.0 * math.pi) - math.pi
                left_v = max(0.8, min(3.2, 2.10 - orbit_err * 1.4))
                right_v = max(0.8, min(3.2, 2.10 + orbit_err * 1.4))

            else:
                self.standoff_logged = False
                self.vlm_inspect_timer = 0

                # Potential-field path planning
                vec_att_x = (target_tx - self.rx) / dist_to_target
                vec_att_y = (target_ty - self.ry) / dist_to_target

                u_rep_x, u_rep_y = 0.0, 0.0
                for k, obj in ARENA_OBJECTS.items():
                    if k == locked_key:
                        continue
                    ox, oy = target_positions.get(k, obj.coords)
                    dx = self.rx - ox
                    dy = self.ry - oy
                    d_o = math.hypot(dx, dy)
                    clearance = d_o - obj.radius - ROVER_RADIUS
                    if clearance < 0.60 and d_o > 1e-4:
                        k_rep = (0.60 - clearance) / 0.60
                        u_rep_x += (dx / d_o) * k_rep * 1.6
                        u_rep_y += (dy / d_o) * k_rep * 1.6

                if abs(self.rx) > 1.25:
                    u_rep_x -= math.copysign(1.2 * ((abs(self.rx) - 1.25) / 0.49), self.rx)
                if abs(self.ry) > 1.25:
                    u_rep_y -= math.copysign(1.2 * ((abs(self.ry) - 1.25) / 0.49), self.ry)

                v_tot_x = vec_att_x + u_rep_x
                v_tot_y = vec_att_y + u_rep_y
                desired_yaw = math.atan2(v_tot_y, v_tot_x)
                yaw_err = (desired_yaw - self.ryaw + math.pi) % (2.0 * math.pi) - math.pi

                if dist_to_target > 1.20:
                    chase_speed = 4.40
                else:
                    scale = (dist_to_target - standoff_threshold) / (1.20 - standoff_threshold)
                    chase_speed = 2.20 + max(0.0, min(1.0, scale)) * (4.40 - 2.20)

                turn = max(-2.4, min(2.4, yaw_err * 2.4))
                left_v = max(-MAX_SPEED, min(MAX_SPEED, chase_speed - turn))
                right_v = max(-MAX_SPEED, min(MAX_SPEED, chase_speed + turn))

        # Kinematics Update
        v_lin = ((left_v + right_v) / (2.0 * MAX_SPEED)) * 0.42
        omega = ((right_v - left_v) / (2.0 * MAX_SPEED)) * 1.65
        self.ryaw = (self.ryaw + omega * dt + math.pi) % (2.0 * math.pi) - math.pi
        self.rx += v_lin * math.cos(self.ryaw) * dt
        self.ry += v_lin * math.sin(self.ryaw) * dt

        # Solid-body collision pushback
        self.rx, self.ry, _ = self.resolve_collisions(self.rx, self.ry)
        yaw_deg = math.degrees(self.ryaw)

        # Render 3D frame
        with self.state.lock:
            semantic_scene = self.state.semantic_scene
            semantic_score = self.state.semantic_score

        frame = self.renderer.render(
            self.rx, self.ry, self.ryaw, target_positions, mode,
            locked_key, locked_label, self.vlm_inspect_timer,
            semantic_scene, semantic_score, left_v, right_v
        )

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        depth_jpeg, haz_det, haz_score, l_mean, r_mean = self.perception.compute_monocular_depth_and_hazard(frame)

        # Update State
        with self.state.lock:
            self.state.frame_raw = frame
            self.state.frame_jpeg = jpeg.tobytes()
            self.state.depth_jpeg = depth_jpeg
            self.state.depth_obstacle_detected = haz_det
            self.state.depth_hazard_score = haz_score
            self.state.depth_left_mean = l_mean
            self.state.depth_right_mean = r_mean
            self.state.pos = [round(self.rx, 3), round(self.ry, 3), 0.06]
            self.state.orientation = round(self.ryaw, 3)
            self.state.yaw_deg = round(yaw_deg, 1)
            self.state.left_speed = round(left_v, 2)
            self.state.right_speed = round(right_v, 2)
            self.state.ps_left_val = round(ps_l, 1)
            self.state.ps_right_val = round(ps_r, 1)
            self.state.locked_target = locked_label
            self.state.trail.append((self.rx, self.ry))
            if len(self.state.trail) > 500:
                self.state.trail = self.state.trail[-500:]

    def run_loop(self):
        """Dedicated 30Hz simulation loop."""
        print("🚀 30Hz Autonomous Robotics Physics & Camera Engine Active.")
        dt = TIMESTEP / 1000.0
        while self._is_running:
            t_start = time.time()
            try:
                self.step_simulation(dt)
            except Exception as e:
                print(f"[ERROR in physics loop]: {e}")
            elapsed = time.time() - t_start
            time.sleep(max(0.005, dt - elapsed))

    def start(self):
        """Starts physics simulator thread."""
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self.run_loop, daemon=True, name="PhysicsSimulatorThread")
        self._thread.start()

    def stop(self):
        """Stops physics simulator."""
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=1.0)
