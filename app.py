#!/usr/bin/env python3
"""
app.py — Master Hugging Face Spaces & Container Launcher for AI Vision Patrol Robot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features:
  1. Sets up Webots R2025a environment variables (DISPLAY, WEBOTS_HOME, LD_LIBRARY_PATH).
  2. Boots Xvfb (Virtual Framebuffer) on DISPLAY=:99 for headless OpenGL rendering.
  3. Launches Webots in headless batch realtime mode with worlds/ai_arena.wbt.
  4. Starts background AI worker and Webots Supervisor controller threads.
  5. Unifies FastAPI backend (/dashboard, /vr, /ws, /video_feed, /depth_feed)
     with an interactive Gradio Interface on Hugging Face port 7860.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import time
import shutil
import subprocess
import threading
import uvicorn

# ──────────────────────────────────────────────────────────────────────────────
# 1. Project Root & Environment Configuration
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set headless display
os.environ["DISPLAY"] = os.environ.get("DISPLAY", ":99")

# Locate or configure WEBOTS_HOME
DEFAULT_WEBOTS_HOME = "/usr/local/webots"
if "WEBOTS_HOME" not in os.environ:
    if os.path.exists(DEFAULT_WEBOTS_HOME):
        os.environ["WEBOTS_HOME"] = DEFAULT_WEBOTS_HOME
    elif os.path.exists("/Applications/Webots.app"):
        os.environ["WEBOTS_HOME"] = "/Applications/Webots.app"
    else:
        os.environ["WEBOTS_HOME"] = DEFAULT_WEBOTS_HOME

webots_home = os.environ["WEBOTS_HOME"]

# Configure Python controller library paths
controller_python_candidates = [
    os.path.join(webots_home, "lib", "controller", "python"),
    os.path.join(webots_home, "Contents", "lib", "controller", "python"),
    "/usr/local/webots/lib/controller/python",
]
for cp in controller_python_candidates:
    if os.path.exists(cp) and cp not in sys.path:
        sys.path.insert(0, cp)
        break

# Configure Dynamic Linker library paths
controller_lib = os.path.join(webots_home, "lib", "controller")
if os.path.exists(controller_lib):
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{controller_lib}:{existing_ld}".rstrip(":")

# Ensure Ultralytics writes cache to /tmp
os.environ["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"
os.makedirs("/tmp/ultralytics", exist_ok=True)

# Port configuration (Hugging Face Spaces defaults to 7860)
PORT = int(os.environ.get("PORT", 7860))
HOST = os.environ.get("HOST", "0.0.0.0")

print("=" * 65)
print("  🤖 AI Vision Patrol Robot — Hugging Face Space Launcher")
print("=" * 65)
print(f"  Project Root:     {PROJECT_ROOT}")
print(f"  WEBOTS_HOME:      {os.environ.get('WEBOTS_HOME')}")
print(f"  DISPLAY:          {os.environ.get('DISPLAY')}")
print(f"  Host / Port:      {HOST}:{PORT}")
print("=" * 65)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Virtual Framebuffer (Xvfb) Boot Routine
# ──────────────────────────────────────────────────────────────────────────────
def boot_xvfb():
    """Starts Xvfb background process on DISPLAY=:99 if running on Linux."""
    if sys.platform != "linux":
        print(f"ℹ️  OS is '{sys.platform}'. Skipping Xvfb initialization.")
        return

    xvfb_bin = shutil.which("Xvfb")
    if not xvfb_bin:
        print("[WARN] 'Xvfb' not found in PATH. Ensure packages.txt includes 'xvfb'.")
        return

    # Check if display :99 is already responding or running
    lock_file = "/tmp/.X99-lock"
    if os.path.exists(lock_file):
        print(f"✅ Xvfb lockfile {lock_file} detected. Virtual display :99 is ready.")
        return

    display = os.environ.get("DISPLAY", ":99")
    print(f"🚀 Booting Xvfb virtual framebuffer on display {display}...")
    try:
        proc = subprocess.Popen(
            [xvfb_bin, display, "-screen", "0", "1024x768x24", "-ac"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(1.5)
        print(f"✅ Xvfb active on {display} (PID: {proc.pid}).")
    except Exception as exc:
        print(f"[WARN] Failed to launch Xvfb: {exc}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Webots Simulation Boot Routine
# ──────────────────────────────────────────────────────────────────────────────
def boot_webots():
    """Launches Webots in headless batch realtime mode with worlds/ai_arena.wbt."""
    world_path = os.path.join(PROJECT_ROOT, "worlds", "ai_arena.wbt")
    if not os.path.exists(world_path):
        print(f"[ERROR] World file not found at: {world_path}")
        return None

    # Resolve webots binary location
    webots_candidates = [
        shutil.which("webots"),
        os.path.join(webots_home, "webots"),
        os.path.join(webots_home, "Contents", "MacOS", "webots"),
        "/usr/local/webots/webots",
        "/Applications/Webots.app/Contents/MacOS/webots",
    ]
    webots_bin = next((b for b in webots_candidates if b and os.path.exists(b)), None)

    if not webots_bin:
        print(f"[WARN] Webots executable not found. Looked in: {webots_candidates}")
        print("       Server will run in standalone/simulated mode.")
        return None

    print(f"🚀 Launching Webots simulation (headless batch mode)...")
    print(f"   Binary: {webots_bin}")
    print(f"   World:  {world_path}")

    try:
        proc = subprocess.Popen(
            [webots_bin, "--batch", "--mode=realtime", "--stdout", "--stderr", world_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"✅ Webots started (PID: {proc.pid}). Waiting 4s for IPC initialization...")
        time.sleep(4)
        return proc
    except Exception as exc:
        print(f"[ERROR] Failed to start Webots: {exc}")
        return None

# ──────────────────────────────────────────────────────────────────────────────
# 4. Import FastAPI Mission Control & Start Worker Threads
# ──────────────────────────────────────────────────────────────────────────────
boot_xvfb()
boot_webots()

from controllers.mission_control.mission_control import (
    app as fastapi_app,
    start_background_threads,
    STATE,
    ControlMode
)

# Launch background AI worker loop and Webots supervisor controller thread
start_background_threads()

# ──────────────────────────────────────────────────────────────────────────────
# 5. Gradio Interface Definition
# ──────────────────────────────────────────────────────────────────────────────
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("[WARN] Gradio library not found. Falling back to pure FastAPI.")

if GRADIO_AVAILABLE:
    custom_css = """
    .gradio-container { max-width: 100% !important; padding: 0 !important; }
    iframe { width: 100%; height: 780px; border: none; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 13px; margin-right: 6px; }
    .badge-green { background: #e8f5e9; color: #2e7d32; }
    .badge-blue { background: #e3f2fd; color: #1565c0; }
    .badge-purple { background: #f3e5f5; color: #7b1fa2; }
    """

    def send_mode_cmd(mode_name: str):
        try:
            with STATE.lock:
                STATE.mode = ControlMode(mode_name)
                STATE.manual_cmd = "STOP"
            return f"Mode successfully switched to: {mode_name}"
        except Exception as e:
            return f"Error setting mode: {e}"

    def send_manual_move(cmd: str):
        try:
            with STATE.lock:
                STATE.manual_cmd = cmd
            return f"Manual command delivered: {cmd}"
        except Exception as e:
            return f"Error: {e}"

    def trigger_vlm():
        try:
            from controllers.mission_control.mission_control import select_vlm_goal
            goal, reason = select_vlm_goal()
            return f"🎯 VLM Selected: {goal}\nReason: {reason}"
        except Exception as e:
            return f"Error triggering VLM: {e}"

    def set_openvocab(prompt: str):
        try:
            with STATE.lock:
                STATE.openvocab_prompt = prompt.strip()
            return f"Open-vocabulary prompt updated: '{prompt}'"
        except Exception as e:
            return f"Error: {e}"

    def get_flight_log():
        log_path = os.path.join(PROJECT_ROOT, "mission_log.md")
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return "".join(lines[-35:])
        return "Mission flight log initializing..."

    with gr.Blocks(title="AI Vision Patrol Robot", css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🤖 AI Vision Patrol Robot — Mission Control & 3D WebXR Digital Twin
            <span class="badge badge-green">● Webots R2025a Engine</span>
            <span class="badge badge-blue">● YOLOv8 + OpenAI CLIP ViT-B/32</span>
            <span class="badge badge-purple">● Monocular Depth MAGMA Stream</span>
            <span class="badge badge-blue">● A-Frame WebXR Digital Twin</span>
            """
        )

        with gr.Row():
            gr.HTML(
                """
                <div style="display:flex; gap:10px; margin-bottom:12px; flex-wrap:wrap;">
                    <a href="/dashboard" target="_blank" style="text-decoration:none; background:#1976d2; color:white; padding:8px 16px; border-radius:6px; font-weight:600;">↗ Open Fullscreen 2D Mission Control</a>
                    <a href="/vr" target="_blank" style="text-decoration:none; background:#7b1fa2; color:white; padding:8px 16px; border-radius:6px; font-weight:600;">↗ Open Fullscreen 3D WebXR Digital Twin</a>
                    <a href="/video_feed" target="_blank" style="text-decoration:none; background:#388e3c; color:white; padding:8px 16px; border-radius:6px; font-weight:600;">📹 Raw RGB Stream</a>
                    <a href="/depth_feed" target="_blank" style="text-decoration:none; background:#d32f2f; color:white; padding:8px 16px; border-radius:6px; font-weight:600;">🗺️ Monocular Depth Stream</a>
                </div>
                """
            )

        with gr.Tabs():
            with gr.TabItem("🖥️ 2D Mission Control Dashboard"):
                gr.HTML('<iframe src="/dashboard" allow="fullscreen"></iframe>')

            with gr.TabItem("🥽 3D WebXR Digital Twin"):
                gr.HTML('<iframe src="/vr" allow="fullscreen; xr-spatial-tracking; accelerometer; gyroscope"></iframe>')

            with gr.TabItem("📹 Dual Vision Feeds"):
                with gr.Row():
                    gr.HTML('<h3>Live RGB Camera + HUD</h3><iframe src="/video_feed" style="height:480px;"></iframe>')
                    gr.HTML('<h3>Monocular Depth Map (MAGMA)</h3><iframe src="/depth_feed" style="height:480px;"></iframe>')

            with gr.TabItem("🕹️ Remote Controls & VLM Deck"):
                status_box = gr.Textbox(label="Command Status", interactive=False)
                with gr.Row():
                    btn_patrol = gr.Button("🧠 AI Patrol Mode", variant="primary")
                    btn_chase = gr.Button("🎯 AI Chase Target", variant="secondary")
                    btn_manual = gr.Button("🕹️ Manual Teleop Mode")

                gr.Markdown("#### Manual Teleoperation")
                with gr.Row():
                    btn_fwd = gr.Button("⬆️ Forward")
                with gr.Row():
                    btn_left = gr.Button("⬅️ Left")
                    btn_stop = gr.Button("⏹️ Stop", variant="stop")
                    btn_right = gr.Button("➡️ Right")
                with gr.Row():
                    btn_back = gr.Button("⬇️ Backward")

                gr.Markdown("#### Multimodal AI Actions")
                with gr.Row():
                    btn_vlm = gr.Button("🧭 Trigger Dynamic VLM Goal Selection", variant="primary")
                with gr.Row():
                    openvocab_in = gr.Textbox(label="Purpose C: Open-Vocabulary Text Prompt", placeholder="e.g. wooden crate, blue workstation, red beacon")
                    btn_vocab = gr.Button("🔍 Lock & Ground Target")

                btn_patrol.click(lambda: send_mode_cmd("AI_PATROL"), outputs=status_box)
                btn_chase.click(lambda: send_mode_cmd("AI_CHASE"), outputs=status_box)
                btn_manual.click(lambda: send_mode_cmd("MANUAL"), outputs=status_box)

                btn_fwd.click(lambda: send_manual_move("FORWARD"), outputs=status_box)
                btn_left.click(lambda: send_manual_move("LEFT"), outputs=status_box)
                btn_stop.click(lambda: send_manual_move("STOP"), outputs=status_box)
                btn_right.click(lambda: send_manual_move("RIGHT"), outputs=status_box)
                btn_back.click(lambda: send_manual_move("BACKWARD"), outputs=status_box)

                btn_vlm.click(trigger_vlm, outputs=status_box)
                btn_vocab.click(set_openvocab, inputs=openvocab_in, outputs=status_box)

            with gr.TabItem("📋 Mission Flight Logs"):
                log_box = gr.Textbox(value=get_flight_log, label="Recent Automated Flight Captions (mission_log.md)", lines=20)
                btn_refresh_log = gr.Button("🔄 Refresh Logs")
                btn_refresh_log.click(get_flight_log, outputs=log_box)

    # Mount Gradio onto the unified FastAPI application on root path "/"
    app = gr.mount_gradio_app(fastapi_app, demo, path="/")
else:
    app = fastapi_app

# ──────────────────────────────────────────────────────────────────────────────
# 6. Standalone Execution / Uvicorn Launcher
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"🚀 Starting Uvicorn server on http://{HOST}:{PORT}")
    try:
        uvicorn.run(app, host=HOST, port=PORT, log_level="info")
    except KeyboardInterrupt:
        print("\nShutdown requested by user.")

