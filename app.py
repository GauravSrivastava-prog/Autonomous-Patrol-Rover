import spaces

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CRITICAL: Top-level ZeroGPU hook required for Hugging Face static AST validation
# This unconditioned, unmasked statement fulfills the ZeroGPU runtime contract.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@spaces.GPU(duration=60)
def zero_gpu_warmup():
    """Required top-level ZeroGPU hook for Hugging Face Spaces runtime detection."""
    return "GPU ready"


# Backwards compatibility aliases
zero_gpu_warmup_engine = zero_gpu_warmup
spaces_gpu_warmup = zero_gpu_warmup

"""
app.py — Master Hugging Face Spaces Entrypoint for AI Vision Patrol Robot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Architecture (Gradio-Primary + Non-Blocking Launch + Post-Launch Route Injection):
  1. `import spaces` as absolute first import (ZeroGPU CUDA guard)
  2. Top-level @spaces.GPU decorated zero_gpu_warmup() for static AST scan
  3. Safe lazy loading of heavy models (YOLOv8 & CLIP deferred outside import)
  4. Environment configuration (WEBOTS_HOME, DISPLAY, PYTHONPATH, LD_LIBRARY_PATH)
  5. Boot Xvfb + Webots in background (non-blocking, tolerant of missing binaries)
  6. Import FastAPI mission control app — carries all routes (/dashboard, /vr,
     /ws, /video_feed, /depth_feed, /telemetry, /mission_logs, etc.)
  7. Build Gradio Blocks UI as `demo` (Gradio 6 compatible, title via demo.title)
  8. Non-blocking demo.launch(prevent_thread_lock=True) triggers ZeroGPU worker hook
  9. Post-launch injection of all FastAPI routes directly onto demo.app.router.routes
  10. Start background 30Hz simulation loop and AI worker engine
  11. demo.block_thread() keeps container alive serving both Gradio and FastAPI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import time
import shutil
import subprocess

# ──────────────────────────────────────────────────────────────────────────────
# 1. Project Root & Environment Configuration (before any heavy imports)
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Headless display for Xvfb
os.environ.setdefault("DISPLAY", ":99")

# Locate or configure WEBOTS_HOME
if "WEBOTS_HOME" not in os.environ:
    for candidate in ["/usr/local/webots", "/Applications/Webots.app"]:
        if os.path.exists(candidate):
            os.environ["WEBOTS_HOME"] = candidate
            break
    else:
        os.environ["WEBOTS_HOME"] = "/usr/local/webots"

webots_home = os.environ["WEBOTS_HOME"]

# Add Webots Python controller to sys.path
for _subpath in ["lib/controller/python", "Contents/lib/controller/python"]:
    _cp = os.path.join(webots_home, _subpath)
    if os.path.exists(_cp) and _cp not in sys.path:
        sys.path.insert(0, _cp)
        break

# Dynamic linker paths for Webots native libraries
for _subpath in ["lib/controller", "Contents/lib/controller"]:
    _cl = os.path.join(webots_home, _subpath)
    if os.path.exists(_cl):
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{_cl}:{existing_ld}".rstrip(":")
        break

# Ultralytics cache in writable /tmp
os.environ["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"
os.makedirs("/tmp/ultralytics", exist_ok=True)

# Port configuration (Hugging Face Spaces defaults to 7860)
PORT = int(os.environ.get("PORT", 7860))
HOST = os.environ.get("HOST", "0.0.0.0")

print("=" * 65)
print("  🤖 AI Vision Patrol Robot — Hugging Face Space Launcher")
print("=" * 65)
print(f"  Project Root:     {PROJECT_ROOT}")
print(f"  WEBOTS_HOME:      {webots_home}")
print(f"  DISPLAY:          {os.environ.get('DISPLAY')}")
print(f"  Host / Port:      {HOST}:{PORT}")
print("=" * 65)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Virtual Framebuffer (Xvfb) Boot Routine
# ──────────────────────────────────────────────────────────────────────────────
def boot_xvfb():
    """Starts Xvfb on DISPLAY=:99 for headless OpenGL rendering (Linux only)."""
    if sys.platform != "linux":
        print(f"ℹ️  OS is '{sys.platform}'. Skipping Xvfb.")
        return

    xvfb_bin = shutil.which("Xvfb")
    if not xvfb_bin:
        print("[WARN] Xvfb not in PATH. Headless rendering may fail.")
        return

    if os.path.exists("/tmp/.X99-lock"):
        print("✅ Xvfb lockfile detected — display :99 already active.")
        return

    display = os.environ.get("DISPLAY", ":99")
    print(f"🚀 Booting Xvfb on {display}...")
    try:
        subprocess.Popen(
            [xvfb_bin, display, "-screen", "0", "1024x768x24", "-ac"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.5)
        print(f"✅ Xvfb active on {display}.")
    except Exception as exc:
        print(f"[WARN] Failed to launch Xvfb: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Webots Simulation Boot Routine
# ──────────────────────────────────────────────────────────────────────────────
def boot_webots():
    """Launches Webots in headless batch realtime mode (non-blocking)."""
    world_path = os.path.join(PROJECT_ROOT, "worlds", "ai_arena.wbt")
    if not os.path.exists(world_path):
        print(f"[WARN] World file not found: {world_path}")
        return None

    webots_candidates = [
        shutil.which("webots"),
        os.path.join(webots_home, "webots"),
        os.path.join(webots_home, "Contents", "MacOS", "webots"),
        "/usr/local/webots/webots",
    ]
    webots_bin = next((b for b in webots_candidates if b and os.path.exists(b)), None)

    if not webots_bin:
        print("[INFO] Webots binary not found → running synthetic 3D arena simulation.")
        return None

    print(f"🚀 Launching Webots: {webots_bin}")
    try:
        proc = subprocess.Popen(
            [webots_bin, "--batch", "--mode=realtime", "--stdout", "--stderr", world_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        print(f"✅ Webots started (PID: {proc.pid}). Waiting 4s for IPC init...")
        time.sleep(4)
        return proc
    except Exception as exc:
        print(f"[ERROR] Webots launch failed: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 4. Boot infrastructure, then import the heavy controller module
#    (mission_control.py imports torch/CLIP/YOLO lazily — safe outside import)
# ──────────────────────────────────────────────────────────────────────────────
boot_xvfb()
boot_webots()

from controllers.mission_control.mission_control import (  # noqa: E402
    app as fastapi_app,
    STATE,
    ControlMode,
    plan_vlm_goal,
)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Build the Gradio 6 Interface
#    IMPORTANT: In Gradio 6, title/theme/css are NOT allowed in gr.Blocks()
#    constructor. Styling is injected via inline <style> tags in gr.HTML().
# ──────────────────────────────────────────────────────────────────────────────
import gradio as gr  # noqa: E402

CUSTOM_CSS = """
<style>
.gradio-container { max-width: 100% !important; padding: 0 !important; }
iframe { width: 100%; height: 780px; border: none; border-radius: 8px;
         box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
.badge { display: inline-block; padding: 4px 10px; border-radius: 12px;
         font-weight: 600; font-size: 13px; margin-right: 6px; }
.badge-green  { background: #e8f5e9; color: #2e7d32; }
.badge-blue   { background: #e3f2fd; color: #1565c0; }
.badge-purple { background: #f3e5f5; color: #7b1fa2; }
</style>
"""


def send_mode_cmd(mode_name: str):
    try:
        with STATE.lock:
            STATE.mode = ControlMode(mode_name)
            STATE.manual_cmd = "STOP"
        return f"✅ Mode switched to: {mode_name}"
    except Exception as e:
        return f"❌ Error: {e}"


def send_manual_move(cmd: str):
    try:
        with STATE.lock:
            STATE.manual_cmd = cmd
        return f"✅ Manual command: {cmd}"
    except Exception as e:
        return f"❌ Error: {e}"


@spaces.GPU(duration=60)
def trigger_vlm():
    try:
        inst, target = plan_vlm_goal()
        return f"🎯 VLM Selected: {target}\n{inst}"
    except Exception as e:
        return f"❌ VLM Error: {e}"


def set_openvocab(prompt: str):
    try:
        with STATE.lock:
            STATE.openvocab_prompt = prompt.strip()
        return f"✅ Open-vocab prompt: '{prompt}'"
    except Exception as e:
        return f"❌ Error: {e}"


def get_flight_log():
    log_path = os.path.join(PROJECT_ROOT, "mission_log.md")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return "".join(lines[-35:])
    return "Mission flight log initializing..."


# Gradio 6 Blocks (exposed at module level as `demo`)
# Note: In Gradio 6, constructor arguments like title, css, theme are removed.
with gr.Blocks() as demo:
    # Inject scoped custom CSS
    gr.HTML(CUSTOM_CSS)

    gr.Markdown(
        """
        # 🤖 AI Vision Patrol Robot — Mission Control & 3D WebXR Digital Twin
        <span class="badge badge-green">● Webots R2025a</span>
        <span class="badge badge-blue">● YOLOv8 + OpenAI CLIP ViT-B/32</span>
        <span class="badge badge-purple">● Monocular Depth MAGMA</span>
        <span class="badge badge-blue">● A-Frame WebXR Digital Twin</span>
        """
    )

    with gr.Row():
        gr.HTML(
            """
            <div style="display:flex; gap:10px; margin-bottom:12px; flex-wrap:wrap;">
                <a href="/dashboard" target="_blank"
                   style="text-decoration:none; background:#1976d2; color:white;
                          padding:8px 16px; border-radius:6px; font-weight:600;">
                   ↗ Fullscreen 2D Mission Control</a>
                <a href="/vr" target="_blank"
                   style="text-decoration:none; background:#7b1fa2; color:white;
                          padding:8px 16px; border-radius:6px; font-weight:600;">
                   ↗ Fullscreen 3D WebXR Twin</a>
                <a href="/video_frame.jpg" target="_blank"
                   style="text-decoration:none; background:#388e3c; color:white;
                          padding:8px 16px; border-radius:6px; font-weight:600;">
                   📷 RGB Snapshot</a>
                <a href="/depth_frame.jpg" target="_blank"
                   style="text-decoration:none; background:#d32f2f; color:white;
                          padding:8px 16px; border-radius:6px; font-weight:600;">
                   🗺️ Depth Snapshot</a>
            </div>
            """
        )

    with gr.Tabs():
        with gr.TabItem("🖥️ 2D Mission Control Dashboard"):
            gr.HTML('<iframe src="/dashboard" allow="fullscreen"></iframe>')

        with gr.TabItem("🥽 3D WebXR Digital Twin"):
            gr.HTML(
                '<iframe src="/vr" allow="fullscreen; xr-spatial-tracking; '
                'accelerometer; gyroscope"></iframe>'
            )

        with gr.TabItem("📹 Dual Vision Feeds"):
            with gr.Row():
                gr.HTML(
                    """
                    <div style="display:flex; flex-direction:column; gap:8px;">
                        <h3 style="margin:0 0 4px 0; color:#eee;">Live RGB Camera + HUD</h3>
                        <img id="tab3_rgb_feed" src="/video_frame.jpg"
                             style="width:100%; max-height:480px; border-radius:8px; border:1px solid #333; object-fit:contain; background:#000;" />
                    </div>
                    """
                )
                gr.HTML(
                    """
                    <div style="display:flex; flex-direction:column; gap:8px;">
                        <h3 style="margin:0 0 4px 0; color:#eee;">Monocular Depth Map (MAGMA)</h3>
                        <img id="tab3_depth_feed" src="/depth_frame.jpg"
                             style="width:100%; max-height:480px; border-radius:8px; border:1px solid #333; object-fit:contain; background:#000;" />
                    </div>
                    <script>
                    (function pollTab3Feeds() {
                        const rgbImg = document.getElementById('tab3_rgb_feed');
                        const depthImg = document.getElementById('tab3_depth_feed');
                        function update() {
                            if (!document.hidden) {
                                const t = Date.now();
                                if (rgbImg) rgbImg.src = '/video_frame.jpg?t=' + t;
                                if (depthImg) depthImg.src = '/depth_frame.jpg?t=' + t;
                            }
                            setTimeout(update, 120);
                        }
                        setTimeout(update, 250);
                    })();
                    </script>
                    """
                )

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
                btn_vlm = gr.Button(
                    "🧭 Trigger Dynamic VLM Goal Selection", variant="primary"
                )
                btn_warmup = gr.Button("⚡ Verify ZeroGPU Worker")
            with gr.Row():
                openvocab_in = gr.Textbox(
                    label="Purpose C: Open-Vocabulary Text Prompt",
                    placeholder="e.g. wooden crate, blue workstation, red beacon",
                )
                btn_vocab = gr.Button("🔍 Lock & Ground Target")

            btn_patrol.click(lambda: send_mode_cmd("AI_PATROL"), outputs=status_box)
            btn_chase.click(lambda: send_mode_cmd("AI_CHASE"), outputs=status_box)
            btn_manual.click(lambda: send_mode_cmd("MANUAL"), outputs=status_box)

            btn_fwd.click(lambda: send_manual_move("FORWARD"), outputs=status_box)
            btn_left.click(lambda: send_manual_move("LEFT"), outputs=status_box)
            btn_stop.click(lambda: send_manual_move("STOP"), outputs=status_box)
            btn_right.click(lambda: send_manual_move("RIGHT"), outputs=status_box)
            btn_back.click(lambda: send_manual_move("BACKWARD"), outputs=status_box)

            btn_vlm.click(fn=trigger_vlm, outputs=status_box)
            btn_warmup.click(fn=zero_gpu_warmup, outputs=status_box)
            btn_vocab.click(fn=set_openvocab, inputs=openvocab_in, outputs=status_box)
            demo.load(fn=zero_gpu_warmup)

        with gr.TabItem("📋 Mission Flight Logs"):
            log_box = gr.Textbox(
                value=get_flight_log,
                label="Recent Automated Flight Captions (mission_log.md)",
                lines=20,
            )
            btn_refresh_log = gr.Button("🔄 Refresh Logs")
            btn_refresh_log.click(fn=get_flight_log, outputs=log_box)

# Set title on the Blocks instance
demo.title = "AI Vision Patrol Robot"

# Expose module-level `app` handle (will reference demo.app once launched)
app = getattr(demo, "app", None)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Non-Blocking Launch, Post-Launch Route Injection & Main Thread Lock
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Launching Gradio Blocks with prevent_thread_lock=True & ssr_mode=False...")
    demo.queue()
    app, local_url, share_url = demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        prevent_thread_lock=True,
        ssr_mode=False,
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Post-Launch FastAPI Route & WebSocket Injection
    # demo.app is now the live, running FastAPI/Starlette instance serving port 7860
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    existing_paths = {getattr(r, "path", None) for r in demo.app.router.routes}
    for route in fastapi_app.routes:
        r_path = getattr(route, "path", None)
        if r_path != "/" and r_path not in existing_paths:
            demo.app.router.routes.insert(0, route)

    # Additional helper routes on demo.app
    @demo.app.get("/gradio", include_in_schema=False)
    def _gradio_redirect():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/")

    @demo.app.get("/gpu_status", include_in_schema=False)
    def _gpu_status():
        status = zero_gpu_warmup()
        return {"status": status, "zerogpu": True}

    # Start background threads (AI worker engine, 3D simulation loop)
    from controllers.mission_control.mission_control import start_background_threads
    start_background_threads()

    print("=" * 65)
    print("  ✅ AI Vision Patrol Robot — All Services Active!")
    print(f"  📡 Local URL:      {local_url}")
    print(f"  🖥️  Gradio UI:      http://0.0.0.0:{PORT}/")
    print(f"  📊 Dashboard:      http://0.0.0.0:{PORT}/dashboard")
    print(f"  🥽 WebXR Twin:     http://0.0.0.0:{PORT}/vr")
    print(f"  📹 Video Feed:     http://0.0.0.0:{PORT}/video_feed")
    print(f"  🗺️  Depth Feed:     http://0.0.0.0:{PORT}/depth_feed")
    print(f"  ⚡ ZeroGPU Status:  http://0.0.0.0:{PORT}/gpu_status")
    print("=" * 65)

    # Hold main thread open so the container stays alive serving Gradio and FastAPI
    demo.block_thread()
