#!/usr/bin/env zsh
# ──────────────────────────────────────────────────────────────────────────────
# run_demo.sh — Master launch script for AI Vision Patrol Robot
#   1. Sets Webots R2025a environment variables
#   2. Opens the world file in Webots (realtime mode)
#   3. Starts the external Python controller + FastAPI server
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

# ── Webots R2025a environment ──
export WEBOTS_HOME="/Applications/Webots.app"
export DYLD_LIBRARY_PATH="$WEBOTS_HOME/Contents/lib/controller:${DYLD_LIBRARY_PATH:-}"
export PYTHONPATH="$WEBOTS_HOME/Contents/lib/controller/python:${PYTHONPATH:-}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🤖 AI Vision Patrol Robot — Webots R2025a Demo"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  WEBOTS_HOME       = $WEBOTS_HOME"
echo "  DYLD_LIBRARY_PATH = $DYLD_LIBRARY_PATH"
echo "  PYTHONPATH        = $PYTHONPATH"
echo "  VENV_PYTHON       = $VENV_PYTHON"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Verify venv ──
if [ ! -f "$VENV_PYTHON" ]; then
  # Fallback to alternate mirror in .nosync if available
  ALT_PYTHON="/Users/gauravsrivastava1212/Documents/My Computer.nosync/SEM-7/Algorithm and Robotics Lab/.venv/bin/python"
  if [ -f "$ALT_PYTHON" ]; then
    VENV_PYTHON="$ALT_PYTHON"
  else
    echo "❌ Virtual environment not found at $VENV_PYTHON"
    echo "   Run: python3.11 -m venv .venv && source .venv/bin/activate && pip install fastapi uvicorn opencv-python-headless numpy ultralytics websockets"
    exit 1
  fi
fi

# ── Kill stale processes on port 8000 ──
echo "🔄 Checking for stale processes on port 8000..."
lsof -ti :8000 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

# ── Launch Webots (if not already running with this world) ──
WORLD_FILE="$PROJECT_DIR/worlds/ai_arena.wbt"
if ! pgrep -f "webots.*ai_arena" > /dev/null 2>&1; then
  echo "🚀 Launching Webots R2025a with $WORLD_FILE ..."
  "$WEBOTS_HOME/Contents/MacOS/webots" --mode=realtime "$WORLD_FILE" &
  WEBOTS_PID=$!
  echo "   Webots PID: $WEBOTS_PID"
  echo "   Waiting 6s for Webots to initialize..."
  sleep 6
else
  echo "✅ Webots already running with ai_arena world."
fi

# ── Start the external controller + FastAPI server ──
echo ""
echo "🧠 Starting Mission Control (Controller + FastAPI + WebSocket + WebXR)..."
echo "   Dashboard:   http://localhost:8000"
echo "   VR Twin:     http://localhost:8000/vr"
echo "   WebSocket:   ws://localhost:8000/ws"
echo ""
exec "$VENV_PYTHON" "$PROJECT_DIR/controllers/mission_control/mission_control.py"
