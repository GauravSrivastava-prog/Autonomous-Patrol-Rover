#!/usr/bin/env python3
"""
generate_synthetic_data.py — Real-to-Sim Domain Adaptation Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Harvests simulation screenshots of realistic 3D assets (Traffic Cone,
Potted Plant, Chair, Crates, Hazard Box), auto-labels them with true
semantic classes, and fine-tunes YOLOv8 on Apple Silicon MPS (or CUDA/CPU)
to continuously improve sim-domain perception accuracy.

Mapped Semantic Classes:
    0: chair
    1: traffic cone
    2: potted plant
    3: wooden crate
    4: hazard box

Usage:
    python3 ml_pipeline/generate_synthetic_data.py
    python3 ml_pipeline/generate_synthetic_data.py --capture --epochs 5 --batch 8
    python3 ml_pipeline/generate_synthetic_data.py --label-only
"""

import os
import sys
import math
import shutil
import argparse
import tempfile
from pathlib import Path
from datetime import datetime

# ── Resolve project root (supports execution from any working directory) ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "models").exists() else SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
SYNTH_DIR = PROJECT_ROOT / "synthetic_data"
RAW_DIR = SYNTH_DIR / "raw_screenshots"
IMAGES_DIR = SYNTH_DIR / "images"
LABELS_DIR = SYNTH_DIR / "labels"

# Ultralytics sandbox-safe config directory
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ── 5 Explicit Semantic Simulation Classes (Eliminates generic "obstacle") ──
SIM_CLASSES = {
    0: "chair",
    1: "traffic cone",
    2: "potted plant",
    3: "wooden crate",
    4: "hazard box",
}

KEYWORD_TO_CLASS = {
    # 0: chair
    "chair": 0,
    "seat": 0,
    "armchair": 0,
    "swivel": 0,
    "desk chair": 0,
    "blue chair": 0,
    # 1: traffic cone
    "traffic cone": 1,
    "cone": 1,
    "beacon": 1,
    "pylon": 1,
    "red cone": 1,
    # 2: potted plant
    "plant pot": 2,
    "potted plant": 2,
    "plant": 2,
    "tree": 2,
    "flower": 2,
    "bush": 2,
    "green plant": 2,
    # 3: wooden crate
    "wooden crate": 3,
    "crate": 3,
    "wooden": 3,
    "box": 3,
    "cargo": 3,
    "bench": 3,
    "cargo box": 3,
    # 4: hazard box
    "hazard box": 4,
    "hazard": 4,
    "electrical box": 4,
    "barrel": 4,
    "dustbin": 4,
    "toolbox": 4,
    "fire hydrant": 4,
    "yellow box": 4,
}


def resolve_device() -> str:
    """Auto-detect best available hardware accelerator."""
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def load_yolo_model(device: str):
    """Load the fine-tuned YOLO model with CUDA→MPS/CPU remapping."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    tuned_path = MODELS_DIR / "rover_yolo_tuned.pt"
    fallback_path = PROJECT_ROOT / "yolov8n.pt"

    if tuned_path.exists() and tuned_path.stat().st_size > 1024:
        print(f"📦 Loading tuned YOLO weights: {tuned_path}")
        try:
            import torch
            _ = torch.load(tuned_path, map_location="cpu", weights_only=False)
        except Exception as torch_load_err:
            print(f"⚠️ Pre-flight torch.load warning: {torch_load_err}")
        model = YOLO(str(tuned_path))
    elif fallback_path.exists():
        print(f"⚠️  Tuned weights not found. Falling back to: {fallback_path}")
        model = YOLO(str(fallback_path))
    else:
        print(f"⚠️  No local weights found. Downloading yolov8n.pt baseline...")
        model = YOLO("yolov8n.pt")

    # Validate MPS compatibility with graceful fallback
    if device == "mps":
        try:
            import torch
            torch.zeros(1, device="mps")
        except Exception:
            print("⚠️  MPS device test failed — falling back to CPU")
            device = "cpu"

    print(f"🔧 Device: {device.upper()}")
    return model, device


def capture_live_screenshots(count: int = 25, port: int = 7860) -> list:
    """
    Harvests screenshots of realistic 3D simulation assets (Traffic Cone, Potted Plant, Chair, Crates, Hazard Box).
    - If FastAPI server is active, fetches frames via HTTP /video_frame.jpg while cycling views.
    - If server is offline, starts an in-process RobotController, sweeps through arena viewpoints, and captures frames.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    captured = []
    print(f"\n📸 Capturing {count} realistic simulation screenshots into {RAW_DIR}...")

    # Strategy A: Check live running FastAPI server
    import urllib.request
    server_online = False
    try:
        req = urllib.request.urlopen(f"http://127.0.0.1:{port}/telemetry", timeout=1.0)
        if req.status == 200:
            server_online = True
            print(f"⚡ Live FastAPI server detected on port {port}.")
    except Exception:
        server_online = False

    if server_online:
        import time
        import urllib.request
        targets = ["chair", "traffic cone", "potted plant", "wooden crate", "hazard box"]
        for i in range(count):
            try:
                if i % 4 == 0:
                    t = targets[(i // 4) % len(targets)]
                    req = urllib.request.Request(f"http://127.0.0.1:{port}/set_target?prompt={t}", method="POST")
                    urllib.request.urlopen(req, timeout=1.0)
                time.sleep(0.2)
                frame_data = urllib.request.urlopen(f"http://127.0.0.1:{port}/video_frame.jpg", timeout=2.0).read()
                if len(frame_data) > 1024:
                    out_path = RAW_DIR / f"live_capture_{int(time.time())}_{i:03d}.jpg"
                    out_path.write_bytes(frame_data)
                    captured.append(out_path)
            except Exception:
                pass
        if captured:
            print(f"✅ Successfully captured {len(captured)} frames from live server.")
            return captured

    # Strategy B: Offline high-fidelity headless RobotController capture
    print("ℹ️ Standalone mode: Initializing local RobotController engine to generate simulation frames...")
    try:
        from core_engine import RobotController, ARENA_OBJECTS

        rc = RobotController(init_perception=False)
        views = ["TPP", "FPP"]
        idx = 0
        for obj_key, obj in ARENA_OBJECTS.items():
            for v in views:
                rc.set_view_mode(v)
                for ang in [0.0, 1.25, 2.5, 3.75, 5.0]:
                    dist = 1.05
                    rx = obj.x - dist * math.cos(ang)
                    ry = obj.y - dist * math.sin(ang)
                    ryaw = ang
                    rc.state.x = max(-1.8, min(1.8, rx))
                    rc.state.y = max(-1.8, min(1.8, ry))
                    rc.state.yaw = ryaw
                    rc._render_frames()
                    if rc.latest_rgb and len(rc.latest_rgb) > 1024:
                        out_path = RAW_DIR / f"sim_asset_{obj_key.lower()}_{v}_{idx:03d}.jpg"
                        out_path.write_bytes(rc.latest_rgb)
                        captured.append(out_path)
                        idx += 1
                        if len(captured) >= count:
                            break
                if len(captured) >= count:
                    break
            if len(captured) >= count:
                break
        del rc
        import gc
        gc.collect()
        print(f"✅ Captured {len(captured)} high-fidelity simulation object frames.")
    except Exception as e:
        print(f"⚠️ Standalone capture encountered: {e}")

    return captured


def collect_raw_images() -> list:
    """Scan RAW_DIR for supported image files."""
    if not RAW_DIR.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(
        p for p in RAW_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.stat().st_size > 0
    )
    return images


def auto_label(model, images: list, device: str, conf_threshold: float = 0.20):
    """
    Run YOLO inference on each raw screenshot, mapping detections strictly
    to the 5 realistic simulation classes:
        0: chair
        1: traffic cone
        2: potted plant
        3: wooden crate
        4: hazard box
    Eliminates generic "obstacle" workaround labels.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    model_names = getattr(model, "names", {})
    labeled_count = 0
    empty_count = 0

    for img_path in images:
        stem = img_path.stem
        dest_img = IMAGES_DIR / f"{stem}.jpg"
        if img_path.suffix.lower() != ".jpg":
            try:
                from PIL import Image
                im = Image.open(str(img_path)).convert("RGB")
                im.save(str(dest_img), "JPEG", quality=95)
            except Exception:
                shutil.copy2(str(img_path), str(dest_img))
        else:
            shutil.copy2(str(img_path), str(dest_img))

        try:
            results = model.predict(
                source=str(img_path),
                device=device,
                conf=conf_threshold,
                imgsz=640,
                verbose=False,
            )
        except Exception as e:
            print(f"  ⚠️  Inference failed on {img_path.name}: {e}")
            continue

        label_path = LABELS_DIR / f"{stem}.txt"
        lines = []

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            img_h, img_w = r.orig_shape
            for box in r.boxes:
                raw_cls_id = int(box.cls[0])
                raw_name = model_names.get(raw_cls_id, str(raw_cls_id)).lower()

                # Map raw YOLO class/name to one of our 5 semantic simulation classes
                target_cls = None
                for kw, cid in KEYWORD_TO_CLASS.items():
                    if kw in raw_name or raw_name in kw:
                        target_cls = cid
                        break

                if target_cls is None and raw_cls_id in SIM_CLASSES:
                    target_cls = raw_cls_id

                # Skip detections that do not map to the 5 semantic classes (no generic obstacles!)
                if target_cls is None:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                lines.append(f"{target_cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # Ground-truth heuristic fallback if YOLO missed the synthetic render
        if not lines:
            lower_stem = stem.lower()
            for kw, cid in [
                ("cone", 1), ("red", 1),
                ("plant", 2), ("green", 2),
                ("chair", 0), ("blue", 0),
                ("crate", 3),
                ("hazard", 4), ("yellow", 4),
            ]:
                if kw in lower_stem:
                    lines.append(f"{cid} 0.500000 0.520000 0.380000 0.420000")
                    break

        label_path.write_text("\n".join(lines) + "\n" if lines else "")
        if lines:
            labeled_count += 1
            classes_detected = [SIM_CLASSES.get(int(l.split()[0]), "?") for l in lines]
            print(f"  ✅ {img_path.name} → {len(lines)} detections {classes_detected} (conf≥{conf_threshold})")
        else:
            empty_count += 1
            print(f"  ⬚  {img_path.name} → 0 detections (empty label)")

    print(f"\n📊 Auto-Labeling Summary: {labeled_count} labeled / {empty_count} empty / {len(images)} total")
    return labeled_count


def create_dataset_yaml() -> Path:
    """Generate a dataset.yaml strictly mapped to the 5 realistic simulation classes."""
    names_yaml = "\n".join(f"  {k}: \"{v}\"" for k, v in SIM_CLASSES.items())
    yaml_content = f"""# Auto-generated Real-to-Sim synthetic dataset (Realistic 3D Assets)
# Generated: {datetime.now().isoformat()}
path: {SYNTH_DIR}
train: images
val: images

nc: {len(SIM_CLASSES)}
names:
{names_yaml}
"""
    yaml_path = SYNTH_DIR / "synthetic_dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"📝 Dataset YAML written to: {yaml_path} ({len(SIM_CLASSES)} classes: {list(SIM_CLASSES.values())})")
    return yaml_path


def fine_tune(model, dataset_yaml: Path, device: str, epochs: int, batch: int):
    """Run Ultralytics fine-tuning loop on the auto-labeled synthetic data."""
    output_dir = SYNTH_DIR / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🏋️ Starting {epochs}-epoch fine-tuning loop on {device.upper()}...")
    print(f"   Device: {device.upper()} | Batch: {batch} | ImgSz: 640")
    print(f"   Dataset: {dataset_yaml}")
    print(f"   Output:  {output_dir}")
    print("─" * 60)

    try:
        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            batch=batch,
            imgsz=640,
            device=device,
            project=str(output_dir),
            name=f"synth_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            exist_ok=True,
            pretrained=True,
            optimizer="AdamW",
            lr0=0.0005,
            lrf=0.01,
            warmup_epochs=1,
            mosaic=0.5,
            plots=True,
            verbose=True,
        )
        print(f"\n✅ Fine-tuning complete!")
    except Exception as e:
        print(f"\n❌ Fine-tuning failed: {e}")
        return None

    # Copy best weights back to models/
    best_pt = None
    for run_dir in sorted(output_dir.iterdir(), reverse=True):
        candidate = run_dir / "weights" / "best.pt"
        if candidate.exists():
            best_pt = candidate
            break

    if best_pt and best_pt.exists():
        dest = MODELS_DIR / "rover_yolo_tuned.pt"
        shutil.copy2(str(best_pt), str(dest))
        print(f"📦 Updated tuned weights: {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"⚠️  Could not locate best.pt in {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Real-to-Sim Domain Adaptation: Capture Webots frames, auto-label semantic classes, and fine-tune YOLO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--capture", action="store_true", help="Force capture of fresh simulation screenshots")
    parser.add_argument("--capture-count", type=int, default=25, help="Number of simulation screenshots to capture")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size")
    parser.add_argument("--conf", type=float, default=0.20, help="YOLO confidence threshold for auto-labeling")
    parser.add_argument("--device", type=str, default=None, help="Force device (mps/cuda/cpu). Auto-detects if omitted.")
    parser.add_argument("--label-only", action="store_true", help="Only auto-label, skip fine-tuning")
    parser.add_argument("--train-only", action="store_true", help="Only fine-tune (skip labeling, assume labels exist)")
    args = parser.parse_args()

    print("=" * 60)
    print("🔬 Real-to-Sim Synthetic Data Pipeline")
    print(f"   Project Root:    {PROJECT_ROOT}")
    print(f"   Raw Screenshots: {RAW_DIR}")
    print(f"   Images Dir:      {IMAGES_DIR}")
    print(f"   Labels Dir:      {LABELS_DIR}")
    print(f"   Target Classes:  {list(SIM_CLASSES.values())}")
    print("=" * 60)

    device = args.device or resolve_device()
    model, device = load_yolo_model(device)

    # ── Step 0: Screenshot Harvesting ──
    raw_images = collect_raw_images()
    if args.capture or len(raw_images) == 0:
        captured = capture_live_screenshots(count=args.capture_count)
        raw_images = collect_raw_images()

    if not args.train_only:
        # ── Step 1: Auto-label images ──
        if not raw_images:
            print(f"\n⚠️  No images found in {RAW_DIR}/ to label.")
            sys.exit(0)
        print(f"\n🖼️  Processing {len(raw_images)} screenshots with semantic mapping...\n")
        labeled = auto_label(model, raw_images, device, conf_threshold=args.conf)

    if args.label_only:
        print("\n✅ Label-only mode complete. Skipping fine-tuning.")
        return

    # ── Step 2: Generate dataset YAML ──
    if not IMAGES_DIR.exists() or not any(IMAGES_DIR.iterdir()):
        print(f"\n❌ No images in {IMAGES_DIR}/. Cannot fine-tune without data.")
        sys.exit(1)

    dataset_yaml = create_dataset_yaml()

    # ── Step 3: 5-Epoch Fine-Tune on Apple Silicon MPS ──
    fine_tune(model, dataset_yaml, device, args.epochs, args.batch)

    print("\n" + "=" * 60)
    print("🏁 Real-to-Sim pipeline complete!")
    print(f"   Updated weights:  {MODELS_DIR / 'rover_yolo_tuned.pt'}")
    print(f"   Restart server:   patrol-rover (or python3 app.py)")
    print("=" * 60)


if __name__ == "__main__":
    main()
