#!/usr/bin/env python3
"""
generate_synthetic_data.py — Real-to-Sim Domain Adaptation Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Watches `synthetic_data/raw_screenshots/` for Webots screenshots,
auto-labels them with the tuned YOLO model, then runs a 5-epoch
fine-tuning loop on MPS/CUDA/CPU to continuously improve sim-domain
perception accuracy.

Usage:
    python3 ml_pipeline/generate_synthetic_data.py
    python3 ml_pipeline/generate_synthetic_data.py --epochs 10 --batch 4
    python3 ml_pipeline/generate_synthetic_data.py --label-only
"""

import os
import sys
import shutil
import argparse
import tempfile
from pathlib import Path
from datetime import datetime

# ── Resolve project root (supports execution from any working directory) ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
SYNTH_DIR = PROJECT_ROOT / "synthetic_data"
RAW_DIR = SYNTH_DIR / "raw_screenshots"
IMAGES_DIR = SYNTH_DIR / "images"
LABELS_DIR = SYNTH_DIR / "labels"

# Ultralytics sandbox-safe config directory
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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


def collect_raw_images() -> list:
    """Scan RAW_DIR for supported image files."""
    if not RAW_DIR.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(
        p for p in RAW_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.stat().st_size > 0
    )
    return images


def auto_label(model, images: list, device: str, conf_threshold: float = 0.25):
    """
    Run YOLO inference on each raw screenshot and write YOLO-format
    annotation .txt files into LABELS_DIR + copy images into IMAGES_DIR.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    labeled_count = 0
    empty_count = 0

    for img_path in images:
        stem = img_path.stem
        # Copy image to training images directory
        dest_img = IMAGES_DIR / f"{stem}.jpg"
        if img_path.suffix.lower() != ".jpg":
            # Convert to JPEG using PIL for consistency
            try:
                from PIL import Image
                im = Image.open(str(img_path)).convert("RGB")
                im.save(str(dest_img), "JPEG", quality=95)
            except ImportError:
                shutil.copy2(str(img_path), str(dest_img))
        else:
            shutil.copy2(str(img_path), str(dest_img))

        # Run inference
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

        # Extract bounding boxes in YOLO normalized format
        label_path = LABELS_DIR / f"{stem}.txt"
        lines = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            img_h, img_w = r.orig_shape
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Convert to YOLO format: class x_center y_center width height (normalized)
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                conf = float(box.conf[0])
                lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        label_path.write_text("\n".join(lines) + "\n" if lines else "")
        if lines:
            labeled_count += 1
            print(f"  ✅ {img_path.name} → {len(lines)} detections (conf≥{conf_threshold})")
        else:
            empty_count += 1
            print(f"  ⬚  {img_path.name} → 0 detections (empty label)")

    print(f"\n📊 Auto-Labeling Summary: {labeled_count} labeled / {empty_count} empty / {len(images)} total")
    return labeled_count


def create_dataset_yaml() -> Path:
    """Generate a temporary dataset.yaml for Ultralytics fine-tuning."""
    # Detect class names from the model or use generic
    yaml_content = f"""# Auto-generated Real-to-Sim synthetic dataset
# Generated: {datetime.now().isoformat()}
path: {SYNTH_DIR}
train: images
val: images

# Class names inherited from the base model
# (Ultralytics auto-resolves class map from the pretrained weights)
nc: 80
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
"""
    yaml_path = SYNTH_DIR / "synthetic_dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"📝 Dataset YAML written to: {yaml_path}")
    return yaml_path


def fine_tune(model, dataset_yaml: Path, device: str, epochs: int, batch: int):
    """Run Ultralytics fine-tuning loop on the auto-labeled synthetic data."""
    output_dir = SYNTH_DIR / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🏋️ Starting {epochs}-epoch fine-tuning loop...")
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
            # Prevent Ultralytics from opening plots
            plots=True,
            verbose=True,
        )
        print(f"\n✅ Fine-tuning complete!")
    except Exception as e:
        print(f"\n❌ Fine-tuning failed: {e}")
        return None

    # Copy best weights back to models/
    best_pt = output_dir / "synth_finetune_latest" / "weights" / "best.pt"
    # Search for the actual output
    for run_dir in sorted(output_dir.iterdir(), reverse=True):
        candidate = run_dir / "weights" / "best.pt"
        if candidate.exists():
            best_pt = candidate
            break

    if best_pt.exists():
        dest = MODELS_DIR / "rover_yolo_tuned.pt"
        shutil.copy2(str(best_pt), str(dest))
        print(f"📦 Updated tuned weights: {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"⚠️  Could not locate best.pt in {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Real-to-Sim Domain Adaptation: Auto-label Webots screenshots and fine-tune YOLO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold for auto-labeling")
    parser.add_argument("--device", type=str, default=None, help="Force device (mps/cuda/cpu). Auto-detects if omitted.")
    parser.add_argument("--label-only", action="store_true", help="Only auto-label, skip fine-tuning")
    parser.add_argument("--train-only", action="store_true", help="Only fine-tune (skip labeling, assume labels exist)")
    args = parser.parse_args()

    print("=" * 60)
    print("🔬 Real-to-Sim Synthetic Data Pipeline")
    print(f"   Project Root:  {PROJECT_ROOT}")
    print(f"   Raw Screenshots: {RAW_DIR}")
    print(f"   Images Dir:      {IMAGES_DIR}")
    print(f"   Labels Dir:      {LABELS_DIR}")
    print("=" * 60)

    device = args.device or resolve_device()
    model, device = load_yolo_model(device)

    if not args.train_only:
        # ── Step 1: Collect raw screenshots ──
        images = collect_raw_images()
        if not images:
            print(f"\n⚠️  No images found in {RAW_DIR}/")
            print(f"   Place Webots screenshots (.jpg/.png) there and re-run.")
            print(f"   Tip: Press Ctrl+Shift+S in Webots to save a screenshot.")
            if not args.label_only:
                print(f"\n   Checking if pre-labeled data exists for training...")
                existing_labels = list(LABELS_DIR.glob("*.txt")) if LABELS_DIR.exists() else []
                if not existing_labels:
                    print(f"   No labels found either. Exiting gracefully.")
                    sys.exit(0)
                else:
                    print(f"   Found {len(existing_labels)} existing labels. Proceeding to fine-tune.")
            else:
                sys.exit(0)
        else:
            print(f"\n🖼️  Found {len(images)} raw screenshots. Starting auto-labeling...\n")
            labeled = auto_label(model, images, device, conf_threshold=args.conf)

    if args.label_only:
        print("\n✅ Label-only mode complete. Skipping fine-tuning.")
        return

    # ── Step 2: Generate dataset YAML ──
    # Verify we have labeled data
    if not IMAGES_DIR.exists() or not any(IMAGES_DIR.iterdir()):
        print(f"\n❌ No images in {IMAGES_DIR}/. Cannot fine-tune without data.")
        sys.exit(1)

    dataset_yaml = create_dataset_yaml()

    # ── Step 3: Fine-tune ──
    fine_tune(model, dataset_yaml, device, args.epochs, args.batch)

    print("\n" + "=" * 60)
    print("🏁 Real-to-Sim pipeline complete!")
    print(f"   Updated weights:  {MODELS_DIR / 'rover_yolo_tuned.pt'}")
    print(f"   Restart server:   patrol-rover (or python3 app.py)")
    print("=" * 60)


if __name__ == "__main__":
    main()
