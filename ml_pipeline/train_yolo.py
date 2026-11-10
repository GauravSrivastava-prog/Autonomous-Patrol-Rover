#!/usr/bin/env python3
"""
train_yolo.py — Enterprise YOLOv8 Fine-Tuning Algorithm
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features:
  1. Ultralytics YOLOv8 fine-tuning with custom hyperparameters (AdamW, Cosine LR).
  2. Hardware detection & explicit device pinning (Apple Silicon MPS / NVIDIA CUDA).
  3. Dynamic augmentation injection (Mosaic, MixUp, HSV jitter, perspective shift).
  4. Real-time metric tracking: mAP@50, mAP@50-95, Precision, Recall.
  5. Checkpoint management saving top-performing weights to runs/train/weights/best.pt.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml

import torch

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("YOLOTrainer")

def resolve_compute_device(requested_device: Optional[str] = None) -> str:
    """
    Detects hardware capabilities and returns the optimal compute device.
    Prioritizes Apple Metal Performance Shaders (MPS) on macOS and CUDA on Linux.
    """
    if requested_device:
        logger.info(f"⚙️ Device explicitly requested by user: '{requested_device}'")
        return requested_device

    if torch.backends.mps.is_available():
        logger.info("🍎 Apple Silicon GPU detected — Pinning execution to Metal Performance Shaders (MPS).")
        return "mps"
    elif torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"🚀 NVIDIA GPU detected: {device_name} — Pinning execution to CUDA:0.")
        return "0"
    else:
        logger.warning("⚠️ No hardware accelerator detected. Falling back to multi-core CPU execution.")
        return "cpu"

def load_yaml_config(path: Path) -> dict:
    """Safely loads a YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_training(
    dataset_yaml: Path,
    hyperparams_yaml: Path,
    override_epochs: Optional[int] = None,
    override_batch: Optional[int] = None,
    override_device: Optional[str] = None,
    resume: bool = False
):
    """Executes the Ultralytics YOLOv8 training and validation lifecycle."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("❌ 'ultralytics' library is not installed. Run: pip install ultralytics")
        sys.exit(1)

    # 1. Load configurations
    hp = load_yaml_config(hyperparams_yaml)
    device = resolve_compute_device(override_device)

    base_model_name = hp.get("model", "yolov8n.pt")
    epochs = override_epochs if override_epochs is not None else hp.get("epochs", 50)
    batch = override_batch if override_batch is not None else hp.get("batch", 16)
    imgsz = hp.get("imgsz", 640)

    # Output directory layout
    project_dir = Path("ml_pipeline/runs/train")
    run_name = "rover_yolo_run"

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("🚀 INITIALIZING YOLOV8 REPRODUCIBLE FINE-TUNING REGIME")
    logger.info(f"  • Base Model:       {base_model_name}")
    logger.info(f"  • Dataset Spec:     {dataset_yaml}")
    logger.info(f"  • Target Device:    {device}")
    logger.info(f"  • Epochs:           {epochs}")
    logger.info(f"  • Batch Size:       {batch}")
    logger.info(f"  • Resolution:       {imgsz}x{imgsz}")
    logger.info(f"  • Optimizer:        {hp.get('optimizer', 'AdamW')} (Cosine LR: {hp.get('cos_lr', True)})")
    logger.info(f"  • Augmentations:    Mosaic ({hp.get('mosaic', 1.0)}), MixUp ({hp.get('mixup', 0.15)}), Perspective ({hp.get('perspective', 0.0005)})")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # 2. Instantiate YOLO Model
    model = YOLO(base_model_name)

    # 3. Assemble training arguments
    train_args = {
        "data": str(dataset_yaml.resolve()),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": True,
        "pretrained": hp.get("pretrained", True),
        "optimizer": hp.get("optimizer", "AdamW"),
        "lr0": float(hp.get("lr0", 0.001)),
        "lrf": float(hp.get("lrf", 0.01)),
        "momentum": float(hp.get("momentum", 0.937)),
        "weight_decay": float(hp.get("weight_decay", 0.0005)),
        "warmup_epochs": float(hp.get("warmup_epochs", 3.0)),
        "warmup_momentum": float(hp.get("warmup_momentum", 0.8)),
        "cos_lr": bool(hp.get("cos_lr", True)),
        "box": float(hp.get("box", 7.5)),
        "cls": float(hp.get("cls", 0.5)),
        "dfl": float(hp.get("dfl", 1.5)),
        "hsv_h": float(hp.get("hsv_h", 0.015)),
        "hsv_s": float(hp.get("hsv_s", 0.7)),
        "hsv_v": float(hp.get("hsv_v", 0.4)),
        "degrees": float(hp.get("degrees", 10.0)),
        "translate": float(hp.get("translate", 0.1)),
        "scale": float(hp.get("scale", 0.5)),
        "shear": float(hp.get("shear", 2.0)),
        "perspective": float(hp.get("perspective", 0.0005)),
        "flipud": float(hp.get("flipud", 0.0)),
        "fliplr": float(hp.get("fliplr", 0.5)),
        "mosaic": float(hp.get("mosaic", 1.0)),
        "mixup": float(hp.get("mixup", 0.15)),
        "copy_paste": float(hp.get("copy_paste", 0.10)),
        "erasing": float(hp.get("erasing", 0.20)),
        "patience": int(hp.get("patience", 12)),
        "save": bool(hp.get("save", True)),
        "seed": int(hp.get("seed", 42)),
        "deterministic": bool(hp.get("deterministic", True)),
        "verbose": True
    }

    if resume:
        train_args["resume"] = True

    # 4. Execute Fine-Tuning Loop
    logger.info("🏋️ Training loop starting...")
    results = model.train(**train_args)

    # 5. Evaluate on Validation / Test Set & Extract Key Metrics
    logger.info("📊 Running rigorous post-training evaluation...")
    metrics = model.val(data=str(dataset_yaml.resolve()), split="val", device=device)

    mp = metrics.box.mp             # Mean Precision
    mr = metrics.box.mr             # Mean Recall
    map50 = metrics.box.map50       # mAP @ IoU 0.50
    map50_95 = metrics.box.map      # mAP @ IoU 0.50:0.95

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("🏆 FINAL VALIDATION PERFORMANCE DASHBOARD:")
    logger.info(f"  • Precision (Mean):  {mp * 100:.2f}%")
    logger.info(f"  • Recall (Mean):     {mr * 100:.2f}%")
    logger.info(f"  • mAP @ 50:          {map50 * 100:.2f}%")
    logger.info(f"  • mAP @ 50-95:       {map50_95 * 100:.2f}%")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    best_weights_path = project_dir / run_name / "weights" / "best.pt"
    if best_weights_path.exists():
        logger.info(f"💾 Best checkpoint saved to: {best_weights_path.resolve()}")
    else:
        logger.warning(f"⚠️ Checkpoint file not found at expected path: {best_weights_path}")

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Ultralytics YOLOv8 Fine-Tuning for AI Patrol Rover")
    parser.add_argument(
        "--data",
        type=str,
        default="ml_pipeline/configs/dataset.yaml",
        help="Path to dataset YAML file"
    )
    parser.add_argument(
        "--hyperparams",
        type=str,
        default="ml_pipeline/configs/yolov8_hyperparams.yaml",
        help="Path to hyperparameters YAML file"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override mini-batch size")
    parser.add_argument("--device", type=str, default=None, help="Pin compute device ('mps', '0', 'cpu')")
    parser.add_argument("--resume", action="store_true", help="Resume from last training checkpoint")
    args = parser.parse_args()

    data_path = Path(args.data)
    hp_path = Path(args.hyperparams)

    if not data_path.exists():
        logger.error(f"❌ Dataset specification '{data_path}' not found. Run fetch_kaggle_data.py first.")
        sys.exit(1)

    run_training(
        dataset_yaml=data_path,
        hyperparams_yaml=hp_path,
        override_epochs=args.epochs,
        override_batch=args.batch,
        override_device=args.device,
        resume=args.resume
    )

if __name__ == "__main__":
    main()
