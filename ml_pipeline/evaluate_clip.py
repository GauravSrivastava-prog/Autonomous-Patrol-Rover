#!/usr/bin/env python3
"""
evaluate_clip.py — OpenAI CLIP Calibration & Prompt Engineering Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features:
  1. Loads 'openai/clip-vit-base-patch32' for Zero-Shot multimodal calibration.
  2. Multi-template Prompt Engineering matrix ensemble for robust open-vocabulary matching.
  3. Evaluates Zero-Shot Top-1 & Top-K accuracy on rover perception test sets.
  4. Generates a pre-computed Semantic Cache (.json / .pkl) to drop runtime inference
     latency to <0.2ms (sub-millisecond dot-product lookup without re-encoding text).
  5. Deploys the semantic cache directly into the application backend.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import json
import time
import pickle
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import numpy as np
from PIL import Image

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("CLIPCalibration")

# Standard target taxonomy for AI Patrol Rover
TARGET_CLASSES = [
    "blue_chair",
    "green_plant",
    "red_cone",
    "wooden_crate",
    "hazard_box",
    "obstacle"
]

# Prompt Engineering Matrix for Robotics / Indoor Navigation
PROMPT_TEMPLATES = [
    "a photo of a {}",
    "a {} in an indoor environment",
    "a cropped view of a {}",
    "a low-angle view of a {} from a robot camera",
    "a rover camera image of a {}",
    "an indoor object: {}",
    "a realistic photograph of a {}"
]

def resolve_device(requested_device: Optional[str] = None) -> str:
    """Detects available acceleration hardware for CLIP."""
    if requested_device:
        return requested_device
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def build_ensemble_text_embeddings(
    classes: List[str],
    templates: List[str],
    model,
    processor,
    device: str
) -> Dict[str, np.ndarray]:
    """
    Computes ensemble normalized text embeddings by averaging across
    the prompt engineering matrix for each target class.
    """
    logger.info("🧠 Generating ensemble text embeddings across prompt engineering matrix...")
    class_embeddings = {}

    with torch.no_grad():
        for cls_name in classes:
            # Human-readable label representation (e.g., 'blue chair' instead of 'blue_chair')
            readable_name = cls_name.replace("_", " ")
            prompts = [template.format(readable_name) for template in templates]

            # Tokenize and encode
            inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
            text_features = model.get_text_features(**inputs)
            if hasattr(text_features, "text_embeds") and text_features.text_embeds is not None:
                text_features = text_features.text_embeds
            elif hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
                text_features = text_features.pooler_output
            elif isinstance(text_features, (list, tuple)) and len(text_features) > 0:
                text_features = text_features[0]

            # L2 Normalize individual prompt embeddings
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            # Average ensemble embeddings across all prompt variants
            ensemble_vec = text_features.mean(dim=0)
            # Re-normalize ensemble vector to unit length
            ensemble_vec = ensemble_vec / ensemble_vec.norm(p=2, dim=-1)

            class_embeddings[cls_name] = ensemble_vec.cpu().numpy()

    return class_embeddings

def evaluate_test_dataset(
    test_dir: Path,
    class_embeddings: Dict[str, np.ndarray],
    model,
    processor,
    device: str
) -> Dict[str, float]:
    """
    Evaluates CLIP zero-shot classification performance on test set images.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    test_images = [p for p in test_dir.glob("*") if p.suffix.lower() in valid_exts]

    if not test_images:
        logger.warning(f"⚠️ No test images found in {test_dir}. Skipping dataset evaluation.")
        return {"top1_accuracy": 0.0, "samples_evaluated": 0}

    logger.info(f"📊 Evaluating Zero-Shot accuracy across {len(test_images)} test samples...")
    class_names = list(class_embeddings.keys())
    # Matrix of shape (num_classes, embedding_dim)
    emb_matrix = np.stack([class_embeddings[k] for k in class_names], axis=0)

    correct_top1 = 0
    correct_top2 = 0
    evaluated = 0

    with torch.no_grad():
        for img_path in test_images:
            try:
                img = Image.open(img_path).convert("RGB")
                inputs = processor(images=img, return_tensors="pt").to(device)
                img_features = model.get_image_features(**inputs)
                if hasattr(img_features, "image_embeds") and img_features.image_embeds is not None:
                    img_features = img_features.image_embeds
                elif hasattr(img_features, "pooler_output") and img_features.pooler_output is not None:
                    img_features = img_features.pooler_output
                elif isinstance(img_features, (list, tuple)) and len(img_features) > 0:
                    img_features = img_features[0]
                img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
                img_vec = img_features.cpu().numpy().squeeze(0)  # (embedding_dim,)

                # Cosine similarity dot product: (num_classes,)
                sims = np.dot(emb_matrix, img_vec)
                ranked_indices = np.argsort(sims)[::-1]
                pred_class = class_names[ranked_indices[0]]

                # Match against ground truth encoded in filename or label
                evaluated += 1
                for idx, c in enumerate(class_names):
                    if c in img_path.name.lower():
                        if ranked_indices[0] == idx:
                            correct_top1 += 1
                        if idx in ranked_indices[:2]:
                            correct_top2 += 1
                        break
            except Exception as e:
                logger.debug(f"Error evaluating {img_path}: {e}")

    top1_acc = (correct_top1 / evaluated) * 100 if evaluated > 0 else 0.0
    top2_acc = (correct_top2 / evaluated) * 100 if evaluated > 0 else 0.0

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("🎯 CLIP ZERO-SHOT EVALUATION PERFORMANCE:")
    logger.info(f"  • Samples Evaluated:  {evaluated}")
    logger.info(f"  • Top-1 Accuracy:     {top1_acc:.2f}%")
    logger.info(f"  • Top-2 Accuracy:     {top2_acc:.2f}%")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return {
        "top1_accuracy": round(top1_acc, 2),
        "top2_accuracy": round(top2_acc, 2),
        "samples_evaluated": evaluated
    }

def benchmark_cache_lookup(class_embeddings: Dict[str, np.ndarray]) -> float:
    """Demonstrates sub-millisecond dot-product retrieval using pre-computed cache."""
    class_names = list(class_embeddings.keys())
    emb_matrix = np.stack([class_embeddings[k] for k in class_names], axis=0)
    dim = emb_matrix.shape[1]

    # Synthetic image feature vector (unit normalized)
    dummy_vec = np.random.randn(dim).astype(np.float32)
    dummy_vec /= np.linalg.norm(dummy_vec)

    runs = 1000
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = np.dot(emb_matrix, dummy_vec)
    total_time = time.perf_counter() - t0

    latency_us = (total_time / runs) * 1_000_000
    latency_ms = (total_time / runs) * 1_000
    logger.info(f"⚡ Pre-computed Semantic Cache Dot-Product Latency: {latency_us:.1f} µs ({latency_ms:.4f} ms)")
    return latency_ms

def save_and_deploy_semantic_cache(
    class_embeddings: Dict[str, np.ndarray],
    eval_metrics: Dict[str, float],
    model_name: str,
    output_cache_path: Path,
    backend_dest_path: Path
):
    """Serializes semantic cache dictionary to JSON & PKL and copies to backend."""
    # Convert numpy arrays to float lists for standard JSON serialization
    json_cache = {
        "metadata": {
            "model": model_name,
            "classes": list(class_embeddings.keys()),
            "templates_count": len(PROMPT_TEMPLATES),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "evaluation": eval_metrics
        },
        "templates": PROMPT_TEMPLATES,
        "embeddings": {k: v.tolist() for k, v in class_embeddings.items()}
    }

    # 1. Save JSON representation
    output_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_cache_path, "w", encoding="utf-8") as f:
        json.dump(json_cache, f, indent=2)
    logger.info(f"💾 Semantic cache saved to: {output_cache_path}")

    # 2. Save binary Pickle representation for zero-deserialization-overhead Python runtime
    pkl_path = output_cache_path.with_suffix(".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(class_embeddings, f)
    logger.info(f"💾 Binary semantic cache saved to: {pkl_path}")

    # 3. Deploy to rover backend directory
    backend_dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_cache_path, backend_dest_path)
    shutil.copy2(pkl_path, backend_dest_path.with_suffix(".pkl"))
    logger.info(f"🚚 Semantic cache successfully deployed to rover backend: {backend_dest_path}")

def main():
    parser = argparse.ArgumentParser(description="OpenAI CLIP Calibration & Semantic Cache Generator")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model identifier"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="ml_pipeline/datasets/rover_indoor_objects/images/test",
        help="Directory containing test images for zero-shot evaluation"
    )
    parser.add_argument(
        "--cache-out",
        type=str,
        default="ml_pipeline/models/clip_semantic_cache.json",
        help="Output path for semantic cache JSON"
    )
    parser.add_argument(
        "--backend-dest",
        type=str,
        default="models/clip_semantic_cache.json",
        help="Destination path in backend"
    )
    parser.add_argument("--device", type=str, default=None, help="Device ('mps', 'cuda', 'cpu')")
    args = parser.parse_args()

    device = resolve_device(args.device)
    logger.info(f"🚀 Initializing CLIP Calibration on device: {device.upper()}...")

    try:
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        logger.error("❌ 'transformers' library not installed. Run: pip install transformers")
        sys.exit(1)

    logger.info(f"📦 Loading pre-trained model: '{args.model}'...")
    model = CLIPModel.from_pretrained(args.model).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)
    model.eval()

    # Step 1: Prompt Engineering Ensemble Embeddings
    class_embeddings = build_ensemble_text_embeddings(
        classes=TARGET_CLASSES,
        templates=PROMPT_TEMPLATES,
        model=model,
        processor=processor,
        device=device
    )

    # Step 2: Evaluation on Test Dataset
    test_path = Path(args.test_dir)
    eval_metrics = {}
    if test_path.exists():
        eval_metrics = evaluate_test_dataset(test_path, class_embeddings, model, processor, device)
    else:
        logger.info(f"ℹ️ Test dataset path '{test_path}' does not exist yet. Emitting cache with baseline metrics.")
        eval_metrics = {"top1_accuracy": 0.0, "status": "pre_dataset_calibration"}

    # Step 3: Benchmark Runtime Cache Latency
    benchmark_cache_lookup(class_embeddings)

    # Step 4: Serialize and Deploy Cache
    save_and_deploy_semantic_cache(
        class_embeddings=class_embeddings,
        eval_metrics=eval_metrics,
        model_name=args.model,
        output_cache_path=Path(args.cache_out),
        backend_dest_path=Path(args.backend_dest)
    )

    logger.info("✨ CLIP Multimodal Calibration & Semantic Cache Pipeline Completed!")

if __name__ == "__main__":
    main()
