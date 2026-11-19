# Autonomous Patrol Rover

> **Real-Time Edge AI Vision, 3D Robotics Kinematics & Multimodal Telemetry Platform**  
> *Course Project • Algorithm and Robotics Lab • Department of Computer Science & Robotics, UPES*

[![Interactive Manifesto](https://img.shields.io/badge/Live_Manifesto-Vercel_Production-00f0ff?style=flat-square&logo=vercel)](https://autonomous-patrol-rover.vercel.app)
[![MLOps Training Pipeline](https://img.shields.io/badge/Google_Colab-MLOps_Pipeline-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com/drive/1QgUtht2vBS_Bt9jFmo3osvr-AtRW7B4F?usp=sharing)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://python.org)
[![Webots R2025a](https://img.shields.io/badge/Webots-R2025a-00599C?style=flat-square)](https://cyberbotics.com)
[![PyTorch MPS/CUDA](https://img.shields.io/badge/PyTorch-Apple_MPS_%7C_CUDA-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)

---

## Abstract

The **Autonomous Patrol Rover** is an edge-native robotics platform that combines 3D rigid-body kinematics in Webots with local multimodal neural inference. Operating without cloud compute or external API dependencies, the system pairs high-speed spatial perception (**Ultralytics YOLOv8**) with zero-shot semantic reasoning (**OpenAI CLIP ViT-B/32**) running directly on Apple Silicon Metal Performance Shaders (MPS) and NVIDIA CUDA cores.

A high-throughput **FastAPI / WebSocket IPC bridge (30 Hz)** synchronizes sensor raycasting, monocular RGB feeds, differential drive telemetry, and dynamic waypoints with an interactive web-based Mission Control deck.

---

## Live Evaluator Links

* **Interactive Project Manifesto & Architectural Deck:**  
  [https://autonomous-patrol-rover.vercel.app](https://autonomous-patrol-rover.vercel.app)  
  *Features a full-bleed WebGL2 beam, React Bits SplitText animation, interactive 3D physics runbook, and 4K telemetry lightbox.*

* **Google Colab MLOps Fine-Tuning & Validation Notebook:**  
  [Open in Google Colab](https://colab.research.google.com/drive/1QgUtht2vBS_Bt9jFmo3osvr-AtRW7B4F?usp=sharing)  
  *Contains full training scripts, Kaggle dataset fusion, 29-epoch convergence curves, confusion matrices, and FP16 CoreML/ONNX export pipelines.*

---

## Quickstart: Evaluator CLI Launch

The rover platform includes a unified CLI executable (`patrol-rover`) that starts the core IPC engine, suppresses framework noise, displays an ANSI diagnostics terminal dashboard, and serves the mission deck on `http://127.0.0.1:7860`.

### Option 1: Zero-Install Launch (Recommended via `uvx`)
Execute immediately in an isolated virtual environment without manual cloning or dependency installation:
```bash
uvx --from git+https://github.com/GauravSrivastava-prog/Autonomous-Patrol-Rover.git patrol-rover
```

### Option 2: Clone & Local Editable Installation
```bash
git clone https://github.com/GauravSrivastava-prog/Autonomous-Patrol-Rover.git
cd Autonomous-Patrol-Rover
pip install -e .
patrol-rover
```

### Option 3: Direct Python Launch
```bash
git clone https://github.com/GauravSrivastava-prog/Autonomous-Patrol-Rover.git
cd Autonomous-Patrol-Rover
pip install -r requirements.txt
python3 app.py
```

*The application automatically binds to `http://127.0.0.1:7860` and attempts to launch your system browser.*

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Webots R2025a Simulator                         │
│  - Differential Drive Kinematics (v = 0.45 m/s, w = 0.18 rad/s)        │
│  - 360° LiDAR Sensor Array (10 Hz Raycasting)                          │
│  - 640×480 HD Camera Sensor Node                                       │
│  - Supervisor API (World Pose, Obstacle Translation, Arena Reset)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ IPC Bridge (<extern> Socket)
┌───────────────────────────────────▼────────────────────────────────────┐
│                  FastAPI & Asyncio Core IPC Engine                     │
│  - Asynchronous IPC Event Loop (~30 Hz)                                │
│  - Low-Latency JPEG Streaming Buffer                                   │
│  - WebSocket Telemetry Protocol (60 FPS Teleop & Pose Updates)         │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
    (Perception)    ▼                                ▼    (Reasoning)
┌───────────────────────────────┐        ┌───────────────────────────────┐
│     Spatial Reflex Cortex     │        │    Semantic Target Reasoner   │
│  - Fine-Tuned YOLOv8n         │        │  - OpenAI CLIP ViT-B/32       │
│  - 4.9 ms Inference (MPS)     │        │  - Zero-Shot Natural Language │
│  - 6 Hazardous Obstacle Classes│       │  - 2.6 µs Local Embedding Cache│
└───────────────────────────────┘        └───────────────────────────────┘
```

### Dual-Model Perception Stack
1. **Reflex Cortex (YOLOv8n):** Processes continuous 640×480 video frames at 30+ FPS, regressing 2D bounding boxes and obstacle class probabilities to prevent physical collisions in real time.
2. **Cognitive Reasoner (OpenAI CLIP ViT-B/32):** Computes normalized cosine similarities between candidate camera crops and arbitrary natural language prompts (e.g., *"red fire extinguisher mounted on concrete wall"*). Pre-computed embedding vectors are indexed in a local binary cache, enabling target verification in 2.6 microseconds without network round-trips.

---

## Quantitative Validation & Benchmarks

The model was fine-tuned across 29 epochs on an NVIDIA T4 GPU using a composite dataset merging real-world indoor office obstacles (Kaggle) with synthetic domain-randomized perspectives.

| Metric | Validated Result | Notes / Conditions |
| :--- | :--- | :--- |
| **mAP@50 (All Hazard Classes)** | **0.594** (59.4%) | Evaluated on held-out validation batch |
| **mAP@50-95** | **0.466** (46.6%) | Strict IoU threshold range |
| **YOLOv8 Latency (Apple Silicon MPS)** | **4.9 ms** | FP16 Half-Precision (204 FPS peak) |
| **YOLOv8 Latency (NVIDIA T4 CUDA)** | **3.2 ms** | Tensor Cores enabled (312 FPS peak) |
| **CLIP Semantic Cache Lookup** | **2.6 µs** | Pre-computed ViT-B/32 embedding dot product |
| **Simulation Sync Rate** | **60.2 FPS** | Webots rigid-body kinematics step (16ms) |
| **Cloud API Calls** | **0** | 100% on-device local execution |

### Class-Specific Detection Accuracy
* **Traffic Cones:** `99.5%` precision (tight bounding box regression)
* **Chairs & Office Furniture:** `99.5%` precision
* **Floor Hazards & Manholes:** `89.3%` precision
* **Overall Hazard Recall:** `88.4%` across challenging lighting and occlusion conditions

---

## Repository Structure

```
Autonomous-Patrol-Rover/
├── app.py                     # Entry point: CLI dashboard, FastAPI & Webots launcher
├── core_engine.py             # Multimodal AI perception worker, IPC sockets, and kinematics
├── pyproject.toml             # Packaging configuration for 'patrol-rover' CLI
├── requirements.txt           # Production dependencies
│
├── controllers/               # Webots controller implementations
│   └── mission_control/       # Mission control supervisor & robot driver
│
├── ml_pipeline/               # MLOps pipeline & training utilities
│   ├── train_yolo.py          # YOLOv8 fine-tuning script
│   ├── evaluate_clip.py       # CLIP zero-shot evaluation benchmarks
│   ├── export_model.py        # ONNX / CoreML / TorchScript export pipelines
│   └── configs/               # Hyperparameter and dataset YAML specifications
│
├── models/                    # Model weights and semantic caches
│   ├── rover_yolo_tuned.pt    # Fine-tuned YOLOv8 hazard weights
│   └── clip_semantic_cache.pkl # Pre-computed semantic embedding dictionary
│
├── worlds/                    # Webots 3D simulation environments
│   ├── patrol_arena.wbt       # 4m×4m obstacle patrol arena
│   └── textures/              # Surface textures and hazard decals
│
└── docs/                      # Interactive manifesto web application
    ├── index.html             # React 18, GSAP SplitText, WebGL2 LaserFlow, and CSS Glassmorphism
    ├── media/                 # 4K PR curves, confusion matrices, and demo video
    └── laserflow-reveal.png   # Texture mask for interactive hover lens
```

---

## Verification & Testing

To execute the automated headless verification suite testing WebGL contexts, layout geometry, and responsive viewports:

```bash
# Verify docs/index.html across desktop (1440x900) and mobile (390x844) viewports
python3 scratch/verify_part11.py
```

---

## Academic Context & Attribution

* **Course:** Algorithm and Robotics Lab (Semester 7)  
* **Lead Author:** Gaurav Srivastava ([Gaurav.122467@stu.upes.ac.in](mailto:Gaurav.122467@stu.upes.ac.in))  
* **License:** [MIT License](LICENSE)
