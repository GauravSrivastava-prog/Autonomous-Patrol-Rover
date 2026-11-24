# Autonomous Patrol Rover — Complete Website Content Export

**Source File:** `docs/index.html`  
**Total Sections:** 8 Chronological Sections + Floating Global Navigation Controls + Interactive 4K Lightbox Modal  
**Extraction Standard:** 100% Exhaustive & Verbatim (Zero Summarization, Zero Omissions, Full State-Gated Coverage)

---

## Global Navigation Controls: Replay Laser Intro Button

* **Section Context:** Fixed floating control button situated at the top-right viewport corner (`fixed top-4 right-4 z-50`). Active on WebGL2-compatible browsers when reduced motion is not requested. Triggers an instantaneous replay of the falling hero laser beam animation.
* **Verbatim Content Tree:**
  * **Element Type:** HTML `<button type="button">`
  * **Tooltip (`title` attribute):** `Replay Laser Intro`
  * **Accessibility Label (`aria-label` attribute):** `Replay animation`
  * **Visual Icon:** SVG 24×24 circular loop / reload icon path:
    ```
    M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8
    M3 3v5h5
    M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16
    M16 21h5v-5
    ```

---

## Section 1: Hero Display & Mission Manifesto

* **Section Context:** Full-bleed WebGL2 hero header container (`.laser-hero` / `.hero-header-group`) featuring animated character-splitting display typography (`<SplitText />`), a soft dark radial vignette backdrop (`.hero-header-backdrop`), and the project's foundational architectural mission statement.
* **Verbatim Content Tree:**
  * **Main Display Heading (`<h1>` via `<SplitText />`):**
    `Autonomous Patrol Rover`
    * *Animation & Physics Specifications:* Staggered character entrance (`splitType="chars"`, 21 character spans, `delay={15}ms`, `duration={1.25}s`, `ease="power3.out"`, accent color flash `#60a5fa`, interactive spring-physics lift hover `STIFFNESS=180`, `DAMPING=19`).
  * **Subheading (`<p class="hero-subheading">`):**
    > A custom 3D robotics simulation running real-time AI vision entirely on local hardware. We engineered an edge platform using YOLOv8 and OpenAI CLIP to navigate, detect hazards, and patrol complex environments without cloud latency.
    * **Underlined Technical Anchor Highlights:**
      1. `real-time AI vision` (styled: `font-semibold text-white underline decoration-[#93c5fd]/60 underline-offset-4 decoration-1`)
      2. `local hardware` (styled: `font-semibold text-white underline decoration-[#93c5fd]/60 underline-offset-4 decoration-1`)
      3. `YOLOv8` (styled: `font-semibold text-white underline decoration-[#93c5fd]/60 underline-offset-4 decoration-1`)
      4. `OpenAI CLIP` (styled: `font-semibold text-white underline decoration-[#93c5fd]/60 underline-offset-4 decoration-1`)

---

## Section 2: Stage 0 — Sovereign Edge Terminal Dashboard (`#card-hero`)

* **Section Context:** Centered hero terminal card (`#card-hero`) wrapped in a macOS-styled window shell with three-dot traffic lights, host user prompt, an exact boxed ASCII execution dashboard, an edge architecture summary paragraph, and the initial interactive progression prompt (`PromptRow 0`).
* **Verbatim Content Tree:**
  * **macOS Title Bar:**
    * **Traffic Light Indicators:** Red (`#ff5f56`), Yellow (`#ffbd2e`), Green (`#27c93f`)
    * **Window Title:** `gauravsrivastava1212@Gauravs-MacBook-Pro ~ % patrol-rover`
    * **Status Pill (Right):** `MPS 60FPS`
  * **Host Terminal Shell Prompt:**
    * **User & Host:** `gauravsrivastava1212@Gauravs-MacBook-Pro`
    * **Path:** `~ %`
    * **Command:** `patrol-rover`
  * **Boxed CLI Dashboard Execution Logs (Verbatim):**
    ```text
╭──────────────────────────────────────────────────────────────────────────────╮
│                                                                              │
│       ___   __  __ ______  ___   _  __ ___   __  __ ___   __  __  ____       │
│      / _ | / / / //_  __/ / _ \ / |/ // _ \ /  |/ // _ \ / / / / / __/       │
│     / __ |/ /_/ /  / /   / // //    // // // /|_/ // // / /_/ / _\ \         │
│    /_/ |_|\____/  /_/    \___//_/|_/ \___//_/  /_/ \___/ \____/ /___/        │
│                                                                              │
│          P A T R O L   R O V E R   |   S O V E R E I G N   E D G E           │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ ==> Auto-updating Homebrew...                                                │
│ ==> Auto-updated Homebrew!                                                   │
│ Updated 1 tap (homebrew/core).                                               │
│                                                                              │
│ ==> Caveats                                                                  │
│ Autonomous Patrol Rover runs 100% locally on sovereign edge silicon.         │
│ Zero cloud virtualization. Zero 429 rate limits. 60 FPS rigid-body sync.     │
│                                                                              │
│ ==> Checking host silicon accelerator...                                     │
│ ✔ Detected host silicon engine: Apple Metal Performance Shaders (MPS)        │
│                                                                              │
│ ==> Pouring OpenAI CLIP Semantic Cache... 🍺                                 │
│ ✔ Loaded 6 class zero-shot embeddings from ./models (100% verified)          │
│                                                                              │
│ ==> Pouring YOLOv8 Tensor Core weights... 🍺                                 │
│ ✔ Loaded tuned neural weights: rover_yolo_tuned.pt (6.18 MB, MPS)            │
│                                                                              │
│ ==> Initializing 60 FPS Rigid-Body Physics & Zero-Latency IPC Engine... 🚀   │
│ ✔ Local WebSocket telemetry bridge bound to ws://127.0.0.1:7860/ws           │
│ ✔ GTA 3rd-Person Chase Engine calibrated (1280x720 HD @ 30 FPS)              │
│ ✔ Solid PBR Laboratory Arena & Military Gunmetal Chassis initialized         │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  🚀 [MISSION] GET / ✔ 200 OK                                                 │
│  🧭 [MODE] WS SET_MODE MANUAL ✔ 200 OK                                       │
│  🧭 [MODE] POST /set_mode/MANUAL ✔ 200 OK                                    │
│  🧭 [MODE] WS SET_MODE MANUAL ✔ 200 OK                                       │
│  🕹️ [TELEOP] WS MANUAL BACKWARD ✔ 200 OK                                     │
╰──────────────────────────────────────────────────────────────────────────────╯
    ```
  * **Human Architectural Summary Paragraph:**
    > Autonomous robotics requires deterministic 60 FPS physics and sub-millisecond IPC loops. Instead of offloading video feeds to high-latency cloud servers, our architecture runs dual neural networks directly on Apple Silicon MPS and NVIDIA CUDA cores at zero latency.
  * **Stage 0 Interactive Prompt (`PromptRow 0`):**
    * **Prompt Prefix:** `root@rover:~#`
    * **Prompt Text:** `Inspect YOLOv8 Spatial Cortex & Metrics? [Y/N]:`
    * **Interactive Buttons (When Active):**
      * Button 1: `[ Y ] Proceed` (Tooltip: `Press Y to proceed`)
      * Button 2: `[ N ] Skip` (Tooltip: `Press N to skip`)
      * Keyboard Shortcut Hint: `[ Press Y / N ]`
    * **Logged Choice Output (When Answered):** ` Y` or ` N` (appended to prompt line)

---

## Section 3: Stage 1 — Model Performance & Telemetry (`#section-telemetry`)

* **Section Context:** Conditionally rendered upon Stage 0 progression (`stage >= 1`). Contains section header, external Google Colab fine-tuning link, a 4-stat quantitative metrics grid, two 4K native visual showcase frames with lightbox zoom triggers, the Stage 1 YOLO inference terminal, and `PromptRow 1`.
* **Verbatim Content Tree:**
  * **Section Header:**
    * **Title (`<h2>`):** `Model Performance & Telemetry`
    * **Description:** `Hardware-accelerated edge inference benchmarks on Apple Silicon MPS and NVIDIA T4 Tensor Cores.`
    * **External Action Button:**
      * **Button Label:** `Run Fine-Tuning in Google Colab`
      * **Target URL:** `https://colab.research.google.com/drive/1gTfE_N5gQjZ3B6Lz3rC9eB5w6FzXvQp?usp=sharing`
      * **Target Behavior:** Opens in new window (`target="_blank" rel="noopener noreferrer"`)
      * **Indicator Pill:** Glowing cyan animated ping dot
      * **Icon:** SVG external link icon
  * **4-Stat Quantitative Metric Grid:**
    1. **Metric Card 1:**
       * **Label (Mono):** `mAP@0.5 (All Hazards)`
       * **Primary Value:** `94.2%` (color: `#00ff9d`)
       * **Sub-Caption:** `+18.4% over baseline COCO`
    2. **Metric Card 2:**
       * **Label (Mono):** `Edge MPS Latency`
       * **Primary Value:** `3.2 ms` (color: cyan-400)
       * **Sub-Caption:** `312 FPS peak throughput`
    3. **Metric Card 3:**
       * **Label (Mono):** `Precision (Class Peak)`
       * **Primary Value:** `91.8%` (color: sky-400)
       * **Sub-Caption:** `Low false-positive rate`
    4. **Metric Card 4:**
       * **Label (Mono):** `Recall (Class Peak)`
       * **Primary Value:** `88.4%` (color: purple-400)
       * **Sub-Caption:** `Zero missed critical hazards`
  * **4K Visual Showcase Frames:**
    * **Showcase Frame 1 (`MediaBreakout`):**
      * **Frame Title:** `Training Convergence & Precision-Recall Analysis`
      * **Frame Badge:** `PR_curve.png • 2250x1500 4K Native`
      * **Frame Caption:** `High-resolution Precision-Recall curves showing rapid mAP@0.5 convergence across all 6 hazard classes over 29 fine-tuning epochs on NVIDIA T4 GPU.`
      * **Hover Inspection Badge:** `Click to inspect in full resolution` (with magnifying glass SVG icon)
      * **Image Visual Content:** *[Image content omitted: `./media/PR_curve.png`, fallback `./media/yolo_metrics.png`]*
    * **Showcase Frame 2 (`MediaBreakout`):**
      * **Frame Title:** `Live Multi-Object Bounding Boxes & Confidence Tracking`
      * **Frame Badge:** `val_batch0_pred.jpg • 1920x1098 Native`
      * **Frame Caption:** `Validation batch predictions demonstrating tight bounding box regression, 99.5% accuracy on traffic cones and chairs, and real-to-sim spatial generalization.`
      * **Hover Inspection Badge:** `Click to inspect in full resolution` (with magnifying glass SVG icon)
      * **Image Visual Content:** *[Image content omitted: `./media/val_batch0_pred.jpg`, fallback `./media/yolo_vision.png`]*
  * **Stage 1 Terminal Window (`#card-yolo` / `MacWindow`):**
    * **Window Title:** `bash - yolo-inference - 80x24`
    * **Status Badge:** `30 FPS CoreML Active`
    * **Terminal Log Lines (Verbatim):**
      ```text
[MPS] Initializing Metal backend: Apple M3 Pro GPU (18 cores)
[DETECT] Frame #0842: chair (99.4%), traffic_cone (96.8%) in 3.2ms
[SPATIAL] Obstacle vector: [x: +0.42m, y: -0.12m, dist: 1.84m]
      ```
  * **Stage 1 Interactive Prompt (`PromptRow 1`):**
    * **Prompt Prefix:** `root@rover:~#`
    * **Prompt Text:** `Explore Dual-Model Cognitive Architecture? [Y/N]:`
    * **Interactive Buttons (When Active):** `[ Y ] Proceed` | `[ N ] Skip` | `[ Press Y / N ]`
    * **Logged Choice Output (When Answered):** ` Y` or ` N`

---

## Section 4: Stage 2 — How It Works: Reflexes vs. Reasoning (`#section-folder-architecture`)

* **Section Context:** Conditionally rendered upon Stage 1 progression (`stage >= 2`). Presents the Extended Rover View: the dual-model perception and semantic reasoning stack. Features interactive 3D Folder A (`label="Architecture"`), 3 zero-skeleton mini spec-sheet cards, CLIP visual target acquisition showcase, the CLIP reasoning terminal (`#card-clip`), and `PromptRow 2`.
* **Verbatim Content Tree:**
  * **Section Header:**
    * **Title (`<h2>`):** `How It Works: Reflexes vs. Reasoning`
    * **Description:** `The Extended Rover View: how two complementary AI models give the rover eyes to dodge hazards and a brain to understand instructions.`
  * **Interactive Folder Component A (`<Folder label="Architecture" />`):**
    * **Folder Tab Label:** `Architecture`
    * **Card 1 (`PERCEPTION`):**
      * **Category Tag:** `PERCEPTION`
      * **Card Title:** `The Eyes: YOLOv8`
      * **Body Copy:** `Real-time obstacle detection running every frame. Draws tight bounding boxes around physical hazards—chairs, stairs, cones, and manholes—so the rover avoids collisions autonomously.`
      * **Bottom Metric Strip:** `4.9 ms / img • 60 FPS Loop`
    * **Card 2 (`SEMANTICS`):**
      * **Category Tag:** `SEMANTICS`
      * **Card Title:** `The Brain: OpenAI CLIP`
      * **Body Copy:** `Zero-shot vision-language reasoning (ViT-B/32). Connects raw camera crops to plain English prompts like "find the red fire extinguisher" without needing retraining for new targets.`
      * **Bottom Metric Strip:** `2.6 µs Cache • Zero-Shot`
    * **Card 3 (`EDGE RUNTIME`):**
      * **Category Tag:** `EDGE RUNTIME`
      * **Card Title:** `Sovereign Edge Loop`
      * **Body Copy:** `Both models run 100% locally on Apple Silicon MPS / CUDA via a zero-latency FastAPI WebSocket bridge to Webots R2025a—no cloud APIs, no rate limits.`
      * **Bottom Metric Strip:** `0 Cloud Calls • Local IPC`
    * **Interactive Interaction Hint (Below Folder):**
      `● Click to open • Drag cards out • Click any card to zoom`
  * **CLIP Target Acquisition Showcase (`MediaBreakout`):**
    * **Frame Title:** `Zero-Shot Semantic Target Acquisition`
    * **Frame Badge:** `ViT-B/32 Cosine Similarity`
    * **Frame Caption:** `Our embedded CLIP model matching raw camera frames against natural language prompts in under 13 milliseconds without network overhead.`
    * **Lightbox Metadata Trigger:** `CLIP Zero-Shot Semantic Acquisition` | `ViT-B/32 • 12.4ms` | `Live semantic target matching comparing visual crops against arbitrary text queries in embedding space.`
    * **Image Visual Content:** *[Image content omitted: `./media/clip_reasoning.png`]*
  * **Stage 2 Terminal Window (`#card-clip` / `MacWindow`):**
    * **Window Title:** `bash - clip-reasoning - 80x24`
    * **Status Badge:** `CLIP ViT-B/32`
    * **Terminal Log Lines (Verbatim):**
      ```text
[QUERY] Prompt: "red fire extinguisher mounted on concrete wall"
[SIMILARITY] Crop #3 Cosine Score: 0.892 (Threshold: 0.65)
[NAV] Target locked. Issuing waypoints to differential drive.
      ```
  * **Stage 2 Interactive Prompt (`PromptRow 2`):**
    * **Prompt Prefix:** `root@rover:~#`
    * **Prompt Text:** `Examine Dataset & Training Provenance? [Y/N]:`
    * **Interactive Buttons (When Active):** `[ Y ] Proceed` | `[ N ] Skip` | `[ Press Y / N ]`
    * **Logged Choice Output (When Answered):** ` Y` or ` N`

---

## Section 5: Stage 3 — Dataset & Training Provenance (`#section-training-provenance`)

* **Section Context:** Conditionally rendered upon Stage 2 progression (`stage >= 3`). Documents dataset origins, micro-split augmentations, and PyTorch AMP export pipelines. Features interactive 3D Folder B (`label="Training Data"`), 3 zero-skeleton mini spec-sheet cards, the dataset verification pipeline terminal (`#card-pipeline`), and `PromptRow 3`.
* **Verbatim Content Tree:**
  * **Section Header:**
    * **Title (`<h2>`):** `Dataset & Training Provenance`
    * **Description:** `Curated synthetic hazard dataset and edge optimization pipeline.`
  * **Interactive Folder Component B (`<Folder label="Training Data" />`):**
    * **Folder Tab Label:** `Training Data`
    * **Card 1 (`DATA SOURCE`):**
      * **Category Tag:** `DATA SOURCE`
      * **Card Title:** `Fused Kaggle Datasets`
      * **Body Copy:** `Combined Kaggle's Office Object Detection and Obstacle Detection datasets, normalized into a unified 25-class YOLO format covering indoor and outdoor hazards.`
      * **Bottom Metric Strip:** `25 Classes • 80/20 Split`
    * **Card 2 (`PREPROCESSING`):**
      * **Category Tag:** `PREPROCESSING`
      * **Card Title:** `Real-to-Sim Augmentation`
      * **Body Copy:** `Compressed into a curated 800-image micro-dataset and trained with Mosaic, Mixup, perspective shifts, and HSV color jitter to bridge real-world photos to 3D simulation textures.`
      * **Bottom Metric Strip:** `800 Images • Mosaic + Mixup`
    * **Card 3 (`DEPLOYMENT`):**
      * **Category Tag:** `DEPLOYMENT`
      * **Card Title:** `T4 Training → M3 Export`
      * **Body Copy:** `Fine-tuned over 29 epochs on an NVIDIA T4 GPU in Google Colab (AdamW, cosine LR, early stopping), then exported to FP16 ONNX & PyTorch weights for local M3 Pro execution.`
      * **Bottom Metric Strip:** `0.594 mAP@50 • 6.18 MB FP16`
    * **Interactive Interaction Hint (Below Folder):**
      `● Click to open • Drag cards out • Click any card to zoom`
  * **Stage 3 Terminal Window (`#card-pipeline` / `MacWindow`):**
    * **Window Title:** `Terminal - dataset-pipeline - 80x24`
    * **Status Badge:** `Pipeline Verified`
    * **Terminal Log Lines (Verbatim):**
      ```text
[1/4] Scanning local asset cache: ./data/synthetic_raw/       ✔ 1,240 frames
[2/4] Domain randomization: HSV jitter, Mosaic 4x, Blur       ✔ augmented
[3/4] Exporting ONNX / CoreML engine: FP16 precision           ✔ mps_graph.mlpackage
[4/4] Hardware IPC validation: Shared memory buffer @ 60Hz    ✔ 0.12ms loop
      ```
  * **Stage 3 Interactive Prompt (`PromptRow 3`):**
    * **Prompt Prefix:** `root@rover:~#`
    * **Prompt Text:** `Launch Webots 3D Physics Simulation? [Y/N]:`
    * **Interactive Buttons (When Active):** `[ Y ] Proceed` | `[ N ] Skip` | `[ Press Y / N ]`
    * **Logged Choice Output (When Answered):** ` Y` or ` N`

---

## Section 6: Stage 4 — 3D Simulation & Telemetry Deck (`#section-simulation`)

* **Section Context:** Conditionally rendered upon Stage 3 progression (`stage >= 4`). Showcases the Webots R2025a physics environment, 3D rigid-body kinematics, the GTA-style 3rd-person chase camera video stream, the real-time simulation bridge terminal (`#card-sim`), and `PromptRow 4`.
* **Verbatim Content Tree:**
  * **Simulation Showcase Frame (`MediaBreakout`):**
    * **Frame Title:** `3D Rigid-Body Physics & GTA Chase Cam`
    * **Frame Badge:** `60 FPS MPS`
    * **Frame Caption:** `Real-time rigid-body dynamics computed in Webots with differential drive kinematics, live LiDAR raycasting, and Apple Silicon MPS neural rendering.`
    * **Video Fallback Warning (DOM):** `Your browser does not support the video tag.`
    * **Video Visual Content:** *[Video content omitted: `./media/patrol_mode.mp4`]*
  * **Stage 4 Terminal Window (`#card-sim` / `MacWindow`):**
    * **Window Title:** `bash - webots-sim - 80x24`
    * **Status Badge:** `60.0 FPS Sync`
    * **Terminal Log Lines (Verbatim):**
      ```text
[SIM] Differential drive kinematics: v=0.45 m/s, w=0.18 rad/s
[LIDAR] 360 rays @ 10Hz: Obstacle clearance 1.42m
[HUD] 1280x720 30 FPS stream pushing to /video_feed
      ```
  * **Stage 4 Interactive Prompt (`PromptRow 4`):**
    * **Prompt Prefix:** `root@rover:~#`
    * **Prompt Text:** `Review Local Edge Runbook & Deployment? [Y/N]:`
    * **Interactive Buttons (When Active):** `[ Y ] Proceed` | `[ N ] Skip` | `[ Press Y / N ]`
    * **Logged Choice Output (When Answered):** ` Y` or ` N`

---

## Section 7: Stage 5 — Execution Runbook & Deployment Hub (`#section-runbook`)

* **Section Context:** Conditionally rendered upon Stage 4 progression (`stage >= 5`). Contains the complete developer execution runbook (`#card-runbook`) featuring three launch commands with single-click copy buttons, hardware runtime specs grid, the final repository launch CTA card (`#card-stage-5`), and the global project footer.
* **Verbatim Content Tree:**
  * **Runbook Terminal Window (`#card-runbook` / `MacWindow`):**
    * **Window Title:** `Terminal - local-runbook - 80x24`
    * **Status Badge:** `Edge Runbook`
    * **Execution Option 1 (Zero-Install Launch via `uvx`):**
      * **Header Label:** `Option 1: Zero-Install Launch (Recommended)`
      * **Sub-Badge:** `uvx • Zero Setup`
      * **Shell Prompt Prefix:** `root@rover:~#`
      * **Copy Button:** `COPY` (flashes `COPIED! ✔` upon click)
      * **Command (Verbatim):**
        ```bash
        uvx --from git+https://github.com/GauravSrivastava-prog/Autonomous-Patrol-Rover.git patrol-rover
        ```
      * **Inline Annotation:** `# Automatically binds http://127.0.0.1:7860`
    * **Execution Option 2 (Clone & Global CLI Installation):**
      * **Header Label:** `Option 2: Clone & Global CLI (Run Anywhere)`
      * **Sub-Badge:** `pip install -e .`
      * **Shell Prompt Prefix:** `root@rover:~#`
      * **Copy Button:** `COPY` (flashes `COPIED! ✔` upon click)
      * **Command (Verbatim):**
        ```bash
        git clone https://github.com/GauravSrivastava-prog/Autonomous-Patrol-Rover.git && cd Autonomous-Patrol-Rover && pip install -e . && patrol-rover
        ```
      * **Inline Annotation:** `# Installs global 'patrol-rover' binary • binds 127.0.0.1:7860`
    * **Execution Option 3 (Direct Python Launch):**
      * **Header Label:** `Option 3: Direct Python Launch`
      * **Sub-Badge:** `Python 3.10+`
      * **Shell Prompt Prefix:** `root@rover:~#`
      * **Copy Button:** `COPY` (flashes `COPIED! ✔` upon click)
      * **Command (Verbatim):**
        ```bash
        python3 app.py
        ```
      * **Inline Annotation:** `# Launches FastAPI + Webots bridge on http://127.0.0.1:7860`
    * **Hardware Runtime Benchmarks Strip (4 Blocks):**
      1. Block 1: `Cloud APIs` | `ZERO` (color: `#22c55e`)
      2. Block 2: `YOLOv8 Infer` | `3.2 ms` (color: `#38bdf8`)
      3. Block 3: `CLIP Cosine` | `12.4 ms` (color: `#a855f7`)
      4. Block 4: `Physics Loop` | `60.2 FPS` (color: `#eab308`)
  * **Repository Launch Card (`#card-stage-5`):**
    * **Main Heading (`<h3>`):** `Ready to run locally.`
    * **Mission Paragraph:**
      > Autonomous robotics requires sub-millisecond IPC loops and predictable frame times. By targeting host Apple Silicon MPS and NVIDIA CUDA cores directly, we eliminate cloud latency and API rate limits. The entire codebase, simulation worlds, and neural weights are open source.
    * **GitHub CTA Button:**
      * **Main Button Text:** `EXPLORE ON GITHUB`
      * **Target URL:** `https://github.com/GauravSrivastava-prog/Autonomous-Patrol-Rover`
      * **Target Behavior:** Opens in new window (`target="_blank" rel="noopener noreferrer"`)
      * **Version Pill (Right):** `v2.4.0`
      * **Icon:** SVG GitHub Octocat logo path

---

## Section 8: Global Project Footer (`#footer-section`)

* **Section Context:** Clean, bottom-of-page copyright and academic institutional attribution bar with direct hyperlink back to the source repository.
* **Verbatim Content Tree:**
  * **Institutional Attribution (Left):**
    `UPES Department of Computer Science & Robotics`
  * **Repository Hyperlink (Right):**
    * **Visible Link Text:** `github.com/GauravSrivastava-prog/Autonomous-Patrol-Rover`
    * **Target URL (`href`):** `https://github.com/GauravSrivastava-prog/Autonomous-Patrol-Rover`
    * **Link Behavior:** Opens in new window (`target="_blank" rel="noopener noreferrer"`)

---

## Section 9: Interactive 4K Lightbox Modal (`InteractiveLightbox`)

* **Section Context:** Global full-viewport modal overlay (`#lightbox-modal`, `fixed inset-0 z-[9999]`) activated whenever a user clicks any 4K showcase image. Features custom keyboard shortcuts (`Escape`, `+`, `-`, `0`), click-to-toggle 1.0x/2.5x zoom, mouse dragging/panning, zoom percentage badge, and dedicated controls.
* **Verbatim Content Tree:**
  * **Modal Header Bar Controls:**
    * **Traffic Light Close/Zoom Triggers:**
      * Red dot: Closes lightbox modal
      * Yellow dot: Resets zoom to `1.0x` and pan to `{x: 0, y: 0}`
      * Green dot: Expands zoom to `2.5x`
    * **Header Image Title:** Dynamically reflects clicked image's title:
      1. `Training Convergence & Precision-Recall Analysis`
      2. `Live Multi-Object Bounding Boxes & Confidence Tracking`
      3. `CLIP Zero-Shot Semantic Acquisition`
    * **Header Badge:** Dynamically reflects clicked image's badge:
      1. `PR_curve.png • 2250x1500 4K Native`
      2. `val_batch0_pred.jpg • 1920x1098 Native`
      3. `ViT-B/32 • 12.4ms`
    * **Zoom Toolset:**
      * **Zoom Out Button:** `−` (Tooltip: `Zoom Out (-)`, disabled when `zoom <= 1.0`)
      * **Reset Zoom Button:** Displays current percentage, e.g., `100%` or `250%` (Tooltip: `Reset Zoom (0)`)
      * **Zoom In Button:** `+` (Tooltip: `Zoom In (+)`, disabled when `zoom >= 3.5`)
    * **Modal Close Button:** `✕` (Tooltip: `Close Lightbox (ESC)`)
  * **Main Viewport Area:**
    * *[Image visual pixel content omitted]*
  * **Modal Footer Information & Hint Bar:**
    * **Left Caption Display:** Reflects image caption:
      1. `High-resolution Precision-Recall curves showing rapid mAP@0.5 convergence across all 6 hazard classes over 29 fine-tuning epochs on NVIDIA T4 GPU.`
      2. `Validation batch predictions demonstrating tight bounding box regression, 99.5% accuracy on traffic cones and chairs, and real-to-sim spatial generalization.`
      3. `Live semantic target matching comparing visual crops against arbitrary text queries in embedding space.`
    * **Right Interaction Hints (Mono):**
      * `Click to Zoom 2.5x` (or `Click to Reset` when zoomed)
      * `•`
      * `Drag to Pan`
      * `•`
      * `[ESC] to Close`

---

## Verbatim String Inventory & Checklist

| Section / Element | Exact Extracted String | Status |
| :--- | :--- | :--- |
| **Replay Button** | `Replay animation` / `Replay Laser Intro` | **VERIFIED** |
| **Hero Heading** | `Autonomous Patrol Rover` | **VERIFIED** |
| **Hero Subheading** | `A custom 3D robotics simulation running real-time AI vision entirely on local hardware...` | **VERIFIED** |
| **Hero CLI Title** | `gauravsrivastava1212@Gauravs-MacBook-Pro ~ % patrol-rover` | **VERIFIED** |
| **Hero CLI Pill** | `MPS 60FPS` | **VERIFIED** |
| **Hero Box Title** | `P A T R O L   R O V E R   \|   S O V E R E I G N   E D G E` | **VERIFIED** |
| **Stage 0 Prompt** | `Inspect YOLOv8 Spatial Cortex & Metrics? [Y/N]:` | **VERIFIED** |
| **Telemetry Heading** | `Model Performance & Telemetry` | **VERIFIED** |
| **Telemetry Colab Link**| `Run Fine-Tuning in Google Colab` (`https://colab.research.google.com/...`) | **VERIFIED** |
| **Stat 1** | `mAP@0.5 (All Hazards)` / `94.2%` / `+18.4% over baseline COCO` | **VERIFIED** |
| **Stat 2** | `Edge MPS Latency` / `3.2 ms` / `312 FPS peak throughput` | **VERIFIED** |
| **Stat 3** | `Precision (Class Peak)` / `91.8%` / `Low false-positive rate` | **VERIFIED** |
| **Stat 4** | `Recall (Class Peak)` / `88.4%` / `Zero missed critical hazards` | **VERIFIED** |
| **Stage 1 Prompt** | `Explore Dual-Model Cognitive Architecture? [Y/N]:` | **VERIFIED** |
| **Folder 1 Card 1** | `PERCEPTION` / `The Eyes: YOLOv8` / `4.9 ms / img • 60 FPS Loop` | **VERIFIED** |
| **Folder 1 Card 2** | `SEMANTICS` / `The Brain: OpenAI CLIP` / `2.6 µs Cache • Zero-Shot` | **VERIFIED** |
| **Folder 1 Card 3** | `EDGE RUNTIME` / `Sovereign Edge Loop` / `0 Cloud Calls • Local IPC` | **VERIFIED** |
| **Folder 1 Hint** | `Click to open • Drag cards out • Click any card to zoom` | **VERIFIED** |
| **Stage 2 Prompt** | `Examine Dataset & Training Provenance? [Y/N]:` | **VERIFIED** |
| **Folder 2 Card 1** | `DATA SOURCE` / `Fused Kaggle Datasets` / `25 Classes • 80/20 Split` | **VERIFIED** |
| **Folder 2 Card 2** | `PREPROCESSING` / `Real-to-Sim Augmentation` / `800 Images • Mosaic + Mixup` | **VERIFIED** |
| **Folder 2 Card 3** | `DEPLOYMENT` / `T4 Training → M3 Export` / `0.594 mAP@50 • 6.18 MB FP16` | **VERIFIED** |
| **Folder 2 Hint** | `Click to open • Drag cards out • Click any card to zoom` | **VERIFIED** |
| **Stage 3 Prompt** | `Launch Webots 3D Physics Simulation? [Y/N]:` | **VERIFIED** |
| **Simulation Title**| `3D Rigid-Body Physics & GTA Chase Cam` / `60 FPS MPS` | **VERIFIED** |
| **Stage 4 Prompt** | `Review Local Edge Runbook & Deployment? [Y/N]:` | **VERIFIED** |
| **Runbook Option 1**| `uvx --from git+https://github.com/GauravSrivastava-prog/Autonomous-Patrol-Rover.git patrol-rover` | **VERIFIED** |
| **Runbook Option 2**| `git clone ... && pip install -e . && patrol-rover` | **VERIFIED** |
| **Runbook Option 3**| `python3 app.py` | **VERIFIED** |
| **Runbook Benchmarks**| `Cloud APIs ZERO` / `YOLOv8 Infer 3.2 ms` / `CLIP Cosine 12.4 ms` / `Physics Loop 60.2 FPS` | **VERIFIED** |
| **CTA Card Heading**| `Ready to run locally.` | **VERIFIED** |
| **CTA GitHub Link** | `EXPLORE ON GITHUB` (`v2.4.0`) → `https://github.com/...` | **VERIFIED** |
| **Footer Left** | `UPES Department of Computer Science & Robotics` | **VERIFIED** |
| **Footer Right** | `github.com/GauravSrivastava-prog/Autonomous-Patrol-Rover` | **VERIFIED** |
| **Lightbox Controls**| `Zoom Out (-)` / `Reset Zoom (0)` / `Zoom In (+)` / `Close Lightbox (ESC)` | **VERIFIED** |
