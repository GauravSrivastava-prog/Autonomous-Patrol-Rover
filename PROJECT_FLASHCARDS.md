# Project Flashcards: Autonomous AI Robot & Web Mission Control

---

## Section 1: Tech Stack

---

### Card 1: Python
- **What it is**: A programming language.
- **Role in this project**: It runs the robot movement logic, camera image processing, machine learning models, and the web server.

---

### Card 2: Webots R2025a
- **What it is**: A 3D robot simulation program.
- **Role in this project**: It creates the 3D space, computes physical movement and collisions, and provides simulated motors and camera data.

---

### Card 3: FastAPI
- **What it is**: A Python library used to build web backends.
- **Role in this project**: It hosts the website pages, sends telemetry numbers, and receives user commands from the browser.

---

### Card 4: Uvicorn
- **What it is**: A server program that runs Python web applications.
- **Role in this project**: It accepts network requests from web browsers and directs them to FastAPI.

---

### Card 5: WebSockets
- **What it is**: A network communication method that stays open between browser and server.
- **Role in this project**: It sends robot positions to the browser 30 times per second and sends user commands back to the robot without delay.

---

### Card 6: OpenCV (`cv2`)
- **What it is**: A computer program library for image and video processing.
- **Role in this project**: It reads raw camera pixel bytes from Webots, converts color formats, finds colored objects, draws boxes, and encodes frames into JPEG pictures.

---

### Card 7: PyTorch (`torch`)
- **What it is**: A machine learning library for mathematical calculations on number grids (tensors).
- **Role in this project**: It executes the neural network layers of the CLIP vision model on the computer CPU.

---

### Card 8: HTML, CSS, JavaScript
- **What they are**: Standard code languages for web browsers.
- **Role in this project**:
  - HTML creates the layout structure.
  - CSS sets colors, sizes, and grids.
  - JavaScript updates telemetry numbers, draws the radar map, and sends clicks to the server.

---

### Card 9: A-Frame and Three.js
- **What they are**: 3D graphics libraries that run inside web browsers.
- **Role in this project**: They render the 3D virtual copy of the arena, robot, and targets directly on the `/vr` web page.

---

## Section 2: Functionality and Features

---

### Card 10: AI Patrol Mode (`AI_PATROL`)
- **What it is**: An automated driving mode.
- **How it works**: The robot moves forward in the arena. When distance sensor numbers rise above a threshold, the robot turns away from the obstacle.

---

### Card 11: AI Chase Mode (`AI_CHASE`)
- **What it is**: An automated target-following mode.
- **How it works**: The robot scans camera images for red, green, or blue colors. It steers left or right to center the target in the frame, moves forward when far, and reverses if closer than 15 centimeters.

---

### Card 12: Manual Mode (`MANUAL`)
- **What it is**: A user control mode.
- **How it works**: The autonomous driving stops. The user presses W, A, S, D on the keyboard or clicks browser buttons to send move and stop commands.

---

### Card 13: Teleport Reset
- **What it is**: A function that moves the robot back to the starting point.
- **How it works**: The Python script calls the Webots Supervisor interface to rewrite the robot coordinate values directly to (0, 0, 0.06).

---

### Card 14: Live Camera Stream
- **What it is**: Continuous transmission of video frames to the browser.
- **How it works**: The robot camera captures 640x480 pixel arrays. OpenCV writes text and shapes on them, compresses them to JPEG format, and sends them via an HTTP video stream.

---

### Card 15: 2D Radar Map
- **What it is**: A top-down graphical display on the web page.
- **How it works**: An HTML canvas draws arena boundary lines, dots for target coordinates, a moving dot for robot position, a line showing robot heading angle, and a trail of past coordinates.

---

### Card 16: 3D Digital Twin (`/vr`)
- **What it is**: A 3D browser representation that copies the simulation.
- **How it works**: The `/vr` web page creates 3D shapes for the floor, walls, objects, and robot. Every time a WebSocket message arrives with new coordinates, JavaScript moves the 3D shapes to match.

---

### Card 17: 3D Drag-and-Drop Teleportation
- **What it is**: Moving virtual objects by clicking and dragging in the web browser.
- **How it works**: When a user clicks and drags a 3D target on the `/vr` page, the browser calculates the floor coordinates and sends them over WebSocket. Webots updates the target position and resets its physics.

---

### Card 18: Semantic Scene Safety Throttling
- **What it is**: Automatic speed reduction based on image meaning.
- **How it works**: A vision model evaluates the camera frame. If the category is an obstacle or plant with score above 0.65, motor speeds are reduced by 50 percent.

---

## Section 3: File Structure and Main Files

---

### Card 19: File Structure Overview
- **What it is**: The organization of files in the project folder.
- **Folder layout**:
  - `worlds/` &rarr; holds 3D scene files.
  - `controllers/` &rarr; holds the robot code.
  - `.venv/` &rarr; holds Python libraries.
  - `run_demo.sh` &rarr; launch script.
  - `yolov8n.pt` &rarr; object detection model weights.
  - `README.md` &rarr; written documentation.

---

### Card 20: `worlds/ai_arena.wbt`
- **What it is**: The Webots simulation world file.
- **Contents**: Text definitions of the 4m x 4m floor, four glass walls with metal rails, three target objects, two lights, and the robot with camera and distance sensors.

---

### Card 21: `controllers/mission_control/mission_control.py`
- **What it is**: The main program file for the entire project.
- **Contents**:
  - Connects to Webots sensors and motors.
  - Runs the control loop (Patrol, Chase, Manual).
  - Runs OpenCV color detection, YOLOv8, and CLIP models.
  - Runs FastAPI, WebSocket broadcasts, and hosts both web dashboards (2D and 3D).

---

### Card 22: `run_demo.sh`
- **What it is**: A shell command script.
- **Contents**: Sets paths for Webots libraries, terminates older running instances, opens Webots with `ai_arena.wbt`, and starts `mission_control.py` with the virtual environment.

---

### Card 23: `yolov8n.pt`
- **What it is**: A binary file containing saved numbers (weights) for a neural network.
- **Contents**: The parameters needed by YOLOv8 Nano to detect 80 types of objects.

---

### Card 24: `.venv/`
- **What it is**: A local Python virtual environment folder.
- **Contents**: A private copy of Python and all installed libraries (FastAPI, Uvicorn, OpenCV, PyTorch, Ultralytics, CLIP).

---

## Section 4: Components of Webots

---

### Card 25: World (`.wbt`)
- **What it is**: A plain text file defining a simulation environment.
- **Function**: Lists every 3D object, its shape dimensions, material colors, positions, light sources, and physics rules.

---

### Card 26: Supervisor API
- **What it is**: A control permission level in Webots.
- **Function**: Gives code the ability to read or change positions, orientations, and velocities of any object in the world directly, without applying physical forces.

---

### Card 27: Robot Node
- **What it is**: The base container in Webots for an controllable agent.
- **Function**: Groups motors, sensors, and physical body parts under one name and connects them to a controller script.

---

### Card 28: RotationalMotor
- **What it is**: A simulated electric motor node.
- **Function**: Turns a wheel. Code sets position to infinity to enable continuous rotation, then sets velocity values to move the robot.

---

### Card 29: Camera Sensor
- **What it is**: A virtual optical sensor node.
- **Function**: Captures a grid of pixels from the 3D scene (640 wide by 480 high) at each simulation step and returns them as a byte array.

---

### Card 30: DistanceSensor
- **What it is**: A virtual proximity sensor node.
- **Function**: Emits a simulated ray forward. It returns a low number when the path is clear and a high number when a surface is close.

---

### Card 31: Solid Node and DEF Name
- **What it is**:
  - `Solid`: An object in Webots that has physical existence and position.
  - `DEF`: A unique label assigned to a node in the `.wbt` file so that Python code can find and control it by name.

---

## Section 5: Python Libraries, Models, and Training Datasets

---

### Card 32: Ultralytics YOLOv8
- **Library**: `ultralytics`
- **Model**: `yolov8n` (YOLOv8 Nano, size ~6.5 MB)
- **What it does**: Takes an image, passes it through a convolutional neural network in a single evaluation step, and outputs bounding box coordinates and class names.
- **Trained on dataset**: COCO (Common Objects in Context).

---

### Card 33: COCO Dataset
- **Full name**: Common Objects in Context.
- **What it contains**: Over 330,000 images labeled with 1.5 million object instances across 80 everyday categories (such as person, chair, potted plant, bottle, cup).
- **Role in this project**: Allowed the YOLOv8 model to recognize objects in the camera view without extra training.

---

### Card 34: OpenAI CLIP
- **Library**: `clip`
- **Model**: `ViT-B/32` (Vision Transformer, Base size, 32x32 pixel image patch size)
- **What it does**: Converts an image into a number vector, converts text sentences into number vectors, and calculates similarity between them to find the best text description.
- **Trained on dataset**: WIT (WebImageText).

---

### Card 35: WIT Dataset
- **Full name**: WebImageText.
- **What it contains**: 400 million pairs of images and text descriptions collected from public internet pages.
- **Role in this project**: Enabled zero-shot image classification, allowing the robot to classify custom phrases (such as "a safe open hallway" or "a robotic obstacle ahead") without training on new images.

---

### Card 36: OpenCV Color Tracking
- **Library**: `opencv-python-headless`
- **Model / Method**: Deterministic mathematical filtering (No machine learning model, no dataset).
- **How it works**: Converts BGR pixels to HSV (Hue, Saturation, Value). Checks each pixel against lower and upper numeric limits. Groups matching pixels into contours to find center coordinates and bounding boxes.

---

### Card 37: PyTorch
- **Library**: `torch`
- **Model / Method**: General numerical computing engine for neural networks.
- **What it does in this project**: Loads the CLIP model architecture, reads the binary parameter values, and executes matrix multiplications on the CPU when classifying a frame.

