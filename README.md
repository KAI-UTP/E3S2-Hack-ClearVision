# E3S2-Hack-ClearVision

ClearVision is an AI-powered assistive vision navigation system developed by team SUS Developers for ElectroHack 2.0 (SENSE-AI: Inclusion in Motion) at Universiti Teknologi PETRONAS.

The platform leverages computer vision techniques to empower visually impaired and blind individuals by scanning real-world surroundings, estimating obstacle proximity, resolving traffic light statuses, and providing intuitive auditory and map-based navigation data to support independent mobility.

---

## Key Features

* **Real-time Object Prioritization:** Dynamically categorizes detected surrounding hazards (such as traffic lights, pedestrians, cars, walls, and stairs) using prioritized risk hierarchies (Levels 1 to 5).
* **Geometric Distance Estimation:** Computes metric distances dynamically based on camera pixel bounding-box heights mapping to known physical item defaults.
* **Corridor Walking Zone and Path Planning:** Monitors the central walking corridor of the camera stream to safely evaluate path availability and issue alternative orientation feedback (left or right).
* **Dual Traffic Light Resolution Engine:** Implements precise HSV color space masking to distinguish real-world operational traffic signals (Red, Yellow, Green) alongside a custom virtual phone test pattern recognition mode for controlled indoor debugging.
* **Integrated Interactive Web Map:** Frontend tracking layer built using Leaflet.js supporting dynamic orientation tracking, path guidance, and Text-to-Speech (TTS) announcement engines.
* **Thread-Safe Concurrency Management:** Driven by a single-worker backend execution queue allowing safe sequential processing of frames over network request streams.

---

## Architecture and Tech Stack
                 +---------------------------------------+
                 |         Frontend Web Client           |
                 |  - Leaflet.js Geolocation Map Tracker |
                 |  - HTML5 Camera Capture Pipeline      |
                 |  - Web Speech TTS Audio Generator     |
                 +-------------------+-------------------+
                                     |
                            (POST /detect)
                                     v
                 +---------------------------------------+
                 |            FastAPI Server             |
                 |  - Single-worker ThreadPoolExecutor   |
                 |  - Multi-thread Background Warmup     |
                 +-------------------+-------------------+
                                     |
                                     v
                 +---------------------------------------+
                 |         Scene Inference Core          |
                 |  - Ultralytics YOLO Object Detector   |
                 |  - Custom Ratio & HSV Color Masking   |
                 |  - Metric Focal Height Distance Unit  |
                 +---------------------------------------+

                 
### Frameworks and Libraries
* **Backend Application:** FastAPI, Uvicorn, Jinja2 Templates
* **Machine Learning and Computer Vision:** Ultralytics YOLOv8, OpenCV (headless), NumPy, Pillow
* **Frontend UI Layer:** Leaflet.js (Mapping), Vanilla HTML5/CSS3/JavaScript (Web Audio API / Web Speech Synthesis)

---

## Project Repository Structure

```text
├── templates/
│   └── index.html            # Frontend user interface and mapping scripts
├── .dockerignore             # Excluded files for container deployments
├── .gcloudignore             # Deployment configuration rules for Google Cloud
├── app.py                    # Main FastAPI application server interface
├── detector.py               # Core YOLO inference and HSV traffic light processor
├── distance_utils.py         # Distance estimation module and metrics
├── Dockerfile                # Multi-stage production container build guide
└── requirements.txt          # Explicit pip python ecosystem dependencies

---


