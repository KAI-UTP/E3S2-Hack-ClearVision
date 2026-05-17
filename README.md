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

### Local Installation and Environment Setup

1. Prerequisites
Ensure you have Python 3.11 installed locally.

2. Clone and Initialize the Repository
Bash

git clone <your-repository-url>
cd clearvision
3. Establish a Virtual Environment
Bash

python -m venv venv
source venv/bin/activate  # On Windows use: venv\\Scripts\\activate
4. Install Dependencies
Bash

pip install -r requirements.txt
Note: On your first execution instance, the backend asynchronously retrieves the necessary standard tracking weights file (yolov8s-worldv2.pt) automatically if not present in the current folder path.

Running the Application
Launch the development web application server via:

Bash

python app.py
Once initialized, open your browser and navigate to http://localhost:8000 to view the interactive application dashboard.

Container Deployment (Docker and Cloud)
The code contains pre-configured instructions for deployment to cloud platforms like Google Cloud Run via Docker.

Build the Docker Image Locally:
Bash

docker build -t clearvision-app:latest .
Launch the Container Locally:
Bash

docker run -p 8080:8080 clearvision-app:latest
API Specification
1. Health Diagnostic Check
URL: /health

Method: GET

Response Summary:

JSON

{
  "status": "ok",
  "model_ready": true,
  "model_error": null
}
2. Contextual Inference Handler
URL: /detect

Method: POST

Payload Format: multipart/form-data containing binary image file mapped to the form key "frame"

Response Summary: Contains generated announcements, bounding-box data arrays, structural metric calculations, and safe-path steering parameters.

Team SUS Developers
Developed for ElectroHack 2.0 (2026) at Universiti Teknologi PETRONAS by:

Chan Li Kai - Bachelor of Computer Engineering with Honours

Yap Wei Ming - Bachelor of Electrical and Electronics Engineering with Honours

Irvin Chang Hou Ceng - Bachelor of Computer Engineering with Honours

Tee Teck An - Bachelor of Electrical and Electronics Engineering with Honours
"""
