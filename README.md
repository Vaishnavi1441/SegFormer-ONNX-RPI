# Autonomous Multi-Sensor UAV Scanner & Ground Control System

Complete end-to-end software stack for an autonomous 10-inch quadcopter equipped with an **ArduPilot / Pixhawk** flight controller, **Raspberry Pi 5 (8GB)** companion computer, **triple camera payload (RGB + Multispectral + Thermal)**, **R12 RC Receiver**, and a **Speech & Text Ground Control Station (GCS)** web application.

---

## System Architecture

```
                                +-----------------------------------+
                                |    GROUND CONTROL STATION (GCS)   |
                                |  Speech / Text NLP Intent Engine  |
                                |  Interactive Map & Multi-Cam HUD  |
                                +-----------------+-----------------+
                                                  | (WiFi / Telemetry Data Link)
                                                  v
+-----------------------+       +-----------------+-----------------+
|   R12 RC RECEIVER     |       |    RASPBERRY PI 5 (8GB COMPANION) |
| Manual Safety Override|       |  - Triple Camera Hub (RGB/MS/LWIR)|
| Direct to Pixhawk RC  |       |  - YOLO Detection + Fast NDVI     |
+-----------+-----------+       |  - Radiometric Thermal Hotspots   |
            |                   |  - GeoTagger (Pixel-to-GPS)       |
            | (SBUS / PPM)      |  - Mission Executive State Machine|
            v                   +-----------------+-----------------+
+-----------+-------------------------------------+-----------------+
|               ARDUPILOT / PIXHAWK FLIGHT CONTROLLER               |
|      EKF3 Navigation, Rate Controllers, Motors & ESCs (10-Inch)   |
+-------------------------------------------------------------------+
```

---

## Directory Structure

```
uav_autonomous_scanner/
├── config/
│   ├── cameras.yaml              # Camera IDs, resolutions, calibrations & FOV
│   ├── mavlink.yaml              # MAVLink port (UART/UDP), baud (921600), stream rates
│   └── autonomy.yaml             # Altitudes, survey spacing, tracking PID parameters
├── gcs/                          # Ground Control Station Application
│   ├── backend/
│   │   ├── nlp_parser.py         # Speech / Text NLP intent parser
│   │   └── gcs_server.py         # FastAPI + WebSockets + MJPEG Multi-Cam Video
│   └── frontend/
│       ├── index.html            # Tactical GCS layout (Map, HUD, Voice mic)
│       ├── style.css             # Cybernetic dark aerospace styling
│       └── app.js                # WebSockets, Leaflet Map & Speech Recognition
├── onboard_pi/                   # Runs on Raspberry Pi 5
│   ├── drivers/
│   │   ├── mavlink_bridge.py     # High-speed ArduPilot communication
│   │   ├── camera_hub.py         # Async multi-camera capture (RGB, MS, Thermal)
│   │   └── thermal_radiometric.py# Raw 16-bit to Celsius mapping & hotspot detection
│   ├── vision/
│   │   ├── detector.py           # Multi-modal detection (RGB + Thermal)
│   │   ├── segmentor.py          # City-aware semantic segmentation + NDVI/NDWI
│   │   └── geotag.py             # Ray-ground intersection (Pixel -> GPS Lat/Lon)
│   ├── autonomy/
│   │   ├── guidance.py           # Lawnmower survey grid & visual tracking PID
│   │   └── mission_executive.py  # Autonomous state machine & command handler
│   └── main_agent.py             # Main daemon entrypoint on Pi 5
├── simulation/
│   └── sitl_launcher.py          # ArduPilot SITL setup guide
├── training/
│   └── train_detector.py         # YOLO training script for RTX 4050 GPU
├── test_suite.py                 # Automated unit and integration tests
└── requirements.txt
```

---

## Quick Start Guide

### 1. Installation on Development PC (Windows 11) or Raspberry Pi 5

```bash
# Clone or navigate to the workspace
cd C:\Users\thisi\.gemini\antigravity\scratch\uav_autonomous_scanner

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare the city semantic segmentation model

```bash
python models/download_models.py
```

The model is SegFormer-B0 fine-tuned on ADE20K and runs through ONNX Runtime. The project maps ADE20K classes into UAV classes such as building, road, sidewalk, vehicle, person, vegetation, water, bare ground and sky. If the model is absent, the application uses a conservative visual fallback and explicitly labels it as fallback rather than reporting fabricated statistics.

### 3. Running the Ground Control Station (GCS)

```bash
python gcs/backend/gcs_server.py
```
Open your browser to: **`http://localhost:8000`**

- **Voice Commands**: Click the microphone icon and speak naturally (e.g., *"Take off to 30 meters"*, *"Start grid survey"*, *"Follow red vehicle"*, *"Return home"*).
- **Video Feeds**: Switch between **RGB Optical**, **Thermal LWIR (Ironbow)**, and **Multispectral NDVI** tabs in real-time.
- **Georeferenced Map**: Watch detections automatically pinned on the satellite map with precise GPS coordinates.

### 4. Deploying to Raspberry Pi 5

1. Connect the Pixhawk `TELEM2` port to Raspberry Pi 5 GPIO UART pins (`TX -> RX`, `RX -> TX`, `GND -> GND`).
2. Set Pixhawk `SERIAL2_BAUD = 921` (921600 baud) and `SERIAL2_PROTOCOL = 2` (MAVLink2).
3. Connect cameras:
   - RGB Camera: CSI / USB3
   - Thermal Camera (e.g., FLIR Lepton via PureThermal): USB
   - Multispectral Camera: USB / CSI
4. Start the onboard daemon on boot:
```bash
python onboard_pi/main_agent.py
```

### 5. Safety & R12 RC Manual Override
- The **R12 Receiver** is wired directly to the Pixhawk RC IN.
- The pilot maintains absolute override priority at all times. Switching the flight mode switch on your transmitter to `POSHOLD` or `STABILIZE` instantly disengages companion computer autonomous control.
