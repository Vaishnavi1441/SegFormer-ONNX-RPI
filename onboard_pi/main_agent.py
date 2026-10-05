"""
Main Autonomous UAV Companion Daemon (Pi 5 Entrypoint)
Coordinates MAVLink, Triple-Camera Ingestion, Detection, Segmentation & Mission Executive.
"""

import sys
import os

# Automatically add project root directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import yaml
import logging
import cv2
from typing import Dict, Any

from onboard_pi.drivers.mavlink_bridge import MAVLinkBridge
from onboard_pi.drivers.camera_hub import CameraHub
from onboard_pi.drivers.thermal_radiometric import ThermalRadiometricProcessor
from onboard_pi.vision.segmentor import TerrainSegmentor
from onboard_pi.vision.detector import ObjectDetector
from onboard_pi.vision.geotag import GeoTagger
from onboard_pi.autonomy.mission_executive import MissionExecutive

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("OnboardUAVDaemon")

def load_yaml(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}

class UAVAutonomousDaemon:
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = os.path.join(PROJECT_ROOT, "config")

        # Load configs
        self.cam_config = load_yaml(os.path.join(config_dir, "cameras.yaml"))
        self.mav_config = load_yaml(os.path.join(config_dir, "mavlink.yaml"))
        self.auto_config = load_yaml(os.path.join(config_dir, "autonomy.yaml"))

        # Initialize MAVLink
        conn_str = self.mav_config.get("connection", {}).get("address", "udp:127.0.0.1:14550")
        baud = self.mav_config.get("connection", {}).get("baudrate", 921600)
        self.mavlink = MAVLinkBridge(connection_str=conn_str, baudrate=baud)

        # Initialize Sensors & Vision
        self.camera_hub = CameraHub(self.cam_config)
        self.thermal_processor = ThermalRadiometricProcessor(
            min_temp_c=self.cam_config.get("thermal_camera", {}).get("min_temp_celsius", 10.0),
            max_temp_c=self.cam_config.get("thermal_camera", {}).get("max_temp_celsius", 60.0),
            hotspot_thresh_c=self.cam_config.get("thermal_camera", {}).get("hotspot_threshold_celsius", 42.0)
        )
        self.segmentor = TerrainSegmentor()
        self.detector = ObjectDetector()
        
        # Initialize Geotagger
        rgb_cfg = self.cam_config.get("rgb_camera", {})
        self.geotagger = GeoTagger(
            fov_horizontal_deg=rgb_cfg.get("fov_horizontal_deg", 75.0),
            image_width=rgb_cfg.get("width", 1280),
            image_height=rgb_cfg.get("height", 720),
            camera_pitch_mount_deg=self.auto_config.get("geotagging", {}).get("camera_mount_pitch_deg", -90.0)
        )

        # Initialize Autonomy State Machine
        self.mission_executive = MissionExecutive(self.mavlink, self.geotagger)

    def start(self):
        logger.info("==================================================")
        logger.info("   AUTONOMOUS UAV ONBOARD DAEMON INITIALIZING     ")
        logger.info("==================================================")

        # 1. Connect MAVLink
        self.mavlink.connect()

        # 2. Start Cameras
        self.camera_hub.start_all()

        # 3. Start Mission Executive
        self.mission_executive.start()

        logger.info("All systems online. Running perception & autonomy loop...")

        try:
            while True:
                t0 = time.time()

                # Ingest Frames
                rgb_frame = self.camera_hub.get_frame("rgb")
                ms_frame = self.camera_hub.get_frame("multispectral")
                thermal_raw = self.camera_hub.get_frame("thermal")

                # City semantic segmentation is performed on RGB.
                # Multispectral is reserved for spectral indices such as NDVI/NDWI.
                if rgb_frame is not None:
                    _, city_stats, _ = self.segmentor.classifier.segment_frame(rgb_frame, alpha=0.45)
                    self.mission_executive.update_terrain_stats(city_stats)

                # Multispectral indices remain independent from RGB semantic segmentation.
                if ms_frame is not None:
                    ndvi = self.segmentor.compute_ndvi(ms_frame)
                    # The GCS can expose this grid through the NDVI stream.

                # Process Thermal
                thermal_color = None
                if thermal_raw is not None:
                    temp_grid = self.thermal_processor.raw_to_celsius(thermal_raw)
                    thermal_color = self.thermal_processor.celsius_to_colorized(temp_grid, "ironbow")
                    hotspots, _ = self.thermal_processor.detect_hotspots(temp_grid)

                # Multi-modal Object Detection
                detections = self.detector.detect(rgb_frame, thermal_raw)

                # Update Georeferenced Detections with latest Telemetry
                telem = self.mavlink.get_telemetry()
                self.mission_executive.update_vision_detections(detections, telem)

                # Loop Rate Control (~15-20 Hz on Pi 5)
                elapsed = time.time() - t0
                sleep_time = (1.0 / 20.0) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Shutting down UAV Daemon...")
        finally:
            self.stop()

    def stop(self):
        self.mission_executive.stop()
        self.camera_hub.stop_all()
        self.mavlink.close()
        logger.info("UAV Daemon stopped cleanly.")

if __name__ == "__main__":
    daemon = UAVAutonomousDaemon()
    daemon.start()
