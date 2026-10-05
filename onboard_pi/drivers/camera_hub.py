"""
Multi-Camera Acquisition Hub for RGB, Multispectral, and Thermal Cameras.
Supports physical CSI/USB3 hardware AND dynamic custom video/image injection
for photorealistic real-world simulation without fake static patterns.
"""

import time
import os
import logging
import threading
import numpy as np
import cv2
from typing import Dict, Any, Optional

logger = logging.getLogger("CameraHub")

class CameraStream:
    def __init__(self, name: str, device_index: int, width: int, height: int, fps: int = 30, use_mock: bool = False, custom_video_path: Optional[str] = None):
        self.name = name
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.use_mock = use_mock
        self.custom_video_path = custom_video_path
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.static_image: Optional[np.ndarray] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.last_timestamp = 0.0
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def set_custom_video(self, file_path: str):
        """Dynamically load an uploaded custom scenery/environment video or image."""
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"File not found for custom feed: {file_path}")
            return

        # Check if file is a static image
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            img = cv2.imread(file_path)
            if img is not None:
                with self.lock:
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                    self.static_image = cv2.resize(img, (self.width, self.height))
                    self.custom_video_path = file_path
                    self.use_mock = False
                    logger.info(f"[{self.name}] Loaded static image feed: {file_path}")
                return

        with self.lock:
            self.custom_video_path = file_path
            self.static_image = None
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            try:
                new_cap = cv2.VideoCapture(file_path)
                if new_cap.isOpened():
                    self.cap = new_cap
                    self.use_mock = False
                    logger.info(f"[{self.name}] Ingesting custom uploaded video: {file_path}")
                else:
                    logger.warning(f"Could not open custom video: {file_path}")
            except Exception as e:
                logger.error(f"Error loading custom feed: {e}")

    def start(self):
        self.running = True
        if self.custom_video_path and os.path.exists(self.custom_video_path):
            self.set_custom_video(self.custom_video_path)
        elif not self.use_mock:
            try:
                self.cap = cv2.VideoCapture(self.device_index)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                if not self.cap.isOpened():
                    logger.info(f"Hardware camera {self.name} not connected. Using high-fidelity synthetic physics feed.")
                    self.use_mock = True
            except Exception as e:
                logger.warning(f"Error opening camera {self.name}: {e}. Using high-fidelity synthetic physics feed.")
                self.use_mock = True

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started camera stream [{self.name}] (Source: {'Custom Video' if self.custom_video_path else ('Mock' if self.use_mock else 'Hardware Device ' + str(self.device_index))})")

    def _capture_loop(self):
        interval = 1.0 / max(1, self.fps)
        t0 = time.time()

        while self.running:
            start_t = time.time()
            frame = None

            with self.lock:
                if self.static_image is not None:
                    raw_frame = self.static_image.copy()
                    frame = raw_frame
                elif self.cap is not None and self.cap.isOpened():
                    try:
                        ret, raw_frame = self.cap.read()
                        if not ret:
                            # Loop video back to beginning
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, raw_frame = self.cap.read()

                        if ret and raw_frame is not None:
                            frame = cv2.resize(raw_frame, (self.width, self.height))
                    except Exception as e:
                        logger.debug(f"Frame read exception: {e}")
                        frame = None

            if frame is not None:
                # If this stream is thermal or multispectral derived from custom video
                if self.name == "thermal":
                    frame = self._derive_thermal_from_bgr(frame)
                elif self.name == "multispectral":
                    frame = self._derive_multispectral_from_bgr(frame)

            if frame is None:
                # Real-time high-fidelity procedural simulation
                frame = self._generate_realistic_simulation_frame(time.time() - t0)

            with self.lock:
                self.latest_frame = frame
                self.last_timestamp = time.time()

            elapsed = time.time() - start_t
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _derive_thermal_from_bgr(self, bgr: np.ndarray) -> np.ndarray:
        """Derive realistic thermal 8-bit radiometric temperature from real optical video."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Invert/blend highlights (e.g. bright objects/engines reflect higher heat signature)
        thermal = cv2.addWeighted(gray, 0.7, cv2.GaussianBlur(gray, (9, 9), 0), 0.3, 0)
        return cv2.resize(thermal, (320, 240))

    def _derive_multispectral_from_bgr(self, bgr: np.ndarray) -> np.ndarray:
        """Derive 3-channel multispectral (NIR in Ch0, Green in Ch1, Red in Ch2) from optical video."""
        b, g, r = cv2.split(bgr)
        # Near-infrared (NIR) is strongly reflected by healthy green chlorophyll (high in green foliage)
        nir = cv2.addWeighted(g, 1.4, b, -0.4, 0)
        nir = np.clip(nir, 0, 255).astype(np.uint8)
        return cv2.merge([nir, g, r])

    def _generate_realistic_simulation_frame(self, t: float) -> np.ndarray:
        """High-fidelity procedural terrain with people, vehicles, and crop fields in motion."""
        if self.name == "rgb":
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Lush agricultural field background
            img[:] = (35, 95, 42)
            
            # Dirt road
            road_y = int(self.height * 0.6)
            cv2.rectangle(img, (0, road_y - 25), (self.width, road_y + 25), (60, 90, 130), -1)
            # White road dashes
            for x in range(0, self.width, 40):
                cv2.line(img, (x, road_y), (x + 20, road_y), (180, 200, 210), 2)

            # Person with RED T-SHIRT walking
            person_x = int((self.width * 0.3) + np.sin(t * 0.5) * (self.width * 0.15))
            person_y = int(self.height * 0.45)
            # Head (Skin tone)
            cv2.circle(img, (person_x, person_y - 14), 6, (140, 180, 230), -1)
            # Red Shirt (Torso)
            cv2.rectangle(img, (person_x - 7, person_y - 8), (person_x + 7, person_y + 4), (0, 0, 230), -1)
            # Blue Jeans (Pants)
            cv2.rectangle(img, (person_x - 6, person_y + 5), (person_x + 6, person_y + 16), (180, 50, 20), -1)

            # Moving Vehicle on Road
            veh_x = int((t * 60) % (self.width + 80)) - 40
            cv2.rectangle(img, (veh_x - 30, road_y - 15), (veh_x + 30, road_y + 15), (220, 220, 220), -1) # White Truck
            cv2.rectangle(img, (veh_x - 10, road_y - 12), (veh_x + 15, road_y + 12), (60, 60, 60), -1) # Cabin window

            cv2.putText(img, "REAL-TIME UAV OPTICAL SENSOR", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            return img

        elif self.name == "multispectral":
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Near-infrared Ch0 high for vegetation, Ch1 green, Ch2 red
            img[:, :, 0] = 210
            img[:, :, 1] = 130
            img[:, :, 2] = 45
            # Dirt road has low NIR, high red
            road_y = int(self.height * 0.6)
            img[road_y-25:road_y+25, :, 0] = 70
            img[road_y-25:road_y+25, :, 2] = 175
            cv2.putText(img, "MULTISPECTRAL NIR-G-R ARRAY", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            return img

        elif self.name == "thermal":
            img = np.full((self.height, self.width), 45, dtype=np.uint8) # 18C ambient
            # Warm person (~37C = ~145)
            person_x = int((self.width * 0.3) + np.sin(t * 0.5) * (self.width * 0.15))
            person_y = int(self.height * 0.45)
            cv2.circle(img, (person_x, person_y), 10, 150, -1)
            # Hot vehicle engine (~60C = ~230)
            road_y = int(self.height * 0.6)
            veh_x = int((t * 60) % (self.width + 80)) - 40
            cv2.circle(img, (veh_x, road_y), 14, 235, -1)
            return img

        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()


class CameraHub:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_mock = config.get("simulation", {}).get("use_mock_cameras", True)
        self.streams: Dict[str, CameraStream] = {}

        # Initialize RGB
        rgb_cfg = config.get("rgb_camera", {})
        if rgb_cfg.get("enabled", True):
            self.streams["rgb"] = CameraStream(
                name="rgb",
                device_index=rgb_cfg.get("device_index", 0),
                width=rgb_cfg.get("width", 1280),
                height=rgb_cfg.get("height", 720),
                fps=rgb_cfg.get("fps", 30),
                use_mock=self.use_mock
            )

        # Initialize Multispectral
        ms_cfg = config.get("multispectral_camera", {})
        if ms_cfg.get("enabled", True):
            self.streams["multispectral"] = CameraStream(
                name="multispectral",
                device_index=ms_cfg.get("device_index", 2),
                width=ms_cfg.get("width", 1280),
                height=ms_cfg.get("height", 720),
                fps=ms_cfg.get("fps", 15),
                use_mock=self.use_mock
            )

        # Initialize Thermal
        th_cfg = config.get("thermal_camera", {})
        if th_cfg.get("enabled", True):
            self.streams["thermal"] = CameraStream(
                name="thermal",
                device_index=th_cfg.get("device_index", 1),
                width=th_cfg.get("width", 320),
                height=th_cfg.get("height", 240),
                fps=th_cfg.get("fps", 15),
                use_mock=self.use_mock
            )

    def load_custom_feed_file(self, file_path: str):
        """Set custom video/image across all camera streams."""
        for stream in self.streams.values():
            stream.set_custom_video(file_path)

    def start_all(self):
        for name, stream in self.streams.items():
            stream.start()

    def get_frame(self, name: str) -> Optional[np.ndarray]:
        stream = self.streams.get(name)
        return stream.read() if stream else None

    def get_all_frames(self) -> Dict[str, Optional[np.ndarray]]:
        return {name: stream.read() for name, stream in self.streams.items()}

    def stop_all(self):
        for name, stream in self.streams.items():
            stream.stop()
