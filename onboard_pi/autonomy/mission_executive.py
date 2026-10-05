"""
Mission Executive State Machine & Command Dispatcher
Orchestrates autonomous flight behaviors, processes GCS / voice commands,
tracks terrain scanning progress (NDVI & land cover), and guides ArduPilot along survey paths.
"""

import time
import math
import logging
import threading
from typing import Dict, Any, List, Optional
from onboard_pi.drivers.mavlink_bridge import MAVLinkBridge
from onboard_pi.vision.geotag import GeoTagger
from onboard_pi.vision.detector import ObjectDetector
from onboard_pi.autonomy.guidance import LawnmowerSurveyPlanner, VisualTrackingController

logger = logging.getLogger("MissionExecutive")

class MissionExecutive:
    def __init__(self, mavlink: MAVLinkBridge, geotagger: GeoTagger, detector: Optional[ObjectDetector] = None):
        self.mavlink = mavlink
        self.geotagger = geotagger
        self.detector = detector
        self.tracker = VisualTrackingController()
        
        # State Machine: IDLE, TAKEOFF, SURVEY_GRID, TRACK_TARGET, ORBIT, RTL, LAND, HOLD, FILTER_SEARCH
        self.current_state = "IDLE"
        self.target_locked: Optional[Dict[str, Any]] = None
        self.survey_waypoints: List[Dict[str, float]] = []
        self.current_wp_idx = 0
        self.total_survey_area_m2 = 10000.0 # Default 100x100m
        self.scanned_area_m2 = 0.0
        self.scan_progress_pct = 0.0
        
        self.latest_terrain_stats: Dict[str, float] = {
            "healthy_vegetation_pct": 74.2,
            "stressed_vegetation_pct": 12.8,
            "bare_soil_pct": 9.5,
            "water_pct": 3.5,
            "mean_ndvi": 0.68
        }

        self.active_filter: Optional[Dict[str, Any]] = None
        self.detected_objects: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._state_machine_loop, daemon=True)
        self._thread.start()
        logger.info("Mission Executive State Machine started.")

    def set_camera_mount_angle(self, angle_deg: float):
        """Set camera pitch mount angle (e.g. -45.0 for oblique or -90.0 for nadir)."""
        self.geotagger.set_mount_angle(angle_deg)
        logger.info(f"Camera mount angle updated to: {angle_deg}°")

    def process_command(self, cmd_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute structured command received from GCS (Speech or Text)."""
        action = cmd_payload.get("action", "").upper()
        params = cmd_payload.get("parameters", {})
        logger.info(f"Processing Autonomous Command: {action} with params: {params}")

        with self._lock:
            if action == "TAKEOFF":
                alt = params.get("altitude_m", 25.0)
                self.mavlink.takeoff(alt)
                self.current_state = "TAKEOFF"
                return {"status": "SUCCESS", "message": f"Taking off to {alt}m"}

            elif action == "SURVEY_GRID" or action == "SCAN_AREA" or action == "SCAN_TERRAIN":
                width = params.get("width_m", 120.0)
                height = params.get("height_m", 120.0)
                alt = params.get("altitude_m", 35.0)
                telem = self.mavlink.get_telemetry()
                center_lat = params.get("lat", telem["lat"])
                center_lon = params.get("lon", telem["lon"])

                self.total_survey_area_m2 = width * height
                self.survey_waypoints = LawnmowerSurveyPlanner.generate_grid(
                    center_lat, center_lon, width_m=width, height_m=height, altitude_m=alt
                )
                self.current_wp_idx = 0
                self.scan_progress_pct = 0.0
                self.scanned_area_m2 = 0.0
                self.current_state = "SURVEY_GRID"

                # Command drone to first waypoint
                if self.survey_waypoints:
                    wp0 = self.survey_waypoints[0]
                    self.mavlink.goto_location(wp0["lat"], wp0["lon"], wp0["alt_m"], ground_speed=8.0)

                logger.info(f"Generated {len(self.survey_waypoints)} survey waypoints for {int(self.total_survey_area_m2)}m² terrain scan.")
                return {"status": "SUCCESS", "message": f"Initiating {int(self.total_survey_area_m2)}m² terrain scan grid ({len(self.survey_waypoints)} waypoints) at {alt}m."}

            elif action == "FILTER_SEARCH":
                self.active_filter = params
                if self.detector:
                    self.detector.set_query_filter(params)
                desc = f"{params.get('color', '')} {params.get('target_class', '')}".strip()
                return {"status": "SUCCESS", "message": f"Visual filter engaged for '{desc}'. Overlays active."}

            elif action == "TRACK_TARGET" or action == "FOLLOW_TARGET":
                self.active_filter = params
                if self.detector:
                    self.detector.set_query_filter(params)
                self.current_state = "TRACK_TARGET"
                desc = f"{params.get('color', '')} {params.get('target_class', 'target')}".strip()
                return {"status": "SUCCESS", "message": f"Autonomous visual tracking locked onto '{desc}'."}

            elif action == "RETURN_TO_LAUNCH" or action == "RTL" or action == "GO_HOME":
                self.current_state = "RTL"
                self.mavlink.return_to_launch()
                return {"status": "SUCCESS", "message": "Returning to Launch (RTL)"}

            elif action == "LAND":
                self.current_state = "LAND"
                self.mavlink.land()
                return {"status": "SUCCESS", "message": "Landing immediately"}

            elif action == "HOVER" or action == "HOLD":
                self.current_state = "HOLD"
                self.mavlink.set_mode("LOITER")
                return {"status": "SUCCESS", "message": "Holding position in Loiter mode"}

            elif action == "SET_ALTITUDE":
                target_alt = params.get("altitude_m", 30.0)
                telem = self.mavlink.get_telemetry()
                self.mavlink.goto_location(telem["lat"], telem["lon"], target_alt)
                return {"status": "SUCCESS", "message": f"Adjusting altitude to {target_alt}m"}

            else:
                return {"status": "ERROR", "message": f"Unknown action: {action}"}

    def update_terrain_stats(self, stats: Dict[str, float]):
        with self._lock:
            self.latest_terrain_stats = stats

    def update_vision_detections(self, detections: List[Dict[str, Any]], telemetry: Dict[str, Any]):
        """Georeference new vision detections and store with global GPS coordinates."""
        with self._lock:
            lat = telemetry["lat"]
            lon = telemetry["lon"]
            alt = telemetry["alt_relative_m"]
            heading = telemetry["heading_deg"]
            pitch = telemetry["pitch_deg"]
            roll = telemetry["roll_deg"]

            for det in detections:
                cx, cy = det["center"]
                target_lat, target_lon = self.geotagger.pixel_to_gps(
                    cx, cy, lat, lon, alt, heading, pitch, roll
                )
                det["gps"] = {
                    "lat": round(target_lat, 7),
                    "lon": round(target_lon, 7)
                }
                det["timestamp"] = time.time()

            self.detected_objects = detections

            if self.current_state == "TRACK_TARGET" and len(detections) > 0:
                matched = [d for d in detections if d.get("is_matched", True)]
                self.target_locked = matched[0] if len(matched) > 0 else detections[0]

    def _state_machine_loop(self):
        while self._running:
            try:
                telem = self.mavlink.get_telemetry()
                
                with self._lock:
                    state = self.current_state

                    # Handle Survey Waypoints Navigation & Progress
                    if state == "SURVEY_GRID" and self.survey_waypoints:
                        if self.current_wp_idx < len(self.survey_waypoints):
                            target_wp = self.survey_waypoints[self.current_wp_idx]
                            
                            R_earth = 6378137.0
                            d_north = (target_wp["lat"] - telem["lat"]) * (math.pi / 180.0) * R_earth
                            d_east = (target_wp["lon"] - telem["lon"]) * (math.pi / 180.0) * R_earth * math.cos(math.radians(telem["lat"]))
                            dist = math.sqrt(d_north**2 + d_east**2)

                            if dist < 2.5: # Reached waypoint
                                self.current_wp_idx += 1
                                self.scan_progress_pct = round((self.current_wp_idx / len(self.survey_waypoints)) * 100, 1)
                                self.scanned_area_m2 = round((self.scan_progress_pct / 100.0) * self.total_survey_area_m2, 0)
                                logger.info(f"Reached survey waypoint {self.current_wp_idx}/{len(self.survey_waypoints)} ({self.scan_progress_pct}% scanned)")
                                
                                if self.current_wp_idx < len(self.survey_waypoints):
                                    next_wp = self.survey_waypoints[self.current_wp_idx]
                                    self.mavlink.goto_location(next_wp["lat"], next_wp["lon"], next_wp["alt_m"], ground_speed=8.0)
                            else:
                                self.mavlink.goto_location(target_wp["lat"], target_wp["lon"], target_wp["alt_m"], ground_speed=8.0)
                        else:
                            logger.info("Terrain Survey Grid Complete! Holding position.")
                            self.current_state = "HOLD"
                            self.scan_progress_pct = 100.0
                            self.scanned_area_m2 = self.total_survey_area_m2

                    # Handle Dynamic Visual Tracking
                    elif state == "TRACK_TARGET" and self.target_locked:
                        norm_x, norm_y = self.target_locked["center_normalized"]
                        vx, vy, vz, yaw_rate = self.tracker.compute_command(
                            norm_x, norm_y, target_alt_m=25.0, current_alt_m=telem["alt_relative_m"]
                        )
                        self.mavlink.set_velocity_body(vx, vy, vz, yaw_rate_deg_s=yaw_rate)

            except Exception as e:
                logger.error(f"Mission state machine error: {e}")

            time.sleep(0.1)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self.current_state,
                "survey_waypoints": self.survey_waypoints,
                "active_waypoints_count": len(self.survey_waypoints),
                "current_waypoint_idx": self.current_wp_idx,
                "scan_progress_pct": self.scan_progress_pct,
                "scanned_area_m2": self.scanned_area_m2,
                "total_survey_area_m2": self.total_survey_area_m2,
                "terrain_stats": self.latest_terrain_stats,
                "active_filter": self.active_filter,
                "camera_mount_angle_deg": self.geotagger.mount_pitch_deg,
                "latest_detections": self.detected_objects,
                "target_locked": self.target_locked
            }

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
