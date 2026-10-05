"""
Autonomous Guidance & Flight Pattern Generator
Includes Lawnmower Survey Grid Generator, Visual Target Tracker Controller (PID Body Velocity),
and Circular Orbit Path Planner.
"""

import math
from typing import List, Tuple, Dict, Any

class LawnmowerSurveyPlanner:
    @staticmethod
    def generate_grid(center_lat: float, 
                      center_lon: float, 
                      width_m: float = 100.0, 
                      height_m: float = 100.0, 
                      grid_spacing_m: float = 18.0, 
                      altitude_m: float = 35.0) -> List[Dict[str, float]]:
        """
        Generate a serpentine (lawnmower) survey path around a center GPS point.
        """
        R_earth = 6378137.0
        num_legs = max(2, int(height_m / grid_spacing_m))
        waypoints = []

        # Half extents in meters
        half_w = width_m / 2.0
        half_h = height_m / 2.0

        for i in range(num_legs + 1):
            north_offset = -half_h + (i * grid_spacing_m)
            # Alternate East-West direction for serpentine scan
            if i % 2 == 0:
                east_start = -half_w
                east_end = half_w
            else:
                east_start = half_w
                east_end = -half_w

            # Point 1 (Start of leg)
            d_lat1 = (north_offset / R_earth) * (180.0 / math.pi)
            d_lon1 = (east_start / (R_earth * math.cos(math.radians(center_lat)))) * (180.0 / math.pi)
            waypoints.append({
                "lat": center_lat + d_lat1,
                "lon": center_lon + d_lon1,
                "alt_m": altitude_m
            })

            # Point 2 (End of leg)
            d_lat2 = (north_offset / R_earth) * (180.0 / math.pi)
            d_lon2 = (east_end / (R_earth * math.cos(math.radians(center_lat)))) * (180.0 / math.pi)
            waypoints.append({
                "lat": center_lat + d_lat2,
                "lon": center_lon + d_lon2,
                "alt_m": altitude_m
            })

        return waypoints


class VisualTrackingController:
    """
    Proportional Visual Servoing Controller.
    Computes body velocity commands (vx, vy, vz, yaw_rate) to keep a target centered in the camera frame.
    """
    def __init__(self, kp_linear: float = 3.0, kp_yaw: float = 45.0, max_speed_mps: float = 5.0):
        self.kp_linear = kp_linear
        self.kp_yaw = kp_yaw
        self.max_speed_mps = max_speed_mps

    def compute_command(self, target_norm_x: float, target_norm_y: float, target_alt_m: float = 25.0, current_alt_m: float = 25.0) -> Tuple[float, float, float, float]:
        """
        target_norm_x, target_norm_y: normalized coordinates from 0.0 to 1.0 (center is 0.5, 0.5)
        Returns: (vx, vy, vz, yaw_rate_deg_s) in Body NED Frame
        """
        # Error from center (-0.5 to +0.5)
        err_x = target_norm_x - 0.5  # Positive = Target is to the right
        err_y = target_norm_y - 0.5  # Positive = Target is below center (behind or forward depending on mount)

        # Body Frame Velocity Commands:
        # vy (Right/Left): Proportional to horizontal pixel error
        vy = max(-self.max_speed_mps, min(self.max_speed_mps, err_x * self.kp_linear))
        
        # vx (Forward/Backward): Proportional to vertical pixel error (assuming downward-tilted camera)
        vx = max(-self.max_speed_mps, min(self.max_speed_mps, -err_y * self.kp_linear))

        # vz (Altitude correction): Climb/descend to maintain tracking altitude
        err_alt = current_alt_m - target_alt_m
        vz = max(-1.5, min(1.5, err_alt * 0.5)) # Negative vz is climb in NED

        # Yaw Rate: Turn towards target
        yaw_rate = max(-self.kp_yaw, min(self.kp_yaw, err_x * self.kp_yaw))

        return vx, vy, vz, yaw_rate
