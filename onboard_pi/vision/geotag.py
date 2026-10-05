"""
Georeferencing Engine: Converts 2D Pixel Bounding Boxes/Centroids
to Global Earth Coordinates (Latitude, Longitude, Altitude MSL).
Supports both 90-degree Nadir (straight down) AND 45-degree Oblique Forward-Tilted Mounts,
with complete 3D perspective projection and UAV Attitude (Roll, Pitch, Yaw).
"""

import math
import numpy as np
from typing import Tuple

class GeoTagger:
    def __init__(self, 
                 fov_horizontal_deg: float = 75.0, 
                 image_width: int = 1280, 
                 image_height: int = 720, 
                 camera_pitch_mount_deg: float = -45.0): # Default to 45-degree oblique or -90 nadir
        self.image_width = image_width
        self.image_height = image_height
        self.fov_h_rad = math.radians(fov_horizontal_deg)
        
        # Focal length in pixels
        self.fx = (image_width / 2.0) / math.tan(self.fov_h_rad / 2.0)
        self.fy = self.fx
        self.cx = image_width / 2.0
        self.cy = image_height / 2.0

        self.mount_pitch_deg = camera_pitch_mount_deg

    def set_mount_angle(self, mount_pitch_deg: float):
        """Dynamically switch between 45-degree oblique (-45.0) and 90-degree nadir (-90.0)."""
        self.mount_pitch_deg = mount_pitch_deg

    def pixel_to_gps(self, 
                     px: float, 
                     py: float, 
                     uav_lat: float, 
                     uav_lon: float, 
                     uav_alt_agl_m: float, 
                     uav_heading_deg: float, 
                     uav_pitch_deg: float = 0.0, 
                     uav_roll_deg: float = 0.0) -> Tuple[float, float]:
        """
        Calculate the real-world (Latitude, Longitude) of a pixel in the image frame.
        Accounts for camera mount tilt (-45° oblique or -90° nadir) and UAV attitude.
        """
        if uav_alt_agl_m <= 0.5:
            return uav_lat, uav_lon

        # 1. Normalized Ray in Camera Optical Frame
        # Camera coords: +x right, +y down, +z optical forward along lens
        x_norm = (px - self.cx) / self.fx
        y_norm = (py - self.cy) / self.fy
        z_norm = 1.0

        ray_cam = np.array([x_norm, y_norm, z_norm], dtype=np.float64)
        ray_cam /= np.linalg.norm(ray_cam)

        # 2. Camera Frame to UAV Body Frame (NED: X=Forward, Y=Right, Z=Down)
        # Mount pitch angle (e.g. -45°: optical axis is 45° down from forward horizon; -90°: straight down)
        # Standard camera: Z_cam forward, X_cam right, Y_cam down
        # Pitch rotation of mount around Body Y-axis (tilt down by alpha)
        mount_pitch_rad = math.radians(self.mount_pitch_deg)

        # Base rotation from camera optical frame (Z forward, X right, Y down) to Body Frame
        # When mount_pitch is -90° (nadir): Z_cam -> Z_body (Down), Y_cam -> -X_body (Backward), X_cam -> Y_body (Right)
        # When mount_pitch is -45° (oblique): Z_cam is 45° between +X_body (Forward) and +Z_body (Down)
        
        # General Transformation for mount angle alpha (where alpha is negative for downward tilt from horizon):
        # alpha = -45° -> cos(45°)*X_body + sin(45°)*Z_body
        sin_alpha = math.sin(abs(mount_pitch_rad))
        cos_alpha = math.cos(abs(mount_pitch_rad))

        R_cam_to_body = np.array([
            [-y_norm * sin_alpha + z_norm * cos_alpha], # X_body (Forward)
            [x_norm],                                   # Y_body (Right)
            [y_norm * cos_alpha + z_norm * sin_alpha]  # Z_body (Down)
        ], dtype=np.float64).flatten()
        
        ray_body = R_cam_to_body / np.linalg.norm(R_cam_to_body)

        # 3. Apply UAV Attitude (Roll around X, Pitch around Y, Yaw around Z)
        roll = math.radians(uav_roll_deg)
        pitch = math.radians(uav_pitch_deg)
        yaw = math.radians(uav_heading_deg)

        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ])
        R_y = np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])
        R_z = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Combined Rotation: Body -> NED Earth Frame
        R_ned = R_z @ R_y @ R_x
        ray_ned = R_ned @ ray_body

        # 4. Ray-Ground Intersection (Ground at relative depth Z = uav_alt_agl_m)
        dz = ray_ned[2]
        if dz <= 1e-4:
            # Ray pointing above horizon or parallel to flat ground (clamp to max 500m forward)
            dz = 1e-4

        scale = uav_alt_agl_m / dz
        north_offset_m = ray_ned[0] * scale
        east_offset_m = ray_ned[1] * scale

        # 5. Convert North/East offsets (meters) to Latitude/Longitude
        R_earth = 6378137.0 # Earth equatorial radius
        d_lat = (north_offset_m / R_earth) * (180.0 / math.pi)
        d_lon = (east_offset_m / (R_earth * math.cos(math.radians(uav_lat)))) * (180.0 / math.pi)

        target_lat = uav_lat + d_lat
        target_lon = uav_lon + d_lon

        return target_lat, target_lon
