"""
Thermal Radiometric Image Processor
Handles raw thermal sensor conversion, temperature estimation, false-color palettization
(Ironbow, Jet, White Hot), and hotspot threshold detection.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any

class ThermalRadiometricProcessor:
    def __init__(self, min_temp_c: float = 10.0, max_temp_c: float = 60.0, hotspot_thresh_c: float = 42.0):
        self.min_temp_c = min_temp_c
        self.max_temp_c = max_temp_c
        self.hotspot_thresh_c = hotspot_thresh_c

    def raw_to_celsius(self, raw_frame: np.ndarray) -> np.ndarray:
        """
        Convert raw thermal array (16-bit Lepton/Boson raw counts or normalized 8-bit)
        to floating point Celsius temperature grid.
        """
        if raw_frame.dtype == np.uint16:
            # FLIR Radiometric format: raw count / 100 - 273.15 (Kelvin to Celsius)
            # or linear scale depending on camera AGC mode:
            temp_c = (raw_frame.astype(np.float32) / 100.0) - 273.15
        elif raw_frame.dtype == np.uint8:
            # Linear mapping from 0-255 to min_temp_c ... max_temp_c
            temp_c = self.min_temp_c + (raw_frame.astype(np.float32) / 255.0) * (self.max_temp_c - self.min_temp_c)
        else:
            temp_c = raw_frame.astype(np.float32)
        return temp_c

    def celsius_to_colorized(self, temp_grid: np.ndarray, colormap: str = "ironbow") -> np.ndarray:
        """Apply false-color thermal colormap to Celsius temperature grid."""
        # Normalize to 0-255 based on dynamic or fixed range
        t_min = max(self.min_temp_c, np.percentile(temp_grid, 1))
        t_max = min(self.max_temp_c, np.percentile(temp_grid, 99))
        if t_max <= t_min:
            t_max = t_min + 1.0

        normalized = np.clip((temp_grid - t_min) / (t_max - t_min) * 255.0, 0, 255).astype(np.uint8)

        if colormap.lower() == "ironbow":
            # Approximated Ironbow using COLORMAP_INFERNO or COLORMAP_JET
            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
        elif colormap.lower() == "jet":
            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        elif colormap.lower() == "black_hot":
            colored = cv2.cvtColor(255 - normalized, cv2.COLOR_GRAY2BGR)
        else: # white_hot
            colored = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

        return colored

    def detect_hotspots(self, temp_grid: np.ndarray, min_area_px: int = 15) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Detect heat anomalies / hotspots exceeding hotspot_thresh_c.
        Returns a list of hotspot dicts with bbox, center_px, max_temp_c, and average_temp_c.
        """
        hotspot_mask = (temp_grid >= self.hotspot_thresh_c).astype(np.uint8) * 255
        
        # Morphological opening to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hotspot_mask = cv2.morphologyEx(hotspot_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(hotspot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hotspots = []

        h, w = temp_grid.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_px:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            roi_temp = temp_grid[y:y+bh, x:x+bw]
            max_t = float(np.max(roi_temp))
            mean_t = float(np.mean(roi_temp))

            cx = int(x + bw / 2)
            cy = int(y + bh / 2)

            hotspots.append({
                "bbox": [x, y, x + bw, y + bh],
                "center": (cx, cy),
                "center_normalized": (cx / w, cy / h),
                "max_temp_c": round(max_t, 1),
                "mean_temp_c": round(mean_t, 1),
                "area_px": int(area)
            })

        return hotspots, hotspot_mask
