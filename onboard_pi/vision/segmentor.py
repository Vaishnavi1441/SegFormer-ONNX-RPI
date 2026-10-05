"""
Terrain & Multispectral Segmentation Engine
Computes real-time NDVI (Vegetation Index), NDWI (Water Index), and Multi-Class Semantic Land Cover
(Forest, Grassland, Water, Beach/Sand, Mountain/Rock, Roads, Clouds/Snow).
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple
from onboard_pi.vision.terrain_classifier import TerrainClassifier

class TerrainSegmentor:
    def __init__(self, nir_ch: int = 0, green_ch: int = 1, red_ch: int = 2):
        self.nir_ch = nir_ch
        self.green_ch = green_ch
        self.red_ch = red_ch
        self.classifier = TerrainClassifier()

    def compute_ndvi(self, ms_frame: np.ndarray) -> np.ndarray:
        """Compute Normalized Difference Vegetation Index: NDVI = (NIR - Red) / (NIR + Red)"""
        nir = ms_frame[:, :, self.nir_ch].astype(np.float32)
        red = ms_frame[:, :, self.red_ch].astype(np.float32)

        denom = nir + red
        denom[denom == 0] = 1e-5

        ndvi = (nir - red) / denom
        return np.clip(ndvi, -1.0, 1.0)

    def compute_ndwi(self, ms_frame: np.ndarray) -> np.ndarray:
        """Compute Normalized Difference Water Index: NDWI = (Green - NIR) / (Green + NIR)"""
        nir = ms_frame[:, :, self.nir_ch].astype(np.float32)
        green = ms_frame[:, :, self.green_ch].astype(np.float32)

        denom = green + nir
        denom[denom == 0] = 1e-5

        ndwi = (green - nir) / denom
        return np.clip(ndwi, -1.0, 1.0)

    def colorize_ndvi(self, ndvi_grid: np.ndarray) -> np.ndarray:
        """Colorize NDVI grid for GCS display using Turbo colormap."""
        scaled = ((ndvi_grid + 1.0) * 127.5).astype(np.uint8)
        colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        return colored

    def segment_land_cover(self, frame_bgr: np.ndarray, alpha: float = 0.5) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Segment real-world scenery into 7 distinct semantic terrain classes.
        Returns:
            - blended_overlay: Drone video with colored semi-transparent semantic mask
            - stats: Complete land cover percentages
        """
        blended, stats, _ = self.classifier.segment_frame(frame_bgr, alpha=alpha)
        return blended, stats
