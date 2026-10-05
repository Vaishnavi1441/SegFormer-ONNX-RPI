"""
Attribute-Aware Object Detection & Fine-Grained Filtering Engine
Performs fine-grained color segmentation, clothing/region extraction, and natural-language
attribute matching (e.g., "search people with red tshirt", "find white trucks", "track blue car").
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("AttributeDetector")

# Color ranges in HSV space: (Hue 0-180, Sat 0-255, Val 0-255)
COLOR_HSV_RANGES = {
    "red": [
        (np.array([0, 80, 50]), np.array([10, 255, 255])),
        (np.array([170, 80, 50]), np.array([180, 255, 255]))
    ],
    "blue": [
        (np.array([95, 80, 50]), np.array([130, 255, 255]))
    ],
    "green": [
        (np.array([35, 70, 50]), np.array([85, 255, 255]))
    ],
    "yellow": [
        (np.array([20, 80, 80]), np.array([35, 255, 255]))
    ],
    "orange": [
        (np.array([10, 100, 100]), np.array([22, 255, 255]))
    ],
    "white": [
        (np.array([0, 0, 190]), np.array([180, 45, 255]))
    ],
    "black": [
        (np.array([0, 0, 0]), np.array([180, 255, 55]))
    ]
}

class AttributeAwareDetector:
    def __init__(self, base_detector=None):
        self.base_detector = base_detector

    def extract_color_attributes(self, crop_img: np.ndarray, region_type: str = "general") -> Dict[str, float]:
        """
        Analyze the HSV color breakdown within a cropped object region.
        For people: upper 45% represents shirts/jackets.
        """
        if crop_img.size == 0:
            return {}

        h, w = crop_img.shape[:2]
        if region_type == "upper_body" and h > 15:
            # Crop upper 45% (torso / shirt)
            sample_roi = crop_img[:int(h * 0.45), :]
        else:
            sample_roi = crop_img

        hsv = cv2.cvtColor(sample_roi, cv2.COLOR_BGR2HSV)
        total_pixels = max(1, sample_roi.shape[0] * sample_roi.shape[1])

        color_scores = {}
        for color_name, ranges in COLOR_HSV_RANGES.items():
            mask_combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                mask_combined |= mask

            color_pixel_count = np.sum(mask_combined > 0)
            percentage = color_pixel_count / total_pixels
            if percentage > 0.08: # At least 8% dominant in region
                color_scores[color_name] = round(float(percentage), 2)

        return color_scores

    def match_detection_attributes(self, 
                                   detections: List[Dict[str, Any]], 
                                   frame_bgr: np.ndarray, 
                                   query_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Augment detections with detected attributes (colors, shirt/torso tags) and
        mark 'is_matched: True/False' based on query filter.
        query_filter: e.g. {'target_class': 'person', 'color': 'red', 'item': 'tshirt'}
        """
        if frame_bgr is None:
            return detections

        h_img, w_img = frame_bgr.shape[:2]
        annotated_detections = []

        target_cls = query_filter.get("target_class", "").lower() if query_filter else ""
        target_color = query_filter.get("color", "").lower() if query_filter else ""

        for det in detections:
            det_copy = dict(det)
            x1, y1, x2, y2 = det["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            crop = frame_bgr[y1:y2, x1:x2]
            det_cls = det.get("class", "object").lower()

            region_mode = "upper_body" if det_cls == "person" else "general"
            color_breakdown = self.extract_color_attributes(crop, region_type=region_mode)

            # Determine dominant color
            dominant_color = "unknown"
            if color_breakdown:
                dominant_color = max(color_breakdown.items(), key=lambda item: item[1])[0]

            det_copy["attributes"] = {
                "dominant_color": dominant_color,
                "color_breakdown": color_breakdown,
                "region": region_mode
            }

            # Evaluate filter match
            is_match = True
            if target_cls and target_cls not in det_cls and det_cls not in target_cls:
                is_match = False
            if target_color and target_color not in color_breakdown:
                is_match = False

            det_copy["is_matched"] = is_match
            if is_match and query_filter:
                det_copy["display_label"] = f"[MATCH] {det_cls.upper()} ({dominant_color})"
            else:
                det_copy["display_label"] = f"{det_cls.upper()} ({dominant_color})"

            annotated_detections.append(det_copy)

        return annotated_detections
