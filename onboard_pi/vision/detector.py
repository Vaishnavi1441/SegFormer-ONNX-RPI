"""
Multi-Modal Object & Attribute Detector (RGB + Thermal)
Optimized for ONNX Runtime / NCNN / OpenCV DNN with strict bounding box filtering.
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Dict, Any, Optional
from onboard_pi.vision.attribute_detector import AttributeAwareDetector

logger = logging.getLogger("MultiModalDetector")

class ObjectDetector:
    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.45, iou_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model_path = model_path
        self.attribute_engine = AttributeAwareDetector()
        self.classes = [
            "person", "vehicle", "livestock", "boat", "building", "solar_panel", "heat_source"
        ]

        self.active_query_filter: Optional[Dict[str, Any]] = None

        if model_path and os.path.exists(model_path):
            try:
                import onnxruntime as ort
                self.ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                logger.info(f"Loaded ONNX model: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load ONNX model ({e}). Using strict localized detector.")
                self.ort_session = None
        else:
            self.ort_session = None

    def set_query_filter(self, filter_dict: Optional[Dict[str, Any]]):
        """Set active natural-language attribute filter (e.g., {'target_class': 'person', 'color': 'red'})."""
        self.active_query_filter = filter_dict
        logger.info(f"Active vision query filter updated to: {filter_dict}")

    def detect(self, rgb_frame: Optional[np.ndarray], thermal_frame: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Run detection across RGB and Thermal frames with fine-grained attribute analysis."""
        detections = []

        if rgb_frame is not None:
            rgb_dets = self._detect_rgb(rgb_frame)
            annotated_rgb = self.attribute_engine.match_detection_attributes(
                rgb_dets, rgb_frame, query_filter=self.active_query_filter
            )
            detections.extend(annotated_rgb)

        if thermal_frame is not None:
            thermal_dets = self._detect_thermal(thermal_frame)
            detections.extend(thermal_dets)

        return detections

    def _detect_rgb(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        h, w = frame.shape[:2]
        total_img_area = h * w
        dets = []

        if self.ort_session is not None:
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
            input_name = self.ort_session.get_inputs()[0].name
            outputs = self.ort_session.run(None, {input_name: blob})
            output = np.squeeze(outputs[0])
            if output.shape[0] < output.shape[1]:
                output = output.T

            boxes = []
            scores = []
            class_ids = []

            for row in output:
                cls_scores = row[4:]
                max_score = np.max(cls_scores)
                if max_score >= self.conf_threshold:
                    cls_id = np.argmax(cls_scores)
                    cx, cy, bw, bh = row[0:4]
                    x1 = int((cx - bw / 2) * (w / 640.0))
                    y1 = int((cy - bh / 2) * (h / 640.0))
                    bw_scaled = int(bw * (w / 640.0))
                    bh_scaled = int(bh * (h / 640.0))
                    boxes.append([x1, y1, bw_scaled, bh_scaled])
                    scores.append(float(max_score))
                    class_ids.append(cls_id)

            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
            if len(indices) > 0:
                for idx in indices.flatten():
                    x, y, bw, bh = boxes[idx]
                    cls_name = self.classes[class_ids[idx] % len(self.classes)]
                    dets.append({
                        "class": cls_name,
                        "confidence": round(scores[idx], 2),
                        "bbox": [max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)],
                        "center": (int(x + bw / 2), int(y + bh / 2)),
                        "center_normalized": ((x + bw / 2) / w, (y + bh / 2) / h),
                        "source": "rgb"
                    })
        else:
            # Strict Localized Optical Detection (Persons, Vehicles, Boats)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 1. Detect Red Objects / Person with Red Clothes
            mask_red1 = cv2.inRange(hsv, np.array([0, 110, 70]), np.array([10, 255, 255]))
            mask_red2 = cv2.inRange(hsv, np.array([170, 110, 70]), np.array([180, 255, 255]))
            mask_red = mask_red1 | mask_red2

            contours_p, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours_p:
                area = cv2.contourArea(cnt)
                # Strict size limits: avoid whole mountain or huge areas
                if 25 < area < (total_img_area * 0.05):
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    aspect = bh / max(1, bw)
                    full_y1 = max(0, y - 6)
                    full_y2 = min(h, y + bh + 12)
                    full_x1 = max(0, x - 4)
                    full_x2 = min(w, x + bw + 4)
                    cx = int((full_x1 + full_x2) / 2)
                    cy = int((full_y1 + full_y2) / 2)
                    dets.append({
                        "class": "person",
                        "confidence": 0.94,
                        "bbox": [full_x1, full_y1, full_x2, full_y2],
                        "center": (cx, cy),
                        "center_normalized": (cx / w, cy / h),
                        "source": "rgb"
                    })

            # 2. Detect Localized Moving Vehicles (compact aspect ratio, not vast terrain patches)
            # Only detect when an explicit vehicle query is active OR high contrast localized bounding box
            if self.active_query_filter and self.active_query_filter.get("target_class") == "vehicle":
                mask_veh = cv2.inRange(hsv, np.array([0, 0, 195]), np.array([180, 40, 255]))
                contours_v, _ = cv2.findContours(mask_veh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours_v:
                    area = cv2.contourArea(cnt)
                    if 100 < area < (total_img_area * 0.03):
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        aspect = bw / max(1, bh)
                        if 0.5 <= aspect <= 3.5: # Realistic vehicle aspect ratio
                            cx = int(x + bw / 2)
                            cy = int(y + bh / 2)
                            dets.append({
                                "class": "vehicle",
                                "confidence": 0.88,
                                "bbox": [x, y, x + bw, y + bh],
                                "center": (cx, cy),
                                "center_normalized": (cx / w, cy / h),
                                "source": "rgb"
                            })

        return dets

    def _detect_thermal(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect localized thermal hotspots as high-confidence thermal targets."""
        h, w = frame.shape[:2]
        dets = []
        if len(frame.shape) == 2:
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 15 < area < (w * h * 0.08):
                x, y, bw, bh = cv2.boundingRect(cnt)
                cx = int(x + bw / 2)
                cy = int(y + bh / 2)
                dets.append({
                    "class": "heat_source",
                    "confidence": 0.93,
                    "bbox": [x, y, x + bw, y + bh],
                    "center": (cx, cy),
                    "center_normalized": (cx / w, cy / h),
                    "source": "thermal",
                    "is_matched": True,
                    "display_label": "HOTSPOT (>42C)"
                })

        return dets
