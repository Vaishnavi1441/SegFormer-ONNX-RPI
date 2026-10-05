"""City-aware semantic segmentation for the UAV scanner.

Primary path: SegFormer-B0 ADE20K exported to ONNX (CPU-friendly and Raspberry Pi 5
compatible through ONNX Runtime).  The model is downloaded separately by
models/download_models.py so the source tree stays small.

The network predicts 150 ADE20K classes.  We collapse them into an explicit UAV
city/terrain ontology instead of assigning unknown classes to an arbitrary class.
"""
import logging
import os
import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("CitySemanticSegmentor")

UAV_CLASSES = [
    "building", "road", "sidewalk", "vehicle", "person", "vegetation",
    "water", "bare_ground", "sky", "other"
]

COLORS_BGR = {
    "building": (70, 70, 190),
    "road": (75, 75, 75),
    "sidewalk": (130, 130, 130),
    "vehicle": (40, 170, 230),
    "person": (180, 70, 190),
    "vegetation": (40, 170, 55),
    "water": (210, 100, 35),
    "bare_ground": (80, 150, 205),
    "sky": (235, 190, 90),
    "other": (100, 100, 100),
}

# ADE20K class index -> UAV ontology.  Names are the standard ADE20K 150-class order.
ADE20K_NAMES = [
    "wall","building","sky","floor","tree","ceiling","road","bed","windowpane","grass",
    "cabinet","sidewalk","person","earth","door","table","mountain","plant","curtain","chair",
    "car","water","painting","sofa","shelf","house","sea","mirror","rug","field",
    "armchair","seat","fence","desk","rock","wardrobe","lamp","bathtub","railing","cushion",
    "base","box","column","signboard","chest of drawers","counter","sand","sink","skyscraper","fireplace",
    "refrigerator","grandstand","path","stairs","runway","case","pool table","pillow","screen door","stairway",
    "river","bridge","bookcase","blind","coffee table","toilet","flower","book","hill","bench",
    "countertop","stove","palm","kitchen island","computer","swivel chair","boat","bar","arcade machine","hovel",
    "bus","towel","light","truck","tower","chandelier","awning","streetlight","booth","television receiver",
    "airplane","dirt track","apparel","pole","land","bannister","escalator","ottoman","bottle","buffet",
    "poster","stage","van","ship","fountain","conveyer belt","canopy","washer","plaything","swimming pool",
    "stool","barrel","basket","waterfall","tent","bag","minibike","cradle","oven","ball",
    "food","step","tank","trade name","microwave","pot","animal","bicycle","lake","dishwasher",
    "screen","blanket","sculpture","hood","sconce","vase","traffic light","tray","ashcan","fan",
    "pier","crt screen","plate","monitor","bulletin board","shower","radiator","glass","clock","flag"
]

ADE20K_TO_UAV = {}
for name in ADE20K_NAMES:
    target = "other"
    if name in {"wall", "building", "house", "skyscraper", "tower", "bridge", "fence", "roof"}:
        target = "building"
    elif name in {"road", "path", "dirt track", "runway", "streetlight", "traffic light", "pole", "bannister", "stairs", "stairway", "escalator"}:
        target = "road"
    elif name in {"sidewalk", "floor", "step"}:
        target = "sidewalk"
    elif name in {"car", "bus", "truck", "van", "minibike", "bicycle", "boat", "ship", "airplane"}:
        target = "vehicle"
    elif name in {"person"}:
        target = "person"
    elif name in {"tree", "grass", "plant", "flower", "palm", "field", "canopy"}:
        target = "vegetation"
    elif name in {"water", "sea", "river", "lake", "fountain", "swimming pool", "waterfall", "pier"}:
        target = "water"
    elif name in {"earth", "sand", "land", "rock", "mountain", "hill"}:
        target = "bare_ground"
    elif name in {"sky", "cloud"}:
        target = "sky"
    ADE20K_TO_UAV[name] = target


def _build_lut() -> np.ndarray:
    lut = np.full(150, UAV_CLASSES.index("other"), dtype=np.uint8)
    for i, name in enumerate(ADE20K_NAMES):
        lut[i] = UAV_CLASSES.index(ADE20K_TO_UAV.get(name, "other"))
    return lut


class TerrainClassifier:
    """Semantic city/terrain segmentation with an honest no-model fallback."""

    def __init__(self, default_alpha: float = 0.50, model_path: Optional[str] = None,
                 input_size: int = 512, max_inference_hz: float = 3.0):
        self.alpha = float(np.clip(default_alpha, 0.0, 1.0))
        self.input_size = int(input_size)
        self.max_inference_hz = float(max_inference_hz)
        self._lock = threading.Lock()
        self._session = None
        self._input_name = None
        self._last_run = 0.0
        self._last_result = None
        self.model_path = model_path or os.getenv(
            "UAV_SEGMENTATION_MODEL",
            os.path.join(os.path.dirname(__file__), "../../models/segformer_b0_ade20k_quantized.onnx")
        )
        self._lut = _build_lut()
        self._init_onnx()

    def _init_onnx(self):
        if not os.path.exists(self.model_path):
            logger.warning("Segmentation model not found at %s. Run models/download_models.py.", self.model_path)
            return
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
            logger.info("Loaded city segmentation model: %s", self.model_path)
        except Exception as exc:
            logger.exception("Could not load ONNX segmentation model: %s", exc)
            self._session = None

    @property
    def model_loaded(self) -> bool:
        return self._session is not None

    def set_alpha(self, alpha: float):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        x = rgb.astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return np.transpose(x, (2, 0, 1))[None, ...]

    def _infer_labels(self, frame_bgr: np.ndarray) -> np.ndarray:
        inp = self._preprocess(frame_bgr)
        outputs = self._session.run(None, {self._input_name: inp})
        logits = np.asarray(outputs[0])
        if logits.ndim == 4:
            labels_small = np.argmax(logits, axis=1)[0].astype(np.uint8)
        elif logits.ndim == 3:
            labels_small = np.argmax(logits, axis=0).astype(np.uint8)
        else:
            raise RuntimeError(f"Unexpected segmentation output shape: {logits.shape}")
        return cv2.resize(labels_small, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    def _fallback(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Conservative visual partition used only when no neural model is installed.
        It never fabricates percentages: uncertain pixels remain 'other'."""
        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lab = np.full((h, w), UAV_CLASSES.index("other"), dtype=np.uint8)
        # Sky: upper image, low texture and blue/cyan bias.
        upper = np.indices((h, w))[0] < int(h * 0.38)
        sky = upper & (hsv[..., 2] > 120) & (hsv[..., 1] < 150)
        lab[sky] = UAV_CLASSES.index("sky")
        # Vegetation: green dominance + saturation.
        b, g, r = cv2.split(frame_bgr.astype(np.int16))
        veg = (g > r + 10) & (g > b + 5) & (g > 55)
        lab[veg] = UAV_CLASSES.index("vegetation")
        # Water: blue dominance, moderate/high saturation.
        water = (b > r + 15) & (b > g + 5) & (hsv[..., 1] > 50)
        lab[water] = UAV_CLASSES.index("water")
        # Road-like neutral dark regions; deliberately conservative.
        road = (hsv[..., 1] < 55) & (hsv[..., 2] < 125) & (np.indices((h, w))[0] > h * 0.25)
        lab[road] = UAV_CLASSES.index("road")
        return lab

    def segment_frame(self, frame_bgr: np.ndarray, alpha: Optional[float] = None):
        if frame_bgr is None:
            return frame_bgr, {}, None
        h, w = frame_bgr.shape[:2]
        now = time.monotonic()
        min_dt = 1.0 / max(0.1, self.max_inference_hz)

        with self._lock:
            needs_infer = (self._session is not None and 
                           (now - self._last_run >= min_dt or self._last_result is None or self._last_result.shape[:2] != (h, w)))
            if needs_infer:
                try:
                    self._last_result = self._infer_labels(frame_bgr)
                    self._last_run = now
                except Exception as exc:
                    logger.error("Segmentation inference failed: %s", exc)

            if self._last_result is not None:
                if self._last_result.shape[:2] != (h, w):
                    self._last_result = cv2.resize(self._last_result, (w, h), interpolation=cv2.INTER_NEAREST)
                semantic = self._lut[self._last_result]
                model_status = "segformer-b0-ade20k"
            else:
                semantic = self._fallback(frame_bgr)
                model_status = "fallback"

        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        stats = {f"{c}_pct": 0.0 for c in UAV_CLASSES}
        total_pixels = float(h * w) if (h * w) > 0 else 1.0

        for idx, cls in enumerate(UAV_CLASSES):
            mask = (semantic == idx)
            count = int(np.count_nonzero(mask))
            if count > 0:
                color_mask[mask] = COLORS_BGR[cls]
                stats[f"{cls}_pct"] = round((count / total_pixels) * 100.0, 1)

        stats["model"] = model_status
        stats["segmentation_confidence"] = "model" if self._session is not None else "fallback"
        blend_alpha = self.alpha if alpha is None else float(np.clip(alpha, 0, 1))
        blended = cv2.addWeighted(frame_bgr, 1.0 - blend_alpha, color_mask, blend_alpha, 0)
        cv2.rectangle(blended, (0, 0), (w, 32), (15, 18, 24), -1)
        text = (f"CITY SEGMENTATION [{model_status}]  "
                f"Bldg {stats['building_pct']}% | Road {stats['road_pct']}% | "
                f"Veg {stats['vegetation_pct']}% | Water {stats['water_pct']}%")
        cv2.putText(blended, text[:145], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 240, 245), 1, cv2.LINE_AA)
        return blended, stats, color_mask
