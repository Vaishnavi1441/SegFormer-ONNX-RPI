"""
Ground Control Station (GCS) Backend Server
Provides FastAPI REST API, WebSockets Telemetry Relay, Multi-Camera MJPEG Video Streaming,
Dual Raw/Segmented Feeds, Multi-Level Terrain Classification, and Dynamic Alpha Transparency.
"""

import sys
import os
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import asyncio
import time
import json
import logging
import cv2
import numpy as np
from typing import Dict, Any, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

from gcs.backend.nlp_parser import UAVCommandParser
from onboard_pi.drivers.mavlink_bridge import MAVLinkBridge
from onboard_pi.drivers.camera_hub import CameraHub
from onboard_pi.drivers.thermal_radiometric import ThermalRadiometricProcessor
from onboard_pi.vision.segmentor import TerrainSegmentor
from onboard_pi.vision.terrain_classifier import TerrainClassifier
from onboard_pi.vision.detector import ObjectDetector
from onboard_pi.vision.geotag import GeoTagger
from onboard_pi.autonomy.mission_executive import MissionExecutive

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("GCSServer")

app = FastAPI(title="Autonomous UAV Ground Control Station")

UPLOADS_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Core UAV Perception & Guidance Subsystems
mavlink = MAVLinkBridge("udp:127.0.0.1:14550")
mavlink.connect()

cam_config = {
    "rgb_camera": {"width": 640, "height": 360, "fps": 30, "fov_horizontal_deg": 75.0},
    "multispectral_camera": {"width": 640, "height": 360, "fps": 15},
    "thermal_camera": {"width": 320, "height": 240, "fps": 15},
    "simulation": {"use_mock_cameras": True}
}
camera_hub = CameraHub(cam_config)
camera_hub.start_all()

default_video_path = os.path.join(UPLOADS_DIR, "Anjuna Lake  Drone View  Goa_1080p.mp4")
if os.path.exists(default_video_path):
    logger.info(f"Auto-loading default drone video: {default_video_path}")
    camera_hub.load_custom_feed_file(default_video_path)

thermal_processor = ThermalRadiometricProcessor()
terrain_classifier = TerrainClassifier(default_alpha=0.5)
segmentor = TerrainSegmentor()
detector = ObjectDetector()
geotagger = GeoTagger(fov_horizontal_deg=75.0, image_width=640, image_height=360, camera_pitch_mount_deg=-45.0)
mission_executive = MissionExecutive(mavlink, geotagger, detector=detector)
mission_executive.start()

nlp_parser = UAVCommandParser()

perception_mode = "TERRAIN"

class CommandRequest(BaseModel):
    command: str

class CameraAngleRequest(BaseModel):
    angle_deg: float

class PerceptionModeRequest(BaseModel):
    mode: str # "TERRAIN", "OBJECTS", "COMBINED"

class AlphaRequest(BaseModel):
    alpha: float # 0.0 to 1.0

active_connections: List[WebSocket] = []

@app.post("/api/command")
async def execute_command(req: CommandRequest):
    """Parse speech/text command and execute on the UAV."""
    parsed = nlp_parser.parse(req.command)
    logger.info(f"GCS Received: '{req.command}' -> Parsed: {parsed['action']} (Params: {parsed['parameters']})")
    
    if parsed["action"] != "UNKNOWN":
        result = mission_executive.process_command(parsed)
        return {
            "status": "OK",
            "parsed": parsed,
            "execution": result
        }
    else:
        return {
            "status": "UNRECOGNIZED",
            "parsed": parsed,
            "execution": {"status": "ERROR", "message": parsed["feedback_msg"]}
        }

@app.post("/api/set_camera_angle")
async def set_camera_angle(req: CameraAngleRequest):
    """Set camera mount pitch angle (e.g. -45.0 for oblique or -90.0 for nadir)."""
    mission_executive.set_camera_mount_angle(req.angle_deg)
    return {
        "status": "OK",
        "camera_mount_angle_deg": req.angle_deg,
        "mode": "45° OBLIQUE" if abs(req.angle_deg - (-45.0)) < 1.0 else "90° NADIR"
    }

@app.post("/api/set_perception_mode")
async def set_perception_mode(req: PerceptionModeRequest):
    """Switch perception mode between TERRAIN, OBJECTS, and COMBINED."""
    global perception_mode
    mode_clean = req.mode.upper()
    if mode_clean in ["TERRAIN", "OBJECTS", "COMBINED"]:
        perception_mode = mode_clean
        logger.info(f"Perception mode switched to: {perception_mode}")
        return {"status": "OK", "mode": perception_mode}
    return {"status": "ERROR", "message": f"Invalid mode: {req.mode}"}

@app.post("/api/set_terrain_alpha")
async def set_terrain_alpha(req: AlphaRequest):
    """Set semantic terrain mask opacity (0.0 to 1.0)."""
    terrain_classifier.set_alpha(req.alpha)
    return {"status": "OK", "alpha": req.alpha}

class FeedSelectRequest(BaseModel):
    filename: str

@app.get("/api/list_feeds")
async def list_available_feeds():
    """List all available uploaded and preset drone videos and images."""
    files = []
    if os.path.exists(UPLOADS_DIR):
        for f in sorted(os.listdir(UPLOADS_DIR)):
            if not f.startswith(".") and f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png")):
                files.append(f)
    return {"status": "OK", "feeds": files}

@app.post("/api/select_feed")
async def select_preset_feed(req: FeedSelectRequest):
    """Switch active perception stream to an existing file in data/uploads."""
    target_path = os.path.join(UPLOADS_DIR, req.filename)
    if os.path.exists(target_path):
        logger.info(f"Switching feed to: {target_path}")
        camera_hub.load_custom_feed_file(target_path)
        return {
            "status": "SUCCESS",
            "filename": req.filename,
            "message": f"Successfully activated feed '{req.filename}'"
        }
    return JSONResponse(status_code=404, content={"status": "ERROR", "message": f"File '{req.filename}' not found."})

@app.post("/api/upload_feed")
async def upload_custom_feed(file: UploadFile = File(...)):
    """Upload custom real-world aerial scenery video (MP4/AVI) or image."""
    try:
        clean_filename = os.path.basename(file.filename)
        dest_path = os.path.join(UPLOADS_DIR, clean_filename)
        
        contents = await file.read()
        with open(dest_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"Uploaded custom feed ({len(contents)} bytes): {dest_path}. Injecting into perception engine...")
        camera_hub.load_custom_feed_file(dest_path)
        
        return {
            "status": "SUCCESS",
            "filename": clean_filename,
            "message": f"Successfully loaded '{clean_filename}' into multi-sensor AI perception pipeline."
        }
    except Exception as e:
        logger.error(f"Failed to process uploaded feed: {e}")
        return JSONResponse(status_code=500, content={"status": "ERROR", "message": str(e)})

@app.get("/api/telemetry")
async def get_telemetry():
    """Fetch instantaneous UAV telemetry and mission state."""
    telem = mavlink.get_telemetry()
    status = mission_executive.get_status()
    return {
        "telemetry": telem,
        "mission_status": status,
        "perception_mode": perception_mode
    }

@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """High-rate WebSocket streaming telemetry, terrain scan progress & AI detections."""
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            telem = mavlink.get_telemetry()
            status = mission_executive.get_status()
            
            payload = {
                "telemetry": telem,
                "mission_status": status,
                "perception_mode": perception_mode,
                "timestamp": asyncio.get_event_loop().time()
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

# --- VIDEO STREAM GENERATORS ---

def gen_raw_frames():
    """Pristine raw original camera/video footage without any overlays."""
    while True:
        try:
            frame = camera_hub.get_frame("rgb")
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logger.debug(f"Raw frame error: {e}")
        time.sleep(0.033)

def gen_terrain_seg_frames():
    """Live Multi-Level Semantic Terrain Segmentation Overlay Stream."""
    while True:
        try:
            frame = camera_hub.get_frame("rgb")
            if frame is not None:
                blended_terrain, terrain_stats, _ = terrain_classifier.segment_frame(frame)
                mission_executive.update_terrain_stats(terrain_stats)

                ret, buffer = cv2.imencode('.jpg', blended_terrain, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logger.debug(f"Terrain frame error: {e}")
        time.sleep(0.033)

def gen_rgb_frames():
    """RGB Optical Feed with Object Detection Bounding Boxes."""
    while True:
        try:
            frame = camera_hub.get_frame("rgb")
            if frame is not None:
                dets = detector.detect(frame)
                telem = mavlink.get_telemetry()
                mission_executive.update_vision_detections(dets, telem)
                
                _, terrain_stats, _ = terrain_classifier.segment_frame(frame)
                mission_executive.update_terrain_stats(terrain_stats)

                for d in dets:
                    if d.get("source") == "rgb":
                        x1, y1, x2, y2 = d["bbox"]
                        is_matched = d.get("is_matched", True)
                        box_color = (0, 255, 0) if is_matched else (120, 120, 120)
                        thickness = 2 if is_matched else 1

                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
                        label = d.get("display_label", f"{d['class']} ({int(d['confidence']*100)}%)")
                        
                        if is_matched:
                            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + len(label) * 10, y1), (0, 180, 0), -1)
                            cv2.putText(frame, label, (x1 + 4, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
                        else:
                            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logger.debug(f"RGB frame error: {e}")
        time.sleep(0.033)

def gen_combined_frames():
    """Combined Stream: Multi-Level Terrain Segmentation + Object Detections."""
    while True:
        try:
            frame = camera_hub.get_frame("rgb")
            if frame is not None:
                blended_terrain, terrain_stats, _ = terrain_classifier.segment_frame(frame, alpha=0.38)
                mission_executive.update_terrain_stats(terrain_stats)

                dets = detector.detect(frame)
                telem = mavlink.get_telemetry()
                mission_executive.update_vision_detections(dets, telem)

                for d in dets:
                    if d.get("source") == "rgb":
                        x1, y1, x2, y2 = d["bbox"]
                        is_matched = d.get("is_matched", True)
                        box_color = (0, 255, 0) if is_matched else (120, 120, 120)
                        cv2.rectangle(blended_terrain, (x1, y1), (x2, y2), box_color, 2)
                        label = d.get("display_label", f"{d['class']} ({int(d['confidence']*100)}%)")
                        if is_matched:
                            cv2.rectangle(blended_terrain, (x1, max(0, y1 - 22)), (x1 + len(label) * 10, y1), (0, 180, 0), -1)
                            cv2.putText(blended_terrain, label, (x1 + 4, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

                ret, buffer = cv2.imencode('.jpg', blended_terrain, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logger.debug(f"Combined frame error: {e}")
        time.sleep(0.033)

def gen_thermal_frames():
    while True:
        try:
            frame = camera_hub.get_frame("thermal")
            if frame is not None:
                temp_grid = thermal_processor.raw_to_celsius(frame)
                colorized = thermal_processor.celsius_to_colorized(temp_grid, "ironbow")
                hotspots, _ = thermal_processor.detect_hotspots(temp_grid)
                
                for hs in hotspots:
                    x1, y1, x2, y2 = hs["bbox"]
                    cv2.rectangle(colorized, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(colorized, f"{hs['max_temp_c']}C", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                ret, buffer = cv2.imencode('.jpg', colorized, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logger.debug(f"Thermal frame error: {e}")
        time.sleep(0.05)

def gen_ndvi_frames():
    while True:
        try:
            frame = camera_hub.get_frame("multispectral")
            if frame is not None:
                ndvi = segmentor.compute_ndvi(frame)
                colored_ndvi = segmentor.colorize_ndvi(ndvi)
                ret, buffer = cv2.imencode('.jpg', colored_ndvi, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logger.debug(f"NDVI frame error: {e}")
        time.sleep(0.05)

@app.get("/stream/raw")
def stream_raw():
    return StreamingResponse(gen_raw_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stream/terrain_seg")
def stream_terrain_seg():
    return StreamingResponse(gen_terrain_seg_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stream/rgb")
def stream_rgb():
    return StreamingResponse(gen_rgb_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stream/combined")
def stream_combined():
    return StreamingResponse(gen_combined_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stream/thermal")
def stream_thermal():
    return StreamingResponse(gen_thermal_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stream/ndvi")
def stream_ndvi():
    return StreamingResponse(gen_ndvi_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def get_root():
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h1>UAV Ground Control Station Backend Running</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
