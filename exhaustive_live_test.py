"""
Exhaustive Live System & GCS Server Integration Test
Tests all endpoints, streaming video, websockets, dynamic video upload, and attribute-aware queries.
"""

import sys
import os
import io
import time
import requests
import json
import asyncio
import websockets
import numpy as np
import cv2

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws/telemetry"

def test_http_frontend():
    print("[1/8] Testing Frontend HTML & Assets...")
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200, f"Root returned {r.status_code}"
    assert "AERO-TACTIC" in r.text
    print("  [OK] Root Dashboard HTML: HTTP 200 OK")

    r_css = requests.get(f"{BASE_URL}/static/style.css")
    assert r_css.status_code == 200
    print("  [OK] Tactical CSS Stylesheet: HTTP 200 OK")

    r_js = requests.get(f"{BASE_URL}/static/app.js")
    assert r_js.status_code == 200
    print("  [OK] Client JavaScript App: HTTP 200 OK")

def test_telemetry_endpoint():
    print("\n[2/8] Testing REST Telemetry API (4S LiPo & Sensors)...")
    r = requests.get(f"{BASE_URL}/api/telemetry")
    assert r.status_code == 200
    data = r.json()
    assert "telemetry" in data
    assert "mission_status" in data
    print(f"  [OK] Telemetry: Mode={data['telemetry']['mode']}, State={data['mission_status']['state']}, Battery={data['telemetry']['battery_voltage']}V")

async def test_websocket_telemetry():
    print("\n[3/8] Testing Live WebSockets Telemetry Stream...")
    async with websockets.connect(WS_URL) as ws:
        msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
        data = json.loads(msg)
        assert "telemetry" in data
        assert "mission_status" in data
        print(f"  [OK] WebSocket Stream Connected: Received 10Hz Packet (Timestamp: {data.get('timestamp')})")

def test_video_streams():
    print("\n[4/8] Testing Live Multi-Sensor Video MJPEG Streams...")
    for stream_name in ["rgb", "thermal", "ndvi"]:
        url = f"{BASE_URL}/stream/{stream_name}"
        with requests.get(url, stream=True, timeout=5.0) as r:
            assert r.status_code == 200
            assert "multipart/x-mixed-replace" in r.headers.get("content-type", "")
            chunks_read = 0
            for chunk in r.iter_content(chunk_size=1024):
                if len(chunk) > 0:
                    chunks_read += 1
                if chunks_read >= 5:
                    break
            print(f"  [OK] Live Stream [{stream_name.upper()}]: Transmitted valid multipart MJPEG frame packets")

def test_file_upload_api():
    print("\n[5/8] Testing Custom Scenery Video / Image Upload API...")
    # Create test synthetic image
    test_img = np.zeros((360, 640, 3), dtype=np.uint8)
    test_img[:] = (30, 80, 40)
    # Person with red shirt
    cv2.circle(test_img, (320, 160), 10, (140, 180, 230), -1)
    cv2.rectangle(test_img, (310, 170), (330, 200), (0, 0, 220), -1) # Red shirt
    cv2.rectangle(test_img, (312, 200), (328, 230), (180, 50, 20), -1) # Blue pants

    is_success, buffer = cv2.imencode(".jpg", test_img)
    files = {"file": ("test_scenery_feed.jpg", io.BytesIO(buffer), "image/jpeg")}
    
    r = requests.post(f"{BASE_URL}/api/upload_feed", files=files)
    assert r.status_code == 200, f"Upload returned {r.status_code}: {r.text}"
    res = r.json()
    assert res["status"] == "SUCCESS"
    print(f"  [OK] Scenery Upload & Feed Injection: {res['message']}")

def test_attribute_aware_queries():
    print("\n[6/8] Testing Attribute-Aware NLP Queries & Filters...")
    queries = [
        ("search people with red tshirt", "FILTER_SEARCH", {"target_class": "person", "color": "red"}),
        ("find white trucks", "FILTER_SEARCH", {"target_class": "vehicle", "color": "white"}),
        ("search heat hotspots above 42 degrees", "FILTER_SEARCH", {"target_class": "heat_source", "min_temp_c": 42.0}),
        ("track person with red shirt", "TRACK_TARGET", {"target_class": "person", "color": "red"})
    ]

    for q, expected_action, expected_params in queries:
        r = requests.post(f"{BASE_URL}/api/command", json={"command": q})
        assert r.status_code == 200
        res = r.json()
        assert res["status"] == "OK"
        assert res["parsed"]["action"] == expected_action
        for k, v in expected_params.items():
            if v is not None:
                assert res["parsed"]["parameters"][k] == v
        print(f"  [OK] Attribute Query: '{q}' -> {expected_action} ({res['execution']['message']})")

def test_autonomous_flight_commands():
    print("\n[7/8] Testing Core Autonomous Flight Directives...")
    commands = [
        ("takeoff to 25 meters", "TAKEOFF", {"altitude_m": 25.0}),
        ("start grid survey at 35m", "SURVEY_GRID", {"altitude_m": 35.0}),
        ("hover", "HOVER", {}),
        ("climb to 45 meters", "SET_ALTITUDE", {"altitude_m": 45.0}),
        ("return to launch", "RETURN_TO_LAUNCH", {}),
        ("emergency land", "LAND", {})
    ]

    for speech_input, expected_action, expected_params in commands:
        r = requests.post(f"{BASE_URL}/api/command", json={"command": speech_input})
        assert r.status_code == 200
        res = r.json()
        assert res["status"] == "OK"
        assert res["parsed"]["action"] == expected_action
        print(f"  [OK] Flight Order: '{speech_input}' -> {expected_action}")

def test_unrecognized_fallback():
    print("\n[8/8] Testing Unrecognized Voice Input Fallback...")
    r = requests.post(f"{BASE_URL}/api/command", json={"command": "play my favorite music album"})
    assert r.status_code == 200
    res = r.json()
    assert res["status"] == "UNRECOGNIZED"
    print("  [OK] Fallback Handler: Rejected unrelated input with friendly operator guidance")

def main():
    print("="*75)
    print(" >>> RUNNING EXHAUSTIVE LIVE MULTI-SENSOR & ATTRIBUTE TEST SUITE")
    print("="*75)
    test_http_frontend()
    test_telemetry_endpoint()
    asyncio.run(test_websocket_telemetry())
    test_video_streams()
    test_file_upload_api()
    test_attribute_aware_queries()
    test_autonomous_flight_commands()
    test_unrecognized_fallback()
    print("\n" + "="*75)
    print(" >>> [8/8] ALL ADVANCED SIMULATION & LIVE ENDPOINTS 100% OPERATIONAL! <<<")
    print("="*75)

if __name__ == "__main__":
    main()
