"""
End-to-End Test for Ground Control Station (GCS) API & Command Dispatch
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from gcs.backend.gcs_server import app

def test_gcs_api_endpoints():
    client = TestClient(app)

    # 1. Test Root endpoint (loads HTML dashboard)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "AERO-TACTIC" in resp.text
    print("[PASSED] GCS Frontend HTML Serving Test")

    # 2. Test Telemetry endpoint
    resp = client.get("/api/telemetry")
    assert resp.status_code == 200
    data = resp.json()
    assert "telemetry" in data
    assert "mission_status" in data
    print("[PASSED] GCS Telemetry API Test")

    # 3. Test Voice/Text Command API: Takeoff
    resp = client.post("/api/command", json={"command": "take off to 35 meters"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "OK"
    assert data["parsed"]["action"] == "TAKEOFF"
    assert data["parsed"]["parameters"]["altitude_m"] == 35.0
    print("[PASSED] Voice Command Execution: 'take off to 35 meters'")

    # 4. Test Voice/Text Command API: Survey Grid
    resp = client.post("/api/command", json={"command": "start grid survey at 40m"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "OK"
    assert data["parsed"]["action"] == "SURVEY_GRID"
    print("[PASSED] Voice Command Execution: 'start grid survey at 40m'")

    # 5. Test Voice/Text Command API: Follow Vehicle
    resp = client.post("/api/command", json={"command": "follow the red vehicle"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "OK"
    assert data["parsed"]["action"] == "TRACK_TARGET"
    print("[PASSED] Voice Command Execution: 'follow the red vehicle'")

    # 6. Test Voice/Text Command API: Return Home
    resp = client.post("/api/command", json={"command": "return home"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "OK"
    assert data["parsed"]["action"] == "RETURN_TO_LAUNCH"
    print("[PASSED] Voice Command Execution: 'return home'")

    print("\n=======================================================")
    print(" >>>  ALL GCS API & COMMAND TESTS PASSED!          <<<")
    print("=======================================================")

if __name__ == "__main__":
    test_gcs_api_endpoints()
