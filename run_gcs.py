"""
UAV Ground Control Station Launcher
Run with: python run_gcs.py
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Auto-re-exec with project virtualenv if needed
venv_python = os.path.join(PROJECT_ROOT, "venv", "bin", "python")
if os.path.exists(venv_python) and sys.executable != venv_python:
    try:
        import uvicorn
        import fastapi
    except ImportError:
        os.execv(venv_python, [venv_python] + sys.argv)

import uvicorn
from gcs.backend.gcs_server import app

if __name__ == "__main__":
    print("\n=======================================================")
    print(" >>> STARTING UAV GROUND CONTROL STATION (GCS)...    <<<")
    print(" >>> Open Browser: http://localhost:8000             <<<")
    print(" >>> Or:          http://127.0.0.1:8000             <<<")
    print("=======================================================\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
