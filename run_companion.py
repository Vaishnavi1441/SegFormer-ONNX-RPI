"""
Onboard UAV Companion Daemon Launcher
Run with: python run_companion.py
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
        import pymavlink
        import cv2
        import onnxruntime
    except ImportError:
        os.execv(venv_python, [venv_python] + sys.argv)

from onboard_pi.main_agent import UAVAutonomousDaemon

if __name__ == "__main__":
    print("\n=======================================================")
    print(" >>> STARTING AUTONOMOUS UAV ONBOARD DAEMON...       <<<")
    print("=======================================================\n")
    daemon = UAVAutonomousDaemon()
    daemon.start()
