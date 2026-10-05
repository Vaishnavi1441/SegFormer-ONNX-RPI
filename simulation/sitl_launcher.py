"""
ArduPilot SITL (Software In The Loop) Setup & Launcher
Guide for testing autonomous missions on Windows 11 without hardware.
"""

import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SITLLauncher")

SITL_SETUP_GUIDE = """
================================================================================
ARDUPILOT SITL (SOFTWARE IN THE LOOP) SIMULATION INSTRUCTIONS
================================================================================

To run ArduPilot SITL on your Windows 11 development machine (with WSL2):

1. IN WSL2 (Ubuntu):
   $ git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git
   $ cd ardupilot
   $ Tools/environment_install/install-prereqs-ubuntu.sh -y
   $ . ~/.profile
   $ cd ArduCopter
   $ sim_vehicle.py -v ArduCopter --console --map --out=127.0.0.1:14550

2. CONNECTING THE GCS & COMPANION DAEMON:
   The MAVLink bridge connects to 'udp:127.0.0.1:14550'.
   All Guided mode commands (takeoff, survey, visual tracking, RTL) will 
   control the virtual 10-inch quadcopter in real-time.

3. OPTIONAL: CONNECT MISSION PLANNER ON WINDOWS:
   Open Mission Planner -> Connect via UDP port 14550 / 14551 to view 
   the 3D flight instrumentation side-by-side with our custom Web GCS!
================================================================================
"""

def main():
    print(SITL_SETUP_GUIDE)

if __name__ == "__main__":
    main()
