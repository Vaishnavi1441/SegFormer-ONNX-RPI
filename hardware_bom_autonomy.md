# Hardware Component Architecture & Bill of Materials (BOM)

To achieve **full level-4/level-5 UAV autonomy**, **precision terrain following**, **360-degree obstacle avoidance**, and **GPS-denied navigation** on your 10-inch quadcopter with Raspberry Pi 5 and ArduPilot/Pixhawk:

---

## 1. Core Flight & Power System
| Component | Recommended Model | Specifications & Role | Interface |
| :--- | :--- | :--- | :--- |
| **Airframe** | 10-Inch Quadcopter Frame (e.g. Tarot / Holybro X500 / Martian) | 450mm–500mm wheelbase, carbon fiber, vibration isolation damping plate | - |
| **Motors & Props** | Sunnysky / T-Motor 2814 or 3110 (700–900KV) + 10x4.5 Carbon/Nylon Props | High-efficiency lift for ~2.2–2.8kg All-Up-Weight (AUW) | PWM / DShot |
| **ESCs** | 4-in-1 45A–55A BLHeli_32 / BLHeli_S | Low internal resistance, DShot600 telemetry capable | Pixhawk I/O |
| **Battery** | **14.8V 5200mAh 40C 4S LiPo** *(Your Spec)* | ~18–25 minutes payload flight endurance; 16.8V max -> 14.8V nominal -> 13.6V critical RTL | XT60 / XT90 |
| **Power Module** | Holybro PM02D / Mauch Power Module | Accurate current & voltage sensing for ArduPilot battery failsafe | I2C / ADC |
| **Companion Power Regulator** | 5V 5A High-Power Step-Down UBEC | Converts 14.8V (4S) directly to stable 5.1V @ 5A for Raspberry Pi 5 | Pi 5 USB-C / GPIO 5V |

---

## 2. Autonomy, Obstacle Avoidance & Rangefinding Suite
| Purpose | Recommended Sensor | Features | Connection |
| :--- | :--- | :--- | :--- |
| **Downwards Precision Terrain Altitude (AGL)** | **Benewake TFmini-S / TF-Luna (0.1m–12m)** or **LightWare SF11/c (0.1m–100m)** | High-precision laser rangefinder for ground clearance, auto-takeoff, and precision terrain following | Pixhawk I2C or UART (`SERIAL4`) |
| **360° Horizontal Obstacle Avoidance** | **Slamtec RPLIDAR S2** or **8x Benewake TF-Luna Ring Array** | 360-degree 2D planar LiDAR (30m range) communicating via MAVLink `DISTANCE_SENSOR` with ArduPilot **BendyRuler / Dijkstra Path Planning** | Pi 5 USB / Pixhawk UART |
| **GPS-Denied Precision Hover & Optical Flow** | **Holybro PMW3901** or **Matek 3901-L0X Optical Flow + LiDAR** | Optical surface velocity tracking for rock-solid indoor / tree-canopy hover without GPS | Pixhawk SPI / I2C |
| **GNSS & Heading** | **Holybro Micro M9N GPS + Compass** *(Pixhawk GPS)* | Multi-constellation (GPS, GLONASS, Galileo, BeiDou) + internal IST8310 magnetometer | Pixhawk `GPS1` + `I2C` |

---

## 3. Triple Camera Payload & Gimbal Stabilization
| Payload | Hardware Model | Specs & Capabilities | Interface |
| :--- | :--- | :--- | :--- |
| **RGB Optical Camera** | **Raspberry Pi Camera Module 3 (IMX708)** or **Sony IMX477 (12.3MP HQ)** | 1080p/720p 30-60FPS, autofocus/global shutter, 75°–120° FOV | Pi 5 CSI-2 Ribbon Cable |
| **Thermal Radiometric Camera** | **FLIR Lepton 3.5 + PureThermal 2/3 Board** or **Infiray T2-Search** | 160x120 radiometric thermal LWIR ($8\mu m-14\mu m$), $-10^\circ\text{C}$ to $140^\circ\text{C}$ spot temperature | Pi 5 USB UVC |
| **Multispectral Camera** | **Mapir Survey3 (R-G-NIR)** or **MicaSense RedEdge-MX** | Calibrated Near-Infrared (850nm), Green (550nm), Red (660nm) for NDVI | Pi 5 USB / CSI |
| **Camera Gimbal** | **Tarot T-2D / 3-Axis Brushless Gimbal** | Active roll/pitch brushless stabilization to ensure nadir vertical orientation during wind gusts | Pixhawk PWM Servo Gimbal Output |

---

## 4. Communication & Safety Hierarchy
| Link | Equipment | Frequency & Range | Function |
| :--- | :--- | :--- | :--- |
| **Pilot Manual RC Override** | **R12 Receiver** *(Your Spec)* -> Pixhawk RC IN | 2.4GHz / Long-range (SBUS/CRSF) | **Master Safety Override**: Instantly disengages companion control if transmitter switch is toggled. |
| **Telemetry & GCS Data Link** | **915MHz / 433MHz 1000mW Telemetry Radios** (SiK / Holybro) or **5GHz Wi-Fi / LTE 4G Modem Hat (Sixfab)** | 2km–10km RF / Global LTE | Transmits live GCS telemetry, speech commands, AI detection geolocations, and video. |
