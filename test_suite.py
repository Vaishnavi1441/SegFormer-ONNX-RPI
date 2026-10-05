"""
UAV Autonomous Scanner Comprehensive Unit Test Suite
"""

import numpy as np
import cv2
import os
from onboard_pi.drivers.thermal_radiometric import ThermalRadiometricProcessor
from onboard_pi.vision.segmentor import TerrainSegmentor
from onboard_pi.vision.geotag import GeoTagger
from onboard_pi.autonomy.guidance import LawnmowerSurveyPlanner
from gcs.backend.nlp_parser import UAVCommandParser

def test_nlp_parser():
    parser = UAVCommandParser()
    p1 = parser.parse("takeoff to 30 meters")
    assert p1["action"] == "TAKEOFF"
    assert p1["parameters"]["altitude_m"] == 30.0

    p2 = parser.parse("start grid survey at 40m")
    assert p2["action"] == "SURVEY_GRID"
    assert p2["parameters"]["altitude_m"] == 40.0

    p3 = parser.parse("search people with red tshirt")
    assert p3["action"] == "FILTER_SEARCH"
    assert p3["parameters"]["target_class"] == "person"
    assert p3["parameters"]["color"] == "red"

    p4 = parser.parse("return home")
    assert p4["action"] == "RETURN_TO_LAUNCH"
    print("[PASSED] NLP Intent Parser Test")

def test_thermal_processor():
    proc = ThermalRadiometricProcessor(hotspot_thresh_c=35.0)
    raw = np.full((120, 160), 30500, dtype=np.uint16)
    celsius = proc.raw_to_celsius(raw)
    assert abs(celsius[0, 0] - 31.85) < 0.1

    raw[40:50, 40:50] = 32000
    celsius_hot = proc.raw_to_celsius(raw)
    hotspots, mask = proc.detect_hotspots(celsius_hot, min_area_px=10)
    assert len(hotspots) >= 1
    print("[PASSED] Thermal Radiometric Processor Test")

def test_terrain_segmentor_ndvi():
    seg = TerrainSegmentor()
    ms_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    ms_frame[:, :, 0] = 220 # NIR (Ch 0)
    ms_frame[:, :, 1] = 120 # Green (Ch 1)
    ms_frame[:, :, 2] = 30  # Red (Ch 2)

    ndvi = seg.compute_ndvi(ms_frame)
    assert ndvi.shape == (100, 100)
    assert np.mean(ndvi) > 0.65

    # Test SegFormer Neural Network Land Cover Segmentation on Real Scenery
    test_img_path = "data/test_user_scenery_1.jpg"
    if os.path.exists(test_img_path):
        real_scenery = cv2.imread(test_img_path)
    else:
        real_scenery = np.zeros((240, 320, 3), dtype=np.uint8)
        real_scenery[:] = (40, 120, 30)

    blended, stats = seg.segment_land_cover(real_scenery)
    assert "building_pct" in stats
    assert "road_pct" in stats
    assert "vegetation_pct" in stats
    assert "water_pct" in stats
    assert abs(sum(stats[f"{k}_pct"] for k in ["building", "road", "sidewalk", "vehicle", "person", "vegetation", "water", "bare_ground", "sky", "other"]) - 100.0) < 1.0
    print("[PASSED] Terrain Segmentor NDVI & Deep Learning Land Cover Test")

def test_geotagging():
    gt = GeoTagger(fov_horizontal_deg=75.0, image_width=640, image_height=360, camera_pitch_mount_deg=-45.0)
    lat, lon = gt.pixel_to_gps(
        px=320, py=180,
        uav_lat=37.7749, uav_lon=-122.4194,
        uav_alt_agl_m=30.0,
        uav_heading_deg=0.0
    )
    assert lat > 37.7749
    print("[PASSED] GeoTagger 45° Oblique Perspective Math Test")

def test_survey_planner():
    waypoints = LawnmowerSurveyPlanner.generate_grid(
        center_lat=37.7749, center_lon=-122.4194,
        width_m=100.0, height_m=100.0,
        altitude_m=30.0, grid_spacing_m=20.0
    )
    assert len(waypoints) >= 4
    for wp in waypoints:
        assert wp["alt_m"] == 30.0
    print("[PASSED] Lawnmower Survey Planner Test")

def main():
    test_nlp_parser()
    test_thermal_processor()
    test_terrain_segmentor_ndvi()
    test_geotagging()
    test_survey_planner()
    print("\n" + "="*55)
    print(" >>>  ALL UNIT TEST SUITES PASSED SUCCESSFULLY! <<<")
    print("="*55)

if __name__ == "__main__":
    main()
