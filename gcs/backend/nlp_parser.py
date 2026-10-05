"""
NLP Intent Parser & Speech Command Compiler for UAV GCS
Translates natural speech or typed user commands into structured autonomous UAV action payloads
with support for fine-grained multi-attribute queries (e.g. "search people with red tshirt", "find white trucks").
"""

import re
import logging
from typing import Dict, Any

logger = logging.getLogger("NLPParser")

class UAVCommandParser:
    def __init__(self):
        pass

    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse raw speech/text input and extract action + parameters.
        Returns: {
            "raw_text": str,
            "action": str,
            "parameters": dict,
            "confidence": float,
            "feedback_msg": str
        }
        """
        cleaned = text.strip().lower()
        logger.info(f"Parsing Command: '{cleaned}'")

        # 1. Return to Launch / Go Home
        if re.search(r'\b(?:return|go\s+home|come\s+back|rtl|rth|back\s+to\s+base)\b', cleaned):
            return {
                "raw_text": text,
                "action": "RETURN_TO_LAUNCH",
                "parameters": {},
                "confidence": 0.98,
                "feedback_msg": "Command received: Executing Return-to-Launch (RTL)."
            }

        # 2. Land
        if re.search(r'\b(?:land|touch\s*down)\b', cleaned):
            return {
                "raw_text": text,
                "action": "LAND",
                "parameters": {},
                "confidence": 0.96,
                "feedback_msg": "Command received: Landing immediately in place."
            }

        # 3. Takeoff Commands
        match_takeoff = re.search(r'\b(?:take\s*off|takeoff|launch|ascend\s+to\s+start)\b(?:\s+(?:to|at)?\s*(\d+(?:\.\d+)?)\s*(?:m|meters|meter)?)?', cleaned)
        if match_takeoff:
            alt = 25.0
            if match_takeoff.group(1):
                alt = float(match_takeoff.group(1))
            return {
                "raw_text": text,
                "action": "TAKEOFF",
                "parameters": {"altitude_m": alt},
                "confidence": 0.95,
                "feedback_msg": f"Command received: Taking off to {alt} meters altitude."
            }

        # 4. Attribute-Filtered Search (e.g., "search people with red tshirt", "find white trucks", "detect solar panels")
        if re.search(r'\b(?:search|find|detect|look\s+for|highlight|filter|locate)\b', cleaned) and not re.search(r'\b(?:grid|survey|mapping)\b', cleaned):
            color = self._extract_color(cleaned)
            target_cls = self._extract_class(cleaned)
            
            temp_match = re.search(r'(?:above|over|exceeding|\>)\s*(\d+(?:\.\d+)?)\s*(?:c|degrees|deg|celsius)?', cleaned)
            min_temp = float(temp_match.group(1)) if temp_match else None

            params = {
                "target_class": target_cls,
                "color": color,
                "min_temp_c": min_temp
            }
            desc = f"{color or ''} {target_cls}".strip()
            return {
                "raw_text": text,
                "action": "FILTER_SEARCH",
                "parameters": params,
                "confidence": 0.94,
                "feedback_msg": f"Visual Filter Active: Overlaying and prioritizing targets matching '{desc}'."
            }

        # 5. Grid Survey / Terrain Scanning Commands
        if re.search(r'\b(?:survey|scan|mapping|grid\s*scan)\b', cleaned):
            alt = 35.0
            width = 100.0
            height = 100.0
            alt_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:m|meters|meter)', cleaned)
            if alt_match:
                alt = float(alt_match.group(1))
            
            return {
                "raw_text": text,
                "action": "SURVEY_GRID",
                "parameters": {"altitude_m": alt, "width_m": width, "height_m": height},
                "confidence": 0.92,
                "feedback_msg": f"Command received: Commencing autonomous terrain survey grid at {alt}m altitude."
            }

        # 6. Object / Target Tracking Commands (with optional attributes, e.g. "track person with red shirt")
        if re.search(r'\b(?:follow|pursue|lock\s+onto)\b', cleaned) or re.search(r'\btrack\s+(?:the\s+|a\s+)?(?:target|object|vehicle|car|truck|person|human|man|woman|heat|thermal|hotspot|livestock|animal|boat)', cleaned) or cleaned.startswith("track "):
            target_cls = self._extract_class(cleaned)
            color = self._extract_color(cleaned)

            params = {"target_class": target_cls, "color": color}
            desc = f"{color or ''} {target_cls}".strip()
            return {
                "raw_text": text,
                "action": "TRACK_TARGET",
                "parameters": params,
                "confidence": 0.94,
                "feedback_msg": f"Command received: Autonomous visual tracking engaged for '{desc}'."
            }

        # 7. Hover / Hold Position
        if re.search(r'\b(?:hover|hold|pause|stay|stop)\b', cleaned):
            return {
                "raw_text": text,
                "action": "HOVER",
                "parameters": {},
                "confidence": 0.90,
                "feedback_msg": "Command received: Holding position (Loiter)."
            }

        # 8. Altitude Adjustment
        alt_change = re.search(r'\b(?:climb|descend|go|change\s+altitude)\b\s+(?:to\s+)?(\d+(?:\.\d+)?)\s*(?:m|meters)?', cleaned)
        if alt_change:
            target_alt = float(alt_change.group(1))
            return {
                "raw_text": text,
                "action": "SET_ALTITUDE",
                "parameters": {"altitude_m": target_alt},
                "confidence": 0.93,
                "feedback_msg": f"Command received: Adjusting altitude to {target_alt} meters."
            }

        # Unknown / Unparsed Fallback
        return {
            "raw_text": text,
            "action": "UNKNOWN",
            "parameters": {},
            "confidence": 0.0,
            "feedback_msg": f"Could not recognize command: '{text}'. Try 'search people with red tshirt', 'takeoff', 'start survey', 'follow vehicle', or 'return home'."
        }

    def _extract_color(self, text: str) -> str:
        for c in ["red", "blue", "green", "yellow", "orange", "white", "black"]:
            if re.search(rf'\b{c}\b', text):
                return c
        return ""

    def _extract_class(self, text: str) -> str:
        if re.search(r'\b(?:person|human|man|woman|people|persons|tshirt|shirt)s?\b', text):
            return "person"
        elif re.search(r'\b(?:thermal|heat|hotspot)s?\b', text):
            return "heat_source"
        elif re.search(r'\b(?:livestock|animal|cattle|deer|cow|sheep)s?\b', text):
            return "livestock"
        elif re.search(r'\b(?:boat|ship|vessel)s?\b', text):
            return "boat"
        elif re.search(r'\b(?:solar|solar\s*panel)s?\b', text):
            return "solar_panel"
        elif re.search(r'\b(?:water|river|lake|pond)s?\b', text):
            return "water"
        elif re.search(r'\b(?:vehicle|car|truck|suv|van|automobile)s?\b', text):
            return "vehicle"
        return "object"
