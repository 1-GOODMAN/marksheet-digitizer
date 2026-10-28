"""
Configuration constants for Marksheet Digitizer.
Includes OCR model paths, beam search sizes, institutional priors, and visual heuristic thresholds.
"""

from pathlib import Path
import os

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CROPS_DIR = DATA_DIR / "crops"
OUTPUT_DIR = DATA_DIR / "output"

# Multi-subject input directories (relative names under DATA_DIR)
SUBJECT_DIRS = ["raw", "Sub1", "Sub2"]

# OCR Model settings
OCR_MODEL_NAME = os.getenv("OCR_MODEL_NAME", "microsoft/trocr-small-handwritten")
BEAM_SIZE = int(os.getenv("BEAM_SIZE", 4))

# Fast Heuristic Thresholds for Blank / Struck-out Cells
BLANK_STD_THRESHOLD = 4.5
BLANK_MEAN_BRIGHTNESS_THRESHOLD = 245.0

# Smudge / Scribble / Cross-out strike detector
STRIKE_OUT_MAX_BRIGHTNESS = 60.0
STRIKE_OUT_AREA_RATIO = 0.40

# Default confidence assigned to strike-outs and blanks
BLANK_PROBABILITY = 1.0
STRIKE_OUT_PROBABILITY = 0.95

# Institutional Priors: USN and Course Code
USN_PREFIX = "4MT24MC"
USN_PREFIX_LEN = len(USN_PREFIX)  # 7
USN_TOTAL_LEN = 10
VALID_COURSE_CODES = ["24MCPC522", "24MCPC517", "24MCPC521"]

# Default course code mapping per subject folder
FOLDER_COURSE_CODES = {
    "raw": "24MCPC522",
    "Sub1": "24MCPC517",
    "sub1": "24MCPC517",
    "Sub2": "24MCPC521",
    "sub2": "24MCPC521",
}

# Char-level confusion map used in USN suffix decoding
USN_CHAR_CONFUSION = {
    "O": "0", "D": "0", "Q": "0", "o": "0",
    "I": "1", "L": "1", "l": "1", "|": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "G": "6", "g": "6",
    "B": "8", "b": "8", "P": "8", "p": "8",
}
