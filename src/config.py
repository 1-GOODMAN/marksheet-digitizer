"""
Configuration constants for Marksheet Digitizer.
Includes OCR model paths, beam search sizes, institutional priors, and visual heuristic thresholds.
"""

from pathlib import Path
from typing import Optional, List, Dict
import os

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CROPS_DIR = DATA_DIR / "crops"
OUTPUT_DIR = DATA_DIR / "output"

def discover_subject_dirs(data_dir: Optional[Path] = None) -> List[str]:
    """
    Dynamically discovers all subdirectories inside data/ that contain images
    (e.g., raw, Sub1, Sub2, Sub3, Sub4, etc.).
    Excludes internal output/cache directories.
    """
    d_dir = Path(data_dir or DATA_DIR)
    if not d_dir.exists():
        return []

    ignored_names = {
        "crops", "output", "cloud_vault", "cloud_bucket", "debug",
        "debug_sub1_sub2", "test_crops", "__pycache__", ".git"
    }
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

    discovered = []
    for item in sorted(d_dir.iterdir()):
        if item.is_dir() and item.name not in ignored_names and not item.name.startswith("."):
            try:
                has_images = any(
                    f.is_file() and f.suffix.lower() in image_exts
                    for f in item.iterdir()
                )
                if has_images:
                    discovered.append(item.name)
            except Exception:
                continue

    return discovered

# Dynamically discovered input directories (re-evaluated on call or initialized here)
SUBJECT_DIRS = discover_subject_dirs()

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
VALID_COURSE_CODES = ["24MCPC522", "24MCPC521"]

# Default course code mapping per subject folder
FOLDER_COURSE_CODES = {
    "raw": "24MCPC522",
    "Sub2": "24MCPC521",
    "sub2": "24MCPC521",
    "24MCPC522": "24MCPC522",
    "24MCPC521": "24MCPC521",
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


def resolve_subject_from_filename(filename: str, data_dir: Optional[Path] = None) -> Optional[str]:
    """
    Checks if filename exists in data/Sub2/ (-> 24MCPC521) or data/raw/ (-> 24MCPC522),
    and uses that as a strong prior alongside visual OCR.
    Also handles stem matching and substring hints.
    """
    if not filename:
        return None
    d_dir = Path(data_dir or DATA_DIR)
    name = Path(filename).name.lower()
    stem = Path(filename).stem.lower()

    # 0. Check screenshot of Sub2 booklet
    if "203006" in name or "screenshot" in name:
        return "24MCPC521"

    # 1. Check Sub2 directory
    sub2_dir = d_dir / "Sub2"
    if sub2_dir.exists():
        for f in sub2_dir.iterdir():
            if f.name.lower() == name or f.stem.lower() == stem:
                return "24MCPC521"

    # 2. Check raw directory
    raw_dir = d_dir / "raw"
    if raw_dir.exists():
        for f in raw_dir.iterdir():
            if f.name.lower() == name or f.stem.lower() == stem:
                return "24MCPC522"

    # 3. Substring hints in filename
    if "sub2" in name or "521" in name:
        return "24MCPC521"
    if "raw" in name or "522" in name or "booklet" in name:
        return "24MCPC522"

    return None

    return None

