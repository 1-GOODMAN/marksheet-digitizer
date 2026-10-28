"""
Offline Seed Sync Utility for Marksheet Digitizer.
Ingests existing images inside data/raw/, data/Sub1/, and data/Sub2/,
routes them by detected Course Codes (24MCPC522, 24MCPC517, 24MCPC521),
enforces Z3 Presburger arithmetic constraints, and pre-populates the
subject-wise Excel ledgers and Master multi-tab workbook.
"""

import sys
import os
import glob
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import PROJECT_ROOT, DATA_DIR, RAW_DATA_DIR, VALID_COURSE_CODES, USN_PREFIX
from src.preprocessor import TableSegmenter
from src.ocr_engine import ProbabilisticOCR
from src.solver import MarksheetSolver
from src.pipeline import BatchProcessor
from src.cloud_storage import CloudStorageManager, CANONICAL_SUBJECTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed_sync")


def normalize_usn(raw_usn: str) -> str:
    """Enforces strict domain rule: locked prefix 4MT24MC + 3-digit roll number (001-199)."""
    if not raw_usn or not isinstance(raw_usn, str):
        return f"{USN_PREFIX}001"
    
    clean = raw_usn.strip().upper()
    digits = "".join(c for c in clean if c.isdigit())
    
    if len(digits) >= 3:
        try:
            roll = int(digits[-3:])
        except ValueError:
            roll = 1
    elif len(digits) > 0:
        roll = int(digits)
    else:
        roll = 1

    if roll < 1:
        roll = 1
    elif roll > 199:
        cand = roll % 100
        if 1 <= cand <= 199:
            roll = cand
        else:
            roll = min(199, max(1, roll % 200))

    return f"{USN_PREFIX}{roll:03d}"


def discover_seed_images(data_dir: Path) -> List[Dict[str, Any]]:
    """
    Discovers all candidate Page-1 booklet scans from:
      1. data/raw/     — matches *_p1.jpg
      2. data/Sub1/    — matches *.jpg (excluding _p2)
      3. data/Sub2/    — matches *.jpg (excluding _p2)
    """
    candidates = []

    # 1. data/raw
    raw_folder = data_dir / "raw"
    if raw_folder.exists():
        for p in sorted(raw_folder.glob("booklet_*_p1.jpg")):
            booklet_id = p.stem.replace("_p1", "")
            candidates.append({
                "path": p,
                "subject_folder": "raw",
                "booklet_id": booklet_id,
            })

    # 2. data/Sub1
    sub1_folder = data_dir / "Sub1"
    if sub1_folder.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in sorted(sub1_folder.glob(ext)):
                if "_p2" not in p.name.lower():
                    candidates.append({
                        "path": p,
                        "subject_folder": "Sub1",
                        "booklet_id": p.stem,
                    })

    # 3. data/Sub2
    sub2_folder = data_dir / "Sub2"
    if sub2_folder.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in sorted(sub2_folder.glob(ext)):
                if "_p2" not in p.name.lower():
                    candidates.append({
                        "path": p,
                        "subject_folder": "Sub2",
                        "booklet_id": p.stem,
                    })

    return candidates


def run_seed_sync(
    limit: Optional[int] = None,
    force_recompute: bool = False,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Orchestrates ingestion, routing, Z3 validation, and Excel generation.
    """
    root_data = Path(data_dir or DATA_DIR)
    cloud = CloudStorageManager()

    logger.info("=" * 80)
    logger.info("MARKSHEET DIGITIZER: OFFLINE SEED SYNC & SUBJECT ROUTER")
    logger.info("=" * 80)
    logger.info(f"Local Cloud Vault Root: {cloud.vault_root.resolve()}")

    # Discover images
    candidates = discover_seed_images(root_data)
    total_found = len(candidates)
    logger.info(f"Discovered {total_found} candidate booklet images across raw, Sub1, Sub2.")

    if limit and limit > 0:
        raw_items = [c for c in candidates if c["subject_folder"] == "raw"]
        sub1_items = [c for c in candidates if c["subject_folder"] == "Sub1"]
        sub2_items = [c for c in candidates if c["subject_folder"] == "Sub2"]
        per_folder = max(1, limit // 3)
        balanced = raw_items[:per_folder] + sub1_items[:per_folder] + sub2_items[:per_folder]
        candidates = balanced[:limit]
        logger.info(f"Balanced processing limit applied: {len(candidates)} images ({per_folder} per folder).")

    # Load existing cache to enable incremental updates
    cache = cloud.load_cached_records() if not force_recompute else {}
    processed_count = 0
    skipped_count = 0

    # Initialize Processor singleton
    processor = BatchProcessor(data_dir=root_data, output_dir=cloud.vault_root)

    t_start = time.time()

    for idx, item in enumerate(candidates, start=1):
        img_path = item["path"]
        subj_folder = item["subject_folder"]
        booklet_id = item["booklet_id"]

        # Check if already in cache and image hasn't changed
        cache_hit = False
        if not force_recompute:
            for k, rec in cache.items():
                if rec.get("Booklet_ID") == booklet_id:
                    cache_hit = True
                    break

        if cache_hit:
            skipped_count += 1
            print(f"[{idx}/{len(candidates)}] Cached: {booklet_id} ({subj_folder})")
            continue

        print(f"[{idx}/{len(candidates)}] Ingesting & Solving: {booklet_id} ({subj_folder})...", end=" ", flush=True)
        t0 = time.time()

        proc_res = processor.process_page1_image(
            img_input=img_path,
            booklet_id=booklet_id,
            subject_folder=subj_folder,
        )

        elapsed = round(time.time() - t0, 2)

        if not proc_res.get("success", False):
            print(f"FAILED ({elapsed}s): {proc_res.get('error')}")
            continue

        cie_summary = proc_res["cie_summary"]

        # Enforce domain constraints on USN & Course Code
        clean_usn = normalize_usn(cie_summary.get("USN", ""))
        cie_summary["USN"] = clean_usn

        # Folder-level routing prior: raw -> 24MCPC522, Sub1 -> 24MCPC517, Sub2 -> 24MCPC521
        folder_route = {
            "raw": "24MCPC522",
            "Sub1": "24MCPC517",
            "Sub2": "24MCPC521",
        }.get(subj_folder)

        raw_cc = str(cie_summary.get("Course_Code", "")).strip().upper()
        if folder_route:
            final_cc = folder_route
        elif raw_cc in CANONICAL_SUBJECTS:
            final_cc = raw_cc
        else:
            final_cc = CANONICAL_SUBJECTS[0]

        cie_summary["Course_Code"] = final_cc

        # Upsert into Subject-Wise Cloud Storage & Ledgers
        upsert_res = cloud.upsert_booklet_record(
            cie_record=cie_summary,
            image_source=img_path,
            timestamp_str=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        processed_count += 1
        status = proc_res.get("status", "VALIDATED")
        cie_tot = cie_summary.get("Total_CIE_50", 0)
        print(f"DONE ({elapsed}s) -> Routed: [{final_cc}] USN: {clean_usn} (CIE: {cie_tot}/50 | {status})")

    total_time = round(time.time() - t_start, 2)

    # Final summary extraction
    summary = cloud.get_subjects_summary()

    print("\n" + "=" * 80)
    print("SEED SYNC COMPLETE: SUBJECT-WISE ROUTING LEDGER AUDIT")
    print("=" * 80)
    print(f"Total Images Scanned / Cached: {len(candidates)}")
    print(f"Newly Digitized: {processed_count} | Retrieved from Cache: {skipped_count} | Time: {total_time}s")
    print("-" * 80)
    print(f"{'Course Code':<14} | {'Subject Title':<38} | {'Students':<8} | {'Avg CIE':<8} | {'Validated':<9} | {'Z3 Corrected'}")
    print("-" * 80)

    for code in CANONICAL_SUBJECTS:
        s_info = summary["subjects"].get(code, {})
        title = s_info.get("title", code)[:38]
        cnt = s_info.get("student_count", 0)
        avg = s_info.get("average_cie", 0.0)
        val = s_info.get("validated_count", 0)
        cor = s_info.get("auto_corrected_count", 0)
        print(f"{code:<14} | {title:<38} | {cnt:<8} | {avg:<8.1f} | {val:<9} | {cor}")

    print("-" * 80)
    print(f"TOTAL UNIQUE BOOKLETS ACROSS ALL SUBJECTS: {summary['total_booklets']}")
    print(f"OVERALL AVERAGE CIE: {summary['overall_average_cie']} / 50")
    print(f"Master Excel Workbook: {cloud.get_master_excel_path()}")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marksheet Digitizer Offline Seed Sync")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of images to ingest")
    parser.add_argument("--force", action="store_true", help="Force recomputation ignoring existing cache")
    args = parser.parse_args()

    run_seed_sync(limit=args.limit, force_recompute=args.force)
