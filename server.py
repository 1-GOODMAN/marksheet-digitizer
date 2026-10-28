"""
FastAPI Cloud Backend Service for Marksheet Digitizer.
Continuous Smartphone Batch-Scanning Cloud System with Automated
Subject-Wise Routing (24MCPC522, 24MCPC517, 24MCPC521) and Ledger Organization.
"""

import os
import json
import time
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import PROJECT_ROOT, DATA_DIR, RAW_DATA_DIR, OUTPUT_DIR, VALID_COURSE_CODES, USN_PREFIX
from src.preprocessor import TableSegmenter
from src.ocr_engine import ProbabilisticOCR
from src.solver import MarksheetSolver
from src.pipeline import BatchProcessor
from src.stitcher import DualPageStitcher
from src.cloud_storage import CloudStorageManager, CANONICAL_SUBJECTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("marksheet_server")

# -----------------------------------------------------------------------------
# Directory Initialization
# -----------------------------------------------------------------------------
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"
CLOUD_VAULT_DIR = PROJECT_ROOT / "cloud_vault"
EXCEL_LEDGER_PATH = CLOUD_VAULT_DIR / "exports" / "master_all_subjects_marksheet.xlsx"

TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
CLOUD_VAULT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# FastAPI App & Middleware
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Marksheet Digitizer Cloud API",
    description="Continuous Smartphone Batch-Scanning Cloud System with Automated Subject-Wise Routing & Ledger Organization",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mounts
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/cloud-vault", StaticFiles(directory=str(CLOUD_VAULT_DIR)), name="cloud_vault")
# Backward-compatibility alias
app.mount("/cloud-bucket", StaticFiles(directory=str(CLOUD_VAULT_DIR)), name="cloud_bucket")

if RAW_DATA_DIR.exists():
    app.mount("/raw-samples", StaticFiles(directory=str(RAW_DATA_DIR)), name="raw_samples")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# -----------------------------------------------------------------------------
# Service Singletons
# -----------------------------------------------------------------------------
_processor: Optional[BatchProcessor] = None
_stitcher = DualPageStitcher(jpeg_quality=92)
_cloud = CloudStorageManager(cloud_vault_root=CLOUD_VAULT_DIR)


def get_processor() -> BatchProcessor:
    """Returns singleton BatchProcessor (TrOCR + TableSegmenter + MarksheetSolver)."""
    global _processor
    if _processor is None:
        logger.info("Initializing BatchProcessor (TableSegmenter + TrOCR + Z3 Solver)...")
        _processor = BatchProcessor(
            data_dir=DATA_DIR,
            output_dir=CLOUD_VAULT_DIR,
            ocr=ProbabilisticOCR(),
            segmenter=TableSegmenter(),
            solver=MarksheetSolver(),
        )
    return _processor


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


# -----------------------------------------------------------------------------
# Web Portal
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def faculty_portal(request: Request):
    """Serves the Continuous Smartphone Batch-Scanning & Subject Routing Web Portal."""
    return templates.TemplateResponse(request, "index.html", {})


# -----------------------------------------------------------------------------
# System Status & Diagnostics
# -----------------------------------------------------------------------------
@app.get("/api/status")
async def system_status():
    """Returns backend health, active hardware device, cloud storage diagnostics, and totals."""
    proc = get_processor()
    cloud_info = _cloud.get_status()
    summary = _cloud.get_subjects_summary()

    return {
        "status": "online",
        "device": str(proc.ocr.device).upper(),
        "ocr_model": "microsoft/trocr-small-handwritten",
        "solver": "Z3 MaxSMT Presburger (Arithmetic Verification)",
        "cloud": cloud_info,
        "total_records": summary["total_booklets"],
        "subjects": list(summary["subjects"].keys()),
    }


# -----------------------------------------------------------------------------
# Endpoint 1: Continuous Smartphone Batch-Upload & Subject-Wise Router
# -----------------------------------------------------------------------------
@app.post("/api/booklets/batch-upload")
async def batch_upload_endpoint(
    files: Optional[List[UploadFile]] = File(default=None),
    photos: Optional[List[UploadFile]] = File(default=None),
    page1: Optional[UploadFile] = File(default=None),
    sample_ids: Optional[str] = Form(default=None),
    forced_course_code: Optional[str] = Form(default=None),
    subject_folder: Optional[str] = Form(default="live_mobile"),
):
    """
    Continuous Smartphone Batch-Upload & Automatic Subject Routing:
      - Ingests single or multiple smartphone photos (list[UploadFile] or sample_ids).
      - Applies optional forced_course_code override (24MCPC522, 24MCPC517, 24MCPC521).
      - Runs TableSegmenter -> ProbabilisticOCR -> MarksheetSolver (Z3 C1, C2, C3).
      - Enforces strict USN domain rule: 4MT24MC + 3-digit roll number (001-199).
      - Routes scans to: cloud_vault/subjects/{course_code}/scans/{usn}_{timestamp}.jpg.
      - Deduplicates & upserts in ascending USN order into subject and master Excel workbooks.
    """
    t0 = time.time()
    processor = get_processor()

    # Collect all image sources
    upload_items: List[Dict[str, Any]] = []

    # Check file lists
    all_incoming_files = (files or []) + (photos or [])
    if page1 is not None:
        all_incoming_files.append(page1)

    for up_file in all_incoming_files:
        if up_file and up_file.filename:
            raw_bytes = await up_file.read()
            if len(raw_bytes) > 0:
                upload_items.append({
                    "type": "bytes",
                    "data": raw_bytes,
                    "filename": up_file.filename,
                    "booklet_id": Path(up_file.filename).stem,
                })

    # Check sample_ids if any (comma-separated or single)
    if sample_ids:
        ids = [s.strip() for s in sample_ids.split(",") if s.strip()]
        for s_id in ids:
            p1_path = RAW_DATA_DIR / f"{s_id}_p1.jpg"
            if not p1_path.exists():
                p1_path = RAW_DATA_DIR / f"{s_id}.jpg"
            if p1_path.exists():
                upload_items.append({
                    "type": "file_path",
                    "data": p1_path,
                    "filename": p1_path.name,
                    "booklet_id": s_id,
                })

    if not upload_items:
        raise HTTPException(
            status_code=400,
            detail="No valid marksheet photos or benchmark sample IDs provided for upload.",
        )

    # Validate forced_course_code override
    active_forced_code = None
    if forced_course_code and forced_course_code.strip():
        f_clean = forced_course_code.strip().upper()
        if f_clean in CANONICAL_SUBJECTS:
            active_forced_code = f_clean

    processed_results: List[Dict[str, Any]] = []

    for item in upload_items:
        b_id = item["booklet_id"]
        ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Decode image
        if item["type"] == "bytes":
            nparr = np.frombuffer(item["data"], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            raw_image_source = item["data"]
        else:
            img = cv2.imread(str(item["data"]))
            raw_image_source = item["data"]

        if img is None:
            logger.warning(f"Failed to decode image: {item['filename']}")
            continue

        # Neuro-symbolic extraction
        proc_res = processor.process_page1_image(
            img_input=img,
            booklet_id=b_id,
            subject_folder=subject_folder or "live_mobile",
        )

        if not proc_res.get("success", False):
            logger.warning(f"Processing failed for {b_id}: {proc_res.get('error')}")
            continue

        cie_summary = proc_res["cie_summary"]

        # 1. Subject Routing Override
        if active_forced_code:
            cie_summary["Course_Code"] = active_forced_code
        else:
            raw_cc = str(cie_summary.get("Course_Code", CANONICAL_SUBJECTS[0])).strip().upper()
            if raw_cc not in CANONICAL_SUBJECTS:
                raw_cc = CANONICAL_SUBJECTS[0]
            cie_summary["Course_Code"] = raw_cc

        final_course_code = cie_summary["Course_Code"]

        # 2. Strict USN Domain Rule (4MT24MC + 3 digits 001-199)
        final_usn = normalize_usn(cie_summary.get("USN", ""))
        cie_summary["USN"] = final_usn

        # 3. Cloud Archival & Atomic Ledger Upsert
        upsert_res = _cloud.upsert_booklet_record(
            cie_record=cie_summary,
            image_source=raw_image_source,
            timestamp_str=ts_now,
        )

        rec = upsert_res["record"]

        processed_results.append({
            "filename": item["filename"],
            "booklet_id": b_id,
            "usn": final_usn,
            "course_code": final_course_code,
            "test1_15": rec["Test1_15"],
            "test2_15": rec["Test2_15"],
            "aba_10": rec["ABA_10"],
            "theory_total_40": rec["Theory_Total_40"],
            "viva_5": rec["Viva_5"],
            "lab_5": rec["Lab_5"],
            "practical_total_10": rec["Practical_Total_10"],
            "total_cie_50": rec["Total_CIE_50"],
            "status": rec["Status"],
            "overrides": proc_res.get("overrides", []),
            "solver_overrides_str": rec["Solver_Overrides"],
            "image_url": rec["Image_Url"],
            "cloud_key": rec["Cloud_Key"],
            "timestamp": ts_now,
        })

    elapsed_total = round(time.time() - t0, 2)
    current_summary = _cloud.get_subjects_summary()

    return {
        "success": True,
        "count": len(processed_results),
        "processing_time_sec": elapsed_total,
        "results": processed_results,
        "subjects_summary": current_summary,
    }


# Backward-compatible single booklet upload route
@app.post("/api/booklets/upload")
async def single_upload_endpoint(
    page1: Optional[UploadFile] = File(default=None),
    sample_id: Optional[str] = Form(default=None),
    forced_course_code: Optional[str] = Form(default=None),
    course_code_override: Optional[str] = Form(default=None),
    booklet_id: Optional[str] = Form(default=None),
    subject_folder: Optional[str] = Form(default="live_mobile"),
):
    """Wraps batch-upload for drop-in backward compatibility."""
    active_code = forced_course_code or course_code_override
    batch_res = await batch_upload_endpoint(
        page1=page1,
        sample_ids=sample_id,
        forced_course_code=active_code,
        subject_folder=subject_folder,
    )
    if not batch_res.get("results"):
        raise HTTPException(status_code=400, detail="Failed to digitize booklet image.")
    first = batch_res["results"][0]
    return {
        "success": True,
        **first,
        "cie_summary": first,
        "cloud_storage": {
            "provider": _cloud.provider,
            "composite_key": first.get("cloud_key", ""),
            "composite_url": first.get("image_url", ""),
            "excel_key": f"subjects/{first['course_code']}/{first['course_code']}_marksheet.xlsx",
        },
        "processing_time_sec": batch_res.get("processing_time_sec", 0.0),
    }


# -----------------------------------------------------------------------------
# Endpoint 2: Subject-Wise Summary Statistics & Dashboard Data
# -----------------------------------------------------------------------------
@app.get("/api/subjects/summary")
async def get_subjects_summary_endpoint():
    """
    Returns live statistics per subject (24MCPC522, 24MCPC517, 24MCPC521):
      - total booklets scanned
      - average CIE score (out of 50)
      - validation & Z3 auto-correction counts
      - complete list of digitized student rows sorted by USN
    """
    return _cloud.get_subjects_summary()


# -----------------------------------------------------------------------------
# Endpoint 3: Standalone Subject Excel Download
# -----------------------------------------------------------------------------
@app.get("/api/subjects/{course_code}/download")
async def download_subject_marksheet(course_code: str):
    """
    Streams the standalone Excel marksheet for that specific subject
    ({course_code}_marksheet.xlsx), cleanly formatted and sorted by USN.
    """
    code = course_code.strip().upper()
    if code not in CANONICAL_SUBJECTS:
        raise HTTPException(
            status_code=400,
            detail=f"Course code '{course_code}' is not valid. Must be one of {CANONICAL_SUBJECTS}",
        )

    excel_path = _cloud.get_subject_excel_path(code)
    if not excel_path.exists():
        # Build empty formatted workbook if none exists yet
        _cloud.write_subject_workbook(code, [])

    return FileResponse(
        path=str(excel_path),
        filename=f"{code}_marksheet.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# -----------------------------------------------------------------------------
# Endpoint 4: Consolidated Master Multi-Sheet Workbook Download
# -----------------------------------------------------------------------------
@app.get("/api/booklets/download-master")
async def download_master_workbook():
    """
    Streams the consolidated multi-sheet workbook (master_all_subjects_marksheet.xlsx)
    containing:
      - Sheet 1: All_Records_Combined
      - Sheet 2: 24MCPC522
      - Sheet 3: 24MCPC517
      - Sheet 4: 24MCPC521
    All sheets sorted in ascending order by USN, with zero duplicates.
    """
    master_path = _cloud.get_master_excel_path()
    if not master_path.exists():
        cache = _cloud.load_cached_records()
        _cloud.write_master_workbook(list(cache.values()))

    return FileResponse(
        path=str(master_path),
        filename="master_all_subjects_marksheet.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# Legacy Excel download endpoint alias
@app.get("/api/booklets/download-excel")
async def download_excel_legacy(course_code: Optional[str] = Query(default=None)):
    if course_code and course_code.upper() in CANONICAL_SUBJECTS:
        return await download_subject_marksheet(course_code.upper())
    return await download_master_workbook()


# -----------------------------------------------------------------------------
# Digitized Records List & Samples
# -----------------------------------------------------------------------------
@app.get("/api/booklets/records")
async def list_digitized_records():
    """
    Returns list of all digitized student booklets across all subjects,
    merged with cloud image URLs and audit statuses.
    """
    summary = _cloud.get_subjects_summary()
    all_recs: List[Dict[str, Any]] = []

    for code, subj in summary["subjects"].items():
        for r in subj.get("records", []):
            ov_raw = str(r.get("Solver_Overrides", "")).strip()
            ov_chips = [s.strip() for s in ov_raw.split(";") if s.strip()] if ov_raw else []
            all_recs.append({
                "subject_folder": r.get("Booklet_ID", "live_scan"),
                "booklet_id": r.get("Booklet_ID", "live_scan"),
                "usn": r.get("USN", ""),
                "course_code": r.get("Course_Code", code),
                "test1_15": r.get("Test1_15", 0),
                "test2_15": r.get("Test2_15", 0),
                "aba_10": r.get("ABA_10", 0),
                "theory_total_40": r.get("Theory_Total_40", 0),
                "viva_5": r.get("Viva_5", 0),
                "lab_5": r.get("Lab_5", 0),
                "practical_total_10": r.get("Practical_Total_10", 0),
                "total_cie_50": r.get("Total_CIE_50", 0),
                "status": r.get("Status", "VALIDATED"),
                "solver_overrides": ov_raw,
                "overrides_chips": ov_chips,
                "composite_url": r.get("Image_Url", ""),
                "cloud_key": r.get("Cloud_Key", ""),
                "timestamp": r.get("Scanned_At", ""),
            })

    # Sort descending by timestamp for recent scans on top
    all_recs.sort(key=lambda x: str(x.get("timestamp", "")), reverse=True)

    return {
        "records": all_recs,
        "summary": {
            "total": summary["total_booklets"],
            "validated": summary["total_validated"],
            "auto_corrected": summary["total_auto_corrected"],
            "manual_review": summary["total_manual_review"],
        },
    }


@app.get("/api/booklets/samples")
async def list_sample_booklets():
    """Lists pre-scanned MITE sample booklets from data/raw/ for instant 1-click mobile testing."""
    samples = []
    if RAW_DATA_DIR.exists():
        p1_files = sorted(RAW_DATA_DIR.glob("booklet_*_p1.jpg"))
        for p1 in p1_files:
            b_id = p1.name.replace("_p1.jpg", "")
            samples.append({
                "booklet_id": b_id,
                "p1_url": f"/raw-samples/{p1.name}",
            })
    return {"samples": samples, "count": len(samples)}


# -----------------------------------------------------------------------------
# Main Application Launcher
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
