"""
Production-Ready Marksheet Digitizer Web Application.
FastAPI Backend with Automated Neuro-Symbolic Pipeline, Template Detection
(Integrated-23 & Theory-23), Active Learning Human Corrections, Strict
24MCPC517 / Decimal Purging, and One-Click Clean Excel Export.
"""

import os
import sys
import io
import time
import json
import uuid
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Setup Project Root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR,
    OUTPUT_DIR,
    VALID_COURSE_CODES,
    FOLDER_COURSE_CODES,
    resolve_subject_from_filename,
)
from src.pipeline import BatchProcessor
from src.active_learner import get_active_learner

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("marksheet_webapp")

# Directory Config
VAULT_DIR = PROJECT_ROOT / "cloud_vault" / "batch_uploads"
OUTPUT_DIR_PATH = Path(OUTPUT_DIR)
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"

VAULT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

FINAL_CLEAN_EXCEL_PATH = OUTPUT_DIR_PATH / "web_clean_digitized_marksheet.xlsx"
AUDITED_EXCEL_PATH = OUTPUT_DIR_PATH / "audited_marksheet.xlsx"

# -----------------------------------------------------------------------------
# FastAPI App Initialization
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Marksheet Digitizer Web Application",
    description="Neuro-Symbolic Mobile Marksheet Scoring, Template Routing & Clean Excel Export",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Global Pipeline Singleton
processor_singleton: Optional[BatchProcessor] = None


def get_processor() -> BatchProcessor:
    global processor_singleton
    if processor_singleton is None:
        logger.info("Initializing BatchProcessor singleton...")
        processor_singleton = BatchProcessor()
    return processor_singleton


# In-memory session store for current processed batch
CURRENT_BATCH_DATA: Dict[str, Any] = {
    "records": [],
    "purged_records": [],
    "stats": {"total_in": 0, "dropped_517": 0, "dropped_decimal": 0, "passed": 0},
    "last_updated": None,
}


# -----------------------------------------------------------------------------
# Formatted Clean Excel Generator (OpenPyXL)
# -----------------------------------------------------------------------------
def generate_formatted_clean_excel(records: List[Dict[str, Any]], output_path: Path) -> Path:
    """
    Generates a production-ready, beautifully styled Excel workbook with:
      - Sheet 1: CIE_Summary (Unified Master Sheet)
      - Sheet 2: 24MCPC522 (Integrated-23 Subset)
      - Sheet 3: 24MCPC521 (Theory-23 Subset)
    Applies column formatting, status color-fills, and auto column widths.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")
    center_align = Alignment(horizontal="center", vertical="center")
    left_align = Alignment(horizontal="left", vertical="center")
    thin_border = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1"),
    )

    fill_validated = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid")  # soft green
    fill_corrected = PatternFill(start_color="DBEAFE", end_color="DBEAFE", fill_type="solid")  # soft blue
    fill_review = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")     # soft red

    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    master_columns = [
        "Subject_Folder", "Booklet_ID", "USN", "Course_Code",
        "Test1_15", "Test2_15", "Test_Total_30",
        "ABA_10", "ABA_I_10", "ABA_II_10", "Total_ABA_20",
        "Theory_Total_40", "Viva_5", "Lab_5", "Practical_Total_10",
        "Total_CIE_50", "Status", "Solver_Overrides"
    ]

    def _write_sheet(sheet_name: str, sheet_records: List[Dict[str, Any]], column_list: List[str]):
        ws = wb.create_sheet(title=sheet_name[:31])
        ws.views.sheetView[0].showGridLines = True

        # Header
        for col_idx, col_name in enumerate(column_list, start=1):
            cell = ws.cell(row=1, column=col_idx, value=col_name)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center_align
            cell.border = thin_border
        ws.row_dimensions[1].height = 26

        # Data rows
        for row_idx, r in enumerate(sheet_records, start=2):
            ws.row_dimensions[row_idx].height = 20
            status_val = str(r.get("Status", "VALIDATED")).upper()

            for col_idx, col_name in enumerate(column_list, start=1):
                val = r.get(col_name, "")
                if pd.isna(val):
                    val = ""

                cell = ws.cell(row=row_idx, column=col_idx, value=val)
                cell.border = thin_border

                if col_name in ["Booklet_ID", "Solver_Overrides", "Reviewer_Notes"]:
                    cell.alignment = left_align
                else:
                    cell.alignment = center_align

                # Color-code status cell
                if col_name == "Status":
                    if "VALIDATED" in status_val or "APPROVED" in status_val:
                        cell.fill = fill_validated
                    elif "CORRECTED" in status_val:
                        cell.fill = fill_corrected
                    elif "REVIEW" in status_val or "UNSAT" in status_val:
                        cell.fill = fill_review

        # Auto width
        for col in ws.columns:
            col_letter = get_column_letter(col[0].column)
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    # 1. Master Sheet
    _write_sheet("CIE_Summary", records, master_columns)

    # 2. 24MCPC522 Sheet (Integrated-23)
    rec_522 = [r for r in records if "522" in str(r.get("Course_Code", ""))]
    if rec_522:
        cols_522 = [
            "Booklet_ID", "USN", "Course_Code",
            "Test1_15", "Test2_15", "ABA_10", "Theory_Total_40",
            "Viva_5", "Lab_5", "Practical_Total_10",
            "Total_CIE_50", "Status", "Solver_Overrides"
        ]
        _write_sheet("24MCPC522", rec_522, cols_522)

    # 3. 24MCPC521 Sheet (Theory-23)
    rec_521 = [r for r in records if "521" in str(r.get("Course_Code", ""))]
    if rec_521:
        cols_521 = [
            "Booklet_ID", "USN", "Course_Code",
            "Test1_15", "Test2_15", "Test_Total_30",
            "ABA_I_10", "ABA_II_10", "Total_ABA_20",
            "Total_CIE_50", "Status", "Solver_Overrides"
        ]
        _write_sheet("24MCPC521", rec_521, cols_521)

    wb.save(output_path)
    logger.info(f"Clean formatted Excel saved to {output_path} ({len(records)} records)")
    return output_path


# -----------------------------------------------------------------------------
# Web Routes
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """Serves the clean MITE marksheet portal."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/status")
async def get_system_status():
    """Returns real-time pipeline, OCR device, and Active Learning cache status."""
    learner = get_active_learner()
    processor = get_processor()
    device_info = "GPU (CUDA)" if getattr(processor.ocr, "device", "cpu") != "cpu" else "CPU"
    return {
        "status": "ready",
        "device": device_info,
        "active_learner_cached_records": len(learner.corrections),
        "total_current_batch": len(CURRENT_BATCH_DATA["purged_records"]),
        "valid_course_codes": VALID_COURSE_CODES,
    }


@app.post("/api/upload-batch")
async def upload_and_process_batch(
    files: List[UploadFile] = File(...),
    subject_hint: str = Form("auto"),
    is_first: str = Form("false"),
    is_last: str = Form("false"),
):
    """
    Batch Upload Ingestion Endpoint:
      1. Saves uploaded files to cloud_vault/batch_uploads/<batch_id>
      2. Executes Page 1 neuro-symbolic extraction
      3. Automatically applies Active Learning corrections (from audited_marksheet.xlsx)
      4. Strictly drops excluded subject 24MCPC517 and decimal-containing records
      5. Generates the clean formatted Excel file for one-click download
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    reset_batch = is_first.lower() in ("true", "1")
    if reset_batch:
        CURRENT_BATCH_DATA["records"] = []
        CURRENT_BATCH_DATA["purged_records"] = []
        CURRENT_BATCH_DATA["stats"] = {"total_in": 0, "dropped_517": 0, "dropped_decimal": 0, "passed": 0}

    processor = get_processor()
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = VAULT_DIR / f"batch_{batch_timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    saved_image_paths: List[Tuple[Path, str]] = []
    supported_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

    for file in files:
        filename = file.filename or "unknown.jpg"
        clean_name = Path(filename).name
        ext = Path(clean_name).suffix.lower()
        if ext not in supported_extensions:
            continue

        # Automatically skip any Page-2 booklet images
        if "_p2" in clean_name.lower():
            logger.info(f"Skipping Page-2 image: {clean_name}")
            continue

        target_path = batch_dir / clean_name
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_image_paths.append((target_path, clean_name))

    if not saved_image_paths:
        raise HTTPException(
            status_code=400,
            detail="No valid image files found. Please upload .jpg, .jpeg, or .png scans.",
        )

    logger.info(f"Ingested {len(saved_image_paths)} images into {batch_dir}. Starting extraction...")

    # Process all images
    extracted_rows: List[Dict[str, Any]] = []

    for img_path, original_filename in saved_image_paths:
        booklet_stem = Path(original_filename).stem

        # Subject determination
        if subject_hint != "auto" and subject_hint in FOLDER_COURSE_CODES:
            folder_arg = subject_hint
        else:
            folder_arg = "raw"  # pipeline uses visual OCR & resolve_subject_from_filename

        try:
            res = processor.process_page1_image(
                img_input=img_path,
                booklet_id=booklet_stem,
                subject_folder=folder_arg,
            )
            if res.get("success", False) and "cie_summary" in res:
                cie_row = dict(res["cie_summary"])
                cie_row["Filename"] = original_filename
                cie_row["Vault_Path"] = str(img_path)
                extracted_rows.append(cie_row)
            else:
                logger.warning(f"Processing failed for {original_filename}: {res.get('error')}")
        except Exception as e:
            logger.error(f"Error processing {original_filename}: {e}", exc_info=True)

    # Apply strict purge filter: drop 24MCPC517 and any decimal marks
    purged_records, purge_stats = BatchProcessor.purge_invalid_export_records(extracted_rows)

    # Accumulate into current session
    CURRENT_BATCH_DATA["records"].extend(extracted_rows)
    CURRENT_BATCH_DATA["purged_records"].extend(purged_records)
    CURRENT_BATCH_DATA["stats"]["total_in"] += purge_stats.get("total_in", 0)
    CURRENT_BATCH_DATA["stats"]["dropped_517"] += purge_stats.get("dropped_517", 0)
    CURRENT_BATCH_DATA["stats"]["dropped_decimal"] += purge_stats.get("dropped_decimal", 0)
    CURRENT_BATCH_DATA["stats"]["passed"] += purge_stats.get("passed", 0)
    CURRENT_BATCH_DATA["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate clean formatted Excel
    if CURRENT_BATCH_DATA["purged_records"]:
        generate_formatted_clean_excel(CURRENT_BATCH_DATA["purged_records"], FINAL_CLEAN_EXCEL_PATH)

    return {
        "success": True,
        "message": (
            f"Processed {len(saved_image_paths)} booklet scan(s). "
            f"{len(CURRENT_BATCH_DATA['purged_records'])} record(s) extracted successfully."
        ),
        "total_uploaded": len(saved_image_paths),
        "total_processed": len(extracted_rows),
        "passed_count": len(CURRENT_BATCH_DATA["purged_records"]),
        "records": CURRENT_BATCH_DATA["purged_records"],
        "download_url": "/api/download-excel",
    }


@app.get("/api/download-excel")
async def download_clean_excel():
    """
    High-Visibility One-Click Excel Download:
    Serves the filtered, production-ready spreadsheet.
    If no new batch was processed, falls back to the audited dataset purged of 24MCPC517.
    """
    if FINAL_CLEAN_EXCEL_PATH.exists() and CURRENT_BATCH_DATA["purged_records"]:
        return FileResponse(
            path=FINAL_CLEAN_EXCEL_PATH,
            filename="clean_digitized_marksheet.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # Fallback to existing audited marksheet (with 24MCPC517 strictly purged)
    if AUDITED_EXCEL_PATH.exists():
        try:
            df = pd.read_excel(AUDITED_EXCEL_PATH, sheet_name=0)
            records = df.to_dict(orient="records")
            purged, _ = BatchProcessor.purge_invalid_export_records(records)
            if purged:
                generate_formatted_clean_excel(purged, FINAL_CLEAN_EXCEL_PATH)
                return FileResponse(
                    path=FINAL_CLEAN_EXCEL_PATH,
                    filename="clean_digitized_marksheet.xlsx",
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        except Exception as e:
            logger.error(f"Error preparing fallback Excel: {e}")

    raise HTTPException(
        status_code=404,
        detail="No clean marksheet records found. Please upload booklet scans first.",
    )


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    host = "0.0.0.0"
    port = 8000
    print("\n" + "=" * 60)
    print("🚀 Marksheet Digitizer Web Application")
    print(f"👉 Local Web Portal: http://127.0.0.1:{port}")
    print(f"👉 Network Portal:   http://{host}:{port}")
    print(f"👉 API Docs:         http://127.0.0.1:{port}/docs")
    print("=" * 60 + "\n")
    uvicorn.run("app:app", host=host, port=port, reload=False)
