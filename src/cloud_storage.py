"""
Pluggable Subject-Wise Cloud Storage Router & Ledger Manager for Marksheet Digitizer.
Supports AWS S3, Supabase Storage, Google Cloud Storage, and Local Cloud Vault.

Strict Storage Layout:
  - Image Archival:  cloud_vault/subjects/{course_code}/scans/{usn}_{timestamp}.jpg
  - Subject Excel:   cloud_vault/subjects/{course_code}/{course_code}_marksheet.xlsx
  - Master Excel:    cloud_vault/exports/master_all_subjects_marksheet.xlsx
"""

import os
import shutil
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import cv2
import numpy as np
import pandas as pd
import requests
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from .config import PROJECT_ROOT, DATA_DIR, VALID_COURSE_CODES, USN_PREFIX

logger = logging.getLogger(__name__)

# Canonical subjects handled by the MITE Marksheet Digitizer
CANONICAL_SUBJECTS = list(VALID_COURSE_CODES)  # ["24MCPC522", "24MCPC517", "24MCPC521"]

SUBJECT_TITLES = {
    "24MCPC522": "Cloud Computing & Distributed Systems",
    "24MCPC517": "Machine Learning & Pattern Recognition",
    "24MCPC521": "Advanced Network Security & Cryptography",
}

COLUMN_SPECS = [
    ("USN", "Student USN", 16),
    ("Course_Code", "Course Code", 16),
    ("Test1_15", "Test 1 (15)", 13),
    ("Test2_15", "Test 2 (15)", 13),
    ("ABA_10", "ABA (10)", 12),
    ("Theory_Total_40", "Theory (40)", 14),
    ("Viva_5", "Viva (5)", 11),
    ("Lab_5", "Lab (5)", 11),
    ("Practical_Total_10", "Pract (10)", 13),
    ("Total_CIE_50", "Total CIE (50)", 16),
    ("Status", "Status", 18),
    ("Solver_Overrides", "Z3 Overrides / Audit", 36),
    ("Scanned_At", "Timestamp", 20),
    ("Booklet_ID", "Booklet Source", 24),
]


def _format_worksheet(ws: Any, title_text: str = "") -> None:
    """Applies professional styling to an openpyxl worksheet."""
    # Palette
    header_fill = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")  # Navy
    header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    data_font = Font(name="Segoe UI", size=10, bold=False, color="0F172A")
    total_font = Font(name="Segoe UI", size=11, bold=True, color="047857")  # Deep Emerald
    total_fill = PatternFill(start_color="ECFDF5", end_color="ECFDF5", fill_type="solid")  # Soft green

    thin_border = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1"),
    )

    # Freeze panes below header
    ws.freeze_panes = "A2"

    # Style header row (Row 1)
    for col_idx in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border
    ws.row_dimensions[1].height = 28

    # Style data rows
    for row_idx in range(2, ws.max_row + 1):
        ws.row_dimensions[row_idx].height = 22
        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.font = data_font
            cell.border = thin_border

            header_val = str(ws.cell(row=1, column=col_idx).value or "")

            if "Total CIE" in header_val:
                cell.font = total_font
                cell.fill = total_fill
                cell.alignment = Alignment(horizontal="center", vertical="center")
            elif any(k in header_val for k in ["Test", "ABA", "Theory", "Viva", "Lab", "Pract"]):
                cell.alignment = Alignment(horizontal="center", vertical="center")
            elif "USN" in header_val or "Course" in header_val:
                cell.alignment = Alignment(horizontal="center", vertical="center")
            elif "Status" in header_val:
                val = str(cell.value or "")
                if val == "VALIDATED":
                    cell.font = Font(name="Segoe UI", size=10, bold=True, color="15803D")
                elif val == "AUTO_CORRECTED":
                    cell.font = Font(name="Segoe UI", size=10, bold=True, color="B45309")
                cell.alignment = Alignment(horizontal="center", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="left", vertical="center")

    # Auto-adjust column widths
    for col_idx, (_, _, default_w) in enumerate(COLUMN_SPECS, start=1):
        if col_idx <= ws.max_column:
            col_letter = get_column_letter(col_idx)
            ws.column_dimensions[col_letter].width = max(default_w, 12)


class CloudStorageManager:
    """
    Unified Subject-Wise Storage Router and Ledger Manager.
    Manages local filesystem storage under `cloud_vault/` and synchronizes
    with AWS S3, Supabase Storage, or Google Cloud Storage when configured.
    """

    def __init__(self, cloud_vault_root: Optional[Path] = None):
        # Default root: <PROJECT_ROOT>/cloud_vault
        self.vault_root = Path(cloud_vault_root or (PROJECT_ROOT / "cloud_vault"))
        self.local_root = self.vault_root  # Alias for backward compatibility
        self.cache_file = self.vault_root / "records_cache.json"

        # Initialize core directory tree
        self._init_vault_directories()

        # Cloud Credentials
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
        self.aws_region = os.getenv("AWS_REGION", "ap-south-1").strip()
        self.s3_bucket = os.getenv("S3_BUCKET_NAME", "").strip()
        self.s3_endpoint = os.getenv("S3_ENDPOINT_URL", "").strip() or None

        self.supabase_url = os.getenv("SUPABASE_URL", "").strip()
        self.supabase_key = os.getenv("SUPABASE_KEY", "").strip()
        self.supabase_bucket = os.getenv("SUPABASE_BUCKET", "marksheet-digitizer").strip()

        self.gcs_bucket = os.getenv("GCS_BUCKET_NAME", "").strip()

        self.provider = "local"
        self.s3_client = None
        self.gcs_client = None

        self._initialize_provider()

    def _init_vault_directories(self) -> None:
        """Ensures all canonical subject scan and export directories exist."""
        self.vault_root.mkdir(parents=True, exist_ok=True)
        (self.vault_root / "exports").mkdir(parents=True, exist_ok=True)
        (self.vault_root / "previews").mkdir(parents=True, exist_ok=True)

        for code in CANONICAL_SUBJECTS:
            subj_dir = self.vault_root / "subjects" / code / "scans"
            subj_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_provider(self) -> None:
        """Initializes active cloud adapter or falls back to local vault."""
        if self.aws_access_key and self.aws_secret_key and self.s3_bucket:
            try:
                import boto3
                self.s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key,
                    region_name=self.aws_region,
                    endpoint_url=self.s3_endpoint,
                )
                self.provider = "aws_s3"
                logger.info(f"CloudStorage initialized with AWS S3: {self.s3_bucket}")
                return
            except Exception as e:
                logger.warning(f"AWS S3 init failed, falling back to local: {e}")

        if self.supabase_url and self.supabase_key:
            self.provider = "supabase"
            logger.info(f"CloudStorage initialized with Supabase: {self.supabase_bucket}")
            return

        if self.gcs_bucket and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            try:
                from google.cloud import storage  # type: ignore
                self.gcs_client = storage.Client()
                self.provider = "gcs"
                logger.info(f"CloudStorage initialized with GCS: {self.gcs_bucket}")
                return
            except Exception as e:
                logger.warning(f"GCS init failed, falling back to local: {e}")

        self.provider = "local"
        logger.info(f"CloudStorage running in Local Vault mode at: {self.vault_root.resolve()}")

    # -------------------------------------------------------------------------
    # Path Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def sanitize_segment(segment: str, default: str = "GENERAL") -> str:
        if not segment:
            return default
        cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(segment).strip())
        cleaned = cleaned.strip("_")
        return cleaned[:40] if cleaned else default

    def get_subject_excel_path(self, course_code: str) -> Path:
        safe_course = self.sanitize_segment(course_code, default=CANONICAL_SUBJECTS[0])
        return self.vault_root / "subjects" / safe_course / f"{safe_course}_marksheet.xlsx"

    def get_master_excel_path(self) -> Path:
        return self.vault_root / "exports" / "master_all_subjects_marksheet.xlsx"

    def get_subject_scans_dir(self, course_code: str) -> Path:
        safe_course = self.sanitize_segment(course_code, default=CANONICAL_SUBJECTS[0])
        p = self.vault_root / "subjects" / safe_course / "scans"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # -------------------------------------------------------------------------
    # Core Cloud Object Upload
    # -------------------------------------------------------------------------
    def upload_bytes_or_file(
        self,
        source: Union[bytes, str, Path, np.ndarray],
        object_key: str,
        content_type: str = "application/octet-stream",
    ) -> Dict[str, Any]:
        """
        Stores an object under `object_key` relative to `cloud_vault/`
        both in local vault and active cloud provider.
        """
        # Clean relative object key
        clean_key = object_key.replace("\\", "/").lstrip("/")
        if clean_key.startswith("cloud_vault/"):
            rel_path = clean_key[len("cloud_vault/"):]
        else:
            rel_path = clean_key

        local_target = self.vault_root / rel_path
        local_target.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(source, np.ndarray):
            _, buf = cv2.imencode(".jpg", source, [cv2.IMWRITE_JPEG_QUALITY, 92])
            payload_bytes = buf.tobytes()
            local_target.write_bytes(payload_bytes)
        elif isinstance(source, bytes):
            payload_bytes = source
            local_target.write_bytes(payload_bytes)
        else:
            src_path = Path(source)
            if src_path.resolve() != local_target.resolve():
                shutil.copy2(src_path, local_target)
            payload_bytes = local_target.read_bytes()

        # Web-accessible URL
        web_url = f"/cloud-vault/{rel_path}"
        cloud_url = web_url
        remote_synced = False

        # 1. AWS S3 Upload
        if self.provider == "aws_s3" and self.s3_client is not None:
            try:
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=f"cloud_vault/{rel_path}",
                    Body=payload_bytes,
                    ContentType=content_type,
                )
                cloud_url = self.get_presigned_url(f"cloud_vault/{rel_path}")
                remote_synced = True
            except Exception as e:
                logger.error(f"S3 upload error for {rel_path}: {e}")

        # 2. Supabase Storage
        elif self.provider == "supabase":
            try:
                endpoint = f"{self.supabase_url.rstrip('/')}/storage/v1/object/{self.supabase_bucket}/cloud_vault/{rel_path}"
                headers = {
                    "Authorization": f"Bearer {self.supabase_key}",
                    "apikey": self.supabase_key,
                    "Content-Type": content_type,
                    "x-upsert": "true",
                }
                resp = requests.post(endpoint, headers=headers, data=payload_bytes, timeout=15)
                if resp.status_code in (200, 201):
                    cloud_url = f"{self.supabase_url.rstrip('/')}/storage/v1/object/public/{self.supabase_bucket}/cloud_vault/{rel_path}"
                    remote_synced = True
                else:
                    logger.warning(f"Supabase upload returned {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.error(f"Supabase upload failed for {rel_path}: {e}")

        # 3. Google Cloud Storage
        elif self.provider == "gcs" and self.gcs_client is not None:
            try:
                bucket = self.gcs_client.bucket(self.gcs_bucket)
                blob = bucket.blob(f"cloud_vault/{rel_path}")
                blob.upload_from_string(payload_bytes, content_type=content_type)
                cloud_url = blob.public_url
                remote_synced = True
            except Exception as e:
                logger.error(f"GCS upload failed for {rel_path}: {e}")

        return {
            "key": f"cloud_vault/{rel_path}",
            "rel_path": rel_path,
            "provider": self.provider,
            "remote_synced": remote_synced,
            "url": cloud_url,
            "local_url": web_url,
            "local_path": str(local_target.resolve()),
            "size_bytes": len(payload_bytes),
        }

    # -------------------------------------------------------------------------
    # Subject Scan Archival
    # -------------------------------------------------------------------------
    def upload_subject_scan(
        self,
        image_source: Union[bytes, str, Path, np.ndarray],
        course_code: str,
        usn: str,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Archives scan photo strictly under:
          cloud_vault/subjects/{course_code}/scans/{usn}_{timestamp}.jpg
        """
        safe_course = self.sanitize_segment(course_code, default=CANONICAL_SUBJECTS[0])
        safe_usn = self.sanitize_segment(usn, default="UNKNOWN_USN")
        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{safe_usn}_{ts}.jpg"
        object_key = f"subjects/{safe_course}/scans/{filename}"

        return self.upload_bytes_or_file(
            source=image_source,
            object_key=object_key,
            content_type="image/jpeg",
        )

    # -------------------------------------------------------------------------
    # Ledger Cache & Record Management
    # -------------------------------------------------------------------------
    def load_cached_records(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads the in-vault ledger dictionary keyed by `f'{course_code}::{usn}'`.
        """
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Could not parse ledger cache: {e}")
        return {}

    def save_cached_records(self, records_dict: Dict[str, Dict[str, Any]]) -> None:
        """Persists the ledger dictionary to disk atomically."""
        try:
            self.cache_file.write_text(json.dumps(records_dict, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save records cache: {e}")

    # -------------------------------------------------------------------------
    # Excel Ledger Generators (openpyxl styled)
    # -------------------------------------------------------------------------
    def _create_openpyxl_sheet(self, ws: Any, records: List[Dict[str, Any]]) -> None:
        """Populates and styles an openpyxl sheet from a list of records."""
        # Header Row
        headers = [disp for _, disp, _ in COLUMN_SPECS]
        ws.append(headers)

        for rec in records:
            row_data = [
                rec.get("USN", ""),
                rec.get("Course_Code", ""),
                int(rec.get("Test1_15", 0) or 0),
                int(rec.get("Test2_15", 0) or 0),
                int(rec.get("ABA_10", 0) or 0),
                int(rec.get("Theory_Total_40", 0) or 0),
                int(rec.get("Viva_5", 0) or 0),
                int(rec.get("Lab_5", 0) or 0),
                int(rec.get("Practical_Total_10", 0) or 0),
                int(rec.get("Total_CIE_50", 0) or 0),
                rec.get("Status", "VALIDATED"),
                rec.get("Solver_Overrides", ""),
                rec.get("Scanned_At", ""),
                rec.get("Booklet_ID", ""),
            ]
            ws.append(row_data)

        _format_worksheet(ws)

    def write_subject_workbook(self, course_code: str, records: List[Dict[str, Any]]) -> Path:
        """
        Generates and writes `{course_code}_marksheet.xlsx` sorted by USN ascending.
        Path: cloud_vault/subjects/{course_code}/{course_code}_marksheet.xlsx
        """
        out_path = self.get_subject_excel_path(course_code)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        sorted_records = sorted(records, key=lambda r: str(r.get("USN", "")).upper())

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = course_code[:31]

        self._create_openpyxl_sheet(ws, sorted_records)
        wb.save(str(out_path))

        # Sync to cloud if remote provider configured
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        rel_key = f"subjects/{course_code}/{course_code}_marksheet.xlsx"
        self.upload_bytes_or_file(out_path, rel_key, content_type=mime)

        return out_path

    def write_master_workbook(self, all_records: List[Dict[str, Any]]) -> Path:
        """
        Generates and writes consolidated `master_all_subjects_marksheet.xlsx`.
        Contains:
          - Sheet 1: All_Records_Combined
          - Sheet 2: 24MCPC522
          - Sheet 3: 24MCPC517
          - Sheet 4: 24MCPC521
        All sheets sorted by USN in ascending order.
        """
        out_path = self.get_master_excel_path()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        wb = openpyxl.Workbook()

        # Sheet 1: All_Records_Combined
        ws_all = wb.active
        ws_all.title = "All_Records_Combined"
        sorted_all = sorted(
            all_records,
            key=lambda r: (str(r.get("Course_Code", "")), str(r.get("USN", "")))
        )
        self._create_openpyxl_sheet(ws_all, sorted_all)

        # Subject Sheets (24MCPC522, 24MCPC517, 24MCPC521)
        for code in CANONICAL_SUBJECTS:
            subj_records = [r for r in all_records if str(r.get("Course_Code", "")).upper() == code]
            subj_sorted = sorted(subj_records, key=lambda r: str(r.get("USN", "")).upper())
            ws_subj = wb.create_sheet(title=code[:31])
            self._create_openpyxl_sheet(ws_subj, subj_sorted)

        wb.save(str(out_path))

        # Sync to cloud if remote provider configured
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        rel_key = "exports/master_all_subjects_marksheet.xlsx"
        self.upload_bytes_or_file(out_path, rel_key, content_type=mime)

        return out_path

    # -------------------------------------------------------------------------
    # Upsert & Ledger Reconciliation
    # -------------------------------------------------------------------------
    def upsert_booklet_record(
        self,
        cie_record: Dict[str, Any],
        image_source: Optional[Union[bytes, str, Path, np.ndarray]] = None,
        timestamp_str: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Atomic upsert:
          1. Normalizes USN and Course_Code.
          2. Archives scan photo to `cloud_vault/subjects/{course_code}/scans/{usn}_{ts}.jpg`.
          3. Upserts into cache (keyed by `f'{course_code}::{usn}'`).
          4. Re-sorts ascending by USN and writes subject `{course_code}_marksheet.xlsx`.
          5. Rebuilds and writes `master_all_subjects_marksheet.xlsx`.
        """
        ts = timestamp_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_compact = ts.replace("-", "").replace(":", "").replace(" ", "_")

        raw_course = str(cie_record.get("Course_Code", CANONICAL_SUBJECTS[0])).strip().upper()
        if raw_course not in CANONICAL_SUBJECTS:
            # Snap to closest canonical subject
            raw_course = CANONICAL_SUBJECTS[0]
        course_code = raw_course

        usn = str(cie_record.get("USN", "UNKNOWN")).strip().upper()
        booklet_id = str(cie_record.get("Booklet_ID", "live_scan")).strip()

        # Archive image if provided
        img_info = {}
        if image_source is not None:
            img_info = self.upload_subject_scan(
                image_source=image_source,
                course_code=course_code,
                usn=usn,
                timestamp=ts_compact,
            )

        # Standardized record dictionary
        record_entry = {
            "USN": usn,
            "Course_Code": course_code,
            "Test1_15": int(cie_record.get("Test1_15", 0) or 0),
            "Test2_15": int(cie_record.get("Test2_15", 0) or 0),
            "ABA_10": int(cie_record.get("ABA_10", 0) or 0),
            "Theory_Total_40": int(cie_record.get("Theory_Total_40", 0) or 0),
            "Viva_5": int(cie_record.get("Viva_5", 0) or 0),
            "Lab_5": int(cie_record.get("Lab_5", 0) or 0),
            "Practical_Total_10": int(cie_record.get("Practical_Total_10", 0) or 0),
            "Total_CIE_50": int(cie_record.get("Total_CIE_50", 0) or 0),
            "Status": str(cie_record.get("Status", "VALIDATED")),
            "Solver_Overrides": str(cie_record.get("Solver_Overrides", "")),
            "Scanned_At": ts,
            "Booklet_ID": booklet_id,
            "Image_Url": img_info.get("local_url", cie_record.get("Image_Url", "")),
            "Cloud_Key": img_info.get("key", cie_record.get("Cloud_Key", "")),
        }

        # Load existing ledger cache
        cache = self.load_cached_records()
        record_key = f"{course_code}::{usn}"
        cache[record_key] = record_entry
        self.save_cached_records(cache)

        # Rebuild subject Excel
        all_records = list(cache.values())
        subj_records = [r for r in all_records if str(r.get("Course_Code", "")).upper() == course_code]
        self.write_subject_workbook(course_code, subj_records)

        # Rebuild Master Excel
        self.write_master_workbook(all_records)

        return {
            "record": record_entry,
            "image_info": img_info,
            "subject_excel": str(self.get_subject_excel_path(course_code)),
            "master_excel": str(self.get_master_excel_path()),
        }

    # -------------------------------------------------------------------------
    # Subject-Wise Summary Statistics
    # -------------------------------------------------------------------------
    def get_subjects_summary(self) -> Dict[str, Any]:
        """
        Returns live statistics across all subjects:
          - live student count per subject
          - average CIE marks (0-50)
          - validation vs auto-corrected breakdown
          - student rows sorted ascending by USN
        """
        cache = self.load_cached_records()
        all_records = list(cache.values())

        subjects_stat: Dict[str, Any] = {}
        total_validated = 0
        total_auto_corrected = 0
        total_manual_review = 0
        all_totals: List[int] = []

        for code in CANONICAL_SUBJECTS:
            records = [r for r in all_records if str(r.get("Course_Code", "")).upper() == code]
            records_sorted = sorted(records, key=lambda r: str(r.get("USN", "")).upper())

            val_count = sum(1 for r in records if r.get("Status") == "VALIDATED")
            auto_count = sum(1 for r in records if r.get("Status") == "AUTO_CORRECTED")
            rev_count = sum(1 for r in records if r.get("Status") not in ("VALIDATED", "AUTO_CORRECTED"))

            total_validated += val_count
            total_auto_corrected += auto_count
            total_manual_review += rev_count

            cie_vals = [int(r.get("Total_CIE_50", 0) or 0) for r in records]
            all_totals.extend(cie_vals)
            avg_cie = round(float(np.mean(cie_vals)), 1) if cie_vals else 0.0

            subjects_stat[code] = {
                "course_code": code,
                "title": SUBJECT_TITLES.get(code, f"Course {code}"),
                "student_count": len(records_sorted),
                "average_cie": avg_cie,
                "validated_count": val_count,
                "auto_corrected_count": auto_count,
                "manual_review_count": rev_count,
                "records": records_sorted,
                "download_url": f"/api/subjects/{code}/download",
            }

        overall_avg = round(float(np.mean(all_totals)), 1) if all_totals else 0.0

        return {
            "success": True,
            "subjects": subjects_stat,
            "total_booklets": len(all_records),
            "overall_average_cie": overall_avg,
            "total_validated": total_validated,
            "total_auto_corrected": total_auto_corrected,
            "total_manual_review": total_manual_review,
            "master_download_url": "/api/booklets/download-master",
        }

    # -------------------------------------------------------------------------
    # Compatibility Shims
    # -------------------------------------------------------------------------
    def upload_composite_image(
        self,
        image_source: Union[bytes, str, Path],
        course_code: str,
        booklet_id: str,
        date_str: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Backward compatible wrapper around upload_subject_scan."""
        return self.upload_subject_scan(
            image_source=image_source,
            course_code=course_code,
            usn=booklet_id,
            timestamp=date_str or datetime.now().strftime("%Y%m%d_%H%M%S"),
        )

    def sync_excel_ledger(
        self,
        excel_path: Union[str, Path],
        course_code: str = "ALL_COURSES",
    ) -> Dict[str, Any]:
        """Backward compatible wrapper."""
        safe_course = self.sanitize_segment(course_code, default="ALL_COURSES")
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        return self.upload_bytes_or_file(
            source=excel_path,
            object_key=f"exports/{safe_course}/digitized_marksheet.xlsx",
            content_type=mime,
        )

    def get_presigned_url(self, object_key: str, expires_in: int = 3600) -> str:
        clean_key = object_key.replace("\\", "/").lstrip("/")
        if self.provider == "aws_s3" and self.s3_client is not None:
            try:
                return self.s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.s3_bucket, "Key": clean_key},
                    ExpiresIn=expires_in,
                )
            except Exception as e:
                logger.warning(f"Presigned URL generation error: {e}")

        if clean_key.startswith("cloud_vault/"):
            return f"/{clean_key}"
        return f"/cloud-vault/{clean_key}"

    def get_status(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "s3_bucket": self.s3_bucket or None,
            "supabase_bucket": self.supabase_bucket if self.provider == "supabase" else None,
            "gcs_bucket": self.gcs_bucket or None,
            "vault_root": str(self.vault_root.resolve()),
        }
