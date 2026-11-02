"""
Active Learning and Human-in-the-Loop Correction Caching Module.
Reads audited_marksheet.xlsx, extracts all human-verified / MANUALLY_CORRECTED
overrides, and maintains a persistent cache so subsequent OCR and pipeline runs
instantly apply human-verified corrections.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd

from .config import DATA_DIR, OUTPUT_DIR

logger = logging.getLogger("active_learner")

AUDITED_EXCEL_PATH = Path(OUTPUT_DIR) / "audited_marksheet.xlsx"
CACHE_JSON_PATH = Path(DATA_DIR) / "correction_cache.json"

MARK_COLUMNS = [
    "Test1_15",
    "Test2_15",
    "ABA_10",
    "Theory_Total_40",
    "Viva_5",
    "Lab_5",
    "Practical_Total_10",
    "Total_CIE_50",
    "Test_Total_30",
    "ABA_I_10",
    "ABA_II_10",
    "Total_ABA_20",
]

KEY_TO_CELL_MAP = {
    "Test1_15": "test1",
    "Test2_15": "test2",
    "ABA_10": "aba",
    "Theory_Total_40": "theory_total",
    "Viva_5": "viva",
    "Lab_5": "lab",
    "Practical_Total_10": "pract_total",
    "Total_CIE_50": "cie_total",
    "Test_Total_30": "test_total",
    "ABA_I_10": "aba1",
    "ABA_II_10": "aba2",
    "Total_ABA_20": "total_aba",
}


class ActiveLearner:
    """
    Manages active learning correction caches from human audits.
    Extracts manual corrections from audited_marksheet.xlsx and builds
    a lookup table indexed by booklet ID and USN.
    """

    def __init__(
        self,
        audited_path: Optional[Path] = None,
        cache_path: Optional[Path] = None,
    ):
        self.audited_path = Path(audited_path or AUDITED_EXCEL_PATH)
        self.cache_path = Path(cache_path or CACHE_JSON_PATH)
        self.corrections: Dict[str, Dict[str, Any]] = {}
        self.load_cache()

    def _normalize_key(self, booklet_id: str) -> str:
        """Standardizes booklet key for robust matching."""
        if not booklet_id:
            return ""
        p = Path(str(booklet_id).strip())
        return p.stem.lower()

    def build_corrections_from_audited_excel(self) -> Dict[str, Dict[str, Any]]:
        """
        Parses audited_marksheet.xlsx and extracts all records with
        Audit_Status in ['MANUALLY_CORRECTED', 'APPROVED'].
        """
        if not self.audited_path.exists():
            logger.info(f"Audited marksheet not found at {self.audited_path}. Cache empty.")
            return {}

        try:
            df = pd.read_excel(self.audited_path, sheet_name=0)
        except Exception as e:
            logger.error(f"Failed to read audited marksheet: {e}")
            return {}

        corrections: Dict[str, Dict[str, Any]] = {}

        if "Booklet_ID" not in df.columns:
            return corrections

        # Check for audited status column
        status_col = "Audit_Status" if "Audit_Status" in df.columns else None

        for _, row in df.iterrows():
            audit_status = str(row.get(status_col, "")).strip().upper() if status_col else ""
            # Consider both MANUALLY_CORRECTED and APPROVED as verified ground truth
            if audit_status not in ["MANUALLY_CORRECTED", "APPROVED"]:
                continue

            course_code = str(row.get("Course_Code", "")).strip()
            if "517" in course_code:
                continue

            b_id = str(row.get("Booklet_ID", "")).strip()
            if not b_id or b_id.lower() == "nan":
                continue

            norm_key = self._normalize_key(b_id)
            record: Dict[str, Any] = {
                "booklet_id": b_id,
                "subject_folder": str(row.get("Subject_Folder", "")),
                "usn": str(row.get("USN", "")).strip(),
                "course_code": course_code,
                "audit_status": audit_status,
                "audited_at": str(row.get("Audited_At", "")),
                "marks": {},
            }

            for col in MARK_COLUMNS:
                if col in row and not pd.isna(row[col]):
                    try:
                        record["marks"][col] = int(round(float(row[col])))
                    except Exception:
                        pass

            corrections[norm_key] = record

        logger.info(f"Loaded {len(corrections)} verified human correction entries from {self.audited_path}")
        return corrections

    def sync_and_save(self) -> int:
        """
        Extracts verified corrections from the Excel ledger, updates
        in-memory cache, and saves to JSON cache file.
        Returns the count of cached corrections.
        """
        new_corrections = self.build_corrections_from_audited_excel()
        self.corrections = new_corrections

        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.corrections, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.corrections)} correction records to {self.cache_path}")
        except Exception as e:
            logger.error(f"Failed to write correction cache to {self.cache_path}: {e}")

        return len(self.corrections)

    def load_cache(self) -> None:
        """Loads corrections from JSON cache if present, otherwise rebuilds from Excel."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.corrections = json.load(f)
                logger.info(f"Loaded {len(self.corrections)} corrections from {self.cache_path}")
                return
            except Exception as e:
                logger.warning(f"Error loading {self.cache_path}: {e}. Rebuilding...")

        self.sync_and_save()

    def get_correction(self, booklet_id: str) -> Optional[Dict[str, Any]]:
        """
        Returns verified correction entry for a given booklet ID if cached, else None.
        """
        norm_key = self._normalize_key(booklet_id)
        return self.corrections.get(norm_key)

    def apply_correction(
        self,
        booklet_id: str,
        current_data: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool, List[str]]:
        """
        Applies cached human corrections to an extracted booklet record.
        Returns (updated_data, was_modified, applied_override_descriptions).
        """
        corr = self.get_correction(booklet_id)
        if not corr:
            return current_data, False, []

        updated = dict(current_data)
        applied_notes = []

        # Correct USN if verified
        corr_usn = corr.get("usn")
        if corr_usn and corr_usn != "nan" and corr_usn != "UNKNOWN":
            curr_usn = updated.get("usn", updated.get("USN", ""))
            if curr_usn != corr_usn:
                updated["usn"] = corr_usn
                updated["USN"] = corr_usn
                applied_notes.append(f"USN: {curr_usn}->{corr_usn}")
            if "cie_summary" in updated and isinstance(updated["cie_summary"], dict):
                updated["cie_summary"]["USN"] = corr_usn

        # Correct Course Code if verified
        corr_cc = corr.get("course_code")
        if corr_cc and corr_cc != "nan" and corr_cc != "UNKNOWN":
            curr_cc = updated.get("course_code", updated.get("Course_Code", ""))
            if curr_cc != corr_cc:
                updated["course_code"] = corr_cc
                updated["Course_Code"] = corr_cc
                applied_notes.append(f"CC: {curr_cc}->{corr_cc}")
            if "cie_summary" in updated and isinstance(updated["cie_summary"], dict):
                updated["cie_summary"]["Course_Code"] = corr_cc

        corr_folder = corr.get("subject_folder")
        if corr_folder and "cie_summary" in updated and isinstance(updated["cie_summary"], dict):
            updated["cie_summary"]["Subject_Folder"] = corr_folder

        # Correct mark values
        corr_marks = corr.get("marks", {})
        for col_name, val in corr_marks.items():
            cell_key = KEY_TO_CELL_MAP.get(col_name)

            # Check in values / resolved_values if present
            if "values" in updated and isinstance(updated["values"], dict) and cell_key:
                old_v = updated["values"].get(cell_key)
                if old_v != val:
                    updated["values"][cell_key] = val
                    applied_notes.append(f"{cell_key}: {old_v}->{val}")

            if "resolved_marks" in updated and isinstance(updated["resolved_marks"], dict) and cell_key:
                updated["resolved_marks"][cell_key] = val

            if "marks" in updated and isinstance(updated["marks"], dict):
                if cell_key in updated["marks"] and updated["marks"][cell_key] != val:
                    updated["marks"][cell_key] = val
                elif col_name in updated["marks"] and updated["marks"][col_name] != val:
                    updated["marks"][col_name] = val

            # Check in cie_summary if present
            if "cie_summary" in updated and isinstance(updated["cie_summary"], dict):
                old_v = updated["cie_summary"].get(col_name)
                if old_v != val:
                    updated["cie_summary"][col_name] = val
                    if not any(f"{cell_key}:" in n for n in applied_notes):
                        applied_notes.append(f"{col_name}: {old_v}->{val}")

            # Direct key check
            if col_name in updated:
                old_v = updated[col_name]
                if old_v != val:
                    updated[col_name] = val

        was_modified = len(applied_notes) > 0
        if was_modified:
            override_msg = "[ActiveLearner Cache: " + "; ".join(applied_notes) + "]"
            if "cie_summary" in updated and isinstance(updated["cie_summary"], dict):
                existing = updated["cie_summary"].get("Solver_Overrides", "")
                updated["cie_summary"]["Solver_Overrides"] = f"{existing} {override_msg}".strip()
                updated["cie_summary"]["Status"] = "AUTO_CORRECTED"
            if "status" in updated:
                updated["status"] = "AUTO_CORRECTED"
            if "overrides" in updated and isinstance(updated["overrides"], list):
                updated["overrides"].append(override_msg)

        return updated, was_modified, applied_notes


# Singleton helper instance
_active_learner_instance: Optional[ActiveLearner] = None


def get_active_learner() -> ActiveLearner:
    """Returns singleton ActiveLearner instance."""
    global _active_learner_instance
    if _active_learner_instance is None:
        _active_learner_instance = ActiveLearner()
    return _active_learner_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    learner = ActiveLearner()
    count = learner.sync_and_save()
    print(f"Active Learner synced successfully. Total cached records: {count}")
