"""
End-to-End Batch Pipeline Orchestrator for Marksheet Digitizer.
Page-1-Only Architecture: integrates Table Extraction (CV), Probabilistic TrOCR (Neural),
and Z3 Arithmetic Constraint Reconciliation (Symbolic).
Supports multi-subject folder processing (raw, Sub1, Sub2).
"""

import glob
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import cv2
import numpy as np

from .config import DATA_DIR, RAW_DATA_DIR, OUTPUT_DIR, SUBJECT_DIRS, FOLDER_COURSE_CODES
from .preprocessor import TableSegmenter
from .ocr_engine import ProbabilisticOCR
from .solver import MarksheetSolver


class BatchProcessor:
    """
    Orchestrates booklet image ingestion, table cell extraction, probabilistic OCR,
    and neuro-symbolic constraint satisfaction for Page-1-Only CIE mark extraction.

    Supports multi-subject folder mode: iterates across raw/, Sub1/, Sub2/ directories.
    For 'raw/': discovers files matching *_p1.jpg (booklet pairs).
    For 'Sub1/', 'Sub2/': discovers any .jpg/.png without '_p2' in the filename.
    Strictly deduplicates file paths using set(Path.resolve()).
    """

    P1_KEYS = [
        "test1", "test2", "aba", "theory_total",
        "viva", "lab", "pract_total", "cie_total",
    ]

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        ocr: Optional[ProbabilisticOCR] = None,
        segmenter: Optional[TableSegmenter] = None,
        solver: Optional[MarksheetSolver] = None,
    ):
        self.data_dir = Path(data_dir or DATA_DIR)
        self.output_dir = Path(output_dir or OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ocr = ocr or ProbabilisticOCR()
        self.segmenter = segmenter or TableSegmenter()
        self.solver = solver or MarksheetSolver()

    # ------------------------------------------------------------------
    # Discovery helper
    # ------------------------------------------------------------------

    def _discover_subject_images(self, subject_folder: str) -> List[Path]:
        """
        Returns a sorted, strictly deduplicated list of image paths from DATA_DIR/subject_folder.
        - 'raw': returns *_p1.jpg files only (Page 1 of booklet pairs).
        - 'Sub1', 'Sub2', others: returns all .jpg/.png files that do NOT contain '_p2'.
        Uses set(f.resolve()) to avoid Windows case-insensitive duplicate globbing!
        """
        folder = self.data_dir / subject_folder
        if not folder.exists():
            return []

        resolved_files = [
            f.resolve() for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        if subject_folder.lower() == "raw":
            filtered = [f for f in resolved_files if f.name.lower().endswith("_p1.jpg")]
        else:
            filtered = [f for f in resolved_files if "_p2" not in f.name.lower()]

        # Deduplicate using set on resolved paths and sort by name
        unique_paths = sorted(list(set(filtered)), key=lambda p: p.name)
        return [Path(p) for p in unique_paths]

    # ------------------------------------------------------------------
    # Core single-image processor
    # ------------------------------------------------------------------

    def process_page1_image(
        self,
        img_input: Union[str, Path, np.ndarray],
        booklet_id: str = "unknown",
        subject_folder: str = "raw",
    ) -> Dict[str, Any]:
        """
        Executes the full Page-1-Only pipeline on a single image.

        Steps:
            A. Preprocessing   — USN grid, course code strip, 8 CIE mark cells
            B. OCR Inference   — USN hard-lock, folder-aware course code snap, digit candidates
            C. Symbolic Solve  — Z3 MaxSMT enforcing C1/C2/C3 with total-anchoring
            D. Output Packaging
        """
        booklet_name = str(booklet_id)
        try:
            # ----------------------------------------------------------------
            # Step A: Preprocessing
            # ----------------------------------------------------------------
            p1_data = self.segmenter.extract_page1_cells(img_input)

            # ----------------------------------------------------------------
            # Step B: OCR Inference
            # ----------------------------------------------------------------
            # USN: hard-locked prefix 4MT24MC + 3-digit suffix (18% border shave)
            usn_text = self.ocr.predict_usn(
                p1_data.get("usn_crop"),
                p1_data.get("usn_char_boxes"),
                p1_data.get("image"),
            )

            # Course Code: folder-aware mapping
            folder_default = FOLDER_COURSE_CODES.get(subject_folder, "24MCPC522")
            course_code_text = self.ocr.predict_course_code(
                p1_data.get("course_code_crop"),
                folder_default=folder_default,
            )

            # Marks: batch digit candidate prediction over all 8 P1 cells
            all_crops: Dict[str, np.ndarray] = {}
            crop_stds: Dict[str, float] = {}
            for k, c in p1_data.get("p1_marks_crops", {}).items():
                if k in self.P1_KEYS:
                    all_crops[k] = c
                    if c is not None and c.size > 0:
                        c_g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if len(c.shape) == 3 else c
                        crop_stds[k] = float(c_g.std())
                    else:
                        crop_stds[k] = 0.0

            if hasattr(self.ocr, "predict_digit_candidates_batch"):
                candidate_distributions = self.ocr.predict_digit_candidates_batch(
                    all_crops, beam_size=5
                )
            else:
                candidate_distributions = {
                    k: self.ocr.predict_digit_candidates(c, beam_size=5)
                    for k, c in all_crops.items()
                }

            # Top-1 raw predictions (pre-solver)
            raw_top1: Dict[str, int] = {
                k: max(dist, key=dist.get)
                for k, dist in candidate_distributions.items()
            }

            # ----------------------------------------------------------------
            # Step C: Symbolic Constraint Verification & Z3 Optimization
            # ----------------------------------------------------------------
            t1 = raw_top1.get("test1", 0)
            t2 = raw_top1.get("test2", 0)
            aba = raw_top1.get("aba", 0)
            theory = raw_top1.get("theory_total", 0)
            viva = raw_top1.get("viva", 0)
            lab = raw_top1.get("lab", 0)
            pract = raw_top1.get("pract_total", 0)
            cie = raw_top1.get("cie_total", 0)

            c1 = (t1 + t2 + aba == theory)
            c2 = (viva + lab == pract)
            c3 = (theory + pract == cie)
            raw_valid = c1 and c2 and c3

            overrides_list: List[str] = []

            if raw_valid:
                status = "VALIDATED"
                resolved_values = dict(raw_top1)
            else:
                solve_res = self.solver.solve(candidate_distributions, crop_stds=crop_stds)
                if solve_res.get("success", False):
                    resolved_values = solve_res["values"]
                    for k in sorted(resolved_values.keys()):
                        if resolved_values[k] != raw_top1.get(k):
                            overrides_list.append(
                                f"{k}: {raw_top1.get(k)}->{resolved_values[k]}"
                            )
                    status = "AUTO_CORRECTED" if overrides_list else "VALIDATED"
                else:
                    status = "MANUAL_REVIEW_REQUIRED"
                    resolved_values = dict(raw_top1)
                    overrides_list = ["UNSAT_ARITHMETIC_ERROR"]

            solver_overrides_str = "; ".join(overrides_list)

            # ----------------------------------------------------------------
            # Step D: Package Output
            # ----------------------------------------------------------------
            cie_summary_row = {
                "Subject_Folder": subject_folder,
                "Booklet_ID": booklet_name,
                "USN": usn_text or "UNKNOWN",
                "Course_Code": course_code_text or "UNKNOWN",
                "Test1_15": resolved_values.get("test1", 0),
                "Test2_15": resolved_values.get("test2", 0),
                "ABA_10": resolved_values.get("aba", 0),
                "Theory_Total_40": resolved_values.get("theory_total", 0),
                "Viva_5": resolved_values.get("viva", 0),
                "Lab_5": resolved_values.get("lab", 0),
                "Practical_Total_10": resolved_values.get("pract_total", 0),
                "Total_CIE_50": resolved_values.get("cie_total", 0),
                "Status": status,
                "Solver_Overrides": solver_overrides_str,
            }

            return {
                "success": True,
                "booklet_id": booklet_name,
                "status": status,
                "values": resolved_values,
                "raw_top1": raw_top1,
                "overrides": overrides_list,
                "cie_summary": cie_summary_row,
                "candidate_distributions": candidate_distributions,
                "p1_data": p1_data,
            }

        except Exception as e:
            import traceback
            err_detail = traceback.format_exc()
            return {
                "success": False,
                "booklet_id": booklet_name,
                "status": "MANUAL_REVIEW_REQUIRED",
                "error": str(e),
                "traceback": err_detail,
                "cie_summary": {
                    "Subject_Folder": subject_folder,
                    "Booklet_ID": booklet_name,
                    "USN": "ERROR",
                    "Course_Code": "ERROR",
                    "Test1_15": 0,
                    "Test2_15": 0,
                    "ABA_10": 0,
                    "Theory_Total_40": 0,
                    "Viva_5": 0,
                    "Lab_5": 0,
                    "Practical_Total_10": 0,
                    "Total_CIE_50": 0,
                    "Status": "MANUAL_REVIEW_REQUIRED",
                    "Solver_Overrides": f"EXCEPTION: {str(e)[:80]}",
                },
            }

    # ------------------------------------------------------------------
    # Legacy two-page compatibility shim (kept for server.py back-compat)
    # ------------------------------------------------------------------

    def process_images(
        self,
        p1_input: Union[str, Path, np.ndarray],
        p2_input: Union[str, Path, np.ndarray, None] = None,
        booklet_id: str = "custom_booklet",
        subject_folder: str = "raw",
    ) -> Dict[str, Any]:
        """
        Compatibility shim: processes Page 1 only (p2_input is accepted but ignored).
        Returns the same dict shape as process_page1_image() for drop-in server.py usage.
        """
        return self.process_page1_image(p1_input, booklet_id=booklet_id, subject_folder=subject_folder)

    def process_booklet(self, booklet_id: Union[int, str]) -> Dict[str, Any]:
        """
        Legacy raw-folder booklet processor (booklet_XXX_p1.jpg).
        Kept for backwards compatibility. Calls process_page1_image internally.
        """
        if isinstance(booklet_id, int):
            booklet_name = f"booklet_{booklet_id:03d}"
        else:
            booklet_name = str(booklet_id)

        p1_path = self.data_dir / "raw" / f"{booklet_name}_p1.jpg"
        if not p1_path.exists():
            return {
                "success": False,
                "booklet_id": booklet_name,
                "status": "MANUAL_REVIEW_REQUIRED",
                "error": f"Missing image: {p1_path}",
                "cie_summary": {
                    "Subject_Folder": "raw",
                    "Booklet_ID": booklet_name,
                    "USN": "UNKNOWN",
                    "Course_Code": "UNKNOWN",
                    "Test1_15": 0, "Test2_15": 0, "ABA_10": 0,
                    "Theory_Total_40": 0, "Viva_5": 0, "Lab_5": 0,
                    "Practical_Total_10": 0, "Total_CIE_50": 0,
                    "Status": "MANUAL_REVIEW_REQUIRED",
                    "Solver_Overrides": "MISSING_IMAGE_FILE",
                },
            }

        return self.process_page1_image(p1_path, booklet_id=booklet_name, subject_folder="raw")
