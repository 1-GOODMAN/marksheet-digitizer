"""
Main Entry Point for Marksheet Digitizer — Page-1-Only Multi-Subject Batch Processing.
Iterates across subject folders (raw, Sub1, Sub2), runs the Page-1 pipeline on every
discovered image, and exports a multi-sheet Excel audit report.
"""

import sys
import argparse
import time
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.pipeline import BatchProcessor
from src.config import OUTPUT_DIR, DATA_DIR, VALID_COURSE_CODES, discover_subject_dirs


# ---------------------------------------------------------------------------
# Excel Export
# ---------------------------------------------------------------------------

def _save_excel(records: list, out_path: str, purge_invalid: bool = True) -> None:
    """
    Writes the multi-sheet Excel audit report.
    Applies strict purge filter: drops records with Course_Code == '24MCPC517'
    and any decimal-containing rows before generating the final Excel workbook.

    Sheet layout:
      CIE_Summary   — all records from valid subject folders
      raw           — records from data/raw only
      Sub2          — records from data/Sub2 only
      <course_code> — one sheet per valid course code (e.g. 24MCPC522, 24MCPC521)
    """
    if not records:
        print("  [WARN] No records to export.")
        return

    # Apply strict purge filter if enabled
    if purge_invalid:
        records_to_export, purge_stats = BatchProcessor.purge_invalid_export_records(records)
        print(f"  [Purge Filter] Input: {purge_stats['total_in']} | Dropped 24MCPC517: {purge_stats['dropped_517']} | Dropped Decimals: {purge_stats['dropped_decimal']} | Exporting: {purge_stats['passed']}")
    else:
        records_to_export = records

    if not records_to_export:
        print("  [WARN] All records were filtered out by export purge filter.")
        return

    cie_columns = [
        "Subject_Folder", "Booklet_ID", "USN", "Course_Code",
        "Test1_15", "Test2_15", "ABA_10", "Theory_Total_40",
        "Viva_5", "Lab_5", "Practical_Total_10", "Total_CIE_50",
        "Status", "Solver_Overrides",
    ]

    df_all = pd.DataFrame(records_to_export)
    available_cols = [c for c in cie_columns if c in df_all.columns]
    df_all = df_all[available_cols]

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(str(out_file), engine="openpyxl") as writer:
        # Master sheet
        df_all.to_excel(writer, sheet_name="CIE_Summary", index=False)

        # Per-subject-folder sheets (dynamically discovered from processed records)
        subject_folders = []
        if "Subject_Folder" in df_all.columns:
            for s in df_all["Subject_Folder"].dropna().unique():
                if s and str(s) not in subject_folders:
                    subject_folders.append(str(s))

        for subdir in subject_folders:
            subset = df_all[df_all["Subject_Folder"] == subdir]
            sheet_name = str(subdir)[:31]  # Excel sheet name max 31 chars
            if not subset.empty:
                subset.to_excel(writer, sheet_name=sheet_name, index=False)

        # Per-course-code sheets
        for code in VALID_COURSE_CODES:
            if "Course_Code" in df_all.columns:
                subset = df_all[df_all["Course_Code"] == code]
                if not subset.empty:
                    subset.to_excel(writer, sheet_name=code[:31], index=False)

    print(f"  Excel saved → {out_file.resolve()}")


# ---------------------------------------------------------------------------
# Arithmetic verification (post-export audit)
# ---------------------------------------------------------------------------

from typing import Tuple


def _verify_constraints(records: list) -> Tuple[int, int, int]:
    """
    Verifies that all three arithmetic constraints hold for every output row.
    Returns (total, pass_count, fail_count).
    """
    total = len(records)
    fails = 0
    for r in records:
        c1 = r.get("Test1_15", 0) + r.get("Test2_15", 0) + r.get("ABA_10", 0) == r.get("Theory_Total_40", 0)
        c2 = r.get("Viva_5", 0) + r.get("Lab_5", 0) == r.get("Practical_Total_10", 0)
        c3 = r.get("Theory_Total_40", 0) + r.get("Practical_Total_10", 0) == r.get("Total_CIE_50", 0)
        status = r.get("Status", "")
        if not (c1 and c2 and c3) and status != "MANUAL_REVIEW_REQUIRED":
            fails += 1
    return total, total - fails, fails


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Marksheet Digitizer — Page-1-Only Multi-Subject Batch Pipeline"
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Subject folders to process (default: all folders in data/ containing images)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max images to process per subject folder",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR / "digitized_marksheet.xlsx"),
        help="Excel export path",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 90)
    print("  Marksheet Digitizer — Page-1-Only Neuro-Symbolic Batch Pipeline")
    print("=" * 90)

    discovered = discover_subject_dirs(DATA_DIR)
    subjects_to_process = args.subjects if args.subjects is not None else discovered
    if not subjects_to_process:
        subjects_to_process = [d for d in ["raw", "Sub1", "Sub2", "Sub3"] if (DATA_DIR / d).exists()]

    processor = BatchProcessor()
    print(f"  Hardware device : {processor.ocr.device}")
    print(f"  Subject folders : {subjects_to_process}")
    print(f"  Output path     : {args.output}")
    print("-" * 90)

    all_records: list = []
    global_status_counts = {"VALIDATED": 0, "AUTO_CORRECTED": 0, "MANUAL_REVIEW_REQUIRED": 0}
    grand_start = time.time()

    for subject_folder in subjects_to_process:
        images = processor._discover_subject_images(subject_folder)
        if not images:
            print(f"  [SKIP] {subject_folder}: no images found in {DATA_DIR / subject_folder}")
            continue

        if args.limit:
            images = images[: args.limit]

        n = len(images)
        print(f"\n[{subject_folder}] Processing {n} image(s) from data/{subject_folder}/")
        print("-" * 90)

        folder_start = time.time()

        for idx, img_path in enumerate(images, start=1):
            booklet_id = img_path.stem  # filename without extension
            t0 = time.time()
            res = processor.process_page1_image(
                img_path, booklet_id=booklet_id, subject_folder=subject_folder
            )
            elapsed = time.time() - t0

            status = res["status"]
            global_status_counts[status] = global_status_counts.get(status, 0) + 1

            cie = res["cie_summary"]
            all_records.append(cie)

            usn_disp = cie.get("USN", "N/A")
            cc_disp = cie.get("Course_Code", "N/A")
            cie_total = cie.get("Total_CIE_50", "?")

            print(
                f"  [{idx:03d}/{n:03d}] {booklet_id:<30} | USN: {usn_disp:<12} | "
                f"CC: {cc_disp:<12} | CIE: {cie_total:>2} | {status} ({elapsed:.1f}s)"
            )
            if status == "AUTO_CORRECTED" and cie.get("Solver_Overrides"):
                print(f"           └─ Overrides: {cie['Solver_Overrides']}")
            if not res["success"] and res.get("error"):
                print(f"           └─ Error: {res['error'][:100]}")

            # Checkpoint save every 10 images
            if idx % 10 == 0 or idx == n:
                try:
                    _save_excel(all_records, args.output)
                except Exception as exc:
                    print(f"  [WARN] Checkpoint save failed: {exc}")

        folder_elapsed = time.time() - folder_start
        print(f"  [{subject_folder}] Completed {n} images in {folder_elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # Final Export & Verification
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - grand_start
    print("\n" + "=" * 90)
    print("  Final Export")
    print("=" * 90)

    _save_excel(all_records, args.output)

    total, passed, failed = _verify_constraints(all_records)
    print(f"\n  Arithmetic constraint verification:")
    print(f"    Total records  : {total}")
    print(f"    Constraints OK : {passed}")
    print(f"    Violations     : {failed}  (excludes MANUAL_REVIEW rows)")

    val = global_status_counts.get("VALIDATED", 0)
    auto = global_status_counts.get("AUTO_CORRECTED", 0)
    rev = global_status_counts.get("MANUAL_REVIEW_REQUIRED", 0)
    n_total = max(1, len(all_records))

    print(f"\n{'=' * 55}")
    print(f"  {'BATCH EXECUTION SUMMARY':^51}")
    print(f"{'=' * 55}")
    print(f"  Total Images Processed  : {len(all_records):>4d}")
    print(f"  Validated               : {val:>4d}  ({val / n_total * 100:>5.1f}%)")
    print(f"  Auto-Corrected by Z3    : {auto:>4d}  ({auto / n_total * 100:>5.1f}%)")
    print(f"  Flagged for Review      : {rev:>4d}  ({rev / n_total * 100:>5.1f}%)")
    print(f"  Total runtime           : {total_elapsed:.1f}s  "
          f"({total_elapsed / n_total:.2f}s/image)")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
