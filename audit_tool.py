"""
Marksheet Visual Audit and Verification Dashboard.
A Streamlit web application for human verification and correction of OCR and Z3-extracted CIE marks.
Supports smart merging of new batches (e.g. Sub2, Sub3), folder-specific filtering, and live on-demand processing.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import io
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import streamlit as st

# Setup Project Root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import discover_subject_dirs, DATA_DIR
from src.active_learner import get_active_learner

# Paths configuration
DEFAULT_DIGITIZED_EXCEL = PROJECT_ROOT / "data" / "output" / "digitized_marksheet.xlsx"
FALLBACK_DIGITIZED_EXCEL = PROJECT_ROOT / "data" / "output" / "review1_marksheet.xlsx"
DEFAULT_AUDITED_EXCEL = PROJECT_ROOT / "data" / "output" / "audited_marksheet.xlsx"

# Standard columns expected in the dataset
CORE_FIELDS = [
    "USN",
    "Course_Code",
    "Test1_15",
    "Test2_15",
    "ABA_10",
    "Theory_Total_40",
    "Viva_5",
    "Lab_5",
    "Practical_Total_10",
    "Total_CIE_50",
]


def resolve_image_file(subject_folder: Optional[str], booklet_id: Optional[str], image_file: Optional[str] = None) -> Optional[Path]:
    """
    Resolve the original booklet image file path across multiple search patterns.
    """
    candidates = []
    
    # 1. Direct image_file check
    if image_file and str(image_file).strip() and str(image_file).strip().lower() != "nan":
        img_p = Path(str(image_file).strip())
        candidates.append(img_p)
        candidates.append(PROJECT_ROOT / img_p)
        candidates.append(PROJECT_ROOT / "data" / img_p)
        candidates.append(PROJECT_ROOT / "cloud_vault" / "review1_uploads" / img_p.name)

    if not booklet_id or str(booklet_id).strip() == "" or str(booklet_id).strip().lower() == "nan":
        for c in candidates:
            if c.exists() and c.is_file():
                return c.resolve()
        return None

    b_id = str(booklet_id).strip()
    s_folder = str(subject_folder).strip() if subject_folder and str(subject_folder).lower() != "nan" else ""

    # Common extensions to try
    extensions = ["", ".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"]

    # Subject-specific search paths
    base_folders = []
    if s_folder:
        base_folders.append(PROJECT_ROOT / "data" / s_folder)
    for sub in discover_subject_dirs(PROJECT_ROOT / "data"):
        base_folders.append(PROJECT_ROOT / "data" / sub)
    base_folders.extend([
        PROJECT_ROOT / "data" / "Sub3",
        PROJECT_ROOT / "data" / "Sub2",
        PROJECT_ROOT / "data" / "Sub1",
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "cloud_vault" / "review1_uploads",
    ])

    for folder in base_folders:
        for ext in extensions:
            if ext == "":
                candidates.append(folder / b_id)
            else:
                candidates.append(folder / f"{b_id}{ext}")

    for c in candidates:
        try:
            if c.exists() and c.is_file():
                return c.resolve()
        except Exception:
            continue

    # Fallback: case-insensitive match on file stem inside candidate folders
    for folder in base_folders:
        if folder.exists() and folder.is_dir():
            try:
                for f in folder.iterdir():
                    if f.is_file() and f.stem.lower() == Path(b_id).stem.lower():
                        return f.resolve()
            except Exception:
                continue

    return None


def export_dataframe_to_excel(df: pd.DataFrame, output_path: Path) -> bytes:
    """
    Save audited DataFrame to formatted Excel workbook and return raw bytes.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "CIE_Summary"

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

    fill_approved = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid")
    fill_corrected = PatternFill(start_color="DBEAFE", end_color="DBEAFE", fill_type="solid")
    fill_pending = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")

    columns = list(df.columns)
    ws.append(columns)

    for col_idx, col_name in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_align
        cell.border = thin_border

    for row_idx, row_data in enumerate(df.itertuples(index=False), start=2):
        row_list = list(row_data)
        ws.append(row_list)
        status_val = str(getattr(row_data, "Audit_Status", "PENDING")).upper()

        for col_idx in range(1, len(columns) + 1):
            c = ws.cell(row=row_idx, column=col_idx)
            c.border = thin_border
            c.alignment = left_align if columns[col_idx - 1] in ["Booklet_ID", "Solver_Overrides", "Reviewer_Notes"] else center_align

            if columns[col_idx - 1] == "Audit_Status":
                if "APPROVED" in status_val:
                    c.fill = fill_approved
                elif "CORRECTED" in status_val:
                    c.fill = fill_corrected
                else:
                    c.fill = fill_pending

    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    wb.save(output_path)

    # Automatically keep the Active Learning correction cache synchronized
    if output_path.resolve() == DEFAULT_AUDITED_EXCEL.resolve():
        try:
            get_active_learner().sync_and_save()
        except Exception:
            pass

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


def load_initial_data() -> pd.DataFrame:
    """
    Smart Merge on Startup:
    Reads both digitized_marksheet.xlsx (latest extracted records) and audited_marksheet.xlsx.
    Preserves all already-audited rows (with their human corrections, approval statuses, and notes),
    and appends any newly processed records (e.g. from Sub2 or Sub3) as PENDING.
    """
    df_audited: Optional[pd.DataFrame] = None
    df_digitized: Optional[pd.DataFrame] = None

    # 1. Read existing audited marksheet
    if DEFAULT_AUDITED_EXCEL.exists():
        try:
            df_audited = pd.read_excel(DEFAULT_AUDITED_EXCEL, sheet_name="CIE_Summary")
        except Exception:
            try:
                df_audited = pd.read_excel(DEFAULT_AUDITED_EXCEL)
            except Exception:
                df_audited = None

    # 2. Read latest digitized marksheet
    if DEFAULT_DIGITIZED_EXCEL.exists():
        try:
            df_digitized = pd.read_excel(DEFAULT_DIGITIZED_EXCEL, sheet_name="CIE_Summary")
        except Exception:
            try:
                df_digitized = pd.read_excel(DEFAULT_DIGITIZED_EXCEL)
            except Exception:
                df_digitized = None
    elif FALLBACK_DIGITIZED_EXCEL.exists():
        try:
            df_digitized = pd.read_excel(FALLBACK_DIGITIZED_EXCEL, sheet_name="CIE_Summary")
        except Exception:
            try:
                df_digitized = pd.read_excel(FALLBACK_DIGITIZED_EXCEL)
            except Exception:
                df_digitized = None

    # Merge logic
    if df_audited is not None and df_digitized is not None:
        # Ensure audit columns exist in audited df
        if "Audit_Status" not in df_audited.columns:
            df_audited["Audit_Status"] = "PENDING"
        else:
            df_audited["Audit_Status"] = df_audited["Audit_Status"].fillna("PENDING").astype(str)

        if "Audited_At" not in df_audited.columns:
            df_audited["Audited_At"] = ""

        if "Reviewer_Notes" not in df_audited.columns:
            df_audited["Reviewer_Notes"] = ""

        # Identify existing records by (Subject_Folder, Booklet_ID)
        existing_keys = set(
            zip(
                df_audited["Subject_Folder"].astype(str).str.strip(),
                df_audited["Booklet_ID"].astype(str).str.strip()
            )
        )

        new_rows = []
        for _, row in df_digitized.iterrows():
            key = (str(row.get("Subject_Folder", "")).strip(), str(row.get("Booklet_ID", "")).strip())
            if key not in existing_keys:
                r_dict = row.to_dict()
                r_dict["Audit_Status"] = "PENDING"
                r_dict["Audited_At"] = ""
                r_dict["Reviewer_Notes"] = ""
                new_rows.append(r_dict)
                existing_keys.add(key)

        if new_rows:
            df = pd.concat([df_audited, pd.DataFrame(new_rows)], ignore_index=True)
            # Auto-save merged ledger so newly merged records persist
            try:
                export_dataframe_to_excel(df, DEFAULT_AUDITED_EXCEL)
            except Exception:
                pass
        else:
            df = df_audited

    elif df_audited is not None:
        df = df_audited
    elif df_digitized is not None:
        df = df_digitized
    else:
        df = pd.DataFrame(columns=[
            "Subject_Folder", "Booklet_ID", "USN", "Course_Code",
            "Test1_15", "Test2_15", "ABA_10", "Theory_Total_40",
            "Viva_5", "Lab_5", "Practical_Total_10", "Total_CIE_50",
            "Status", "Solver_Overrides"
        ])

    # Ensure required audit tracking columns exist
    if "Audit_Status" not in df.columns:
        df["Audit_Status"] = "PENDING"
    else:
        df["Audit_Status"] = df["Audit_Status"].fillna("PENDING").astype(str)

    if "Audited_At" not in df.columns:
        df["Audited_At"] = ""

    if "Reviewer_Notes" not in df.columns:
        df["Reviewer_Notes"] = ""

    numeric_cols = [
        "Test1_15", "Test2_15", "ABA_10", "Theory_Total_40",
        "Viva_5", "Lab_5", "Practical_Total_10", "Total_CIE_50"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df


def get_unprocessed_images(current_df: pd.DataFrame) -> Dict[str, List[Path]]:
    """
    Scans disk folders (raw, Sub1, Sub2, Sub3, etc.) and returns any images
    that have not been processed into the digitized/audited dataset yet.
    """
    unprocessed: Dict[str, List[Path]] = {}
    existing_keys = set()
    if "Subject_Folder" in current_df.columns and "Booklet_ID" in current_df.columns:
        existing_keys = set(
            zip(
                current_df["Subject_Folder"].astype(str).str.strip(),
                current_df["Booklet_ID"].astype(str).str.strip()
            )
        )

    # Folders to check
    candidate_folders = []
    for d in ["Sub3", "Sub2", "Sub1", "raw"]:
        candidate_folders.append(d)
    for d in discover_subject_dirs(PROJECT_ROOT / "data"):
        if d not in candidate_folders:
            candidate_folders.append(d)

    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

    for f_name in candidate_folders:
        dir_p = PROJECT_ROOT / "data" / f_name
        if not dir_p.exists() or not dir_p.is_dir():
            continue

        folder_imgs = []
        for file_p in sorted(dir_p.iterdir()):
            if not file_p.is_file() or file_p.suffix.lower() not in image_exts:
                continue
            if f_name.lower() == "raw" and not file_p.name.lower().endswith("_p1.jpg"):
                continue
            if f_name.lower() != "raw" and "_p2" in file_p.name.lower():
                continue

            stem = file_p.stem
            if (f_name, stem) not in existing_keys and (f_name, file_p.name) not in existing_keys:
                folder_imgs.append(file_p)

        if folder_imgs:
            unprocessed[f_name] = folder_imgs

    return unprocessed


def process_missing_images_batch(unprocessed_dict: Dict[str, List[Path]]) -> int:
    """
    Runs the Page-1 neuro-symbolic pipeline on unprocessed images and updates
    both digitized_marksheet.xlsx and st.session_state.audited_df.
    """
    from src.pipeline import BatchProcessor
    processor = BatchProcessor()

    new_records = []
    total_imgs = sum(len(imgs) for imgs in unprocessed_dict.values())
    if total_imgs == 0:
        return 0

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    processed_count = 0

    for subject_folder, img_list in unprocessed_dict.items():
        for img_path in img_list:
            processed_count += 1
            status_text.text(f"Processing ({processed_count}/{total_imgs}): {subject_folder}/{img_path.name}")
            progress_bar.progress(processed_count / total_imgs)

            booklet_id = img_path.stem
            try:
                res = processor.process_page1_image(
                    img_path, booklet_id=booklet_id, subject_folder=subject_folder
                )
                cie = res.get("cie_summary", {})
                cie["Audit_Status"] = "PENDING"
                cie["Audited_At"] = ""
                cie["Reviewer_Notes"] = ""
                new_records.append(cie)
            except Exception as e:
                # Add fallback error record
                err_record = {
                    "Subject_Folder": subject_folder,
                    "Booklet_ID": booklet_id,
                    "USN": "UNKNOWN",
                    "Course_Code": "UNKNOWN",
                    "Test1_15": 0, "Test2_15": 0, "ABA_10": 0, "Theory_Total_40": 0,
                    "Viva_5": 0, "Lab_5": 0, "Practical_Total_10": 0, "Total_CIE_50": 0,
                    "Status": "MANUAL_REVIEW_REQUIRED",
                    "Solver_Overrides": f"Processing error: {str(e)}",
                    "Audit_Status": "PENDING",
                    "Audited_At": "",
                    "Reviewer_Notes": f"Failed during batch processing: {str(e)}"
                }
                new_records.append(err_record)

    progress_bar.empty()
    status_text.empty()

    if new_records:
        new_df = pd.DataFrame(new_records)
        updated_df = pd.concat([st.session_state.audited_df, new_df], ignore_index=True)
        st.session_state.audited_df = updated_df
        export_dataframe_to_excel(updated_df, DEFAULT_AUDITED_EXCEL)

        # Also append to digitized_marksheet.xlsx
        try:
            if DEFAULT_DIGITIZED_EXCEL.exists():
                dig_df = pd.read_excel(DEFAULT_DIGITIZED_EXCEL, sheet_name="CIE_Summary")
                combined_dig = pd.concat([dig_df, pd.DataFrame(new_records)], ignore_index=True)
                export_dataframe_to_excel(combined_dig, DEFAULT_DIGITIZED_EXCEL)
        except Exception:
            pass

    return len(new_records)


def main():
    # Set Streamlit Page Configuration
    st.set_page_config(
        page_title="MITE Marksheet Visual Audit & Verification",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom Styling
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 1.85rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 0.25rem;
        }
        .sub-title {
            font-size: 0.95rem;
            color: #64748b;
            margin-bottom: 1.25rem;
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
        .badge-approved {
            background-color: #dcfce7;
            color: #15803d;
            border: 1px solid #86efac;
        }
        .badge-corrected {
            background-color: #dbeafe;
            color: #1d4ed8;
            border: 1px solid #93c5fd;
        }
        .badge-pending {
            background-color: #fef3c7;
            color: #b45309;
            border: 1px solid #fde68a;
        }
        .alert-card {
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 16px;
            border-left: 4px solid;
        }
        .alert-card-warning {
            background-color: #fffbeb;
            border-color: #f59e0b;
            color: #92400e;
        }
        .alert-card-info {
            background-color: #eff6ff;
            border-color: #3b82f6;
            color: #1e40af;
        }
        .stat-box {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 10px 14px;
            text-align: center;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        }
        .stat-number {
            font-size: 1.4rem;
            font-weight: 700;
            color: #0f172a;
        }
        .stat-label {
            font-size: 0.75rem;
            color: #64748b;
            text-transform: uppercase;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize Session State
    if "audited_df" not in st.session_state:
        st.session_state.audited_df = load_initial_data()

    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    df = st.session_state.audited_df
    total_records = len(df)

    # Check for unprocessed images on disk
    unprocessed_dict = get_unprocessed_images(df)
    unprocessed_count = sum(len(v) for v in unprocessed_dict.values())

    # -------------------------------------------------------------
    # Sidebar: Audit Progress & Controls
    # -------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🎓 **Audit & Verification**")
        st.caption("Human-in-the-Loop AI Verification System")
        st.markdown("---")

        # ---------------------------------------------------------
        # Dynamic Processing Button for Missing Sub2 / Sub3 Images
        # ---------------------------------------------------------
        if unprocessed_count > 0:
            st.markdown(
                f"""
                <div class='alert-card alert-card-info' style='padding: 8px 12px; margin-bottom: 10px;'>
                    <strong>🔔 {unprocessed_count} New Image(s) Detected!</strong><br/>
                    <small>Found in {', '.join(unprocessed_dict.keys())} waiting for digitization.</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("🔄 Process Missing Sub2/Sub3 Images Now", type="primary", use_container_width=True):
                with st.spinner("Processing new booklet images with TrOCR and Z3 solver..."):
                    n_done = process_missing_images_batch(unprocessed_dict)
                    st.success(f"Successfully processed and added {n_done} new booklet(s)!")
                    st.rerun()
            st.markdown("---")

        # ---------------------------------------------------------
        # Sidebar Folder Filter (Jump Straight to Sub3 / Sub2)
        # ---------------------------------------------------------
        st.markdown("#### 📁 **Subject Folder Filter**")

        # Gather available folders
        present_folders = []
        if "Subject_Folder" in df.columns:
            present_folders = [str(f) for f in df["Subject_Folder"].dropna().unique() if str(f).strip()]

        folder_options = ["All Folders"]
        # Prioritize Sub3, Sub2, Sub1, raw in order
        for preferred in ["Sub3", "Sub2", "Sub1", "raw"]:
            if preferred not in folder_options:
                folder_options.append(preferred)

        for f in present_folders:
            if f not in folder_options:
                folder_options.append(f)

        # Default to "Sub3" if Sub3 exists in records or on disk; otherwise "All Folders"
        default_folder_idx = 0
        if "Sub3" in folder_options:
            sub3_has_records = "Sub3" in present_folders
            sub3_has_files = (PROJECT_ROOT / "data" / "Sub3").exists()
            if sub3_has_records or sub3_has_files:
                default_folder_idx = folder_options.index("Sub3")

        selected_folder = st.selectbox(
            "Select Folder to Audit:",
            folder_options,
            index=default_folder_idx,
            help="Filter and navigate records belonging only to the selected subject folder.",
        )

        # ---------------------------------------------------------
        # Status Filter
        # ---------------------------------------------------------
        selected_status_filter = st.selectbox(
            "🔍 Audit Status Filter:",
            ["All", "PENDING", "APPROVED", "MANUALLY_CORRECTED", "Auto-Corrected (Z3 Overrides)"],
            index=0,
        )

        # ---------------------------------------------------------
        # Filter Indices Computation
        # ---------------------------------------------------------
        filter_mask = pd.Series(True, index=df.index)

        # Folder condition
        if selected_folder != "All Folders":
            if "Subject_Folder" in df.columns:
                filter_mask = filter_mask & (df["Subject_Folder"].astype(str) == selected_folder)
            else:
                filter_mask = pd.Series(False, index=df.index)

        # Status condition
        if selected_status_filter == "PENDING":
            filter_mask = filter_mask & (df["Audit_Status"].astype(str) == "PENDING")
        elif selected_status_filter == "APPROVED":
            filter_mask = filter_mask & (df["Audit_Status"].astype(str) == "APPROVED")
        elif selected_status_filter == "MANUALLY_CORRECTED":
            filter_mask = filter_mask & (df["Audit_Status"].astype(str) == "MANUALLY_CORRECTED")
        elif selected_status_filter == "Auto-Corrected (Z3 Overrides)":
            filter_mask = filter_mask & (df["Status"].astype(str).str.contains("AUTO_CORRECTED", case=False, na=False))

        valid_indices = df.index[filter_mask].tolist()

        # Subset metrics for the selected folder
        folder_subset = df[df["Subject_Folder"].astype(str) == selected_folder] if selected_folder != "All Folders" and "Subject_Folder" in df.columns else df
        folder_total = len(folder_subset)

        if folder_total > 0:
            f_approved = int((folder_subset["Audit_Status"] == "APPROVED").sum())
            f_corrected = int((folder_subset["Audit_Status"] == "MANUALLY_CORRECTED").sum())
            f_audited = f_approved + f_corrected
            f_pending = folder_total - f_audited
            pct = (f_audited / folder_total) * 100 if folder_total > 0 else 0

            st.markdown(f"**{selected_folder} Progress: {f_audited} / {folder_total} ({pct:.1f}%)**")
            st.progress(f_audited / folder_total if folder_total > 0 else 0)

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.markdown(f"<div class='stat-box'><div class='stat-number' style='color:#15803d;'>{f_approved}</div><div class='stat-label'>Approved</div></div>", unsafe_allow_html=True)
            with col_s2:
                st.markdown(f"<div class='stat-box'><div class='stat-number' style='color:#1d4ed8;'>{f_corrected}</div><div class='stat-label'>Edited</div></div>", unsafe_allow_html=True)
            with col_s3:
                st.markdown(f"<div class='stat-box'><div class='stat-number' style='color:#b45309;'>{f_pending}</div><div class='stat-label'>Pending</div></div>", unsafe_allow_html=True)

        st.markdown("---")

        # ---------------------------------------------------------
        # Jump to Record
        # ---------------------------------------------------------
        if valid_indices:
            def format_record_label(rec_idx):
                row_rec = df.loc[rec_idx]
                b_id = row_rec.get("Booklet_ID", f"Row {rec_idx}")
                usn_val = row_rec.get("USN", "N/A")
                stat_val = row_rec.get("Audit_Status", "PENDING")
                auto_val = " [Z3]" if "AUTO_CORRECTED" in str(row_rec.get("Status", "")) else ""
                pos_in_view = valid_indices.index(rec_idx) + 1
                return f"#{pos_in_view}: {usn_val} ({b_id}) - {stat_val}{auto_val}"

            current_idx = st.session_state.current_index
            if current_idx not in valid_indices:
                current_idx = valid_indices[0]
                st.session_state.current_index = current_idx

            selected_label = st.selectbox(
                f"Go to Record ({len(valid_indices)} in view):",
                options=valid_indices,
                format_func=format_record_label,
                index=valid_indices.index(current_idx) if current_idx in valid_indices else 0,
            )
            if selected_label != st.session_state.current_index:
                st.session_state.current_index = selected_label
                st.rerun()

        st.markdown("---")
        st.markdown("### 💾 **Export & Save**")

        excel_bytes = export_dataframe_to_excel(df, DEFAULT_AUDITED_EXCEL)
        st.success("Auto-saved to `data/output/audited_marksheet.xlsx`")

        st.download_button(
            label="📥 Download Audited Excel",
            data=excel_bytes,
            file_name="audited_marksheet.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        st.markdown("---")
        if st.button("🔄 Reload & Re-Merge Data", use_container_width=True):
            st.session_state.audited_df = load_initial_data()
            st.session_state.current_index = 0
            st.rerun()

    # -------------------------------------------------------------
    # Main Content View
    # -------------------------------------------------------------
    if total_records == 0:
        st.error("No marksheet records found in `data/output/digitized_marksheet.xlsx` or `review1_marksheet.xlsx`.")
        return

    if not valid_indices:
        st.warning(f"No records match the current filter: Folder = **{selected_folder}**, Status = **{selected_status_filter}**.")
        if selected_folder in ["Sub2", "Sub3"]:
            st.info(f"If you have images in `data/{selected_folder}/`, click **'🔄 Process Missing Sub2/Sub3 Images Now'** in the sidebar to extract them.")
        return

    # Ensure current_index is valid
    if st.session_state.current_index not in valid_indices:
        st.session_state.current_index = valid_indices[0]

    idx = st.session_state.current_index
    row = df.loc[idx]

    view_position = valid_indices.index(idx) + 1
    view_total = len(valid_indices)

    subject_folder = str(row.get("Subject_Folder", "raw"))
    booklet_id = str(row.get("Booklet_ID", f"record_{idx+1}"))
    image_file_col = row.get("Image_File", None)
    solver_status = str(row.get("Status", "OPTIMAL"))
    solver_overrides = str(row.get("Solver_Overrides", ""))
    audit_status = str(row.get("Audit_Status", "PENDING")).upper()

    # Resolve Image
    resolved_img_path = resolve_image_file(subject_folder, booklet_id, image_file_col)

    # Header Section
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.markdown(
            f"<div class='main-title'>{selected_folder if selected_folder != 'All Folders' else 'All'} &mdash; Booklet {view_position} of {view_total}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='sub-title'><b>Booklet:</b> <code>{booklet_id}</code> &nbsp;|&nbsp; <b>Folder:</b> <code>{subject_folder}</code> &nbsp;|&nbsp; <b>Global Index:</b> #{idx+1} of {total_records}</div>",
            unsafe_allow_html=True,
        )
    with col_head2:
        if audit_status == "APPROVED":
            st.markdown("<div style='text-align:right;'><span class='badge badge-approved'>✅ APPROVED</span></div>", unsafe_allow_html=True)
        elif audit_status == "MANUALLY_CORRECTED":
            st.markdown("<div style='text-align:right;'><span class='badge badge-corrected'>✏️ EDITED</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align:right;'><span class='badge badge-pending'>⏳ PENDING</span></div>", unsafe_allow_html=True)

    # Highlight Auto-Corrected / Overrides
    is_auto_corrected = "AUTO_CORRECTED" in solver_status or (solver_overrides and solver_overrides != "nan" and len(solver_overrides.strip()) > 0)
    if is_auto_corrected:
        st.markdown(
            f"""
            <div class='alert-card alert-card-warning'>
                <strong>⚠️ Z3 Neuro-Symbolic Auto-Correction Detected:</strong><br/>
                <code>{solver_overrides if (solver_overrides and solver_overrides != 'nan') else 'Satisfiability adjustment triggered for this booklet.'}</code><br/>
                <small style='color:#b45309;'>The mathematical constraint solver adjusted one or more raw OCR readings to satisfy sum equations. Please double-check these marks against the original image.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # Two-Column Workspace: Left = Image, Right = Form
    # -------------------------------------------------------------
    col_img, col_form = st.columns([1, 1], gap="medium")

    # --- LEFT COLUMN: Image Viewer ---
    with col_img:
        st.markdown("#### 📷 Original Marksheet Scan")
        if resolved_img_path and resolved_img_path.exists():
            st.image(str(resolved_img_path), use_container_width=True, caption=f"File: {resolved_img_path.name}")
            with st.expander("🔍 Image Details & File Path"):
                st.code(str(resolved_img_path))
                file_size_kb = resolved_img_path.stat().st_size / 1024
                st.caption(f"File size: {file_size_kb:.1f} KB")
        else:
            st.error("⚠️ Image file could not be located on disk.")
            st.info(
                f"Searched paths for Booklet: `{booklet_id}`:\n"
                f"- `data/{subject_folder}/{booklet_id}.jpg`\n"
                f"- `data/Sub3/{booklet_id}.jpg`\n"
                f"- `data/Sub2/{booklet_id}.jpg`\n"
                f"- `data/Sub1/{booklet_id}.jpg`\n"
                f"- `data/raw/{booklet_id}.jpg`\n"
                f"- `cloud_vault/review1_uploads/{booklet_id}`"
            )

    # --- RIGHT COLUMN: Form & Audit Controls ---
    with col_form:
        st.markdown("#### 📋 Verify Extracted Data")

        # Header student identifiers
        col_f_usn, col_f_code = st.columns(2)
        with col_f_usn:
            curr_usn = str(row.get("USN", "4MT24MC000"))
            new_usn = st.text_input("USN (University Seat Number)", value=curr_usn, key=f"usn_{idx}")
        with col_f_code:
            curr_code = str(row.get("Course_Code", "24MCPC522"))
            new_code = st.text_input("Course Code", value=curr_code, key=f"code_{idx}")

        # Check template type: Theory-23 (24MCPC521 / Sub2) vs Integrated-23 (24MCPC522 / 24MCPC517)
        is_theory_23 = (
            "521" in str(new_code)
            or str(subject_folder).lower() in ["sub2", "theory-23", "theory_23"]
        )

        if is_theory_23:
            # -------------------------------------------------------------
            # Template: Theory-23 (7 Data Rows, No Viva/Lab)
            # -------------------------------------------------------------
            st.markdown(
                "<div style='margin-bottom:8px;'><span class='badge' style='background-color:#ede9fe; color:#6d28d9; border:1px solid #c4b5fd;'>📘 Theory-23 Template (No Viva/Lab)</span></div>",
                unsafe_allow_html=True,
            )

            st.markdown("**1. Continuous Internal Evaluation Tests (Max 30)**")
            val_t1 = int(row.get("Test1_15", 0))
            val_t2 = int(row.get("Test2_15", 0))

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                new_test1 = st.number_input(
                    "Test 1 (0-15)",
                    min_value=0,
                    max_value=max(50, val_t1 + 5),
                    value=val_t1,
                    step=1,
                    key=f"t1_{idx}",
                )
            with col_t2:
                new_test2 = st.number_input(
                    "Test 2 (0-15)",
                    min_value=0,
                    max_value=max(50, val_t2 + 5),
                    value=val_t2,
                    step=1,
                    key=f"t2_{idx}",
                )

            calc_test_total = new_test1 + new_test2
            default_test_total = int(row.get("Test_Total_30", calc_test_total) if "Test_Total_30" in row and not pd.isna(row.get("Test_Total_30")) else calc_test_total)

            col_tt_val, col_tt_stat = st.columns([2, 1])
            with col_tt_val:
                new_test_total = st.number_input(
                    "Test Total (0-30)",
                    min_value=0,
                    max_value=max(50, default_test_total + 5),
                    value=default_test_total,
                    step=1,
                    key=f"tt_{idx}",
                )
            with col_tt_stat:
                if new_test_total == calc_test_total:
                    st.markdown("<div style='margin-top:30px; color:#15803d; font-size:0.85rem;'>✅ T1 + T2 Match</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='margin-top:30px; color:#b45309; font-size:0.85rem;'>⚠️ T1+T2 = {calc_test_total}</div>", unsafe_allow_html=True)

            st.markdown("<hr style='margin: 8px 0;'/>", unsafe_allow_html=True)
            st.markdown("**2. Activity Based Assessment (ABA I & II, Max 20)**")

            # Parse initial ABA components if available
            stored_aba_total = int(row.get("Total_ABA_20", row.get("ABA_10", 0)) if not pd.isna(row.get("Total_ABA_20", row.get("ABA_10", 0))) else 0)
            initial_aba1 = int(row.get("ABA_I_10", min(10, stored_aba_total)) if "ABA_I_10" in row and not pd.isna(row.get("ABA_I_10")) else min(10, stored_aba_total))
            initial_aba2 = int(row.get("ABA_II_10", max(0, stored_aba_total - initial_aba1)) if "ABA_II_10" in row and not pd.isna(row.get("ABA_II_10")) else max(0, stored_aba_total - initial_aba1))

            col_aba1, col_aba2 = st.columns(2)
            with col_aba1:
                new_aba1 = st.number_input(
                    "ABA I (0-10)",
                    min_value=0,
                    max_value=max(50, initial_aba1 + 5),
                    value=initial_aba1,
                    step=1,
                    key=f"aba1_{idx}",
                )
            with col_aba2:
                new_aba2 = st.number_input(
                    "ABA II (0-10)",
                    min_value=0,
                    max_value=max(50, initial_aba2 + 5),
                    value=initial_aba2,
                    step=1,
                    key=f"aba2_{idx}",
                )

            calc_total_aba = new_aba1 + new_aba2
            col_aba_tot, col_aba_stat = st.columns([2, 1])
            with col_aba_tot:
                new_total_aba = st.number_input(
                    "Total ABA (0-20)",
                    min_value=0,
                    max_value=max(50, calc_total_aba + 5),
                    value=calc_total_aba,
                    step=1,
                    key=f"total_aba_{idx}",
                )
            with col_aba_stat:
                if new_total_aba == calc_total_aba:
                    st.markdown("<div style='margin-top:30px; color:#15803d; font-size:0.85rem;'>✅ ABA I+II Match</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='margin-top:30px; color:#b45309; font-size:0.85rem;'>⚠️ Sum = {calc_total_aba}</div>", unsafe_allow_html=True)

            st.markdown("<hr style='margin: 8px 0;'/>", unsafe_allow_html=True)
            st.markdown("**3. Total CIE Marks (Max 50)**")

            calc_cie = new_test_total + new_total_aba
            curr_cie = int(row.get("Total_CIE_50", calc_cie))

            col_cie_tot, col_cie_stat = st.columns([2, 1])
            with col_cie_tot:
                new_cie = st.number_input(
                    "Total CIE Marks (0-50)",
                    min_value=0,
                    max_value=max(60, curr_cie + 5),
                    value=curr_cie,
                    step=1,
                    key=f"cie_{idx}",
                )
            with col_cie_stat:
                if new_cie == calc_cie:
                    st.markdown("<div style='margin-top:30px; color:#15803d; font-size:0.85rem;'>✅ Sum Match (30+20)</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='margin-top:30px; color:#b45309; font-size:0.85rem;'>⚠️ Tests+ABA = {calc_cie}</div>", unsafe_allow_html=True)

            # Theory-23 has 0 practical/viva/lab
            new_theory = new_cie
            new_viva = 0
            new_lab = 0
            new_pract = 0
            new_aba = new_total_aba

            if st.button("🔄 Auto-Calculate Totals from Components", key=f"sync_btn_{idx}"):
                df.at[idx, "Test_Total_30"] = calc_test_total
                df.at[idx, "ABA_I_10"] = new_aba1
                df.at[idx, "ABA_II_10"] = new_aba2
                df.at[idx, "Total_ABA_20"] = calc_total_aba
                df.at[idx, "ABA_10"] = calc_total_aba
                df.at[idx, "Theory_Total_40"] = calc_test_total + calc_total_aba
                df.at[idx, "Viva_5"] = 0
                df.at[idx, "Lab_5"] = 0
                df.at[idx, "Practical_Total_10"] = 0
                df.at[idx, "Total_CIE_50"] = calc_test_total + calc_total_aba
                export_dataframe_to_excel(df, DEFAULT_AUDITED_EXCEL)
                st.rerun()

        else:
            # -------------------------------------------------------------
            # Template: Integrated-23 (8 Data Rows, Viva & Lab Included)
            # -------------------------------------------------------------
            st.markdown(
                "<div style='margin-bottom:8px;'><span class='badge' style='background-color:#e0f2fe; color:#0369a1; border:1px solid #bae6fd;'>📗 Integrated-23 Template (Theory + Practical)</span></div>",
                unsafe_allow_html=True,
            )

            st.markdown("**1. Theory Assessment (Max 40)**")
            val_t1 = int(row.get("Test1_15", 0))
            val_t2 = int(row.get("Test2_15", 0))
            val_aba = int(row.get("ABA_10", 0))

            col_t1, col_t2, col_aba = st.columns(3)
            with col_t1:
                new_test1 = st.number_input(
                    "Test 1 (0-15)",
                    min_value=0,
                    max_value=max(50, val_t1 + 5),
                    value=val_t1,
                    step=1,
                    key=f"t1_{idx}",
                )
            with col_t2:
                new_test2 = st.number_input(
                    "Test 2 (0-15)",
                    min_value=0,
                    max_value=max(50, val_t2 + 5),
                    value=val_t2,
                    step=1,
                    key=f"t2_{idx}",
                )
            with col_aba:
                new_aba = st.number_input(
                    "ABA (0-10)",
                    min_value=0,
                    max_value=max(50, val_aba + 5),
                    value=val_aba,
                    step=1,
                    key=f"aba_{idx}",
                )

            calc_theory = new_test1 + new_test2 + new_aba
            curr_theory = int(row.get("Theory_Total_40", calc_theory))
            
            col_th_tot, col_th_stat = st.columns([2, 1])
            with col_th_tot:
                new_theory = st.number_input(
                    "Theory Total (0-40)",
                    min_value=0,
                    max_value=max(50, curr_theory + 5),
                    value=curr_theory,
                    step=1,
                    key=f"th_{idx}",
                )
            with col_th_stat:
                if new_theory == calc_theory:
                    st.markdown("<div style='margin-top:30px; color:#15803d; font-size:0.85rem;'>✅ Sum Match</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='margin-top:30px; color:#b45309; font-size:0.85rem;'>⚠️ Sum = {calc_theory}</div>", unsafe_allow_html=True)

            st.markdown("<hr style='margin: 8px 0;'/>", unsafe_allow_html=True)
            st.markdown("**2. Practical Assessment (Max 10)**")

            val_viva = int(row.get("Viva_5", 0))
            val_lab = int(row.get("Lab_5", 0))

            col_viva, col_lab = st.columns(2)
            with col_viva:
                new_viva = st.number_input(
                    "Viva (0-5)",
                    min_value=0,
                    max_value=max(50, val_viva + 5),
                    value=val_viva,
                    step=1,
                    key=f"viva_{idx}",
                )
            with col_lab:
                new_lab = st.number_input(
                    "Lab (0-5)",
                    min_value=0,
                    max_value=max(50, val_lab + 5),
                    value=val_lab,
                    step=1,
                    key=f"lab_{idx}",
                )

            calc_pract = new_viva + new_lab
            curr_pract = int(row.get("Practical_Total_10", calc_pract))

            col_pr_tot, col_pr_stat = st.columns([2, 1])
            with col_pr_tot:
                new_pract = st.number_input(
                    "Practical Total (0-10)",
                    min_value=0,
                    max_value=max(50, curr_pract + 5),
                    value=curr_pract,
                    step=1,
                    key=f"pr_{idx}",
                )
            with col_pr_stat:
                if new_pract == calc_pract:
                    st.markdown("<div style='margin-top:30px; color:#15803d; font-size:0.85rem;'>✅ Sum Match</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='margin-top:30px; color:#b45309; font-size:0.85rem;'>⚠️ Sum = {calc_pract}</div>", unsafe_allow_html=True)

            st.markdown("<hr style='margin: 8px 0;'/>", unsafe_allow_html=True)
            st.markdown("**3. Grand Total CIE (Max 50)**")

            calc_cie = new_theory + new_pract
            curr_cie = int(row.get("Total_CIE_50", calc_cie))

            col_cie_tot, col_cie_stat = st.columns([2, 1])
            with col_cie_tot:
                new_cie = st.number_input(
                    "Total CIE Marks (0-50)",
                    min_value=0,
                    max_value=max(60, curr_cie + 5),
                    value=curr_cie,
                    step=1,
                    key=f"cie_{idx}",
                )
            with col_cie_stat:
                if new_cie == calc_cie:
                    st.markdown("<div style='margin-top:30px; color:#15803d; font-size:0.85rem;'>✅ Sum Match</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='margin-top:30px; color:#b45309; font-size:0.85rem;'>⚠️ Sum = {calc_cie}</div>", unsafe_allow_html=True)

            new_test_total = new_test1 + new_test2
            new_aba1 = None
            new_aba2 = None
            new_total_aba = new_aba

            if st.button("🔄 Auto-Calculate Totals from Components", key=f"sync_btn_{idx}"):
                df.at[idx, "Theory_Total_40"] = calc_theory
                df.at[idx, "Practical_Total_10"] = calc_pract
                df.at[idx, "Total_CIE_50"] = calc_theory + calc_pract
                export_dataframe_to_excel(df, DEFAULT_AUDITED_EXCEL)
                st.rerun()

        new_notes = st.text_input(
            "Reviewer Notes (Optional)",
            value=str(row.get("Reviewer_Notes", "") if row.get("Reviewer_Notes") and str(row.get("Reviewer_Notes")) != "nan" else ""),
            key=f"notes_{idx}"
        )

        st.markdown("<hr style='margin: 12px 0;'/>", unsafe_allow_html=True)

        # -------------------------------------------------------------
        # Action Buttons (Scoped strictly to valid_indices)
        # -------------------------------------------------------------
        def advance_to_next_index():
            current_pos = valid_indices.index(idx)
            if current_pos + 1 < len(valid_indices):
                st.session_state.current_index = valid_indices[current_pos + 1]
            elif len(valid_indices) > 0:
                st.session_state.current_index = valid_indices[0]

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            if st.button("✅ Approve (Correct)", type="primary", use_container_width=True, key=f"app_{idx}"):
                df.at[idx, "Audit_Status"] = "APPROVED"
                df.at[idx, "Audited_At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if new_notes:
                    df.at[idx, "Reviewer_Notes"] = new_notes
                export_dataframe_to_excel(df, DEFAULT_AUDITED_EXCEL)
                st.toast(f"Record {view_position} of {view_total} ({new_usn}) Approved!", icon="✅")
                advance_to_next_index()
                st.rerun()

        with btn_col2:
            if st.button("💾 Save Corrections (Wrong)", use_container_width=True, key=f"corr_{idx}"):
                changes = []
                if new_usn != str(row.get("USN", "")):
                    changes.append(f"USN: {row.get('USN')}->{new_usn}")
                if new_code != str(row.get("Course_Code", "")):
                    changes.append(f"Code: {row.get('Course_Code')}->{new_code}")
                if new_test1 != int(row.get("Test1_15", 0)):
                    changes.append(f"T1: {row.get('Test1_15')}->{new_test1}")
                if new_test2 != int(row.get("Test2_15", 0)):
                    changes.append(f"T2: {row.get('Test2_15')}->{new_test2}")
                if new_aba != int(row.get("ABA_10", 0)):
                    changes.append(f"ABA: {row.get('ABA_10')}->{new_aba}")
                if new_theory != int(row.get("Theory_Total_40", 0)):
                    changes.append(f"Theory: {row.get('Theory_Total_40')}->{new_theory}")
                if new_viva != int(row.get("Viva_5", 0)):
                    changes.append(f"Viva: {row.get('Viva_5')}->{new_viva}")
                if new_lab != int(row.get("Lab_5", 0)):
                    changes.append(f"Lab: {row.get('Lab_5')}->{new_lab}")
                if new_pract != int(row.get("Practical_Total_10", 0)):
                    changes.append(f"Pract: {row.get('Practical_Total_10')}->{new_pract}")
                if new_cie != int(row.get("Total_CIE_50", 0)):
                    changes.append(f"CIE: {row.get('Total_CIE_50')}->{new_cie}")

                # Save standard columns
                df.at[idx, "USN"] = new_usn
                df.at[idx, "Course_Code"] = new_code
                df.at[idx, "Test1_15"] = new_test1
                df.at[idx, "Test2_15"] = new_test2
                df.at[idx, "ABA_10"] = new_aba
                df.at[idx, "Theory_Total_40"] = new_theory
                df.at[idx, "Viva_5"] = new_viva
                df.at[idx, "Lab_5"] = new_lab
                df.at[idx, "Practical_Total_10"] = new_pract
                df.at[idx, "Total_CIE_50"] = new_cie

                # Save Theory-23 specific columns if applicable
                if is_theory_23:
                    df.at[idx, "Test_Total_30"] = new_test_total
                    if new_aba1 is not None:
                        df.at[idx, "ABA_I_10"] = new_aba1
                    if new_aba2 is not None:
                        df.at[idx, "ABA_II_10"] = new_aba2
                    df.at[idx, "Total_ABA_20"] = new_total_aba

                df.at[idx, "Audit_Status"] = "MANUALLY_CORRECTED"
                df.at[idx, "Audited_At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                change_summary = "; ".join(changes) if changes else "Values verified and saved"
                existing_overrides = str(row.get("Solver_Overrides", ""))
                df.at[idx, "Solver_Overrides"] = f"{existing_overrides} [Manual: {change_summary}]".strip(" []")
                if new_notes:
                    df.at[idx, "Reviewer_Notes"] = new_notes

                export_dataframe_to_excel(df, DEFAULT_AUDITED_EXCEL)
                st.toast(f"Corrections saved for {new_usn}!", icon="💾")
                advance_to_next_index()
                st.rerun()

        # Free Navigation Row (strictly navigating filtered valid_indices)
        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
        current_pos = valid_indices.index(idx)

        with nav_col1:
            prev_disabled = (current_pos == 0)
            if st.button("⬅️ Previous", disabled=prev_disabled, use_container_width=True, key=f"prev_{idx}"):
                if current_pos > 0:
                    st.session_state.current_index = valid_indices[current_pos - 1]
                    st.rerun()

        with nav_col2:
            st.markdown(
                f"<div style='text-align:center; padding-top:8px; font-size:0.85rem; color:#64748b;'>{view_position} / {view_total} in {selected_folder}</div>",
                unsafe_allow_html=True
            )

        with nav_col3:
            next_disabled = (current_pos >= len(valid_indices) - 1)
            if st.button("Next ➡️", disabled=next_disabled, use_container_width=True, key=f"next_{idx}"):
                if current_pos < len(valid_indices) - 1:
                    st.session_state.current_index = valid_indices[current_pos + 1]
                    st.rerun()

    # -------------------------------------------------------------
    # Bottom Section: Audited Data Ledger (Live Table)
    # -------------------------------------------------------------
    st.markdown("---")
    with st.expander(f"📊 View Audited Data Ledger ({selected_folder})", expanded=False):
        display_cols = [
            "Audit_Status", "Subject_Folder", "Booklet_ID", "USN", "Course_Code",
            "Test1_15", "Test2_15", "ABA_10", "Theory_Total_40",
            "Viva_5", "Lab_5", "Practical_Total_10", "Total_CIE_50",
            "Status", "Solver_Overrides", "Reviewer_Notes"
        ]
        avail_cols = [c for c in display_cols if c in df.columns]
        view_df = df.loc[valid_indices, avail_cols] if valid_indices else df[avail_cols]
        st.dataframe(
            view_df,
            use_container_width=True,
            height=320,
        )


if __name__ == "__main__":
    main()
