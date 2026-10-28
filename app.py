"""
Streamlit Web Application for Faculty Booklet Uploads and Excel Logging.
Integrates OpenCV Table Segmentation, TrOCR Neural Inference, and Z3 MaxSMT Solver.
"""

import os
import io
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.config import OUTPUT_DIR, RAW_DATA_DIR
from src.ocr_engine import ProbabilisticOCR
from src.preprocessor import TableSegmenter
from src.solver import MarksheetSolver
from src.pipeline import BatchProcessor


# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Marksheet Digitizer | Faculty Portal",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-validated {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-corrected {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-review {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Resource Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading TrOCR Neural Inference Engine...")
def get_ocr_engine() -> ProbabilisticOCR:
    return ProbabilisticOCR()


@st.cache_resource(show_spinner="Initializing OpenCV Table Segmenter...")
def get_table_segmenter() -> TableSegmenter:
    return TableSegmenter()


@st.cache_resource(show_spinner="Initializing Z3 Neuro-Symbolic Solver...")
def get_marksheet_solver() -> MarksheetSolver:
    return MarksheetSolver()


@st.cache_resource(show_spinner="Setting up End-to-End Batch Processor...")
def get_pipeline() -> BatchProcessor:
    ocr = get_ocr_engine()
    segmenter = get_table_segmenter()
    solver = get_marksheet_solver()
    return BatchProcessor(ocr=ocr, segmenter=segmenter, solver=solver)


# -----------------------------------------------------------------------------
# Excel Persistence Helper
# -----------------------------------------------------------------------------
EXCEL_PATH = OUTPUT_DIR / "digitized_marksheet.xlsx"


def append_record_to_excel(cie_record: Dict[str, Any], q_record: Dict[str, Any], excel_path: Path = EXCEL_PATH):
    """
    Appends or updates the processed student marksheet in the consolidated Excel spreadsheet.
    """
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    df_cie = pd.DataFrame()
    df_q = pd.DataFrame()

    if excel_path.exists():
        try:
            df_cie = pd.read_excel(excel_path, sheet_name="CIE_Summary")
            df_q = pd.read_excel(excel_path, sheet_name="Question_Level_Marks")
        except Exception:
            df_cie = pd.DataFrame()
            df_q = pd.DataFrame()

    booklet_id = cie_record.get("Booklet_ID", "custom_booklet")
    new_cie_df = pd.DataFrame([cie_record])
    new_q_df = pd.DataFrame([q_record])

    # Reorder standard columns
    cie_columns = [
        "Booklet_ID", "USN", "Course_Code",
        "Test1_Raw", "Test1_Scaled", "Test2_Raw", "Test2_Scaled",
        "ABA", "Theory_Total", "Viva", "Lab",
        "Practical_Total", "Total_CIE_50", "Status", "Solver_Overrides"
    ]
    q_columns = [
        "Booklet_ID", "USN",
        *[f"T1_Q{i}" for i in range(1, 11)], "T1_Raw_Total",
        *[f"T2_Q{i}" for i in range(1, 11)], "T2_Raw_Total"
    ]

    if not df_cie.empty and "Booklet_ID" in df_cie.columns and booklet_id in df_cie["Booklet_ID"].values:
        df_cie = df_cie[df_cie["Booklet_ID"] != booklet_id]
        df_cie = pd.concat([df_cie, new_cie_df], ignore_index=True)
    else:
        df_cie = pd.concat([df_cie, new_cie_df], ignore_index=True)

    if not df_q.empty and "Booklet_ID" in df_q.columns and booklet_id in df_q["Booklet_ID"].values:
        df_q = df_q[df_q["Booklet_ID"] != booklet_id]
        df_q = pd.concat([df_q, new_q_df], ignore_index=True)
    else:
        df_q = pd.concat([df_q, new_q_df], ignore_index=True)

    # Filter available columns
    avail_cie = [c for c in cie_columns if c in df_cie.columns]
    df_cie = df_cie[avail_cie]
    avail_q = [c for c in q_columns if c in df_q.columns]
    df_q = df_q[avail_q]

    with pd.ExcelWriter(str(excel_path), engine="openpyxl") as writer:
        df_cie.to_excel(writer, sheet_name="CIE_Summary", index=False)
        df_q.to_excel(writer, sheet_name="Question_Level_Marks", index=False)


def get_excel_bytes(excel_path: Path = EXCEL_PATH) -> Optional[bytes]:
    if excel_path.exists():
        with open(excel_path, "rb") as f:
            return f.read()
    return None


def read_uploaded_image(file_or_path) -> Optional[np.ndarray]:
    """Reads a Streamlit UploadedFile or file path into a BGR numpy array."""
    if file_or_path is None:
        return None
    if isinstance(file_or_path, (str, Path)):
        return cv2.imread(str(file_or_path))
    try:
        file_bytes = np.frombuffer(file_or_path.read(), np.uint8)
        file_or_path.seek(0)  # Reset buffer pointer
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        st.error(f"Failed to read image: {e}")
        return None


# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
def main():
    st.markdown('<div class="main-title">📝 Marksheet Digitizer & Verification Portal</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Automated 2-Internal MITE Marksheet Ingestion with TrOCR and Z3 Neuro-Symbolic Verification</div>',
        unsafe_allow_html=True,
    )

    pipeline = get_pipeline()

    # -------------------------------------------------------------------------
    # Sidebar: Model status, Dataset stats & Excel Download
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ System Status")
        st.success(f"Device: **{str(pipeline.ocr.device).upper()}**")
        st.info("TrOCR: `microsoft/trocr-small-handwritten`\n\nSolver: `Z3 MaxSMT Presburger`")

        st.markdown("---")
        st.header("📊 Consolidated Records")
        excel_bytes = get_excel_bytes(EXCEL_PATH)
        if excel_bytes and EXCEL_PATH.exists():
            try:
                df_cur = pd.read_excel(EXCEL_PATH, sheet_name="CIE_Summary")
                st.metric("Total Records in Sheet", len(df_cur))
                if "Status" in df_cur.columns:
                    val_c = (df_cur["Status"] == "VALIDATED").sum()
                    cor_c = (df_cur["Status"] == "AUTO_CORRECTED").sum()
                    st.caption(f"✅ Validated: **{val_c}** | 🔧 Auto-Corrected: **{cor_c}**")
            except Exception:
                st.caption("Excel file exists.")

            st.download_button(
                label="📥 Download Updated Excel",
                data=excel_bytes,
                file_name="digitized_marksheet.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.warning("No records logged to Excel yet.")

        st.markdown("---")
        st.markdown(
            "**Rules Enforced by Z3:**\n"
            "- $\sum Q_i (T1) = T1_{raw}$\n"
            "- $\sum Q_i (T2) = T2_{raw}$\n"
            "- $T_{scaled} = \lfloor(3 T_{raw} + 5)/10\\rfloor$\n"
            "- $Theory = T1_s + T2_s + ABA$\n"
            "- $Practical = Viva + Lab$\n"
            "- $CIE = Theory + Practical$"
        )

    # -------------------------------------------------------------------------
    # Input Selection Mode
    # -------------------------------------------------------------------------
    st.subheader("1. Select or Upload Marksheet Booklet")
    input_mode = st.radio(
        "Input Source:",
        ["Upload Custom Images", "Choose from Benchmark Dataset (51 Booklets)"],
        horizontal=True,
    )

    p1_img: Optional[np.ndarray] = None
    p2_img: Optional[np.ndarray] = None
    booklet_identifier = "custom_upload"

    if input_mode == "Upload Custom Images":
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            p1_file = st.file_uploader(
                "Upload Page 1 (Front: USN, Course Code, CIE Marks)",
                type=["jpg", "jpeg", "png"],
                key="uploader_p1",
            )
        with col_up2:
            p2_file = st.file_uploader(
                "Upload Page 2 (Back: Test 1 & Test 2 Question Grids)",
                type=["jpg", "jpeg", "png"],
                key="uploader_p2",
            )

        custom_id = st.text_input("Booklet ID / Identifier (optional):", value="booklet_upload")
        booklet_identifier = custom_id.strip() or "booklet_upload"

        if p1_file:
            p1_img = read_uploaded_image(p1_file)
        if p2_file:
            p2_img = read_uploaded_image(p2_file)

    else:
        sample_booklets = [f"booklet_{i:03d}" for i in range(1, 52)]
        selected_booklet = st.selectbox("Select Booklet Pair:", sample_booklets, index=0)
        booklet_identifier = selected_booklet

        p1_path = RAW_DATA_DIR / f"{selected_booklet}_p1.jpg"
        p2_path = RAW_DATA_DIR / f"{selected_booklet}_p2.jpg"

        if p1_path.exists():
            p1_img = cv2.imread(str(p1_path))
        if p2_path.exists():
            p2_img = cv2.imread(str(p2_path))

    # -------------------------------------------------------------------------
    # Side-by-Side Image Previews
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("2. Booklet Preview")
    prev_col1, prev_col2 = st.columns(2)

    with prev_col1:
        st.markdown("**Page 1: Front Cover (CIE Summary)**")
        if p1_img is not None:
            # Display scaled RGB preview
            p1_rgb = cv2.cvtColor(p1_img, cv2.COLOR_BGR2RGB)
            st.image(p1_rgb, caption=f"Page 1 ({p1_img.shape[1]}x{p1_img.shape[0]} px)", use_container_width=True)
        else:
            st.info("Please upload or select Page 1 image.")

    with prev_col2:
        st.markdown("**Page 2: Back Cover (Question Breakdowns)**")
        if p2_img is not None:
            p2_rgb = cv2.cvtColor(p2_img, cv2.COLOR_BGR2RGB)
            st.image(p2_rgb, caption=f"Page 2 ({p2_img.shape[1]}x{p2_img.shape[0]} px)", use_container_width=True)
        else:
            st.info("Please upload or select Page 2 image.")

    # -------------------------------------------------------------------------
    # Processing Action
    # -------------------------------------------------------------------------
    st.markdown("---")
    ready_to_process = (p1_img is not None) and (p2_img is not None)

    process_btn = st.button(
        "🚀 Process Marksheet",
        type="primary",
        disabled=not ready_to_process,
        use_container_width=True,
    )

    if process_btn and ready_to_process:
        with st.spinner("Extracting cells with OpenCV, performing TrOCR neural inference, and resolving Z3 constraints..."):
            t_start = time.time()
            result = pipeline.process_images(p1_img, p2_img, booklet_id=booklet_identifier)
            t_elapsed = time.time() - t_start

        if not result.get("success", False):
            st.error(f"❌ Processing failed: {result.get('error', 'Unknown Error')}")
            return

        # Store result in session state
        st.session_state["last_result"] = result
        st.session_state["last_elapsed"] = t_elapsed

        # Automatically append/update to Excel
        cie_row = result["cie_summary"]
        q_row = result["question_marks"]
        append_record_to_excel(cie_row, q_row, EXCEL_PATH)
        st.session_state["saved_to_excel"] = True

    # -------------------------------------------------------------------------
    # Results Display
    # -------------------------------------------------------------------------
    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        t_el = st.session_state.get("last_elapsed", 0.0)
        status = res.get("status", "UNKNOWN")
        cie_summary = res.get("cie_summary", {})
        q_marks = res.get("question_marks", {})
        overrides = res.get("overrides", [])

        st.markdown("---")
        st.subheader("3. Digitization & Verification Results")

        # Top summary metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Student USN", cie_summary.get("USN", "N/A"))
        with m_col2:
            st.metric("Course Code", cie_summary.get("Course_Code", "N/A"))
        with m_col3:
            st.metric("Total CIE Mark (out of 50)", cie_summary.get("Total_CIE_50", "N/A"))
        with m_col4:
            if status == "VALIDATED":
                badge_html = '<div class="badge-validated">✅ VALIDATED</div>'
            elif status == "AUTO_CORRECTED":
                badge_html = '<div class="badge-corrected">🔧 AUTO_CORRECTED</div>'
            else:
                badge_html = '<div class="badge-review">⚠️ MANUAL REVIEW</div>'
            st.markdown(f"**Status**<br>{badge_html}", unsafe_allow_html=True)
            st.caption(f"Processed in {t_el:.2f}s")

        if overrides:
            with st.expander(f"🔍 Solver Overrides Audit Log ({len(overrides)} adjustments by Z3)", expanded=True):
                st.info(
                    "The neural OCR candidates were reconciled using Presburger arithmetic constraints:\n\n"
                    + "\n".join([f"- `{ov}`" for ov in overrides])
                )

        # Tabbed detailed views
        tab_cie, tab_q, tab_crops = st.tabs(["📋 CIE Marks Breakdown", "📝 Question-Level Marks", "🔍 Extracted Cell Crops"])

        with tab_cie:
            st.markdown("##### Page 1 CIE Summary Table")
            df_display_cie = pd.DataFrame([cie_summary])
            st.dataframe(df_display_cie, use_container_width=True, hide_index=True)

            c_sub1, c_sub2, c_sub3, c_sub4 = st.columns(4)
            c_sub1.info(f"**Test I:** {cie_summary.get('Test1_Raw')} raw $\\rightarrow$ **{cie_summary.get('Test1_Scaled')}** scaled")
            c_sub2.info(f"**Test II:** {cie_summary.get('Test2_Raw')} raw $\\rightarrow$ **{cie_summary.get('Test2_Scaled')}** scaled")
            c_sub3.info(f"**Theory Total:** {cie_summary.get('Theory_Total')}/40 (with ABA: {cie_summary.get('ABA')})")
            c_sub4.info(f"**Practical:** {cie_summary.get('Practical_Total')}/10 (Viva: {cie_summary.get('Viva')}, Lab: {cie_summary.get('Lab')})")

        with tab_q:
            st.markdown("##### Page 2 Question-Level Marks (Test 1 & Test 2)")
            q_cols_t1 = [f"T1_Q{i}" for i in range(1, 11)] + ["T1_Raw_Total"]
            q_cols_t2 = [f"T2_Q{i}" for i in range(1, 11)] + ["T2_Raw_Total"]

            df_t1 = pd.DataFrame([{col.replace("T1_", ""): q_marks.get(col, 0) for col in q_cols_t1}])
            df_t2 = pd.DataFrame([{col.replace("T2_", ""): q_marks.get(col, 0) for col in q_cols_t2}])

            st.markdown("**Test I Marks (Questions 1 - 10 + Total):**")
            st.dataframe(df_t1, use_container_width=True, hide_index=True)

            st.markdown("**Test II Marks (Questions 1 - 10 + Total):**")
            st.dataframe(df_t2, use_container_width=True, hide_index=True)

        with tab_crops:
            st.markdown("##### Extracted Image Cell Crops")
            p1_data = res.get("p1_data", {})
            p2_data = res.get("p2_data", {})

            col_cr1, col_cr2 = st.columns(2)
            with col_cr1:
                usn_crop = p1_data.get("usn_crop")
                if usn_crop is not None:
                    st.image(cv2.cvtColor(usn_crop, cv2.COLOR_BGR2RGB), caption=f"USN Strip: {cie_summary.get('USN')}")

                cc_crop = p1_data.get("course_code_crop")
                if cc_crop is not None:
                    st.image(cv2.cvtColor(cc_crop, cv2.COLOR_BGR2RGB), caption=f"Course Code: {cie_summary.get('Course_Code')}")

            with col_cr2:
                p1_crops = p1_data.get("p1_marks_crops", {})
                if p1_crops:
                    st.markdown("**Page 1 Marks Cells:**")
                    crops_to_show = ["test1", "test2", "aba", "theory_total", "cie_total"]
                    crop_imgs = [cv2.cvtColor(p1_crops[c], cv2.COLOR_BGR2RGB) for c in crops_to_show if c in p1_crops]
                    crop_caps = [f"{c}: {cie_summary.get(c, '')}" for c in crops_to_show if c in p1_crops]
                    st.image(crop_imgs, caption=crop_caps, width=90)

        # Persistence confirmation banner
        st.success(f"💾 Record successfully logged to `{EXCEL_PATH.resolve()}` across both sheets ('CIE_Summary' and 'Question_Level_Marks')!")


if __name__ == "__main__":
    main()
