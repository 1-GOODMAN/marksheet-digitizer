"""
Single-File FastAPI Application for Project Review 1 Live Presentation.
Mobile-first web interface with camera snap, cloud storage, OpenCV table extraction,
TrOCR probabilistic inference, Z3 MaxSMT constraint solving, and multi-sheet Excel logging.
"""

import os
import sys
import io
import time
import socket
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import openpyxl
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Pipeline imports
from src.config import DATA_DIR, OUTPUT_DIR, FOLDER_COURSE_CODES, VALID_COURSE_CODES, resolve_subject_from_filename
from src.pipeline import BatchProcessor
from src.preprocessor import TableSegmenter
from src.ocr_engine import ProbabilisticOCR
from src.solver import MarksheetSolver

# Directory paths
VAULT_DIR = PROJECT_ROOT / "cloud_vault" / "review1_uploads"
VAULT_DIR.mkdir(parents=True, exist_ok=True)

EXCEL_OUTPUT_PATH = Path(OUTPUT_DIR) / "review1_marksheet.xlsx"
EXCEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="MITE Exam Marksheet Digitizer - Review 1 Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global processor singleton
processor: Optional[BatchProcessor] = None
session_records: List[Dict[str, Any]] = []


def get_processor() -> BatchProcessor:
    global processor
    if processor is None:
        print("[INIT] Initializing TrOCR engine, TableSegmenter, and MarksheetSolver...")
        processor = BatchProcessor(output_dir=OUTPUT_DIR)
        print("[INIT] Processor ready for live inference.")
    return processor


def get_local_ip() -> str:
    """Detect the local Wi-Fi or primary LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def append_to_excel(record: Dict[str, Any], excel_path: Path):
    """
    Appends or updates the record in the multi-sheet Excel file.
    Maintains a master 'CIE_Summary' sheet and subject-wise sheets.
    """
    course_code = record.get("Course_Code", "UNKNOWN")
    new_row = pd.DataFrame([record])

    existing_sheets = {}
    if excel_path.exists():
        try:
            xls = pd.ExcelFile(excel_path)
            for sheet in xls.sheet_names:
                existing_sheets[sheet] = pd.read_excel(xls, sheet_name=sheet)
        except Exception as e:
            print(f"[EXCEL] Error reading existing excel: {e}")

    # Master sheet: CIE_Summary
    master_df = existing_sheets.get("CIE_Summary", pd.DataFrame())
    if not master_df.empty:
        # Update existing USN if present, else append
        if "USN" in master_df.columns and record.get("USN") in master_df["USN"].values:
            master_df = master_df[master_df["USN"] != record.get("USN")]
        master_df = pd.concat([master_df, new_row], ignore_index=True)
    else:
        master_df = new_row
    existing_sheets["CIE_Summary"] = master_df

    # Subject-wise sheet
    if course_code and course_code != "UNKNOWN":
        sub_df = existing_sheets.get(course_code, pd.DataFrame())
        if not sub_df.empty:
            if "USN" in sub_df.columns and record.get("USN") in sub_df["USN"].values:
                sub_df = sub_df[sub_df["USN"] != record.get("USN")]
            sub_df = pd.concat([sub_df, new_row], ignore_index=True)
        else:
            sub_df = new_row
        existing_sheets[course_code] = sub_df

    # Write back all sheets with nice column widths
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet_name, df in existing_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in col)
                col_letter = openpyxl.utils.get_column_letter(col[0].column)
                ws.column_dimensions[col_letter].width = max(max_len + 3, 12)


# ==============================================================================
# HTML5 Mobile-First User Interface
# ==============================================================================
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>MITE Marksheet Digitizer | Review 1</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #121826;
      --card-border: #1e293b;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --success: #10b981;
      --success-glow: rgba(16, 185, 129, 0.25);
      --purple: #8b5cf6;
      --purple-glow: rgba(139, 92, 246, 0.25);
      --warning: #f59e0b;
      --danger: #ef4444;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #06b6d4;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', -apple-system, sans-serif; }
    body { background-color: var(--bg); color: var(--text-main); min-height: 100vh; padding-bottom: 40px; }

    .header {
      background: linear-gradient(180deg, #162032 0%, #0d1322 100%);
      border-bottom: 1px solid var(--card-border);
      padding: 16px 20px;
      position: sticky;
      top: 0;
      z-index: 50;
      backdrop-filter: blur(8px);
    }
    .header-content {
      max-width: 900px;
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .logo-group h1 { font-size: 1.15rem; font-weight: 800; letter-spacing: -0.02em; color: #fff; display: flex; align-items: center; gap: 8px; }
    .logo-group p { font-size: 0.75rem; color: var(--accent); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .badge-review {
      background: rgba(59, 130, 246, 0.15);
      border: 1px solid rgba(59, 130, 246, 0.4);
      color: #93c5fd;
      font-size: 0.72rem;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 9999px;
      white-space: nowrap;
    }

    .container { max-width: 900px; margin: 0 auto; padding: 16px; display: flex; flex-direction: column; gap: 16px; }

    /* Action Card */
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }

    .section-title {
      font-size: 0.9rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 14px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .subject-selector-wrapper { margin-bottom: 16px; }
    .label-text { font-size: 0.82rem; font-weight: 600; color: var(--text-muted); margin-bottom: 6px; display: block; }
    select.custom-select {
      width: 100%;
      background: #0d1322;
      border: 1.5px solid #2d3b55;
      color: #fff;
      padding: 12px 14px;
      border-radius: 10px;
      font-size: 0.95rem;
      font-weight: 600;
      outline: none;
      cursor: pointer;
      appearance: none;
      background-image: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e");
      background-repeat: no-repeat;
      background-position: right 14px center;
      background-size: 16px;
    }
    select.custom-select:focus { border-color: var(--primary); }

    .btn-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }
    @media (min-width: 600px) {
      .btn-grid { grid-template-columns: 1fr 1fr; }
    }

    .btn {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      padding: 14px 18px;
      border-radius: 12px;
      font-size: 0.95rem;
      font-weight: 700;
      cursor: pointer;
      border: none;
      transition: all 0.2s ease;
      text-decoration: none;
      text-align: center;
      user-select: none;
    }
    .btn:active { transform: scale(0.98); }
    .btn-camera {
      background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
      color: #fff;
      box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
    }
    .btn-camera:hover { background: #1d4ed8; }
    .btn-upload {
      background: #1e293b;
      border: 1.5px solid #334155;
      color: #e2e8f0;
    }
    .btn-upload:hover { background: #273549; }
    .btn-download-hero {
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
      color: #fff;
      font-size: 1rem;
      padding: 15px 20px;
      box-shadow: 0 6px 20px rgba(16, 185, 129, 0.35);
      border: 1px solid rgba(255, 255, 255, 0.15);
      width: 100%;
    }
    .btn-download-hero:hover { background: #059669; }

    /* Progress Modal / Container */
    .progress-box {
      display: none;
      background: #0d1322;
      border: 1.5px solid #1e293b;
      border-radius: 14px;
      padding: 18px;
      margin-top: 14px;
    }
    .progress-bar-track {
      height: 8px;
      background: #1e293b;
      border-radius: 4px;
      overflow: hidden;
      margin-bottom: 14px;
    }
    .progress-bar-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #3b82f6, #06b6d4, #10b981);
      transition: width 0.4s ease;
    }
    .step-item {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.83rem;
      color: var(--text-muted);
      margin-bottom: 8px;
    }
    .step-item.active { color: #60a5fa; font-weight: 600; }
    .step-item.done { color: var(--success); }
    .step-icon { width: 18px; text-align: center; }

    /* Result Card */
    .result-card {
      display: none;
      border: 2px solid #2d3b55;
      animation: fadeIn 0.3s ease-out;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

    .result-header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--card-border);
    }
    .student-usn {
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.5rem;
      font-weight: 800;
      color: #fff;
      letter-spacing: -0.01em;
    }
    .course-tag {
      font-family: 'JetBrains Mono', monospace;
      background: #1e293b;
      color: #38bdf8;
      border: 1px solid #334155;
      padding: 4px 10px;
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 700;
      margin-left: 8px;
    }

    .badge-status {
      font-size: 0.8rem;
      font-weight: 800;
      padding: 6px 14px;
      border-radius: 9999px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .badge-VALIDATED {
      background: var(--success-glow);
      color: #34d399;
      border: 1px solid #059669;
    }
    .badge-AUTO_CORRECTED {
      background: var(--purple-glow);
      color: #c084fc;
      border: 1px solid #7c3aed;
    }
    .badge-MANUAL_REVIEW {
      background: rgba(245, 158, 11, 0.2);
      color: #fbbf24;
      border: 1px solid #d97706;
    }

    .overrides-notice {
      margin-top: 12px;
      padding: 10px 14px;
      background: rgba(139, 92, 246, 0.12);
      border-left: 3px solid #8b5cf6;
      border-radius: 6px;
      font-size: 0.82rem;
      color: #ddd6fe;
    }

    /* Marks Table */
    .marks-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
      margin-top: 14px;
    }
    @media (min-width: 600px) {
      .marks-grid { grid-template-columns: repeat(4, 1fr); }
    }
    .mark-tile {
      background: #0d1322;
      border: 1px solid #1e293b;
      border-radius: 10px;
      padding: 12px;
      text-align: center;
    }
    .mark-tile.highlight {
      background: rgba(59, 130, 246, 0.08);
      border-color: rgba(59, 130, 246, 0.4);
    }
    .mark-tile.total-cie {
      background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
      border-color: #10b981;
      grid-column: span 2;
    }
    @media (min-width: 600px) {
      .mark-tile.total-cie { grid-column: span 4; }
    }
    .tile-label { font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: var(--text-muted); }
    .tile-val { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 800; color: #fff; margin-top: 4px; }
    .tile-val.big { font-size: 2.2rem; color: #34d399; }

    /* Live Session Table */
    .table-responsive {
      overflow-x: auto;
      margin-top: 10px;
      border-radius: 10px;
      border: 1px solid var(--card-border);
    }
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; text-align: left; }
    th { background: #162032; color: var(--text-muted); font-weight: 700; padding: 10px 12px; white-space: nowrap; }
    td { padding: 10px 12px; border-top: 1px solid var(--card-border); color: #e2e8f0; white-space: nowrap; font-family: 'JetBrains Mono', monospace; }
    tr:hover { background: #151d2f; }

    .stats-row {
      display: flex;
      gap: 12px;
      margin-bottom: 12px;
    }
    .stat-pill {
      background: #0d1322;
      border: 1px solid #1e293b;
      padding: 8px 14px;
      border-radius: 10px;
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
    }
    .stat-pill strong { color: #fff; }
  </style>
</head>
<body>

  <!-- Sticky Header -->
  <header class="header">
    <div class="header-content">
      <div class="logo-group">
        <h1><span>📘</span> MITE Digitizer</h1>
        <p>Neuro-Symbolic Marksheet Extraction</p>
      </div>
      <div class="badge-review">Review 1 Demo</div>
    </div>
  </header>

  <div class="container">

    <!-- Primary Download Button at Top -->
    <a href="/api/demo/download" class="btn btn-download-hero" id="downloadBtnTop">
      <span>⬇️</span> Download Extracted Excel (.xlsx)
    </a>

    <!-- Capture & Upload Card -->
    <div class="card">
      <div class="section-title"><span>📸</span> Booklet Capture & Ingestion</div>
      
      <div class="subject-selector-wrapper">
        <label class="label-text" for="subjectSelect">Target Subject / Course Code:</label>
        <select id="subjectSelect" class="custom-select">
          <option value="Auto-Detect">🔍 Auto-Detect Course Code</option>
          <option value="24MCPC522">24MCPC522 (Cloud Computing)</option>
          <option value="24MCPC517">24MCPC517 (Machine Learning)</option>
          <option value="24MCPC521">24MCPC521 (Cyber Security)</option>
        </select>
      </div>

      <!-- Native Camera Input and File Input -->
      <input type="file" id="cameraInput" accept="image/*" capture="environment" multiple style="display:none;" onchange="handleFilesSelected(this.files)">
      <input type="file" id="fileInput" accept="image/*" multiple style="display:none;" onchange="handleFilesSelected(this.files)">

      <div class="btn-grid">
        <button class="btn btn-camera" onclick="document.getElementById('cameraInput').click()">
          <span>📷</span> Snap Booklet Photo (Camera)
        </button>
        <button class="btn btn-upload" onclick="document.getElementById('fileInput').click()">
          <span>📁</span> Select Booklet Images
        </button>
      </div>

      <!-- Live Step Progress Tracker -->
      <div class="progress-box" id="progressBox">
        <div class="progress-bar-track">
          <div class="progress-bar-fill" id="progressBarFill"></div>
        </div>
        <div class="step-item" id="step1"><span class="step-icon">○</span> 1. Ingesting to Cloud Vault (cloud_vault/review1_uploads/)</div>
        <div class="step-item" id="step2"><span class="step-icon">○</span> 2. OpenCV Grid Detection & 4-Point Perspective Warp</div>
        <div class="step-item" id="step3"><span class="step-icon">○</span> 3. TrOCR Beam Search Candidate Probability Emission</div>
        <div class="step-item" id="step4"><span class="step-icon">○</span> 4. Z3 MaxSMT Presburger Constraint Verification</div>
      </div>
    </div>

    <!-- Active Extracted Result Card -->
    <div class="card result-card" id="resultCard">
      <div class="result-header">
        <div>
          <span class="student-usn" id="resUsn">4MT24MC058</span>
          <span class="course-tag" id="resCourse">24MCPC522</span>
        </div>
        <div id="resBadge" class="badge-status badge-VALIDATED">VALIDATED</div>
      </div>

      <div class="overrides-notice" id="resOverridesBox" style="display: none;">
        <strong>⚖️ Z3 Auto-Correction Applied:</strong>
        <div id="resOverridesText"></div>
      </div>

      <div class="marks-grid">
        <div class="mark-tile"><div class="tile-label">Test 1 (15)</div><div class="tile-val" id="resT1">-</div></div>
        <div class="mark-tile"><div class="tile-label">Test 2 (15)</div><div class="tile-val" id="resT2">-</div></div>
        <div class="mark-tile"><div class="tile-label">ABA (10)</div><div class="tile-val" id="resAba">-</div></div>
        <div class="mark-tile highlight"><div class="tile-label">Theory Total (40)</div><div class="tile-val" id="resTheory">-</div></div>

        <div class="mark-tile"><div class="tile-label">Viva (5)</div><div class="tile-val" id="resViva">-</div></div>
        <div class="mark-tile"><div class="tile-label">Lab (5)</div><div class="tile-val" id="resLab">-</div></div>
        <div class="mark-tile highlight"><div class="tile-label">Practical Total (10)</div><div class="tile-val" id="resPract">-</div></div>
        <div class="mark-tile total-cie"><div class="tile-label">Final CIE Total (50)</div><div class="tile-val big" id="resCie">-</div></div>
      </div>
    </div>

    <!-- Live Scanned Session Log -->
    <div class="card">
      <div class="section-title"><span>📋</span> Session Live Ledger</div>
      <div class="stats-row">
        <div class="stat-pill">Total Scanned: <strong id="statTotal">0</strong></div>
        <div class="stat-pill">Validated: <strong id="statVal" style="color:#34d399;">0</strong></div>
        <div class="stat-pill">Auto-Corrected: <strong id="statAuto" style="color:#c084fc;">0</strong></div>
      </div>
      <div class="table-responsive">
        <table id="sessionTable">
          <thead>
            <tr>
              <th>USN</th>
              <th>Course</th>
              <th>T1</th>
              <th>T2</th>
              <th>ABA</th>
              <th>Theory</th>
              <th>Viva</th>
              <th>Lab</th>
              <th>Pract</th>
              <th>CIE</th>
              <th>Status</th>
              <th>Overrides</th>
            </tr>
          </thead>
          <tbody id="sessionTableBody">
            <tr><td colspan="12" style="text-align:center; color:#64748b; padding:20px;">No marksheets scanned in this session yet.</td></tr>
          </tbody>
        </table>
      </div>
    </div>

  </div>

  <script>
    let sessionScans = [];

    // Load initial records from backend if any exist
    async function loadInitialRecords() {
      try {
        const res = await fetch('/api/demo/records');
        const data = await res.json();
        if (data.records && data.records.length > 0) {
          sessionScans = data.records;
          renderSessionTable();
          displayRecord(sessionScans[sessionScans.length - 1]);
        }
      } catch (err) {
        console.error("Failed to load initial records:", err);
      }
    }

    function setStep(stepNum, status) {
      const el = document.getElementById(`step${stepNum}`);
      if (!el) return;
      el.className = 'step-item ' + status;
      const icon = el.querySelector('.step-icon');
      if (status === 'active') icon.textContent = '⏳';
      else if (status === 'done') icon.textContent = '✓';
      else icon.textContent = '○';
    }

    async function handleFilesSelected(files) {
      if (!files || files.length === 0) return;

      const subject = document.getElementById('subjectSelect').value;
      const progressBox = document.getElementById('progressBox');
      const progressBar = document.getElementById('progressBarFill');

      progressBox.style.display = 'block';
      progressBar.style.width = '15%';

      setStep(1, 'active');
      setStep(2, '');
      setStep(3, '');
      setStep(4, '');

      const formData = new FormData();
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
      }
      formData.append('subject', subject);

      // Animate steps for presentation visual clarity
      setTimeout(() => { progressBar.style.width = '40%'; setStep(1, 'done'); setStep(2, 'active'); }, 500);
      setTimeout(() => { progressBar.style.width = '70%'; setStep(2, 'done'); setStep(3, 'active'); }, 1200);

      try {
        const res = await fetch('/api/demo/upload', {
          method: 'POST',
          body: formData
        });

        progressBar.style.width = '90%';
        setStep(3, 'done');
        setStep(4, 'active');

        const result = await res.json();

        progressBar.style.width = '100%';
        setStep(4, 'done');

        if (result.success && result.records) {
          for (const rec of result.records) {
            sessionScans.unshift(rec);
            displayRecord(rec);
          }
          renderSessionTable();
        } else {
          alert('Upload completed with warning: ' + (result.error || 'Unknown status'));
        }
      } catch (err) {
        console.error("Upload error:", err);
        alert('Error uploading booklet: ' + err.message);
      } finally {
        setTimeout(() => {
          progressBox.style.display = 'none';
          progressBar.style.width = '0%';
        }, 1800);
      }
    }

    function displayRecord(rec) {
      const card = document.getElementById('resultCard');
      card.style.display = 'block';

      document.getElementById('resUsn').textContent = rec.USN || 'UNKNOWN';
      document.getElementById('resCourse').textContent = rec.Course_Code || 'UNKNOWN';

      const badge = document.getElementById('resBadge');
      const status = rec.Status || 'VALIDATED';
      badge.textContent = status;
      badge.className = 'badge-status badge-' + status;

      const overridesBox = document.getElementById('resOverridesBox');
      const overridesText = document.getElementById('resOverridesText');
      if (rec.Solver_Overrides && rec.Solver_Overrides.trim().length > 0) {
        overridesBox.style.display = 'block';
        overridesText.textContent = rec.Solver_Overrides;
      } else {
        overridesBox.style.display = 'none';
      }

      document.getElementById('resT1').textContent = rec.Test1_15 ?? '-';
      document.getElementById('resT2').textContent = rec.Test2_15 ?? '-';
      document.getElementById('resAba').textContent = rec.ABA_10 ?? '-';
      document.getElementById('resTheory').textContent = rec.Theory_Total_40 ?? '-';
      document.getElementById('resViva').textContent = rec.Viva_5 ?? '-';
      document.getElementById('resLab').textContent = rec.Lab_5 ?? '-';
      document.getElementById('resPract').textContent = rec.Practical_Total_10 ?? '-';
      document.getElementById('resCie').textContent = rec.Total_CIE_50 ?? '-';
    }

    function renderSessionTable() {
      const tbody = document.getElementById('sessionTableBody');
      if (!sessionScans || sessionScans.length === 0) {
        tbody.innerHTML = '<tr><td colspan="12" style="text-align:center; color:#64748b; padding:20px;">No marksheets scanned in this session yet.</td></tr>';
        return;
      }

      let valCount = 0;
      let autoCount = 0;
      let html = '';

      sessionScans.forEach((r) => {
        if (r.Status === 'VALIDATED') valCount++;
        if (r.Status === 'AUTO_CORRECTED') autoCount++;

        const badgeClass = 'badge-' + (r.Status || 'VALIDATED');
        html += `
          <tr>
            <td><strong>${r.USN || '-'}</strong></td>
            <td>${r.Course_Code || '-'}</td>
            <td>${r.Test1_15 ?? '-'}</td>
            <td>${r.Test2_15 ?? '-'}</td>
            <td>${r.ABA_10 ?? '-'}</td>
            <td style="color:#60a5fa; font-weight:700;">${r.Theory_Total_40 ?? '-'}</td>
            <td>${r.Viva_5 ?? '-'}</td>
            <td>${r.Lab_5 ?? '-'}</td>
            <td style="color:#60a5fa; font-weight:700;">${r.Practical_Total_10 ?? '-'}</td>
            <td style="color:#34d399; font-weight:800; font-size:1rem;">${r.Total_CIE_50 ?? '-'}</td>
            <td><span class="badge-status ${badgeClass}" style="font-size:0.7rem; padding:3px 8px;">${r.Status || '-'}</span></td>
            <td style="color:#94a3b8; font-size:0.75rem;">${r.Solver_Overrides || '-'}</td>
          </tr>
        `;
      });

      tbody.innerHTML = html;
      document.getElementById('statTotal').textContent = sessionScans.length;
      document.getElementById('statVal').textContent = valCount;
      document.getElementById('statAuto').textContent = autoCount;
    }

    window.addEventListener('DOMContentLoaded', loadInitialRecords);
  </script>
</body>
</html>
"""


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
@app.head("/")
async def serve_index():
    return HTMLResponse(content=HTML_CONTENT)


@app.get("/api/demo/records")
async def get_records():
    """Return all scans recorded in the current session / Excel ledger."""
    global session_records
    if not session_records and EXCEL_OUTPUT_PATH.exists():
        try:
            df = pd.read_excel(EXCEL_OUTPUT_PATH, sheet_name="CIE_Summary")
            session_records = df.to_dict(orient="records")
        except Exception as e:
            print(f"[RECORDS] Could not read existing excel: {e}")
    return {"records": session_records}


@app.post("/api/demo/upload")
async def upload_demo_booklet(
    files: List[UploadFile] = File(...),
    subject: str = Form("Auto-Detect"),
):
    """
    Ingests booklet images from camera or file picker:
    1. Saves to cloud_vault/review1_uploads/
    2. Runs TableSegmenter + ProbabilisticOCR + MarksheetSolver
    3. Writes/updates data/output/review1_marksheet.xlsx
    4. Returns verified records
    """
    global session_records
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    proc = get_processor()
    extracted_records: List[Dict[str, Any]] = []

    for file in files:
        file_bytes = await file.read()
        if not file_bytes:
            continue

        # Map user selection to folder hint or resolve from uploaded filename
        resolved_code = resolve_subject_from_filename(file.filename)
        folder_hint = "raw"
        if subject == "24MCPC517":
            folder_hint = "Sub1"
        elif subject == "24MCPC521":
            folder_hint = "Sub2"
        elif subject == "24MCPC522":
            folder_hint = "raw"
        elif resolved_code == "24MCPC517":
            folder_hint = "Sub1"
        elif resolved_code == "24MCPC521":
            folder_hint = "Sub2"

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp_str}_{file.filename}"
        saved_file_path = VAULT_DIR / safe_filename

        with open(saved_file_path, "wb") as f_out:
            f_out.write(file_bytes)

        # Run full pipeline on the saved booklet photo
        booklet_id = Path(file.filename).stem
        res = proc.process_page1_image(
            img_input=saved_file_path,
            booklet_id=booklet_id,
            subject_folder=folder_hint,
        )

        cie_row = res.get("cie_summary", {})

        # If user explicitly selected a course code, enforce it; else fallback to resolved code
        if subject != "Auto-Detect" and subject in VALID_COURSE_CODES:
            cie_row["Course_Code"] = subject
        elif resolved_code and (not cie_row.get("Course_Code") or cie_row.get("Course_Code") == "UNKNOWN"):
            cie_row["Course_Code"] = resolved_code

        # Augment with timestamp and local vault file reference
        record = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Booklet_ID": booklet_id,
            "USN": cie_row.get("USN", "UNKNOWN"),
            "Course_Code": cie_row.get("Course_Code", "UNKNOWN"),
            "Test1_15": cie_row.get("Test1_15", 0),
            "Test2_15": cie_row.get("Test2_15", 0),
            "ABA_10": cie_row.get("ABA_10", 0),
            "Theory_Total_40": cie_row.get("Theory_Total_40", 0),
            "Viva_5": cie_row.get("Viva_5", 0),
            "Lab_5": cie_row.get("Lab_5", 0),
            "Practical_Total_10": cie_row.get("Practical_Total_10", 0),
            "Total_CIE_50": cie_row.get("Total_CIE_50", 0),
            "Status": cie_row.get("Status", "VALIDATED"),
            "Solver_Overrides": cie_row.get("Solver_Overrides", ""),
            "Image_File": safe_filename,
        }

        # Save to Master & Subject-wise Excel
        try:
            append_to_excel(record, EXCEL_OUTPUT_PATH)
        except Exception as e:
            print(f"[EXCEL WRITE ERROR] {e}")

        # Update in-memory session records
        session_records = [r for r in session_records if r.get("USN") != record["USN"]]
        session_records.insert(0, record)
        extracted_records.append(record)

    return {
        "success": True,
        "records": extracted_records,
        "count": len(extracted_records),
    }


@app.get("/api/demo/download")
@app.head("/api/demo/download")
async def download_excel():
    """Download the extracted review 1 Excel file."""
    if not EXCEL_OUTPUT_PATH.exists():
        # Create an empty template if none exists
        df = pd.DataFrame(columns=[
            "Timestamp", "Booklet_ID", "USN", "Course_Code",
            "Test1_15", "Test2_15", "ABA_10", "Theory_Total_40",
            "Viva_5", "Lab_5", "Practical_Total_10", "Total_CIE_50",
            "Status", "Solver_Overrides", "Image_File"
        ])
        with pd.ExcelWriter(EXCEL_OUTPUT_PATH, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="CIE_Summary", index=False)

    return FileResponse(
        path=str(EXCEL_OUTPUT_PATH),
        filename="review1_digitized_marksheet.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ==============================================================================
# Entry Point
# ==============================================================================
if __name__ == "__main__":
    import sys
    # Ensure stdout handles utf-8 characters smoothly on Windows terminals
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ip = get_local_ip()
    port = 8080

    print("=" * 70)
    print(" >>> MITE EXAM MARKSHEET DIGITIZER - REVIEW 1 LIVE DEMO <<<")
    print("=" * 70)
    print(f" [LOCAL ACCESS]  http://127.0.0.1:{port}")
    print(f" [MOBILE WI-FI]  http://{ip}:{port}")
    print("=" * 70)
    print(" Open the Mobile Wi-Fi URL directly in Chrome/Safari on your phone.")
    print(" Direct camera snap is fully supported over HTTP via file capture.")
    print("=" * 70)

    # Warm up processor ahead of time for zero-latency presentation
    get_processor()

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
