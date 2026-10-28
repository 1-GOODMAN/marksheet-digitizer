# Neuro-Symbolic Exam Marksheet Digitizer & Verification System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Z3-Solver](https://img.shields.io/badge/SMT-Z3--Solver-purple.svg)](https://github.com/Z3Prover/z3)
[![TrOCR](https://img.shields.io/badge/Transformer-Microsoft%20TrOCR-orange.svg)](https://huggingface.co/microsoft/trocr-small-handwritten)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, privacy-preserving **Neuro-Symbolic Computer Vision & Formal Reasoning System** designed to automate university booklet digitization, handwritten continuous internal evaluation (CIE) score extraction, and multi-tier institutional arithmetic verification.

The system combines **adaptive morphological table extraction** and **Transformer-based handwritten text recognition (TrOCR)** with a **Z3 Presburger MaxSMT symbolic solver**, guaranteeing $100\%$ arithmetic consistency across student evaluation sheets without manual spreadsheet reconciliation.

---

## Architecture Overview

```
                                      +---------------------------------------------+
                                      |          Mobile / Scanned Booklet           |
                                      +---------------------------------------------+
                                                             |
                                                             v
+-----------------------------------------------------------------------------------------------------------------------+
|  Phase 1: Computer Vision & Perspective Normalization (OpenCV)                                                        |
|  - PII Masking: Automated blanking of student confidential header sections                                           |
|  - Dynamic Morphological Grid Filtering: Resolves scale invariance across raw scans & mobile photography              |
|  - 4-Point Perspective Warp: Rectifies distorted captures onto a canonical 1000x600 coordinate grid                  |
|  - Optical Cell Extraction: 10-char USN grid (with 18% border shave) & 8-row Column 5 ('Marks Obtained') crops        |
+-----------------------------------------------------------------------------------------------------------------------+
                                                             |
                                                             v
+-----------------------------------------------------------------------------------------------------------------------+
|  Phase 2: Neural Probabilistic Inference Engine (Microsoft TrOCR)                                                     |
|  - Red-Ink Isolation Filter: Inverts green channel + saturation masking (S > 35) to strip printed black table lines   |
|  - Cell Quality & Blank Classifier: Fast heuristic short-circuiting on empty paper texture (std < 4.5, mean > 245)     |
|  - Beam-Search Posterior Distribution: Extracts candidate likelihoods P(val | cell) with eff_beam = 5                |
|  - Institutional Domain Priors: Hard-locked USN prefix (4MT24MCxxx) & folder-aware Course Code snapping               |
+-----------------------------------------------------------------------------------------------------------------------+
                                                             |
                                                             v
+-----------------------------------------------------------------------------------------------------------------------+
|  Phase 3: Formal Symbolic Arithmetic Resolution (Z3 MaxSMT / Presburger Arithmetic)                                  |
|  - Hard Invariant Enforcement:                                                                                        |
|      (C1) Test1_15 + Test2_15 + ABA_10 == Theory_Total_40                                                            |
|      (C2) Viva_5 + Lab_5 == Practical_Total_10                                                                        |
|      (C3) Theory_Total_40 + Practical_Total_10 == Total_CIE_50                                                        |
|  - Total-Anchored Confidence Multipliers: 3.5x penalty preservation on verified summary totals (25..50)              |
|  - Algebraic Missing-Cell Inference: Back-solves ambiguous single cells from reliable totals                         |
+-----------------------------------------------------------------------------------------------------------------------+
                                                             |
                                                             v
+-----------------------------------------------------------------------------------------------------------------------+
|  Phase 4: Persistence, Cloud Synchronization & Multi-Sheet Excel Ledger                                                |
|  - Automated In-Memory Deduplication: Ensures unique file paths across multi-subject folders (raw, Sub1, Sub2)        |
|  - Multi-Tab Excel Export: Generates Master 'CIE_Summary', per-subject tabs, and per-course-code audit records        |
|  - Cloud Storage Vault: Auto-archives processed booklet imagery and synced ledgers via S3/Local abstraction           |
+-----------------------------------------------------------------------------------------------------------------------+
```

---

## Key Features

1. **Grid-Anchored Table Normalization**
   - Automatically identifies the CIE marks table contour in the lower page quadrant regardless of framing, zoom, or camera distance.
   - Warps the bounding polygon to an exact `1000 x 600` canonical coordinate space before extracting individual cell bounding boxes.

2. **Red-Ink Stroke Isolation & Edge Cleanup**
   - Faculty marks written in red or blue ink are isolated via HSV chroma filtering (`S > 35`) and green-channel inversion (`255 - G_channel`).
   - Connected-component edge stripping removes overlapping black grid borders and printed text columns (`Maximum Marks`), eliminating digit misrecognition.

3. **USN 10-Box Character Grid Parser**
   - Crops individual character slots and shaves $18\%$ off horizontal margins and $15\%$ off vertical margins to erase outer box borders.
   - Completely resolves the common `4MT24MC111` border hallucination bug where vertical box lines were misread as `1`.
   - Strictly validates roll numbers within institutional boundaries (`001` - `199`).

4. **Total-Anchored Neuro-Symbolic Reconciliation**
   - Neural OCR provides posterior probability distributions over possible digit values.
   - When low-contrast cells are ambiguous, Z3 MaxSMT guarantees that valid, high-confidence grand totals are never degraded. The solver algebraically computes the missing addend:
     $$\text{Test2} = \text{Theory Total} - \text{Test1} - \text{ABA}$$
     $$\text{Viva} = \text{Practical Total} - \text{Lab}$$

5. **Cross-Platform Deduplicated Multi-Subject Ingestion**
   - Ingests multiple subject folders (`raw/`, `Sub1/`, `Sub2/`).
   - Automatically maps folder domain priors (`24MCPC522`, `24MCPC517`, `24MCPC521`).
   - Employs case-insensitive path resolution to guarantee $0$ duplicate rows across Windows and UNIX environments.

6. **Full-Stack Deployment & Audit UI**
   - **FastAPI Backend (`server.py`)**: Asynchronous REST endpoints for mobile camera uploads, live OCR execution, cloud sync, and ledger queries.
   - **Faculty Web Portal (`templates/index.html` + `static/app.js`)**: Responsive single-page interface with live viewfinder, real-time stage progress, and searchable ledger gallery.
   - **Streamlit Interactive Audit Dashboard (`app.py`)**: Rapid prototyping app with image inspection and editable score reviews.

---

## Repository Structure

```
marksheet_digitizer/
│
├── src/                               # Core Neuro-Symbolic Library
│   ├── __init__.py
│   ├── config.py                      # Institutional priors, prefixes, model settings
│   ├── preprocessor.py                # OpenCV table segmentation, perspective warp, USN slicing
│   ├── ocr_engine.py                  # Probabilistic TrOCR inference, red-ink isolation, priors
│   ├── solver.py                      # Z3 Presburger MaxSMT arithmetic reconciliation solver
│   ├── pipeline.py                    # End-to-end single-page ingestion orchestrator
│   ├── cloud_storage.py               # Cloud storage abstraction (S3 & Local Bucket)
│   └── stitcher.py                    # Multi-page compositing & document stitching utility
│
├── static/                            # Web Portal Assets
│   ├── app.js                         # Camera controls, async fetch, ledger renderer
│   └── styles.css                     # Dark-mode dashboard styling
│
├── templates/
│   └── index.html                     # Faculty upload & live digitization portal
│
├── data/                              # Data Directories (Git-tracked structure only)
│   ├── raw/                           # Raw benchmark booklets (.gitkeep)
│   ├── Sub1/                          # Subject 1 test images (.gitkeep)
│   ├── Sub2/                          # Subject 2 test images (.gitkeep)
│   ├── crops/                         # Intermediate visual inspection crops (.gitkeep)
│   └── output/                        # Exported Excel ledgers (.gitkeep)
│
├── app.py                             # Streamlit inspection & evaluation dashboard
├── server.py                          # FastAPI production backend server
├── main.py                            # CLI multi-subject batch processing pipeline
├── requirements.txt                   # Production dependencies
├── .gitignore                         # Strict exclusion of PII images, venv, and caches
└── README.md                          # System documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- NVIDIA CUDA-capable GPU (recommended for acceleration, though CPU inference is fully supported)

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_ORGANIZATION_OR_USERNAME>/marksheet-digitizer.git
cd marksheet-digitizer
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### 1. Running the Batch Digitization Pipeline (CLI)
Process all marksheets across `data/raw/`, `data/Sub1/`, and `data/Sub2/`:
```bash
python main.py
```

Optional arguments:
```bash
# Process specific subject folders
python main.py --subjects raw Sub1

# Limit number of images per folder (e.g., testing first 5 images)
python main.py --limit 5

# Custom output destination
python main.py --output data/output/custom_audit_report.xlsx
```

### 2. Launching the Web Portal (FastAPI)
Start the mobile-ready faculty web server:
```bash
python server.py
```
Open your browser and navigate to:
```
http://localhost:8000
```
- Capture Page 1 via web camera or upload an image file.
- View real-time neuro-symbolic execution progress.
- Search previously processed student booklets and download the latest multi-sheet Excel ledger with one click.

### 3. Launching the Streamlit Inspection App
```bash
streamlit run app.py
```

---

## Data Schema & Excel Export

The pipeline outputs an auditable, multi-tab Microsoft Excel workbook (`digitized_marksheet.xlsx`):

| Sheet Name | Description |
|:---|:---|
| **`CIE_Summary`** | Consolidated master ledger of all digitized student records. |
| **`raw`** | Subset of records processed from `data/raw/`. |
| **`Sub1`** | Subset of records processed from `data/Sub1/`. |
| **`Sub2`** | Subset of records processed from `data/Sub2/`. |
| **`24MCPC522`** | Records filtered by Course Code `24MCPC522`. |
| **`24MCPC517`** | Records filtered by Course Code `24MCPC517`. |
| **`24MCPC521`** | Records filtered by Course Code `24MCPC521`. |

### Column Layout
1. `Subject_Folder` — Origin directory (`raw`, `Sub1`, `Sub2`).
2. `Booklet_ID` — Identifier of the marksheet image.
3. `USN` — Formatted University Student Number (e.g., `4MT24MC058`).
4. `Course_Code` — Verified course identifier (`24MCPC522`, `24MCPC517`, or `24MCPC521`).
5. `Test1_15` — Scaled Test 1 mark (range: $0 - 15$).
6. `Test2_15` — Scaled Test 2 mark (range: $0 - 15$).
7. `ABA_10` — Activity-Based Assessment mark (range: $0 - 10$).
8. `Theory_Total_40` — Total theory score (range: $0 - 40$).
9. `Viva_5` — Viva-voce mark (range: $0 - 5$).
10. `Lab_5` — Laboratory continuous assessment mark (range: $0 - 5$).
11. `Practical_Total_10` — Total practical score (range: $0 - 10$).
12. `Total_CIE_50` — Grand Total CIE mark (range: $0 - 50$).
13. `Status` — Validation flag (`VALIDATED`, `AUTO_CORRECTED`, or `MANUAL_REVIEW_REQUIRED`).
14. `Solver_Overrides` — Traceable audit log of any values adjusted by the symbolic solver.

---

## Formal Verification Guarantees

Every processed record satisfies Presburger arithmetic constraints verified through the Z3 SMT Theorem Prover:

$$\forall \text{ record } r:$$
$$\begin{cases}
r.\text{Test1} + r.\text{Test2} + r.\text{ABA} = r.\text{Theory Total} \\
r.\text{Viva} + r.\text{Lab} = r.\text{Practical Total} \\
r.\text{Theory Total} + r.\text{Practical Total} = r.\text{Total CIE}
\end{cases}$$

If raw visual OCR predictions violate any constraint, the SMT solver identifies the optimal configuration that maximizes visual log-likelihood while maintaining strict invariance satisfaction.

---

## Security & Privacy (PII Protection)

- **Automated PII Obfuscation**: The top header region containing confidential student identification is automatically masked prior to table processing.
- **Strict Data Exclusion**: All raw student marksheet scans, intermediate crop binaries, and generated spreadsheets are strictly excluded from version control via `.gitignore`.
- **Stateless Cloud Integration**: Image artifacts can be archived to private cloud storage (AWS S3) with pre-signed ephemeral URLs.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
