"""
Standalone verification script for Phase 3 Probabilistic TrOCR Neural Inference Engine.
Loads cropped cells from data/crops/test_run/, predicts candidates, compares with visual ground truth,
and exports full predictions to trocr_raw_predictions.json.
"""

from pathlib import Path
import json
import cv2
import numpy as np

from src.ocr_engine import ProbabilisticOCR


import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    print("=" * 80)
    print("Phase 3: Probabilistic TrOCR Neural Inference Engine Verification")
    print("=" * 80)

    test_run_dir = Path("data/crops/test_run")
    if not test_run_dir.exists():
        raise FileNotFoundError(f"Directory {test_run_dir} not found. Run test_preprocessor.py first.")

    # Initialize OCR engine
    print("\n[1/4] Initializing TrOCR engine...")
    ocr = ProbabilisticOCR()
    print(f"       Hardware Device: {ocr.device}")
    print(f"       Model Checkpoint: {ocr.model_name}")

    predictions = {
        "metadata": {
            "booklet_id": "booklet_001",
            "model_name": ocr.model_name,
            "device": str(ocr.device),
        },
        "usn": "",
        "course_code": "",
        "p1_marks": {},
        "p2_test1": {},
        "p2_test2": {},
    }

    # Visual ground truth labels for booklet_001
    ground_truth = {
        "usn": "4MT24MC058",
        "course_code": "24MCPC522",
        "p1_marks": {
            "test1": 12,
            "test2": 11,
            "aba": 10,
            "theory_total": 33,
            "viva": 5,
            "lab": 3,
            "pract_total": 8,
            "cie_total": 41,
        },
        "p2_test1": {
            "q1": 9,
            "q2": 0,
            "q3": 6,
            "q4": 0,
            "q5": 0,
            "q6": 8,
            "q7": 8,
            "q8": 0,
            "q9": 0,
            "q10": 9,
            "total": 40,
        },
        "p2_test2": {
            "q1": 7,
            "q2": 0,
            "q3": 7,
            "q4": 0,
            "q5": 8,
            "q6": 0,
            "q7": 0,
            "q8": 8,
            "q9": 0,
            "q10": 6,
            "total": 36,
        },
    }

    # 1. Alphanumeric Text Extraction
    print("\n[2/4] Extracting Alphanumeric Fields (USN & Course Code)...")
    usn_path = test_run_dir / "usn_strip.png"
    cc_path = test_run_dir / "course_code.png"

    if usn_path.exists():
        usn_img = cv2.imread(str(usn_path))
        usn_pred = ocr.predict_text(usn_img)
        predictions["usn"] = usn_pred
        print(f"       USN Strip:         Predicted: '{usn_pred}' | Ground Truth: '{ground_truth['usn']}'")
    else:
        print("       Warning: usn_strip.png not found")

    if cc_path.exists():
        cc_img = cv2.imread(str(cc_path))
        cc_pred = ocr.predict_text(cc_img)
        predictions["course_code"] = cc_pred
        print(f"       Course Code Strip: Predicted: '{cc_pred}' | Ground Truth: '{ground_truth['course_code']}'")
    else:
        print("       Warning: course_code.png not found")

    # 2. Mark Distributions Extraction
    print("\n[3/4] Running Probabilistic Digit Inference on Marks Crops...")
    table_rows = []

    sections = [
        ("p1_marks", "Page 1 CIE Marks", [
            "test1", "test2", "aba", "theory_total",
            "viva", "lab", "pract_total", "cie_total"
        ]),
        ("p2_test1", "Page 2 Test 1 (Secured)", [
            "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "total"
        ]),
        ("p2_test2", "Page 2 Test 2 (Secured)", [
            "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "total"
        ]),
    ]

    for sec_key, sec_title, ordered_keys in sections:
        sec_dir = test_run_dir / sec_key
        for cell_key in ordered_keys:
            img_file = sec_dir / f"{cell_key}.png"
            if not img_file.exists():
                print(f"       Warning: {img_file} not found")
                continue

            crop = cv2.imread(str(img_file))
            dist = ocr.predict_digit_candidates(crop, beam_size=4)
            predictions[sec_key][cell_key] = dist

            # Top-1 candidate
            sorted_candidates = sorted(dist.items(), key=lambda x: x[1], reverse=True)
            top1_val, top1_prob = sorted_candidates[0]
            gt_val = ground_truth.get(sec_key, {}).get(cell_key, "N/A")

            # Formatted distribution string
            dist_str = ", ".join(f"{k}:{v:.2f}" for k, v in sorted_candidates)

            is_match = (top1_val == gt_val)
            match_marker = "[OK]" if is_match else "[DIFF]"

            table_rows.append({
                "section": sec_title,
                "cell": cell_key,
                "gt": str(gt_val),
                "top1": f"{top1_val} ({top1_prob:.2f}) {match_marker}",
                "dist": f"{{{dist_str}}}",
            })

    # Print Formatted Breakdown Table
    print("\n" + "=" * 105)
    print(f"{'Section':<24} | {'Cell':<12} | {'Target Visual':<14} | {'Top-1 Candidate':<20} | {'Full Beam Distribution'}")
    print("-" * 105)
    current_sec = ""
    for r in table_rows:
        sec_display = r["section"] if r["section"] != current_sec else ""
        current_sec = r["section"]
        print(f"{sec_display:<24} | {r['cell']:<12} | {r['gt']:<14} | {r['top1']:<20} | {r['dist']}")
    print("=" * 105)

    # 3. Specific Page 1 CIE Marks Verification
    print("\n" + "=" * 80)
    print("Page 1 CIE Marks Verification Check:")
    print("=" * 80)
    p1_expected = {
        "test1": 12,
        "test2": 11,
        "aba": 10,
        "theory_total": 33,
        "cie_total": 41,
    }
    all_p1_passed = True
    for cell, exp_val in p1_expected.items():
        cell_dist = predictions["p1_marks"].get(cell, {})
        contains_val = exp_val in cell_dist
        status = "[PASS]" if contains_val else "[FAIL]"
        all_p1_passed = all_p1_passed and contains_val
        print(f" {status} {cell:<14}: Expected candidate {exp_val} -> Output: {cell_dist}")

    if all_p1_passed:
        print("\n [SUCCESS] All Page 1 CIE marks target candidates verified successfully!")
    else:
        print("\n [WARNING] Some Page 1 target candidates were not found in distributions.")

    # 4. Export predictions to JSON
    out_json_path = test_run_dir / "trocr_raw_predictions.json"
    print(f"\n[4/4] Saving raw predictions to: {out_json_path.resolve()}...")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    print("Verification completed successfully!")


if __name__ == "__main__":
    main()
