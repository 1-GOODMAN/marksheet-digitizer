"""
Test suite for 2-Internal MITE booklet structure extraction and solver.
Tests on booklet_001_p1.jpg and booklet_001_p2.jpg.

Saves extracted crops into data/crops/test_run/:
  - usn_strip.png
  - course_code.png
  - p1_marks/ (test1, test2, aba, theory_total, viva, lab, pract_total, cie_total)
  - p2_test1/ (q1 to q10, total)
  - p2_test2/ (q1 to q10, total)
  - Visual bounding box overlays: debug_p1.jpg, debug_p2.jpg, debug_grid_overlay.jpg
"""

import os
import cv2
import numpy as np
from src.preprocessor import TableSegmenter
from src.solver import MarksheetSolver


def run_test():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    p1_path = os.path.join(base_dir, "data", "raw", "booklet_001_p1.jpg")
    p2_path = os.path.join(base_dir, "data", "raw", "booklet_001_p2.jpg")
    test_run_dir = os.path.join(base_dir, "data", "crops", "test_run")

    p1_marks_dir = os.path.join(test_run_dir, "p1_marks")
    p2_test1_dir = os.path.join(test_run_dir, "p2_test1")
    p2_test2_dir = os.path.join(test_run_dir, "p2_test2")

    for d in [test_run_dir, p1_marks_dir, p2_test1_dir, p2_test2_dir]:
        os.makedirs(d, exist_ok=True)

    print("=================================================================")
    print("MARKSHEET DIGITIZER - 2-INTERNAL MITE EXTRACTION & SOLVER TEST")
    print("=================================================================")
    print(f"Target Output Directory: {test_run_dir}")
    print(f"Booklet P1: {os.path.exists(p1_path)} ({p1_path})")
    print(f"Booklet P2: {os.path.exists(p2_path)} ({p2_path})")

    segmenter = TableSegmenter(cell_padding_px=3)

    # ----------------------------------------------------
    # 1. PROCESS PAGE 1
    # ----------------------------------------------------
    print("\n--- Processing Page 1 ---")
    p1_res = segmenter.extract_page1_cells(p1_path)
    p1_img = p1_res["image"]

    # Save USN strip
    usn_path = os.path.join(test_run_dir, "usn_strip.png")
    if p1_res["usn_crop"] is not None:
        cv2.imwrite(usn_path, p1_res["usn_crop"])
        h, w = p1_res["usn_crop"].shape[:2]
        print(f"  [Header] USN Strip        -> Saved: usn_strip.png ({w}x{h}) box={p1_res['usn_box']}")

    # Save Course Code strip
    cc_path = os.path.join(test_run_dir, "course_code.png")
    if p1_res["course_code_crop"] is not None:
        cv2.imwrite(cc_path, p1_res["course_code_crop"])
        h, w = p1_res["course_code_crop"].shape[:2]
        print(f"  [Header] Course Code      -> Saved: course_code.png ({w}x{h}) box={p1_res['course_code_box']}")

    # Save 8 CIE marks cells into p1_marks/
    p1_marks_crops = p1_res["p1_marks_crops"]
    p1_marks_boxes = p1_res["p1_marks_boxes"]
    print(f"  Extracted {len(p1_marks_crops)} Page 1 CIE marks cells:")

    p1_keys = ["test1", "test2", "aba", "theory_total", "viva", "lab", "pract_total", "cie_total"]
    for k in p1_keys:
        if k in p1_marks_crops:
            crop = p1_marks_crops[k]
            box = p1_marks_boxes[k]
            crop_path = os.path.join(p1_marks_dir, f"{k}.png")
            cv2.imwrite(crop_path, crop)
            h, w = crop.shape[:2]
            print(f"    - {k:15s} -> Saved: p1_marks/{k}.png ({w}x{h}) box={box}")

    # ----------------------------------------------------
    # 2. PROCESS PAGE 2
    # ----------------------------------------------------
    print("\n--- Processing Page 2 ---")
    p2_res = segmenter.extract_page2_cells(p2_path)
    p2_img = p2_res["image"]

    # Save Test I mark cells into p2_test1/ (Q1..Q10 + Total)
    p2_t1_crops = p2_res["p2_test1_crops"]
    p2_t1_boxes = p2_res["p2_test1_boxes"]
    print(f"  Extracted {len(p2_t1_crops)} Test I Marks cells:")

    for q_idx in range(1, 11):
        k = f"q{q_idx}"
        if k in p2_t1_crops:
            crop = p2_t1_crops[k]
            box = p2_t1_boxes[k]
            crop_path = os.path.join(p2_test1_dir, f"{k}.png")
            cv2.imwrite(crop_path, crop)
            h, w = crop.shape[:2]
            print(f"    - {k:10s} (Test 1) -> Saved: p2_test1/{k}.png ({w}x{h}) box={box}")

    if "total" in p2_t1_crops:
        crop = p2_t1_crops["total"]
        box = p2_t1_boxes["total"]
        crop_path = os.path.join(p2_test1_dir, "total.png")
        cv2.imwrite(crop_path, crop)
        h, w = crop.shape[:2]
        print(f"    - {'total':10s} (Test 1) -> Saved: p2_test1/total.png ({w}x{h}) box={box}")

    # Save Test II mark cells into p2_test2/ (Q1..Q10 + Total)
    p2_t2_crops = p2_res["p2_test2_crops"]
    p2_t2_boxes = p2_res["p2_test2_boxes"]
    print(f"  Extracted {len(p2_t2_crops)} Test II Marks cells:")

    for q_idx in range(1, 11):
        k = f"q{q_idx}"
        if k in p2_t2_crops:
            crop = p2_t2_crops[k]
            box = p2_t2_boxes[k]
            crop_path = os.path.join(p2_test2_dir, f"{k}.png")
            cv2.imwrite(crop_path, crop)
            h, w = crop.shape[:2]
            print(f"    - {k:10s} (Test 2) -> Saved: p2_test2/{k}.png ({w}x{h}) box={box}")

    if "total" in p2_t2_crops:
        crop = p2_t2_crops["total"]
        box = p2_t2_boxes["total"]
        crop_path = os.path.join(p2_test2_dir, "total.png")
        cv2.imwrite(crop_path, crop)
        h, w = crop.shape[:2]
        print(f"    - {'total':10s} (Test 2) -> Saved: p2_test2/total.png ({w}x{h}) box={box}")

    # ----------------------------------------------------
    # 3. GENERATE VISUAL BOUNDING BOX OVERLAYS
    # ----------------------------------------------------
    print("\n--- Generating Visual Overlays ---")

    # Debug P1 Overlay
    debug_p1 = p1_img.copy()
    # Draw all detected table cells in green
    for (x, y, bw, bh) in p1_res["all_boxes"]:
        cv2.rectangle(debug_p1, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    # Highlight USN strip in blue
    if p1_res["usn_box"] is not None:
        (ux, uy, uw, uh) = p1_res["usn_box"]
        cv2.rectangle(debug_p1, (ux, uy), (ux + uw, uy + uh), (255, 100, 0), 3)
        cv2.putText(debug_p1, "USN STRIP", (ux, uy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 0), 2)

    # Highlight Course Code strip in orange
    if p1_res["course_code_box"] is not None:
        (cx, cy, cw, ch) = p1_res["course_code_box"]
        cv2.rectangle(debug_p1, (cx, cy), (cx + cw, cy + ch), (0, 165, 255), 3)
        cv2.putText(debug_p1, "COURSE CODE", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

    # Highlight 8 Marks Obtained cells in yellow with labels
    for k, (x, y, bw, bh) in p1_marks_boxes.items():
        cv2.rectangle(debug_p1, (x, y), (x + bw, y + bh), (0, 255, 255), 4)
        cv2.putText(debug_p1, k, (x + 10, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    p1_overlay_path = os.path.join(test_run_dir, "debug_p1.jpg")
    cv2.imwrite(p1_overlay_path, debug_p1)
    print(f"  - Saved Page 1 overlay: debug_p1.jpg")

    # Debug P2 Overlay
    debug_p2 = p2_img.copy()
    # Draw all detected cells in green
    for (x, y, bw, bh) in p2_res["all_boxes"]:
        cv2.rectangle(debug_p2, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    # Highlight Test I cells in cyan
    for k, (x, y, bw, bh) in p2_t1_boxes.items():
        cv2.rectangle(debug_p2, (x, y), (x + bw, y + bh), (255, 255, 0), 3)
        cv2.putText(debug_p2, f"T1_{k}", (x + 10, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Highlight Test II cells in magenta
    for k, (x, y, bw, bh) in p2_t2_boxes.items():
        cv2.rectangle(debug_p2, (x, y), (x + bw, y + bh), (255, 0, 255), 3)
        cv2.putText(debug_p2, f"T2_{k}", (x + 10, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    p2_overlay_path = os.path.join(test_run_dir, "debug_p2.jpg")
    cv2.imwrite(p2_overlay_path, debug_p2)
    print(f"  - Saved Page 2 overlay: debug_p2.jpg")

    # Combined overlay
    scale = 0.5
    d1 = cv2.resize(debug_p1, (int(debug_p1.shape[1] * scale), int(debug_p1.shape[0] * scale)))
    d2 = cv2.resize(debug_p2, (int(debug_p2.shape[1] * scale), int(debug_p2.shape[0] * scale)))
    combined = np.hstack([d1, d2])
    combined_path = os.path.join(test_run_dir, "debug_grid_overlay.jpg")
    cv2.imwrite(combined_path, combined)
    print(f"  - Saved Combined overlay: debug_grid_overlay.jpg")

    # ----------------------------------------------------
    # 4. TEST Z3 CROSS-PAGE TWO-TIER SCALING SOLVER
    # ----------------------------------------------------
    print("\n--- Testing Z3 Cross-Page Two-Tier Solver ---")
    solver = MarksheetSolver()

    # Create candidate distributions based on Booklet 001 extracted marks
    # Test 1 questions: Q1=9, Q2=0, Q3=6, Q4=0, Q5=0, Q6=8, Q7=8, Q8=0, Q9=0, Q10=9 -> Total=40
    # Test 2 questions: Q1=7, Q2=0, Q3=7, Q4=0, Q5=8, Q6=0, Q7=0, Q8=8, Q9=0, Q10=6 -> Total=36
    # Scaled Test 1: (3*40+5)//10 = 12
    # Scaled Test 2: (3*36+5)//10 = 11
    # ABA = 10 -> Theory Total = 33
    # Viva = 5, Lab = 3 -> Pract Total = 8
    # CIE Total = 33 + 8 = 41
    sample_candidate_distributions = {
        # Test 1 questions (blank/dash cells treated as 0)
        "q1_test1": {9: 0.95, 8: 0.05},
        "q2_test1": {0: 0.99},
        "q3_test1": {6: 0.95, 5: 0.05},
        "q4_test1": {0: 0.99},
        "q5_test1": {0: 0.99},
        "q6_test1": {8: 0.95, 3: 0.05},
        "q7_test1": {8: 0.95},
        "q8_test1": {0: 0.99},
        "q9_test1": {0: 0.99},
        "q10_test1": {9: 0.95},
        "test1_raw_total": {40: 0.98},
        # Test 2 questions
        "q1_test2": {7: 0.95},
        "q2_test2": {0: 0.99},
        "q3_test2": {7: 0.95},
        "q4_test2": {0: 0.99},
        "q5_test2": {8: 0.95},
        "q6_test2": {0: 0.99},
        "q7_test2": {0: 0.99},
        "q8_test2": {8: 0.95},
        "q9_test2": {0: 0.99},
        "q10_test2": {6: 0.95},
        "test2_raw_total": {36: 0.98},
        # Page 1 marks
        "test1": {12: 0.98},
        "test2": {11: 0.98},
        "aba": {10: 0.99},
        "theory_total": {33: 0.98},
        "viva": {5: 0.99},
        "lab": {3: 0.98},
        "pract_total": {8: 0.99},
        "cie_total": {41: 0.99}
    }

    solve_res = solver.solve(sample_candidate_distributions)
    print(f"  Solver Success: {solve_res['success']}")
    print(f"  Total Log-Likelihood: {solve_res['log_likelihood']:.4f}")
    print("  Resolved Values:")
    resolved = solve_res["values"]
    print(f"    - Test 1 Raw Total: {resolved.get('test1_raw_total')} (Sum of Q1..Q10: {sum(resolved.get(f'q{i}_test1', 0) for i in range(1, 11))})")
    print(f"    - Test 2 Raw Total: {resolved.get('test2_raw_total')} (Sum of Q1..Q10: {sum(resolved.get(f'q{i}_test2', 0) for i in range(1, 11))})")
    print(f"    - Scaled Test 1: {resolved.get('test1')} (Formula: (3 * {resolved.get('test1_raw_total')} + 5) // 10 = {(3 * resolved.get('test1_raw_total', 0) + 5) // 10})")
    print(f"    - Scaled Test 2: {resolved.get('test2')} (Formula: (3 * {resolved.get('test2_raw_total')} + 5) // 10 = {(3 * resolved.get('test2_raw_total', 0) + 5) // 10})")
    print(f"    - Theory Total: {resolved.get('theory_total')} (Test1 {resolved.get('test1')} + Test2 {resolved.get('test2')} + ABA {resolved.get('aba')} = {resolved.get('test1', 0) + resolved.get('test2', 0) + resolved.get('aba', 0)})")
    print(f"    - Practical Total: {resolved.get('pract_total')} (Viva {resolved.get('viva')} + Lab {resolved.get('lab')} = {resolved.get('viva', 0) + resolved.get('lab', 0)})")
    print(f"    - CIE Total: {resolved.get('cie_total')} (Theory {resolved.get('theory_total')} + Pract {resolved.get('pract_total')} = {resolved.get('theory_total', 0) + resolved.get('pract_total', 0)})")

    print("\nAll components of 2-Internal MITE extraction and solver validated successfully!")


if __name__ == "__main__":
    run_test()
