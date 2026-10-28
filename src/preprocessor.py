"""
Table, Header, and Cell Extraction Module (Phase 2)
Extracts tabular marksheet cells, USN grid, course code, and test totals
from 2-Internal MITE booklet images using OpenCV.
"""

import os
from typing import Dict, List, Optional, Tuple, Any, Union
import cv2
import numpy as np


class TableSegmenter:
    """
    Detects table grids, USN character boxes, course code strips, and individual marks cells
    from marksheet booklet images. Applies resolution-scaled dynamic morphological filters,
    grid line reconstruction, contour hierarchy analysis, and inner margin padding.

    Leading-zero fix: all marks-column crops use MARKS_COL_RIGHT_EXTRA_PX=8 additional
    pixels on the right boundary so the second digit of values like 05, 08, 03 is never
    clipped by the crop window.
    """

    CELL_PADDING_PX: int = 3
    MARKS_COL_RIGHT_EXTRA_PX: int = 8

    def __init__(self, cell_padding_px: int = 3):
        self.cell_padding_px = cell_padding_px

    def mask_pii(self, img: np.ndarray, mask_ratio: float = 0.20) -> np.ndarray:
        masked = img.copy()
        h, _ = masked.shape[:2]
        cut_h = int(h * mask_ratio)
        masked[:cut_h, :] = 0
        return masked

    def detect_grid(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        thresh = cv2.adaptiveThreshold(
            ~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
        )

        h_kernel_len = max(10, int(w * 0.025))
        v_kernel_len = max(10, int(h * 0.025))

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))

        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

        h_connected = cv2.dilate(h_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
        v_connected = cv2.dilate(v_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15)))

        table_grid = cv2.add(h_connected, v_connected)
        return table_grid, h_lines, v_lines

    def extract_bounding_boxes(
        self,
        table_grid: np.ndarray,
        min_width: int = 80,
        min_height: int = 35,
        max_width_ratio: float = 0.85,
        max_height_ratio: float = 0.45,
    ) -> List[Tuple[int, int, int, int]]:
        h, w = table_grid.shape[:2]
        cnts, hier = cv2.findContours(table_grid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        boxes: List[Tuple[int, int, int, int]] = []
        if hier is None or len(cnts) == 0:
            return boxes

        for i, c in enumerate(cnts):
            if hier[0][i][2] != -1:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if (
                min_width < bw < int(w * max_width_ratio)
                and min_height < bh < int(h * max_height_ratio)
            ):
                boxes.append((x, y, bw, bh))

        return boxes

    def crop_cell(
        self,
        img: np.ndarray,
        box: Tuple[int, int, int, int],
        padding: Optional[int] = None,
        right_extra: int = 0,
    ) -> np.ndarray:
        """
        Crops a cell with inner margin padding.
        right_extra expands the right edge by N pixels to capture
        the second digit of leading-zero two-digit values.
        """
        if padding is None:
            padding = self.cell_padding_px

        x, y, bw, bh = box
        img_h, img_w = img.shape[:2]

        x_start = min(max(0, x + padding), img_w - 1)
        y_start = min(max(0, y + padding), img_h - 1)
        x_end = max(x_start + 1, min(img_w, x + bw - padding + right_extra))
        y_end = max(y_start + 1, min(img_h, y + bh - padding))

        return img[y_start:y_end, x_start:x_end].copy()

    def extract_usn_grid(
        self, img: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        """
        Extracts the 10-character USN grid strip in the top 20% of Page 1.
        Returns (usn_crop, strip_box, char_boxes_sorted_left_to_right).
        char_boxes coordinates are absolute (relative to full img).
        """
        h, w = img.shape[:2]
        top_h = int(h * 0.22)
        top_roi = img[:top_h, :]

        gray = cv2.cvtColor(top_roi, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            ~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
        )

        h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.005), 1))
        v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.005)))
        h_l = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_k)
        v_l = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_k)
        grid = cv2.add(h_l, v_l)

        cnts, hier = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 1. Attempt to find the outer rectangular strip of the 10-box USN grid
        strip_cnt = None
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            if 600 <= bw <= 1300 and 65 <= bh <= 200 and (float(bw) / float(bh)) >= 4.5:
                if strip_cnt is None or bw > cv2.boundingRect(strip_cnt)[2]:
                    strip_cnt = c

        if strip_cnt is not None:
            sx, sy, sw, sh = cv2.boundingRect(strip_cnt)
            strip_box = (sx, sy, sw, sh)
            usn_crop = img[sy : sy + sh, sx : sx + sw].copy()
            slot_w = sw / 10.0
            slots: List[Tuple[int, int, int, int]] = [
                (int(sx + i * slot_w), sy, int(slot_w), sh) for i in range(10)
            ]
            return usn_crop, strip_box, slots

        # 2. Contour-based detection of individual character boxes
        candidate_boxes = []
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            ar = float(bw) / float(bh) if bh > 0 else 0
            if 45 <= bw <= 140 and 45 <= bh <= 150 and 0.4 <= ar <= 1.4:
                candidate_boxes.append((bx, by, bw, bh))

        candidate_boxes.sort(key=lambda b: (b[1], b[0]))
        rows: List[List[Tuple[int, int, int, int]]] = []
        for b in candidate_boxes:
            placed = False
            for r in rows:
                if abs(b[1] - r[0][1]) < 35:
                    r.append(b)
                    placed = True
                    break
            if not placed:
                rows.append([b])

        usn_row: List[Tuple[int, int, int, int]] = []
        for r in rows:
            sorted_r = sorted(r, key=lambda b: b[0])
            dedup_r: List[Tuple[int, int, int, int]] = []
            for b in sorted_r:
                if not dedup_r or abs(b[0] - dedup_r[-1][0]) > 20:
                    dedup_r.append(b)
            if len(dedup_r) >= 8:
                usn_row = dedup_r
                break

        if usn_row:
            min_x = usn_row[0][0]
            max_x = usn_row[-1][0] + usn_row[-1][2]
            min_y = min(b[1] for b in usn_row)
            max_y = max(b[1] + b[3] for b in usn_row)
            pad = 4
            strip_box = (
                max(0, min_x - pad),
                max(0, min_y - pad),
                min(w, max_x + pad) - max(0, min_x - pad),
                min(top_h, max_y + pad) - max(0, min_y - pad),
            )
            usn_crop = img[
                strip_box[1] : strip_box[1] + strip_box[3],
                strip_box[0] : strip_box[0] + strip_box[2],
            ].copy()
            return usn_crop, strip_box, usn_row

        # Fallback
        strip_box = (int(w * 0.18), int(h * 0.095), int(w * 0.30), int(h * 0.035))
        usn_crop = img[
            strip_box[1] : strip_box[1] + strip_box[3],
            strip_box[0] : strip_box[0] + strip_box[2],
        ].copy()
        return usn_crop, strip_box, []

    def extract_course_code_strip(
        self, img: np.ndarray, table_y_top: int
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        h, w = img.shape[:2]
        y_start = max(0, table_y_top - int(h * 0.065))
        y_end = max(y_start + 1, table_y_top - int(h * 0.035))
        x_start = int(w * 0.13)
        x_end = int(w * 0.62)
        strip_box = (x_start, y_start, x_end - x_start, y_end - y_start)
        crop = img[y_start:y_end, x_start:x_end].copy()
        return crop, strip_box

    def extract_page1_cells(
        self, img_input: Union[str, os.PathLike, np.ndarray], debug_save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extracts all Page 1 components using Grid-Anchored 4-point Perspective Table Normalization:
          - USN 10-char strip + individual char boxes
          - Course Code strip
          - 8 CIE marks cells normalized to flat canonical grid (1000x600)
            Column 5 ('Marks Obtained') extracted with 6px inward margin
        """
        if isinstance(img_input, np.ndarray):
            img = img_input.copy()
        else:
            img = cv2.imread(str(img_input))
            if img is None:
                raise FileNotFoundError(f"Could not load image at {img_input}")

        h, w = img.shape[:2]

        # 1. USN Grid extraction
        usn_crop, usn_box, usn_char_boxes = self.extract_usn_grid(img)

        # 2. Locate the CIE Table Bounding Box dynamically in lower half of page
        lower_offset_y = int(h * 0.35)
        lower_roi = img[lower_offset_y : int(h * 0.95), :]
        lower_gray = cv2.cvtColor(lower_roi, cv2.COLOR_BGR2GRAY)
        lower_thresh = cv2.adaptiveThreshold(
            ~lower_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
        )
        h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.03), 1))
        v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        hl = cv2.morphologyEx(lower_thresh, cv2.MORPH_OPEN, h_k)
        vl = cv2.morphologyEx(lower_thresh, cv2.MORPH_OPEN, v_k)
        grid = cv2.add(hl, vl)

        cnts, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_cnts = [c for c in cnts if cv2.contourArea(c) > (w * h * 0.04)]

        if large_cnts:
            table_cnt = max(large_cnts, key=cv2.contourArea)
            peri = cv2.arcLength(table_cnt, True)
            approx = cv2.approxPolyDP(table_cnt, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
            else:
                rect = cv2.minAreaRect(table_cnt)
                pts = cv2.boxPoints(rect).astype(np.float32)

            pts[:, 1] += lower_offset_y
            # Sort corners: TL, TR, BR, BL
            pts = pts[np.argsort(pts[:, 1])]
            top_2 = pts[:2][np.argsort(pts[:2, 0])]
            bot_2 = pts[2:][np.argsort(pts[2:, 0])]
            src_pts = np.float32([top_2[0], top_2[1], bot_2[1], bot_2[0]])
            table_y_top = int(min(top_2[0][1], top_2[1][1]))
        else:
            # Fallback table coordinates
            table_y_top = int(h * 0.48)
            src_pts = np.float32([
                [int(w * 0.10), int(h * 0.48)],
                [int(w * 0.90), int(h * 0.48)],
                [int(w * 0.90), int(h * 0.82)],
                [int(w * 0.10), int(h * 0.82)],
            ])

        # 3. Perspective Warp to Normalized 1000x600 Canonical Grid
        dst_pts = np.float32([[0, 0], [1000, 0], [1000, 600], [0, 600]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_table = cv2.warpPerspective(img, M, (1000, 600))

        # 4. Extract Course Code strip (located right above table_y_top)
        cc_crop, cc_box = self.extract_course_code_strip(img, table_y_top)

        # 5. Extract 8 data rows from Column 5 ('Marks Obtained')
        # Canonical layout: Header (0..85px), then 8 rows of ~64.3px
        # Column 5 ('Marks Obtained') is at x: 650..820
        row_dividers = [int(85 + i * 64.375) for i in range(9)]
        col_x1 = 650
        col_x2 = 820
        inner_margin = 6

        p1_keys = [
            "test1", "test2", "aba", "theory_total",
            "viva", "lab", "pract_total", "cie_total",
        ]

        crops: Dict[str, np.ndarray] = {}
        boxes: Dict[str, Tuple[int, int, int, int]] = {}

        for i, key in enumerate(p1_keys):
            y_start = row_dividers[i] + inner_margin
            y_end = row_dividers[i + 1] - inner_margin
            x_start = col_x1 + inner_margin
            x_end = col_x2 - inner_margin
            cell_crop = warped_table[y_start:y_end, x_start:x_end].copy()
            crops[key] = cell_crop
            boxes[key] = (x_start, y_start, x_end - x_start, y_end - y_start)

        # 6. Save visual debug overlay if requested
        if debug_save_path:
            debug_img = warped_table.copy()
            # Draw row dividers
            for r in row_dividers:
                cv2.line(debug_img, (0, r), (1000, r), (0, 255, 0), 2)
            # Draw Column 5 boundaries
            cv2.line(debug_img, (col_x1, 85), (col_x1, 600), (255, 0, 0), 2)
            cv2.line(debug_img, (col_x2, 85), (col_x2, 600), (255, 0, 0), 2)
            # Draw cell crops boxes
            for key, (bx, by, bw, bh) in boxes.items():
                cv2.rectangle(debug_img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                cv2.putText(
                    debug_img, key, (bx + 4, by + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1
                )
            os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
            cv2.imwrite(debug_save_path, debug_img)

        return {
            "usn_crop": usn_crop,
            "usn_box": usn_box,
            "usn_char_boxes": usn_char_boxes,
            "course_code_crop": cc_crop,
            "course_code_box": cc_box,
            "p1_marks_crops": crops,
            "p1_marks_boxes": boxes,
            "crops": crops,
            "boxes": boxes,
            "warped_table": warped_table,
            "image": img,
        }

    def extract_page2_cells(
        self, img_input: Union[str, os.PathLike, np.ndarray]
    ) -> Dict[str, Any]:
        """Extracts Page 2 question-mark cells (preserved for two-page workflows)."""
        if isinstance(img_input, np.ndarray):
            img = img_input.copy()
        else:
            img = cv2.imread(str(img_input))
            if img is None:
                raise FileNotFoundError(f"Could not load image at {img_input}")

        table_grid, _, _ = self.detect_grid(img)
        raw_boxes = self.extract_bounding_boxes(table_grid)

        t1_cells = [b for b in raw_boxes if 1150 <= b[0] + b[2] / 2 <= 1650 and b[1] >= 720]
        t2_cells = [b for b in raw_boxes if 2200 <= b[0] + b[2] / 2 <= 2800 and b[1] >= 720]
        t1_cells.sort(key=lambda b: b[1])
        t2_cells.sort(key=lambda b: b[1])

        if len(t1_cells) > 11:
            t1_data = t1_cells[1:12]
        else:
            t1_data = t1_cells

        if len(t1_data) == 11:
            t1_questions, t1_total_box = t1_data[:10], [t1_data[10]]
        elif len(t1_data) > 0:
            t1_questions, t1_total_box = t1_data[:-1], [t1_data[-1]]
        else:
            t1_questions, t1_total_box = [], []

        if len(t2_cells) > 11:
            t2_data = t2_cells[1:12]
        else:
            t2_data = t2_cells

        if len(t2_data) == 11:
            t2_questions, t2_total_box = t2_data[:10], [t2_data[10]]
        elif len(t2_data) > 0:
            t2_questions, t2_total_box = t2_data[:-1], [t2_data[-1]]
        else:
            t2_questions, t2_total_box = [], []

        p2_test1_crops: Dict[str, np.ndarray] = {}
        p2_test1_boxes: Dict[str, Tuple[int, int, int, int]] = {}
        p2_test2_crops: Dict[str, np.ndarray] = {}
        p2_test2_boxes: Dict[str, Tuple[int, int, int, int]] = {}

        for q_idx in range(1, 11):
            key = f"q{q_idx}"
            if q_idx - 1 < len(t1_questions):
                box = t1_questions[q_idx - 1]
                p2_test1_boxes[key] = box
                p2_test1_crops[key] = self.crop_cell(img, box)

        if t1_total_box:
            p2_test1_boxes["total"] = t1_total_box[0]
            p2_test1_crops["total"] = self.crop_cell(img, t1_total_box[0])

        for q_idx in range(1, 11):
            key = f"q{q_idx}"
            if q_idx - 1 < len(t2_questions):
                box = t2_questions[q_idx - 1]
                p2_test2_boxes[key] = box
                p2_test2_crops[key] = self.crop_cell(img, box)

        if t2_total_box:
            p2_test2_boxes["total"] = t2_total_box[0]
            p2_test2_crops["total"] = self.crop_cell(img, t2_total_box[0])

        combined_crops: Dict[str, np.ndarray] = {}
        combined_boxes: Dict[str, Tuple[int, int, int, int]] = {}
        for k, v in p2_test1_crops.items():
            combined_crops[f"{k}_test1"] = v
            combined_boxes[f"{k}_test1"] = p2_test1_boxes[k]
        for k, v in p2_test2_crops.items():
            combined_crops[f"{k}_test2"] = v
            combined_boxes[f"{k}_test2"] = p2_test2_boxes[k]

        return {
            "p2_test1_crops": p2_test1_crops,
            "p2_test1_boxes": p2_test1_boxes,
            "p2_test2_crops": p2_test2_crops,
            "p2_test2_boxes": p2_test2_boxes,
            "crops": combined_crops,
            "boxes": combined_boxes,
            "all_boxes": raw_boxes,
            "table_grid": table_grid,
            "image": img,
        }
