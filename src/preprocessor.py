"""
Table, Header, and Cell Extraction Module (Phase 2)
Extracts tabular marksheet cells, USN grid, course code, and test totals
from 2-Internal MITE booklet images using OpenCV.
"""

import os
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps


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

    def crop_to_page_sheet(
        self, img_bgr: np.ndarray, target_size: Tuple[int, int] = (1200, 1600)
    ) -> Tuple[np.ndarray, bool]:
        """
        Detects the bright white A4 booklet page contour against the background desk,
        crops/warps tightly to the page boundary using perspective transform,
        and resizes to a standard canonical resolution (default 1200 x 1600).
        """
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr, False

        h, w = img_bgr.shape[:2]
        scale = 800.0 / max(h, w)
        sw, sh = int(w * scale), int(h * scale)
        small = cv2.resize(img_bgr, (sw, sh))

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 0.40 * (sw * sh):
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32) / scale
                else:
                    rect = cv2.minAreaRect(c)
                    pts = cv2.boxPoints(rect).astype(np.float32) / scale

                pts = pts[np.argsort(pts[:, 1])]
                top_2 = pts[:2][np.argsort(pts[:2, 0])]
                bot_2 = pts[2:][np.argsort(pts[2:, 0])]
                src_pts = np.float32([top_2[0], top_2[1], bot_2[1], bot_2[0]])
                dst_pts = np.float32([
                    [0, 0],
                    [target_size[0], 0],
                    [target_size[0], target_size[1]],
                    [0, target_size[1]],
                ])
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(img_bgr, M, target_size)
                return warped, True

        return cv2.resize(img_bgr, target_size), False

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
        Extracts the 10-character USN grid strip in the top portion of Page 1.
        Returns (usn_crop, strip_box, char_boxes_sorted_left_to_right).
        char_boxes coordinates are absolute (relative to full img).
        """
        h, w = img.shape[:2]
        top_h = int(h * 0.25)
        top_roi = img[:top_h, :]

        gray = cv2.cvtColor(top_roi, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            ~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
        )

        h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(w * 0.015)), 1))
        v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(h * 0.015))))
        h_l = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_k)
        v_l = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_k)
        grid = cv2.add(h_l, v_l)

        cnts, hier = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 1. Contour-based detection of individual character boxes (w in 15..55, h in 20..65)
        candidate_boxes = []
        box_min_w = max(15, int(w * 0.015))
        box_max_w = max(55, int(w * 0.06))
        box_min_h = max(20, int(h * 0.015))
        box_max_h = max(65, int(h * 0.06))

        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            ar = float(bw) / float(bh) if bh > 0 else 0
            if box_min_w <= bw <= box_max_w and box_min_h <= bh <= box_max_h and 0.4 <= ar <= 1.4:
                candidate_boxes.append((bx, by, bw, bh))

        candidate_boxes.sort(key=lambda b: (b[1], b[0]))
        rows: List[List[Tuple[int, int, int, int]]] = []
        for b in candidate_boxes:
            placed = False
            for r in rows:
                if abs(b[1] - r[0][1]) < 22:
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
                if not dedup_r or abs(b[0] - dedup_r[-1][0]) > 15:
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

        # 2. Outer rectangular strip fallback
        strip_cnt = None
        min_w = int(w * 0.20)
        max_w = int(w * 0.55)
        min_h = int(h * 0.02)
        max_h = int(h * 0.08)

        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            if min_w <= bw <= max_w and min_h <= bh <= max_h and (float(bw) / float(bh)) >= 4.0:
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
        y_start = max(0, table_y_top - int(h * 0.09))
        y_end = max(y_start + 1, table_y_top - int(h * 0.02))
        x_start = int(w * 0.10)
        x_end = int(w * 0.65)
        strip_box = (x_start, y_start, x_end - x_start, y_end - y_start)
        crop = img[y_start:y_end, x_start:x_end].copy()
        return crop, strip_box

    def extract_page1_cells(
        self, img_input: Union[str, os.PathLike, np.ndarray], debug_save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extracts all Page 1 components using:
          1. EXIF orientation normalization via PIL ImageOps.exif_transpose.
          2. Automatic White-Page boundary cropping & warp to canonical (1200x1600).
          3. Grid-anchored 4-point perspective table normalization (1000x600).
          4. Morphological line detection to find exact intersection grid of 8 rows
             and Column 5 ('Marks Obtained').
        """
        # 1. EXIF Orientation Normalization
        if isinstance(img_input, np.ndarray):
            img_orig = img_input.copy()
        else:
            pil_img = ImageOps.exif_transpose(Image.open(str(img_input)))
            img_orig = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 2. Automatic White-Page Boundary Cropping to Canonical 1200x1600
        page, page_found = self.crop_to_page_sheet(img_orig, target_size=(1200, 1600))
        h, w = page.shape[:2]

        # 3. USN Grid extraction
        usn_crop, usn_box, usn_char_boxes = self.extract_usn_grid(page)

        # 4. Locate CIE Table Bounding Box dynamically in lower half
        lower_offset_y = int(h * 0.35)
        lower_roi = page[lower_offset_y : int(h * 0.95), :]
        lower_gray = cv2.cvtColor(lower_roi, cv2.COLOR_BGR2GRAY)
        lower_thresh = cv2.adaptiveThreshold(
            ~lower_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
        )
        h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.035), 1))
        v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        hl = cv2.morphologyEx(lower_thresh, cv2.MORPH_OPEN, h_k)
        vl = cv2.morphologyEx(lower_thresh, cv2.MORPH_OPEN, v_k)
        # Dilate lines to close any tiny gap at borders
        grid = cv2.add(
            cv2.dilate(hl, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))),
            cv2.dilate(vl, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))),
        )

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
            pts = pts[np.argsort(pts[:, 1])]
            top_2 = pts[:2][np.argsort(pts[:2, 0])]
            bot_2 = pts[2:][np.argsort(pts[2:, 0])]
            src_pts = np.float32([top_2[0], top_2[1], bot_2[1], bot_2[0]])
            table_y_top = int(min(top_2[0][1], top_2[1][1]))
        else:
            table_y_top = int(h * 0.48)
            src_pts = np.float32([
                [int(w * 0.10), int(h * 0.48)],
                [int(w * 0.90), int(h * 0.48)],
                [int(w * 0.90), int(h * 0.82)],
                [int(w * 0.10), int(h * 0.82)],
            ])

        # 5. Perspective Warp to Normalized 1000x600 Canonical Grid
        dst_pts = np.float32([[0, 0], [1000, 0], [1000, 600], [0, 600]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_table = cv2.warpPerspective(page, M, (1000, 600))

        # 6. Extract Course Code strip (located right above table_y_top)
        cc_crop, cc_box = self.extract_course_code_strip(page, table_y_top)

        # 7. Grid-Line Anchored Column 5 and 8-Row Detection in warped_table
        w_gray = cv2.cvtColor(warped_table, cv2.COLOR_BGR2GRAY)
        w_thresh = cv2.adaptiveThreshold(
            ~w_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
        )
        w_hl = cv2.morphologyEx(w_thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))
        w_vl = cv2.morphologyEx(w_thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40)))

        # Find Column 5 ('Marks Obtained') bounds
        v_proj = np.sum(w_vl > 0, axis=0)
        col_lines = []
        for x in range(len(v_proj)):
            if v_proj[x] > 150:
                if not col_lines or x - col_lines[-1] > 25:
                    col_lines.append(x)

        m_x1, m_x2 = None, None
        for i in range(len(col_lines) - 1):
            x_a, x_b = col_lines[i], col_lines[i + 1]
            if 590 <= x_a <= 670 and 750 <= x_b <= 840 and (130 <= x_b - x_a <= 190):
                m_x1, m_x2 = x_a, x_b
                break

        if m_x1 is None:
            if len(col_lines) <= 6:
                m_x1, m_x2 = 618, 778
            else:
                m_x1, m_x2 = 648, 820

        # Find row lines within this column
        col_strip = w_hl[:, m_x1:m_x2]
        h_proj = np.sum(col_strip > 0, axis=1)
        row_lines = []
        for y in range(len(h_proj)):
            if h_proj[y] > 25:
                if not row_lines or y - row_lines[-1] > 20:
                    row_lines.append(y)

        # Build data intervals from detected horizontal grid lines
        data_intervals = []
        if len(row_lines) >= 8:
            for i in range(1, len(row_lines) - 1):
                data_intervals.append((row_lines[i], row_lines[i + 1]))

        p1_keys = [
            "test1", "test2", "aba", "theory_total",
            "viva", "lab", "pract_total", "cie_total",
        ]
        crops: Dict[str, np.ndarray] = {}
        boxes: Dict[str, Tuple[int, int, int, int]] = {}
        inner_margin = 4

        def make_cell(y1: int, y2: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
            ys = max(0, y1 + inner_margin)
            ye = min(600, y2 - inner_margin)
            xs = max(0, m_x1 + inner_margin)
            xe = min(1000, m_x2 - inner_margin + self.MARKS_COL_RIGHT_EXTRA_PX)
            c = warped_table[ys:ye, xs:xe].copy()
            b = (xs, ys, xe - xs, ye - ys)
            return c, b

        blank_cell = np.full((40, m_x2 - m_x1, 3), 255, dtype=np.uint8)

        # Structure Detection:
        # Template 2: Theory-23 (7 Data Rows — Used in 24MCPC521 / Sub2)
        # Rows:
        #   1. test1 (Max 15)
        #   2. test2 (Max 15)
        #   3. test_total (Max 30) -> test1 + test2
        #   4. aba1 (Max 10)
        #   5. aba2 (Max 10)
        #   6. aba_total (Max 20) -> aba1 + aba2
        #   7. cie_total (Max 50) -> test_total + aba_total
        #
        # Template 1: Integrated-23 (8 Data Rows — Used in 24MCPC522 & 24MCPC517)
        # Rows:
        #   1. test1 (Max 15)
        #   2. test2 (Max 15)
        #   3. aba (Max 10)
        #   4. theory_total (Max 40)
        #   5. viva (Max 05)
        #   6. lab (Max 05)
        #   7. pract_total (Max 10)
        #   8. cie_total (Max 50)

        is_theory_23 = (len(data_intervals) == 7)

        if is_theory_23:
            # 7-row Theory-23 Layout
            c_t1, b_t1 = make_cell(*data_intervals[0])
            c_t2, b_t2 = make_cell(*data_intervals[1])
            c_test_tot, b_test_tot = make_cell(*data_intervals[2])
            c_aba1, b_aba1 = make_cell(*data_intervals[3])
            c_aba2, b_aba2 = make_cell(*data_intervals[4])
            c_aba_tot, b_aba_tot = make_cell(*data_intervals[5])
            c_cie, b_cie = make_cell(*data_intervals[6])

            # Composite ABA combining ABA 1 and ABA 2
            c_aba_comp, b_aba_comp = make_cell(data_intervals[3][0], data_intervals[4][1])

            crops["test1"] = c_t1; boxes["test1"] = b_t1
            crops["test2"] = c_t2; boxes["test2"] = b_t2
            crops["test_total"] = c_test_tot; boxes["test_total"] = b_test_tot
            crops["aba1"] = c_aba1; boxes["aba1"] = b_aba1
            crops["aba2"] = c_aba2; boxes["aba2"] = b_aba2
            crops["aba_total"] = c_aba_tot; boxes["aba_total"] = b_aba_tot
            # Map aba to aba_total crop (or composite)
            crops["aba"] = c_aba_tot; boxes["aba"] = b_aba_tot
            # Theory total in unified schema maps to cie_total (or test_total + aba_total)
            crops["theory_total"] = c_cie; boxes["theory_total"] = b_cie
            crops["viva"] = blank_cell.copy(); boxes["viva"] = (m_x1, data_intervals[5][0], m_x2 - m_x1, 20)
            crops["lab"] = blank_cell.copy(); boxes["lab"] = (m_x1, data_intervals[5][0] + 20, m_x2 - m_x1, 20)
            crops["pract_total"] = blank_cell.copy(); boxes["pract_total"] = (m_x1, data_intervals[5][0] + 40, m_x2 - m_x1, 20)
            crops["cie_total"] = c_cie; boxes["cie_total"] = b_cie

        elif len(data_intervals) >= 9:
            # 2-ABA layout with separate viva and lab (Test I, Test II, ABA I, ABA II, Theory Total, Viva, Lab, Practical Total, CIE Total)
            c_t1, b_t1 = make_cell(*data_intervals[0])
            c_t2, b_t2 = make_cell(*data_intervals[1])
            c_aba1, b_aba1 = make_cell(*data_intervals[2])
            c_aba2, b_aba2 = make_cell(*data_intervals[3])
            c_aba_comp, b_aba_comp = make_cell(data_intervals[2][0], data_intervals[3][1])
            c_th, b_th = make_cell(*data_intervals[4])
            c_viva, b_viva = make_cell(*data_intervals[5])
            c_lab, b_lab = make_cell(*data_intervals[6])
            c_pr, b_pr = make_cell(*data_intervals[7])
            c_cie, b_cie = make_cell(*data_intervals[8])

            crops["test1"] = c_t1; boxes["test1"] = b_t1
            crops["test2"] = c_t2; boxes["test2"] = b_t2
            crops["aba1"] = c_aba1; boxes["aba1"] = b_aba1
            crops["aba2"] = c_aba2; boxes["aba2"] = b_aba2
            crops["aba"] = c_aba_comp; boxes["aba"] = b_aba_comp
            crops["theory_total"] = c_th; boxes["theory_total"] = b_th
            crops["viva"] = c_viva; boxes["viva"] = b_viva
            crops["lab"] = c_lab; boxes["lab"] = b_lab
            crops["pract_total"] = c_pr; boxes["pract_total"] = b_pr
            crops["cie_total"] = c_cie; boxes["cie_total"] = b_cie

        else:
            # Standard 8-row Integrated-23 layout (or proportional fallback)
            if len(data_intervals) < 8:
                if m_x1 < 635:
                    r_divs = [105, 168, 230, 292, 358, 425, 485, 548, 600]
                else:
                    r_divs = [83, 140, 195, 305, 360, 415, 470, 525, 590]
                data_intervals = [(r_divs[k], r_divs[k + 1]) for k in range(8)]

            p1_keys = [
                "test1", "test2", "aba", "theory_total",
                "viva", "lab", "pract_total", "cie_total",
            ]
            for i, key in enumerate(p1_keys):
                y1, y2 = data_intervals[i]
                c, b = make_cell(y1, y2)
                crops[key] = c
                boxes[key] = b

        # 8. Save visual debug overlay
        debug_img = warped_table.copy()
        cv2.line(debug_img, (m_x1, 0), (m_x1, 600), (255, 0, 0), 2)
        cv2.line(debug_img, (m_x2, 0), (m_x2, 600), (255, 0, 0), 2)
        for i, key in enumerate(p1_keys):
            bx, by, bw, bh = boxes[key]
            cv2.rectangle(debug_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.putText(
                debug_img, key, (bx + 4, by + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1
            )

        if debug_save_path:
            os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
            cv2.imwrite(debug_save_path, debug_img)

        # Also auto-save overlay in debug_sub1_sub2 for visual verification
        try:
            auto_debug_dir = Path("data/crops/debug_sub1_sub2")
            auto_debug_dir.mkdir(parents=True, exist_ok=True)
            if not isinstance(img_input, np.ndarray):
                stem = Path(img_input).stem
                cv2.imwrite(str(auto_debug_dir / f"overlay_{stem}.jpg"), debug_img)
        except Exception:
            pass

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
            "image": page,
        }

    def extract_page2_cells(
        self, img_input: Union[str, os.PathLike, np.ndarray]
    ) -> Dict[str, Any]:
        """Extracts Page 2 question-mark cells (preserved for two-page workflows)."""
        if isinstance(img_input, np.ndarray):
            img = img_input.copy()
        else:
            pil_img = ImageOps.exif_transpose(Image.open(str(img_input)))
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
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
