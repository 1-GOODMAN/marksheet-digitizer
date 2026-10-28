"""
Probabilistic TrOCR Neural Inference Engine
Performs handwritten digit and alphanumeric recognition with beam-search
posterior probability estimation, red-ink contrast enhancement, institutional
domain priors (USN prefix lock, course-code snapping), and leading-zero fix.
"""

from typing import Dict, Optional, Union, List, Tuple
import re
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .config import (
    OCR_MODEL_NAME,
    BEAM_SIZE,
    BLANK_STD_THRESHOLD,
    BLANK_MEAN_BRIGHTNESS_THRESHOLD,
    STRIKE_OUT_MAX_BRIGHTNESS,
    STRIKE_OUT_AREA_RATIO,
    BLANK_PROBABILITY,
    STRIKE_OUT_PROBABILITY,
    USN_PREFIX,
    USN_TOTAL_LEN,
    VALID_COURSE_CODES,
    USN_CHAR_CONFUSION,
)


def _levenshtein(a: str, b: str) -> int:
    """Pure-Python Levenshtein distance (no external deps)."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[n]


class ProbabilisticOCR:
    """
    TrOCR-based inference engine with:
    - Red-ink contrast enhancement (green-channel CLAHE)
    - Blank / struck-out cell heuristics
    - Leading-zero fix (05->0 truncation corrected by right-half ink detection)
    - USN hard-lock prefix decoding (4MT24MCxxx)
    - Course code Levenshtein snapping to {24MCPC522, 24MCPC517, 24MCPC521}
    """

    VISUAL_CONFUSIONS = {
        "cog": 9, "bog": 9, "cg": 9,
        "color": 6, "co": 6, "do": 6, "oo": 6,
        "was": 8, "ever": 8, "era": 8, "or": 8, "oz": 8, "log": 8, "og": 8,
        "of": 7, "if": 7,
        "what": 5, "who": 5,
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        blank_std_threshold: float = BLANK_STD_THRESHOLD,
    ):
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = model_name or OCR_MODEL_NAME
        self.blank_std_threshold = blank_std_threshold
        try:
            self.processor = TrOCRProcessor.from_pretrained(self.model_name, local_files_only=True)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name, local_files_only=True).to(self.device)
        except Exception:
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return img
        elif img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return img

    def enhance_digit_contrast(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Enhances handwritten strokes for TrOCR consumption:
        1. Strips any black border lines reaching the outer edges via connected components.
        2. When red ink is detected (S > 35), suppresses low-saturation black pixels and
           isolates red ink strokes (255 - G_channel masked by saturation).
        3. Returns 3-channel RGB with dark ink on clean white background.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr

        if len(crop_bgr.shape) == 2:
            crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)

        h, w = crop_bgr.shape[:2]

        # 1. Strip black border lines touching outer 3px border using connected components
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        _, border_mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(border_mask, connectivity=8)
        clean_bgr = crop_bgr.copy()
        for i in range(1, nlabels):
            bx, by, bw, bh, area = stats[i]
            # If component touches the outer 3px edge and is long/thin (a border line)
            touches_edge = (bx <= 2) or (by <= 2) or (bx + bw >= w - 2) or (by + bh >= h - 2)
            if touches_edge and (bw >= int(w * 0.4) or bh >= int(h * 0.4)):
                clean_bgr[labels == i] = [255, 255, 255]

        # 2. Red ink isolation
        hsv = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2HSV)
        mask_r = ((hsv[:, :, 0] <= 15) | (hsv[:, :, 0] >= 165)) & (hsv[:, :, 1] > 35)
        red_pixel_count = int(np.sum(mask_r))

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        if red_pixel_count > 40:
            g = clean_bgr[:, :, 1]
            inv_g = 255 - g
            sat = hsv[:, :, 1]
            # Stroke mask: high saturation and significant inverted green
            stroke_mask = (sat > 35) & (inv_g > 25)
            clean_gray = np.zeros_like(g)
            clean_gray[stroke_mask] = inv_g[stroke_mask]
            enhanced = 255 - clean_gray

            # Center on ink bounding box with generous padding
            coords = cv2.findNonZero(clean_gray)
            if coords is not None:
                cx, cy, cw, ch = cv2.boundingRect(coords)
                pad_x = max(30, int(cw * 0.35))
                pad_y = max(15, int(ch * 0.20))
                y1 = max(0, cy - pad_y)
                y2 = min(enhanced.shape[0], cy + ch + pad_y)
                x1 = max(0, cx - pad_x)
                x2 = min(enhanced.shape[1], cx + cw + pad_x)
                enhanced = enhanced[y1:y2, x1:x2]
        else:
            c_gray = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY)
            if float(c_gray.std()) < self.blank_std_threshold:
                enhanced = c_gray
            else:
                enhanced = clahe.apply(c_gray)

        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    def is_blank_or_paper_texture(self, gray: np.ndarray) -> bool:
        std = float(gray.std())
        mean = float(gray.mean())
        if mean > BLANK_MEAN_BRIGHTNESS_THRESHOLD:
            return True
        if std < self.blank_std_threshold:
            return True
        dark_ink_count = int(np.sum(gray < 130))
        if std < 6.0 and dark_ink_count < 50:
            return True
        return False

    def is_struck_out_or_scribble(self, gray: np.ndarray) -> bool:
        dark_ratio = float(np.sum(gray < STRIKE_OUT_MAX_BRIGHTNESS)) / float(gray.size)
        if dark_ratio >= STRIKE_OUT_AREA_RATIO or float(gray.mean()) < STRIKE_OUT_MAX_BRIGHTNESS:
            return True
        return False

    def _has_right_half_ink(self, crop_bgr: np.ndarray) -> bool:
        """Returns True if the right half of the crop has meaningful ink (>=40 dark px)."""
        if crop_bgr is None or crop_bgr.size == 0:
            return False
        h, w = crop_bgr.shape[:2]
        right_half = crop_bgr[:, w // 2:, :]
        gray_right = self._to_gray(right_half)
        return int(np.sum(gray_right < 128)) > 40

    def _parse_digit_from_text(self, text: str) -> Optional[int]:
        """
        Parses numeric mark value from decoded TrOCR text.
        S/s always maps to 5 (critical for '0S'->05 leading-zero fix).
        """
        if not text:
            return None
        t = text.strip()

        # 1. Exact 2-digit number
        direct = re.findall(r"\b\d{2}\b", t)
        if direct and int(direct[0]) <= 50:
            return int(direct[0])

        # 2. Visual confusion word mappings
        words = re.findall(r"[a-zA-Z]+", t.lower())
        for w in words:
            if w in self.VISUAL_CONFUSIONS:
                return self.VISUAL_CONFUSIONS[w]

        # 3. Space-separated single digits
        tokens = t.split()
        single_digits = [tok for tok in tokens if tok.isdigit() and len(tok) == 1]
        if len(single_digits) >= 2:
            combined = int("".join(single_digits[:2]))
            if combined <= 50:
                return combined

        # 4. Stroke-level substitution (S->5 regardless of position)
        mapped = ""
        for ch in t:
            if ch.isdigit():
                mapped += ch
            elif ch in ["I", "l", "|", "!", ";", ",", "j", "(", ")", "'"]:
                mapped += "1"
            elif ch in ["O", "o"]:
                mapped += "0"
            elif ch in ["S", "s"]:
                mapped += "5"
            elif ch.isspace() or ch == ".":
                continue
            else:
                mapped += " "

        m_nums = re.findall(r"\d+", mapped)
        if m_nums:
            for m in m_nums:
                if len(m) == 2 and int(m) <= 50:
                    return int(m)
            v = int(m_nums[0])
            if v <= 50:
                return v
            if len(str(v)) >= 2 and int(str(v)[:2]) <= 50:
                return int(str(v)[:2])

        all_d = re.findall(r"\d+", t)
        if all_d:
            v = int(all_d[0])
            if v <= 50:
                return v
        return None

    def _apply_leading_zero_prior(
        self,
        candidate_scores: Dict[int, float],
        crop_bgr: Optional[np.ndarray],
    ) -> Dict[int, float]:
        """
        Leading-zero fix:
        When ink std > 12 but P(0) > 0.70 AND right-half of crop also has ink,
        inject P(5), P(3), P(8) as alternative second-digit candidates and
        down-weight standalone 0 by x0.15.
        This corrects '05'->'0', '03'->'0', '08'->'0' OCR truncation.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return candidate_scores
        gray = self._to_gray(crop_bgr)
        std = float(gray.std())
        dominant_is_zero = candidate_scores.get(0, 0.0) > 0.70 and 0 in candidate_scores
        if not (std > 12.0 and dominant_is_zero):
            return candidate_scores
        if not self._has_right_half_ink(crop_bgr):
            return candidate_scores

        adjusted = dict(candidate_scores)
        adjusted[0] = adjusted[0] * 0.15
        if 5 not in adjusted:
            adjusted[5] = 0.55
        if 3 not in adjusted:
            adjusted[3] = 0.15
        if 8 not in adjusted:
            adjusted[8] = 0.10

        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        return adjusted

    def predict_digit_candidates(
        self, crop_img: np.ndarray, beam_size: int = 5, is_usn_cell: bool = False
    ) -> Dict[int, float]:
        """Single-cell digit candidate prediction with all priors applied."""
        if crop_img is None or crop_img.size == 0:
            return {0: BLANK_PROBABILITY}

        enhanced_rgb = self.enhance_digit_contrast(crop_bgr=crop_img)
        enhanced_gray = self._to_gray(enhanced_rgb)

        if not is_usn_cell and self.is_blank_or_paper_texture(enhanced_gray):
            return {0: BLANK_PROBABILITY}
        if not is_usn_cell and self.is_struck_out_or_scribble(enhanced_gray):
            return {0: STRIKE_OUT_PROBABILITY}

        pil_img = Image.fromarray(enhanced_rgb)
        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values.to(self.device)

        eff_beam = max(beam_size, 5)
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                num_beams=eff_beam,
                num_return_sequences=eff_beam,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=8,
            )

        candidate_scores: Dict[int, float] = {}
        for i, seq in enumerate(outputs.sequences):
            decoded = self.processor.decode(seq, skip_special_tokens=True).strip()
            score = float(outputs.sequences_scores[i])
            prob = float(torch.exp(torch.tensor(score)))
            digit = self._parse_digit_from_text(decoded)
            if digit is not None:
                candidate_scores[digit] = candidate_scores.get(digit, 0.0) + prob

        if not is_usn_cell:
            candidate_scores = self._apply_leading_zero_prior(candidate_scores, crop_img)

        if candidate_scores:
            total_prob = sum(candidate_scores.values())
            normalized = {k: round(v / total_prob, 4) for k, v in candidate_scores.items()}
            prob_sum = sum(normalized.values())
            if prob_sum > 0:
                first_key = next(iter(normalized))
                normalized[first_key] = round(normalized[first_key] + (1.0 - prob_sum), 4)
            return normalized
        return {0: BLANK_PROBABILITY}

    def predict_digit_candidates_batch(
        self, crop_dict: Dict[str, np.ndarray], beam_size: int = BEAM_SIZE
    ) -> Dict[str, Dict[int, float]]:
        """Batch digit candidate prediction (single TrOCR forward pass)."""
        results: Dict[str, Dict[int, float]] = {}
        active_keys: List[str] = []
        active_pil: List[Image.Image] = []
        active_orig: List[np.ndarray] = []

        for key, crop_img in crop_dict.items():
            if crop_img is None or crop_img.size == 0:
                results[key] = {0: BLANK_PROBABILITY}
                continue
            enhanced_rgb = self.enhance_digit_contrast(crop_bgr=crop_img)
            enhanced_gray = self._to_gray(enhanced_rgb)
            if self.is_blank_or_paper_texture(enhanced_gray):
                results[key] = {0: BLANK_PROBABILITY}
            elif self.is_struck_out_or_scribble(enhanced_gray):
                results[key] = {0: STRIKE_OUT_PROBABILITY}
            else:
                active_keys.append(key)
                active_pil.append(Image.fromarray(enhanced_rgb))
                active_orig.append(crop_img)

        if active_pil:
            pixel_values = self.processor(active_pil, return_tensors="pt").pixel_values.to(self.device)
            eff_beam = max(beam_size, 4)
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    num_beams=eff_beam,
                    num_return_sequences=eff_beam,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=8,
                )

            for j, key in enumerate(active_keys):
                candidate_scores: Dict[int, float] = {}
                start_b = j * eff_beam
                for i in range(start_b, start_b + eff_beam):
                    seq = outputs.sequences[i]
                    decoded = self.processor.decode(seq, skip_special_tokens=True).strip()
                    score = float(outputs.sequences_scores[i])
                    prob = float(torch.exp(torch.tensor(score)))
                    digit = self._parse_digit_from_text(decoded)
                    if digit is not None:
                        candidate_scores[digit] = candidate_scores.get(digit, 0.0) + prob

                candidate_scores = self._apply_leading_zero_prior(candidate_scores, active_orig[j])

                if candidate_scores:
                    total_prob = sum(candidate_scores.values())
                    normalized = {k: round(v / total_prob, 4) for k, v in candidate_scores.items()}
                    prob_sum = sum(normalized.values())
                    if prob_sum > 0:
                        first_k = next(iter(normalized))
                        normalized[first_k] = round(normalized[first_k] + (1.0 - prob_sum), 4)
                    results[key] = normalized
                else:
                    results[key] = {0: BLANK_PROBABILITY}

        return results

    def predict_usn(
        self,
        usn_crop: Optional[np.ndarray],
        char_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        full_img: Optional[np.ndarray] = None,
    ) -> str:
        """
        Decodes USN with hard-locked prefix '4MT24MC'.
        Runs digit OCR on the 3 suffix character boxes (slots 7, 8, 9).
        Shaves 18% off left/right and 15% off top/bottom of each box to remove
        black grid borders completely (avoiding '111' border hallucination).
        Never applies leading-zero prior on USN character boxes!
        """
        suffix_digits: List[str] = []

        if char_boxes and len(char_boxes) >= 3 and full_img is not None:
            for box in char_boxes[-3:]:
                bx, by, bw, bh = box
                h_img, w_img = full_img.shape[:2]
                # Shave 18% horizontal and 15% vertical to eliminate border lines
                shave_x = max(2, int(bw * 0.18))
                shave_y = max(2, int(bh * 0.15))
                x1, y1 = max(0, bx + shave_x), max(0, by + shave_y)
                x2, y2 = min(w_img, bx + bw - shave_x), min(h_img, by + bh - shave_y)
                char_crop = full_img[y1:y2, x1:x2]

                raw_t = self.predict_text(char_crop).strip()
                # Apply character confusion mapping
                mapped_ch = ""
                for ch in raw_t:
                    mapped_ch += USN_CHAR_CONFUSION.get(ch, ch)
                digits_found = re.findall(r"\d", mapped_ch)
                if digits_found:
                    suffix_digits.append(digits_found[0])
                else:
                    # Fallback to beam search without leading zero prior
                    dist = self.predict_digit_candidates(char_crop, beam_size=5, is_usn_cell=True)
                    best_digit = max(dist, key=dist.get)
                    suffix_digits.append(str(best_digit % 10))

        if len(suffix_digits) < 3 and usn_crop is not None:
            raw_text = self.predict_text(usn_crop)
            mapped = "".join(USN_CHAR_CONFUSION.get(ch, ch) for ch in raw_text)
            digits_only = re.sub(r"[^0-9]", "", mapped)
            if len(digits_only) >= 3:
                suffix_digits = list(digits_only[-3:])
            elif len(digits_only) > 0:
                suffix_digits = list(digits_only.zfill(3))
            else:
                suffix_digits = ["0", "0", "0"]

        if len(suffix_digits) < 3:
            suffix_digits = (suffix_digits + ["0", "0", "0"])[:3]

        roll_str = "".join(suffix_digits[:3])
        try:
            roll_num = int(roll_str)
        except ValueError:
            roll_num = 1

        # Enforce strict institutional domain rule: roll number strictly within (001 - 199)
        if roll_num < 1:
            roll_num = 1
        elif roll_num > 199:
            cand = roll_num % 100
            if 1 <= cand <= 199:
                roll_num = cand
            else:
                roll_num = min(199, max(1, roll_num % 200))

        return f"{USN_PREFIX}{roll_num:03d}"

    def predict_course_code(self, cc_crop: Optional[np.ndarray], folder_default: str = "24MCPC522") -> str:
        """
        Extracts course code and snaps to nearest VALID_COURSE_CODES entry.
        Uses folder-aware default unless OCR explicitly detects '522', '517', or '521' with confidence.
        """
        if cc_crop is None or cc_crop.size == 0:
            return folder_default

        raw_text = self.predict_text(cc_crop).upper()
        if not raw_text:
            return folder_default

        # Look for explicit subject suffixes
        if "522" in raw_text:
            return "24MCPC522"
        if "517" in raw_text:
            return "24MCPC517"
        if "521" in raw_text:
            return "24MCPC521"

        # Levenshtein distance fallback
        best_code = folder_default
        best_dist: float = float("inf")
        for code in VALID_COURSE_CODES:
            full_dist = _levenshtein(raw_text, code)
            raw_suffix = raw_text[-3:] if len(raw_text) >= 3 else raw_text
            suffix_dist = _levenshtein(raw_suffix, code[-3:])
            combined = full_dist + suffix_dist * 2
            if combined < best_dist:
                best_dist = combined
                best_code = code

        # Only override folder_default if match is strong (dist <= 3)
        if best_dist <= 3:
            return best_code
        return folder_default

    def predict_text(self, crop_img: Optional[np.ndarray]) -> str:
        """Generic alphanumeric OCR (uppercase, digits + letters only)."""
        if crop_img is None or crop_img.size == 0:
            return ""
        if len(crop_img.shape) == 2:
            rgb = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(pixel_values, num_beams=BEAM_SIZE, max_new_tokens=32)
        raw_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        return "".join(c for c in raw_text if c.isalnum()).upper()
