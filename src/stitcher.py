"""
Dual-Page Image Stitcher & Audit Watermarking Module.
Merges Booklet Front (Page 1: CIE) and Back (Page 2: Tests) side-by-side
with aspect-ratio normalization, visual divider, and audit header tags.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import cv2
import numpy as np


class DualPageStitcher:
    """
    Stitches Page 1 (Front) and Page 2 (Back) marksheet scans into a single
    audit-compliant composite image: `composite_<timestamp>_<uuid>.jpg`.
    Also supports splitting a composite image back into Page 1 and Page 2.
    """

    HEADER_RATIO: float = 0.035  # Top audit header bar height relative to page height
    DIVIDER_PX: int = 16         # Center vertical divider width in pixels

    def __init__(self, target_height: Optional[int] = None, jpeg_quality: int = 92):
        self.target_height = target_height
        self.jpeg_quality = jpeg_quality

    @staticmethod
    def load_image(source: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
        """Loads an image from a path, raw bytes, or numpy array into BGR format."""
        if isinstance(source, np.ndarray):
            return source.copy()
        if isinstance(source, (str, Path)):
            img = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not read image from path: {source}")
            return img
        if isinstance(source, (bytes, bytearray)):
            arr = np.frombuffer(source, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image from bytes buffer")
            return img
        raise TypeError(f"Unsupported image source type: {type(source)}")

    @staticmethod
    def _ensure_portrait(img: np.ndarray) -> np.ndarray:
        """Rotates landscape single-page phone captures to portrait orientation."""
        h, w = img.shape[:2]
        if w > h * 1.15:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img

    def normalize_pair(
        self,
        p1_img: np.ndarray,
        p2_img: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalizes Page 1 and Page 2 to matching portrait height while preserving
        aspect ratio for accurate side-by-side stitching.
        """
        p1 = self._ensure_portrait(p1_img)
        p2 = self._ensure_portrait(p2_img)

        h1, w1 = p1.shape[:2]
        h2, w2 = p2.shape[:2]

        target_h = self.target_height or max(h1, h2)

        if h1 != target_h:
            scale1 = target_h / float(h1)
            new_w1 = max(1, int(round(w1 * scale1)))
            p1 = cv2.resize(p1, (new_w1, target_h), interpolation=cv2.INTER_AREA if scale1 < 1 else cv2.INTER_CUBIC)

        if h2 != target_h:
            scale2 = target_h / float(h2)
            new_w2 = max(1, int(round(w2 * scale2)))
            p2 = cv2.resize(p2, (new_w2, target_h), interpolation=cv2.INTER_AREA if scale2 < 1 else cv2.INTER_CUBIC)

        return p1, p2

    def stitch_pages(
        self,
        p1_source: Union[str, Path, bytes, np.ndarray],
        p2_source: Union[str, Path, bytes, np.ndarray],
        booklet_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Stitches Page 1 and Page 2 side-by-side with an audit header bar,
        visual center divider, and 'PAGE 1: CIE' / 'PAGE 2: TESTS' tags.

        Returns:
            Dictionary with `composite_img`, `jpeg_bytes`, `filename`, `p1_img`, `p2_img`, and `metadata`.
        """
        raw_p1 = self.load_image(p1_source)
        raw_p2 = self.load_image(p2_source)

        p1_norm, p2_norm = self.normalize_pair(raw_p1, raw_p2)
        page_h, w1 = p1_norm.shape[:2]
        _, w2 = p2_norm.shape[:2]

        header_h = max(64, int(round(page_h * self.HEADER_RATIO)))
        divider_w = self.DIVIDER_PX
        total_w = w1 + divider_w + w2
        total_h = header_h + page_h

        # Create composite canvas (dark slate header + divider)
        composite = np.full((total_h, total_w, 3), (30, 41, 59), dtype=np.uint8)

        # Place Page 1 (left) and Page 2 (right)
        composite[header_h:header_h + page_h, 0:w1] = p1_norm
        composite[header_h:header_h + page_h, w1 + divider_w:w1 + divider_w + w2] = p2_norm

        # Draw vertical center divider line (accent blue/cyan)
        div_x_center = w1 + divider_w // 2
        cv2.line(
            composite,
            (div_x_center, 0),
            (div_x_center, total_h),
            (235, 99, 37),  # BGR for #2563EB (Royal Blue)
            thickness=max(4, divider_w // 2),
        )

        # Generate timestamp and unique ID
        now = datetime.now()
        ts_filename = now.strftime("%Y%m%d_%H%M%S")
        ts_display = now.strftime("%Y-%m-%d %H:%M:%S")
        short_uuid = uuid.uuid4().hex[:8]
        filename = f"composite_{ts_filename}_{short_uuid}.jpg"

        # Draw Header Tags ("PAGE 1: CIE" and "PAGE 2: TESTS")
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.75, header_h / 68.0)
        thickness = max(2, int(round(font_scale * 2.2)))

        # Left Badge: PAGE 1: CIE
        p1_tag = "PAGE 1: CIE"
        if booklet_id:
            p1_tag = f"PAGE 1: CIE  [{booklet_id}]"
        cv2.rectangle(
            composite,
            (18, int(header_h * 0.14)),
            (min(w1 - 18, int(w1 * 0.58)), int(header_h * 0.86)),
            (89, 130, 22),  # BGR Dark Emerald
            -1,
        )
        cv2.putText(
            composite,
            p1_tag,
            (36, int(header_h * 0.64)),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        # Right Badge: PAGE 2: TESTS
        p2_x_offset = w1 + divider_w
        p2_tag = f"PAGE 2: TESTS  |  AUDIT ID: {short_uuid.upper()}"
        cv2.rectangle(
            composite,
            (p2_x_offset + 18, int(header_h * 0.14)),
            (p2_x_offset + min(w2 - 18, int(w2 * 0.78)), int(header_h * 0.86)),
            (175, 64, 30),  # BGR Deep Royal Blue
            -1,
        )
        cv2.putText(
            composite,
            p2_tag,
            (p2_x_offset + 36, int(header_h * 0.64)),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        # Timestamp watermark on top-right
        ts_scale = font_scale * 0.65
        cv2.putText(
            composite,
            ts_display,
            (max(p2_x_offset + 20, total_w - int(w2 * 0.28)), int(header_h * 0.62)),
            font,
            ts_scale,
            (203, 213, 225),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )

        # Encode composite image to JPEG bytes
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        ok, buf = cv2.imencode(".jpg", composite, encode_params)
        if not ok:
            raise RuntimeError("Failed to encode composite image to JPEG")

        return {
            "composite_img": composite,
            "jpeg_bytes": buf.tobytes(),
            "filename": filename,
            "uuid": short_uuid,
            "timestamp": ts_display,
            "p1_img": raw_p1,
            "p2_img": raw_p2,
            "metadata": {
                "composite_width": total_w,
                "composite_height": total_h,
                "header_height": header_h,
                "divider_width": divider_w,
                "p1_shape": list(raw_p1.shape),
                "p2_shape": list(raw_p2.shape),
            },
        }

    def split_composite(
        self,
        composite_source: Union[str, Path, bytes, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits a pre-merged side-by-side composite image back into (Page 1, Page 2)
        so it can be processed by `TableSegmenter`.
        """
        comp = self.load_image(composite_source)
        total_h, total_w = comp.shape[:2]

        # Check if top audit header bar is present (dark slate strip at top)
        top_strip = comp[: max(10, int(total_h * 0.03)), :, :]
        mean_top_brightness = float(np.mean(top_strip))

        if mean_top_brightness < 85.0:
            # Estimate header height constructed with HEADER_RATIO = 0.035
            est_header_h = int(round(total_h * (self.HEADER_RATIO / (1.0 + self.HEADER_RATIO))))
            header_h = max(64, est_header_h)
        else:
            header_h = 0

        body = comp[header_h:, :, :]
        _, body_w = body.shape[:2]
        mid_x = body_w // 2
        half_div = self.DIVIDER_PX // 2

        p1_img = body[:, : max(1, mid_x - half_div)].copy()
        p2_img = body[:, min(body_w - 1, mid_x + half_div):].copy()

        return p1_img, p2_img
