import os
from pathlib import Path
from PIL import Image, ExifTags, ImageOps
import cv2
import numpy as np
from src.preprocessor import TableSegmenter

raw_imgs = [Path('data/raw/booklet_001_p1.jpg'), Path('data/raw/booklet_002_p1.jpg')]
sub1_imgs = list(Path('data/Sub1').glob('*.jpg'))[:2]
sub2_imgs = list(Path('data/Sub2').glob('*.jpg'))[:2]

samples = [('raw', p) for p in raw_imgs] + [('Sub1', p) for p in sub1_imgs] + [('Sub2', p) for p in sub2_imgs]

print("=== 1. IMAGE METADATA & EXIF ===")
for cat, p in samples:
    with Image.open(p) as img:
        w, h = img.size
        exif = img.getexif()
        orient = exif.get(0x0112, 'None')
        print(f"{cat:5s} | {p.name:30s} | Size: {w}x{h} | AR: {w/h:.3f} | EXIF Orientation: {orient}")

debug_dir = Path('data/crops/debug_sub1_sub2')
debug_dir.mkdir(parents=True, exist_ok=True)

segmenter = TableSegmenter()

print("\n=== 2. CURRENT TABLE SEGMENTER OVERLAYS ===")
for cat, p in samples:
    out_path = debug_dir / f"{cat}_{p.stem}_warped_overlay.jpg"
    res = segmenter.extract_page1_cells(p, debug_save_path=str(out_path))
    orig = cv2.imread(str(p))
    if orig is not None:
        ov = orig.copy()
        if res.get('usn_box'):
            ux, uy, uw, uh = res['usn_box']
            cv2.rectangle(ov, (ux, uy), (ux+uw, uy+uh), (255, 0, 0), 4)
            cv2.putText(ov, 'USN', (ux, uy-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        if res.get('course_code_box'):
            cx, cy, cw, ch = res['course_code_box']
            cv2.rectangle(ov, (cx, cy), (cx+cw, cy+ch), (0, 255, 255), 4)
            cv2.putText(ov, 'COURSE_CODE', (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # Save full overlay
        ov_resized = cv2.resize(ov, (1000, int(1000 * orig.shape[0] / orig.shape[1])))
        cv2.imwrite(str(debug_dir / f"{cat}_{p.stem}_orig_overlay.jpg"), ov_resized)
        
        # Save course code crop
        cc_crop = res.get('course_code_crop')
        if cc_crop is not None and cc_crop.size > 0:
            cv2.imwrite(str(debug_dir / f"{cat}_{p.stem}_cc_crop.jpg"), cc_crop)
            # Analyze color of course code crop
            b, g, r = cv2.split(cc_crop)
            print(f"  [{cat} {p.name}] CC Crop shape: {cc_crop.shape}, Mean BGR: B={b.mean():.1f}, G={g.mean():.1f}, R={r.mean():.1f}")
        
        # Save marks crops
        for k, c in res.get('p1_marks_crops', {}).items():
            if c is not None and c.size > 0:
                cv2.imwrite(str(debug_dir / f"{cat}_{p.stem}_{k}.jpg"), c)
                b, g, r = cv2.split(c)
                # Check ink color: find dark pixels
                dark_mask = (b < 150) & (g < 150) & (r < 150)
                if np.sum(dark_mask) > 10:
                    db = b[dark_mask].mean()
                    dg = g[dark_mask].mean()
                    dr = r[dark_mask].mean()
                    print(f"  [{cat} {p.name}] Cell {k:12s} Ink Dark Pixels: Mean B={db:.1f}, G={dg:.1f}, R={dr:.1f}")
                else:
                    print(f"  [{cat} {p.name}] Cell {k:12s} No dark ink found (mean BGR={b.mean():.1f}, {g.mean():.1f}, {r.mean():.1f})")

print("Diagnostics complete.")

