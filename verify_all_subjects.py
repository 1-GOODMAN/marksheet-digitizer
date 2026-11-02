import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import BatchProcessor

def main():
    processor = BatchProcessor()

    raw_files = list(Path("data/raw").glob("*p1.jpg"))[:3]
    sub1_files = list(Path("data/Sub1").glob("*.jpg"))[:3]
    sub2_files = list(Path("data/Sub2").glob("*.jpg"))[:3]

    tasks = [
        ("raw", f) for f in raw_files
    ] + [
        ("Sub1", f) for f in sub1_files
    ] + [
        ("Sub2", f) for f in sub2_files
    ]

    print(f"Running pipeline verification on {len(tasks)} images...\n")
    results = []

    for folder, file_path in tasks:
        print(f"Processing [{folder}] {file_path.name}...")
        res = processor.process_page1_image(
            img_input=file_path,
            booklet_id=file_path.stem,
            subject_folder=folder,
        )
        if not res.get("success", False):
            print(f"  [ERROR] {res.get('error')}")
            continue

        cie = res["cie_summary"]
        results.append({
            "filename": file_path.name,
            "folder": folder,
            "course_code": cie.get("Course_Code", "UNKNOWN"),
            "usn": cie.get("USN", "UNKNOWN"),
            "t1": cie.get("Test1_15", 0),
            "t2": cie.get("Test2_15", 0),
            "aba": cie.get("ABA_10", 0),
            "theory": cie.get("Theory_Total_40", 0),
            "viva": cie.get("Viva_5", 0),
            "lab": cie.get("Lab_5", 0),
            "pract": cie.get("Practical_Total_10", 0),
            "total_cie": cie.get("Total_CIE_50", 0),
            "status": cie.get("Status", "VALIDATED"),
            "overrides": res.get("overrides", []),
        })

    # Print markdown table
    print("\n" + "=" * 120)
    print("FINAL SIDE-BY-SIDE VERIFICATION TABLE")
    print("=" * 120)
    hdr = f"{'Filename':30s} | {'Source':6s} | {'Course Code':11s} | {'USN':10s} | {'T1':3s} | {'T2':3s} | {'ABA':3s} | {'Theory':6s} | {'Viva':4s} | {'Lab':3s} | {'Pract':5s} | {'Total CIE':9s} | {'Status'}"
    print(hdr)
    print("-" * 120)

    for r in results:
        row = (
            f"{r['filename']:30s} | {r['folder']:6s} | {r['course_code']:11s} | {r['usn']:10s} | "
            f"{r['t1']:3d} | {r['t2']:3d} | {r['aba']:3d} | {r['theory']:6d} | "
            f"{r['viva']:4d} | {r['lab']:3d} | {r['pract']:5d} | {r['total_cie']:9d} | {r['status']}"
        )
        print(row)
        if r['overrides']:
            print(f"   -> Overrides: {r['overrides']}")

    print("=" * 120)

if __name__ == "__main__":
    main()

