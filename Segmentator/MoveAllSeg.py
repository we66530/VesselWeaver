import os
import shutil
import argparse
from pathlib import Path


# ============================================================
# 1. Fixed Paths
# ============================================================
# AllSeg.py currently writes its segmentation mask here:
#   C:\Users\User\Desktop\AbdVesselGen\Seg_Done\<case_id>_All_segmentation.nii
#
# This script moves that file into:
#   C:\Users\User\Desktop\AbdVesselGen\Seg_Done\<case_id>_All_segmentation\<case_id>_All_segmentation.nii
SEG_DONE_ROOT = Path(r"C:\Users\User\Desktop\AbdVesselGen\Seg_Done")


# ============================================================
# 2. Utility Functions
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_source_mask(case_id: str) -> Path:
    """
    Locate the AllSeg output mask for this case.
    The primary expected file is .nii, but .nii.gz is also supported.
    """
    candidates = [
        SEG_DONE_ROOT / f"{case_id}_All_segmentation.nii",
        SEG_DONE_ROOT / f"{case_id}_All_segmentation.nii.gz",
    ]

    for path in candidates:
        if path.exists():
            return path

    candidate_text = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "AllSeg output mask was not found. Checked:\n"
        f"{candidate_text}"
    )


def move_allseg_mask(case_id: str) -> Path:
    """
    Move the AllSeg mask from the Seg_Done root folder into its own case-specific folder.
    Returns the final moved file path.
    """
    source_mask = resolve_source_mask(case_id)

    target_dir = SEG_DONE_ROOT / f"{case_id}_All_segmentation"
    ensure_dir(target_dir)

    target_mask = target_dir / source_mask.name

    print("=" * 100)
    print("[MOVE ALLSEG MASK]")
    print(f"[CASE ID] {case_id}")
    print(f"[SOURCE ] {source_mask}")
    print(f"[TARGET ] {target_mask}")
    print("=" * 100)

    if target_mask.exists():
        print(f"[SKIP] Target file already exists: {target_mask}")
        print("[SKIP] Source file will not be moved.")
        return target_mask

    shutil.move(str(source_mask), str(target_mask))

    if not target_mask.exists():
        raise RuntimeError(f"Move operation failed. Target file was not created: {target_mask}")

    print(f"[DONE] AllSeg mask moved successfully: {target_mask}")
    return target_mask


# ============================================================
# 3. Command-Line Interface for run.py
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Move the AllSeg output mask into its case-specific segmentation folder."
    )

    # Kept for compatibility with run.py, although this script does not use the CT path directly.
    parser.add_argument(
        "--ct_path",
        required=True,
        help="CT path passed by run.py. Accepted for pipeline compatibility.",
    )

    # Kept for compatibility with run.py, although this script uses SEG_DONE_ROOT as its actual root.
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory passed by run.py. Accepted for pipeline compatibility.",
    )

    parser.add_argument(
        "--case_id",
        required=True,
        help="Case ID passed by run.py.",
    )

    return parser.parse_args()


# ============================================================
# 4. Main Entry Point
# ============================================================
def main():
    args = parse_args()
    case_id = args.case_id.strip()

    if not case_id:
        raise ValueError("case_id is empty.")

    ensure_dir(SEG_DONE_ROOT)
    move_allseg_mask(case_id)


if __name__ == "__main__":
    main()
