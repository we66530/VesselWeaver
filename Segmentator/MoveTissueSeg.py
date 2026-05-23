import argparse
import shutil
from pathlib import Path
from typing import List


# ============================================================
# 1. Fixed Paths
# ============================================================
# TissueSeg outputs are expected to be created under this folder first.
SEG_DONE_ROOT = Path(r"C:\Users\User\Desktop\AbdVesselGen\Seg_Done")


# ============================================================
# 2. Utility Functions
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_target_dir(case_id: str) -> Path:
    """
    Example:
        C:/Users/User/Desktop/AbdVesselGen/Seg_Done/43_24055_tissue_4_types
    """
    return SEG_DONE_ROOT / f"{case_id}_tissue_4_types"


def build_source_file(case_id: str) -> Path:
    """
    TissueSeg.py is expected to create this root-level NIfTI file first.

    Example:
        C:/Users/User/Desktop/AbdVesselGen/Seg_Done/43_24055_tissue_4_types.nii
    """
    return SEG_DONE_ROOT / f"{case_id}_tissue_4_types.nii"


def build_target_file(case_id: str) -> Path:
    """
    Final moved TissueSeg file location.

    Example:
        C:/Users/User/Desktop/AbdVesselGen/Seg_Done/43_24055_tissue_4_types/43_24055_tissue_4_types.nii
    """
    target_dir = build_target_dir(case_id)
    return target_dir / f"{case_id}_tissue_4_types.nii"


def move_tissue_seg_output(case_id: str) -> Path:
    """
    Move the TissueSeg NIfTI output from the Seg_Done root folder into its
    case-specific tissue folder.
    """
    ensure_dir(SEG_DONE_ROOT)

    source_file = build_source_file(case_id)
    target_dir = build_target_dir(case_id)
    target_file = build_target_file(case_id)

    ensure_dir(target_dir)

    print("=" * 100)
    print("[MOVE TISSUE SEGMENTATION OUTPUT]")
    print(f"[CASE ID] {case_id}")
    print(f"[SOURCE ] {source_file}")
    print(f"[TARGET ] {target_file}")
    print("=" * 100)

    # If the file has already been moved, this step is complete.
    if target_file.exists():
        print(f"[SKIP] Target file already exists: {target_file}")
        print("[SKIP] TissueSeg output appears to have already been moved.")
        return target_file

    # If the source file is not present and the target also does not exist,
    # we should stop because there is nothing valid to move.
    if not source_file.exists():
        raise FileNotFoundError(
            "TissueSeg output file was not found. Checked:\n"
            f"{source_file}\n"
            "The target file also does not exist, so the move step cannot proceed."
        )

    shutil.move(str(source_file), str(target_file))

    if not target_file.exists():
        raise RuntimeError(
            "Move operation failed. Target file was not created:\n"
            f"{target_file}"
        )

    print(f"[DONE] TissueSeg output moved successfully: {target_file}")
    print("=" * 100)
    print("==> Tissue segmentation output move completed successfully.")
    print("=" * 100)

    return target_file


# ============================================================
# 3. Command-Line Interface for run.py
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Move TissueSeg output into its case-specific tissue folder."
    )

    # Kept for compatibility with run.py.
    parser.add_argument(
        "--ct_path",
        required=True,
        help="CT path passed by run.py. Accepted for pipeline compatibility.",
    )

    # Kept for compatibility with run.py.
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
def main() -> None:
    args = parse_args()
    case_id = args.case_id.strip()

    if not case_id:
        raise ValueError("case_id is empty.")

    move_tissue_seg_output(case_id)


if __name__ == "__main__":
    main()
