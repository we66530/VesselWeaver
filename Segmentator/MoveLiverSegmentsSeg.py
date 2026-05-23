import argparse
import shutil
from pathlib import Path


# ============================================================
# 1. Fixed Paths
# ============================================================
# LiverSegmentsSeg.py first writes its output here:
#   C:\Users\User\Desktop\AbdVesselGen\Seg_Done\<case_id>_liver_segments.nii
#
# This script moves it into:
#   C:\Users\User\Desktop\AbdVesselGen\Seg_Done\<case_id>_liver_segments\<case_id>_liver_segments.nii
SEG_DONE_ROOT = Path(r"C:\Users\User\Desktop\AbdVesselGen\Seg_Done")


# ============================================================
# 2. Utility Functions
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_source_file(case_id: str) -> Path:
    """
    Root-level output produced by LiverSegmentsSeg.py.

    Example:
        C:/Users/User/Desktop/AbdVesselGen/Seg_Done/43_24055_liver_segments.nii
    """
    return SEG_DONE_ROOT / f"{case_id}_liver_segments.nii"


def build_target_dir(case_id: str) -> Path:
    """
    Case-specific liver segments folder.

    Example:
        C:/Users/User/Desktop/AbdVesselGen/Seg_Done/43_24055_liver_segments
    """
    return SEG_DONE_ROOT / f"{case_id}_liver_segments"


def build_target_file(case_id: str) -> Path:
    """
    Final moved liver segments NIfTI path.

    Example:
        C:/Users/User/Desktop/AbdVesselGen/Seg_Done/43_24055_liver_segments/43_24055_liver_segments.nii
    """
    return build_target_dir(case_id) / f"{case_id}_liver_segments.nii"


# ============================================================
# 3. Main Move Function
# ============================================================
def move_liver_segments_output(case_id: str) -> Path:
    """
    Move the LiverSegmentsSeg NIfTI output from the Seg_Done root folder into
    its case-specific liver-segments folder.
    """
    ensure_dir(SEG_DONE_ROOT)

    source_file = build_source_file(case_id)
    target_dir = build_target_dir(case_id)
    target_file = build_target_file(case_id)

    ensure_dir(target_dir)

    print("=" * 100)
    print("[MOVE LIVER SEGMENTS OUTPUT]")
    print(f"[CASE ID] {case_id}")
    print(f"[SOURCE ] {source_file}")
    print(f"[TARGET ] {target_file}")
    print("=" * 100)

    # If the file has already been moved, this step is complete.
    if target_file.exists():
        print(f"[SKIP] Target file already exists: {target_file}")
        print("[SKIP] LiverSegmentsSeg output appears to have already been moved.")
        return target_file

    # If the source file is not present and target does not exist, stop clearly.
    if not source_file.exists():
        raise FileNotFoundError(
            "LiverSegmentsSeg output file was not found. Checked:\n"
            f"{source_file}\n"
            "The target file also does not exist, so the move step cannot proceed."
        )

    shutil.move(str(source_file), str(target_file))

    if not target_file.exists():
        raise RuntimeError(
            "Move operation failed. Target file was not created:\n"
            f"{target_file}"
        )

    print(f"[DONE] LiverSegmentsSeg output moved successfully: {target_file}")
    print("=" * 100)
    print("==> Liver segments output move completed successfully.")
    print("=" * 100)

    return target_file


# ============================================================
# 4. Command-Line Interface for run.py
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Move LiverSegmentsSeg output into its case-specific liver-segments folder."
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
# 5. Main Entry Point
# ============================================================
def main() -> None:
    args = parse_args()
    case_id = args.case_id.strip()

    if not case_id:
        raise ValueError("case_id is empty.")

    move_liver_segments_output(case_id)


if __name__ == "__main__":
    main()
