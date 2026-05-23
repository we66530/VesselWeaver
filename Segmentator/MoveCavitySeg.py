import argparse
import shutil
from pathlib import Path
from typing import List, Optional


# ============================================================
# 1. Fixed Paths
# ============================================================
# CavitySeg outputs are expected to be found somewhere under this root.
SEG_DONE_ROOT = Path(r"C:\Users\User\Desktop\AbdVesselGen\Seg_Done")


# ============================================================
# 2. Utility Functions
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_target_dir(case_id: str) -> Path:
    """
    Example:
        C:/Users/User/Desktop/AbdVesselGen/Seg_Done/43_24055_trunk_cavities
    """
    return SEG_DONE_ROOT / f"{case_id}_trunk_cavities"


def candidate_source_files(case_id: str) -> List[Path]:
    """
    Candidate files that may be produced by different versions of CavitySeg.py.

    This makes the move step robust to small naming differences, such as:
      - <case_id>_trunk_cavities_atlas.nii
      - <case_id>_trunk_cavities_atlas.nii.gz
      - <case_id>_trunk_cavities.nii
      - <case_id>_trunk_cavities.nii.gz
      - corresponding .nrrd copies
    """
    return [
        SEG_DONE_ROOT / f"{case_id}_trunk_cavities_atlas.nii",
        SEG_DONE_ROOT / f"{case_id}_trunk_cavities_atlas.nii.gz",
        SEG_DONE_ROOT / f"{case_id}_trunk_cavities_atlas.nrrd",
        SEG_DONE_ROOT / f"{case_id}_trunk_cavities.nii",
        SEG_DONE_ROOT / f"{case_id}_trunk_cavities.nii.gz",
        SEG_DONE_ROOT / f"{case_id}_trunk_cavities.nrrd",
    ]


def candidate_already_moved_files(case_id: str) -> List[Path]:
    """
    Files that may already exist inside the case-specific target folder.
    """
    target_dir = build_target_dir(case_id)
    return [
        target_dir / f"{case_id}_trunk_cavities_atlas.nii",
        target_dir / f"{case_id}_trunk_cavities_atlas.nii.gz",
        target_dir / f"{case_id}_trunk_cavities_atlas.nrrd",
        target_dir / f"{case_id}_trunk_cavities.nii",
        target_dir / f"{case_id}_trunk_cavities.nii.gz",
        target_dir / f"{case_id}_trunk_cavities.nrrd",
    ]


def move_one_file(source_path: Path, target_dir: Path) -> Path:
    """Move one file into target_dir and return its final path."""
    ensure_dir(target_dir)
    target_path = target_dir / source_path.name

    if target_path.exists():
        print(f"[SKIP] Target file already exists: {target_path}")
        print(f"[SKIP] Source file will not be moved: {source_path}")
        return target_path

    shutil.move(str(source_path), str(target_path))

    if not target_path.exists():
        raise RuntimeError(f"Move operation failed. Target file was not created: {target_path}")

    print(f"[DONE] Moved file: {target_path}")
    return target_path


def move_cavity_seg_outputs(case_id: str) -> List[Path]:
    """
    Move CavitySeg outputs from SEG_DONE_ROOT into:
        SEG_DONE_ROOT/<case_id>_trunk_cavities/

    The function can move multiple matching outputs, e.g. both .nii and .nrrd.
    """
    ensure_dir(SEG_DONE_ROOT)
    target_dir = build_target_dir(case_id)
    ensure_dir(target_dir)

    existing_sources = [path for path in candidate_source_files(case_id) if path.exists()]
    already_moved = [path for path in candidate_already_moved_files(case_id) if path.exists()]

    print("=" * 100)
    print("[MOVE CAVITY SEGMENTATION OUTPUTS]")
    print(f"[CASE ID] {case_id}")
    print(f"[ROOT   ] {SEG_DONE_ROOT}")
    print(f"[TARGET ] {target_dir}")
    print("=" * 100)

    if not existing_sources:
        if already_moved:
            print("[SKIP] No source files found in Seg_Done root, but cavity outputs already exist in the target folder.")
            for path in already_moved:
                print(f"[FOUND] {path}")
            return already_moved

        checked_text = "\n".join(str(path) for path in candidate_source_files(case_id))
        raise FileNotFoundError(
            "No movable CavitySeg output file was found. Checked:\n"
            f"{checked_text}"
        )

    moved_paths: List[Path] = []
    for source_path in existing_sources:
        print(f"[SOURCE] {source_path}")
        moved_paths.append(move_one_file(source_path, target_dir))

    print("=" * 100)
    print("==> Cavity segmentation outputs moved successfully.")
    print("=" * 100)

    return moved_paths


# ============================================================
# 3. Command-Line Interface for run.py
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Move CavitySeg output files into their case-specific trunk-cavities folder."
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

    move_cavity_seg_outputs(case_id)


if __name__ == "__main__":
    main()
