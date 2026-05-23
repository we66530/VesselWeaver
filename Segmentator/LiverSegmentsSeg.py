import os
import json
import argparse
import pathlib
from pathlib import Path

import torch
from totalsegmentator.python_api import totalsegmentator


# ============================================================
# 1. Global Settings
# ============================================================
# Final LiverSegmentsSeg output is written here first,
# regardless of where the source CT is located.
SEG_DONE_ROOT = Path(r"C:\Users\User\Desktop\AbdVesselGen\Seg_Done")

# If the expected final NIfTI output already exists, skip this step.
SKIP_IF_FINAL_EXISTS = True

# TotalSegmentator task used by the original script.
# Note: this task is named liver_segments_mr in TotalSegmentator.
TOTALSEG_TASK = "liver_segments_mr"

# Use multilabel output directly.
# With ml=True, TotalSegmentator writes one multilabel image directly to output path.
USE_MULTILABEL_OUTPUT = True

# License configuration used by your existing workflow.
LICENSE_CODE = "aca_E27OH4P3RQKWRI"


# ============================================================
# 2. Utility Functions
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def strip_nii_extension(filename: str) -> str:
    """Remove .nii or .nii.gz extension."""
    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        return filename[:-7]
    if lower.endswith(".nii"):
        return filename[:-4]
    return os.path.splitext(filename)[0]


def setup_license_manual() -> None:
    """Ensure that the TotalSegmentator license file exists."""
    config_dir = pathlib.Path.home() / ".totalsegmentator"
    config_file = config_dir / "config.json"
    config_dir.mkdir(parents=True, exist_ok=True)

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump({"license_number": LICENSE_CODE}, f)

    print("==> License configuration confirmed.")


def build_final_nifti_path(case_id: str) -> Path:
    """
    Example:
        C:/Users/User/Desktop/AbdVesselGen/Seg_Done/43_24055_liver_segments.nii
    """
    return SEG_DONE_ROOT / f"{case_id}_liver_segments.nii"


# ============================================================
# 3. Single-Case Liver Segments Segmentation
# ============================================================
def run_liver_segments_segmentation(
    ct_path: Path,
    case_id: str,
    device: str,
) -> Path:
    """
    Run TotalSegmentator liver_segments_mr for one CT/NIfTI file only.

    This pipeline version outputs only one .nii file to Seg_Done:
        <case_id>_liver_segments.nii

    The original script manually merged individual liver_segment_*.nii.gz files
    into an NRRD. Here, ml=True is used so TotalSegmentator directly writes one
    multilabel NIfTI file.
    """
    ensure_dir(SEG_DONE_ROOT)

    final_nii = build_final_nifti_path(case_id)

    if SKIP_IF_FINAL_EXISTS and final_nii.exists():
        print("=" * 100)
        print("[SKIP] Liver segments atlas already exists.")
        print(f"[CASE ID] {case_id}")
        print(f"[NIfTI  ] {final_nii}")
        print("=" * 100)
        return final_nii

    print("\n" + "=" * 100)
    print("[LIVER SEGMENTS SEGMENTATION]")
    print(f"[CASE ID] {case_id}")
    print(f"[INPUT  ] {ct_path}")
    print(f"[OUTPUT ] {final_nii}")
    print(f"[TASK   ] {TOTALSEG_TASK}")
    print(f"[DEVICE ] {device}")
    print("=" * 100)

    print(f"==> Starting TotalSegmentator '{TOTALSEG_TASK}' task on {device}...")
    print("==> Note: liver_segments_mr may be optimized for MR, but this follows the original script.")

    totalsegmentator(
        input=str(ct_path),
        output=str(final_nii),
        task=TOTALSEG_TASK,
        ml=USE_MULTILABEL_OUTPUT,
        device=device,
    )

    if not final_nii.exists():
        raise FileNotFoundError(
            "TotalSegmentator finished, but the expected liver-segments atlas was not found:\n"
            f"{final_nii}"
        )

    file_size_mb = final_nii.stat().st_size / (1024 ** 2)
    print(f"==> Saved liver segments NIfTI atlas: {final_nii}")
    print(f"==> File size: {file_size_mb:.2f} MB")
    print("==> Label IDs follow TotalSegmentator liver_segments_mr multilabel output.")
    print("\n" + "=" * 100)
    print("==> Single-case liver segments segmentation completed successfully.")
    print("=" * 100)

    return final_nii


# ============================================================
# 4. Command-Line Interface for run.py
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TotalSegmentator liver_segments_mr segmentation for one NIfTI file."
    )

    parser.add_argument(
        "--ct_path",
        required=True,
        help="Path to the single CT .nii or .nii.gz file loaded by run.py.",
    )

    # Kept for compatibility with run.py.
    # This script writes to SEG_DONE_ROOT instead of using this argument.
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory passed by run.py. Accepted for pipeline compatibility.",
    )

    parser.add_argument(
        "--case_id",
        required=False,
        default=None,
        help="Case ID provided by run.py. If omitted, it will be inferred from the CT filename.",
    )

    return parser.parse_args()


# ============================================================
# 5. Main Entry Point
# ============================================================
def main() -> None:
    args = parse_args()

    ct_path = Path(args.ct_path).resolve()
    if not ct_path.exists():
        raise FileNotFoundError(f"Input NIfTI file not found: {ct_path}")

    lower = ct_path.name.lower()
    if not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
        raise ValueError(f"Input file must be .nii or .nii.gz: {ct_path}")

    case_id = args.case_id.strip() if args.case_id else strip_nii_extension(ct_path.name)
    if not case_id:
        raise ValueError("case_id is empty.")

    setup_license_manual()
    device = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"==> Using device: {device}")

    run_liver_segments_segmentation(
        ct_path=ct_path,
        case_id=case_id,
        device=device,
    )


if __name__ == "__main__":
    main()
