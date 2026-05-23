import os
import argparse
from pathlib import Path


import torch
from totalsegmentator.python_api import totalsegmentator


# ([github.com](https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/nnunet.py))================================
# 1. Global Settings
# ============================================================
# AllSeg outputs are always written here, regardless of where the input CT is located.
SEG_DONE_ROOT = Path(r"C:\Users\User\Desktop\AbdVesselGen\Seg_Done")

# If the final multilabel atlas already exists, skip this case.
SKIP_IF_FINAL_EXISTS = True

# TotalSegmentator 'total' task setting.
TOTAL_SEG_FAST = False


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


def build_output_mask_path(case_id: str) -> Path:
    """
    Build the exact multilabel atlas output path expected by the next pipeline step.

    Example:
        C:/Users/User/Desktop/AbdVesselGen/Seg_Done/43_24055_All_segmentation.nii
    """
    return SEG_DONE_ROOT / f"{case_id}_All_segmentation.nii"


# ============================================================
# 3. Single-Case Processing
# ============================================================
def process_one_case(input_path: Path, case_id: str, device: str) -> Path:
    """
    Run TotalSegmentator for one CT only.

    Important:
    With ml=True, TotalSegmentator saves the multilabel atlas directly to the
    exact output file path provided in `output=...`.
    Therefore, this script does NOT search for class_map.nii.gz afterward.
    """
    input_path = input_path.resolve()
    ensure_dir(SEG_DONE_ROOT)

    final_mask_path = build_output_mask_path(case_id)

    if SKIP_IF_FINAL_EXISTS and final_mask_path.exists():
        print("=" * 100)
        print("[SKIP] AllSeg multilabel atlas already exists.")
        print(f"[CASE ID] {case_id}")
        print(f"[OUTPUT ] {final_mask_path}")
        print("=" * 100)
        return final_mask_path

    print("\n" + "=" * 100)
    print("[ALL SEGMENTATION]")
    print(f"[CASE ID] {case_id}")
    print(f"[INPUT  ] {input_path}")
    print(f"[OUTPUT ] {final_mask_path}")
    print(f"[DEVICE ] {device}")
    print("=" * 100)

    # ----------------------------------------------------
    # A. Run TotalSegmentator total task
    # ----------------------------------------------------
    print(f"==> Starting TotalSegmentator 'total' task on {device}...")

    totalsegmentator(
        input=str(input_path),
        output=str(final_mask_path),
        task="total",
        ml=True,
        fast=TOTAL_SEG_FAST,
        device=device,
    )

    # ----------------------------------------------------
    # B. Verify that the expected multilabel atlas was created
    # ----------------------------------------------------
    if not final_mask_path.exists():
        raise FileNotFoundError(
            "TotalSegmentator finished, but the expected multilabel atlas was not found:\n"
            f"{final_mask_path}"
        )

    file_size_mb = final_mask_path.stat().st_size / (1024 ** 2)
    print(f"==> Saved AllSeg multilabel atlas: {final_mask_path}")
    print(f"==> File size: {file_size_mb:.2f} MB")

    print("\n" + "=" * 100)
    print("==> Single-case total segmentation completed successfully.")
    print("=" * 100)

    return final_mask_path


# ============================================================
# 4. Command-Line Interface for run.py
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TotalSegmentator total-task multilabel atlas generation for one CT NIfTI file."
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
def main():
    args = parse_args()

    ct_path = Path(args.ct_path).resolve()

    if not ct_path.exists():
        raise FileNotFoundError(f"CT file not found: {ct_path}")

    lower = ct_path.name.lower()
    if not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
        raise ValueError(f"Input file must be .nii or .nii.gz: {ct_path}")

    case_id = args.case_id.strip() if args.case_id else strip_nii_extension(ct_path.name)
    if not case_id:
        raise ValueError("case_id is empty.")

    device = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"==> Using device: {device}")

    process_one_case(
        input_path=ct_path,
        case_id=case_id,
        device=device,
    )


if __name__ == "__main__":
    main()
