import argparse
from pathlib import Path
from typing import Iterable, Tuple

import nibabel as nib
import numpy as np


# ============================================================
# 1. Global Settings
# ============================================================
BASE_DIR = Path(r"C:\Users\User\Desktop\AbdVesselGen")
SEG_DONE_ROOT = BASE_DIR / "Seg_Done"
ROI_CT_ROOT = BASE_DIR / "ROI_CT"

# Output filename will be:
#   <case_id>_ROI_CT.nii.gz
OUTPUT_EXTENSION = ".nii.gz"

# Outside-ROI CT value.
OUT_OF_FRAME_HU = -1024.0

# Trunk cavities label to keep.
# Based on your previous logic: keep only Trunk Segment 1.
TRUNK_LABEL_TO_KEEP = 1

# AllSeg labels to keep inside the trunk ROI.
# Your original comment said: keep TotalSeg background 0 or label 64.
# Therefore all other non-zero labels are excluded.
ALLSEG_LABELS_TO_KEEP = {0}

# Tissue labels: exclude all non-zero voxels.
EXCLUDE_ALL_TISSUE_LABELS = True

# If final ROI CT already exists, skip this step.
SKIP_IF_FINAL_EXISTS = True


# ============================================================
# 2. Path Builders
# ============================================================
def build_allseg_path(case_id: str) -> Path:
    return SEG_DONE_ROOT / f"{case_id}_All_segmentation" / f"{case_id}_All_segmentation.nii"


def build_tissue_path(case_id: str) -> Path:
    return SEG_DONE_ROOT / f"{case_id}_tissue_4_types" / f"{case_id}_tissue_4_types.nii"


def build_trunk_cavity_path(case_id: str) -> Path:
    return SEG_DONE_ROOT / f"{case_id}_trunk_cavities" / f"{case_id}_trunk_cavities_atlas.nii"


def build_output_roi_ct_path(case_id: str) -> Path:
    return ROI_CT_ROOT / f"{case_id}_ROI_CT{OUTPUT_EXTENSION}"


# ============================================================
# 3. Utility Functions
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def strip_nii_extension(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        return filename[:-7]
    if lower.endswith(".nii"):
        return filename[:-4]
    return Path(filename).stem


def load_nifti(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found:\n{path}")
    img = nib.load(str(path))
    data = img.get_fdata()
    return img, data


def check_same_geometry(
    reference_name: str,
    reference_img,
    reference_data: np.ndarray,
    target_name: str,
    target_img,
    target_data: np.ndarray,
) -> None:
    """
    Shape mismatch is fatal. Affine mismatch is warned but not fatal,
    because some previous outputs may preserve geometry with minor header differences.
    """
    if reference_data.shape != target_data.shape:
        raise ValueError(
            f"{target_name} shape does not match {reference_name}.\n"
            f"{reference_name} shape: {reference_data.shape}\n"
            f"{target_name} shape:    {target_data.shape}\n"
            "All masks must be aligned voxel-by-voxel before ROI CT generation."
        )

    if not np.allclose(reference_img.affine, target_img.affine, atol=1e-4):
        print(f"[WARNING] {target_name} affine does not exactly match {reference_name} affine.")
        print("[WARNING] This script will continue because voxel shapes match.")
        print("[WARNING] Please confirm spatial alignment in 3D Slicer if needed.\n")


def isin_labels(data: np.ndarray, labels: Iterable[int]) -> np.ndarray:
    labels = list(labels)
    return np.isin(data.astype(np.int64), labels)


# ============================================================
# 4. ROI CT Generation
# ============================================================
def generate_roi_ct(
    ct_path: Path,
    allseg_path: Path,
    tissue_path: Path,
    trunk_path: Path,
    output_path: Path,
) -> Path:
    print("=" * 100)
    print("[GENERATE ROI CT]")
    print(f"[CT     ] {ct_path}")
    print(f"[ALLSEG ] {allseg_path}")
    print(f"[TISSUE ] {tissue_path}")
    print(f"[TRUNK  ] {trunk_path}")
    print(f"[OUTPUT ] {output_path}")
    print("=" * 100)

    ensure_dir(output_path.parent)

    if SKIP_IF_FINAL_EXISTS and output_path.exists():
        print(f"[SKIP] ROI CT already exists: {output_path}")
        return output_path

    # --------------------------------------------------------
    # A. Load CT and masks
    # --------------------------------------------------------
    print("[INFO] Loading CT and masks...")
    ct_img, ct_data = load_nifti(ct_path)
    allseg_img, allseg_data = load_nifti(allseg_path)
    tissue_img, tissue_data = load_nifti(tissue_path)
    trunk_img, trunk_data = load_nifti(trunk_path)

    # --------------------------------------------------------
    # B. Geometry checks
    # --------------------------------------------------------
    print("\n[INFO] Geometry check:")
    print(f"CT shape:      {ct_data.shape}")
    print(f"AllSeg shape:  {allseg_data.shape}")
    print(f"Tissue shape:  {tissue_data.shape}")
    print(f"Trunk shape:   {trunk_data.shape}")

    check_same_geometry("CT", ct_img, ct_data, "AllSeg", allseg_img, allseg_data)
    check_same_geometry("CT", ct_img, ct_data, "Tissue", tissue_img, tissue_data)
    check_same_geometry("CT", ct_img, ct_data, "Trunk", trunk_img, trunk_data)

    # --------------------------------------------------------
    # C. Build ROI logic
    # --------------------------------------------------------
    print("\n[INFO] Building ROI mask...")

    # Condition A: keep only trunk cavity label 1.
    is_in_trunk = trunk_data.astype(np.int64) == TRUNK_LABEL_TO_KEEP

    # Condition B: exclude AllSeg labels except labels listed in ALLSEG_LABELS_TO_KEEP.
    # Default: keep AllSeg background 0 and label 64; exclude everything else.
    allseg_keep = isin_labels(allseg_data, ALLSEG_LABELS_TO_KEEP)
    to_exclude_allseg = ~allseg_keep

    # Condition C: exclude all non-zero tissue labels.
    if EXCLUDE_ALL_TISSUE_LABELS:
        to_exclude_tissue = tissue_data != 0
    else:
        to_exclude_tissue = np.zeros_like(tissue_data, dtype=bool)

    keep_mask = (
        is_in_trunk
        & (~to_exclude_allseg)
        & (~to_exclude_tissue)
    )

    # --------------------------------------------------------
    # D. Create ROI CT
    # --------------------------------------------------------
    roi_data = np.full(ct_data.shape, OUT_OF_FRAME_HU, dtype=np.float32)
    roi_data[keep_mask] = ct_data[keep_mask].astype(np.float32)

    # --------------------------------------------------------
    # E. Statistics
    # --------------------------------------------------------
    total_voxels = int(ct_data.size)
    trunk_voxels = int(np.count_nonzero(is_in_trunk))
    excluded_by_allseg_inside_trunk = int(np.count_nonzero(to_exclude_allseg & is_in_trunk))
    excluded_by_tissue_inside_trunk = int(np.count_nonzero(to_exclude_tissue & is_in_trunk))
    kept_voxels = int(np.count_nonzero(keep_mask))

    print("\n" + "=" * 100)
    print("[ROI CT STATISTICS]")
    print("=" * 100)
    print(f"Total CT voxels:                         {total_voxels:,}")
    print(f"Voxels in trunk label {TRUNK_LABEL_TO_KEEP}:               {trunk_voxels:,}")
    print(f"Excluded by AllSeg rule inside trunk:    {excluded_by_allseg_inside_trunk:,}")
    print(f"Excluded by tissue rule inside trunk:    {excluded_by_tissue_inside_trunk:,}")
    print(f"Final kept voxels:                       {kept_voxels:,}")
    print("=" * 100)

    if kept_voxels == 0:
        print("[WARNING] Final ROI contains zero voxels. Please check label IDs and geometry.")

    # --------------------------------------------------------
    # F. Save ROI CT
    # --------------------------------------------------------
    print(f"\n[INFO] Saving ROI CT to:\n{output_path}")

    output_header = ct_img.header.copy()
    output_header.set_data_dtype(np.float32)

    out_img = nib.Nifti1Image(
        roi_data.astype(np.float32),
        affine=ct_img.affine,
        header=output_header,
    )
    nib.save(out_img, str(output_path))

    if not output_path.exists():
        raise RuntimeError(f"ROI CT output was not created: {output_path}")

    print("[DONE] ROI CT generation completed.")
    print("=" * 100)
    return output_path


# ============================================================
# 5. Command-Line Interface for run.py
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ROI CT using AllSeg, tissue_4_types, and trunk_cavities masks."
    )

    parser.add_argument(
        "--ct_path",
        required=True,
        help="CT path passed by run.py. This is the raw CT to crop into ROI CT.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory passed by run.py. Accepted for pipeline compatibility.",
    )

    parser.add_argument(
        "--case_id",
        required=False,
        default=None,
        help="Case ID passed by run.py. If omitted, inferred from CT filename.",
    )

    return parser.parse_args()


# ============================================================
# 6. Main Entry Point
# ============================================================
def main() -> None:
    args = parse_args()

    ct_path = Path(args.ct_path).resolve()
    if not ct_path.exists():
        raise FileNotFoundError(f"CT file not found: {ct_path}")

    case_id = args.case_id.strip() if args.case_id else strip_nii_extension(ct_path.name)
    if not case_id:
        raise ValueError("case_id is empty.")

    allseg_path = build_allseg_path(case_id)
    tissue_path = build_tissue_path(case_id)
    trunk_path = build_trunk_cavity_path(case_id)
    output_path = build_output_roi_ct_path(case_id)

    generate_roi_ct(
        ct_path=ct_path,
        allseg_path=allseg_path,
        tissue_path=tissue_path,
        trunk_path=trunk_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
