import os
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage


# ============================================================
# 1. Global Settings
# ============================================================
# Colon vessel segmentation outputs are stored here.
SEG_DONE_ROOT = Path(r"C:\Users\User\Desktop\AbdVesselGen\Seg_Done")

# If the final pruned mask already exists, skip this step.
SKIP_IF_FINAL_EXISTS = True

# Keep only the largest connected component.
TOP_K_COMPONENTS = 1


# ============================================================
# 2. Path Builders
# ============================================================
def build_case_vein_dir(case_id: str) -> Path:
    """
    Example:
        C:/Users/User/Desktop/AbdVesselGen/Seg_Done/43_24055_veins_segmentation
    """
    return SEG_DONE_ROOT / f"{case_id}_veins_segmentation"


def build_input_mask_path(case_id: str) -> Path:
    """
    Prediction produced by inference.py.

    Example:
        C:/Users/User/Desktop/AbdVesselGen/Seg_Done/43_24055_veins_segmentation/43_24055_pred.nii.gz
    """
    return build_case_vein_dir(case_id) / f"{case_id}_pred.nii.gz"


def build_output_mask_path(case_id: str) -> Path:
    """
    Pruned prediction after retaining only the largest connected component.

    Example:
        C:/Users/User/Desktop/AbdVesselGen/Seg_Done/43_24055_veins_segmentation/43_24055_pred_top1_largest_components.nii.gz
    """
    return build_case_vein_dir(case_id) / f"{case_id}_pred_top1_largest_components.nii.gz"


# ============================================================
# 3. Core Pruning Function
# ============================================================
def keep_top_largest_connected_components(
    input_path: Path,
    output_path: Path,
    top_k: int = TOP_K_COMPONENTS,
) -> Path:
    """
    Keep the top-k largest 3D connected components from a binary prediction mask.
    Connectivity: 26-neighborhood in 3D.
    """
    print("=" * 100)
    print("[PRUNE COLON VESSEL PREDICTION]")
    print(f"[INPUT ] {input_path}")
    print(f"[OUTPUT] {output_path}")
    print(f"[TOP K ] {top_k}")
    print("=" * 100)

    if not input_path.exists():
        raise FileNotFoundError(f"Input prediction mask not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if SKIP_IF_FINAL_EXISTS and output_path.exists():
        print(f"[SKIP] Pruned prediction already exists: {output_path}")
        return output_path

    print("[INFO] Loading NIfTI...")
    nii = nib.load(str(input_path))
    data = nii.get_fdata()

    print(f"[INFO] Shape: {data.shape}")
    print(f"[INFO] Data range: min={data.min()}, max={data.max()}")

    # --------------------------------------------------------
    # Treat all voxels > 0 as positive foreground.
    # --------------------------------------------------------
    binary_mask = data > 0
    total_positive = int(binary_mask.sum())
    print(f"[INFO] Original positive voxels: {total_positive:,}")

    if total_positive == 0:
        print("[WARNING] Input mask contains no positive voxel.")
        output_data = np.zeros_like(data, dtype=np.uint8)

        output_header = nii.header.copy()
        output_header.set_data_dtype(np.uint8)
        out_nii = nib.Nifti1Image(
            output_data,
            affine=nii.affine,
            header=output_header,
        )
        nib.save(out_nii, str(output_path))

        print(f"[INFO] Empty output saved to: {output_path}")
        return output_path

    # --------------------------------------------------------
    # 3D 26-connectivity connected components.
    # --------------------------------------------------------
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled_mask, num_components = ndimage.label(binary_mask, structure=structure)

    print(f"[INFO] Number of connected components found: {num_components}")

    if num_components == 0:
        print("[WARNING] No connected component found after labeling.")
        output_data = np.zeros_like(data, dtype=np.uint8)

        output_header = nii.header.copy()
        output_header.set_data_dtype(np.uint8)
        out_nii = nib.Nifti1Image(
            output_data,
            affine=nii.affine,
            header=output_header,
        )
        nib.save(out_nii, str(output_path))
        print(f"[INFO] Empty output saved to: {output_path}")
        return output_path

    # --------------------------------------------------------
    # Compute the voxel size of each connected component.
    # Component label 0 is background and is excluded.
    # --------------------------------------------------------
    component_sizes = ndimage.sum(
        binary_mask,
        labeled_mask,
        index=np.arange(1, num_components + 1),
    )
    component_sizes = np.asarray(component_sizes, dtype=np.int64)

    sorted_component_indices = np.argsort(component_sizes)[::-1]
    top_k = max(1, min(int(top_k), num_components))
    top_component_labels = sorted_component_indices[:top_k] + 1

    print("\n[INFO] Connected components retained:")
    for rank, label_id in enumerate(top_component_labels, start=1):
        size = int(component_sizes[label_id - 1])
        print(f"  Rank {rank}: Component label = {label_id}, voxels = {size:,}")

    # --------------------------------------------------------
    # Build the final pruned binary mask.
    # --------------------------------------------------------
    output_mask = np.isin(labeled_mask, top_component_labels).astype(np.uint8)

    retained_voxels = int(output_mask.sum())
    removed_voxels = total_positive - retained_voxels
    print(f"\n[INFO] Retained positive voxels: {retained_voxels:,}")
    print(f"[INFO] Removed positive voxels: {removed_voxels:,}")

    # --------------------------------------------------------
    # Save NIfTI while preserving affine and header geometry.
    # --------------------------------------------------------
    output_header = nii.header.copy()
    output_header.set_data_dtype(np.uint8)

    out_nii = nib.Nifti1Image(
        output_mask,
        affine=nii.affine,
        header=output_header,
    )
    nib.save(out_nii, str(output_path))

    if not output_path.exists():
        raise RuntimeError(f"Pruned output was not created: {output_path}")

    print("\n[INFO] Pruned output saved to:")
    print(output_path)
    print("=" * 100)

    return output_path


# ============================================================
# 4. Command-Line Interface for run.py
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only the largest connected component from colon vessel prediction output."
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

    parser.add_argument(
        "--top_k",
        type=int,
        default=TOP_K_COMPONENTS,
        help="Number of largest connected components to retain. Default: 1.",
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

    input_path = build_input_mask_path(case_id)
    output_path = build_output_mask_path(case_id)

    keep_top_largest_connected_components(
        input_path=input_path,
        output_path=output_path,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
