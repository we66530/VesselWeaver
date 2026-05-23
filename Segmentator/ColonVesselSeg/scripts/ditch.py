import argparse
from pathlib import Path
from typing import List

import numpy as np
import nibabel as nib
from scipy import ndimage


# ============================================================
# 1. Global Settings
# ============================================================
SEG_DONE_ROOT = Path(r"C:\Users\User\Desktop\AbdVesselGen\Seg_Done")

# Remove skeleton voxels overlapping CT HU lower than this threshold.
# This version uses HU < 30 by default.
HU_THRESHOLD = 30

# Final skeleton tube target width.
# In voxel morphology, an exact even-number diameter is discretized approximately.
# radius=3 gives an approximately 6-voxel-wide vessel tube around the skeleton core.
TARGET_WIDTH_VOXELS = 4
DILATION_RADIUS = int(np.ceil(TARGET_WIDTH_VOXELS / 2.0))

# If final output already exists, skip this step.
SKIP_IF_FINAL_EXISTS = True


# ============================================================
# 2. Path Builders
# ============================================================
def build_case_vein_dir(case_id: str) -> Path:
    return SEG_DONE_ROOT / f"{case_id}_veins_segmentation"


def build_candidate_skeleton_paths(case_id: str) -> List[Path]:
    """
    Support both possible skeleton filenames:
      1. <case_id>_pred_skeleton.nii.gz
      2. <case_id>_pred_skeleton_.nii.gz

    The second filename matches the earlier skeleton.py version.
    """
    case_dir = build_case_vein_dir(case_id)
    return [
        case_dir / f"{case_id}_pred_skeleton.nii.gz",
        case_dir / f"{case_id}_pred_skeleton_.nii.gz",
    ]


def resolve_input_skeleton_path(case_id: str) -> Path:
    candidates = build_candidate_skeleton_paths(case_id)
    for path in candidates:
        if path.exists():
            return path

    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Skeleton input mask was not found. Checked:\n"
        f"{checked}"
    )


def build_output_path(case_id: str) -> Path:
    """
    Final output after:
      1. removing skeleton voxels overlapping CT HU < HU_THRESHOLD
      2. retaining only the largest connected component
      3. expanding the skeleton to approximately 6-voxel width
    """
    return build_case_vein_dir(case_id) / f"{case_id}_pred_skeleton_HUge{HU_THRESHOLD}_top1_width{TARGET_WIDTH_VOXELS}.nii.gz"


# ============================================================
# 3. Utility Functions
# ============================================================
def load_nifti(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found:\n{path}")

    img = nib.load(str(path))
    data = img.get_fdata()
    return img, data


def check_same_geometry(mask_img, ct_img, mask_data, ct_data) -> None:
    """
    This script assumes mask and CT are already voxel-wise aligned.
    Shape mismatch is fatal. Affine mismatch is warned but not fatal.
    """
    if mask_data.shape != ct_data.shape:
        raise ValueError(
            "Mask and CT have different shapes.\n"
            f"Mask shape: {mask_data.shape}\n"
            f"CT shape:   {ct_data.shape}\n\n"
            "They must be aligned before voxel-wise HU filtering."
        )

    if not np.allclose(mask_img.affine, ct_img.affine, atol=1e-4):
        print("[WARNING] Mask and CT affine matrices are not exactly the same.")
        print("[WARNING] This script will continue because voxel shapes match.")
        print("[WARNING] Please confirm spatial alignment in 3D Slicer if needed.\n")


def create_sphere_structure(radius: int) -> np.ndarray:
    """Create a 3D spherical structuring element for dilation."""
    radius = int(radius)
    axis = np.arange(-radius, radius + 1)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    sphere = (x ** 2 + y ** 2 + z ** 2) <= (radius ** 2)
    return sphere.astype(bool)


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 3D 26-connected component from a binary mask."""
    mask = mask.astype(bool)
    positive_voxels = int(mask.sum())

    if positive_voxels == 0:
        print("[WARNING] No positive voxel before connected-component filtering.")
        return np.zeros_like(mask, dtype=bool)

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, num_components = ndimage.label(mask, structure=structure)

    print(f"[INFO] Number of connected components after HU filtering: {num_components}")

    if num_components == 0:
        return np.zeros_like(mask, dtype=bool)

    component_sizes = ndimage.sum(
        mask,
        labeled,
        index=np.arange(1, num_components + 1),
    )
    component_sizes = np.asarray(component_sizes, dtype=np.int64)

    largest_component_label = int(np.argmax(component_sizes) + 1)
    largest_size = int(component_sizes[largest_component_label - 1])

    print(f"[INFO] Largest component label: {largest_component_label}")
    print(f"[INFO] Largest component voxels: {largest_size:,}")

    largest_mask = labeled == largest_component_label
    return largest_mask


def expand_to_width(mask: np.ndarray, dilation_radius: int) -> np.ndarray:
    """
    Expand the mask using spherical dilation.

    For TARGET_WIDTH_VOXELS=6, dilation_radius=3 is used as a practical discrete
    approximation of a 6-voxel-wide tube.
    """
    if int(mask.sum()) == 0:
        print("[WARNING] Empty mask before dilation. Returning empty mask.")
        return np.zeros_like(mask, dtype=np.uint8)

    structure = create_sphere_structure(dilation_radius)
    expanded = ndimage.binary_dilation(mask.astype(bool), structure=structure)
    return expanded.astype(np.uint8)


# ============================================================
# 4. Main Processing Function
# ============================================================
def process_skeleton_with_hu_filter_and_width_expansion(
    skeleton_path: Path,
    ct_path: Path,
    output_path: Path,
) -> Path:
    print("=" * 100)
    print("[PROCESS SKELETON MASK: HU FILTER + TOP1 CC + WIDTH EXPANSION]")
    print(f"[SKELETON] {skeleton_path}")
    print(f"[CT      ] {ct_path}")
    print(f"[OUTPUT  ] {output_path}")
    print(f"[HU THR  ] remove voxels where CT HU < {HU_THRESHOLD}")
    print(f"[WIDTH   ] approximately {TARGET_WIDTH_VOXELS} voxels")
    print("=" * 100)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if SKIP_IF_FINAL_EXISTS and output_path.exists():
        print(f"[SKIP] Final processed skeleton already exists: {output_path}")
        return output_path

    print("[INFO] Loading skeleton mask...")
    mask_img, mask_data = load_nifti(skeleton_path)

    print("[INFO] Loading CT...")
    ct_img, ct_data = load_nifti(ct_path)

    check_same_geometry(mask_img, ct_img, mask_data, ct_data)

    print(f"[INFO] Mask shape: {mask_data.shape}")
    print(f"[INFO] CT shape:   {ct_data.shape}")
    print(f"[INFO] Mask range: min={mask_data.min()}, max={mask_data.max()}")
    print(f"[INFO] CT range:   min={ct_data.min()}, max={ct_data.max()}")

    # --------------------------------------------------------
    # A. Binarize skeleton mask
    # --------------------------------------------------------
    mask_bin = mask_data > 0
    original_voxels = int(mask_bin.sum())
    print(f"[INFO] Original skeleton voxels: {original_voxels:,}")

    if original_voxels == 0:
        print("[WARNING] Input skeleton mask is empty. Saving empty output.")
        final_mask = np.zeros_like(mask_bin, dtype=np.uint8)
    else:
        # ----------------------------------------------------
        # B. Remove skeleton voxels where CT HU < threshold
        # ----------------------------------------------------
        low_hu_mask = ct_data < HU_THRESHOLD
        overlap_remove = mask_bin & low_hu_mask
        remove_voxels = int(overlap_remove.sum())
        print(f"[INFO] Voxels removed because CT HU < {HU_THRESHOLD}: {remove_voxels:,}")

        cleaned_mask = mask_bin.copy()
        cleaned_mask[overlap_remove] = False

        cleaned_voxels = int(cleaned_mask.sum())
        print(f"[INFO] Voxels after HU filtering: {cleaned_voxels:,}")

        # ----------------------------------------------------
        # C. Keep only the largest connected component
        # ----------------------------------------------------
        largest_mask = keep_largest_connected_component(cleaned_mask)
        largest_voxels = int(largest_mask.sum())
        print(f"[INFO] Voxels after retaining top1 component: {largest_voxels:,}")

        # ----------------------------------------------------
        # D. Expand skeleton width to approximately 6 voxels
        # ----------------------------------------------------
        print(f"[INFO] Expanding skeleton with spherical dilation radius = {DILATION_RADIUS} voxels...")
        final_mask = expand_to_width(largest_mask, dilation_radius=DILATION_RADIUS)
        final_voxels = int(final_mask.sum())
        print(f"[INFO] Final expanded skeleton voxels: {final_voxels:,}")

    # --------------------------------------------------------
    # E. Save final NIfTI
    # --------------------------------------------------------
    output_header = mask_img.header.copy()
    output_header.set_data_dtype(np.uint8)

    out_img = nib.Nifti1Image(
        final_mask.astype(np.uint8),
        affine=mask_img.affine,
        header=output_header,
    )
    nib.save(out_img, str(output_path))

    if not output_path.exists():
        raise RuntimeError(f"Output was not created: {output_path}")

    print("[DONE] Finished processing skeleton mask.")
    print(f"[SAVED] {output_path}")
    print("=" * 100)

    return output_path


# ============================================================
# 5. Command-Line Interface for run.py
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process colon vessel skeleton: remove low-HU voxels, keep top1 CC, and expand to width 6."
    )

    parser.add_argument(
        "--ct_path",
        required=True,
        help="CT path passed by run.py. This is used for HU filtering.",
    )

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
# 6. Main Entry Point
# ============================================================
def main() -> None:
    args = parse_args()
    case_id = args.case_id.strip()

    if not case_id:
        raise ValueError("case_id is empty.")

    ct_path = Path(args.ct_path).resolve()
    skeleton_path = resolve_input_skeleton_path(case_id)
    output_path = build_output_path(case_id)

    process_skeleton_with_hu_filter_and_width_expansion(
        skeleton_path=skeleton_path,
        ct_path=ct_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
