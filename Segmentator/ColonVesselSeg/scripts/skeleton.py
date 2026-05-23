import os
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import kimimaro
from scipy.ndimage import binary_dilation


# ============================================================
# 1. Global Settings
# ============================================================
# Colon vessel segmentation outputs are stored here.
SEG_DONE_ROOT = Path(r"C:\Users\User\Desktop\AbdVesselGen\Seg_Done")

# If the final skeleton output already exists, skip this step.
SKIP_IF_FINAL_EXISTS = True

# Skeleton tube width = 3 voxels.
# Implementation: centerline + spherical dilation with radius = 1 voxel.
SKELETON_TUBE_RADIUS = 1

# Kimimaro TEASAR parameters.
# Conservative settings suitable for vessel-like structures.
TEASAR_PARAMS = {
    "scale": 4,
    "const": 1.0,
    "pdrf_scale": 100000,
    "pdrf_exponent": 4,
    "soma_acceptance_threshold": 999999,
    "soma_detection_threshold": 999999,
    "soma_invalidation_const": 0,
    "soma_invalidation_scale": 0,
    "max_paths": None,
}


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
    Input produced by prune.py.

    Example:
        C:/Users/User/Desktop/AbdVesselGen/Seg_Done/43_24055_veins_segmentation/43_24055_pred_top1_largest_components.nii.gz
    """
    return build_case_vein_dir(case_id) / f"{case_id}_pred_top1_largest_components.nii.gz"


def build_output_skeleton_path(case_id: str) -> Path:
    """
    Final skeleton tube output.

    Example:
        C:/Users/User/Desktop/AbdVesselGen/Seg_Done/43_24055_veins_segmentation/43_24055_pred_skeleton.nii.gz
    """
    return build_case_vein_dir(case_id) / f"{case_id}_pred_skeleton.nii.gz"


# ============================================================
# 3. Utility Functions
# ============================================================
def create_sphere_structure(radius: int) -> np.ndarray:
    """Create a 3D spherical structuring element."""
    r = int(radius)
    axis = np.arange(-r, r + 1)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    sphere = (x**2 + y**2 + z**2) <= (r**2)
    return sphere.astype(bool)


def draw_line_3d(mask: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> None:
    """
    Draw a centerline segment from p1 to p2 into a 3D binary mask.
    Coordinates are interpreted as array indices.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)

    dist = np.linalg.norm(p2 - p1)
    num_points = max(int(np.ceil(dist * 2)) + 1, 2)

    line_points = np.linspace(p1, p2, num_points)
    line_points = np.rint(line_points).astype(np.int64)

    shape = np.array(mask.shape, dtype=np.int64)
    valid = np.all((line_points >= 0) & (line_points < shape), axis=1)
    line_points = line_points[valid]

    if len(line_points) > 0:
        mask[line_points[:, 0], line_points[:, 1], line_points[:, 2]] = 1


# ============================================================
# 4. Skeletonization Core
# ============================================================
def create_kimimaro_skeleton_tube(
    input_path: Path,
    output_path: Path,
) -> Path:
    """
    Skeletonize a binary vessel mask using kimimaro, rasterize the centerline,
    dilate it to a 3-voxel-wide tube, and save only this final skeleton mask.
    """
    print("=" * 100)
    print("[COLON VESSEL SKELETONIZATION]")
    print(f"[INPUT ] {input_path}")
    print(f"[OUTPUT] {output_path}")
    print("=" * 100)

    if not input_path.exists():
        raise FileNotFoundError(f"Input pruned vessel mask not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if SKIP_IF_FINAL_EXISTS and output_path.exists():
        print(f"[SKIP] Skeleton output already exists: {output_path}")
        return output_path

    print("[INFO] Loading input NIfTI...")
    nii = nib.load(str(input_path))
    data = nii.get_fdata()

    print(f"[INFO] Shape: {data.shape}")
    print(f"[INFO] Data range: min={data.min()}, max={data.max()}")

    binary_mask = data > 0
    positive_voxels = int(binary_mask.sum())
    print(f"[INFO] Positive voxels in input mask: {positive_voxels:,}")

    output_header = nii.header.copy()
    output_header.set_data_dtype(np.uint8)

    if positive_voxels == 0:
        print("[WARNING] Input mask is empty. Saving an empty skeleton mask.")
        empty = np.zeros_like(data, dtype=np.uint8)
        out_nii = nib.Nifti1Image(empty, affine=nii.affine, header=output_header)
        nib.save(out_nii, str(output_path))
        print(f"[INFO] Empty skeleton output saved to: {output_path}")
        return output_path

    # Kimimaro requires a labeled volume.
    # Since this input should already contain the top-1 component only,
    # all positive voxels are assigned label 1.
    labels = binary_mask.astype(np.uint32)

    print("\n" + "=" * 100)
    print("[INFO] Running kimimaro skeletonization...")

    skels = kimimaro.skeletonize(
        labels,
        teasar_params=TEASAR_PARAMS,
        dust_threshold=0,
        anisotropy=(1, 1, 1),
        fix_branching=True,
        fix_borders=False,
        fill_holes=False,
        progress=True,
        parallel=1,
    )

    if 1 not in skels:
        print("[WARNING] Kimimaro did not return skeleton for label 1. Saving empty output.")
        empty = np.zeros_like(labels, dtype=np.uint8)
        out_nii = nib.Nifti1Image(empty, affine=nii.affine, header=output_header)
        nib.save(out_nii, str(output_path))
        return output_path

    skel = skels[1]
    vertices = np.asarray(skel.vertices)
    edges = np.asarray(skel.edges)

    print("\n" + "=" * 100)
    print("[INFO] Skeletonization completed.")
    print(f"[INFO] Number of skeleton vertices: {len(vertices):,}")
    print(f"[INFO] Number of skeleton edges:    {len(edges):,}")

    if len(vertices) == 0 or len(edges) == 0:
        print("[WARNING] Skeleton has no vertices or edges. Saving empty output.")
        empty = np.zeros_like(labels, dtype=np.uint8)
        out_nii = nib.Nifti1Image(empty, affine=nii.affine, header=output_header)
        nib.save(out_nii, str(output_path))
        return output_path

    # --------------------------------------------------------
    # Rasterize 1-voxel centerline.
    # --------------------------------------------------------
    print("\n[INFO] Rasterizing skeleton centerline...")
    skeleton_centerline = np.zeros_like(labels, dtype=np.uint8)

    for edge in edges:
        idx1, idx2 = int(edge[0]), int(edge[1])
        p1 = vertices[idx1]
        p2 = vertices[idx2]
        draw_line_3d(skeleton_centerline, p1, p2)

    centerline_voxels = int(skeleton_centerline.sum())
    print(f"[INFO] Centerline voxels: {centerline_voxels:,}")

    # --------------------------------------------------------
    # Expand centerline to 3-voxel-wide tube.
    # --------------------------------------------------------
    print("[INFO] Expanding centerline to 3-voxel-wide tube...")
    tube_structure = create_sphere_structure(SKELETON_TUBE_RADIUS)

    skeleton_tube = binary_dilation(
        skeleton_centerline.astype(bool),
        structure=tube_structure,
    ).astype(np.uint8)

    tube_voxels = int(skeleton_tube.sum())
    print(f"[INFO] Skeleton tube voxels: {tube_voxels:,}")

    # --------------------------------------------------------
    # Save only the requested skeleton output.
    # --------------------------------------------------------
    out_nii = nib.Nifti1Image(
        skeleton_tube.astype(np.uint8),
        affine=nii.affine,
        header=output_header,
    )
    nib.save(out_nii, str(output_path))

    if not output_path.exists():
        raise RuntimeError(f"Skeleton output was not created: {output_path}")

    print("\n[INFO] Skeleton output saved to:")
    print(output_path)
    print("=" * 100)
    print("[DONE] Skeletonization finished.")

    return output_path


# ============================================================
# 5. Command-Line Interface for run.py
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a kimimaro skeleton tube from the pruned colon vessel prediction."
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
# 6. Main Entry Point
# ============================================================
def main() -> None:
    args = parse_args()
    case_id = args.case_id.strip()

    if not case_id:
        raise ValueError("case_id is empty.")

    input_path = build_input_mask_path(case_id)
    output_path = build_output_skeleton_path(case_id)

    create_kimimaro_skeleton_tube(
        input_path=input_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
