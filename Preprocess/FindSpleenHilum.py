import argparse
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage


# ============================================================
# 1. Global Settings
# ============================================================
BASE_DIR = Path(r"C:\Users\User\Desktop\AbdVesselGen")
SEG_DONE_ROOT = BASE_DIR / "Seg_Done"
ANCHOR_POINTS_ROOT = BASE_DIR / "AnchorPoints"

# TotalSegmentator / AllSeg spleen label.
# Keep this as 1 if your AllSeg output uses label 1 for spleen, matching the original script.
SPLEEN_LABEL = 1

# Sphere settings inherited from the original script.
SPHERE_RADIUS = 4
SIGMA_DIST = 20.0

# If final output already exists, skip this step.
SKIP_IF_FINAL_EXISTS = True


# ============================================================
# 2. Path Builders
# ============================================================
def build_allseg_path(case_id: str) -> Path:
    """
    Example:
        C:/Users/User/Desktop/AbdVesselGen/Seg_Done/43_24055_All_segmentation/43_24055_All_segmentation.nii
    """
    return SEG_DONE_ROOT / f"{case_id}_All_segmentation" / f"{case_id}_All_segmentation.nii"


def build_output_path(case_id: str) -> Path:
    """
    Example:
        C:/Users/User/Desktop/AbdVesselGen/AnchorPoints/43_24055_SpleenHilum.nii
    """
    return ANCHOR_POINTS_ROOT / f"{case_id}_SpleenHilum.nii"


def build_summary_path(case_id: str) -> Path:
    return ANCHOR_POINTS_ROOT / f"{case_id}_SpleenHilum.txt"


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


# ============================================================
# 4. Core Hilum Finder
# ============================================================
def find_central_hilum_lps(
    mask_path: Path,
    output_path: Path,
    summary_path: Path,
    label: int = SPLEEN_LABEL,
    sphere_radius: int = SPHERE_RADIUS,
    sigma_dist: float = SIGMA_DIST,
) -> Path:
    """
    Find a central concavity-like point on the anterior spleen surface and draw
    a small sphere at that point.

    This is the run.py-integrated version of the original FindSpleenHilum.py.
    The original algorithm:
      1. Loads a label mask.
      2. Finds the selected spleen label.
      3. Computes its bounding box.
      4. Builds an anterior surface map using minimum Y for each Z-X column.
      5. Finds the weighted deepest surface point near the bbox center.
      6. Draws a sphere at that point.

    Note:
      The original script names the axes as Z/Y/X. NIfTI arrays loaded by nibabel
      are indexed as data[i, j, k]. This version preserves the original axis logic
      to avoid changing behavior.
    """
    ensure_dir(output_path.parent)

    print("=" * 100)
    print("[FIND SPLEEN HILUM]")
    print(f"[MASK INPUT] {mask_path}")
    print(f"[OUTPUT    ] {output_path}")
    print(f"[SUMMARY   ] {summary_path}")
    print(f"[LABEL     ] {label}")
    print(f"[RADIUS    ] {sphere_radius}")
    print(f"[SIGMA DIST] {sigma_dist}")
    print("=" * 100)

    if SKIP_IF_FINAL_EXISTS and output_path.exists() and summary_path.exists():
        print(f"[SKIP] Spleen hilum output already exists: {output_path}")
        print(f"[SKIP] Summary already exists: {summary_path}")
        return output_path

    img, data_float = load_nifti(mask_path)
    data = np.rint(data_float).astype(np.int32)
    affine = img.affine
    header = img.header.copy()

    coords = np.argwhere(data == int(label))
    if coords.size == 0:
        raise RuntimeError(
            f"Spleen label {label} was not found in mask:\n{mask_path}\n"
            "Please confirm the spleen label ID in your AllSeg output."
        )

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    print(f"[INFO] Label voxels: {len(coords):,}")
    print(f"[INFO] BBox min, original-axis naming: Z={z_min}, Y={y_min}, X={x_min}")
    print(f"[INFO] BBox max, original-axis naming: Z={z_max}, Y={y_max}, X={x_max}")

    z_center_local = (z_max - z_min) / 2.0
    x_center_local = (x_max - x_min) / 2.0

    # Surface height map: for every local (Z, X), find the minimum Y.
    surf_height = np.full((z_max - z_min + 1, x_max - x_min + 1), 1e6, dtype=np.float64)

    for c in coords:
        zz = int(c[0] - z_min)
        yy = int(c[1])
        xx = int(c[2] - x_min)
        if yy < surf_height[zz, xx]:
            surf_height[zz, xx] = yy

    valid_mask = surf_height != 1e6
    depth_map = np.zeros_like(surf_height, dtype=np.float64)
    depth_map[valid_mask] = surf_height[valid_mask] - y_min

    zz_grid, xx_grid = np.indices(depth_map.shape)
    dist_sq = (zz_grid - z_center_local) ** 2 + (xx_grid - x_center_local) ** 2
    weight_map = np.exp(-dist_sq / (2 * float(sigma_dist) ** 2))

    weighted_depth = depth_map * weight_map
    smoothed = ndimage.gaussian_filter(weighted_depth, sigma=1.2)

    local_max_filter = ndimage.maximum_filter(smoothed, size=10)
    candidates = (smoothed == local_max_filter) & (smoothed > 0)

    cand_indices = np.argwhere(candidates)
    if cand_indices.size == 0:
        idx = np.unravel_index(np.argmax(weighted_depth), weighted_depth.shape)
        selected_mode = "global_weighted_depth_max"
    else:
        best_val = -np.inf
        idx = cand_indices[0]
        for c_idx in cand_indices:
            val = smoothed[c_idx[0], c_idx[1]]
            if val > best_val:
                best_val = val
                idx = c_idx
        selected_mode = "local_max_smoothed_weighted_depth"

    z_final = int(idx[0] + z_min)
    x_final = int(idx[1] + x_min)
    y_final = int(surf_height[idx[0], idx[1]])

    center_zx = (z_center_local + z_min, x_center_local + x_min)

    print("=" * 100)
    print("[SPLEEN HILUM RESULT]")
    print(f"[BBox center Z/X] ({center_zx[0]:.1f}, {center_zx[1]:.1f})")
    print(f"[Selected mode  ] {selected_mode}")
    print(f"[Hilum point    ] Z={z_final}, Y={y_final}, X={x_final}")
    print("=" * 100)

    output_data = np.zeros_like(data, dtype=np.uint8)
    z_g, y_g, x_g = np.ogrid[:data.shape[0], :data.shape[1], :data.shape[2]]
    dist_ball = np.sqrt(
        (z_g - z_final) ** 2
        + (y_g - y_final) ** 2
        + (x_g - x_final) ** 2
    )
    output_data[dist_ball <= int(sphere_radius)] = 1

    out_header = header.copy()
    out_header.set_data_dtype(np.uint8)

    new_img = nib.Nifti1Image(output_data.astype(np.uint8), affine, out_header)
    nib.save(new_img, str(output_path))

    if not output_path.exists():
        raise RuntimeError(f"Output was not created: {output_path}")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("item\tvalue\n")
        f.write(f"mask_path\t{mask_path}\n")
        f.write(f"output_path\t{output_path}\n")
        f.write(f"label\t{label}\n")
        f.write(f"sphere_radius\t{sphere_radius}\n")
        f.write(f"sigma_dist\t{sigma_dist}\n")
        f.write(f"bbox_z_min\t{z_min}\n")
        f.write(f"bbox_y_min\t{y_min}\n")
        f.write(f"bbox_x_min\t{x_min}\n")
        f.write(f"bbox_z_max\t{z_max}\n")
        f.write(f"bbox_y_max\t{y_max}\n")
        f.write(f"bbox_x_max\t{x_max}\n")
        f.write(f"bbox_center_z\t{center_zx[0]:.6f}\n")
        f.write(f"bbox_center_x\t{center_zx[1]:.6f}\n")
        f.write(f"selected_mode\t{selected_mode}\n")
        f.write(f"hilum_z\t{z_final}\n")
        f.write(f"hilum_y\t{y_final}\n")
        f.write(f"hilum_x\t{x_final}\n")
        f.write(f"sphere_voxels\t{int(np.count_nonzero(output_data))}\n")

    print(f"[DONE] Spleen hilum sphere saved: {output_path}")
    print(f"[DONE] Summary saved: {summary_path}")
    print(f"[INFO] Sphere voxels: {int(np.count_nonzero(output_data)):,}")

    return output_path


# ============================================================
# 5. Command-Line Interface for run.py
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find spleen hilum sphere from AllSeg spleen label and save it to AnchorPoints."
    )

    parser.add_argument(
        "--ct_path",
        required=True,
        help="CT path passed by run.py. Accepted for compatibility but not directly used.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory passed by run.py. Accepted for compatibility.",
    )

    parser.add_argument(
        "--case_id",
        required=False,
        default=None,
        help="Case ID passed by run.py. If omitted, inferred from CT filename.",
    )

    parser.add_argument(
        "--spleen_label",
        required=False,
        type=int,
        default=SPLEEN_LABEL,
        help="Spleen label ID in AllSeg mask. Default: 1.",
    )

    return parser.parse_args()


# ============================================================
# 6. Main Entry Point
# ============================================================
def main() -> None:
    args = parse_args()
    ct_path = Path(args.ct_path).resolve()

    case_id = args.case_id.strip() if args.case_id else strip_nii_extension(ct_path.name)
    if not case_id:
        raise ValueError("case_id is empty.")

    mask_in = build_allseg_path(case_id)
    output_path = build_output_path(case_id)
    summary_path = build_summary_path(case_id)

    find_central_hilum_lps(
        mask_path=mask_in,
        output_path=output_path,
        summary_path=summary_path,
        label=int(args.spleen_label),
        sphere_radius=SPHERE_RADIUS,
        sigma_dist=SIGMA_DIST,
    )


if __name__ == "__main__":
    main()
