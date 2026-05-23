import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import nibabel as nib
from scipy import ndimage


# ============================================================
# 1. Global Settings
# ============================================================
BASE_DIR = Path(r"C:\Users\User\Desktop\AbdVesselGen")
SEG_DONE_ROOT = BASE_DIR / "Seg_Done"
ROI_CT_ROOT = BASE_DIR / "ROI_CT"
ANCHOR_POINTS_ROOT = BASE_DIR / "AnchorPoints"

AORTA_LABEL = 52
INVALID_HU_VALUE = -1024

# Only keep connected components covering more than this number of unique Z slices.
# Original scripts used: unique Z slices > 1.
MIN_UNIQUE_Z_SLICES_EXCLUSIVE = 1
CONNECTIVITY_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)

# If outputs already exist, skip generation.
SKIP_IF_FINAL_EXISTS = True


# ============================================================
# 2. Search Task Configuration
# ============================================================
@dataclass
class SearchTask:
    name: str
    plane_suffix: str
    output_suffix: str
    hu_threshold: float
    min_fill_ratio: float
    box_width: int
    box_length: int
    angle_min_deg: float = -30.0
    angle_max_deg: float = 30.0
    angle_step_deg: float = 1.0
    ray_step_size: float = 0.25
    max_ray_distance: float = 100.0


SEARCH_TASKS: List[SearchTask] = [
    SearchTask(
        name="SearchStart2_3231",
        plane_suffix="seg32_highest_seg31_lowest_Zplanes",
        output_suffix="search3231_volume",
        hu_threshold=60,
        min_fill_ratio=0.80,
        box_width=5,
        box_length=15,
    ),
    SearchTask(
        name="SearchStart3_2928",
        plane_suffix="seg29_highest_seg28_lowest_Zplanes",
        output_suffix="search2928_volume",
        hu_threshold=15,
        min_fill_ratio=0.60,
        box_width=5,
        box_length=5,
    ),
]

# Axial-plane anterior direction used in your original scripts.
BASE_DIRECTION = np.array([0.0, -1.0], dtype=np.float64)


# ============================================================
# 3. Path Builders
# ============================================================
def build_roi_ct_path(case_id: str) -> Path:
    return ROI_CT_ROOT / f"{case_id}_ROI_CT.nii.gz"


def build_allseg_path(case_id: str) -> Path:
    return SEG_DONE_ROOT / f"{case_id}_All_segmentation" / f"{case_id}_All_segmentation.nii"


def build_zplane_path(case_id: str, plane_suffix: str) -> Path:
    return ANCHOR_POINTS_ROOT / f"{case_id}_{plane_suffix}.nii.gz"


def build_output_path(case_id: str, output_suffix: str) -> Path:
    # Simple output names, one NIfTI per task.
    return ANCHOR_POINTS_ROOT / f"{case_id}_{output_suffix}.nii.gz"


# ============================================================
# 4. Basic Utilities
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


def check_same_shape(reference_name: str, reference_data: np.ndarray, target_name: str, target_data: np.ndarray) -> None:
    if reference_data.shape != target_data.shape:
        raise ValueError(
            f"Shape mismatch: {reference_name} shape {reference_data.shape} "
            f"!= {target_name} shape {target_data.shape}"
        )


def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize zero vector.")
    return v / norm


def rotate_vector_2d(v: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]], dtype=np.float64)
    return rotation_matrix @ v


def is_inside_slice(shape_2d: Tuple[int, int], x: int, y: int) -> bool:
    sx, sy = shape_2d
    return 0 <= x < sx and 0 <= y < sy


# ============================================================
# 5. Z-Plane Range Utility
# ============================================================
def get_z_range_from_plane_mask(zplane_data: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Z-plane label map convention:
      Label 1 = upper segment highest Z-plane
      Label 2 = lower segment lowest Z-plane

    Returns:
      z_label1, z_label2, z_low, z_high
    """
    label1_coords = np.argwhere(zplane_data == 1)
    label2_coords = np.argwhere(zplane_data == 2)

    if len(label1_coords) == 0:
        raise ValueError("Label 1 not found in Z-plane mask.")
    if len(label2_coords) == 0:
        raise ValueError("Label 2 not found in Z-plane mask.")

    label1_z_values = np.unique(label1_coords[:, 2])
    label2_z_values = np.unique(label2_coords[:, 2])

    if len(label1_z_values) != 1:
        raise ValueError(f"Label 1 appears on multiple Z slices: {label1_z_values.tolist()}")
    if len(label2_z_values) != 1:
        raise ValueError(f"Label 2 appears on multiple Z slices: {label2_z_values.tolist()}")

    z_label1 = int(label1_z_values[0])
    z_label2 = int(label2_z_values[0])
    z_low = min(z_label1, z_label2)
    z_high = max(z_label1, z_label2)

    return z_label1, z_label2, z_low, z_high


# ============================================================
# 6. Aorta-Surface Search Box Utilities
# ============================================================
def find_first_outside_point_from_aorta_surface(
    aorta_slice: np.ndarray,
    centroid_xy: np.ndarray,
    direction_xy: np.ndarray,
    ray_step_size: float = 0.25,
    max_ray_distance: float = 100.0,
):
    """
    Move from the aorta centroid along direction_xy and return the first point
    outside the aorta mask.
    """
    shape_2d = aorta_slice.shape
    centroid_xy = np.asarray(centroid_xy, dtype=np.float64)
    direction_xy = normalize_vector(direction_xy)

    cx_int = int(np.round(centroid_xy[0]))
    cy_int = int(np.round(centroid_xy[1]))

    if not is_inside_slice(shape_2d, cx_int, cy_int):
        return None, None
    if not aorta_slice[cx_int, cy_int]:
        return None, None

    last_inside_point = centroid_xy.copy()
    num_steps = int(max_ray_distance / ray_step_size)

    for step_idx in range(1, num_steps + 1):
        distance = step_idx * ray_step_size
        point = centroid_xy + distance * direction_xy

        x = int(np.round(point[0]))
        y = int(np.round(point[1]))

        if not is_inside_slice(shape_2d, x, y):
            return None, None

        if aorta_slice[x, y]:
            last_inside_point = point.copy()
        else:
            first_outside_point = point.copy()
            return first_outside_point, last_inside_point

    return None, None


def build_surface_based_search_box_mask(
    slice_shape: Tuple[int, int],
    start_outside_xy: np.ndarray,
    direction_xy: np.ndarray,
    box_length: int,
    box_width: int,
) -> np.ndarray:
    """
    Build a surface-based searching box from the first outside-aorta point.
    """
    sx, sy = slice_shape
    start_outside_xy = np.asarray(start_outside_xy, dtype=np.float64)
    direction_xy = normalize_vector(direction_xy)

    perpendicular_xy = np.array([-direction_xy[1], direction_xy[0]], dtype=np.float64)
    box_mask = np.zeros(slice_shape, dtype=bool)

    length_offsets = np.arange(0, box_length, dtype=np.float64)
    half_width = box_width // 2
    width_offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)

    for l in length_offsets:
        row_center = start_outside_xy + l * direction_xy
        for w in width_offsets:
            p = row_center + w * perpendicular_xy
            x = int(np.round(p[0]))
            y = int(np.round(p[1]))
            if 0 <= x < sx and 0 <= y < sy:
                box_mask[x, y] = True

    return box_mask


# ============================================================
# 7. Connected Component Thickness Filter
# ============================================================
def keep_components_with_axial_thickness_gt1(mask: np.ndarray) -> np.ndarray:
    """
    Keep only connected components covering more than 1 unique axial Z slice.
    Uses 26-connectivity.
    """
    mask = mask.astype(bool)
    total_positive = int(mask.sum())

    print(f"[INFO] Raw candidate voxels before thickness filtering: {total_positive:,}")

    if total_positive == 0:
        print("[WARNING] Candidate mask is empty before thickness filtering.")
        return np.zeros_like(mask, dtype=np.uint8)

    labeled_mask, num_components = ndimage.label(mask, structure=CONNECTIVITY_STRUCTURE)
    print(f"[INFO] Number of connected components found: {num_components}")

    if num_components == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    output_mask = np.zeros_like(mask, dtype=np.uint8)
    kept_component_count = 0
    removed_component_count = 0

    print("\n" + "-" * 110)
    print("[INFO] Per-component axial thickness check:")
    print("-" * 110)

    for comp_label in range(1, num_components + 1):
        coords = np.argwhere(labeled_mask == comp_label)
        if len(coords) == 0:
            continue

        voxel_count = len(coords)
        unique_z = np.unique(coords[:, 2])
        num_unique_z = len(unique_z)
        z_min = int(unique_z.min())
        z_max = int(unique_z.max())

        keep = num_unique_z > MIN_UNIQUE_Z_SLICES_EXCLUSIVE

        print(
            f"[Component {comp_label:4d}] "
            f"voxels = {voxel_count:6d} | "
            f"unique Z slices = {num_unique_z:3d} | "
            f"Z range = {z_min:4d} ~ {z_max:4d} | "
            f"{'KEEP' if keep else 'REMOVE'}"
        )

        if keep:
            output_mask[labeled_mask == comp_label] = 1
            kept_component_count += 1
        else:
            removed_component_count += 1

    retained_voxels = int(output_mask.sum())
    removed_voxels = total_positive - retained_voxels

    print("\n" + "=" * 110)
    print("[INFO] Thickness filtering completed.")
    print(f"[INFO] Kept components:    {kept_component_count}")
    print(f"[INFO] Removed components: {removed_component_count}")
    print(f"[INFO] Retained voxels:    {retained_voxels:,}")
    print(f"[INFO] Removed voxels:     {removed_voxels:,}")
    print("=" * 110)

    return output_mask.astype(np.uint8)


# ============================================================
# 8. Main Search Logic
# ============================================================
def run_search_task(
    task: SearchTask,
    case_id: str,
    ct_img,
    ct_data: np.ndarray,
    allseg_data: np.ndarray,
) -> Path:
    zplane_path = build_zplane_path(case_id, task.plane_suffix)
    output_path = build_output_path(case_id, task.output_suffix)

    print("\n" + "=" * 120)
    print(f"[RUN SEARCH TASK] {task.name}")
    print(f"[Z-PLANE] {zplane_path}")
    print(f"[OUTPUT ] {output_path}")
    print("=" * 120)

    ensure_dir(output_path.parent)

    if SKIP_IF_FINAL_EXISTS and output_path.exists():
        print(f"[SKIP] Output already exists: {output_path}")
        return output_path

    zplane_img, zplane_data = load_nifti(zplane_path)
    check_same_shape("ROI CT", ct_data, "Z-plane", zplane_data)

    z_label1, z_label2, z_low, z_high = get_z_range_from_plane_mask(zplane_data)

    print("-" * 110)
    print(f"[INFO] Label 1 Z-plane = {z_label1}")
    print(f"[INFO] Label 2 Z-plane = {z_label2}")
    print(f"[INFO] Searching Z range, inclusive: {z_low} ~ {z_high}")
    print("-" * 110)

    sx, sy, sz = ct_data.shape

    aorta_mask = allseg_data.astype(np.int64) == AORTA_LABEL
    aorta_total_voxels = int(aorta_mask.sum())

    print(f"[INFO] Aorta label = Segment_{AORTA_LABEL}")
    print(f"[INFO] Total aorta voxels: {aorta_total_voxels:,}")

    if aorta_total_voxels == 0:
        raise RuntimeError(f"No voxels found for Segment_{AORTA_LABEL}.")

    search_angles_deg = np.arange(
        task.angle_min_deg,
        task.angle_max_deg + 1e-6,
        task.angle_step_deg,
    )

    raw_output_mask = np.zeros_like(ct_data, dtype=np.uint8)
    qualified_records = []

    print("\n" + "=" * 110)
    print("[INFO] Scanning only within the Z-plane interval...")
    print(
        f"[INFO] Angle sweep: {task.angle_min_deg:.1f}° ~ "
        f"{task.angle_max_deg:.1f}°, step = {task.angle_step_deg:.1f}°"
    )
    print(f"[INFO] HU threshold: > {task.hu_threshold}")
    print(f"[INFO] Minimum fill ratio: {task.min_fill_ratio * 100:.1f}%")
    print(f"[INFO] Box size: width={task.box_width}, length={task.box_length}")
    print("=" * 110)

    for z in range(z_high, z_low - 1, -1):
        aorta_slice = aorta_mask[:, :, z]
        if not np.any(aorta_slice):
            continue

        ct_slice = ct_data[:, :, z]
        aorta_coords = np.argwhere(aorta_slice)
        centroid_xy = aorta_coords.mean(axis=0)

        best_candidate = None

        for angle_deg in search_angles_deg:
            search_direction = rotate_vector_2d(BASE_DIRECTION, float(angle_deg))

            outside_start_xy, _ = find_first_outside_point_from_aorta_surface(
                aorta_slice=aorta_slice,
                centroid_xy=centroid_xy,
                direction_xy=search_direction,
                ray_step_size=task.ray_step_size,
                max_ray_distance=task.max_ray_distance,
            )

            if outside_start_xy is None:
                continue

            raw_box_mask = build_surface_based_search_box_mask(
                slice_shape=(sx, sy),
                start_outside_xy=outside_start_xy,
                direction_xy=search_direction,
                box_length=task.box_length,
                box_width=task.box_width,
            )

            if int(raw_box_mask.sum()) == 0:
                continue

            # Avoid including aorta itself.
            box_mask = raw_box_mask & (~aorta_slice)
            box_voxel_count = int(box_mask.sum())

            if box_voxel_count == 0:
                continue

            hu_positive_count = int(np.logical_and(box_mask, ct_slice > task.hu_threshold).sum())
            fill_ratio = hu_positive_count / box_voxel_count

            candidate = {
                "angle_deg": float(angle_deg),
                "box_mask": box_mask,
                "fill_ratio": float(fill_ratio),
                "positive_count": hu_positive_count,
                "box_voxel_count": box_voxel_count,
                "surface_start_xy": tuple(np.round(outside_start_xy, 2)),
            }

            if best_candidate is None or candidate["fill_ratio"] > best_candidate["fill_ratio"]:
                best_candidate = candidate

        if best_candidate is not None and best_candidate["fill_ratio"] >= task.min_fill_ratio:
            raw_output_mask[:, :, z][best_candidate["box_mask"]] = 1
            qualified_records.append({
                "z": z,
                "centroid_xy": tuple(np.round(centroid_xy, 2)),
                "best_candidate": best_candidate,
            })

            print(
                f"[QUALIFIED] Z={z:4d} | "
                f"best angle={best_candidate['angle_deg']:+.1f}° | "
                f"fill={best_candidate['fill_ratio'] * 100:.1f}% | "
                f"HU>{task.hu_threshold}: "
                f"{best_candidate['positive_count']}/"
                f"{best_candidate['box_voxel_count']} | "
                f"start={best_candidate['surface_start_xy']}"
            )

    print("\n" + "=" * 110)
    print(f"[INFO] Search completed for {task.name}.")
    print(f"[INFO] Number of qualified Z slices: {len(qualified_records)}")

    if len(qualified_records) > 0:
        qualified_z_values = [r["z"] for r in qualified_records]
        print("[INFO] Qualified Z values, from high to low:")
        print(qualified_z_values)
    else:
        print("[WARNING] No qualified slice found within the given Z-plane interval.")
    print("=" * 110)

    final_output_mask = keep_components_with_axial_thickness_gt1(raw_output_mask)

    out_header = ct_img.header.copy()
    out_header.set_data_dtype(np.uint8)

    out_img = nib.Nifti1Image(
        final_output_mask.astype(np.uint8),
        affine=ct_img.affine,
        header=out_header,
    )

    nib.save(out_img, str(output_path))

    if not output_path.exists():
        raise RuntimeError(f"Output was not created: {output_path}")

    print("\n[INFO] Final candidate volume saved to:")
    print(output_path)
    print("[DONE] Finished task.")

    return output_path


# ============================================================
# 9. Combined Pipeline Entry
# ============================================================
def run_combined_search(case_id: str) -> List[Path]:
    roi_ct_path = build_roi_ct_path(case_id)
    allseg_path = build_allseg_path(case_id)

    print("=" * 120)
    print("[COMBINED SEARCH START VOLUME GENERATION]")
    print(f"[CASE ID] {case_id}")
    print(f"[ROI CT ] {roi_ct_path}")
    print(f"[ALLSEG ] {allseg_path}")
    print(f"[OUTPUT ROOT] {ANCHOR_POINTS_ROOT}")
    print("=" * 120)

    ct_img, ct_data = load_nifti(roi_ct_path)
    allseg_img, allseg_data = load_nifti(allseg_path)

    print(f"[INFO] ROI CT shape: {ct_data.shape}")
    print(f"[INFO] ROI CT HU range: min={ct_data.min():.1f}, max={ct_data.max():.1f}")
    print(f"[INFO] AllSeg shape: {allseg_data.shape}")

    check_same_shape("ROI CT", ct_data, "AllSeg", allseg_data)

    output_paths = []
    for task in SEARCH_TASKS:
        output_paths.append(
            run_search_task(
                task=task,
                case_id=case_id,
                ct_img=ct_img,
                ct_data=ct_data,
                allseg_data=allseg_data,
            )
        )

    print("=" * 120)
    print("[DONE] Combined search start volume generation completed.")
    for path in output_paths:
        print(f"[OUTPUT] {path}")
    print("=" * 120)

    return output_paths


# ============================================================
# 10. Command-Line Interface for run.py
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate two candidate search volumes between vertebrae-derived Z-plane intervals."
    )

    # Kept for run.py compatibility.
    parser.add_argument(
        "--ct_path",
        required=True,
        help="CT path passed by run.py. Accepted for compatibility; this script uses ROI_CT output.",
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

    return parser.parse_args()


# ============================================================
# 11. Main Entry Point
# ============================================================
def main() -> None:
    args = parse_args()
    ct_path = Path(args.ct_path).resolve()

    case_id = args.case_id.strip() if args.case_id else strip_nii_extension(ct_path.name)
    if not case_id:
        raise ValueError("case_id is empty.")

    run_combined_search(case_id=case_id)


if __name__ == "__main__":
    main()
