import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import nibabel as nib


# ============================================================
# 1. Global Settings
# ============================================================
BASE_DIR = Path(r"C:\Users\User\Desktop\AbdVesselGen")
ROI_CT_ROOT = BASE_DIR / "ROI_CT"
SEG_DONE_ROOT = BASE_DIR / "Seg_Done"
ANCHOR_POINTS_ROOT = BASE_DIR / "AnchorPoints"

# Optional: write one combined label map as well.
# 0 = background
# 1 = renal artery Lt start candidate
# 2 = renal artery Rt start candidate
# 3 = renal vein Rt start candidate
# 4 = renal vein Lt start candidate
SAVE_COMBINED_LABEL_MAP = True

# If final combined output already exists, skip this step.
SKIP_IF_FINAL_EXISTS = True


# ============================================================
# 2. Task Definition
# ============================================================
@dataclass
class SearchTask:
    name: str
    target_label: int
    target_name: str
    output_label_value: int
    hu_threshold: float
    min_fill_ratio: float
    box_width: int
    box_length: int
    search_angles_deg: List[float]
    base_direction: Tuple[float, float]
    ray_step_size: float = 0.25
    max_ray_distance: float = 100.0


TASKS = [
    SearchTask(
        name="renal_artery_Lt",
        target_label=52,  # Aorta
        target_name="Aorta / Segment_52",
        output_label_value=1,
        hu_threshold=60,
        min_fill_ratio=0.80,
        box_width=5,
        box_length=10,
        search_angles_deg=[0, 30, -30],
        base_direction=(-1.0, 0.0),
    ),
    SearchTask(
        name="renal_artery_Rt",
        target_label=52,  # Aorta
        target_name="Aorta / Segment_52",
        output_label_value=2,
        hu_threshold=60,
        min_fill_ratio=0.80,
        box_width=5,
        box_length=5,
        search_angles_deg=[0, 10, -10],
        base_direction=(1.0, 0.0),
    ),
    SearchTask(
        name="renal_vein_Rt",
        target_label=63,  # IVC
        target_name="IVC / Segment_63",
        output_label_value=3,
        hu_threshold=60,
        min_fill_ratio=0.80,
        box_width=5,
        box_length=10,
        search_angles_deg=[0, 10, -10],
        base_direction=(-1.0, 0.0),
    ),
    SearchTask(
        name="renal_vein_Lt",
        target_label=63,  # IVC
        target_name="IVC / Segment_63",
        output_label_value=4,
        hu_threshold=60,
        min_fill_ratio=0.80,
        box_width=5,
        box_length=20,
        search_angles_deg=[0, 0, -10],  # Kept exactly as the original script.
        base_direction=(1.0, 0.0),
    ),
]


# ============================================================
# 3. Path Builders
# ============================================================
def build_roi_ct_path(case_id: str) -> Path:
    return ROI_CT_ROOT / f"{case_id}_ROI_CT.nii.gz"


def build_allseg_path(case_id: str) -> Path:
    return SEG_DONE_ROOT / f"{case_id}_All_segmentation" / f"{case_id}_All_segmentation.nii"


def build_z_plane_mask_path(case_id: str) -> Path:
    return ANCHOR_POINTS_ROOT / f"{case_id}_seg32_highest_seg31_lowest_Zplanes.nii.gz"


def build_task_output_path(case_id: str, task_name: str) -> Path:
    return ANCHOR_POINTS_ROOT / f"{case_id}_{task_name}_start_restrict.nii.gz"


def build_combined_output_path(case_id: str) -> Path:
    return ANCHOR_POINTS_ROOT / f"{case_id}_renal_artery_vein_4starts_combined_restrict.nii.gz"


def build_summary_path(case_id: str) -> Path:
    return ANCHOR_POINTS_ROOT / f"{case_id}_renal_artery_vein_4starts_summary.txt"


# ============================================================
# 4. Utility Functions
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


# ============================================================
# 5. Geometry Helpers
# ============================================================
def normalize_vector(v):
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize zero vector.")
    return v / norm


def rotate_vector_2d(v, angle_deg):
    """Rotate a 2D vector in the axial XY plane."""
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]], dtype=np.float64)
    return rotation_matrix @ np.asarray(v, dtype=np.float64)


def is_inside_slice(shape_2d, x, y):
    sx, sy = shape_2d
    return 0 <= x < sx and 0 <= y < sy


def find_first_outside_point_from_vessel_surface(
    vessel_slice,
    centroid_xy,
    direction_xy,
    ray_step_size=0.25,
    max_ray_distance=100.0,
):
    """
    From the target vessel centroid, shoot a ray along direction_xy.
    Return the first point outside the vessel mask.
    """
    shape_2d = vessel_slice.shape
    centroid_xy = np.asarray(centroid_xy, dtype=np.float64)
    direction_xy = normalize_vector(direction_xy)

    cx_int = int(np.round(centroid_xy[0]))
    cy_int = int(np.round(centroid_xy[1]))

    if not is_inside_slice(shape_2d, cx_int, cy_int):
        return None, None

    if not vessel_slice[cx_int, cy_int]:
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

        if vessel_slice[x, y]:
            last_inside_point = point.copy()
        else:
            first_outside_point = point.copy()
            return first_outside_point, last_inside_point

    return None, None


def build_surface_based_search_box_mask(
    slice_shape,
    start_outside_xy,
    direction_xy,
    box_length=10,
    box_width=5,
):
    """
    Build an oriented search box starting from the first outside-surface point.
    The first row is adjacent to the target vessel surface.
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
# 6. Z-Plane Restriction
# ============================================================
def get_z_scan_range_from_two_plane_mask(z_plane_mask_path: Path, expected_shape):
    """
    Read a NIfTI mask containing two Z-plane segments and return inclusive
    scan range: z_min <= z <= z_max.
    """
    print("\n[INFO] Loading Z-plane restriction mask:")
    print(z_plane_mask_path)

    z_img, z_data = load_nifti(z_plane_mask_path)

    print(f"[INFO] Z-plane mask shape: {z_data.shape}")

    if z_data.shape != expected_shape:
        raise ValueError(
            f"Shape mismatch: CT shape {expected_shape} "
            f"!= Z-plane mask shape {z_data.shape}"
        )

    nonzero_mask = z_data > 0
    if not np.any(nonzero_mask):
        raise RuntimeError("No nonzero voxels found in Z-plane restriction mask.")

    labels = np.unique(z_data[nonzero_mask])
    labels = labels[labels > 0]

    plane_records = []

    if len(labels) >= 2:
        print(f"[INFO] Nonzero Z-plane labels found: {labels.tolist()}")

        for label in labels:
            coords = np.argwhere(z_data == label)
            if coords.size == 0:
                continue
            z_values = coords[:, 2]
            plane_z = int(np.round(np.median(z_values)))
            plane_records.append({
                "label": label,
                "z": plane_z,
                "z_min": int(z_values.min()),
                "z_max": int(z_values.max()),
                "voxels": int(coords.shape[0]),
            })

        if len(plane_records) < 2:
            raise RuntimeError("Could not derive two valid Z planes from labeled mask.")

        z_positions = [r["z"] for r in plane_records]
        z_min = int(min(z_positions))
        z_max = int(max(z_positions))

        print("[INFO] Derived Z planes:")
        for r in sorted(plane_records, key=lambda x: x["z"], reverse=True):
            print(
                f"       label={r['label']}: z={r['z']} "
                f"(raw z range {r['z_min']}-{r['z_max']}), voxels={r['voxels']:,}"
            )
    else:
        coords = np.argwhere(nonzero_mask)
        z_values = coords[:, 2]
        z_min = int(z_values.min())
        z_max = int(z_values.max())
        print(
            "[WARNING] Only one nonzero label found in Z-plane mask; "
            "using all nonzero voxels' min/max Z instead."
        )

    if z_min > z_max:
        z_min, z_max = z_max, z_min

    print(f"[INFO] Restricted scan Z range: {z_min} to {z_max} inclusive")
    return z_min, z_max


# ============================================================
# 7. Core Search Runner
# ============================================================
def run_one_task(
    task: SearchTask,
    case_id: str,
    ct_img,
    ct_data,
    seg_data,
    z_scan_min,
    z_scan_max,
):
    sx, sy, sz = ct_data.shape
    base_direction = np.asarray(task.base_direction, dtype=np.float64)

    vessel_mask = seg_data == task.target_label
    vessel_total_voxels = int(vessel_mask.sum())

    output_path = build_task_output_path(case_id, task.name)

    print("\n" + "=" * 100)
    print(f"[TASK] {task.name}")
    print(f"[INFO] Target = {task.target_name}")
    print(f"[INFO] Total target voxels: {vessel_total_voxels:,}")
    print(
        "[INFO] Criteria: "
        f"HU > {task.hu_threshold}, fill ratio >= {task.min_fill_ratio:.2f}, "
        f"box_width={task.box_width}, box_length={task.box_length}, "
        f"angles={task.search_angles_deg}, base_direction={base_direction.tolist()}"
    )
    print(f"[INFO] Z restriction: {z_scan_min} <= z <= {z_scan_max}")
    print(f"[OUTPUT] {output_path}")
    print("=" * 100)

    if vessel_total_voxels == 0:
        raise RuntimeError(f"No voxels found for Segment_{task.target_label} in task {task.name}.")

    output_mask = np.zeros_like(ct_data, dtype=np.uint8)
    qualified_records = []

    scan_start_z = min(z_scan_max, sz - 1)
    scan_end_z = max(z_scan_min, 0)

    for z in range(scan_start_z, scan_end_z - 1, -1):
        vessel_slice = vessel_mask[:, :, z]
        if not np.any(vessel_slice):
            continue

        ct_slice = ct_data[:, :, z]
        vessel_coords = np.argwhere(vessel_slice)
        centroid_xy = vessel_coords.mean(axis=0)

        qualified_boxes_this_slice = []

        for angle_deg in task.search_angles_deg:
            search_direction = rotate_vector_2d(base_direction, angle_deg)

            outside_start_xy, last_inside_xy = find_first_outside_point_from_vessel_surface(
                vessel_slice=vessel_slice,
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

            raw_box_voxels = int(raw_box_mask.sum())
            if raw_box_voxels == 0:
                continue

            overlap_with_vessel = raw_box_mask & vessel_slice
            overlap_voxels = int(overlap_with_vessel.sum())

            box_mask = raw_box_mask & (~vessel_slice)
            box_voxel_count = int(box_mask.sum())
            if box_voxel_count == 0:
                continue

            hu_positive_count = int(np.logical_and(box_mask, ct_slice > task.hu_threshold).sum())
            fill_ratio = hu_positive_count / box_voxel_count

            if fill_ratio >= task.min_fill_ratio:
                qualified_boxes_this_slice.append({
                    "angle_deg": angle_deg,
                    "box_mask": box_mask,
                    "fill_ratio": fill_ratio,
                    "positive_count": hu_positive_count,
                    "box_voxel_count": box_voxel_count,
                    "raw_box_voxels": raw_box_voxels,
                    "vessel_overlap_removed": overlap_voxels,
                    "surface_start_xy": tuple(np.round(outside_start_xy, 2)),
                })

        if len(qualified_boxes_this_slice) > 0:
            for record in qualified_boxes_this_slice:
                output_mask[:, :, z][record["box_mask"]] = 1

            qualified_records.append({
                "z": z,
                "centroid_xy": tuple(np.round(centroid_xy, 2)),
                "boxes": qualified_boxes_this_slice,
            })

            angle_summary = ", ".join([
                f"{r['angle_deg']:+g}deg ({r['fill_ratio'] * 100:.1f}%, start={r['surface_start_xy']})"
                for r in qualified_boxes_this_slice
            ])

            print(
                f"[QUALIFIED][{task.name}] Z = {z:4d} | "
                f"centroid = ({centroid_xy[0]:.2f}, {centroid_xy[1]:.2f}) | "
                f"{angle_summary}"
            )

    print("\n" + "-" * 100)
    print(f"[INFO][{task.name}] Scan completed.")
    print(f"[INFO][{task.name}] Number of qualified Z slices: {len(qualified_records)}")

    if len(qualified_records) > 0:
        qualified_z_values = [record["z"] for record in qualified_records]
        print(f"[INFO][{task.name}] Qualified Z values, from high to low:")
        print(qualified_z_values)
    else:
        print(f"[WARNING][{task.name}] No qualified axial slice found.")

    print(f"[INFO][{task.name}] Saving output mask:")
    print(output_path)
    ensure_dir(output_path.parent)

    out_header = ct_img.header.copy()
    out_header.set_data_dtype(np.uint8)
    out_img = nib.Nifti1Image(output_mask.astype(np.uint8), affine=ct_img.affine, header=out_header)
    nib.save(out_img, str(output_path))
    print(f"[DONE][{task.name}] Output mask saved.")

    return output_mask, qualified_records, output_path


# ============================================================
# 8. Main Pipeline Function
# ============================================================
def run_renal_start_search(case_id: str) -> Path:
    ct_path = build_roi_ct_path(case_id)
    vessel_seg_path = build_allseg_path(case_id)
    z_plane_mask_path = build_z_plane_mask_path(case_id)
    combined_output_path = build_combined_output_path(case_id)
    summary_path = build_summary_path(case_id)

    print("=" * 100)
    print("[RENAL ARTERY / RENAL VEIN START SEARCH - RUN.PY VERSION]")
    print(f"[CASE ID    ] {case_id}")
    print(f"[ROI CT     ] {ct_path}")
    print(f"[VESSEL SEG ] {vessel_seg_path}")
    print(f"[Z PLANES   ] {z_plane_mask_path}")
    print(f"[OUTPUT DIR ] {ANCHOR_POINTS_ROOT}")
    print("=" * 100)

    ensure_dir(ANCHOR_POINTS_ROOT)

    if SKIP_IF_FINAL_EXISTS and combined_output_path.exists() and summary_path.exists():
        print(f"[SKIP] Combined output already exists: {combined_output_path}")
        print(f"[SKIP] Summary already exists: {summary_path}")
        return combined_output_path

    ct_img, ct_data = load_nifti(ct_path)
    seg_img, seg_data_float = load_nifti(vessel_seg_path)
    seg_data = np.rint(seg_data_float).astype(np.int32)

    print(f"[INFO] CT shape: {ct_data.shape}")
    print(f"[INFO] CT HU range: min={ct_data.min():.1f}, max={ct_data.max():.1f}")
    print(f"[INFO] Segmentation shape: {seg_data.shape}")

    check_same_shape("ROI CT", ct_data, "vessel segmentation", seg_data)

    z_scan_min, z_scan_max = get_z_scan_range_from_two_plane_mask(
        z_plane_mask_path=z_plane_mask_path,
        expected_shape=ct_data.shape,
    )

    combined_label_map = np.zeros_like(ct_data, dtype=np.uint8)
    all_summary = []

    for task in TASKS:
        task_mask, records, task_output_path = run_one_task(
            task=task,
            case_id=case_id,
            ct_img=ct_img,
            ct_data=ct_data,
            seg_data=seg_data,
            z_scan_min=z_scan_min,
            z_scan_max=z_scan_max,
        )

        combined_label_map[task_mask > 0] = task.output_label_value
        all_summary.append((task.name, len(records), str(task_output_path)))

    if SAVE_COMBINED_LABEL_MAP:
        print("\n" + "=" * 100)
        print("[INFO] Saving combined 4-task label map:")
        print(combined_output_path)
        out_header = ct_img.header.copy()
        out_header.set_data_dtype(np.uint8)
        combined_img = nib.Nifti1Image(
            combined_label_map.astype(np.uint8),
            affine=ct_img.affine,
            header=out_header,
        )
        nib.save(combined_img, str(combined_output_path))
        print("[DONE] Combined label map saved.")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"case_id\t{case_id}\n")
        f.write(f"ct_path\t{ct_path}\n")
        f.write(f"vessel_seg_path\t{vessel_seg_path}\n")
        f.write(f"z_plane_mask_path\t{z_plane_mask_path}\n")
        f.write(f"z_scan_min\t{z_scan_min}\n")
        f.write(f"z_scan_max\t{z_scan_max}\n")
        f.write("\n")
        f.write("task_name\tqualified_slices\toutput_path\n")
        for task_name, n_slices, out_path in all_summary:
            f.write(f"{task_name}\t{n_slices}\t{out_path}\n")
        if SAVE_COMBINED_LABEL_MAP:
            f.write(f"combined_label_map\tNA\t{combined_output_path}\n")

    print("\n" + "=" * 100)
    print("[SUMMARY]")
    for task_name, n_slices, out_path in all_summary:
        print(f"  {task_name}: qualified_slices={n_slices}, output={out_path}")
    if SAVE_COMBINED_LABEL_MAP:
        print(f"  combined_label_map: {combined_output_path}")
    print(f"  summary: {summary_path}")
    print("=" * 100)

    return combined_output_path


# ============================================================
# 9. Command-Line Interface for run.py
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search renal artery and renal vein starting boxes using ROI_CT, AllSeg, and vertebral Z-plane restriction."
    )

    parser.add_argument(
        "--ct_path",
        required=True,
        help="CT path passed by run.py. Accepted for compatibility; this script uses ROI_CT output.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory passed by run.py. Accepted for compatibility; output is written to AnchorPoints.",
    )

    parser.add_argument(
        "--case_id",
        required=False,
        default=None,
        help="Case ID passed by run.py. If omitted, inferred from CT filename.",
    )

    return parser.parse_args()


# ============================================================
# 10. Main Entry Point
# ============================================================
def main() -> None:
    args = parse_args()
    ct_path = Path(args.ct_path).resolve()

    case_id = args.case_id.strip() if args.case_id else strip_nii_extension(ct_path.name)
    if not case_id:
        raise ValueError("case_id is empty.")

    run_renal_start_search(case_id=case_id)


if __name__ == "__main__":
    main()
