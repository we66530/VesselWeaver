import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
from scipy import ndimage


# ============================================================
# 1. Global Settings
# ============================================================
BASE_DIR = Path(r"C:\Users\User\Desktop\AbdVesselGen")
ANCHOR_POINTS_ROOT = BASE_DIR / "AnchorPoints"
ROI_CT_ROOT = BASE_DIR / "ROI_CT"

INVALID_HU_VALUE = -1024
MAX_SEARCH_RADIUS = 200

# 3D 26-connectivity
CONNECTIVITY_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)

# If the final TXT already exists, skip this step.
SKIP_IF_FINAL_EXISTS = True

SEARCH_VOLUME_SUFFIXES = [
    "search2928_volume",
    "search3231_volume",
]


# ============================================================
# 2. Path Builders
# ============================================================
def build_roi_ct_path(case_id: str) -> Path:
    return ROI_CT_ROOT / f"{case_id}_ROI_CT.nii.gz"


def build_search_volume_path(case_id: str, suffix: str) -> Path:
    return ANCHOR_POINTS_ROOT / f"{case_id}_{suffix}.nii.gz"


def build_output_txt_path(case_id: str) -> Path:
    return ANCHOR_POINTS_ROOT / f"{case_id}_anchor_points_from_search_volumes.txt"


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


def check_same_shape(reference_name: str, reference_data: np.ndarray, target_name: str, target_data: np.ndarray) -> None:
    if reference_data.shape != target_data.shape:
        raise ValueError(
            f"Shape mismatch: {reference_name} shape {reference_data.shape} "
            f"!= {target_name} shape {target_data.shape}"
        )


# ============================================================
# 4. Nearest Valid ROI CT Voxel Search
# ============================================================
def find_nearest_valid_voxel_to_centroid(
    roi_ct_data: np.ndarray,
    centroid_xyz: np.ndarray,
    invalid_hu_value: float = INVALID_HU_VALUE,
    max_search_radius: int = MAX_SEARCH_RADIUS,
) -> Tuple[Tuple[int, int, int], float, float, int]:
    """
    Search outward from the component centroid and return the nearest voxel where
    ROI_CT HU != invalid_hu_value.

    Returns:
        best_coord_tuple, best_hu, distance_to_centroid, found_radius
    """
    shape = np.array(roi_ct_data.shape, dtype=np.int64)
    centroid_xyz = np.asarray(centroid_xyz, dtype=np.float64)
    rounded_centroid = np.rint(centroid_xyz).astype(np.int64)

    # First test the nearest integer voxel.
    if np.all((rounded_centroid >= 0) & (rounded_centroid < shape)):
        rounded_tuple = tuple(int(x) for x in rounded_centroid.tolist())
        rounded_hu = float(roi_ct_data[rounded_tuple])
        if rounded_hu != invalid_hu_value:
            distance = float(np.linalg.norm(rounded_centroid.astype(np.float64) - centroid_xyz))
            return rounded_tuple, rounded_hu, distance, 0

    # Expand a cubic search window until a valid voxel is found.
    for radius in range(1, max_search_radius + 1):
        x_min = max(0, int(rounded_centroid[0]) - radius)
        x_max = min(int(shape[0]) - 1, int(rounded_centroid[0]) + radius)

        y_min = max(0, int(rounded_centroid[1]) - radius)
        y_max = min(int(shape[1]) - 1, int(rounded_centroid[1]) + radius)

        z_min = max(0, int(rounded_centroid[2]) - radius)
        z_max = min(int(shape[2]) - 1, int(rounded_centroid[2]) + radius)

        sub_ct = roi_ct_data[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
        valid_local_mask = sub_ct != invalid_hu_value

        if not np.any(valid_local_mask):
            continue

        local_coords = np.argwhere(valid_local_mask)
        global_coords = local_coords + np.array([x_min, y_min, z_min], dtype=np.int64)

        distances = np.linalg.norm(global_coords.astype(np.float64) - centroid_xyz[None, :], axis=1)
        best_idx = int(np.argmin(distances))

        best_coord = global_coords[best_idx]
        best_coord_tuple = tuple(int(x) for x in best_coord.tolist())
        best_hu = float(roi_ct_data[best_coord_tuple])
        best_distance = float(distances[best_idx])

        return best_coord_tuple, best_hu, best_distance, radius

    raise RuntimeError(
        f"No voxel with HU != {invalid_hu_value} found within radius "
        f"{max_search_radius} around centroid {centroid_xyz.tolist()}."
    )


# ============================================================
# 5. Component Centroid Extraction
# ============================================================
def extract_anchor_points_from_volume(
    volume_name: str,
    volume_path: Path,
    volume_data: np.ndarray,
    roi_ct_data: np.ndarray,
) -> List[Dict]:
    """
    For every disconnected component in one search volume:
      1. compute component centroid
      2. find the nearest ROI_CT voxel where HU != -1024
      3. return records for TXT export
    """
    print("=" * 100)
    print(f"[PROCESS SEARCH VOLUME] {volume_name}")
    print(f"[INPUT] {volume_path}")
    print("=" * 100)

    binary_mask = volume_data > 0
    positive_voxels = int(binary_mask.sum())

    print(f"[INFO] Positive voxels: {positive_voxels:,}")

    if positive_voxels == 0:
        print(f"[WARNING] Search volume is empty: {volume_path}")
        return []

    labeled_components, num_components = ndimage.label(binary_mask, structure=CONNECTIVITY_STRUCTURE)

    print(f"[INFO] Number of disconnected components: {num_components}")

    records: List[Dict] = []

    for component_id in range(1, num_components + 1):
        component_mask = labeled_components == component_id
        component_coords = np.argwhere(component_mask)

        if len(component_coords) == 0:
            continue

        component_voxels = int(len(component_coords))
        centroid_xyz = component_coords.mean(axis=0)

        starting_point, starting_hu, distance_to_centroid, found_radius = find_nearest_valid_voxel_to_centroid(
            roi_ct_data=roi_ct_data,
            centroid_xyz=centroid_xyz,
            invalid_hu_value=INVALID_HU_VALUE,
            max_search_radius=MAX_SEARCH_RADIUS,
        )

        print("-" * 100)
        print(f"[COMPONENT] {volume_name} component {component_id}")
        print(f"  Voxels             : {component_voxels:,}")
        print(f"  Centroid xyz       : ({centroid_xyz[0]:.3f}, {centroid_xyz[1]:.3f}, {centroid_xyz[2]:.3f})")
        print(f"  Selected point xyz : {starting_point}")
        print(f"  HU at point        : {starting_hu:.1f}")
        print(f"  Distance to centroid: {distance_to_centroid:.4f}")
        print(f"  Search radius      : {found_radius}")

        records.append({
            "volume_name": volume_name,
            "component_id": component_id,
            "component_voxels": component_voxels,
            "centroid_x": float(centroid_xyz[0]),
            "centroid_y": float(centroid_xyz[1]),
            "centroid_z": float(centroid_xyz[2]),
            "point_x": int(starting_point[0]),
            "point_y": int(starting_point[1]),
            "point_z": int(starting_point[2]),
            "point_hu": float(starting_hu),
            "distance_to_centroid": float(distance_to_centroid),
            "found_radius": int(found_radius),
        })

    return records


# ============================================================
# 6. TXT Export
# ============================================================
def write_records_to_txt(output_txt_path: Path, case_id: str, records: List[Dict]) -> Path:
    ensure_dir(output_txt_path.parent)

    print("=" * 100)
    print("[WRITE TXT OUTPUT]")
    print(f"[OUTPUT] {output_txt_path}")
    print(f"[N RECORDS] {len(records)}")
    print("=" * 100)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(f"case_id\t{case_id}\n")
        f.write(f"invalid_hu_value\t{INVALID_HU_VALUE}\n")
        f.write(f"max_search_radius\t{MAX_SEARCH_RADIUS}\n")
        f.write("\n")

        header = [
            "volume_name",
            "component_id",
            "component_voxels",
            "centroid_x",
            "centroid_y",
            "centroid_z",
            "point_x",
            "point_y",
            "point_z",
            "point_hu",
            "distance_to_centroid",
            "found_radius",
        ]
        f.write("\t".join(header) + "\n")

        for r in records:
            row = [
                r["volume_name"],
                str(r["component_id"]),
                str(r["component_voxels"]),
                f"{r['centroid_x']:.6f}",
                f"{r['centroid_y']:.6f}",
                f"{r['centroid_z']:.6f}",
                str(r["point_x"]),
                str(r["point_y"]),
                str(r["point_z"]),
                f"{r['point_hu']:.3f}",
                f"{r['distance_to_centroid']:.6f}",
                str(r["found_radius"]),
            ]
            f.write("\t".join(row) + "\n")

    print(f"[DONE] TXT saved: {output_txt_path}")
    return output_txt_path


# ============================================================
# 7. Main Pipeline Function
# ============================================================
def find_anchor_points_from_search_volumes(case_id: str) -> Path:
    roi_ct_path = build_roi_ct_path(case_id)
    output_txt_path = build_output_txt_path(case_id)

    print("=" * 100)
    print("[FIND ANCHOR POINTS FROM SEARCH VOLUMES]")
    print(f"[CASE ID] {case_id}")
    print(f"[ROI CT ] {roi_ct_path}")
    print(f"[OUTPUT ] {output_txt_path}")
    print("=" * 100)

    if SKIP_IF_FINAL_EXISTS and output_txt_path.exists():
        print(f"[SKIP] TXT already exists: {output_txt_path}")
        return output_txt_path

    roi_ct_img, roi_ct_data = load_nifti(roi_ct_path)

    print(f"[INFO] ROI CT shape: {roi_ct_data.shape}")
    print(f"[INFO] ROI CT HU range: min={roi_ct_data.min():.1f}, max={roi_ct_data.max():.1f}")

    all_records: List[Dict] = []

    for suffix in SEARCH_VOLUME_SUFFIXES:
        volume_path = build_search_volume_path(case_id, suffix)
        volume_img, volume_data = load_nifti(volume_path)

        check_same_shape("ROI CT", roi_ct_data, suffix, volume_data)

        records = extract_anchor_points_from_volume(
            volume_name=suffix,
            volume_path=volume_path,
            volume_data=volume_data,
            roi_ct_data=roi_ct_data,
        )
        all_records.extend(records)

    if len(all_records) == 0:
        print("[WARNING] No anchor point records were generated. TXT will still be created with header only.")

    return write_records_to_txt(output_txt_path, case_id, all_records)


# ============================================================
# 8. Command-Line Interface for run.py
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find nearest valid ROI_CT points to each component centroid in search volumes."
    )

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
# 9. Main Entry Point
# ============================================================
def main() -> None:
    args = parse_args()
    ct_path = Path(args.ct_path).resolve()

    case_id = args.case_id.strip() if args.case_id else strip_nii_extension(ct_path.name)
    if not case_id:
        raise ValueError("case_id is empty.")

    find_anchor_points_from_search_volumes(case_id=case_id)


if __name__ == "__main__":
    main()
