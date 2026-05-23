import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import nibabel as nib


# ============================================================
# 1. Global Settings
# ============================================================
BASE_DIR = Path(r"C:\Users\User\Desktop\AbdVesselGen")
SEG_DONE_ROOT = BASE_DIR / "Seg_Done"
ANCHOR_POINTS_ROOT = BASE_DIR / "AnchorPoints"

# Renal-related labels from AllSeg used by the original script.
TARGET_LABELS = [2, 3]

# Diameter = 6 voxels, therefore radius = 3 voxels.
SPHERE_RADIUS_VOX = 3.0

# If final output already exists, skip this step.
SKIP_IF_FINAL_EXISTS = True


# ============================================================
# 2. Path Builders
# ============================================================
def build_input_nii_path(case_id: str) -> Path:
    """
    Example:
        C:/Users/User/Desktop/AbdVesselGen/Seg_Done/43_24055_All_segmentation/43_24055_All_segmentation.nii
    """
    return SEG_DONE_ROOT / f"{case_id}_All_segmentation" / f"{case_id}_All_segmentation.nii"


def build_output_nii_path(case_id: str) -> Path:
    """
    Example:
        C:/Users/User/Desktop/AbdVesselGen/AnchorPoints/43_24055_renal_hilum_seg2_seg3_centroid_spheres_diameter6.nii.gz
    """
    return ANCHOR_POINTS_ROOT / f"{case_id}_renal_hilum_seg2_seg3_centroid_spheres_diameter6.nii.gz"


def build_summary_path(case_id: str) -> Path:
    return ANCHOR_POINTS_ROOT / f"{case_id}_renal_hilum_seg2_seg3_centroid_spheres_diameter6.txt"


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
# 4. Core Functions
# ============================================================
def compute_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return coords.mean(axis=0)


def draw_sphere(
    output_arr: np.ndarray,
    center: np.ndarray,
    radius: float,
    label: int,
) -> None:
    """
    Draw a voxel-space sphere centered at `center`.

    center is in voxel index coordinates: [x, y, z]
    """
    shape = output_arr.shape
    cx, cy, cz = center

    x_min = max(int(np.floor(cx - radius)), 0)
    x_max = min(int(np.ceil(cx + radius)) + 1, shape[0])

    y_min = max(int(np.floor(cy - radius)), 0)
    y_max = min(int(np.ceil(cy + radius)) + 1, shape[1])

    z_min = max(int(np.floor(cz - radius)), 0)
    z_max = min(int(np.ceil(cz + radius)) + 1, shape[2])

    xs = np.arange(x_min, x_max)
    ys = np.arange(y_min, y_max)
    zs = np.arange(z_min, z_max)

    x_grid, y_grid, z_grid = np.meshgrid(xs, ys, zs, indexing="ij")

    dist = np.sqrt(
        (x_grid - cx) ** 2
        + (y_grid - cy) ** 2
        + (z_grid - cz) ** 2
    )

    sphere_mask = dist <= radius
    output_arr[x_min:x_max, y_min:y_max, z_min:z_max][sphere_mask] = int(label)


def create_renal_hilum_centroid_spheres(
    input_nii_path: Path,
    output_nii_path: Path,
    summary_path: Path,
    target_labels: List[int] = TARGET_LABELS,
    sphere_radius_vox: float = SPHERE_RADIUS_VOX,
) -> Path:
    print("=" * 100)
    print("[FIND RENAL HILUM CENTROID SPHERES]")
    print(f"[INPUT ] {input_nii_path}")
    print(f"[OUTPUT] {output_nii_path}")
    print(f"[SUMMARY] {summary_path}")
    print(f"[TARGET LABELS] {target_labels}")
    print(f"[SPHERE RADIUS VOX] {sphere_radius_vox}")
    print("=" * 100)

    ensure_dir(output_nii_path.parent)

    if SKIP_IF_FINAL_EXISTS and output_nii_path.exists() and summary_path.exists():
        print(f"[SKIP] Output already exists: {output_nii_path}")
        print(f"[SKIP] Summary already exists: {summary_path}")
        return output_nii_path

    nii, data_float = load_nifti(input_nii_path)
    data = np.rint(data_float).astype(np.int32)

    output = np.zeros(data.shape, dtype=np.uint8)

    print(f"[INFO] Image shape: {data.shape}")

    summary_records = []

    for label in target_labels:
        print("\n" + "-" * 100)
        print(f"[INFO] Processing Segment_{label}")

        mask = data == int(label)
        voxel_count = int(mask.sum())

        if voxel_count == 0:
            print(f"[WARNING] Segment_{label} not found. Skipped.")
            summary_records.append({
                "label": label,
                "voxel_count": 0,
                "centroid_x": "NA",
                "centroid_y": "NA",
                "centroid_z": "NA",
                "status": "not_found",
            })
            continue

        centroid = compute_centroid(mask)
        if centroid is None:
            print(f"[WARNING] Segment_{label} centroid could not be computed. Skipped.")
            summary_records.append({
                "label": label,
                "voxel_count": voxel_count,
                "centroid_x": "NA",
                "centroid_y": "NA",
                "centroid_z": "NA",
                "status": "centroid_failed",
            })
            continue

        print(f"[INFO] Segment_{label} voxel count: {voxel_count:,}")
        print(f"[INFO] Segment_{label} centroid voxel coordinate: {centroid}")

        draw_sphere(
            output_arr=output,
            center=centroid,
            radius=sphere_radius_vox,
            label=int(label),
        )

        summary_records.append({
            "label": label,
            "voxel_count": voxel_count,
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),
            "status": "ok",
        })

    out_header = nii.header.copy()
    out_header.set_data_dtype(np.uint8)

    out_nii = nib.Nifti1Image(
        output.astype(np.uint8),
        affine=nii.affine,
        header=out_header,
    )
    nib.save(out_nii, str(output_nii_path))

    if not output_nii_path.exists():
        raise RuntimeError(f"Output was not created: {output_nii_path}")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("item\tvalue\n")
        f.write(f"input_nii\t{input_nii_path}\n")
        f.write(f"output_nii\t{output_nii_path}\n")
        f.write(f"target_labels\t{target_labels}\n")
        f.write(f"sphere_radius_vox\t{sphere_radius_vox}\n")
        f.write(f"sphere_diameter_vox\t{sphere_radius_vox * 2}\n")
        f.write(f"output_nonzero_voxels\t{int(np.count_nonzero(output))}\n")
        f.write("\n")
        f.write("label\tvoxel_count\tcentroid_x\tcentroid_y\tcentroid_z\tstatus\n")
        for r in summary_records:
            f.write(
                f"{r['label']}\t{r['voxel_count']}\t"
                f"{r['centroid_x']}\t{r['centroid_y']}\t{r['centroid_z']}\t{r['status']}\n"
            )

    print("\n" + "=" * 100)
    print("[DONE] Saved centroid sphere NIfTI:")
    print(output_nii_path)
    print("[DONE] Saved summary:")
    print(summary_path)
    print(f"[INFO] Output nonzero voxels: {int(np.count_nonzero(output)):,}")
    print("=" * 100)

    return output_nii_path


# ============================================================
# 5. Command-Line Interface for run.py
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create centroid spheres for Segment_2 and Segment_3 from AllSeg output."
    )

    parser.add_argument(
        "--ct_path",
        required=True,
        help="CT path passed by run.py. Accepted for compatibility but not directly used.",
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
# 6. Main Entry Point
# ============================================================
def main() -> None:
    args = parse_args()
    ct_path = Path(args.ct_path).resolve()

    case_id = args.case_id.strip() if args.case_id else strip_nii_extension(ct_path.name)
    if not case_id:
        raise ValueError("case_id is empty.")

    input_nii = build_input_nii_path(case_id)
    output_nii = build_output_nii_path(case_id)
    summary_path = build_summary_path(case_id)

    create_renal_hilum_centroid_spheres(
        input_nii_path=input_nii,
        output_nii_path=output_nii,
        summary_path=summary_path,
        target_labels=TARGET_LABELS,
        sphere_radius_vox=SPHERE_RADIUS_VOX,
    )


if __name__ == "__main__":
    main()
