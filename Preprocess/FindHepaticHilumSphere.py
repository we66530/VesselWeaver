import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from scipy.spatial import KDTree


# ============================================================
# 1. Global Settings
# ============================================================
BASE_DIR = Path(r"C:\Users\User\Desktop\AbdVesselGen")
SEG_DONE_ROOT = BASE_DIR / "Seg_Done"
ANCHOR_POINTS_ROOT = BASE_DIR / "AnchorPoints"

# AllSeg portal vein / portal structure label.
PV_LABEL_IN_ALLSEG = 64

# Hepatic hilum sphere target liver segments.
# This follows HepaticHilum2.py: generate one sphere for liver segment 4 and one for liver segment 5.
TARGET_LIVER_SEGMENTS = [4, 5]

# Diameter in voxel units, following HepaticHilum2.py.
# diameter=20 -> radius=10 voxels.
SPHERE_DIAMETER_VOXELS = 20

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


def build_liver_segments_path(case_id: str) -> Path:
    """
    Example:
        C:/Users/User/Desktop/AbdVesselGen/Seg_Done/43_24055_liver_segments/43_24055_liver_segments.nii
    """
    return SEG_DONE_ROOT / f"{case_id}_liver_segments" / f"{case_id}_liver_segments.nii"


def build_output_path(case_id: str) -> Path:
    """
    Example:
        C:/Users/User/Desktop/AbdVesselGen/AnchorPoints/43_24055_hepatic_hilum_spheres.nii
    """
    return ANCHOR_POINTS_ROOT / f"{case_id}_hepatic_hilum_spheres.nii"


def build_summary_path(case_id: str) -> Path:
    return ANCHOR_POINTS_ROOT / f"{case_id}_hepatic_hilum_spheres.txt"


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


def warn_if_affine_mismatch(reference_name: str, reference_img, target_name: str, target_img) -> None:
    if not np.allclose(reference_img.affine, target_img.affine, atol=1e-4):
        print(f"[WARNING] {reference_name} affine does not exactly match {target_name} affine.")
        print("[WARNING] This script will continue because voxel shapes match.")
        print("[WARNING] Please verify spatial alignment if needed.\n")


# ============================================================
# 4. Sphere Drawing
# ============================================================
def draw_voxel_sphere(
    output: np.ndarray,
    center_voxel: np.ndarray,
    radius_voxels: float,
    label_value: int,
) -> None:
    """
    Draw a voxel-space sphere into output.

    This follows HepaticHilum2.py: sphere size is defined in voxel units, not mm.
    """
    center_voxel = np.asarray(center_voxel, dtype=np.float64)

    x_grid, y_grid, z_grid = np.ogrid[
        0:output.shape[0],
        0:output.shape[1],
        0:output.shape[2],
    ]

    dist_sq = (
        (x_grid - center_voxel[0]) ** 2
        + (y_grid - center_voxel[1]) ** 2
        + (z_grid - center_voxel[2]) ** 2
    )

    output[dist_sq <= radius_voxels ** 2] = int(label_value)


# ============================================================
# 5. Core Processing Function
# ============================================================
def create_hepatic_hilum_spheres(
    allseg_path: Path,
    liver_segments_path: Path,
    output_path: Path,
    summary_path: Path,
    pv_label: int = PV_LABEL_IN_ALLSEG,
    target_liver_segments: List[int] = TARGET_LIVER_SEGMENTS,
    sphere_diameter_voxels: float = SPHERE_DIAMETER_VOXELS,
) -> Path:
    """
    Generate hepatic hilum anchor spheres using the HepaticHilum2.py logic:

    1. Load AllSeg and liver segment mask.
    2. Build PV mask from AllSeg label 64.
    3. Remove liver segments 4 and 5 from PV to obtain residual PV.
    4. For liver segment 4:
       - find residual PV candidates close to segment 4
       - choose the candidate with score = y - z minimized, matching the original script's behavior
    5. For liver segment 5:
       - choose the residual PV point with minimum distance to segment 5
    6. Draw two spheres:
       - Label 1 = segment 4 interface sphere
       - Label 2 = segment 5 interface sphere
    """
    ensure_dir(output_path.parent)

    print("=" * 100)
    print("[CREATE HEPATIC HILUM SPHERES - TWO-SPHERE VERSION]")
    print(f"[ALLSEG      ] {allseg_path}")
    print(f"[LIVER SEG   ] {liver_segments_path}")
    print(f"[OUTPUT      ] {output_path}")
    print(f"[SUMMARY     ] {summary_path}")
    print(f"[PV LABEL    ] {pv_label}")
    print(f"[TARGET SEGS ] {target_liver_segments}")
    print(f"[DIAMETER VX ] {sphere_diameter_voxels}")
    print("=" * 100)

    if SKIP_IF_FINAL_EXISTS and output_path.exists() and summary_path.exists():
        print(f"[SKIP] Output already exists: {output_path}")
        print(f"[SKIP] Summary already exists: {summary_path}")
        return output_path

    allseg_img, allseg_data = load_nifti(allseg_path)
    liver_img, liver_data = load_nifti(liver_segments_path)

    check_same_shape("AllSeg", allseg_data, "LiverSegments", liver_data)
    warn_if_affine_mismatch("AllSeg", allseg_img, "LiverSegments", liver_img)

    allseg_int = np.rint(allseg_data).astype(np.int32)
    liver_int = np.rint(liver_data).astype(np.int32)

    pv_mask = allseg_int == int(pv_label)
    liver_4_5_mask = np.isin(liver_int, target_liver_segments)

    residual_pv_mask = pv_mask & (~liver_4_5_mask)
    residual_pv_coords = np.argwhere(residual_pv_mask)

    print(f"[INFO] PV label {pv_label} voxels: {int(pv_mask.sum()):,}")
    print(f"[INFO] Liver segment 4/5 voxels: {int(liver_4_5_mask.sum()):,}")
    print(f"[INFO] Residual PV voxels after excluding liver 4/5: {len(residual_pv_coords):,}")

    if len(residual_pv_coords) == 0:
        raise RuntimeError(
            "Residual PV is empty after excluding liver segments 4 and 5.\n"
            "Please check whether AllSeg label 64 and liver segments 4/5 are aligned."
        )

    output_spheres = np.zeros(allseg_data.shape, dtype=np.uint8)
    radius = float(sphere_diameter_voxels) / 2.0

    records: List[Dict] = []

    for out_label, seg_id in enumerate(target_liver_segments, start=1):
        print("\n" + "-" * 100)
        print(f"[TARGET] Liver Segment {seg_id}")

        seg_coords = np.argwhere(liver_int == int(seg_id))
        if len(seg_coords) == 0:
            print(f"[WARNING] Liver Segment {seg_id} not found. Skipping this sphere.")
            continue

        tree_seg = KDTree(seg_coords)
        distances, nearest_indices = tree_seg.query(residual_pv_coords)
        min_dist = float(np.min(distances))

        if int(seg_id) == 4:
            # Original HepaticHilum2.py special filtering for segment 4.
            threshold = min_dist + 1.5
            candidate_indices = np.where(distances <= threshold)[0]
            candidates = residual_pv_coords[candidate_indices]

            if len(candidates) == 0:
                raise RuntimeError("No segment-4 candidates found despite non-empty residual PV.")

            # Original code used: scores = candidates[:, 1] - candidates[:, 2]
            # and chose argmin(scores). Keep this exact behavior for compatibility.
            scores = candidates[:, 1] - candidates[:, 2]
            best_point = candidates[int(np.argmin(scores))].astype(np.int64)
            selection_rule = "seg4_special_score_min_y_minus_z_among_near_interface_candidates"
            print(f"[INFO] Segment 4 candidate threshold: min_dist + 1.5 = {threshold:.4f}")
            print(f"[INFO] Segment 4 candidates: {len(candidates):,}")
            print(f"[INFO] Selected point by y-z score: {tuple(int(x) for x in best_point.tolist())}")

        else:
            best_idx = int(np.argmin(distances))
            best_point = residual_pv_coords[best_idx].astype(np.int64)
            selection_rule = "minimum_distance_to_liver_segment"
            print(f"[INFO] Selected minimum-distance point: {tuple(int(x) for x in best_point.tolist())}")

        nearest_liver_index = int(nearest_indices[int(np.argmin(np.linalg.norm(residual_pv_coords - best_point[None, :], axis=1)))])
        nearest_liver_point = seg_coords[nearest_liver_index].astype(np.int64)

        draw_voxel_sphere(
            output=output_spheres,
            center_voxel=best_point,
            radius_voxels=radius,
            label_value=out_label,
        )

        print(f"[INFO] Output label {out_label} sphere drawn at: {tuple(int(x) for x in best_point.tolist())}")
        print(f"[INFO] Closest distance to liver segment {seg_id}: {min_dist:.4f} voxels")

        records.append({
            "output_label": out_label,
            "target_liver_segment": int(seg_id),
            "selection_rule": selection_rule,
            "center_x": int(best_point[0]),
            "center_y": int(best_point[1]),
            "center_z": int(best_point[2]),
            "nearest_liver_x": int(nearest_liver_point[0]),
            "nearest_liver_y": int(nearest_liver_point[1]),
            "nearest_liver_z": int(nearest_liver_point[2]),
            "min_distance_voxels": float(min_dist),
        })

    if len(records) == 0:
        raise RuntimeError("No hepatic hilum sphere was generated.")

    header = allseg_img.header.copy()
    header.set_data_dtype(np.uint8)

    out_img = nib.Nifti1Image(
        output_spheres.astype(np.uint8),
        affine=allseg_img.affine,
        header=header,
    )
    nib.save(out_img, str(output_path))

    if not output_path.exists():
        raise RuntimeError(f"Output was not created: {output_path}")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("item\tvalue\n")
        f.write(f"allseg_path\t{allseg_path}\n")
        f.write(f"liver_segments_path\t{liver_segments_path}\n")
        f.write(f"pv_label_allseg\t{pv_label}\n")
        f.write(f"target_liver_segments\t{target_liver_segments}\n")
        f.write(f"sphere_diameter_voxels\t{sphere_diameter_voxels}\n")
        f.write(f"sphere_radius_voxels\t{radius}\n")
        f.write(f"total_sphere_voxels\t{int(np.count_nonzero(output_spheres))}\n")
        f.write("\n")
        f.write("output_label\ttarget_liver_segment\tselection_rule\tcenter_x\tcenter_y\tcenter_z\tnearest_liver_x\tnearest_liver_y\tnearest_liver_z\tmin_distance_voxels\n")
        for r in records:
            f.write(
                f"{r['output_label']}\t{r['target_liver_segment']}\t{r['selection_rule']}\t"
                f"{r['center_x']}\t{r['center_y']}\t{r['center_z']}\t"
                f"{r['nearest_liver_x']}\t{r['nearest_liver_y']}\t{r['nearest_liver_z']}\t"
                f"{r['min_distance_voxels']:.6f}\n"
            )

    print("\n" + "=" * 100)
    print("[DONE] Hepatic hilum spheres saved:")
    print(output_path)
    print("[DONE] Summary saved:")
    print(summary_path)
    print(f"[INFO] Positive output voxels: {int(np.count_nonzero(output_spheres)):,}")
    print(f"[INFO] Output labels present: {[int(x) for x in np.unique(output_spheres) if x > 0]}")
    print("=" * 100)

    return output_path


# ============================================================
# 6. Command-Line Interface for run.py
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create two hepatic hilum spheres from AllSeg label 64 interfaces with liver segments 4 and 5."
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

    return parser.parse_args()


# ============================================================
# 7. Main Entry Point
# ============================================================
def main() -> None:
    args = parse_args()
    ct_path = Path(args.ct_path).resolve()
    case_id = args.case_id.strip() if args.case_id else strip_nii_extension(ct_path.name)

    if not case_id:
        raise ValueError("case_id is empty.")

    allseg_path = build_allseg_path(case_id)
    liver_segments_path = build_liver_segments_path(case_id)
    output_path = build_output_path(case_id)
    summary_path = build_summary_path(case_id)

    create_hepatic_hilum_spheres(
        allseg_path=allseg_path,
        liver_segments_path=liver_segments_path,
        output_path=output_path,
        summary_path=summary_path,
        pv_label=PV_LABEL_IN_ALLSEG,
        target_liver_segments=TARGET_LIVER_SEGMENTS,
        sphere_diameter_voxels=SPHERE_DIAMETER_VOXELS,
    )


if __name__ == "__main__":
    main()
