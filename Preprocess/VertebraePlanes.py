import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import nibabel as nib


# ============================================================
# 1. Global Settings
# ============================================================
BASE_DIR = Path(r"C:\Users\User\Desktop\AbdVesselGen")
SEG_DONE_ROOT = BASE_DIR / "Seg_Done"
ANCHOR_POINTS_ROOT = BASE_DIR / "AnchorPoints"

# If both outputs already exist, skip this step.
SKIP_IF_FINAL_EXISTS = True

# Each tuple means:
#   (upper_segment_id, lower_segment_id, output_suffix)
#
# Output label convention inside each generated plane map:
#   Label 1 = highest Z-plane of upper_segment_id
#   Label 2 = lowest  Z-plane of lower_segment_id
PLANE_TASKS = [
    (32, 31, "seg32_highest_seg31_lowest_Zplanes"),
    (29, 28, "seg29_highest_seg28_lowest_Zplanes"),
]


# ============================================================
# 2. Path Builders
# ============================================================
def build_allseg_path(case_id: str) -> Path:
    """
    Example:
        C:/Users/User/Desktop/AbdVesselGen/Seg_Done/43_24055_All_segmentation/43_24055_All_segmentation.nii
    """
    return SEG_DONE_ROOT / f"{case_id}_All_segmentation" / f"{case_id}_All_segmentation.nii"


def build_output_path(case_id: str, suffix: str) -> Path:
    """
    Example:
        C:/Users/User/Desktop/AbdVesselGen/AnchorPoints/43_24055_seg32_highest_seg31_lowest_Zplanes.nii.gz
    """
    return ANCHOR_POINTS_ROOT / f"{case_id}_{suffix}.nii.gz"


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


def load_allseg(allseg_path: Path):
    if not allseg_path.exists():
        raise FileNotFoundError(f"AllSeg mask not found:\n{allseg_path}")

    img = nib.load(str(allseg_path))
    data = img.get_fdata().astype(np.int32)

    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI, but got shape: {data.shape}")

    return img, data


# ============================================================
# 4. Core Plane Generation
# ============================================================
def generate_z_plane_labelmap(
    allseg_img,
    allseg_data: np.ndarray,
    upper_segment_id: int,
    lower_segment_id: int,
    output_path: Path,
) -> Path:
    """
    Generate a 3D label map containing two axial planes:
      Label 1 = highest Z-plane of upper_segment_id
      Label 2 = lowest  Z-plane of lower_segment_id
    """
    print("=" * 100)
    print("[GENERATE VERTEBRAE-BASED Z PLANES]")
    print(f"[UPPER SEGMENT] Segment_{upper_segment_id}: highest Z")
    print(f"[LOWER SEGMENT] Segment_{lower_segment_id}: lowest Z")
    print(f"[OUTPUT       ] {output_path}")
    print("=" * 100)

    ensure_dir(output_path.parent)

    if SKIP_IF_FINAL_EXISTS and output_path.exists():
        print(f"[SKIP] Output already exists: {output_path}")
        return output_path

    upper_coords = np.argwhere(allseg_data == upper_segment_id)
    if len(upper_coords) == 0:
        raise ValueError(f"Segment_{upper_segment_id} was not found in the AllSeg mask.")

    lower_coords = np.argwhere(allseg_data == lower_segment_id)
    if len(lower_coords) == 0:
        raise ValueError(f"Segment_{lower_segment_id} was not found in the AllSeg mask.")

    upper_highest_z = int(upper_coords[:, 2].max())
    lower_lowest_z = int(lower_coords[:, 2].min())

    print(f"[RESULT] Segment_{upper_segment_id} highest Z slice = {upper_highest_z}")
    print(f"[RESULT] Segment_{lower_segment_id} lowest  Z slice = {lower_lowest_z}")

    output = np.zeros(allseg_data.shape, dtype=np.uint8)

    # Label 1: upper segment highest Z-plane.
    output[:, :, upper_highest_z] = 1

    # Label 2: lower segment lowest Z-plane.
    if lower_lowest_z == upper_highest_z:
        print(
            "[WARNING] The two Z-planes are on the same Z slice. "
            "Label 2 will overwrite Label 1 on that plane."
        )
    output[:, :, lower_lowest_z] = 2

    header = allseg_img.header.copy()
    header.set_data_dtype(np.uint8)

    out_img = nib.Nifti1Image(
        output,
        affine=allseg_img.affine,
        header=header,
    )

    nib.save(out_img, str(output_path))

    if not output_path.exists():
        raise RuntimeError(f"Output was not created: {output_path}")

    print("[SUCCESS] Z-plane label map saved:")
    print(output_path)
    print("[LABEL DEFINITION]")
    print(f"  Label 1 = Z-plane at Segment_{upper_segment_id} highest point")
    print(f"  Label 2 = Z-plane at Segment_{lower_segment_id} lowest point")
    print("=" * 100)

    return output_path


def generate_all_vertebrae_planes(case_id: str) -> Tuple[Path, Path]:
    allseg_path = build_allseg_path(case_id)

    print("=" * 100)
    print("[VERTEBRAE PLANE GENERATION]")
    print(f"[CASE ID] {case_id}")
    print(f"[ALLSEG ] {allseg_path}")
    print(f"[OUTPUT ROOT] {ANCHOR_POINTS_ROOT}")
    print("=" * 100)

    allseg_img, allseg_data = load_allseg(allseg_path)

    print(f"[INFO] AllSeg shape: {allseg_data.shape}")
    print(f"[INFO] AllSeg dtype after loading: {allseg_data.dtype}")

    output_paths = []
    for upper_segment_id, lower_segment_id, suffix in PLANE_TASKS:
        output_path = build_output_path(case_id, suffix)
        generated_path = generate_z_plane_labelmap(
            allseg_img=allseg_img,
            allseg_data=allseg_data,
            upper_segment_id=upper_segment_id,
            lower_segment_id=lower_segment_id,
            output_path=output_path,
        )
        output_paths.append(generated_path)

    print("=" * 100)
    print("[DONE] Vertebrae plane generation completed.")
    for path in output_paths:
        print(f"[OUTPUT] {path}")
    print("=" * 100)

    return tuple(output_paths)


# ============================================================
# 5. Command-Line Interface for run.py
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate vertebrae-based Z-plane label maps from the AllSeg mask."
    )

    # Kept for run.py compatibility.
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
# 6. Main Entry Point
# ============================================================
def main() -> None:
    args = parse_args()

    ct_path = Path(args.ct_path).resolve()
    case_id = args.case_id.strip() if args.case_id else strip_nii_extension(ct_path.name)

    if not case_id:
        raise ValueError("case_id is empty.")

    generate_all_vertebrae_planes(case_id=case_id)


if __name__ == "__main__":
    main()
