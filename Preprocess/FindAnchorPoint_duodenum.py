import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from tqdm import tqdm


# ============================================================
# 1. Global Settings
# ============================================================
BASE_DIR = Path(r"C:\Users\User\Desktop\AbdVesselGen")
ANCHOR_POINTS_ROOT = BASE_DIR / "AnchorPoints"

# Input produced by InteriorFaceFinder_duodenum.py
INPUT_INNER_SURFACE_SUFFIX = "bowel_inner_surface_toward_dynamic_center_duodenum"

# Output name kept simple and explicit.
OUTPUT_SUFFIX = "duodenum_surface_anchor_spheres"

# Number of sampled points on the inner surface.
NUM_POINTS = 3

# Discrete sphere radius in voxels.
# Note: radius=2 gives a diameter of 5 voxels.
# Your original script used SPHERE_RADIUS = 5, so this version preserves that value.
SPHERE_RADIUS = 5

# Output single-segment label value.
OUTPUT_LABEL = 1

# Fixed random seed for reproducibility.
RANDOM_SEED = 42

# If final output already exists, skip this step.
SKIP_IF_FINAL_EXISTS = True


# ============================================================
# 2. Path Builders
# ============================================================
def build_input_inner_surface_path(case_id: str) -> Path:
    """
    Example:
        C:/Users/User/Desktop/AbdVesselGen/AnchorPoints/43_24055_bowel_inner_surface_toward_dynamic_center_duodenum.nii.gz
    """
    return ANCHOR_POINTS_ROOT / f"{case_id}_{INPUT_INNER_SURFACE_SUFFIX}.nii.gz"


def build_output_point_spheres_path(case_id: str) -> Path:
    """
    Example:
        C:/Users/User/Desktop/AbdVesselGen/AnchorPoints/43_24055_duodenum_surface_anchor_spheres.nii.gz
    """
    return ANCHOR_POINTS_ROOT / f"{case_id}_{OUTPUT_SUFFIX}.nii.gz"


def build_output_points_txt_path(case_id: str) -> Path:
    """
    Optional TXT summary of sampled point coordinates.
    """
    return ANCHOR_POINTS_ROOT / f"{case_id}_{OUTPUT_SUFFIX}.txt"


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
# 4. Sphere Utilities
# ============================================================
def create_sphere_offsets(radius: int) -> np.ndarray:
    """
    Create 3D sphere offsets with the given voxel radius.
    """
    radius = int(radius)
    offsets = []

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx * dx + dy * dy + dz * dz <= radius * radius:
                    offsets.append((dx, dy, dz))

    return np.array(offsets, dtype=np.int16)


# ============================================================
# 5. Farthest Point Sampling
# ============================================================
def farthest_point_sampling(points: np.ndarray, num_samples: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """
    Select spatially distributed points from surface voxels using farthest point sampling.

    Args:
        points: (N, 3) voxel coordinates.
        num_samples: number of points to sample.

    Returns:
        sampled_points: (num_samples, 3) voxel coordinates.
    """
    rng = np.random.default_rng(seed)
    n_points = len(points)

    if n_points == 0:
        raise RuntimeError("No surface voxels found.")

    if n_points <= num_samples:
        print(
            f"[WARNING] Surface only contains {n_points} voxels, "
            f"which is <= requested {num_samples}. Returning all available voxels."
        )
        return points.copy()

    points_f = points.astype(np.float32)

    first_idx = int(rng.integers(0, n_points))
    selected_indices = [first_idx]

    first_point = points_f[first_idx]
    min_dist_sq = np.sum((points_f - first_point) ** 2, axis=1)

    for _ in tqdm(range(1, num_samples), desc="Farthest Point Sampling", unit="point"):
        next_idx = int(np.argmax(min_dist_sq))
        selected_indices.append(next_idx)

        new_point = points_f[next_idx]
        dist_sq_to_new = np.sum((points_f - new_point) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq_to_new)

    sampled_points = points[selected_indices]
    return sampled_points


# ============================================================
# 6. Draw Spheres
# ============================================================
def draw_spheres(
    shape: tuple[int, int, int],
    centers: np.ndarray,
    sphere_offsets: np.ndarray,
    label_value: int = OUTPUT_LABEL,
) -> np.ndarray:
    """
    Draw all sampled-point spheres into one labelmap.
    """
    output = np.zeros(shape, dtype=np.uint8)
    shape_arr = np.array(shape, dtype=np.int64)

    for center in tqdm(centers, desc="Drawing spheres", unit="sphere"):
        sphere_coords = center[None, :] + sphere_offsets

        valid = np.all(
            (sphere_coords >= 0)
            & (sphere_coords < shape_arr[None, :]),
            axis=1,
        )

        sphere_coords = sphere_coords[valid]

        output[
            sphere_coords[:, 0],
            sphere_coords[:, 1],
            sphere_coords[:, 2],
        ] = label_value

    return output


# ============================================================
# 7. Main Processing Function
# ============================================================
def generate_duodenum_surface_anchor_spheres(
    input_inner_surface_path: Path,
    output_spheres_path: Path,
    output_points_txt_path: Path,
) -> Path:
    print("=" * 100)
    print("[GENERATE DUODENUM SURFACE ANCHOR SPHERES]")
    print(f"[INPUT ] {input_inner_surface_path}")
    print(f"[OUTPUT] {output_spheres_path}")
    print(f"[TXT   ] {output_points_txt_path}")
    print(f"[NUM POINTS] {NUM_POINTS}")
    print(f"[SPHERE RADIUS] {SPHERE_RADIUS}")
    print("=" * 100)

    ensure_dir(output_spheres_path.parent)

    if SKIP_IF_FINAL_EXISTS and output_spheres_path.exists() and output_points_txt_path.exists():
        print(f"[SKIP] Output already exists: {output_spheres_path}")
        print(f"[SKIP] TXT already exists: {output_points_txt_path}")
        return output_spheres_path

    nii, inner_surface_data = load_nifti(input_inner_surface_path)

    inner_surface_mask = inner_surface_data > 0
    surface_coords = np.argwhere(inner_surface_mask)

    print(f"[INFO] Volume shape: {inner_surface_mask.shape}")
    print(f"[INFO] Inner surface voxel count: {len(surface_coords):,}")

    if len(surface_coords) == 0:
        raise RuntimeError("The inner surface mask contains no nonzero voxels.")

    print(f"[INFO] Sampling {NUM_POINTS} spatially distributed surface points...")
    sampled_points = farthest_point_sampling(
        surface_coords,
        NUM_POINTS,
        seed=RANDOM_SEED,
    )

    print("[INFO] Sampled point coordinates:")
    for i, p in enumerate(sampled_points, start=1):
        print(f"  Point {i:02d}: ({int(p[0])}, {int(p[1])}, {int(p[2])})")

    print(f"[INFO] Creating sphere offsets with radius={SPHERE_RADIUS} voxel...")
    sphere_offsets = create_sphere_offsets(SPHERE_RADIUS)
    print(f"[INFO] Voxels per sphere template: {len(sphere_offsets)}")

    output = draw_spheres(
        shape=inner_surface_mask.shape,
        centers=sampled_points,
        sphere_offsets=sphere_offsets,
        label_value=OUTPUT_LABEL,
    )

    print(f"[INFO] Total nonzero voxels in sphere mask: {int(np.sum(output > 0)):,}")

    header = nii.header.copy()
    header.set_data_dtype(np.uint8)

    out_nii = nib.Nifti1Image(
        output.astype(np.uint8),
        affine=nii.affine,
        header=header,
    )

    nib.save(out_nii, str(output_spheres_path))

    if not output_spheres_path.exists():
        raise RuntimeError(f"Output was not created: {output_spheres_path}")

    with open(output_points_txt_path, "w", encoding="utf-8") as f:
        f.write("point_index\tx\ty\tz\n")
        for i, p in enumerate(sampled_points, start=1):
            f.write(f"{i}\t{int(p[0])}\t{int(p[1])}\t{int(p[2])}\n")

    print("=" * 100)
    print("[DONE] Saved duodenum surface anchor spheres to:")
    print(output_spheres_path)
    print("[DONE] Saved sampled point coordinates to:")
    print(output_points_txt_path)
    print("=" * 100)

    return output_spheres_path


# ============================================================
# 8. Command-Line Interface for run.py
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate anchor-point spheres on the dynamic-center duodenum inner surface."
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
# 9. Main Entry Point
# ============================================================
def main() -> None:
    args = parse_args()

    ct_path = Path(args.ct_path).resolve()
    case_id = args.case_id.strip() if args.case_id else strip_nii_extension(ct_path.name)

    if not case_id:
        raise ValueError("case_id is empty.")

    input_inner_surface_path = build_input_inner_surface_path(case_id)
    output_spheres_path = build_output_point_spheres_path(case_id)
    output_points_txt_path = build_output_points_txt_path(case_id)

    generate_duodenum_surface_anchor_spheres(
        input_inner_surface_path=input_inner_surface_path,
        output_spheres_path=output_spheres_path,
        output_points_txt_path=output_points_txt_path,
    )


if __name__ == "__main__":
    main()
