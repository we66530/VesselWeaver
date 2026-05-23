import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage

try:
    import nrrd
except Exception as e:
    nrrd = None
    NRRD_IMPORT_ERROR = e
else:
    NRRD_IMPORT_ERROR = None

try:
    from nilearn.image import resample_to_img
    HAS_NILEARN = True
except Exception:
    HAS_NILEARN = False


# ============================================================
# 1. Global Settings
# ============================================================
BASE_DIR = Path(r"C:\Users\User\Desktop\AbdVesselGen")
RESULTS_ROOT = BASE_DIR / "Results"
RESULTS_INTERMEDIATE_ROOT = RESULTS_ROOT / "Intermediate"
SEG_DONE_ROOT = BASE_DIR / "Seg_Done"

OUTPUT_SEG_NAME = "AbdominalVessels.seg.nrrd"
SUMMARY_NAME = "AbdominalVessels_summary.txt"

CONNECTIVITY_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)

# For GDA/hepatic masks:
# - If multiple positive labels exist, preserve labels as individual segments.
# - If only one positive label exists but it contains multiple disconnected components,
#   split connected components so GDA_1, GDA_2, ... can be generated.
SPLIT_BINARY_MASK_INTO_CONNECTED_COMPONENTS = True

# Small components below this size are ignored for GDA/hepatic auto-splitting.
MIN_COMPONENT_VOXELS = 1

# If final output already exists, skip this step.
SKIP_IF_FINAL_EXISTS = True


COLORS = [
    [0.8, 0.0, 0.0],
    [0.0, 0.7, 0.0],
    [0.0, 0.2, 0.9],
    [0.9, 0.7, 0.0],
    [0.7, 0.0, 0.9],
    [0.0, 0.8, 0.8],
    [1.0, 0.45, 0.0],
    [0.45, 1.0, 0.0],
    [0.0, 0.5, 1.0],
    [1.0, 0.0, 0.45],
    [0.45, 0.0, 1.0],
    [0.2, 0.8, 0.2],
    [0.8, 0.4, 0.4],
    [0.4, 0.8, 0.4],
    [0.4, 0.4, 0.8],
]


# ============================================================
# 2. Path Builders
# ============================================================
def build_gda_path(case_id: str) -> Path:
    return RESULTS_INTERMEDIATE_ROOT / f"{case_id}_GDA.nii.gz"


def build_hepatic_vessels_path(case_id: str) -> Path:
    return RESULTS_INTERMEDIATE_ROOT / f"{case_id}_hepatic_vessels.nii.gz"


def build_renal_paths_path(case_id: str) -> Path:
    return RESULTS_INTERMEDIATE_ROOT / f"{case_id}_renal_artery_vein_paths.nii.gz"


def build_splenic_artery_path(case_id: str) -> Path:
    return RESULTS_INTERMEDIATE_ROOT / f"{case_id}_splenic_artery_path.nii.gz"


def build_portal_vein_path(case_id: str) -> Path:
    return SEG_DONE_ROOT / f"{case_id}_veins_segmentation" / f"{case_id}_pred_skeleton_HUge30_top1_width4.nii.gz"


def build_output_seg_path(case_id: str) -> Path:
    return RESULTS_ROOT / OUTPUT_SEG_NAME


def build_summary_path(case_id: str) -> Path:
    return RESULTS_ROOT / f"{case_id}_{SUMMARY_NAME}"


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
        raise FileNotFoundError(f"NIfTI file not found:\n{path}")
    img = nib.load(str(path))
    data = img.get_fdata()
    return img, data


def load_and_align_mask_to_reference(
    mask_path: Path,
    reference_img,
    reference_shape: Tuple[int, int, int],
    mask_name: str,
) -> np.ndarray:
    """
    Load one mask and align it to the original CT geometry if needed.

    The output array always matches reference_shape.
    """
    mask_img, mask_data = load_nifti(mask_path)

    same_shape = mask_data.shape == reference_shape
    same_affine = np.allclose(mask_img.affine, reference_img.affine, atol=1e-4)

    print("-" * 100)
    print(f"[LOAD MASK] {mask_name}")
    print(f"[PATH] {mask_path}")
    print(f"[MASK SHAPE] {mask_data.shape}")
    print(f"[REF  SHAPE] {reference_shape}")
    print(f"[SAME SHAPE] {same_shape}")
    print(f"[SAME AFFINE] {same_affine}")

    if same_shape and same_affine:
        aligned_data = mask_data
        print(f"[INFO] {mask_name} already matches reference CT geometry.")
    else:
        print(f"[WARNING] {mask_name} geometry differs from reference CT.")
        print("[INFO] Resampling with nearest-neighbor interpolation...")

        if not HAS_NILEARN:
            raise RuntimeError(
                f"{mask_name} requires resampling but nilearn is not available.\n"
                "Install it with:\n"
                "    pip install nilearn\n"
                "or make sure all masks match the original CT geometry."
            )

        aligned_img = resample_to_img(
            source_img=mask_img,
            target_img=reference_img,
            interpolation="nearest",
        )
        aligned_data = aligned_img.get_fdata()
        print(f"[INFO] Resampled shape: {aligned_data.shape}")

    if aligned_data.shape != reference_shape:
        raise RuntimeError(
            f"After alignment, {mask_name} shape {aligned_data.shape} still does not match "
            f"reference CT shape {reference_shape}."
        )

    return aligned_data


# ============================================================
# 4. Segment Extraction Helpers
# ============================================================
def extract_labeled_or_connected_segments(
    data_float: np.ndarray,
    base_name: str,
    split_binary: bool = SPLIT_BINARY_MASK_INTO_CONNECTED_COMPONENTS,
) -> List[Dict]:
    """
    Extract segment masks from a labelmap.

    Rules:
      1. If multiple positive labels exist, each label is one segment.
      2. If only one positive label exists and split_binary=True, split into
         connected components.
      3. Segment names become base_name_1, base_name_2, ... in sorted order.
    """
    data_int = np.rint(data_float).astype(np.int32)
    positive_labels = [int(x) for x in np.unique(data_int) if int(x) > 0]

    segments: List[Dict] = []

    if len(positive_labels) == 0:
        print(f"[WARNING] {base_name}: no positive labels found. Skipped.")
        return segments

    if len(positive_labels) == 1 and split_binary:
        binary = data_int > 0
        labeled, num_components = ndimage.label(binary, structure=CONNECTIVITY_STRUCTURE)
        print(f"[INFO] {base_name}: binary-like mask. Connected components found: {num_components}")

        component_records = []
        for component_id in range(1, num_components + 1):
            mask = labeled == component_id
            voxels = int(mask.sum())
            if voxels < MIN_COMPONENT_VOXELS:
                continue
            coords = np.argwhere(mask)
            centroid = coords.astype(np.float64).mean(axis=0)
            component_records.append({
                "source_label": component_id,
                "mask": mask,
                "voxel_count": voxels,
                "centroid": centroid,
                "mean_z": float(centroid[2]),
            })

        # Keep deterministic anatomical-ish order: high Z to low Z.
        component_records = sorted(component_records, key=lambda r: r["mean_z"], reverse=True)

        for i, rec in enumerate(component_records, start=1):
            rec["name"] = f"{base_name}_{i}"
            segments.append(rec)

    else:
        print(f"[INFO] {base_name}: preserving positive labels: {positive_labels}")
        label_records = []

        for label_value in positive_labels:
            mask = data_int == label_value
            voxels = int(mask.sum())
            if voxels < MIN_COMPONENT_VOXELS:
                continue
            coords = np.argwhere(mask)
            centroid = coords.astype(np.float64).mean(axis=0)
            label_records.append({
                "source_label": label_value,
                "mask": mask,
                "voxel_count": voxels,
                "centroid": centroid,
                "mean_z": float(centroid[2]),
            })

        # Preserve numeric label order for already-labeled paths.
        label_records = sorted(label_records, key=lambda r: int(r["source_label"]))

        for i, rec in enumerate(label_records, start=1):
            rec["name"] = f"{base_name}_{i}"
            segments.append(rec)

    for seg in segments:
        c = seg["centroid"]
        print(
            f"[ADD] {seg['name']}: source_label={seg['source_label']}, "
            f"voxels={seg['voxel_count']:,}, centroid=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})"
        )

    return segments


def extract_fixed_label_segments(
    data_float: np.ndarray,
    label_name_map: Dict[int, str],
    source_group: str,
) -> List[Dict]:
    """
    Extract specifically named labels from a labelmap.

    Missing labels are skipped with warning.
    """
    data_int = np.rint(data_float).astype(np.int32)
    positive_labels = [int(x) for x in np.unique(data_int) if int(x) > 0]
    print(f"[INFO] {source_group}: positive labels found: {positive_labels}")

    segments: List[Dict] = []

    for label_value, segment_name in label_name_map.items():
        mask = data_int == int(label_value)
        voxels = int(mask.sum())

        if voxels == 0:
            print(f"[WARNING] {source_group}: label {label_value} ({segment_name}) not found or empty. Skipped.")
            continue

        coords = np.argwhere(mask)
        centroid = coords.astype(np.float64).mean(axis=0)

        seg = {
            "name": segment_name,
            "source_label": int(label_value),
            "mask": mask,
            "voxel_count": voxels,
            "centroid": centroid,
            "mean_z": float(centroid[2]),
        }
        segments.append(seg)

        print(
            f"[ADD] {segment_name}: source_label={label_value}, voxels={voxels:,}, "
            f"centroid=({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})"
        )

    return segments


def extract_single_segment(
    data_float: np.ndarray,
    segment_name: str,
    source_group: str,
) -> List[Dict]:
    """
    Convert all positive voxels into one segment.

    This is used for Splenic_artery and Portal_Vein_System.
    """
    mask = data_float > 0
    voxels = int(mask.sum())

    if voxels == 0:
        print(f"[WARNING] {source_group}: empty mask. Skipped.")
        return []

    coords = np.argwhere(mask)
    centroid = coords.astype(np.float64).mean(axis=0)

    seg = {
        "name": segment_name,
        "source_label": 1,
        "mask": mask,
        "voxel_count": voxels,
        "centroid": centroid,
        "mean_z": float(centroid[2]),
    }

    print(
        f"[ADD] {segment_name}: voxels={voxels:,}, "
        f"centroid=({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})"
    )

    return [seg]


# ============================================================
# 5. Branch Collection
# ============================================================
def collect_branch_segments(
    case_id: str,
    reference_img,
    reference_shape: Tuple[int, int, int],
) -> List[Dict]:
    all_segments: List[Dict] = []

    # --------------------------------------------------------
    # GDA: GDA_1, GDA_2, GDA_3, ...
    # --------------------------------------------------------
    gda_path = build_gda_path(case_id)
    gda_data = load_and_align_mask_to_reference(gda_path, reference_img, reference_shape, "GDA")
    all_segments.extend(
        extract_labeled_or_connected_segments(
            data_float=gda_data,
            base_name="GDA",
            split_binary=True,
        )
    )

    # --------------------------------------------------------
    # Hepatic artery: Hepatic_artery_1, Hepatic_artery_2, ...
    # --------------------------------------------------------
    hepatic_path = build_hepatic_vessels_path(case_id)
    hepatic_data = load_and_align_mask_to_reference(hepatic_path, reference_img, reference_shape, "hepatic_vessels")
    all_segments.extend(
        extract_labeled_or_connected_segments(
            data_float=hepatic_data,
            base_name="Hepatic_artery",
            split_binary=True,
        )
    )

    # --------------------------------------------------------
    # Renal artery/vein paths.
    # User-defined naming rule:
    #   Segment_1 = Lt_Renal_vein
    #   Segment_2 = Rt_Renal_vein
    #   Segment_3 = Lt_Renal_artery
    #   Segment_4 = Rt_Renal_artery
    # --------------------------------------------------------
    renal_path = build_renal_paths_path(case_id)
    renal_data = load_and_align_mask_to_reference(renal_path, reference_img, reference_shape, "renal_artery_vein_paths")
    renal_name_map = {
        1: "Lt_Renal_vein",
        2: "Rt_Renal_vein",
        3: "Lt_Renal_artery",
        4: "Rt_Renal_artery",
    }
    all_segments.extend(
        extract_fixed_label_segments(
            data_float=renal_data,
            label_name_map=renal_name_map,
            source_group="renal_artery_vein_paths",
        )
    )

    # --------------------------------------------------------
    # Splenic artery: one segment named Splenic_artery.
    # --------------------------------------------------------
    splenic_path = build_splenic_artery_path(case_id)
    splenic_data = load_and_align_mask_to_reference(splenic_path, reference_img, reference_shape, "splenic_artery_path")
    all_segments.extend(
        extract_single_segment(
            data_float=splenic_data,
            segment_name="Splenic_artery",
            source_group="splenic_artery_path",
        )
    )

    # --------------------------------------------------------
    # Portal vein system: one segment named Portal_Vein_System.
    # --------------------------------------------------------
    portal_path = build_portal_vein_path(case_id)
    portal_data = load_and_align_mask_to_reference(portal_path, reference_img, reference_shape, "portal_vein_system")
    all_segments.extend(
        extract_single_segment(
            data_float=portal_data,
            segment_name="Portal_Vein_System",
            source_group="portal_vein_system",
        )
    )

    if not all_segments:
        raise RuntimeError("No branch segments were collected. Nothing to write.")

    # Assign final layer order.
    for i, seg in enumerate(all_segments):
        seg["segment_index"] = i
        seg["segment_id"] = f"Segment_{i + 1}"
        seg["segment_layer"] = i
        seg["segment_label_value"] = 1

    print("=" * 100)
    print("[FINAL SEGMENT LIST]")
    for seg in all_segments:
        c = seg["centroid"]
        print(
            f"Layer {seg['segment_layer']:02d}: {seg['name']} | voxels={seg['voxel_count']:,} | "
            f"centroid=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})"
        )
    print("=" * 100)

    return all_segments


# ============================================================
# 6. Slicer .seg.nrrd Writer
#    Follows the proven CombineName.py style:
#       seg_data shape = (segments, X, Y, Z)
#       kinds = ["list", "domain", "domain", "domain"]
#       space directions = [[nan,nan,nan], affine columns...]
#
#    This allows overlapping segments because each segment is an independent layer.
# ============================================================
def build_segmentation_array(segments: List[Dict]) -> np.ndarray:
    segment_arrays = []

    for seg in segments:
        mask = seg["mask"].astype(np.uint8)
        if int(mask.sum()) == 0:
            print(f"[SKIP] Empty final segment: {seg['name']}")
            continue
        segment_arrays.append(mask)

    if not segment_arrays:
        raise RuntimeError("All final branch masks are empty.")

    seg_data = np.stack(segment_arrays, axis=0).astype(np.uint8)
    print(f"[INFO] Output seg_data shape: {seg_data.shape}")
    return seg_data


def build_seg_nrrd_header(
    segments: List[Dict],
    reference_affine: np.ndarray,
) -> Dict:
    header: Dict = {
        "type": "uint8",
        "dimension": 4,
        "encoding": "gzip",
        "space": "right-anterior-superior",
        "space origin": reference_affine[:3, 3].astype(float),
        "space directions": np.vstack([
            [np.nan, np.nan, np.nan],
            reference_affine[:3, 0],
            reference_affine[:3, 1],
            reference_affine[:3, 2],
        ]).astype(float),
        "kinds": ["list", "domain", "domain", "domain"],
        "Segmentation_MasterRepresentation": "Binary labelmap",
        "Segmentation_ContainedRepresentationNames": "Binary labelmap",
    }

    for i, seg in enumerate(segments):
        mask = seg["mask"]
        nz = np.argwhere(mask > 0)

        if nz.size == 0:
            continue

        xmin, ymin, zmin = nz.min(axis=0)
        xmax, ymax, zmax = nz.max(axis=0)

        color = COLORS[i % len(COLORS)]
        prefix = f"Segment{i}"

        header[f"{prefix}_ID"] = f"Segment_{i + 1}"
        header[f"{prefix}_Name"] = seg["name"]
        header[f"{prefix}_NameAutoGenerated"] = "0"
        header[f"{prefix}_Color"] = f"{color[0]} {color[1]} {color[2]}"
        header[f"{prefix}_ColorAutoGenerated"] = "0"
        header[f"{prefix}_LabelValue"] = "1"
        header[f"{prefix}_Layer"] = str(i)
        header[f"{prefix}_Extent"] = f"{xmin} {xmax} {ymin} {ymax} {zmin} {zmax}"

    return header


def save_seg_nrrd(
    output_path: Path,
    seg_data: np.ndarray,
    header: Dict,
) -> None:
    if nrrd is None:
        raise ImportError(
            "The 'pynrrd' package is required to write .seg.nrrd files.\n"
            "Install it with:\n"
            "    pip install pynrrd\n"
            f"Original import error: {NRRD_IMPORT_ERROR}"
        )

    ensure_dir(output_path.parent)

    if output_path.exists():
        output_path.unlink()

    # Same style as the provided CombineName.py.
    nrrd.write(str(output_path), seg_data.astype(np.uint8), header)

    if not output_path.exists():
        raise RuntimeError(f"Output .seg.nrrd was not created: {output_path}")


# ============================================================
# 7. Main Function
# ============================================================
def combine_abdominal_vessel_branches(
    case_id: str,
    reference_ct_path: Path,
) -> Tuple[Path, Path]:
    output_seg_path = build_output_seg_path(case_id)
    summary_path = build_summary_path(case_id)

    print("=" * 100)
    print("[BRANCHES COMBINE - RUN.PY VERSION]")
    print(f"[CASE ID      ] {case_id}")
    print(f"[REFERENCE CT ] {reference_ct_path}")
    print(f"[OUTPUT SEG   ] {output_seg_path}")
    print(f"[SUMMARY      ] {summary_path}")
    print("=" * 100)

    ensure_dir(RESULTS_ROOT)

    if SKIP_IF_FINAL_EXISTS and output_seg_path.exists() and summary_path.exists():
        print(f"[SKIP] Output already exists: {output_seg_path}")
        print(f"[SKIP] Summary already exists: {summary_path}")
        return output_seg_path, summary_path

    reference_img, reference_data = load_nifti(reference_ct_path)
    reference_shape = reference_data.shape

    print(f"[INFO] Reference CT shape: {reference_shape}")
    print(f"[INFO] Reference CT affine:\n{reference_img.affine}")

    segments = collect_branch_segments(
        case_id=case_id,
        reference_img=reference_img,
        reference_shape=reference_shape,
    )

    seg_data = build_segmentation_array(segments)
    header = build_seg_nrrd_header(
        segments=segments,
        reference_affine=reference_img.affine,
    )

    save_seg_nrrd(output_seg_path, seg_data, header)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"case_id\t{case_id}\n")
        f.write(f"reference_ct_path\t{reference_ct_path}\n")
        f.write(f"output_seg_nrrd\t{output_seg_path}\n")
        f.write(f"num_segments\t{len(segments)}\n")
        f.write("\n")
        f.write(
            "segment_index\tsegment_name\tsource_label\tvoxel_count\t"
            "centroid_x\tcentroid_y\tcentroid_z\n"
        )

        for seg in segments:
            c = seg["centroid"]
            f.write(
                f"{seg['segment_index']}\t"
                f"{seg['name']}\t"
                f"{seg['source_label']}\t"
                f"{seg['voxel_count']}\t"
                f"{c[0]:.6f}\t{c[1]:.6f}\t{c[2]:.6f}\n"
            )

    print("=" * 100)
    print("[DONE] Abdominal vessel branch segmentation saved.")
    print(f"[OUTPUT] {output_seg_path}")
    print(f"[SUMMARY] {summary_path}")
    print("[SEGMENTS]")
    for seg in segments:
        print(f"  Layer {seg['segment_layer']:02d}: {seg['name']} ({seg['voxel_count']:,} voxels)")
    print("=" * 100)

    return output_seg_path, summary_path


# ============================================================
# 8. Command-Line Interface for run.py
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine abdominal vessel branch masks into AbdominalVessels.seg.nrrd."
    )

    parser.add_argument(
        "--ct_path",
        required=True,
        help="Original CT path passed by run.py. Used as reference geometry.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory passed by run.py. Accepted for compatibility; output is written to Results.",
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
    reference_ct_path = Path(args.ct_path).resolve()

    case_id = args.case_id.strip() if args.case_id else strip_nii_extension(reference_ct_path.name)
    if not case_id:
        raise ValueError("case_id is empty.")

    combine_abdominal_vessel_branches(
        case_id=case_id,
        reference_ct_path=reference_ct_path,
    )


if __name__ == "__main__":
    main()
