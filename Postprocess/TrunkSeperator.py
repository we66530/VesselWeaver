import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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
ANCHOR_POINTS_ROOT = BASE_DIR / "AnchorPoints"
RESULTS_ROOT = BASE_DIR / "Results"

INPUT_SUFFIX = "ATrace_connected_components"
OUTPUT_SEG_NAME = "VesselsTrunk.seg.nrrd"
SUMMARY_NAME = "VesselsTrunk_summary.txt"

CONNECTIVITY_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)

# If input has only one positive label, treat it as binary and split connected components.
# If input has labels 1/2/3 etc., preserve these labels as separate segments.
RECOMPUTE_CONNECTED_COMPONENTS_FOR_BINARY_INPUT = True

SKIP_IF_FINAL_EXISTS = True


COLORS = [
    [1.0, 0.2, 0.2],   # red
    [0.2, 0.8, 0.2],   # green
    [0.2, 0.4, 1.0],   # blue
    [1.0, 0.7, 0.2],
    [0.7, 0.2, 1.0],
]


# ============================================================
# 2. Path Builders
# ============================================================
def build_input_path(case_id: str) -> Path:
    return ANCHOR_POINTS_ROOT / f"{case_id}_{INPUT_SUFFIX}.nii.gz"


def build_output_seg_path(case_id: str) -> Path:
    # User requested fixed output name.
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


def load_and_align_input_to_reference(
    input_path: Path,
    reference_ct_path: Path,
):
    """
    Load ATrace_connected_components and align it to the original CT passed by run.py.

    This keeps the output .seg.nrrd in the same coordinate system as the CT
    that was selected in the GUI.
    """
    input_img, input_data = load_nifti(input_path)
    ref_img, ref_data = load_nifti(reference_ct_path)

    print("=" * 100)
    print("[GEOMETRY CHECK]")
    print(f"[REFERENCE CT] {reference_ct_path}")
    print(f"[REFERENCE CT SHAPE] {ref_data.shape}")
    print(f"[REFERENCE CT AFFINE]\n{ref_img.affine}")
    print("-" * 100)
    print(f"[INPUT MASK] {input_path}")
    print(f"[INPUT MASK SHAPE] {input_data.shape}")
    print(f"[INPUT MASK AFFINE]\n{input_img.affine}")
    print("=" * 100)

    same_shape = input_data.shape == ref_data.shape
    same_affine = np.allclose(input_img.affine, ref_img.affine, atol=1e-4)

    if same_shape and same_affine:
        print("[INFO] Input mask already matches reference CT geometry.")
        aligned_img = input_img
        aligned_data = input_data
    else:
        print("[WARNING] Input mask geometry differs from reference CT.")
        print("[INFO] Resampling input mask to reference CT with nearest-neighbor interpolation...")

        if not HAS_NILEARN:
            raise RuntimeError(
                "nilearn is required for geometry alignment but is not available.\n"
                "Please install it with:\n"
                "    pip install nilearn\n"
                "or make sure the ATrace mask already matches the original CT geometry."
            )

        aligned_img = resample_to_img(
            source_img=input_img,
            target_img=ref_img,
            interpolation="nearest",
        )
        aligned_data = aligned_img.get_fdata()

        print(f"[INFO] Resampled mask shape: {aligned_data.shape}")
        print(f"[INFO] Resampled mask affine:\n{aligned_img.affine}")

    if aligned_data.shape != ref_data.shape:
        raise RuntimeError(
            f"Aligned mask shape {aligned_data.shape} does not match reference CT shape {ref_data.shape}."
        )

    return ref_img, ref_data, aligned_img, aligned_data


# ============================================================
# 4. Component Extraction and Naming
# ============================================================
def extract_components(data_float: np.ndarray) -> List[Dict]:
    """
    Preserve existing positive labels if input is labeled.

    If input is binary only, split it into connected components.
    """
    data_int = np.rint(data_float).astype(np.int32)
    positive_labels = [int(x) for x in np.unique(data_int) if int(x) > 0]

    if len(positive_labels) == 0:
        raise RuntimeError("Input mask contains no positive voxels.")

    components: List[Dict] = []

    if len(positive_labels) == 1 and RECOMPUTE_CONNECTED_COMPONENTS_FOR_BINARY_INPUT:
        print("[INFO] Input appears binary. Recomputing connected components with 26-connectivity...")

        binary = data_int > 0
        labeled, num_components = ndimage.label(binary, structure=CONNECTIVITY_STRUCTURE)

        print(f"[INFO] Connected components found: {num_components}")

        for component_id in range(1, num_components + 1):
            mask = labeled == component_id
            coords = np.argwhere(mask)
            if coords.size == 0:
                continue

            components.append({
                "source_label": component_id,
                "mask": mask,
                "voxel_count": int(coords.shape[0]),
                "mean_z": float(coords[:, 2].mean()),
                "min_z": int(coords[:, 2].min()),
                "max_z": int(coords[:, 2].max()),
                "centroid": coords.astype(np.float64).mean(axis=0),
            })

    else:
        print(f"[INFO] Input already has positive labels. Preserving labels: {positive_labels}")

        for label_value in positive_labels:
            mask = data_int == label_value
            coords = np.argwhere(mask)
            if coords.size == 0:
                continue

            components.append({
                "source_label": label_value,
                "mask": mask,
                "voxel_count": int(coords.shape[0]),
                "mean_z": float(coords[:, 2].mean()),
                "min_z": int(coords[:, 2].min()),
                "max_z": int(coords[:, 2].max()),
                "centroid": coords.astype(np.float64).mean(axis=0),
            })

    if len(components) == 0:
        raise RuntimeError("No valid component could be extracted.")

    # Sort by Z-axis height from highest to lowest.
    components = sorted(components, key=lambda x: x["mean_z"], reverse=True)

    print("=" * 100)
    print("[COMPONENTS SORTED BY Z AXIS, HIGH TO LOW]")
    for rank, comp in enumerate(components, start=1):
        c = comp["centroid"]
        print(
            f"Rank {rank}: source_label={comp['source_label']}, "
            f"voxels={comp['voxel_count']:,}, mean_z={comp['mean_z']:.3f}, "
            f"z_range={comp['min_z']}-{comp['max_z']}, "
            f"centroid=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})"
        )
    print("=" * 100)

    return components


def assign_trunk_names(components: List[Dict]) -> List[Dict]:
    n = len(components)

    if n == 3:
        names = ["Celiac_trunk", "SMA_trunk", "IMA_trunk"]
    elif n == 2:
        names = ["Celiac+SMA_trunk", "IMA_trunk"]
    else:
        raise RuntimeError(
            f"Expected 2 or 3 segments, but found {n}.\n"
            "The naming rule is only defined for 2 or 3 trunk components."
        )

    assigned: List[Dict] = []

    for i, comp in enumerate(components):
        new_comp = dict(comp)
        new_comp["segment_index"] = i
        new_comp["segment_id"] = f"Segment_{i + 1}"
        new_comp["segment_name"] = names[i]
        new_comp["segment_layer"] = i
        new_comp["segment_label_value"] = 1
        assigned.append(new_comp)

    print("=" * 100)
    print("[TRUNK NAME ASSIGNMENT]")
    for comp in assigned:
        print(
            f"Layer {comp['segment_layer']}: {comp['segment_name']} "
            f"<- source_label={comp['source_label']}, "
            f"mean_z={comp['mean_z']:.3f}, voxels={comp['voxel_count']:,}"
        )
    print("=" * 100)

    return assigned


# ============================================================
# 5. Slicer .seg.nrrd Writer
#    This follows CombineName.py style:
#       seg_data shape = (segments, X, Y, Z)
#       kinds = ["list", "domain", "domain", "domain"]
#       space directions = [[nan,nan,nan], affine columns...]
# ============================================================
def build_segmentation_array(
    components: List[Dict],
) -> Tuple[np.ndarray, List[str]]:
    segment_arrays = []
    segment_names = []

    for comp in components:
        mask = comp["mask"].astype(np.uint8)
        voxel_count = int(mask.sum())

        if voxel_count == 0:
            print(f"[SKIP] Empty segment after extraction: {comp['segment_name']}")
            continue

        segment_arrays.append(mask)
        segment_names.append(comp["segment_name"])

        print(f"[ADD] {comp['segment_name']}: {voxel_count:,} voxels")

    if not segment_arrays:
        raise RuntimeError("All trunk components are empty. Nothing to write.")

    # IMPORTANT: this is the same as CombineName.py
    # Shape = (segments, X, Y, Z)
    seg_data = np.stack(segment_arrays, axis=0).astype(np.uint8)

    print(f"[INFO] Output seg_data shape: {seg_data.shape}")
    print(f"[INFO] Output segment names: {segment_names}")

    return seg_data, segment_names


def build_seg_nrrd_header(
    components: List[Dict],
    reference_affine: np.ndarray,
) -> Dict:
    """
    Build Slicer-compatible NRRD header, following CombineName.py.

    Note:
      This intentionally uses "right-anterior-superior" and reference_affine
      directly, matching the proven script supplied by the user.
    """
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

    for i, comp in enumerate(components):
        mask = comp["mask"]
        nz = np.argwhere(mask > 0)

        if nz.size == 0:
            continue

        xmin, ymin, zmin = nz.min(axis=0)
        xmax, ymax, zmax = nz.max(axis=0)

        color = COLORS[i % len(COLORS)]

        header[f"Segment{i}_ID"] = f"Segment_{i + 1}"
        header[f"Segment{i}_Name"] = comp["segment_name"]
        header[f"Segment{i}_NameAutoGenerated"] = "0"
        header[f"Segment{i}_Color"] = f"{color[0]} {color[1]} {color[2]}"
        header[f"Segment{i}_ColorAutoGenerated"] = "0"
        header[f"Segment{i}_LabelValue"] = "1"
        header[f"Segment{i}_Layer"] = str(i)
        header[f"Segment{i}_Extent"] = f"{xmin} {xmax} {ymin} {ymax} {zmin} {zmax}"

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

    # Same style as CombineName.py:
    # nrrd.write(output_path, seg_data, header)
    nrrd.write(str(output_path), seg_data.astype(np.uint8), header)

    if not output_path.exists():
        raise RuntimeError(f"Output .seg.nrrd was not created: {output_path}")


# ============================================================
# 6. Main Conversion Function
# ============================================================
def convert_trunk_components_to_seg_nrrd(
    case_id: str,
    reference_ct_path: Path,
) -> Tuple[Path, Path]:
    input_path = build_input_path(case_id)
    output_path = build_output_seg_path(case_id)
    summary_path = build_summary_path(case_id)

    print("=" * 100)
    print("[TRUNK SEPARATOR - RUN.PY VERSION]")
    print(f"[CASE ID      ] {case_id}")
    print(f"[REFERENCE CT ] {reference_ct_path}")
    print(f"[INPUT        ] {input_path}")
    print(f"[OUTPUT       ] {output_path}")
    print(f"[SUMMARY      ] {summary_path}")
    print("=" * 100)

    ensure_dir(RESULTS_ROOT)

    if SKIP_IF_FINAL_EXISTS and output_path.exists() and summary_path.exists():
        print(f"[SKIP] Output already exists: {output_path}")
        print(f"[SKIP] Summary already exists: {summary_path}")
        return output_path, summary_path

    ref_img, ref_data, aligned_img, aligned_data = load_and_align_input_to_reference(
        input_path=input_path,
        reference_ct_path=reference_ct_path,
    )

    print(f"[INFO] Aligned input shape: {aligned_data.shape}")
    print(
        f"[INFO] Aligned value range: "
        f"min={float(np.nanmin(aligned_data)):.3f}, "
        f"max={float(np.nanmax(aligned_data)):.3f}"
    )

    components = extract_components(aligned_data)
    components = assign_trunk_names(components)

    seg_data, segment_names = build_segmentation_array(components)

    # Use the original CT affine, exactly like CombineName.py uses ct_img.affine.
    header = build_seg_nrrd_header(
        components=components,
        reference_affine=ref_img.affine,
    )

    save_seg_nrrd(output_path, seg_data, header)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"case_id\t{case_id}\n")
        f.write(f"reference_ct_path\t{reference_ct_path}\n")
        f.write(f"input_path\t{input_path}\n")
        f.write(f"output_seg_nrrd\t{output_path}\n")
        f.write(f"num_segments\t{len(components)}\n")
        f.write(f"segment_names\t{segment_names}\n")
        f.write("\n")
        f.write(
            "segment_index\t"
            "segment_name\t"
            "source_label\t"
            "voxel_count\t"
            "mean_z\t"
            "min_z\t"
            "max_z\t"
            "centroid_x\t"
            "centroid_y\t"
            "centroid_z\n"
        )

        for comp in components:
            c = comp["centroid"]
            f.write(
                f"{comp['segment_index']}\t"
                f"{comp['segment_name']}\t"
                f"{comp['source_label']}\t"
                f"{comp['voxel_count']}\t"
                f"{comp['mean_z']:.6f}\t"
                f"{comp['min_z']}\t{comp['max_z']}\t"
                f"{c[0]:.6f}\t{c[1]:.6f}\t{c[2]:.6f}\n"
            )

    print("=" * 100)
    print("[DONE] Vessels trunk segmentation saved.")
    print(f"[OUTPUT] {output_path}")
    print(f"[SUMMARY] {summary_path}")
    print("[SEGMENTS]")
    for comp in components:
        print(
            f"  Layer {comp['segment_layer']} = {comp['segment_name']} "
            f"(source_label={comp['source_label']}, mean_z={comp['mean_z']:.3f})"
        )
    print("=" * 100)

    return output_path, summary_path


# ============================================================
# 7. Command-Line Interface for run.py
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ATrace connected components into a named Slicer .seg.nrrd trunk segmentation."
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
# 8. Main Entry Point
# ============================================================
def main() -> None:
    args = parse_args()
    reference_ct_path = Path(args.ct_path).resolve()

    case_id = args.case_id.strip() if args.case_id else strip_nii_extension(reference_ct_path.name)
    if not case_id:
        raise ValueError("case_id is empty.")

    convert_trunk_components_to_seg_nrrd(
        case_id=case_id,
        reference_ct_path=reference_ct_path,
    )


if __name__ == "__main__":
    main()