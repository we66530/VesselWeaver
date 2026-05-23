import argparse
import heapq
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation

try:
    from nilearn.image import resample_to_img
    HAS_NILEARN = True
except Exception:
    HAS_NILEARN = False


# ============================================================
# 1. Global Settings
# ============================================================
BASE_DIR = Path(r"C:\Users\User\Desktop\AbdVesselGen")
ROI_CT_ROOT = BASE_DIR / "ROI_CT"
ANCHOR_POINTS_ROOT = BASE_DIR / "AnchorPoints"
RESULTS_INTERMEDIATE_ROOT = BASE_DIR / "Results" / "Intermediate"

INVALID_HU_VALUE = -1024

# Starting volume from SearchStartCombined.py.
START_VOLUME_NAME = "search3231_volume"

# Input/output suffixes.
SPLENIC_HILUM_SUFFIX = "SpleenHilum"
RENAL_PATHS_SUFFIX = "renal_artery_vein_paths"
OUTPUT_FILENAME = "splenic_artery_path.nii.gz"
SUMMARY_FILENAME = "splenic_artery_path_summary.txt"

# Path rendering width.
PATH_DILATION_ITERATIONS = 1

# Pathfinding settings.
USE_26_NEIGHBOR_PATHFINDING = True
DIJKSTRA_BBOX_MARGINS = [None]

# ------------------------------------------------------------
# Initial forward-only constraint for splenic artery pathfinding.
# For the first N path steps, only allow movement toward decreasing Y.
# This prevents the path from immediately going backward into wrong high-HU regions.
# ------------------------------------------------------------
ENABLE_INITIAL_FORWARD_ONLY_CONSTRAINT = True
FORWARD_ONLY_INITIAL_STEPS = 5
FORWARD_AXIS_INDEX = 1  # 0=x, 1=y, 2=z
FORWARD_AXIS_DELTA = -1  # y-axis value must decrease
ALLOW_SIDEWAYS_DURING_FORWARD_ONLY = False

# Start extraction settings.
START_COMPONENT_CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)
START_NEARBY_SEARCH_RADIUS = 30
START_MIN_HU_PREFERRED = 20

# Endpoint extraction settings.
ENDPOINT_NEARBY_SEARCH_RADIUS = 80
ENDPOINT_DISTANCE_TOLERANCE = 1.5

# Obstacle settings.
# The renal artery/vein paths are treated as forbidden obstacle masks.
# ROI_CT voxels with HU == -1024 are also forbidden.
DILATE_FORBIDDEN_PATHS = True
FORBIDDEN_PATH_DILATION_ITERATIONS = 1

# Allow path to leave start/end even if the forbidden mask touches those areas.
ROOT_RELAX_DISTANCE = 3.0
END_RELAX_DISTANCE = 3.0

SKIP_IF_FINAL_EXISTS = True


# ============================================================
# 2. Path Builders
# ============================================================
def build_roi_ct_path(case_id: str) -> Path:
    return ROI_CT_ROOT / f"{case_id}_ROI_CT.nii.gz"


def build_renal_paths_path(case_id: str) -> Path:
    return RESULTS_INTERMEDIATE_ROOT / f"{case_id}_{RENAL_PATHS_SUFFIX}.nii.gz"


def build_start_volume_path(case_id: str) -> Path:
    return ANCHOR_POINTS_ROOT / f"{case_id}_{START_VOLUME_NAME}.nii.gz"


def build_spleen_hilum_path(case_id: str) -> Path:
    return ANCHOR_POINTS_ROOT / f"{case_id}_{SPLENIC_HILUM_SUFFIX}.nii"


def build_output_path(case_id: str) -> Path:
    return RESULTS_INTERMEDIATE_ROOT / f"{case_id}_{OUTPUT_FILENAME}"


def build_summary_path(case_id: str) -> Path:
    return RESULTS_INTERMEDIATE_ROOT / f"{case_id}_{SUMMARY_FILENAME}"


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


def is_inside_shape(point: Tuple[int, int, int], shape: Tuple[int, int, int]) -> bool:
    x, y, z = point
    sx, sy, sz = shape
    return 0 <= x < sx and 0 <= y < sy and 0 <= z < sz


def align_mask_to_ct_if_needed(mask_img, mask_data: np.ndarray, ct_img, name: str):
    ct_shape = ct_img.shape[:3]

    if mask_data.shape == ct_shape and np.allclose(mask_img.affine, ct_img.affine, atol=1e-4):
        print(f"[INFO] {name} already matches ROI_CT geometry.")
        return mask_img, mask_data

    print(f"[WARNING] {name} geometry does not match ROI_CT geometry.")

    if not HAS_NILEARN:
        raise RuntimeError(
            f"{name} geometry differs from ROI_CT, but nilearn is not available for resampling.\n"
            "Please install nilearn or ensure the mask is already aligned."
        )

    print(f"[INFO] Resampling {name} to ROI_CT using nearest-neighbor interpolation...")
    aligned_img = resample_to_img(mask_img, ct_img, interpolation="nearest")
    aligned_data = aligned_img.get_fdata()
    return aligned_img, aligned_data


def get_neighbor_offsets(use_26_neighbor: bool = USE_26_NEIGHBOR_PATHFINDING) -> List[Tuple[int, int, int, float]]:
    offsets: List[Tuple[int, int, int, float]] = []

    if use_26_neighbor:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    offsets.append((dx, dy, dz, float(np.sqrt(dx * dx + dy * dy + dz * dz))))
    else:
        for dx, dy, dz in [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ]:
            offsets.append((dx, dy, dz, 1.0))

    return offsets


# ============================================================
# 4. Start Point Extraction from search3231_volume
# ============================================================
def find_nearest_valid_voxel_inside_component(
    ct_data: np.ndarray,
    component_coords: np.ndarray,
    centroid: np.ndarray,
) -> Optional[Tuple[int, int, int]]:
    hu_values = ct_data[
        component_coords[:, 0],
        component_coords[:, 1],
        component_coords[:, 2],
    ]
    valid_mask = hu_values != INVALID_HU_VALUE

    if not np.any(valid_mask):
        return None

    valid_coords = component_coords[valid_mask]
    valid_hu = hu_values[valid_mask]

    preferred_mask = valid_hu > START_MIN_HU_PREFERRED
    if np.any(preferred_mask):
        candidate_coords = valid_coords[preferred_mask]
        candidate_hu = valid_hu[preferred_mask]
    else:
        candidate_coords = valid_coords
        candidate_hu = valid_hu

    distances = np.linalg.norm(candidate_coords.astype(np.float64) - centroid[None, :], axis=1)
    order = np.lexsort((-candidate_hu, distances))
    best = candidate_coords[int(order[0])]
    return tuple(int(v) for v in best.tolist())


def find_nearest_valid_voxel_near_point(
    ct_data: np.ndarray,
    point: np.ndarray,
    max_radius: int,
) -> Optional[Tuple[int, int, int]]:
    shape = np.array(ct_data.shape, dtype=np.int64)
    center = np.rint(point).astype(np.int64)

    for radius in range(0, max_radius + 1):
        x_min = max(0, int(center[0]) - radius)
        x_max = min(int(shape[0]) - 1, int(center[0]) + radius)
        y_min = max(0, int(center[1]) - radius)
        y_max = min(int(shape[1]) - 1, int(center[1]) + radius)
        z_min = max(0, int(center[2]) - radius)
        z_max = min(int(shape[2]) - 1, int(center[2]) + radius)

        sub_ct = ct_data[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
        valid = sub_ct != INVALID_HU_VALUE

        if not np.any(valid):
            continue

        local_coords = np.argwhere(valid)
        global_coords = local_coords + np.array([x_min, y_min, z_min], dtype=np.int64)
        hu_values = ct_data[global_coords[:, 0], global_coords[:, 1], global_coords[:, 2]]
        distances = np.linalg.norm(global_coords.astype(np.float64) - point[None, :], axis=1)
        order = np.lexsort((-hu_values, distances))
        best = global_coords[int(order[0])]
        return tuple(int(v) for v in best.tolist())

    return None


def extract_start_points_from_search_volume(
    ct_data: np.ndarray,
    start_volume_data: np.ndarray,
) -> List[Dict]:
    binary_mask = start_volume_data > 0
    total_voxels = int(binary_mask.sum())

    print("=" * 100)
    print("[START POINT EXTRACTION FROM SEARCH3231 VOLUME]")
    print(f"[INFO] Search volume positive voxels: {total_voxels:,}")
    print("=" * 100)

    if total_voxels == 0:
        raise RuntimeError("search3231_volume is empty; no splenic artery starting point candidates found.")

    labeled, num_components = ndimage.label(binary_mask, structure=START_COMPONENT_CONNECTIVITY)
    print(f"[INFO] Connected components found: {num_components}")

    starts: List[Dict] = []

    for component_id in range(1, num_components + 1):
        component_mask = labeled == component_id
        component_coords = np.argwhere(component_mask)
        component_voxels = int(component_coords.shape[0])
        centroid = component_coords.astype(np.float64).mean(axis=0)

        point = find_nearest_valid_voxel_inside_component(
            ct_data=ct_data,
            component_coords=component_coords,
            centroid=centroid,
        )
        method = "nearest_valid_inside_component"

        if point is None:
            point = find_nearest_valid_voxel_near_point(
                ct_data=ct_data,
                point=centroid,
                max_radius=START_NEARBY_SEARCH_RADIUS,
            )
            method = "nearest_valid_near_component_centroid"

        if point is None:
            print(f"[WARNING] Component {component_id}: no valid HU point found. Skipped.")
            continue

        record = {
            "component_id": component_id,
            "component_voxels": component_voxels,
            "centroid": centroid,
            "point": point,
            "point_hu": float(ct_data[point]),
            "method": method,
        }
        starts.append(record)

        print(
            f"[START] component={component_id}, voxels={component_voxels:,}, "
            f"centroid={np.round(centroid, 2).tolist()}, point={point}, "
            f"HU={float(ct_data[point]):.1f}, method={method}"
        )

    if len(starts) == 0:
        raise RuntimeError("No valid start point could be extracted from search3231_volume.")

    return starts


# ============================================================
# 5. Endpoint Extraction from SpleenHilum.nii
# ============================================================
def extract_spleen_hilum_endpoint(
    ct_data: np.ndarray,
    spleen_hilum_data: np.ndarray,
) -> Dict:
    coords = np.argwhere(spleen_hilum_data > 0)

    print("=" * 100)
    print("[SPLEEN HILUM ENDPOINT EXTRACTION]")
    print(f"[INFO] SpleenHilum positive voxels: {len(coords):,}")
    print("=" * 100)

    if len(coords) == 0:
        raise RuntimeError("SpleenHilum mask contains no positive voxel.")

    centroid = coords.astype(np.float64).mean(axis=0)
    rounded = tuple(int(v) for v in np.rint(centroid).astype(np.int64).tolist())

    if is_inside_shape(rounded, ct_data.shape) and float(ct_data[rounded]) != INVALID_HU_VALUE:
        point = rounded
        method = "rounded_centroid_valid"
    else:
        point = find_nearest_valid_voxel_near_point(
            ct_data=ct_data,
            point=centroid,
            max_radius=ENDPOINT_NEARBY_SEARCH_RADIUS,
        )
        method = "nearest_valid_to_hilum"
        if point is None:
            raise RuntimeError(
                f"No valid endpoint found near SpleenHilum within radius {ENDPOINT_NEARBY_SEARCH_RADIUS}."
            )

    endpoint = {
        "centroid": centroid,
        "point": point,
        "point_hu": float(ct_data[point]),
        "method": method,
    }

    print(
        f"[ENDPOINT] centroid={np.round(centroid, 2).tolist()}, point={point}, "
        f"HU={endpoint['point_hu']:.1f}, method={method}"
    )

    return endpoint


# ============================================================
# 6. Forbidden Zone and Dijkstra Pathfinding
# ============================================================
def build_forbidden_zone(
    ct_data: np.ndarray,
    renal_paths_data: np.ndarray,
    start_point: Tuple[int, int, int],
    endpoint: Tuple[int, int, int],
) -> np.ndarray:
    out_of_roi = ct_data == INVALID_HU_VALUE
    renal_obstacle = renal_paths_data > 0

    if DILATE_FORBIDDEN_PATHS:
        renal_obstacle = binary_dilation(
            renal_obstacle,
            structure=np.ones((3, 3, 3), dtype=bool),
            iterations=FORBIDDEN_PATH_DILATION_ITERATIONS,
        )

    forbidden = out_of_roi | renal_obstacle
    forbidden[start_point] = False
    forbidden[endpoint] = False
    return forbidden


def voxel_step_cost(hu_val: float) -> float:
    base_cost = 1.0
    if hu_val < 40:
        hu_penalty = (40 - hu_val) ** 2 + 500
    else:
        hu_penalty = max(0, 250 - hu_val)
    return float(base_cost + hu_penalty)


def make_bbox_bounds(
    shape: Tuple[int, int, int],
    start: Tuple[int, int, int],
    end: Tuple[int, int, int],
    margin: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    shape_arr = np.array(shape, dtype=np.int64)
    if margin is None:
        return np.array([0, 0, 0], dtype=np.int64), shape_arr - 1

    start_arr = np.array(start, dtype=np.int64)
    end_arr = np.array(end, dtype=np.int64)
    lower = np.maximum(0, np.minimum(start_arr, end_arr) - int(margin))
    upper = np.minimum(shape_arr - 1, np.maximum(start_arr, end_arr) + int(margin))
    return lower, upper


def is_allowed_by_initial_forward_constraint(
    curr_pos: Tuple[int, int, int],
    next_pos: Tuple[int, int, int],
    steps_taken: int,
) -> bool:
    if not ENABLE_INITIAL_FORWARD_ONLY_CONSTRAINT:
        return True

    if steps_taken >= FORWARD_ONLY_INITIAL_STEPS:
        return True

    delta = next_pos[FORWARD_AXIS_INDEX] - curr_pos[FORWARD_AXIS_INDEX]

    if FORWARD_AXIS_DELTA < 0:
        if ALLOW_SIDEWAYS_DURING_FORWARD_ONLY:
            return delta <= 0
        return delta < 0

    if FORWARD_AXIS_DELTA > 0:
        if ALLOW_SIDEWAYS_DURING_FORWARD_ONLY:
            return delta >= 0
        return delta > 0

    return True


def dijkstra_pathfinder_once(
    ct_data: np.ndarray,
    start: Tuple[int, int, int],
    end: Tuple[int, int, int],
    forbidden_zone: np.ndarray,
    bbox_margin: Optional[int] = None,
) -> Tuple[Optional[List[Tuple[int, int, int]]], float, int]:
    rows, cols, slices = ct_data.shape

    if not is_inside_shape(start, ct_data.shape):
        raise ValueError(f"Start point out of bounds: {start}")
    if not is_inside_shape(end, ct_data.shape):
        raise ValueError(f"End point out of bounds: {end}")

    if float(ct_data[start]) == INVALID_HU_VALUE:
        print(f"[WARNING] Start point has invalid HU: {start}")
        return None, float("inf"), 0
    if float(ct_data[end]) == INVALID_HU_VALUE:
        print(f"[WARNING] End point has invalid HU: {end}")
        return None, float("inf"), 0

    lower, upper = make_bbox_bounds(ct_data.shape, start, end, bbox_margin)

    start_state = (start, 0)
    queue = [(0.0, start_state)]
    distances = {start_state: 0.0}
    predecessor = {start_state: None}
    neighbors = get_neighbor_offsets(USE_26_NEIGHBOR_PATHFINDING)
    visited = set()
    expanded_nodes = 0

    start_arr = np.array(start, dtype=np.float64)
    end_arr = np.array(end, dtype=np.float64)

    while queue:
        curr_cost, curr_state = heapq.heappop(queue)
        curr_pos, steps_taken = curr_state

        if curr_state in visited:
            continue
        visited.add(curr_state)
        expanded_nodes += 1

        if curr_pos == end:
            path = []
            state = curr_state
            while state is not None:
                pos, _ = state
                path.append(pos)
                state = predecessor[state]
            path = path[::-1]
            return path, float(curr_cost), expanded_nodes

        if curr_cost > distances.get(curr_state, float("inf")):
            continue

        for dx, dy, dz, movement_length in neighbors:
            next_pos = (curr_pos[0] + dx, curr_pos[1] + dy, curr_pos[2] + dz)

            if not (0 <= next_pos[0] < rows and 0 <= next_pos[1] < cols and 0 <= next_pos[2] < slices):
                continue

            if not is_allowed_by_initial_forward_constraint(curr_pos, next_pos, steps_taken):
                continue

            next_arr = np.array(next_pos, dtype=np.int64)
            if np.any(next_arr < lower) or np.any(next_arr > upper):
                continue

            hu_val = float(ct_data[next_pos])
            if hu_val == INVALID_HU_VALUE:
                continue

            if forbidden_zone[next_pos]:
                next_arr_float = np.array(next_pos, dtype=np.float64)
                dist_to_start = float(np.linalg.norm(next_arr_float - start_arr))
                dist_to_end = float(np.linalg.norm(next_arr_float - end_arr))

                if dist_to_start > ROOT_RELAX_DISTANCE and dist_to_end > END_RELAX_DISTANCE:
                    continue

            next_steps = min(steps_taken + 1, FORWARD_ONLY_INITIAL_STEPS)
            next_state = (next_pos, next_steps)
            new_cost = curr_cost + movement_length * voxel_step_cost(hu_val)

            if new_cost < distances.get(next_state, float("inf")):
                distances[next_state] = new_cost
                predecessor[next_state] = curr_state
                heapq.heappush(queue, (new_cost, next_state))

    return None, float("inf"), expanded_nodes


def dijkstra_pathfinder(
    ct_data: np.ndarray,
    start: Tuple[int, int, int],
    end: Tuple[int, int, int],
    forbidden_zone: np.ndarray,
) -> Tuple[Optional[List[Tuple[int, int, int]]], float]:
    print(f"  [Dijkstra] Connectivity: {'26-neighbor' if USE_26_NEIGHBOR_PATHFINDING else '6-neighbor'}")
    print(
        f"  [Dijkstra] Initial forward-only constraint: {ENABLE_INITIAL_FORWARD_ONLY_CONSTRAINT}, "
        f"steps={FORWARD_ONLY_INITIAL_STEPS}, axis={FORWARD_AXIS_INDEX}, delta={FORWARD_AXIS_DELTA}"
    )

    for margin in DIJKSTRA_BBOX_MARGINS:
        margin_text = "full-volume" if margin is None else f"bbox margin {margin}"
        print(f"  [Dijkstra] Trying {margin_text}...")

        path, cost, expanded = dijkstra_pathfinder_once(
            ct_data=ct_data,
            start=start,
            end=end,
            forbidden_zone=forbidden_zone,
            bbox_margin=margin,
        )

        if path is not None:
            print(f"  [Dijkstra] Success with {margin_text}. Expanded nodes: {expanded:,}")
            return path, cost

        print(f"  [Dijkstra] Failed with {margin_text}. Expanded nodes: {expanded:,}")

    return None, float("inf")


# ============================================================
# 7. Rendering and Summary
# ============================================================
def render_path_to_labelmap(shape: Tuple[int, int, int], path: List[Tuple[int, int, int]]) -> np.ndarray:
    path_mask = np.zeros(shape, dtype=bool)
    for p in path:
        path_mask[p] = True

    output = binary_dilation(
        path_mask,
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=PATH_DILATION_ITERATIONS,
    ).astype(np.uint8)

    return output


def write_summary(
    summary_path: Path,
    case_id: str,
    ct_path: Path,
    renal_paths_path: Path,
    start_volume_path: Path,
    spleen_hilum_path: Path,
    output_path: Path,
    start_records: List[Dict],
    endpoint_record: Dict,
    selected_result: Dict,
) -> None:
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"case_id\t{case_id}\n")
        f.write(f"ct_path\t{ct_path}\n")
        f.write(f"renal_paths_path\t{renal_paths_path}\n")
        f.write(f"start_volume_path\t{start_volume_path}\n")
        f.write(f"spleen_hilum_path\t{spleen_hilum_path}\n")
        f.write(f"output_path\t{output_path}\n")
        f.write(f"invalid_hu_value\t{INVALID_HU_VALUE}\n")
        f.write(f"use_26_neighbor_pathfinding\t{USE_26_NEIGHBOR_PATHFINDING}\n")
        f.write(f"dijkstra_bbox_margins\t{DIJKSTRA_BBOX_MARGINS}\n")
        f.write(f"enable_initial_forward_only_constraint\t{ENABLE_INITIAL_FORWARD_ONLY_CONSTRAINT}\n")
        f.write(f"forward_only_initial_steps\t{FORWARD_ONLY_INITIAL_STEPS}\n")
        f.write(f"forward_axis_index\t{FORWARD_AXIS_INDEX}\n")
        f.write(f"forward_axis_delta\t{FORWARD_AXIS_DELTA}\n")
        f.write(f"allow_sideways_during_forward_only\t{ALLOW_SIDEWAYS_DURING_FORWARD_ONLY}\n")
        f.write(f"dilate_forbidden_paths\t{DILATE_FORBIDDEN_PATHS}\n")
        f.write(f"forbidden_path_dilation_iterations\t{FORBIDDEN_PATH_DILATION_ITERATIONS}\n")
        f.write("\n")

        f.write("ENDPOINT\n")
        endpoint = endpoint_record["point"]
        f.write("centroid_x\tcentroid_y\tcentroid_z\tpoint_x\tpoint_y\tpoint_z\tpoint_hu\tmethod\n")
        f.write(
            f"{endpoint_record['centroid'][0]:.6f}\t"
            f"{endpoint_record['centroid'][1]:.6f}\t"
            f"{endpoint_record['centroid'][2]:.6f}\t"
            f"{endpoint[0]}\t{endpoint[1]}\t{endpoint[2]}\t"
            f"{endpoint_record['point_hu']:.6f}\t{endpoint_record['method']}\n"
        )
        f.write("\n")

        f.write("START_CANDIDATES\n")
        f.write("component_id\tcomponent_voxels\tcentroid_x\tcentroid_y\tcentroid_z\tpoint_x\tpoint_y\tpoint_z\tpoint_hu\tmethod\tcost\tpath_voxels\tstatus\n")
        for rec in start_records:
            centroid = rec["centroid"]
            point = rec["point"]
            f.write(
                f"{rec['component_id']}\t{rec['component_voxels']}\t"
                f"{centroid[0]:.6f}\t{centroid[1]:.6f}\t{centroid[2]:.6f}\t"
                f"{point[0]}\t{point[1]}\t{point[2]}\t{rec['point_hu']:.6f}\t"
                f"{rec['method']}\t{rec.get('cost', float('inf')):.6f}\t"
                f"{rec.get('path_voxels', 0)}\t{rec.get('status', 'not_run')}\n"
            )
        f.write("\n")

        f.write("SELECTED_PATH\n")
        start = selected_result["start_point"]
        end = selected_result["endpoint"]
        f.write("start_component_id\tstart_x\tstart_y\tstart_z\tend_x\tend_y\tend_z\tcost\tpath_voxels\n")
        f.write(
            f"{selected_result['component_id']}\t"
            f"{start[0]}\t{start[1]}\t{start[2]}\t"
            f"{end[0]}\t{end[1]}\t{end[2]}\t"
            f"{selected_result['cost']:.6f}\t{selected_result['path_voxels']}\n"
        )


# ============================================================
# 8. Main Pipeline
# ============================================================
def run_splenic_artery_pathfinder(case_id: str) -> Tuple[Path, Path]:
    ct_path = build_roi_ct_path(case_id)
    renal_paths_path = build_renal_paths_path(case_id)
    start_volume_path = build_start_volume_path(case_id)
    spleen_hilum_path = build_spleen_hilum_path(case_id)
    output_path = build_output_path(case_id)
    summary_path = build_summary_path(case_id)

    print("=" * 100)
    print("[SPLENIC ARTERY PATHFINDER - RUN.PY VERSION]")
    print(f"[CASE ID      ] {case_id}")
    print(f"[ROI CT       ] {ct_path}")
    print(f"[RENAL PATHS  ] {renal_paths_path}")
    print(f"[START VOLUME ] {start_volume_path}")
    print(f"[SPLEEN HILUM ] {spleen_hilum_path}")
    print(f"[OUTPUT       ] {output_path}")
    print("=" * 100)

    ensure_dir(output_path.parent)

    if SKIP_IF_FINAL_EXISTS and output_path.exists() and summary_path.exists():
        print(f"[SKIP] Output already exists: {output_path}")
        print(f"[SKIP] Summary already exists: {summary_path}")
        return output_path, summary_path

    ct_img, ct_data = load_nifti(ct_path)

    renal_img_raw, renal_data_raw = load_nifti(renal_paths_path)
    start_img_raw, start_data_raw = load_nifti(start_volume_path)
    spleen_img_raw, spleen_data_raw = load_nifti(spleen_hilum_path)

    _, renal_data = align_mask_to_ct_if_needed(renal_img_raw, renal_data_raw, ct_img, "renal paths obstacle mask")
    _, start_data = align_mask_to_ct_if_needed(start_img_raw, start_data_raw, ct_img, "search3231 start volume")
    _, spleen_data = align_mask_to_ct_if_needed(spleen_img_raw, spleen_data_raw, ct_img, "SpleenHilum endpoint mask")

    print(f"[INFO] ROI_CT shape: {ct_data.shape}")
    print(f"[INFO] ROI_CT HU range: min={ct_data.min():.1f}, max={ct_data.max():.1f}")
    print(f"[INFO] Renal obstacle voxels: {int((renal_data > 0).sum()):,}")

    start_records = extract_start_points_from_search_volume(ct_data, start_data)
    endpoint_record = extract_spleen_hilum_endpoint(ct_data, spleen_data)
    endpoint = endpoint_record["point"]

    print("=" * 100)
    print("[COMPUTE SPLENIC ARTERY PATHS FROM ALL START CANDIDATES]")
    print(f"[N STARTS] {len(start_records)}")
    print(f"[ENDPOINT] {endpoint}, HU={endpoint_record['point_hu']:.1f}")
    print("=" * 100)

    best_result = None

    for rec in start_records:
        start_point = rec["point"]
        print("-" * 100)
        print(
            f"[PATH CANDIDATE] component={rec['component_id']}, start={start_point}, "
            f"HU={rec['point_hu']:.1f} -> endpoint={endpoint}"
        )

        forbidden_zone = build_forbidden_zone(
            ct_data=ct_data,
            renal_paths_data=renal_data,
            start_point=start_point,
            endpoint=endpoint,
        )

        path, cost = dijkstra_pathfinder(
            ct_data=ct_data,
            start=start_point,
            end=endpoint,
            forbidden_zone=forbidden_zone,
        )

        if path is None:
            rec["cost"] = float("inf")
            rec["path_voxels"] = 0
            rec["status"] = "failed"
            print(f"[FAILED] component={rec['component_id']}")
            continue

        rec["cost"] = float(cost)
        rec["path_voxels"] = len(path)
        rec["status"] = "ok"
        rec["path"] = path

        print(f"[OK] component={rec['component_id']}, cost={cost:.3f}, path_voxels={len(path):,}")

        if best_result is None or cost < best_result["cost"]:
            best_result = {
                "component_id": rec["component_id"],
                "start_point": start_point,
                "endpoint": endpoint,
                "cost": float(cost),
                "path_voxels": len(path),
                "path": path,
            }

    if best_result is None:
        raise RuntimeError("No feasible splenic artery path found from any search3231 start component.")

    print("=" * 100)
    print("[BEST SPLENIC ARTERY PATH]")
    print(f"[COMPONENT] {best_result['component_id']}")
    print(f"[START    ] {best_result['start_point']}, HU={float(ct_data[best_result['start_point']]):.1f}")
    print(f"[ENDPOINT ] {best_result['endpoint']}, HU={float(ct_data[best_result['endpoint']]):.1f}")
    print(f"[COST     ] {best_result['cost']:.3f}")
    print(f"[VOXELS   ] {best_result['path_voxels']:,}")
    print("=" * 100)

    output_labelmap = render_path_to_labelmap(ct_data.shape, best_result["path"])

    out_header = ct_img.header.copy()
    out_header.set_data_dtype(np.uint8)

    out_img = nib.Nifti1Image(
        output_labelmap.astype(np.uint8),
        affine=ct_img.affine,
        header=out_header,
    )
    nib.save(out_img, str(output_path))

    if not output_path.exists():
        raise RuntimeError(f"Output was not created: {output_path}")

    write_summary(
        summary_path=summary_path,
        case_id=case_id,
        ct_path=ct_path,
        renal_paths_path=renal_paths_path,
        start_volume_path=start_volume_path,
        spleen_hilum_path=spleen_hilum_path,
        output_path=output_path,
        start_records=start_records,
        endpoint_record=endpoint_record,
        selected_result=best_result,
    )

    print("=" * 100)
    print("[DONE] Splenic artery pathfinding completed.")
    print(f"[OUTPUT MASK] {output_path}")
    print(f"[SUMMARY    ] {summary_path}")
    print("=" * 100)

    return output_path, summary_path


# ============================================================
# 9. Command-Line Interface for run.py
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace splenic artery path from search3231 starts to SpleenHilum while avoiding renal artery/vein paths."
    )

    parser.add_argument(
        "--ct_path",
        required=True,
        help="CT path passed by run.py. Accepted for compatibility; this script uses ROI_CT output.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory passed by run.py. Accepted for compatibility; output is written to Results/Intermediate.",
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

    run_splenic_artery_pathfinder(case_id=case_id)


if __name__ == "__main__":
    main()
