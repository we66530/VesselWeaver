import argparse
import heapq
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, label as cc_label
from scipy.spatial import KDTree

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

HEPATIC_HILUM_SPHERES_SUFFIX = "hepatic_hilum_spheres"
ANCHOR_POINTS_TXT_SUFFIX = "anchor_points_from_search_volumes"

# Use only starting points derived from this search volume.
START_VOLUME_NAME = "search3231_volume"

OUTPUT_FILENAME = "hepatic_vessels.nii.gz"
SUMMARY_FILENAME = "hepatic_vessels_summary.txt"

ENDPOINT_MIN_HU = 20
INVALID_HU_VALUE = -1024

PATH_DILATION_ITERATIONS = 1
AVOID_INVALID_HU = True
USE_26_NEIGHBOR_PATHFINDING = True

# Dijkstra can be slow on a full 500x500x540 volume.
# Try local boxes first, then full-volume search only if needed.
DIJKSTRA_BBOX_MARGINS = [None]

# Fallback endpoint search settings.
# If a sphere endpoint cannot be reached, search for a valid ROI_CT voxel nearest
# to the sphere. Among similarly close voxels, prefer higher HU.
ENABLE_FALLBACK_ENDPOINT = True
FALLBACK_MAX_RADIUS = 160
FALLBACK_DISTANCE_TOLERANCE = 1.5
FALLBACK_REQUIRE_SAME_VALID_COMPONENT_AS_START = True

SKIP_IF_FINAL_EXISTS = True


# ============================================================
# 2. Path Builders
# ============================================================
def build_roi_ct_path(case_id: str) -> Path:
    return ROI_CT_ROOT / f"{case_id}_ROI_CT.nii.gz"


def build_spheres_path(case_id: str) -> Path:
    return ANCHOR_POINTS_ROOT / f"{case_id}_{HEPATIC_HILUM_SPHERES_SUFFIX}.nii"


def build_anchor_points_txt_path(case_id: str) -> Path:
    return ANCHOR_POINTS_ROOT / f"{case_id}_{ANCHOR_POINTS_TXT_SUFFIX}.txt"


def build_output_path(case_id: str) -> Path:
    return RESULTS_INTERMEDIATE_ROOT / f"{case_id}_{OUTPUT_FILENAME}"


def build_summary_path(case_id: str) -> Path:
    return RESULTS_INTERMEDIATE_ROOT / f"{case_id}_{SUMMARY_FILENAME}"


# ============================================================
# 3. Basic Utilities
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


def align_spheres_to_ct_if_needed(spheres_img, spheres_data: np.ndarray, ct_img):
    ct_shape = ct_img.shape[:3]

    if spheres_data.shape == ct_shape and np.allclose(spheres_img.affine, ct_img.affine, atol=1e-4):
        print("[INFO] Sphere mask already matches ROI CT geometry.")
        return spheres_img, spheres_data

    print("[WARNING] Sphere mask geometry does not match ROI CT geometry.")

    if not HAS_NILEARN:
        raise RuntimeError(
            "Sphere mask geometry differs from ROI_CT, but nilearn is not available for resampling.\n"
            "Please install nilearn or ensure the sphere mask is already aligned."
        )

    print("[INFO] Resampling sphere mask to ROI_CT using nearest-neighbor interpolation...")
    aligned_img = resample_to_img(spheres_img, ct_img, interpolation="nearest")
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
# 4. Starting Points from SearchStartCombined / Anchor TXT
# ============================================================
def parse_anchor_points_txt(txt_path: Path) -> List[Dict]:
    """
    Parse the tab-separated anchor point file generated by
    FindAnchorPointsFromSearchVolumes.py.

    Expected columns:
      volume_name, component_id, point_x, point_y, point_z, point_hu, ...
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"Anchor points TXT not found:\n{txt_path}")

    records: List[Dict] = []
    header = None

    with open(txt_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split("\t")

            if len(parts) == 2 and parts[0] in {"case_id", "invalid_hu_value", "max_search_radius"}:
                continue

            if parts[0] == "volume_name":
                header = parts
                continue

            if header is None:
                continue

            if len(parts) != len(header):
                print(f"[WARNING] Skipping malformed TXT row: {line}")
                continue

            row = dict(zip(header, parts))

            try:
                row["component_id"] = int(row["component_id"])
                row["component_voxels"] = int(row["component_voxels"])
                row["centroid_x"] = float(row["centroid_x"])
                row["centroid_y"] = float(row["centroid_y"])
                row["centroid_z"] = float(row["centroid_z"])
                row["point_x"] = int(row["point_x"])
                row["point_y"] = int(row["point_y"])
                row["point_z"] = int(row["point_z"])
                row["point_hu"] = float(row["point_hu"])
                row["distance_to_centroid"] = float(row["distance_to_centroid"])
                row["found_radius"] = int(row["found_radius"])
            except Exception as e:
                print(f"[WARNING] Skipping row because parsing failed: {line}")
                print(f"[WARNING] Error: {e}")
                continue

            records.append(row)

    return records


def get_start_points_from_search3231(anchor_txt_path: Path, ct_data: np.ndarray) -> List[Tuple[int, int, int]]:
    records = parse_anchor_points_txt(anchor_txt_path)
    selected = [r for r in records if r.get("volume_name") == START_VOLUME_NAME]

    if len(selected) == 0:
        raise RuntimeError(
            f"No starting point records found from volume_name={START_VOLUME_NAME} in:\n{anchor_txt_path}"
        )

    starts: List[Tuple[int, int, int]] = []
    seen = set()

    for r in selected:
        point = (int(r["point_x"]), int(r["point_y"]), int(r["point_z"]))
        if point in seen:
            continue
        if not is_inside_shape(point, ct_data.shape):
            print(f"[WARNING] Skipping out-of-bounds start point: {point}")
            continue
        if AVOID_INVALID_HU and float(ct_data[point]) == INVALID_HU_VALUE:
            print(f"[WARNING] Skipping invalid-HU start point: {point}, HU={float(ct_data[point]):.1f}")
            continue
        seen.add(point)
        starts.append(point)

    if len(starts) == 0:
        raise RuntimeError("No valid in-bounds search3231 starting point was found.")

    print("=" * 100)
    print("[STARTING POINTS FROM SEARCH3231]")
    for i, p in enumerate(starts, start=1):
        print(f"  Start {i}: {p}, ROI_CT HU={float(ct_data[p]):.1f}")
    print("=" * 100)

    return starts


def build_reachable_valid_component_mask(
    ct_data: np.ndarray,
    start_point: Tuple[int, int, int],
) -> Optional[np.ndarray]:
    valid_mask = ct_data != INVALID_HU_VALUE

    if not valid_mask[start_point]:
        print(f"[WARNING] Start point {start_point} is not in a valid ROI component.")
        return None

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, num_components = cc_label(valid_mask, structure=structure)
    start_component = int(labeled[start_point])

    if start_component == 0:
        print(f"[WARNING] Start point component is 0 for point {start_point}.")
        return None

    reachable_mask = labeled == start_component
    print(
        f"[INFO] Start {start_point} reachable component: label={start_component}, "
        f"voxels={int(reachable_mask.sum()):,}, total_valid_components={num_components}"
    )
    return reachable_mask


def build_reachable_masks_for_starts(
    ct_data: np.ndarray,
    start_points: List[Tuple[int, int, int]],
) -> List[Optional[np.ndarray]]:
    if not FALLBACK_REQUIRE_SAME_VALID_COMPONENT_AS_START:
        return [None for _ in start_points]

    masks = []
    for p in start_points:
        masks.append(build_reachable_valid_component_mask(ct_data, p))
    return masks


# ============================================================
# 5. Endpoint Selection from Hepatic Hilum Spheres
# ============================================================
def find_best_endpoint_inside_sphere(
    label_id: int,
    spheres_data: np.ndarray,
    ct_data: np.ndarray,
    min_hu: float = ENDPOINT_MIN_HU,
) -> Optional[Tuple[int, int, int]]:
    coords = np.argwhere(spheres_data == label_id)
    if len(coords) == 0:
        return None

    ideal_center = np.mean(coords, axis=0)
    mask_hu = ct_data[coords[:, 0], coords[:, 1], coords[:, 2]]

    valid_idx = (mask_hu != INVALID_HU_VALUE) & (mask_hu > min_hu)
    valid_coords = coords[valid_idx]
    valid_hu = mask_hu[valid_idx]

    if len(valid_coords) == 0:
        print(f"[WARNING] Label {label_id}: no valid endpoint voxel inside sphere with HU > {min_hu}.")
        return None

    dists = np.linalg.norm(valid_coords - ideal_center, axis=1)
    best_idx = int(np.argmin(dists))
    best_coord = tuple(int(v) for v in valid_coords[best_idx].tolist())

    print("-" * 100)
    print(f"[PRIMARY ENDPOINT LABEL {label_id}]")
    print(f"  Ideal center: {np.round(ideal_center, 2)}")
    print(f"  Selected endpoint inside sphere: {best_coord}")
    print(f"  HU at endpoint: {float(valid_hu[best_idx]):.1f}")
    print(f"  Distance to center: {float(dists[best_idx]):.3f} voxels")
    return best_coord


def find_fallback_endpoint_near_sphere(
    label_id: int,
    spheres_data: np.ndarray,
    ct_data: np.ndarray,
    reachable_mask: Optional[np.ndarray] = None,
    max_radius: int = FALLBACK_MAX_RADIUS,
    distance_tolerance: float = FALLBACK_DISTANCE_TOLERANCE,
) -> Tuple[int, int, int]:
    sphere_coords = np.argwhere(spheres_data == label_id)
    if len(sphere_coords) == 0:
        raise RuntimeError(f"Cannot find fallback endpoint because sphere label {label_id} is empty.")

    shape = np.array(ct_data.shape, dtype=np.int64)
    sphere_min = sphere_coords.min(axis=0)
    sphere_max = sphere_coords.max(axis=0)
    sphere_tree = KDTree(sphere_coords)

    print("-" * 100)
    print(f"[FALLBACK ENDPOINT SEARCH] label={label_id}")
    print(f"[INFO] Sphere bbox min: {sphere_min.tolist()}, max: {sphere_max.tolist()}")
    print(f"[INFO] Max search radius: {max_radius}")
    print(f"[INFO] Distance tolerance for HU preference: {distance_tolerance}")
    print(f"[INFO] Require same reachable component: {reachable_mask is not None}")

    for radius in range(0, max_radius + 1):
        min_corner = np.maximum(0, sphere_min - radius)
        max_corner = np.minimum(shape - 1, sphere_max + radius)
        sx0, sy0, sz0 = [int(x) for x in min_corner]
        sx1, sy1, sz1 = [int(x) for x in max_corner]

        sub_ct = ct_data[sx0:sx1 + 1, sy0:sy1 + 1, sz0:sz1 + 1]
        valid = sub_ct != INVALID_HU_VALUE

        if reachable_mask is not None:
            sub_reachable = reachable_mask[sx0:sx1 + 1, sy0:sy1 + 1, sz0:sz1 + 1]
            valid = valid & sub_reachable

        if not np.any(valid):
            continue

        local_coords = np.argwhere(valid)
        global_coords = local_coords + np.array([sx0, sy0, sz0], dtype=np.int64)
        hu_values = ct_data[global_coords[:, 0], global_coords[:, 1], global_coords[:, 2]]

        distances, _ = sphere_tree.query(global_coords)
        min_distance = float(np.min(distances))
        close_mask = distances <= (min_distance + distance_tolerance)

        close_coords = global_coords[close_mask]
        close_hu = hu_values[close_mask]
        close_dist = distances[close_mask]

        # Highest HU among similarly close voxels; if tied, closer wins.
        order = np.lexsort((close_dist, -close_hu))
        best_idx = int(order[0])
        best_coord = close_coords[best_idx]
        best_tuple = tuple(int(v) for v in best_coord.tolist())

        print(f"[FALLBACK FOUND] radius={radius}")
        print(f"  Closest distance to sphere: {float(close_dist[best_idx]):.3f} voxels")
        print(f"  Selected fallback endpoint: {best_tuple}")
        print(f"  HU at fallback endpoint: {float(close_hu[best_idx]):.1f}")
        print(f"  Candidates within tolerance: {len(close_coords):,}")
        return best_tuple

    raise RuntimeError(f"No fallback endpoint found near sphere label {label_id} within radius {max_radius}.")


def get_endpoint_records_from_spheres(
    spheres_data: np.ndarray,
    ct_data: np.ndarray,
    min_hu: float = ENDPOINT_MIN_HU,
) -> List[Dict]:
    labels_found = np.unique(spheres_data)
    labels_found = labels_found[labels_found > 0]

    print("=" * 100)
    print("[HEPATIC HILUM ENDPOINT DETECTION]")
    print(f"[N LABELS FOUND] {len(labels_found)}")
    print(f"[LABELS] {[int(x) for x in labels_found.tolist()]}")
    print("=" * 100)

    if len(labels_found) == 0:
        raise RuntimeError("No positive label was found in hepatic hilum sphere mask.")

    endpoint_records: List[Dict] = []
    for label_id in labels_found:
        label_int = int(label_id)
        endpoint_records.append({
            "sphere_label": label_int,
            "primary_endpoint": find_best_endpoint_inside_sphere(
                label_int,
                spheres_data,
                ct_data,
                min_hu=min_hu,
            ),
        })

    if len(endpoint_records) > 2:
        print(f"[WARNING] More than 2 sphere labels detected ({len(endpoint_records)}). Only the first 2 will be used.")
        endpoint_records = endpoint_records[:2]

    print("=" * 100)
    print("[ENDPOINT RECORDS]")
    for i, rec in enumerate(endpoint_records, start=1):
        print(f"  Target {i}: sphere_label={rec['sphere_label']}, primary_endpoint={rec['primary_endpoint']}")
    print("=" * 100)
    return endpoint_records


# ============================================================
# 6. Dijkstra Pathfinding
# ============================================================
def voxel_step_cost(hu_val: float) -> float:
    step_cost = 1.0
    if hu_val < 40:
        hu_penalty = (40 - hu_val) ** 2 + 500
    else:
        hu_penalty = max(0, 250 - hu_val)
    return float(step_cost + hu_penalty)


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


def point_in_bounds(point: Tuple[int, int, int], lower: np.ndarray, upper: np.ndarray) -> bool:
    p = np.array(point, dtype=np.int64)
    return bool(np.all(p >= lower) and np.all(p <= upper))


def dijkstra_pathfinder_once(
    ct_data: np.ndarray,
    start: Tuple[int, int, int],
    end: Tuple[int, int, int],
    bbox_margin: Optional[int] = None,
) -> Tuple[Optional[List[Tuple[int, int, int]]], float, int]:
    rows, cols, slices = ct_data.shape

    if not is_inside_shape(start, ct_data.shape):
        raise ValueError(f"Start point out of bounds: {start}")
    if not is_inside_shape(end, ct_data.shape):
        raise ValueError(f"End point out of bounds: {end}")

    if AVOID_INVALID_HU and float(ct_data[start]) == INVALID_HU_VALUE:
        print(f"[WARNING] Dijkstra start point is invalid HU: {start}")
        return None, float("inf"), 0
    if AVOID_INVALID_HU and float(ct_data[end]) == INVALID_HU_VALUE:
        print(f"[WARNING] Dijkstra endpoint is invalid HU: {end}")
        return None, float("inf"), 0

    lower, upper = make_bbox_bounds(ct_data.shape, start, end, bbox_margin)
    if not point_in_bounds(start, lower, upper) or not point_in_bounds(end, lower, upper):
        return None, float("inf"), 0

    queue = [(0.0, start)]
    distances = {start: 0.0}
    predecessor = {start: None}
    neighbors = get_neighbor_offsets(USE_26_NEIGHBOR_PATHFINDING)
    visited = set()
    expanded_nodes = 0

    while queue:
        curr_cost, curr_pos = heapq.heappop(queue)

        if curr_pos in visited:
            continue
        visited.add(curr_pos)
        expanded_nodes += 1

        if curr_pos == end:
            path = []
            p = curr_pos
            while p is not None:
                path.append(p)
                p = predecessor[p]
            path = path[::-1]
            return path, float(curr_cost), expanded_nodes

        if curr_cost > distances.get(curr_pos, float("inf")):
            continue

        for dx, dy, dz, movement_length in neighbors:
            next_pos = (curr_pos[0] + dx, curr_pos[1] + dy, curr_pos[2] + dz)

            if not (0 <= next_pos[0] < rows and 0 <= next_pos[1] < cols and 0 <= next_pos[2] < slices):
                continue

            next_arr = np.array(next_pos, dtype=np.int64)
            if np.any(next_arr < lower) or np.any(next_arr > upper):
                continue

            hu_val = float(ct_data[next_pos])
            if AVOID_INVALID_HU and hu_val == INVALID_HU_VALUE:
                continue

            new_cost = curr_cost + (movement_length * voxel_step_cost(hu_val))

            if new_cost < distances.get(next_pos, float("inf")):
                distances[next_pos] = new_cost
                predecessor[next_pos] = curr_pos
                heapq.heappush(queue, (new_cost, next_pos))

    return None, float("inf"), expanded_nodes


def dijkstra_pathfinder(
    ct_data: np.ndarray,
    start: Tuple[int, int, int],
    end: Tuple[int, int, int],
) -> Tuple[Optional[List[Tuple[int, int, int]]], float]:
    print(f"  [Dijkstra] Connectivity: {'26-neighbor' if USE_26_NEIGHBOR_PATHFINDING else '6-neighbor'}")

    for margin in DIJKSTRA_BBOX_MARGINS:
        margin_text = "full-volume" if margin is None else f"bbox margin {margin}"
        print(f"  [Dijkstra] Trying {margin_text}...")
        path, cost, expanded = dijkstra_pathfinder_once(
            ct_data=ct_data,
            start=start,
            end=end,
            bbox_margin=margin,
        )
        if path is not None:
            print(f"  [Dijkstra] Success with {margin_text}. Expanded nodes: {expanded:,}")
            return path, cost
        print(f"  [Dijkstra] Failed with {margin_text}. Expanded nodes: {expanded:,}")

    return None, float("inf")


def trace_path_for_start_and_sphere(
    ct_data: np.ndarray,
    spheres_data: np.ndarray,
    start: Tuple[int, int, int],
    sphere_label: int,
    primary_endpoint: Optional[Tuple[int, int, int]],
    reachable_mask: Optional[np.ndarray],
) -> Dict:
    """
    Try one start point to one hepatic hilum sphere.

    The path can use the primary endpoint inside the sphere, or a fallback endpoint
    near the sphere if the primary endpoint cannot be reached.
    """
    if primary_endpoint is not None:
        print(f"[TRY PRIMARY ENDPOINT] start={start}, sphere_label={sphere_label}, endpoint={primary_endpoint}")
        path, cost = dijkstra_pathfinder(ct_data, start, primary_endpoint)
        if path is not None:
            return {
                "path": path,
                "cost": cost,
                "endpoint": primary_endpoint,
                "endpoint_type": "primary_endpoint",
                "num_voxels": len(path),
                "status": "ok",
            }
        print(f"[WARNING] Primary endpoint path failed: start={start}, sphere_label={sphere_label}")
    else:
        print(f"[WARNING] No primary endpoint available: start={start}, sphere_label={sphere_label}")

    if not ENABLE_FALLBACK_ENDPOINT:
        return {
            "path": None,
            "cost": float("inf"),
            "endpoint": None,
            "endpoint_type": "none",
            "num_voxels": 0,
            "status": "failed_no_fallback",
        }

    try:
        fallback_endpoint = find_fallback_endpoint_near_sphere(
            label_id=sphere_label,
            spheres_data=spheres_data,
            ct_data=ct_data,
            reachable_mask=reachable_mask if FALLBACK_REQUIRE_SAME_VALID_COMPONENT_AS_START else None,
            max_radius=FALLBACK_MAX_RADIUS,
            distance_tolerance=FALLBACK_DISTANCE_TOLERANCE,
        )
    except Exception as e:
        print(f"[WARNING] Fallback endpoint search failed: {e}")
        return {
            "path": None,
            "cost": float("inf"),
            "endpoint": None,
            "endpoint_type": "fallback_search_failed",
            "num_voxels": 0,
            "status": f"failed: {e}",
        }

    print(f"[TRY FALLBACK ENDPOINT] start={start}, sphere_label={sphere_label}, endpoint={fallback_endpoint}")
    path, cost = dijkstra_pathfinder(ct_data, start, fallback_endpoint)

    if path is None:
        return {
            "path": None,
            "cost": float("inf"),
            "endpoint": fallback_endpoint,
            "endpoint_type": "fallback_endpoint_failed_path",
            "num_voxels": 0,
            "status": "failed_fallback_path",
        }

    return {
        "path": path,
        "cost": cost,
        "endpoint": fallback_endpoint,
        "endpoint_type": "fallback_endpoint_nearest_sphere_valid_hu_high_preferred",
        "num_voxels": len(path),
        "status": "ok",
    }


# ============================================================
# 7. Pairing Optimization
# ============================================================
def compute_all_start_sphere_paths(
    ct_data: np.ndarray,
    spheres_data: np.ndarray,
    start_points: List[Tuple[int, int, int]],
    endpoint_records: List[Dict],
    reachable_masks: List[Optional[np.ndarray]],
) -> Dict[Tuple[int, int], Dict]:
    results: Dict[Tuple[int, int], Dict] = {}

    print("=" * 100)
    print("[COMPUTE ALL START-SPHERE PATHS]")
    print(f"[N STARTS ] {len(start_points)}")
    print(f"[N SPHERES] {len(endpoint_records)}")
    print("=" * 100)

    for si, start in enumerate(start_points):
        for ti, rec in enumerate(endpoint_records):
            sphere_label = int(rec["sphere_label"])
            print("-" * 100)
            print(
                f"[PATH CANDIDATE] start {si + 1}/{len(start_points)} {start} "
                f"-> sphere {ti + 1}/{len(endpoint_records)} label={sphere_label}"
            )

            result = trace_path_for_start_and_sphere(
                ct_data=ct_data,
                spheres_data=spheres_data,
                start=start,
                sphere_label=sphere_label,
                primary_endpoint=rec["primary_endpoint"],
                reachable_mask=reachable_masks[si],
            )

            if result["path"] is None:
                print(f"  [FAILED] {result['status']}")
            else:
                print(
                    f"  [OK] cost={result['cost']:.3f}, "
                    f"path voxels={result['num_voxels']:,}, endpoint_type={result['endpoint_type']}"
                )

            results[(si, ti)] = result

    return results


def choose_best_assignment(
    path_results: Dict[Tuple[int, int], Dict],
    num_starts: int,
    num_targets: int,
) -> Tuple[List[int], float]:
    """
    Choose one start for each sphere target to minimize total path cost.

    Important: starts are reusable. Therefore both hepatic hilum sphere paths can
    choose the same search3231 start point if that produces the lowest total cost.
    """
    best_assignment = None
    best_total_cost = float("inf")

    for assignment in itertools.product(range(num_starts), repeat=num_targets):
        total_cost = 0.0
        feasible = True

        for target_idx, start_idx in enumerate(assignment):
            cost = path_results[(start_idx, target_idx)]["cost"]
            if not np.isfinite(cost):
                feasible = False
                break
            total_cost += cost

        if feasible and total_cost < best_total_cost:
            best_total_cost = float(total_cost)
            best_assignment = list(assignment)

    if best_assignment is None:
        raise RuntimeError("No feasible start-sphere assignment was found.")

    print("=" * 100)
    print("[BEST START-SPHERE ASSIGNMENT]")
    print(f"[TOTAL COST] {best_total_cost:.3f}")
    for target_idx, start_idx in enumerate(best_assignment):
        result = path_results[(start_idx, target_idx)]
        print(
            f"  Sphere target {target_idx + 1} uses Start {start_idx + 1} | "
            f"cost={result['cost']:.3f}, endpoint={result['endpoint']}, type={result['endpoint_type']}"
        )
    print("=" * 100)

    return best_assignment, best_total_cost


# ============================================================
# 8. Rendering and Summary
# ============================================================
def render_paths_to_labelmap(
    ct_data: np.ndarray,
    selected_paths: List[List[Tuple[int, int, int]]],
) -> np.ndarray:
    vessel_mask = np.zeros(ct_data.shape, dtype=np.uint16)
    struct = np.ones((3, 3, 3), dtype=bool)

    for i, path in enumerate(selected_paths):
        label_value = i + 1
        temp_path_mask = np.zeros(ct_data.shape, dtype=bool)

        for p in path:
            temp_path_mask[p] = True

        dilated_path = binary_dilation(
            temp_path_mask,
            structure=struct,
            iterations=PATH_DILATION_ITERATIONS,
        )

        write_mask = dilated_path & (vessel_mask == 0)
        vessel_mask[write_mask] = label_value

    return vessel_mask


def write_summary(
    summary_path: Path,
    case_id: str,
    ct_path: Path,
    anchor_txt_path: Path,
    start_points: List[Tuple[int, int, int]],
    endpoint_records: List[Dict],
    best_assignment: List[int],
    best_total_cost: float,
    path_results: Dict[Tuple[int, int], Dict],
) -> None:
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"case_id\t{case_id}\n")
        f.write("ct_source\tROI_CT\n")
        f.write(f"ct_path\t{ct_path}\n")
        f.write(f"anchor_points_txt\t{anchor_txt_path}\n")
        f.write(f"start_source\t{START_VOLUME_NAME}\n")
        f.write(f"invalid_hu_value\t{INVALID_HU_VALUE}\n")
        f.write(f"avoid_invalid_hu\t{AVOID_INVALID_HU}\n")
        f.write(f"use_26_neighbor_pathfinding\t{USE_26_NEIGHBOR_PATHFINDING}\n")
        f.write(f"dijkstra_bbox_margins\t{DIJKSTRA_BBOX_MARGINS}\n")
        f.write(f"fallback_enabled\t{ENABLE_FALLBACK_ENDPOINT}\n")
        f.write(f"fallback_max_radius\t{FALLBACK_MAX_RADIUS}\n")
        f.write(f"fallback_distance_tolerance\t{FALLBACK_DISTANCE_TOLERANCE}\n")
        f.write(f"fallback_require_same_component_as_start\t{FALLBACK_REQUIRE_SAME_VALID_COMPONENT_AS_START}\n")
        f.write(f"best_total_cost\t{best_total_cost:.6f}\n")
        f.write("\n")

        f.write("START_POINTS\n")
        f.write("start_index\tx\ty\tz\thu\n")
        # HU values are written later by lookup in selected path output; this section keeps coords only.
        for i, p in enumerate(start_points, start=1):
            f.write(f"{i}\t{p[0]}\t{p[1]}\t{p[2]}\tNA\n")
        f.write("\n")

        f.write("SPHERE_TARGETS\n")
        f.write("target_index\tsphere_label\tprimary_endpoint\n")
        for i, rec in enumerate(endpoint_records, start=1):
            f.write(f"{i}\t{rec['sphere_label']}\t{rec['primary_endpoint']}\n")
        f.write("\n")

        f.write("SELECTED_PATHS\n")
        f.write(
            "path_label\ttarget_index\tsphere_label\tstart_index\t"
            "start_x\tstart_y\tstart_z\t"
            "endpoint_x\tendpoint_y\tendpoint_z\tendpoint_type\t"
            "cost\tpath_voxels\n"
        )

        for target_idx, start_idx in enumerate(best_assignment):
            start = start_points[start_idx]
            rec = endpoint_records[target_idx]
            result = path_results[(start_idx, target_idx)]
            endpoint = result["endpoint"]
            f.write(
                f"{target_idx + 1}\t{target_idx + 1}\t{rec['sphere_label']}\t{start_idx + 1}\t"
                f"{start[0]}\t{start[1]}\t{start[2]}\t"
                f"{endpoint[0]}\t{endpoint[1]}\t{endpoint[2]}\t{result['endpoint_type']}\t"
                f"{result['cost']:.6f}\t{result['num_voxels']}\n"
            )


# ============================================================
# 9. Main Pipeline Function
# ============================================================
def run_hepatic_vessel_pathfinder(case_id: str) -> Tuple[Path, Path]:
    ct_path = build_roi_ct_path(case_id)
    spheres_path = build_spheres_path(case_id)
    anchor_txt_path = build_anchor_points_txt_path(case_id)
    output_path = build_output_path(case_id)
    summary_path = build_summary_path(case_id)

    print("=" * 100)
    print("[HEPATIC VESSEL PATHFINDER - SEARCH3231 DUAL-START VERSION]")
    print(f"[CASE ID  ] {case_id}")
    print(f"[ROI CT   ] {ct_path}")
    print(f"[SPHERES  ] {spheres_path}")
    print(f"[START TXT] {anchor_txt_path}")
    print(f"[OUTPUT   ] {output_path}")
    print("=" * 100)

    ensure_dir(output_path.parent)

    if SKIP_IF_FINAL_EXISTS and output_path.exists() and summary_path.exists():
        print(f"[SKIP] Hepatic vessel output already exists: {output_path}")
        print(f"[SKIP] Summary already exists: {summary_path}")
        return output_path, summary_path

    ct_img, ct_data = load_nifti(ct_path)
    spheres_img_raw, spheres_data_raw = load_nifti(spheres_path)
    spheres_img, spheres_data = align_spheres_to_ct_if_needed(spheres_img_raw, spheres_data_raw, ct_img)

    print(f"[INFO] ROI CT shape: {ct_data.shape}")
    print(f"[INFO] ROI CT HU range: min={ct_data.min():.1f}, max={ct_data.max():.1f}")
    print(f"[INFO] Sphere mask shape: {spheres_data.shape}")

    start_points = get_start_points_from_search3231(anchor_txt_path, ct_data)
    reachable_masks = build_reachable_masks_for_starts(ct_data, start_points)

    endpoint_records = get_endpoint_records_from_spheres(
        spheres_data=spheres_data,
        ct_data=ct_data,
        min_hu=ENDPOINT_MIN_HU,
    )

    if len(endpoint_records) == 0:
        raise RuntimeError("No endpoint records were found from hepatic hilum spheres.")

    path_results = compute_all_start_sphere_paths(
        ct_data=ct_data,
        spheres_data=spheres_data,
        start_points=start_points,
        endpoint_records=endpoint_records,
        reachable_masks=reachable_masks,
    )

    best_assignment, best_total_cost = choose_best_assignment(
        path_results=path_results,
        num_starts=len(start_points),
        num_targets=len(endpoint_records),
    )

    selected_paths: List[List[Tuple[int, int, int]]] = []
    for target_idx, start_idx in enumerate(best_assignment):
        selected_path = path_results[(start_idx, target_idx)]["path"]
        if selected_path is None:
            raise RuntimeError("Internal error: selected assignment contains a failed path.")
        selected_paths.append(selected_path)

    final_mask = render_paths_to_labelmap(ct_data, selected_paths)

    output_header = ct_img.header.copy()
    output_header.set_data_dtype(np.uint16)

    output_img = nib.Nifti1Image(
        final_mask.astype(np.uint16),
        affine=ct_img.affine,
        header=output_header,
    )
    nib.save(output_img, str(output_path))

    if not output_path.exists():
        raise RuntimeError(f"Output was not created: {output_path}")

    write_summary(
        summary_path=summary_path,
        case_id=case_id,
        ct_path=ct_path,
        anchor_txt_path=anchor_txt_path,
        start_points=start_points,
        endpoint_records=endpoint_records,
        best_assignment=best_assignment,
        best_total_cost=best_total_cost,
        path_results=path_results,
    )

    print("=" * 100)
    print("[DONE] Hepatic vessel pathfinding completed.")
    print(f"[OUTPUT MASK] {output_path}")
    print(f"[SUMMARY    ] {summary_path}")
    print("=" * 100)

    return output_path, summary_path


# ============================================================
# 10. Command-Line Interface for run.py
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hepatic vessel pathfinding from search3231 starting points to hepatic hilum spheres."
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
# 11. Main Entry Point
# ============================================================
def main() -> None:
    args = parse_args()
    ct_path = Path(args.ct_path).resolve()

    case_id = args.case_id.strip() if args.case_id else strip_nii_extension(ct_path.name)
    if not case_id:
        raise ValueError("case_id is empty.")

    run_hepatic_vessel_pathfinder(case_id=case_id)


if __name__ == "__main__":
    main()
