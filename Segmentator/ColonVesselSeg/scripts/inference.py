"""
Single-case colon vessel segmentation inference script for run.py integration.

Expected project structure:

ColonVesselSeg/
├─ scripts/
│  ├─ inference.py
│  └─ utils_inference.py
└─ weights/
   └─ veins_segmentation_model.pth

This script:
1. Receives one CT NIfTI file from run.py through --ct_path.
2. Automatically loads weights/veins_segmentation_model.pth.
3. Runs inference only on that single CT volume.
4. Saves the prediction into AbdVesselGen/Seg_Done/<case_id>_veins_segmentation/.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
)
from monai.networks.nets import UNet

# Ensure that utils_inference.py in the same scripts folder is importable.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils_inference import predict_volume, save_nifti, resample_3d


# ============================================================
# 1. Global Settings
# ============================================================
SEG_DONE_ROOT = Path(r"C:\Users\User\Desktop\AbdVesselGen\Seg_Done")
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "veins_segmentation_model.pth"

SKIP_IF_FINAL_EXISTS = True

# Model / inference configuration inherited from your original script.
IN_CHANNELS = 1
OUT_CHANNELS = 2
ROI_SIZE = (96, 96, 96)
SW_BATCH_SIZE = 4
SW_OVERLAP = 0.75


# ============================================================
# 2. Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# 3. MONAI Preprocessing
# ============================================================
test_transforms = Compose([
    LoadImaged(keys=["image"], image_only=False),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000,
        a_max=1000,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),
    EnsureTyped(keys=["image"]),
])


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
    return os.path.splitext(filename)[0]


def build_case_output_dir(case_id: str) -> Path:
    return SEG_DONE_ROOT / f"{case_id}_veins_segmentation"


def build_prediction_path(case_id: str) -> Path:
    return build_case_output_dir(case_id) / f"{case_id}_pred.nii.gz"


def validate_single_case_paths(ct_path: Path, weights_path: Path) -> None:
    if not ct_path.exists():
        raise FileNotFoundError(f"CT file not found: {ct_path}")

    lower = ct_path.name.lower()
    if not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
        raise ValueError(f"Input file must be .nii or .nii.gz: {ct_path}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")


def to_numpy_affine(value: Any) -> np.ndarray:
    """Convert MONAI affine metadata into a standard 4x4 NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    else:
        value = np.asarray(value)

    value = np.asarray(value)
    if value.ndim == 3:
        value = value[0]

    return value.astype(np.float64)


def to_shape_tuple(value: Any, fallback_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Convert MONAI original_shape / spatial_shape metadata into a Python tuple."""
    if value is None:
        return fallback_shape

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    else:
        value = np.asarray(value)

    value = np.asarray(value)
    if value.ndim >= 2:
        value = value[0]

    shape_tuple = tuple(int(x) for x in value.tolist())
    if len(shape_tuple) != 3:
        return fallback_shape
    return shape_tuple


# ============================================================
# 5. Model Loading
# ============================================================
def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """Load the trained 3D MONAI UNet model."""
    try:
        model = UNet(
            spatial_dims=3,
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

        state_dict = torch.load(str(model_path), map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        logger.info(f"Loaded model weights from: {model_path}")
        return model

    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        raise
    except RuntimeError as e:
        logger.error(f"Failed to load model weights: {str(e)}")
        raise


# ============================================================
# 6. Single-Case Inference
# ============================================================
def run_single_case_inference(
    ct_path: Path,
    case_id: str,
    weights_path: Path,
    device: torch.device,
) -> Path:
    """Run inference for one CT and save one NIfTI prediction."""
    validate_single_case_paths(ct_path, weights_path)
    ensure_dir(SEG_DONE_ROOT)

    output_dir = build_case_output_dir(case_id)
    output_path = build_prediction_path(case_id)
    ensure_dir(output_dir)

    if SKIP_IF_FINAL_EXISTS and output_path.exists():
        print("=" * 100)
        print("[SKIP] Colon vessel prediction already exists.")
        print(f"[CASE ID] {case_id}")
        print(f"[OUTPUT ] {output_path}")
        print("=" * 100)
        return output_path

    print("\n" + "=" * 100)
    print("[COLON VESSEL SEGMENTATION INFERENCE]")
    print(f"[CASE ID] {case_id}")
    print(f"[INPUT  ] {ct_path}")
    print(f"[WEIGHTS] {weights_path}")
    print(f"[OUTPUT ] {output_path}")
    print(f"[DEVICE ] {device}")
    print("=" * 100)

    model = load_model(weights_path, device)

    dataset = Dataset(
        data=[{"image": str(ct_path)}],
        transform=test_transforms,
    )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    for batch_data in dataloader:
        img_meta = batch_data["image"].meta

        original_affine_raw = img_meta.get("original_affine", img_meta.get("affine"))
        fallback_affine = np.eye(4, dtype=np.float64)
        original_affine = (
            to_numpy_affine(original_affine_raw)
            if original_affine_raw is not None
            else fallback_affine
        )

        fallback_shape = tuple(int(x) for x in batch_data["image"].shape[-3:])
        original_shape = to_shape_tuple(
            img_meta.get("original_shape", img_meta.get("spatial_shape")),
            fallback_shape=fallback_shape,
        )

        logger.info(f"Original affine:\n{original_affine}")
        logger.info(f"Original shape: {original_shape}")
        logger.info(f"Preprocessed tensor shape: {tuple(batch_data['image'].shape)}")

        pred = predict_volume(
            model=model,
            input_data=batch_data,
            device=device,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            overlap=SW_OVERLAP,
            mode="constant",
        )

        pred_resampled = resample_3d(pred, original_shape)
        save_nifti(
            data=pred_resampled,
            affine=original_affine,
            path=str(output_path),
            dtype=np.uint8,
        )

        if not output_path.exists():
            raise RuntimeError(f"Prediction save failed: {output_path}")

        file_size_mb = output_path.stat().st_size / (1024 ** 2)
        logger.info(f"Saved prediction to: {output_path}")
        logger.info(f"Prediction file size: {file_size_mb:.2f} MB")

    torch.cuda.empty_cache()

    print("=" * 100)
    print("==> Single-case colon vessel segmentation inference completed successfully.")
    print("=" * 100)

    return output_path


# ============================================================
# 7. Command-Line Interface for run.py
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-case colon vessel segmentation inference for run.py."
    )

    parser.add_argument(
        "--ct_path",
        required=True,
        help="Path to the single CT .nii or .nii.gz file loaded by run.py.",
    )

    # Kept for compatibility with run.py.
    # This script writes to SEG_DONE_ROOT instead of using this argument.
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory passed by run.py. Accepted for pipeline compatibility.",
    )

    parser.add_argument(
        "--case_id",
        required=False,
        default=None,
        help="Case ID provided by run.py. If omitted, it will be inferred from the CT filename.",
    )

    # Optional override for debugging. run.py does not need to pass this.
    parser.add_argument(
        "--weights_file",
        required=False,
        default=None,
        help="Optional custom model weights path. Defaults to weights/veins_segmentation_model.pth.",
    )

    return parser.parse_args()


# ============================================================
# 8. Main Entry Point
# ============================================================
def main() -> None:
    start_time = time.time()
    args = parse_args()

    ct_path = Path(args.ct_path).resolve()
    case_id = args.case_id.strip() if args.case_id else strip_nii_extension(ct_path.name)
    if not case_id:
        raise ValueError("case_id is empty.")

    weights_path = (
        Path(args.weights_file).resolve()
        if args.weights_file
        else DEFAULT_WEIGHTS_PATH.resolve()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    run_single_case_inference(
        ct_path=ct_path,
        case_id=case_id,
        weights_path=weights_path,
        device=device,
    )

    total_time = time.time() - start_time
    logger.info(f"Total inference time: {total_time:.2f} sec")


if __name__ == "__main__":
    main()
