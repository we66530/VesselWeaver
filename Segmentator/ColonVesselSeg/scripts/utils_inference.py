"""
Medical image segmentation utilities for processing and evaluating 3D volumes.

This module provides functions for reading and saving NIfTI, performing
segmentation inference with MONAI, calculating evaluation metrics, and post-processing
segmentation outputs. It is designed for use with PyTorch and MONAI models.
"""

import os
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Optional
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import scipy.ndimage as ndimage
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import euclidean

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def validate_paths(model_path: str, input_dir: str, ground_truth_dir: Optional[str], output_dir: str) -> None:
    """Validate input and output paths."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist")
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    if ground_truth_dir and not os.path.exists(ground_truth_dir):
        raise FileNotFoundError(f"Ground truth directory {ground_truth_dir} does not exist")
    os.makedirs(output_dir, exist_ok=True)

def read_nifti(path):
    nii = nib.load(path)
    data = nii.get_fdata()
    affine = nii.affine
    return data, affine

def save_nifti(data, affine, path, dtype=np.float32):
    nii = nib.Nifti1Image(data.astype(dtype), affine)
    nib.save(nii, path)

def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled

def calculate_metrics(pred, gt, class_idx=1):
    """
    Calculate multiple metrics for a specific class
    Args:
        pred: Prediction array (numpy array)
        gt: Ground truth array (numpy array)
        class_idx: Class index to calculate metrics for (1 for foreground)
    Returns:
        Dictionary containing metrics or NaN if no ground truth
    """
    # Flatten arrays for sklearn metrics
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    # Create binary masks for the class of interest
    pred_mask = (pred_flat == class_idx)
    gt_mask = (gt_flat == class_idx)
    
    if not np.any(gt_mask):  # if no ground truth, return NaN for all metrics
        return {
            'dice': np.nan,
            'accuracy': np.nan,
            'sensitivity': np.nan,
            'precision': np.nan,
            'specificity': np.nan,
            'tp': np.nan,
            'tn': np.nan,
            'fp': np.nan,
            'fn': np.nan,
            'n_pred': np.nan,
            'n_ref': np.nan
        }
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(gt_mask, pred_mask, labels=[False, True]).ravel()
    # Calculate number of predicted and reference foreground pixels
    n_pred = np.sum(pred_mask)
    n_ref = np.sum(gt_mask)
    # Calculate specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    
    # Calculate metrics
    metrics = {
        'dice': compute_dice(
            torch.from_numpy(pred_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(gt_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        ).item(),
        'accuracy': accuracy_score(gt_flat, pred_flat),
        'sensitivity': recall_score(gt_mask, pred_mask, zero_division=np.nan),
        'precision': precision_score(gt_mask, pred_mask, zero_division=np.nan),
        'specificity': specificity,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'n_pred': n_pred,
        'n_ref': n_ref
    }
    
    return metrics

def calculate_and_save_metrics(pred_resized: np.ndarray, gt_path: str, patient_id: str, output_dir: str, metrics_df: pd.DataFrame) -> None:
    """Calculate and log segmentation metrics."""
    if not os.path.exists(gt_path):
        logger.warning(f"No ground truth found for {patient_id}")
        metrics_df.loc[len(metrics_df)] = {
            "Patient": patient_id,
            "Dice": np.nan,
            "Accuracy": np.nan,
            "Sensitivity": np.nan,
            "Precision": np.nan,
            "Specificity": np.nan,
            "TP": np.nan,
            "TN": np.nan,
            "FP": np.nan,
            "FN": np.nan,
            "N_Pred": np.nan,
            "N_Ref": np.nan
        }
        return

    gt_img, _ = read_nifti(gt_path)
    metrics = calculate_metrics(pred_resized, gt_img)
    expected_keys = {"dice", "accuracy", "sensitivity", "precision", "specificity", "tp", "tn", "fp", "fn", "n_pred", "n_ref"}

    if not all(k in metrics for k in expected_keys):
        logger.warning(f"Missing metrics keys for {patient_id}: {metrics.keys()}")
        metrics = {k: np.nan for k in expected_keys}

    metrics_df.loc[len(metrics_df)] = {
        "Patient": patient_id,
        "Dice": metrics["dice"],
        "Accuracy": metrics["accuracy"],
        "Sensitivity": metrics["sensitivity"],
        "Precision": metrics["precision"],
        "Specificity": metrics["specificity"],
        "TP": metrics["tp"],
        "TN": metrics["tn"],
        "FP": metrics["fp"],
        "FN": metrics["fn"],
        "N_Pred": metrics["n_pred"],
        "N_Ref": metrics["n_ref"]
    }
    logger.info(
        f"Metrics for {patient_id}: "
        f"Dice={metrics['dice']:.4f}, "
        f"Accuracy={metrics['accuracy']:.4f}, "
        f"Sensitivity={metrics['sensitivity']:.4f}, "
        f"Precision={metrics['precision']:.4f}, "
        f"Specificity={metrics['specificity']:.4f}, "
        f"TP={metrics['tp']}, "
        f"TN={metrics['tn']}, "
        f"FP={metrics['fp']}, "
        f"FN={metrics['fn']}, "
        f"N_Pred={metrics['n_pred']}, "
        f"N_Ref={metrics['n_ref']}"
    )
    
def predict_volume(
    model: torch.nn.Module,
    input_data: dict,
    device: torch.device,
    roi_size: tuple = (96, 96, 96),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    mode: str = "constant"
) -> np.ndarray:
    """Predict segmentation for a single volume using sliding window inference"""
    with torch.no_grad():
        input_tensor = input_data["image"].to(device)
        
        output = sliding_window_inference(
            inputs=input_tensor,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
            mode=mode
        )

        # Get the class with highest probability
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    return pred

def post_process_far_clusters(pred, max_distance=50, reference_cluster="largest"):
    """
    Post-process segmentation to remove clusters far from the main cluster.
    
    Args:
        pred (np.ndarray): Binary or integer segmentation mask (foreground > 0, background = 0)
        max_distance (float): Maximum distance (in voxels) from the reference cluster's center of mass
        reference_cluster (str): Method to select reference cluster ("largest" or "centroid")
    
    Returns:
        np.ndarray: Post-processed segmentation mask
    """
    # Ensure pred is binary for connected component analysis
    pred_binary = (pred > 0).astype(np.uint8)
    
    # Label connected components
    labeled, num_features = label(pred_binary)
    
    if num_features == 0:
        return pred  # No foreground clusters found
    
    # Get cluster centers of mass
    cluster_centers = []
    for i in range(1, num_features + 1):
        cluster_mask = (labeled == i)
        if np.sum(cluster_mask) > 0:  # Ensure cluster is non-empty
            cluster_centers.append(center_of_mass(cluster_mask))
        else:
            cluster_centers.append(None)
    
    # Select reference cluster
    if reference_cluster == "largest":
        # Find the largest cluster
        cluster_sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        ref_idx = np.argmax(cluster_sizes)  # Index of largest cluster (0-based)
    elif reference_cluster == "centroid":
        # Choose cluster closest to image center
        image_center = np.array(pred.shape) / 2
        distances_to_center = [
            euclidean(center, image_center) if center is not None else np.inf
            for center in cluster_centers
        ]
        ref_idx = np.argmin(distances_to_center)  # Index of closest cluster
    else:
        raise ValueError("reference_cluster must be 'largest' or 'centroid'")
    
    ref_center = cluster_centers[ref_idx]
    if ref_center is None:
        return pred  # Reference cluster is empty
    
    # Filter clusters based on distance
    output = np.zeros_like(pred)
    for i in range(1, num_features + 1):
        cluster_mask = (labeled == i)
        if np.sum(cluster_mask) == 0:
            continue
        center = cluster_centers[i - 1]
        if center is None:
            continue
        distance = euclidean(center, ref_center)
        
        # Keep cluster if it is within max_distance
        if distance <= max_distance:
            output[cluster_mask] = pred[cluster_mask]
    
    return output

