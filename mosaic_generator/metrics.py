"""
Similarity metrics module for mosaic quality evaluation.

This module provides functions to compute various similarity metrics
between images, including MSE, SSIM, and MS-SSIM for quantitative
assessment of mosaic quality.

Functions:
    compute_mse: Mean Squared Error
    compute_ssim: Structural Similarity Index
    compute_ms_ssim: Multi-Scale Structural Similarity Index
    
Author: Pratham Satani
Course: CS5130 - Applied Programming and Data Processing for AI
Institution: Northeastern University
"""

import numpy as np
import torch
from torch import Tensor
from typing import Union, Tuple
from pytorch_msssim import ms_ssim, ssim
import cv2

from .utils import numpy_to_tensor, tensor_to_numpy, match_dimensions, logger


def compute_mse(image1: np.ndarray, image2: np.ndarray, 
                match_size: bool = True) -> float:
    """
    Compute Mean Squared Error between two images.
    
    MSE measures the average squared difference between corresponding
    pixels. Lower values indicate higher similarity (0 = identical).
    
    Args:
        image1: First image as NumPy array
        image2: Second image as NumPy array
        match_size: If True, crops images to match dimensions
        
    Returns:
        float: MSE value (lower is better, 0 = identical)
        
    Raises:
        ValueError: If image shapes don't match and match_size=False
        
    Example:
        >>> img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        >>> img2 = img1.copy()
        >>> mse = compute_mse(img1, img2)
        >>> print(f"MSE: {mse:.2f}")
        MSE: 0.00
    """
    # Match dimensions if requested
    if match_size and image1.shape != image2.shape:
        target_shape = (
            min(image1.shape[0], image2.shape[0]),
            min(image1.shape[1], image2.shape[1])
        )
        image1 = match_dimensions(image1, target_shape)
        image2 = match_dimensions(image2, target_shape)
    
    if image1.shape != image2.shape:
        raise ValueError(
            f"Image shapes must match: {image1.shape} vs {image2.shape}"
        )
    
    # Compute MSE
    mse = np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)
    
    logger.debug(f"Computed MSE: {mse:.4f}")
    return float(mse)


def compute_ssim(image1: Union[np.ndarray, Tensor],
                image2: Union[np.ndarray, Tensor],
                data_range: int = 255,
                match_size: bool = True) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.
    
    SSIM is a perceptual metric that quantifies image degradation
    as perceived change in structural information. Values range
    from -1 to 1, where 1 indicates perfect similarity.
    
    Args:
        image1: First image (NumPy array or Tensor)
        image2: Second image (NumPy array or Tensor)
        data_range: Dynamic range of pixel values (255 for uint8)
        match_size: If True, crops images to match dimensions
        
    Returns:
        float: SSIM value (higher is better, 1 = identical)
        
    Example:
        >>> img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        >>> img2 = img1 + np.random.randint(-10, 10, img1.shape, dtype=np.int16)
        >>> img2 = np.clip(img2, 0, 255).astype(np.uint8)
        >>> ssim_val = compute_ssim(img1, img2)
        >>> print(f"SSIM: {ssim_val:.4f}")
    """
    # Convert to tensors if needed
    if isinstance(image1, np.ndarray):
        if match_size and image1.shape != image2.shape:
            target_shape = (
                min(image1.shape[0], image2.shape[0]),
                min(image1.shape[1], image2.shape[1])
            )
            image1 = match_dimensions(image1, target_shape)
            image2 = match_dimensions(image2, target_shape)
        
        img1_tensor = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0).float()
        img2_tensor = torch.from_numpy(image2).permute(2, 0, 1).unsqueeze(0).float()
    else:
        img1_tensor = image1.unsqueeze(0) if image1.dim() == 3 else image1
        img2_tensor = image2.unsqueeze(0) if image2.dim() == 3 else image2
    
    # Compute SSIM
    ssim_val = ssim(
        img1_tensor,
        img2_tensor,
        data_range=data_range,
        size_average=True
    )
    
    result = ssim_val.item()
    logger.debug(f"Computed SSIM: {result:.4f}")
    return result


def compute_ms_ssim(image1: Union[np.ndarray, Tensor],
                   image2: Union[np.ndarray, Tensor],
                   data_range: int = 255,
                   match_size: bool = True) -> float:
    """
    Compute Multi-Scale Structural Similarity Index (MS-SSIM).
    
    MS-SSIM extends SSIM by evaluating similarity at multiple scales,
    providing a more robust perceptual quality metric. This is the
    recommended metric for mosaic quality evaluation.
    
    Args:
        image1: First image (NumPy array or Tensor)
        image2: Second image (NumPy array or Tensor)
        data_range: Dynamic range of pixel values (255 for uint8)
        match_size: If True, crops images to match dimensions
        
    Returns:
        float: MS-SSIM value (higher is better, 1 = identical)
        
    Raises:
        RuntimeError: If images are too small for MS-SSIM computation
        
    Example:
        >>> original = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        >>> mosaic = original.copy()  # Simplified example
        >>> ms_ssim_val = compute_ms_ssim(original, mosaic)
        >>> print(f"MS-SSIM: {ms_ssim_val:.4f}")
        MS-SSIM: 1.0000
    """
    # Convert to tensors if needed
    if isinstance(image1, np.ndarray):
        if match_size and image1.shape != image2.shape:
            target_shape = (
                min(image1.shape[0], image2.shape[0]),
                min(image1.shape[1], image2.shape[1])
            )
            image1 = match_dimensions(image1, target_shape)
            image2 = match_dimensions(image2, target_shape)
        
        img1_tensor = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0).float()
        img2_tensor = torch.from_numpy(image2).permute(2, 0, 1).unsqueeze(0).float()
    else:
        img1_tensor = image1.unsqueeze(0) if image1.dim() == 3 else image1
        img2_tensor = image2.unsqueeze(0) if image2.dim() == 3 else image2
    
    try:
        # Compute MS-SSIM
        ms_ssim_val = ms_ssim(
            img1_tensor,
            img2_tensor,
            data_range=data_range,
            size_average=True
        )
        
        result = ms_ssim_val.item()
        logger.debug(f"Computed MS-SSIM: {result:.4f}")
        return result
        
    except RuntimeError as e:
        raise RuntimeError(
            f"MS-SSIM computation failed (images may be too small): {str(e)}"
        )


def compute_psnr(image1: np.ndarray, image2: np.ndarray,
                match_size: bool = True) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    PSNR is expressed in decibels (dB). Higher values indicate better
    quality. Typical values range from 20-50 dB.
    
    Args:
        image1: First image
        image2: Second image
        match_size: If True, crops images to match dimensions
        
    Returns:
        float: PSNR value in dB (higher is better)
        
    Example:
        >>> img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        >>> img2 = img1.copy()
        >>> psnr = compute_psnr(img1, img2)
        >>> print(f"PSNR: {psnr:.2f} dB")
    """
    # Match dimensions if requested
    if match_size and image1.shape != image2.shape:
        target_shape = (
            min(image1.shape[0], image2.shape[0]),
            min(image1.shape[1], image2.shape[1])
        )
        image1 = match_dimensions(image1, target_shape)
        image2 = match_dimensions(image2, target_shape)
    
    # Compute MSE
    mse = compute_mse(image1, image2, match_size=False)
    
    if mse == 0:
        return float('inf')  # Images are identical
    
    # Compute PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    logger.debug(f"Computed PSNR: {psnr:.2f} dB")
    return float(psnr)


def evaluate_mosaic_quality(original: np.ndarray,
                           mosaic: np.ndarray,
                           metrics: Tuple[str, ...] = ("ms_ssim", "ssim", "mse", "psnr")) -> dict:
    """
    Evaluate mosaic quality using multiple metrics.
    
    Args:
        original: Original image
        mosaic: Generated mosaic
        metrics: Tuple of metric names to compute
        
    Returns:
        dict: Dictionary with computed metrics
        
    Example:
        >>> original = load_image("original.jpg")
        >>> mosaic = load_image("mosaic.jpg")
        >>> results = evaluate_mosaic_quality(original, mosaic)
        >>> for metric, value in results.items():
        ...     print(f"{metric}: {value:.4f}")
    """
    results = {}
    
    if "mse" in metrics:
        results["mse"] = compute_mse(original, mosaic)
    
    if "ssim" in metrics:
        results["ssim"] = compute_ssim(original, mosaic)
    
    if "ms_ssim" in metrics:
        results["ms_ssim"] = compute_ms_ssim(original, mosaic)
    
    if "psnr" in metrics:
        results["psnr"] = compute_psnr(original, mosaic)
    
    logger.info(f"Quality evaluation: {results}")
    return results
