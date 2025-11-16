"""
Utility functions for the mosaic generator package.

This module provides helper functions used across the package, including
image format conversions, color space transformations, and validation utilities.

Functions:
    rgb_to_text: Convert RGB values to color name
    tensor_to_numpy: Convert PyTorch tensor to NumPy array
    numpy_to_tensor: Convert NumPy array to PyTorch tensor
    validate_image: Validate image format and dimensions
    match_dimensions: Crop image to match target dimensions
    setup_logging: Configure logging for the package

Author: Pratham Satani
Course: CS5130 - Applied Programming and Data Processing for AI
Institution: Northeastern University
"""

import numpy as np
import torch
from torch import Tensor
from typing import Union, Tuple, Optional
import logging
from pathlib import Path

from .config import LOG_LEVEL, SUPPORTED_IMAGE_FORMATS


def setup_logging(level: str = LOG_LEVEL, name: str = "mosaic_generator") -> logging.Logger:
    """
    Configure and return a logger for the package.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        name: Logger name
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        >>> logger = setup_logging("INFO")
        >>> logger.info("Starting mosaic generation")
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


# Initialize package logger
logger = setup_logging()


def rgb_to_text(r: int, g: int, b: int) -> str:
    """
    Convert RGB values to a human-readable color name.
    
    Determines the dominant color based on the relative magnitudes
    of the red, green, and blue channels.
    
    Args:
        r: Red channel value (0-255)
        g: Green channel value (0-255)
        b: Blue channel value (0-255)
        
    Returns:
        str: Color name (e.g., 'Red', 'Green', 'Blue', 'Yellow', etc.)
             Returns 'Unknown' if no clear dominant color
        
    Example:
        >>> rgb_to_text(255, 0, 0)
        'Red'
        >>> rgb_to_text(255, 255, 0)
        'Yellow'
        >>> rgb_to_text(128, 128, 128)
        'Unknown'
    """
    # Primary colors
    if r > g and r > b:
        return "Red"
    elif g > r and g > b:
        return "Green"
    elif b > r and b > g:
        return "Blue"
    
    # Secondary colors (equal channel combinations)
    elif r == g and r > b:
        return "Yellow"
    elif r == b and r > g:
        return "Magenta"
    elif g == b and g > r:
        return "Cyan"
    
    # No clear dominant color
    return "Unknown"


def tensor_to_numpy(tensor: Tensor, denormalize: bool = True) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.
    
    Handles common tensor formats including:
    - Single images (C, H, W)
    - Batches (N, C, H, W)
    - Normalized tensors [0, 1] or [-1, 1]
    
    Args:
        tensor: Input PyTorch tensor
        denormalize: If True, converts [0, 1] range to [0, 255] uint8
        
    Returns:
        np.ndarray: Image as NumPy array
        
    Raises:
        ValueError: If tensor has unexpected dimensions
        
    Example:
        >>> tensor = torch.rand(3, 256, 256)
        >>> image = tensor_to_numpy(tensor)
        >>> print(image.shape, image.dtype)
        (256, 256, 3) uint8
    """
    # Move to CPU and detach from computation graph
    arr = tensor.cpu().detach()
    
    if arr.dim() == 3:  # Single image (C, H, W)
        if arr.shape[0] == 1:  # Grayscale
            arr = arr.squeeze(0).numpy()
        else:  # Color (CHW to HWC)
            arr = arr.permute(1, 2, 0).numpy()
    elif arr.dim() == 4:  # Batch (N, C, H, W)
        # Return first image in batch
        arr = arr[0].permute(1, 2, 0).numpy()
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {arr.dim()}D")
    
    # Denormalize if needed
    if denormalize:
        # Handle different normalization ranges
        if arr.min() < 0:  # [-1, 1] range
            arr = (arr + 1) / 2
        
        # Convert to uint8
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    
    return arr


def numpy_to_tensor(image: np.ndarray, 
                   normalize: bool = True,
                   device: Optional[torch.device] = None) -> Tensor:
    """
    Convert a NumPy array to a PyTorch tensor.
    
    Args:
        image: Input image as NumPy array (H, W, C) or (H, W)
        normalize: If True, converts to [0, 1] range
        device: Target device for the tensor (default: CPU)
        
    Returns:
        Tensor: Image as PyTorch tensor (C, H, W)
        
    Raises:
        TypeError: If input is not a NumPy array
        
    Example:
        >>> image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        >>> tensor = numpy_to_tensor(image)
        >>> print(tensor.shape, tensor.dtype)
        torch.Size([3, 256, 256]) torch.float32
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected NumPy array, got {type(image).__name__}")
    
    # Convert to float
    if normalize and image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif normalize:
        image = image.astype(np.float32)
    
    # Handle grayscale
    if len(image.shape) == 2:
        tensor = torch.from_numpy(image).unsqueeze(0)
    else:  # Color (HWC to CHW)
        tensor = torch.from_numpy(image).permute(2, 0, 1)
    
    # Move to device if specified
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


def validate_image(image: np.ndarray, 
                  min_size: Optional[Tuple[int, int]] = None,
                  max_size: Optional[Tuple[int, int]] = None) -> None:
    """
    Validate that an image meets required specifications.
    
    Args:
        image: Image to validate
        min_size: Minimum (height, width), optional
        max_size: Maximum (height, width), optional
        
    Raises:
        TypeError: If image is not a NumPy array
        ValueError: If image dimensions are invalid
        
    Example:
        >>> image = np.zeros((256, 256, 3), dtype=np.uint8)
        >>> validate_image(image, min_size=(64, 64))
        >>> validate_image(image, min_size=(512, 512))  # Raises ValueError
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Image must be a NumPy array, got {type(image).__name__}")
    
    if image.ndim not in [2, 3]:
        raise ValueError(f"Image must be 2D or 3D array, got {image.ndim}D")
    
    h, w = image.shape[:2]
    
    if min_size is not None:
        min_h, min_w = min_size
        if h < min_h or w < min_w:
            raise ValueError(
                f"Image size ({h}x{w}) is smaller than minimum required ({min_h}x{min_w})"
            )
    
    if max_size is not None:
        max_h, max_w = max_size
        if h > max_h or w > max_w:
            raise ValueError(
                f"Image size ({h}x{w}) exceeds maximum allowed ({max_h}x{max_w})"
            )
    
    # Validate color channels if 3D
    if image.ndim == 3:
        channels = image.shape[2]
        if channels not in [1, 3, 4]:
            raise ValueError(
                f"Image must have 1, 3, or 4 color channels, got {channels}"
            )


def match_dimensions(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Crop image to match target dimensions by removing excess from right/bottom.
    
    Args:
        image: Input image array
        target_shape: Desired (height, width)
        
    Returns:
        np.ndarray: Cropped image
        
    Raises:
        ValueError: If image is smaller than target shape
        
    Example:
        >>> image = np.ones((300, 400, 3), dtype=np.uint8)
        >>> cropped = match_dimensions(image, (256, 256))
        >>> print(cropped.shape)
        (256, 256, 3)
    """
    h, w = image.shape[:2]
    target_h, target_w = target_shape
    
    if h < target_h or w < target_w:
        raise ValueError(
            f"Image size ({h}x{w}) is smaller than target shape ({target_h}x{target_w})"
        )
    
    if (h, w) == target_shape:
        return image
    
    # Crop from top-left
    return image[:target_h, :target_w]


def validate_file_path(path: Union[str, Path], 
                       must_exist: bool = True,
                       check_extension: bool = True) -> Path:
    """
    Validate and normalize a file path.
    
    Args:
        path: File path to validate
        must_exist: If True, raises error if file doesn't exist
        check_extension: If True, validates file extension
        
    Returns:
        Path: Validated Path object
        
    Raises:
        FileNotFoundError: If must_exist=True and file doesn't exist
        ValueError: If check_extension=True and extension not supported
        
    Example:
        >>> path = validate_file_path("image.jpg")
        >>> print(type(path))
        <class 'pathlib.Path'>
    """
    path = Path(path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if check_extension:
        ext = path.suffix.lower()
        if ext not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported formats: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
            )
    
    return path


def compute_chunks_shape(image_shape: Tuple[int, int], 
                        grid_size: int) -> Tuple[int, int]:
    """
    Compute the dimensions of individual chunks given image size and grid.
    
    Args:
        image_shape: Image dimensions (height, width)
        grid_size: Number of chunks per dimension
        
    Returns:
        Tuple[int, int]: Chunk dimensions (chunk_height, chunk_width)
        
    Example:
        >>> compute_chunks_shape((512, 512), 16)
        (32, 32)
        >>> compute_chunks_shape((1024, 768), 32)
        (32, 24)
    """
    h, w = image_shape
    chunk_h = h // grid_size
    chunk_w = w // grid_size
    return chunk_h, chunk_w


def rearrange_for_ms_ssim(image: np.ndarray) -> Tensor:
    """
    Rearrange image for MS-SSIM computation (HWC -> NCHW format).
    
    Args:
        image: Input image in NumPy format (H, W, C)
        
    Returns:
        Tensor: Rearranged tensor (1, C, H, W) ready for MS-SSIM
        
    Example:
        >>> image = np.zeros((256, 256, 3), dtype=np.uint8)
        >>> tensor = rearrange_for_ms_ssim(image)
        >>> print(tensor.shape)
        torch.Size([1, 3, 256, 256])
    """
    tensor = torch.tensor(image, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor
