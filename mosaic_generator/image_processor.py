"""
Image processing module for the mosaic generator package.

This module provides functions for image loading, preprocessing, resizing,
and grid creation. It handles various image formats and ensures proper
formatting for mosaic generation.

Functions:
    load_image: Load image from file path
    resize_image: Resize image to target dimensions
    create_image_grid: Divide image into grid of chunks
    preprocess_image: Apply preprocessing pipeline
    
Author: Pratham Satani
Course: CS5130 - Applied Programming and Data Processing for AI
Institution: Northeastern University
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional, List
import torch
from torch import Tensor

from .config import DEFAULT_GRID_SIZE, validate_grid_size
from .utils import (
    validate_image, 
    validate_file_path, 
    compute_chunks_shape,
    logger
)


def load_image(path: Union[str, Path], 
               color_mode: str = "RGB",
               resize_to: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load an image from a file path.
    
    Supports multiple image formats (JPG, PNG, BMP, TIFF, WebP) and handles
    color space conversions automatically.
    
    Args:
        path: Path to the image file
        color_mode: Color mode ('RGB', 'BGR', 'GRAY'). Defaults to 'RGB'
        resize_to: Optional (width, height) to resize the loaded image
        
    Returns:
        np.ndarray: Loaded image as NumPy array
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded or has invalid format
        
    Example:
        >>> image = load_image("photo.jpg")
        >>> print(image.shape, image.dtype)
        (1024, 768, 3) uint8
        
        >>> image = load_image("photo.jpg", resize_to=(512, 512))
        >>> print(image.shape)
        (512, 512, 3)
    """
    # Validate path
    path = validate_file_path(path, must_exist=True, check_extension=True)
    
    try:
        # Load using PIL for better format support
        pil_image = Image.open(path)
        
        # Convert to desired color mode
        if color_mode == "RGB":
            pil_image = pil_image.convert("RGB")
        elif color_mode == "GRAY":
            pil_image = pil_image.convert("L")
        elif color_mode == "BGR":
            pil_image = pil_image.convert("RGB")
        
        # Resize if needed
        if resize_to is not None:
            pil_image = pil_image.resize(resize_to, Image.Resampling.LANCZOS)
        
        # Convert to NumPy array
        image = np.array(pil_image)
        
        # Convert RGB to BGR if requested
        if color_mode == "BGR" and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        logger.debug(f"Loaded image from {path} with shape {image.shape}")
        return image
        
    except Exception as e:
        raise ValueError(f"Failed to load image from {path}: {str(e)}")


def resize_image(image: np.ndarray, 
                size: Tuple[int, int],
                interpolation: str = "lanczos") -> np.ndarray:
    """
    Resize an image to target dimensions.
    
    Args:
        image: Input image as NumPy array
        size: Target size as (width, height)
        interpolation: Interpolation method ('lanczos', 'bilinear', 'nearest', 'cubic')
        
    Returns:
        np.ndarray: Resized image
        
    Raises:
        ValueError: If interpolation method is invalid
        
    Example:
        >>> image = np.zeros((1024, 768, 3), dtype=np.uint8)
        >>> resized = resize_image(image, (512, 512))
        >>> print(resized.shape)
        (512, 512, 3)
    """
    validate_image(image)
    
    # Map interpolation names to OpenCV constants
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "area": cv2.INTER_AREA,
    }
    
    if interpolation.lower() not in interp_map:
        raise ValueError(
            f"Invalid interpolation method: {interpolation}. "
            f"Must be one of {list(interp_map.keys())}"
        )
    
    interp_flag = interp_map[interpolation.lower()]
    
    # OpenCV expects (width, height)
    resized = cv2.resize(image, size, interpolation=interp_flag)
    
    logger.debug(f"Resized image from {image.shape[:2]} to {resized.shape[:2]}")
    return resized


def create_image_grid(image: np.ndarray, 
                     grid_size: int = DEFAULT_GRID_SIZE,
                     return_format: str = "array") -> Union[np.ndarray, List[List[np.ndarray]]]:
    """
    Divide an image into a grid of equal-sized chunks.
    
    The image is divided into grid_size × grid_size chunks. If the image
    dimensions are not perfectly divisible, the excess pixels on the right
    and bottom are truncated.
    
    Args:
        image: Input image as NumPy array (H, W, C) or (H, W)
        grid_size: Number of chunks per dimension (creates grid_size × grid_size grid)
        return_format: Format of returned chunks ('array' or 'list')
            - 'array': Returns 4D/5D NumPy array (n, n, chunk_h, chunk_w[, channels])
            - 'list': Returns 2D list of chunk arrays
        
    Returns:
        Union[np.ndarray, List]: Grid of image chunks
        
    Raises:
        ValueError: If grid_size is invalid or image is too small
        
    Example:
        >>> image = np.zeros((512, 512, 3), dtype=np.uint8)
        >>> chunks = create_image_grid(image, grid_size=16)
        >>> print(chunks.shape)
        (16, 16, 32, 32, 3)
        
        >>> chunks_list = create_image_grid(image, grid_size=16, return_format='list')
        >>> print(len(chunks_list), len(chunks_list[0]))
        16 16
    """
    validate_image(image)
    validate_grid_size(grid_size)
    
    h, w = image.shape[:2]
    
    # Validate image is large enough
    if h < grid_size or w < grid_size:
        raise ValueError(
            f"Image size ({h}×{w}) is too small for {grid_size}×{grid_size} grid. "
            f"Minimum image size: {grid_size}×{grid_size}"
        )
    
    # Calculate chunk dimensions
    chunk_h, chunk_w = compute_chunks_shape((h, w), grid_size)
    
    if chunk_h == 0 or chunk_w == 0:
        raise ValueError(
            f"Image too small to create {grid_size}×{grid_size} chunks. "
            f"Each chunk would be {chunk_h}×{chunk_w}"
        )
    
    # Truncate to exact multiple of chunk size
    truncated = image[:chunk_h * grid_size, :chunk_w * grid_size]
    
    # Reshape into grid
    if len(image.shape) == 3:  # Color image
        channels = image.shape[2]
        # Reshape: (n*chunk_h, n*chunk_w, c) -> (n, chunk_h, n, chunk_w, c)
        reshaped = truncated.reshape(grid_size, chunk_h, grid_size, chunk_w, channels)
        # Transpose: -> (n, n, chunk_h, chunk_w, c)
        chunks = reshaped.transpose(0, 2, 1, 3, 4)
    else:  # Grayscale image
        # Reshape: (n*chunk_h, n*chunk_w) -> (n, chunk_h, n, chunk_w)
        reshaped = truncated.reshape(grid_size, chunk_h, grid_size, chunk_w)
        # Transpose: -> (n, n, chunk_h, chunk_w)
        chunks = reshaped.transpose(0, 2, 1, 3)
    
    logger.debug(
        f"Created {grid_size}×{grid_size} grid from image of shape {image.shape}. "
        f"Chunk size: {chunk_h}×{chunk_w}"
    )
    
    # Convert to list format if requested
    if return_format == "list":
        chunks_list = []
        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                row.append(chunks[i, j])
            chunks_list.append(row)
        return chunks_list
    
    return chunks


def create_image_grid_tensor(image: Union[np.ndarray, Tensor],
                            grid_size: int,
                            device: Optional[torch.device] = None) -> Tensor:
    """
    Create image grid using PyTorch tensors for GPU acceleration.
    
    This version uses PyTorch's unfold operation for efficient chunking
    on GPU if available.
    
    Args:
        image: Input image as NumPy array or Tensor
        grid_size: Number of chunks per dimension
        device: Target device (if None, uses CPU)
        
    Returns:
        Tensor: Chunks as tensor (n*n, C, chunk_h, chunk_w)
        
    Example:
        >>> image = np.zeros((512, 512, 3), dtype=np.uint8)
        >>> chunks = create_image_grid_tensor(image, 16)
        >>> print(chunks.shape)
        torch.Size([256, 3, 32, 32])
    """
    validate_grid_size(grid_size)
    
    # Convert to tensor if needed
    if isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        if len(image.shape) == 2:
            img_tensor = torch.from_numpy(image).unsqueeze(0)
        else:
            img_tensor = torch.from_numpy(image).permute(2, 0, 1)
    else:
        img_tensor = image
    
    # Move to device
    if device is not None:
        img_tensor = img_tensor.to(device)
    
    C, H, W = img_tensor.shape
    chunk_h, chunk_w = H // grid_size, W // grid_size
    
    # Truncate to exact multiple
    img_truncated = img_tensor[:, :chunk_h * grid_size, :chunk_w * grid_size]
    
    # Use unfold for efficient chunking
    chunks = img_truncated.unfold(1, chunk_h, chunk_h).unfold(2, chunk_w, chunk_w)
    
    # Reshape to (grid_size*grid_size, C, chunk_h, chunk_w)
    chunks = chunks.permute(1, 2, 0, 3, 4).contiguous()
    chunks = chunks.view(grid_size * grid_size, C, chunk_h, chunk_w)
    
    logger.debug(f"Created tensor grid with shape {chunks.shape}")
    return chunks


def preprocess_image(image: np.ndarray,
                    target_size: Optional[Tuple[int, int]] = None,
                    normalize: bool = False,
                    ensure_divisible: Optional[int] = None) -> np.ndarray:
    """
    Apply preprocessing pipeline to an image.
    
    Args:
        image: Input image
        target_size: Resize to (width, height) if provided
        normalize: Normalize to [0, 1] range if True
        ensure_divisible: Ensure dimensions are divisible by this value (crops if needed)
        
    Returns:
        np.ndarray: Preprocessed image
        
    Example:
        >>> image = load_image("photo.jpg")
        >>> processed = preprocess_image(image, ensure_divisible=32)
        >>> print(processed.shape[0] % 32, processed.shape[1] % 32)
        0 0
    """
    validate_image(image)
    processed = image.copy()
    
    # Resize if target size specified
    if target_size is not None:
        processed = resize_image(processed, target_size)
    
    # Ensure dimensions are divisible by specified value
    if ensure_divisible is not None:
        h, w = processed.shape[:2]
        new_h = (h // ensure_divisible) * ensure_divisible
        new_w = (w // ensure_divisible) * ensure_divisible
        
        if new_h != h or new_w != w:
            processed = processed[:new_h, :new_w]
            logger.debug(f"Cropped image to {new_h}×{new_w} for divisibility by {ensure_divisible}")
    
    # Normalize if requested
    if normalize and processed.dtype == np.uint8:
        processed = processed.astype(np.float32) / 255.0
    
    return processed


def stitch_chunks(chunks: Union[np.ndarray, List[List[np.ndarray]]]) -> np.ndarray:
    """
    Reconstruct an image from a grid of chunks.
    
    This is the inverse operation of create_image_grid.
    
    Args:
        chunks: Grid of chunks as 4D/5D array or 2D list
        
    Returns:
        np.ndarray: Reconstructed image
        
    Raises:
        ValueError: If chunks have inconsistent dimensions
        
    Example:
        >>> chunks = create_image_grid(image, 16)
        >>> reconstructed = stitch_chunks(chunks)
        >>> print(reconstructed.shape)
        (512, 512, 3)
    """
    # Convert list to array if needed
    if isinstance(chunks, list):
        # Validate list structure
        if not chunks or not chunks[0]:
            raise ValueError("Chunks list cannot be empty")
        
        # Stitch using numpy operations
        rows = []
        for row_chunks in chunks:
            row_image = np.hstack(row_chunks)
            rows.append(row_image)
        
        stitched = np.vstack(rows)
        
    elif isinstance(chunks, np.ndarray):
        if chunks.ndim not in [4, 5]:
            raise ValueError(f"Expected 4D or 5D chunks array, got {chunks.ndim}D")
        
        grid_size = chunks.shape[0]
        
        if len(chunks.shape) == 5:  # Color image
            chunk_h, chunk_w, channels = chunks.shape[2:5]
            # Transpose and reshape back
            stitched = chunks.transpose(0, 2, 1, 3, 4).reshape(
                grid_size * chunk_h, grid_size * chunk_w, channels
            )
        else:  # Grayscale
            chunk_h, chunk_w = chunks.shape[2:4]
            stitched = chunks.transpose(0, 2, 1, 3).reshape(
                grid_size * chunk_h, grid_size * chunk_w
            )
    else:
        raise TypeError(f"Chunks must be array or list, got {type(chunks).__name__}")
    
    logger.debug(f"Stitched chunks into image of shape {stitched.shape}")
    return stitched
