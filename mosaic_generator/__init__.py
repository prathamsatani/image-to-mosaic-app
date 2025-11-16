"""
Mosaic Generator Package

A modular, high-performance Python package for creating image mosaics using tile-matching algorithms.
This package provides GPU-accelerated operations through PyTorch and optimized vectorized operations
through NumPy.

Main Components:
    - TileManager: Manages tile loading, caching, and feature extraction
    - MosaicBuilder: Main mosaic construction with various matching algorithms
    - Image processing utilities for loading and grid creation
    - Similarity metrics (MSE, SSIM) for quality evaluation

Example:
    >>> from mosaic_generator import TileManager, MosaicBuilder
    >>> 
    >>> # Initialize components
    >>> tile_manager = TileManager(tile_directory='images/')
    >>> builder = MosaicBuilder(tile_manager, grid_size=(32, 32))
    >>> 
    >>> # Generate mosaic
    >>> import cv2
    >>> image = cv2.imread('input.jpg')
    >>> mosaic = builder.create_mosaic(image)
    >>> similarity = builder.compute_similarity(image, mosaic)
    >>> print(f"Similarity: {similarity:.4f}")

Author: Pratham Satani
Course: CS5130 - Applied Programming and Data Processing for AI
Institution: Northeastern University
"""

from .tile_manager import TileManager
from .mosaic_builder import MosaicBuilder
from .image_processor import load_image, resize_image, create_image_grid
from .metrics import compute_mse, compute_ssim, compute_ms_ssim
from .config import (
    DEFAULT_GRID_SIZE,
    DEFAULT_TILE_SIZE,
    DEFAULT_BLEND_ALPHA,
    SUPPORTED_IMAGE_FORMATS,
    DEFAULT_DEVICE
)

__version__ = "2.0.0"
__author__ = "Pratham Satani"
__email__ = "satani.p@northeastern.edu"

__all__ = [
    # Main classes
    "TileManager",
    "MosaicBuilder",
    
    # Image processing functions
    "load_image",
    "resize_image",
    "create_image_grid",
    
    # Metrics
    "compute_mse",
    "compute_ssim",
    "compute_ms_ssim",
    
    # Configuration constants
    "DEFAULT_GRID_SIZE",
    "DEFAULT_TILE_SIZE",
    "DEFAULT_BLEND_ALPHA",
    "SUPPORTED_IMAGE_FORMATS",
    "DEFAULT_DEVICE",
]
