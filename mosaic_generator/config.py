"""
Configuration constants for the mosaic generator package.

This module contains all configurable parameters and constants used throughout
the mosaic generation process, including default values for grid sizes, tile
dimensions, blending parameters, and device settings.

Constants:
    DEFAULT_GRID_SIZE: Default number of chunks per dimension
    DEFAULT_TILE_SIZE: Standard tile dimensions (height, width)
    DEFAULT_BLEND_ALPHA: Default blending factor for tile superimposition
    SUPPORTED_IMAGE_FORMATS: List of supported image file extensions
    DEFAULT_TILES_DIR: Default directory for tile images
    DEFAULT_METADATA_FILE: Default CSV file for tile metadata
    DEFAULT_DEVICE: Default computation device (auto-detect GPU/CPU)
    USE_HALF_PRECISION: Whether to use FP16 on GPU for memory efficiency
    
Quality Settings:
    MIN_GRID_SIZE: Minimum allowed grid size
    MAX_GRID_SIZE: Maximum recommended grid size
    MIN_TILE_SIZE: Minimum tile dimensions
    MAX_TILE_SIZE: Maximum tile dimensions

Performance Settings:
    CACHE_TILES: Whether to cache tiles in memory
    BATCH_SIZE: Batch size for GPU operations
    NUM_WORKERS: Number of parallel workers for CPU operations

Author: Pratham Satani
Course: CS5130 - Applied Programming and Data Processing for AI
Institution: Northeastern University
"""

from typing import Tuple
import torch

# ========== Grid and Tile Settings ==========
DEFAULT_GRID_SIZE: int = 32
"""Default number of chunks per dimension (creates 32x32 grid)"""

DEFAULT_TILE_SIZE: Tuple[int, int] = (32, 32)
"""Standard tile dimensions in pixels (height, width)"""

DEFAULT_BLEND_ALPHA: float = 0.5
"""Default blending factor: 0.5 = equal blend of original and tile"""

# ========== Quality Constraints ==========
MIN_GRID_SIZE: int = 4
"""Minimum allowed grid size"""

MAX_GRID_SIZE: int = 128
"""Maximum recommended grid size"""

MIN_TILE_SIZE: Tuple[int, int] = (8, 8)
"""Minimum tile dimensions"""

MAX_TILE_SIZE: Tuple[int, int] = (256, 256)
"""Maximum tile dimensions"""

# ========== File and Directory Settings ==========
DEFAULT_TILES_DIR: str = "images"
"""Default directory containing tile images"""

DEFAULT_METADATA_FILE: str = "tiles_metadata.csv"
"""Default CSV file with tile metadata (colors, features)"""

SUPPORTED_IMAGE_FORMATS: Tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
)
"""Supported image file extensions"""

# ========== Device and Performance Settings ==========
DEFAULT_DEVICE: str = "auto"
"""
Default computation device:
    - 'auto': Automatically detect CUDA GPU or fallback to CPU
    - 'cuda': Force GPU computation
    - 'cpu': Force CPU computation
"""

USE_HALF_PRECISION: bool = True
"""Use FP16 (half precision) on GPU to save memory and increase speed"""

CACHE_TILES: bool = True
"""Whether to cache tiles in memory for faster access"""

BATCH_SIZE: int = 64
"""Batch size for GPU tensor operations"""

NUM_WORKERS: int = 4
"""Number of parallel workers for CPU-based operations"""

# ========== Color Matching Settings ==========
COLOR_SPACE: str = "RGB"
"""Color space for tile matching: 'RGB', 'LAB', or 'HSV'"""

DISTANCE_METRIC: str = "euclidean"
"""Distance metric for color matching: 'euclidean', 'manhattan', 'cosine'"""

DOMINANT_COLOR_MATCHING: bool = True
"""Use dominant color filtering before detailed matching"""

# ========== Blending Settings ==========
BLEND_MODE_NEAREST: float = 0.5
"""Blending alpha for 'nearest' tile matching mode"""

BLEND_MODE_RANDOM: float = 0.3
"""Blending alpha for 'random' tile selection mode"""

# ========== Logging Settings ==========
LOG_LEVEL: str = "INFO"
"""Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'"""

VERBOSE: bool = True
"""Enable verbose output during mosaic generation"""

# ========== Optimization Flags ==========
USE_VECTORIZED_OPS: bool = True
"""Use vectorized NumPy operations (always recommended)"""

USE_GPU_ACCELERATION: bool = True
"""Enable GPU acceleration if available"""

ENABLE_PROFILING: bool = False
"""Enable performance profiling (for debugging)"""

# ========== Metadata Column Names ==========
METADATA_COLUMNS = {
    "filename": "filename",
    "avg_red": "average-red",
    "avg_green": "average-green",
    "avg_blue": "average-blue",
    "dominant_color": "dominant-color",
}
"""Column names in the tile metadata CSV file"""

# ========== Tile Retrieval Modes ==========
TILE_RETRIEVAL_NEAREST: str = "nearest"
"""Nearest color matching mode"""

TILE_RETRIEVAL_RANDOM: str = "random"
"""Random tile selection mode"""

VALID_RETRIEVAL_MODES: Tuple[str, ...] = (
    TILE_RETRIEVAL_NEAREST,
    TILE_RETRIEVAL_RANDOM,
)
"""Valid tile retrieval modes"""


def get_device() -> torch.device:
    """
    Get the appropriate computation device based on configuration.
    
    Returns:
        torch.device: CUDA device if available and enabled, otherwise CPU
        
    Example:
        >>> device = get_device()
        >>> print(f"Using device: {device}")
        Using device: cuda:0
    """
    if DEFAULT_DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif DEFAULT_DEVICE == "cuda" and not torch.cuda.is_available():
        import warnings
        warnings.warn("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    else:
        return torch.device(DEFAULT_DEVICE)


def validate_grid_size(grid_size: int) -> None:
    """
    Validate that grid size is within acceptable bounds.
    
    Args:
        grid_size: Number of chunks per dimension
        
    Raises:
        ValueError: If grid_size is outside valid range
        
    Example:
        >>> validate_grid_size(32)  # Valid
        >>> validate_grid_size(200)  # Raises ValueError
    """
    if not isinstance(grid_size, int):
        raise TypeError(f"Grid size must be an integer, got {type(grid_size).__name__}")
    
    if grid_size < MIN_GRID_SIZE:
        raise ValueError(
            f"Grid size {grid_size} is too small. "
            f"Minimum allowed: {MIN_GRID_SIZE}"
        )
    
    if grid_size > MAX_GRID_SIZE:
        raise ValueError(
            f"Grid size {grid_size} is too large. "
            f"Maximum recommended: {MAX_GRID_SIZE}"
        )


def validate_tile_size(tile_size: Tuple[int, int]) -> None:
    """
    Validate that tile size is within acceptable bounds.
    
    Args:
        tile_size: Tile dimensions (height, width)
        
    Raises:
        ValueError: If tile_size is outside valid range
        TypeError: If tile_size is not a tuple
        
    Example:
        >>> validate_tile_size((32, 32))  # Valid
        >>> validate_tile_size((2, 2))  # Raises ValueError
    """
    if not isinstance(tile_size, tuple) or len(tile_size) != 2:
        raise TypeError("Tile size must be a tuple of (height, width)")
    
    h, w = tile_size
    
    if h < MIN_TILE_SIZE[0] or w < MIN_TILE_SIZE[1]:
        raise ValueError(
            f"Tile size {tile_size} is too small. "
            f"Minimum allowed: {MIN_TILE_SIZE}"
        )
    
    if h > MAX_TILE_SIZE[0] or w > MAX_TILE_SIZE[1]:
        raise ValueError(
            f"Tile size {tile_size} is too large. "
            f"Maximum allowed: {MAX_TILE_SIZE}"
        )


def validate_blend_alpha(alpha: float) -> None:
    """
    Validate that blend alpha is in valid range [0, 1].
    
    Args:
        alpha: Blending factor
        
    Raises:
        ValueError: If alpha is outside [0, 1]
        TypeError: If alpha is not numeric
        
    Example:
        >>> validate_blend_alpha(0.5)  # Valid
        >>> validate_blend_alpha(1.5)  # Raises ValueError
    """
    if not isinstance(alpha, (int, float)):
        raise TypeError(f"Blend alpha must be numeric, got {type(alpha).__name__}")
    
    if not 0 <= alpha <= 1:
        raise ValueError(f"Blend alpha must be in range [0, 1], got {alpha}")
