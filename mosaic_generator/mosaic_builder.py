"""
Mosaic builder module - main mosaic construction logic.

This module provides the MosaicBuilder class that orchestrates the mosaic
generation process using the TileManager and image processing utilities.

Classes:
    MosaicBuilder: Main class for creating image mosaics
    
Author: Pratham Satani
Course: CS5130 - Applied Programming and Data Processing for AI
Institution: Northeastern University
"""

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Union, Tuple, Optional
import time

from .tile_manager import TileManager
from .image_processor import create_image_grid_tensor, stitch_chunks
from .metrics import compute_ms_ssim
from .config import (
    DEFAULT_GRID_SIZE,
    DEFAULT_BLEND_ALPHA,
    TILE_RETRIEVAL_NEAREST,
    TILE_RETRIEVAL_RANDOM,
    VALID_RETRIEVAL_MODES,
    validate_grid_size,
    validate_blend_alpha,
)
from .utils import (
    validate_image,
    numpy_to_tensor,
    tensor_to_numpy,
    match_dimensions,
    rearrange_for_ms_ssim,
    logger,
)


class MosaicBuilder:
    """
    Main class for creating image mosaics using tile matching.
    
    The MosaicBuilder coordinates the mosaic generation process:
    1. Divides input image into grid of chunks
    2. Computes features for each chunk (average color)
    3. Matches chunks to tiles using TileManager
    4. Blends tiles with original chunks
    5. Reconstructs final mosaic image
    
    Supports both CPU and GPU acceleration through PyTorch.
    
    Attributes:
        tile_manager: TileManager instance for tile operations
        grid_size: Default grid size (n×n)
        device: Computation device
        
    Example:
        >>> from mosaic_generator import TileManager, MosaicBuilder
        >>> 
        >>> # Initialize
        >>> tile_manager = TileManager(tile_directory='images/')
        >>> builder = MosaicBuilder(tile_manager, grid_size=(32, 32))
        >>> 
        >>> # Generate mosaic
        >>> image = load_image("input.jpg")
        >>> mosaic = builder.create_mosaic(image)
        >>> 
        >>> # Compute similarity
        >>> similarity = builder.compute_similarity(image, mosaic)
        >>> print(f"MS-SSIM: {similarity:.4f}")
    """
    
    def __init__(self,
                 tile_manager: TileManager,
                 grid_size: Union[int, Tuple[int, int]] = DEFAULT_GRID_SIZE,
                 blend_alpha: float = DEFAULT_BLEND_ALPHA):
        """
        Initialize MosaicBuilder.
        
        Args:
            tile_manager: TileManager instance with loaded tiles
            grid_size: Default grid size as int (n×n) or tuple (rows, cols)
            blend_alpha: Default blending factor [0, 1]
            
        Raises:
            ValueError: If grid_size or blend_alpha is invalid
            TypeError: If tile_manager is not a TileManager instance
        """
        if not isinstance(tile_manager, TileManager):
            raise TypeError(
                f"tile_manager must be a TileManager instance, "
                f"got {type(tile_manager).__name__}"
            )
        
        self.tile_manager = tile_manager
        
        # Handle grid size
        if isinstance(grid_size, int):
            validate_grid_size(grid_size)
            self.grid_size = (grid_size, grid_size)
        elif isinstance(grid_size, tuple) and len(grid_size) == 2:
            validate_grid_size(grid_size[0])
            validate_grid_size(grid_size[1])
            self.grid_size = grid_size
        else:
            raise ValueError(
                "grid_size must be an integer or tuple of (rows, cols)"
            )
        
        validate_blend_alpha(blend_alpha)
        self.blend_alpha = blend_alpha
        
        # Use same device as tile manager
        self.device = tile_manager.device
        
        logger.info(
            f"Initialized MosaicBuilder with {self.grid_size[0]}×{self.grid_size[1]} "
            f"grid on {self.device}"
        )
    
    def create_mosaic(self,
                     image: Union[np.ndarray, Tensor],
                     grid_size: Optional[int] = None,
                     tile_retrieval: str = TILE_RETRIEVAL_NEAREST,
                     blend_alpha: Optional[float] = None,
                     return_tensor: bool = False) -> Union[np.ndarray, Tensor]:
        """
        Create a mosaic from an input image.
        
        This is the main method that orchestrates the entire mosaic
        generation process using GPU-accelerated operations.
        
        Args:
            image: Input image as NumPy array (H, W, C) or Tensor
            grid_size: Grid size (overrides default). Creates grid_size×grid_size grid
            tile_retrieval: 'nearest' for color matching or 'random' for random tiles
            blend_alpha: Blending factor (overrides default)
            return_tensor: If True, returns PyTorch tensor instead of NumPy array
            
        Returns:
            Union[np.ndarray, Tensor]: Generated mosaic
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If mosaic generation fails
            
        Example:
            >>> builder = MosaicBuilder(tile_manager)
            >>> image = load_image("photo.jpg")
            >>> mosaic = builder.create_mosaic(image, grid_size=32)
            >>> 
            >>> # Random mosaic
            >>> random_mosaic = builder.create_mosaic(
            ...     image, 
            ...     grid_size=32,
            ...     tile_retrieval="random"
            ... )
        """
        # Validate inputs
        if isinstance(image, np.ndarray):
            validate_image(image)
        
        if tile_retrieval not in VALID_RETRIEVAL_MODES:
            raise ValueError(
                f"tile_retrieval must be one of {VALID_RETRIEVAL_MODES}, "
                f"got '{tile_retrieval}'"
            )
        
        # Use defaults if not specified
        n = grid_size if grid_size is not None else self.grid_size[0]
        alpha = blend_alpha if blend_alpha is not None else self.blend_alpha
        
        validate_grid_size(n)
        validate_blend_alpha(alpha)
        
        logger.info(
            f"Creating {n}×{n} mosaic using '{tile_retrieval}' retrieval "
            f"on {self.device}"
        )
        
        start_time = time.time()
        
        try:
            # Convert to tensor
            img_tensor = self._prepare_image(image)
            
            # Create chunks (all on device)
            chunks = create_image_grid_tensor(img_tensor, n, self.device)
            
            if tile_retrieval == TILE_RETRIEVAL_NEAREST:
                # Compute features
                chunk_colors = self._compute_chunk_features(chunks)
                
                # Find best matching tiles (GPU-accelerated)
                tile_indices = self.tile_manager.find_nearest_tiles_batch(chunk_colors)
                
                # Retrieve tiles
                chunk_h, chunk_w = chunks.shape[2:]
                tiles = self.tile_manager.get_tiles_batch(tile_indices, (chunk_h, chunk_w))
                
                # Blend
                result = self._blend_chunks_tiles(chunks, tiles, alpha)
            else:
                # Random selection
                n_chunks_total = len(chunks)
                chunk_h, chunk_w = chunks.shape[2:]
                tiles = self.tile_manager.get_random_tiles(n_chunks_total, (chunk_h, chunk_w))
                result = self._blend_chunks_tiles(chunks, tiles, alpha) 
            
            # Reconstruct image
            mosaic_tensor = self._stitch_chunks_tensor(result, n)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Mosaic created in {elapsed:.3f}s on {self.device}")
            
            if return_tensor:
                return mosaic_tensor
            else:
                return tensor_to_numpy(mosaic_tensor)
                
        except Exception as e:
            logger.error(f"Mosaic generation failed: {str(e)}")
            raise RuntimeError(f"Failed to create mosaic: {str(e)}")
    
    def _prepare_image(self, image: Union[np.ndarray, Tensor]) -> Tensor:
        """Convert image to tensor on device."""
        if isinstance(image, np.ndarray):
            return numpy_to_tensor(image, normalize=True, device=self.device)
        else:
            return image.to(self.device)
    
    def _compute_chunk_features(self, chunks: Tensor) -> Tensor:
        """
        Compute average colors for chunks.
        
        Args:
            chunks: Tensor of chunks (n*n, C, H, W)
            
        Returns:
            Average colors (n*n, C) scaled to [0, 255]
        """
        # Global average pooling
        avg_colors = chunks.mean(dim=(2, 3))
        
        # Scale to 0-255 range for matching with tile metadata
        avg_colors = avg_colors * 255
        
        return avg_colors
    
    def _blend_chunks_tiles(self, chunks: Tensor, tiles: Tensor, alpha: float) -> Tensor:
        """
        Blend chunks and tiles using weighted average.
        
        Args:
            chunks: Original chunks (n, C, H, W)
            tiles: Matched tiles (n, C, H, W)
            alpha: Blend factor [0, 1]
            
        Returns:
            Blended result
        """
        return alpha * chunks + (1 - alpha) * tiles
    
    def _stitch_chunks_tensor(self, chunks: Tensor, n: int) -> Tensor:
        """
        Reconstruct image from chunks using tensor operations.
        
        Args:
            chunks: Tensor of chunks (n*n, C, H, W)
            n: Grid size
            
        Returns:
            Reconstructed image (C, H, W)
        """
        batch_size, C, chunk_h, chunk_w = chunks.shape
        
        # Reshape to grid
        chunks_grid = chunks.view(n, n, C, chunk_h, chunk_w)
        
        # Transpose and reshape to reconstruct
        chunks_grid = chunks_grid.permute(2, 0, 3, 1, 4)  # (C, n, chunk_h, n, chunk_w)
        reconstructed = chunks_grid.contiguous().view(C, n * chunk_h, n * chunk_w)
        
        return reconstructed
    
    def compute_similarity(self,
                          original: Union[np.ndarray, Tensor],
                          mosaic: Union[np.ndarray, Tensor],
                          metric: str = "ms_ssim") -> float:
        """
        Compute similarity between original image and mosaic.
        
        Args:
            original: Original image
            mosaic: Generated mosaic
            metric: Similarity metric ('ms_ssim', 'ssim', 'mse', 'psnr')
            
        Returns:
            float: Similarity score
            
        Example:
            >>> similarity = builder.compute_similarity(image, mosaic)
            >>> print(f"Similarity: {similarity:.4f}")
        """
        # Convert to numpy if needed
        if isinstance(original, Tensor):
            original = tensor_to_numpy(original)
        if isinstance(mosaic, Tensor):
            mosaic = tensor_to_numpy(mosaic)
        
        # Match dimensions
        if original.shape != mosaic.shape:
            target_shape = mosaic.shape[:2]
            original = match_dimensions(original, target_shape)
        
        if metric == "ms_ssim":
            return compute_ms_ssim(original, mosaic)
        elif metric == "mse":
            from .metrics import compute_mse
            return compute_mse(original, mosaic)
        elif metric == "ssim":
            from .metrics import compute_ssim
            return compute_ssim(original, mosaic)
        elif metric == "psnr":
            from .metrics import compute_psnr
            return compute_psnr(original, mosaic)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def get_performance_stats(self) -> dict:
        """
        Get performance statistics.
        
        Returns:
            dict: Performance metrics and device info
        """
        stats = {
            'device': str(self.device),
            'device_type': self.device.type,
            'grid_size': self.grid_size,
            'blend_alpha': self.blend_alpha,
            'num_tiles': len(self.tile_manager),
        }
        
        if self.device.type == 'cuda':
            stats.update({
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
            })
        
        return stats
