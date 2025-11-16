"""
Tile management module for the mosaic generator package.

This module provides the TileManager class for loading, caching, and managing
tile images. It handles tile feature extraction, color analysis, and efficient
retrieval for mosaic generation.

Classes:
    TileCacheTensor: Dataclass for cached tile data
    TileManager: Main class for tile management and retrieval
    
Author: Pratham Satani
Course: CS5130 - Applied Programming and Data Processing for AI
Institution: Northeastern University
"""

import numpy as np
import pandas as pd
import cv2
import torch
from torch import Tensor
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import os

from .config import (
    DEFAULT_TILES_DIR,
    DEFAULT_METADATA_FILE,
    DEFAULT_TILE_SIZE,
    METADATA_COLUMNS,
    get_device,
    USE_HALF_PRECISION,
)
from .utils import rgb_to_text, validate_file_path, logger


@dataclass
class TileCacheTensor:
    """
    Cached tile data stored as PyTorch tensors for GPU operations.
    
    Attributes:
        image: Tile image tensor (C, H, W)
        avg_color: Average RGB color tensor (3,)
        dominant_color: Dominant color name as string
    """
    image: Tensor
    avg_color: Tensor
    dominant_color: str


class TileManager:
    """
    Manages tile images for mosaic generation.
    
    This class handles:
    - Loading tiles from directory
    - Caching tiles in memory (CPU or GPU)
    - Computing tile features (average color, dominant color)
    - Efficient tile retrieval and matching
    
    The TileManager supports both CPU and GPU acceleration, automatically
    caching tiles on the appropriate device for fast access during mosaic
    generation.
    
    Attributes:
        tiles_dir: Directory containing tile images
        metadata_file: Path to CSV with tile metadata
        tile_size: Standard size for cached tiles (H, W)
        device: Computation device (cuda/cpu)
        tile_cache: Dictionary mapping tile indices to cached data
        tiles_metadata: DataFrame with tile metadata
        
    Example:
        >>> manager = TileManager(tile_directory='images/')
        >>> print(f"Loaded {len(manager)} tiles")
        Loaded 1000 tiles
        
        >>> # Find best matching tile for a color
        >>> tile_idx = manager.find_nearest_tile([255, 0, 0])
        >>> tile_image = manager.get_tile(tile_idx)
    """
    
    def __init__(self,
                 tile_directory: str = DEFAULT_TILES_DIR,
                 metadata_file: str = DEFAULT_METADATA_FILE,
                 tile_size: Tuple[int, int] = DEFAULT_TILE_SIZE,
                 device: Optional[str] = None,
                 use_half_precision: bool = USE_HALF_PRECISION):
        """
        Initialize TileManager with tiles from directory.
        
        Args:
            tile_directory: Directory containing tile images
            metadata_file: CSV file with tile metadata
            tile_size: Standard size for cached tiles (H, W)
            device: Force specific device ('cuda', 'cpu') or None for auto
            use_half_precision: Use float16 on GPU for memory savings
            
        Raises:
            FileNotFoundError: If tile directory or metadata file not found
            ValueError: If no valid tiles found
        """
        self.tiles_dir = tile_directory
        self.metadata_file = metadata_file
        self.tile_size = tile_size
        
        # Setup device
        if device:
            self.device = torch.device(device)
        else:
            self.device = get_device()
        
        # Use half precision on GPU for memory efficiency
        self.dtype = torch.float16 if (use_half_precision and self.device.type == 'cuda') else torch.float32
        
        logger.info(f"🚀 Initializing TileManager on {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"   GPU: {torch.cuda.get_device_name()}")
            logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Validate paths
        if not os.path.exists(tile_directory):
            raise FileNotFoundError(f"Tile directory not found: {tile_directory}")
        
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # Load metadata
        logger.info("Loading tile metadata...")
        self.tiles_metadata = pd.read_csv(metadata_file)
        
        # Initialize cache
        self.tile_cache: Dict[int, TileCacheTensor] = {}
        
        # Load and cache all tiles
        logger.info("Loading and caching tiles to device memory...")
        self._cache_all_tiles()
        
        # Build tensor indices for fast matching
        logger.info("Building tensor indices for fast matching...")
        self._build_tensor_indices()
        
        logger.info(f"✅ Initialized with {len(self.tile_cache)} tiles on {self.device}")
        
        # Print memory usage if GPU
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"   GPU Memory Used: {allocated:.2f} GB")
    
    def _cache_all_tiles(self) -> None:
        """
        Load all tiles as PyTorch tensors on the target device.
        
        Raises:
            ValueError: If no valid tiles are loaded
        """
        tile_colors_list = []
        
        for idx, row in self.tiles_metadata.iterrows():
            filename = row[METADATA_COLUMNS['filename']]
            filepath = os.path.join(self.tiles_dir, filename)
            
            try:
                # Load image using PIL for consistency
                pil_image = Image.open(filepath).convert('RGB')
                
                # Resize to standard size
                pil_image = pil_image.resize(
                    self.tile_size[::-1],  # (width, height)
                    Image.Resampling.LANCZOS
                )
                
                # Convert to tensor and normalize to [0, 1]
                img_array = np.array(pil_image)
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                
                # Move to device
                img_tensor = img_tensor.to(self.device, dtype=self.dtype)
                
                # Get average color from metadata
                avg_color = torch.tensor([
                    row[METADATA_COLUMNS['avg_red']],
                    row[METADATA_COLUMNS['avg_green']],
                    row[METADATA_COLUMNS['avg_blue']]
                ], dtype=self.dtype, device=self.device)
                
                # Cache the tile
                self.tile_cache[idx] = TileCacheTensor(
                    image=img_tensor,
                    avg_color=avg_color,
                    dominant_color=row.get(METADATA_COLUMNS['dominant_color'], 'Unknown')
                )
                
                tile_colors_list.append(avg_color)
                
            except Exception as e:
                logger.warning(f"Failed to load tile {filename}: {e}")
        
        if not tile_colors_list:
            raise ValueError(f"No valid tiles found in {self.tiles_dir}")
        
        # Stack all colors into a single tensor for vectorized operations
        self.tile_colors_tensor = torch.stack(tile_colors_list)
        self.tile_indices = torch.tensor(
            list(self.tile_cache.keys()),
            device=self.device,
            dtype=torch.long
        )
    
    def _build_tensor_indices(self) -> None:
        """Build tensors for efficient batch operations."""
        if len(self.tile_colors_tensor) > 0:
            # Pre-compute normalized colors for faster matching
            self.tile_colors_norm = F.normalize(self.tile_colors_tensor, p=2, dim=1)
        else:
            self.tile_colors_norm = self.tile_colors_tensor
    
    def find_nearest_tile(self, color: Union[List[float], np.ndarray, Tensor]) -> int:
        """
        Find the tile with the nearest average color.
        
        Uses Euclidean distance in RGB space to find the best match.
        
        Args:
            color: RGB color values (3,) in range [0, 255]
            
        Returns:
            int: Index of the best matching tile
            
        Example:
            >>> manager = TileManager()
            >>> tile_idx = manager.find_nearest_tile([255, 0, 0])  # Red
            >>> print(f"Best match: tile {tile_idx}")
        """
        # Convert to tensor if needed
        if isinstance(color, (list, np.ndarray)):
            color_tensor = torch.tensor(color, dtype=self.dtype, device=self.device)
        else:
            color_tensor = color.to(self.device, dtype=self.dtype)
        
        # Compute distances
        distances = torch.norm(
            self.tile_colors_tensor - color_tensor.unsqueeze(0),
            dim=1
        )
        
        # Find minimum
        min_idx = distances.argmin()
        return self.tile_indices[min_idx].item()
    
    def find_nearest_tiles_batch(self, colors: Tensor) -> Tensor:
        """
        Find nearest tiles for a batch of colors (GPU-optimized).
        
        Args:
            colors: Batch of RGB colors (N, 3) in range [0, 255]
            
        Returns:
            Tensor: Indices of best matching tiles (N,)
            
        Example:
            >>> manager = TileManager()
            >>> colors = torch.rand(100, 3) * 255
            >>> tile_indices = manager.find_nearest_tiles_batch(colors)
            >>> print(tile_indices.shape)
            torch.Size([100])
        """
        if len(self.tile_colors_tensor) == 0:
            # Random selection if no tiles
            return torch.randint(
                0, max(1, len(self.tile_cache)),
                (len(colors),),
                device=self.device
            )
        
        # Ensure colors are on device
        colors = colors.to(self.device, dtype=self.dtype)
        
        # Use cdist for efficient batched distance computation
        distances = torch.cdist(
            colors.unsqueeze(0),
            self.tile_colors_tensor.unsqueeze(0),
            p=2  # Euclidean distance
        ).squeeze(0)
        
        # Find minimum distance tiles
        min_indices = distances.argmin(dim=1)
        
        # Map to actual tile indices
        matched_tile_ids = self.tile_indices[min_indices]
        
        return matched_tile_ids
    
    def get_tile(self, idx: int, target_size: Optional[Tuple[int, int]] = None) -> Tensor:
        """
        Retrieve a cached tile by index.
        
        Args:
            idx: Tile index
            target_size: Optional (H, W) to resize tile
            
        Returns:
            Tensor: Tile image (C, H, W)
            
        Raises:
            KeyError: If tile index not found
            
        Example:
            >>> manager = TileManager()
            >>> tile = manager.get_tile(0, target_size=(64, 64))
            >>> print(tile.shape)
            torch.Size([3, 64, 64])
        """
        if idx not in self.tile_cache:
            raise KeyError(f"Tile index {idx} not found in cache")
        
        tile = self.tile_cache[idx].image
        
        # Resize if needed
        if target_size is not None and tile.shape[-2:] != target_size:
            tile = F.interpolate(
                tile.unsqueeze(0),
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        return tile
    
    def get_tiles_batch(self, indices: Tensor, target_size: Tuple[int, int]) -> Tensor:
        """
        Retrieve multiple tiles efficiently in a batch.
        
        Args:
            indices: Tile indices (N,)
            target_size: Target size (H, W)
            
        Returns:
            Tensor: Batch of tiles (N, C, H, W)
            
        Example:
            >>> manager = TileManager()
            >>> indices = torch.tensor([0, 1, 2, 3])
            >>> tiles = manager.get_tiles_batch(indices, (32, 32))
            >>> print(tiles.shape)
            torch.Size([4, 3, 32, 32])
        """
        # Stack all tiles at once - no loop!
        tiles_list = [self.tile_cache[idx.item()].image for idx in indices]
        all_tiles = torch.stack(tiles_list)
        
        # Single batched resize operation if needed
        if all_tiles.shape[-2:] != target_size:
            all_tiles = F.interpolate(
                all_tiles,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
        
        return all_tiles
    
    def get_random_tiles(self, count: int, target_size: Tuple[int, int]) -> Tensor:
        """
        Get random tiles for random mosaic mode.
        
        Args:
            count: Number of random tiles to retrieve
            target_size: Target size (H, W)
            
        Returns:
            Tensor: Batch of random tiles (count, C, H, W)
            
        Example:
            >>> manager = TileManager()
            >>> tiles = manager.get_random_tiles(16, (32, 32))
            >>> print(tiles.shape)
            torch.Size([16, 3, 32, 32])
        """
        random_indices = torch.randint(
            0, len(self.tile_cache),
            (count,),
            device=self.device
        )
        
        return self.get_tiles_batch(
            self.tile_indices[random_indices],
            target_size
        )
    
    def get_tile_by_dominant_color(self, color_name: str) -> Optional[int]:
        """
        Find a tile with the specified dominant color.
        
        Args:
            color_name: Color name (e.g., 'Red', 'Blue', 'Green')
            
        Returns:
            Optional[int]: Tile index if found, None otherwise
            
        Example:
            >>> manager = TileManager()
            >>> red_tile = manager.get_tile_by_dominant_color('Red')
        """
        for idx, tile_data in self.tile_cache.items():
            if tile_data.dominant_color == color_name:
                return idx
        return None
    
    def get_tiles_by_dominant_color(self, color_name: str) -> List[int]:
        """
        Get all tiles with the specified dominant color.
        
        Args:
            color_name: Color name
            
        Returns:
            List[int]: List of tile indices
        """
        return [
            idx for idx, tile_data in self.tile_cache.items()
            if tile_data.dominant_color == color_name
        ]
    
    def __len__(self) -> int:
        """Return number of cached tiles."""
        return len(self.tile_cache)
    
    def __getitem__(self, idx: int) -> TileCacheTensor:
        """Get tile data by index."""
        return self.tile_cache[idx]
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the tile collection.
        
        Returns:
            Dict: Statistics including count, dominant colors, memory usage
            
        Example:
            >>> manager = TileManager()
            >>> stats = manager.get_statistics()
            >>> print(f"Total tiles: {stats['total_tiles']}")
        """
        stats = {
            'total_tiles': len(self.tile_cache),
            'tile_size': self.tile_size,
            'device': str(self.device),
            'dtype': str(self.dtype),
        }
        
        # Count by dominant color
        color_counts = {}
        for tile_data in self.tile_cache.values():
            color = tile_data.dominant_color
            color_counts[color] = color_counts.get(color, 0) + 1
        stats['dominant_colors'] = color_counts
        
        # Memory usage
        if self.device.type == 'cuda':
            stats['gpu_memory_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            total_memory = sum(
                tile.image.element_size() * tile.image.nelement()
                for tile in self.tile_cache.values()
            )
            stats['cpu_memory_mb'] = total_memory / (1024 * 1024)
        
        return stats
