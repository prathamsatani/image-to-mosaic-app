"""
Unit tests for the mosaic_generator package.

This module contains comprehensive tests for the mosaic generation functionality,
including tile management, image processing, metrics computation, and mosaic building.

Author: Pratham Satani
Course: CS5130 - Applied Programming and Data Processing for AI
Institution: Northeastern University
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import cv2

# Import package modules
from mosaic_generator import TileManager, MosaicBuilder
from mosaic_generator.image_processor import (
    load_image,
    resize_image,
    create_image_grid,
    stitch_chunks,
    preprocess_image,
)
from mosaic_generator.metrics import (
    compute_mse,
    compute_ssim,
    compute_ms_ssim,
    compute_psnr,
)
from mosaic_generator.utils import (
    rgb_to_text,
    numpy_to_tensor,
    tensor_to_numpy,
    validate_image,
    match_dimensions,
)
from mosaic_generator.config import (
    validate_grid_size,
    validate_tile_size,
    validate_blend_alpha,
)


class TestUtils:
    """Test utility functions."""
    
    def test_rgb_to_text(self):
        """Test RGB to color name conversion."""
        assert rgb_to_text(255, 0, 0) == "Red"
        assert rgb_to_text(0, 255, 0) == "Green"
        assert rgb_to_text(0, 0, 255) == "Blue"
        assert rgb_to_text(255, 255, 0) == "Yellow"
        assert rgb_to_text(128, 128, 128) == "Unknown"
    
    def test_numpy_to_tensor(self):
        """Test NumPy to tensor conversion."""
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        tensor = numpy_to_tensor(image)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 256, 256)
        assert tensor.dtype == torch.float32
    
    def test_tensor_to_numpy(self):
        """Test tensor to NumPy conversion."""
        tensor = torch.rand(3, 256, 256)
        image = tensor_to_numpy(tensor)
        
        assert isinstance(image, np.ndarray)
        assert image.shape == (256, 256, 3)
        assert image.dtype == np.uint8
    
    def test_match_dimensions(self):
        """Test dimension matching."""
        image = np.ones((300, 400, 3), dtype=np.uint8)
        cropped = match_dimensions(image, (256, 256))
        
        assert cropped.shape == (256, 256, 3)
    
    def test_validate_image(self):
        """Test image validation."""
        valid_image = np.zeros((256, 256, 3), dtype=np.uint8)
        validate_image(valid_image)  # Should not raise
        
        with pytest.raises(ValueError):
            validate_image(valid_image, min_size=(512, 512))


class TestConfig:
    """Test configuration validation."""
    
    def test_validate_grid_size(self):
        """Test grid size validation."""
        validate_grid_size(32)  # Should not raise
        
        with pytest.raises(ValueError):
            validate_grid_size(1)  # Too small
        
        with pytest.raises(ValueError):
            validate_grid_size(200)  # Too large
    
    def test_validate_tile_size(self):
        """Test tile size validation."""
        validate_tile_size((32, 32))  # Should not raise
        
        with pytest.raises(ValueError):
            validate_tile_size((2, 2))  # Too small
    
    def test_validate_blend_alpha(self):
        """Test blend alpha validation."""
        validate_blend_alpha(0.5)  # Should not raise
        
        with pytest.raises(ValueError):
            validate_blend_alpha(1.5)  # Out of range


class TestImageProcessor:
    """Test image processing functions."""
    
    def test_resize_image(self):
        """Test image resizing."""
        image = np.zeros((1024, 768, 3), dtype=np.uint8)
        resized = resize_image(image, (512, 512))
        
        assert resized.shape == (512, 512, 3)
    
    def test_create_image_grid(self):
        """Test grid creation."""
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        chunks = create_image_grid(image, grid_size=16)
        
        assert chunks.shape == (16, 16, 32, 32, 3)
    
    def test_stitch_chunks(self):
        """Test chunk stitching."""
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        chunks = create_image_grid(image, grid_size=16)
        stitched = stitch_chunks(chunks)
        
        assert stitched.shape == (512, 512, 3)
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        image = np.random.randint(0, 256, (513, 513, 3), dtype=np.uint8)
        processed = preprocess_image(image, ensure_divisible=32)
        
        assert processed.shape[0] % 32 == 0
        assert processed.shape[1] % 32 == 0


class TestMetrics:
    """Test similarity metrics."""
    
    def test_compute_mse_identical(self):
        """Test MSE for identical images."""
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        mse = compute_mse(image, image)
        
        assert mse == 0.0
    
    def test_compute_mse_different(self):
        """Test MSE for different images."""
        image1 = np.zeros((256, 256, 3), dtype=np.uint8)
        image2 = np.ones((256, 256, 3), dtype=np.uint8) * 255
        mse = compute_mse(image1, image2)
        
        assert mse > 0
    
    def test_compute_ssim(self):
        """Test SSIM computation."""
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        ssim_val = compute_ssim(image, image)
        
        assert 0 <= ssim_val <= 1
        assert ssim_val > 0.99  # Should be very close to 1 for identical images
    
    def test_compute_ms_ssim(self):
        """Test MS-SSIM computation."""
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        ms_ssim_val = compute_ms_ssim(image, image)
        
        assert 0 <= ms_ssim_val <= 1
        assert ms_ssim_val > 0.99
    
    def test_compute_psnr(self):
        """Test PSNR computation."""
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        psnr = compute_psnr(image, image)
        
        assert psnr == float('inf')  # Identical images


class TestMosaicBuilder:
    """Test mosaic building functionality."""
    
    @pytest.fixture
    def setup_builder(self):
        """Setup fixture for mosaic builder tests."""
        # Create a minimal tile manager for testing
        # This assumes tiles_metadata.csv and images/ directory exist
        try:
            tile_manager = TileManager(
                tile_directory="images",
                metadata_file="tiles_metadata.csv"
            )
            builder = MosaicBuilder(tile_manager, grid_size=16)
            return builder
        except FileNotFoundError:
            pytest.skip("Tile data not available for testing")
    
    def test_create_mosaic_nearest(self, setup_builder):
        """Test mosaic creation with nearest matching."""
        builder = setup_builder
        
        # Create a test image
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        
        # Generate mosaic
        mosaic = builder.create_mosaic(image, grid_size=16, tile_retrieval="nearest")
        
        assert isinstance(mosaic, np.ndarray)
        assert mosaic.shape[0] > 0 and mosaic.shape[1] > 0
        assert mosaic.dtype == np.uint8
    
    def test_create_mosaic_random(self, setup_builder):
        """Test mosaic creation with random tiles."""
        builder = setup_builder
        
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        
        mosaic = builder.create_mosaic(image, grid_size=16, tile_retrieval="random")
        
        assert isinstance(mosaic, np.ndarray)
        assert mosaic.shape[0] > 0 and mosaic.shape[1] > 0
    
    def test_compute_similarity(self, setup_builder):
        """Test similarity computation."""
        builder = setup_builder
        
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        
        # Similarity with self should be high
        similarity = builder.compute_similarity(image, image)
        
        assert 0 <= similarity <= 1
        assert similarity > 0.99


class TestIntegration:
    """Integration tests for complete workflow."""
    
    @pytest.mark.skipif(
        not Path("images").exists() or not Path("tiles_metadata.csv").exists(),
        reason="Tile data not available"
    )
    def test_full_workflow(self):
        """Test complete mosaic generation workflow."""
        # Initialize components
        tile_manager = TileManager(
            tile_directory="images",
            metadata_file="tiles_metadata.csv"
        )
        builder = MosaicBuilder(tile_manager, grid_size=32)
        
        # Create test image
        test_image = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        
        # Generate mosaic
        mosaic = builder.create_mosaic(
            test_image,
            grid_size=32,
            tile_retrieval="nearest"
        )
        
        # Compute quality
        similarity = builder.compute_similarity(test_image, mosaic)
        
        # Assertions
        assert mosaic.shape[0] > 0
        assert mosaic.shape[1] > 0
        assert 0 <= similarity <= 1
        
        print(f"✅ Integration test passed! Similarity: {similarity:.4f}")


def test_package_imports():
    """Test that all package components can be imported."""
    from mosaic_generator import (
        TileManager,
        MosaicBuilder,
        load_image,
        resize_image,
        create_image_grid,
        compute_mse,
        compute_ssim,
        compute_ms_ssim,
    )
    
    assert TileManager is not None
    assert MosaicBuilder is not None
    assert load_image is not None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
