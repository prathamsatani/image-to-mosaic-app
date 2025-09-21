import cv2
import numpy as np
import random
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorizedMosaicGenerator:
    """
    Optimized vectorized implementation for creating image mosaics.
    
    This class provides methods to divide an image into chunks, apply random
    rotations to each chunk, and stitch them back together to create a mosaic effect.
    Uses vectorized NumPy operations for improved performance.
    
    Attributes:
        rotation_seed (Optional[int]): Seed value for reproducible random rotations.
    
    Example:
        >>> generator = VectorizedMosaicGenerator()
        >>> generator.set_seed(42)
        >>> mosaic = generator.create_mosaic(image_array, n_chunks=5)
    """
    
    def __init__(self):
        """Initialize the VectorizedMosaicGenerator with no seed set."""
        self.rotation_seed: Optional[int] = None
    
    def set_seed(self, seed: int) -> None:
        """
        Set seed for reproducible random rotations.
        
        Args:
            seed: Integer seed value for the random number generator.
        
        Raises:
            TypeError: If seed is not an integer.
            ValueError: If seed is negative.
        """
        if not isinstance(seed, int):
            raise TypeError(f"Seed must be an integer, got {type(seed).__name__}")
        if seed < 0:
            raise ValueError(f"Seed must be non-negative, got {seed}")
            
        self.rotation_seed = seed
        np.random.seed(seed)
        logger.info(f"Random seed set to {seed}")
    
    def convert_to_chunks(self, img: np.ndarray, n: int = 3) -> np.ndarray:
        """
        Convert an image into a grid of chunks using fully vectorized operations.
        
        This method divides the input image into n×n equal-sized chunks.
        The image is truncated if dimensions are not perfectly divisible by n.
        
        Args:
            img: Input image as a NumPy array. Can be grayscale (2D) or color (3D).
            n: Number of chunks per dimension (creates n×n grid). Defaults to 3.
        
        Returns:
            4D array (grayscale) or 5D array (color) containing the image chunks.
            Shape: (n, n, chunk_height, chunk_width[, channels])
        
        Raises:
            TypeError: If img is not a NumPy array or n is not an integer.
            ValueError: If n is less than 1 or greater than image dimensions.
        """
        try:
            # Type checking
            if not isinstance(img, np.ndarray):
                logger.warning("Converting input to NumPy array")
                img = np.array(img)
            
            if not isinstance(n, int):
                raise TypeError(f"n must be an integer, got {type(n).__name__}")
            
            if n < 1:
                raise ValueError(f"n must be at least 1, got {n}")
            
            h, w = img.shape[:2]
            
            if n > h or n > w:
                raise ValueError(f"n ({n}) should be less than both image dimensions ({h}×{w})")
            
            chunk_h = h // n
            chunk_w = w // n
            
            if chunk_h == 0 or chunk_w == 0:
                raise ValueError(f"Image too small to create {n}×{n} chunks")
            
            # Truncate image to fit exact chunks
            truncated = img[:chunk_h*n, :chunk_w*n]
            
            # Reshape based on image type (grayscale vs color)
            if len(img.shape) == 3:
                channels = img.shape[2]
                reshaped = truncated.reshape(n, chunk_h, n, chunk_w, channels)
                chunks = reshaped.transpose(0, 2, 1, 3, 4)
            else:
                reshaped = truncated.reshape(n, chunk_h, n, chunk_w)
                chunks = reshaped.transpose(0, 2, 1, 3)
            
            logger.debug(f"Created {n}×{n} chunks from image of shape {img.shape}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error converting image to chunks: {str(e)}")
            raise
        
    def average_chunks_color(self, chunks: np.ndarray) -> np.ndarray:   
        """
        Compute the average color of each chunk.
        
        Args:
            chunks: Array of image chunks from convert_to_chunks.
        
        Returns:
            Array of average colors for each chunk.
            Shape: (n, n, channels) for color images or (n, n) for grayscale.
        
        Raises:
            TypeError: If chunks is not a NumPy array.
            ValueError: If chunks array has unexpected shape.
        """
        try:
            if not isinstance(chunks, np.ndarray):
                raise TypeError(f"chunks must be a NumPy array, got {type(chunks).__name__}")

            if chunks.ndim not in [4, 5]:
                raise ValueError(f"Expected 4D or 5D array, got {chunks.ndim}D")

            # Compute mean across height and width of each chunk
            avg_colors = chunks.mean(axis=(2, 3), keepdims=True)

            # Broadcast back to original shape
            avg_colors = np.broadcast_to(avg_colors, chunks.shape)
            avg_colors = avg_colors.astype(chunks.dtype, copy=False)
            logger.debug(f"Computed average colors for {chunks.shape[0]}×{chunks.shape[1]} chunks")
            return avg_colors   
            
        except Exception as e:
            logger.error(f"Error computing average chunk colors: {str(e)}")
            raise
        
    def stitch_chunks(self, chunks: np.ndarray) -> np.ndarray:
        """
        Stitch chunks back together into a single image using vectorized operations.
        
        This method reconstructs the image from the chunk grid, maintaining
        the original arrangement but with applied transformations.
        
        Args:
            chunks: Array of image chunks, typically after rotation.
                    Shape: (n, n, chunk_height, chunk_width[, channels])
        
        Returns:
            Reconstructed image as a NumPy array.
        
        Raises:
            TypeError: If chunks is not a NumPy array.
            ValueError: If chunks array has unexpected dimensions.
        """
        try:
            if not isinstance(chunks, np.ndarray):
                raise TypeError(f"chunks must be a NumPy array, got {type(chunks).__name__}")
            
            if chunks.ndim not in [4, 5]:
                raise ValueError(f"Expected 4D or 5D array, got {chunks.ndim}D")
            
            n = chunks.shape[0]
            
            if len(chunks.shape) == 5:  # Color image
                chunk_h, chunk_w, channels = chunks.shape[2:5]
                stitched = chunks.transpose(0, 2, 1, 3, 4).reshape(n*chunk_h, n*chunk_w, channels)
            else:  # Grayscale image
                chunk_h, chunk_w = chunks.shape[2:4]
                stitched = chunks.transpose(0, 2, 1, 3).reshape(n*chunk_h, n*chunk_w)
            
            logger.debug(f"Stitched {n}×{n} chunks into image of shape {stitched.shape}")
            return stitched
            
        except Exception as e:
            logger.error(f"Error stitching chunks: {str(e)}")
            raise
    
    def create_mosaic(self, image_array: np.ndarray, n_chunks: int) -> np.ndarray:
        """
        Create a mosaic effect on an image using fully vectorized operations.
        
        This method combines all steps: chunking, rotating, and stitching.
        
        Args:
            image_array: Input image as a NumPy array.
            n_chunks: Number of chunks per dimension (creates n_chunks×n_chunks grid).
        
        Returns:
            Mosaic image as a NumPy array with the same type as input.
        
        Raises:
            TypeError: If inputs are not of the expected type.
            ValueError: If n_chunks is invalid or image is too small.
        
        Example:
            >>> generator = VectorizedMosaicGenerator()
            >>> image = cv2.imread('input.jpg')
            >>> mosaic = generator.create_mosaic(image, 5)
            >>> cv2.imwrite('mosaic.jpg', mosaic)
        """
        try:
            if not isinstance(image_array, np.ndarray):
                raise TypeError(f"image_array must be a NumPy array, got {type(image_array).__name__}")
            
            if not isinstance(n_chunks, int):
                raise TypeError(f"n_chunks must be an integer, got {type(n_chunks).__name__}")
            
            if n_chunks < 1:
                raise ValueError(f"n_chunks must be at least 1, got {n_chunks}")
            
            logger.info(f"Creating {n_chunks}×{n_chunks} mosaic from image of shape {image_array.shape}")
            
            chunks = self.convert_to_chunks(image_array, n_chunks)
            averaged_chunks = self.average_chunks_color(chunks)
            mosaic = self.stitch_chunks(averaged_chunks)

            logger.info(f"Successfully created mosaic of shape {mosaic.shape}")
            return mosaic
            
        except Exception as e:
            logger.error(f"Failed to create mosaic: {str(e)}")
            raise




