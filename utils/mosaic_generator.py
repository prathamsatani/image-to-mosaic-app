import cv2
import numpy as np
import random
from typing import Optional, Union, Tuple
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
    
    def rotate_chunk(self, mat: np.ndarray, times: int = 1) -> np.ndarray:
        """
        Rotate a matrix/image chunk by 90-degree increments.
        
        Args:
            mat: Input matrix/chunk as a NumPy array.
            times: Number of 90-degree clockwise rotations to apply.
                   Negative values rotate counter-clockwise.
        
        Returns:
            Rotated array with the same shape as input.
        
        Raises:
            TypeError: If inputs are not of the expected type.
            ValueError: If the chunk is empty.
        """
        try:
            if not isinstance(mat, np.ndarray):
                mat = np.asarray(mat)
            
            if not isinstance(times, int):
                raise TypeError(f"times must be an integer, got {type(times).__name__}")
            
            if mat.size == 0:
                raise ValueError("Cannot rotate empty chunk")
            
            dims = mat.shape
            effective_rotations = times % 4
            
            if effective_rotations == 0:
                return mat
            elif effective_rotations == 1:
                return cv2.rotate(mat, cv2.ROTATE_90_CLOCKWISE).reshape(dims)
            elif effective_rotations == 2:
                return cv2.rotate(mat, cv2.ROTATE_180).reshape(dims)
            elif effective_rotations == 3:
                return cv2.rotate(mat, cv2.ROTATE_90_COUNTERCLOCKWISE).reshape(dims)
                
        except Exception as e:
            logger.error(f"Error rotating chunk: {str(e)}")
            # Return original chunk if rotation fails
            return mat
    
    def apply_random_rotations(self, chunks: np.ndarray, n: int) -> np.ndarray:
        """
        Apply random rotations to all chunks in the grid.
        
        Each chunk is randomly rotated by 0, 90, 180, or 270 degrees.
        
        Args:
            chunks: Array of image chunks from convert_to_chunks.
            n: Grid dimension (n×n chunks).
        
        Returns:
            Array with the same shape as input, containing rotated chunks.
        
        Raises:
            TypeError: If inputs are not of the expected type.
            ValueError: If chunks array has unexpected shape.
        """
        try:
            if not isinstance(chunks, np.ndarray):
                raise TypeError(f"chunks must be a NumPy array, got {type(chunks).__name__}")
            
            if not isinstance(n, int) or n < 1:
                raise ValueError(f"n must be a positive integer, got {n}")
            
            if chunks.shape[0] != n or chunks.shape[1] != n:
                raise ValueError(f"Expected {n}×{n} chunks, got {chunks.shape[0]}×{chunks.shape[1]}")
            
            # Generate random rotation values (0-10 for backward compatibility)
            rotations = np.random.randint(0, 11, size=(n, n))
            
            rotated_chunks = np.empty_like(chunks)
            for i in range(n):
                for j in range(n):
                    rotated_chunks[i, j] = self.rotate_chunk(chunks[i, j], rotations[i, j])
            
            logger.debug(f"Applied random rotations to {n}×{n} chunks")
            return rotated_chunks
            
        except Exception as e:
            logger.error(f"Error applying rotations: {str(e)}")
            # Return original chunks if rotation fails
            return chunks
    
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
            rotated_chunks = self.apply_random_rotations(chunks, n_chunks)
            mosaic = self.stitch_chunks(rotated_chunks)
            
            logger.info(f"Successfully created mosaic of shape {mosaic.shape}")
            return mosaic
            
        except Exception as e:
            logger.error(f"Failed to create mosaic: {str(e)}")
            raise


class MosaicGenerator:
    """
    Basic implementation for creating image mosaics using nested lists.
    
    This class provides methods to divide an image into chunks, apply random
    rotations to each chunk, and stitch them back together. Uses traditional
    loop-based approaches.
    
    Note: VectorizedMosaicGenerator is recommended for better performance.
    
    Example:
        >>> generator = MosaicGenerator()
        >>> mosaic = generator.create_mosaic(image_array, n_chunks=5)
    """
    
    def convert_to_chunks(self, img: np.ndarray, n: int = 3) -> list[list[np.ndarray]]:
        """
        Convert an image into a grid of chunks using nested lists.
        
        Args:
            img: Input image as a NumPy array. Can be grayscale (2D) or color (3D).
            n: Number of chunks per dimension (creates n×n grid). Defaults to 3.
        
        Returns:
            List of lists containing image chunks.
        
        Raises:
            TypeError: If img is not array-like or n is not an integer.
            ValueError: If n is invalid or greater than image area.
        """
        try:
            if not isinstance(img, np.ndarray):
                logger.warning("Converting input to NumPy array")
                img = np.array(img)
            
            if not isinstance(n, int):
                raise TypeError(f"n must be an integer, got {type(n).__name__}")
            
            if n < 1:
                raise ValueError(f"n must be at least 1, got {n}")
            
            dims = img.shape
            if n > dims[0] * dims[1]:
                raise ValueError(f"n ({n}) should be less than image area ({dims[0]}×{dims[1]})")
            
            chunks = []
            for j in range(n):
                row = []
                for k in range(n):
                    start_row = int(len(img) * j / n)
                    end_row = int(len(img) * (j + 1) / n)
                    start_col = int(dims[1] * k / n)
                    end_col = int(dims[1] * (k + 1) / n)
                    
                    chunk = np.array([img[i][start_col:end_col] 
                                    for i in range(start_row, end_row)])
                    row.append(chunk)
                chunks.append(row)
            
            logger.debug(f"Created {n}×{n} chunks from image of shape {img.shape}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error converting image to chunks: {str(e)}")
            raise

    def rotate_matrix(self, mat: np.ndarray, times: int = 1) -> np.ndarray:
        """
        Rotate a matrix/image chunk by 90-degree increments.
        
        Args:
            mat: Input matrix/chunk as a NumPy array.
            times: Number of 90-degree clockwise rotations to apply.
        
        Returns:
            Rotated array with the same shape as input.
        
        Raises:
            TypeError: If inputs are not of the expected type.
            ValueError: If the matrix is empty.
        """
        try:
            if not isinstance(mat, np.ndarray):
                mat = np.asarray(mat)
            
            if not isinstance(times, int):
                raise TypeError(f"times must be an integer, got {type(times).__name__}")
            
            if mat.size == 0:
                raise ValueError("Cannot rotate empty matrix")
            
            dims = mat.shape
            effective_rotations = times % 4
            
            if effective_rotations == 0:
                return mat
            elif effective_rotations == 1:
                return cv2.rotate(mat, cv2.ROTATE_90_CLOCKWISE).reshape(dims)
            elif effective_rotations == 2:
                return cv2.rotate(mat, cv2.ROTATE_180).reshape(dims)
            elif effective_rotations == 3:
                return cv2.rotate(mat, cv2.ROTATE_90_COUNTERCLOCKWISE).reshape(dims)
            
            return mat
            
        except Exception as e:
            logger.error(f"Error rotating matrix: {str(e)}")
            return mat

    def stitch_chunks(self, chunks: list[list[np.ndarray]]) -> np.ndarray:
        """
        Stitch chunks back together into a single image using hstack/vstack.
        
        Args:
            chunks: List of lists containing image chunks.
        
        Returns:
            Reconstructed image as a NumPy array.
        
        Raises:
            TypeError: If chunks is not a list of lists.
            ValueError: If chunks is empty or malformed.
        """
        try:
            if not isinstance(chunks, list):
                raise TypeError(f"chunks must be a list, got {type(chunks).__name__}")
            
            if not chunks:
                raise ValueError("Cannot stitch empty chunks list")
            
            if not all(isinstance(row, list) for row in chunks):
                raise TypeError("chunks must be a list of lists")
            
            rows = []
            for i in range(len(chunks)):
                if not chunks[i]:
                    raise ValueError(f"Row {i} in chunks is empty")
                row_stitched = np.hstack(chunks[i])
                rows.append(row_stitched)
            
            stitched = np.vstack(rows)
            logger.debug(f"Stitched chunks into image of shape {stitched.shape}")
            return stitched
            
        except Exception as e:
            logger.error(f"Error stitching chunks: {str(e)}")
            raise

    def create_mosaic(self, image_array: np.ndarray, n_chunks: int) -> np.ndarray:
        """
        Create a mosaic effect on an image using traditional loop-based approach.
        
        Args:
            image_array: Input image as a NumPy array.
            n_chunks: Number of chunks per dimension (creates n_chunks×n_chunks grid).
        
        Returns:
            Mosaic image as a NumPy array.
        
        Raises:
            TypeError: If inputs are not of the expected type.
            ValueError: If n_chunks is invalid or image is too small.
        
        Example:
            >>> generator = MosaicGenerator()
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
            
            # Apply random rotations
            for i in range(len(chunks)):
                for j in range(len(chunks[0])):
                    t = random.randint(0, 10)
                    chunks[i][j] = self.rotate_matrix(chunks[i][j], t)
            
            mosaic = self.stitch_chunks(chunks)
            
            logger.info(f"Successfully created mosaic of shape {mosaic.shape}")
            return mosaic
            
        except Exception as e:
            logger.error(f"Failed to create mosaic: {str(e)}")
            raise


def validate_image(image: Union[np.ndarray, list]) -> Tuple[bool, str]:
    """
    Validate if the input is a valid image array.
    
    Args:
        image: Input to validate.
    
    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.
    """
    try:
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        if image.ndim not in [2, 3]:
            return False, f"Image must be 2D or 3D, got {image.ndim}D"
        
        if image.ndim == 3 and image.shape[2] not in [1, 3, 4]:
            return False, f"Color image must have 1, 3, or 4 channels, got {image.shape[2]}"
        
        if image.size == 0:
            return False, "Image is empty"
        
        if np.any(np.array(image.shape[:2]) < 2):
            return False, f"Image dimensions must be at least 2×2, got {image.shape[:2]}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


# Example usage
if __name__ == "__main__":
    try:
        # Create a sample image for testing
        sample_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        
        # Validate image
        is_valid, error_msg = validate_image(sample_image)
        if not is_valid:
            raise ValueError(f"Invalid image: {error_msg}")
        
        # Test vectorized implementation
        vec_generator = VectorizedMosaicGenerator()
        vec_generator.set_seed(42)
        vec_mosaic = vec_generator.create_mosaic(sample_image, n_chunks=5)
        print(f"Vectorized mosaic created: {vec_mosaic.shape}")
        
        # Test basic implementation
        basic_generator = MosaicGenerator()
        basic_mosaic = basic_generator.create_mosaic(sample_image, n_chunks=5)
        print(f"Basic mosaic created: {basic_mosaic.shape}")
        
    except Exception as e:
        logger.error(f"Example failed: {str(e)}")