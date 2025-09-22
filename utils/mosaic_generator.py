import os
import cv2
import numpy as np
import pandas as pd
import logging

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
        self.tiles_metadata = pd.read_csv("tiles_metadata.csv")
        
    
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
            
            truncated = img[:chunk_h*n, :chunk_w*n]
            
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

            avg_colors = chunks.mean(axis=(2, 3), keepdims=True)

            avg_colors = np.broadcast_to(avg_colors, chunks.shape)
            avg_colors = avg_colors.astype(chunks.dtype, copy=False)
            logger.debug(f"Computed average colors for {chunks.shape[0]}×{chunks.shape[1]} chunks")
            return avg_colors   
            
        except Exception as e:
            logger.error(f"Error computing average chunk colors: {str(e)}")
            raise
    
    def retrieve_tile_images(self, chunks: np.ndarray) -> np.ndarray:
        """Retrieve tile images for a batch of chunks, resizing tiles to match chunk dimensions.
        
        Args:
            chunks (np.ndarray): Batch of chunks with shape (n, n, height, width, channels)
            
        Returns:
            np.ndarray: Tile images organized as (n, n, chunk_height, chunk_width, channels)
        """
        tile_rgbs = self.tiles_metadata[["average-red", "average-green", "average-blue"]].values

        n_rows, n_cols, chunk_h, chunk_w, channels = chunks.shape
        
        chunks_flat = chunks.reshape(-1, chunk_h, chunk_w, channels)
        chunks_avg = chunks_flat.mean(axis=(1, 2))
        
        distances = np.linalg.norm(
            chunks_avg[:, np.newaxis, :] - tile_rgbs[np.newaxis, :, :], 
            axis=2
        )
        
        closest_indices = distances.argmin(axis=1)
        tile_filenames = self.tiles_metadata.iloc[closest_indices]['filename'].values
        
        tile_images = np.zeros((n_rows, n_cols, chunk_h, chunk_w, channels), dtype=np.uint8)
        
        for idx, filename in enumerate(tile_filenames):
            row, col = divmod(idx, n_cols)
            
            image_path = os.path.join(os.getcwd(), "images", filename)
            tile = cv2.imread(image_path)
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            
            resized_tile = cv2.resize(tile, (chunk_w, chunk_h), interpolation=cv2.INTER_AREA)
            
            tile_images[row, col] = resized_tile
        
        return tile_images
    
    def retrieve_tile_images_randomly(self, chunks: np.ndarray) -> np.ndarray:
        
        n_rows, n_cols, chunk_h, chunk_w, channels = chunks.shape
        chunks_flat = chunks.reshape(-1, chunk_h, chunk_w, channels)

        rng = np.random.default_rng()
        num_tiles = len(self.tiles_metadata)
        random_indices = rng.integers(0, num_tiles, size=chunks_flat.shape[0])

        tile_filenames = self.tiles_metadata.iloc[random_indices]['filename'].values

        tile_images = np.zeros((n_rows, n_cols, chunk_h, chunk_w, channels), dtype=np.uint8)

        for idx, filename in enumerate(tile_filenames):
            row, col = divmod(idx, n_cols)
            image_path = os.path.join(os.getcwd(), "images", filename)

            tile = cv2.imread(image_path)
            if tile is None:
                tile = np.zeros((chunk_h, chunk_w, channels), dtype=np.uint8)
            else:
                tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                if tile.ndim == 2 and channels == 3:
                    tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
                elif tile.ndim == 3 and tile.shape[2] != channels:
                    if channels == 1:
                        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)[..., np.newaxis]
                    else:
                        tile = tile[..., :channels]

                resized_tile = cv2.resize(tile, (chunk_w, chunk_h), interpolation=cv2.INTER_AREA)
                tile = resized_tile

            tile_images[row, col] = tile

        return tile_images
    
    def superimpose_tiles_and_chunks(self, chunks: np.ndarray, tiles: np.ndarray, tile_retrieval: str) -> np.ndarray:
        """
        Superimpose tile images onto their corresponding chunks.

        Args:
            chunks (np.ndarray): Array of image chunks with shape (n, n, chunk_height, chunk_width[, channels])
            tiles (np.ndarray): Array of tile images with shape (n, n, chunk_height, chunk_width[, channels])

        Returns:
            np.ndarray: Superimposed image as a NumPy array.
        """
        try:
            if not isinstance(chunks, np.ndarray):
                raise TypeError(f"chunks must be a NumPy array, got {type(chunks).__name__}")

            if not isinstance(tiles, np.ndarray):
                raise TypeError(f"tiles must be a NumPy array, got {type(tiles).__name__}")

            if chunks.shape != tiles.shape:
                raise ValueError(f"Chunks and tiles must have the same shape, got {chunks.shape} and {tiles.shape}")

            for i in range(chunks.shape[0]):
                for j in range(chunks.shape[1]):
                    alpha = 0.5 if tile_retrieval == "nearest" else 1
                    beta = 0.5 if tile_retrieval == "nearest" else 0.3
                    chunks[i, j] = cv2.addWeighted(chunks[i, j], alpha, tiles[i, j], beta, 0)

            logger.debug(f"Superimposed tiles onto chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error superimposing tiles and chunks: {str(e)}")
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
            
            if len(chunks.shape) == 5:  
                chunk_h, chunk_w, channels = chunks.shape[2:5]
                stitched = chunks.transpose(0, 2, 1, 3, 4).reshape(n*chunk_h, n*chunk_w, channels)
            else:  
                chunk_h, chunk_w = chunks.shape[2:4]
                stitched = chunks.transpose(0, 2, 1, 3).reshape(n*chunk_h, n*chunk_w)
            
            logger.debug(f"Stitched {n}×{n} chunks into image of shape {stitched.shape}")
            return stitched
            
        except Exception as e:
            logger.error(f"Error stitching chunks: {str(e)}")
            raise


    def create_mosaic(self, image_array: np.ndarray, n_chunks: int, tile_retrieval: str) -> np.ndarray:
        """
        Create a mosaic effect on an image using fully vectorized operations.
        
        This method combines all steps: chunking, rotating, and stitching.

        Args:
            image_array: Input image as a NumPy array.
            n_chunks: Number of chunks per dimension (creates n_chunks×n_chunks grid).
            tile_retrieval: Method for retrieving tiles ("nearest" or "random").

        Returns:
            Mosaic image as a NumPy array with the same type as input.

        Raises:
            TypeError: If inputs are not of the expected type.
            ValueError: If n_chunks is invalid or image is too small.
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
            if tile_retrieval == "nearest":
                tile_images = self.retrieve_tile_images(chunks)
            else:
                tile_images = self.retrieve_tile_images_randomly(chunks)
                
            superimposed_chunks = self.superimpose_tiles_and_chunks(averaged_chunks, tile_images, tile_retrieval)
            mosaic = self.stitch_chunks(superimposed_chunks)

            logger.info(f"Successfully created mosaic of shape {mosaic.shape}")
            return mosaic
            
        except Exception as e:
            logger.error(f"Failed to create mosaic: {str(e)}")
            raise
