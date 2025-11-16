"""
Gradio Web Interface for Image to Mosaic Generator

This application provides an interactive web interface for creating image mosaics
using the modular mosaic_generator package. It supports multiple grid sizes,
tile selection modes, and provides quality metrics.

Features:
    - Multiple grid sizes (16x16, 32x32, 64x64)
    - Tile selection modes (Nearest match, Random tiles)
    - Reproducible results with optional seed
    - MS-SSIM quality metric display
    - Processing time tracking
    - Example images for quick testing

Author: Pratham Satani
Course: CS5130 - Applied Programming and Data Processing for AI
Institution: Northeastern University
"""

import gradio as gr
import numpy as np
import time
import random
from typing import Tuple, Optional

# Import from the new modular package
from mosaic_generator import TileManager, MosaicBuilder
from mosaic_generator.metrics import compute_ms_ssim
from mosaic_generator.utils import match_dimensions, rearrange_for_ms_ssim
from mosaic_generator.config import get_device
import torch

# Initialize the mosaic generation components
print("🚀 Initializing Mosaic Generator...")
tile_manager = TileManager(tile_directory="images", metadata_file="tiles_metadata.csv")
mosaic_builder = MosaicBuilder(tile_manager, grid_size=32)

# Get device information
device = get_device()
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
device_info = f"{'🎮 GPU' if device.type == 'cuda' else '💻 CPU'}: {device_name}"
print(f"✅ Initialization complete! Using {device_info}")


def generate_mosaic(
    image: np.ndarray, 
    chunks: int, 
    tile_retrieval: str = "nearest"
) -> Tuple[np.ndarray, str, str]:
    """
    Core mosaic generation wrapper with performance tracking.
    
    Args:
        image: Input image as NumPy array
        chunks: Grid size (n×n)
        tile_retrieval: 'nearest' or 'random'
        
    Returns:
        Tuple of (mosaic_image, score_markdown, time_markdown)
    """
    if image is None:
        raise ValueError("Input is required")
    
    image_array = np.asarray(image)
    
    # Track processing time
    start_time = time.time()
    
    # Generate mosaic using the new modular builder
    mosaic = mosaic_builder.create_mosaic(
        image_array, 
        grid_size=chunks, 
        tile_retrieval=tile_retrieval
    )
    
    processing_time = time.time() - start_time
    
    # Compute MS-SSIM score
    matched_input = match_dimensions(image_array, mosaic.shape[:2])
    
    score = compute_ms_ssim(
        rearrange_for_ms_ssim(matched_input),
        rearrange_for_ms_ssim(mosaic),
        data_range=255
    )
    
    score_md = f"## Multi-SSIM Score: {score:.4f}"
    time_md = f"**Processing Time:** {processing_time:.3f}s"
    
    return mosaic, score_md, time_md


def run_mosaic_adapter(image, grid_choice, selection_mode, seed_value):
    """
    Adapter for Gradio button that handles UI updates and error handling.
    
    Yields intermediate updates for progress indication and final results.
    
    Args:
        image: Input image from Gradio
        grid_choice: Grid size selection ('16x16', '32x32', '64x64')
        selection_mode: Tile selection mode ('Nearest match' or 'Random tiles')
        seed_value: Optional random seed for reproducibility
        
    Yields:
        Tuple of (mosaic_image, info_markdown, time_markdown)
    """
    if image is None:
        yield None, "❗ Please upload an input image.", ""
        return
    
    # Parse grid size
    try:
        grid_size = int(str(grid_choice).split("x", 1)[0])
    except Exception:
        grid_size = 32
    
    # Determine tile retrieval mode
    tile_retrieval = "nearest" if str(selection_mode).lower().startswith("nearest") else "random"
    
    # Show progress update
    yield None, "⏳ Running mosaic generation...", ""
    
    # Set random seed if provided (for random mode)
    try:
        if tile_retrieval == "random" and seed_value is not None:
            seed_int = int(seed_value)
            np.random.seed(seed_int)
            random.seed(seed_int)
            import torch
            torch.manual_seed(seed_int)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed_int)
    except Exception as e:
        print(f"Warning: Could not set seed: {e}")
    
    # Generate mosaic
    try:
        mosaic, score_md, time_md = generate_mosaic(image, grid_size, tile_retrieval=tile_retrieval)
        yield mosaic, score_md, time_md
    except Exception as e:
        yield None, f"❗ Error while generating mosaic: {str(e)}", ""


# Create the Gradio interface
with gr.Blocks(fill_width=True, title="Image to Mosaic Generator") as demo:
    gr.Markdown(
        f"""
        # 🎨 Image to Mosaic Generator
        
        Transform your images into beautiful mosaics using tile-matching algorithms!
        
        **Device:** {device_info}
        
        Upload an image, choose settings, and click **Run** to generate your mosaic.
        
        ### Features:
        - **Grid Sizes**: Choose between 16×16, 32×32, or 64×64 tile grids
        - **Selection Modes**:
          - *Nearest Match*: Intelligently matches tiles based on color similarity
          - *Random Tiles*: Creates artistic variations with random tile placement
        - **Quality Metrics**: MS-SSIM score and processing time tracking
        - **Reproducibility**: Set a random seed for consistent random mosaics
        
        ---
        """
    )
    
    with gr.Row():
        # Left column - Input
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Input Image")
            input_image = gr.Image(
                label="Upload Image",
                type="numpy",
                interactive=True,
                placeholder="Upload or drop an image here",
            )
            
            gr.Markdown("### 📋 Examples")
            gr.Markdown("*Click an example to load it*")
            
            # Example inputs
            examples_list = [
                ["examples/example_1.png", "64x64", "Random tiles", None],
                ["examples/example_2.png", "32x32", "Nearest Match", 42],
                ["examples/example_3.png", "16x16", "Nearest match", None],
                ["examples/example_4.png", "32x32", "Random tiles", 123],
                ["examples/example_5.png", "64x64", "Nearest match", None]
            ]
            
            gr.Examples(
                examples=examples_list,
                inputs=[input_image],
                label="Example Images"
            )
        
        # Right column - Output
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Mosaic Output")
            mosaic_out = gr.Image(label="Generated Mosaic", type="numpy")
            mosaic_score = gr.Markdown("")
            processing_time = gr.Markdown("")
    
    # Settings row
    gr.Markdown("---")
    gr.Markdown("### ⚙️ Settings")
    
    with gr.Row():
        grid_radio = gr.Radio(
            choices=["16x16", "32x32", "64x64"],
            value="32x32",
            label="Grid Size",
            info="Number of tiles per dimension (larger = more detail, slower)",
        )
        
        selection_mode = gr.Radio(
            choices=["Nearest match", "Random tiles"],
            value="Nearest match",
            label="Tile Selection Mode",
            info="How tiles are chosen for each grid cell",
        )
        
        seed_input = gr.Number(
            label="Random Seed (Optional)",
            value=None,
            precision=0,
            interactive=True,
            info="For reproducible random mosaics",
        )
    
    # Run button
    btn = gr.Button("🚀 Generate Mosaic", variant="primary", size="lg")
    
    # Connect the button click to the processing function
    btn.click(
        fn=run_mosaic_adapter,
        inputs=[input_image, grid_radio, selection_mode, seed_input],
        outputs=[mosaic_out, mosaic_score, processing_time],
        show_progress=True,
    )
    
    # Footer
    gr.Markdown(
        f"""
        ---
        
        ### 📊 Performance Information
        
        **Current Device:** {device_info}
        
        This application uses a **modular, GPU-accelerated architecture** with the following optimizations:
        - PyTorch tensor operations for GPU acceleration
        - Batched tile matching for efficiency
        - Pre-cached tiles in device memory
        - Vectorized color distance computations
        
        **Built with:**
        - `mosaic_generator` - Modular mosaic generation package
        - `Gradio` - Interactive web interface
        - `PyTorch` - GPU-accelerated operations
        - `pytorch-msssim` - Quality metrics
        
        **Author:** Pratham Satani | **Course:** CS5130 | **Institution:** Northeastern University
        """
    )


# Launch the application
if __name__ == "__main__":
    demo.launch(share=True)
