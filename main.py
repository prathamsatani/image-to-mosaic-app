import gradio as gr
import numpy as np
from utils.mosaic_generator import VectorizedMosaicGenerator, MosaicGenerator
from pytorch_msssim import ms_ssim
import torch

custom_css = """
.gradio-container {
    margin: 0 !important;
}
"""

mosaic_generator = MosaicGenerator()
vectorized_mosaic_generator = VectorizedMosaicGenerator()

def rearrange_image(image: np.ndarray) -> torch.Tensor:
    new_image = torch.tensor(image, dtype=torch.float32)
    new_image = new_image.permute(2, 0, 1).unsqueeze(0)
    return new_image

def match_dimensions(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if image.shape[0] < target_shape[0] or image.shape[1] < target_shape[1]:
        raise ValueError("Input image is smaller than target shape")
    
    if image.shape == target_shape:
        return image
    
    return image[:target_shape[0], :target_shape[1], :]

def generate_mosaic(image: np.ndarray, chunks: int):
    if image is None:
        raise ValueError("Input is required")
    image_array = np.asarray(image)
    
    mosaic = mosaic_generator.create_mosaic(image_array, chunks)
    vectorized_mosaic = vectorized_mosaic_generator.create_mosaic(image_array, chunks)
    
    score = ms_ssim(
        rearrange_image(match_dimensions(image_array, mosaic.shape[:2])), 
        rearrange_image(mosaic), 
        data_range=255,  
        size_average=True
    )
    
    score_vectorized = ms_ssim(
        rearrange_image(match_dimensions(image_array, vectorized_mosaic.shape[:2])), 
        rearrange_image(vectorized_mosaic), 
        data_range=255,  
        size_average=True
    )
    return mosaic, vectorized_mosaic, f"## Multi-SSIM Score: {score.item():.4f}", f"## Vectorized Multi-SSIM Score: {score_vectorized.item():.4f}"

with gr.Blocks(fill_width=True) as demo:
    gr.Markdown("# Image to Mosaic Generator")
    with gr.Row():
        image = gr.Image(placeholder="Upload an image")
        vectorized_out = gr.Image()
        out = gr.Image()
    slider = gr.Slider(minimum=1, maximum=200, step=5, label="Number of chunks")
    vectorized_score_output = gr.Markdown()
    score_output = gr.Markdown()
    btn = gr.Button("Run")
    btn.click(fn=generate_mosaic, inputs=[image, slider], outputs=[out, vectorized_out, score_output, vectorized_score_output])

demo.launch(pwa=True)
