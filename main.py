import gradio as gr
import numpy as np
from utils.mosaic_generator import VectorizedMosaicGenerator
from pytorch_msssim import ms_ssim
import torch
import random
import os

vectorized_mosaic_generator = VectorizedMosaicGenerator()


def rearrange_image(image: np.ndarray) -> torch.Tensor:
    new_image = torch.tensor(image, dtype=torch.float32)
    new_image = new_image.permute(2, 0, 1).unsqueeze(0)
    return new_image


def match_dimensions(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Crop the input image to match target height/width.
    target_shape: (height, width)
    """
    if image.shape[0] < target_shape[0] or image.shape[1] < target_shape[1]:
        raise ValueError("Input image is smaller than target shape")

    if (image.shape[0], image.shape[1]) == target_shape:
        return image

    return image[: target_shape[0], : target_shape[1], :]


def generate_mosaic(image: np.ndarray, chunks: int, tile_retrieval: str = "nearest") -> tuple[np.ndarray, str]:
    """
    Core mosaic generation wrapper. Returns (mosaic_image_numpy, markdown_score).
    This function assumes VectorizedMosaicGenerator.create_mosaic(image_array, chunks, tile_retrieval)
    returns a numpy array image in RGB with uint8 dtype and shape (H, W, C).
    """
    if image is None:
        raise ValueError("Input is required")

    image_array = np.asarray(image)

    # Create mosaic
    vectorized_mosaic = vectorized_mosaic_generator.create_mosaic(image_array, chunks, tile_retrieval)

    # Ensure types and shapes for MS-SSIM
    score_vectorized = ms_ssim(
        rearrange_image(match_dimensions(image_array, vectorized_mosaic.shape[:2])),
        rearrange_image(vectorized_mosaic),
        data_range=255,
        size_average=True,
    )
    return vectorized_mosaic, f"## Multi-SSIM Score: {score_vectorized.item():.4f}"


def _run_mosaic_adapter(image, grid_choice, selection_mode, seed_value):
    """
    Adapter used by Gradio button. Yields an immediate "Running..." update,
    then yields the final results so the UI shows progress.
    - grid_choice: '16x16', '32x32', '64x64'
    - selection_mode: 'Nearest match' | 'Random tiles'
    - seed_value: optional numeric seed for reproducibility (only used for random tiles)
    """
    # First quick validation and immediate feedback
    if image is None:
        yield None, "❗ Please upload an input image."
        return

    try:
        grid_size = int(str(grid_choice).split("x", 1)[0])
    except Exception:
        grid_size = 32  # fallback

    # Map selection_mode UI -> generator parameter
    tile_retrieval = "nearest" if str(selection_mode).lower().startswith("nearest") else "random"

    # Provide an immediate UI update
    yield None, "⏳ Running mosaic generation..."

    # If user asked for randomness and provided a seed, set seeds for reproducibility.
    try:
        if tile_retrieval == "random" and seed_value is not None:
            # Accept float/int-ish inputs, coerce to int if possible
            seed_int = int(seed_value)
            np.random.seed(seed_int)
            random.seed(seed_int)
            # If VectorizedMosaicGenerator supports injection of RNG, you could do:
            # vectorized_mosaic_generator.rng = np.random.default_rng(seed_int)
    except Exception:
        # ignore seed errors but continue; reproducibility won't be guaranteed
        pass

    # Run mosaic generation and catch exceptions so UI shows them
    try:
        mosaic, score_md = generate_mosaic(image, grid_size, tile_retrieval=tile_retrieval)
        yield mosaic, score_md
    except Exception as e:
        # Return the exception message to the UI instead of crashing
        yield None, f"❗ Error while generating mosaic: {str(e)}"


with gr.Blocks(fill_width=True, title="Image to Mosaic Generator") as demo:
    gr.Markdown(
        """
        # Image to Mosaic Generator
        Upload an image on the left, choose a grid size and options, then click **Run**.
        - Grid size `16x16` means the image will be divided into a 16×16 tile grid.
        - Use *Tile selection mode* to pick nearest-match tiles or random tiles.
        - (Optional) supply a random seed for reproducible random mosaics.
        """
    )

    # Top row: input (left) and output (right)
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Input Image",
                type="numpy",
                interactive=True,
                placeholder="Upload or drop an image here",
            )
        with gr.Column(scale=1):
            mosaic_out = gr.Image(label="Mosaic Output", type="numpy")
            mosaic_score = gr.Markdown(label="Score / Info")

    # Full-width parameter row spanning both columns
    with gr.Row():
        # Use a single Column with double scale so it visually spans the layout
        with gr.Column(scale=2):
            # place controls in a nested Row so they appear side-by-side
            with gr.Row():
                grid_radio = gr.Radio(
                    choices=["16x16", "32x32", "64x64"],
                    value="32x32",
                    label="Grid size",
                    info="Choose number of chunks in each dimension",
                )
                selection_mode = gr.Radio(
                    choices=["Nearest match", "Random tiles"],
                    value="Nearest match",
                    label="Tile selection mode",
                )
                seed_input = gr.Number(
                    label="Random seed (optional)",
                    value=None,
                    precision=0,
                    interactive=True,
                )
            btn = gr.Button("Run", variant="primary")

    # Hook up the button as before
    btn.click(
        fn=_run_mosaic_adapter,
        inputs=[input_image, grid_radio, selection_mode, seed_input],
        outputs=[mosaic_out, mosaic_score],
        show_progress=True,
    )

# If running standalone:
if __name__ == "__main__":
    demo.launch(pwa=True)
