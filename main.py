import gradio as gr
import numpy as np
from utils.mosaic_generator import OptimizedMosaicGenerator
from pytorch_msssim import ms_ssim
import torch
import random
import os

vectorized_mosaic_generator = OptimizedMosaicGenerator()


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

    vectorized_mosaic = vectorized_mosaic_generator.create_mosaic(image_array, chunks, tile_retrieval)

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
    if image is None:
        yield None, "❗ Please upload an input image."
        return

    try:
        grid_size = int(str(grid_choice).split("x", 1)[0])
    except Exception:
        grid_size = 32  

    tile_retrieval = "nearest" if str(selection_mode).lower().startswith("nearest") else "random"

    yield None, "⏳ Running mosaic generation..."

    try:
        if tile_retrieval == "random" and seed_value is not None:
            seed_int = int(seed_value)
            np.random.seed(seed_int)
            random.seed(seed_int)
    except Exception:
        pass

    try:
        mosaic, score_md = generate_mosaic(image, grid_size, tile_retrieval=tile_retrieval)
        yield mosaic, score_md
    except Exception as e:
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

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Input Image",
                type="numpy",
                interactive=True,
                placeholder="Upload or drop an image here",
            )

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
                label="Click an example to populate the Input Image"
            )

        with gr.Column(scale=1):
            mosaic_out = gr.Image(label="Mosaic Output", type="numpy")
            mosaic_score = gr.Markdown(label="Score / Info")

    with gr.Row():
        with gr.Column(scale=2):
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

    btn.click(
        fn=_run_mosaic_adapter,
        inputs=[input_image, grid_radio, selection_mode, seed_input],
        outputs=[mosaic_out, mosaic_score],
        show_progress=True,
    )

if __name__ == "__main__":
    demo.launch(share=True)
