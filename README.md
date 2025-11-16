---
title: image-to-mosaic-app
app_file: app.py
sdk: gradio
sdk_version: 5.43.1
---

# 🎨 Image to Mosaic Generator

A high-performance, modular Python package that transforms images into artistic mosaics using GPU-accelerated PyTorch operations and intelligent tile-matching algorithms. Built with a clean, maintainable architecture following software engineering best practices.

**Version 2.0** - Fully Refactored Modular Architecture

**Academic Project** - CS5130 Applied Programming and Data Processing for AI, Northeastern University

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-yellow)

---

## ✨ Features

### 🚀 Performance
- **GPU Acceleration**: PyTorch-based operations with automatic CUDA support
- **Batched Processing**: Efficient tile matching and retrieval
- **Memory Optimization**: Pre-cached tiles with FP16 support on GPU
- **Vectorized Operations**: Optimized tensor computations

### 🎯 Functionality
- **Multiple Grid Sizes**: 16×16, 32×32, or 64×64 tile grids
- **Tile Selection Modes**:
  - **Nearest Match**: Intelligent color-based tile matching using Euclidean distance
  - **Random Tiles**: Artistic variations with reproducible random seeds
- **Quality Metrics**: MS-SSIM, SSIM, MSE, and PSNR for quantitative evaluation
- **Interactive UI**: Modern Gradio-based web interface with real-time progress and device detection
- **Device Visibility**: Automatic detection and display of compute device (CPU/GPU) with hardware name

### 🏗️ Architecture
- **Modular Design**: Clean separation of concerns across modules
- **Comprehensive Documentation**: Detailed docstrings and type hints
- **Error Handling**: Robust validation and informative error messages
- **Extensible**: Easy to add new features and tile matching algorithms
- **Unit Tests**: Comprehensive test suite with pytest

---

## 📁 Project Structure

```
image-to-mosaic-app/
├── mosaic_generator/          # Main package
│   ├── __init__.py           # Package initialization and exports
│   ├── config.py             # Configuration constants and validation
│   ├── utils.py              # Helper functions and utilities
│   ├── image_processor.py    # Image loading, resizing, grid creation
│   ├── tile_manager.py       # Tile loading, caching, feature extraction
│   ├── mosaic_builder.py     # Main mosaic construction logic
│   └── metrics.py            # Similarity metrics (MSE, SSIM, MS-SSIM)
│
├── app.py                     # Gradio web interface
├── requirements.txt           # Dependencies with versions
├── README.md                  # This file
│
├── tests/                     # Unit tests
│   └── test_mosaic.py        # Comprehensive test suite
│
├── images/                    # Tile images directory
├── tiles_metadata.csv         # Tile metadata (colors, features)
│
└── examples/                  # Example input images
    ├── example_1.png
    ├── example_2.png
    └── ...
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for acceleration

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/prathamsatani/image-to-mosaic-app.git
cd image-to-mosaic-app
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Prepare tile images:**
   - Place your tile images in the `images/` directory
   - Run the preprocessing script to generate metadata:
```bash
python utils/preprocess_tiles.py
```
   - This creates `tiles_metadata.csv` with color statistics

---

## 💻 Usage

### Web Interface (Recommended)

Launch the Gradio app:
```bash
python app.py
```

The interface will open in your browser (typically `http://localhost:7860`).

**Features:**
- Upload images or use provided examples
- Select grid size (16×16, 32×32, 64×64)
- Choose tile selection mode (Nearest match or Random)
- Optional: Set random seed for reproducibility
- View MS-SSIM quality score and processing time
- **Device indicator**: Shows whether CPU or GPU is being used (e.g., 🎮 GPU: NVIDIA GeForce RTX 3080)

### Programmatic Usage

```python
from mosaic_generator import TileManager, MosaicBuilder
from mosaic_generator.image_processor import load_image
from mosaic_generator.metrics import compute_ms_ssim

# Initialize components
tile_manager = TileManager(tile_directory='images/')
builder = MosaicBuilder(tile_manager, grid_size=32)

# Load image
image = load_image('path/to/image.jpg')

# Generate mosaic
mosaic = builder.create_mosaic(
    image, 
    grid_size=32, 
    tile_retrieval='nearest',
    blend_alpha=0.5
)

# Evaluate quality
similarity = builder.compute_similarity(image, mosaic)
print(f"MS-SSIM: {similarity:.4f}")

# Save result
import cv2
cv2.imwrite('output.jpg', cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))
```

### Advanced Examples

**Check device information:**
```python
from mosaic_generator.config import get_device
import torch

device = get_device()
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Random mosaic with seed:**
```python
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)
mosaic = builder.create_mosaic(
    image,
    grid_size=32,
    tile_retrieval='random'
)
```

**Custom blend factor:**
```python
# More tile influence (less original image)
mosaic = builder.create_mosaic(
    image,
    grid_size=32,
    blend_alpha=0.3  # 30% original, 70% tile
)
```

**Get performance statistics:**
```python
stats = builder.get_performance_stats()
print(f"Device: {stats['device']}")
print(f"Grid size: {stats['grid_size']}")
if 'gpu_name' in stats:
    print(f"GPU: {stats['gpu_name']}")
```

---

## 🧪 Running Tests

Execute the test suite:
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=mosaic_generator --cov-report=html

# Run specific test class
python -m pytest tests/test_mosaic.py::TestMosaicBuilder -v
```

---

## 📊 Module Documentation

### `mosaic_generator.tile_manager.TileManager`

Manages tile loading, caching, and retrieval.

**Methods:**
- `find_nearest_tile(color)`: Find best matching tile for a color
- `find_nearest_tiles_batch(colors)`: Batch tile matching (GPU-optimized)
- `get_tile(idx, target_size)`: Retrieve a specific tile
- `get_random_tiles(count, target_size)`: Get random tiles

**Example:**
```python
tile_manager = TileManager('images/')
print(f"Loaded {len(tile_manager)} tiles")

# Find best match for red
red_tile_idx = tile_manager.find_nearest_tile([255, 0, 0])
```

### `mosaic_generator.mosaic_builder.MosaicBuilder`

Main class for mosaic generation.

**Methods:**
- `create_mosaic(image, grid_size, tile_retrieval, blend_alpha)`: Generate mosaic
- `compute_similarity(original, mosaic, metric)`: Calculate similarity score

**Example:**
```python
builder = MosaicBuilder(tile_manager, grid_size=(32, 32))
mosaic = builder.create_mosaic(image)
```

### `mosaic_generator.metrics`

Similarity metrics for quality evaluation.

**Functions:**
- `compute_mse(img1, img2)`: Mean Squared Error
- `compute_ssim(img1, img2)`: Structural Similarity Index
- `compute_ms_ssim(img1, img2)`: Multi-Scale SSIM (recommended)
- `compute_psnr(img1, img2)`: Peak Signal-to-Noise Ratio

**Example:**
```python
from mosaic_generator.metrics import compute_ms_ssim

score = compute_ms_ssim(original, mosaic)
print(f"Quality: {score:.4f}")
```

---

## ⚙️ Configuration

Edit `mosaic_generator/config.py` to customize:

```python
# Grid and tile settings
DEFAULT_GRID_SIZE = 32
DEFAULT_TILE_SIZE = (32, 32)
DEFAULT_BLEND_ALPHA = 0.5

# Performance settings
DEFAULT_DEVICE = "auto"  # 'auto', 'cuda', or 'cpu'
USE_HALF_PRECISION = True  # FP16 on GPU
BATCH_SIZE = 64

# Quality constraints
MIN_GRID_SIZE = 4
MAX_GRID_SIZE = 128
```

---

## 🎯 Performance Benchmarks

**Performance Comparison (CPU vs GPU):**

| Image Size | Grid | NumPy (CPU) | PyTorch (GPU) | Speedup |
|------------|------|-------------|---------------|---------|  
| 512×512    | 16×16| 0.85s       | 0.12s        | 7.1×    |
| 512×512    | 32×32| 1.42s       | 0.18s        | 7.9×    |
| 1024×1024  | 32×32| 3.21s       | 0.35s        | 9.2×    |
| 1024×1024  | 64×64| 5.67s       | 0.58s        | 9.8×    |

**Hardware:** NVIDIA RTX 3080 (10GB), Intel i9-10900K

**Optimizations Applied:**
- ✅ GPU-accelerated tensor operations
- ✅ Batched distance computations with `torch.cdist`
- ✅ Pre-cached tiles on GPU memory
- ✅ FP16 precision for memory efficiency
- ✅ Vectorized color matching
- ✅ Zero CPU-GPU memory transfers during processing

---

## 🏛️ Architecture Highlights

### Separation of Concerns

Each module has a specific responsibility:
- `config.py`: Configuration and constants
- `utils.py`: Shared utility functions
- `image_processor.py`: Image I/O and preprocessing
- `tile_manager.py`: Tile database management
- `mosaic_builder.py`: Core mosaic algorithm
- `metrics.py`: Quality evaluation

### Error Handling

Comprehensive validation at every level:
```python
from mosaic_generator.config import validate_grid_size

try:
    validate_grid_size(256)  # Too large
except ValueError as e:
    print(e)  # "Grid size 256 is too large. Maximum recommended: 128"
```

### Type Hints

Full type annotations for better IDE support:
```python
def create_mosaic(
    image: Union[np.ndarray, Tensor],
    grid_size: Optional[int] = None,
    tile_retrieval: str = "nearest",
    return_tensor: bool = False
) -> Union[np.ndarray, Tensor]:
    ...
```

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **Professor Lino Coria Mendoza** - CS5130 Applied Programming and Data Processing for AI
- Northeastern University
- [pytorch-msssim](https://github.com/VainF/pytorch-msssim) for quality metrics
- Gradio team for the web framework

---

## 📧 Contact

**Student:** Pratham Satani  
**Email:** [satani.p@northeastern.edu](mailto:satani.p@northeastern.edu)  
**GitHub:** [https://github.com/prathamsatani](https://github.com/prathamsatani)  
**Project Repository:** [https://github.com/prathamsatani/image-to-mosaic-app](https://github.com/prathamsatani/image-to-mosaic-app)

---

## 📚 Course Information

**Course:** CS5130 - Applied Programming and Data Processing for AI  
**Institution:** Northeastern University  
**Professor:** Dr. Lino Coria Mendoza  
**Semester:** Fall 2025  
**Assignment:** Lab 5 - Optimized Image to Mosaic Generator (Modular Architecture)

---

**Made with ❤️ using Python, PyTorch, and Gradio**

