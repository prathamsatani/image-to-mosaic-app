---
title: image-to-mosaic-app
app_file: main.py
sdk: gradio
sdk_version: 5.43.1
---
# Image to Mosaic Generator

A high-performance Python application that transforms images into artistic mosaics using vectorized NumPy operations and a tile-matching algorithm. Built with Gradio for an intuitive web interface.

**Academic Project** - CS5130 Applied Programming and Data Processing for AI, Northeastern University

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-orange)
![NEU](https://img.shields.io/badge/NEU-CS5800-red)

## 🎨 Features

- **Vectorized Processing**: Fully optimized NumPy operations for fast mosaic generation
- **Multiple Grid Sizes**: Support for 16x16, 32x32, and 64x64 tile grids
- **Tile Selection Modes**: 
  - **Nearest Match**: Intelligently matches tiles based on average color similarity
  - **Random**: Creates artistic variations with random tile placement
- **Quality Assessment**: MS-SSIM (Multi-Scale Structural Similarity Index) scoring for mosaic quality evaluation
- **Reproducible Results**: Optional seed support for consistent random tile generation
- **Interactive Web UI**: Clean, responsive Gradio interface with real-time progress updates

## 📸 How It Works

1. **Image Chunking**: Divides the input image into a grid of equal-sized chunks
2. **Color Analysis**: Computes average RGB values for each chunk
3. **Tile Matching**: 
   - In "Nearest Match" mode: Finds tiles with the closest color match using Euclidean distance
   - In "Random" mode: Selects tiles randomly from the database
4. **Superimposition**: Blends selected tiles with original chunks using weighted averaging
5. **Reconstruction**: Stitches processed chunks back into a complete mosaic image

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-to-mosaic-generator.git
cd image-to-mosaic-generator
```

2. Install required dependencies:
```bash
pip install gradio numpy opencv-python pandas pytorch-msssim torch
```

3. Prepare your tile images:
   - Create an `images/` directory in the project root
   - Add your tile images (JPG or PNG format)
   - Run the preprocessing script:
```bash
python utils/preprocess_tiles.py
```

This will generate `tiles_metadata.csv` containing color statistics for all tiles.

## 💻 Usage

### Web Interface (Recommended)

Launch the Gradio interface:
```bash
python main.py
```

The web interface will open automatically in your browser (typically at `http://localhost:7860`).

### Programmatic Usage

```python
from utils.mosaic_generator import VectorizedMosaicGenerator
import cv2

# Initialize generator
generator = VectorizedMosaicGenerator()

# Load image
image = cv2.imread('path/to/image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create mosaic
mosaic = generator.create_mosaic(
    image_array=image,
    n_chunks=32,  # 32x32 grid
    tile_retrieval="nearest"  # or "random"
)

# Save result
cv2.imwrite('mosaic_output.jpg', cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))
```

## 📁 Project Structure

```
image-to-mosaic-generator/
│
├── main.py                      # Gradio web interface
├── tiles_metadata.csv           # Generated tile color database
├── utils/
│   ├── mosaic_generator.py     # Core mosaic generation logic
│   └── preprocess_tiles.py     # Tile preprocessing utility
├── images/                     # Directory for tile images
│   ├── tile1.jpg
│   ├── tile2.png
│   └── ...
└── README.md
```

## 🔧 Configuration Options

### Grid Sizes
- **16x16**: Coarse mosaic with larger tiles (faster processing)
- **32x32**: Balanced detail and performance (default)
- **64x64**: Fine-grained mosaic with smaller tiles (higher detail)

### Tile Selection Modes
- **Nearest Match**: Best for photorealistic results
  - Alpha blend: 0.5 (original) / 0.5 (tile)
- **Random Tiles**: Best for artistic effects
  - Alpha blend: 1.0 (original) / 0.3 (tile)

### Performance Optimization
The generator uses several optimization techniques:
- Vectorized NumPy operations for chunk processing
- Batch distance calculations for tile matching
- Efficient memory broadcasting for color averaging
- OpenCV's optimized resize algorithms

## 🎯 Use Cases

- **Digital Art Creation**: Transform photos into unique mosaic artwork
- **Data Visualization**: Create visual representations using themed tile sets
- **Image Compression Visualization**: Demonstrate chunking and reconstruction concepts
- **Educational Tool**: Teach image processing and computer vision concepts
- **Social Media Content**: Generate eye-catching mosaic effects for posts

## 📊 Technical Details

### Algorithms
- **Color Matching**: Euclidean distance in RGB color space
- **Image Quality**: MS-SSIM metric for perceptual similarity assessment
- **Chunk Processing**: Fully vectorized using NumPy's advanced indexing

### Dependencies
- `gradio`: Web interface framework
- `numpy`: Vectorized array operations
- `opencv-python`: Image processing and I/O
- `pandas`: Tile metadata management
- `pytorch-msssim`: MS-SSIM quality metric
- `torch`: Tensor operations for MS-SSIM

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Professor [Lino Coria Mendoza](https://scholar.google.com/citations?user=UYvHwn8AAAAJ&hl=en)** - CS5800 Algorithms, Northeastern University
- Lab assignment focusing on vectorized operations and algorithmic optimization
- Inspired by classical photomosaic techniques
- MS-SSIM implementation from [pytorch-msssim](https://github.com/VainF/pytorch-msssim)
- Gradio team for the excellent web framework

## 🎓 Academic Information

**Course**: CS5130 - Applied Programming and Data Processing for AI  
**Institution**: Northeastern University  
**Professor**: Dr. Lino Coria Mendoza  
**Semester**: Fall 2025  
**Assignment**: Lab Activity - Interactive Image Mosaic Generator Using Gradio

## 📧 Contact

**Student**: Pratham Satani  
**Email**: [satani.p@northeastern.edu](mailto:satani.p@northeastern.edu)  
**GitHub**: [https://github.com/prathamsatani](https://github.com/prathamsatani)

**Project Link**: [https://github.com/prathamsatani/image-to-mosaic-app](https://github.com/prathamsatani/image-to-mosaic-app)

---

**Note**: Ensure you have a diverse collection of tile images for best results. The quality and variety of your tile database directly impacts the final mosaic quality.