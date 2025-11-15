import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import time
import tracemalloc
import json
import torch
from typing import Dict, List, Tuple
import logging

# Import the implementations
# Adjust the import paths according to your project structure
from utils.mosaic_generator import VectorizedMosaicGenerator, MosaicGenerator, OptimizedMosaicGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FourWayPerformanceTester:
    """
    Run performance tests comparing MosaicGenerator (baseline), VectorizedMosaicGenerator,
    PyTorchMosaicGenerator(CPU), and PyTorchMosaicGenerator(GPU) across different configurations.
    """
    
    def __init__(self, test_image_path: str = None, test_image_size: Tuple[int, int] = (1024, 1024)):
        """
        Initialize the performance tester with all implementations.
        
        Args:
            test_image_path: Path to test image. If None, generates a synthetic test image.
            test_image_size: Size for the test image (default 1024x1024)
        """
        # Initialize baseline and NumPy implementations
        self.loop_gen = MosaicGenerator()
        self.vectorized_gen = VectorizedMosaicGenerator()
        
        # Initialize PyTorch CPU implementation
        try:
            self.pytorch_cpu_gen = OptimizedMosaicGenerator(
                device='cpu',  # Force CPU
                use_half_precision=False  # FP32 on CPU
            )
            self.pytorch_cpu_available = True
            logger.info("PyTorch CPU initialized successfully")
        except Exception as e:
            logger.warning(f"PyTorch CPU initialization failed: {e}")
            self.pytorch_cpu_gen = None
            self.pytorch_cpu_available = False
        
        # Initialize PyTorch GPU implementation if available
        self.pytorch_gpu_available = False
        self.pytorch_gpu_gen = None
        self.gpu_info = None
        
        if torch.cuda.is_available():
            try:
                self.pytorch_gpu_gen = OptimizedMosaicGenerator(
                    device='cuda',  # Force GPU
                    use_half_precision=True  # FP16 on GPU for efficiency
                )
                
                # Get GPU information
                gpu_stats = self.pytorch_gpu_gen.get_performance_stats()
                self.gpu_info = {
                    'name': gpu_stats.get('gpu_name', 'Unknown GPU'),
                    'memory_gb': gpu_stats.get('gpu_memory_total_gb', 0),
                    'device': 'cuda:0'
                }
                
                self.pytorch_gpu_available = True
                logger.info(f"PyTorch GPU initialized: {self.gpu_info['name']}")
                
            except Exception as e:
                logger.warning(f"PyTorch GPU initialization failed: {e}")
                self.pytorch_gpu_gen = None
                
        # Load or generate test image
        if test_image_path and os.path.exists(test_image_path):
            self.test_image = cv2.imread(test_image_path)
            self.test_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2RGB)
        else:
            logger.info(f"Generating synthetic test image {test_image_size}")
            self.test_image = self._generate_test_image(test_image_size[0], test_image_size[1])
        
        # Ensure image is correct size
        if self.test_image.shape[:2] != test_image_size:
            self.test_image = cv2.resize(self.test_image, (test_image_size[1], test_image_size[0]))
        
        self.results = {}
        
    def _generate_test_image(self, height: int, width: int) -> np.ndarray:
        """Generate a synthetic test image with varied colors and patterns."""
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create interesting patterns
        for i in range(height):
            for j in range(width):
                r = int(128 + 127 * np.sin(i * 0.02))
                g = int(128 + 127 * np.cos(j * 0.02))
                b = int(128 + 127 * np.sin((i + j) * 0.01))
                image[i, j] = [r, g, b]
        
        # Add some geometric shapes for visual interest
        cv2.circle(image, (width//3, height//3), min(width, height)//6, (255, 200, 100), -1)
        cv2.rectangle(image, (width//2, height//2), 
                     (3*width//4, 3*height//4), (100, 200, 255), -1)
        cv2.ellipse(image, (2*width//3, 2*height//3), 
                   (width//8, height//6), 45, 0, 360, (200, 100, 255), -1)
        
        return image
    
    def measure_performance(self, func, *args, warmup_runs: int = 2, test_runs: int = 5, 
                           sync_cuda: bool = False, **kwargs) -> Tuple[any, float, float, float]:
        """
        Measure time and memory usage of a function call with warmup and multiple runs.
        
        Args:
            func: Function to test
            warmup_runs: Number of warmup runs (not measured)
            test_runs: Number of test runs to average
            sync_cuda: Whether to synchronize CUDA before/after timing
            
        Returns:
            Tuple of (function result, avg execution time, std execution time, peak memory in MB)
        """
        # Warmup runs (important for PyTorch JIT compilation)
        for _ in range(warmup_runs):
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            _ = func(*args, **kwargs)
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
        
        times = []
        result = None
        
        # Measure memory
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]
        
        # Run tests
        for _ in range(test_runs):
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
                
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
                
            execution_time = time.perf_counter() - start_time
            times.append(execution_time)
        
        # Get peak memory
        peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        
        memory_used_mb = (peak_memory - initial_memory) / (1024 * 1024)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        return result, avg_time, std_time, memory_used_mb
    
    def run_implementation_comparison(self) -> Dict:
        """
        Compare all four implementations across multiple grid sizes.
        """
        logger.info("Starting Four-Way Implementation Comparison...")
        
        grid_sizes = [16, 32, 64]
        
        # Build list of implementations to test
        implementations = ['Loop-based (Baseline)', 'NumPy Vectorized']
        if self.pytorch_cpu_available:
            implementations.append('PyTorch (CPU)')
        if self.pytorch_gpu_available:
            implementations.append('PyTorch (GPU)')
        
        results = {
            'grid_sizes': grid_sizes,
            'implementations': implementations,
            'processing_times': {impl: [] for impl in implementations},
            'processing_stds': {impl: [] for impl in implementations},
            'memory_usage': {impl: [] for impl in implementations},
            'speedups': {impl: [] for impl in implementations},  # Relative to loop-based
            'gpu_info': self.gpu_info if self.pytorch_gpu_available else None
        }
        
        for grid_size in grid_sizes:
            logger.info(f"\nTesting {grid_size}x{grid_size} grid...")
            
            # Test 1: Loop-based implementation (Baseline)
            logger.info("  Testing Loop-based (Baseline) implementation...")
            try:
                _, time_loop, std_loop, memory_loop = self.measure_performance(
                    self.loop_gen.create_mosaic,
                    self.test_image, grid_size,
                    warmup_runs=1, test_runs=3  # Fewer runs for slow implementation
                )
                results['processing_times']['Loop-based (Baseline)'].append(time_loop)
                results['processing_stds']['Loop-based (Baseline)'].append(std_loop)
                results['memory_usage']['Loop-based (Baseline)'].append(memory_loop)
                results['speedups']['Loop-based (Baseline)'].append(1.0)  # Baseline
                logger.info(f"    Time: {time_loop:.3f}±{std_loop:.3f}s, Memory: {memory_loop:.1f} MB")
            except Exception as e:
                logger.warning(f"    Loop-based test failed: {e}")
                results['processing_times']['Loop-based (Baseline)'].append(0)
                results['processing_stds']['Loop-based (Baseline)'].append(0)
                results['memory_usage']['Loop-based (Baseline)'].append(0)
                results['speedups']['Loop-based (Baseline)'].append(1.0)
                time_loop = 1.0  # Default for speedup calculation
            
            # Test 2: NumPy vectorized implementation
            logger.info("  Testing NumPy Vectorized implementation...")
            try:
                _, time_vec, std_vec, memory_vec = self.measure_performance(
                    self.vectorized_gen.create_mosaic,
                    self.test_image, grid_size, "nearest",
                    warmup_runs=2, test_runs=5
                )
                speedup_vec = time_loop / time_vec if time_vec > 0 else 0
                results['processing_times']['NumPy Vectorized'].append(time_vec)
                results['processing_stds']['NumPy Vectorized'].append(std_vec)
                results['memory_usage']['NumPy Vectorized'].append(memory_vec)
                results['speedups']['NumPy Vectorized'].append(speedup_vec)
                logger.info(f"    Time: {time_vec:.3f}±{std_vec:.3f}s, Memory: {memory_vec:.1f} MB, Speedup: {speedup_vec:.1f}×")
            except Exception as e:
                logger.warning(f"    NumPy vectorized test failed: {e}")
                results['processing_times']['NumPy Vectorized'].append(0)
                results['processing_stds']['NumPy Vectorized'].append(0)
                results['memory_usage']['NumPy Vectorized'].append(0)
                results['speedups']['NumPy Vectorized'].append(0)
            
            # Test 3: PyTorch CPU implementation
            if self.pytorch_cpu_available:
                logger.info("  Testing PyTorch (CPU) implementation...")
                try:
                    _, time_pytorch_cpu, std_pytorch_cpu, memory_pytorch_cpu = self.measure_performance(
                        self.pytorch_cpu_gen.create_mosaic,
                        self.test_image, grid_size, "nearest",
                        warmup_runs=3, test_runs=5,  # Extra warmup for JIT
                        blend_alpha=0.5,
                        return_tensor=False
                    )
                    speedup_pytorch_cpu = time_loop / time_pytorch_cpu if time_pytorch_cpu > 0 else 0
                    results['processing_times']['PyTorch (CPU)'].append(time_pytorch_cpu)
                    results['processing_stds']['PyTorch (CPU)'].append(std_pytorch_cpu)
                    results['memory_usage']['PyTorch (CPU)'].append(memory_pytorch_cpu)
                    results['speedups']['PyTorch (CPU)'].append(speedup_pytorch_cpu)
                    logger.info(f"    Time: {time_pytorch_cpu:.3f}±{std_pytorch_cpu:.3f}s, Memory: {memory_pytorch_cpu:.1f} MB, Speedup: {speedup_pytorch_cpu:.1f}×")
                except Exception as e:
                    logger.warning(f"    PyTorch CPU test failed: {e}")
                    results['processing_times']['PyTorch (CPU)'].append(0)
                    results['processing_stds']['PyTorch (CPU)'].append(0)
                    results['memory_usage']['PyTorch (CPU)'].append(0)
                    results['speedups']['PyTorch (CPU)'].append(0)
            
            # Test 4: PyTorch GPU implementation
            if self.pytorch_gpu_available:
                logger.info(f"  Testing PyTorch (GPU) implementation on {self.gpu_info['name']}...")
                try:
                    _, time_pytorch_gpu, std_pytorch_gpu, memory_pytorch_gpu = self.measure_performance(
                        self.pytorch_gpu_gen.create_mosaic,
                        self.test_image, grid_size, "nearest",
                        warmup_runs=3, test_runs=5,
                        sync_cuda=True,  # Important for accurate GPU timing
                        blend_alpha=0.5,
                        return_tensor=False
                    )
                    speedup_pytorch_gpu = time_loop / time_pytorch_gpu if time_pytorch_gpu > 0 else 0
                    results['processing_times']['PyTorch (GPU)'].append(time_pytorch_gpu)
                    results['processing_stds']['PyTorch (GPU)'].append(std_pytorch_gpu)
                    results['memory_usage']['PyTorch (GPU)'].append(memory_pytorch_gpu)
                    results['speedups']['PyTorch (GPU)'].append(speedup_pytorch_gpu)
                    logger.info(f"    Time: {time_pytorch_gpu:.3f}±{std_pytorch_gpu:.3f}s, Memory: {memory_pytorch_gpu:.1f} MB, Speedup: {speedup_pytorch_gpu:.1f}×")
                except Exception as e:
                    logger.warning(f"    PyTorch GPU test failed: {e}")
                    results['processing_times']['PyTorch (GPU)'].append(0)
                    results['processing_stds']['PyTorch (GPU)'].append(0)
                    results['memory_usage']['PyTorch (GPU)'].append(0)
                    results['speedups']['PyTorch (GPU)'].append(0)
        
        self.results['implementation_comparison'] = results
        return results
    
    def run_scaling_analysis(self) -> Dict:
        """
        Analyze how each implementation scales with image size.
        """
        logger.info("\nStarting Scaling Analysis...")
        
        image_sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
        grid_size = 32  # Fixed grid size for scaling analysis
        
        # Build list of implementations
        implementations = ['Loop-based (Baseline)', 'NumPy Vectorized']
        if self.pytorch_cpu_available:
            implementations.append('PyTorch (CPU)')
        if self.pytorch_gpu_available:
            implementations.append('PyTorch (GPU)')
        
        results = {
            'image_sizes': image_sizes,
            'total_pixels': [w*h for w, h in image_sizes],
            'implementations': implementations,
            'processing_times': {impl: [] for impl in implementations},
            'processing_stds': {impl: [] for impl in implementations},
            'memory_usage': {impl: [] for impl in implementations}
        }
        
        for img_size in image_sizes:
            logger.info(f"\nTesting {img_size[0]}x{img_size[1]} image...")
            
            # Generate test image of specific size
            test_img = cv2.resize(self.test_image, img_size)
            
            # Test loop-based
            logger.info("  Testing Loop-based...")
            try:
                _, time_loop, std_loop, memory_loop = self.measure_performance(
                    self.loop_gen.create_mosaic,
                    test_img, grid_size,
                    warmup_runs=1, test_runs=2  # Fewer runs for large images
                )
                results['processing_times']['Loop-based (Baseline)'].append(time_loop)
                results['processing_stds']['Loop-based (Baseline)'].append(std_loop)
                results['memory_usage']['Loop-based (Baseline)'].append(memory_loop)
            except Exception as e:
                logger.warning(f"    Failed: {e}")
                results['processing_times']['Loop-based (Baseline)'].append(0)
                results['processing_stds']['Loop-based (Baseline)'].append(0)
                results['memory_usage']['Loop-based (Baseline)'].append(0)
            
            # Test NumPy vectorized
            logger.info("  Testing NumPy Vectorized...")
            try:
                _, time_vec, std_vec, memory_vec = self.measure_performance(
                    self.vectorized_gen.create_mosaic,
                    test_img, grid_size, "nearest",
                    warmup_runs=2, test_runs=3
                )
                results['processing_times']['NumPy Vectorized'].append(time_vec)
                results['processing_stds']['NumPy Vectorized'].append(std_vec)
                results['memory_usage']['NumPy Vectorized'].append(memory_vec)
            except Exception as e:
                logger.warning(f"    Failed: {e}")
                results['processing_times']['NumPy Vectorized'].append(0)
                results['processing_stds']['NumPy Vectorized'].append(0)
                results['memory_usage']['NumPy Vectorized'].append(0)
            
            # Test PyTorch CPU
            if self.pytorch_cpu_available:
                logger.info("  Testing PyTorch CPU...")
                try:
                    _, time_pytorch_cpu, std_pytorch_cpu, memory_pytorch_cpu = self.measure_performance(
                        self.pytorch_cpu_gen.create_mosaic,
                        test_img, grid_size, "nearest",
                        warmup_runs=3, test_runs=3,
                        blend_alpha=0.5,
                        return_tensor=False
                    )
                    results['processing_times']['PyTorch (CPU)'].append(time_pytorch_cpu)
                    results['processing_stds']['PyTorch (CPU)'].append(std_pytorch_cpu)
                    results['memory_usage']['PyTorch (CPU)'].append(memory_pytorch_cpu)
                except Exception as e:
                    logger.warning(f"    Failed: {e}")
                    results['processing_times']['PyTorch (CPU)'].append(0)
                    results['processing_stds']['PyTorch (CPU)'].append(0)
                    results['memory_usage']['PyTorch (CPU)'].append(0)
            
            # Test PyTorch GPU
            if self.pytorch_gpu_available:
                logger.info("  Testing PyTorch GPU...")
                try:
                    _, time_pytorch_gpu, std_pytorch_gpu, memory_pytorch_gpu = self.measure_performance(
                        self.pytorch_gpu_gen.create_mosaic,
                        test_img, grid_size, "nearest",
                        warmup_runs=3, test_runs=3,
                        sync_cuda=True,
                        blend_alpha=0.5,
                        return_tensor=False
                    )
                    results['processing_times']['PyTorch (GPU)'].append(time_pytorch_gpu)
                    results['processing_stds']['PyTorch (GPU)'].append(std_pytorch_gpu)
                    results['memory_usage']['PyTorch (GPU)'].append(memory_pytorch_gpu)
                except Exception as e:
                    logger.warning(f"    Failed: {e}")
                    results['processing_times']['PyTorch (GPU)'].append(0)
                    results['processing_stds']['PyTorch (GPU)'].append(0)
                    results['memory_usage']['PyTorch (GPU)'].append(0)
        
        self.results['scaling_analysis'] = results
        return results
    
    def save_results(self, filename: str = 'four_way_test_results.json'):
        """Save test results to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        serializable_results = convert_to_serializable(self.results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Results saved to {filename}")
    
    def run_all_tests(self):
        """Run all performance tests."""
        print("\n" + "="*70)
        print("FOUR-WAY MOSAIC GENERATOR PERFORMANCE TESTING")
        print("="*70)
        print("\nImplementations being tested:")
        print("  1. Loop-based (Baseline)")
        print("  2. NumPy Vectorized")
        if self.pytorch_cpu_available:
            print("  3. PyTorch (CPU)")
        if self.pytorch_gpu_available:
            print(f"  4. PyTorch (GPU) - {self.gpu_info['name']}")
        print("="*70 + "\n")
        
        self.run_implementation_comparison()
        self.run_scaling_analysis()
        
        self.save_results()
        
        print("\n" + "="*70)
        print("TESTING COMPLETE")
        print("="*70 + "\n")
        
        return self.results


class FourWayReportVisualizer:
    """Generate publication-ready visualizations comparing all four implementations."""
    
    def __init__(self, results: Dict = None, results_file: str = None):
        """Initialize visualizer with test results."""
        if results:
            self.results = results
        elif results_file and os.path.exists(results_file):
            with open(results_file, 'r') as f:
                self.results = json.load(f)
        else:
            raise ValueError("Must provide either results dict or valid results_file path")
        
        # Define color scheme for four implementations
        self.colors = {
            'Loop-based (Baseline)': '#C73E1D',  # Red
            'NumPy Vectorized': '#2E86AB',       # Blue
            'PyTorch (CPU)': '#A23B72',          # Purple
            'PyTorch (GPU)': '#F18F01',          # Orange
            'primary': '#2E86AB',
            'secondary': '#F18F01',
            'accent': '#A23B72'
        }
        
        # Check which implementations are available
        self.implementations = self.results.get('implementation_comparison', {}).get('implementations', [])
    
    def create_implementation_comparison_figure(self):
        """Create Figure 1: Four-way implementation comparison across grid sizes."""
        data = self.results['implementation_comparison']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Figure 1: Four-Way Implementation Performance Comparison', 
                     fontsize=16, fontweight='bold')
        
        grid_sizes = data['grid_sizes']
        implementations = data['implementations']
        x = np.arange(len(grid_sizes))
        width = 0.2  # Narrower bars for 4 implementations
        
        # Processing Time Comparison
        ax = axes[0, 0]
        for i, impl in enumerate(implementations):
            times = data['processing_times'][impl]
            stds = data['processing_stds'][impl]
            offset = (i - len(implementations)/2 + 0.5) * width
            bars = ax.bar(x + offset, times, width, 
                         label=impl, color=self.colors[impl], alpha=0.8)
            
            # Add error bars
            ax.errorbar(x + offset, times, yerr=stds, 
                       fmt='none', color='black', capsize=3, alpha=0.5)
            
            # Add value labels
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=7, rotation=90 if height > 1 else 0)
        
        ax.set_xlabel('Grid Size', fontsize=11)
        ax.set_ylabel('Processing Time (seconds)', fontsize=11)
        ax.set_title('Processing Time by Grid Size', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{g}×{g}' for g in grid_sizes])
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for better visibility
        
        # Memory Usage Comparison
        ax = axes[0, 1]
        for i, impl in enumerate(implementations):
            memories = data['memory_usage'][impl]
            offset = (i - len(implementations)/2 + 0.5) * width
            bars = ax.bar(x + offset, memories, width,
                         label=impl, color=self.colors[impl], alpha=0.8)
            
            # Add value labels
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.0f}', ha='center', va='bottom', fontsize=7)
        
        ax.set_xlabel('Grid Size', fontsize=11)
        ax.set_ylabel('Memory Usage (MB)', fontsize=11)
        ax.set_title('Memory Usage by Grid Size', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{g}×{g}' for g in grid_sizes])
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Speedup Comparison
        ax = axes[0, 2]
        for impl in implementations[1:]:  # Skip baseline
            speedups = data['speedups'][impl]
            ax.plot(grid_sizes, speedups, marker='o', linewidth=2, 
                   markersize=8, label=impl, color=self.colors[impl])
            
            # Add value labels
            for i, (x_val, y_val) in enumerate(zip(grid_sizes, speedups)):
                if y_val > 0:
                    ax.annotate(f'{y_val:.1f}×', (x_val, y_val), 
                              textcoords="offset points", xytext=(0,5), 
                              ha='center', fontsize=8)
        
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        ax.set_xlabel('Grid Size', fontsize=11)
        ax.set_ylabel('Speedup (vs Loop-based)', fontsize=11)
        ax.set_title('Speedup Factors', fontsize=12, fontweight='bold')
        ax.set_xticks(grid_sizes)
        ax.set_xticklabels([f'{g}×{g}' for g in grid_sizes])
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for large speedups
        
        # Scaling Analysis
        if 'scaling_analysis' in self.results:
            scaling_data = self.results['scaling_analysis']
            
            # Time vs Image Size
            ax = axes[1, 0]
            image_labels = [f'{w}×{h}' for w, h in scaling_data['image_sizes']]
            
            for impl in scaling_data['implementations']:
                times = scaling_data['processing_times'][impl]
                ax.plot(range(len(image_labels)), times, marker='o', 
                       linewidth=2, markersize=8, label=impl, color=self.colors[impl])
            
            ax.set_xlabel('Image Size', fontsize=11)
            ax.set_ylabel('Processing Time (seconds)', fontsize=11)
            ax.set_title('Scaling with Image Size (32×32 grid)', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(image_labels)))
            ax.set_xticklabels(image_labels, rotation=45)
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            # Memory Scaling
            ax = axes[1, 1]
            for impl in scaling_data['implementations']:
                memories = scaling_data['memory_usage'][impl]
                ax.plot(range(len(image_labels)), memories, marker='s', 
                       linewidth=2, markersize=8, label=impl, color=self.colors[impl])
            
            ax.set_xlabel('Image Size', fontsize=11)
            ax.set_ylabel('Memory Usage (MB)', fontsize=11)
            ax.set_title('Memory Scaling with Image Size', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(image_labels)))
            ax.set_xticklabels(image_labels, rotation=45)
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Performance Summary
            ax = axes[1, 2]
            
            # Calculate average speedups
            avg_speedups = {}
            for impl in implementations[1:]:
                speedups = data['speedups'][impl]
                avg_speedups[impl] = np.mean([s for s in speedups if s > 0])
            
            summary_text = "Performance Summary\n" + "="*30 + "\n\n"
            
            if data.get('gpu_info'):
                summary_text += f"GPU: {data['gpu_info']['name']}\n"
                summary_text += f"Memory: {data['gpu_info']['memory_gb']:.1f} GB\n\n"
            
            summary_text += "Average Speedups (vs Baseline):\n"
            for impl, speedup in avg_speedups.items():
                summary_text += f"  {impl}: {speedup:.1f}×\n"
            
            # Add PyTorch CPU vs GPU comparison if both available
            if 'PyTorch (CPU)' in avg_speedups and 'PyTorch (GPU)' in avg_speedups:
                gpu_vs_cpu = avg_speedups['PyTorch (GPU)'] / avg_speedups['PyTorch (CPU)']
                summary_text += f"\nGPU vs CPU: {gpu_vs_cpu:.1f}× faster\n"
            
            summary_text += "\nBest Performance (32×32 grid):\n"
            best_time_idx = 1  # 32x32 grid index
            valid_impls = [impl for impl in implementations 
                          if data['processing_times'][impl][best_time_idx] > 0]
            if valid_impls:
                best_impl = min(valid_impls, 
                              key=lambda x: data['processing_times'][x][best_time_idx])
                best_time = data['processing_times'][best_impl][best_time_idx]
                summary_text += f"  {best_impl}: {best_time:.4f}s\n"
            
            ax.text(0.05, 0.5, summary_text, fontsize=10, verticalalignment='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('figure1_four_way_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Figure 1 saved as 'figure1_four_way_comparison.png'")
    
    def create_summary_table_figure(self):
        """Create Figure 2: Comprehensive summary table."""
        fig, ax = plt.subplots(figsize=(18, 12))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare summary data
        summary_data = []
        summary_data.append(['Implementation', '16×16 Grid', '32×32 Grid', '64×64 Grid', 'Avg Speedup', 'Memory (Avg)', 'Device'])
        summary_data.append(['='*20, '='*15, '='*15, '='*15, '='*12, '='*15, '='*10])
        
        impl_data = self.results['implementation_comparison']
        
        for impl in impl_data['implementations']:
            row = [impl]
            
            # Add timing for each grid size
            for i, grid_size in enumerate(impl_data['grid_sizes']):
                time = impl_data['processing_times'][impl][i]
                std = impl_data['processing_stds'][impl][i]
                if time > 0:
                    row.append(f'{time:.4f}±{std:.4f}s')
                else:
                    row.append('N/A')
            
            # Add average speedup
            speedups = impl_data['speedups'][impl]
            avg_speedup = np.mean([s for s in speedups if s > 0])
            if impl == 'Loop-based (Baseline)':
                row.append('1.0× (base)')
            else:
                row.append(f'{avg_speedup:.1f}×')
            
            # Add average memory
            avg_memory = np.mean([m for m in impl_data['memory_usage'][impl] if m > 0])
            row.append(f'{avg_memory:.0f} MB')
            
            # Add device info
            if 'GPU' in impl:
                row.append('CUDA')
            elif 'CPU' in impl:
                row.append('CPU')
            else:
                row.append('CPU')
            
            summary_data.append(row)
        
        # Add scaling analysis if available
        if 'scaling_analysis' in self.results:
            summary_data.append(['', '', '', '', '', '', ''])
            summary_data.append(['SCALING ANALYSIS (32×32 grid)', '', '', '', '', '', ''])
            
            # Create a simplified header row for scaling analysis
            scaling_header = ['Image Size', '256×256', '512×512', '1024×1024', '2048×2048', '', '']
            summary_data.append(scaling_header)
            
            scaling_data = self.results['scaling_analysis']
            # Add rows for each implementation
            for impl in scaling_data['implementations']:
                row = [impl]
                for i, size in enumerate(scaling_data['image_sizes']):
                    time = scaling_data['processing_times'][impl][i]
                    row.append(f'{time:.3f}s' if time > 0 else 'N/A')
                while len(row) < 7:
                    row.append('')
                summary_data.append(row)
        
        # Add key findings
        summary_data.append(['', '', '', '', '', '', ''])
        summary_data.append(['KEY FINDINGS', '', '', '', '', '', ''])
        
        # Find best performer
        valid_impls = [impl for impl in impl_data['implementations']
                      if any(t > 0 for t in impl_data['processing_times'][impl])]
        if valid_impls:
            best_impl = min(valid_impls, 
                          key=lambda x: np.mean([t for t in impl_data['processing_times'][x] if t > 0]))
            summary_data.append(['Best Overall', best_impl, '', '', '', '', ''])
        
        # Add GPU advantage if available
        if 'PyTorch (GPU)' in impl_data['implementations']:
            gpu_speedups = [s for s in impl_data['speedups']['PyTorch (GPU)'] if s > 0]
            if gpu_speedups:
                max_speedup = max(gpu_speedups)
                summary_data.append(['Max GPU Speedup', f'{max_speedup:.1f}× vs baseline', '', '', '', '', ''])
        
        # Create table
        table = ax.table(cellText=summary_data, cellLoc='left', loc='center',
                        colWidths=[0.22, 0.15, 0.15, 0.15, 0.11, 0.11, 0.11])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        
        # Color coding
        for i in range(len(summary_data)):
            if i == 0:  # Header
                for j in range(7):
                    table[(i, j)].set_facecolor('#2E86AB')
                    table[(i, j)].set_text_props(weight='bold', color='white')
            elif 'ANALYSIS' in str(summary_data[i][0]) or 'FINDINGS' in str(summary_data[i][0]):
                for j in range(7):
                    table[(i, j)].set_facecolor('#E8F4F8')
                    table[(i, j)].set_text_props(weight='bold')
            elif i > 2 and i < len(summary_data) - 4:  # Data rows
                # Highlight GPU speedups > 100×
                if '×' in str(summary_data[i][4]) and summary_data[i][4] != '1.0× (base)':
                    try:
                        speedup_str = summary_data[i][4].replace('×', '')
                        speedup = float(speedup_str)
                        if speedup > 100:
                            table[(i, 4)].set_facecolor('#90EE90')
                        elif speedup > 10:
                            table[(i, 4)].set_facecolor('#FFFACD')
                    except:
                        pass
        
        plt.title('Table 1: Four-Way Mosaic Generator Performance Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.savefig('figure2_summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Figure 2 saved as 'figure2_summary_table.png'")
    
    def generate_all_figures(self):
        """Generate all figures from test results."""
        print("\nGenerating visualizations from test results...")
        print("-" * 40)
        
        self.create_implementation_comparison_figure()
        self.create_summary_table_figure()
        
        print("-" * 40)
        print("All visualizations generated successfully!")


def main():
    """Main function to run four-way comparison tests and generate visualizations."""
    print("="*70)
    print("FOUR-WAY MOSAIC GENERATOR PERFORMANCE ANALYSIS")
    print("Comparing:")
    print("  1. Loop-based (Baseline)")
    print("  2. NumPy Vectorized")
    print("  3. PyTorch (CPU)")
    print("  4. PyTorch (GPU) - if available")
    print("="*70)
    
    # Step 1: Run performance tests
    print("\nStep 1: Running performance tests...")
    tester = FourWayPerformanceTester(test_image_size=(1024, 1024))
    results = tester.run_all_tests()
    
    # Step 2: Generate visualizations from results
    print("\nStep 2: Generating visualizations from results...")
    visualizer = FourWayReportVisualizer(results=results)
    visualizer.generate_all_figures()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - four_way_test_results.json (raw test data)")
    print("  - figure1_four_way_comparison.png")
    print("  - figure2_summary_table.png")
    
    # Print quick summary
    if 'implementation_comparison' in results:
        impl_comp = results['implementation_comparison']
        print("\nQuick Summary (32×32 grid on 1024×1024 image):")
        for impl in impl_comp['implementations']:
            if impl_comp['processing_times'][impl][1] > 0:  # Index 1 is 32x32 grid
                print(f"  {impl}: {impl_comp['processing_times'][impl][1]:.4f}s")
                if impl != 'Loop-based (Baseline)':
                    print(f"    Speedup: {impl_comp['speedups'][impl][1]:.1f}×")
    
    return results, visualizer


if __name__ == "__main__":
    try:
        results, visualizer = main()
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print("\nIf tests failed, you can load previous results:")
        print("  visualizer = FourWayReportVisualizer(results_file='four_way_test_results.json')")
        print("  visualizer.generate_all_figures()")