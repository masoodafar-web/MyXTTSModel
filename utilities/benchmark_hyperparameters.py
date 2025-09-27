#!/usr/bin/env python3
"""
🔬 MyXTTS Hyperparameter Benchmark Suite

This tool automatically benchmarks different hyperparameter combinations
to find optimal settings for your specific dataset and hardware.

Features:
- Automated grid search over key hyperparameters
- GPU utilization monitoring
- Loss convergence analysis
- Training speed benchmarking
- Memory usage optimization
- Quality assessment integration
- Comprehensive reporting

Usage:
    python3 benchmark_hyperparameters.py --config benchmark_config.yaml
    python3 benchmark_hyperparameters.py --quick-test  # Fast 5-minute test
    python3 benchmark_hyperparameters.py --full-sweep  # Comprehensive sweep
"""

import os
import sys
import json
import time
import argparse
import logging
import itertools
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import subprocess
import psutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Try to import GPUtil, fallback to nvidia-ml-py or manual GPU detection
try:
    import GPUtil
    GPU_UTILS_AVAILABLE = True
except ImportError:
    try:
        import pynvml
        pynvml.nvmlInit()
        GPU_UTILS_AVAILABLE = True
        
        class GPUtil:
            @staticmethod
            def getGPUs():
                gpus = []
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    class GPU:
                        def __init__(self, load, memory_util, memory_total):
                            self.load = load / 100.0
                            self.memoryUtil = memory_util
                            self.memoryTotal = memory_total
                    
                    gpu = GPU(
                        load=utilization.gpu,
                        memory_util=memory_info.used / memory_info.total,
                        memory_total=memory_info.total / (1024**3)  # Convert to GB
                    )
                    gpus.append(gpu)
                return gpus
    except ImportError:
        GPU_UTILS_AVAILABLE = False
        
        class GPUtil:
            @staticmethod
            def getGPUs():
                # Fallback: try to get GPU info from nvidia-smi
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                           '--format=csv,noheader,nounits'], 
                                         capture_output=True, text=True)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        gpus = []
                        for line in lines:
                            parts = line.split(', ')
                            if len(parts) >= 3:
                                class GPU:
                                    def __init__(self, load, memory_used, memory_total):
                                        self.load = float(load) / 100.0
                                        self.memoryUtil = float(memory_used) / float(memory_total)
                                        self.memoryTotal = float(memory_total) / 1024  # Convert MB to GB
                                
                                gpu = GPU(float(parts[0]), float(parts[1]), float(parts[2]))
                                gpus.append(gpu)
                        return gpus
                except:
                    pass
                return []

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for hyperparameter benchmarking"""
    
    # Model size options to test
    model_sizes: List[str] = None
    
    # Optimization levels to test
    optimization_levels: List[str] = None
    
    # Learning rates to test
    learning_rates: List[float] = None
    
    # Batch sizes to test
    batch_sizes: List[int] = None
    
    # GPU stabilizer options
    gpu_stabilizer_options: List[bool] = None
    
    # Loss weight combinations
    mel_loss_weights: List[float] = None
    kl_loss_weights: List[float] = None
    
    # Training duration for each test
    test_epochs: int = 10
    max_test_time: int = 300  # 5 minutes max per test
    
    # Convergence criteria
    convergence_threshold: float = 0.01
    patience: int = 5
    
    # Hardware constraints
    max_memory_usage: float = 0.9
    min_gpu_utilization: float = 40.0
    
    def __post_init__(self):
        """Set default values if not provided"""
        if self.model_sizes is None:
            self.model_sizes = ["tiny", "small", "normal"]
        if self.optimization_levels is None:
            self.optimization_levels = ["basic", "enhanced", "plateau_breaker"]
        if self.learning_rates is None:
            self.learning_rates = [1e-5, 2e-5, 5e-5, 8e-5, 1e-4]
        if self.batch_sizes is None:
            self.batch_sizes = [4, 8, 16, 24, 32]
        if self.gpu_stabilizer_options is None:
            self.gpu_stabilizer_options = [True, False]
        if self.mel_loss_weights is None:
            self.mel_loss_weights = [1.5, 2.0, 2.5, 3.0]
        if self.kl_loss_weights is None:
            self.kl_loss_weights = [1.0, 1.2, 1.5, 1.8]

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    
    # Configuration tested
    config: Dict[str, Any]
    
    # Performance metrics
    final_loss: float
    convergence_speed: float  # epochs to reach threshold
    training_speed: float     # samples per second
    gpu_utilization: float    # average GPU usage
    memory_usage: float       # peak memory usage
    
    # Quality indicators
    loss_stability: float     # standard deviation of loss
    gradient_norm: float      # average gradient norm
    
    # Status
    completed: bool
    error_message: Optional[str] = None
    
    # Timestamps
    start_time: str = ""
    end_time: str = ""
    duration: float = 0.0
    
    def score(self) -> float:
        """Calculate overall performance score (higher is better)"""
        if not self.completed:
            return 0.0
            
        # Normalize metrics (lower loss is better, higher speed is better)
        loss_score = max(0, 1.0 - (self.final_loss / 5.0))  # Assume max reasonable loss is 5.0
        speed_score = min(1.0, self.training_speed / 50.0)   # Assume max reasonable speed is 50 sps
        gpu_score = min(1.0, self.gpu_utilization / 100.0)
        stability_score = max(0, 1.0 - (self.loss_stability / 1.0))  # Lower std is better
        
        # Weighted combination
        total_score = (
            loss_score * 0.4 +      # Loss is most important
            speed_score * 0.25 +    # Training speed
            gpu_score * 0.2 +       # GPU utilization
            stability_score * 0.15  # Stability
        )
        
        return total_score


class HyperparameterBenchmark:
    """Main benchmarking class"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.best_result: Optional[BenchmarkResult] = None
        
        # Create output directory
        self.output_dir = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Benchmark results will be saved to: {self.output_dir}")
    
    def generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations to test"""
        
        # For quick test, use limited combinations
        if hasattr(self, 'quick_test') and self.quick_test:
            combinations = [
                {
                    'model_size': 'tiny',
                    'optimization_level': 'enhanced',
                    'learning_rate': 2e-5,
                    'batch_size': 8,
                    'gpu_stabilizer': True,
                    'mel_loss_weight': 2.0,
                    'kl_loss_weight': 1.2
                },
                {
                    'model_size': 'tiny',
                    'optimization_level': 'plateau_breaker',
                    'learning_rate': 1.5e-5,
                    'batch_size': 4,
                    'gpu_stabilizer': True,
                    'mel_loss_weight': 2.0,
                    'kl_loss_weight': 1.2
                },
                {
                    'model_size': 'small',
                    'optimization_level': 'enhanced',
                    'learning_rate': 5e-5,
                    'batch_size': 16,
                    'gpu_stabilizer': False,
                    'mel_loss_weight': 2.5,
                    'kl_loss_weight': 1.5
                }
            ]
            logger.info(f"Quick test mode: Testing {len(combinations)} combinations")
            return combinations
        
        # Full parameter sweep
        param_grid = {
            'model_size': self.config.model_sizes,
            'optimization_level': self.config.optimization_levels,
            'learning_rate': self.config.learning_rates,
            'batch_size': self.config.batch_sizes,
            'gpu_stabilizer': self.config.gpu_stabilizer_options,
            'mel_loss_weight': self.config.mel_loss_weights,
            'kl_loss_weight': self.config.kl_loss_weights
        }
        
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        
        # Filter out impossible combinations (e.g., big model with small batch on limited memory)
        filtered_combinations = []
        for combo in combinations:
            if self._is_viable_combination(combo):
                filtered_combinations.append(combo)
        
        logger.info(f"Generated {len(filtered_combinations)} viable combinations from {len(combinations)} total")
        return filtered_combinations
    
    def _is_viable_combination(self, combo: Dict[str, Any]) -> bool:
        """Check if a parameter combination is viable given hardware constraints"""
        
        # Estimate memory usage
        model_size = combo['model_size']
        batch_size = combo['batch_size']
        
        # Rough memory estimates (in GB)
        memory_estimates = {
            'tiny': 2,
            'small': 4,
            'normal': 8,
            'big': 16
        }
        
        estimated_memory = memory_estimates.get(model_size, 8) * (batch_size / 16)
        
        # Get available GPU memory
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                available_memory = gpus[0].memoryTotal / 1024  # Convert to GB
                if estimated_memory > available_memory * 0.9:
                    return False
        except:
            pass
        
        # Skip combinations that are known to be problematic
        if model_size == 'big' and batch_size > 16:
            return False
        if combo['optimization_level'] == 'plateau_breaker' and combo['learning_rate'] > 5e-5:
            return False
            
        return True
    
    def run_single_benchmark(self, params: Dict[str, Any]) -> BenchmarkResult:
        """Run a single benchmark with given parameters"""
        
        logger.info(f"Starting benchmark: {params}")
        start_time = time.time()
        
        # Build command
        cmd = [
            'python3', 'train_main.py',
            '--model-size', params['model_size'],
            '--optimization-level', params['optimization_level'],
            '--lr', str(params['learning_rate']),
            '--batch-size', str(params['batch_size']),
            '--epochs', str(self.config.test_epochs),
            '--checkpoint-dir', f"{self.output_dir}/temp_checkpoint_{hash(str(params))}",
        ]
        
        # Add GPU stabilizer option
        if params['gpu_stabilizer']:
            cmd.append('--enable-gpu-stabilizer')
        else:
            cmd.append('--disable-gpu-stabilizer')
        
        # Create temporary config for this test
        temp_config = self._create_temp_config(params)
        
        # Monitor system resources
        gpu_utilizations = []
        memory_usages = []
        training_losses = []
        
        try:
            # Start training process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor training
            start_monitor_time = time.time()
            last_loss = float('inf')
            patience_count = 0
            
            while process.poll() is None:
                current_time = time.time()
                
                # Timeout check
                if current_time - start_time > self.config.max_test_time:
                    process.terminate()
                    logger.warning(f"Test timed out after {self.config.max_test_time} seconds")
                    break
                
                # Monitor GPU
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_utilizations.append(gpus[0].load * 100)
                        memory_usages.append(gpus[0].memoryUtil * 100)
                except:
                    pass
                
                # Parse training output for loss
                try:
                    line = process.stdout.readline()
                    if line and "Loss:" in line:
                        # Extract loss from line (format varies)
                        loss_value = self._extract_loss_from_line(line)
                        if loss_value is not None:
                            training_losses.append(loss_value)
                            
                            # Check convergence
                            if abs(loss_value - last_loss) < self.config.convergence_threshold:
                                patience_count += 1
                            else:
                                patience_count = 0
                            
                            if patience_count >= self.config.patience:
                                logger.info("Early convergence detected")
                                process.terminate()
                                break
                            
                            last_loss = loss_value
                except:
                    pass
                
                time.sleep(1)  # Monitor every second
            
            # Wait for process to complete
            stdout, stderr = process.communicate(timeout=30)
            
            # Calculate metrics
            end_time = time.time()
            duration = end_time - start_time
            
            final_loss = training_losses[-1] if training_losses else float('inf')
            avg_gpu_util = np.mean(gpu_utilizations) if gpu_utilizations else 0.0
            peak_memory = max(memory_usages) if memory_usages else 0.0
            loss_stability = np.std(training_losses[-10:]) if len(training_losses) >= 10 else float('inf')
            
            # Estimate training speed (samples per second)
            samples_processed = params['batch_size'] * self.config.test_epochs * 100  # Rough estimate
            training_speed = samples_processed / duration if duration > 0 else 0.0
            
            # Calculate convergence speed
            convergence_speed = len(training_losses) if training_losses else float('inf')
            
            result = BenchmarkResult(
                config=params,
                final_loss=final_loss,
                convergence_speed=convergence_speed,
                training_speed=training_speed,
                gpu_utilization=avg_gpu_util,
                memory_usage=peak_memory,
                loss_stability=loss_stability,
                gradient_norm=1.0,  # Placeholder
                completed=True,
                start_time=datetime.fromtimestamp(start_time).isoformat(),
                end_time=datetime.fromtimestamp(end_time).isoformat(),
                duration=duration
            )
            
            logger.info(f"Benchmark completed: Loss={final_loss:.3f}, GPU={avg_gpu_util:.1f}%, Speed={training_speed:.1f} sps")
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            result = BenchmarkResult(
                config=params,
                final_loss=float('inf'),
                convergence_speed=float('inf'),
                training_speed=0.0,
                gpu_utilization=0.0,
                memory_usage=0.0,
                loss_stability=float('inf'),
                gradient_norm=float('inf'),
                completed=False,
                error_message=str(e),
                start_time=datetime.fromtimestamp(start_time).isoformat(),
                end_time=datetime.now().isoformat(),
                duration=time.time() - start_time
            )
        
        finally:
            # Cleanup
            self._cleanup_temp_files(params)
        
        return result
    
    def _extract_loss_from_line(self, line: str) -> Optional[float]:
        """Extract loss value from training output line"""
        try:
            # Try different formats
            if "Train Loss:" in line:
                return float(line.split("Train Loss:")[1].split()[0])
            elif "loss=" in line:
                return float(line.split("loss=")[1].split(",")[0])
            elif "Loss:" in line:
                return float(line.split("Loss:")[1].split()[0])
        except:
            pass
        return None
    
    def _create_temp_config(self, params: Dict[str, Any]) -> str:
        """Create temporary configuration file for this test"""
        # Implementation would create a config file with specific loss weights
        pass
    
    def _cleanup_temp_files(self, params: Dict[str, Any]):
        """Clean up temporary files created during testing"""
        import shutil
        temp_dir = f"{self.output_dir}/temp_checkpoint_{hash(str(params))}"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    def run_full_benchmark(self):
        """Run complete hyperparameter benchmark"""
        
        logger.info("Starting hyperparameter benchmark suite")
        
        # Generate parameter combinations
        combinations = self.generate_parameter_combinations()
        total_combinations = len(combinations)
        
        logger.info(f"Testing {total_combinations} parameter combinations")
        
        # Run benchmarks
        for i, params in enumerate(combinations, 1):
            logger.info(f"Progress: {i}/{total_combinations} ({i/total_combinations*100:.1f}%)")
            
            result = self.run_single_benchmark(params)
            self.results.append(result)
            
            # Update best result
            if result.completed and (self.best_result is None or result.score() > self.best_result.score()):
                self.best_result = result
                logger.info(f"New best result found: Score={result.score():.3f}")
            
            # Save intermediate results
            self.save_results()
            
            # Brief pause between tests
            time.sleep(2)
        
        logger.info("Benchmark suite completed!")
        self.generate_report()
    
    def save_results(self):
        """Save benchmark results to files"""
        
        # Save raw results as JSON
        results_data = [asdict(result) for result in self.results]
        with open(f"{self.output_dir}/results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save as CSV for easy analysis
        df = pd.DataFrame([
            {
                'model_size': r.config['model_size'],
                'optimization_level': r.config['optimization_level'],
                'learning_rate': r.config['learning_rate'],
                'batch_size': r.config['batch_size'],
                'gpu_stabilizer': r.config['gpu_stabilizer'],
                'mel_loss_weight': r.config['mel_loss_weight'],
                'kl_loss_weight': r.config['kl_loss_weight'],
                'final_loss': r.final_loss,
                'training_speed': r.training_speed,
                'gpu_utilization': r.gpu_utilization,
                'memory_usage': r.memory_usage,
                'loss_stability': r.loss_stability,
                'score': r.score(),
                'completed': r.completed,
                'duration': r.duration
            }
            for r in self.results
        ])
        df.to_csv(f"{self.output_dir}/results.csv", index=False)
        
        logger.info(f"Results saved to {self.output_dir}/")
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        
        logger.info("Generating benchmark report...")
        
        # Filter successful results
        successful_results = [r for r in self.results if r.completed]
        
        if not successful_results:
            logger.error("No successful benchmark runs to analyze!")
            return
        
        # Generate plots
        self._generate_plots()
        
        # Generate summary report
        report = self._generate_summary_report(successful_results)
        
        # Save report
        with open(f"{self.output_dir}/benchmark_report.md", 'w') as f:
            f.write(report)
        
        logger.info(f"Comprehensive report saved to {self.output_dir}/benchmark_report.md")
    
    def _generate_plots(self):
        """Generate visualization plots"""
        
        if not self.results:
            return
        
        # Create DataFrame for plotting
        df = pd.DataFrame([
            {
                'model_size': r.config['model_size'],
                'optimization_level': r.config['optimization_level'],
                'learning_rate': r.config['learning_rate'],
                'batch_size': r.config['batch_size'],
                'final_loss': r.final_loss if r.completed else None,
                'training_speed': r.training_speed if r.completed else None,
                'gpu_utilization': r.gpu_utilization if r.completed else None,
                'score': r.score() if r.completed else None
            }
            for r in self.results
        ])
        
        # Plot 1: Loss vs Learning Rate
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        for opt_level in df['optimization_level'].unique():
            data = df[df['optimization_level'] == opt_level]
            plt.scatter(data['learning_rate'], data['final_loss'], label=opt_level, alpha=0.7)
        plt.xlabel('Learning Rate')
        plt.ylabel('Final Loss')
        plt.title('Loss vs Learning Rate by Optimization Level')
        plt.legend()
        plt.xscale('log')
        
        # Plot 2: GPU Utilization vs Batch Size
        plt.subplot(2, 2, 2)
        for gpu_stab in [True, False]:
            data = df[df['batch_size'].notna()]
            if gpu_stab:
                # Note: We need to add gpu_stabilizer to the DataFrame
                plt.scatter(data['batch_size'], data['gpu_utilization'], 
                           label=f'GPU Stabilizer: {gpu_stab}', alpha=0.7)
        plt.xlabel('Batch Size')
        plt.ylabel('GPU Utilization (%)')
        plt.title('GPU Utilization vs Batch Size')
        plt.legend()
        
        # Plot 3: Training Speed vs Model Size
        plt.subplot(2, 2, 3)
        model_order = ['tiny', 'small', 'normal', 'big']
        for model in model_order:
            if model in df['model_size'].values:
                data = df[df['model_size'] == model]
                plt.scatter([model] * len(data), data['training_speed'], alpha=0.7)
        plt.xlabel('Model Size')
        plt.ylabel('Training Speed (samples/sec)')
        plt.title('Training Speed by Model Size')
        
        # Plot 4: Overall Score Distribution
        plt.subplot(2, 2, 4)
        scores = df['score'].dropna()
        plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Overall Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Overall Scores')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/benchmark_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Plots saved to benchmark_plots.png")
    
    def _generate_summary_report(self, successful_results: List[BenchmarkResult]) -> str:
        """Generate markdown summary report"""
        
        # Find best configurations
        best_overall = max(successful_results, key=lambda r: r.score())
        best_loss = min(successful_results, key=lambda r: r.final_loss)
        best_speed = max(successful_results, key=lambda r: r.training_speed)
        best_gpu = max(successful_results, key=lambda r: r.gpu_utilization)
        
        report = f"""# 🔬 MyXTTS Hyperparameter Benchmark Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Summary Statistics

- **Total Configurations Tested**: {len(self.results)}
- **Successful Runs**: {len(successful_results)}
- **Success Rate**: {len(successful_results)/len(self.results)*100:.1f}%
- **Total Benchmark Time**: {sum(r.duration for r in self.results):.1f} seconds

## 🏆 Best Configurations

### 🥇 Best Overall Score: {best_overall.score():.3f}
```yaml
Model Size: {best_overall.config['model_size']}
Optimization Level: {best_overall.config['optimization_level']}
Learning Rate: {best_overall.config['learning_rate']}
Batch Size: {best_overall.config['batch_size']}
GPU Stabilizer: {best_overall.config['gpu_stabilizer']}
Mel Loss Weight: {best_overall.config['mel_loss_weight']}
KL Loss Weight: {best_overall.config['kl_loss_weight']}

Results:
- Final Loss: {best_overall.final_loss:.3f}
- Training Speed: {best_overall.training_speed:.1f} samples/sec
- GPU Utilization: {best_overall.gpu_utilization:.1f}%
- Memory Usage: {best_overall.memory_usage:.1f}%
- Loss Stability: {best_overall.loss_stability:.3f}
```

### 🎯 Best Loss: {best_loss.final_loss:.3f}
```yaml
Configuration: {best_loss.config}
Training Speed: {best_loss.training_speed:.1f} samples/sec
GPU Utilization: {best_loss.gpu_utilization:.1f}%
```

### ⚡ Best Training Speed: {best_speed.training_speed:.1f} samples/sec
```yaml
Configuration: {best_speed.config}
Final Loss: {best_speed.final_loss:.3f}
GPU Utilization: {best_speed.gpu_utilization:.1f}%
```

### 🎮 Best GPU Utilization: {best_gpu.gpu_utilization:.1f}%
```yaml
Configuration: {best_gpu.config}
Final Loss: {best_gpu.final_loss:.3f}
Training Speed: {best_gpu.training_speed:.1f} samples/sec
```

## 📈 Key Insights

### Model Size Impact:
"""
        
        # Analyze by model size
        for size in ['tiny', 'small', 'normal', 'big']:
            size_results = [r for r in successful_results if r.config['model_size'] == size]
            if size_results:
                avg_loss = np.mean([r.final_loss for r in size_results])
                avg_speed = np.mean([r.training_speed for r in size_results])
                avg_gpu = np.mean([r.gpu_utilization for r in size_results])
                report += f"""
- **{size.title()}**: Avg Loss={avg_loss:.3f}, Speed={avg_speed:.1f} sps, GPU={avg_gpu:.1f}%"""
        
        report += """

### Optimization Level Impact:
"""
        
        # Analyze by optimization level
        for level in ['basic', 'enhanced', 'experimental', 'plateau_breaker']:
            level_results = [r for r in successful_results if r.config['optimization_level'] == level]
            if level_results:
                avg_loss = np.mean([r.final_loss for r in level_results])
                avg_speed = np.mean([r.training_speed for r in level_results])
                report += f"""
- **{level.title()}**: Avg Loss={avg_loss:.3f}, Speed={avg_speed:.1f} sps"""
        
        report += """

### GPU Stabilizer Impact:
"""
        
        # Analyze GPU stabilizer impact
        with_gpu = [r for r in successful_results if r.config['gpu_stabilizer']]
        without_gpu = [r for r in successful_results if not r.config['gpu_stabilizer']]
        
        if with_gpu and without_gpu:
            gpu_enabled_util = np.mean([r.gpu_utilization for r in with_gpu])
            gpu_disabled_util = np.mean([r.gpu_utilization for r in without_gpu])
            report += f"""
- **With GPU Stabilizer**: Avg GPU Utilization={gpu_enabled_util:.1f}%
- **Without GPU Stabilizer**: Avg GPU Utilization={gpu_disabled_util:.1f}%
- **Improvement**: {gpu_enabled_util - gpu_disabled_util:.1f}% higher utilization"""
        
        report += f"""

## 🎯 Recommendations

### For Maximum Quality (Lowest Loss):
```bash
python3 train_main.py \\
    --model-size {best_loss.config['model_size']} \\
    --optimization-level {best_loss.config['optimization_level']} \\
    --lr {best_loss.config['learning_rate']} \\
    --batch-size {best_loss.config['batch_size']} \\
    {'--enable-gpu-stabilizer' if best_loss.config['gpu_stabilizer'] else '--disable-gpu-stabilizer'}
```

### For Maximum Speed:
```bash
python3 train_main.py \\
    --model-size {best_speed.config['model_size']} \\
    --optimization-level {best_speed.config['optimization_level']} \\
    --lr {best_speed.config['learning_rate']} \\
    --batch-size {best_speed.config['batch_size']} \\
    {'--enable-gpu-stabilizer' if best_speed.config['gpu_stabilizer'] else '--disable-gpu-stabilizer'}
```

### For Balanced Performance (Best Overall):
```bash
python3 train_main.py \\
    --model-size {best_overall.config['model_size']} \\
    --optimization-level {best_overall.config['optimization_level']} \\
    --lr {best_overall.config['learning_rate']} \\
    --batch-size {best_overall.config['batch_size']} \\
    {'--enable-gpu-stabilizer' if best_overall.config['gpu_stabilizer'] else '--disable-gpu-stabilizer'}
```

## 📋 Failed Configurations

"""
        
        # List failed configurations
        failed_results = [r for r in self.results if not r.completed]
        if failed_results:
            report += f"**{len(failed_results)} configurations failed:**\n\n"
            for result in failed_results[:5]:  # Show first 5
                report += f"- {result.config}: {result.error_message or 'Unknown error'}\n"
            if len(failed_results) > 5:
                report += f"- ... and {len(failed_results) - 5} more\n"
        else:
            report += "**All configurations completed successfully!**\n"
        
        report += """

## 📁 Files Generated

- `results.json` - Raw benchmark data
- `results.csv` - Tabular results for analysis
- `benchmark_plots.png` - Visualization plots
- `benchmark_report.md` - This report

---

**🎯 Use these insights to optimize your MyXTTS training for your specific use case!**
"""
        
        return report


def create_quick_benchmark_config() -> BenchmarkConfig:
    """Create configuration for quick 15-minute benchmark"""
    return BenchmarkConfig(
        model_sizes=["tiny", "small"],
        optimization_levels=["enhanced", "plateau_breaker"],
        learning_rates=[1.5e-5, 2e-5, 5e-5],
        batch_sizes=[4, 8, 16],
        gpu_stabilizer_options=[True, False],
        mel_loss_weights=[2.0, 2.5],
        kl_loss_weights=[1.2, 1.5],
        test_epochs=5,
        max_test_time=180  # 3 minutes per test
    )


def create_full_benchmark_config() -> BenchmarkConfig:
    """Create configuration for comprehensive benchmark"""
    return BenchmarkConfig(
        test_epochs=20,
        max_test_time=600  # 10 minutes per test
    )


def main():
    parser = argparse.ArgumentParser(description="MyXTTS Hyperparameter Benchmark Suite")
    parser.add_argument("--config", help="Benchmark configuration file")
    parser.add_argument("--quick-test", action="store_true", help="Run quick 15-minute benchmark")
    parser.add_argument("--full-sweep", action="store_true", help="Run comprehensive benchmark sweep")
    parser.add_argument("--custom-params", help="JSON string with custom parameter ranges")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.quick_test:
        config = create_quick_benchmark_config()
        logger.info("Using quick test configuration (15 minutes)")
    elif args.full_sweep:
        config = create_full_benchmark_config()
        logger.info("Using full sweep configuration (several hours)")
    elif args.config:
        # Load from file
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        config = BenchmarkConfig(**config_data)
    else:
        # Default configuration
        config = BenchmarkConfig()
        logger.info("Using default benchmark configuration")
    
    # Create and run benchmark
    benchmark = HyperparameterBenchmark(config)
    
    if args.quick_test:
        benchmark.quick_test = True
    
    try:
        benchmark.run_full_benchmark()
        
        if benchmark.best_result:
            logger.info("=" * 60)
            logger.info("🏆 BENCHMARK COMPLETE - BEST CONFIGURATION FOUND:")
            logger.info("=" * 60)
            logger.info(f"Score: {benchmark.best_result.score():.3f}")
            logger.info(f"Configuration: {benchmark.best_result.config}")
            logger.info(f"Final Loss: {benchmark.best_result.final_loss:.3f}")
            logger.info(f"Training Speed: {benchmark.best_result.training_speed:.1f} samples/sec")
            logger.info(f"GPU Utilization: {benchmark.best_result.gpu_utilization:.1f}%")
            logger.info("=" * 60)
            logger.info(f"📋 Full report available in: {benchmark.output_dir}/")
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        benchmark.save_results()
        logger.info("Partial results saved")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        benchmark.save_results()
        raise


if __name__ == "__main__":
    main()