# Dataset Preprocessing Control for MyXTTS

This document describes the new dataset preprocessing control feature that allows users to optimize GPU utilization by controlling how and when dataset preprocessing occurs.

## Problem Statement

GPU utilization was low during training due to CPU preprocessing bottlenecks. The model would show 0% GPU utilization while CPU usage remained at 100%, indicating that data preprocessing was becoming a bottleneck during training.

## Solution: Configurable Preprocessing Modes

MyXTTS now supports three preprocessing modes that give users control over when and how dataset preprocessing occurs:

### 1. AUTO Mode (Default)
- **Behavior**: Attempts to precompute dataset before training, falls back gracefully if preprocessing fails
- **Use Case**: General-purpose training, good starting point for most users
- **GPU Impact**: Moderate improvement in GPU utilization
- **Disk Usage**: Creates cache files when possible

```bash
# Default behavior
python trainTestFile.py --mode train --data-path ./data/ljspeech
python trainTestFile.py --mode train --preprocessing-mode auto
```

### 2. PRECOMPUTE Mode (Recommended for GPU optimization)
- **Behavior**: Forces complete dataset preprocessing before training starts
- **Use Case**: Maximize GPU utilization by eliminating preprocessing bottlenecks during training
- **GPU Impact**: Maximum GPU utilization (70-90% target)
- **Disk Usage**: Requires sufficient disk space for mel spectrograms and token caches
- **Failure Mode**: Training fails if preprocessing cannot be completed

```bash
# Force preprocessing before training
python trainTestFile.py --mode train --preprocessing-mode precompute --batch-size 48
```

### 3. RUNTIME Mode (For limited disk space)
- **Behavior**: Processes data on-the-fly during training, no disk caching
- **Use Case**: Limited disk space scenarios
- **GPU Impact**: May have lower GPU utilization due to CPU processing during training
- **Disk Usage**: Minimal (no cache files created)

```bash
# On-the-fly processing
python trainTestFile.py --mode train --preprocessing-mode runtime --batch-size 16
```

## Configuration

### Command Line Usage

```bash
# Create configuration with specific preprocessing mode
python trainTestFile.py --mode create-config --output gpu_optimized.yaml --preprocessing-mode precompute

# Train with specific preprocessing mode
python trainTestFile.py --mode train --preprocessing-mode precompute --batch-size 48

# Override YAML config preprocessing mode
python trainTestFile.py --mode train --config my_config.yaml --preprocessing-mode precompute
```

### YAML Configuration

```yaml
data:
  dataset_path: ./data/ljspeech
  batch_size: 48
  preprocessing_mode: precompute  # "auto", "precompute", "runtime"
  num_workers: 12
  prefetch_buffer_size: 8
```

### Programmatic Configuration

```python
from myxtts.config.config import XTTSConfig
from trainTestFile import create_default_config

# Create config with specific preprocessing mode
config = create_default_config(
    data_path="./data/ljspeech",
    preprocessing_mode="precompute",
    batch_size=48
)

# Or use XTTSConfig directly
config = XTTSConfig(
    data_path="./data/ljspeech",
    preprocessing_mode="precompute",
    batch_size=48
)
```

## Performance Impact

### Expected GPU Utilization Improvements

| Mode | GPU Utilization | CPU Usage | Disk Usage | Training Speed |
|------|----------------|-----------|------------|----------------|
| auto | 40-70% | Medium | Moderate | Good |
| precompute | 70-90% | Low | High | Excellent |
| runtime | 20-50% | High | Low | Slower |

### Recommended Settings by Mode

#### PRECOMPUTE Mode (Maximum GPU utilization)
```yaml
data:
  preprocessing_mode: precompute
  batch_size: 48          # Larger batch size possible
  num_workers: 12         # More workers for initial preprocessing
  prefetch_buffer_size: 8 # Larger prefetch buffer
```

#### RUNTIME Mode (Limited disk space)
```yaml
data:
  preprocessing_mode: runtime
  batch_size: 16          # Smaller batch size to account for runtime processing
  num_workers: 8          # Fewer workers to avoid overloading CPU
  prefetch_buffer_size: 4 # Smaller prefetch buffer
```

## Implementation Details

### Configuration Parameter

The new `preprocessing_mode` parameter is added to `DataConfig`:

```python
@dataclass
class DataConfig:
    # ... other parameters ...
    
    # Dataset preprocessing control
    preprocessing_mode: str = "auto"  # "auto", "precompute", "runtime"
    
    def __post_init__(self):
        # Validate preprocessing_mode
        valid_modes = ["auto", "precompute", "runtime"]
        if self.preprocessing_mode not in valid_modes:
            raise ValueError(f"preprocessing_mode must be one of {valid_modes}")
```

### Trainer Implementation

The `XTTSTrainer.prepare_datasets()` method now respects the preprocessing mode:

- **precompute**: Forces complete preprocessing, fails if any preprocessing step fails
- **runtime**: Disables all caching and preprocessing
- **auto**: Attempts preprocessing with graceful fallback (original behavior)

## Usage Examples

### Example 1: Maximum GPU Utilization Setup

```bash
# Create GPU-optimized configuration
python trainTestFile.py --mode create-config \
  --output gpu_optimized.yaml \
  --preprocessing-mode precompute \
  --batch-size 48 \
  --epochs 1000

# Train with GPU optimization
python trainTestFile.py --mode train --config gpu_optimized.yaml
```

### Example 2: Limited Disk Space Setup

```bash
# Create low-storage configuration
python trainTestFile.py --mode create-config \
  --output low_storage.yaml \
  --preprocessing-mode runtime \
  --batch-size 16 \
  --epochs 100

# Train with minimal disk usage
python trainTestFile.py --mode train --config low_storage.yaml
```

### Example 3: Quick Test with Runtime Override

```bash
# Use existing config but override preprocessing mode
python trainTestFile.py --mode train \
  --config existing_config.yaml \
  --preprocessing-mode precompute \
  --batch-size 64
```

## Monitoring and Troubleshooting

### GPU Utilization Monitoring

```bash
# Monitor GPU utilization during training
python gpu_monitor.py --log-file --duration 3600

# Test GPU utilization improvements
python test_gpu_utilization.py
```

### Common Issues and Solutions

1. **Low GPU utilization with precompute mode**:
   - Verify all data was successfully preprocessed
   - Check cache file integrity
   - Increase batch size if GPU memory allows

2. **Preprocessing fails in precompute mode**:
   - Check disk space availability
   - Verify dataset file permissions
   - Try auto mode as fallback

3. **High CPU usage in runtime mode**:
   - Reduce batch size
   - Reduce number of workers
   - Consider switching to auto or precompute mode

### Performance Validation

Target metrics after implementing preprocessing control:

- **GPU Utilization**: 70-90% during training (precompute mode)
- **CPU Usage**: 40-60% during training (precompute mode)  
- **Training Speed**: 2-3x improvement over runtime mode
- **Memory Efficiency**: Optimal with proper batch sizing

## Backward Compatibility

The implementation maintains full backward compatibility:

- Default mode is "auto" which preserves existing behavior
- Existing YAML configurations continue to work without modification
- Command line scripts work without changes
- Only new functionality is added, no existing functionality is removed

## Future Enhancements

Potential future improvements to preprocessing control:

1. **Automatic mode selection** based on available disk space and dataset size
2. **Hybrid preprocessing** that preprocesses frequently accessed items
3. **Distributed preprocessing** for multi-node training setups
4. **Progressive preprocessing** that preprocesses data while training early batches