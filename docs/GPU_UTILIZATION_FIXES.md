# GPU Utilization Fixes for MyXTTS

This document describes the improvements made to fix GPU utilization issues in MyXTTS training.

## Problem Description

The original issue was that GPU utilization remained at 0% during training while CPU usage was at maximum, indicating that the model was not properly utilizing available GPU resources. This led to inefficient training and poor performance.

## Root Causes Identified

1. **Missing explicit GPU device placement** - Tensors and operations were not explicitly placed on GPU
2. **Lack of distribution strategy** - No proper GPU distribution strategy for optimal utilization
3. **Suboptimal data pipeline** - Data loading was creating CPU bottlenecks
4. **Missing GPU optimizations** - XLA compilation, mixed precision, and other GPU optimizations were not properly configured
5. **Inefficient memory management** - GPU memory was not being optimally utilized

## Fixes Implemented

### 1. Enhanced GPU Device Management (`myxtts/utils/commons.py`)

```python
def setup_gpu_strategy():
    """Set up GPU distribution strategy for optimal GPU utilization."""
    
def ensure_gpu_placement(tensor):
    """Ensure a tensor is placed on GPU if available."""
```

- Added proper GPU strategy setup for single and multi-GPU configurations
- Implemented explicit GPU tensor placement functions
- Enhanced GPU memory growth configuration

### 2. Distributed Training Support (`myxtts/training/trainer.py`)

```python
@tf.function
def distributed_train_step(self, dist_inputs):
    """Distributed training step for multi-GPU training."""
    
@tf.function
def distributed_validation_step(self, dist_inputs):
    """Distributed validation step for multi-GPU validation."""
```

- Added distributed training and validation steps
- Implemented proper loss scaling for distributed training
- Added explicit GPU tensor placement in training loops
- Enhanced mixed precision support with proper scaling

### 3. Optimized Data Pipeline

- **Increased prefetch buffer size** from 4 to 8 batches
- **Enhanced parallel processing** with more workers
- **GPU-optimized dataset distribution** using `strategy.experimental_distribute_dataset`
- **Improved batch size defaults** from 16 to 32+ for better GPU utilization

### 4. GPU-Specific Optimizations

```python
# Enable XLA compilation
if config.data.enable_xla:
    tf.config.optimizer.set_jit(True)

# Enhanced mixed precision
if config.data.mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
```

- **XLA compilation** for better GPU kernel optimization
- **Enhanced mixed precision training** with proper loss scaling
- **Memory pinning** for faster CPU-GPU data transfer
- **Improved gradient clipping** with distributed training support

### 5. GPU Monitoring and Diagnostics

Created comprehensive monitoring tools:

- **`gpu_monitor.py`** - Real-time GPU utilization monitoring
- **`test_gpu_utilization.py`** - Validation script for GPU improvements
- **Enhanced performance monitoring** with detailed GPU metrics

## Configuration Changes

### Updated `config.yaml`

```yaml
training:
  batch_size: 48              # Increased for better GPU utilization

data:
  batch_size: 48              # Increased for better GPU utilization
  num_workers: 12             # Increased for better CPU utilization
  prefetch_buffer_size: 8     # Increased for better GPU utilization
  shuffle_buffer_multiplier: 20
  enable_xla: true            # Enable XLA compilation
  mixed_precision: true       # Enable mixed precision
  pin_memory: true            # Pin memory for faster GPU transfer
```

## Usage Instructions

### 1. Training with GPU Optimizations

```bash
# Use the optimized configuration
python trainTestFile.py --mode train --config config.yaml

# Or use programmatic configuration with GPU optimizations
python trainTestFile.py --mode train --batch-size 48 --epochs 1000
```

### 2. Monitor GPU Utilization

```bash
# Start GPU monitoring during training
python gpu_monitor.py --log-file --duration 3600

# Test GPU utilization improvements
python test_gpu_utilization.py
```

### 3. Validate Fixes

```bash
# Run comprehensive GPU utilization test
python test_gpu_utilization.py --stress-duration 60
```

## Expected Results

After implementing these fixes, you should see:

1. **GPU utilization increase** from 0% to 70-90% during training
2. **CPU usage decrease** from 100% to more reasonable levels (40-60%)
3. **Faster training speed** due to proper GPU utilization
4. **Better memory utilization** with optimized batch sizes
5. **More stable training** with proper distributed training setup

## Monitoring and Troubleshooting

### Key Metrics to Monitor

- **GPU Utilization**: Should be 70-90% during training
- **GPU Memory Usage**: Should be 80-95% of available memory
- **Data Loading Time**: Should be < 50% of compute time
- **Batch Processing Speed**: Should increase significantly

### Common Issues and Solutions

1. **Still low GPU utilization (<30%)**:
   - Increase `prefetch_buffer_size` in config
   - Increase `num_workers` for data loading
   - Check data preprocessing bottlenecks

2. **GPU memory errors**:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use memory growth settings

3. **Slow data loading**:
   - Increase `num_workers`
   - Use cached datasets
   - Optimize data preprocessing

## Technical Details

### Distribution Strategy Selection

- **Single GPU**: Uses `OneDeviceStrategy` with explicit device placement
- **Multi-GPU**: Uses `MirroredStrategy` for data parallelism
- **CPU-only**: Falls back to default strategy

### Mixed Precision Training

- Uses `mixed_float16` policy for forward pass
- Maintains `float32` precision for loss computation
- Includes proper loss scaling for gradient computation

### XLA Compilation

- Automatically enables JIT compilation for GPU kernels
- Optimizes computation graphs for better performance
- Reduces memory overhead and improves speed

## Performance Benchmarks

Expected improvements after applying fixes:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| GPU Utilization | 0-5% | 70-90% | 15-20x |
| Training Speed | Baseline | 2-3x faster | 200-300% |
| CPU Usage | 100% | 40-60% | 40-60% reduction |
| Memory Efficiency | Low | High | Significantly better |

## Files Modified

1. `myxtts/utils/commons.py` - Enhanced GPU device management
2. `myxtts/training/trainer.py` - Added distributed training support
3. `myxtts/config/config.py` - Added GPU optimization settings
4. `myxtts/utils/performance.py` - Enhanced GPU monitoring
5. `config.yaml` - Optimized default configuration
6. `trainTestFile.py` - Updated default batch sizes

## New Files Added

1. `gpu_monitor.py` - Real-time GPU monitoring tool
2. `test_gpu_utilization.py` - GPU validation and testing script
3. `GPU_UTILIZATION_FIXES.md` - This documentation

These fixes should resolve the GPU utilization issue and significantly improve training performance.