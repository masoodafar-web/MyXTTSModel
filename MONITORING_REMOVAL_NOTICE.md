# GPU Monitoring Removal Notice

## What Changed

As requested, detailed GPU monitoring and graphics logging have been removed from the MyXTTS codebase to reduce complexity and overhead.

## Removed Components

- **Detailed GPU monitoring** from `gpu_monitor.py`
- **PerformanceMonitor** detailed GPU utilization tracking
- **DataLoadingProfiler** comprehensive profiling
- **GPU utilization reporting** in training loops
- **Performance monitoring test files**

## What Remains (Essential Components)

- **Basic GPU detection** in `gpu_monitor.py` for device availability checking
- **Standard TensorFlow GPU configuration** in `commons.py`
- **Essential GPU setup functions** (configure_gpus, setup_gpu_strategy)
- **Basic CPU/memory monitoring** for essential system checks
- **Standard logging** for training progress

## Updated Files

- `gpu_monitor.py` - Now only provides basic GPU detection
- `myxtts/utils/performance.py` - Simplified to basic CPU/memory monitoring
- `myxtts/training/trainer.py` - Removed performance monitoring integration
- `myxtts/data/ljspeech.py` - Removed data loading profiler
- `test_performance.py` - Removed (was testing removed features)
- `test_gpu_utilization.py` - Removed (was testing removed features)

## Migration Notes

If you need basic GPU information, use:

```python
from gpu_monitor import get_gpu_info, check_gpu_availability

# Check if GPU is available
has_gpu = check_gpu_availability()

# Get basic GPU information
gpu_info = get_gpu_info()
print(f"GPU available: {gpu_info['available']}")
print(f"GPU count: {gpu_info['count']}")
```

For essential GPU configuration:

```python
from myxtts.utils.commons import configure_gpus, setup_gpu_strategy

# Configure GPU settings
configure_gpus(memory_growth=True)

# Setup GPU strategy for training
strategy = setup_gpu_strategy()
```

## Documentation Updates Needed

The following documentation files contain references to removed monitoring features and should be considered outdated:

- `PERFORMANCE_IMPROVEMENTS.md`
- `GPU_FIX_USAGE_GUIDE.md`
- `MEMORY_OPTIMIZATION_FIXES.md`
- `GPU_UTILIZATION_FIXES.md`
- `ADVANCED_MEMORY_OPTIMIZATION_GUIDE.md`
- `MYXTTSTRAIN_COMPLETION_FIX.md`

These files may reference removed functions and should be updated to reflect the current simplified monitoring approach.