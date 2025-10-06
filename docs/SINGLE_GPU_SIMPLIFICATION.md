# Single GPU Simplification

## Overview

This document describes the removal of all multi-GPU and MirroredStrategy support from MyXTTS, simplifying the codebase to support only single GPU or CPU training.

## Changes Made

### Core Functions Modified

#### `myxtts/utils/commons.py`

- **`setup_gpu_strategy()`**: Simplified to always return default strategy (no parameters)
  - Removed `enable_multi_gpu` parameter
  - Always returns `tf.distribute.get_strategy()` regardless of GPU count
  
- **`ensure_gpu_placement()`**: Simplified GPU placement logic
  - Removed multi-GPU replica checks
  - Directly places tensors on GPU:0 when available
  
- **`get_device_context()`**: Simplified device context selection
  - Removed multi-GPU replica checks
  - Returns GPU:0 or CPU:0 device context

#### `myxtts/config/config.py`

- **`TrainingConfig`**: Removed multi-GPU parameters
  - Removed `multi_gpu: bool` parameter
  - Removed `visible_gpus: Optional[str]` parameter
  - Added comment indicating single GPU training only

#### `myxtts/training/trainer.py`

- **Removed Methods**:
  - `distributed_train_step()` - Multi-GPU training method
  - `distributed_validation_step()` - Multi-GPU validation method

- **Simplified Logic**:
  - Removed all `multi_gpu` configuration checks
  - Removed all `MirroredStrategy` type checks
  - Removed dataset distribution logic
  - Removed distributed warmup logic
  - Simplified `cleanup_gpu_memory()` - no MirroredStrategy checks
  - Direct training/validation step calls (no strategy.run wrapping)

#### `train_main.py`

- **Removed CLI Arguments**:
  - `--multi-gpu` - Enable multi-GPU training flag
  - `--visible-gpus` - GPU selection argument

- **Simplified Code**:
  - Removed multi_gpu configuration setting
  - Cleaned up commented multi-GPU code blocks

### Documentation Changes

#### Deleted Files

- `docs/GPU_STRATEGY_CONTROL.md` - Multi-GPU configuration guide
- `docs/IMPLEMENTATION_SUMMARY.md` - Multi-GPU implementation details
- `examples/gpu_strategy_demo.py` - Multi-GPU demonstration script

#### Updated Files

- `README.md` - Changed "Multi-GPU Support" to "Single GPU Training"
- `examples/example_usage.py` - Removed multi-GPU examples
- `utilities/validate_memory_fixes.py` - Updated tests for single GPU
- `monitoring/debug_cpu_usage.py` - Updated to use single GPU

### Test Updates

#### `tests/test_gpu_strategy.py`

- Simplified to test single GPU behavior only
- Removed multi-GPU test cases
- Tests now verify default strategy is returned regardless of GPU count

### Minor Comment Updates

- `myxtts/models/layers.py` - Removed MirroredStrategy mention from comment
- `myxtts/training/simple_loss.py` - Removed MirroredStrategy from docstring
- `myxtts/data/ljspeech.py` - Simplified GPU prefetch logic for single GPU

## Benefits

1. **Simpler Codebase**: Removed ~600+ lines of complex distributed training logic
2. **Easier Maintenance**: No need to maintain and test multi-GPU paths
3. **Clearer Intent**: Code explicitly states single GPU training only
4. **Reduced Complexity**: No strategy contexts, replica handling, or distributed dataset management
5. **Better Focus**: Resources focused on single GPU optimization

## Usage

Training is now straightforward with automatic GPU detection:

```bash
# Train on GPU (if available) or CPU
python train_main.py --config configs/config.yaml
```

The system will automatically:
- Use GPU:0 if a GPU is available
- Fall back to CPU if no GPU is detected
- Apply single GPU optimizations

## Note

The `configure_gpus()` function in `commons.py` still has a `visible_gpus` parameter. This is intentional and allows selecting which single GPU to use in multi-GPU systems:

```python
# Use GPU 1 instead of GPU 0
configure_gpus(visible_gpus="1")
```

This is different from multi-GPU distributed training - it simply selects which GPU to use for single GPU training.
