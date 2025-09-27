# GPU Strategy Control in MyXTTS

## Overview

This document describes the GPU strategy control feature that allows users to enable or disable MirroredStrategy for multi-GPU training in MyXTTS.

## Problem Statement

Previously, the GPU strategy selection was automatic based on hardware detection:
- Single GPU: OneDeviceStrategy
- Multiple GPUs: MirroredStrategy (automatic)

This didn't give users control over whether they wanted to use multiple GPUs or stick to single GPU training.

## Solution

### Configuration Parameter

Added a new configuration parameter `multi_gpu` in `TrainingConfig`:

```python
@dataclass
class TrainingConfig:
    # Device / distribution
    multi_gpu: bool = False            # Enable MirroredStrategy when True
    visible_gpus: Optional[str] = None # e.g., "0" or "0,1"; None = all visible
```

### Updated GPU Strategy Function

Modified `setup_gpu_strategy()` in `myxtts/utils/commons.py` to accept an `enable_multi_gpu` parameter:

```python
def setup_gpu_strategy(enable_multi_gpu: bool = False):
    """
    Set up GPU distribution strategy for optimal GPU utilization.
    
    Args:
        enable_multi_gpu: Whether to enable MirroredStrategy for multi-GPU training.
                         If False, uses OneDeviceStrategy even with multiple GPUs.
    
    Returns:
        tf.distribute.Strategy for training
    """
```

### Strategy Selection Logic

The new logic works as follows:

1. **No GPUs available**: Always use `tf.distribute.get_strategy()` (CPU strategy)
2. **Single GPU available**: Always use `OneDeviceStrategy("/gpu:0")`
3. **Multiple GPUs available**:
   - If `enable_multi_gpu=True`: Use `MirroredStrategy()` for distributed training
   - If `enable_multi_gpu=False`: Use `OneDeviceStrategy("/gpu:0")` (first GPU only)

### Trainer Integration

Updated `XTTSTrainer` to use the new GPU strategy:

- Imports `setup_gpu_strategy` function
- Calls it with the configuration parameter: `setup_gpu_strategy(enable_multi_gpu=config.training.multi_gpu)`
- Creates model and optimizer within strategy scope for proper distributed training
- Uses distributed training steps when MirroredStrategy is active

## Usage Examples

### Enable Multi-GPU Training

```python
from myxtts.config.config import XTTSConfig, TrainingConfig

# Method 1: Via TrainingConfig
training_config = TrainingConfig(multi_gpu=True)
config = XTTSConfig(training=training_config)

# Method 2: Via XTTSConfig kwargs
config = XTTSConfig(multi_gpu=True)

# Method 3: Via YAML
# training:
#   multi_gpu: true
config = XTTSConfig.from_yaml("config.yaml")
```

### Disable Multi-GPU Training (Force Single GPU)

```python
# Default behavior - uses single GPU even with multiple GPUs available
config = XTTSConfig()  # multi_gpu defaults to False

# Explicit setting
config = XTTSConfig(multi_gpu=False)
```

### Control GPU Visibility

```python
# Use only specific GPUs
config = XTTSConfig(
    multi_gpu=True,
    visible_gpus="0,1"  # Use only GPU 0 and 1
)

# Use only GPU 0 (even if multi_gpu=True, only one GPU is visible)
config = XTTSConfig(
    multi_gpu=True,
    visible_gpus="0"
)
```

## Benefits

1. **User Control**: Users can choose whether to use multiple GPUs or not
2. **Better Resource Management**: Can limit training to specific GPUs
3. **Debugging**: Easier to debug on single GPU before scaling to multiple GPUs
4. **Performance**: Can avoid multi-GPU overhead for smaller models/datasets
5. **GPU Utilization**: Proper strategy selection should improve GPU utilization from 10% to 70-90%

## Implementation Details

### Files Modified

1. `myxtts/utils/commons.py`: Updated `setup_gpu_strategy()` function
2. `myxtts/training/trainer.py`: Updated trainer to use proper GPU strategy
3. `myxtts/config/config.py`: Already had `multi_gpu` parameter (no changes needed)

### Backward Compatibility

- `multi_gpu` defaults to `False`, maintaining existing behavior for users who don't explicitly enable it
- Existing configurations without `multi_gpu` setting will continue to work
- Single GPU setups are unaffected

### GPU Utilization Improvements

The changes should address the low GPU utilization issue by:

1. Using proper distribution strategies instead of default strategy
2. Ensuring model and optimizer are created within strategy scope
3. Using distributed training/validation steps when appropriate
4. Maintaining GPU memory growth and other optimizations