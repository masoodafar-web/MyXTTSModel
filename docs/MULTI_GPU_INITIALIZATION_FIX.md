# Multi-GPU Initialization Fix

## Problem

When using Multi-GPU mode with `--data-gpu` and `--model-gpu` arguments, the training script would fail with the error:

```
⚠️  Multi-GPU setup failed, falling back to default: Physical devices cannot be modified after being initialized
```

### Root Cause

TensorFlow GPUs cannot be reconfigured after initialization. The previous implementation had the following problematic order:

1. `train_main.py` imports `tensorflow as tf` (line 248)
2. Various TF operations happen (e.g., `tf.config.list_physical_devices('GPU')` at line 1250)
3. Data pipeline tries to configure GPUs in `ljspeech.py` (line 1214-1216)
4. **ERROR**: Attempting to modify physical devices after initialization

## Solution

The fix implements early GPU configuration that happens **before** any TensorFlow operations:

### 1. New `early_gpu_configuration()` Function

Added to `myxtts/utils/commons.py`:

```python
def early_gpu_configuration(data_gpu: Optional[int] = None, model_gpu: Optional[int] = None) -> bool:
    """Configure GPUs before any TensorFlow operations.
    
    This function MUST be called before any TensorFlow GPU operations to avoid the error:
    "Physical devices cannot be modified after being initialized"
    
    Args:
        data_gpu: GPU ID for data processing (Multi-GPU mode)
        model_gpu: GPU ID for model training (Multi-GPU mode)
    
    Returns:
        bool: True if multi-GPU mode was successfully configured, False otherwise
    """
```

**Key Features:**
- Validates GPU indices before configuration
- Sets visible devices for multi-GPU mode
- Configures memory growth for each GPU
- Sets device policy to 'silent' for automatic tensor copying
- Returns clear success/failure status
- Provides detailed logging for debugging

### 2. Modified Training Script (`train_main.py`)

**Import the function:**
```python
from myxtts.utils.commons import setup_logging, find_latest_checkpoint, early_gpu_configuration
```

**Call it immediately after argument parsing:**
```python
args = parser.parse_args()
logger = setup_logging()

# CRITICAL: Configure GPUs BEFORE any TensorFlow operations
is_multi_gpu_mode = early_gpu_configuration(data_gpu=args.data_gpu, model_gpu=args.model_gpu)

if args.data_gpu is not None and args.model_gpu is not None and not is_multi_gpu_mode:
    logger.error("❌ Multi-GPU mode was requested but configuration failed")
    sys.exit(1)

# Now safe to do TensorFlow operations
gpu_available = bool(tf.config.list_physical_devices('GPU'))
```

### 3. Updated Data Pipeline (`myxtts/data/ljspeech.py`)

Removed the problematic GPU configuration code that tried to modify devices after initialization. The data pipeline now:

1. Assumes GPUs are already configured by `early_gpu_configuration()`
2. Uses remapped GPU indices (original `data_gpu` -> `GPU:0`, original `model_gpu` -> `GPU:1`)
3. Only configures the data pipeline, not the physical devices

**Before:**
```python
# Configure GPU memory growth
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)  # ❌ Fails!
```

**After:**
```python
# Note: GPU configuration is done early in train_main.py via early_gpu_configuration()
# After early configuration, visible devices are remapped: data_gpu -> GPU:0, model_gpu -> GPU:1
data_device = '/GPU:0'  # Use remapped device
```

## GPU Device Remapping

When multi-GPU mode is configured, TensorFlow remaps the visible devices:

| Original Device | After `set_visible_devices` | Usage |
|----------------|---------------------------|-------|
| GPU N (data_gpu) | GPU:0 | Data processing |
| GPU M (model_gpu) | GPU:1 | Model training |

**Example:**
```bash
# User specifies:
python train_main.py --data-gpu 0 --model-gpu 1

# After early_gpu_configuration():
# - Only GPUs 0 and 1 are visible
# - GPU 0 becomes /GPU:0 (data processing)
# - GPU 1 becomes /GPU:1 (model training)
```

## Error Handling

The fix includes comprehensive error handling:

### 1. Insufficient GPUs
```
❌ Multi-GPU requires at least 2 GPUs, found 1
   Falling back to single-GPU mode
```

### 2. Invalid GPU Indices
```
❌ Invalid data_gpu=5, must be 0-1
```

### 3. Configuration Failure
```
❌ Multi-GPU mode was requested but configuration failed
   Please check your GPU indices and ensure you have at least 2 GPUs
```

### 4. Already Initialized
```
❌ Multi-GPU setup failed: Physical devices cannot be modified after being initialized
   This error typically means TensorFlow was already initialized.
   Ensure early_gpu_configuration() is called BEFORE any TF operations.
```

## Usage

### Multi-GPU Mode
```bash
# Use GPU 0 for data processing, GPU 1 for model training
python train_main.py \
    --data-gpu 0 \
    --model-gpu 1 \
    --train-data ../dataset/dataset_train \
    --val-data ../dataset/dataset_eval
```

### Single-GPU Mode (Default)
```bash
# Uses all available GPUs with memory growth
python train_main.py \
    --train-data ../dataset/dataset_train \
    --val-data ../dataset/dataset_eval
```

## Verification

To verify the fix is working:

1. **Check for the error message:** The error "Physical devices cannot be modified after being initialized" should no longer appear

2. **Look for success messages:**
   ```
   🎯 Configuring Multi-GPU Mode...
      Data Processing GPU: 0
      Model Training GPU: 1
      Set visible devices: GPU 0 and GPU 1
      Configured memory growth for data GPU
      Configured memory growth for model GPU
   ✅ Multi-GPU configuration completed successfully
   ```

3. **Monitor GPU usage:**
   ```bash
   watch -n 1 nvidia-smi
   ```
   Both GPUs should show activity during training.

## Testing

Run the test suite to verify configuration:
```bash
python tests/test_intelligent_gpu_pipeline.py
```

All tests should pass, including:
- `test_dataconfig_multi_gpu_mode`
- `test_gpu_mode_detection`
- `test_buffer_size_validation`

## Benefits

1. ✅ **Eliminates the initialization error** completely
2. ✅ **Clear separation of concerns**: GPU configuration happens once, early
3. ✅ **Better error messages**: Users get actionable feedback
4. ✅ **Explicit failure**: No silent fallback that might confuse users
5. ✅ **Proper device remapping**: Code correctly uses remapped device indices
6. ✅ **Backward compatible**: Single-GPU mode continues to work as before

## Migration Notes

- **No changes required for single-GPU users**: The fix is transparent
- **Multi-GPU users**: Should see improved reliability and clear error messages
- **Developers**: When adding TF operations, they can safely assume GPUs are already configured
