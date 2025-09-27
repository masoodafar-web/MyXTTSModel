# Device Placement Fix for MyXTTS Model

## Problem Description

The MyXTTS model was encountering an `InvalidArgumentError` during initialization when GPUs were available:

```
InvalidArgumentError: Tensors on conflicting devices: cannot compute Cast as input #0 was expected to be on /job:localhost/replica:0/task:0/device:GPU:0 but is actually on /job:localhost/replica:0/task:0/device:CPU:0
```

This error occurred specifically in the `MultiHeadAttention` layer initialization when creating `Dropout` layers, due to TensorFlow's internal seed generator creating tensors on CPU while the model expected GPU placement.

## Root Cause Analysis

1. **Device Policy Conflict**: TensorFlow was configured with `explicit` device policy, which prevented automatic tensor copying between devices
2. **Seed Generator Issue**: Dropout layer seed generators created tensors on CPU by default
3. **Mixed Device Context**: Model components were being created with inconsistent device placement

## Solution Implemented

### 1. Updated GPU Configuration (`myxtts/utils/commons.py`)

**Before:**
```python
tf.config.experimental.set_device_policy('explicit')
```

**After:**
```python
tf.config.experimental.set_device_policy('silent')
```

The `silent` policy allows TensorFlow to automatically copy tensors between devices when needed, preventing device placement conflicts.

### 2. Added Device-Aware Layer Creation Utilities

**New Functions:**
- `get_device_context()`: Returns appropriate device context (GPU if available, CPU otherwise)
- `create_dropout_layer()`: Creates dropout layers with proper device placement

### 3. Updated Layer Implementations (`myxtts/models/layers.py`)

**All Dropout Layer Creation:**
```python
# Before
self.dropout = tf.keras.layers.Dropout(dropout)

# After  
self.dropout = create_dropout_layer(dropout, name="dropout")
```

**Positional Encoding Variable Creation:**
```python
# Before
self.pe = tf.Variable(...)

# After
with get_device_context():
    self.pe = tf.Variable(...)
```

### 4. Early GPU Configuration (`myxtts/models/__init__.py`)

Added automatic GPU configuration when the models module is imported to ensure proper device policy is set before any layer creation.

## Benefits

✅ **Resolves Original Error**: Eliminates the InvalidArgumentError with device placement conflicts  
✅ **Maintains Functionality**: All layers work identically to before, just with better device handling  
✅ **Backward Compatible**: No changes to public APIs or usage patterns  
✅ **Automatic Configuration**: GPU settings are applied automatically when needed  
✅ **Robust Handling**: Graceful fallback to CPU when GPU is not available  

## Files Modified

1. `myxtts/utils/commons.py`:
   - Changed device policy to 'silent'
   - Added `get_device_context()` and `create_dropout_layer()` utilities

2. `myxtts/models/layers.py`:
   - Updated all Dropout layer creations to use `create_dropout_layer()`
   - Added device context for PositionalEncoding Variable creation
   - Added import for device utilities

3. `myxtts/models/__init__.py`:
   - Added automatic GPU configuration on module import

## Testing

The fix has been validated with comprehensive tests that verify:
- Device-aware dropout creation works correctly
- All layer types handle device placement consistently  
- Functionality remains identical to original implementation
- Proper fallback behavior when GPU is not available

## Usage

The fix is applied automatically when importing the MyXTTS models:

```python
from myxtts.models import MultiHeadAttention  # GPU config applied automatically

# This will now work without device placement errors
attention = MultiHeadAttention(d_model=512, num_heads=8, dropout=0.1)
```

For manual GPU configuration:
```python
from myxtts.utils.commons import configure_gpus

configure_gpus()  # Applies the device placement fix
```

## Migration Notes

- **No Code Changes Required**: Existing code will work without modifications
- **Automatic Activation**: Fix is applied automatically when importing models
- **Performance**: No performance impact - only affects device placement handling
- **Compatibility**: Works with both GPU and CPU-only environments