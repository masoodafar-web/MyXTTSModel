# XTTS Duration Predictor and Attention Alignment Support

## Problem Resolved

The MyXTTS model was missing duration predictor and attention alignment outputs, while the loss functions (`myxtts/training/losses.py`) expected these outputs for training stability. This mismatch led to:

- Disabled duration and attention loss components (weights set to 0.0)
- Potential training instability and alignment issues
- Repetition and discontinuity problems in generated speech

## Solution Implemented

### 1. Added Duration Predictor
- **File**: `myxtts/models/xtts.py` - TextEncoder class
- **Change**: Added `duration_predictor` Dense layer with ReLU activation
- **Output**: Returns duration predictions for each text token during training

### 2. Enhanced Attention Weight Extraction
- **File**: `myxtts/models/layers.py` - MultiHeadAttention and TransformerBlock classes
- **Change**: Added `return_attention_weights` parameter to optionally return attention weights
- **Output**: Cross-attention weights from decoder transformer blocks

### 3. Updated Model Architecture
- **File**: `myxtts/models/xtts.py` - XTTS class
- **Training Mode**: Returns `duration_pred` and `attention_weights`
- **Inference Mode**: Maintains backward compatibility (no additional outputs)

### 4. Enabled Loss Components
- **File**: `myxtts/training/losses.py`
- **Change**: Enabled attention_loss_weight (0.1) and duration_loss_weight (0.1)
- **File**: `myxtts/config/config.py`
- **Change**: Added attention_loss_weight to training configuration

## Technical Implementation Details

```python
# Duration predictor in TextEncoder
self.duration_predictor = tf.keras.layers.Dense(1, activation='relu', name="duration_predictor")

# Attention weights extraction in MultiHeadAttention
def call(self, ..., return_attention_weights=False):
    if return_attention_weights:
        return output, attention_weights
    else:
        return output

# Model outputs during training
if training:
    outputs["duration_pred"] = duration_pred
    outputs["attention_weights"] = attention_weights
```

## Validation Results

✅ **Unit Tests**: All new functionality tested and working
✅ **Integration Tests**: Training step with gradient computation successful
✅ **Backward Compatibility**: Inference mode unchanged
✅ **Loss Computation**: All loss components active and balanced

## Benefits

1. **Training Stability**: Monotonic attention loss guides proper alignment
2. **Duration Modeling**: Duration predictor helps with speech timing
3. **Reduced Artifacts**: Less repetition and skipping in generated speech
4. **Configurable**: Loss weights can be tuned per use case
5. **Compatible**: No breaking changes to existing inference code

## Usage

The enhanced model automatically provides duration and attention outputs during training. No code changes needed for basic usage:

```python
# Training mode - includes duration_pred and attention_weights
outputs = model(text_inputs, mel_inputs, training=True)

# Inference mode - compatible with existing code
outputs = model(text_inputs, mel_inputs, training=False)
```

## Files Modified

- `myxtts/models/layers.py` - Enhanced attention layers
- `myxtts/models/xtts.py` - Added duration predictor and outputs
- `myxtts/training/losses.py` - Enabled attention/duration loss weights
- `myxtts/config/config.py` - Added attention loss weight configuration

## Test Files Added

- `test_attention_duration_outputs.py` - Comprehensive unit tests
- `test_integration_attention_duration.py` - Training integration test
- `demo_attention_duration_improvements.py` - Demonstration script

The implementation resolves the Persian issue statement:
> پشتیبانی از پیشبینیگر مدت و توجه صریح: در myxtts/models/xtts.py هیچ duration predictor یا attention alignment خروجی نمیدهد، درحالیکه در myxtts/training/losses.py ضرر duration/attention پیشبینی شده.

Translation: Support for duration predictor and explicit attention - the model now provides the missing outputs that the loss functions expected.