# TensorBoard Spectrogram Logging

This document describes the spectrogram logging feature in MyXTTS, which helps visualize model training progress by comparing target and predicted mel spectrograms.

## Overview

During training, the system automatically logs spectrogram comparisons to TensorBoard at configurable intervals. This provides visual feedback on how well the model is learning to predict mel spectrograms.

## Features

- **Side-by-side comparison**: Target, prediction, and difference shown together
- **Matplotlib-based visualization**: Professional appearance with labels and colorbars
- **Configurable intervals**: Control logging frequency to balance detail vs. overhead
- **Multiple sample support**: Log one or more samples per event
- **Reference sample mode**: Track the same sample across training for consistent comparison

## Configuration

Configure spectrogram logging in `myxtts/config/config.py` or via the training configuration:

```python
from myxtts.config.config import TrainingConfig

training_config = TrainingConfig(
    # Logging interval (0 disables spectrogram logging)
    spectrogram_log_interval=5,  # Log every 5 steps
    
    # Number of samples to log per event
    spectrogram_log_num_examples=1,
    
    # Fixed sample index (None = sequential from batch)
    spectrogram_log_example_index=None,
    
    # Source of samples: "batch", "train", or "val"
    spectrogram_reference_subset="batch",
    
    # Dataset index when using fixed sample mode
    spectrogram_reference_index=None,
)
```

## Configuration Options

### `spectrogram_log_interval`
- **Type**: `int`
- **Default**: `5`
- **Description**: Log spectrograms every N training steps. Set to 0 to disable.
- **Recommendation**: 
  - Small datasets: 5-10
  - Large datasets: 50-100
  - Production: 100-500

### `spectrogram_log_num_examples`
- **Type**: `int`
- **Default**: `1`
- **Description**: Number of samples to visualize per logging event
- **Recommendation**: Keep at 1-3 to avoid cluttering TensorBoard

### `spectrogram_log_example_index`
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: 
  - `None`: Cycle through batch samples sequentially
  - `int`: Always use the same index from the batch

### `spectrogram_reference_subset`
- **Type**: `str`
- **Default**: `"batch"`
- **Options**: `"batch"`, `"train"`, `"val"`
- **Description**:
  - `"batch"`: Use samples from the current training batch
  - `"train"`: Use a fixed sample from the training set
  - `"val"`: Use a fixed sample from the validation set

### `spectrogram_reference_index`
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: When using `"train"` or `"val"` mode, which dataset index to use

## Visualization Output

Each logged spectrogram comparison contains three subplots:

1. **Target Spectrogram** (Left)
   - Ground truth mel spectrogram from the dataset
   - Shows what the model should predict
   - Colormap: viridis (blue to yellow)

2. **Predicted Spectrogram** (Center)
   - Model's prediction for the same input
   - Shows what the model actually produces
   - Colormap: viridis (blue to yellow)

3. **Absolute Difference** (Right)
   - Pixel-wise absolute difference: |target - prediction|
   - Shows where the model makes errors
   - Colormap: hot (black to red to white)
   - Lower values (darker) = better prediction

All subplots include:
- Frame count on X-axis
- Mel bin number on Y-axis
- Colorbar showing value range
- Descriptive titles

## Viewing in TensorBoard

1. Start TensorBoard:
```bash
tensorboard --logdir=./checkpoints/tensorboard
```

2. Open browser to `http://localhost:6006`

3. Navigate to the **IMAGES** tab

4. Look for tags like:
   - `train/spectrogram_0/comparison`
   - `train/spectrogram_ref_LJ001-0001/comparison`
   - `val/spectrogram_0/comparison`

5. Use the slider to see how predictions improve over training steps

## Best Practices

### For Development
```python
spectrogram_log_interval=5  # Frequent logging
spectrogram_log_num_examples=2  # Multiple samples
spectrogram_reference_subset="batch"  # See variety
```

### For Production Training
```python
spectrogram_log_interval=100  # Less frequent
spectrogram_log_num_examples=1  # Single sample
spectrogram_reference_subset="val"  # Consistent reference
spectrogram_reference_index=0  # Track same sample
```

### For Debugging
```python
spectrogram_log_interval=1  # Every step
spectrogram_log_num_examples=1
spectrogram_reference_subset="train"
spectrogram_reference_index=0  # Track specific problematic sample
```

## Performance Considerations

- **Overhead**: ~100-200ms per logged sample
- **Frequency**: Logging every 5 steps with 1 sample adds ~2% overhead
- **Memory**: Each comparison image is ~100-200KB
- **Recommendation**: Adjust `spectrogram_log_interval` based on training speed

## Interpreting Results

### Early Training
- High difference values (bright red in difference plot)
- Predicted spectrogram may look noisy or flat
- This is normal

### Mid Training
- Difference values decreasing
- General structure visible in predictions
- Some details may be missing

### Late Training / Converged
- Low difference values (dark in difference plot)
- Predicted spectrogram closely matches target
- Fine details captured

### Signs of Problems

1. **Flat predictions**: Model may not be learning
2. **High-frequency noise**: May need regularization
3. **Consistent bias**: Check normalization/preprocessing
4. **Sudden divergence**: Learning rate may be too high

## Example Usage

```python
from myxtts.config.config import XTTSConfig, TrainingConfig
from myxtts.training.trainer import XTTSTrainer

# Configure logging
config = XTTSConfig()
config.training.spectrogram_log_interval = 10
config.training.spectrogram_log_num_examples = 1

# Create trainer
trainer = XTTSTrainer(config=config)

# Train (spectrograms logged automatically)
trainer.train(train_dataset, val_dataset)
```

## Technical Details

### Implementation
- Located in: `myxtts/training/trainer.py`
- Method: `_log_spectrogram_from_batch()`
- Helper: `_create_spectrogram_comparison_image()`

### Image Generation Process
1. Extract target and predicted spectrograms
2. Normalize to same frame length
3. Create matplotlib figure with 3 subplots
4. Render to PNG buffer
5. Convert to TensorFlow tensor
6. Log to TensorBoard via `tf.summary.image()`

### Memory Management
- Matplotlib figures are properly closed after rendering
- BytesIO buffers are explicitly closed
- No memory leaks in long training runs

## Troubleshooting

### No spectrograms in TensorBoard
- Check `spectrogram_log_interval > 0`
- Verify TensorBoard is reading the correct log directory
- Ensure training has progressed past the first logging interval

### Spectrograms look wrong
- Check data preprocessing pipeline
- Verify mel spectrogram parameters match between training and visualization
- Check for NaN or Inf values in data

### Performance issues
- Increase `spectrogram_log_interval`
- Reduce `spectrogram_log_num_examples`
- Consider disabling for very large models

## Related Files

- `myxtts/training/trainer.py` - Main implementation
- `myxtts/config/config.py` - Configuration options
- `requirements.txt` - Required dependencies (matplotlib, Pillow)

## Dependencies

- matplotlib >= 3.5.0
- Pillow >= 9.0.0
- TensorFlow >= 2.12.0
- numpy >= 1.21.0
