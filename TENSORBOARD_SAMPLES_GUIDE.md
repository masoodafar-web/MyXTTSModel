# TensorBoard Training Samples Logging

## Overview

This feature allows you to log images and audio files from a `training_samples` directory directly to TensorBoard during training. This is useful for monitoring additional samples, visualizations, or reference data alongside your model's training metrics.

## Persian Summary / خلاصه فارسی

این قابلیت به شما امکان می‌دهد تا تصاویر و فایل‌های صوتی از پوشه `training_samples` را مستقیماً به تنسوربورد در حین آموزش لاگ کنید. برای مشاهده تصاویر و صداهای تولید شده در تنسوربورد بسیار مفید است.

## Features

### 1. Image Logging
- Automatically logs all image files from `training_samples` directory to TensorBoard
- Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`
- Images appear under `training_samples/image/*` in TensorBoard

### 2. Audio Logging
- Automatically logs all audio files from `training_samples` directory to TensorBoard
- Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg` (requires soundfile library)
- Audio files appear under `training_samples/audio/*` in TensorBoard

### 3. Generated Audio Samples
- During training, the model generates evaluation audio samples automatically
- These samples are logged to TensorBoard under `text2audio_eval/*`
- Includes both audio waveform and mel spectrogram visualization

## Usage

### Basic Setup

1. **Create the training_samples directory:**
   ```bash
   mkdir training_samples
   ```

2. **Add your images and audio files:**
   ```bash
   # Copy your images
   cp your_images/*.png training_samples/
   
   # Copy your audio files
   cp your_audio/*.wav training_samples/
   ```

3. **Start training with default settings:**
   ```bash
   python train_main.py --train-data ../dataset/dataset_train --val-data ../dataset/dataset_eval
   ```

### Custom Configuration

**Change the training samples directory:**
```bash
python train_main.py \
    --training-samples-dir ./my_samples \
    --train-data ../dataset/dataset_train
```

**Change logging interval (log every N steps):**
```bash
python train_main.py \
    --training-samples-log-interval 500 \
    --train-data ../dataset/dataset_train
```

**Disable training samples logging:**
```bash
python train_main.py \
    --training-samples-log-interval 0 \
    --train-data ../dataset/dataset_train
```

**Custom TensorBoard log directory:**
```bash
python train_main.py \
    --tensorboard-log-dir ./logs \
    --train-data ../dataset/dataset_train
```

## View Results in TensorBoard

### Start TensorBoard

**Default (logs in checkpoint directory):**
```bash
tensorboard --logdir=./checkpointsmain/tensorboard
```

**Custom log directory:**
```bash
tensorboard --logdir=./logs
```

**Persian / فارسی:**
```bash
# اگر آدرس لاگ شما logs است:
tensorboard --logdir=logs
```

Then open your browser and navigate to: `http://localhost:6006`

### Navigate to View Content

1. **Training Samples Images:**
   - Go to the "IMAGES" tab in TensorBoard
   - Look for items under `training_samples/image/`

2. **Training Samples Audio:**
   - Go to the "AUDIO" tab in TensorBoard
   - Look for items under `training_samples/audio/`

3. **Generated Evaluation Audio:**
   - Go to the "AUDIO" tab in TensorBoard
   - Look for items under `text2audio_eval/`

4. **Mel Spectrograms:**
   - Go to the "IMAGES" tab in TensorBoard
   - Look for items under `train/spectrogram_*` or `text2audio_eval/mel_*`

## Configuration Parameters

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--training-samples-dir` | `training_samples` | Directory containing sample images/audio |
| `--training-samples-log-interval` | `100` | Log samples every N steps (0 to disable) |
| `--tensorboard-log-dir` | `{checkpoint_dir}/tensorboard` | TensorBoard log directory |

### Configuration Class (TrainingConfig)

```python
# In your training configuration
training_config = TrainingConfig(
    training_samples_dir="training_samples",  # Directory path
    training_samples_log_interval=100,        # Steps between logging
)
```

## File Organization

```
your_project/
├── training_samples/           # Your sample files
│   ├── sample_image_1.png     # Images to log
│   ├── sample_image_2.jpg
│   ├── sample_audio_1.wav     # Audio to log
│   └── sample_audio_2.wav
├── checkpointsmain/            # Training checkpoints
│   └── tensorboard/            # TensorBoard logs (default)
│       └── events.out.tfevents.*
└── logs/                       # Custom TensorBoard logs (optional)
    └── events.out.tfevents.*
```

## Troubleshooting

### Images Not Appearing

1. **Check directory exists:**
   ```bash
   ls -la training_samples/
   ```

2. **Verify file formats:**
   - Only `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif` are supported
   - Files must have correct extensions

3. **Check logging interval:**
   - By default, logs every 100 steps
   - First log appears at step 100, not step 0
   - Adjust with `--training-samples-log-interval`

### Audio Not Appearing

1. **Check audio format:**
   - `.wav` files work without additional libraries
   - Other formats (`.mp3`, `.flac`, `.ogg`) require `soundfile` library:
     ```bash
     pip install soundfile
     ```

2. **Verify audio file integrity:**
   ```python
   import tensorflow as tf
   audio_bytes = tf.io.read_file('training_samples/sample.wav')
   audio, sr = tf.audio.decode_wav(audio_bytes)
   print(f"Sample rate: {sr}, Shape: {audio.shape}")
   ```

### TensorBoard Shows Wrong Directory

1. **Check the actual TensorBoard directory:**
   - Look at trainer initialization logs
   - Default: `{checkpoint_dir}/tensorboard`
   - With `--tensorboard-log-dir`: uses custom path

2. **Use correct logdir argument:**
   ```bash
   # If using default
   tensorboard --logdir=./checkpointsmain/tensorboard
   
   # If using custom (e.g., --tensorboard-log-dir ./logs)
   tensorboard --logdir=./logs
   ```

## Examples

### Example 1: Basic Training with Samples

```bash
# Prepare samples
mkdir training_samples
cp my_spectrograms/*.png training_samples/
cp my_reference_audio/*.wav training_samples/

# Start training
python train_main.py \
    --train-data ../dataset/dataset_train \
    --val-data ../dataset/dataset_eval \
    --batch-size 16 \
    --epochs 500

# View in TensorBoard
tensorboard --logdir=./checkpointsmain/tensorboard
```

### Example 2: Custom Configuration

```bash
# Create custom directories
mkdir my_samples my_logs

# Add your files
cp samples/*.png my_samples/
cp samples/*.wav my_samples/

# Start training with custom paths
python train_main.py \
    --train-data ../dataset/dataset_train \
    --val-data ../dataset/dataset_eval \
    --training-samples-dir my_samples \
    --training-samples-log-interval 200 \
    --tensorboard-log-dir my_logs \
    --batch-size 32

# View in TensorBoard
tensorboard --logdir=my_logs
```

### Example 3: High-Frequency Logging

```bash
# Log samples every 10 steps for detailed monitoring
python train_main.py \
    --train-data ../dataset/dataset_train \
    --val-data ../dataset/dataset_eval \
    --training-samples-log-interval 10 \
    --batch-size 16
```

## Benefits

1. **Visual Monitoring:** See training progress through reference images
2. **Audio Quality Tracking:** Listen to generated samples during training
3. **Comparison:** Compare model outputs with reference samples
4. **Debugging:** Identify issues by visualizing intermediate results
5. **Documentation:** Automatically document training results

## Technical Details

### Implementation

The feature is implemented in `myxtts/training/trainer.py`:
- `_log_training_samples_to_tensorboard()` method handles the logging
- Called during training at specified intervals
- Uses TensorFlow's `tf.summary.image()` and `tf.summary.audio()` APIs

### Performance Impact

- **Minimal:** Logging only occurs at specified intervals
- **Non-blocking:** Uses TensorFlow's async summary writer
- **Efficient:** Only reads files when logging is triggered

### Supported File Types

**Images:**
- PNG (lossless, recommended)
- JPEG (compressed)
- BMP (uncompressed)
- GIF (animated - only first frame is logged)

**Audio:**
- WAV (recommended, no extra dependencies)
- MP3, FLAC, OGG (requires `soundfile` library)

## Related Features

- **Spectrogram Logging:** Automatic mel spectrogram visualization during training
- **Text-to-Audio Evaluation:** Generates and logs audio samples from evaluation texts
- **WandB Integration:** Can also log to Weights & Biases (use `--use-wandb`)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review TensorBoard documentation: https://www.tensorflow.org/tensorboard
3. Open an issue in the repository

---

**Persian / فارسی:**

برای مشاهده تصاویر و صداها در تنسوربورد:
1. فولدر `training_samples` را بسازید
2. تصاویر و صداهای خود را در آن کپی کنید
3. آموزش را شروع کنید
4. تنسوربورد را با دستور `tensorboard --logdir=logs` باز کنید
5. در مرورگر به `localhost:6006` بروید
6. تب IMAGES برای تصاویر و تب AUDIO برای صداها
