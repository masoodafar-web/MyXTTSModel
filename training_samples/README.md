# Training Samples Directory

This directory is used for logging images and audio to TensorBoard during training.

## Usage

Place your sample files here:

### Images
- Spectrograms (`.png`, `.jpg`)
- Visualizations
- Training progress images
- Any reference images you want to track

Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`

### Audio
- Reference audio samples (`.wav`)
- Audio files for comparison
- Generated samples you want to monitor

Supported formats: `.wav` (recommended), `.mp3`, `.flac`, `.ogg`

## Example Structure

```
training_samples/
├── sample_spectrogram_1.png
├── sample_spectrogram_2.png
├── reference_audio_1.wav
├── reference_audio_2.wav
└── visualization.jpg
```

## How It Works

During training, files in this directory are automatically:
1. Detected based on their extension
2. Logged to TensorBoard at regular intervals (default: every 100 steps)
3. Displayed in TensorBoard's IMAGES and AUDIO tabs

## Viewing in TensorBoard

1. Start TensorBoard:
   ```bash
   tensorboard --logdir=logs
   ```

2. Open `http://localhost:6006` in your browser

3. Navigate to:
   - **IMAGES tab** → Look for `training_samples/image/*`
   - **AUDIO tab** → Look for `training_samples/audio/*`

## Configuration

Control logging behavior with command-line arguments:

```bash
# Change this directory
python train_main.py --training-samples-dir ./my_samples

# Change logging interval (steps)
python train_main.py --training-samples-log-interval 200

# Disable logging
python train_main.py --training-samples-log-interval 0
```

## Tips

- **Images**: Use PNG format for best quality
- **Audio**: Use WAV format (no additional libraries needed)
- **File names**: Use descriptive names (they appear in TensorBoard)
- **File size**: Keep files reasonably sized for faster loading
- **Updates**: You can add/remove files during training

## Persian / فارسی

این پوشه برای لاگ کردن تصاویر و صداها در تنسوربورد استفاده می‌شود.

فایل‌های خود را اینجا قرار دهید:
- تصاویر (PNG, JPG)
- فایل‌های صوتی (WAV)

برای دیدن در تنسوربورد:
```bash
tensorboard --logdir=logs
```

سپس به `http://localhost:6006` بروید.
