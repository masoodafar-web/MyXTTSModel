#!/bin/bash
# Example script to demonstrate TensorBoard training samples logging
# این اسکریپت نحوه استفاده از قابلیت لاگ کردن تصاویر و صداها در تنسوربورد را نشان می‌دهد

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}TensorBoard Training Samples Example${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Step 1: Create training_samples directory
echo -e "${GREEN}Step 1: Creating training_samples directory...${NC}"
mkdir -p training_samples
echo "Created: training_samples/"
echo ""

# Step 2: Create sample images (requires Python)
echo -e "${GREEN}Step 2: Creating sample images...${NC}"
python3 << 'EOF'
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

samples_dir = Path("training_samples")

# Create 3 sample spectrograms
for i in range(3):
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Generate random spectrogram-like data
    time_steps = 200
    mel_bins = 80
    data = np.random.randn(mel_bins, time_steps)
    data = np.cumsum(data, axis=1)  # Make it look more like a spectrogram
    
    im = ax.imshow(data, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title(f'Sample Mel Spectrogram {i+1}')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Mel Bins')
    plt.colorbar(im, ax=ax)
    
    output_path = samples_dir / f'sample_spectrogram_{i+1}.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'Created: {output_path}')

print("Sample images created successfully!")
EOF
echo ""

# Step 3: Create sample audio files (requires Python)
echo -e "${GREEN}Step 3: Creating sample audio files...${NC}"
python3 << 'EOF'
import numpy as np
import tensorflow as tf
from pathlib import Path

samples_dir = Path("training_samples")
sample_rate = 22050
duration = 2  # seconds

# Create 2 sample audio files with different tones
for i in range(2):
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440 * (i + 1)  # A4 and A5 notes
    
    # Main tone
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add harmonics for richer sound
    audio += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
    audio += 0.2 * np.sin(2 * np.pi * frequency * 3 * t)
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.9
    audio = audio.astype(np.float32)
    
    # Save as WAV
    output_path = samples_dir / f'sample_audio_{i+1}.wav'
    audio_tensor = tf.constant(audio, dtype=tf.float32)
    audio_tensor = tf.reshape(audio_tensor, [-1, 1])
    wav_bytes = tf.audio.encode_wav(audio_tensor, sample_rate=sample_rate)
    tf.io.write_file(str(output_path), wav_bytes)
    print(f'Created: {output_path}')

print("Sample audio files created successfully!")
EOF
echo ""

# Step 4: List created files
echo -e "${GREEN}Step 4: Files in training_samples directory:${NC}"
ls -lh training_samples/
echo ""

# Step 5: Show training command
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}Ready to train!${NC}"
echo ""
echo "To start training with TensorBoard logging:"
echo ""
echo "  python train_main.py \\"
echo "      --train-data ../dataset/dataset_train \\"
echo "      --val-data ../dataset/dataset_eval \\"
echo "      --training-samples-dir training_samples \\"
echo "      --training-samples-log-interval 100 \\"
echo "      --tensorboard-log-dir logs \\"
echo "      --batch-size 16 \\"
echo "      --epochs 500"
echo ""
echo "To view results in TensorBoard:"
echo ""
echo "  tensorboard --logdir=logs"
echo ""
echo "Then open: http://localhost:6006"
echo ""
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "${GREEN}Persian / فارسی:${NC}"
echo ""
echo "برای شروع آموزش:"
echo "  python train_main.py --train-data ../dataset/dataset_train"
echo ""
echo "برای دیدن در تنسوربورد:"
echo "  tensorboard --logdir=logs"
echo ""
echo "سپس در مرورگر بروید به: http://localhost:6006"
echo ""
echo -e "${BLUE}================================================${NC}"
