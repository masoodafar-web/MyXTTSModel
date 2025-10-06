# HiFi-GAN Vocoder and Modern Decoding Usage Guide

This guide demonstrates how to use the HiFi-GAN vocoder and modern decoding strategies in MyXTTSModel.

## Key Features

### 1. HiFi-GAN Vocoder
High-quality neural vocoder for mel spectrogram to audio conversion with dramatically improved audio quality.

### 2. Non-Autoregressive Decoder
FastSpeech-style parallel mel generation for faster inference.

### 3. Two-Stage Training
Separate training for mel generation and audio synthesis.

## Quick Start

### Basic Usage with HiFi-GAN Vocoder

```python
from myxtts.config.config import ModelConfig
from myxtts.models.xtts import XTTS
import tensorflow as tf

# Configure model (HiFi-GAN vocoder is used by default)
config = ModelConfig()
config.decoder_strategy = "autoregressive"  # Use autoregressive decoder

# Create model
model = XTTS(config)

# Generate text-to-speech with HiFi-GAN vocoder
text_inputs = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32)
outputs = model.generate(
    text_inputs,
    max_length=100,
    generate_audio=True  # Generate audio directly
)

# Access generated audio
generated_audio = outputs["audio_output"]  # [batch, audio_length, 1]
```

### Non-Autoregressive Generation (Faster Inference)

```python
# Configure for non-autoregressive decoding
config = ModelConfig()
config.decoder_strategy = "non_autoregressive"  # FastSpeech-style

model = XTTS(config)

# Much faster parallel generation
outputs = model.generate(
    text_inputs,
    max_length=100,
    generate_audio=True
)
```

### Two-Stage Training

```python
from myxtts.training.two_stage_trainer import TwoStageTrainer, TwoStageTrainingConfig
from myxtts.config.config import ModelConfig, TrainingConfig

# Configuration
model_config = ModelConfig()
training_config = TrainingConfig()
two_stage_config = TwoStageTrainingConfig(
    stage1_epochs=100,  # Text-to-Mel training
    stage2_epochs=200,  # Vocoder training
    stage1_learning_rate=1e-4,
    stage2_learning_rate=2e-4
)

# Initialize trainer
trainer = TwoStageTrainer(model_config, training_config, two_stage_config)

# Stage 1: Train Text-to-Mel
stage1_history = trainer.train_stage1(train_dataset, val_dataset)

# Stage 2: Train Neural Vocoder
stage2_history = trainer.train_stage2(vocoder_train_dataset, vocoder_val_dataset)

# Create combined model
combined_model = trainer.create_combined_model()
```

## Configuration Options

### Vocoder Configuration

```python
config = ModelConfig()

# HiFi-GAN vocoder parameters (optional, defaults are set)
config.vocoder_upsample_rates = [8, 8, 2, 2]
config.vocoder_upsample_kernel_sizes = [16, 16, 4, 4]
config.vocoder_resblock_kernel_sizes = [3, 7, 11]
config.vocoder_resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
config.vocoder_initial_channel = 512
```

### Decoder Strategy Configuration

```python
config = ModelConfig()

# Decoder strategy
config.decoder_strategy = "autoregressive"  # or "non_autoregressive"

# Enhanced decoder settings
config.decoder_dim = 1536  # Increased for higher quality
config.decoder_layers = 16  # More layers for better modeling
config.decoder_heads = 24   # More attention heads
```

## Audio Processing with Neural Vocoder

```python
from myxtts.utils.audio import AudioProcessor
from myxtts.models.vocoder import Vocoder
import numpy as np

# Initialize audio processor
processor = AudioProcessor(
    sample_rate=22050,
    n_mels=80,
    hop_length=256
)

# Create HiFi-GAN vocoder
config = ModelConfig()
vocoder = Vocoder(config)

# Convert mel to audio with HiFi-GAN vocoder
mel_spectrogram = np.random.randn(80, 100)  # Example mel
audio = processor.mel_to_wav_neural(mel_spectrogram, vocoder)
```

## Performance

### Quality
- **High-quality output**: HiFi-GAN produces natural and clear audio
- **Clean synthesis**: Reduced artifacts and smoother audio
- **Better high frequencies**: Improved clarity and naturalness

### Speed Improvements
- **Non-Autoregressive**: 3-5x faster inference compared to autoregressive
- **Parallel generation**: No sequential dependency for mel frames
- **Reduced latency**: Suitable for real-time applications

## Best Practices

### 1. Training Strategy
```python
# Recommended: Two-stage training for best quality
trainer = TwoStageTrainer(model_config, training_config, two_stage_config)

# Stage 1: Focus on mel quality
stage1_history = trainer.train_stage1(train_dataset)

# Stage 2: Focus on audio quality
stage2_history = trainer.train_stage2(vocoder_dataset)
```

### 2. Inference Strategy
```python
# For highest quality: Autoregressive decoder
config.decoder_strategy = "autoregressive"

# For fastest inference: Non-autoregressive decoder
config.decoder_strategy = "non_autoregressive"
```

### 3. Memory Optimization
```python
config = ModelConfig()
config.enable_gradient_checkpointing = True
config.use_memory_efficient_attention = True
config.enable_mixed_precision = True  # In training config
```

## Troubleshooting

### Common Issues

1. **Dimension Mismatch**: Ensure encoder and decoder dimensions are properly configured
2. **Memory Issues**: Use gradient checkpointing and mixed precision
3. **Quality Issues**: Use two-stage training for best results

### Performance Tips

1. **Use appropriate batch sizes**: Larger batches for vocoder training
2. **Learning rate scheduling**: Lower rates for fine-tuning
3. **Data quality**: High-quality training data essential for neural vocoder

## Example Configurations

### High Quality Configuration
```yaml
# config_hq.yaml
decoder_strategy: "autoregressive"
decoder_dim: 1536
decoder_layers: 16
decoder_heads: 24
vocoder_initial_channel: 512
```

### Fast Inference Configuration
```yaml
# config_fast.yaml
decoder_strategy: "non_autoregressive"
decoder_dim: 1024
decoder_layers: 12
decoder_heads: 16
vocoder_initial_channel: 256
```