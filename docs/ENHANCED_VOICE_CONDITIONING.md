# Enhanced Voice Conditioning Implementation

This document describes the enhanced voice conditioning implementation that replaces the original audio encoder with pre-trained speaker encoders and implements real contrastive/GE2E loss for improved voice cloning quality.

## Overview

The original audio encoder (myxtts/models/xtts.py:184) consisted of simple convolutional layers and Transformer blocks. This implementation replaces it with pre-trained speaker encoders (Resemblyzer, ECAPA-TDNN, or Coqui models) with frozen weights, bringing voice similarity stability to Coqui-level performance.

## Key Components

### 1. PretrainedSpeakerEncoder

Located in `myxtts/models/speaker_encoder.py`, this class supports three types of pre-trained speaker encoders:

- **ECAPA-TDNN**: State-of-the-art speaker recognition architecture
- **Resemblyzer**: LSTM-based encoder similar to the original Resemblyzer
- **Coqui**: Convolutional encoder inspired by Coqui TTS models

### 2. ContrastiveSpeakerLoss

Implements GE2E-style contrastive learning that:
- Minimizes distance between embeddings of the same speaker
- Maximizes distance between embeddings of different speakers
- Uses temperature scaling and margin-based learning

### 3. Enhanced Configuration

New configuration parameters in `ModelConfig`:
- `use_pretrained_speaker_encoder`: Enable enhanced voice conditioning
- `speaker_encoder_type`: Choose encoder architecture
- `freeze_speaker_encoder`: Freeze pre-trained weights

## Usage

### Training

Enable enhanced voice conditioning in `train_main.py`:

```python
# In the build_config function:
use_pretrained_speaker_encoder=True,  # Enable enhanced voice conditioning
speaker_encoder_type="ecapa_tdnn",    # Choose: "ecapa_tdnn", "resemblyzer", "coqui"
freeze_speaker_encoder=True,          # Keep pre-trained weights frozen
```

Then run training as usual:
```bash
python train_main.py --train-data ../dataset/dataset_train --val-data ../dataset/dataset_eval
```

### Inference

Enable enhanced voice conditioning with command line arguments:

```bash
# Basic enhanced voice conditioning
python inference_main.py \
    --text "Hello world" \
    --reference-audio speaker.wav \
    --use-pretrained-speaker-encoder \
    --speaker-encoder-type ecapa_tdnn

# Advanced voice conditioning with custom parameters
python inference_main.py \
    --text "Hello world" \
    --reference-audio speaker.wav \
    --use-pretrained-speaker-encoder \
    --speaker-encoder-type ecapa_tdnn
```

## Speaker Encoder Types

### ECAPA-TDNN (Recommended)
- State-of-the-art speaker recognition performance
- SE-Res2Block architecture with dilated convolutions
- Statistical pooling for robust embeddings
- Best for high-quality voice cloning

### Resemblyzer
- LSTM-based architecture
- Compatible with Resemblyzer pre-trained models
- Good balance of quality and speed
- Suitable for real-time applications

### Coqui
- Convolutional architecture
- Inspired by Coqui TTS speaker encoders
- Fast inference
- Good for production environments

## Loss Function Details

The enhanced implementation includes a real contrastive loss instead of just a parameter:

```python
# Contrastive loss computation
similarities = tf.matmul(embeddings, embeddings, transpose_b=True) / temperature
positive_loss = -tf.reduce_sum(positive_similarities) / num_positive_pairs
negative_loss = tf.reduce_sum(relu(negative_similarities - margin)) / num_negative_pairs
total_loss = positive_loss + negative_loss
```

This encourages:
- Similar speakers to have close embeddings (positive_loss)
- Different speakers to have distant embeddings (negative_loss)
- Temperature controls the sharpness of similarities
- Margin creates a buffer zone for negative pairs

## Performance Benefits

1. **Better Voice Similarity**: Pre-trained speaker encoders provide more robust voice representations
2. **Stable Training**: Frozen pre-trained weights prevent instability during training
3. **Real Loss Function**: Contrastive loss actually trains speaker similarity instead of being unused
4. **Flexible Architecture**: Multiple encoder types for different use cases
5. **Backward Compatible**: Original encoder still available when enhanced mode is disabled

## Migration Guide

To migrate existing models:

1. **Existing Models**: Continue to work without changes (enhanced mode disabled by default)
2. **New Training**: Set `use_pretrained_speaker_encoder=True` in training config
3. **Fine-tuning**: Can enable enhanced mode for existing checkpoints
4. **Inference**: Use new command line flags to enable enhanced voice conditioning

## Technical Implementation

The implementation maintains backward compatibility while adding new capabilities:

1. **AudioEncoder**: Checks `use_pretrained_speaker_encoder` flag to choose encoder type
2. **Training**: Uses existing loss infrastructure with new contrastive loss component  
3. **Configuration**: Extends existing config with new parameters
4. **Inference**: Adds command line arguments for enhanced features

The enhanced voice conditioning brings the model's voice cloning capabilities to professional-grade quality while maintaining the flexibility and ease of use of the original implementation.
