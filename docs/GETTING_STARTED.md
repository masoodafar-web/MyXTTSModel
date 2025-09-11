# Getting Started with MyXTTS

This guide will help you get started with MyXTTS, a TensorFlow-based implementation of the XTTS text-to-speech model.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/masoodafar-web/MyXTTSModel.git
cd MyXTTSModel
```

### 2. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 3. Additional Dependencies (Optional)

For enhanced functionality, install these optional dependencies:

```bash
# For better text processing and phonemization
pip install phonemizer nltk g2p-en

# For training monitoring
pip install wandb tensorboard

# For development
pip install pytest black flake8
```

## Quick Start

### 1. Create a Configuration File

```bash
myxtts create-config --output config.yaml --language en --dataset-path ./data/ljspeech
```

This creates a configuration file with default settings. You can edit it to customize:

- Model architecture parameters
- Training hyperparameters 
- Data processing settings
- Audio parameters

### 2. Prepare Your Dataset

MyXTTS supports LJSpeech format out of the box. Your dataset should have:

```
dataset/
├── metadata.csv          # Format: ID|transcription|normalized_transcription
└── wavs/
    ├── audio1.wav
    ├── audio2.wav
    └── ...
```

For LJSpeech dataset:

```bash
# The dataset will be automatically downloaded during training
myxtts dataset-info --data-path ./data/ljspeech
```

### 3. Train the Model

```bash
myxtts train --config config.yaml --data-path ./data/ljspeech --epochs 100
```

Training parameters can be adjusted in the config file or via command line:

```bash
myxtts train \
  --config config.yaml \
  --data-path ./data/ljspeech \
  --epochs 200 \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --checkpoint-dir ./checkpoints
```

### 4. Generate Speech

Once training is complete, you can synthesize speech:

```bash
# Basic synthesis
myxtts synthesize \
  --config config.yaml \
  --checkpoint ./checkpoints/checkpoint_best \
  --text "Hello, this is MyXTTS!" \
  --output hello.wav

# Voice cloning
myxtts clone-voice \
  --config config.yaml \
  --checkpoint ./checkpoints/checkpoint_best \
  --text "This is my cloned voice" \
  --reference-audio reference.wav \
  --output cloned.wav
```

## Python API

You can also use MyXTTS programmatically:

```python
from myxtts.config.config import XTTSConfig
from myxtts import get_xtts_model, get_trainer, get_inference_engine

# Load configuration
config = XTTSConfig.from_yaml("config.yaml")

# Training
model = get_xtts_model()(config.model)
trainer = get_trainer()(config, model)
train_dataset, val_dataset = trainer.prepare_datasets("./data/ljspeech")
trainer.train(train_dataset, val_dataset)

# Inference
inference = get_inference_engine()(config, checkpoint_path="./checkpoints/best")
result = inference.synthesize("Hello world!")
inference.save_audio(result["audio"], "output.wav")
```

## Configuration Guide

### Model Configuration

Key model parameters in `config.yaml`:

```yaml
model:
  # Architecture
  text_encoder_dim: 512      # Text encoder dimension
  decoder_dim: 1024          # Decoder dimension
  text_encoder_layers: 6     # Number of transformer layers
  
  # Audio settings
  sample_rate: 22050         # Audio sample rate
  n_mels: 80                 # Mel spectrogram channels
  
  # Voice cloning
  use_voice_conditioning: true
  speaker_embedding_dim: 256
  
  # Languages
  languages: [en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, ja, hu, ko]
```

### Training Configuration

```yaml
training:
  epochs: 1000
  learning_rate: 1e-4
  batch_size: 32
  optimizer: adamw
  scheduler: noam
  
  # Loss weights
  mel_loss_weight: 45.0
  kl_loss_weight: 1.0
  
  # Checkpointing
  save_step: 25000
  val_step: 5000
  checkpoint_dir: "./checkpoints"
```

### Data Configuration

```yaml
data:
  dataset_path: "./data/ljspeech"
  dataset_name: "ljspeech"
  language: "en"
  
  # Audio processing
  sample_rate: 22050
  normalize_audio: true
  trim_silence: true
  
  # Text processing
  text_cleaners: ["english_cleaners"]
  add_blank: true
  
  # Training splits
  train_split: 0.9
  val_split: 0.1
  batch_size: 32
```

## Tips and Best Practices

### Training

1. **Start Small**: Begin with a smaller model (256-512 dimensions) for faster experimentation
2. **Monitor Training**: Use tensorboard or wandb to track loss curves
3. **GPU Memory**: Adjust batch size based on available GPU memory
4. **Checkpointing**: Save checkpoints frequently to avoid losing progress

### Dataset Preparation

1. **Audio Quality**: Use high-quality audio (22kHz, 16-bit recommended)
2. **Text Cleaning**: Ensure transcriptions are accurate and properly formatted
3. **Audio Length**: Keep clips between 1-10 seconds for best results
4. **Consistency**: Use consistent recording conditions and speaker

### Voice Cloning

1. **Reference Audio**: Use 3-5 seconds of clean reference audio
2. **Same Language**: Reference and target text should be in the same language
3. **Quality**: Higher quality reference audio gives better cloning results
4. **Multiple References**: Use different reference samples for variety

### Performance Optimization

1. **Mixed Precision**: Enable for faster training on modern GPUs
2. **Batch Size**: Larger batches generally give better gradients
3. **Learning Rate**: Use learning rate scheduling for better convergence
4. **Gradient Clipping**: Helps with training stability

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or model dimensions
2. **Slow Training**: Enable mixed precision, increase batch size
3. **Poor Audio Quality**: Check data preprocessing, increase model size
4. **Training Instability**: Adjust learning rate, enable gradient clipping

### Getting Help

- Check the examples in `examples/` directory
- Run tests with `python tests/run_tests.py`
- Review configuration options in `examples/config_example.yaml`

## Next Steps

1. **Experiment with Languages**: Try training on different language datasets
2. **Fine-tune**: Fine-tune pre-trained models on your specific data
3. **Voice Cloning**: Explore zero-shot voice cloning capabilities
4. **Deployment**: Integrate trained models into applications

For more advanced usage, check out the full documentation and examples in the repository.