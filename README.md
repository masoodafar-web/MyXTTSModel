# MyXTTS: TensorFlow-based XTTS Implementation

A comprehensive TensorFlow implementation of XTTS (eXtreme Text-To-Speech) with multilingual support, voice cloning capabilities, and LJSpeech dataset compatibility.

## Features

- 🎯 **XTTS Architecture**: Complete implementation with text encoder, audio encoder, and mel decoder
- 🌍 **Multilingual Support**: Built-in support for 16+ languages
- 🎭 **Voice Cloning**: Clone voices from reference audio samples
- 📊 **LJSpeech Compatible**: Ready-to-use dataset generator for LJSpeech format
- ⚡ **TensorFlow 2.x**: Optimized for modern TensorFlow with mixed precision training
- 🛠️ **Production Ready**: Complete training and inference pipelines
- 📱 **CLI Interface**: Easy-to-use command-line tools
- 🔧 **Configurable**: Extensive configuration system for all model aspects

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/masoodafar-web/MyXTTSModel.git
cd MyXTTSModel

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Alternative: Using trainTestFile.py (Flexible Configuration)

For even more flexibility, you can use the `trainTestFile.py` script in the root directory, which allows both programmatic configuration (without YAML files) and optional YAML-based configuration:

#### 1. **Train with Programmatic Configuration** (No YAML required):
```bash
python trainTestFile.py --mode train --data-path ./data/ljspeech --epochs 100 --batch-size 16
```

#### 2. **Create and Use YAML Configuration**:
```bash
# Create a configuration file
python trainTestFile.py --mode create-config --output my_config.yaml --epochs 200 --language es

# Train using the configuration file
python trainTestFile.py --mode train --config my_config.yaml

# Override specific parameters from YAML
python trainTestFile.py --mode train --config my_config.yaml --epochs 50 --batch-size 8
```

#### 3. **Test/Inference Mode**:
```bash
python trainTestFile.py --mode test --checkpoint ./checkpoints/best.ckpt --text "Hello world"
```

### Basic Usage

1. **Create Configuration**:
```bash
myxtts create-config --output config.yaml --language en --dataset-path ./data/ljspeech
```

2. **Train Model**:
```bash
myxtts train --config config.yaml --data-path ./data/ljspeech
```

3. **Synthesize Speech**:
```bash
myxtts synthesize --config config.yaml --checkpoint ./checkpoints/checkpoint_best --text "Hello, this is MyXTTS!"
```

4. **Clone Voice**:
```bash
myxtts clone-voice --config config.yaml --checkpoint ./checkpoints/checkpoint_best --text "Hello with cloned voice" --reference-audio reference.wav
```

## Text Processing and Tokenization

MyXTTS supports two tokenization approaches:

### 1. Custom Symbol-based Tokenizer (Default)
- Character-level tokenization with phoneme support
- Smaller vocabulary (~256 symbols)
- Suitable for single-language or limited multilingual use

### 2. NLLB-200 Tokenizer (Recommended for Multilingual)
- Pre-trained multilingual tokenizer from Facebook
- Large vocabulary (256,000+ tokens)
- Excellent multilingual coverage (200+ languages)

```python
from myxtts.utils.text import TextProcessor
from myxtts.config.config import ModelConfig

# Configure for NLLB-200 tokenizer
config = ModelConfig()
config.tokenizer_type = "nllb"
config.tokenizer_model = "facebook/nllb-200-distilled-600M"
config.text_vocab_size = 256_256

# Create text processor
processor = TextProcessor(
    tokenizer_type="nllb",
    tokenizer_model="facebook/nllb-200-distilled-600M"
)

# Tokenize text
tokens = processor.text_to_sequence("Hello, multilingual world!")
```

## Architecture

MyXTTS implements a transformer-based architecture similar to the original XTTS:

- **Text Encoder**: Processes text input with multi-head attention
- **Audio Encoder**: Extracts speaker embeddings from reference audio
- **Mel Decoder**: Generates mel spectrograms with cross-attention
- **Voice Conditioning**: Enables zero-shot voice cloning

## Supported Languages

English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh), Japanese (ja), Hungarian (hu), Korean (ko)

## Dataset Format

MyXTTS supports LJSpeech format out of the box:
```
dataset/
├── metadata.csv          # Format: ID|transcription|normalized_transcription
└── wavs/
    ├── audio1.wav
    ├── audio2.wav
    └── ...
```

## Configuration

All model aspects are configurable through YAML files:

```yaml
model:
  sample_rate: 22050
  n_mels: 80
  use_voice_conditioning: true
  languages: [en, es, fr, de, it]

data:
  dataset_path: "./data/ljspeech"
  batch_size: 32
  language: "en"

training:
  epochs: 1000
  learning_rate: 1e-4
  checkpoint_dir: "./checkpoints"
```

## Python API

```python
from myxtts import XTTS, XTTSConfig, XTTSInference

# Load configuration
config = XTTSConfig.from_yaml("config.yaml")

# Create and train model
model = XTTS(config.model)
trainer = XTTSTrainer(config, model)
trainer.train(train_dataset, val_dataset)

# Inference
inference = XTTSInference(config, checkpoint_path="model.ckpt")
result = inference.synthesize("Hello world!", reference_audio="speaker.wav")
```

## Examples

Check the `examples/` directory for:
- Basic training script
- Inference examples  
- Voice cloning demos
- Configuration templates

## Requirements

- Python 3.8+
- TensorFlow 2.12+
- librosa, soundfile, numpy
- phonemizer (optional, for better text processing)
- espeak-ng (for phonemization)

## Performance

- **Training**: Supports mixed precision for faster training on modern GPUs
- **Inference**: Optimized for real-time synthesis
- **Memory**: Configurable batch sizes and model dimensions

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original XTTS paper and implementation
- LJSpeech dataset creators
- TensorFlow team for the excellent framework