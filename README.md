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

### Alternative: Using trainTestFile.py (Maximum Flexibility)

For maximum control and flexibility, you can use the `trainTestFile.py` script in the root directory. This script supports:
- **Pure programmatic configuration** (no YAML files required)
- **YAML-based configuration** with CLI parameter overrides
- **Smart dataset detection** (automatically skips download if dataset exists)

#### 1. **Train with Direct Parameters** (No YAML required):
```bash
# Train immediately with just CLI parameters - no config file needed
python trainTestFile.py --mode train --data-path ./data/ljspeech --epochs 100 --batch-size 16

# Works with existing datasets - automatically detects and skips download
python trainTestFile.py --mode train --data-path ./my_existing_dataset --epochs 200
```

#### 2. **Create and Use YAML Configuration**:
```bash
# Create a configuration file with your preferred settings
python trainTestFile.py --mode create-config --output my_config.yaml --epochs 200 --language es

# Train using the configuration file
python trainTestFile.py --mode train --config my_config.yaml

# Override specific parameters from YAML config
python trainTestFile.py --mode train --config my_config.yaml --epochs 50 --batch-size 8
```

#### 3. **Test/Inference Mode**:
```bash
python trainTestFile.py --mode test --checkpoint ./checkpoints/best.ckpt --text "Hello world"
```

**Key Benefits of trainTestFile.py:**
- ✅ **No config file required** - set all parameters via CLI
- ✅ **Smart dataset detection** - skips download if dataset exists
- ✅ **Flexible parameter mixing** - combine YAML config with CLI overrides
- ✅ **Self-contained** - works even without full dependency installation for config creation

### Basic Usage

MyXTTS provides flexible configuration options - you can use either configuration files or direct CLI parameters:

#### Option 1: Using Configuration Files (Recommended)

1. **Create Configuration**:
```bash
myxtts create-config --output config.yaml --language en --dataset-path ./data/ljspeech
```

2. **Train Model**:
```bash
myxtts train --config config.yaml
```

3. **Train with Parameter Overrides**:
```bash
# Override specific parameters from the config file
myxtts train --config config.yaml --epochs 500 --batch-size 16
```

#### Option 2: Direct CLI Parameters (No Config File Required)

```bash
# Train directly with CLI parameters
myxtts train --data-path ./data/ljspeech --epochs 100 --batch-size 32 --learning-rate 1e-4
```

#### Inference and Voice Cloning

4. **Synthesize Speech**:
```bash
myxtts synthesize --config config.yaml --checkpoint ./checkpoints/checkpoint_best --text "Hello, this is MyXTTS!"
```

5. **Clone Voice**:
```bash
myxtts clone-voice --config config.yaml --checkpoint ./checkpoints/checkpoint_best --text "Hello with cloned voice" --reference-audio reference.wav
```

## Smart Dataset Detection

MyXTTS automatically detects existing datasets and **skips downloading** if your dataset is already available:

- **Custom dataset structure**: If you have `metadata.csv` and `wavs/` directly in your provided path, MyXTTS will use them
- **LJSpeech structure**: If your dataset follows the standard LJSpeech-1.1 structure, that works too
- **No unnecessary downloads**: If dataset files exist at the provided path, download is automatically skipped

```bash
# If ./my_dataset/ contains metadata.csv and wavs/, no download occurs
myxtts train --data-path ./my_dataset --epochs 100

# Works with both structures:
# ./my_dataset/metadata.csv + ./my_dataset/wavs/
# ./my_dataset/LJSpeech-1.1/metadata.csv + ./my_dataset/LJSpeech-1.1/wavs/
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

## Flexible Data Path Configuration

MyXTTS supports two data organization scenarios:

### Scenario A: Separate Train and Evaluation Data (Different Paths)

Use this when you have separate training and evaluation datasets with potentially different directory structures:

```
project/
├── train_data/
│   ├── metadata_train.csv
│   └── wavs/
│       ├── train_001.wav
│       └── train_002.wav
└── eval_data/
    ├── metadata_eval.csv  
    └── audio/  # Different directory name
        ├── eval_001.wav
        └── eval_002.wav
```

**Configuration:**
```bash
python trainTestFile.py --mode train \
    --metadata-train-file ./train_data/metadata_train.csv \
    --metadata-eval-file ./eval_data/metadata_eval.csv \
    --wavs-train-dir ./train_data/wavs \
    --wavs-eval-dir ./eval_data/audio
```

**YAML Configuration:**
```yaml
data:
  metadata_train_file: ./train_data/metadata_train.csv
  metadata_eval_file: ./eval_data/metadata_eval.csv
  wavs_train_dir: ./train_data/wavs
  wavs_eval_dir: ./eval_data/audio
```

### Scenario B: Single Dataset with Percentage Splits (Default)

Use this for traditional single-dataset training where train/validation/test splits are created automatically:

```
dataset/
├── metadata.csv          # Single metadata file
└── wavs/
    ├── audio1.wav
    ├── audio2.wav
    └── ...
```

**Configuration:**
```bash
python trainTestFile.py --mode train \
    --data-path ./dataset
```

**YAML Configuration:**
```yaml
data:
  dataset_path: ./dataset
  train_split: 0.8  # 80% for training
  val_split: 0.1    # 10% for validation
  # test split is automatically: 1 - train_split - val_split = 0.1 (10%)
```

### Path Resolution Rules

1. **Custom metadata files provided**: Each subset (train/val/test) uses its specific metadata file and wav directory
2. **No custom metadata files**: Single `metadata.csv` file is split into train/val/test by percentages
3. **Relative paths**: Resolved relative to the dataset directory
4. **Absolute paths**: Used as-is
5. **Wav directory fallback**: If not specified, defaults to `{metadata_file_directory}/wavs`

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