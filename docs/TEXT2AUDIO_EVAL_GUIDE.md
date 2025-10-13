# Text-to-Audio Evaluation Callback Guide

## Overview

The Text-to-Audio Evaluation Callback is a built-in feature that automatically generates audio samples from predefined texts during training at regular intervals. This enables real-time quality monitoring and helps detect training issues early.

## Features

- ✅ **Automatic audio generation** at configurable intervals (default: every 200 steps)
- ✅ **Multilingual support** - test with multiple languages simultaneously
- ✅ **WAV file output** - saved to disk for offline comparison
- ✅ **TensorBoard integration** - listen to samples directly in TensorBoard
- ✅ **Text reference files** - saved alongside audio for easy tracking
- ✅ **Non-intrusive** - temporarily switches to eval mode, doesn't affect training
- ✅ **Configurable** - easily customize interval, texts, and output location

## Quick Start

### Default Configuration

The feature is **enabled by default** with sensible defaults:

```python
from myxtts.config.config import XTTSConfig

# Create config with default text-to-audio evaluation
config = XTTSConfig()

# Defaults:
# - Enabled: True
# - Interval: Every 200 steps
# - Output: ./eval_samples/
# - Texts: English + Persian samples
# - TensorBoard logging: Enabled
```

### Custom Configuration

```python
from myxtts.config.config import XTTSConfig

config = XTTSConfig()

# Customize evaluation settings
config.training.enable_text2audio_eval = True
config.training.text2audio_interval_steps = 500  # Generate every 500 steps
config.training.text2audio_output_dir = "./my_eval_samples"
config.training.text2audio_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Testing voice cloning capability.",
    "این یک متن آزمایشی فارسی است.",  # Persian
]
config.training.text2audio_speaker_id = None  # For multi-speaker models
config.training.text2audio_log_tensorboard = True
```

### YAML Configuration

```yaml
training:
  # Text-to-Audio Evaluation Settings
  enable_text2audio_eval: true
  text2audio_interval_steps: 200
  text2audio_output_dir: "./eval_samples"
  text2audio_texts:
    - "The quick brown fox jumps over the lazy dog."
    - "سلام! این یک نمونهی ارزیابی است."
    - "Custom evaluation text here."
  text2audio_speaker_id: null
  text2audio_log_tensorboard: true
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_text2audio_eval` | bool | `True` | Enable/disable the feature |
| `text2audio_interval_steps` | int | `200` | Generate audio every N steps |
| `text2audio_output_dir` | str | `"./eval_samples"` | Output directory for audio files |
| `text2audio_texts` | List[str] | English + Persian | List of texts to synthesize |
| `text2audio_speaker_id` | Optional[int] | `None` | Speaker ID for multi-speaker models |
| `text2audio_log_tensorboard` | bool | `True` | Log audio to TensorBoard |

## Output Structure

```
eval_samples/
├── step_200/
│   ├── eval_00.wav          # Generated audio
│   ├── eval_00.txt          # Reference text
│   ├── eval_01.wav
│   └── eval_01.txt
├── step_400/
│   ├── eval_00.wav
│   ├── eval_00.txt
│   ├── eval_01.wav
│   └── eval_01.txt
└── step_600/
    ├── eval_00.wav
    ├── eval_00.txt
    ├── eval_01.wav
    └── eval_01.txt
```

## Usage Examples

### Example 1: Basic Training with Evaluation

```bash
python train_main.py \
    --train-data ./dataset/train \
    --val-data ./dataset/val \
    --model-size medium
```

Audio will be automatically generated every 200 steps to `./eval_samples/`.

### Example 2: Custom Interval

```python
config = XTTSConfig()
config.training.text2audio_interval_steps = 1000  # Every 1000 steps

trainer = XTTSTrainer(config)
trainer.train(train_dataset, val_dataset)
```

### Example 3: Custom Test Texts

```python
config = XTTSConfig()
config.training.text2audio_texts = [
    "Testing pronunciation accuracy.",
    "Voice quality assessment sample.",
    "نمونه ارزیابی کیفیت صدا",  # Persian
    "Muestra de evaluación de calidad",  # Spanish
]

trainer = XTTSTrainer(config)
trainer.train(train_dataset, val_dataset)
```

### Example 4: Disable Evaluation

```python
config = XTTSConfig()
config.training.enable_text2audio_eval = False

trainer = XTTSTrainer(config)
trainer.train(train_dataset, val_dataset)
```

## How It Works

### Training Loop Integration

1. **After each training step**, the trainer calls `_maybe_eval_text2audio()`
2. **Check interval**: If `current_step % interval == 0`, proceed
3. **Generate audio**:
   - Create output directory: `eval_samples/step_{N}/`
   - Set model to eval mode (temporarily)
   - Process each text:
     - Clean and tokenize text
     - Generate mel spectrogram
     - Convert to audio (HiFi-GAN vocoder or Griffin-Lim)
     - Save WAV file
     - Save text file
   - Log to TensorBoard (if enabled)
   - Restore model to training mode

### Audio Generation Pipeline

```
Text → Clean → Tokenize → Model(eval) → Mel → Vocoder → Audio → Save
```

- **Vocoder**: Uses HiFi-GAN if available, falls back to Griffin-Lim
- **Text Processing**: Uses the same TextProcessor as training
- **Model State**: Temporarily switches to eval mode (`trainable=False`)
- **Error Handling**: Gracefully handles errors without stopping training

## Benefits

### 1. Quality Monitoring
Listen to generated samples at different training stages to assess quality improvements.

### 2. Early Issue Detection
Quickly identify problems like:
- Mode collapse
- Quality degradation
- Language-specific issues
- Voice cloning failures

### 3. Comparative Analysis
Compare samples from different steps to:
- Track convergence
- Evaluate voice consistency
- Assess pronunciation accuracy
- Monitor prosody improvements

### 4. Multilingual Testing
Test multiple languages simultaneously to ensure balanced multilingual performance.

### 5. Documentation
Audio samples serve as training checkpoints for:
- Model versioning
- Experiment tracking
- Result sharing
- Paper submissions

## Best Practices

### 1. Choose Representative Texts
Select texts that cover:
- Common phonemes
- Various sentence lengths
- Different emotions/styles
- Multiple languages (if multilingual)

Example:
```python
config.training.text2audio_texts = [
    "The quick brown fox jumps over the lazy dog.",  # Pangram
    "How are you doing today?",  # Common phrase
    "Testing voice cloning with longer sentences.",  # Length test
    "سلام! چطور هستید؟",  # Persian
]
```

### 2. Set Appropriate Intervals

| Training Duration | Recommended Interval |
|-------------------|---------------------|
| Short (< 10k steps) | 100-200 steps |
| Medium (10k-50k) | 200-500 steps |
| Long (> 50k steps) | 500-1000 steps |

### 3. Organize Output
Use descriptive output directories:
```python
config.training.text2audio_output_dir = f"./eval_samples/{experiment_name}"
```

### 4. Monitor TensorBoard
View and listen to samples in real-time:
```bash
tensorboard --logdir ./checkpoints/tensorboard
```

### 5. Clean Up Old Samples
Periodically clean old evaluation samples to save disk space:
```bash
# Keep only recent samples
find ./eval_samples -type d -name "step_*" | sort -V | head -n -10 | xargs rm -rf
```

## Troubleshooting

### Issue: No audio generated

**Check:**
1. Feature is enabled: `config.training.enable_text2audio_eval = True`
2. Training has reached interval: Check current step vs. interval
3. Output directory is writable

### Issue: Poor audio quality

**Solutions:**
- Check if model is properly trained
- Verify mel spectrogram quality
- Try different vocoder settings
- Increase mel frames limit if clipped

### Issue: Out of memory

**Solutions:**
- Reduce number of evaluation texts
- Increase interval between evaluations
- Generate without TensorBoard logging
- Use Griffin-Lim instead of neural vocoder

### Issue: Errors during generation

The feature handles errors gracefully and logs warnings. Check logs for:
- Text processing errors
- Model inference issues
- File I/O problems

## Performance Impact

The text-to-audio evaluation has **minimal impact** on training:

- **Time**: ~1-5 seconds per evaluation (depends on text length)
- **Memory**: Temporary increase during generation, cleaned up after
- **Frequency**: Configurable interval (default: every 200 steps)
- **Mode**: Non-blocking, doesn't interfere with training loop

For a typical training run:
- Total evaluation time: < 1% of training time
- Memory overhead: < 5% temporary increase
- Disk space: ~1-10 MB per evaluation

## Advanced Usage

### Custom Audio Processing

You can customize audio processing by modifying the AudioProcessor:

```python
from myxtts.utils.audio import AudioProcessor

# Custom processor with different settings
custom_processor = AudioProcessor(
    sample_rate=22050,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=80
)
```

### Multi-Speaker Evaluation

For multi-speaker models, specify speaker IDs:

```python
config.training.text2audio_speaker_id = 0  # Evaluate with speaker 0

# Or evaluate multiple speakers by extending the texts:
config.training.text2audio_texts = [
    "Speaker 0: Hello world",  # Use speaker 0
    "Speaker 1: Hello world",  # Use speaker 1
]
```

### Custom Vocoder Settings

The feature automatically uses available vocoder:
1. HiFi-GAN (if model has vocoder)
2. Griffin-Lim (fallback)

## Integration with Existing Tools

### TensorBoard

Audio samples are logged as TensorBoard audio summaries:
```bash
tensorboard --logdir ./checkpoints/tensorboard
```

Navigate to "AUDIO" tab to listen to samples.

### WandB

To log samples to Weights & Biases, extend the callback:
```python
# In _generate_eval_audio, after saving:
if wandb.run:
    wandb.log({
        f"eval_audio_{idx}": wandb.Audio(audio_waveform, sample_rate=sample_rate),
        "step": self.current_step
    })
```

## FAQ

**Q: Does this slow down training?**  
A: Minimal impact. Evaluation takes 1-5 seconds every N steps (default: 200).

**Q: Can I add more texts during training?**  
A: No, texts are configured before training. Stop and restart with new config.

**Q: What if I want to evaluate with different voice references?**  
A: Currently uses the trained model's default behavior. Voice conditioning would require modifying the callback.

**Q: Can I disable TensorBoard logging but keep file output?**  
A: Yes, set `text2audio_log_tensorboard = False`.

**Q: How do I compare quality across different training runs?**  
A: Use the same evaluation texts across runs and compare WAV files or TensorBoard outputs.

## Examples

See the [demo script](../examples/demo_text2audio_eval.py) for complete examples:

```bash
python examples/demo_text2audio_eval.py
```

## Testing

Run unit tests to verify functionality:

```bash
python -m unittest tests.test_text2audio_eval -v
```

## Contributing

To improve this feature:
1. Add more sophisticated audio quality metrics
2. Implement automatic quality scoring
3. Add voice conditioning support
4. Create comparison visualizations
5. Add multi-speaker evaluation

## License

Same as the main project.

## Support

For issues or questions:
- Check the [troubleshooting section](#troubleshooting)
- Review test cases in `tests/test_text2audio_eval.py`
- Open an issue on GitHub

---

**Happy Training! 🎵**
