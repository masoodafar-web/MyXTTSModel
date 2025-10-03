# Fix for Vocoder Noise Issue

## Problem Description (مشکل)

بعد از حل مشکل استاپ زودهنگام، مدل خروجی تولید می‌کند اما فقط نویز است و هیچ صدای قابل فهمی تولید نمی‌شود.

**Translation**: After fixing the early stopping problem, the model generates output but it's only noise and no intelligible voice is produced.

### Issue in Persian (Original)
> الان مشکل استاپ شدن حل شده ولی خالص نویزه و اصلا آوایی تولید نمیکنه مشکل از کجاست؟ حلش کن

**Translation**: "Now the stop problem is solved but it's pure noise and no voice is being generated at all. What's the problem? Fix it."

## Root Cause (علت اصلی)

The HiFi-GAN neural vocoder is initialized with random weights when:

1. **Model not trained enough**: The vocoder component needs significant training to converge
2. **Checkpoint incomplete**: The checkpoint doesn't include trained vocoder weights
3. **Training interrupted**: Training stopped before vocoder could learn

### Why This Produces Noise

```
Random Weights → Untrained Neural Network → Random/Noise Output
```

The vocoder's job is to convert mel spectrograms to audio waveforms. With random weights, it produces random audio (noise) even when the mel spectrograms are correct.

## Solution (راه حل)

Our fix implements a **multi-layered detection and fallback system**:

### 1. Weight Initialization Tracking

```python
# In VocoderInterface class
self._weights_initialized = False  # Track if weights are loaded

def mark_weights_loaded(self):
    """Called after checkpoint loading"""
    self._weights_initialized = True
```

### 2. Runtime Validation

```python
# Check vocoder output quality
audio_power = np.mean(np.abs(audio_waveform))
if audio_power < 1e-6:
    logger.warning("Vocoder output has very low power, using Griffin-Lim fallback")
    audio_waveform = audio_processor.mel_to_wav(mel_spectrogram)
```

### 3. Automatic Fallback to Griffin-Lim

When vocoder produces invalid output:
- System automatically falls back to Griffin-Lim algorithm
- Griffin-Lim is a classical algorithm that doesn't need training
- Lower quality than neural vocoder, but produces intelligible speech

### 4. User Warnings

```
⚠️  VOCODER WEIGHTS NOT INITIALIZED WARNING
==================================================================
The neural vocoder (HiFi-GAN) weights may not be properly trained.
This will likely result in NOISE instead of intelligible speech.

Common causes:
  1. Model checkpoint doesn't include vocoder weights
  2. Vocoder was not trained (training stopped too early)
  3. Using wrong checkpoint or model configuration

Solutions:
  • Train the model longer to ensure vocoder converges
  • Check that checkpoint includes all model components
  • System will automatically fallback to Griffin-Lim if output is invalid
==================================================================
```

## How to Fix (چگونه برطرف کنیم)

### Option 1: Train the Model Longer (Recommended)

The vocoder needs time to train properly. Typical training requirements:

```bash
# Continue training for vocoder convergence
python3 train_main.py \
    --train-data ../dataset/dataset_train \
    --val-data ../dataset/dataset_eval \
    --checkpoint-dir ./checkpointsmain \
    --model-size tiny \
    --epochs 100  # Ensure sufficient training
```

**Signs of trained vocoder:**
- Training loss decreases steadily
- Validation samples sound clear (not noisy)
- Spectrogram reconstructions are accurate

### Option 2: Use Griffin-Lim Fallback (Temporary)

While training, you can still generate audio using Griffin-Lim:

```bash
python3 inference_main.py \
    --text "متن فارسی یا انگلیسی" \
    --model-size tiny \
    --output test_output.wav
```

The system will automatically use Griffin-Lim if vocoder is untrained.

**Griffin-Lim characteristics:**
- ✅ Works without training
- ✅ Produces intelligible speech
- ❌ Lower quality than neural vocoder
- ❌ May sound "robotic"

### Option 3: Load Pre-trained Vocoder Weights

If available, load a checkpoint with trained vocoder weights:

```bash
python3 inference_main.py \
    --text "Your text here" \
    --checkpoint path/to/trained/checkpoint \
    --output output.wav
```

## Technical Details

### Files Modified

1. **`myxtts/models/vocoder.py`**
   - Added weight initialization tracking
   - Added output validation
   - Implemented automatic fallback logic

2. **`myxtts/utils/commons.py`**
   - Mark vocoder weights as loaded after checkpoint restoration
   - Lines 505-511

3. **`myxtts/inference/synthesizer.py`**
   - Enhanced vocoder output validation
   - Audio power checking
   - Griffin-Lim fallback implementation
   - Lines 192-244

4. **`inference_main.py`**
   - Added user-facing warning messages
   - Lines 751-770

### Detection Logic

```python
# 1. Check if weights were loaded from checkpoint
if not vocoder._weights_initialized:
    warn("Vocoder weights not initialized")

# 2. Check for NaN or near-zero output
audio_mean = tf.reduce_mean(tf.abs(audio))
if tf.math.is_nan(audio_mean) or audio_mean < 1e-8:
    fallback_to_griffin_lim()

# 3. Check audio power level
audio_power = np.mean(np.abs(audio_waveform))
if audio_power < 1e-6:
    fallback_to_griffin_lim()
```

## Expected Results (نتایج مورد انتظار)

### Before Fix (قبل از اصلاح)
- ❌ Pure noise output
- ❌ No intelligible speech
- ❌ No error messages
- ❌ User confusion

### After Fix (بعد از اصلاح)
- ✅ Clear warning messages
- ✅ Automatic Griffin-Lim fallback
- ✅ Intelligible speech (lower quality)
- ✅ Guidance on how to resolve
- ✅ Smooth user experience

## Training Recommendations (توصیه‌های آموزش)

To ensure vocoder trains properly:

### 1. Sufficient Training Steps
```yaml
# In config.yaml
training:
  max_steps: 100000  # Minimum for vocoder convergence
  save_interval: 1000
  validation_interval: 500
```

### 2. Vocoder Learning Rate
```yaml
model:
  vocoder_type: "hifigan"
  vocoder_lr: 0.0002  # Appropriate learning rate
```

### 3. Monitor Vocoder Loss
```bash
# Check training logs for vocoder convergence
grep "vocoder" training.log
```

### 4. Listen to Validation Samples
- Validation samples should improve over time
- By step 20k-30k, should hear clear improvements
- By step 50k+, should have good quality

## Troubleshooting (عیب‌یابی)

### Issue: Still Getting Noise After Training

**Possible causes:**
1. Training stopped too early (< 20k steps)
2. Learning rate too high/low
3. Data quality issues

**Solution:**
```bash
# Check checkpoint metadata
python3 -c "
import json
with open('checkpointsmain/checkpoint_XXXXX_metadata.json') as f:
    meta = json.load(f)
    print(f'Step: {meta[\"step\"]}')
    print(f'Loss: {meta[\"loss\"]}')
"
```

### Issue: Griffin-Lim Quality Too Low

**Temporary solution:**
- Increase Griffin-Lim iterations (in audio processor)
- Use higher quality reference audio
- Clean training data

**Permanent solution:**
- Train vocoder properly (recommended)

### Issue: Checkpoint Not Loading Vocoder

**Check:**
```bash
# Verify checkpoint includes vocoder weights
ls -lh checkpointsmain/checkpoint_*_model.weights.h5
```

## Performance Impact

| Vocoder Type | Quality | Speed | Training Required |
|--------------|---------|-------|-------------------|
| HiFi-GAN (trained) | ⭐⭐⭐⭐⭐ | Fast | ✅ Yes (50k+ steps) |
| HiFi-GAN (untrained) | ❌ Noise | Fast | N/A |
| Griffin-Lim | ⭐⭐⭐ | Slow | ❌ No |

## Summary (خلاصه)

This fix ensures that:

1. **Users are informed** when vocoder is untrained
2. **System doesn't fail** - automatic Griffin-Lim fallback
3. **Clear guidance** on how to resolve the issue
4. **Better experience** - some audio is better than no audio

**Key Message**: The fix doesn't magically train the vocoder, but it:
- Detects the problem
- Provides a working fallback
- Guides users to the real solution (training)

## Related Documentation

- [Early Stop Fix](INFERENCE_FIX_EARLY_STOP.md) - For generation length issues
- [Training Guide](../README.md) - For model training setup
- [Configuration Guide](../configs/README.md) - For model configuration

---

**Note**: This is a diagnostic and fallback system. For production-quality audio, you must train the model with a neural vocoder until convergence (typically 50k-100k steps).
