# Fix for Early Generation Stop Issue

## Problem Description

Users reported that inference was generating very short audio outputs (~0.7 seconds) that contained only noise, even when training loss was low and the model appeared to be converging well.

### Issue in Persian (Original)
> ه الان لاس پایین هست و ولیدیشن لاس هم پایین هست ولی توی inference_main.py
> خروجی که میده کلا 7 دهم ثانیست که غلطه و انتظار میره که شروع کنه به خروجی با کیفیت یا حداقل کم کیفیت بده درصورتی که هیچی نمیده یه نوییز 7 دهم ثانیه میده،
> مشکل کجاست؟

Translation: "The loss is now low and validation loss is also low, but in inference_main.py, the output it gives is only 0.7 seconds which is wrong. It's expected to start giving quality output or at least low quality, but it gives nothing, just 0.7 seconds of noise. What's the problem?"

## Root Cause

The issue was in the autoregressive generation loop in `myxtts/models/xtts.py`:

```python
# OLD CODE (PROBLEMATIC)
for step in range(max_length):
    # ... generation code ...
    
    stop_prob_value = float(tf.reduce_mean(stop_prob))
    if step == 0:
        continue
    elif step > 5 and stop_prob_value > 0.8:  # ❌ Too aggressive!
        break
```

### Why This Caused Problems

1. **Too Few Minimum Steps**: The condition `step > 5` meant generation could stop after only 6 frames
2. **Low Stop Threshold**: A threshold of 0.8 was too low, allowing premature stopping
3. **No Text Length Consideration**: Short and long texts had the same minimum frame requirement
4. **Result**: ~5-10 mel frames = ~0.6-1.2 seconds of audio at typical settings

## Solution

The fix implements more robust stopping criteria:

```python
# NEW CODE (FIXED)
# Calculate minimum frames based on text length
text_len = tf.shape(text_inputs)[1]
min_frames_tensor = tf.maximum(20, text_len * 10)  # At least 20 frames, or 10x text length
try:
    min_frames = int(min_frames_tensor.numpy())
except (AttributeError, RuntimeError):
    min_frames = 50  # Fallback for graph mode

for step in range(max_length):
    # ... generation code ...
    
    stop_prob_value = float(tf.reduce_mean(stop_prob))
    
    # More robust stopping criteria:
    if step < min_frames:
        # Don't check stop condition before minimum length
        continue
    elif step >= min_frames and stop_prob_value > 0.95:  # ✅ Higher threshold
        break
    elif step > max_length * 0.9:
        # Safety break if we're near max length
        break
```

### Key Improvements

1. **Dynamic Minimum Frames**: Calculated as `max(20, text_length * 10)`
   - Short text: At least 20 frames (~1 second)
   - Long text: Proportional to input length
   
2. **Higher Stop Threshold**: Increased from 0.8 to 0.95
   - Requires much higher confidence before stopping
   - Prevents premature stops from noisy predictions
   
3. **Better Safety Check**: Changed from 80% to 90% of max_length
   - Allows more generation room before forcing stop

## Expected Results

After this fix:

- **Minimum audio length**: ~1-2 seconds even for short texts
- **Proper length for long texts**: Proportional to input text length
- **Quality improvement**: Model can generate full utterances instead of truncated clips
- **Better stop behavior**: Only stops when model is genuinely confident

## Testing the Fix

To test if the fix works:

```bash
# Test with a short sentence
python3 inference_main.py \
    --text "Hello world" \
    --model-size tiny \
    --output test_short.wav

# Test with a longer sentence
python3 inference_main.py \
    --text "This is a much longer sentence that should generate several seconds of audio output to verify the fix is working correctly" \
    --model-size tiny \
    --output test_long.wav
```

Expected outcomes:
- `test_short.wav`: At least 1-2 seconds of audio
- `test_long.wav`: 5+ seconds of audio with proper speech

## Configuration Options

Users can still control generation behavior through these parameters:

- `--max-length`: Maximum generation length (default: 1000)
- `--temperature`: Sampling temperature (affects randomness)

The fix is automatic and requires no user configuration changes.

## Technical Details

### File Modified
- `myxtts/models/xtts.py`: Lines 883-935

### Changes Summary
- Added dynamic min_frames calculation based on text length
- Increased stop probability threshold from 0.8 to 0.95
- Added try-except for graph/eager mode compatibility
- Improved comments and documentation in code

### Backwards Compatibility
✅ Fully backwards compatible - no API changes, only internal generation logic improvements
