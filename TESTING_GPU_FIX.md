# راهنمای تست راهکار GPU Utilization
# Testing Guide for GPU Utilization Fix

## هدف / Goal
Test and validate that the GPU utilization fix resolves the 1-5% issue and achieves 80%+ GPU utilization on both RTX 4090 GPUs.

---

## 🧪 Test Plan

### Phase 1: Pre-Test Verification

#### 1.1 Environment Check
```bash
# Check Python version
python3 --version  # Should be 3.8+

# Check TensorFlow
python3 -c "import tensorflow as tf; print(tf.__version__)"  # Should be 2.x

# Check GPUs
nvidia-smi  # Should show 2x RTX 4090
```

#### 1.2 Repository State
```bash
# Verify new files exist
ls -l utilities/configure_max_gpu_utilization.py
ls -l utilities/diagnose_gpu_utilization.py
ls -l myxtts/data/dataset_optimizer.py
ls -l quick_fix_gpu_utilization.sh

# Verify all are executable
chmod +x utilities/configure_max_gpu_utilization.py
chmod +x utilities/diagnose_gpu_utilization.py
chmod +x quick_fix_gpu_utilization.sh
```

---

### Phase 2: Diagnostic Baseline

#### 2.1 Run Diagnostic (Before Fix)
```bash
# Run diagnostic on current configuration
python3 utilities/diagnose_gpu_utilization.py \
    --config configs/config.yaml \
    --skip-realtime

# Expected: Multiple issues detected
# Save output: diagnose_before.txt
```

#### 2.2 Document Current State
Record current metrics:
```bash
# Start a short training run (1-2 minutes)
python3 train_main.py \
    --train-data ../dataset/dataset_train \
    --val-data ../dataset/dataset_eval \
    --epochs 1 &

# In another terminal, monitor
watch -n 1 nvidia-smi

# Record:
# - GPU:0 utilization: _____%
# - GPU:1 utilization: _____%
# - GPU:0 memory used: _____MB
# - GPU:1 memory used: _____MB
# - Approximate step time: _____s
```

---

### Phase 3: Apply Fix

#### 3.1 Quick Fix Script
```bash
# Run the automated fix
bash quick_fix_gpu_utilization.sh

# This should:
# 1. Check prerequisites
# 2. Run diagnostic
# 3. Apply TensorFlow optimizations
# 4. Provide configuration recommendations
```

#### 3.2 Manual Configuration Update

Edit `configs/config.yaml`:

```yaml
data:
  # Before → After
  batch_size: 56 → 128
  num_workers: 16 → 32
  prefetch_buffer_size: 16 → 100
  
  # Ensure these are true
  use_tf_native_loading: true
  prefetch_to_gpu: true
  pad_to_fixed_length: true
  enable_xla: true
  mixed_precision: true
  
  pipeline_buffer_size: 50 → 100
  shuffle_buffer_multiplier: 20 → 50
```

#### 3.3 Verify Fix Applied
```bash
# Run diagnostic again
python3 utilities/diagnose_gpu_utilization.py \
    --config configs/config.yaml \
    --skip-realtime

# Expected: Few or no issues
# Save output: diagnose_after.txt
```

---

### Phase 4: Test Training with Optimized Settings

#### 4.1 Short Test Run (5 minutes)
```bash
# Terminal 1: Start training
python3 train_main.py \
    --train-data ../dataset/dataset_train \
    --val-data ../dataset/dataset_eval \
    --batch-size 128 \
    --num-workers 32 \
    --enable-memory-isolation \
    --data-gpu 0 \
    --model-gpu 1 \
    --data-gpu-memory 10240 \
    --model-gpu-memory 20480 \
    --enable-static-shapes \
    --max-text-length 200 \
    --max-mel-frames 800 \
    --optimization-level enhanced \
    --epochs 1

# Terminal 2: Monitor GPUs
watch -n 1 nvidia-smi

# Terminal 3: Monitor system
htop
```

#### 4.2 Metrics to Record

**During First Epoch (Cache Building):**
- Time for cache building: _____s
- GPU utilization during cache: _____%

**During Training (After Cache):**
- GPU:0 average utilization: _____%
- GPU:1 average utilization: _____%
- GPU:0 memory used: _____MB
- GPU:1 memory used: _____MB
- Average step time: _____s
- Steps per second: _____
- Samples per second: _____

**Training Logs:**
- Look for: "✅ SUCCESS: Using TensorFlow-native data loading"
- Any warnings or errors?
- GPU device placement messages

---

### Phase 5: Batch Size Scaling Test

Test different batch sizes to find optimal:

#### 5.1 Batch Size 64
```bash
python3 train_main.py --batch-size 64 ...
# Record: GPU util, step time, OOM?
```

#### 5.2 Batch Size 128 (Recommended)
```bash
python3 train_main.py --batch-size 128 ...
# Record: GPU util, step time, OOM?
```

#### 5.3 Batch Size 192
```bash
python3 train_main.py --batch-size 192 ...
# Record: GPU util, step time, OOM?
```

#### 5.4 Batch Size 256 (Maximum)
```bash
python3 train_main.py --batch-size 256 ...
# Record: GPU util, step time, OOM?
```

**Find Sweet Spot:**
- Highest batch size without OOM
- GPU utilization > 80%
- Stable step times

---

### Phase 6: Profiling

#### 6.1 Run Profiler
```bash
python3 utilities/dual_gpu_bottleneck_profiler.py \
    --batch-size 128 \
    --num-steps 100

# This will show:
# - Data loading time
# - GPU-to-GPU transfer time
# - Model forward time
# - Loss computation time
# - Backward pass time
# - Optimizer step time
# - Overall throughput
```

#### 6.2 Analyze Results
- Identify remaining bottlenecks
- Check for slow phases
- Verify GPU utilization metrics

---

### Phase 7: Extended Training Test

#### 7.1 Full Epoch Test
```bash
# Run for at least 1 full epoch
python3 train_main.py \
    --batch-size 128 \
    --num-workers 32 \
    --enable-memory-isolation \
    --data-gpu 0 \
    --model-gpu 1 \
    --enable-static-shapes \
    --optimization-level enhanced \
    --epochs 1

# Monitor continuously with nvidia-smi
```

#### 7.2 Stability Check
- GPU utilization stable throughout epoch?
- Step times consistent?
- No OOM errors?
- No performance degradation?

---

## ✅ Success Criteria

### Primary Metrics (Must Pass):

| Metric | Before | Target | Actual |
|--------|--------|--------|--------|
| GPU:0 Utilization | 1-5% | 60-80% | ____% |
| GPU:1 Utilization | 1-5% | 85-95% | ____% |
| Step Time (batch=128) | Very high | <0.5s | ____s |
| Throughput | Very low | >200 samples/s | ____ |

### Secondary Metrics (Should Pass):

- ✅ No OOM errors
- ✅ Stable GPU utilization (low variance)
- ✅ Consistent step times
- ✅ Training logs show TF-native loading
- ✅ No warnings about slow operations

---

## 📊 Expected vs Actual Results

### Expected Improvements:

**Before Fix:**
```
GPU:0: ███                    (1-5%)
GPU:1: ██                     (1-5%)
Step Time: 2-5s
Throughput: 20-50 samples/s
```

**After Fix:**
```
GPU:0: ███████████████       (60-80%)
GPU:1: ████████████████████  (85-95%)
Step Time: 0.3-0.5s
Throughput: 200-300 samples/s
```

### Performance Multipliers:
- GPU utilization: **15-80x increase**
- Training speed: **10-20x faster**
- Step time: **5-10x reduction**

---

## 🔍 Troubleshooting Tests

### Test 1: TF-Native Loading Verification
```bash
# Check training logs for:
grep "TensorFlow-native data loading" train_output.log

# Should see:
# ✅ SUCCESS: Using TensorFlow-native data loading (GPU-optimized)

# If not:
# ❌ WARNING: TF-native loading FAILED
# → Check tf_native_loader.py exists
# → Verify use_tf_native_loading: true in config
```

### Test 2: Static Shapes Verification
```bash
# Enable debug logging
export TF_CPP_MIN_LOG_LEVEL=0

# Run training and check for retracing warnings
python3 train_main.py ...

# Look for: "Tracing" or "Retracing" messages
# If many retracing messages:
# → pad_to_fixed_length not working
# → Check max_text_length and max_mel_frames
```

### Test 3: XLA Verification
```bash
# Check if XLA is enabled
python3 -c "import tensorflow as tf; print(tf.config.optimizer.get_jit())"

# Should print: True

# Check environment
echo $TF_XLA_FLAGS

# Should include: --tf_xla_auto_jit=2
```

### Test 4: Memory Growth Verification
```bash
# Check GPU memory growth
python3 -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    config = tf.config.experimental.get_memory_growth(gpu)
    print(f'{gpu.name}: memory_growth={config}')
"

# Should show: True for all GPUs
```

---

## 📝 Test Report Template

```markdown
# GPU Utilization Fix Test Report

## Environment
- OS: _____
- Python: _____
- TensorFlow: _____
- CUDA: _____
- GPUs: 2x RTX 4090

## Before Fix
- GPU:0 Utilization: _____%
- GPU:1 Utilization: _____%
- Step Time: _____s
- Throughput: _____ samples/s

## After Fix
- GPU:0 Utilization: _____%
- GPU:1 Utilization: _____%
- Step Time: _____s
- Throughput: _____ samples/s

## Improvements
- GPU utilization increase: _____x
- Speed improvement: _____x
- Step time reduction: _____x

## Configuration Used
- batch_size: _____
- num_workers: _____
- prefetch_buffer_size: _____
- use_tf_native_loading: _____
- pad_to_fixed_length: _____

## Issues Encountered
1. _____
2. _____

## Recommendations
1. _____
2. _____

## Conclusion
- [ ] Fix successful (GPU util > 80%)
- [ ] Fix partially successful (GPU util 50-80%)
- [ ] Fix unsuccessful (GPU util < 50%)

## Additional Notes
_____
```

---

## 🚀 Next Steps After Successful Test

1. **Document Final Configuration:**
   - Save optimal batch_size, num_workers
   - Document any hardware-specific tuning

2. **Update Configuration:**
   - Update config.yaml with proven settings
   - Commit changes to repository

3. **Share Results:**
   - Create test report
   - Share performance metrics
   - Document any issues encountered

4. **Production Deployment:**
   - Use validated configuration
   - Monitor first production run
   - Maintain performance logs

---

## 🆘 If Tests Fail

### GPU Utilization Still Low (<50%):

1. **Re-run diagnostic:**
   ```bash
   python3 utilities/diagnose_gpu_utilization.py --config configs/config.yaml
   ```

2. **Check each optimization:**
   - TF-native loading active?
   - Static shapes enabled?
   - Batch size adequate?
   - Workers sufficient?

3. **Run profiler:**
   ```bash
   python3 utilities/dual_gpu_bottleneck_profiler.py --batch-size 128
   ```

4. **Collect logs:**
   - Full training log
   - Diagnostic output
   - Profiler output
   - nvidia-smi output

5. **Report issue:**
   - Share collected logs
   - Include test report
   - Describe hardware setup

---

**Version:** 1.0  
**Date:** 2025-10-10  
**Purpose:** Validate GPU utilization fix for dual RTX 4090 setup
