# راهنمای رفع مشکل استفاده 1-5% از GPU
# GPU Utilization Fix Guide (1-5% Issue on Dual RTX 4090)

## 🎯 Quick Summary | خلاصه سریع

**Problem:** GPU utilization oscillates between 1-5% during training on dual RTX 4090, despite all optimizations (TF-native loading, triple buffering, prefetch 100+, async pipeline, XLA, etc.)

**Solution:** Apply comprehensive fixes including:
1. Aggressive TensorFlow configuration
2. Ultra-optimized data pipeline
3. Increased buffer sizes
4. Critical performance flags

**Expected Result:** GPU:0 = 60-80%, GPU:1 = 85-95%

---

## 🚀 Quick Start (برای شروع سریع)

### Option 1: Automated Fix (Recommended)

```bash
# Run the quick fix script
bash quick_fix_gpu_utilization.sh

# Then start training
python train_main.py \
    --batch-size 128 \
    --num-workers 32 \
    --enable-memory-isolation \
    --data-gpu 0 \
    --model-gpu 1 \
    --enable-static-shapes
```

### Option 2: Manual Fix

1. **Configure TensorFlow:**
   ```bash
   python utilities/configure_max_gpu_utilization.py --verify
   ```

2. **Run diagnostic:**
   ```bash
   python utilities/diagnose_gpu_utilization.py --config configs/config.yaml
   ```

3. **Update config.yaml:**
   ```yaml
   data:
     batch_size: 128
     num_workers: 32
     prefetch_buffer_size: 100
     use_tf_native_loading: true
     prefetch_to_gpu: true
     pad_to_fixed_length: true
     enable_xla: true
     mixed_precision: true
   ```

4. **Start training with optimized settings**

---

## 📋 Complete Fix Implementation

### Phase 1: Install Dependencies (if needed)

```bash
# Install GPUtil for monitoring
pip install gputil

# Install PyYAML for config parsing
pip install pyyaml
```

### Phase 2: Run Diagnostic

```bash
# Diagnose current issues
python utilities/diagnose_gpu_utilization.py --config configs/config.yaml

# This will identify:
# - Missing optimizations
# - Suboptimal configuration
# - Performance bottlenecks
```

### Phase 3: Apply Optimizations

#### A. Configure TensorFlow

```bash
# Apply all TensorFlow optimizations
python utilities/configure_max_gpu_utilization.py

# Or from Python code:
from utilities.configure_max_gpu_utilization import configure_max_gpu_utilization
configure_max_gpu_utilization()
```

This configures:
- Thread pools (inter_op, intra_op)
- GPU memory growth
- XLA JIT compilation
- Mixed precision training
- All experimental optimizations

#### B. Update Configuration File

Edit `configs/config.yaml`:

```yaml
data:
  # CRITICAL: Aggressive batch size for RTX 4090
  batch_size: 128              # Increase from default (32-64)
  
  # CRITICAL: Maximum workers
  num_workers: 32              # Increase from default (8-16)
  
  # CRITICAL: Aggressive prefetch
  prefetch_buffer_size: 100    # Increase significantly
  shuffle_buffer_multiplier: 50
  
  # CRITICAL: Enable all optimizations
  use_tf_native_loading: true
  prefetch_to_gpu: true
  enhanced_gpu_prefetch: true
  optimize_cpu_gpu_overlap: true
  auto_tune_performance: true
  
  # CRITICAL: Enable static shapes (prevents retracing)
  pad_to_fixed_length: true
  max_text_length: 200
  max_mel_frames: 800
  
  # CRITICAL: TensorFlow optimizations
  enable_xla: true
  mixed_precision: true
  pin_memory: true
  persistent_workers: true
  
  # Dual-GPU settings
  pipeline_buffer_size: 100    # Increase from 50
```

#### C. Use Optimized Training Command

```bash
python train_main.py \
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
    --optimization-level enhanced
```

### Phase 4: Monitor Results

```bash
# Terminal 1: Run training (as above)

# Terminal 2: Monitor GPU utilization
watch -n 1 nvidia-smi

# Terminal 3: Run profiler (if needed)
python utilities/dual_gpu_bottleneck_profiler.py --batch-size 128 --num-steps 100
```

---

## 📊 Expected Results | نتایج مورد انتظار

### Before (Current State):
```
GPU:0 Utilization: 1-5%     ❌
GPU:1 Utilization: 1-5%     ❌
Step Time: Very high        ❌
Throughput: Very low        ❌
```

### After (With Fixes):
```
GPU:0 Utilization: 60-80%   ✅ (Data preprocessing)
GPU:1 Utilization: 85-95%   ✅ (Model training)
Step Time: <0.5s            ✅ (batch_size=128)
Throughput: >200 samples/s  ✅
```

---

## 🔧 Troubleshooting | عیبیابی

### Issue 1: Still Low GPU Utilization (<50%)

**Possible Causes:**
1. Batch size too small
2. Insufficient num_workers
3. TF-native loading not active
4. Data pipeline bottleneck

**Solutions:**

```bash
# 1. Increase batch size
python train_main.py --batch-size 256 ...  # Try larger

# 2. Increase workers
python train_main.py --num-workers 48 ...  # If you have many CPU cores

# 3. Verify TF-native loading
# Look for this in training logs:
# "✅ SUCCESS: Using TensorFlow-native data loading (GPU-optimized)"

# 4. Run profiler to identify bottleneck
python utilities/dual_gpu_bottleneck_profiler.py --batch-size 128 --num-steps 100
```

### Issue 2: Out of Memory (OOM)

**Solution: Reduce batch size gradually**

```bash
# Try smaller batch size
python train_main.py --batch-size 96 ...

# Or adjust GPU memory limits
python train_main.py \
    --data-gpu-memory 8192 \
    --model-gpu-memory 18432 \
    ...
```

### Issue 3: TF-native Loading Not Working

**Check training logs for:**
```
❌ WARNING: TF-native loading FAILED
```

**Solution:**
1. Ensure `use_tf_native_loading: true` in config
2. Check that `tf_native_loader.py` is present
3. Verify cache directory is writable

### Issue 4: Oscillating GPU Utilization

**Possible Causes:**
1. Retracing (variable shapes)
2. Insufficient prefetch
3. Sync points in pipeline

**Solutions:**
```yaml
# Enable static shapes
data:
  pad_to_fixed_length: true
  max_text_length: 200
  max_mel_frames: 800
```

---

## 📈 Performance Tuning | تنظیم عملکرد

### For Different Hardware

#### High-End (2x RTX 4090, 64+ CPU cores):
```yaml
data:
  batch_size: 256
  num_workers: 48
  prefetch_buffer_size: 150
  pipeline_buffer_size: 150
```

#### Mid-Range (2x RTX 3090, 32 CPU cores):
```yaml
data:
  batch_size: 128
  num_workers: 32
  prefetch_buffer_size: 100
  pipeline_buffer_size: 100
```

#### Lower-End (2x RTX 3080, 16 CPU cores):
```yaml
data:
  batch_size: 64
  num_workers: 24
  prefetch_buffer_size: 50
  pipeline_buffer_size: 50
```

---

## 🔍 Understanding the Fix | درک راهکار

### Root Causes of 1-5% Utilization:

1. **Insufficient Data Pipeline Parallelism**
   - Default num_parallel_calls not aggressive enough
   - Missing experimental optimization flags

2. **Small Batch Size**
   - Batch size 32-64 too small for RTX 4090
   - GPU finishes computation faster than data arrives

3. **Inadequate Prefetching**
   - Default prefetch buffer too small
   - No GPU prefetching enabled

4. **Variable Shapes Causing Retracing**
   - tf.function recompiles for each shape
   - Causes GPU idle time

5. **Missing TensorFlow Optimizations**
   - XLA not enabled
   - Mixed precision not used
   - Thread pools not configured

### How the Fix Works:

1. **Aggressive Threading**
   ```python
   tf.config.threading.set_inter_op_parallelism_threads(cpu_count)
   tf.config.threading.set_intra_op_parallelism_threads(cpu_count // 2)
   ```
   Maximizes CPU parallelism for data loading

2. **Dataset Options**
   ```python
   options.experimental_optimization.parallel_batch = True
   options.experimental_deterministic = False
   ```
   Enables aggressive parallelization

3. **Multi-Stage Prefetching**
   ```python
   dataset.prefetch(AUTOTUNE)  # Host
   dataset.prefetch_to_device('/GPU:0', buffer_size=100)  # GPU
   dataset.prefetch(AUTOTUNE)  # Final
   ```
   Ensures GPU never starves

4. **Static Shapes**
   ```python
   padded_batch(..., padded_shapes=([200], [800, 80], [], []))
   ```
   Prevents retracing, stable GPU usage

5. **XLA and Mixed Precision**
   ```python
   tf.config.optimizer.set_jit(True)
   tf.keras.mixed_precision.set_global_policy('mixed_float16')
   ```
   Faster computation on RTX 4090

---

## 📚 Additional Resources | منابع اضافی

- **Detailed Documentation:** `CRITICAL_GPU_UTILIZATION_FIX.md`
- **Profiler Tool:** `utilities/dual_gpu_bottleneck_profiler.py`
- **Dataset Optimizer:** `myxtts/data/dataset_optimizer.py`
- **Configuration Tool:** `utilities/configure_max_gpu_utilization.py`
- **Diagnostic Tool:** `utilities/diagnose_gpu_utilization.py`

---

## ✅ Verification Checklist | چک‌لیست تایید

Before starting training, verify:

- [ ] TensorFlow and CUDA properly installed
- [ ] 2 GPUs detected by nvidia-smi
- [ ] Config file updated with aggressive settings
- [ ] `configure_max_gpu_utilization.py` executed
- [ ] `diagnose_gpu_utilization.py` shows no critical issues
- [ ] Batch size ≥ 128 for RTX 4090
- [ ] num_workers ≥ 32
- [ ] `use_tf_native_loading: true`
- [ ] `pad_to_fixed_length: true`
- [ ] `enable_xla: true`

---

## 🆘 Getting Help | دریافت کمک

If you still have issues after applying all fixes:

1. Run full diagnostic:
   ```bash
   python utilities/diagnose_gpu_utilization.py --config configs/config.yaml
   ```

2. Run profiler:
   ```bash
   python utilities/dual_gpu_bottleneck_profiler.py --batch-size 128 --num-steps 100
   ```

3. Check training logs for:
   - "✅ SUCCESS: Using TensorFlow-native data loading"
   - GPU device placement messages
   - Any warnings or errors

4. Monitor nvidia-smi during training

5. Share logs and diagnostic output in issue tracker

---

**Version:** 3.0  
**Date:** 2025-10-10  
**Status:** Production Ready  
**Target:** Dual RTX 4090 with severe GPU underutilization (1-5%)
