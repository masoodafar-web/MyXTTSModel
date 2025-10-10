# Quick Start: Dual-GPU Bottleneck Fix
# راهنمای سریع: رفع Bottleneck دو GPU

## 🎯 برای حل سریع مشکل | Quick Fix

اگر GPU utilization پایین دارید (<70%) با این دستورات شروع کنید:

### Step 1: Run Profiler (تشخیص مشکل)

```bash
cd /path/to/MyXTTSModel

# Profile current setup
python utilities/dual_gpu_bottleneck_profiler.py \
  --batch-size 16 \
  --num-steps 100 \
  --data-gpu 0 \
  --model-gpu 1
```

این profiler به شما می‌گوید:
- کجا bottleneck است
- GPU utilization واقعی
- توصیه‌های خاص برای config شما

### Step 2: Apply Recommended Config (اعمال تنظیمات)

**For RTX 4090 or similar (24GB):**

```bash
python train_main.py \
  --model-size tiny \
  --batch-size 16 \
  --data-gpu 0 \
  --model-gpu 1 \
  --enable-memory-isolation \
  --enable-static-shapes \
  --data-gpu-memory 8192 \
  --model-gpu-memory 16384 \
  --train-data ./data/your_dataset
```

**For smaller GPUs (12-16GB):**

```bash
python train_main.py \
  --model-size tiny \
  --batch-size 8 \
  --data-gpu 0 \
  --model-gpu 1 \
  --enable-memory-isolation \
  --enable-static-shapes \
  --data-gpu-memory 4096 \
  --model-gpu-memory 8192 \
  --train-data ./data/your_dataset
```

### Step 3: Monitor Performance (نظارت)

در terminal دیگر:

```bash
watch -n 1 nvidia-smi
```

**باید ببینید:**
- GPU 0 (Data): 40-60% utilization
- GPU 1 (Model): 80-95% utilization
- Memory stable (not growing)
- Power draw high on both GPUs

---

## 📊 Expected vs Current Performance

### Before Fix (قبل):
```
GPU 0: ████░░░░░░ 40% util   [WAITING]
GPU 1: ███████░░░ 70% util   [WAITING]
Speed: 3-4 steps/sec
Time variation: 50-80% (BAD)
```

### After Fix (بعد):
```
GPU 0: ██████░░░░ 60% util   [BUSY]
GPU 1: █████████░ 90% util   [BUSY]
Speed: 6-8 steps/sec (2x faster!)
Time variation: 15-30% (GOOD)
```

---

## 🔧 Common Issues & Quick Fixes

### Issue 1: "OOM Error with batch-size 32"

**Fix:**
```bash
# Reduce batch size first
--batch-size 16

# OR increase GPU memory limits
--data-gpu-memory 10240 \
--model-gpu-memory 20480
```

### Issue 2: "GPU Utilization still low (<70%)"

**Possible causes and fixes:**

**A. Data loading too slow:**
```yaml
# In configs/config.yaml
data:
  num_workers: 24              # Increase from 16
  prefetch_buffer_size: 6      # Increase from 4
  use_tf_native_loading: true  # Make sure enabled
```

**B. Storage I/O bottleneck:**
- Move dataset to SSD (not HDD)
- Use `--cache-dataset` flag if dataset is small
- Precompute and cache features

**C. Wrong GPU topology:**
```bash
# Check GPU connection
nvidia-smi topo -m

# Should show PIX (good) not NODE/SYS (bad)
```

### Issue 3: "High timing variation (>30%)"

**Fix:**
```bash
# Increase pipeline depth
python train_main.py \
  ... \
  --prefetch-buffer-size 8  # More buffering
```

And in config:
```yaml
data:
  num_workers: 24     # More parallel workers
```

---

## 📈 Configuration Matrix

| GPU Memory | Batch Size | data-gpu-memory | model-gpu-memory | Prefetch |
|------------|------------|-----------------|------------------|----------|
| 12GB each  | 8          | 4096            | 8192             | 6-8      |
| 16GB each  | 12         | 6144            | 10240            | 4-6      |
| 24GB each  | 16-24      | 8192            | 16384            | 2-4      |

---

## 🎓 Understanding the Fix

### What changed?

**1. Pipeline Architecture:**
```
OLD: [Load] → [Prep] → [Transfer] → [Train]
     Sequential, lots of waiting

NEW: [Load] → [Prep] ┐
                      ├→ [Train N-1]
     [Load] → [Prep] ┘
     Parallel, always busy
```

**2. Buffer System:**
- Before: 2 buffers (double buffering)
- After: 3 buffers (triple buffering)
- Result: Smoother pipeline, less waiting

**3. Transfer Optimization:**
- Before: Blocking transfers (wait until complete)
- After: Async transfers (continue immediately)
- Result: GPU:0 prepares next while GPU:1 trains

**4. Dataset Optimization:**
- Added: Direct GPU prefetch
- Added: Parallel batch processing
- Added: Auto-tuning
- Result: Faster data delivery

---

## 📋 Checklist Before Training

- [ ] Profiler run completed
- [ ] No major bottlenecks identified
- [ ] Config adjusted based on profiler recommendations
- [ ] `nvidia-smi` shows both GPUs available
- [ ] Dataset path correct
- [ ] Batch size appropriate for GPU memory
- [ ] `num_workers` increased (16-32 recommended)

---

## 🚀 Performance Targets

### Minimum Acceptable:
- GPU:0 utilization: >40%
- GPU:1 utilization: >70%
- Timing variation: <40%
- No OOM errors

### Good Performance:
- GPU:0 utilization: >50%
- GPU:1 utilization: >80%
- Timing variation: <30%
- Stable training

### Excellent Performance:
- GPU:0 utilization: >60%
- GPU:1 utilization: >90%
- Timing variation: <20%
- 2-3x faster than single GPU

---

## 🆘 Getting Help

### Run Full Diagnostics:

```bash
# 1. System info
nvidia-smi
nvidia-smi topo -m

# 2. Run profiler with full output
python utilities/dual_gpu_bottleneck_profiler.py \
  --batch-size 16 \
  --num-steps 100 \
  --data-gpu 0 \
  --model-gpu 1 \
  > profiler_output.txt 2>&1

# 3. Test training (first 100 steps)
python train_main.py \
  --model-size tiny \
  --batch-size 16 \
  --data-gpu 0 \
  --model-gpu 1 \
  --enable-memory-isolation \
  --train-data ./data \
  --max-steps 100 \
  > training_output.txt 2>&1
```

Share these files for help:
- `profiler_output.txt`
- `training_output.txt`
- Your config.yaml
- `nvidia-smi` output

---

## 📚 Additional Resources

- Full documentation: [DUAL_GPU_BOTTLENECK_FIX.md](./DUAL_GPU_BOTTLENECK_FIX.md)
- Memory isolation guide: [docs/MEMORY_ISOLATION_GUIDE.md](./docs/MEMORY_ISOLATION_GUIDE.md)
- Performance improvements: [docs/PERFORMANCE_IMPROVEMENTS.md](./docs/PERFORMANCE_IMPROVEMENTS.md)

---

**Updated**: 2025-10-10  
**Version**: 2.0  
**Status**: ✅ Ready to Use

موفق باشید! | Good luck! 🚀
