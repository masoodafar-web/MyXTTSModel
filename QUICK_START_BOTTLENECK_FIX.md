# Quick Start: Testing the Bottleneck Fix

## راهنمای سریع تست راه‌حل Bottleneck

این راهنما به شما کمک می‌کند تا سریعاً راه‌حل ارائه شده را تست کنید و بهبود performance را مشاهده کنید.

---

## گام 1: بررسی تنظیمات (2 دقیقه)

```bash
# اعتبارسنجی config
python utilities/validate_gpu_optimization.py --config configs/config.yaml
```

**چه چیزی باید ببینید:**
```
✅ TF-native loading is enabled
✅ num_workers is optimal (16)
✅ prefetch_buffer_size is good
✅ ALL CHECKS PASSED
```

**اگر خطا دیدید:**
- خطاهای CRITICAL را حتماً برطرف کنید
- Warnings را برای بهترین performance برطرف کنید
- در `configs/config.yaml` تغییرات لازم را اعمال کنید

---

## گام 2: تنظیمات ضروری در config.yaml

حداقل این موارد را چک کنید:

```yaml
data:
  # Critical
  use_tf_native_loading: true
  num_workers: 16
  
  # Recommended
  prefetch_buffer_size: 16
  pad_to_fixed_length: true
  max_text_length: 200
  max_mel_frames: 800
  
  # Optional (for dual-GPU)
  data_gpu: 0
  model_gpu: 1
```

---

## گام 3: تحلیل pipeline (5 دقیقه - اختیاری)

برای درک کامل bottleneck‌ها:

```bash
python utilities/analyze_data_pipeline_bottleneck.py
```

این ابزار سه تحلیل انجام می‌دهد:
1. Text preprocessing performance
2. Audio loading variance
3. Pipeline efficiency

**نتیجه مطلوب:**
```
✅ Text preprocessing is efficient
✅ Consistent batch loading performance
✅ Pipeline is efficiently feeding GPU
```

---

## گام 4: پاک کردن cache قدیمی (در صورت نیاز)

اگر قبلاً training کرده‌اید:

```bash
# پاک کردن cache قدیمی
rm -rf data/ljspeech/processed/tf_native_cache_*
```

این کار باعث می‌شود cache جدید با optimizations بسازیم.

---

## گام 5: اولین training run (15-30 دقیقه)

```bash
python train_main.py --config configs/config.yaml
```

### چه چیزی باید ببینید:

#### A. Cache Building (اولین بار - یک‌بار)

```
==================================================================
BUILDING TF-NATIVE CACHE (Text Preprocessing)
==================================================================
Processing 13100 samples...
Using 8 parallel workers for faster preprocessing
  Progress: 13100/13100 (100.0%)
✅ Text preprocessing complete
Saving cache to disk...
✅ Cache saved to disk: data/ljspeech/processed/tf_native_cache_train
==================================================================
```

**زمان مورد انتظار:** 10-30 ثانیه (بسته به CPU و dataset size)

#### B. TF-Native Loading Confirmation

```
==================================================================
✅ SUCCESS: Using TensorFlow-native data loading (GPU-optimized)
==================================================================
Benefits:
  • No CPU bottleneck (no tf.numpy_function)
  • Full graph compilation support
  • GPU-accelerated operations
==================================================================
```

#### C. Training Starts

```
Epoch 1/1000
Step 1/500: loss=3.45, mel_loss=2.12, ...
Step 2/500: loss=3.42, mel_loss=2.10, ...
```

---

## گام 6: monitoring GPU utilization

در terminal جداگانه:

```bash
watch -n 1 nvidia-smi
```

**چه چیزی باید ببینید:**

### حالت Single-GPU:
```
+-----------------------------------------------------------------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|      0  RTX 4090        On   | 00000000:01:00.0 Off |                  Off |
| 85%   85C    P2    420W / 450W |  18000MiB / 24576MiB |     92%      Default |
+-----------------------------------------------------------------------------+
```

### حالت Dual-GPU:
```
+-----------------------------------------------------------------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|      0  RTX 4090        On   | 00000000:01:00.0 Off |                  Off |
| 75%   78C    P2    350W / 450W |  10000MiB / 24576MiB |     85%      Default |
|      1  RTX 4090        On   | 00000000:02:00.0 Off |                  Off |
| 75%   82C    P2    380W / 450W |  17000MiB / 24576MiB |     88%      Default |
+-----------------------------------------------------------------------------+
```

**معیارهای موفقیت:**
- GPU Utilization: **80-95%** (پایدار)
- ❌ نه oscillation (40% → 70% → 40%)
- ❌ نه spike pattern
- ✅ مصرف ثابت و بالا

---

## گام 7: run‌های بعدی (تست سرعت cache loading)

اگر training را متوقف و دوباره شروع کنید:

```bash
# Stop training (Ctrl+C)
# Start again
python train_main.py --config configs/config.yaml
```

**باید ببینید:**

```
==================================================================
LOADING TF-NATIVE CACHE FROM DISK
==================================================================
Loading cached text preprocessing for 13100 samples...
✅ Cache loaded successfully from disk
✅ TF-native cache ready (loaded from disk)
==================================================================
```

**زمان مورد انتظار:** 1-2 ثانیه (خیلی سریع‌تر از اولین بار!)

---

## مقایسه Performance

### قبل از fix:

```
Dataset Initialization: 30-120 seconds
GPU Utilization: 40-70% (oscillating)
Batch Time: 150-250ms
Training Throughput: ~60 samples/s
```

### بعد از fix:

```
First Run:
  Dataset Initialization: 10-30 seconds
  Cache Building: Parallel (8 workers)
  
Subsequent Runs:
  Dataset Initialization: 1-2 seconds
  Cache Loading: From disk

Training:
  GPU Utilization: 80-95% (stable)
  Batch Time: 50-80ms
  Training Throughput: ~250-320 samples/s

Overall Speedup: 4-5x
```

---

## Troubleshooting سریع

### ❌ مشکل: هنوز GPU oscillation دارم

```bash
# 1. بررسی TF-native loading
grep "SUCCESS: Using TensorFlow-native" <training_log>

# 2. تحلیل pipeline
python utilities/analyze_data_pipeline_bottleneck.py

# 3. بررسی config
python utilities/validate_gpu_optimization.py --config configs/config.yaml
```

### ❌ مشکل: Cache هر بار rebuild می‌شود

```bash
# بررسی وجود cache
ls -la data/ljspeech/processed/tf_native_cache_train/

# اگر وجود ندارد، manually بسازید
mkdir -p data/ljspeech/processed/tf_native_cache_train
chmod 755 data/ljspeech/processed/tf_native_cache_train
```

### ❌ مشکل: Cache building خیلی کند است

```yaml
# در config.yaml تعداد workers را کم کنید
data:
  num_workers: 8  # یا 4 اگر CPU ضعیف دارید
```

### ❌ مشکل: Out of Memory

```yaml
# کاهش batch size یا workers
data:
  batch_size: 16  # به جای 32
  num_workers: 8   # به جای 16
```

---

## Checklist موفقیت

قبل از گزارش نتایج، این موارد را چک کنید:

- [ ] `validate_gpu_optimization.py` تمام چک‌ها را pass می‌کند
- [ ] در logs پیام "SUCCESS: Using TensorFlow-native data loading" وجود دارد
- [ ] Cache building با parallel workers اجرا می‌شود
- [ ] Cache روی disk ذخیره و load می‌شود
- [ ] GPU utilization بالای 80% و پایدار است
- [ ] نه oscillation، نه spike pattern
- [ ] Training throughput حداقل 2-3x افزایش یافته

---

## گزارش نتایج

لطفاً این اطلاعات را در issue گزارش دهید:

```markdown
### System Info
- GPU: [e.g., RTX 4090 x2]
- CPU: [e.g., AMD Ryzen 9 5950X]
- RAM: [e.g., 64GB]
- Storage: [SSD/HDD]
- Dataset size: [e.g., 13,100 samples]

### Config
- use_tf_native_loading: [true/false]
- num_workers: [e.g., 16]
- batch_size: [e.g., 32]
- prefetch_buffer_size: [e.g., 16]

### Results

#### Before Fix:
- Dataset init time: [e.g., 90s]
- GPU utilization: [e.g., 45-65% oscillating]
- Batch time: [e.g., 200ms]

#### After Fix:
- First run init time: [e.g., 15s]
- Subsequent init time: [e.g., 2s]
- GPU utilization: [e.g., 85-92% stable]
- Batch time: [e.g., 65ms]
- Speedup: [e.g., 4.2x]

#### Validator Output:
```
[paste output from validate_gpu_optimization.py]
```

#### Pipeline Analysis:
```
[paste output from analyze_data_pipeline_bottleneck.py]
```
```

---

## کمک بیشتر

اگر مشکلی دارید:

1. **مستندات کامل:** `DUAL_GPU_BOTTLENECK_SOLUTION.md`
2. **تحلیل Pipeline:** `DATA_PIPELINE_OPTIMIZATION.md`
3. **GitHub Issue:** گزارش مشکل با اطلاعات بالا

---

**موفق باشید! 🚀**

با این optimizations، باید GPU utilization پایدار 80-95% را ببینید.
