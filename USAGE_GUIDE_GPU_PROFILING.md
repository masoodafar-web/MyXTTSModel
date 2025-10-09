# راهنمای استفاده از ابزارهای Profiling GPU

## 🎯 هدف

این راهنما نحوه استفاده از ابزارهای جدید برای **تشخیص و حل کامل** نوسان GPU و Bottleneck در training را توضیح میدهد.

---

## 📦 نصب وابستگیها

اطمینان حاصل کنید که TensorFlow نصب است:

```bash
pip install tensorflow>=2.12.0
pip install gputil  # اختیاری - برای اطلاعات GPU
```

---

## 🚀 Quick Start

### مرحله ۱: تشخیص سریع

```bash
python utilities/comprehensive_gpu_diagnostic.py --data-path ./data
```

این دستور:
- تمام تنظیمات را بررسی میکند
- مشکلات را شناسایی میکند
- راهحلهای دقیق ارائه میدهد
- گزارش در `diagnostic_report.txt` ذخیره میشود

**مثال خروجی:**

```
DIAGNOSTIC SUMMARY
==================================================
Found 2 issue(s) and 3 recommendation(s)

🔴 ISSUES:
   - Graph mode not enabled
   - High batch time variation detected

💡 RECOMMENDATIONS:
   - Increase num_workers to 8-16
   - Enable XLA compilation
   - Use SSD instead of HDD

⚙️  CONFIGURATION CHANGES:
   training.enable_graph_mode: true
   data.num_workers: 16
```

---

### مرحله ۲: اعمال تغییرات

فایل `configs/config.yaml` را ویرایش کنید:

```yaml
data:
  use_tf_native_loading: true
  prefetch_to_gpu: true
  num_workers: 16
  prefetch_buffer_size: 16

training:
  enable_graph_mode: true
  enable_xla_compilation: true
```

---

### مرحله ۳: تست مجدد

```bash
python utilities/comprehensive_gpu_diagnostic.py --data-path ./data
```

باید ببینید:

```
✅ NO ISSUES DETECTED
   Your configuration appears optimal
```

---

## 🔧 ابزارهای تخصصی

### 1. Comprehensive GPU Diagnostic (تشخیص جامع)

**استفاده:**
```bash
python utilities/comprehensive_gpu_diagnostic.py \
    --config configs/config.yaml \
    --data-path ./data \
    --output my_report.txt
```

**خروجی:**
- گزارش کامل در `my_report.txt`
- خلاصه در console
- لیست مشکلات و راهحلها

**چه زمانی استفاده کنیم:**
- قبل از شروع training
- وقتی GPU oscillation داریم
- بعد از تغییر تنظیمات

---

### 2. Enhanced GPU Profiler (پروفایل Data Pipeline)

**استفاده ساده:**
```bash
python utilities/enhanced_gpu_profiler.py \
    --data-path ./data \
    --batch-size 16 \
    --num-batches 100
```

**Benchmark (پیدا کردن بهترین تنظیمات):**
```bash
python utilities/enhanced_gpu_profiler.py \
    --data-path ./data \
    --benchmark
```

**خروجی:**

```
DATA LOADING - TIMING STATISTICS
==================================================
Average time:        45.23ms
Variation ratio:     11.32%
✅ LOW VARIATION - Stable timing

RECOMMENDED CONFIGURATION
==================================================
batch_size: 32
num_workers: 16
Expected avg time: 35.12ms
```

**چه زمانی استفاده کنیم:**
- وقتی data loading کند است
- برای یافتن بهترین batch_size
- برای یافتن بهترین num_workers

---

### 3. Training Step Profiler (پروفایل کامل Training)

**استفاده:**
```bash
python utilities/training_step_profiler.py \
    --data-path ./data \
    --num-steps 100 \
    --batch-size 16
```

**با تنظیمات خاص:**
```bash
python utilities/training_step_profiler.py \
    --data-path ./data \
    --num-steps 100 \
    --batch-size 32 \
    --no-xla             # غیرفعال XLA
    --no-mixed-precision # غیرفعال mixed precision
```

**خروجی:**

```
TIMING BREAKDOWN:
  Total step:        120.45ms ± 12.30ms
  Data loading:       35.20ms ± 3.10ms (29.2%)
  Training (F+B+O):   85.25ms ± 9.20ms (70.8%)

BOTTLENECK ANALYSIS
==================================================
✅ MODEL TRAINING IS DOMINANT
   Training takes 70.8% of total time
   GPU is well-utilized
```

**چه زمانی استفاده کنیم:**
- برای شناسایی bottleneck اصلی
- وقتی نمیدانیم مشکل data یا model است
- برای بررسی throughput

---

## 📊 تفسیر نتایج

### Variation Ratio (نسبت واریانس)

```
< 20%  ✅ عالی - pipeline پایدار
20-50% ⚠️  قابل قبول - میتوان بهبود داد
> 50%  🔴 مشکل - oscillation شدید
```

### Data Loading Percentage

```
< 30%  ✅ عالی - GPU پر مشغول
30-50% ⚠️  متوسط - قابل بهبود
> 50%  🔴 bottleneck - GPU idle است
```

### Throughput (Steps/Second)

```
> 10   ✅ عالی
5-10   ⚠️  خوب
< 5    🔴 کند - نیاز به بهینهسازی
```

---

## 🔍 سناریوهای رایج

### سناریو ۱: "Variation > 50%"

**مشکل:** نوسان شدید GPU

**راهحل:**
1. فعالسازی TF-native loading:
   ```yaml
   data:
     use_tf_native_loading: true
   ```

2. فعالسازی graph mode:
   ```yaml
   training:
     enable_graph_mode: true
   ```

3. تست:
   ```bash
   python utilities/enhanced_gpu_profiler.py --data-path ./data --num-batches 100
   ```

---

### سناریو ۲: "Data loading > 50%"

**مشکل:** GPU منتظر data است

**راهحل:**
1. افزایش workers:
   ```yaml
   data:
     num_workers: 16
     prefetch_buffer_size: 16
   ```

2. استفاده از SSD (اگر HDD دارید)

3. Precompute features:
   ```yaml
   data:
     preprocessing_mode: "precompute"
   ```

4. تست:
   ```bash
   python utilities/training_step_profiler.py --data-path ./data --num-steps 50
   ```

---

### سناریو ۳: "همه تنظیمات درست است اما هنوز کند است"

**بررسیهای بیشتر:**

1. بررسی hardware:
   ```bash
   nvidia-smi
   ```

2. بررسی RAM:
   ```bash
   free -h
   ```

3. بررسی storage speed:
   ```bash
   # خروجی comprehensive_gpu_diagnostic را ببینید
   python utilities/comprehensive_gpu_diagnostic.py --data-path ./data
   ```

4. بررسی مدل:
   ```bash
   # اگر training time خیلی بالاست، مدل بزرگتر از GPU است
   python utilities/training_step_profiler.py --data-path ./data --batch-size 8
   ```

---

## 📝 Checklist قبل از Training

```bash
# 1. تشخیص
python utilities/comprehensive_gpu_diagnostic.py --data-path ./data

# 2. اعمال تغییرات
# Edit configs/config.yaml

# 3. Benchmark
python utilities/enhanced_gpu_profiler.py --data-path ./data --benchmark

# 4. تست training loop
python utilities/training_step_profiler.py --data-path ./data --num-steps 50

# 5. شروع training
python train_main.py --train-data ./data --batch-size 32
```

---

## 🎓 Tips & Best Practices

### Tip 1: همیشه Benchmark کنید

```bash
python utilities/enhanced_gpu_profiler.py --data-path ./data --benchmark
```

بهترین batch_size و num_workers برای GPU شما متفاوت است.

### Tip 2: Precompute > On-the-fly

```yaml
data:
  preprocessing_mode: "precompute"  # بهترین
```

این روش سریعترین است.

### Tip 3: SSD خیلی مهم است

```
HDD: 50-100ms per batch → Bottleneck
SSD: 10-20ms per batch → Good
```

### Tip 4: XLA می‌تواند ۲-۳x سریعتر باشد

```yaml
training:
  enable_xla_compilation: true
```

اما ممکن است روی همه سیستمها کار نکند.

### Tip 5: Monitor در حین Training

```bash
# Terminal 1
python train_main.py --train-data ./data

# Terminal 2
watch -n 0.5 nvidia-smi
```

GPU utilization باید stable و بالا (70-90%) باشد.

---

## 🆘 Troubleshooting

### مشکل: "TensorFlow not installed"

```bash
pip install tensorflow>=2.12.0
```

### مشکل: "No GPU detected"

بررسی کنید:
```bash
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### مشکل: "Dataset not found"

مسیر درست را بدهید:
```bash
python utilities/comprehensive_gpu_diagnostic.py --data-path /path/to/your/data
```

### مشکل: "XLA compilation failed"

XLA روی همه سیستمها در دسترس نیست:
```yaml
training:
  enable_xla_compilation: false  # غیرفعال کنید
```

---

## 📚 مستندات بیشتر

- [COMPREHENSIVE_GPU_BOTTLENECK_ANALYSIS.md](docs/COMPREHENSIVE_GPU_BOTTLENECK_ANALYSIS.md) - راهنمای جامع فارسی/انگلیسی
- [GPU_OSCILLATION_FIX.md](docs/GPU_OSCILLATION_FIX.md) - توضیحات فنی مشکل
- [GPU_OSCILLATION_SOLUTION_SUMMARY.md](GPU_OSCILLATION_SOLUTION_SUMMARY.md) - خلاصه راهحل
- [QUICK_START_GPU_OSCILLATION_FIX.md](QUICK_START_GPU_OSCILLATION_FIX.md) - شروع سریع

---

## ✅ Summary

**سه ابزار اصلی:**

1. **comprehensive_gpu_diagnostic.py** - تشخیص جامع (شروع از اینجا)
2. **enhanced_gpu_profiler.py** - پروفایل data pipeline
3. **training_step_profiler.py** - پروفایل کامل training

**روند کار:**

```
Diagnostic → Apply Changes → Benchmark → Profile Training → Start Training
```

**هدف:**

```
✅ GPU Utilization: 70-90% (stable)
✅ Data Loading: < 30%
✅ Variation: < 20%
✅ Throughput: > 10 steps/sec
```

---

**تاریخ:** 2024  
**نسخه:** 1.0  
**وضعیت:** ✅ آماده استفاده
