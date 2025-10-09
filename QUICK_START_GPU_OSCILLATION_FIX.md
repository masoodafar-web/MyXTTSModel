# 🚀 Quick Start: رفع مشکل نوسان GPU (2-40%)

## مشکل (Problem)

مصرف GPU به صورت نوسانی (spike/cycle) بین ۲٪ تا ۴۰٪ است و training بسیار کند پیش میرود.

**English**: GPU utilization oscillates between 2-40% in a cyclic pattern, causing very slow training.

---

## ✅ راهحل سریع (Quick Fix)

### گام ۱: تشخیص مشکل (Step 1: Diagnose)

```bash
# اجرای ابزار تشخیص
python utilities/diagnose_gpu_bottleneck.py --batch-size 16 --num-batches 50
```

اگر خروجی این باشد:
- ✅ `High variation detected` - مشکل تشخیص داده شد
- ✅ `tf.numpy_function found` - علت مشکل پیدا شد

### گام ۲: فعالسازی راهحل (Step 2: Enable Fix)

فایل `configs/config.yaml` را ویرایش کنید:

```yaml
data:
  # فعالسازی TensorFlow-native loading
  use_tf_native_loading: true        # حذف CPU bottleneck
  
  # بهینهسازی GPU prefetching
  prefetch_to_gpu: true
  enhanced_gpu_prefetch: true
  optimize_cpu_gpu_overlap: true
  
  # تنظیمات بهینه data loading
  num_workers: 16
  prefetch_buffer_size: 16
  batch_size: 16

training:
  # فعالسازی graph mode
  enable_graph_mode: true
  enable_xla_compilation: true
  enable_eager_debug: false
```

### گام ۳: شروع Training (Step 3: Start Training)

```bash
# اجرای training با تنظیمات بهینه
python train_main.py \
  --train-data ./data/train \
  --val-data ./data/val \
  --batch-size 16 \
  --num-workers 16 \
  --optimization-level enhanced
```

### گام ۴: مانیتورینگ GPU (Step 4: Monitor GPU)

در ترمینال جداگانه:

```bash
# مشاهده GPU utilization به صورت لحظهای
watch -n 0.5 nvidia-smi
```

**انتظار**: GPU utilization پایدار ۷۰-۹۰٪ (نه ۲-۴۰٪ نوسانی)

---

## 📊 نتایج مورد انتظار

### قبل از رفع مشکل:
```
GPU: 2% → 40% → 2% → 40% (نوسانی)
سرعت training: بسیار کند
زمان هر batch: 100-500ms (متغیر)
```

### بعد از رفع مشکل:
```
GPU: 70-90% (پایدار)
سرعت training: 5-10 برابر سریعتر ✅
زمان هر batch: 50-100ms (پایدار)
```

---

## 🔧 تنظیمات پیشرفته (Advanced Settings)

### برای GPUهای قویتر (RTX 4090):

```yaml
data:
  batch_size: 32              # بیشتر برای GPU قویتر
  num_workers: 24             # workers بیشتر
  prefetch_buffer_size: 32    # buffer بزرگتر
```

### برای GPUهای ضعیفتر (RTX 3060):

```yaml
data:
  batch_size: 8               # کمتر برای GPU ضعیفتر
  num_workers: 8              # workers کمتر
  prefetch_buffer_size: 8     # buffer کوچکتر
```

### استفاده از Gradient Accumulation:

اگر GPU memory کم است:

```yaml
training:
  gradient_accumulation_steps: 4  # معادل batch_size × 4

data:
  batch_size: 4  # batch کوچکتر
```

---

## ⚠️ عیبیابی (Troubleshooting)

### مشکل: "TF-native loading failed"

**علت**: فایلهای صوتی WAV نیستند

**راهحل**:
1. تبدیل فایلها به فرمت WAV
2. یا نصب `tensorflow-io`:
   ```bash
   pip install tensorflow-io
   ```

### مشکل: هنوز نوسان وجود دارد

**راهحلهای احتمالی**:

1. **Buffer کوچک است**:
   ```yaml
   prefetch_buffer_size: 32  # افزایش دهید
   ```

2. **Workers کم است**:
   ```yaml
   num_workers: 24  # افزایش دهید
   ```

3. **Storage کند است**:
   - از SSD استفاده کنید (نه HDD)
   - یا داده را به RAM منتقل کنید

4. **Graph mode غیرفعال است**:
   ```yaml
   training:
     enable_graph_mode: true
     enable_xla_compilation: true
   ```

### مشکل: کیفیت صدا کاهش یافته

**راهحل موقت**: TF-native را غیرفعال کنید:

```yaml
data:
  use_tf_native_loading: false  # برگشت به روش قبلی
```

سپس issue گزارش دهید با نمونه صوتی.

---

## 📝 چک‌لیست نهایی

قبل از شروع training، بررسی کنید:

- [ ] `use_tf_native_loading: true` در config
- [ ] `prefetch_to_gpu: true` در config
- [ ] `enable_graph_mode: true` در config
- [ ] `num_workers >= 8` در config
- [ ] `prefetch_buffer_size >= 8` در config
- [ ] فایلهای صوتی در فرمت WAV هستند
- [ ] GPU driver و CUDA نصب است
- [ ] TensorFlow با GPU support نصب است

---

## 🎉 خلاصه

```bash
# 1. تشخیص
python utilities/diagnose_gpu_bottleneck.py

# 2. تنظیمات (در config.yaml)
use_tf_native_loading: true
prefetch_to_gpu: true
enable_graph_mode: true

# 3. اجرا
python train_main.py --batch-size 16 --num-workers 16

# 4. مانیتورینگ
watch -n 0.5 nvidia-smi
```

**نتیجه**: GPU utilization پایدار ۷۰-۹۰٪ و training سرعت ۵-۱۰ برابری! 🚀

---

## 📚 اطلاعات بیشتر

- **مستندات کامل**: [docs/GPU_OSCILLATION_FIX.md](docs/GPU_OSCILLATION_FIX.md)
- **ابزار تشخیص**: `utilities/diagnose_gpu_bottleneck.py`
- **تستها**: `tests/test_gpu_oscillation_fix.py`
- **کد اصلی**: `myxtts/data/tf_native_loader.py`

---

**تاریخ**: 2024

**وضعیت**: ✅ راهحل پیادهسازی شده

**Issue**: تحلیل و رفع مشکل نوسان مصرف GPU (۲-۴۰٪)

---

*برای سوالات یا مشکلات، ابتدا ابزار تشخیص را اجرا کنید*
