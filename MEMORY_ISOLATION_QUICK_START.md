# شروع سریع: Memory-Isolated Dual-GPU Training

## Quick Start: Memory-Isolated Dual-GPU Training

این راهنما برای شروع سریع با سیستم جداسازی حافظه dual-GPU است.

---

## ✅ پیش‌نیازها | Prerequisites

1. **دو GPU**: حداقل دو GPU NVIDIA
2. **CUDA**: نصب شده و فعال
3. **TensorFlow**: نسخه 2.12 یا بالاتر

```bash
# بررسی GPUها
nvidia-smi

# باید حداقل 2 GPU ببینید
```

---

## 🚀 استفاده ساده | Simple Usage

### حالت پایه | Basic Mode

```bash
python train_main.py \
    --data-gpu 0 \
    --model-gpu 1 \
    --enable-memory-isolation \
    --train-data ../dataset/dataset_train \
    --val-data ../dataset/dataset_eval
```

این دستور:
- GPU 0 را برای پردازش داده استفاده می‌کند (محدودیت 8GB)
- GPU 1 را برای آموزش مدل استفاده می‌کند (محدودیت 16GB)
- جداسازی حافظه را فعال می‌کند

---

## 🎯 سناریوهای مختلف | Different Scenarios

### سناریو 1: GPUهای قدرتمند (24GB+)

```bash
python train_main.py \
    --data-gpu 0 \
    --model-gpu 1 \
    --enable-memory-isolation \
    --data-gpu-memory 10240 \
    --model-gpu-memory 20480 \
    --batch-size 64 \
    --train-data ../dataset/dataset_train
```

### سناریو 2: GPUهای متوسط (12-16GB)

```bash
python train_main.py \
    --data-gpu 0 \
    --model-gpu 1 \
    --enable-memory-isolation \
    --data-gpu-memory 6144 \
    --model-gpu-memory 12288 \
    --batch-size 32 \
    --train-data ../dataset/dataset_train
```

### سناریو 3: GPUهای کوچک (8-10GB)

```bash
python train_main.py \
    --data-gpu 0 \
    --model-gpu 1 \
    --enable-memory-isolation \
    --data-gpu-memory 4096 \
    --model-gpu-memory 8192 \
    --batch-size 16 \
    --model-size small \
    --train-data ../dataset/dataset_train
```

---

## 📊 مانیتورینگ | Monitoring

### خروجی مورد انتظار | Expected Output

```
======================================================================
Memory-Isolated Dual-GPU Trainer Initialization
======================================================================
🎯 Setting up GPU Memory Isolation...
   Data GPU 0: 8192MB limit
   Model GPU 1: 16384MB limit
   ✅ Data GPU memory limit set to 8192MB
   ✅ Model GPU memory limit set to 16384MB
   ✅ Set visible devices: GPU 0 and GPU 1

🎯 Device Mapping:
   Physical GPU 0 → Logical /GPU:0 (Data Processing)
   Physical GPU 1 → Logical /GPU:1 (Model Training)

✅ Memory-Isolated Dual-GPU Trainer initialized successfully
======================================================================

Training Progress:
[Step 100] Data GPU 0: 5248/8192MB (64.1%)
[Step 100] Model GPU 1: 14336/16384MB (87.5%)
Epoch 1/500 - Loss: 2.345 - Val Loss: 2.198
```

### نظارت بلادرنگ | Real-time Monitoring

در یک ترمینال جدید:

```bash
# نظارت مداوم GPU
watch -n 1 nvidia-smi
```

شما باید ببینید:
- **GPU 0**: 40-60% استفاده (Data Processing)
- **GPU 1**: 80-95% استفاده (Model Training)

---

## ⚡ نکات مهم | Important Tips

### 1. ترتیب اجرا | Execution Order

**مهم**: جداسازی حافظه باید قبل از هر عملیات TensorFlow اجرا شود.

✅ **درست**:
```bash
python train_main.py --enable-memory-isolation --data-gpu 0 --model-gpu 1 ...
```

❌ **نادرست**:
```python
import tensorflow as tf
# عملیات TensorFlow
setup_gpu_memory_isolation(...)  # خیلی دیر است!
```

### 2. انتخاب حافظه | Memory Selection

قانون طلایی:
- **Data GPU**: 30-40% از کل حافظه GPU
- **Model GPU**: 60-70% از کل حافظه GPU

مثال برای GPU 24GB:
```bash
--data-gpu-memory 8192   # 8GB (33%)
--model-gpu-memory 16384 # 16GB (67%)
```

### 3. Batch Size

با batch size بزرگ شروع کنید:
```bash
--batch-size 64  # سعی کنید
--batch-size 32  # اگر OOM شد
--batch-size 16  # اگر باز هم OOM شد
```

---

## 🐛 عیب‌یابی | Troubleshooting

### مشکل 1: "Out of Memory"

**راهکار**:
```bash
# کاهش batch size
--batch-size 16

# یا کاهش محدودیت حافظه
--model-gpu-memory 12288
```

### مشکل 2: "GPU already initialized"

**راهکار**: این خطا نباید رخ دهد، چون `train_main.py` به درستی مدیریت می‌کند.
اگر استفاده دستی می‌کنید، اطمینان حاصل کنید که `setup_gpu_memory_isolation()` 
قبل از هر `import tensorflow` فراخوانی شود.

### مشکل 3: استفاده پایین GPU

**راهکار**:
```bash
# افزایش batch size
--batch-size 64

# افزایش buffer size
--buffer-size 100
```

### مشکل 4: فقط یک GPU دارم

**راهکار**: از حالت عادی استفاده کنید (بدون `--enable-memory-isolation`):
```bash
python train_main.py \
    --train-data ../dataset/dataset_train \
    --val-data ../dataset/dataset_eval
```

---

## 📈 انتظارات عملکرد | Performance Expectations

### سرعت | Speed
- **2-3x سریعتر** از single GPU
- **1.3-1.5x سریعتر** از dual-GPU بدون جداسازی

### استفاده GPU | GPU Utilization
- **Data GPU**: 40-60%
- **Model GPU**: 80-95%

### استفاده حافظه | Memory Usage
- **Data GPU**: 70-90% از محدودیت
- **Model GPU**: 85-95% از محدودیت

---

## ✨ مقایسه با روش‌های دیگر | Comparison

| Mode | Data GPU | Model GPU | Speed | Complexity |
|------|----------|-----------|-------|------------|
| **Single GPU** | 50% | 50% | 1x | ساده |
| **Dual GPU (عادی)** | 60% | 70% | 1.5-2x | متوسط |
| **Memory-Isolated** | 50% | 90% | 2-3x | ساده |

---

## 🎓 مثال کامل | Complete Example

```bash
# 1. بررسی GPUها
nvidia-smi

# 2. شروع آموزش با جداسازی حافظه
python train_main.py \
    --data-gpu 0 \
    --model-gpu 1 \
    --enable-memory-isolation \
    --data-gpu-memory 8192 \
    --model-gpu-memory 16384 \
    --batch-size 32 \
    --epochs 500 \
    --lr 8e-5 \
    --train-data ../dataset/dataset_train \
    --val-data ../dataset/dataset_eval \
    --checkpoint-dir ./checkpoints

# 3. نظارت (در ترمینال جدید)
watch -n 1 nvidia-smi
```

---

## 📚 مستندات کامل | Full Documentation

برای اطلاعات بیشتر، مراجعه کنید به:
- [MEMORY_ISOLATION_GUIDE.md](docs/MEMORY_ISOLATION_GUIDE.md) - راهنمای کامل
- [DUAL_GPU_SOLUTION_PERSIAN.md](DUAL_GPU_SOLUTION_PERSIAN.md) - راهنمای dual-GPU عادی

---

## ❓ سؤالات متداول | FAQ

**Q: آیا می‌توانم بیش از 2 GPU استفاده کنم؟**
A: در حال حاضر فقط 2 GPU پشتیبانی می‌شود (یکی برای داده، یکی برای مدل).

**Q: آیا می‌توانم GPU 1 را برای داده و GPU 0 را برای مدل استفاده کنم؟**
A: بله، می‌توانید:
```bash
--data-gpu 1 --model-gpu 0
```

**Q: چگونه می‌توانم مطمئن شوم که جداسازی فعال است؟**
A: به دنبال این خطوط در لاگ بگردید:
```
🎯 Setting up GPU Memory Isolation...
✅ GPU Memory Isolation configured successfully
```

**Q: آیا می‌توانم محدودیت‌ها را در حین آموزش تغییر دهم؟**
A: خیر، محدودیت‌ها باید قبل از شروع آموزش تنظیم شوند.

---

## ✅ Checklist شروع سریع | Quick Start Checklist

- [ ] بررسی کردم که 2 GPU دارم (`nvidia-smi`)
- [ ] TensorFlow 2.12+ نصب شده است
- [ ] dataset آماده است
- [ ] محدودیت‌های حافظه مناسب را انتخاب کردم
- [ ] batch size مناسب را انتخاب کردم
- [ ] دستور اجرا را آماده کردم
- [ ] ترمینال دوم برای monitoring آماده است
- [ ] آموزش را شروع کردم و لاگ‌ها را بررسی می‌کنم

---

**موفق باشید! | Good Luck!** 🚀
