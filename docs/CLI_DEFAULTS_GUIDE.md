# راهنمای پیشفرض‌های CLI - CLI Defaults Guide

این راهنما توضیح می‌دهد که چگونه پارامترهای پیشفرض CLI برای بهبود تجربه کاربری تنظیم شده‌اند.

This guide explains how CLI default parameters have been configured to improve user experience.

## 🎯 هدف - Goal

تنظیم پیشفرض‌های منطقی برای پارامترهای خط فرمان (CLI) تا کاربران بتوانند بدون نیاز به مقداردهی دستی، آموزش را شروع کنند.

Set reasonable defaults for command-line parameters so users can start training without manual configuration.

## ✅ پیشفرض‌های جدید - New Defaults

### پارامترهای اصلی - Core Parameters

| پارامتر / Parameter | پیشفرض قبلی / Old Default | پیشفرض جدید / New Default | دلیل / Reason |
|---------------------|---------------------------|---------------------------|----------------|
| `--model-size` | `normal` | `tiny` | مناسب برای مبتدیان و سیستم‌های با منابع محدود / Suitable for beginners and limited resources |
| `--batch-size` | `64` | `16` | محافظه‌کارانه‌تر، با مدل tiny سازگارتر / More conservative, works better with tiny model |
| `--enable-static-shapes` | `False` | `True` | جلوگیری از مشکلات GPU utilization / Prevents GPU utilization issues |
| `--grad-accum` | `2` | `2` | بدون تغییر، مقدار خوبی است / No change, already good |
| `--num-workers` | `8` | `8` | بدون تغییر، مقدار خوبی است / No change, already good |
| `--buffer-size` | `50` | `50` | بدون تغییر، مقدار خوبی است / No change, already good |
| `--data-gpu` | `None` | `None` | حالت تک‌GPU (پیشفرض) / Single-GPU mode (default) |
| `--model-gpu` | `None` | `None` | حالت تک‌GPU (پیشفرض) / Single-GPU mode (default) |
| `--enable-memory-isolation` | `False` | `False` | بدون تغییر، برای راه‌اندازی ساده / No change, for simple setup |

## 🚀 استفاده - Usage

### ساده‌ترین حالت - Simplest Usage

```bash
# فقط یک دستور! - Just one command!
python3 train_main.py
```

این دستور از تمام پیشفرض‌های هوشمند استفاده می‌کند:
- مدل: tiny
- batch size: 16 (تنظیم خودکار بر اساس GPU)
- static shapes: فعال
- حالت: تک‌GPU

This command uses all smart defaults:
- Model: tiny
- Batch size: 16 (auto-adjusted based on GPU)
- Static shapes: enabled
- Mode: single-GPU

### تنظیمات دستی - Manual Override

```bash
# تغییر اندازه مدل - Change model size
python3 train_main.py --model-size small

# تغییر batch size - Change batch size
python3 train_main.py --batch-size 24

# غیرفعال کردن static shapes - Disable static shapes
python3 train_main.py --disable-static-shapes

# آموزش با دو GPU - Dual-GPU training
python3 train_main.py --data-gpu 0 --model-gpu 1
```

## 🧠 تنظیم خودکار - Auto-Adjustment

برخی پارامترها به طور خودکار بر اساس GPU شما تنظیم می‌شوند:

Some parameters are automatically adjusted based on your GPU:

### بر اساس حافظه GPU - Based on GPU Memory

| حافظه GPU | Batch Size | Num Workers | Grad Accum |
|-----------|------------|-------------|------------|
| < 10GB | 8 | 8 | 4 |
| 10-20GB | 24 | 12 | 2 |
| > 20GB | 48 | 16 | 1 |

**نکته**: اگر خودتان مقدار مشخص کنید، از تنظیم خودکار استفاده نمی‌شود.

**Note**: If you specify a value yourself, auto-adjustment is not used.

## 📊 لاگ پارامترها - Parameter Logging

در شروع آموزش، یک خلاصه کامل از پارامترها نمایش داده می‌شود:

At the start of training, a complete summary of parameters is displayed:

```
================================================================================
📋 TRAINING PARAMETERS SUMMARY
================================================================================
Core Training Parameters:
  • Model size: tiny
  • Batch size: 16
  • Gradient accumulation: 2
  • Number of workers: 8
  • Learning rate: 8e-05
  • Epochs: 500
  • Optimization level: enhanced

GPU Configuration:
  • Data GPU: Auto (single-GPU mode)
  • Model GPU: Auto (single-GPU mode)
  • Memory isolation: False
  • Buffer size: 50

Optimization Features:
  • Static shapes: True
    - Max text length: 200
    - Max mel frames: auto

Dataset Paths:
  • Training data: ../dataset/dataset_train
  • Validation data: ../dataset/dataset_eval
================================================================================
```

## 🎓 توصیه‌ها - Recommendations

### برای مبتدیان - For Beginners

```bash
# استفاده از تمام پیشفرض‌ها - Use all defaults
python3 train_main.py
```

### برای کاربران میانی - For Intermediate Users

```bash
# مدل بزرگتر برای کیفیت بهتر - Larger model for better quality
python3 train_main.py --model-size small --batch-size 24
```

### برای کاربران پیشرفته - For Advanced Users

```bash
# تمام بهینه‌سازی‌ها - All optimizations
python3 train_main.py \
    --model-size normal \
    --optimization-level enhanced \
    --batch-size 32 \
    --enable-static-shapes \
    --num-workers 16
```

### برای دو GPU - For Dual-GPU

```bash
# حالت Multi-GPU خودکار فعال می‌شود - Multi-GPU mode automatically activated
python3 train_main.py --data-gpu 0 --model-gpu 1
```

## 📝 یادداشت‌های مهم - Important Notes

### 1. Static Shapes (پیشفرض: فعال)

**چرا فعال شده؟** جلوگیری از retracing و بهبود GPU utilization

**چگونه غیرفعال کنیم؟**
```bash
python3 train_main.py --disable-static-shapes
```

**Why enabled?** Prevents retracing and improves GPU utilization

**How to disable?**
```bash
python3 train_main.py --disable-static-shapes
```

### 2. Model Size (پیشفرض: tiny)

**چرا tiny؟** 
- سریع‌تر برای تست و یادگیری
- کمتر حافظه نیاز دارد
- برای production می‌توانید به small یا normal تغییر دهید

**Why tiny?**
- Faster for testing and learning
- Requires less memory
- For production, you can change to small or normal

### 3. Batch Size (پیشفرض: 16)

**چرا 16؟**
- با مدل tiny سازگار است
- به طور خودکار بر اساس GPU تنظیم می‌شود
- اگر حافظه بیشتری دارید، می‌توانید افزایش دهید

**Why 16?**
- Compatible with tiny model
- Automatically adjusted based on GPU
- Can increase if you have more memory

## 🔍 مثال‌های استفاده - Usage Examples

### مثال 1: آموزش سریع برای تست
```bash
python3 train_main.py --epochs 10
```

### مثال 2: آموزش production با کیفیت بالا
```bash
python3 train_main.py --model-size normal --batch-size 32 --epochs 500
```

### مثال 3: آموزش با دو GPU
```bash
python3 train_main.py \
    --data-gpu 0 \
    --model-gpu 1 \
    --batch-size 48 \
    --buffer-size 100
```

### مثال 4: آموزش با حافظه محدود
```bash
python3 train_main.py \
    --model-size tiny \
    --batch-size 8 \
    --grad-accum 4
```

## 🆘 عیب‌یابی - Troubleshooting

### مشکل: Out of Memory

**راه‌حل:**
```bash
python3 train_main.py --batch-size 8 --grad-accum 4
```

### مشکل: GPU Utilization پایین

**راه‌حل:** اطمینان حاصل کنید static shapes فعال است (پیشفرض)
```bash
# بررسی کنید که static shapes فعال است
python3 train_main.py  # پیشفرض فعال است
```

### مشکل: آموزش خیلی کند

**راه‌حل:** استفاده از مدل کوچکتر یا batch size بزرگتر
```bash
python3 train_main.py --model-size tiny --batch-size 24
```

## 📚 مستندات بیشتر - Further Documentation

- [README.md](../README.md) - راهنمای اصلی / Main guide
- [SOLUTION_PERSIAN.md](../SOLUTION_PERSIAN.md) - راه‌حل مشکلات GPU
- [DUAL_GPU_SOLUTION_PERSIAN.md](../DUAL_GPU_SOLUTION_PERSIAN.md) - راهنمای دو GPU

## 🎉 نتیجه‌گیری - Conclusion

با پیشفرض‌های جدید، آموزش مدل بسیار ساده‌تر شده است:

With the new defaults, model training has become much simpler:

**قبل - Before:**
```bash
python3 train_main.py \
    --model-size tiny \
    --batch-size 16 \
    --enable-static-shapes \
    --data-gpu 0 \
    --num-workers 8 \
    --buffer-size 50 \
    --train-data ../dataset/dataset_train \
    --val-data ../dataset/dataset_eval
```

**حالا - Now:**
```bash
python3 train_main.py
```

همین! 🎯

That's it! 🎯
