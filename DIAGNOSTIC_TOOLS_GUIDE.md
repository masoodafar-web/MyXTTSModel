# راهنمای سریع تشخیص مشکلات / Quick Diagnostic Guide

## ابزارهای تشخیص خودکار / Automatic Diagnostic Tools

این پروژه شامل سه ابزار تشخیص خودکار برای شناسایی مشکلات عملکردی است:

This project includes three automatic diagnostic tools to identify functional issues:

---

## 1. تشخیص مشکلات عملکردی کلی / General Functional Issues

### استفاده / Usage:
```bash
# تشخیص جامع بر اساس فایل config
# Comprehensive diagnostic based on config file
python utilities/diagnose_functional_issues.py --config config.yaml

# با خروجی verbose
# With verbose output
python utilities/diagnose_functional_issues.py --config config.yaml --verbose
```

### چه چیزی را بررسی می‌کند / What It Checks:
- ✅ وزن‌های loss (mel_loss_weight, kl_loss_weight)
- ✅ Loss weights (mel_loss_weight, kl_loss_weight)
- ✅ تنظیمات gradient clipping
- ✅ Gradient clipping settings
- ✅ تطبیق batch_size با model_size
- ✅ Batch size vs model size matching
- ✅ تنظیمات speaker encoder
- ✅ Speaker encoder settings
- ✅ پیکربندی vocoder
- ✅ Vocoder configuration
- ✅ پایداری آموزش
- ✅ Training stability

### خروجی نمونه / Sample Output:
```
======================================================================
              MyXTTS Functional Issue Diagnostic Report               
======================================================================

🟢 [INFO] mel_loss_weight is optimal: 2.5
🟡 [WARNING] batch_size 16 may not be optimal for normal model
❌ [ERROR] No gradient clipping configured
```

---

## 2. تشخیص مشکلات همگرایی / Convergence Issues

### استفاده / Usage:
```bash
# تحلیل از روی فایل لاگ
# Analyze from log file
python utilities/diagnose_convergence.py --log-file training.log

# تحلیل checkpoint
# Analyze checkpoint
python utilities/diagnose_convergence.py --checkpoint path/to/checkpoint

# هر دو
# Both
python utilities/diagnose_convergence.py --log-file training.log --checkpoint path/to/checkpoint
```

### چه چیزی را بررسی می‌کند / What It Checks:
- ✅ Loss plateau (توقف loss)
- ✅ Loss plateau detection
- ✅ Loss divergence (افزایش loss)
- ✅ Loss divergence (increasing loss)
- ✅ نوسانات شدید loss
- ✅ Severe loss oscillations
- ✅ مقادیر NaN/Inf
- ✅ NaN/Inf values
- ✅ Loss سه رقمی
- ✅ Three-digit loss values

### خروجی نمونه / Sample Output:
```
======================================================================
                        Convergence Analysis                          
======================================================================

Parsed 1000 loss values from log file
Loss range: 0.8934 - 283.4512

❌ [ERROR] Initial loss is very high: 283.45 (>100)
   Likely cause: mel_loss_weight too high (should be 2.5-5.0)

⚠️  [WARNING] Loss has plateaued around 2.7834
   Suggestions:
   - Use --optimization-level plateau_breaker
```

---

## 3. تشخیص مشکلات GPU / GPU Issues

### استفاده / Usage:
```bash
# بررسی پیکربندی GPU
# Check GPU configuration
python utilities/diagnose_gpu_issues.py --check-config config.yaml

# پروفایل استفاده از GPU
# Profile GPU utilization
python utilities/diagnose_gpu_issues.py --profile-steps 100

# تشخیص کامل
# Full diagnostic
python utilities/diagnose_gpu_issues.py --check-config config.yaml --profile-steps 100
```

### چه چیزی را بررسی می‌کند / What It Checks:
- ✅ در دسترس بودن GPU
- ✅ GPU availability
- ✅ تنظیمات static shapes
- ✅ Static shapes settings
- ✅ پیکربندی multi-GPU
- ✅ Multi-GPU configuration
- ✅ مشکلات retracing
- ✅ Retracing issues
- ✅ استفاده از حافظه
- ✅ Memory usage
- ✅ data prefetch buffer

### خروجی نمونه / Sample Output:
```
======================================================================
                       GPU Configuration Check                        
======================================================================

❌ [ERROR] Static shapes NOT enabled - will cause severe GPU utilization issues!
   Impact: GPU utilization will oscillate (90% → 5% → 90%)
   Fix: Add 'enable_static_shapes: true' or use --enable-static-shapes

✅ [INFO] Multi-GPU mode detected: data_gpu=0, model_gpu=1
⚠️  [WARNING] Memory isolation not enabled for multi-GPU
```

---

## جدول خلاصه مشکلات رایج / Common Issues Summary

| مشکل / Issue | ابزار / Tool | راه‌حل سریع / Quick Fix |
|-------------|-------------|------------------------|
| Loss سه رقمی / Three-digit loss | `diagnose_functional_issues.py` | `mel_loss_weight: 2.5` |
| Loss plateau | `diagnose_convergence.py` | `--optimization-level plateau_breaker` |
| GPU oscillation 90%→5% | `diagnose_gpu_issues.py` | `enable_static_shapes: true` |
| NaN/Inf loss | `diagnose_convergence.py` | `gradient_clip_norm: 0.5` |
| OOM errors | `diagnose_gpu_issues.py` | کاهش batch_size / Reduce batch_size |
| Poor voice cloning | `diagnose_functional_issues.py` | `enable_gst: true` |

---

## گردش کار توصیه شده / Recommended Workflow

### قبل از شروع آموزش / Before Starting Training:

```bash
# 1. بررسی پیکربندی کلی
# 1. Check general configuration
python utilities/diagnose_functional_issues.py --config config.yaml

# 2. بررسی تنظیمات GPU
# 2. Check GPU settings
python utilities/diagnose_gpu_issues.py --check-config config.yaml

# اگر همه چیز OK باشد، شروع آموزش
# If everything is OK, start training
python train_main.py
```

### در حین آموزش / During Training:

```bash
# بررسی همگرایی از روی لاگ
# Check convergence from logs
python utilities/diagnose_convergence.py --log-file training.log

# اگر مشکلی بود، توقف و رفع مشکل
# If issues found, stop and fix
```

### زمان رفع مشکل / When Troubleshooting:

```bash
# اجرای هر سه ابزار
# Run all three tools
python utilities/diagnose_functional_issues.py --config config.yaml
python utilities/diagnose_convergence.py --log-file training.log
python utilities/diagnose_gpu_issues.py --check-config config.yaml
```

---

## نکات مهم / Important Notes

### ⚠️ اولویت‌ها / Priorities:

1. **🔴 بحرانی / CRITICAL:**
   - `enable_static_shapes: true` (همیشه / always!)
   - `gradient_clip_norm: 0.5` (برای جلوگیری از NaN)
   - `mel_loss_weight: 2.5-5.0` (نه بیشتر!)

2. **🟡 مهم / IMPORTANT:**
   - تطبیق batch_size با model_size
   - Match batch_size to model_size
   - فعال‌سازی GST برای voice cloning
   - Enable GST for voice cloning

3. **🟢 توصیه شده / RECOMMENDED:**
   - `use_mixed_precision: true`
   - `buffer_size: 100`
   - Neural vocoder (HiFiGAN)

---

## خروجی‌های رنگی / Color-Coded Output

ابزارهای تشخیص از رنگ‌ها برای نشان دادن شدت مشکلات استفاده می‌کنند:

The diagnostic tools use colors to indicate severity:

- 🔴 **ERROR** (قرمز / Red): مشکلات بحرانی - باید فوراً رفع شوند
- 🔴 **ERROR** (Red): Critical issues - must be fixed immediately
- 🟡 **WARNING** (زرد / Yellow): مشکلاتی که باید بررسی شوند
- 🟡 **WARNING** (Yellow): Issues that should be reviewed
- 🟢 **INFO** (سبز / Green): اطلاعاتی یا توصیه‌ها
- 🟢 **INFO** (Green): Informational or recommendations

---

## یکپارچه‌سازی با سیستم آموزش / Integration with Training

می‌توانید این ابزارها را در اسکریپت‌های آموزش خود یکپارچه کنید:

You can integrate these tools into your training scripts:

```bash
#!/bin/bash
# اسکریپت آموزش ایمن / Safe training script

echo "Running pre-flight checks..."

# تشخیص مشکلات
# Run diagnostics
python utilities/diagnose_functional_issues.py --config config.yaml
if [ $? -ne 0 ]; then
    echo "Configuration issues found! Fix before training."
    exit 1
fi

python utilities/diagnose_gpu_issues.py --check-config config.yaml
if [ $? -ne 0 ]; then
    echo "GPU issues found! Fix before training."
    exit 1
fi

echo "All checks passed! Starting training..."
python train_main.py --config config.yaml
```

---

## پیوندهای مفید / Useful Links

- [ارزیابی عملکردی کامل / Full Functional Evaluation](FUNCTIONAL_EVALUATION.md)
- [راهنمای رفع مشکل plateau](docs/LOSS_PLATEAU_2.8_TINY_ENHANCED_FIX.md)
- [راهنمای بهینه‌سازی GPU](docs/GPU_UTILIZATION_FIX_GUIDE.md)
- [راهنمای multi-GPU](DUAL_GPU_BOTTLENECK_FIX.md)

---

**نکته:** همیشه قبل از شروع آموزش، ابزارهای تشخیص را اجرا کنید!

**Note:** Always run diagnostic tools before starting training!

**تاریخ / Date:** 2025-10-24  
**نسخه / Version:** 1.0
