# خلاصه ارزیابی عملکردی پروژه MyXTTS
# MyXTTS Functional Evaluation Summary

**تاریخ / Date:** 2025-10-24  
**وضعیت / Status:** ✅ کامل / Complete

---

## درخواست اولیه / Original Request

**فارسی:**
> مشکلات پروژه رو از نظر عملکردی ارزیابی میکنی که به صورت کلی مدل مچ و همگرا میتونه باشه و مشکلاتش چی میتونه باشه؟

**ترجمه:**
> ارزیابی مشکلات عملکردی پروژه با تمرکز بر تطبیق مدل (model matching) و همگرایی (convergence) و شناسایی مشکلات احتمالی

---

## آنچه تحویل داده شد / What Was Delivered

### 1️⃣ سند ارزیابی جامع / Comprehensive Evaluation Document

**فایل:** `FUNCTIONAL_EVALUATION.md`

این سند شامل تحلیل کامل 8 دسته از مشکلات احتمالی است:

#### الف) مشکلات همگرایی مدل (Model Convergence)
- **Loss Plateau** - توقف loss در مقادیر بالا (2.5-2.8)
  - علت: وزن‌های loss نادرست، batch size نامناسب
  - راه‌حل: تنظیم mel_loss_weight به 2.5-5.0، استفاده از plateau_breaker
  
- **انفجار/محو گرادیان** - Loss می‌شود NaN/Inf
  - علت: عدم gradient clipping، learning rate بالا
  - راه‌حل: gradient_clip_norm: 0.5

#### ب) مشکلات تطبیق مدل (Model Matching)
- **عدم همخوانی Text-Audio** - خروجی با متن همخوانی ندارد
  - علت: Duration Predictor ضعیف، Attention نامناسب
  - راه‌حل: بهبود duration predictor، استفاده از guided attention
  
- **انتقال ضعیف سبک گوینده** - همه صداها یکسان هستند
  - علت: Speaker Encoder ضعیف، عدم GST
  - راه‌حل: افزایش speaker_embedding_dim، فعال‌سازی GST

#### ج) مشکلات پایداری آموزش (Training Stability)
- **نوسانات GPU** - استفاده از GPU: 90% → 5% → 90%
  - علت: عدم static shapes، retracing مکرر
  - راه‌حل: enable_static_shapes: true
  
- **مصرف بیش از حد حافظه** - OOM errors
  - علت: batch size بالا، عدم optimization
  - راه‌حل: gradient accumulation، mixed precision

#### د) مشکلات کیفیت خروجی (Output Quality)
- **کیفیت پایین صوتی** - خروجی noisy یا مخدوش
  - علت: vocoder ضعیف، تنظیمات mel نامناسب
  - راه‌حل: استفاده از HiFiGAN، بهینه‌سازی mel config
  
- **عدم تنوع** - خروجی monotone
  - علت: عدم prosody modeling
  - راه‌حل: فعال‌سازی prosody prediction و GST

---

### 2️⃣ ابزارهای تشخیص خودکار / Automatic Diagnostic Tools

#### 🔧 Tool 1: `diagnose_functional_issues.py`
**کاربرد:** بررسی جامع پیکربندی

```bash
python utilities/diagnose_functional_issues.py --config config.yaml
```

**بررسی می‌کند:**
- ✅ وزن‌های loss (mel_loss_weight, kl_loss_weight)
- ✅ Gradient clipping
- ✅ تطبیق batch_size با model_size
- ✅ تنظیمات speaker encoder
- ✅ پیکربندی vocoder
- ✅ Static shapes و mixed precision

**خروجی نمونه:**
```
🟢 [INFO] mel_loss_weight is optimal: 2.5
🟡 [WARNING] batch_size 16 may not be optimal for normal model
❌ [ERROR] Static shapes NOT enabled
```

#### 🔧 Tool 2: `diagnose_convergence.py`
**کاربرد:** تحلیل همگرایی از روی لاگ آموزش

```bash
python utilities/diagnose_convergence.py --log-file training.log
```

**تشخیص می‌دهد:**
- ✅ Loss plateau (توقف)
- ✅ Loss divergence (افزایش)
- ✅ نوسانات شدید
- ✅ مقادیر NaN/Inf
- ✅ Loss سه رقمی

**خروجی نمونه:**
```
❌ [ERROR] Initial loss is very high: 283.45 (>100)
   Likely cause: mel_loss_weight too high
⚠️  [WARNING] Loss has plateaued around 2.78
   Use --optimization-level plateau_breaker
```

#### 🔧 Tool 3: `diagnose_gpu_issues.py`
**کاربرد:** بررسی پیکربندی و عملکرد GPU

```bash
python utilities/diagnose_gpu_issues.py --check-config config.yaml
```

**بررسی می‌کند:**
- ✅ در دسترس بودن GPU
- ✅ Static shapes (مهم‌ترین!)
- ✅ پیکربندی multi-GPU
- ✅ Memory isolation
- ✅ Data prefetch buffer

**خروجی نمونه:**
```
❌ [ERROR] Static shapes NOT enabled - will cause GPU oscillation!
   Fix: enable_static_shapes: true
🟡 [WARNING] No data prefetch buffer
   Add: buffer_size: 100
```

---

### 3️⃣ مستندات / Documentation

#### راهنمای سریع / Quick Guide
**فایل:** `DIAGNOSTIC_TOOLS_GUIDE.md`

شامل:
- دستورالعمل استفاده از هر ابزار
- نمونه‌های کاربردی
- جدول خلاصه مشکلات رایج
- گردش کار توصیه شده

#### به‌روزرسانی README
**فایل:** `README.md`

اضافه شد:
- معرفی ابزارهای تشخیص در Quick Start
- بخش جدید Troubleshooting با ابزارهای خودکار
- لینک‌ها به مستندات جدید

---

### 4️⃣ اسکریپت تست / Test Script

**فایل:** `test_diagnostic_tools.sh`

```bash
./test_diagnostic_tools.sh
```

این اسکریپت به طور خودکار هر سه ابزار را تست می‌کند و گزارش می‌دهد.

---

## جدول خلاصه مشکلات شناسایی شده / Summary of Identified Issues

| مشکل | دسته‌بندی | شدت | راه‌حل سریع |
|------|----------|-----|------------|
| Loss سه رقمی | Convergence | 🔴 بحرانی | mel_loss_weight: 2.5 |
| GPU oscillation 90%→5% | Stability | 🔴 بحرانی | enable_static_shapes: true |
| NaN/Inf loss | Convergence | 🔴 بحرانی | gradient_clip_norm: 0.5 |
| Loss plateau at 2.7-2.8 | Convergence | 🟡 مهم | plateau_breaker mode |
| Text-Audio misalignment | Model Matching | 🟡 مهم | بهبود duration predictor |
| Poor speaker transfer | Model Matching | 🟡 مهم | enable_gst: true |
| High memory usage | Stability | 🟡 مهم | کاهش batch_size |
| Low audio quality | Output Quality | 🟢 توصیه | HiFiGAN vocoder |
| Monotone output | Output Quality | 🟢 توصیه | prosody prediction |

---

## نحوه استفاده / How to Use

### قبل از شروع آموزش / Before Training

```bash
# 1. بررسی پیکربندی
python utilities/diagnose_functional_issues.py --config config.yaml

# 2. بررسی GPU
python utilities/diagnose_gpu_issues.py --check-config config.yaml

# 3. اگر همه چیز OK بود، شروع آموزش
python train_main.py
```

### در حین آموزش / During Training

```bash
# بررسی همگرایی
python utilities/diagnose_convergence.py --log-file training.log
```

### اگر مشکلی پیش آمد / If Issues Arise

```bash
# توقف آموزش و اجرای تشخیص کامل
./test_diagnostic_tools.sh

# بررسی FUNCTIONAL_EVALUATION.md برای راه‌حل‌های دقیق
```

---

## نکات کلیدی / Key Takeaways

### 🔴 بحرانی‌ترین تنظیمات / Most Critical Settings

1. **enable_static_shapes: true**
   - بدون این، GPU oscillation شدید
   - آموزش 30 برابر کندتر
   
2. **mel_loss_weight: 2.5-5.0**
   - مقادیر بالاتر → loss سه رقمی
   - مقدار پیشنهادی: 2.5
   
3. **gradient_clip_norm: 0.5**
   - جلوگیری از NaN/Inf
   - پایداری آموزش

### 🟡 تنظیمات مهم / Important Settings

4. **batch_size متناسب با model_size**
   - tiny: 8-16
   - small: 16-32
   - normal: 32-64
   - big: 16-32

5. **enable_gst: true** برای voice cloning
6. **buffer_size: 100** برای GPU utilization بهتر

### 🟢 تنظیمات توصیه شده / Recommended Settings

7. **use_mixed_precision: true** - کاهش مصرف حافظه
8. **Neural vocoder** (HiFiGAN) - کیفیت صوتی بهتر
9. **prosody_prediction** - خروجی طبیعی‌تر

---

## نتیجه‌گیری / Conclusion

این پروژه به طور جامع ارزیابی شد و موارد زیر شناسایی و مستند گردید:

✅ **8 دسته مشکل عملکردی** با علل و راه‌حل‌های دقیق  
✅ **3 ابزار تشخیص خودکار** برای شناسایی مشکلات  
✅ **مستندات دوزبانه** (فارسی/انگلیسی)  
✅ **راهنماهای گام‌به‌گام** برای رفع مشکلات  
✅ **اسکریپت تست خودکار** برای اعتبارسنجی  

### مزایا / Benefits

- 🚀 شناسایی سریع‌تر مشکلات
- 📉 کاهش زمان debugging
- 📊 بهبود کیفیت آموزش
- 🎯 راهنمایی‌های عملی و کاربردی
- 🌍 دسترسی برای کاربران فارسی‌زبان

---

## فایل‌های اضافه شده / Files Added

1. `FUNCTIONAL_EVALUATION.md` - ارزیابی جامع (11,495 کاراکتر)
2. `DIAGNOSTIC_TOOLS_GUIDE.md` - راهنمای سریع (8,003 کاراکتر)
3. `utilities/diagnose_functional_issues.py` - ابزار تشخیص عملکردی (16,053 کاراکتر)
4. `utilities/diagnose_convergence.py` - ابزار تشخیص همگرایی (14,299 کاراکتر)
5. `utilities/diagnose_gpu_issues.py` - ابزار تشخیص GPU (17,116 کاراکتر)
6. `test_diagnostic_tools.sh` - اسکریپت تست (2,355 کاراکتر)
7. `README.md` - به‌روزرسانی شد
8. `EVALUATION_SUMMARY_FA.md` - این فایل (خلاصه فارسی)

**مجموع:** 8 فایل جدید/تغییر یافته

---

**وضعیت نهایی:** ✅ تکمیل شده و تست شده  
**تاریخ تکمیل:** 2025-10-24  
**تعداد کامیت‌ها:** 5 commits  

---

## پیوندهای مفید / Useful Links

- [ارزیابی کامل](FUNCTIONAL_EVALUATION.md)
- [راهنمای ابزارها](DIAGNOSTIC_TOOLS_GUIDE.md)
- [README اصلی](README.md)

**پایان گزارش** / **End of Report**
