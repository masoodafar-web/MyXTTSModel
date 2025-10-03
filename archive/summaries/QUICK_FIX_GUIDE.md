# راهنمای سریع رفع مشکل نویز | Quick Fix Guide for Noise Issue

## 🔴 مشکل: خروجی فقط نویز است | Problem: Output is Only Noise

### علائم | Symptoms
- ✅ مدل بدون خطا اجرا می‌شود | Model runs without errors
- ✅ فایل صوتی تولید می‌شود | Audio file is generated
- ❌ فقط صدای نویز شنیده می‌شود | Only noise is heard
- ❌ هیچ کلمه‌ای قابل فهم نیست | No intelligible words

## 🟢 راه حل فوری | Immediate Solution

### گام ۱: اجرای استاندارد | Step 1: Run Normally

```bash
python3 inference_main.py \
    --text "متن شما اینجا" \
    --model-size tiny \
    --output test.wav
```

**سیستم خودکار عمل می‌کند:**
- 🔍 نویز را تشخیص می‌دهد
- ⚠️ هشدار نمایش می‌دهد
- 🔄 به گریفین-لیم تغییر می‌کند
- ✅ صدای قابل فهم تولید می‌کند

**System automatically:**
- 🔍 Detects the issue
- ⚠️ Shows warnings
- 🔄 Switches to Griffin-Lim
- ✅ Produces intelligible audio

### گام ۲: بررسی خروجی | Step 2: Check Output

اگر این پیام را دیدید:
```
⚠️  VOCODER WEIGHTS NOT INITIALIZED WARNING
```

**یعنی:**
- مدل به اندازه کافی آموزش ندیده
- واکودر وزن‌های تصادفی دارد
- سیستم از گریفین-لیم استفاده می‌کند

**It means:**
- Model not trained enough
- Vocoder has random weights
- System using Griffin-Lim fallback

## 🔵 راه حل دائمی | Permanent Solution

### روش ۱: آموزش بیشتر (توصیه می‌شود) | Method 1: More Training (Recommended)

```bash
python3 train_main.py \
    --train-data ../dataset/dataset_train \
    --val-data ../dataset/dataset_eval \
    --model-size tiny \
    --epochs 100
```

**زمان مورد نیاز:**
- حداقل ۲۰-۵۰ هزار گام
- برای کیفیت خوب: ۵۰-۱۰۰ هزار گام

**Required time:**
- Minimum: 20-50k steps
- For good quality: 50-100k steps

### روش ۲: بارگذاری چک‌پوینت آموزش‌دیده | Method 2: Load Trained Checkpoint

```bash
python3 inference_main.py \
    --text "متن شما" \
    --checkpoint path/to/trained/checkpoint \
    --output output.wav
```

## 📊 مقایسه کیفیت | Quality Comparison

| وضعیت | کیفیت | سرعت | Status | Quality | Speed |
|-------|--------|------|--------|---------|-------|
| واکودر آموزش‌ندیده | ❌ نویز | ⚡ | Untrained Vocoder | ❌ Noise | ⚡ |
| گریفین-لیم (فعلی) | ⭐⭐⭐ | 🐌 | Griffin-Lim (Current) | ⭐⭐⭐ | 🐌 |
| واکودر آموزش‌دیده | ⭐⭐⭐⭐⭐ | ⚡ | Trained Vocoder | ⭐⭐⭐⭐⭐ | ⚡ |

## ❓ سوالات متداول | FAQ

### س: چرا نویز دارم؟
**ج:** واکودر (HiFi-GAN) به آموزش نیاز دارد. وزن‌های تصادفی = خروجی تصادفی (نویز)

### Q: Why do I get noise?
**A:** Vocoder (HiFi-GAN) needs training. Random weights = random output (noise)

---

### س: چقدر باید آموزش بدهم؟
**ج:**
- حداقل: ۲۰ هزار گام
- خوب: ۵۰ هزار گام
- عالی: ۱۰۰+ هزار گام

### Q: How long to train?
**A:**
- Minimum: 20k steps
- Good: 50k steps
- Excellent: 100k+ steps

---

### س: گریفین-لیم چیست؟
**ج:** الگوریتم کلاسیک که بدون آموزش کار می‌کند. کیفیت کمتر ولی کار می‌کند!

### Q: What is Griffin-Lim?
**A:** Classical algorithm that works without training. Lower quality but it works!

---

### س: چگونه بفهمم واکودر آموزش دیده؟
**ج:** اگر هشدار ندیدید و کیفیت عالی بود = واکودر آموزش دیده ✅

### Q: How to know vocoder is trained?
**A:** If no warning and quality is excellent = vocoder is trained ✅

## 🎯 اقدامات عملی | Action Items

### الان (هم‌اکنون)
1. ✅ از سیستم استفاده کنید (گریفین-لیم خودکار است)
2. ✅ صدای قابل فهم خواهید داشت
3. ⚠️ کیفیت کمی پایین‌تر است

### Now (Immediately)
1. ✅ Use the system (Griffin-Lim is automatic)
2. ✅ You'll get intelligible audio
3. ⚠️ Quality is a bit lower

### بعداً (برنامه‌ریزی کنید)
1. 🎓 مدل را بیشتر آموزش بدهید
2. 🎯 هدف: ۵۰-۱۰۰ هزار گام
3. 🔊 کیفیت عالی خواهید داشت

### Later (Plan for)
1. 🎓 Train the model more
2. 🎯 Target: 50-100k steps  
3. 🔊 You'll get excellent quality

## 📚 اطلاعات بیشتر | More Information

- **جزئیات کامل:** `docs/VOCODER_NOISE_FIX.md`
- **خلاصه راه‌حل:** `VOCODER_NOISE_ISSUE_SOLUTION.md`
- **مشکل استاپ زودهنگام:** `docs/INFERENCE_FIX_EARLY_STOP.md`

- **Full Details:** `docs/VOCODER_NOISE_FIX.md`
- **Solution Summary:** `VOCODER_NOISE_ISSUE_SOLUTION.md`
- **Early Stop Issue:** `docs/INFERENCE_FIX_EARLY_STOP.md`

## ✅ خلاصه | Summary

**الان:**
- سیستم کار می‌کند ✅
- صدا قابل فهم است ✅
- کیفیت متوسط است ⭐⭐⭐

**بعد از آموزش:**
- سیستم کار می‌کند ✅
- صدا قابل فهم است ✅
- کیفیت عالی است ⭐⭐⭐⭐⭐

**Now:**
- System works ✅
- Audio is intelligible ✅
- Quality is medium ⭐⭐⭐

**After Training:**
- System works ✅
- Audio is intelligible ✅
- Quality is excellent ⭐⭐⭐⭐⭐

---

**نکته مهم:** این سیستم مشکل را حل نمی‌کند، بلکه به شما اجازه می‌دهد تا در حین آموزش، از سیستم استفاده کنید.

**Important Note:** This system doesn't magically fix the issue, but allows you to use the system while training progresses.
